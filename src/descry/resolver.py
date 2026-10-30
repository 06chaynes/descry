"""Call resolution: the standard libraries' data and `CallResolver`."""

import contextlib
import os
import posixpath
import re
from functools import cache
from itertools import zip_longest
from pathlib import Path

from descry import _stdlib
from descry._parsing import (
    _TS_TYPEOF_RE,
    _bracketed_parts,
    _split_callee,
    _strip_generics,
)
from descry._reads import Probe
from descry.rust_parser import _RUST_BORROW_RE, _RUST_FN_TYPE_RE, _rust_borrow
from descry.ts_parser import _ts_function_property

_JSTS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")


_C_HEADER_EXTENSIONS = (".h", ".hh", ".hpp", ".hxx")


# The values of the builtins that make a function a static or class method
_PY_METHOD_WRAPPERS = (
    ("library", "builtins.staticmethod"),
    ("library", "builtins.classmethod"),
)


# What an instance of a tuple, and of `type[C]`, starts with as a value
_PY_TUPLES = (("outside", "builtins.tuple"), ("outside", "typing.Tuple"))


_PY_CLASS_OBJECTS = (("outside", "builtins.type"), ("outside", "typing.Type"))


def c_specialization(name, arguments):
    """How a C++ specialization of the template ``name`` is named, by the
    text of its template ``arguments`` after their `<`: as written, each
    `::` in them written `.`, which no source writes, so its ID splits."""
    listed = ", ".join(" ".join(a.split()) for a in _bracketed_parts(arguments))
    listed = re.sub(r"\s*::\s*", ".", listed)
    return f"{name}<{listed}>"


def _c_deduced(pattern, given, params, unknown):
    """Whether the C++ template arguments ``given`` are an instance of the
    ``pattern`` of them a specialization is written with, each of its
    ``params`` standing for one type wherever it is written: True, False,
    or None where it is not told: a pack, a different number of
    arguments, which a default may make up, or a type that opens with a
    name of ``unknown`` on either side, which may stand for any type."""
    if len(pattern) != len(given) or any("..." in p for p in pattern):
        return None

    def tokens(parts):
        """The tokens of ``parts``, each followed by a `,`: a qualified
        name is one, which no alias that stands alone is."""
        found = (re.findall(r"\w+(?:\.\w+)*|\S", part) for part in parts)
        return [t for part in found for t in (*part, ",")]

    def types(held):
        """The lengths of the runs of ``held`` from its start that are a
        type."""
        depth = 0
        for size, token in enumerate(held, 1):
            depth += (token in "<([") - (token in ">)]")
            if depth < 0 or token == "," and not depth:
                return
            if not depth:
                yield size

    def aliased(held):
        """How many tokens of ``held`` the type a name of ``unknown`` opens
        it with spans: the name, after a `typename`, with its template
        arguments and the members named after them."""
        size = 1 + (held[0] == "typename")
        while size < len(held) and held[size] in ("<", "."):
            if held[size] == ".":
                size += 2
                continue
            depth = 0
            while True:
                depth += (held[size] == "<") - (held[size] == ">")
                size += 1
                if not depth or size == len(held):
                    break
        return size

    def fits(written, held, bound, loose):
        if not written or not held:
            return not written and not held
        head = written[0]
        if loose and held[0] in unknown:
            spans = aliased(held)
            if any(
                fits(written[n:], held[spans:], bound, loose) for n in types(written)
            ):
                return True
        if loose and head in unknown and head not in params:
            spans = aliased(written)
            if any(fits(written[spans:], held[n:], bound, loose) for n in types(held)):
                return True
        if head not in params:
            return held[0] == head and fits(written[1:], held[1:], bound, loose)
        return any(
            bound.get(head, run) == run
            and fits(written[1:], held[len(run) :], {**bound, head: run}, loose)
            for run in (tuple(held[:n]) for n in types(held))
        )

    written, held = tokens(pattern), tokens(given)
    if fits(written, held, {}, False):
        return True
    return None if fits(written, held, {}, True) else False


@cache
def _std_aliases():
    """The names in `std` that the C++ data has no class of that name for,
    an alias's or one it lacks, written as `c_specialization` writes them:
    each may stand for any type."""
    data = _stdlib.load("cpp")
    return frozenset(
        f"std.{name}"
        for name, key in data["scope"].items()
        if key not in data["types"] or key.split("::")[-1].split("`")[0] != name
    )


def _member_split(node_id, name):
    """``(owner ID, name)`` of the node ``node_id`` named ``name``, whose
    name, a C++ conversion operator's, may hold `::` itself; no owner for a
    file."""
    if "::" in name and node_id.endswith(f"::{name}"):
        return node_id[: -len(name) - 2], name
    owner, _, last = node_id.rpartition("::")
    return owner, last


def _substituted(written, binding):
    """The type ``written`` without spaces, each type parameter that
    ``binding`` maps written as its argument."""
    held = re.sub(r"[\w$]+", lambda word: binding.get(word[0], word[0]), written)
    return "".join(held.split())


_PARAMETER_MODIFIERS = ("ref", "out", "in", "this", "params", "scoped", "readonly")


def _moded(written, binding):
    """``(mode, type)`` of a C# parameter written ``written``: `ref`, `out`,
    `in` or "" by what it is passed as, and its type as `_substituted`
    gives it, without modifiers that do not change it."""
    words, mode = written.split(), ""
    while len(words) > 1 and words[0] in _PARAMETER_MODIFIERS:
        word = words.pop(0)
        mode = word if word in ("ref", "out", "in") else mode
    return mode, _substituted(" ".join(words), binding)


def _admits(entry, passed):
    """Whether a method overload whose `[least, most, types]` is ``entry``
    takes ``passed`` arguments, any number where that is not known."""
    least, most, _ = entry
    return (
        not isinstance(passed, int)
        or least <= passed
        and (most is None or passed <= most)
    )


def _rust_declared(declared):
    """The declarations of a Rust alias, but the unit type `()` where a
    `#[cfg]` pair writes it as a stand-in for the other: it has no members."""
    return {d for d in declared if d[1] != "()"} or declared


@cache
def _rust_type_parts(text):
    """``(name, type arguments, associated types)`` of a Rust type as
    written, whatever it is borrowed as. A slice, an array and a tuple have
    the names the standard library's data gives them: `&'a [T]` is `[T]`."""
    text = _RUST_BORROW_RE.sub("", text)
    if text.startswith("["):
        element, _, length = ", ".join(_bracketed_parts(text[1:])).partition(";")
        if length:
            return "[T; N]", (element.strip(), length.strip()), {}
        return "[T]", (element,), {}
    if text.startswith("("):
        held = tuple(_bracketed_parts(text[1:]))
        return ("(T,)" if held else "()"), held, {}
    name = re.match(r"[\w:]*", text)[0]
    parts = []
    if text.startswith("<", len(name)):
        parts = _bracketed_parts(text[len(name) + 1 :])
    associated = dict(re.split(r"\s*=\s*", p, maxsplit=1) for p in parts if "=" in p)
    return (
        name,
        tuple(p for p in parts if "=" not in p and not p.startswith("'")),
        associated,
    )


@cache
def _php_type_parts(text):
    """``(name, type arguments, named arguments)`` of a PHP type as the
    parser and the standard library's data write it: a class by its full
    name, `::App::User` or `App\\User`, `static` and `this` as `Self`, with
    its type arguments. A type that may also be `null` or `false` is the
    type; any other union has no name. An `iterable` names its second
    argument, what a `foreach` over it gives, `TValue`."""
    alternatives, depth, start = [], 0, 0
    for at, char in enumerate(f"{text}|"):
        depth += (char in "<({") - (char in ">)}")
        if char == "|" and not depth:
            alternatives.append(text[start:at].strip())
            start = at + 1
    alternatives = [a for a in alternatives if a not in ("null", "false")]
    if len(alternatives) != 1:
        return "", (), {}
    name, opened, rest = alternatives[0].partition("<")
    arguments = tuple(_bracketed_parts(rest)) if opened else ()
    name = name.removeprefix("::").replace("::", "\\")
    if name in ("static", "this"):
        name = "Self"
    named = {"TValue": arguments[1]} if name == "iterable" and arguments[1:] else {}
    return name, arguments, named


@cache
def _go_type_parts(text):
    """``(name, type arguments, {})`` of a Go type as written, whatever it
    points at: `*bytes.Buffer` is `bytes.Buffer`. A type Go writes without
    a name has the one the standard library's data gives it, and as its
    arguments what it holds: `[]T` and `[n]T` are `[]` of `int`, the index,
    and `T`; `map[K]V` is `map` of `K` and `V`; `chan T` is `chan` of `T`;
    the results `(A, B)` are `()` of `A` and `B`. A function has no name."""
    # An origin path writes the `.` of a qualified type as `::`.
    text = text.lstrip("*").replace("::", ".")
    if text.startswith("("):
        return "()", tuple(_bracketed_parts(text[1:])), {}
    if text.startswith("["):
        return "[]", ("int", text[text.index("]") + 1 :]), {}
    if text.startswith("map["):
        key = _bracketed_parts(text[4:])[0]
        return "map", (key, text[text.index(key, 4) + len(key) + 1 :]), {}
    if channel := re.match(r"(?:<-\s*)?chan\b(?:\s*<-)?\s*", text):
        return "chan", (text[channel.end() :],), {}
    name = re.match(r"[\w$./-]*", text)[0]
    arguments = ()
    if text.startswith("[", len(name)):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    return name, arguments, {}


def _go_generics(text: str) -> str:
    """``text``, a Go type, with the `[...]` that instantiates a generic
    type written `<...>`, as other languages write it; `[]T`, `[n]T` and
    `map[K]V` hold no type arguments."""
    written, generic = [], []
    for at, char in enumerate(text):
        if char == "[":
            word = re.search(r"[\w.:]*$", text[:at])[0]
            generic.append(bool(word) and word != "map")
            written.append("<" if generic[-1] else char)
        elif char == "]" and generic:
            written.append(">" if generic.pop() else char)
        else:
            written.append(char)
    return "".join(written)


# What a value of each of Java's primitive types is boxed as
_JAVA_BOXES = {
    "boolean": "Boolean",
    "byte": "Byte",
    "char": "Character",
    "short": "Short",
    "int": "Integer",
    "long": "Long",
    "float": "Float",
    "double": "Double",
}


# The class a Java type has as a base without naming it, by the kind of its
# declaration; `java.lang.Object` for any other
_JAVA_ROOTS = {
    "enum": "java.lang.Enum",
    "record": "java.lang.Record",
    "@interface": "java.lang.annotation.Annotation",
}


@cache
def _java_type_parts(text):
    """``(name, type arguments, {})`` of a Java type as written. A wildcard
    is the bound it extends and a primitive its box; an array, which has
    the methods of `Object`, is that class holding its element type."""
    # An origin path writes the `.` of a qualified type as `::`.
    text = re.sub(r"^\?\s*extends\s+", "", text.replace("::", ".").strip())
    if text.endswith(("]", "...")):
        element = text.rpartition("[" if text.endswith("]") else "...")[0]
        return "java.lang.Object", (element,), {}
    name = re.match(r"[\w$.]*", text)[0]
    arguments = ()
    if text.startswith("<", len(name)):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    return _JAVA_BOXES.get(name, name), arguments, {}


def _angle_arguments(text, name):
    """The type arguments in the angle brackets that follow ``name`` and
    close ``text``, or None where none do. `Type<>` written bare takes one
    argument."""
    if text.startswith("<", len(name)) and text.endswith(">"):
        inner = text[len(name) + 1 : -1]
        return tuple(_bracketed_parts(inner)) or (("",) if not inner else ())
    return None


def _dotnet_type_parts(text):
    """``(name, type arguments, {})`` of a C# type as written. The name
    carries the arity its data keys it by: `List` written with one type
    argument is ``List`1``. An array is `System.Array` holding its
    element type, a nullable value type the type it wraps."""
    text = text.replace("::", ".").strip()
    if text.endswith("[]"):
        return "System.Array", (text[:-2].strip(),), {}
    if text.endswith("?") and "[" not in text:
        return _dotnet_type_parts(text[:-1])
    name = re.match(r"[\w$.`]*", text)[0]
    arguments = _angle_arguments(text, name)
    if arguments is None:
        return name, (), {}
    # A name the data keys by its backticked arity carries it already.
    # `System.Array<e>` is the array of `e`, which the data keys without
    # arity.
    if "`" not in name and name != "System.Array":
        name = f"{name}`{len(arguments)}"
    return name, arguments, {}


def _clang_type_parts(text):
    """``(name, type arguments, {})`` of a C or C++ type as written, or
    the standard library's data writes it: `std::vector<int>` is
    `std::vector` with one argument. The name drops the specifiers and
    the pointer/reference markers a declaration wraps it in."""
    text = re.sub(
        r"^(?:const|volatile|static|inline|constexpr|virtual|struct|class|enum)\s+",
        "",
        text.strip(),
    )
    text = re.sub(r"\s*[*&]+\s*$", "", text).strip()
    name = re.match(r"[\w:$.`]+", text)
    name = name[0] if name else ""
    return name, _angle_arguments(text, name) or (), {}


def _ruby_type_parts(text):
    """``(name, type arguments, {})`` of a Ruby type as written, or the
    standard library's data writes it: `Array[String]` is `Array` with one
    argument; `::Name` keeps the `::` that roots it."""
    text = text.strip()
    match = re.match(r"(?:::)?[A-Z_a-z][\w:$.]*", text)
    if not match:
        return "", (), {}
    name = match[0]
    arguments = ()
    if text.startswith("[", len(name)) and text.endswith("]"):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 : -1]))
    return name, arguments, {}


def _dart_type_parts(text):
    """``(name, type arguments, {})`` of a Dart type as written:
    ``List<E>`` is ``List`` with one argument; a nullable is the type it
    wraps."""
    text = text.strip().rstrip("?")
    match = re.match(r"[A-Za-z_][\w$.]*", text)
    if not match:
        return "", (), {}
    name = match[0]
    arguments = ()
    if text.startswith("<", len(name)) and text.endswith(">"):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    return name, arguments, {}


@cache
def _java_packages():
    """The first name of every package the Java standard library's data
    has a type of."""
    return frozenset(key.partition(".")[0] for key in _stdlib.load("java")["types"])


@cache
def _py_type_parts(text):
    """``(name, type arguments, {})`` of a Python type as an annotation or
    the standard library's data writes it, also between quotes and with the
    `::` an origin path has for its dots. `Optional[T]` and a union of `T`
    with `None` or `Any` are `T`; any other union has no name and the types
    it is of as its arguments. A qualifier (`Final[T]`, `Annotated[T, ...]`)
    is the type it holds. A tuple
    has, before the types of its items, the one type they all are, if any:
    `tuple[T, ...]` has that alone."""
    text = text.strip("'\" ").replace("::", ".")
    members, depth, start = [], 0, 0
    for column, char in enumerate(f"{text}|"):
        depth += (char == "[") - (char == "]")
        if char == "|" and not depth:
            members.append(text[start:column].strip())
            start = column + 1
    members = [m for m in members if m.rpartition(".")[2] not in ("None", "Any")]
    if len(members) != 1:
        return "", tuple(members), {}
    name, bracket, rest = members[0].partition("[")
    arguments = tuple(_bracketed_parts(rest)) if bracket else ()
    short = name.rpartition(".")[2]
    if short in ("Optional", "Union"):
        return _py_type_parts(" | ".join(arguments))
    if short in _stdlib.section("python", "qualifiers") and arguments:
        return _py_type_parts(arguments[0])
    if short == "Self":
        return short, (), {}
    if short in ("tuple", "Tuple") and arguments:
        if arguments[1:] in (("...",), ("Ellipsis",)):
            arguments = arguments[:1]
        else:
            shared = arguments[0] if len(set(arguments)) == 1 else ""
            arguments = (shared, *arguments)
    return name, arguments, {}


@cache
def _py_std_names():
    """The dotted names Python's standard-library data has: its types, its
    functions and variables, and the modules and classes they are in."""
    data = _stdlib.load("python")
    return frozenset(
        key.rsplit(".", cut)[0]
        for key in (*data["types"], *(name.lstrip(".") for name in data["functions"]))
        for cut in range(key.count(".") + 1)
    )


# A `|` between the alternatives of a union, outside any bracket
_TS_UNION_RE = re.compile(r"\s*\|\s*(?![^<>\[\](){}]*[>\])}])")


# A string literal type, as written
_TS_STRING_RE = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`[^`]*`")


def _ts_unquoted(text):
    """``text`` with the text of each string literal in it turned to `_`,
    its quotes and columns kept."""
    return _TS_STRING_RE.sub(
        lambda found: found[0][0] + "_" * (len(found[0]) - 2) + found[0][-1], text
    )


@cache
def _ts_given(alias, arguments):
    """The text of the TypeScript type ``alias``, ``(type parameters,
    text)``, stands for with ``arguments`` in place of its parameters."""
    given = dict(zip(alias[0], arguments))
    return re.sub(
        r"(?<![\w$.])[\w$]+(?![\w$])",
        lambda found: given.get(found[0], found[0]),
        alias[1],
    )


def _ts_type_parts(text):
    """``(name, type arguments, {})`` of a TypeScript type as written,
    whether or not it may be undefined or null instead. An array is an
    `Array`, `this` is `Self`, what a type predicate stands for is a
    `boolean` and a string literal type is a `string`. A tuple is `[]` of
    the types it holds. A union of two types, an object and a function type
    have no name."""
    # An origin path writes the `.` of a qualified type as `::`. A string
    # literal's text holds no mark.
    written = text.replace("::", ".").strip()
    marks = list(_TS_UNION_RE.finditer(_ts_unquoted(written)))
    starts = [0] + [mark.end() for mark in marks]
    ends = [mark.start() for mark in marks] + [len(written)]
    options = {written[start:end] for start, end in zip(starts, ends)}
    options -= {"undefined", "null"}
    text = options.pop() if len(options) == 1 else ""
    if text.endswith("[]"):
        return "Array", (text[:-2].removeprefix("readonly "),), {}
    if re.search(r"\bis\s", text):
        return "boolean", (), {}
    if _TS_STRING_RE.fullmatch(text):
        return "string", (), {}
    tuple_ = text.removeprefix("readonly ").strip()
    if tuple_[:1] == "[" and tuple_[-1:] == "]" and tuple_[1:-1].strip():
        return "[]", tuple(_bracketed_parts(tuple_[1:])), {}
    name = re.match(r"[\w$.]*", text)[0]
    arguments = ()
    if text.startswith("<", len(name)) and text.endswith(">"):
        arguments = tuple(_bracketed_parts(text[len(name) + 1 :]))
    elif len(name) < len(text):
        return "", (), {}
    return "Self" if name == "this" else name, arguments, {}


@cache
def _ts_compound(text):
    """``("?", branches)`` of a TypeScript type that is a conditional type,
    ``("&", operands)`` of an intersection and ``("|", alternatives)`` of
    a union — None for any other. A union's `null` and `undefined`
    alternatives are dropped: a value that may be one reads as the rest."""
    depth, marks = 0, []
    for column, char in enumerate(_ts_unquoted(text)):
        if char in "<([{":
            depth += 1
        elif char in ")]}" or (char == ">" and text[column - 1] != "="):
            depth -= 1
        elif (
            not depth
            and char in "?:&|"
            and "::" not in text[max(column - 1, 0) : column + 2]
        ):
            # A `::` is the `.` of a name an origin path writes, which no
            # TypeScript type has.
            marks.append((column, char))
    asked = [column for column, char in marks if char == "?"]
    if asked:
        # Its `:` is the one that no `?` after its own is still open at.
        pending = 0
        for column, char in marks:
            pending += (column >= asked[0]) * ((char == "?") - (char == ":"))
            if char == ":" and column > asked[0] and not pending:
                branches = text[asked[0] + 1 : column], text[column + 1 :]
                nested = [_ts_compound(branch.strip()) for branch in branches]
                return "?", [
                    part
                    for branch, found in zip(branches, nested)
                    for part in (found[1] if found and found[0] == "?" else [branch])
                ]
        return None
    if {char for _, char in marks} == {"&"}:
        cuts = [-1, *(column for column, _ in marks), len(text)]
        return "&", [text[cuts[n] + 1 : cuts[n + 1]] for n in range(len(marks) + 1)]
    if {char for _, char in marks} in ({"|"}, {"|", "&"}):
        cuts = [-1, *(column for column, char in marks if char == "|"), len(text)]
        alternatives = [text[cuts[n] + 1 : cuts[n + 1]] for n in range(len(cuts) - 1)]
        alternatives = [
            one.strip()
            for one in alternatives
            if one.strip() not in ("undefined", "null", "void")
        ]
        if len(alternatives) > 1:
            return "|", alternatives
    return None


def _gathered(params, number):
    """The value of a rest parameter at position ``number`` of a list
    whose types are ``params``: an array of those it gathers, where they
    are of one type."""
    gathered = list(params[number:])
    if gathered and gathered[-1] and gathered[-1][0] == "rest":
        gathered[-1] = gathered[-1][1]
    one = gathered and all(held == gathered[0] for held in gathered)
    return ("outside", "Array", (gathered[0],)) if one else None


# An indexed access type with a literal key: `Type["key"]`
_TS_INDEXED_RE = re.compile(r"([\w$.]+)\s*\[\s*[\"']([\w$]+)[\"']\s*\]")


# A Rust `self` parameter taken by value
_SELF_BY_VALUE_RE = re.compile(r"(?:mut\s+)?self")


# A possibly qualified name in a C/C++ type
_C_TYPE_NAME_RE = re.compile(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")


# What a C/C++ macro that marks a declaration expands to: nothing, an
# attribute, a specifier or the opening of a namespace
_C_MARK_RE = re.compile(
    r"(?:\s*(?:__\w+\s*(?:\(.*\))?|\[\[.*\]\]|alignas\s*\(.*\)|namespace\s+[\w:]*\s*\{"
    r"|extern(?:\s*\"C\")?|static|inline|constexpr|virtual|explicit))*\s*"
)


# Languages whose methods are reachable only through an explicit receiver:
# a bare `name(...)` there never calls a method of the enclosing class.
_EXPLICIT_RECEIVER_LANGS = frozenset({"python", "rust", "go", "typescript"})


# Languages whose parser reports every member call with its receiver, so a
# bare callee is a free call.
_FREE_BARE_CALL_LANGS = frozenset({"python", "go"})


# Languages where `receiver.name(...)` can reach a top-level definition
# through a namespace object. Only an import shows which; without one the
# definition is a second that may be meant, and is not pinned by name.
_NAMESPACE_RECEIVER_LANGS = frozenset({"typescript"})


# Languages where a file that is no module, one that neither imports nor
# re-exports, shares its top-level definitions with every other file.
_SCRIPT_GLOBAL_LANGS = frozenset({"typescript"})


# Languages where a bare call reaches another file's definition only
# through an import of its name.
_IMPORT_SCOPED_LANGS = frozenset({"rust", "typescript"})


# Languages where a bare call is a call on `self`: it reaches a member of the
# enclosing type or of what that inherits, or a top-level definition, never a
# member of an unrelated type.
_IMPLICIT_SELF_LANGS = frozenset({"ruby"})


# Languages where a call on a type from outside reaches that type's own
# methods only, never those of a type argument.
_OWN_METHOD_LANGS = frozenset({"java", "php"})


# Languages that spread one type's members over several declarations.
_SPLIT_TYPE_LANGS = frozenset({"rust", "go", "clang", "dotnet", "ruby", "typescript"})


# Languages whose libraries nothing lists. On a receiver of unknown type a
# name may be a library's, so one project definition of it is not the only
# one that fits; and what a library's call returns may be a project value.
_UNLISTED_LIBRARY_LANGS = frozenset({"php"})


# What a method calls the object or the class it is used on, by language.
_SELF_RECEIVERS = {
    "python": frozenset({"self", "cls"}),
    "rust": frozenset({"self", "Self"}),
    "ruby": frozenset({"self"}),
    "typescript": frozenset({"this"}),
    "java": frozenset({"this"}),
    "dotnet": frozenset({"this"}),
    "clang": frozenset({"this"}),
    "dart": frozenset({"this"}),
}


_EXTERNAL = "external"


# The name of an npm package, scoped or not
_NPM_PACKAGE_RE = re.compile(r"(?:@[a-z0-9~-][a-z0-9._~-]*/)?[a-z0-9~-][a-z0-9._~-]*")


def _closure(edges, starts):
    """``starts`` and what the map ``edges`` leads them to in any number of
    steps."""
    found, pending = set(starts), list(starts)
    while pending:
        for other in edges.get(pending.pop(), ()):
            if other not in found:
                found.add(other)
                pending.append(other)
    return found


def _ancestors(folder):
    """``folder`` and each directory above it, nearest first."""
    while True:
        yield folder
        parent = posixpath.dirname(folder)
        if parent == folder:
            return
        folder = parent


_UNRESOLVED = "unresolved"


# The implicit numeric conversions C# defines between its built-in numeric
# types (C# language specification, implicit numeric conversions), by the
# runtime names the library data keys the types by
_DOTNET_WIDENING = {
    f"System.{source}": {f"System.{target}" for target in targets.split()}
    for source, targets in {
        "SByte": "Int16 Int32 Int64 Single Double Decimal",
        "Byte": "Int16 UInt16 Int32 UInt32 Int64 UInt64 Single Double Decimal",
        "Int16": "Int32 Int64 Single Double Decimal",
        "UInt16": "Int32 UInt32 Int64 UInt64 Single Double Decimal",
        "Int32": "Int64 Single Double Decimal",
        "UInt32": "Int64 UInt64 Single Double Decimal",
        "Int64": "Single Double Decimal",
        "UInt64": "Single Double Decimal",
        "Char": "UInt16 Int32 UInt32 Int64 UInt64 Single Double Decimal",
        "Single": "Double",
    }.items()
}

# What `_find_member` gives for whether its lookup is complete where the
# several types a base may be have the member: a miss that is not one of
# them, but the candidates.
_SEVERAL = "several"


# What a call returns when nothing is known of it, not even that it comes
# from outside the project.
_OPAQUE = "opaque"


# The GOOS and GOARCH values Go's build constraints know
_GOOS, _GOARCH = (
    frozenset(_stdlib.load("go")["platforms"][axis]) for axis in ("os", "arch")
)


def _go_builds_meet(a, b):
    """The platform constraints two Go files hold are compatible unless one
    requires an atom the other rules out, or they require different values
    of the same axis (GOOS or GOARCH)."""
    req_a, excl_a = a
    req_b, excl_b = b
    if req_a & excl_b or excl_a & req_b:
        return False
    for axis in (_GOOS, _GOARCH):
        if req_a & axis and req_b & axis and req_a & axis != req_b & axis:
            return False
    return True


def _flip_sign(sign):
    return ("-" if sign.startswith("+") else "+") + sign[1:]


def _export_target(target, conditions):
    """The path a package's `exports` target gives under ``conditions``,
    as Node's PACKAGE_TARGET_RESOLVE reads it: of an object the first key
    of ``conditions`` whose value gives a result, of an array the first
    entry that does; "" where that is `null`, which exports nothing; None
    for none."""
    if target is None or isinstance(target, str):
        return target or ""
    entries = (
        target
        if isinstance(target, list)
        else (value for key, value in target.items() if key in conditions)
        if isinstance(target, dict)
        else ()
    )
    return next(
        (path for t in entries if (path := _export_target(t, conditions)) is not None),
        None,
    )


def _star_match(pattern: str, text: str) -> str | None:
    """What the `*` of ``pattern`` stands for where ``pattern`` matches
    ``text``: the empty string for a pattern without one, None for no match."""
    prefix, star, suffix = pattern.partition("*")
    if star and text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix) : len(text) - len(suffix)]
    return "" if pattern == text else None


# What a Rust path that names an item of the crate by its module starts with
_RUST_PATH_ROOTS = ("crate::", "self::", "super::")


# The named type argument a base of a type is written with for what
# iterating the type gives
_ITERATED = {"rust": "Item", "php": "TValue"}


# The member of an iterable type that hands out its iterator, and the
# iterator's member that holds the element, as the language's `for` reads
# them (Dart's for-in, C#'s foreach)
_ITERATOR_MEMBERS = {
    "dart": ("iterator", "current"),
    "dotnet": ("GetEnumerator", "Current"),
}


# The object whose members a language's globals are, as a callee starts with it
_GLOBAL_OBJECT = {"typescript": "globalThis."}


# The type every type of a language has as a base that the data does not
# list among its bases
_ROOT = {"java": "java.lang.Object", "python": "builtins.object"}


def _node_module(specifier):
    """Whether the module specifier ``specifier`` names one of Node's own
    modules."""
    return specifier.startswith("node:") or specifier in _stdlib.section(
        "node", "modules"
    )


# An MSBuild `$(...)`: a property, a method of one, or a function, whose
# arguments may hold `$(...)`s, quoted or bare
_MSBUILD_REFERENCE = (
    r"\$\((?:[^()']|'[^']*'|\((?:[^()']|'[^']*'|\$\((?:[^()']|'[^']*')*\))*\))*\)"
)
# The `[MSBuild]::` version comparisons, by what a comparison must give
_MSBUILD_VERSION = {
    "VersionEquals": (0,),
    "VersionNotEquals": (-1, 1),
    "VersionGreaterThan": (1,),
    "VersionGreaterThanOrEquals": (0, 1),
    "VersionLessThan": (-1,),
    "VersionLessThanOrEquals": (-1, 0),
}
# The string methods read on a property, by what each gives
_MSBUILD_STRING = {
    "StartsWith": str.startswith,
    "EndsWith": str.endswith,
    "Contains": str.__contains__,
    "ToLower": lambda held: held.lower(),
    "ToLowerInvariant": lambda held: held.lower(),
    "ToUpper": lambda held: held.upper(),
    "ToUpperInvariant": lambda held: held.upper(),
    "Trim": str.strip,
}


def _msbuild_expand(text, properties, above):
    """``text`` with each `$(...)` what it gives: a property of
    ``properties``, keyed by lower-case name, a string method of one, a
    `[MSBuild]::` version comparison, or the file or directory a
    `[MSBuild]::GetPathOfFileAbove` or `GetDirectoryNameOfFileAbove` finds,
    the directory ``above(start, name)`` gives. ValueError for any other
    function."""

    def given(found):
        inner = found[0][2:-1].strip()
        if re.fullmatch(r"\w+", inner):
            return properties.get(inner.lower(), "")
        method = re.fullmatch(r"(\w+)\.(\w+)\(\s*(?:'([^']*)')?\s*\)", inner)
        if method and method[2] in _MSBUILD_STRING:
            held = properties.get(method[1].lower(), "")
            call = _MSBUILD_STRING[method[2]]
            value = call(held, method[3]) if method[3] is not None else call(held)
            return str(value).lower() if isinstance(value, bool) else value
        version = re.fullmatch(
            r"\[MSBuild\]::(\w+)\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)", inner
        )
        if version and version[1] in _MSBUILD_VERSION:
            a, b = (
                tuple(
                    int(part)
                    for part in _msbuild_expand(v, properties, above).split(".")
                )
                for v in version.group(2, 3)
            )
            size = max(len(a), len(b))
            a, b = (v + (0,) * (size - len(v)) for v in (a, b))
            return str(((a > b) - (a < b)) in _MSBUILD_VERSION[version[1]]).lower()
        # An argument is quoted, or one property written bare
        argument = rf"'[^']*'|{_MSBUILD_REFERENCE}"
        searched = re.fullmatch(
            r"\[MSBuild\]::(GetPathOfFileAbove|GetDirectoryNameOfFileAbove)"
            rf"\(\s*({argument})\s*(?:,\s*({argument})\s*)?\)",
            inner,
        )
        if searched:
            first, second = (
                _msbuild_expand((v or "").strip("'"), properties, above)
                for v in searched.group(2, 3)
            )
            if searched[1] == "GetDirectoryNameOfFileAbove":
                return above(first, second)
            start = second or properties.get("msbuildthisfiledirectory", "")
            directory = above(start, first)
            return directory and f"{directory}/{first}"
        raise ValueError(inner)

    return re.sub(_MSBUILD_REFERENCE, given, text)


def _msbuild_hold(conditions, properties, exists, above):
    """Whether each MSBuild condition of ``conditions`` holds of
    ``properties``: comparisons of values, quoted or not, by `==`, `!=` and
    the numeric `<`, `<=`, `>`, `>=`, `Exists(path)` as ``exists`` answers
    it, joined by `and`, `or`, `!` and parentheses, with what `$(...)`
    gives (see `_msbuild_expand`). ValueError for a function or form it
    does not read."""
    return all(_msbuild_holds(c, properties, exists, above) for c in conditions)


def _msbuild_holds(condition, properties, exists, above):
    """Whether the MSBuild ``condition`` holds (see `_msbuild_hold`)."""
    tokens = [
        *re.findall(
            rf"'[^']*'|{_MSBUILD_REFERENCE}"
            r"|==|!=|<=|>=|<|>|!|\(|\)|[^\s()=!<>']+",
            condition,
        ),
        "",
    ]
    at = 0
    marks = ("", "(", ")", "!", "==", "!=", "<", "<=", ">", ">=")

    def take():
        nonlocal at
        at += 1
        return tokens[at - 1]

    def value(token):
        if token in marks or token.lower() in ("and", "or"):
            raise ValueError(condition)
        written = token[1:-1] if token.startswith("'") else token
        return _msbuild_expand(written, properties, above)

    def either():
        held = both()
        while tokens[at].lower() == "or":
            take()
            held = both() or held
        return held

    def both():
        held = one()
        while tokens[at].lower() == "and":
            take()
            held = one() and held
        return held

    def operand(token):
        """The value a comparison reads: `Exists(path)`'s `true` or
        `false`, else the token's."""
        if token.lower() == "exists" and tokens[at] == "(":
            take()
            path = value(take())
            if take() != ")":
                raise ValueError(condition)
            return str(exists(path)).lower()
        return value(token).lower()

    def one():
        token = take()
        if token == "!":
            return not one()
        if token == "(":
            held = either()
            if take() != ")":
                raise ValueError(condition)
            return held
        left = operand(token)
        if tokens[at] not in marks[4:]:
            return left == "true"
        op, right = take(), operand(take())
        if op in ("==", "!="):
            return (left == right) == (op == "==")
        a, b = float(left), float(right)
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]

    held = either()
    if tokens[at] != "":
        raise ValueError(condition)
    return held


def _framework_kind(framework):
    """The `TargetFrameworkIdentifier` MSBuild gives the target framework
    moniker ``framework``: `net8.0` and `netcoreapp3.1` are .NETCoreApp,
    `netstandard2.0` .NETStandard, `net48` .NETFramework."""
    if framework.startswith("netstandard"):
        return ".NETStandard"
    if framework.startswith("netcoreapp") or re.match(r"net\d+\.", framework):
        return ".NETCoreApp"
    return ".NETFramework" if re.match(r"net\d+$", framework) else ""


class StandardLibrary:
    """One language's standard library as its generated data has it (see
    ``descry._stdlib``): which members a type has, through its bases, and
    what they return.

    An instance of a standard type is the value ``("outside", key,
    arguments)``: its key in the data's `types` and the values of its
    `params` in order, None for one that is not known. Every other value
    (an instance of a project type, None for an unknown one) is the
    caller's, taken as an argument and handed back untouched.

    A language supplies two functions, and in ``_ITERATED`` the named type
    argument its data writes what iterating a type gives under:

    - ``parts(text)`` reads a type as the language writes it, in the data
      and in source, and returns ``(name, arguments, associated)``: the
      name of the type without what is written around it (a reference, a
      pointer), spelled as the data keys it where the language writes the
      type without a name (`[T]`); its type arguments as written, in the
      order of the type's `params`; and its named type arguments (`Item =
      T`) by name, ``{}`` where the language has none. Text it cannot read
      has the name ``""``. In the data a name is a key of `types`, a name
      of `scope`, a parameter of the type the text belongs to or `Self`,
      or `P::Name` for the named type argument that a base of the type a
      parameter `P` stands for is written with.
    - ``key(name, scope_id)`` says what a type or function name written
      in the source of ``scope_id`` is called in the data: through the
      file's imports, through `scope` unless the project declares the
      name, or as it stands. A name the data lacks is no standard type.
    """

    def __init__(self, lang, parts, key):
        self.lang = lang
        self.parts = parts
        self.key = key

    @property
    def data(self):
        return _stdlib.load(self.lang)

    def named(self, written, scope_id):
        """Key in the data of the type ``written`` in the source of
        ``scope_id``; None for a type the data does not have."""
        key = self.key(self.parts(written)[0], scope_id)
        return key if key in self.data["types"] else None

    def value(self, text, bound):
        """Value of the type the data writes as ``text``. ``bound`` holds
        the values of `Self` and of the type parameters ``text`` may
        name."""
        name, arguments, _ = self.parts(text)
        owner, _, associated = name.partition("::")
        if owner in bound:
            if associated:
                return self.associated(bound[owner], associated)
            return bound[owner]
        name = self.data["scope"].get(name) or self.data["scope"].get(
            re.split(r"[.:]", name)[-1], name
        )
        if name not in self.data["types"] and name != "[]":
            return None
        return ("outside", name, tuple(self.value(a, bound) for a in arguments))

    def canonical(self, key):
        """The key of the type ``key`` names, through the paths and aliases
        the data's `same` gives."""
        same, seen = self.data.get("same", {}), set()
        while key in same and key not in seen:
            seen.add(key)
            key = self.data["scope"].get(same[key], same[key])
        return key

    def lineage(self, key, seen=()):
        """``key`` and the keys of the types it derives from in the data."""
        yield key
        for base in self.data["types"].get(key, {}).get("bases", ()):
            name = self.data["scope"].get(self.parts(base)[0], self.parts(base)[0])
            if name not in seen:
                yield from self.lineage(name, (*seen, key))

    def _entry(self, value, receiver):
        """The data's entry for the instance ``value``, and the values of
        its type parameters and of `Self`: ``receiver``, the instance whose
        bases led to ``value``."""
        # A tuple, which the data has no entry for, has no members of its own.
        entry = self.data["types"].get(value[1]) or {
            "params": [],
            "bases": [],
            "methods": {},
            "statics": {},
        }
        return entry, {**dict(zip(entry["params"], value[2])), "Self": receiver}

    def associated(self, value, name, base=None, receiver=None, seen=()):
        """The type that a base of the instance ``value`` is written with
        as its named type argument ``name``: the base ``base``, or any."""
        if not isinstance(value, tuple) or len(value) < 3 or value[1] in seen:
            return None
        entry, bound = self._entry(value, receiver or value)
        for written in entry["bases"]:
            base_name, _, associated = self.parts(written)
            bound_type = associated.get(name, f"Self::{name}")
            named = base in (None, self.data["scope"].get(base_name, base_name))
            # A base that hands on the type's own `Self::Item` sets nothing.
            if named and bound_type != f"Self::{name}":
                return self.value(bound_type, bound)
            inherited = self.value(written, bound)
            found = self.associated(
                inherited, name, base, bound["Self"], seen + (value[1],)
            )
            if found is not None:
                return found
        return None

    def aliased(self, value, name):
        """Value of the member type ``name`` (`value_type`) that the data
        writes for the instance ``value``; None where it writes none, or
        where the instance has more arguments than the data's parameters:
        non-type ones, whose positions the data does not tell."""
        entry, bound = self._entry(value, value)
        spelled = (entry.get("aliases") or {}).get(name)
        fits = len(value[2]) <= len(entry["params"])
        return self.value(spelled, bound) if spelled and fits else None

    def _where(self, value, name, receiver, table, seen=()):
        """``(key, bound)`` of the first type among the instance ``value``
        and its bases, in their order, whose ``table`` has ``name``, with
        the values of its parameters; ``(None, value)`` where none has:
        ``value`` is then the first base that is no standard type (what a
        pointer points at), itself where every base is one."""
        if len(value) < 3 or value[1] in seen:
            return None, value
        entry, bound = self._entry(value, receiver or value)
        if name in entry[table]:
            return value[1], bound
        for base in entry["bases"]:
            inherited = self.value(base, bound)
            if not isinstance(inherited, tuple) or inherited[0] != "outside":
                return None, inherited
            key, found = self._where(
                inherited, name, bound["Self"], table, seen + (value[1],)
            )
            if key or found != inherited:
                return key, found
        return None, value

    def member(self, value, name, receiver=None, table="methods", results=None):
        """Look ``name`` up in ``table``, `methods` or `statics`, of the
        instance ``value`` and of its bases in their order. Returns
        ``(True, the value the member returns)``, or ``(False, value)``
        where the data has no such member: ``value`` is then the first
        base that is no standard type (what a pointer points at), itself
        where every base is one. ``receiver`` is the instance `Self`
        stands for where that is not ``value``: a project type calling a
        member of a standard base. ``results`` gives, by argument
        position, the value that a callable passed to the member returns."""
        key, bound = self._where(value, name, receiver, table)
        if not key and bound == value and self.lang in _ROOT:
            root = ("outside", _ROOT[self.lang], ())
            key, found = self._where(root, name, receiver or value, table)
            bound = found if key else bound
        if not key:
            return False, bound
        callbacks = self.data.get("callbacks", {}).get(key, {}).get(name, {})
        for position, callback in callbacks.items():
            bound[callback["returns"]] = (results or {}).get(position)
        returns = self.data["types"][key][table][name]
        return True, returns and self.value(returns, bound)

    def extended(self, value, name):
        """The namespaces whose extension methods give the instance
        ``value`` its method ``name``, where it has none of its own; None
        for one it has, or where the data has no such member."""
        key, _ = self._where(value, name, None, "methods")
        return self.data.get("extensions", {}).get(key, {}).get(name)

    def argued(self, value, name, types):
        """Value the method ``name`` of the instance ``value`` returns where
        its data's `by_argument` tells it by the type of an argument,
        ``types`` the keys of the arguments' types by position; None where
        it does not."""
        key, bound = self._where(value, name, None, "methods")
        given = self.data.get("by_argument", {}).get(key, {}).get(name, {})
        for position, returns in given.items():
            at = int(position)
            if at < len(types) and types[at] in returns:
                return self.value(returns[types[at]], bound)
        return None

    def callbacks(self, value, name):
        """``{argument position: values of the parameters}`` of each
        callable that the method ``name`` of the instance ``value`` takes."""
        key, bound = self._where(value, name, None, "methods")
        callbacks = self.data.get("callbacks", {}).get(key, {}).get(name, {})
        return {
            position: [self.value(written, bound) for written in callback["params"]]
            for position, callback in callbacks.items()
        }

    def variant(self, name, passed):
        """The instance that the variant the scope names ``name`` makes of
        the values ``passed`` it in order: of its enum, each type parameter
        what the variant holds in its place. None for no such variant."""
        owner, _, variant = self.data["scope"].get(name, "").rpartition("::")
        fields = self.data.get("variants", {}).get(owner, {}).get(variant)
        if not isinstance(fields, list):
            return None
        held = dict(zip(fields, passed))
        params = self.data["types"][owner]["params"]
        return "outside", owner, tuple(held.get(param) for param in params)

    def held(self, value, variant, field):
        """Value that the variant ``variant`` of the instance ``value``
        holds at ``field``, a position or a name."""
        entry, bound = self._entry(value, value)
        fields = self.data.get("variants", {}).get(value[1], {}).get(variant)
        if fields is None:
            # An alias has the variants of what it stands for.
            for base in entry["bases"]:
                inherited = self.value(base, bound)
                if (
                    isinstance(inherited, tuple)
                    and len(inherited) > 2
                    and inherited[1] != value[1]
                ):
                    found = self.held(inherited, variant, field)
                    if found is not None:
                        return found
            return None
        named = field in fields if isinstance(fields, dict) else field < len(fields)
        return self.value(fields[field], bound) if named else None

    def function(self, name, scope_id):
        """``(True, the value returned)`` where ``name``, written in the
        source of ``scope_id``, is a function of the data."""
        functions = self.data["functions"]
        key = self.key(name, scope_id)
        return key in functions, functions.get(key) and self.value(functions[key], {})


class _ClangLibrary(StandardLibrary):
    """The C and C++ standard libraries read together: `c.json` names the
    C functions, `cpp.json` the types and functions of `std`."""

    def __init__(self, parts, key):
        super().__init__("clang", parts, key)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            out = {"types": {}, "functions": {}, "scope": {}}
            for lang in ("c", "cpp"):
                data = _stdlib.load(lang)
                for kind in ("types", "functions", "scope"):
                    out[kind] |= data.get(kind, {})
            self._data = out
        return self._data


class CallResolver:
    """Decides which project definition, if any, a call site reaches.

    ``resolve`` returns ``(node_id, source)`` for a call pinned to one
    definition, ``(None, "external")`` for a call that cannot reach a project
    definition (no project symbol carries its name, its receiver comes from
    outside the project, or it calls a value held in a variable), and
    ``(None, "unresolved")`` for a call that may reach one of several
    project definitions.
    """

    # What the resolution of a call reads and leaves for ``result`` and
    # ``keep``, which a call resolved within it leaves as it was
    _CALL_STATE = (
        "_result",
        "_receiver",
        "_deferred",
        "_shape_returns",
        "_callables",
        "_this_args",
        "_candidates_now",
        "_name_only",
        "_arguments",
        "_path_call",
        "_passed",
        "_borrows",
        "_callee",
        "_receiver_written",
        "_on",
        "_lambdas",
        "_typed_call",
        "_caller",
        "_looked_up",
        "_line",
        "_in_block",
        "_bound",
        "call_guards",
    )

    # What resolving calls fills in as it goes, kept for one file at a time:
    # all a file's resolution reads is then read while it is resolved.
    _FILE_MEMOS = (
        "alias_blocks",  # see _aliases_named
        "dotnet_owners",  # C# class -> it and the types it derives from
        "base_types",  # (Python class ID, origin path of a base) -> see _type_named
        "blanket_members",  # see _blanket_member
        "body_values",  # (function, type it is called on) -> its body's value
        "_dart_found",  # (library, keyword) -> `_dart_directives`
        "_guards",  # (C/C++ function, caller's file) -> its guards
        "java_fields",  # (class ID, name) -> the field read by it there
        "java_scoped",  # (scope, type name) -> the type it names there
        "loaded",  # Ruby file ID -> the files it loads
        "_macro_events_of",  # see _macro_events
        # (function, arguments passed, type arguments written) -> the
        # signature a call of it is of
        "overloads",
        "reaches",  # file ID -> the files its includes reach (C/C++)
        "rust_in_scope",  # Rust scope ID -> the modules named there
        "rust_uses",  # Rust scope ID -> what the `use`s in force bind
        "shared_declaration",  # candidate node IDs -> node ID or None
        # Language -> the last name segments of the types and functions its
        # standard library's data has, and the names of their members
        "_standard_names",
        "units",  # see _sees
        "_usings_of",  # see `_project_usings`
        "variables",  # (Python module or class ID, name) -> that variable's value
    )

    # What resolving a call keeps for later calls of its own file, or for
    # itself; no other file's resolution reads it
    UNRECORDED = (
        *_FILE_MEMOS,
        "returned",
        "callables",
        "declared_params",
        "this_args",
        "deferred",
        "late",
        "_callables",
        "_this_args",
        "_bounding",
        "_reading",
        "_holding",
        "_ahead_of",
        "_forwarding",
        "open_aliases",
        "standard",
    )

    def begin_file(self):
        """Start resolving the calls of another file."""
        for name in self._FILE_MEMOS:
            setattr(self, name, {})

    def _edge(self, number):
        """The index of the edge an origin path names by ``number``."""
        return self.builder.edge_index.get(int(number), int(number))

    @contextlib.contextmanager
    def _kept(self):
        """Leave the state of the call being resolved as it was, whatever
        is resolved inside."""
        outer = [getattr(self, name) for name in self._CALL_STATE]
        try:
            yield
        finally:
            for name, held in zip(self._CALL_STATE, outer):
                setattr(self, name, held)

    def __init__(self, builder):
        self.begin_file()
        self.builder = builder
        self.lang = cache(builder._get_language)
        self.nodes = {node["id"]: node for node in builder.nodes}
        # npm package name -> where the walk met it; the Cargo crates' folders
        self.package_order = {name: at for at, name in enumerate(builder.js_packages)}
        self.crate_folders = dict.fromkeys(f for f in builder.rust_crates.values() if f)
        self.by_name = {}  # (language, name) -> node IDs
        self.members = {}  # parent ID -> {name: node ID}
        # ID of a value -> ID of the interface declared under its name
        self.companions = {}
        self.renamed_types = {}  # (language, declared name) -> those interfaces
        self.c_spaces = set()  # names of the C++ namespaces that macros open
        # C++ template name -> {the name of each of its specializations: the
        # parameters its arguments deduce}
        self.c_specializations = {}
        self._bounding = set()  # type parameters whose bounds are being read
        self._reading = set()  # (class, attribute) whose values are being read
        self._holding = set()  # constants whose held functions are being read
        self._ahead_of = set()  # calls being resolved ahead of their turn
        for node_id, node in self.nodes.items():
            parent, name = _member_split(node_id, node["metadata"].get("name", ""))
            self.c_spaces.update(node["metadata"].get("opens", ()))
            if parent:
                lang = self.lang(node_id)
                # A JavaScript member given again, `name#2`, is one of its name.
                again = lang in ("typescript", "rust")
                written = re.sub(r"#\d+$", "", name) if again else name
                self.by_name.setdefault((lang, written), []).append(node_id)
                self.members.setdefault(parent, {})[name] = node_id
                declared = self._name(node_id)
                special = name.startswith(f"{declared}<")
                if special and node["type"] == "Class":
                    self.c_specializations.setdefault(declared, {})[name] = node[
                        "metadata"
                    ].get("deduced", ())
                if declared != name and node["type"] == "Class" and not special:
                    # A type is looked up by the name it was declared under;
                    # a C++ specialization by the arguments it is named with.
                    self.renamed_types.setdefault((lang, declared), []).append(node_id)
                    self.companions[f"{parent}::{declared}"] = node_id
                if declared != name and node["metadata"].get("kind") == "macro":
                    # A C macro filed apart from a function of its name
                    # expands calls of its name.
                    self.by_name.setdefault((lang, declared), []).append(node_id)
                if node["metadata"].get("kind") == "prototype":
                    # A function's prototype is what `new` makes an instance of.
                    function = name.removesuffix(".prototype")
                    self.companions[f"{parent}::{function}"] = node_id
        self.macro_functions = {
            f
            for defined in builder.rust_macros.values()
            for _, fs in defined
            for f in fs
        }
        for owner, called in builder.rust_macro_uses.items():
            # The functions of the `macro_rules!` a Rust impl block calls are
            # its methods: the one of its file, else the one of its crate.
            unit, file_id = self._unit(owner, "rust"), owner.split("::")[0]
            for name in called:
                defined = builder.rust_macros.get(name, ())
                written = [f for at, f in defined if at == file_id] or [
                    f for at, f in defined if self._unit(at, "rust") == unit
                ]
                for function in written[0] if len(written) == 1 else ():
                    held = self.members.setdefault(owner, {})
                    held.setdefault(self.nodes[function]["metadata"]["name"], function)
        # Object-like C/C++ macro -> what every definition of it expands
        # to, None where they disagree; and the names scopes are written by
        self.c_macros, self.c_scopes = {}, set(self.c_spaces)
        for node_id, node in self.nodes.items():
            if self.lang(node_id) != "clang":
                continue
            self.c_scopes.update(node_id.split("::")[1:-1])
            if "expands" in node["metadata"]:
                held = node["metadata"]["expands"]
                name = self._name(node_id)
                self.c_macros[name] = (
                    held if self.c_macros.get(name, held) == held else None
                )
            elif node["metadata"].get("kind") == "namespace":
                self.c_scopes.add(self._name(node_id))
        for node_id, node in self.nodes.items():
            # A return type is read as its macros expand.
            head, sep, rest = (node["metadata"].get("return_type") or "").partition(
                "::"
            )
            if sep and self.lang(node_id) == "clang":
                unmacroed = self._c_unmacroed([head])
                if unmacroed != [head]:
                    node["metadata"]["return_type"] = "::".join([*unmacroed, rest])
        # Name -> the C# extension methods declared under it
        self.extensions = {}
        # The namespaces the C# code declares, each with its ancestors, and
        # the dotted path of each of its types, which a `using` resolves
        # its name against
        self.dotnet_spaces, self.dotnet_paths = set(), set()
        for node_id, node in self.nodes.items():
            if "extends" in node["metadata"]:
                self.extensions.setdefault(self._name(node_id), []).append(node_id)
            if self.lang(node_id) != "dotnet":
                continue
            space = node["metadata"].get("namespace") or node["metadata"].get(
                "dotnet_namespace"
            )
            if space:
                parts = space.split(".")
                self.dotnet_spaces.update(
                    ".".join(parts[:size]) for size in range(1, len(parts) + 1)
                )
            if node["type"] == "Class":
                self.dotnet_paths.add(self._dotnet_path(node_id))
        # class ID -> base type names, in Python the origin path of each; a
        # Ruby class's included modules, the last first, before its superclass
        self.bases = {}
        # (Ruby class ID, `prepend` or `extend`) -> those modules, the last
        # first; (Dart mixin ID, `on`) -> the types it is applied on
        self.mixins = {}
        # index of an INHERITS edge -> the base as written, qualifier and all
        self.base_names = {}
        # class ID -> its bases as written, with their type arguments
        self.written_bases = {}
        self.requires = {}  # Ruby file ID -> the paths it requires
        for index, edge in enumerate(builder.edges):
            if edge["relation"] == "IMPORTS" and self.lang(edge["source"]) == "ruby":
                path = edge["target"].removeprefix("MODULE:")
                self.requires.setdefault(edge["source"], []).append(path)
            if edge["relation"] == "INHERITS":
                # A base the parser gave as a node is that node.
                base = written = edge["target"].removeprefix("REF:")
                self.written_bases.setdefault(edge["source"], []).append(written)
                # A PHP base is written by its full name, which names one
                # type; a Ruby constant is looked up by its whole path.
                lang = self.lang(edge["source"])
                if lang not in ("php", "ruby") and base not in self.nodes:
                    qualifier, base = _split_callee(base)
                    written = ".".join([*qualifier, base])
                origin = edge.get("metadata", {}).get("origin", "")
                if lang == "python" or (lang == "rust" and "::" in origin):
                    # The base as written, its path with it
                    base = written = origin or base
                elif origin.startswith("="):
                    # A type parameter names no type.
                    base = written = ""
                # A base is looked up as written, its qualifier with it.
                mixin = edge.get("metadata", {}).get("mixin")
                if mixin in ("prepend", "extend", "on"):
                    self.mixins.setdefault((edge["source"], mixin), []).insert(
                        0, written
                    )
                elif mixin:
                    self.bases.setdefault(edge["source"], []).insert(0, written)
                else:
                    self.bases.setdefault(edge["source"], []).append(written)
                self.base_names[index] = written
        # A Go struct has the methods of the types it embeds, listed under
        # None among its fields.
        for type_id, fields in builder.fields.items():
            if None in fields:
                self.bases[type_id] = fields[None]
        for node_id, node in self.nodes.items():
            if node["type"] == "Class" and self.lang(node_id) == "java":
                root = _JAVA_ROOTS.get(node["metadata"].get("kind"), _ROOT["java"])
                self.bases.setdefault(node_id, []).append(root)
            elif node["type"] == "Class" and self.lang(node_id) == "python":
                root = f"@{_ROOT['python']}"
                self.bases.setdefault(node_id, []).append(root)
        self.containers = {}  # (language, module or package name) -> file IDs
        self._proto = None  # see `_proto_names`
        self.disk = Probe(self._disk)
        self._impl_reading = set()  # (impl block, type) being read
        self.go_packages = {}  # directory -> Go file IDs
        self.go_modules = sorted(  # (go.mod directory, module path), longest first
            builder.go_modules.items(), key=lambda item: -len(item[1])
        )
        self.java_types = {}  # dotted class name -> class ID
        self.c_files = {}  # path a C/C++ file can be included by -> file IDs
        # dotted module name -> {directory its topmost package is in: file ID}
        self.py_modules = {}
        self.py_root_of = {}  # Python file ID -> that directory
        self.py_module_of = {}  # file ID -> dotted module name
        files = {nid for nid, node in self.nodes.items() if node["type"] == "File"}
        py_files = []
        for file_id in sorted(files):
            parts = file_id[5:].rsplit(".", 1)[0].split("/")
            lang = self.lang(file_id)
            if lang == "python":
                top = len(parts) - 1
                while top and f"FILE:{'/'.join(parts[:top])}/__init__.py" in files:
                    top -= 1
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts[top:]:
                    # A directory without `__init__.py` below a package is
                    # a namespace package of it.
                    outer = next(
                        (
                            at - 1
                            for at in range(1, top)
                            if f"FILE:{'/'.join(parts[:at])}/__init__.py" in files
                        ),
                        top,
                    )
                    self.py_module_of[file_id] = ".".join(parts[outer:])
                    py_files.append((top, parts, file_id))
            elif lang == "rust":
                if parts[-1] in ("lib", "main") and parts[-2:-1] == ["src"]:
                    parts = parts[:-2]
                elif parts[-1] == "mod":
                    parts = parts[:-1]
                if parts:
                    key = (lang, parts[-1].replace("-", "_"))
                    self.containers.setdefault(key, []).append(file_id)
            elif lang == "go":
                self.go_packages.setdefault("/".join(parts[:-1]), []).append(file_id)
                if len(parts) > 1:
                    self.containers.setdefault((lang, parts[-2]), []).append(file_id)
            elif lang == "clang":
                # A file is included by any suffix of its path.
                path = file_id[5:].split("/")
                for start in range(len(path)):
                    self.c_files.setdefault("/".join(path[start:]), []).append(file_id)
            elif lang == "java" and f"{file_id}::{parts[-1]}" in self.nodes:
                # A public class is importable by any dotted suffix of its path.
                for start in range(len(parts)):
                    self.java_types.setdefault(
                        ".".join(parts[start:]), f"{file_id}::{parts[-1]}"
                    )
        # The names of the Rust modules: inline ones, and files as their
        # path names them (`a/b.rs`, `a/b/mod.rs`: `b`)
        self.rust_module_names = {
            posixpath.basename(posixpath.splitext(path)[0].removesuffix("/mod"))
            if "::" not in node_id
            else node_id.rpartition("::")[2]
            for node_id, node in self.nodes.items()
            if self.lang(node_id) == "rust"
            and ("::" not in node_id or node["metadata"].get("kind") == "module")
            for path in [node_id[5:].partition("::")[0]]
        }
        self.rust_files, self.rust_parents, self.rust_roots = self._rust_tree(
            builder.rust_mods, files
        )
        self._importing = set()  # (scope ID, name) of the Rust `use`s being read
        # A module is importable by its path below its topmost package and,
        # through namespace packages, by any longer path no other module owns.
        for top, parts, file_id in py_files:
            root = "/".join(parts[:top])
            self.py_root_of[file_id] = root
            self.py_modules.setdefault(".".join(parts[top:]), {}).setdefault(
                root, file_id
            )
        owned = set(self.py_modules)
        for top, parts, file_id in py_files:
            for start in range(top):
                if (name := ".".join(parts[start:])) not in owned:
                    held = self.py_modules.setdefault(name, {})
                    held.setdefault("/".join(parts[:start]), file_id)
        self.py_roots = {module.split(".")[0] for module in self.py_modules}
        # C/C++ file ID -> IDs of the project files it includes and of those
        # that include it; the line of each `#include` and the files it names
        self.includes, self.includers, self.include_lines = {}, {}, {}
        for edge in builder.edges:
            source = edge["source"]
            if edge["relation"] == "IMPORTS" and self.lang(source) == "clang":
                path = posixpath.normpath(edge["target"][len("MODULE:") :])
                beside = posixpath.join(posixpath.dirname(source[5:]), path)
                beside = f"FILE:{posixpath.normpath(beside)}"
                # Else one written from a directory of the include path, which
                # it may climb out of
                held = re.sub(r"^(?:\.\./)+", "", path)
                found = [beside] if beside in self.nodes else self.c_files.get(held, [])
                self.includes.setdefault(source, set()).update(found)
                at = edge["metadata"]["lineno"]
                self.include_lines.setdefault(source, []).append((at, found))
                for included in found:
                    self.includers.setdefault(included, set()).add(source)
        # Language -> its standard library, for those whose data is read
        self.standard = {
            "rust": StandardLibrary("rust", _rust_type_parts, self._rust_std_key),
            "go": StandardLibrary("go", _go_type_parts, self._go_std_key),
            "typescript": StandardLibrary(
                "typescript", _ts_type_parts, self._ts_std_key
            ),
            "java": StandardLibrary("java", _java_type_parts, self._java_std_key),
            "python": StandardLibrary("python", _py_type_parts, self._py_std_key),
            "php": StandardLibrary("php", _php_type_parts, self._php_std_key),
            "dotnet": StandardLibrary(
                "dotnet", _dotnet_type_parts, self._dotnet_std_key
            ),
            "ruby": StandardLibrary("ruby", _ruby_type_parts, self._ruby_std_key),
            "dart": StandardLibrary("dart", _dart_type_parts, self._dart_std_key),
            "clang": _ClangLibrary(_clang_type_parts, self._clang_std_key),
        }
        # Types from outside that the project implements traits for, by
        # name; under "" the blanket impls, which are for a type parameter.
        self.impl_types = {}
        for node_id, node in self.nodes.items():
            name = self._name(node_id)
            if node["metadata"].get("kind") == "impl" and not self._declares_type(name):
                blanket = "bounds" in node["metadata"]
                self.impl_types.setdefault("" if blanket else name, []).append(node_id)
            if node["metadata"].get("kind") == "extension":
                # A Dart extension gives its members to the type it is on.
                for base in self.bases.get(node_id, ()):
                    self.impl_types.setdefault(_split_callee(base)[1], []).append(
                        node_id
                    )
        self.dart_parts = {}  # Dart file ID -> the library file it is a part of
        for file_id, directives in builder.dart_directives.items():
            for keyword, uris, *_ in directives:
                for uri in uris if keyword == "part" else ():
                    if part := self._dart_file(uri, file_id):
                        self.dart_parts[part] = file_id
        # The names of Rust type aliases, and those being read
        self.alias_names = {name for _, name in builder.rust_alias_types}
        self.open_aliases = set()
        self.returned = {}  # edge index of a call -> the value it returns
        # edge index of a call -> {argument position: values of the
        # parameters of the callable passed there}
        self.callables = {}
        # edge index -> (callee, its parameters as written)
        self.declared_params = {}
        self._callables = {}  # those of the call being resolved
        self.this_args = {}  # call index -> {argument: `ThisType` value}
        self._this_args = {}  # those of the call being resolved
        # The function whose call is being kept, the value of each of its
        # type parameters there and the signature the call is of
        self._bound = None, {}, None
        # edge index of a call -> (scope, standard type, method, {argument
        # position: origin of what the callable passed there returns}),
        # until those callables' bodies are resolved
        self.deferred = {}
        self._deferred = None
        # edge index of a call -> (function, call site, scope, receiver),
        # until the calls it is passed the results of are resolved
        self.late = {}
        self._result = None  # value of the call being resolved, where known
        self._shape_returns = None  # what a method an object type declares returns
        self._receiver = None  # type the method being resolved is called on
        # what its parser counted it passes: how many, for Ruby `[positional
        # arguments, whether keywords follow]`
        self._arguments = None
        # The call: its callee, whether written as a path, what it passes and
        # how, as `site` records it, and for C# what the variable it is made
        # on holds
        self._passed = self._on = self._borrows = self._callee = None
        self._path_call = False
        self._receiver_written = None
        # by argument position, how many parameters each C# or Java lambda it
        # passes declares, None for an argument that is no lambda
        self._lambdas = None
        self._typed_call = False  # whether it writes type arguments, `name<>`
        self._caller = None  # the definition that makes it
        # the line the call being resolved is written on, where told
        self._line = None
        self._in_block = False
        # (type, name) of the member lookup `_find_member` last started
        self._looked_up = None
        # Signed `#if` atoms the line of the call being resolved is under
        self.call_guards = None
        # the definitions the call being resolved may reach, where several fit
        self._candidates_now = None
        # what a name-only resolution returns is not to be trusted
        self._name_only = False
        self._forwarding = set()  # methods whose returned call is being followed
        self._build_constants()
        # A constant assigned a maker call whose name is a project constant
        # holds what that constant's `new` returns, and is no type.
        self.unmade = {n for n in self.nodes if self._unmade(n)}
        for node_id in self.unmade:
            node = self.nodes[node_id]
            if node["type"] == "Class" and not node_id.endswith(">"):
                node["type"] = "Constant"
                node["metadata"].pop("kind")
                node["metadata"]["value_type"] = node["metadata"]["maker"]
        # A Ruby block that defines methods has what the type around it
        # has. One run in a class a constant names has what that class has
        # instead, and the class what the block defines.
        for scope_id, node in self.nodes.items():
            if node["metadata"].get("kind") != "block":
                continue
            around = self._enclosing_class(scope_id.rpartition("::")[0])
            opened = node["metadata"].get("reopens")
            if opened:
                self.bases[scope_id] = [opened]
                reopened = self._constant(opened.split("::"), scope_id)
                if reopened not in (None, _EXTERNAL):
                    path = scope_id.partition("::")[2]
                    self.bases.setdefault(reopened, []).append(path)
            elif around:
                self.bases[scope_id] = [around.partition("::")[2]]
        # Indexes over every file, built before any file is resolved
        self._build_heirs()
        self._build_alias_targets()
        # The Rust impl blocks of each type the project declares, written
        # by a path or under a name a `use` binds, in any crate, and of each
        # standard type by its key in the data
        self.impls_for, self.impl_of = {}, {}  # and the type each block is for
        for block, node in self.nodes.items():
            if node["metadata"].get("kind") != "impl" or self.lang(block) != "rust":
                continue
            named = {
                self._type_named(path, module)
                for module, path in node["metadata"].get("impl_paths", ())
            }
            named = [
                t
                for t in named
                if isinstance(t, str)
                and t in self.nodes
                and self.nodes[t]["metadata"].get("kind") != "impl"
            ]
            if len(named) == 1:
                self.impl_of[block] = named[0]
            key = self.impl_of.get(block) or self._standard_type(
                self._name(block), block
            )
            self.impls_for.setdefault(key, []).append(block)
        # What `_type_named` leaves of the paths read is no call's.
        self._candidates_now = None

    def _name(self, node_id):
        """The name ``node_id`` was declared under, which its ID may prefix."""
        declared = self.nodes[node_id]["metadata"].get("name")
        return declared or node_id.rpartition("::")[2]

    def keep(self, index, target):
        """Remember what the call at edge ``index``, just resolved to
        ``target`` if it was, returns and passes to its callables."""
        edge = self.builder.edges[index]
        site = edge.get("metadata", {}).pop("site", {})
        self._bound = None, {}, None
        name = edge["target"][4:]
        if (
            target is None
            and self.lang(edge["source"]) == "rust"
            and "origin" not in edge["metadata"]
            and name not in self._uses(edge["source"])
        ):
            # A variant of the prelude makes its enum of what it is passed.
            passed = [
                self._plain_value(a, edge["source"]) if isinstance(a, str) else None
                for a in site.get("arguments", ())
            ]
            self._result = self.standard["rust"].variant(name, passed) or self._result
        if target in self.builder.signatures:
            self._bind_call(target, site, edge["source"])
            self.declared_params[index] = target, self._bound[2]["params"]
            self._callables = {
                position: [
                    self._rest(written[3:], target, self._receiver)
                    if written.startswith("...")
                    else self._returned_type(written, target, self._receiver)
                    for written in types
                ]
                for position, types in self._bound[2]["callbacks"].items()
            }
            for position, parameter in enumerate(self._bound[2]["params"]):
                if found := re.search(r"ThisType<([^<>]*)>", parameter):
                    written = re.sub(r"::", ".", found[1].strip())
                    self._this_args[str(position)] = self._returned_type(
                        written, target, self._receiver
                    )
            passed = [a for a in site.get("arguments") or () if isinstance(a, str)]
            calls = (re.match(r"#(\d+)", origin) for origin in passed)
            if any(call and self._edge(call[1]) > index for call in calls):
                self.late[index] = (target, site, edge["source"], self._receiver)
        elif target and self.lang(edge["source"]) == "rust":
            # A closure an argument takes has the parameters of the `Fn`
            # trait the parameter is declared with.
            declared = self.nodes[target]["metadata"].get("param_types") or ()
            declared = [t for t in declared if not t.endswith("self")]
            self._callables = {
                str(position): [
                    None if p == "_" else self._returned_type(p, target) for p in params
                ]
                for position, written in enumerate(declared)
                if (found := _RUST_FN_TYPE_RE.search(written))
                and (params := _bracketed_parts(found[1] + ")"))
            }
        elif target and self.lang(edge["source"]) == "java":
            # A lambda an argument position of the method is typed by the
            # functional interface that position declares.
            declared = self.nodes[target]["metadata"].get("params") or ()
            self._callables = {
                str(position): list(params)
                for position, params in (
                    (position, self._functional_params(written, edge["source"]))
                    for position, written in enumerate(declared)
                )
                if params
            }
        self.returned[index] = self.result(target)
        held = site.get("called") and self._returned_function(self.returned[index])
        if held:
            # A call of what the call returns calls the function it returns.
            metadata = {
                "lineno": edge["metadata"]["lineno"],
                "resolution_source": "regex",
            }
            self.builder.add_edge(edge["source"], held, "CALLS_RESOLVED", metadata)
        if site.get("types") and target and self._is_class(target):
            # A generic type called with its type arguments makes a value
            # that holds them.
            made = self._returned_type(f"{self._name(target)}{site['types']}", target)
            self.returned[index] = made or self.returned[index]
        elif target and self._is_class(target) and self.lang(target) == "rust":
            # A tuple struct called makes a value of the type arguments its
            # arguments are of, `Pair(Own)` a `Pair<Own>`.
            passed = dict(enumerate(self._passed or ()))
            given = self._field_arguments(target, passed, self._caller)
            if any(given):
                self.returned[index] = ("self", target, given)
        self.callables[index] = self._callables
        if self._this_args:
            self.this_args[index] = self._this_args
        if target is None and self._candidates_now:
            # Unresolved: the call reaches one of these.
            edge["metadata"]["candidates"] = self._candidates_now
        if self._deferred:
            self.deferred[index] = self._deferred

    def _bind_call(self, target, site, source_id):
        """Make the call ``site`` of the function ``target``, written in
        ``source_id``, the one whose signature and type arguments are read."""
        self._bound = None, {}, None
        signature = self._overload(target, site)
        bound = self._type_arguments(signature, target, site, source_id)
        self._bound = target, bound, signature

    def _overload(self, function_id, site):
        """The signature of ``function_id`` that the call ``site`` is of.
        Of its overloads, those that take as many arguments, where the
        number passed is known, and as many type arguments as the call is
        written with: the first, where nothing but the types of the
        arguments could choose another and all take the same types, or
        any where all return one known type. With none to choose, one
        that has no parameters and returns nothing."""
        declared = self.builder.signatures[function_id]
        # The signature of the body is not one to call where others are.
        fitting = [one for one in declared if not one["body"]] or declared
        if len(fitting) == 1:
            return fitting[0]
        arguments = site.get("arguments")
        passed = None if arguments is None else len(arguments)
        key = function_id, passed, site.get("types")
        if key in self.overloads:
            return self.overloads[key]
        if arguments is not None:
            counted = []
            for one in fitting:
                params = one["params"]
                needed = [
                    param
                    for param in params
                    if not re.match(r"(?:\.\.\.|[\w$]+\?)|.*=(?!>)", param)
                ]
                rest = any(param.startswith("...") for param in params)
                most = len(arguments) if rest else len(params)
                if len(needed) <= len(arguments) <= most:
                    counted.append(one)
            fitting = counted or fitting
        written = len(_bracketed_parts(site.get("types", "<")[1:]))
        fitting = [one for one in fitting if len(one["generics"]) >= written] or fitting
        taken = {
            tuple(param.partition(":")[2].strip() for param in one["params"])
            for one in fitting
        }
        returned = [self._returned_type(one["returns"], function_id) for one in fitting]
        agreed = returned[0] is not None and returned.count(returned[0]) == len(fitting)
        chosen = {"generics": {}, "params": [], "returns": "", "callbacks": {}}
        types = [value and value[:2] for value in returned]
        if not agreed and types[0] and types.count(types[0]) == len(fitting):
            # One type with arguments that differ: that type, its arguments
            # not known.
            chosen = {**fitting[0], "returns": _strip_generics(fitting[0]["returns"])}
        self.overloads[key] = fitting[0] if len(taken) == 1 or agreed else chosen
        return self.overloads[key]

    def _plain_value(self, origin, source_id):
        """Value of an origin path that names no member: what a call
        returned, a parameter, or an instance of a declared type."""
        value, steps, _ = self._root_value(origin, source_id, set())
        for step in steps:
            if step != "()" or not isinstance(value, tuple):
                return None
            if value[0] in ("def", "cls"):
                value = self._called(value)
        return value

    def _type_arguments(self, signature, target, site, source_id):
        """Value of each type parameter of ``signature``, one of the
        function ``target``, in the call ``site`` written in ``source_id``:
        of the type argument the call is written with, else of the
        argument passed for a parameter declared as it, else of the one
        that the type the call's result is declared as has in its place in
        the declared return type, else, where no argument is passed for it,
        of its bound."""
        names = list(signature["generics"])
        if not names:
            return {}
        written = dict(zip(names, _bracketed_parts(site.get("types", "<")[1:])))
        values = {}
        for passed, param in zip(site.get("arguments") or (), signature["params"]):
            taken = re.fullmatch(r"[\w$]+\??\s*:\s*([\w$]+)", param)
            # What the argument's type gives, not the parameters of a
            # function it is declared as
            typed = re.sub(r"\([^()]*\)\s*=>", "", param.partition(":")[2])
            for name in () if written else names:
                if re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", typed):
                    # The argument gives it: one not written as a name, or
                    # of a type it is only part of, leaves it unknown.
                    plain = taken and taken[1] == name and isinstance(passed, str)
                    held = plain and self._plain_value(passed, source_id)
                    values.setdefault(name, held or None)
        if not written and "declared" in site:
            name, arguments, _ = _ts_type_parts(signature["returns"])
            declared = site["declared"]
            returned = self._returned_type(signature["returns"], target)
            given = self._returned_type(declared, source_id)
            if name in names:
                written = {name: declared}
            elif (
                isinstance(returned, tuple)
                and isinstance(given, tuple)
                and returned[:2] == given[:2]
                and (returned[0] == "self" or len(returned) == 3)
            ):
                written = dict(zip(arguments, _ts_type_parts(declared)[1]))
        for name, bound in signature["generics"].items():
            given = (written[name], source_id) if name in written else (bound, target)
            if name in written or name not in values:
                values[name] = self._returned_type(*given) if given[0] else None
        return values

    def resolve(
        self,
        callee,
        source_id,
        origin=None,
        arguments=None,
        within=False,
        lambdas=None,
        passed=None,
        borrows=None,
        receiver=None,
        on=None,
        line=None,
        in_block=False,
        shadowed=False,
        throws=False,
    ):
        """``arguments`` is what the call passes, where the parser counted
        it, ``lambdas`` its C# or Java lambdas' parameter counts, ``passed``
        the origins of a Rust call's arguments, ``borrows`` whether each is
        written as a reference and ``receiver`` whether its receiver is.
        ``within``
        resolves a call, at no `#if` arm of its own, while another is being
        resolved: it gives the value the call returns in place of the
        verdict and leaves the state of the other as it was. ``line`` is
        where the call is written, ``(line, column)``, where its parser
        tells it; ``in_block`` whether a Ruby call on an implicit `self` is
        in a block, which may be run with another `self`; ``shadowed``
        whether a local of a block or function around a JavaScript call
        binds the callee's first name, which read by the name alone is no
        definition of another file; ``throws`` whether reading that name
        there throws, so the call reaches nothing."""
        if within:
            with self._kept():
                self.call_guards = None
                return self.result(
                    self.resolve(callee, source_id, origin, arguments)[0]
                )
        self._result = self._receiver = self._deferred = self._shape_returns = None
        self._callables = {}
        self._this_args = {}
        self._lambdas = lambdas
        self._candidates_now = None
        self._name_only = False
        self._arguments = arguments
        self._path_call = "::" in callee and "." not in callee
        self._passed, self._borrows, self._callee = passed, borrows, callee
        self._receiver_written = receiver
        self._on = on
        self._typed_call = callee.endswith("<>")
        self._caller, self._looked_up, self._line = source_id, None, line
        self._in_block = in_block
        if self.lang(source_id) in _GLOBAL_OBJECT:
            callee = callee.removeprefix(_GLOBAL_OBJECT[self.lang(source_id)])
        # `new Type<arguments>`: the call constructs the type written,
        # whatever the callables it is passed return.
        constructed = None
        if origin and origin.startswith("new "):
            constructed, origin = re.sub(r"\{\d+=.*", "", origin[4:]), None
        if origin and origin.startswith("!") and "." not in origin:
            # A name bound to an item, `let make = Kind::Held;`, calls what
            # a call written with the item's path does.
            callee, origin = origin[1:], None
            self._path_call = "::" in callee
        elif origin and origin.startswith("!"):
            held = self._path_value(origin[1:].partition(".")[0], source_id)
            # A receiver's path that names no value is read as the callee
            # writes it.
            origin = None if held is None else origin
        verdict = self._verdict(callee, source_id, origin)
        held = verdict[0] and self._held_function(verdict[0])
        verdict = (held, verdict[1]) if held else verdict
        if throws or (
            shadowed
            and verdict[0]
            and not origin
            and verdict[0].split("::")[0] != source_id.split("::")[0]
        ):
            verdict = None, _UNRESOLVED
        if verdict[0] and not self._accepts(verdict[0]):
            # A definition that cannot take what the call passes is not
            # the one called: the receiver is another object than it seems.
            verdict = None, _UNRESOLVED
        if verdict[0] and (
            self.lang(verdict[0]) == "java"
            or self.lang(verdict[0]) == "clang"
            and self.nodes[verdict[0]]["type"] == "Method"
        ):
            # A Java or C++ method of the name that a base declares may be
            # the one that takes what the call passes.
            hit = self._overload_by_arity(verdict[0])
            verdict = (hit, verdict[1]) if hit else (None, _UNRESOLVED)
        if verdict[0] and self._candidate_only(verdict[0], source_id):
            # What resolved is a definition an `#if` arm alone builds, and
            # nothing about the call site selects that arm.
            self._candidates_now = self._candidates_now or [verdict[0]]
            verdict = None, _UNRESOLVED
        if (
            verdict[0] == source_id
            and self._path_call
            and self._outside_owner(verdict[0])
        ):
            # A trait's function for a type from outside that calls the
            # type's of its name calls the type's own, found before it.
            verdict = None, _EXTERNAL
        trait_path = (
            verdict[0]
            and self._path_call
            and self._trait_method(verdict[0])
            and _split_callee(callee)[0][-1:]
            == [self._name(verdict[0].rpartition("::")[0])]
        )
        if trait_path:
            # The trait's own path, not a type parameter it bounds: the
            # trait's method for the receiver, whatever else has the name.
            verdict = self._on_receiver(verdict[0])
        elif (
            verdict == (None, _EXTERNAL)
            and self._path_call
            and (owner := self._receiver_type())
        ):
            # An outside trait's path: the receiver's project impl of it.
            path, name = _split_callee(callee)
            own = [
                m
                for m in self._rust_trait_methods(owner, name, path[-1])
                if self._takes_self(m)
            ]
            trait_path = len(own) == 1
            verdict = (own[0], "regex") if trait_path else verdict
        given = (
            not trait_path
            and verdict[0]
            and self.lang(verdict[0]) == "rust"
            and self._alternatives(verdict[0])
        )
        fitting = [g for g in given or () if self._fits(g)]
        receiver = given and not self._path_call and self._receiver_borrowed()
        exact = [g for g in fitting if self._self_borrowed(g) == receiver]
        # A method whose `self` is the receiver's type, a reference or not, is
        # found before one the receiver reaches by a borrow or a dereference.
        fitting = exact if receiver is not None and exact else fitting
        inherent = [
            g for g in fitting if not self.nodes[g]["metadata"].get("trait_impl")
        ]
        # A type's own function is found before a trait's of its name.
        fitting = inherent if len(inherent) == 1 else fitting
        if given and len(fitting) == 1:
            verdict = fitting[0], verdict[1]
        elif given:
            # A Rust function given again in its scope, for another
            # `#[cfg]` or trait argument, is any of those that can take what
            # the call passes; what the call returns is what they all do.
            values = [self._called(("def", g), self._receiver) for g in given]
            first = values[0]
            one = all(
                v == first
                or isinstance(v, tuple)
                and isinstance(first, tuple)
                and v[0] == first[0] == "self"
                and v[1] in self._declarations(first[1])
                for v in values
            )
            # Declarations of one type in several files are the one type.
            self._result = first if one else None
            self._candidates_now = fitting or given
            outside = not fitting and self._outside_impls(given[0])
            verdict = None, _EXTERNAL if outside else _UNRESOLVED
        lang = self.lang(source_id)
        # A Go callee without an origin is a function, or one of a package,
        # as a C++ `std::` one is a function of the standard library and a
        # Rust path one of a type or a module.
        function = origin is None and (
            self._path_call
            if lang == "rust"
            else lang == "go" or (lang == "clang" and "::" in callee)
        )
        if verdict[1] == _EXTERNAL and function:
            # What a function of the standard library returns is in its data.
            known, value = self._std_call(callee, source_id)
            if known:
                self._result = value or _OPAQUE
        if verdict[1] == _EXTERNAL and constructed:
            # A standard type gives an instance of itself; a variable the
            # library declares as another type gives what is not known.
            if self.standard[lang].key(callee, source_id) == callee:
                held = self._returned_type(constructed, source_id)
                self._result = held or self._result
        elif verdict[1] == _EXTERNAL and not origin and lang == "dart":
            # Dart constructs without `new`: a call of a standard type's
            # name gives an instance of it.
            library = self.standard[lang]
            if (
                callee in library.data["types"]
                and library.key(callee, source_id) == callee
            ):
                self._result = self._returned_type(callee, source_id) or self._result
        elif verdict[1] == _EXTERNAL and not origin and lang == "typescript":
            known, value = self._js_global(callee, source_id)
            if known:
                self._result = value or _OPAQUE
        name = _split_callee(callee)[1]
        if verdict[1] == _EXTERNAL and origin and name in self.extensions:
            # A call on a value from outside may reach a C# extension method
            # written for a type from outside, unless that type has the
            # member itself. `Type.Method(...)` is no call on a value.
            written = (
                self.nodes[e]["metadata"]["extends"] for e in self.extensions[name]
            )
            path = re.fullmatch(r"(?!self\.)([A-Za-z_]\w*)(?:\.\w+)+", origin)
            static = path and not self._inherited(path[1], source_id)
            on_value = "." in origin and not static
            # The receiver's declared library type having the member says
            # the call is the library's — no extension of it is meant.
            typed = self._result is not None and self._result not in (
                _OPAQUE,
                _EXTERNAL,
            )
            if (
                on_value
                and not typed
                and not all(self._types("dotnet", w) for w in written)
            ):
                self._result = _OPAQUE
                return None, _UNRESOLVED
        unlisted = self.lang(source_id) in _UNLISTED_LIBRARY_LANGS
        if verdict[1] == _EXTERNAL and self._result is None and not unlisted:
            # What a call returns is from outside too when its receiver or
            # its path is, unless the call has type arguments, `name<>`,
            # one of which may be what it returns.
            self._result = _OPAQUE if callee.endswith("<>") else _EXTERNAL
        return verdict

    def _given_again(self, node_id):
        """The IDs of the Rust function or TypeScript local ``node_id`` and
        of each other its scope gives of its name, `name#2` and on."""
        base = re.sub(r"#\d+$", "", node_id)
        given = [base]
        while f"{base}#{len(given) + 1}" in self.nodes:
            given.append(f"{base}#{len(given) + 1}")
        return given

    def _alternatives(self, node_id):
        """The Rust function ``node_id`` and the others of its name a call
        that reaches it may be: those its scope gives again, and a method's
        of the name in the other impl blocks written for its type's name,
        where the crate defines the type, once."""
        found = self._given_again(node_id)
        owner, name = node_id.rpartition("::")[0], self._name(node_id)
        if self.nodes[node_id]["type"] != "Method" or owner not in self.nodes:
            return found
        # Blocks written for the type's own name, not an alias's
        declarations = [
            d for d in self._declarations(owner) if self._name(d) == self._name(owner)
        ]
        defined = [
            d for d in declarations if self.nodes[d]["metadata"].get("kind") != "impl"
        ]
        if len(defined) != 1:
            # Blocks for a type from outside may be for its other arguments.
            return found
        for declaration in declarations:
            other = self.members.get(declaration, {}).get(name)
            if other and other not in found:
                found += self._given_again(other)
        return found

    def _fits(self, target):
        """Whether the Rust function ``target`` can take what the call being
        resolved passes: as many arguments as it has parameters, `self`
        among them for a path call, a reference where it takes one and a
        value where it takes one, `self` too, and to a trait impl's method
        each argument of a known type of its parameter's, to one for a type
        from outside, which has impls from outside too, of a type known to
        be its parameter's. True where the parser did not record what the
        call passes."""
        metadata = self.nodes[target]["metadata"]
        fields = self.builder.fields.get(target, ())
        if self._is_class(target) and (
            metadata.get("unit")
            or fields
            and not any(isinstance(key, int) for key in fields)
        ):
            # Of structs only a tuple struct is called.
            return False
        receiver = self._receiver
        if (
            isinstance(receiver, tuple)
            and len(receiver) > 2
            and not self._path_call
            and not self._instantiates(target, receiver)
        ):
            # A block for one instantiation, `impl W<Own>`, holds members of
            # that instantiation only.
            return False
        params = metadata.get("param_types")
        if self._passed is None or params is None or target in self.macro_functions:
            # A macro's function is written with the macro's fragments.
            return True
        borrowed = bool(metadata.get("impl_borrowed"))
        by_value = bool(params) and _SELF_BY_VALUE_RE.fullmatch(params[0])
        if not self._path_call and self._takes_self(target):
            # A method called on a value takes it as its `self`.
            params = params[1:]
        if len(self._passed) != len(params):
            return False
        borrows = self._borrows or [None] * len(params)
        for index, (borrow, param) in enumerate(zip(borrows, params)):
            # A reference is passed where one is taken, and only there: a
            # path call's `self` by value is the type its block is for.
            # A reference fits a parameter whose type is a type parameter.
            receiver = index == 0 and self._path_call and by_value
            taken = borrowed if receiver else param.lstrip().startswith("&")
            checked = receiver or (
                metadata.get("trait_impl") and not re.search(r"\bself\b", param)
            )
            typed = taken or receiver or self._returned_type(param, target)
            if checked and borrow is not None and borrow != taken and typed:
                return False
        outside = self._outside_impls(target)
        for argument, param in zip(self._passed, params):
            if not metadata.get("trait_impl") or re.search(r"\bself\b", param):
                continue
            value = isinstance(argument, str) and self._value(
                argument, self._caller, set()
            )
            declared = self._returned_type(param, target)
            # A value of a type the project only writes impl blocks for, `f32`,
            # is of that type from outside.
            value, declared = (
                ("outside", self._name(v[1]), ())
                if isinstance(v, tuple) and v[0] == "self" and self._impl_only(v[1])
                else v
                for v in (value, declared)
            )
            written = _rust_type_parts(param)[0].rpartition("::")[2]
            if (
                declared is None
                and isinstance(value, tuple)
                and value[0] == "self"
                and written not in self.alias_names
                and self._types("rust", written)
                and written != self._name(value[1])
            ):
                # A project type the parameter names, which several
                # definitions leave untold, is not the argument's of another
                # name.
                return False
            project = [
                v[1]
                for v in (value, declared)
                if isinstance(v, tuple) and v[0] == "self"
            ]
            same = len(project) == 2 and project[0] in self._declarations(project[1])
            foreign = [
                v
                for v in (value, declared)
                if v == _EXTERNAL or isinstance(v, tuple) and v[0] == "outside"
            ]
            library = self.standard["rust"]
            named = [library.canonical(v[1]) for v in foreign if v != _EXTERNAL]
            if len(named) == 2:
                # A value of a type from outside is of the types it derives
                # from too, which it dereferences to: `String` is a `str`;
                # an alias is the type it names.
                same = named[1] in library.lineage(named[0])
                if not same:
                    return False
            if len(project) == 2 and not same or len(project) == 1 and foreign:
                return False
            if outside and not same:
                return False
        return True

    def _self_borrowed(self, node_id):
        """Whether the `self` the Rust method ``node_id`` takes is a
        reference: `&self`, or `self` of a block for `&T`; None for none."""
        metadata = self.nodes[node_id]["metadata"]
        first = (metadata.get("param_types") or [""])[0]
        if first.startswith("&"):
            return True
        if _SELF_BY_VALUE_RE.fullmatch(first):
            return bool(metadata.get("impl_borrowed"))
        return None

    def _receiver_borrowed(self):
        """Whether the receiver of the method call being resolved is a
        reference, from how it is written, `(*v)` or `&v`, or for `self` from
        the caller's own `self`; None where neither tells."""
        if self._receiver_written is not None:
            return self._receiver_written
        receiver = self._callee.rpartition(".")[0]
        written = _rust_borrow(receiver)
        if written is not None or receiver != "self":
            return written
        return self._self_borrowed(self._caller) if self._caller in self.nodes else None

    def _trait_method(self, node_id):
        """Whether ``node_id`` is a function a Rust trait declares, not a
        variant of an enum."""
        node = self.nodes[node_id]
        return (
            self.lang(node_id) == "rust"
            and node["type"] == "Function"
            and "param_types" in node["metadata"]
            and self._is_class(node_id.rpartition("::")[0])
        )

    def _takes_self(self, node_id):
        """Whether the Rust function ``node_id`` takes `self` first."""
        params = self.nodes[node_id]["metadata"].get("param_types") or [""]
        return bool(re.search(r"\bself\b", params[0]))

    def _receiver_type(self):
        """The project type of the receiver a Rust path call passes first;
        for a standard type's value, its project impl blocks where their
        headers write it by one name."""
        receiver = (self._passed or [None])[0]
        value = isinstance(receiver, str) and self._value(receiver, self._caller, set())
        if not isinstance(value, tuple) or value[0] not in ("self", "outside"):
            return None
        if value[0] == "self":
            return value[1]
        named = {self._name(b): b for b in self.impls_for.get(value[1], ())}
        return named.popitem()[1] if len(named) == 1 else None

    def _on_receiver(self, declared):
        """``(the method, how it was found)`` that the trait function
        ``declared``, called by its path, reaches on the receiver passed
        first: the receiver type's impl of that trait, else the blanket
        impl's that the type meets, else, for a receiver of the trait's own
        type (a bound), the trait's; unresolved, the impls the candidates,
        where the type has several (one per argument of a generic trait),
        none that is read (a macro writes it, and may override the trait's
        body), or is not known, as for a function without `self`, whose
        type the call's context gives."""
        name = self._name(declared)
        owner = self._takes_self(declared) and self._receiver_type()
        trait = self._name(declared.rpartition("::")[0])
        own = owner and (
            self._rust_trait_methods(owner, name, trait)
            or set(self._rust_blankets(self._declarations(owner), name, trait))
        )
        if own and len(own) == 1:
            return own.pop(), "regex"
        if own:
            self._candidates_now = sorted(own)
            return None, _UNRESOLVED
        if owner and owner == declared.rpartition("::")[0]:
            return declared, "regex"
        self._candidates_now = sorted(
            c
            for c in self.by_name.get(("rust", name), ())
            if c == declared or self.nodes[c]["metadata"].get("trait_impl") == trait
        )
        return None, _UNRESOLVED

    def _rust_trait_methods(self, owner_id, name, trait=None):
        """What a call of ``name`` may reach among the traits the impl
        blocks of the Rust type ``owner_id`` implement, in any file, only
        ``trait``'s where given: each block's method of the name, or its
        trait's own body where a block leaves it."""
        methods = [
            method
            for declaration in self._declarations(owner_id)
            for key, method in self.members.get(declaration, {}).items()
            if key.partition("#")[0] == name and self._trait_of(method)
        ]
        if trait:
            methods = [m for m in methods if self._trait_of(m) == trait]
        found, blocks, bodies = set(methods), {}, {}
        for declaration in self._declarations(owner_id):
            for base in self.bases.get(declaration, ()):
                bare = _split_callee(base)[1]
                if trait and bare != trait:
                    continue
                blocks[bare] = blocks.get(bare, 0) + 1
                for held in self._base_types(base, declaration) or ():
                    if body := self.members.get(held, {}).get(name):
                        bodies[bare] = body
        for bare, count in blocks.items():
            given = sum(self._trait_of(m) == bare for m in methods)
            if count > given and bare in bodies:
                found.add(bodies[bare])
        return found

    def _trait_of(self, node_id):
        """The trait whose impl block gives the Rust method ``node_id``."""
        return self.nodes[node_id]["metadata"].get("trait_impl")

    def _rust_method(self, owner_id, hits, name):
        """``(the method, complete)`` a method call of ``name`` reaches on
        the Rust type ``owner_id`` whose blocks give ``hits``, as rustc
        picks: an inherent one first; else the one method the traits the
        type implements give, those of one trait for several arguments
        left to what the call passes. Where several traits give it, the
        trait in scope decides, so those are the candidates."""
        inherent = [h for h in hits if not self._trait_of(h)]
        if inherent:
            return inherent[0], True
        hit = hits[0]
        found = self._rust_trait_methods(owner_id, name)
        rivals = {
            m for m in found if m == hit or self._trait_of(m) != self._trait_of(hit)
        }
        if len(rivals) <= 1:
            return hit, True
        self._candidates_now = sorted(found)
        return None, _SEVERAL

    def _path_value(self, path, scope_id):
        """Value of the Rust path ``path`` written in ``scope_id`` that names
        a unit struct, `Marker`, a unit variant, `Kind::Unit`, or a constant,
        `Kind::ALL`: an instance of the struct or the enum, a value of the
        type the constant declares; outside for one from outside; None for
        any other, a function or a tuple struct or variant written bare
        being no value but what a call of it reaches."""
        qualifier, _, name = path.rpartition("::")
        typed = self._enclosing_class if "Self" in (path, qualifier) else None
        owner = qualifier and (
            typed(scope_id) if typed else self._type_named(qualifier, scope_id)
        )
        if owner == _EXTERNAL:
            return _EXTERNAL
        # A type's member is a variant or a constant of it, a module's an item.
        member = owner and self.members.get(owner, {}).get(name)
        hit = member or (
            typed(scope_id) if path == "Self" else self._type_named(path, scope_id)
        )
        if hit in (None, _EXTERNAL):
            return hit
        if self.nodes[hit]["type"] == "Constant":
            return self._declared_type(hit)
        if not self.nodes[hit]["metadata"].get("unit"):
            return None
        return "self", owner if member and self._is_class(owner) else hit

    def _rust_variant(self, node_id):
        """Whether ``node_id`` is a variant of a Rust enum."""
        node = self.nodes[node_id]
        return (
            self.lang(node_id) == "rust"
            and node["type"] == "Function"
            and "param_types" not in node["metadata"]
            and self._is_class(node_id.rpartition("::")[0])
        )

    def _outside_impls(self, node_id):
        """Whether the Rust function ``node_id`` is a generic trait's from
        outside for a type from outside, which has impls of that trait from
        outside too; one of a project trait has only the project's."""
        metadata = self.nodes[node_id]["metadata"]
        return bool(
            metadata.get("trait_generic")
            and not self._types("rust", metadata["trait_impl"])
            and self._outside_owner(node_id)
        )

    def _outside_owner(self, node_id):
        """Whether the Rust function ``node_id`` is a trait's for a type from
        outside, one the project only writes impl blocks for; a module of the
        type's name, `mod f32`, is no type."""
        owner = node_id.rpartition("::")[0]
        return bool(
            self.nodes[node_id]["metadata"].get("trait_impl")
            and owner in self.nodes
            and self._impl_only(owner)
        )

    def _impl_only(self, type_id):
        """Whether the project only writes impl blocks for the Rust type
        ``type_id``, one from outside; a module of its name is no type."""
        kinds = {
            self.nodes[d]["metadata"].get("kind") for d in self._declarations(type_id)
        }
        return kinds - {"module"} == {"impl"}

    def _std_inherent(self, key, name):
        """Whether the Rust standard type ``key`` of the data has a method
        ``name`` of its own. rustc picks it before a trait's of the name
        where both take the receiver alike, which the data does not say."""
        types = self.standard["rust"].data["types"]
        return name in types.get(key, {}).get("methods", ())

    def _instantiates(self, member, value):
        """Whether ``member`` is one that ``value``, an instance with its
        type arguments, has: of a block for every instantiation, or for one
        whose arguments are the value's where both are known."""
        fixed = self.nodes[member]["metadata"].get("impl_args") or ()
        for written, held in zip(fixed, value[2]):
            given = written and held and self._returned_type(written, member)
            if given and given[:2] != held[:2]:
                return False
        return True

    def _field_arguments(self, type_id, fields, scope_id):
        """The type arguments of a value of the Rust struct ``type_id`` made
        in ``scope_id`` from the origins ``fields``, by field name or
        position: each type parameter a field is declared as, of the value
        it is given."""
        generics = list(self.nodes[type_id]["metadata"].get("generics") or ())
        given = [None] * len(generics)
        declared = self.builder.fields.get(type_id, {})
        for name, origin in fields.items():
            if declared.get(name) in generics and isinstance(origin, str):
                held = self._value(origin, scope_id, set())
                held = held if isinstance(held, tuple) else None
                given[generics.index(declared[name])] = held
        return tuple(given)

    def _argument_type(self, origin):
        """The key in the standard library's data of the type of the
        argument the call being resolved passes with the origin ``origin``,
        `{integer}` for an integer literal of no suffix, and for a range,
        `{lang:Range}`, the type that language item is; None where not
        known."""
        if origin == "{integer}":
            return origin
        if isinstance(origin, str) and origin.startswith("{lang:"):
            return self.standard["rust"].data.get("lang", {}).get(origin[6:-1])
        held = isinstance(origin, str) and self._value(origin, self._caller, set())
        return held[1] if isinstance(held, tuple) and held[0] == "outside" else None

    def _accepts(self, target):
        """Whether the definition ``target`` can take the arguments of the
        call being resolved, where its parser counted them and recorded
        what the definition takes: ``(positional arguments, whether
        keywords follow)`` against ``(required, most or None for any
        number, keywords: 0 none, 1 taken, 2 required)``. Keywords passed
        to a definition that names none are its last positional argument.
        A module takes no call."""
        if self.nodes[target]["type"] == "File":
            return False
        arity = self.nodes[target]["metadata"].get("arity")
        if not isinstance(self._arguments, list) or not arity:
            return True
        count, keywords = self._arguments
        required, most, named = arity
        count += keywords and not named
        fits = required <= count and (most is None or count <= most)
        return fits and (keywords or named < 2)

    def _proto_names(self):
        """Each protobuf message's and service's full name, `package.Outer.
        Inner`, -> its node IDs; and every leading part of such a name."""
        if self._proto is None:
            full = {}
            for node_id, node in self.nodes.items():
                if node["type"] == "Class" and self.lang(node_id) == "proto":
                    file_id, _, path = node_id.partition("::")
                    package = self.nodes[file_id]["metadata"].get("proto_package")
                    name = ".".join(filter(None, (package, path.replace("::", "."))))
                    full.setdefault(name, []).append(node_id)
            parts = {
                ".".join(name.split(".")[:at])
                for name in full
                for at in range(1, name.count(".") + 2)
            }
            self._proto = full, parts
        return self._proto

    def _proto_verdict(self, written, source_id):
        """The message a protobuf type reference ``written`` in
        ``source_id`` names, as protoc finds it: `.a.B` from the root; else
        the first part from the file's package outward, the rest inside
        what it names. One no project name starts as is from outside."""
        full, parts = self._proto_names()
        package = self.nodes[source_id.split("::")[0]]["metadata"].get(
            "proto_package", ""
        )
        if written.startswith("."):
            name = written[1:]
        else:
            scopes = package.split(".") if package else []
            first = written.split(".")[0]
            name = next(
                (
                    ".".join([*scopes[:at], written])
                    for at in range(len(scopes), -1, -1)
                    if ".".join([*scopes[:at], first]) in parts
                ),
                None,
            )
        if name is None or name.split(".")[0] not in parts:
            return None, _EXTERNAL
        found = full.get(name, [])
        if len(found) == 1:
            return found[0], "regex"
        self._candidates_now = sorted(found) or None
        return None, _UNRESOLVED

    def _verdict(self, callee, source_id, origin):
        lang = self.lang(source_id)
        if lang == "proto":
            return self._proto_verdict(callee, source_id)
        qualifier, name = _split_callee(callee)
        candidates = self.by_name.get((lang, name), [])
        if lang == "typescript" and candidates:
            # A definition elsewhere of the same name is not a candidate
            # the call can reach, nor is a type, which no value is.
            candidates = [
                c
                for c in self._named_from(candidates, source_id)
                if self.nodes[c]["metadata"].get("kind") != "interface"
            ]
        if lang == "php":
            # A PHP call reaches no property or constant of its name.
            candidates = [c for c in candidates if self.nodes[c]["type"] != "Constant"]
        scope_id = source_id
        if origin is None and lang == "go" and qualifier:
            # A package-level variable is read as what it is declared as,
            # where it is declared.
            held = self._go_variable(qualifier[0], source_id)
            if held:
                written = self.nodes[held]["metadata"]["value_type"]
                origin = ".".join([f"{written}()", *qualifier[1:], name])
                origin, scope_id = origin.replace(".", "::", written.count(".")), held
        if lang == "clang" and origin and origin[0] == "^":
            # A bare call written as an argument of a macro is qualified as
            # the macro's text qualifies that parameter: globally, or by a
            # scope or receiver nothing tells. A pasted name is not the one
            # written, and definitions of the macro that disagree leave the
            # call open.
            how = self._c_rewritten(*origin[1:].split("^"), source_id)
            origin = None
            if len(how) > 1:
                return None, _UNRESOLVED if candidates else _EXTERNAL
            if "paste" in how:
                return None, _EXTERNAL
            if "." in how:
                return self._resolve_member(candidates, lang, callee)
            callee = f"::{callee}" if how else callee
        if lang == "clang" and origin is None and name in self.builder.c_aliases:
            # A call of an alias constructs the type it names: unqualified,
            # where C++ looks the name up in a scope around the call that
            # declares the alias first, else where nothing else of that name
            # can be called.
            written = "::".join([*qualifier, name])
            reached = [
                c
                for c in candidates
                if self._is_top_level(c) and self._in_reach(c, source_id)
            ]
            aliased = not qualifier and self._c_alias_first(name, reached, source_id)
            found = (aliased or not reached) and self._type_named(
                written, source_id, True
            )
            if isinstance(found, tuple):
                # An instance of a project type, or of one from outside
                found = found[1] if found[0] == "self" else _EXTERNAL
            if found and found != _EXTERNAL:
                return found, "regex"
            if aliased:
                # A type that is not told may be the project's.
                return None, _EXTERNAL if found == _EXTERNAL else _UNRESOLVED
        if lang == "clang" and origin is None and callee.endswith(">"):
            # A template written with arguments that is called constructs the
            # specialization they are an instance of.
            held = self._c_specialized(callee, source_id)
            if held and held[0] == "self":
                return held[1], "regex"
            if held:
                self._candidates_now = sorted({one[1] for one in held[1] if one})
                return None, _UNRESOLVED if self._candidates_now else _EXTERNAL
        if lang == "clang" and origin is None and "::" in callee:
            # The global namespace and one a macro opens have no node: what
            # they hold is at the top of the files, beside what a namespace
            # of the name holds.
            qualifier = self._c_unmacroed(qualifier)
            if all(q in self.c_spaces for q in qualifier):
                fits = [c for c in candidates if self._c_path(c) == [name]]
                file_id = source_id.split("::")[0]
                return self._resolve_bare(name, fits, file_id, lang, callee)
            qualifier = self._c_written(qualifier, candidates, source_id)
            # A type written with template arguments is the specialization
            # they are an instance of, else the template itself, never
            # another one.
            head = callee.rpartition("::")[0]
            depth, cut = 0, 0
            for at, char in enumerate(head):
                depth += (char == "<") - (char == ">")
                if not depth and head[at : at + 2] == "::":
                    cut = at + 2
            written, opened, arguments = head[cut:].partition("<")
            if opened:
                template = written.strip()
                instances = self._c_instances(c_specialization(template, arguments))
                owned = {
                    c: self._member_owner(c).rpartition("::")[2] for c in candidates
                }
                candidates = [
                    c
                    for c in candidates
                    if owned[c].partition("<")[0] != template or owned[c] in instances
                ]
                fits = [c for c in candidates if owned[c] in instances]
                if len(instances) > 1 and fits:
                    self._candidates_now = sorted(fits)
                    return None, _UNRESOLVED
                if qualifier and len(instances) == 1 and instances != [template]:
                    qualifier = [*qualifier[:-1], instances[0]]
        if origin is not None:
            verdict = self._resolve_origin(origin, scope_id)
            if verdict is None:
                verdict = self._resolve_member(candidates, lang, callee)
            if lang == "dart" and verdict[1] == _EXTERNAL and name.startswith("_"):
                # A private Dart name is declared in the library that uses it.
                verdict = self._resolve_member(candidates, lang, callee)
            # A Python variable or parameter whose value the parser could
            # not tell holds a definition whatever its own name.
            if verdict[0] or candidates or lang == "python" and origin[:1] == "=":
                return verdict
            if self._candidates_now:
                # Several definitions fit the name; the call reaches one.
                return None, _UNRESOLVED
            # No project symbol carries the name, so the call is external;
            # what it returns is unknown unless its receiver is external too.
            if verdict[1] != _EXTERNAL:
                self._result = self._shape_returns or _OPAQUE
            return None, _EXTERNAL
        if lang == "php":
            return self._php_function(callee, candidates, source_id)
        rooted = lang == "rust" and callee.startswith(_RUST_PATH_ROOTS)
        # A path from a project crate's name, where its modules tell
        walked = (
            lang == "rust"
            and not rooted
            and "::" in callee
            and self._rust_rooted(callee, source_id)
            and self._rust_path(qualifier, source_id)
        )
        rooted = rooted or bool(walked)
        if (
            lang == "rust"
            and not qualifier
            and (item := self._rust_local(name, source_id))
        ):
            return item, "regex"
        if rooted:
            imported = walked or self._rust_path(qualifier, source_id)
        elif (
            lang in ("typescript", "java")
            and not qualifier
            and self._shadows_import(name, source_id, lang)
        ):
            imported = None
        else:
            imported = self._imported(qualifier[0] if qualifier else name, source_id)
        if imported == _EXTERNAL:
            return None, _EXTERNAL
        if imported is None and qualifier:
            # A variable in scope that has a declared type, or members of
            # its own, is a receiver as an imported name is.
            held = self._aliased(
                self._lexical(qualifier[0], source_id, skip_classes=True)
            )
            if held and self.nodes[held]["type"] == "Constant":
                declared = self._declared_type(held)
                if declared == _EXTERNAL:
                    return None, _EXTERNAL
                if declared or held in self.members or held in self.builder.reexports:
                    imported = [held]
        if imported is not None:
            # An import names the definition itself or the scope it is in;
            # each further segment names what that scope holds in turn. The
            # scope may export the definition under another name.
            paths = [[] if rooted else qualifier[1:]]
            bound = self.builder.imports.get(source_id.split("::")[0], {})
            if (
                lang in _NAMESPACE_RECEIVER_LANGS
                and qualifier[:1]
                and qualifier[0] in bound
            ):
                # A receiver an import binds to one export, not to the
                # module whole, is what the module holds under its name.
                head = self._exported_as(qualifier[0], source_id)
                paths = [qualifier[1:] if head == "=" else [head, *qualifier[1:]]]
            looked_up = name if qualifier else self._exported_as(name, source_id)
            for path in paths:
                scopes = imported
                for segment in path:
                    found = (self._scope_member(scope, segment) for scope in scopes)
                    scopes = [scope for scope in found if scope]
                # A Rust `use` names the item itself, under any name.
                hits = {
                    i
                    for i in scopes
                    if not qualifier
                    and (lang == "rust" or i.endswith(f"::{looked_up}"))
                }
                hits |= {self._scope_member(scope, looked_up) for scope in scopes}
                hits.discard(None)
                if hits:
                    break
            if hits or (candidates and (len(qualifier) <= 1 or rooted)):
                hit = self._decide(sorted(hits), source_id) if hits else None
                return (hit, "regex") if hit else (None, _UNRESOLVED)
        held = len(qualifier) == 1 and self._written(qualifier[0], source_id, set())
        if self._js_function(held):
            # A property of a function is one assigned to it, else one every
            # function has, else one assigned to it where none is told,
            # which any member of the name may be.
            hit, complete = self._find_member(held, name, static=True)
            if hit or complete == _SEVERAL:
                return (hit, "regex") if hit else (None, _UNRESOLVED)
            function = self.standard[lang].value("Function", {})
            if self.standard[lang].member(function, name)[0]:
                self._result = _OPAQUE
                return None, _EXTERNAL
            self._resolve_member(candidates, lang, callee)
            return None, _UNRESOLVED
        owner = (
            lang in ("python", "typescript")
            and qualifier in (["this"], ["self"])
            and self._enclosing_class(source_id)
        )
        if lang == "rust" and not (qualifier or candidates):
            # A `use` that a glob brings may name an item by another name.
            renamed = self._rust_globbed(
                name, source_id, self._rust_in_scope(source_id)[1:]
            )
            if len(renamed) == 1:
                return renamed[0], "regex"
        if not candidates and not (
            owner
            and (
                self._instance_values(owner, name)
                or lang == "typescript"
                and self._assigned(owner, name)
            )
        ):
            # No project symbol has the name, nor does a field `this` holds
            # one; that alone does not say where the receiver, and so the
            # result, comes from. A Java class's bare call, or a path call
            # of one project type (`H::default()`), has it from a standard
            # trait or base of that type, derived ones too.
            owner = None
            if lang == "java" and not qualifier:
                owner = self._enclosing_class(source_id)
            elif self._path_call and qualifier:
                types = self._types_in(
                    lang, qualifier[-1], source_id, len(qualifier) == 1
                )
                owner = types[0] if len(types) == 1 else None
            self._result = owner and self._std_inherited(owner, name) or _OPAQUE
            return None, _EXTERNAL
        if not qualifier:
            return self._resolve_bare(name, candidates, source_id, lang, callee)
        receiver = qualifier[-1]
        static = None
        # JavaScript's `super` is the base of the class around, or of its
        # constructor in a static member.
        javascript_super = lang == "typescript" and receiver == "super"
        if receiver in _SELF_RECEIVERS.get(lang, ()) or javascript_super:
            # `this` in a JavaScript static member is its constructor.
            static = len(qualifier) == 1 and self._static_owner(source_id)
            owner = static or self._enclosing_class(source_id)
            owners = [owner] if len(qualifier) == 1 else []
            source = "self_type"
            key = lang == "rust" and any(owners) and self._impl_only(owner)
            key = key and self._standard_type(self._name(owner), source_id)
            if key and self._std_inherent(key, name):
                # In a block for a standard type, its own method may be
                # rustc's pick.
                hit = self._instance_member(owner, name)[0]
                self._candidates_now = [hit] if hit else None
                return None, _UNRESOLVED if hit else _EXTERNAL
        else:
            owners = self._types_in(lang, receiver, source_id, len(qualifier) == 1)
            named = lang == "rust" and len(owners) > 1
            named = named and self._type_named("::".join(qualifier), source_id)
            if named and named != _EXTERNAL and named in owners:
                # The one of them the path names where it is written
                owners = [named]
            if (
                lang == "rust"
                and len(qualifier) == 1
                and receiver not in self._uses(source_id)
            ):
                # A module is named only where scope or a glob brings it.
                reached = self._rust_modules_named(receiver, source_id)
                owners = [
                    o
                    for o in owners
                    if self.nodes[o]["metadata"].get("kind") != "module" or o in reached
                ]
            if lang == "clang":
                owners = [o for o in owners if self._in_reach(o, source_id)] or owners
            if lang != "rust":
                owners += self.containers.get((lang, receiver), [])
            elif callee.endswith(f"::{name}"):
                # A Rust module is named in a path, never before a `.`. The
                # first segment of a path may be the name a `use` in force
                # binds the module under.
                uses = self._uses(source_id)
                if len(qualifier) == 1 and receiver not in uses:
                    # A module written alone is one in scope there.
                    owners += self._rust_modules_named(receiver, source_id)
                else:
                    bound = uses.get(receiver, receiver)
                    owners += self.containers.get((lang, bound.rpartition("::")[2]), [])
            source = "regex"
            root = qualifier[0]
            # A path of two or more segments, not a chain of fields, that
            # starts outside the project.
            if (
                lang == "rust"
                and len(qualifier) > 1
                and callee.startswith(f"{root}::")
                and root != "Self"
                and self._use_crate(root, source_id) == ()
            ):
                # `io::Error` is not a project type; the project can
                # still give it methods through a trait impl.
                hits = {
                    self.members[o][name]
                    for o in owners
                    if self.nodes[o]["metadata"].get("kind") == "impl"
                    and name in self.members.get(o, ())
                }
                hit = self._decide(sorted(hits), source_id) if hits else None
                return (hit, source) if hit else (None, _EXTERNAL)
        if any(owners):
            hits, complete = set(), True
            for owner in owners:
                if javascript_super:
                    hit, owner_complete = self._find_member(
                        owner, name, inherited_only=True, static=bool(static)
                    )
                elif source == "self_type" and receiver != "cls" and not static:
                    hit, owner_complete = self._instance_member(owner, name)
                else:
                    # A type named is its constructor, whose members are
                    # the static ones.
                    hit, owner_complete = self._find_member(owner, name, static=True)
                # A namespace may hold more than the project puts in it,
                # unless it is a value too, whose type tells what it holds.
                metadata = self.nodes[owner]["metadata"]
                open_scope = metadata.get("kind") == "namespace" and not metadata.get(
                    "value_type"
                )
                complete = complete and owner_complete and not open_scope
                if hit:
                    hits.add(hit)
            if lang == "clang":
                hits = {h for h in hits if self._in_reach(h, source_id)} or hits
            if hits:
                hit = self._decide(sorted(hits), source_id)
                if hit and lang == "clang":
                    hit = self._c_overload(hit, hits, source_id)
                return (hit, source) if hit else (None, _UNRESOLVED)
            # A member a project type has from outside may return that type;
            # a field it assigns itself is its own.
            self._result = _OPAQUE
            if len(owners) == 1:
                self._result = self._std_inherited(owners[0], name) or _OPAQUE
            complete = complete or any(self._assigns(o, name) for o in owners)
            return None, _UNRESOLVED if complete else _EXTERNAL
        aliased = lang == "clang" and receiver in self.builder.type_aliases
        if lang in ("rust", "clang") and "::" in callee and not aliased:
            return None, _EXTERNAL
        if lang == "typescript" and self._js_global(callee, source_id)[0]:
            return None, _EXTERNAL
        return self._resolve_member(candidates, lang, callee)

    def _php_function(self, callee, candidates, source_id):
        """Verdict on a PHP call without a receiver, which reaches a function
        only. The parser writes its name in full: `\\App\\run` for the
        function of that name, `App\\run` for the one of the caller's
        namespace or, failing that, the global `run`."""
        namespace, _, name = callee.rpartition("\\")
        exact = callee.startswith("\\")
        for space in [namespace.lstrip("\\")] if exact else [namespace, ""]:
            functions = [
                c
                for c in candidates
                if self.nodes[c]["type"] == "Function"
                and self.nodes[c]["metadata"].get("namespace") == space
            ]
            if functions:
                hit = self._decide(functions, source_id)
                return (hit, "regex") if hit else (None, _UNRESOLVED)
        # What a function of PHP's own returns is in the library's data.
        key = f"{namespace}\\{name}".strip("\\") if exact else name
        value = self.standard["php"].function(key, source_id)[1]
        self._result = value or _OPAQUE
        return None, _EXTERNAL

    def _exported_as(self, name, scope_id):
        """The name the module an import in the file of ``scope_id`` binds
        ``name`` from exports it as."""
        renamed = self.builder.import_names.get(scope_id.split("::")[0], {})
        return renamed.get(name, name)

    def _ts_alias(self, name, scope_id):
        """``(type parameters, text)`` of the TypeScript type alias ``name``
        written in ``scope_id`` stands for: one its file declares, else one
        a module it imports the name from does. None for none."""
        file_id = scope_id.split("::")[0]
        found = self.builder.ts_aliases.get((file_id, name))
        imported = None if found else self._imported(name, scope_id)
        for module in imported if isinstance(imported, list) else ():
            exported = self._exported_as(name, scope_id)
            found = found or self.builder.ts_aliases.get(
                (module.split("::")[0], exported)
            )
        return found

    def _imported(self, name, scope_id):
        """What an import in force at ``scope_id`` binds ``name`` to: the
        IDs of the project definitions or files it names (none when they
        cannot be located), ``"external"`` for a target outside the project,
        None when no import binds it."""
        file_id = scope_id.split("::")[0]
        target = self._uses(scope_id).get(name)
        lang = self.lang(file_id)
        if lang == "dart":
            return self._dart_prefixed(name, file_id)
        if target is None or lang not in ("go", "java", "typescript", "rust"):
            return None
        if lang == "typescript":
            required = name in self.builder.required_names.get(file_id, ())
            if name in self.builder.dynamic_names.get(file_id, ()):
                # `import()` loads an ES module whatever the file's format.
                required = None
            return self._js_module(target, file_id, required)
        if lang == "rust":
            # A path from the crate's root or the module leads through the
            # module tree, where it tells; a name each `#[cfg]` binds to
            # another path is any of theirs.
            alternatives = next(
                (
                    held
                    for scope in self._scopes(scope_id)
                    if (held := self.builder.rust_use_alternatives.get((scope, name)))
                ),
                (target,),
            )
            found = None
            if (scope_id, name) not in self._importing:
                # A `use` may lead back to itself through the module tree.
                self._importing.add((scope_id, name))
                try:
                    with self._kept():
                        found = self._rust_reached(alternatives, scope_id)
                finally:
                    self._importing.discard((scope_id, name))
            if found:
                return found
            module, rest = self._rust_walk(target.split("::"), scope_id)
            if module and self._rust_rooted(target, scope_id):
                # Where the module tree leads to a module but not to the
                # item, a `#[cfg]` or a glob re-export may bind it to
                # anything: it is not told; through a module the crate does
                # not have, it leaves the project.
                return [] if set(rest[:-1]) <= self.rust_module_names else None
            *path, leaf = target.split("::")
            modules = path[1:]
            crate = self._use_crate(target, scope_id)
            if not crate:
                return None if crate is None else _EXTERNAL
            found = [
                c
                for c in self.by_name.get((lang, leaf), ())
                if (self._unit(c, lang),) == crate
            ]
            # Without a module tree, a path through modules names the
            # crate's one definition of its name, never one chosen by where
            # its file is.
            defined = [
                c for c in found if self.nodes[c]["metadata"].get("kind") != "impl"
            ]
            if modules and len(defined) == 1:
                return defined
            return [] if modules and defined else found or None
        if lang == "go":
            for folder, module in self.go_modules:
                if target == module or target.startswith(f"{module}/"):
                    parts = (folder, target[len(module) + 1 :])
                    return self.go_packages.get("/".join(p for p in parts if p), [])
            return _EXTERNAL if self.go_modules else None
        if target in self.java_types:
            return [self.java_types[target]]
        owner, _, member = target.rpartition(".")
        if owner in self.java_types:
            hit = self._find_member(self.java_types[owner], member)[0]
            return [hit] if hit else []
        return _EXTERNAL

    def _scope_member(self, scope, name):
        """What ``name`` names in ``scope``, a static member of it: in a Rust
        module also what a `use` of it binds."""
        hit = self._find_member(scope, name, static=True)[0]
        if hit or self.lang(scope) != "rust" or self._rust_module(scope) != scope:
            return hit
        found = self._rust_path(["self", name], scope)
        return found[0] if isinstance(found, list) and len(found) == 1 else None

    def result(self, target):
        """Value returned by the call just resolved, to ``target`` if it was."""
        if target:
            if self._name_only:
                # Resolved by name: what the definition returns is not
                # evidence of what this call's receiver gives back.
                return _OPAQUE
            return self._called(("def", target), self._receiver)
        return None if self._result == _OPAQUE else self._result

    def _rust_tree(self, declared, files):
        """``(files, parents, roots)`` of the Rust modules ``declared`` (file
        ID -> [(module ID, `#[path]`)]): the IDs of the files each module ID
        may be, the module IDs each file is declared as, and the crate roots:
        the files where Cargo finds crates that no `mod` declares. A crate
        root or `mod.rs` holds its modules' files in its directory, any other
        file in a directory of its own name."""

        def lay_out(roots):
            found, parents = {}, {}
            for file_id, modules in declared.items():
                folder, stem = posixpath.split(file_id[5:].removesuffix(".rs"))
                own = stem == "mod" or file_id in roots
                for module_id, written in modules:
                    *inner, name = module_id[len(file_id) + 2 :].split("::")
                    # Inline modules around it name directories; a function
                    # around it does not.
                    inner = [
                        part
                        for at, part in enumerate(inner)
                        if self.nodes.get("::".join([file_id, *inner[: at + 1]]), {})
                        .get("metadata", {})
                        .get("kind")
                        == "module"
                    ]
                    at = posixpath.join(folder if own else f"{folder}/{stem}", *inner)
                    if written:
                        paths = [posixpath.join(at if inner else folder, written)]
                    else:
                        paths = [f"{at}/{name}.rs", f"{at}/{name}/mod.rs"]
                    for path in paths:
                        path = f"FILE:{posixpath.normpath(path.lstrip('/'))}"
                        # A `mod` declared again under another `#[cfg]` is
                        # the one module.
                        if path in files and path not in found.get(module_id, ()):
                            found.setdefault(module_id, []).append(path)
                            parents.setdefault(path, []).append(module_id)
            return found, parents

        targets = {
            f
            for f in files
            if f.endswith(".rs")
            and (
                posixpath.basename(f[5:]) in ("lib.rs", "main.rs", "build.rs")
                or posixpath.basename(posixpath.dirname(f[5:]))
                in ("bin", "tests", "examples", "benches")
            )
        }
        # A target that a `mod` declares is a module, not a root.
        found, parents = lay_out(targets)
        found, parents = lay_out(targets - parents.keys())
        return found, parents, targets - parents.keys()

    def _rust_module(self, node_id):
        """The innermost Rust module around ``node_id``, itself if it is one:
        an inline module's ID, else its file's."""
        return next(
            s
            for s in self._scopes(node_id)
            if "::" not in s
            or (node := self.nodes.get(s))
            and node["metadata"].get("kind") == "module"
        )

    def _rust_parent(self, module):
        """The module that holds the Rust module ``module``; None for a crate
        root, and for a file that several `mod` declarations name."""
        if "::" in module:
            return self._rust_module(module.rpartition("::")[0])
        declared = self.rust_parents.get(module, [])
        return (
            self._rust_module(declared[0].rpartition("::")[0])
            if len(declared) == 1
            else None
        )

    def _rust_child(self, module, name):
        """The module ``name`` of the Rust module ``module``: an inline one,
        else the file its `mod name;` declares; None for none or several."""
        inline = f"{module}::{name}"
        if self._rust_module(inline) == inline:
            return inline
        held = self.rust_files.get(inline, [])
        return held[0] if len(held) == 1 else None

    def _rust_walk(self, path, source_id):
        """``(module, rest)``: the module the leading segments of the Rust
        path ``path`` lead to through the module tree from where
        ``source_id`` writes it, and the segments after it; no module where
        the tree leads nowhere."""
        module = self._rust_module(source_id)
        for at, segment in enumerate(path):
            if segment == "crate":
                while module and module not in self.rust_roots:
                    module = self._rust_parent(module)
            elif segment == "super":
                module = self._rust_parent(module)
            elif segment != "self":
                child = (
                    self._rust_scope_module(segment, source_id)
                    if at == 0
                    else self._rust_child(module, segment)
                )
                if not child:
                    # Of the files `#[cfg]`s declare the module as, the one
                    # the writing file is in is the one its build compiles.
                    own = self._rust_module(source_id)
                    around = set()
                    while own and own not in around:
                        around.add(own)
                        own = self._rust_parent(own)
                    held = self.rust_files.get(f"{module}::{segment}", [])
                    held = [h for h in held if h in around]
                    child = held[0] if len(held) == 1 else None
                if not child and module in self.rust_roots:
                    # `extern crate a as b` at a crate root names `a` `b`.
                    crate = self.builder.rust_externs.get(module, {}).get(segment)
                    child = crate and self._rust_library(crate, module)
                if not (child or at):
                    child = self._rust_library(segment, source_id)
                if not child:
                    return module, path[at:]
                module = child
            if not module:
                return None, []
        return module, []

    def _rust_path(self, path, source_id, seen=frozenset()):
        """IDs of what the Rust path ``path``, its segments from `crate`,
        `self`, `super` or a project crate's name on, names where
        ``source_id`` writes it: a module by
        the module tree, then what each further segment names in the last
        module, by an item of its own or by a `use`. No ID where the module
        tree does not tell it, ``"external"`` where a `use` binds it to an
        item outside the project. ``seen`` holds the `use` bindings being
        followed."""
        module, rest = self._rust_walk(path, source_id)
        if not module:
            return []
        found = [module]
        for segment in rest:
            hits, outside = [], False
            for scope in found:
                hit = self._find_member(scope, segment)[0]
                if hit:
                    hits.append(hit)
                elif self._rust_module(scope) == scope:
                    held = self._rust_used(scope, segment, seen)
                    outside = outside or held == _EXTERNAL
                    hits += held if isinstance(held, list) else []
            if outside:
                # Where one `#[cfg]` alternative leads outside the project,
                # which the name is is not told.
                return [] if hits else _EXTERNAL
            found = list(dict.fromkeys(hits))
        # A path names a type's definition, not the impl blocks of it.
        defined = [
            f
            for f in found
            if self.nodes.get(f, {}).get("metadata", {}).get("kind") != "impl"
        ]
        return defined or found

    def _rust_used(self, module, segment, seen):
        """What a `use` of the Rust module ``module`` binds ``segment`` to,
        a path from the module or from a crate, else what each glob `use`
        of the module's own does: IDs, ``"external"`` where one leads
        outside the project. ``seen`` as `_rust_path`'s."""
        uses = self._uses(module)
        alternatives = self.builder.rust_use_alternatives.get((module, segment))
        written = [uses[segment]] if segment in uses else []
        written = sorted(alternatives or written) or [
            f"{path[:-1]}{segment}" for path in uses.values() if path[-3:] == "::*"
        ]
        hits = []
        for target in written:
            if self._rust_child(module, target.split("::")[0]):
                target = f"self::{target}"
            held = None
            if segment in uses and not self._rust_rooted(target, module):
                held = self._imported(segment, module)
            elif self._rust_rooted(target, module):
                if (module, segment) not in seen:
                    followed = seen | {(module, segment)}
                    held = self._rust_path(target.split("::"), module, followed)
            else:
                # A glob of another crate's module may bring the name.
                held = _EXTERNAL
            if held == _EXTERNAL:
                return _EXTERNAL
            hits += held or []
        return hits

    def _rust_library(self, name, scope_id):
        """The root of the library of the project crate ``name`` written in
        ``scope_id``; None for none."""
        folder = self.builder.rust_crates.get(self._crate_named(name, scope_id))
        library = folder is not None and f"FILE:{posixpath.join(folder, 'src/lib.rs')}"
        return library if library in self.rust_roots else None

    def _rust_rooted(self, path, scope_id):
        """Whether the Rust path ``path`` written in ``scope_id`` starts where
        the module tree tells: at `crate`, `self`, `super`, a project
        crate's library or a module in scope there."""
        head = path.split("::")[0]
        return path.startswith(_RUST_PATH_ROOTS) or bool(
            self._rust_library(head, scope_id)
            or self._rust_scope_module(head, scope_id)
        )

    def _rust_modules_named(self, name, scope_id):
        """The Rust modules ``name`` names written in ``scope_id``: the one
        in scope (see `_rust_scope_module`), else the files `#[cfg]`s declare
        it as there, else those the glob `use`s in force there bring."""
        held = self._rust_scope_module(name, scope_id)
        if held:
            return [held]
        for scope in self._scopes(scope_id):
            # A module `#[cfg]`s declare as several files is any of them.
            several = self.rust_files.get(f"{scope}::{name}", [])
            if len(several) > 1:
                return several
            if self._rust_module(scope) == scope:
                break
        brought = (self._rust_child(m, name) for m in self._rust_in_scope(scope_id)[1:])
        return list(dict.fromkeys(m for m in brought if m))

    def _rust_scope_module(self, name, scope_id):
        """The Rust module ``name`` that a function around ``scope_id``
        declares, innermost first, else that its module does; None for
        none."""
        for scope in self._scopes(scope_id):
            child = self._rust_child(scope, name)
            if child or self._rust_module(scope) == scope:
                return child
        return None

    def _rust_reached(self, paths, scope_id):
        """The project definitions each of the Rust `use` paths ``paths``
        written in ``scope_id`` leads to through the module tree, where
        every one of them does; ``"external"`` where each leads outside the
        project, None where one is not told."""
        found, outside = [], 0
        for path in sorted(paths):
            rooted = self._rust_rooted(path, scope_id) or self._rust_scoped(
                path, scope_id
            )
            reached = rooted and self._rust_path(path.split("::"), scope_id)
            if not reached:
                return None
            if reached == _EXTERNAL:
                outside += 1
            else:
                found += [hit for hit in reached if hit not in found]
        if outside:
            return _EXTERNAL if outside == len(paths) else None
        return found

    def _rust_globbed(self, name, scope_id, modules):
        """What ``name`` names in the Rust ``modules`` that glob `use`s
        bring to ``scope_id``: an item of one, or what a `use` there binds,
        that ``scope_id`` can see."""
        found = []
        for module in modules:
            hit = self._scope_member(module, name)
            used = name in self.builder.imports.get(module, {})
            visibility = (
                self.builder.rust_use_visibility.get((module, name))
                if used
                else self.builder.rust_visibility.get(hit)
            )
            if (
                hit
                and hit not in found
                and self._rust_visible(visibility, module, scope_id)
            ):
                found.append(hit)
        return found

    def _rust_visible(self, visibility, owner, scope_id):
        """Whether an item or a `use` of the Rust module ``owner``, of
        ``visibility`` (see `_rust_visibility`, None for `pub`), can be named
        from ``scope_id``: a private one in ``owner`` and the modules inside
        it, `pub(super)` in its parent's, `pub(crate)` and `pub(in path)` in
        its crate."""
        if visibility in (None, "pub"):
            return True
        if visibility == "crate" or visibility.startswith("in "):
            return self._unit(owner, "rust") == self._unit(scope_id, "rust")
        reach = self._rust_parent(owner) if visibility == "super" else owner
        module = self._rust_module(scope_id)
        while module and module != reach:
            module = self._rust_parent(module)
        return module is not None and module == reach

    def _rust_scoped(self, path, scope_id):
        """Whether the Rust path ``path`` written in ``scope_id`` starts at a
        name in scope there that is the project's: a module or an item that
        ``scope_id``, its module or a module its globs bring declares, or one
        a `use` of those binds into the project; where a `use` path may
        start too."""
        head = path.split("::")[0]
        return any(
            self._rust_child(m, head)
            or f"{m}::{head}" in self.nodes
            or (
                head in self._uses(m)
                and self._imported(head, m) not in (None, _EXTERNAL)
            )
            for m in dict.fromkeys([scope_id, *self._rust_in_scope(scope_id)])
        )

    def _crate_named(self, name, scope_id):
        """The crate that ``name`` names in ``scope_id``: the one an `extern
        crate ... as name` at its crate's root renames, else the package
        its crate's manifest, or the workspace's, gives the dependency key
        ``name``, else ``name``."""
        unit = self._unit(scope_id, "rust")
        for root in self.rust_roots:
            renamed = self.builder.rust_externs.get(root, {})
            if name in renamed and self._unit(root, "rust") == unit:
                return renamed[name]
        path = scope_id.split("::")[0][5:]
        manifests = [
            self.builder.cargo_renames[f]
            for f in _ancestors(path)
            if f in self.builder.cargo_renames
        ]
        own = manifests[0][0].get(name, name) if manifests else name
        shared = (m[1][name] for m in manifests if name in m[1])
        return own if own is not None else next(shared, name)

    def _uses(self, scope_id):
        """``{name: path}`` of what the imports in force at ``scope_id``
        bind: its file's, but in Rust the `use`s of the functions around it
        and of its module, and through `use super::*` the parent module's,
        as the module would write them."""
        file_id = scope_id.split("::")[0]
        if self.lang(file_id) != "rust":
            return self.builder.imports.get(file_id, {})
        if scope_id in self.rust_uses:
            return self.rust_uses[scope_id]
        module, found = self._rust_module(scope_id), {}
        for scope in self._scopes(scope_id):
            for name, path in self.builder.imports.get(scope, {}).items():
                found.setdefault(name, path)
            if scope == module:
                break
        parent = "super::*" in found and self._rust_parent(module)
        for name, path in self._uses(parent).items() if parent else ():
            if f"{module}::{name}" in self.nodes:
                # The module's own item shadows what a glob brings.
                continue
            head = path.split("::")[0]
            if head in ("self", "super") or self._rust_child(parent, head):
                path = f"super::{path.removeprefix('self::')}"
            found.setdefault(name, path)
        self.rust_uses[scope_id] = found
        return found

    def _use_crate(self, path, scope_id):
        """Where a Rust path written in ``scope_id`` leads: ``(crate
        directory,)``, ``()`` for a crate outside the project, None when
        the path starts at a module or type in scope. A name that a `use`
        in force there binds leads where that `use` does."""
        root = path.split("::")[0]
        root = self._uses(scope_id).get(root, root).split("::")[0]
        root = self._crate_named(root, scope_id)
        if root in ("crate", "self", "super"):
            return (self._unit(scope_id, "rust"),)
        if root in self.builder.rust_crates:
            return (self.builder.rust_crates[root],)
        # A module or an item of that name in scope where the path is
        # written; a crate otherwise.
        held = any(
            self._rust_child(module, root) or f"{module}::{root}" in self.nodes
            for module in self._rust_in_scope(scope_id)
        )
        return None if held else ()

    def _rust_in_scope(self, scope_id):
        """The Rust modules whose items are in scope by name in
        ``scope_id``: its own module, and those its glob `use`s bring,
        through the globs of each of those too that ``scope_id`` can see."""
        if scope_id not in self.rust_in_scope:
            self.rust_in_scope[scope_id] = self._rust_brought([scope_id], scope_id)
        return self.rust_in_scope[scope_id]

    def _rust_brought(self, starts, scope_id):
        """The Rust modules of ``starts`` and those their glob `use`s bring,
        through the globs of each of those too: of the `use`s, those
        ``scope_id`` can see, all of its own."""
        modules, todo = [], list(starts)
        while todo:
            at = todo.pop()
            module = self._rust_module(at)
            if module in modules:
                continue
            modules.append(module)
            for name, used in self._uses(at).items():
                seen = at == scope_id or self._rust_visible(
                    self.builder.rust_use_visibility.get((at, name)), at, scope_id
                )
                if used.endswith("::*") and seen:
                    target, rest = self._rust_walk(used[:-3].split("::"), at)
                    if target and not rest:
                        todo.append(target)
        return modules

    def _rust_bare(self, name, candidates, source_id):
        """What the bare Rust name ``name`` in ``source_id`` names by the
        innermost scope, a function's block or its module, that has it,
        where that is what the scope's globs bring: an item the scope
        declares comes first, then its `use`s, then its globs. Several where
        its globs bring more than one; None where the scope declares or
        `use`s the name, or no scope has it."""
        module = self._rust_module(source_id)
        for scope in self._scopes(source_id):
            if scope != module and self.nodes[scope]["type"] not in (
                "Function",
                "Method",
            ):
                continue
            uses = self.builder.imports.get(scope, {})
            if name in uses or any(c.rpartition("::")[0] == scope for c in candidates):
                return None
            targets = []
            for path in uses.values():
                walked = path.endswith("::*") and self._rust_walk(
                    path[:-3].split("::"), scope
                )
                if walked and walked[0] and not walked[1]:
                    targets.append(walked[0])
            reached = self._rust_brought(targets, source_id)
            globbed = self._rust_globbed(name, source_id, reached)
            # A call names a value, never a module.
            if found := [g for g in globbed if self._rust_module(g) != g]:
                return found
            if scope == module:
                return None
        return None

    def _js_module(self, specifier, file_id, required=False):
        """Resolve a module specifier written in ``file_id``, by a `require`
        if ``required``: ``[file ID]`` for a project module, ``[]`` for one
        of the project's that cannot be located, ``"external"`` for a
        declared dependency, None for anything else."""
        folder = posixpath.dirname(file_id[5:])
        if specifier.startswith("."):
            found = self._js_file(posixpath.join(folder, specifier))
            return [found] if found else []
        aliased = self._js_alias(specifier, folder)
        if aliased is not None:
            return aliased
        if specifier[:1] == "#":
            return self._js_imported(specifier, folder, file_id, required)
        # Node resolves its own modules before any package.
        if _node_module(specifier):
            return _EXTERNAL
        if specifier.partition("/")[0] == "$lib":
            # SvelteKit: `$lib` is `src/lib` beside the nearest configuration.
            for above in _ancestors(folder):
                if self._js_file(posixpath.join(above, "svelte.config")):
                    library = posixpath.join(above, "src/lib", specifier[5:])
                    found = self._js_file(library)
                    return [found] if found else []
        # The package of the project the specifier names, the first in the
        # walk where it names several
        parts = specifier.split("/")
        names = ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
        named = [name for name in names if name in self.builder.js_packages]
        for name in sorted(named, key=self.package_order.__getitem__)[:1]:
            folder, exports = self.builder.js_packages[name]
            subpath = f".{specifier[len(name) :]}"
            if exports is None:
                # Without `exports` a subpath is a path inside the package.
                found = self._js_file(posixpath.join(folder, subpath))
                return [found] if found else []
            if not isinstance(exports, dict) or not all(k[:1] == "." for k in exports):
                exports = {".": exports}
            if self._ts_resolution(file_id) in ("node10", "classic"):
                # TypeScript's node10 and classic resolution read no
                # `exports`.
                found = self._js_file(posixpath.join(folder, subpath))
                return [found] if found else []
            path = self._mapped(exports, subpath, file_id, required)
            found = path and self._js_target(posixpath.join(folder, path), file_id)
            return [found] if found else []
        package = "/".join(specifier.split("/")[: 2 if specifier[0] == "@" else 1])
        if package in self.builder.js_dependencies:
            return _EXTERNAL
        # A name no package can have (`$lib`, `#internal`, `@/x`) is an alias
        # of the project's own.
        return None if _NPM_PACKAGE_RE.fullmatch(package) else []

    def _mapped(self, mapping, subpath, file_id, required):
        """The path ``subpath`` maps to in a package's `exports` or `imports`
        ``mapping`` under the conditions of ``file_id``, as Node's
        PACKAGE_EXPORTS_RESOLVE and PACKAGE_IMPORTS_RESOLVE read it: by its
        own key, else of the patterns it matches the one of the longest
        prefix; None where none does."""
        if subpath in mapping and "*" not in subpath:
            key, matched = subpath, ""
        else:
            keys = sorted(
                (k for k in mapping if k.count("*") == 1),
                key=lambda k: (-len(k.partition("*")[0]), -len(k)),
            )
            key = next((k for k in keys if _star_match(k, subpath) is not None), None)
            matched = key and _star_match(key, subpath)
        path = key and _export_target(
            mapping[key], self._js_conditions(file_id, required)
        )
        return path and path.replace("*", matched)

    def _js_imported(self, specifier, folder, file_id, required):
        """Resolve a `#` specifier through the `imports` of the package.json
        nearest above ``folder``: a path in that package, or a package the
        target names; ``[]`` where none maps it."""
        scope = next(
            (
                a or "."
                for a in _ancestors(folder)
                if (a or ".") in self.builder.js_entries
            ),
            None,
        )
        imports = self.builder.js_imports.get(scope)
        path = imports and self._mapped(imports, specifier, file_id, required)
        if not path or path[:1] == "#":
            return []
        if path.startswith("./"):
            found = self._js_target(posixpath.join(scope, path), file_id)
            return [found] if found else []
        return self._js_module(path, file_id, required)

    def _js_alias(self, specifier, folder):
        """Resolve ``specifier`` through the `paths` and `baseUrl` of the
        tsconfig nearest above ``folder``: ``[file ID]``, ``[]`` when a
        pattern matches but no target is a project module, None when the
        configuration says nothing about it."""
        config = self._ts_config(folder)
        if config is None:
            return None
        base, paths, *_ = self._ts_options(config, set())
        # The pattern with the longest prefix wins.
        for pattern in sorted(paths or (), key=lambda p: -len(p.partition("*")[0])):
            matched = _star_match(pattern, specifier)
            if matched is None:
                continue
            targets = (t.replace("*", matched) for t in paths[pattern])
            found = (self._js_file(target) for target in targets)
            return [target for target in found if target][:1]
        found = base is not None and self._js_file(posixpath.join(base, specifier))
        return [found] if found else None

    def _ts_resolution(self, file_id):
        """The module resolution TypeScript reads the file ``file_id`` with,
        as "node16" (node16 to nodenext), "bundler", "node10" or "classic":
        its tsconfig's `moduleResolution`, else what its `module` implies,
        else the default of the TypeScript the nearest package.json depends
        on (6 and on: bundler; before: node10 for CommonJS, else classic),
        the data's 6 where none says. None for a JavaScript file, and where
        a config the tsconfig extends is not read."""
        if not file_id.endswith(_JSTS_EXTENSIONS[:4]):
            return None
        folder = posixpath.dirname(file_id[5:])
        config = self._ts_config(folder)
        resolution, module = (
            self._ts_options(config, set())[3:] if config else (None, None)
        )
        family = ("node16", "node18", "node20", "nodenext")
        if resolution == "?":
            return None
        if resolution:
            return {"node": "node10", **dict.fromkeys(family, "node16")}.get(
                resolution, resolution
            )
        if module in family:
            return "node16"
        version = next(
            (
                self.builder.ts_versions[above or "."]
                for above in _ancestors(folder)
                if (above or ".") in self.builder.ts_versions
            ),
            "6",
        )
        major = re.search(r"\d+", version)
        if module == "preserve" or not major or int(major[0]) >= 6:
            return "bundler"
        return "node10" if module in (None, "commonjs") else "classic"

    def _js_conditions(self, file_id, required):
        """The `exports` conditions a module of ``file_id`` is looked up with,
        where its import is a `require` with ``required``, an `import()` with
        None: Node's for JavaScript; for TypeScript `types`, its config's
        `customConditions`, and `node` but under bundler resolution, a static
        import of a file of CommonJS format (`.cts`, or `.ts` outside a `type:
        module` package) being a `require` under the node16 family."""
        if not file_id.endswith(_JSTS_EXTENSIONS[:4]):
            mode = "require" if required else "import"
            return {"node", "module-sync", "default", mode}
        resolution = self._ts_resolution(file_id)
        path = file_id[5:]
        commonjs = path.endswith(".cts") or (
            path.endswith((".ts", ".tsx")) and not self.builder.js_module(path)
        )
        if resolution == "node16" and commonjs and required is False:
            required = True
        config = self._ts_config(posixpath.dirname(path))
        custom = config and self._ts_options(config, set())[2]
        node = set() if resolution == "bundler" else {"node"}
        return {
            "types",
            "default",
            "require" if required else "import",
            *node,
            *(custom or ()),
        }

    def _ts_config(self, folder):
        """The tsconfig or jsconfig nearest at or above ``folder``; None for
        none."""
        names = (
            posixpath.join(above, name)
            for above in _ancestors(folder)
            for name in ("tsconfig.json", "jsconfig.json")
        )
        return next((name for name in names if name in self.builder.ts_configs), None)

    def _ts_options(self, config, seen):
        """The effective ``(baseUrl directory, {pattern: [target paths]},
        customConditions, moduleResolution, module)`` of the tsconfig at
        ``config``, each None when neither it nor a config it extends sets
        it, the last two "?" where a config it extends is not read."""
        extends, base, paths, *rest = self.builder.ts_configs[config]
        seen.add(config)
        folder = posixpath.dirname(config)
        if base is not None:
            base = posixpath.normpath(posixpath.join(folder, base))
        if paths is not None:
            # Targets are relative to `baseUrl`, else to the config's folder.
            paths = {
                pattern: [posixpath.join(base or folder, t) for t in targets]
                for pattern, targets in paths.items()
            }
        for parent in extends:
            parent = posixpath.normpath(posixpath.join(folder, parent))
            if not parent.endswith(".json"):
                parent += ".json"
            if parent not in self.builder.ts_configs:
                # What a config not read sets is not known.
                rest[1:] = [v or "?" for v in rest[1:]]
            elif parent not in seen:
                inherited = self._ts_options(parent, seen)
                base = inherited[0] if base is None else base
                paths = inherited[1] if paths is None else paths
                rest = [
                    own if own is not None else up
                    for own, up in zip(rest, inherited[2:])
                ]
        return base, paths, *rest

    def _js_target(self, path, file_id):
        """File ID of the module an `exports` target names: the file itself,
        no extension added, and for a TypeScript importer the TypeScript
        file a `.js`, `.mjs` or `.cjs` one stands for."""
        path = posixpath.normpath(path)
        written = [path]
        if file_id.endswith(_JSTS_EXTENSIONS[:4]):
            stem, dot, extension = path.rpartition(".")
            sources = {
                "js": ("ts", "tsx", "d.ts"),
                "mjs": ("mts", "d.mts"),
                "cjs": ("cts", "d.cts"),
            }
            written += [
                f"{stem}.{other}" for other in sources.get(extension, ()) if dot
            ]
        return next((f"FILE:{w}" for w in written if f"FILE:{w}" in self.nodes), None)

    def _js_file(self, path, entry=True):
        """File ID of the JS/TS module at ``path``, written with any or no
        extension, or naming a directory: the module its package.json names
        if ``entry``, else its index module; a declaration file where no
        module is there."""
        path = posixpath.normpath(path)
        if re.search(r"\.[cm]?tsx?$", path) and f"FILE:{path}" in self.nodes:
            # A TypeScript extension names the file itself.
            return f"FILE:{path}"
        stem = re.sub(r"\.[cm]?[jt]sx?$", "", path)

        def module(base):
            base = posixpath.normpath(base)
            for extension in (*_JSTS_EXTENSIONS, ".d.ts", ".d.mts", ".d.cts"):
                if f"FILE:{base}{extension}" in self.nodes:
                    return f"FILE:{base}{extension}"
            return None

        entries = self.builder.js_entries.get(path, ()) if entry else ()
        named = (
            self._js_file(posixpath.join(path, written), False)
            for written in entries
            if isinstance(written, str)
        )
        return (
            module(stem)
            or next(filter(None, named), None)
            or module(posixpath.join(stem, "index"))
        )

    def _dart_file(self, uri, file_id):
        """File ID of the Dart library that ``uri`` names in a directive of
        ``file_id``; None for one outside the project or not indexed."""
        if uri.startswith("package:"):
            package, _, path = uri[len("package:") :].partition("/")
            if package not in self.builder.dart_packages:
                return None
            path = posixpath.join(self.builder.dart_packages[package], "lib", path)
        else:
            path = posixpath.normpath(posixpath.join(file_id[5:], "..", uri))
        return f"FILE:{path}" if f"FILE:{path}" in self.nodes else None

    def _dart_directives(self, library, keyword):
        """``(library files, prefix, names shown, names hidden)`` of each
        directive ``keyword`` of the Dart library ``library``: those of the
        files it names that are the project's, None where no package of the
        project is known and it names one."""
        if (library, keyword) in self._dart_found:
            return self._dart_found[library, keyword]
        found = self._dart_found[library, keyword] = []
        for kind, uris, prefix, show, hide in self.builder.dart_directives.get(
            library, ()
        ):
            if kind == keyword:
                files = [self._dart_file(uri, library) for uri in uris]
                unknown = not self.builder.dart_packages and any(
                    uri.startswith("package:") for uri in uris
                )
                found.append(
                    (None if unknown else [f for f in files if f], prefix, show, hide)
                )
        return found

    def _dart_exports(self, library, name, seen):
        """Top-level definitions the Dart library ``library`` holds under
        ``name``: its own, its parts' and those of the libraries it exports."""
        if library in seen:
            return set()
        seen.add(library)
        units = [library]
        for files, *_ in self._dart_directives(library, "part"):
            units += files or ()
        hits = {self.members.get(unit, {}).get(name) for unit in units} - {None}
        for files, _, show, hide in self._dart_directives(library, "export"):
            if name in show if show else name not in hide:
                for exported in files or ():
                    hits |= self._dart_exports(exported, name, seen)
        return hits

    def _dart_named(self, name, scope_id, prefix=None):
        """Top-level definitions that ``name`` can stand for in the Dart
        library ``scope_id`` belongs to: those the library declares, else
        those its imports bring in, or with ``prefix`` those the imports of
        that prefix do. None where an import cannot be placed in or outside
        the project, so that it may bring the name in."""
        file_id = scope_id.split("::")[0]
        library = self.dart_parts.get(file_id, file_id)
        seen = {
            exported
            for files, *_ in self._dart_directives(library, "export")
            for exported in files or ()
        }
        hits = set() if prefix else self._dart_exports(library, name, seen)
        if hits or name.startswith("_"):
            return sorted(hits)
        for files, named, show, hide in self._dart_directives(library, "import"):
            if named == prefix and (name in show if show else name not in hide):
                if files is None:
                    return None
                for imported in files:
                    hits |= self._dart_exports(imported, name, set())
        return sorted(hits)

    def _dart_prefixed(self, prefix, file_id):
        """Files of the libraries the Dart library ``file_id`` belongs to
        imports as ``prefix``, with those they export: ``"external"`` where
        none is the project's, None where no import has the prefix."""
        library = self.dart_parts.get(file_id, file_id)
        imports = [
            files
            for files, named, *_ in self._dart_directives(library, "import")
            if named == prefix
        ]
        if not imports or None in imports:
            return None
        pending = [imported for files in imports for imported in files]
        found = []
        while pending:
            unit = pending.pop()
            if unit not in found:
                found.append(unit)
                for keyword in ("part", "export"):
                    for files, *_ in self._dart_directives(unit, keyword):
                        pending += files or ()
        return found or _EXTERNAL

    def _dart_root(self, root, steps, scope_id):
        """Value of the name ``root`` that a Dart origin path written in
        ``scope_id`` starts with, and the steps left after it. The name is
        a type written before `()`, or is read in scope: a definition of an
        enclosing scope, a member the enclosing class has or inherits, what
        the library declares or imports, or an import prefix, which names a
        definition together with the step after it. A name none of these
        has is from outside the project."""
        name, generic, _ = root.partition("<")
        if steps[:1] == ["()"]:
            held = self._type_argument(name, scope_id, None)
            if held is not False:
                return held and ("returned", held), steps
            found = self._type_named(name, scope_id)
            outside = found == _EXTERNAL
            made = found and (generic or outside)
            made = made and self._returned_type(root, scope_id)
            if isinstance(made, tuple) and len(made) > 2:
                # A generic type written with its arguments holds them; a
                # standard type has the members the library's data gives it.
                return (made if outside else ("def", found, made[2])), steps
            name = name.rpartition("::")[2]
            if found == _EXTERNAL and (generic or name in self.impl_types):
                # Its methods are its own or those of an extension on it;
                # what it holds may be of the type arguments it is written
                # with.
                return ("outside", name), steps
            return found and (found if found == _EXTERNAL else ("def", found)), steps
        hit = self._lexical(name, scope_id, skip_classes=False)
        owner = self._enclosing_class(scope_id)
        if not hit and owner:
            hit = self._find_member(owner, name)[0]
        if not hit:
            imported = self._imported(name, scope_id)
            if imported and imported != _EXTERNAL and steps:
                named = self._dart_named(steps[0], scope_id, name)
                steps = steps[1:]
            else:
                named = [] if imported else self._dart_named(name, scope_id)
            if named is None or len(named) > 1:
                return None, steps
            if not named:
                return _EXTERNAL, steps
            hit = named[0]
        return ("def", hit), steps

    def _exported(self, file_id, name, seen):
        """Definition a file or an object makes available as ``name``: its
        own, one it re-exports from another module or holds by a name it is
        written with, or a module it re-exports whole under that name. What
        a file exports whole, `export = X`, is its default, ``=``, and holds
        every other name it exports. A JavaScript function's declarations
        are no properties of it."""
        if self._js_function(file_id):
            given = self._assigned(file_id, name)
            return given[0] if len(given) == 1 else None
        entries = self.builder.reexports.get(file_id, ())
        whole = [e for e in entries if e[0] == "="]
        if whole:
            # A module that exports itself whole, through others or not,
            # exports nothing so.
            if ("=", file_id) in seen:
                return None
            seen.add(("=", file_id))
            held = self._given(whole[-1], file_id, "=", seen)
            if name in ("=", "default"):
                return held
            # What is exported whole is a class or a function, whose own
            # members are its static ones; a CommonJS module exports what
            # is assigned to `exports.name` besides.
            hit = held and self._find_member(held, name, static=True)[0]
        elif name == "=":
            # A module that exports nothing whole is its exports object.
            return file_id
        elif file_id not in self.builder.commonjs:
            hit = self.members.get(file_id, {}).get(name)
        else:
            # A CommonJS module exports what its exports object holds: none
            # of its own top-level names besides.
            hit = None
        if hit or file_id in seen:
            return hit
        seen.add(file_id)
        # `export *` passes on neither a default nor what is exported whole.
        star = "*" if name not in ("default", "=") else None
        for entry in entries:
            if entry[0] not in (star, name):
                continue
            held = self._given(entry, file_id, name, seen)
            # `{ ...a }` holds what `a` does.
            hit = (
                held
                if entry[0] == name or entry[1]
                else held and self._find_member(held, name)[0]
            )
            if hit:
                return hit
        return None

    def _given(self, entry, file_id, name, seen):
        """What the export entry ``entry`` of ``file_id`` gives as ``name``:
        what the name it holds stands for where it is written, the entry's
        fourth element, or what the module it names exports as it, of `*`
        the module itself, of a `require`, ``=``, what the module exports
        whole. None where it is not known."""
        _, specifier, original, scope = entry
        if not specifier:
            return original and self._written(original, scope, seen)
        modules = self._js_module(specifier, file_id, original == "=")
        for module in modules if isinstance(modules, list) else ():
            if original == "*":
                return module
            hit = self._exported(module, original or name, seen)
            if hit:
                return hit
        return None

    def _assigned(self, owner_id, name, static=False):
        """The definitions of the member ``name`` of the JavaScript or
        TypeScript type or function ``owner_id``, under ``name`` and, given
        again, `name#2` and on: each a node, what a member that holds a name
        holds, or what a field typed `typeof f` holds. Of a class, those
        ``static`` says, the constructor's or its instances'; of a function,
        its static ones, a declaration in its body being none."""
        node = self.nodes[owner_id]
        function = self._js_function(owner_id)
        split = function or node["type"] == "Class" and not node["metadata"].get("kind")
        found, key, count = [], name, 1
        while hit := self.members.get(owner_id, {}).get(key):
            metadata = self.nodes[hit]["metadata"]
            if not split or bool(metadata.get("static")) == (static or function):
                value = (
                    metadata.get("value")
                    if self.nodes[hit]["type"] == "Constant"
                    else None
                )
                held = value and self._returned_type(f"typeof {value}", owner_id)
                found.append(held[1] if held else hit)
            count += 1
            key = f"{name}#{count}"
        written = self.builder.fields.get(owner_id, {}).get(name, "")
        held = not found and _TS_TYPEOF_RE.fullmatch(written)
        held = held and self._returned_type(written, owner_id)
        return found + [held[1]] if held else found

    def _ahead(self, index):
        """What the call at edge ``index`` returns, which is not resolved
        yet: one written inside the arguments of the call being resolved."""
        edge = self.builder.edges[index]
        metadata = edge.get("metadata", {})
        if index in self._ahead_of or not edge["target"].startswith("REF:"):
            return None
        self._ahead_of.add(index)
        value = self.resolve(
            edge["target"][4:],
            edge["source"],
            metadata.get("origin"),
            metadata.get("args"),
            within=True,
        )
        self._ahead_of.discard(index)
        return value

    def _held_function(self, node_id):
        """The function of the project that the JavaScript constant
        ``node_id`` holds, the one its value names or the one a call it is
        given returns, where that is told: a call of the constant calls it.
        None elsewhere."""
        node = self.nodes[node_id]
        written = node["type"] == "Constant" and node["metadata"].get("value") or ""
        call = re.fullmatch(r"([\w$]+(?:\.[\w$]+)*)\s*\([^()]*\)\s*;?", written)
        if self.lang(node_id) != "typescript" or node_id in self._holding:
            return None
        self._holding.add(node_id)
        if call:
            held = self._returned_function(self.resolve(call[1], node_id, within=True))
        else:
            held = self._named(node_id)
        self._holding.discard(node_id)
        return held

    def _returned_function(self, value):
        """The JavaScript function of the project that ``value``, what a
        call returns, is; None where it is none."""
        held = isinstance(value, tuple) and value[0] == "def" and value[1]
        return held if held and self._js_function(held) else None

    def _js_function(self, node_id):
        """Whether ``node_id`` is a JavaScript or TypeScript function."""
        node = self.nodes.get(node_id)
        return bool(
            node
            and self.lang(node_id) == "typescript"
            and node["type"] in ("Function", "Method")
        )

    def _written(self, name, scope_id, seen):
        """What ``name`` written in the JS/TS ``scope_id`` stands for: the
        definition in scope, else what an import binds it to, the module
        itself for one bound whole."""
        hit = self._lexical(name, scope_id, skip_classes=True)
        required = self.builder.required_names.get(scope_id.split("::")[0], {})
        if hit and self.nodes[hit]["metadata"].get("lineno") == required.get(name):
            # The constant a `require` binds is the module.
            hit = None
        imported = None if hit else self._imported(name, scope_id)
        exported = self._exported_as(name, scope_id)
        for module in imported if isinstance(imported, list) else ():
            hit = hit or self._exported(module, exported, seen)
        return hit

    def _resolve_bare(self, name, candidates, source_id, lang, callee):
        file_id = source_id.split("::")[0]
        if lang == "dotnet":
            # A bare C# call reaches a local function, a member of an
            # enclosing type or one of a type named by `using static`, and
            # no method of another type.
            hit = self._inherited(name, source_id)
            hit = hit and self._overload_by_arity(hit)
            types = self.dotnet_paths | set(self.standard["dotnet"].data["types"])
            targets = [
                *self._dotnet_imports(file_id).values(),
                *self._dotnet_imported(source_id, True, types),
            ]
            for target in targets:
                if hit or self._candidates_now:
                    break
                held = self._type_named(target, file_id)
                if held not in (None, _EXTERNAL):
                    hit = self._find_member(held, name)[0]
                    hit = hit and self._overload_by_arity(hit, static=True)
            if not hit:
                self._result = _OPAQUE
                return None, _UNRESOLVED if self._candidates_now else _EXTERNAL
            return hit, "regex"
        if lang in _IMPLICIT_SELF_LANGS:
            return self._resolve_self_call(name, candidates, source_id)
        if lang == "clang":
            # The preprocessor expands a macro before any name is looked up:
            # the one macro of the name in reach is the call's where every
            # build of the call defines it, or where every other definition
            # of the name is built only where it is, the macro then naming
            # it. Which of several macros is defined last is not told. A
            # macro not defined where the call is written is no macro there.
            candidates = [
                c
                for c in candidates
                if self.nodes[c]["metadata"].get("kind") != "macro"
                or self._macro_in_force(c, source_id)
            ]
            macros = [
                c
                for c in candidates
                if self.nodes[c]["metadata"].get("kind") == "macro"
                and self._in_reach(c, source_id)
            ]
            held = set(macros and self.nodes[macros[0]]["metadata"].get("guards", ()))
            caller = self._caller_guards(source_id)
            rivals = [
                c
                for c in candidates
                if c not in macros
                and not held <= set(self._effective_guards(c, source_id))
            ]
            if len(macros) == 1 and (held <= set(caller) or not rivals):
                return macros[0], "regex"
        found = lang == "rust" and self._rust_bare(name, candidates, source_id)
        if found:
            if len(found) == 1:
                return found[0], "regex"
            self._candidates_now = sorted(found)
            return None, _UNRESOLVED
        explicit = lang in _EXPLICIT_RECEIVER_LANGS
        # A Java class's members, inherited ones too, are looked up class by
        # class below, the innermost first (JLS 15.12.1).
        hit = self._lexical(name, source_id, skip_classes=explicit or lang == "java")
        if (
            hit
            and lang == "clang"
            and self.nodes[hit]["metadata"].get("kind") == "macro"
            and not self._macro_in_force(hit, source_id)
        ):
            hit = None
        if hit and lang == "clang":
            hit = self._c_overload(hit, candidates, source_id)
            if not hit:
                return None, _UNRESOLVED
        guards = hit and lang == "clang" and self._effective_guards(hit, source_id)
        caller = guards and self._caller_guards(source_id)
        if guards and not set(guards) <= set(caller):
            # A definition that an `#if` arm the call is not under alone
            # builds is one of those of its name the call may reach.
            path = self._c_path(hit)
            fits = [
                c
                for c in self._fitting(candidates, source_id)
                if self._c_path(c) == path
            ]
            if fits != [hit]:
                self._candidates_now = sorted(fits) or None
                return (fits[0], "regex") if len(fits) == 1 else (None, _UNRESOLVED)
        if hit:
            return hit, "regex"
        owner = None if explicit else self._enclosing_class(source_id)
        if owner:
            hit, complete = self._find_member(owner, name)
            if hit:
                return hit, "regex"
            if complete == _SEVERAL:
                return None, _UNRESOLVED
        inherited = False
        around = filter(self._is_class, self._scopes(owner))
        for scope in around if lang == "java" else ():
            # The innermost class around a Java call that has the method is
            # the one called; one with a supertype from outside may have it
            # from there.
            hit, complete = self._find_member(scope, name)
            if hit:
                return (None, _UNRESOLVED) if inherited else (hit, "regex")
            inherited = inherited or not complete
        if inherited:
            self._result = self._std_inherited(owner, name) or _OPAQUE
            return None, _EXTERNAL
        free = [c for c in candidates if self._is_top_level(c)]
        imports = self._uses(source_id)
        if lang in _IMPORT_SCOPED_LANGS and name not in imports:
            # Only a glob import can bring the name in, from the crate it
            # names; one that starts at something in scope stays in this one.
            own = (self._unit(file_id, lang),)
            crates = set()
            for path in imports:
                if path.endswith("::*"):
                    found = self._use_crate(path, source_id)
                    crates.update(own if found is None else found)
            free = [c for c in free if self._unit(c, lang) in crates]
        if lang == "go":
            # A bare Go call stays inside its package, which is its
            # directory, and one of a variable calls the value it holds.
            package = self._unit(file_id, lang)
            free = [
                c
                for c in free
                if self._unit(c, lang) == package
                and self.nodes[c]["type"] != "Constant"
            ]
        if lang == "clang":
            free = [c for c in free if self._in_reach(c, source_id)]
        if lang == "dart":
            # A bare Dart call that reaches no member of the enclosing class
            # reaches what its library declares or imports, or leaves the
            # project.
            named = self._dart_named(name, file_id)
            free = free if named is None else named
            if not free:
                self._result = _OPAQUE
                return None, _EXTERNAL
        if free:
            hit = self._decide(free, source_id)
            if hit and lang == "clang":
                hit = self._c_overload(hit, free, source_id)
            return (hit, "regex") if hit else (None, _UNRESOLVED)
        if lang in _FREE_BARE_CALL_LANGS:
            self._result = _OPAQUE
            return None, _EXTERNAL
        if explicit or lang == "clang":
            # Without a receiver a method is out of reach.
            candidates = [c for c in candidates if self.nodes[c]["type"] != "Method"]
        if lang == "rust":
            # A glob import brings, before the prelude's names, the items
            # of a type or a module its path leads to, and what the `use`s
            # of such a module bind, through the globs of each too.
            globs = [path[:-3] for path in imports if path.endswith("::*")]
            reached = set()
            for path in globs:
                held = self._rust_path(path.split("::"), source_id)
                reached.update(held if held != _EXTERNAL else ())
            exact = [
                c
                for c in candidates
                if c.rpartition("::")[0] in reached
                and self._rust_visible(
                    self.builder.rust_visibility.get(c),
                    c.rpartition("::")[0],
                    source_id,
                )
            ]
            globbed = self._rust_globbed(
                name, source_id, self._rust_in_scope(source_id)[1:]
            )
            exact = list(dict.fromkeys(exact + globbed))
            if len(exact) == 1:
                return exact[0], "regex"
            globbed = {path.rpartition("::")[2] for path in globs}
            candidates = [
                c
                for c in candidates
                if self._is_top_level(c) or self._name(c.rpartition("::")[0]) in globbed
            ]
        if lang in _SCRIPT_GLOBAL_LANGS:
            # A bare name is no member of a class, nor what another module
            # declares.
            named = [c for c in candidates if self._named_in(c, file_id)]
            return self._resolve_member(named, lang, callee, top_level=True)
        return self._resolve_member(candidates, lang, callee)

    def _resolve_self_call(self, name, candidates, source_id):
        """Resolve a call made on an implicit `self`: to a member of the
        enclosing type or of what it inherits, else to a top-level
        definition. In a module `self` is an instance of a type that
        includes it, so the call may also reach a member of one of those.
        A block may be run with another `self`, so a name that none of
        these has is not shown to come from outside the project there; in a
        class's method, outside any block, one the class's whole ancestry
        lacks is `Object`'s, where the library has it."""
        owner = self._enclosing_class(source_id)
        complete = False
        if owner:
            hit, complete = self._find_member(owner, name)
            source = self.nodes[source_id]["metadata"]
            if source_id == owner or source.get("is_singleton"):
                # In the body of a class and in a singleton method `self`
                # is the class.
                hit = self._find_member(owner, f"self.{name}")[0] or hit
            if hit:
                return hit, "self_type"
            if self.nodes[owner]["metadata"].get("kind") == "module":
                heirs = self._heirs(owner, set())
                hits = {self._find_member(heir, name)[0] for heir in heirs} - {None}
                if hits:
                    return (
                        (hits.pop(), "regex") if len(hits) == 1 else (None, _UNRESOLVED)
                    )
        # A top-level definition is there once its file is loaded.
        loaded = self._loaded(source_id.split("::")[0])
        free = [
            c
            for c in candidates
            if self._is_top_level(c) and c.split("::")[0] in loaded
        ]
        if free:
            hit = self._decide(free, source_id)
            return (hit, "regex") if hit else (None, _UNRESOLVED)
        if (
            complete is True
            and not self._in_block
            and source_id != owner
            and self.nodes[owner]["metadata"].get("kind") != "module"
        ):
            # A class's ancestry ends in `Object` and what the library says
            # it inherits, unless it derives from `BasicObject` itself; the
            # project may reopen them.
            root = (
                "BasicObject" if "BasicObject" in self._ancestors(owner) else "Object"
            )
            roots = [root, *self.standard["ruby"].data["types"][root]["bases"]]
            for held in roots:
                for declaration in self.constants.get(held, ()):
                    if hit := self.members.get(declaration, {}).get(name):
                        return hit, "self_type"
            if self.standard["ruby"].member(("outside", root, ()), name)[0]:
                self._result = _OPAQUE
                return None, _EXTERNAL
        return None, _UNRESOLVED

    def _loaded(self, file_id):
        """The Ruby files that ``file_id`` loads: itself and those it
        requires, directly or through one of them. A required path names
        the file whose path, without its extension, ends with it."""
        if file_id not in self.loaded:
            paths = {
                other: other[5:].rpartition(".")[0]
                for other in self.nodes
                if "::" not in other and self.lang(other) == "ruby"
            }
            found, todo = {file_id}, [file_id]
            while todo:
                for required in self.requires.get(todo.pop(), ()):
                    required = re.sub(r"^(?:\.{1,2}/)+", "", required).removesuffix(
                        ".rb"
                    )
                    for other, path in paths.items():
                        named = path == required or path.endswith(f"/{required}")
                        if named and other not in found:
                            found.add(other)
                            todo.append(other)
            self.loaded[file_id] = found
        return self.loaded[file_id]

    def _build_heirs(self):
        """Path of each Ruby type -> the types that name it as a base."""
        self.heirs = {}
        for declaration, bases in self.bases.items():
            for base in bases if self.lang(declaration) == "ruby" else ():
                named = self._type_named(base, declaration)
                if named not in (None, _EXTERNAL):
                    path = named.partition("::")[2]
                    self.heirs.setdefault(path, []).append(declaration)

    def _heirs(self, type_id, seen):
        """Ruby types that have ``type_id`` as a base, directly or through
        another type."""
        found = []
        for heir in self.heirs.get(type_id.partition("::")[2], ()):
            if heir not in seen:
                seen.add(heir)
                found += [heir, *self._heirs(heir, seen)]
        return found

    def _resolve_member(self, candidates, lang, callee, top_level=False):
        """Resolve a call through a receiver whose type is not known, or
        with ``top_level`` a bare call of a global."""
        pool = [
            c
            for c in candidates
            if (
                self._is_top_level(c)
                if top_level
                else self._has_members(c.rpartition("::")[0])
            )
        ]
        namespace = (
            not top_level
            and lang in _NAMESPACE_RECEIVER_LANGS
            and any(map(self._is_top_level, candidates))
        )
        # A call on a receiver whose type is not known is not pinned to the
        # one project definition of the member's name; the definitions that
        # fit stay its candidates. Nor is a bare call whose name the
        # standard library also has pinned to the project's definition, nor
        # dropped.
        if (
            not top_level
            or not pool
            or _split_callee(callee)[1] in _stdlib.members(lang)
        ):
            self._result = _OPAQUE
            if pool:
                self._candidates_now = sorted(pool)
            elif namespace:
                self._candidates_now = sorted(candidates)
            return None, _UNRESOLVED if pool or namespace else _EXTERNAL
        pool = tuple(pool)
        if pool not in self.shared_declaration:
            self.shared_declaration[pool] = self._shared_declaration(pool)
        hit = self.shared_declaration[pool]
        if hit:
            # Resolved by name alone: what the definition returns is not
            # evidence of this call's.
            self._name_only = True
        elif len(pool) > 1:
            self._candidates_now = sorted(pool)
        return (hit, "regex") if hit else (None, _UNRESOLVED)

    def _property_params(self, call, argument):
        """The parameters of the function the property `A:key` of an object
        literal passed to call ``call`` holds: those of the function type
        the property has in the type of the call's parameter `A`."""
        target, declared = self.declared_params.get(call, (None, ()))
        position, *path = argument.split(":")
        if int(position) >= len(declared):
            return ()
        written, scope = declared[int(position)].partition(":")[2].strip(), target
        for key in path:
            written, scope = self._property_type(written, key, scope)
            if not written:
                return ()
        return self._functional_params(written, scope)

    def _property_type(self, written, key, scope_id):
        """``(type, scope)`` of the property ``key`` of the TypeScript type
        ``written`` in ``scope_id``: an object type's, or a member's of the
        type it names, a method's as a function type; None for none."""
        if written.startswith("{") and written.endswith("}"):
            for part in _bracketed_parts(written[1:-1].replace(";", ",")):
                part = re.sub(r"^\s*readonly\s+", "", part)
                method = _ts_function_property(part) or (
                    re.match(r"\s*([\w$]+)\??\s*(\(.*\))\s*:\s*(.+)", part)
                )
                if method and isinstance(method, tuple) and method[0] == key:
                    name, signature = method
                    params, _, returned = signature[len(name) :].rpartition("):")
                    return f"{params}) => {returned.strip()}", scope_id
                if method and not isinstance(method, tuple) and method[1] == key:
                    return f"{method[2]} => {method[3]}", scope_id
                name, colon, held = part.partition(":")
                if colon and name.strip().rstrip("?") == key:
                    return held.strip(), scope_id
            return None, scope_id
        type_id = self._type_named(written, scope_id)
        if type_id in (None, _EXTERNAL):
            return None, scope_id
        member = self._find_member(type_id, key)[0]
        if member in self.builder.signatures:
            signature = self.builder.signatures[member][0]
            returned = signature["returns"] or "void"
            return f"({', '.join(signature['params'])}) => {returned}", member
        return self.builder.fields.get(type_id, {}).get(key), type_id

    def _rest(self, written, scope_id, receiver=None):
        """``("rest", value)`` of a TypeScript rest parameter declared
        ``written``: it takes in each later argument, as an element of its
        array."""
        element = _ts_type_parts(written)
        held = element[1][:1] if element[0] == "Array" else ()
        value = self._returned_type(held[0], scope_id, receiver) if held else None
        return "rest", value

    def _functional_params(self, written, scope_id):
        """``[values]`` of the parameters of the function ``written``'s
        type declares: the single abstract method of the project
        functional interface a Java writing names, or the parameters a
        TypeScript function type writes — ``(kind, node ID)`` values stand
        for an interface of that name — the last ``("rest", value)`` where
        that and every later parameter has the element type of a rest one.
        ``()`` where the writing is neither."""
        named = isinstance(written, str) and _ts_type_parts(written)
        if named and named[0] and self.lang(scope_id) == "typescript":
            # An alias stands for the type it names, its type arguments in
            # place of its parameters.
            alias = self._ts_alias(named[0], scope_id)
            if alias and alias[1] != written:
                return self._functional_params(_ts_given(alias, named[1]), scope_id)
        if (
            isinstance(written, str)
            and written.startswith(("(", "<"))
            and "=>" in written
            and self.lang(scope_id) == "typescript"
        ):
            # `(...) => ...` or, where an origin path carries the writing,
            # `<...>` in place of `(...)`: the parameter list the function
            # type takes, the parentheses a type is written in aside. A
            # union of function types, and a generic one, whose type
            # parameters a parameter's type can name, take none known; an
            # intersection is called as its one function type is.
            written = written.strip()
            compound = _ts_compound(written)
            if compound and compound[0] == "&":
                functions = [part.strip() for part in compound[1] if "=>" in part]
                if len(functions) != 1:
                    return ()
                return self._functional_params(functions[0], scope_id)
            while True:
                left, right = written[0], ")" if written[0] == "(" else ">"
                depth, close = 0, -1
                for at, char in enumerate(written):
                    if char == ">" and written[at - 1 : at] == "=":
                        continue
                    depth += (char == left) - (char == right)
                    if not depth:
                        close = at
                        break
                rest = written[close + 1 :].strip()
                if close < 0 or rest:
                    break
                written = written[1:close].strip()
                if not written.startswith(("(", "<")):
                    return ()
            if close < 0 or not rest.startswith("=>"):
                return ()
            # A parameter's name is what its list writes before a `:` no
            # bracket holds, and its type what follows that `:`; `this`
            # names the receiver's type, no parameter, and the type of a
            # rest parameter, whose `.` the origin path writes as `::`, is
            # the list's, not that of a parameter it takes in.
            values = []
            for param in _bracketed_parts(written[1:close] + ")"):
                # An origin path writes a rest parameter's `...` as `::::::`.
                rest = param.startswith(("...", "::::::"))
                param = param.removeprefix("...").removeprefix("::::::")
                name = (_bracketed_parts(param + ")", ":") or [""])[0]
                typed = param[len(name) :].lstrip()
                if param.startswith(":") or not typed.startswith(":"):
                    typed = ":"
                typed = typed[1:].strip() or "any"
                if rest:
                    return (*values, self._rest(typed, scope_id))
                if name != "this":
                    values.append(self._returned_type(typed, scope_id))
            return tuple(values)
        if self.lang(scope_id) != "java":
            return ()
        arguments = ()
        type_id = None
        if isinstance(written, tuple) and len(written) > 1:
            type_id = written[1]
        elif isinstance(written, str) and written:
            name, arguments, _ = _java_type_parts(written)
            types = [
                t
                for t in self._types("java", name)
                if self.nodes[t]["metadata"].get("kind") == "interface"
            ]
            type_id = types[0] if len(types) == 1 else None
        if (
            type_id is None
            or self.nodes[type_id]["metadata"].get("kind") != "interface"
        ):
            return ()
        abstract = [
            member
            for member in self.members.get(type_id, {}).values()
            if self.nodes[member]["type"] == "Method"
            and not self.nodes[member]["metadata"].get("body")
            and not self.nodes[member]["metadata"].get("static")
            and not self.nodes[member]["metadata"].get("default")
        ]
        if len(abstract) != 1:
            return ()
        node = self.nodes[type_id]
        generics = node["metadata"].get("generics", ())
        substitute = dict(zip(generics, arguments))
        params = self.nodes[abstract[0]]["metadata"].get("params", ())
        return tuple(
            self._returned_type(substitute.get(param, param), scope_id)
            for param in params
        )

    def _shared_declaration(self, pool):
        """The one definition every other one in ``pool`` overrides, if any."""
        if len(pool) == 1:
            return pool[0]
        owners = {c: c.rpartition("::")[0] for c in pool}
        ancestors = {owner: self._ancestors(owner) for owner in owners.values()}
        for candidate, owner in owners.items():
            base = self._name(owner)
            if all(c == candidate or base in ancestors[o] for c, o in owners.items()):
                return candidate
        return None

    def _resolve_origin(self, origin, source_id, seen=None):
        """Follow an origin path (see ``PythonParser._origin``) to a verdict.
        Returns None when the receiver of the last attribute is of unknown
        type. With ``seen``, the fixtures being evaluated, the path is read
        rather than called: the value it leads to is returned."""
        read = seen is not None
        lang = self.lang(source_id)
        python = lang == "python"
        # `path{0=origin}`: what the callable passed as argument 0 of the
        # call the path ends with returns.
        results = {}
        head, brace, passed = origin.rpartition("{")
        if brace and re.fullmatch(r"[\d?]+=[^,=]+(?:,[\d?]+=[^,=]+)*\}", passed):
            origin = head
            results = dict(item.split("=") for item in passed[:-1].split(","))
        value, steps, source = self._root_value(
            origin, source_id, seen if read else set()
        )
        if (
            value is None
            and python
            and not read
            and (reached := self._through_imports(origin, source_id, steps))
        ):
            self._candidates_now = reached
            return None, _UNRESOLVED
        if value is None and origin.startswith("@") and not read:
            return None, _UNRESOLVED
        reads_parameter = steps[:1] and steps[0].startswith("^")
        if value is None and origin.startswith("=") and not reads_parameter:
            # A value held in a variable is no definition. What calling it
            # returns is what the function it holds declares, `>Type`, and
            # not known without that; a parameter of that function is of a
            # type not known. A Python variable or parameter whose value the
            # parser could not tell may hold a project definition.
            declared = origin.partition(">")[2]
            held = declared and self._returned_type(declared, source_id)
            self._result = held or _OPAQUE
            return None, _UNRESOLVED if python else _EXTERNAL
        library = self.standard.get(lang)
        # The library's types are classes, as the project's are.
        classed = lang in ("java", "php", "ruby")
        rust = lang == "rust"
        for index, step in enumerate(steps):
            last = index == len(steps) - 1 and not read
            if value == _EXTERNAL and rust and step == "?":
                continue
            if python and step == "await":
                # What awaiting gives: the type argument the data says an
                # awaitable of the library hands on; a project function's
                # call is typed by what it returns already.
                if isinstance(value, tuple) and value[0] == "outside":
                    at = _stdlib.load("python").get("awaited", {}).get(value[1])
                    held = value[2] if len(value) > 2 else ()
                    value = held[at] if at is not None and at < len(held) else None
                continue
            field = (
                rust
                and re.fullmatch(r"[A-Za-z_]\w*", step)
                and index < len(steps) - 1
                and steps[index + 1] != "()"
            )
            if value == _EXTERNAL and field:
                # A field of a Rust value from outside may hold a project type.
                value = None
                break
            if value in (_OPAQUE, _EXTERNAL):
                # A call's opaque or outside-typed result: its members are
                # not known.
                break
            if isinstance(value, tuple) and value[0] == "pointer":
                # `->` and a subscript reach what a raw pointer points at, as
                # does a member read after a `*` that the path does not write.
                value = value[1]
                if step in ("?", "[]", "->"):
                    continue
            if step == "->" and isinstance(value, tuple) and value[0] != "wrapped":
                # `->` reaches what the type's `operator->` points at: the
                # first type argument of a library template the data says
                # it reaches, what a project type's member returns; a value
                # of a project type with none is a pointer to it.
                if value[0] == "outside":
                    pointer = value[1].split("`")[0].removeprefix("std::")
                    pointed = pointer in _stdlib.section("cpp", "pointers")
                    value = value[2][0] if pointed and value[2:] and value[2] else None
                elif member := self._attr(value, "operator->", called=True):
                    value = self._called(member, value if len(value) > 2 else value[1])
                    value = value[1] if value and value[0] == "pointer" else value
                continue
            if step == "[]" and isinstance(value, tuple):
                # A subscript gives what the type's `operator[]` returns, in
                # Rust its `Index` impl's `Output`, the element an iterator
                # gives where the index decides it: by the data for a
                # standard type, its member's for one of the project.
                if value[0] == "outside" and rust:
                    value = library.associated(
                        value, "Output", "std::ops::Index"
                    ) or library.associated(value, _ITERATED[lang])
                elif value[0] == "outside":
                    known, held = library.member(value, "operator[]")
                    value = held if known else None
                else:
                    member = self._attr(
                        value, "index" if rust else "operator[]", called=True
                    )
                    value = member and self._called(
                        member, value if len(value) > 2 else value[1]
                    )
                continue
            if python and step == "__getitem__" and self._is_class_value(value):
                # A class written with type arguments is that class.
                value = ("returned", value)
                continue
            if isinstance(value, tuple) and value[0] == "library":
                # A name the data has under another names that; any other
                # is a member of what the first gives.
                if f"{value[1]}.{step}" in _py_std_names():
                    value = ("library", f"{value[1]}.{step}")
                    continue
                value = ("returned", self._library_value(value[1]))
            if isinstance(value, tuple) and value[0] == "returned":
                # What a member the data has gives, read or called.
                value = value[1]
                if step == "()":
                    continue
            if python:
                value = self._class_object(value)
            if (
                python
                and step != "()"
                and isinstance(value, tuple)
                and value[0] == "def"
            ):
                # An attribute of a property, or of a function a decorator
                # is applied to, is one of what the name is bound to.
                owner, _, member = value[1].rpartition("::")
                if member in self.builder.fields.get(owner, ()):
                    value = self._variable(owner, member) or value
            if step == "*" or "@" in step:
                value = self._part(value, step, source_id)
                continue
            if (
                isinstance(value, tuple)
                and value[0] == "def"
                and step not in ("()", "?")
                and self.nodes[value[1]]["type"] != "Class"
                and lang == "dart"
            ):
                # A Dart variable or getter is read as its declared type.
                value = self._declared_type(value[1]) or value
            if classed and isinstance(value, tuple) and value[0] == "def":
                # A field of a standard type is read as the value it holds.
                held = self._declared_type(value[1])
                if isinstance(held, tuple) and held[0] == "outside":
                    value = held
            if step.startswith("^"):
                # `^N`: parameter N of the single abstract method of a
                # functional type — how a Java lambda's parameters are
                # written.
                params = ()
                if isinstance(value, tuple) and value[0] == "functional":
                    params = self._functional_params(value[1], source_id)
                elif isinstance(value, tuple) and value[0] in ("def", "self"):
                    if self._is_class(value[1]):
                        params = self._functional_params(value, source_id)
                    else:
                        held = self.nodes[value[1]]["metadata"].get("value_type")
                        params = self._functional_params(held, source_id)
                number = int(step[1:]) if step[1:].isdigit() else None
                rest = (
                    params and isinstance(params[-1], tuple) and params[-1][0] == "rest"
                )
                value = None
                if number is not None and steps[index + 1 : index + 2] == ["&"]:
                    value = _gathered(params, number)
                elif number is not None and number < len(params) - bool(rest):
                    value = params[number]
                elif number is not None and rest:
                    value = params[-1][1]
                continue
            if step == "&":
                # What the `^N` before it gathered
                continue
            if isinstance(value, tuple) and value[0] == "literal":
                verdict = self._each(value[1], steps[index:], source_id)
                if len(value) == 2 or (verdict and verdict[0]):
                    return verdict
                # What the object leads to no definition of is read in the
                # type its function declares.
                value = value[2]
            outside = isinstance(value, tuple) and value[0] == "outside"
            if outside and value[1] == "(T,)" and step.isdigit():
                # A field of a Rust tuple is the element at its position.
                held = value[2] if len(value) > 2 else ()
                value = held[int(step)] if int(step) < len(held) else None
                continue
            if outside and step not in ("()", "?"):
                # A method of a standard type is the standard library's and
                # returns what the library's data says, unless the project
                # implements a trait with it for that type or for every
                # type. One the type has not is a method of what it
                # dereferences to.
                if python and not last and steps[index + 1 : index + 2] != ["()"]:
                    # An attribute that is read, which the data names
                    # behind a dot.
                    known, held = library.member(value, f".{step}")
                    value = held if known else None
                    continue
                name = value[1].rpartition("::")[2]
                # A Dart extension on a standard type is one on every type
                # derived from it; one on a project type is not.
                dart = lang == "dart"
                direct = {
                    self._find_member(t, step)[0]
                    for named in (library.lineage(name) if dart else [name])
                    for t in self.impl_types.get(named, [])
                    if not dart or self._type_named(named, t) == _EXTERNAL
                }
                aliased = self.impls_for.get(value[1], ()) if rust else ()
                direct |= {self._find_member(t, step)[0] for t in aliased}
                blanket = {self._find_member(t, step)[0] for t in self._blanket(name)}
                given = (direct | blanket) - {None}
                known, held = False, value
                if library:
                    known, held = library.member(value, step)
                # A Rust value that is the type written with its arguments,
                # no instance of it
                typed = rust and "()" not in steps[:index]
                typed = typed and not origin.startswith(("#", "=", "{", "?"))
                if (classed or python or typed) and not known:
                    # A static method is called on the type or on a value; a
                    # Rust one on the type written with its arguments.
                    known, held = library.member(value, step, table="statics")
                if (
                    lang == "dotnet"
                    and not known
                    and value[1].split("<")[0] == "System.Array"
                    and value[2]
                    and "System.Collections.Generic.IEnumerable`1"
                    in library.data["types"]
                ):
                    # An array is also an IEnumerable<T> of its element:
                    # what that interface's members give is the call's.
                    known, held = library.member(
                        (
                            "outside",
                            "System.Collections.Generic.IEnumerable`1",
                            value[2],
                        ),
                        step,
                    )
                # The library's own extension methods of the name, which a
                # project one may come before.
                spaces = lang == "dotnet" and known and library.extended(value, step)
                if (
                    lang == "dotnet"
                    and last
                    and (not known or spaces)
                    and not given
                    and value[1].split("<")[0] in library.data["types"]
                ):
                    # The library's type has no member of the name: a
                    # project extension method written for it is the call's.
                    hits, nearer = self._dotnet_extension(value, step), True
                    if spaces:
                        hits, nearer = self._dotnet_rivals(hits, spaces, source_id)
                    caller_arms = self._caller_guards(source_id)
                    if caller_arms:
                        hits = [
                            h
                            for h in hits
                            if not self._guard_conflict(
                                caller_arms,
                                self._effective_guards(h, source_id),
                            )
                        ]
                    else:
                        # One every build has is meant before one an `#if`
                        # arm alone builds.
                        guarded = [
                            h for h in hits if self._effective_guards(h, source_id)
                        ]
                        hits = [h for h in hits if h not in guarded] or [
                            h for h in guarded if not self._candidate_only(h, source_id)
                        ]
                        if not hits and guarded:
                            self._candidates_now = sorted(guarded)
                            return None, _UNRESOLVED
                    fits = self._extension_fits(hits, value)
                    if len(fits) == 1 and nearer:
                        value = ("def", fits[0])
                        continue
                    if fits:
                        self._candidates_now = fits
                        return None, _UNRESOLVED
                    if not spaces:
                        # The data has the type's every member: a name it
                        # lacks that no project extension gives is external.
                        value = _EXTERNAL
                        continue
                extended = lang == "typescript" and len(value) > 2
                if last and extended and not known:
                    # A project type may extend the standard one and be
                    # what the value is: the member may be the project's.
                    self._result = _OPAQUE
                    return None, _UNRESOLVED
                if last and given and rust and self._std_inherent(value[1], step):
                    self._candidates_now = sorted(given)
                    return None, _UNRESOLVED
                if known or given or held == value:
                    # Every Rust type has `Any`; what implements a trait
                    # has more methods than the trait's.
                    typed = known or len(value) < 3 or lang != "rust"
                    typed = typed or library.member(value, "type_id")[0]
                    if last and not given and typed:
                        if lang == "dart" and value[1] not in library.data["types"]:
                            # A Dart type the data does not have may gain
                            # the member through an extension; the name
                            # alone decides it.
                            return None
                        if known and rust:
                            # A return type an argument's type decides
                            types = [self._argument_type(a) for a in self._passed or ()]
                            held = library.argued(value, step, types) or held
                        self._result = (known and held) or _OPAQUE
                        if known:
                            self._callables = library.callbacks(value, step)
                        if known and results:
                            self._deferred = (source_id, value, step, results)
                        return None, _EXTERNAL
                    sole = len(given) == 1 and not known
                    value = ("def", given.pop()) if sole else held
                    if sole and not direct:
                        # A trait's method found through an impl that holds
                        # for every type is no evidence of this receiver's
                        # type: what the call returns is not read.
                        self._name_only = True
                    if not known and lang in ("go", "java", "php"):
                        # The data has every method: this is a field, whose
                        # type is the library's as its type is.
                        value = _EXTERNAL
                    if python and known:
                        value = ("returned", held)
                    continue
                value = held
            if value is None or value == _EXTERNAL:
                break
            if value[0] == "outside":
                if step == "?" and lang == "typescript":
                    # What `await` gives: what a thenable hands to the first
                    # callable of its `then`, any other value itself.
                    passed = library.callbacks(value, "then").get("0")
                    value = passed[0] if passed else value
                elif step == "?" and lang == "dart":
                    # `await` gives the type a `Future` or `FutureOr` holds.
                    if (
                        value[1] in ("Future", "FutureOr")
                        and len(value) > 2
                        and value[2]
                    ):
                        value = value[2][0]
                elif step == "?":
                    if lang == "clang" and len(value) > 2 and value[2]:
                        # A container iterates its `value_type` — `map`'s
                        # `pair<const K, V>`, `vector`'s `T` — or its first
                        # type argument.
                        value = library.aliased(value, "value_type") or value[2][0]
                    else:
                        # What Rust's `Try` says the type holds.
                        held = ("Output", "std::ops::Try")
                        value = library and library.associated(value, *held)
            elif step == "()":
                # A standard type called constructs a value of it.
                if not (
                    outside
                    and library
                    and value[1].split("<")[0] in library.data["types"]
                ):
                    value = self._called(value, self._receiver)
            elif value[0] == "union":
                # `A | B`: a member is called on whichever alternative has
                # it — resolve only where every alternative agrees, else
                # the call is ambiguous.
                if step == "?":
                    continue
                outcomes = [
                    self._attr(one, step, called=last)
                    for one in value[1]
                    if isinstance(one, tuple)
                ]
                if (
                    outcomes
                    and len(outcomes) == len(value[1])
                    and outcomes[0] is not None
                    and all(one == outcomes[0] for one in outcomes)
                ):
                    value = outcomes[0]
                else:
                    self._candidates_now = (
                        sorted(
                            {
                                one[1]
                                for one in outcomes
                                if isinstance(one, tuple) and one[0] == "def"
                            }
                        )
                        or None
                    )
                    return None, _UNRESOLVED
            elif value[0] == "wrapped":
                unwrapped = ("self", value[1])
                if last and step != "?" and lang == "rust":
                    # A method of a Rust type from outside is its own, one
                    # of a trait the project implements for it, or through
                    # `Deref` the wrapped type's.
                    impls = self.impl_types.get(value[2], []) + self._blanket(value[2])
                    given = any(self._find_member(t, step)[0] for t in impls)
                    held, complete = self._find_member(value[1], step)
                    if not given and not held and complete is True:
                        return None, _EXTERNAL
                if last and step not in ("?", "->") and lang in ("dotnet", "clang"):
                    # A C# type or a C++ smart pointer from outside has its
                    # own members only; what one returns may be the project
                    # type it wraps.
                    self._result = _OPAQUE
                    return None, _EXTERNAL
                # A C++ smart pointer's own members are from outside.
                own = _EXTERNAL if lang == "clang" else None
                value = unwrapped if step in ("?", "->") else own
            elif step != "?":
                # A value of a generic type carries its type arguments, as
                # does the type written with them, `Boxed<Dog>.new`.
                self._receiver = value[1] if value[0] == "self" else None
                if value[0] in ("self", "def") and len(value) > 2:
                    self._receiver = ("self", *value[1:])
                field = value[0] == "def" and self.nodes[value[1]]["type"] == "Constant"
                untyped = field and not self._declared_type(value[1])
                value = self._attr(value, step, called=last)
                if value is None and last:
                    # A field declared no type holds a value of unknown type.
                    return None if untyped else (None, _UNRESOLVED)
        if read:
            returned = isinstance(value, tuple) and value[0] == "returned"
            return value[1] if returned else value
        if python:
            value = self._class_object(value)
        if isinstance(value, tuple) and value[0] == "library":
            self._result = self._library_value(value[1]) or _OPAQUE
            # The data has the callables a function takes under its module.
            module, _, name = value[1].rpartition(".")
            passed = library.data["callbacks"].get(module, {}).get(name, {})
            self._callables = {
                argument: [library.value(written, {}) for written in taken["params"]]
                for argument, taken in passed.items()
            }
            return None, _EXTERNAL
        if isinstance(value, tuple) and value[0] in ("def", "cls"):
            return value[1], source
        if value == _EXTERNAL:
            if not steps:
                # A type from outside constructs a value from outside.
                self._result = _EXTERNAL
            return None, _EXTERNAL
        if classed and not steps and isinstance(value, tuple) and value[0] == "outside":
            # A standard type called by its name makes an instance of it.
            self._result = value
            return None, _EXTERNAL
        if python and isinstance(value, tuple) and value[0] == "self":
            hit = self._find_member(value[1], "__call__")[0]
            if hit:
                return hit, source
        standard = isinstance(value, tuple) and value[0] == "outside"
        if python and (standard or origin.startswith(":") and not steps):
            # A value of a standard type, or of a type that is no class.
            self._result = _OPAQUE
            return None, _EXTERNAL
        # A call of a value held in a variable, or returned by another call.
        if origin.endswith(")") or "." not in origin:
            self._result = _OPAQUE
            return None, _UNRESOLVED
        return None

    def _each(self, literal, steps, source_id):
        """Verdict of a call that the origin steps ``steps`` name on an
        element of the array literal ``literal``: the one that every element
        gives, None where the type of one is not known, unresolved where
        they lead to different definitions."""
        path = "".join(step if step == "()" else f".{step}" for step in steps)
        verdicts, results = set(), set()
        # What a function returns names what is in that function's scope.
        source_id = self.builder.literal_scopes.get(literal, source_id)
        for element in self.builder.literals[literal]:
            origin = element and f"{element}{path}"
            if isinstance(element, dict):
                origin = element.get(steps[0])
                origin = origin and origin + path[len(steps[0]) + 1 :]
            verdicts.add(origin and self._resolve_origin(origin, source_id))
            results.add((self._result, self._receiver))
        if None in verdicts:
            return None
        if len(results) > 1:
            # What the call returns is known where the elements agree on it.
            self._result, self._receiver = _OPAQUE, None
            self._callables, self._deferred = {}, None
        return verdicts.pop() if len(verdicts) == 1 else (None, _UNRESOLVED)

    def _root_value(self, origin, source_id, seen):
        """Evaluate the first step of an origin path written in
        ``source_id``: an imported path behind `@`, `self`, `cls` or
        `super`, `#N` for what the call at edge index N returned, `~name`
        for what pytest passes as the parameter `name`, `=name` for a
        parameter or variable of ``source_id`` (a definition only where it
        is one of ``source_id``'s own) and `=name>Type` for one that holds a
        function declared to return `Type`, `!Path` for a Rust path read as
        a value, an instance of the unit struct that a name in scope or that
        one import names and else of no type known, `?` for a value of no
        known origin, or a name read in scope, which in Java is read with the
        names after it where it is not called (see ``_java_path``). Returns
        its value, the steps after it, and the resolution source a call
        found through it has. ``seen`` holds the fixtures being evaluated."""
        file_id = source_id.split("::")[0]
        path = origin.lstrip("@")
        steps = re.findall(r"\(\)|[^.()]+", path)
        if origin.startswith("@"):
            dots = len(path) - len(path.lstrip("."))
            return (*self._import_value(steps, dots, file_id), "regex")
        source = "regex"
        root, steps = steps[0], steps[1:]
        owner = self._enclosing_class(source_id)
        static = root == "self" and self._static_owner(source_id)
        if static:
            # `this` in a JavaScript static member is its constructor.
            source, value = "self_type", ("cls", static)
        elif root in ("self", "cls", "super") and owner:
            source = "self_type"
            value = (root, owner)
            if root == "super":
                steps = steps[1:]
        elif root in ("self", "cls", "super", "?"):
            value = None
        elif root in self.builder.parameter_types:
            value = self._returned_type(*self.builder.parameter_types[root])
        elif root in self.builder.narrowings:
            # Promoted where each object's type derives from the type
            values, held, before = self.builder.narrowings[root]
            target = self._instance_class(held, source_id)
            kept = target is not None and all(
                value
                and target
                in self._lineage(self._instance_class(value, source_id), set())
                for value in values
            )
            value, ahead, source = self._root_value(
                held if kept else before, source_id, seen
            )
            steps = ahead + steps
        elif root.startswith(":") and not root.startswith("::"):
            # `::App::User` is a name read in scope, PHP's in full.
            value = self._annotated(root[1:], source_id)
        elif root.startswith("#") and "|" in root.partition(">")[0]:
            # `#N|A|P`: parameter P of the callable passed to call N as
            # argument A, or as the only one that is a callable (`?`), or
            # given as the property `key` of an object passed as A (`A:key`);
            # `#N|A|this` is the `ThisType` parameter A declares.
            call, argument, parameter = root[1:].split("|")
            call = self._edge(call)
            if parameter == "this":
                value = self.this_args.get(call, {}).get(argument)
            else:
                callables = self.callables.get(call, {})
                if argument == "?" and len(callables) == 1:
                    argument = next(iter(callables))
                params = callables.get(argument, ())
                if ":" in argument:
                    params = self._property_params(call, argument)
                value = params[int(parameter)] if int(parameter) < len(params) else None
                if steps[:1] == ["&"]:
                    value = _gathered(params, int(parameter))
                    steps = steps[1:]
        elif root.startswith("#"):
            # `#N>Type`: what the call at edge index N returned, read for
            # the members the type a variable it is bound to is declared
            # with has where the object itself leads to none.
            call, _, declared = root[1:].partition(">")
            call = self._edge(call)
            if call in self.deferred:
                # The closures the call was passed are resolved by now.
                scope_id, held, name, results = self.deferred.pop(call)
                results = {
                    argument: self._root_value(origin, scope_id, set())[0]
                    for argument, origin in results.items()
                }
                library = self.standard[self.lang(scope_id)]
                self.returned[call] = library.member(held, name, results=results)[1]
            if call in self.late:
                # The calls whose results it was passed are resolved by now.
                target, site, scope_id, receiver = self.late.pop(call)
                self._bind_call(target, site, scope_id)
                self.returned[call] = self._called(("def", target), receiver)
            value = self.returned[call] if call in self.returned else self._ahead(call)
            if declared and (value is None or value == _EXTERNAL):
                # A call that gives nothing of the project's leaves the name
                # read as the type it is declared with.
                path = "".join(step if step == "()" else f".{step}" for step in steps)
                return self._root_value(f"{declared}(){path}", source_id, seen)
            if declared and isinstance(value, tuple) and value[0] == "literal":
                # The type reads the members the object leads to none of.
                value = (*value[:2], self._plain_value(f"{declared}()", source_id))
        elif root.startswith("%"):
            # `%N`: what the origins the parser holds under the key all give
            # (the arms of a Rust `if` or `match`); not known where they part.
            held = [
                self._value(o, source_id, set(seen))
                for o in self.builder.literals[root]
            ]
            agreed = {v[:2] if isinstance(v, tuple) else v for v in held}
            value = held[0] if len(agreed) == 1 and isinstance(held[0], tuple) else None
        elif root.startswith("~"):
            value = self._fixture_value(root[1:], source_id, seen)
        elif root.startswith(">"):
            # `>Type`: a type written without a variable, as a cast or an
            # assignment gives it — a `^N` step takes its functional
            # parameters.
            value = ("functional", root[1:])
        elif root.startswith("!"):
            value = self._path_value(root[1:], source_id)
        elif root.startswith("="):
            expr = root[1:]
            if self.lang(source_id) == "clang" and ("<" in expr or "." in expr):
                # `=Type<args>`: a written type a range-for iterates, not a
                # name bound in scope.
                hit = None
                value = self._name_value(expr, source_id, set())
            else:
                hit = self.members.get(source_id, {}).get(expr.partition(">")[0])
                value = ("def", hit) if hit else None
            if not hit and steps and self.lang(source_id) == "dart":
                # `=value.call`: a variable holding `value` is called.
                return self._root_value(origin[1:], source_id, seen)
        elif self.lang(source_id) == "java" and steps and steps[0] != "()":
            value, steps = self._java_path([root, *steps], source_id)
        elif self.lang(source_id) == "dart":
            value, steps = self._dart_root(root, steps, source_id)
        else:
            # A C# name read for a member of it is a member of an enclosing
            # type before it is a type.
            reads = steps[:1] not in ([], ["()"]) and self.lang(source_id) == "dotnet"
            hit = self._inherited(root, source_id) if reads else None
            value = ("def", hit) if hit else self._name_value(root, source_id, set())
            unbuilt = steps[:1] != ["()"]
            if self.lang(source_id) == "rust" and "::" in root and unbuilt and not hit:
                # A Rust path to a unit struct or variant, or a constant, not
                # called, is a value of it.
                held = self._path_value(root, source_id)
                value = held if isinstance(held, tuple) and held[0] != "def" else value
            # Else a path through namespaces to a type, `App.Lib.Bar`
            for size in range(1, steps.index("()") if "()" in steps else len(steps)):
                if value not in (None, _EXTERNAL) or not reads:
                    break
                found = self._dotnet_type(
                    [root, *steps[: size - 1]], steps[size - 1], source_id
                )
                if found not in (None, _EXTERNAL):
                    value, steps = ("def", found), steps[size:]
        return value, steps, source

    def _fixture_value(self, name, scope_id, seen):
        """Value pytest passes for the parameter ``name`` of the test or
        fixture ``scope_id``: what the nearest fixture of that name gives,
        looking in the enclosing classes, the module, and then each
        `conftest.py` from the module's directory upwards."""
        file_id = scope_id.split("::")[0]
        scopes = [s for s in self._scopes(scope_id) if self._is_class(s)]
        scopes.append(file_id)
        folder = posixpath.dirname(file_id[len("FILE:") :])
        while True:
            scopes.append(f"FILE:{posixpath.join(folder, 'conftest.py')}")
            parent = posixpath.dirname(folder)
            if parent == folder:
                break
            folder = parent
        for scope in scopes:
            function_id, origin, yields = self.builder.fixtures.get(
                (scope, name), (None, None, False)
            )
            # A fixture that takes a parameter of its own name gets the one
            # it overrides.
            if function_id is None or function_id == scope_id or function_id in seen:
                continue
            seen.add(function_id)
            # What a fixture yields is not the generator it is annotated as.
            value = None if yields else self._called(("def", function_id))
            if value is None and origin:
                value = self._value(origin, function_id, seen)
            return value
        return None

    def _name_value(self, name, scope_id, seen):
        """Value of a name read inside ``scope_id``."""
        lang = self.lang(scope_id)
        if lang == "typescript":
            hit = self._written(name, scope_id, set())
            alias = not hit and self._ts_alias(name, scope_id)
            if alias and _ts_type_parts(alias[1])[0] == "[]":
                # A tuple an alias stands for, a value of it once called
                return ("returned", self._returned_type(name, scope_id))
        else:
            hit = self._lexical(name, scope_id, skip_classes=True)
        if hit and lang in ("dotnet", "ruby") and self._is_class(hit):
            # A C# type is the one its rules find: one nested in an enclosing
            # type before one of the namespace; a Ruby constant is looked up
            # by `Module.nesting`, then the ancestors, then `Object`.
            hit = None
        if hit and lang == "rust" and self._rust_alias(name, scope_id):
            # A type and an alias of one name are each other's twin for
            # other builds: the alias reads which.
            hit = None
        if (
            hit
            and lang == "rust"
            and self._impl_only(hit)
            and self._standard_type(name, scope_id)
        ):
            # The project's impl blocks for a standard type the name stands
            # for are no type of it.
            hit = None
        if hit:
            if (
                lang == "python"
                and hit.rpartition("::")[0] == (file_id := scope_id.split("::")[0])
                and (binding := self.builder.import_binding.get(file_id, {}).get(name))
            ):
                # The module imported the name after defining it: the import
                # is what binds, or one of the two where a conditional holds
                # one of them. A star import binds it where a module it
                # names exports it.
                imports = self.builder.imports[file_id]
                if name in imports:
                    imported = self._imported_as(name, imports[name], file_id, set())
                else:
                    found, outside = self._starred(file_id, name, set())
                    if not (found or outside):
                        return ("def", hit)
                    imported = found.pop() if len(found) == 1 and not outside else None
                if binding == "import":
                    return imported
                if isinstance(imported, tuple) and imported[0] == "def":
                    self._candidates_now = sorted({hit, imported[1]})
                    return None
            return ("def", hit)
        if lang == "clang" and name.endswith("*"):
            # A raw pointer to the type the rest names
            held = self._returned_type(name, scope_id)
            return held and ("returned", held)
        if lang != "python":
            # `Type<>` is a type written with type arguments, in Go `Type[]`.
            written = name
            name, generic, _ = name.partition("[" if lang == "go" else "<")
            if self._standard_type(written, scope_id):
                return self._returned_type(written, scope_id)
            if lang == "rust" and (
                written == "Self" or self._rust_alias(written, scope_id)
            ):
                # `Self` is the type of the impl block around, an alias the
                # type it names.
                return ("returned", self._returned_type(written, scope_id))
            package, _, member = name.rpartition("::")
            if lang == "go" and package:
                # `pkg::Name` may be a package-level variable of a project
                # package, read as what it is declared as.
                scopes = self._imported(package, scope_id)
                scopes = scopes if isinstance(scopes, list) else []
                held = (self.members.get(scope, {}).get(member) for scope in scopes)
                held = [h for h in held if h and self.nodes[h]["type"] == "Constant"]
                if held:
                    return self._declared_type(held[0])
            # A type parameter is its bound, or of no type known.
            held = self._type_argument(name, scope_id, None)
            if held is not False:
                return held and ("returned", held)
            given = generic and written.endswith("]" if lang == "go" else ">")
            if lang == "clang" and given:
                held = self._c_specialized(written, scope_id)
                if held is not False:
                    return held and ("returned", held)
            generic = generic and lang not in _OWN_METHOD_LANGS
            found = self._type_named(name, scope_id, not generic)
            if isinstance(found, tuple):
                return ("returned", found)
            name = name.rpartition("::")[2]
            if given and found and found != _EXTERNAL:
                # A generic type written with its arguments: an instance of
                # it holds them.
                made = self._returned_type(written, scope_id)
                if made and len(made) > 2:
                    return ("def", found, made[2])
            if generic and found == _EXTERNAL:
                # A wrapper from outside around what may be a project type;
                # in C# its members are its own.
                if lang == "dotnet":
                    written_name, arguments, _ = _dotnet_type_parts(written)
                    key = self._dotnet_std_key(written_name, scope_id)
                    if key:
                        args = tuple(
                            self._returned_type(a, scope_id) or _OPAQUE
                            for a in arguments
                        )
                        return "outside", key, args
                return None
            if found is None and not self._types(lang, name):
                # An alias may stand for a project type, as may what a
                # generic type wraps; one written with its arguments is
                # read as a declared type is.
                held = given and self._returned_type(written, scope_id)
                if held:
                    return ("returned", held)
                aliased = generic or name in self.builder.type_aliases
                return None if aliased else _EXTERNAL
            return ("def", found) if found not in (None, _EXTERNAL) else found
        file_id = scope_id.split("::")[0]
        imports = self.builder.imports.get(file_id, {})
        if name in imports and (file_id, name) not in seen:
            return self._imported_as(name, imports[name], file_id, seen)
        if name in self.builder.fields.get(file_id, {}):
            return self._variable(file_id, name)
        found, outside = self._starred(file_id, name, seen)
        if found or outside:
            return found.pop() if len(found) == 1 else None
        # Neither defined nor imported: a builtin, if the data has one.
        key = self.standard["python"].data["scope"].get(name)
        return ("library", key) if key else _EXTERNAL

    def _through_imports(self, origin, source_id, steps):
        """The definitions the path ``origin`` in ``source_id`` may reach
        where its first name is one the file imports from several places:
        the path read through each, by its attributes. Empty for any other
        path."""
        file_id = source_id.split("::")[0]
        root = re.match(r"[A-Za-z_]\w*", origin)
        targets = root and self.builder.imports.get(file_id, {}).get(root[0])
        if not isinstance(targets, tuple) or not all(
            re.fullmatch(r"\w+", step) for step in steps
        ):
            return []
        reached = set()
        for target in targets:
            value = self._imported_as(root[0], target, file_id, set())
            for at, step in enumerate(steps):
                if not isinstance(value, tuple):
                    break
                value = self._attr(value, step, called=at == len(steps) - 1)
            if isinstance(value, tuple) and value[0] == "def":
                reached.add(value[1])
        return sorted(reached)

    def _imported_as(self, name, targets, file_id, seen):
        """Value the file ``file_id`` imports ``name`` as, of the targets
        ``targets``: the one value they all give; None, with the
        definitions that fit as the candidates, where they differ."""
        seen.add((file_id, name))
        targets = targets if isinstance(targets, tuple) else (targets,)
        values = []
        for target in targets:
            dots = len(target) - len(target.lstrip("."))
            value, steps = self._import_value(
                target.strip(".").split("."), dots, file_id
            )
            for step in steps:
                if value is None or value == _EXTERNAL:
                    break
                value = self._attr(value, step, seen)
            values.append(value)
        if len(set(values)) <= 1:
            return values[0] if values else None
        # Two imports of the name fit; which one it names is not told.
        self._candidates_now = (
            sorted({v[1] for v in values if isinstance(v, tuple) and v[0] == "def"})
            or None
        )
        return None

    def _variable(self, owner_id, name):
        """Value of the variable ``name`` that the Python module or class
        ``owner_id`` binds, from the origin paths of what it is bound to:
        the value they agree on, or some standard value where each is one
        from outside the project."""
        key = (owner_id, name)
        if key not in self.variables:
            # A variable bound to itself has no value.
            self.variables[key] = None
            origins = self.builder.fields[owner_id][name]
            values = {self._value(origin, owner_id, set()) for origin in origins}
            outside = (v == _EXTERNAL or v and v[0] == "outside" for v in values)
            if len(values) == 1:
                self.variables[key] = values.pop()
            elif all(outside):
                self.variables[key] = ("outside", "builtins.object", ())
        return self.variables[key]

    def _starred(self, file_id, name, seen):
        """``(values, outside)``: the values that the modules ``file_id``
        imports every public name of have under ``name``, and whether one of
        those modules is not the project's."""
        found, outside = set(), False
        for target in self.builder.imports.get(file_id, {}):
            if (
                not isinstance(target, str)
                or not target.endswith(".*")
                or (file_id, target, name) in seen
            ):
                continue
            seen.add((file_id, target, name))
            dots = len(target) - len(target.lstrip("."))
            names = target.strip(".").split(".")[:-1]
            module = self._import_value(names, dots, file_id)[0]
            if not isinstance(module, tuple) or module[0] != "module":
                outside = True
                continue
            # A module's `__all__`, where it writes one, lists what it
            # exports, or may export any name where it is not written out;
            # else every public name.
            exports = self.builder.py_all
            module_id = self.py_modules[module[1]][module[2]]
            listed = exports.get(module_id)
            if module_id not in exports:
                exported = not name.startswith("_")
            else:
                exported = listed is None or name in listed
            value = self._attr(module, name, seen) if exported else None
            found.add(value)
            # A name an export list not written out holds may not be bound.
            outside |= value is not None and listed is None and module_id in exports
        return found - {None}, outside

    def _value(self, origin, scope_id, seen):
        """Value the origin path ``origin`` leads to in ``scope_id``, read
        while a call is being resolved. ``seen`` holds the fixtures being
        evaluated."""
        with self._kept():
            return self._resolve_origin(origin, scope_id, seen)

    def _instance_class(self, origin, scope_id):
        """The project class of the object that the origin path ``origin``
        leads to in ``scope_id``, a variable read as its declared type; None
        for any other."""
        value = self._value(origin, scope_id, set())
        if (
            isinstance(value, tuple)
            and value[0] == "def"
            and not self._is_class(value[1])
        ):
            value = self._declared_type(value[1])
        return value[1] if isinstance(value, tuple) and value[0] == "self" else None

    def _is_class_value(self, value):
        """Whether ``value`` is a class, the project's or the standard
        library's, rather than an instance of one."""
        if not isinstance(value, tuple):
            return False
        if value[0] == "library":
            return value[1] in self.standard["python"].data["types"]
        return value[0] == "def" and self._is_class(value[1])

    def _class_object(self, value):
        """The class that ``value`` is where it is an instance of
        `type[C]`: C as a project class, None for any other. Every other
        value is returned as it is."""
        if not isinstance(value, tuple) or value[:2] not in _PY_CLASS_OBJECTS:
            return value
        held = value[2][0] if value[2] else None
        project = isinstance(held, tuple) and held[0] == "self"
        return ("cls", held[1]) if project else None

    def _library_value(self, key):
        """What the name ``key`` of Python's standard library gives, called
        or read: an instance of the class, what the function returns or the
        variable holds, ``"external"`` for a module."""
        library = self.standard["python"]
        if key in library.data["types"]:
            return ("outside", key, (None,) * len(library.data["types"][key]["params"]))
        functions = library.data["functions"]
        returns = functions.get(key, functions.get(f".{key}", _EXTERNAL))
        if returns in (None, _EXTERNAL):
            return returns
        return library.value(returns, {})

    def _annotated(self, written, scope_id, receiver=None):
        """Value of an instance of the Python type ``written`` in
        ``scope_id``, as an annotation has it: of a project class, of a
        standard type that holds what its type arguments are,
        ``"external"``, or None. `Self` is ``receiver``, the type the method
        was called on, where that is known."""
        library = self.standard["python"]
        name, arguments, _ = library.parts(written)
        if name == "Self":
            owner = receiver or self._enclosing_class(scope_id)
            return owner and ("self", owner)
        held = tuple(self._annotated(a, scope_id, receiver) for a in arguments)
        if not name:
            # A union of standard types is some standard value.
            standard = (isinstance(it, tuple) and it[0] == "outside" for it in held)
            if held and all(standard):
                return ("outside", "builtins.object", ())
            return None
        value = self._value(name, scope_id, set())
        if isinstance(value, tuple) and value[0] == "library":
            if value[1] not in library.data["types"]:
                return None
            return ("outside", value[1], held)
        if isinstance(value, tuple) and value[0] == "def":
            return ("self", value[1]) if self._is_class(value[1]) else None
        return value if value == _EXTERNAL else None

    def _py_std_key(self, name, scope_id):
        """What the Python name ``name``, written in ``scope_id`` or as an
        origin path, is called in the standard library's data."""
        value = self._value(name, scope_id, set()) if name else None
        library = isinstance(value, tuple) and value[0] == "library"
        return value[1] if library else None

    def _blanket(self, name):
        """The blanket impls that are for the outside type ``name``: those
        whose every bound it may meet. A trait the project declares it meets
        only where the project implements that trait for it."""
        met = {
            _split_callee(base)[1]
            for given in self.impl_types.get(name, ())
            for base in self.bases.get(given, ())
        }
        return [
            given
            for given in self.impl_types.get("", ())
            if all(
                bound in met or not self._declares_type(bound)
                for bound in self.nodes[given]["metadata"]["bounds"]
            )
        ]

    def _declares_type(self, name):
        """Whether the project declares a Rust type ``name``, rather than
        only implementing traits for one."""
        return any(
            self.nodes[t]["metadata"].get("kind") != "impl"
            for t in self._types("rust", name)
        )

    def _rust_std_key(self, name, scope_id):
        """What the Rust name ``name`` written in ``scope_id`` is called in
        the standard library's data: from the crate its first segment is
        imported from (`io::Result` is `std::io::Result` under `use
        std::io`), or by the prelude unless the project declares a type or
        an alias of that name."""
        file_id = scope_id.split("::")[0]
        if name in self.builder.rust_aliases.get(file_id, ()):
            return None
        root, _, rest = name.partition("::")
        root = self._uses(scope_id).get(root, root)
        path = f"{root}::{rest}" if rest else root
        if path == name and not self._declares_type(name):
            path = self.standard["rust"].data["scope"].get(name, name)
        return path

    def _part(self, value, step, scope_id):
        """Value of the part of ``value`` that a pattern binds a name to.
        The step `*` is what iterating ``value`` gives, `Name@key` what the
        variant or the struct `Name` holds at ``key``, a position or a
        field name, and `@key` an element of a tuple. None where the type
        of ``value`` does not say."""
        if value == _EXTERNAL and self.lang(scope_id) == "go":
            # What a Go package outside the project hands out is its own.
            return _EXTERNAL
        if (
            self.lang(scope_id) in ("php", "dotnet", "dart")
            and isinstance(value, tuple)
            and value[0] == "def"
        ):
            # A PHP property, a C# field or property, a Dart field or getter,
            # is read as the value it holds.
            value = self._declared_type(value[1])
        if not isinstance(value, tuple) or value[0] not in ("outside", "self"):
            return None
        name, _, key = step.partition("@")
        key = int(key) if key.isdigit() else key
        if value[0] == "self":
            # A struct is matched by its own name, an enum by a variant's.
            owner = value[1]
            if name and name != self._name(owner):
                owner = self.members.get(owner, {}).get(name)
            for declaration in self._declarations(owner) if owner else ():
                written = self.builder.fields.get(declaration, {}).get(key)
                if written:
                    return self._returned_type(written, declaration, value)
            return None
        if len(value) < 3:
            return None
        if value[1] == "[]" and step == "*":
            # A tuple's elements share a type only where it writes one.
            return value[2][0] if len(set(value[2])) == 1 else None
        if self.lang(scope_id) == "typescript" and value[1] != "[]" and not name:
            # An element at a position of what is no tuple is what
            # iterating it gives.
            step = "*"
        library = self.standard.get(self.lang(scope_id))
        if step == "*" and value[1] == "iterable":
            # PHP's own name for what a `foreach` reads.
            return value[2][1] if value[2][1:] else None
        if step == "*" and library and library.lang in _ITERATED:
            return library.associated(value, _ITERATED[library.lang])
        if step == "*" and value[1] == "System.Array":
            # A C# array's element, as `foreach` reads it.
            return value[2][0] if value[2] else None
        if step == "*" and library and library.lang in _ITERATOR_MEMBERS:
            # What the iterator the type hands out holds.
            iterator, current = _ITERATOR_MEMBERS[library.lang]
            found, iterator = library.member(value, iterator)
            if not found or not iterator:
                return None
            found, element = library.member(iterator, current)
            return element if found else None
        if step == "*":
            # The first type argument of the iterator a type hands out.
            if not library:
                return None
            iterator = library.member(value, "[Symbol.iterator]")[1]
            return iterator[2][0] if iterator and iterator[2] else None
        if name:
            return library.held(value, name, key) if library else None
        if library and library.lang == "python":
            # A tuple has the type its items share before those of each.
            if value[:2] not in _PY_TUPLES:
                return None
            key = key + 1 if len(value[2]) > 1 else 0
        return value[2][key] if key in range(len(value[2])) else None

    def _go_variable(self, name, scope_id):
        """The package-level Go variable ``name`` that ``scope_id`` has in
        scope and that has a declared type: in any file of its package."""
        package = self._unit(scope_id, "go")
        found = [
            c
            for c in self.by_name.get(("go", name), ())
            if self.nodes[c]["metadata"].get("value_type")
            and self._is_top_level(c)
            and self._unit(c, "go") == package
        ]
        return found[0] if len(found) == 1 else None

    def _go_std_key(self, name, scope_id):
        """What the Go name ``name`` written in ``scope_id`` is called in
        the standard library's data: `http.Request` is `net/http.Request`
        under `import "net/http"`, and a predeclared name is itself unless
        the project declares a type of that name."""
        package, dot, member = name.partition(".")
        imports = self.builder.imports.get(scope_id.split("::")[0], {})
        if dot:
            return f"{imports.get(package, package)}.{member}"
        return None if self._types("go", name) else name

    def _dotnet_std_key(self, name, scope_id):
        """What the C# name ``name``, written in ``scope_id``, is called in
        the standard library's data: the type it names fully, or one of
        an enclosing namespace or of a namespace a `using` gives by that
        name, or the language alias for it — unless the name names a
        project type in scope."""
        types = self.standard["dotnet"].data["types"]
        name = name.replace("::", ".").removeprefix("global.")
        if name in types:
            return name
        head, dot, rest = name.partition(".")
        aliased = self._dotnet_aliased(head, scope_id)
        if dot:
            full = aliased and f"{aliased}.{rest}"
            return full if full in types else None
        if aliased:
            return aliased if aliased in types else None
        # A project type of the name is in scope before a standard one;
        # several it could mean still shadow it.
        if self._dotnet_type([], name, scope_id) != _EXTERNAL:
            return None
        for spaces, _ in self._dotnet_scopes(scope_id):
            keys = sorted(f"{s}.{name}".lstrip(".") for s in spaces)
            key = next((key for key in keys if key in types), None)
            if key:
                return key
        # `string`, `int` and the language's other aliases are its types'.
        return self.standard["dotnet"].data["scope"].get(name)

    def _std_declares(self, name):
        """Whether C++'s library declares ``name`` in `std`: a class
        template of any arity, a function, or a typedef `scope` names."""
        data = self.standard["clang"].data
        return (
            f"std::{name}" in data["functions"]
            or any(f"std::{name}`{n}" in data["types"] for n in range(10))
            or data["scope"].get(name, "").startswith("std::")
        )

    def _clang_std_key(self, name, scope_id):
        """What the C or C++ name ``name``, written in ``scope_id``, is
        called in the standard library's data: `std::vector` is the type
        `std::vector`1`, `vector` the same under `using namespace std`,
        `printf` the C function of that name — unless the name may name a
        project type in scope."""
        data = self.standard["clang"].data
        name = name.lstrip(":")
        if name in data["types"] or name in data["functions"]:
            return name
        base = name.rpartition("`")[0]
        written = base or name

        def generic(path):
            # The data's key of the type ``path`` names, of the most type
            # parameters it has one of.
            arities = (f"{path}`{number}" for number in range(9, -1, -1))
            return next((key for key in arities if key in data["types"]), None)

        if written.startswith("std::"):
            found = generic(written) or (
                written if written in data["functions"] else None
            )
            # A typedef of `std` (`std::true_type`) is what `scope` names,
            # unless the project declares the type.
            scoped = data["scope"].get(written[5:], "")
            declared = self._c_type(["std"], written[5:], scope_id) != _EXTERNAL
            typedef = scoped.startswith("std::") and not declared
            return found or (scoped if typedef else None)
        *qualifier, last = written.split("::")
        if self._c_type(qualifier, last, scope_id) != _EXTERNAL or (
            written in self.builder.c_aliases or written in self.builder.type_aliases
        ):
            return None
        path = f"std::{written}"
        found = path if path in data["types"] else generic(path)
        if found:
            return found
        scoped = data["scope"].get(name) or data["scope"].get(written)
        return (
            self._clang_std_key(scoped, scope_id)
            if scoped and scoped not in (name, written)
            else None
        )

    def _php_std_key(self, name, scope_id):
        """What the PHP class or function ``name``, written in full, is
        called in the standard library's data: itself, unless the project
        declares a class of that name."""
        declared = self._type_named(f"::{name}".replace("\\", "::"), scope_id)
        return name if declared == _EXTERNAL else None

    def _ruby_std_key(self, name, scope_id):
        """What the Ruby class or module ``name``, written bare, is called
        in the standard library's data: itself, unless the project declares
        a constant of that name in scope."""
        declared = self._constant([*name.split("::")], scope_id)
        return name.removeprefix("::") if declared == _EXTERNAL else None

    def _dart_std_key(self, name, scope_id):
        """What the Dart name ``name``, written in ``scope_id``, is called
        in the standard library's data: itself, unless the project's own
        libraries or its `package:` imports declare a type of that name in
        scope."""
        named = self._dart_named(name, scope_id)
        return None if named else name

    def _ts_std_key(self, name, scope_id):
        """What the JavaScript name ``name`` written in ``scope_id`` is
        called in the standard library's data: a global is itself or the
        type it is declared as, unless its file binds the name or the
        project declares a type of that name that is named there."""
        root = name.partition(".")[0]
        bound = self.builder.js_bound.get(scope_id.split("::")[0], ())
        declared = root in self.builder.type_aliases or self._types_in(
            "typescript", root, scope_id
        )
        if root in bound or declared:
            return None
        return self.standard["typescript"].data["scope"].get(name, name)

    def _js_global(self, callee, source_id):
        """``(True, the value returned)`` where the JavaScript ``callee``,
        which has no origin, is a function of the standard library, a
        member of one of its types or a method that the type of one of its
        global objects has."""
        library = self.standard["typescript"]
        owner, _, name = callee.rpartition(".")
        root = owner.partition(".")[0]
        if not owner:
            return library.function(callee, source_id)
        key = library.named(owner, source_id)
        if key and library.key(owner, source_id) == owner:
            return True, library.member(("outside", key, ()), name, table="statics")[1]
        if key:
            return library.member(("outside", key, ()), name)
        # A path through a type, `Object.prototype.toString.call`.
        own = library.key(root, source_id) == root
        return bool(own and library.named(root, source_id)), None

    def _java_std_key(self, name, scope_id):
        """What the Java type ``name`` written in ``scope_id`` is called in
        the standard library's data: by the import of its first name
        (`Map.Entry` is `java.util.Map.Entry` under `import java.util.Map`),
        in a package or class imported on demand, or as a class of
        `java.lang`, unless a project type of that name is in scope."""
        head, dot, rest = name.partition(".")
        imports = self.builder.imports.get(scope_id.split("::")[0], {})
        if self._java_member_type(head, scope_id):
            return None
        if head in imports:
            return f"{imports[head]}{dot}{rest}"
        if self._java_type(head, scope_id):
            return None
        data = self.standard["java"].data
        for star, package in imports.items():
            if star.endswith("*") and f"{package}.{name}" in data["types"]:
                return f"{package}.{name}"
        return f"{data['scope'].get(head, head)}{dot}{rest}"

    def _java_type(self, name, scope_id):
        """The project type that ``name`` stands for where it is written
        in ``scope_id``, a Java type name that no import of the file binds:
        a member type of a class around ``scope_id`` or of one of its
        supertypes, a type of the same package, one of a package or class
        imported on demand, or the class of that full name."""
        file_id = scope_id.split("::")[0]
        key = (self._enclosing_class(scope_id) or file_id, name)
        if key in self.java_scoped:
            return self.java_scoped[key]
        # Inside its own lookup, for a supertype, the name stands for nothing.
        self.java_scoped[key] = None
        head, *path = name.split(".")
        package = self.nodes[file_id]["metadata"].get("java_package")
        found = [self._java_member_type(head, scope_id)]
        found += [
            t
            for t in self._types("java", head)
            if self._is_top_level(t)
            and self.nodes[t.split("::")[0]]["metadata"].get("java_package") == package
        ]
        for star, imported in self.builder.imports.get(file_id, {}).items():
            if star.endswith("*"):
                found.append(self.java_types.get(f"{imported}.{head}"))
                if imported in self.java_types:
                    found.append(self._find_member(self.java_types[imported], head)[0])
        hit = next((f for f in found if f and self._is_class(f)), None)
        for segment in path if hit else ():
            hit = hit and self._find_member(hit, segment)[0]
        # A name written in full, never a simple one, is a class anywhere.
        hit = hit or (path and self.java_types.get(name))
        self.java_scoped[key] = hit if hit and self._is_class(hit) else None
        return self.java_scoped[key]

    def _java_member_type(self, name, scope_id):
        """The member type ``name`` of a class around ``scope_id`` or of one
        of its supertypes, innermost first, which shadows any import of the
        name (JLS 6.4.1); None for none."""
        key = ("member", self._enclosing_class(scope_id), name)
        if key not in self.java_scoped:
            # Inside its own lookup, for a supertype, the name is no member.
            self.java_scoped[key] = None
            around = filter(self._is_class, self._scopes(scope_id))
            found = ((scope, self._find_member(scope, name)[0]) for scope in around)
            hit = next(
                (
                    f
                    for scope, f in found
                    if f
                    and self._is_class(f)
                    # A supertype's private member type is not inherited
                    # (JLS 8.5).
                    and not (
                        self.nodes[f]["metadata"].get("private")
                        and self._member_owner(f) not in self._declarations(scope)
                    )
                ),
                None,
            )
            self.java_scoped[key] = hit
        return self.java_scoped[key]

    def _java_path(self, names, scope_id):
        """Split a dotted Java path read in ``scope_id``, whose first name
        no local declares, into the value it starts with and the names left
        to read from it. The first name is a field of a class around
        ``scope_id`` or of one of its supertypes, failing that a type or
        what an import binds; a longer start may spell out a class with its
        package. ``"external"`` for a path into a package of the standard
        library that names no type of its data, None for any other."""
        root = names[0]
        owner = self._enclosing_class(scope_id)
        if (owner, root) not in self.java_fields:
            # Java keeps a field that shares a method's name under `field_`.
            found = (
                self._find_member(scope, name)[0]
                for scope in filter(self._is_class, self._scopes(owner))
                for name in (root, f"field_{root}")
            )
            fields = (
                hit for hit in found if hit and self.nodes[hit]["type"] == "Constant"
            )
            self.java_fields[owner, root] = next(fields, None)
        if field := self.java_fields[owner, root]:
            # Read without a receiver, it is of the type as written, a type
            # parameter's bound for one.
            written = self.nodes[field]["metadata"]["value_type"]
            held = self._type_argument(written, field, None)
            if held is False:
                held = self._name_value(written.replace(".", "::"), field, set())
            return held, names[1:]
        # A standard type, written with the types it is nested in.
        sizes = range(len(names) - 1, 0, -1)
        nested = (
            n for n in sizes if self._standard_type(".".join(names[:n]), scope_id)
        )
        if size := next(nested, 0):
            return self._returned_type(".".join(names[:size]), scope_id), names[size:]
        # A member type shadows any import of its name (JLS 6.4.1).
        member = self._java_member_type(root, scope_id)
        imported = not member and self._imported(root, scope_id)
        if imported == _EXTERNAL:
            return _EXTERNAL, []
        found = imported[0] if imported else self._java_type(root, scope_id)
        if found:
            return ("def", found), names[1:]
        for size in range(len(names) - 1, 1, -1):
            dotted = ".".join(names[:size])
            if dotted in self.java_types:
                return ("def", self.java_types[dotted]), names[size:]
            if dotted in self.standard["java"].data["types"]:
                return ("outside", dotted, ()), names[size:]
        return (_EXTERNAL if root in _java_packages() else None), []

    def _standard_type(self, written, scope_id):
        """Key in the standard library's data of the type ``written`` in
        ``scope_id``; None for a type the data does not have."""
        library = self.standard.get(self.lang(scope_id))
        return library and library.named(written, scope_id)

    def _std_call(self, callee, source_id):
        """``(True, the value returned)`` where ``callee``, a Rust path or
        a Go name, names a function of the standard library or of one of its
        types, or in Go a type of it, which converts to an instance."""
        library = self.standard[self.lang(source_id)]
        if library.lang == "go":
            named = library.named(callee, source_id)
            if named:
                return True, ("outside", named, ())
            return library.function(callee, source_id)
        owner, _, name = callee.rpartition("::")
        owner = library.named(owner, source_id)
        if owner:
            return library.member(("outside", owner, ()), name, table="statics")
        if library.lang == "clang":
            base, arguments, _ = library.parts(callee)
            key = library.key(base, source_id)
            functions = library.data["functions"]
            if key not in functions:
                return False, None
            returns = functions[key]
            if returns is None:
                return True, None
            # The arguments the call writes are the parameters the
            # written return names, in their order of appearance.
            params = list(dict.fromkeys(re.findall(r"\b_[A-Z]\w*", returns)))
            bound = dict(
                zip(
                    params,
                    (self._returned_type(a, source_id) for a in arguments),
                )
            )
            return True, library.value(returns, bound)
        return library.function(callee, source_id)

    def _py_root(self, module, file_id):
        """The directory the module ``module`` imported in ``file_id`` is
        below: the importer's own where the module is there, else the only
        one it is below; None where it is below several others."""
        roots = self.py_modules[module]
        own = self.py_root_of.get(file_id)
        if own in roots:
            return own
        return next(iter(roots)) if len(roots) == 1 else None

    def _import_value(self, steps, dots, file_id):
        """Split an imported dotted path into the project module it starts
        with and the steps left to take inside it."""
        names = steps[: steps.index("()")] if "()" in steps else steps
        package = []
        if dots:
            package = self.py_module_of.get(file_id, "").split(".")
            if not file_id.endswith("/__init__.py"):
                package = package[:-1]
            package = package[: len(package) - (dots - 1)]
        for size in range(len(names), -1, -1):
            module = ".".join(package + names[:size])
            if module in self.py_modules:
                root = self._py_root(module, file_id)
                if root is None:
                    return None, []
                return ("module", module, root), steps[size:]
        if dots or names[0] in self.py_roots:
            return None, []
        if names[0] in _py_std_names():
            return ("library", names[0]), steps[1:]
        return _EXTERNAL, []

    def _attr(self, value, name, seen=None, called=False):
        """Value of attribute ``name`` of ``value``; ``called`` says the
        attribute is the callee itself rather than part of its receiver."""
        kind, ref = value[:2]
        if kind == "outside":
            # A standard type's member is the library's, read by none of this.
            return None
        if kind == "shape":
            written, scope = self._property_type(ref, name, value[2])
            if called and written and written[:1] in "(<":
                # A method an object type declares has no definition; the
                # call gives what its type returns.
                depth = 0
                for at, char in enumerate(written):
                    depth += (char == "(") - (char == ")")
                    if not depth:
                        break
                returned = written[at + 1 :].strip().removeprefix("=>").strip()
                self._shape_returns = self._returned_type(returned, scope)
                self._result = self._shape_returns or _OPAQUE
                return None
            return written and self._returned_type(written, scope)
        if kind == "library":
            dotted = f"{ref}.{name}"
            return ("library", dotted) if dotted in _py_std_names() else _EXTERNAL
        if kind == "module":
            root = value[2]
            if root in self.py_modules.get(f"{ref}.{name}", ()):
                return ("module", f"{ref}.{name}", root)
            file_id = self.py_modules[ref][root]
            hit = self.members.get(file_id, {}).get(name)
            if hit:
                return ("def", hit)
            seen = set() if seen is None else seen
            # A name the module itself imports is re-exported by it.
            if name in self.builder.imports.get(file_id, {}):
                return self._name_value(name, file_id, seen)
            if name in self.builder.fields.get(file_id, {}):
                return self._variable(file_id, name)
            found = self._starred(file_id, name, seen)[0]
            return found.pop() if len(found) == 1 else None
        if kind == "def" and self.nodes[ref]["type"] != "Class":
            # A field read gives an instance of the type it is declared as;
            # one declared no type holds what it is written with.
            held = self._declared_type(ref)
            hit = not held and self._exported(ref, name, set())
            if hit:
                return ("def", hit)
            if held == _EXTERNAL:
                return _EXTERNAL
            if held and held[0] == "outside":
                # A field declared a type of the standard library has that
                # type's members; a C# field of it may extend them.
                library = self.standard.get(self.lang(ref))
                if not library:
                    return held if not called else None
                known, val = library.member(held, name)
                if known:
                    self._result = val or _OPAQUE
                    return _EXTERNAL if called else val
                if (
                    called
                    and self.lang(ref) == "dotnet"
                    and held[1].split("<")[0] in library.data["types"]
                ):
                    fits = self._extension_fits(
                        self._dotnet_extension(held, name), held
                    )
                    if len(fits) == 1:
                        return ("def", fits[0])
                    if fits:
                        self._candidates_now = fits
                        return None
                    self._result = _OPAQUE
                    return _EXTERNAL
                return None
            if called and held and self.lang(ref) == "dotnet" and held[0] != "self":
                # A C# type from outside has its own members only; what
                # one returns may be the project type it wraps.
                self._result = _OPAQUE
                return _EXTERNAL
            if not held or held[0] != "self":
                return None
            if len(held) > 2:
                # A member of a generic type is read on the instance the
                # field holds, with its arguments.
                self._receiver = held
            return self._attr(held, name, seen, called)
        if kind == "self":
            hit, complete = self._instance_member(ref, name)
        else:
            # A class named, or `this` in a static member, is the
            # constructor, whose members are the static ones.
            hit, complete = self._find_member(
                ref, name, inherited_only=kind == "super", static=kind != "super"
            )
        declared = hit and not called and len(value) > 2 and self.nodes[hit]["metadata"]
        if declared and declared.get("value_type"):
            # A field of a generic type's value is of its type with the
            # value's arguments.
            return self._returned_type(declared["value_type"], hit, value)
        python = self.lang(ref) == "python"
        if hit and not called and self.nodes[hit]["type"] == "Method":
            # Java keeps a field that shares a method's name under `field_`;
            # a Rust one is read as its struct declares it.
            hit = self._find_member(ref, f"field_{name}")[0] or hit
            fields = (self.builder.fields.get(d, {}) for d in self._declarations(ref))
            if self.lang(ref) == "rust" and any(name in f for f in fields):
                hit = None
        if called and self.lang(ref) == "dotnet":
            # A call on a value is of a method of its type or of an extension
            # method written for it that takes what the call passes; one that
            # passes nothing to a method that takes nothing is the method's.
            member = hit and self._overload_by_arity(hit, static=kind == "def")
            if hit and not member and self._candidates_now:
                return None
            extended = (
                kind == "self"
                and not (member and self._arguments == 0)
                and self._extension_hits(ref, name, complete)
            )
            fits = sorted(
                {member, *self._extension_fits(extended or (), ("self", ref))} - {None}
            )
            if len(fits) > 1:
                self._candidates_now = fits
                return None
            if fits or hit:
                return ("def", fits[0]) if fits else None
        if kind in ("def", "cls") and self.lang(ref) == "ruby":
            # A Ruby class that has a singleton method and an instance
            # method of one name has the first as `self.name`, and `new`
            # makes an instance of a class that defines no `new` of its own.
            hit = self._find_member(ref, f"self.{name}")[0] or hit
            made = self.nodes[ref]["metadata"].get("kind") != "module"
            if not hit and name == "new" and made:
                return ("cls", ref)
        if hit:
            return ("def", hit)
        if (
            called
            and kind == "self"
            and self.lang(ref) == "dart"
            and self.nodes[ref]["metadata"].get("kind") == "typedef"
            and name == "call"
        ):
            # A typedef is a function type; calling its `call` calls it.
            self._result = _OPAQUE
            return _EXTERNAL
        # A Go struct has, besides its methods and those of what it embeds,
        # only fields: one that is called holds a function, and the call
        # returns what the field's type, `>Type`, declares.
        if called and complete and self.lang(ref) == "go":
            for declaration in self._declarations(ref):
                if self.nodes[declaration]["metadata"].get("kind") == "struct":
                    written = self.builder.fields.get(declaration, {}).get(name) or ""
                    declared = written.startswith(">") and written[1:]
                    held = declared and self._returned_type(declared, declaration)
                    self._result = held or _OPAQUE
                    return _EXTERNAL
        for owner in (
            self._lineage(ref, set()) if python and complete != _SEVERAL else ()
        ):
            if name in self.builder.fields.get(owner, ()):
                # What is called is a definition the attribute holds.
                held = self._variable(owner, name)
                kept = held == _EXTERNAL or held and held[0] in ("def", "library")
                return held if kept or not called else None
            if self._assigns(owner, name):
                break
        # A C/C++ field that is called holds a function, or an object that
        # is called.
        if called and self.lang(ref) == "clang" and self._c_field(ref, name):
            self._result = _OPAQUE
            return _EXTERNAL
        # A method missing from a type with a base outside the project is
        # inherited from that base, and may return the type; a missing data
        # attribute need not be inherited. A field the type assigns itself is
        # its own, of what value is not known.
        if called and not complete and self._assigns(ref, name):
            return None
        if called and not complete:
            self._result = self._std_inherited(ref, name) or _OPAQUE
            return _EXTERNAL
        if python and not complete:
            return ("returned", self._std_inherited(ref, name))
        if called and self.lang(ref) == "clang" and complete != _SEVERAL:
            # A member the type declares without a definition is defined by
            # the types derived from it: the call reaches one of theirs.
            mine = self._name(ref)
            derived = (
                hit
                for hit in self.by_name.get(("clang", name), ())
                if mine in self._ancestors(hit.rpartition("::")[0])
            )
            self._candidates_now = sorted(derived) or None
        if called or kind != "self":
            return None
        return self._part(value, f"{self._name(ref)}@{name}", ref)

    def _declared_type(self, node_id):
        """Value of the type the variable ``node_id`` is declared as, read
        as ``_returned_type`` reads a return type."""
        declared = self.nodes[node_id]["metadata"].get("value_type")
        called = self.nodes[node_id]["metadata"].get("value_call")
        if called and not declared:
            # A Dart variable without a type holds what the call that is
            # its value returns.
            root, *steps = called.split(".")
            value, steps = self._dart_root(root, steps, node_id)
            for step in steps:
                if not isinstance(value, tuple):
                    break
                if value[0] == "outside":
                    # A standard type's member is the library's.
                    known, value = self.standard["dart"].member(value, step)
                    value = value if known else None
                else:
                    value = self._attr(value, step)
            if isinstance(value, tuple) and value[0] == "outside":
                # What the library says the standard method called returns
                return value
            return self._called(value) if isinstance(value, tuple) else value
        return declared and self._returned_type(declared, node_id)

    def _type_named(self, name, scope_id, full=False):
        """The project type a possibly qualified type name written in
        ``scope_id`` refers to, ``"external"`` for one an import places
        outside the project, None when it cannot be told. With ``full`` a
        C++ alias gives the value of the type it names, its template
        arguments and pointer with it."""
        lang = self.lang(scope_id)
        if lang == "python":
            # A base class is written as its origin path.
            if (scope_id, name) not in self.base_types:
                value = self._value(name, scope_id, set())
                found = None
                if value == _EXTERNAL or value and value[0] == "library":
                    found = _EXTERNAL
                elif value and value[0] == "def" and self._is_class(value[1]):
                    found = value[1]
                self.base_types[scope_id, name] = found
            return self.base_types[scope_id, name]
        if lang == "php":
            # The parser writes a PHP class by its full name, which only the
            # class declared in that namespace has: `::App::Models::User`.
            namespace, _, name = name.removeprefix("::").rpartition("::")
            namespace = namespace.replace("::", "\\")
            types = [
                t
                for t in self._types(lang, name)
                if self.nodes[t]["metadata"].get("namespace") == namespace
            ]
            return self._decide(types, scope_id) if types else _EXTERNAL
        # `::Name` is looked up at the top level alone in Ruby.
        rooted = [""] if lang == "ruby" and name.startswith("::") else []
        parts = [part for part in re.split(r"::|\.", name) if part]
        if not parts:
            # A base that is the value of an expression names no type.
            return None
        *qualifier, name = parts
        if lang == "dotnet":
            return self._dotnet_type(qualifier, name, scope_id)
        if lang == "ruby":
            return self._constant([*rooted, *qualifier, name], scope_id)
        if lang == "clang":
            return self._c_type(qualifier, name, scope_id, full)
        path = "::".join([*qualifier, name])
        if lang == "rust" and qualifier and self._rust_rooted(path, scope_id):
            # A path the module tree tells; one from a crate's name it does
            # not is read as written.
            found = self._rust_path(path.split("::"), scope_id)
            if found == _EXTERNAL:
                return _EXTERNAL
            if found or f"{qualifier[0]}::" in _RUST_PATH_ROOTS:
                one = len(found) == 1 and self._is_class(found[0])
                return found[0] if one else None
        head = qualifier[0] if qualifier else name
        for scope in self._scopes(scope_id) if lang == "rust" else ():
            # An item a function around declares comes before the module's.
            if self._rust_module(scope) == scope:
                break
            hit = f"{scope}::{head}"
            if (
                self.nodes[scope]["type"] in ("Function", "Method")
                and hit in self.nodes
            ):
                for segment in [*qualifier[1:], name][: len(qualifier)]:
                    hit = hit and self._scope_member(hit, segment)
                return hit if hit and self._is_class(hit) else None
        imported = self._imported(head, scope_id)
        if lang == "java" and (
            imported is None or self._java_member_type(head, scope_id)
        ):
            return self._java_type(".".join([*qualifier, name]), scope_id)
        if imported == _EXTERNAL:
            return _EXTERNAL
        if lang == "rust" and imported == []:
            # A `use` binds it to what the module tree does not tell.
            return None
        types = [] if qualifier else [i for i in imported or () if self._is_class(i)]
        if not types and imported:
            # The import names the scope the type is in, in Rust that of
            # the path's first segment, which the others lead on from.
            exported = name if qualifier else self._exported_as(name, scope_id)
            scopes = imported
            for segment in qualifier[1:] if lang == "rust" else ():
                scopes = [h for s in scopes if (h := self._scope_member(s, segment))]
            found = (
                self._scope_member(scope, exported)
                if lang == "rust"
                else self._find_member(scope, exported)[0]
                for scope in scopes
            )
            found = (self.companions.get(hit, hit) for hit in found)
            types = [hit for hit in found if hit and self._is_class(hit)]
        if lang == "dart" and (imported or not qualifier):
            # Only the library's own types and those it imports are in scope.
            named = self._dart_named(name, scope_id, *qualifier[:1])
            if named is not None:
                types = [hit for hit in named if self._is_class(hit)]
                if not types:
                    # `void` and `dynamic` say nothing of the value.
                    unknown = named or name in ("void", "dynamic")
                    return None if unknown else _EXTERNAL
        if lang == "rust" and not (types or qualifier or imported):
            # A type of the module's, or one a glob `use` of it brings
            found = self._rust_path(["self", name], scope_id)
            types = (
                [t for t in found if self._is_class(t)] if found != _EXTERNAL else []
            )
        if not types:
            types = self._types_in(lang, name, scope_id)
            # A JavaScript function names its prototype, which its
            # instances and a class that extends it take their members from.
            held = (
                lang == "typescript"
                and not (types or qualifier)
                and self.companions.get(
                    self._lexical(name, scope_id, skip_classes=True)
                )
            )
            types = [held] if held else types
            if (
                lang == "typescript"
                and imported is None
                and not qualifier
                and not types
                and self._types(lang, name)
            ):
                # Is none of them in reach and no import binds the name, it
                # is not the project's: the writing names the language's
                # global.
                return _EXTERNAL
        if lang == "rust" and qualifier:
            crate = self._use_crate("::".join(qualifier), scope_id)
            if crate == ():
                return _EXTERNAL
            if crate:
                # A crate may re-export the type from another one; which of
                # several of that name is then not told.
                inside = [t for t in types if (self._unit(t, lang),) == crate]
                if not inside and len(types) > 1:
                    return None
                types = inside or types
        hit = self._decide(types, scope_id) if types else None
        defined = [t for t in types if self.nodes[t]["metadata"].get("kind") != "impl"]
        if (
            hit is None
            and len({self._unit(t, lang) for t in types}) == 1
            and (lang != "rust" or len(defined) <= 1)
        ):
            # Declarations of one split type, in one crate or package; a
            # Rust type defined twice is one of two `#[cfg]`s.
            hit = types[0]
        return hit

    def _dotnet_imports(self, file_id):
        """``{name or "<path>.*": dotted path}`` of what the `using`
        directives in force in the whole C# file ``file_id`` name: under a
        name what an alias stands for, under a path with `.*` the namespace
        or the type whose members are in scope; the project's and the
        `global using`s of its project's files with the file's own."""
        folder = posixpath.dirname(file_id[len("FILE:") :])
        imports = {
            alias or f"{space}.*": space
            for space, (alias, static) in self._project_usings(folder).items()
            if alias or not static
        }
        imports.update(self.builder.imports.get(file_id, {}))
        for directory in self._dotnet_compiled_with(folder):
            imports.update(self.builder.dotnet_global_usings.get(directory, {}))
        return imports

    def _dotnet_compiled_with(self, folder):
        """The directories holding C# files of the project ``folder``'s
        files are compiled in: those whose nearest project file is its."""
        own = self._dotnet_nearest(".csproj", folder)
        found = {
            *self.builder.dotnet_global_usings,
            *self.builder.dotnet_global_statics,
        }
        return sorted(d for d in found if self._dotnet_nearest(".csproj", d) == own)

    def _dotnet_nearest(self, name, folder):
        """The .NET project file whose name ends with ``name`` nearest at or
        above ``folder``; None for none."""
        for directory in _ancestors(folder):
            found = sorted(
                path
                for path in self.builder.dotnet_projects
                if posixpath.dirname(path) == directory and path.endswith(name)
            )
            if found:
                return found[0]
        return None

    def _disk(self, question):
        """What the project's directory answers ``question``: for
        ("exists", directory, path), whether ``path`` from ``directory`` is
        there inside the root; for ("above", start, name), the directory at
        or above ``start``, inside the root, holding the file ``name``, ""
        for none."""
        root = Path(self.builder.root_dir).resolve()
        kind, start, path = question
        if kind == "exists":
            target = (root / start / path.replace("\\", "/")).resolve()
            return target.is_relative_to(root) and target.exists()
        directory = Path(os.path.normpath(root / start.replace("\\", "/")))
        while directory.is_relative_to(root):
            if (directory / path).is_file():
                return directory.as_posix()
            if directory == root:
                break
            directory = directory.parent
        return ""

    def _project_usings(self, folder):
        """The `<Using>` items in force for the C# files in ``folder`` as
        MSBuild evaluates their .NET project, ``{namespace or type: (alias,
        static)}``: those of the nearest `Directory.Build.props` at or above
        the project file, then the SDK's implicit usings where
        `<ImplicitUsings>` is on, then the project file's own and those of
        the nearest `Directory.Build.targets`; for several target
        frameworks, those of any of its builds."""
        if folder in self._usings_of:
            return self._usings_of[folder]
        projects = self.builder.dotnet_projects
        project = self._dotnet_nearest(".csproj", folder)
        start = posixpath.dirname(project) if project else folder
        root = Path(self.builder.root_dir).resolve()
        file = posixpath.basename(project) if project else ""
        name = file.rpartition(".")[0]
        files = [
            self._dotnet_nearest("Directory.Build.props", start),
            project,
            self._dotnet_nearest("Directory.Build.targets", start),
        ]

        def exists(here):
            return lambda path: self.disk(("exists", here, path))

        def above(start, name):
            return self.disk(("above", start, name))

        def evaluate(fixed):
            """The items of a build whose global properties are ``fixed``,
            and the properties it ends with."""
            # Property names, which MSBuild reads in any case, in lower
            # case: the reserved ones, a C# project's `Language`, and those
            # of the build
            properties = {
                "language": "C#",
                "msbuildprojectname": name,
                "msbuildprojectfile": file,
                "msbuildprojectextension": file and f".{file.rpartition('.')[2]}",
                "msbuildprojectdirectory": (root / start).as_posix(),
                "msbuildprojectfullpath": (root / (project or "")).as_posix(),
                "msbuildruntimetype": "Core",
                "os": "Windows_NT" if os.name == "nt" else "Unix",
                **fixed,
            }

            def within(path):
                """The properties as read in the MSBuild file ``path``."""
                whole = (root / path).as_posix()
                return {
                    **properties,
                    "msbuildthisfile": posixpath.basename(path),
                    "msbuildthisfilename": posixpath.basename(path).rpartition(".")[0],
                    "msbuildthisfileextension": posixpath.splitext(path)[1],
                    "msbuildthisfilefullpath": whole,
                    "msbuildthisfiledirectory": f"{posixpath.dirname(whole)}/",
                }

            def read(path, seen):
                """The `<Using>` items of the MSBuild file ``path`` and of
                those it imports, in order, as ``(file, kind, name, [alias,
                static], conditions)``, the properties set as read."""
                here, items = posixpath.dirname(path), []
                for kind, written, *rest in projects[path][1]:
                    if kind not in ("import", "property"):
                        items.append((path, kind, written, *rest))
                        continue
                    held = within(path)
                    try:
                        holds = _msbuild_hold(rest[-1], held, exists(here), above)
                        value = _msbuild_expand(
                            rest[0] if kind == "property" else written, held, above
                        )
                    except ValueError:
                        # A form or function MSBuild reads and this does not
                        continue
                    if holds and kind == "property" and written.lower() not in fixed:
                        properties[written.lower()] = value
                    elif holds and kind == "import":
                        target = os.path.relpath(
                            os.path.normpath(root / here / value.replace("\\", "/")),
                            root,
                        ).replace(os.sep, "/")
                        if target in projects and target not in seen:
                            items += read(target, seen | {target})
                return items

            # MSBuild reads `Directory.Build.props`, the project file, then
            # `Directory.Build.targets`: every property first, then every
            # item. The SDK gives `Configuration` after the first.
            groups = []
            for found in files:
                groups.append(read(found, {found}) if found else [])
                properties.setdefault("configuration", "Debug")
            # What the SDK gives a property the project leaves unset
            framework = (
                properties.get("targetframework")
                or properties.get("targetframeworks", "").partition(";")[0]
            )
            properties.setdefault(
                "targetframeworkidentifier", _framework_kind(framework)
            )
            properties.setdefault("rootnamespace", name)
            properties.setdefault("assemblyname", name)
            usings = {}

            def apply(kind, written, how, held):
                if held:
                    written = _msbuild_expand(written, properties, above)
                    if kind == "Include":
                        usings[written] = how
                    else:
                        usings.pop(written, None)

            def apply_all(items):
                for path, kind, written, how, conditions in items:
                    here = posixpath.dirname(path)
                    try:
                        held = _msbuild_hold(
                            conditions, within(path), exists(here), above
                        )
                    except ValueError:
                        # Unread, a removal is made, an inclusion is not:
                        # what is in scope is never widened.
                        held = kind == "Remove"
                    apply(kind, written, how, held)

            apply_all(groups[0])
            if properties.get("implicitusings", "").lower() in ("enable", "true"):
                listed = self.standard["dotnet"].data["implicit_usings"]
                sdk = projects[project][0] if project else None
                for kind, space, *conditions in [
                    *listed.get("Microsoft.NET.Sdk", ()),
                    *(listed.get(sdk, ()) if sdk != "Microsoft.NET.Sdk" else ()),
                ]:
                    held = all(
                        (properties.get(name.lower(), "").lower() == value.lower())
                        == (op == "==")
                        for name, op, value in (conditions[0] if conditions else ())
                    )
                    apply(kind, space, ("", False), held)
            apply_all(groups[1] + groups[2])
            return usings, properties

        usings, properties = evaluate({})
        frameworks = properties.get("targetframeworks", "").split(";")
        frameworks = [f.strip() for f in frameworks if f.strip()]
        if len(frameworks) > 1 and not properties.get("targetframework"):
            # One build per framework, which sets `TargetFramework`
            usings = {}
            for framework in frameworks:
                usings.update(evaluate({"targetframework": framework})[0])
        self._usings_of[folder] = usings
        return usings

    def _dotnet_file_usings(self, file_id):
        """A C# file's `using` directives, and the `global using static`
        ones of its project's files and items, as ``dotnet_usings``
        entries."""
        folder = posixpath.dirname(file_id[len("FILE:") :])
        usings = list(self.builder.dotnet_usings.get(file_id, ()))
        usings += [
            (space, "", False, True, False, None, None)
            for space, (alias, static) in self._project_usings(folder).items()
            if static and not alias
        ]
        for directory in self._dotnet_compiled_with(folder):
            usings += self.builder.dotnet_global_statics.get(directory, ())
        return usings

    def _dotnet_usings_at(self, scope_id):
        """The ``dotnet_usings`` entries in force at the C# code of
        ``scope_id``: of the file, and of each namespace declaration around
        it — not of another declaration of the same namespace."""
        levels = self._dotnet_levels(scope_id)
        line = self.nodes.get(scope_id, {}).get("metadata", {}).get("lineno")
        return [
            entry
            for entry in self._dotnet_file_usings(scope_id.split("::")[0])
            if entry[1] in levels
            and (
                entry[6] is None
                or line is not None
                and entry[6][0] <= line <= entry[6][1]
            )
        ]

    def _dotnet_namespace(self, node_id):
        """The namespace the C# code of ``node_id`` is written in."""
        top = "::".join(node_id.split("::")[:2])
        metadata = self.nodes[top if top in self.nodes else node_id.split("::")[0]][
            "metadata"
        ]
        return metadata.get("namespace") or metadata.get("dotnet_namespace")

    def _dotnet_levels(self, node_id):
        """The namespaces around the C# code of ``node_id``, innermost first,
        down to the global one, ""."""
        here = self._dotnet_namespace(node_id)
        parts = here.split(".") if here else []
        return [".".join(parts[:size]) for size in range(len(parts), -1, -1)]

    def _dotnet_rivals(self, extensions, spaces, source_id):
        """``(candidates, nearer)`` for a C# call in ``source_id`` of a
        method that the standard library's extension methods of the
        namespaces ``spaces`` give the receiver, and the project extension
        methods ``extensions`` may give it. Lookup goes out from the
        innermost namespace around the call to the file: at each, the
        types of that namespace and of those the `using`s written in its
        body import; the first that has an applicable one is the call's.
        A `using` written in a namespace is in scope only for code it
        encloses, and names the first of its path under each enclosing
        namespace that is one the project or the library declares,
        `global::` the path alone; `using static` names a type's members,
        extension methods among them. The candidates are those of
        ``extensions`` at the first level that has any, none where the
        library's come first, and they are nearer where they come before
        the library's. A library namespace the file does not import may be
        imported for every file, at the outermost level; a file that
        imports one of them in an `#if` arm is read as any build."""
        file_id = source_id.split("::")[0]
        levels = self._dotnet_levels(source_id)
        usings = self._dotnet_usings_at(source_id)
        everywhere = set(self._dotnet_imports(file_id).values()) - set(
            self.builder.imports.get(file_id, {}).values()
        )
        named = {e: self._dotnet_namespace(e) or "" for e in extensions}
        typed = {e: self._dotnet_path(e.rpartition("::")[0]) for e in extensions}
        types = self.dotnet_paths | set(self.standard["dotnet"].data["types"])
        namespaces = self.dotnet_spaces | set(spaces)
        # The namespace each `using` in scope imports, and the type each
        # `using static` brings the members of, at the level of the
        # namespace it is written in.
        imported, static, armed = {}, {}, []
        for written, at, arm, is_static, rooted, alias, _ in usings:
            if alias:
                continue
            resolved = self._dotnet_using(
                written, at, rooted, types if is_static else namespaces
            )
            if arm:
                armed.append(resolved)
            held = static if is_static else imported
            held[resolved] = min(levels.index(at), held.get(resolved, len(levels)))

        def level(space):
            found = [at for at, name in enumerate(levels) if name == space]
            if space in imported:
                found.append(imported[space])
            return min(found + [len(levels) - 1] * (space in everywhere), default=None)

        found = {}
        for e, space in named.items():
            given = [level(space), static.get(typed[e])]
            at = min((a for a in given if a is not None), default=None)
            if at is not None:
                found[e] = at
        first = min(found.values(), default=None)
        library = min(
            [
                *[at for at in map(level, spaces) if at is not None],
                *(
                    at
                    for path, at in static.items()
                    if path.rpartition(".")[0] in spaces
                ),
            ],
            default=len(levels) - 1,
        )
        relevant = {*spaces, *named.values(), *typed.values()}
        guarded = any(path in relevant for path in armed)
        if first is None or first > library and not guarded:
            return [], False
        return [e for e, at in found.items() if at == first], (
            first < library and not guarded
        )

    def _dotnet_scopes(self, scope_id):
        """``(namespaces, {alias: path})`` groups a C# type name written in
        ``scope_id`` is looked up in, in turn: for each namespace around the
        code, innermost first, the namespace itself, then what the `using`s
        of the declaration of it around the code import and alias; at the
        global one the file's own `using`s with the project's and the global
        ones."""
        file_id = scope_id.split("::")[0]
        levels = self._dotnet_levels(scope_id)
        imported = [set() for _ in levels]
        aliases = [{} for _ in levels]
        for written, at, _, is_static, rooted, alias, _ in self._dotnet_usings_at(
            scope_id
        ):
            if at in levels[:-1] and not is_static:
                # An alias may name a type, read from the namespace around
                # it as a namespace is.
                pool = self.dotnet_spaces | (self.dotnet_paths if alias else set())
                held = self._dotnet_using(written, at, rooted, pool)
                if alias:
                    aliases[levels.index(at)][alias] = held
                else:
                    imported[levels.index(at)].add(held)
        for key, held in self._dotnet_imports(file_id).items():
            if key.endswith(".*"):
                imported[-1].add(held)
            else:
                aliases[-1][key] = held
        return [
            group
            for space, more, named in zip(levels, imported, aliases)
            for group in (({space}, {}), (more, named))
        ]

    def _dotnet_owners(self, scope):
        """The declarations of the C# class ``scope`` and of the types it
        derives from, whose nested types its code names bare."""
        if scope not in self.dotnet_owners:
            # Inside its own lookup, for a base, the class alone
            self.dotnet_owners[scope] = set(self._declarations(scope))
            self.dotnet_owners[scope] = {
                d
                for t, _ in self._lookup_order(scope, "")
                for d in self._declarations(t)
            }
        return self.dotnet_owners[scope]

    def _dotnet_aliased(self, name, scope_id):
        """The path the C# alias ``name`` stands for where it is written in
        ``scope_id``: the alias of the nearest group (see `_dotnet_scopes`)
        that has one, unless a type of the name comes first; None for
        none."""
        library = self.standard["dotnet"].data["types"]
        spaces = {
            self._dotnet_path(t).rpartition(".")[0]
            for t in self._types("dotnet", name)
            if self._is_top_level(t)
        }
        for held, aliases in self._dotnet_scopes(scope_id):
            if name in aliases:
                return aliases[name]
            if held & spaces or any(f"{s}.{name}".lstrip(".") in library for s in held):
                return None
        return None

    def _dotnet_expands(self, path, at):
        """The names a path written in the C# namespace ``at`` may name:
        the path under each enclosing namespace, innermost first, then as
        written."""
        parts = at.split(".") if at else []
        return [".".join([*parts[:size], path]) for size in range(len(parts), -1, -1)]

    def _dotnet_imported(self, scope_id, static, pool):
        """What the `using`s in scope at ``scope_id`` import, each resolved
        against ``pool``: namespaces, or with ``static`` the types whose
        members `using static` brings in."""
        return [
            self._dotnet_using(written, at, rooted, pool)
            for written, at, _, is_static, rooted, alias, _ in self._dotnet_usings_at(
                scope_id
            )
            if is_static == static and not alias
        ]

    def _dotnet_using(self, written, at, rooted, pool):
        """The name a `using` of ``written`` in the C# namespace ``at``
        imports: the first of the names it expands to that ``pool`` holds,
        else the name as written; `global::` names ``written`` itself."""
        if rooted:
            return written
        names = self._dotnet_expands(written, at)
        return next((name for name in names if name in pool), names[-1])

    def _dotnet_path(self, type_id):
        """Dotted name of the C# type ``type_id`` from the global namespace."""
        parts = type_id.split("::")
        namespace = self.nodes["::".join(parts[:2])]["metadata"].get("namespace")
        names = [
            self._name("::".join(parts[:size])) for size in range(2, len(parts) + 1)
        ]
        return ".".join(filter(None, [namespace, *names]))

    def _dotnet_type(self, qualifier, name, scope_id):
        """The project type that the C# name ``name`` means where it is
        written behind ``qualifier`` in ``scope_id``: one nested in an
        enclosing type or in a base of one, else one of the nearest
        enclosing namespace that has one, else one of a namespace the file
        imports. ``"external"`` where no type the project declares is in
        scope by the name, None where several are."""
        file_id = scope_id.split("::")[0]
        aliased = self._dotnet_aliased(qualifier[0] if qualifier else name, scope_id)
        if aliased:
            written = [*qualifier[1:], name] if qualifier else []
            *qualifier, name = [*aliased.split("."), *written]
        paths = {t: self._dotnet_path(t) for t in self._types("dotnet", name)}
        hits = []
        if qualifier and aliased:
            # An alias stands for a path from the global namespace.
            written = ".".join([*qualifier, name])
            hits = [t for t, path in paths.items() if path == written]
        elif qualifier:
            # The first name is a namespace or a type of a namespace around
            # the code, or a type a `using` imports.
            written = ".".join([*qualifier, name])
            for index, (spaces, _) in enumerate(self._dotnet_scopes(scope_id)):
                hits = [
                    t
                    for t, path in paths.items()
                    for space in spaces
                    if path == f"{space}.{written}".lstrip(".")
                    and (
                        index % 2 == 0
                        or f"{space}.{qualifier[0]}".lstrip(".") in self.dotnet_paths
                    )
                ]
                if hits:
                    break
        if qualifier and not hits:
            return _EXTERNAL
        if not paths:
            return _EXTERNAL
        nested = [t for t in paths if self._is_class(t.rpartition("::")[0])]
        for scope in self._scopes(scope_id):
            if not hits and self._is_class(scope):
                owners = self._dotnet_owners(scope)
                hits = [t for t in nested if t.rpartition("::")[0] in owners]
        if not hits:
            library = self.standard["dotnet"].data["types"]
            outer = {
                t: path.rpartition(".")[0]
                for t, path in paths.items()
                if self._is_top_level(t)
            }
            for spaces, _ in self._dotnet_scopes(scope_id):
                hits = [t for t, space in outer.items() if space in spaces]
                if hits or any(f"{s}.{name}".lstrip(".") in library for s in spaces):
                    break
            if not hits:
                return _EXTERNAL
        if len({paths[t] for t in hits}) == 1:
            # One type, declared in parts: the caller's file's part, if any.
            own = [t for t in hits if t.split("::")[0] == file_id]
            return (own or hits)[0]
        self._candidates_now = sorted(hits)
        return None

    def _c_unmacroed(self, qualifier):
        """``qualifier`` as the object-like macros among its names expand: a
        macro that stands for a namespace as that namespace, one that stands
        for nothing, an attribute or a specifier as nothing. A name that also
        names a scope is the scope's."""
        unmacroed = []
        for name in qualifier:
            held = None if name in self.c_scopes else self.c_macros.get(name)
            if held is None:
                unmacroed.append(name)
            elif held.strip(":").split("::")[0] in self.c_scopes:
                unmacroed += held.strip(":").split("::")
            elif not _C_MARK_RE.fullmatch(held):
                unmacroed.append(name)
        return unmacroed

    def _c_written(self, qualifier, held, scope_id):
        """``qualifier``, written in ``scope_id``, as it names a definition:
        without the namespaces macros open, which have no node, unless it
        names, from ``scope_id`` or a namespace around it, one that holds
        one of ``held``. A macro among its names is read as it expands."""
        qualifier = self._c_unmacroed(qualifier)
        parents = {p for h in held if (p := h.rpartition("::")[0]) in self.nodes}
        if any(self._c_names(p, qualifier, scope_id) for p in parents):
            return qualifier
        return [q for q in qualifier if q not in self.c_spaces]

    def _c_names(self, parent, qualifier, scope_id):
        """How deep in ``scope_id`` the namespace is from which
        ``qualifier``, written there, names the scope ``parent``, counting
        the global one; 0 where it names it from none."""
        path, around = parent.split("::")[1:], scope_id.split("::")[1:]
        outer = path[: len(path) - len(qualifier)]
        named = (
            qualifier
            and path[len(outer) :] == qualifier
            and around[: len(outer)] == outer
        )
        return len(outer) + 1 if named else 0

    def _c_path(self, node_id):
        """The qualified name of ``node_id`` without the namespaces macros
        open, which have no node: the name it has however they are opened,
        a macro filed apart from a function of its name by its own."""
        *path, last = [p for p in node_id.split("::")[1:] if p not in self.c_spaces]
        macro = self.nodes[node_id]["metadata"].get("kind") == "macro"
        return [*path, self._name(node_id) if macro else last]

    def _c_instances(self, special):
        """The names of the C++ classes that an instance of a template, named
        with its arguments as `c_specialization` writes it in ``special``,
        may be of: of the specializations its arguments match, the one more
        specialized than each other, else the template; where that is not
        told, as an alias or a type parameter may make one match, each that
        may be the one. Both are read with the template's default arguments
        for those they leave out."""
        template, _, arguments = special.partition("<")
        written = self.c_specializations.get(template)
        if not written:
            return [template]
        primaries = set()
        for t in self._types("clang", template):
            metadata = self.nodes[t]["metadata"]
            if "generics" in metadata:
                # C++ merges the default arguments of every declaration.
                declared = self.builder.c_defaults.get(t.partition("::")[2], ())
                given = zip_longest(
                    metadata.get("defaults", ()), *declared, fillvalue=""
                )
                merged = tuple(next(filter(None, d), "") for d in given)
                primaries.add((metadata["generics"], merged))
        generics, defaults = primaries.pop() if len(primaries) == 1 else ((), ())

        def completed(listed, params=()):
            """``listed``, with the default arguments of the primary
            template for the parameters it leaves out, written with their
            parameters' arguments, and `?` for each argument a macro other
            than ``params`` is written in, which may stand for anything."""
            rest = defaults[len(listed) :]
            binding = dict(zip(generics, listed))
            listed = listed + [
                re.sub(
                    r"\s*::\s*",
                    ".",
                    re.sub(r"\w+", lambda w: binding.get(w[0], w[0]), d),
                )
                for d in (rest if all(rest) else ())
            ]
            return [
                "?"
                if any(
                    w in self.c_macros and w not in params
                    for w in re.findall(r"\w+", a)
                )
                else a
                for a in listed
            ]

        parts = {
            name: completed(_bracketed_parts(name.partition("<")[2]), params)
            for name, params in written.items()
        }
        given = completed(_bracketed_parts(arguments))

        def classed(template):
            """Whether ``template`` names a class template of the project
            or of the library."""
            if template.startswith("std."):
                return bool(
                    self.standard["clang"].named(template.replace(".", "::"), "")
                )
            return any(
                "generics" in self.nodes[t]["metadata"]
                for t in self._types("clang", template.rpartition(".")[2])
            )

        # A project alias, a library alias, a member a dependent `typename`
        # names and a template that names no class may stand for any type.
        spelled = [*given, *(p for pattern in parts.values() for p in pattern)]
        templates = re.findall(r"([\w.]+)\s*<", " ".join(spelled))
        unknown = (
            self.builder.type_aliases
            | _std_aliases()
            | {"typename", "?"}
            | {t for t in templates if not classed(t)}
        )
        told = {
            name: _c_deduced(parts[name], given, params, unknown)
            for name, params in written.items()
        }

        def under(general, special, unknown=()):
            """Whether ``special``'s pattern is an instance of ``general``'s,
            where names of ``unknown`` may stand for any type."""
            return _c_deduced(parts[general], parts[special], written[general], unknown)

        sure = [name for name in written if told[name]]
        best = [
            name
            for name in sure
            if all(
                other == name or under(other, name) and not under(name, other)
                for other in sure
            )
        ]
        maybe = [name for name in written if told[name] is None]
        if len(best) == 1:
            # Valid code matches no two specializations neither of which is
            # more specialized: one that may match is the instance's only
            # where it may be more specialized than the one that does.
            maybe = [
                m
                for m in maybe
                if under(best[0], m, unknown.difference(written[m])) is not False
            ]
        return (best if len(best) == 1 else sure or [template]) + maybe

    def _c_specialized(self, written, scope_id):
        """What an instance of the C++ type ``written`` with template
        arguments in ``scope_id`` is: one of the project's specialization it
        is of, or where that is not told a union of one of each type it may
        be of, None for a type of no project's; False where it is of the
        template itself."""
        name, _, arguments = written.partition("<")
        *qualifier, template = name.strip().split("::")
        instances = self._c_instances(c_specialization(template, arguments))
        if instances == [template]:
            return False
        values = []
        for instance in instances:
            found = self._c_type([q for q in qualifier if q], instance, scope_id)
            values.append(("self", found) if found and found != _EXTERNAL else None)
        return values[0] if len(values) == 1 else ("union", tuple(values))

    def _c_type(self, qualifier, name, scope_id, full=False):
        """``_type_named`` for C and C++, which import no names: the type
        is the one of that name nearest in scope, else the one among the
        files ``scope_id`` is compiled with, and a name that no type or
        alias of the project carries is from outside it. An alias is the
        type it names where it is declared."""
        types = [
            t
            for t in self._types("clang", name)
            if self.nodes[t]["metadata"].get("kind") != "namespace"
        ]
        qualifier = self._c_written(qualifier, types, scope_id)
        aliased = {name, *qualifier[-1:]} & self.builder.type_aliases
        aliases = self.builder.c_aliases.get(name, ())
        seen = self._sees(scope_id.split("::")[0])
        if qualifier:
            # The types the qualifier names in full, else those in a scope
            # of its last name.
            # Of those it names in full, the ones from the innermost scope.
            depths = {
                t: self._c_names(t.rpartition("::")[0], qualifier, scope_id)
                for t in types
            }
            inside = [
                t for t in types if depths[t] and depths[t] == max(depths.values())
            ] or [
                t for t in types if self._name(t.rpartition("::")[0]) == qualifier[-1]
            ]
            # The qualifier may be an alias of the type the member is in.
            owner = not inside and self._c_type(qualifier[:-1], qualifier[-1], scope_id)
            if owner and owner != _EXTERNAL:
                owners = self._declarations(owner)
                inside = [t for t in types if t.rpartition("::")[0] in owners]
            types = inside
            aliases = [a for a in aliases if self._name(a[0]) == qualifier[-1]]
        else:
            # A member of an enclosing scope, or of the type that an
            # enclosing scope holds the out-of-class members of.
            for scope in self._scopes(scope_id):
                scopes = self._declarations(scope) if self._is_class(scope) else [scope]
                inner = {self.members.get(s, {}).get(name) for s in scopes}
                if inner & set(types):
                    hit = (inner & set(types)).pop()
                    reached = self._candidates_now
                    hit = self._c_overload(hit, types, scope_id)
                    self._candidates_now = reached
                    return hit
                written = [a for a in aliases if a[0] in scopes]
                if written:
                    types, aliases = [], written
                    break
                # The library's members of `std` are found in `namespace
                # std` before the scopes around it.
                std = self.nodes.get(scope, {}).get("metadata", {}).get("kind")
                if (
                    std == "namespace"
                    and self._name(scope) == "std"
                    and self._is_top_level(scope)
                    and self._std_declares(name)
                ):
                    return _EXTERNAL
            # A member of any other type takes that type's name to reach,
            # as does an alias of a namespace that is not around the scope.
            types = [t for t in types if self._is_top_level(t)]
            path = scope_id.split("::")[1:]
            aliases = [
                a for a in aliases if (at := a[0].split("::")[1:]) == path[: len(at)]
            ]
        # A type of a file it does not see may come from a header an include
        # not followed brings; one of another source file is not in scope.
        known = any(t.split("::")[0].endswith(_C_HEADER_EXTENSIONS) for t in types)
        types = [t for t in types if self._in_reach(t, scope_id)]
        if not types:
            # The aliases agree on the type, whatever its arguments; one of
            # a pointer names the type it points at.
            named, values = set(), set()
            for scope, written in aliases:
                if scope.split("::")[0] in seen:
                    value = self._returned_type(written, scope)
                    values.add(value)
                    held = value[1] if value and value[0] == "pointer" else value
                    named.add(held[:2] if isinstance(held, tuple) else held)
            held = named.pop() if len(named) == 1 else None
            if full and len(values) == 1 and held and held[0] in ("self", "outside"):
                return values.pop()
            if held == _EXTERNAL or (held and held[0] == "self"):
                return held if held == _EXTERNAL else held[1]
            return None if aliased or known else _EXTERNAL
        # The innermost of the namespaces around the scope that has one.
        scope = scope_id.split("::")[1:]
        depth = {t: len(t.split("::")) - 2 for t in types}
        inside = [t for t in types if t.split("::")[1:-1] == scope[: depth[t]]]
        types = [t for t in inside if depth[t] == max(map(depth.get, inside))] or types
        # Two definitions are different types: specialisations of a template.
        if len({self._unit(t, "clang") for t in types}) == 1:
            return types[0]
        self._candidates_now = sorted(types)
        return None

    def _sees(self, file_id):
        """IDs of the C/C++ files ``file_id`` is compiled with: those that
        it, and every file that includes it, include in any number of
        steps."""
        if file_id not in self.units:
            units = _closure(self.includers, {file_id})
            self.units[file_id] = _closure(self.includes, units)
        return self.units[file_id]

    def _macro_in_force(self, macro, source_id):
        """Whether the macro ``macro`` is defined at the call being resolved
        in ``source_id``: the last `#define` or `#undef` of its name above
        the call is its own `#define`, a file included above read where it
        is included. One the call's file meets nowhere may be defined where
        that file is included. A macro's body is read where the macro is
        used, so a call in one takes any."""
        if self.nodes[source_id]["metadata"].get("kind") == "macro":
            return True
        file_id, line = source_id.split("::")[0], self._line and self._line[0]
        name = self._name(macro)
        if line is None:
            return True
        events = self._macro_events(file_id, name)
        above = [(defined, path) for at, defined, path in events if at < line]
        if not above:
            return not events
        defined, path = above[-1]
        return defined and path == macro.split("::")[0]

    def _macro_events(self, file_id, name):
        """``[(line, defined, file)]``: each `#define` (True) and `#undef`
        (False) of the macro ``name`` as compiling ``file_id`` reads them, in
        order, at the line of ``file_id`` read, an included file's at its
        `#include`, and the file it is written in. A file its include guard
        or `#pragma once` keeps is read once; any other at each `#include`,
        but within itself."""
        key = (file_id, name)
        if key not in self._macro_events_of:
            events, read = [], {file_id}

            def walk(path, at, reading):
                entries = [
                    *self.builder.c_macro_lines.get(path, {}).get(name, ()),
                    *self.include_lines.get(path, ()),
                ]
                for line, event in sorted(entries, key=lambda e: e[0]):
                    here = line if at is None else at
                    if isinstance(event, bool):
                        events.append((here, event, path))
                        continue
                    for included in event:
                        once = included in self.builder.c_once
                        if included not in reading and not (once and included in read):
                            read.add(included)
                            walk(included, here, reading | {included})

            walk(file_id, None, frozenset({file_id}))
            self._macro_events_of[key] = events
        return self._macro_events_of[key]

    def _in_reach(self, node_id, scope_id):
        """Whether C/C++ code in ``scope_id`` can name ``node_id``. A type,
        a macro, a function with internal linkage and whatever a header
        defines are known only to the files compiled with the one that
        holds them."""
        node = self.nodes[node_id]
        kind = node["metadata"].get("kind")
        file_id = node_id.split("::")[0]
        local = (
            kind == "macro"
            or node["metadata"].get("linkage") == "internal"
            or (node["type"] == "Class" and kind != "namespace")
            or file_id.endswith(_C_HEADER_EXTENSIONS)
        )
        return not local or file_id in self._sees(scope_id.split("::")[0])

    def _c_alias_first(self, name, reached, source_id):
        """Whether, of the scopes around the C/C++ code of ``source_id``,
        innermost first, one that declares an alias ``name`` the code sees
        comes before one that has a member of the name, its own or a
        base's, or holds one of the definitions ``reached``."""
        seen = self._sees(source_id.split("::")[0])
        aliases = [
            scope
            for scope, _ in self.builder.c_aliases[name]
            if scope.split("::")[0] in seen
        ]
        for scope in self._scopes(source_id):
            level = scope.count("::")
            if any(self._c_depth(alias, source_id) == level for alias in aliases):
                return True
            if (self._is_class(scope) and self._find_member(scope, name)[0]) or any(
                self._c_depth(c.rpartition("::")[0], source_id) == level
                for c in reached
            ):
                return False
        return False

    def _c_depth(self, scope, source_id):
        """How many scopes deep ``scope`` encloses the C/C++ code of
        ``source_id``, -1 where it does not: a function by its own node, a
        namespace, type or file wherever it is declared, by its path."""
        at = scope.split("::")[1:]
        if self.nodes.get(scope, {}).get("type") in ("Function", "Method"):
            inside = source_id == scope or source_id.startswith(f"{scope}::")
        else:
            inside = source_id.split("::")[1 : len(at) + 1] == at
        return len(at) if inside else -1

    def _c_field(self, owner_id, name):
        """Whether the C/C++ type ``owner_id`` declares a field ``name``,
        itself or through a macro that stands in its body for fields."""
        declared = set()
        for declaration in self._declarations(owner_id):
            declared.update(self.builder.fields.get(declaration, ()))
        pending = [key[1:-1] for key in declared if key.startswith("<")]
        while pending and name not in declared:
            for node_id in self.by_name.get(("clang", pending.pop()), ()):
                for field, macro in self.nodes[node_id]["metadata"].get("fields", ()):
                    if (field or f"<{macro}>") not in declared:
                        declared.add(field or f"<{macro}>")
                        pending += [macro] if macro else []
        return name in declared

    def _c_rewritten(self, macro, index, scope_id, seen=()):
        """How the C/C++ macros named ``macro`` that ``scope_id`` can name
        write their parameter ``index`` (see ``clang_parser._rewrites``),
        followed through the macros they pass it on to."""
        found = set()
        for node_id in self.by_name.get(("clang", macro), ()):
            rewrites = self.nodes[node_id]["metadata"].get("rewrites", ())
            if (macro, index) in seen or not self._in_reach(node_id, scope_id):
                continue
            for at, how in rewrites:
                if str(at) == index and how.endswith(")"):
                    passed = (*how[:-1].split("("), scope_id, {*seen, (macro, index)})
                    found |= self._c_rewritten(*passed)
                elif str(at) == index:
                    found.add(how)
        return found

    def _c_overload(self, hit, candidates, scope_id):
        """The C++ definition the call that reached ``hit`` is a call of,
        among those of its qualified name that ``scope_id`` is compiled
        with: the one of them that takes what the call passes, else None,
        with them as the candidates. An overload or a specialisation in
        another file is among ``candidates``, in a file
        compiled with it or, of external linkage, under a name a prototype
        declares in a header both files are compiled with, where the caller
        is C++ (a `.c` file is compiled as C, which has no overloads).
        Definitions an `#if` arm alone builds are alternatives, not
        overloads."""
        path = self._c_path(hit)
        file_id = scope_id.split("::")[0]
        seen = self._sees(file_id)
        defined = {c for c in candidates if "end_lineno" in self.nodes[c]["metadata"]}
        macro = self.nodes[hit]["metadata"].get("kind") == "macro"
        guards = self._caller_guards(scope_id)
        if hit not in defined or macro:
            return hit
        others = [
            c
            for c in candidates
            if c != hit
            and c in defined
            and self._c_path(c) == path
            and self.takes(c, self._arguments)
            and (
                c.split("::")[0] in seen
                or not file_id.endswith(".c")
                and self.nodes[c]["metadata"].get("linkage") != "internal"
                and any(
                    c.rpartition("::")[2] in self.builder.declares.get(file, ())
                    for file in seen & self._sees(c.split("::")[0])
                )
            )
            and not self._guard_conflict(guards, self._effective_guards(c, scope_id))
        ]
        fits = [c for c in [hit, *others] if self.takes(c, self._arguments)]
        if not others or len(fits) == 1:
            return fits[0] if others else hit
        self._candidates_now = sorted(fits or [hit, *others])
        return None

    def _build_constants(self):
        """Path of each Ruby type and constant -> its declarations, and the
        paths that only prefix one: `A` of a `class A::B`."""
        self.constants, self.namespaces = {}, set()
        for node_id, node in self.nodes.items():
            if self.lang(node_id) == "ruby" and node["type"] in ("Class", "Constant"):
                written = node_id.partition("::")[2]
                self.constants.setdefault(written, []).append(node_id)
                while written := written.rpartition("::")[0]:
                    self.namespaces.add(written)

    def _constant(self, path, scope_id, seen=frozenset()):
        """The definition that the Ruby constant ``path``, a list of names
        written in ``scope_id``, refers to. Its first name is looked up in
        each enclosing class or module, innermost first, then in what the
        innermost one inherits from, then at the top level; the rest of
        the path names what that defines. ``"external"`` for a path the
        project does not define, None when it cannot be told. ``seen``
        holds the types whose bases are being looked up. A path whose first
        name is empty, `::Name`, starts at the top level."""
        namespaces = [] if path[:1] == [""] else self._ruby_nesting(scope_id)
        path = path[1:] if path[:1] == [""] else path
        if not path:
            return None
        innermost = next((n for n in namespaces if n in self.constants), None)
        inherited = [] if innermost in seen else [innermost]
        for namespace in inherited:
            for declaration in self.constants.get(namespace, ()):
                # A prepended module comes before the class in its ancestry.
                for base in [
                    *self.mixins.get((declaration, "prepend"), ()),
                    *self.bases.get(declaration, ()),
                ]:
                    named = self._constant(
                        base.split("::"), declaration, seen | {innermost}
                    )
                    if named and named != _EXTERNAL:
                        found_path = named.partition("::")[2]
                        if found_path not in inherited and found_path not in namespaces:
                            inherited.append(found_path)
        # The top level's constants are `Object`'s, which a class deriving
        # from `BasicObject` alone does not have.
        basic = any(
            _split_callee(base)[1] == "BasicObject"
            for d in self.constants.get(innermost, ())
            for base in self.bases.get(d, ())
        )
        for namespace in [*namespaces, *inherited[1:], *[""][: not basic]]:
            root = f"{namespace}::{path[0]}" if namespace else path[0]
            if root in self.constants or root in self.namespaces:
                break
        else:
            # A project constant of the name that lookup does not reach may
            # still be meant, unless the core library has the name.
            library = path[0] in self.standard["ruby"].data["types"]
            return (
                None
                if self.by_name.get(("ruby", path[0])) and not library
                else _EXTERNAL
            )
        found = self.constants.get("::".join([root, *path[1:]]))
        if found:
            # Of several declarations the one in the file is the nearest.
            own = [f for f in found if f.split("::")[0] == scope_id.split("::")[0]]
            found = (own or found)[0]
            # A constant assigned a constant is another name for it.
            alias = self.nodes[found]["metadata"].get("value")
            if alias and found not in seen:
                return self._constant(alias.split("::"), found, seen | {found})
            return found
        # A constant of a type's base is that type's too.
        open_ended = any(d in self.bases for d in self.constants.get(root, ()))
        return None if open_ended and path[1:] else _EXTERNAL

    def _unmade(self, node_id):
        """Whether ``node_id`` is a Ruby type a maker call makes, or an
        accessor one gives its type, `Struct.new(:a)` and its `a`, where the
        maker's name, looked up where the call is written, is a project
        constant rather than the core class."""
        node = self.nodes[node_id]
        maker = node["metadata"].get("maker")
        written = node_id.rpartition("::")[0]
        if node["type"] == "Method":
            written = written.rpartition("::")[0]
        return bool(maker) and self._constant(maker.split("::"), written) != _EXTERNAL

    def _ruby_nesting(self, scope_id):
        """The paths of the Ruby scopes around ``scope_id``, innermost first:
        a `class A::B` body is in `A::B` and in what encloses the statement,
        not in `A`."""
        nesting = []
        while "::" in scope_id:
            node = self.nodes.get(scope_id, {})
            if cref := node.get("metadata", {}).get("cref"):
                # A method's `Module.nesting`, as its statements write it
                return [*nesting, scope_id.partition("::")[2], *cref]
            if node.get("type") == "Class":
                nesting.append(scope_id.partition("::")[2])
            scope_id = (
                node.get("metadata", {}).get("nesting") or scope_id.rpartition("::")[0]
            )
        return nesting

    def type_at(self, path, scope_id):
        """The project type that the Rust path ``path``, written in
        ``scope_id``, leads to; ``"external"`` where it leads outside the
        project, None where it does not tell."""
        found = self._type_named(path, scope_id)
        if found in (None, _EXTERNAL):
            return found
        # A type the project only writes impl blocks for is declared outside.
        declared = self.nodes[found]["metadata"].get("kind") != "impl"
        return found if declared else _EXTERNAL

    def base_type(self, index):
        """The project type that the base the INHERITS edge ``index`` names
        stands for where its subtype is declared. None for a type outside
        the project, where several fit and nothing tells which, and where
        the name leads back to the subtype, which is no base of itself."""
        subtype = self.builder.edges[index]["source"]
        named = self._named_base(self.base_names[index], subtype)[0]
        return None if named in (_EXTERNAL, subtype) else named

    def _named_base(self, base, declaration):
        """What the base ``base`` of ``declaration`` names, as
        ``_type_named`` reads it, and the types in reach it could not tell
        between; a JavaScript class may extend a function given no
        prototype members, which it then names, whose static members the
        class has."""
        # The types several fit are no call's candidates.
        reached, self._candidates_now = self._candidates_now, None
        named = self._type_named(base, declaration)
        several = self._candidates_now or []
        self._candidates_now = reached
        held = named is None and self._written(base, declaration, set())
        return (held if held and self._js_function(held) else named), several

    def outside_base(self, index):
        """Whether the base the INHERITS edge ``index`` names is a type the
        standard library's data has where its subtype is declared."""
        subtype = self.builder.edges[index]["source"]
        written = self.base_names[index]
        lang = self.lang(subtype)
        if lang == "clang":
            # A C++ base is kept with its qualifier joined by dots.
            written = written.replace(".", "::")
        if lang == "dotnet":
            # C# names a generic type by its arity: `List<T>` is ``List`1``.
            written = self.builder.edges[index]["target"].removeprefix("REF:")
        if lang == "typescript":
            # A class of one of Node's modules, which the data names alone
            specifier = self._uses(subtype).get(written.partition(".")[0])
            if specifier and _node_module(specifier):
                return True
        if lang == "python":
            # Any name of the library: `typing.Generic` is no class there.
            return bool(self._py_std_key(written, subtype))
        return bool(self._standard_type(written, subtype))

    def type_in_scope(self, origin, scope_id):
        """The project type the origin path ``origin``, written in
        ``scope_id``, leads to; None where it leads to none."""
        verdict = self._resolve_origin(origin, scope_id)
        found = verdict[:1] if verdict else ()
        for hit in found:
            # A type the project only writes impl blocks for is declared
            # outside it.
            if (
                hit != scope_id
                and self._is_class(hit)
                and self.nodes[hit]["metadata"].get("kind") != "impl"
            ):
                return hit
        return None

    def _called(self, value, receiver=None):
        """Value returned by calling ``value``, a method of the type
        ``receiver`` where that is known."""
        kind, ref = value[:2]
        if kind == "cls" or (kind == "def" and self.nodes[ref]["type"] == "Class"):
            return ("self", ref, *value[2:])
        if kind == "def" and self._rust_variant(ref):
            # A Rust variant called builds a value of its enum.
            return "self", ref.rpartition("::")[0]
        if kind != "def":
            return None
        returns = self.nodes[ref]["metadata"].get("return_type") or ""
        body = None
        if ref in self.builder.signatures:
            # As written, of the signature the call being kept is of.
            chosen = self._bound[2] if ref == self._bound[0] else None
            chosen = chosen or self._overload(ref, {})
            returns, body = chosen["returns"], chosen.get("returned")
        if self.lang(ref) != "python":
            # A value declared under the name of an interface constructs it,
            # unless its return type says otherwise.
            companion = self.companions.get(ref)
            found = self._returned_type(returns, ref, receiver)
            if body:
                # The object a body builds is what is returned, whatever
                # type it is declared as.
                made = self._body_value(ref, receiver)
                if not returns:
                    found = made
                elif made and made[0] == "literal":
                    found = (*made, found)
            found = found or (companion and ("self", companion)) or self._alias(ref)
            return found or self._forwarded(ref)
        if self.nodes[ref]["metadata"].get("coroutine"):
            # A coroutine, which awaited gives what the function returns.
            returned = self._python_called(ref, returns, receiver)
            return "outside", "types.CoroutineType", (None, None, returned)
        return self._python_called(ref, returns, receiver)

    def _python_called(self, ref, returns, receiver):
        """What running the Python function ``ref``, annotated ``returns``,
        on ``receiver`` gives."""
        # The annotation is read where the function is defined.
        scope_id = ref.rpartition("::")[0]
        held = self.builder.fields.get(ref, {})
        for maker in _stdlib.section("python", "context_managers"):
            if maker in held:
                # The context manager the data says the decorator makes,
                # of what the function yields.
                library = self.standard["python"]
                made = library.parts(library.data["functions"][maker])[1][-1]
                return (
                    "outside",
                    library.parts(made)[0],
                    (self._variable(ref, maker),),
                )
        taken = self.nodes[ref]["metadata"].get("param_types") or [""]
        fluent = held.get("()") == ["self"]
        if self._is_class(scope_id) and (fluent or returns == taken[0] != "Any"):
            # What its `self` is, or is annotated as, is the type it is
            # called on.
            return ("self", receiver or scope_id)
        if "()" in held:
            return self._variable(ref, "()")
        return self._annotated(returns, scope_id, receiver)

    def _body_value(self, function_id, receiver):
        """Value that every `return` in the body of ``function_id``, which
        declares no return type, gives when called on ``receiver``: `this`,
        what a call or a construction returns, or a name of a declared
        type. None where one of them is not known or they differ."""
        returned = self.builder.signatures[function_id][-1]["returned"]
        key = function_id, receiver
        receiver = receiver[1] if isinstance(receiver, tuple) else receiver
        if key not in self.body_values:
            # A function that returns its own call has no value from that.
            self.body_values[key] = None
            values = []
            for callee, origin in (record or ("", "?") for record in returned):
                if origin == "self" and not callee:
                    owner = receiver or self._enclosing_class(function_id)
                    values.append(owner and ("self", owner))
                elif origin in self.builder.literals:
                    values.append(("literal", origin))
                elif not callee:
                    values.append(self._plain_value(origin, function_id))
                else:
                    scope = receiver if callee.startswith("this.") else None
                    scope = scope or function_id
                    values.append(self.resolve(callee, scope, origin, within=True))
            if (
                values
                and values[0] is not None
                and values.count(values[0]) == len(values)
            ):
                self.body_values[key] = values[0]
        return self.body_values.get(key)

    def _forwarded(self, function_id):
        """Value that ``function_id`` returns where its parser recorded,
        as the ``(callee, origin)`` of a call written in it, the one call
        whose result every way out of it hands back."""
        call = self.nodes[function_id]["metadata"].get("returned")
        if not call or function_id in self._forwarding:
            return None
        self._forwarding.add(function_id)
        with self._kept():
            self._result = self._receiver = self._arguments = self._looked_up = None
            self._passed = self._on = None
            self._caller = function_id
            target, how = self._verdict(*call[:1], function_id, call[1])
            if target:
                value = self._called(("def", target), self._receiver)
            elif how == _EXTERNAL and self._result != _OPAQUE:
                value = self._result or _EXTERNAL
            else:
                value = None
        self._forwarding.discard(function_id)
        return value

    def _alias(self, constant_id):
        """Value returned by calling a constant whose value is the name of
        a function or method: what that one returns."""
        target = self._named(constant_id)
        if target and self.nodes[target]["type"] != "Constant":
            return self._called(("def", target))
        return None

    def _named(self, constant_id):
        """What a constant whose value is a name names, through constants
        that are names of others (`const b = a; const a = helper`): the
        first definition that is no such constant; None for a constant of
        any other value, a name that reaches nothing, or a cycle."""
        seen, target = set(), constant_id
        while target not in seen:
            seen.add(target)
            node = self.nodes[target]
            path = str(node["metadata"].get("value") or "").strip().rstrip(";")
            named = re.fullmatch(r"[\w$]+(?:\.[\w$]+)*", path)
            if node["type"] != "Constant" or not named:
                return target if target != constant_id else None
            with self._kept():
                self._arguments, self._caller = None, target
                self._passed = self._on = None
                target = (self._resolve_origin(path, target) or (None,))[0]
            if not target:
                return None
        return None

    def _returned_type(self, returns, function_id, receiver=None):
        """Value of the declared return type ``returns`` of ``function_id``:
        an instance of a project type, a wrapper around one (`Result<T, E>`,
        `Promise<T>`), ``"external"``, or None. `this` and `Self` are
        ``receiver``, the type the method was called on, where that is known."""
        instance = receiver if isinstance(receiver, tuple) else None
        receiver = instance[1] if instance else receiver
        if self.lang(function_id) == "python":
            return self._annotated(returns, function_id, receiver)
        if function_id == self._bound[0] and returns in self._bound[1]:
            return self._bound[1][returns]
        if returns in self.builder.literals:
            # A literal sequence, not known where its elements are not
            held = self.builder.literals[returns]
            return None if held is None else ("literal", returns)
        queried = self.lang(function_id) == "typescript" and _TS_TYPEOF_RE.fullmatch(
            returns.strip()
        )
        if queried:
            # `typeof x` is the type of the value `x`: a value of it is `x`.
            root, *steps = re.split(r"\.|::", queried[1])
            held = self._written(root, function_id, set())
            for step in steps:
                held = held and self._find_member(held, step)[0]
            return ("def", held) if held else None
        compound = self.lang(function_id) == "typescript" and _ts_compound(returns)
        indexed = compound is None and _TS_INDEXED_RE.fullmatch(returns.strip())
        if indexed:
            # `Type["key"]`: the member of the type under that key
            held = self._type_named(indexed[1], function_id)
            member = (
                held not in (None, _EXTERNAL) and self._find_member(held, indexed[2])[0]
            )
            return ("def", member) if member else None
        if compound is None and returns[:1] == "{" and returns[-1:] == "}":
            # An object type: a value of it has the properties it writes.
            return ("shape", returns, function_id)
        name, parts, _ = _ts_type_parts(returns) if compound is None else ("", (), {})
        alias = name and name != "[]" and self._ts_alias(name, function_id)
        if alias and _ts_type_parts(alias[1])[0] == "[]":
            # An alias of a tuple type is that tuple.
            name, parts, _ = _ts_type_parts(_ts_given(alias, parts))
        if name == "[]" and returns not in self.builder.literals:
            # A tuple type holds the types it writes, by position.
            return (
                "outside",
                "[]",
                tuple(self._returned_type(p, function_id) for p in parts),
            )
        if compound:
            values = [
                self._returned_type(part.strip(), function_id, instance or receiver)
                for part in compound[1]
            ]
            if compound[0] == "|":
                # The value's type is one of the alternatives: members it
                # calls are looked up on each of them.
                return ("union", tuple(values))
            if compound[0] == "?":
                # A conditional type is the type its branches agree on.
                return values[0] if values.count(values[0]) == len(values) else None
            if all(value and value[0] == "shape" for value in values):
                # Object types together have the properties each writes.
                held = "; ".join(value[1][1:-1].strip() for value in values)
                return ("shape", f"{{ {held} }}", function_id)
            # An intersection is the operand derived from every other.
            known = [value for value in values if value is not None]
            for value in known:
                if all(
                    other == value
                    or "self" == value[0] == other[0]
                    and self._name(other[1]) in self._ancestors(value[1])
                    for other in known
                ):
                    return value
            return None
        parts = _bracketed_parts(returns) if "," in returns else []
        if self.lang(function_id) == "dotnet" and len(parts) > 1:
            # A C# method that hands values to its `out` arguments returns
            # them beside its result, which a `#N.@P` origin picks up.
            return (
                "outside",
                "()",
                tuple(
                    self._returned_type(part, function_id, instance or receiver)
                    for part in parts
                ),
            )
        if self.lang(function_id) == "clang" and returns.rstrip().endswith("*"):
            # A raw pointer
            held = self._returned_type(
                returns.rstrip()[:-1], function_id, instance or receiver
            )
            return held and ("pointer", held)
        owned = self.lang(function_id) == "rust" and _RUST_BORROW_RE.sub("", returns)
        if owned and owned.startswith("Self::"):
            # An associated type of the impl block the method is in, behind
            # a reference too: `&Self::Output`
            node = self.nodes.get(function_id, {})
            key = (
                function_id.rpartition("::")[0],
                node.get("metadata", {}).get("trait_impl"),
                owned[6:],
            )
            if key in self.builder.rust_associated:
                held = self.builder.rust_associated[key]
                return self._returned_type(held, function_id, instance or receiver)
        aliased = self.lang(function_id) == "rust" and self._rust_alias(
            returns, function_id
        )
        if aliased:
            return self._alias_value(
                aliased, returns, function_id, instance or receiver
            )
        library = self.standard.get(self.lang(function_id))
        member = self.lang(function_id) == "clang" and re.fullmatch(
            r"(.*>)\s*::\s*(\w+(?:\s*::\s*\w+)*)[\s&]*", returns.strip()
        )
        if member:
            # `X<A>::m` is the member type `m` of the instance `X<A>`: the
            # one the library's data writes, a project template's own, else
            # not known.
            held = self._returned_type(member[1], function_id, instance or receiver)
            kind = isinstance(held, tuple) and held[0]
            if kind == "outside" and len(held) > 2 and member[2].isidentifier():
                return library.aliased(held, member[2])
            if kind == "self" and self._c_specialized(member[1], function_id) is False:
                written = f"{_strip_generics(member[1])}::{member[2]}"
                return self._returned_type(written, function_id, instance or receiver)
            return None
        name = library and library.named(returns, function_id)
        if name:
            # A standard type holds what its type arguments are.
            arguments = library.parts(returns)[1]
            held = (
                self._returned_type(a, function_id, instance or receiver)
                for a in arguments
            )
            return ("outside", name, tuple(held))
        if self.lang(function_id) == "rust":
            returns = _RUST_BORROW_RE.sub("", returns)
        if self.lang(function_id) == "go":
            returns = _go_generics(returns)
        match = re.match(
            r"[&*\s]*(?:'\w+\s+)?(?:mut\s+)?([\w$.:]+)\s*"
            r"(<\s*(?:'\w+\s*,\s*)*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?([\w$.:]+)?)?",
            returns,
        )
        if not match:
            return None
        owner = receiver or self._enclosing_class(function_id)
        held = self._type_argument(match[1], function_id, instance or receiver)
        if held is not False:
            return held
        if self.lang(function_id) == "clang" and match[2]:
            held = self._c_specialized(returns[match.start(1) :], function_id)
            if held is not False:
                return held
        standard = None
        for kind, name in zip(("self", "wrapped"), match.group(1, 3)):
            if name in ("Self", "this"):
                if kind == "self" and instance and not match[2]:
                    # The receiver itself, with its type arguments
                    return instance
                found = owner
            else:
                plain = kind == "self" and not match[2]
                found = name and self._type_named(name, function_id, plain)
                if isinstance(found, tuple):
                    # An alias is the type it names, written as it writes it.
                    return found
            if found == _EXTERNAL and kind == "self" and match[2]:
                # A type from outside written with a type argument: its
                # methods are its own, unless it wraps a project type.
                standard = ("outside", name)
                continue
            lang = self.lang(function_id)
            if (
                not found
                and kind == "self"
                and self._types(lang, re.split(r"::|\.", name)[-1])
            ):
                # A project type that cannot be told apart wraps nothing.
                return None
            if found == _EXTERNAL:
                bare = re.split(r"::|\.", name)[-1]
                rust = self.lang(function_id) == "rust"
                if rust and kind == "self" and self.impl_types.get(bare):
                    # A type from outside the project implements traits for:
                    # a value of it has their methods.
                    return "outside", bare, ()
                return _EXTERNAL if kind == "self" else standard
            generic = found and kind == "self" and match[2]
            if generic and self.nodes[found]["metadata"].get("generics"):
                # A generic type written with its arguments holds them; one
                # no import places outside, as a function's own type
                # parameter, is not known.
                written = _bracketed_parts(returns[returns.index("<") + 1 :])
                fields = self.builder.literals.get(written[0]) if written else None
                if isinstance(fields, dict):
                    # A struct literal holds each parameter as the field
                    # declared of it holds.
                    return (
                        kind,
                        found,
                        self._field_arguments(found, fields, function_id),
                    )
                held = []
                for a in written:
                    if a.startswith("'"):
                        # A lifetime takes no type parameter's place.
                        continue
                    value = self._returned_type(a, function_id, instance or receiver)
                    imported = self._imported(a.partition("<")[0].strip(), function_id)
                    outside = value != _EXTERNAL or imported == _EXTERNAL
                    held.append(value if outside else None)
                return kind, found, tuple(held)
            if found and kind == "wrapped":
                # A standard type holding a project type is what the data
                # says it is; one the data lacks has its own members.
                key = self._standard_type(match[1], function_id)
                if key:
                    return "outside", key, (("self", found),)
                return kind, found, re.split(r"::|\.", match[1])[-1]
            if found:
                return (kind, found)
        return standard

    def _rust_alias(self, written, scope_id):
        """``(module, name, declarations)`` of the Rust type alias the type
        ``written`` in ``scope_id`` names: one its module declares, or the
        one a `use` there or the path binds; None for none."""
        *qualifier, name = _rust_type_parts(_RUST_BORROW_RE.sub("", written))[0].split(
            "::"
        )
        leaf = self._uses(scope_id).get(name, name).rpartition("::")[2]
        if name not in self.alias_names and leaf not in self.alias_names:
            return None
        module = self._rust_module(scope_id)
        if qualifier and self._rust_child(module, qualifier[0]):
            qualifier = ["self", *qualifier]
        if qualifier:
            found = self._rust_rooted("::".join([*qualifier, name]), scope_id) and (
                self._rust_path(qualifier, scope_id)
            )
            module = found[0] if isinstance(found, list) and len(found) == 1 else None
        return module and self._alias_in(module, name)

    def _alias_in(self, module, name, seen=frozenset()):
        """``(module, name, declarations)`` of the Rust type alias ``name``
        names in ``module``: one it declares, else the one a `use` of it
        binds, by the name or by a glob."""
        found = self.builder.rust_alias_types.get((module, name))
        if found:
            return module, name, _rust_declared(found)
        uses = self._uses(module)
        written = [uses[name]] if name in uses else []
        written = written or [
            f"{path[:-1]}{name}" for path in uses.values() if path[-3:] == "::*"
        ]
        for target in written:
            *qualifier, leaf = target.split("::")
            if qualifier and self._rust_child(module, qualifier[0]):
                qualifier = ["self", *qualifier]
            path = "::".join([*qualifier, leaf])
            if not qualifier or (module, path) in seen:
                continue
            if not self._rust_rooted(path, module):
                continue
            held = self._rust_path(qualifier, module)
            for scope in held if isinstance(held, list) else ():
                hit = self._alias_in(scope, leaf, seen | {(module, path)})
                if hit:
                    return hit
        return None

    def _alias_value(self, alias, written, scope_id, receiver):
        """Value of the type ``written`` in ``scope_id`` that names the Rust
        type alias ``alias``: of the type it stands for, read in its module
        with its parameters the arguments ``written`` gives them; None where
        its declarations differ or its module declares a type of its name."""
        module, name, declared = alias
        # A type the module also declares of the name is the alias's twin
        # for other builds (`#[cfg]`): a value of the name is either.
        twin = any(
            t.rpartition("::")[0] == module
            and self.nodes[t]["metadata"].get("kind") != "impl"
            for t in self._types("rust", name)
        )
        if twin or len(declared) > 1 or (module, name) in self.open_aliases:
            return None
        ((params, held),) = declared
        arguments = _rust_type_parts(_RUST_BORROW_RE.sub("", written))[1]
        values = {
            param: self._returned_type(argument, scope_id, receiver)
            for param, argument in zip(params, arguments)
        }
        # An impl block whose name is the alias where it is written, not a
        # type of the name elsewhere
        owners = [
            t
            for t in self._types("rust", name)
            if self.nodes[t]["metadata"].get("kind") == "impl"
            and (self._rust_alias(name, t) or ())[:2] == (module, name)
        ]
        if owners:
            # Impl blocks written for the alias: an instance of their type,
            # which has the aliased type's members after its own
            return "self", owners[0], tuple(values.values())
        bound, self._bound = self._bound, (module, values, None)
        self.open_aliases.add((module, name))
        try:
            return self._returned_type(_RUST_BORROW_RE.sub("", held), module)
        finally:
            self._bound = bound
            self.open_aliases.discard((module, name))

    def _type_argument(self, name, scope_id, receiver):
        """What the type parameter ``name`` of a type around ``scope_id``
        stands for: the argument ``receiver``, a value of that type or of
        one derived from it or the ID of such a type, gives it, else its
        bound, the receiver's own type for one bound by a type of itself
        (`B extends Builder<B>`), None for none; False where no type around
        declares the parameter."""
        instance = receiver if isinstance(receiver, tuple) else None
        instance = instance or (receiver and ("self", receiver))
        # A Go method's receiver names the type's parameters by position.
        renamed = self.nodes.get(scope_id, {}).get("metadata", {})
        renamed = renamed.get("receiver_generics") or ()
        for scope in self._scopes(scope_id):
            metadata = self.nodes.get(scope, {}).get("metadata", {})
            generics = list(metadata.get("generics") or ())
            if name in renamed and len(renamed) == len(generics):
                name = generics[renamed.index(name)]
            if name in generics and self._is_class(scope):
                given = self._instance_arguments(scope, instance)
                at = generics.index(name)
                held = self.nodes.get(scope_id, {}).get("metadata", {})
                fixed = held.get("impl_args") or ()
                if at < len(fixed) and fixed[at]:
                    # A Rust block for one instantiation, `impl W<Own>`
                    return self._returned_type(fixed[at], scope_id)
                if at < len(given):
                    # An instance's argument, known or not: its bound only
                    # limits what it may be.
                    return given[at]
                bound = (metadata.get("type_bounds") or {}).get(name)
                where = scope
                if not bound:
                    # A Rust `impl` block bounds them for its own items.
                    bound, where = held.get("impl_bounds", {}).get(name), scope_id
                if bound and re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", bound):
                    head = bound.partition("<")[0].strip()
                    reached = self._candidates_now
                    named = self._type_named(head, where)
                    self._candidates_now = reached
                    if named in self._declarations(scope):
                        # A bound by its own type: the receiver's own type.
                        return instance and instance[:2]
                    # One whose arguments name the parameter: that type.
                    bound = head
                if not bound or (scope, name) in self._bounding:
                    # A bound through the parameter itself bounds nothing.
                    return None
                self._bounding.add((scope, name))
                try:
                    return self._returned_type(bound, where)
                finally:
                    self._bounding.discard((scope, name))
        return False

    def _instance_arguments(self, type_id, instance):
        """The type arguments ``instance`` gives the generic type
        ``type_id``: its own where it is one, those its type writes a base
        of that type with where it derives from it."""
        if not instance:
            return ()
        own, given = instance[1], instance[2] if len(instance) > 2 else ()
        if type_id in self._declarations(own):
            return given
        reached = self._candidates_now
        for base in self.written_bases.get(own, ()):
            name, opened, written = base.partition("<")
            if opened and self._type_named(name, own) == type_id:
                self._candidates_now = reached
                held = ("self", own, given)
                return tuple(
                    self._returned_type(a, own, held) for a in _bracketed_parts(written)
                )
        self._candidates_now = reached
        return ()

    def _has_members(self, owner):
        """Whether ``owner`` is what a receiver can be: a class, or a
        constant or a variable that holds an object of members."""
        return self._is_class(owner) or (
            owner in self.members
            and self.nodes.get(owner, {}).get("type") in ("Constant", "Variable")
        )

    def _aliased(self, held):
        """``held``, or where it is a constant whose value is the name of
        another, the constant the parser read that name as, in turn: the
        same value."""
        seen = set()
        while held in self.builder.js_aliases and held not in seen:
            seen.add(held)
            held = self.builder.js_aliases[held]
        return held

    def _rust_local(self, name, source_id):
        """The item named ``name`` that a function around ``source_id``
        declares, which hides what a `use` outside it binds the name to;
        None where a `use` in a function nearer binds it, or none
        declares it."""
        for scope in self._scopes(source_id):
            if not self._is_function_scope(scope):
                return None
            if hit := self.members.get(scope, {}).get(name):
                return hit
            if name in self.builder.imports.get(scope, {}):
                return None
        return None

    def _scopes(self, node_id):
        """Yield ``node_id`` and each scope enclosing it, out to its file."""
        while node_id:
            yield node_id
            node_id = node_id.rpartition("::")[0]

    def _is_class(self, node_id):
        node = self.nodes.get(node_id)
        # A Ruby module has members as a class has; a Rust one is a scope.
        # A callback's owner is a scope source cannot name, not a type: its
        # members are read where they are written, not through a receiver.
        return bool(
            node
            and node["type"] == "Class"
            and (
                node["metadata"].get("kind") not in ("module", "namespace", "callback")
                or self.lang(node_id) == "ruby"
            )
        )

    def _is_top_level(self, node_id):
        parent = self.nodes.get(node_id.rpartition("::")[0])
        return (
            parent is None
            or parent["type"] == "File"
            or parent["metadata"].get("kind") == "namespace"
        )

    def _named_from(self, node_ids, scope_id):
        """Those of the TypeScript definitions ``node_ids`` that are named
        where ``scope_id`` is written. An ECMA-262 declaration is
        block-scoped: one written inside a function or a callback is named
        only inside it. A function's static member is a property of it,
        named where the function is."""
        around = set(self._scopes(scope_id))
        return [
            node_id
            for node_id in node_ids
            if all(
                scope in around
                for scope in self._scopes(
                    self._member_owner(node_id).rpartition("::")[0]
                    if self.nodes[node_id]["metadata"].get("static")
                    else node_id.rpartition("::")[0]
                )
                if self._is_function_scope(scope)
            )
        ]

    def _named_in(self, node_id, file_id):
        """Whether the TypeScript definition ``node_id`` is named in the file
        ``file_id`` without an import: one of a module only in the module."""
        defining = node_id.split("::")[0]
        return defining == file_id or defining not in self.builder.js_modules

    def _is_function_scope(self, node_id):
        """Whether ``node_id`` is a scope names written inside it stay in:
        a function, a method or a callback."""
        node = self.nodes.get(node_id)
        return bool(
            node
            and (
                node["type"] in ("Function", "Method")
                or node["metadata"].get("kind") == "callback"
            )
        )

    def _static_owner(self, node_id):
        """The JavaScript class or function whose static member ``node_id``
        is code of, which `this` is there; None where it is code of none."""
        for scope in (
            self._scopes(node_id) if self.lang(node_id) == "typescript" else ()
        ):
            if self.nodes.get(scope, {}).get("metadata", {}).get("static"):
                return self._member_owner(scope)
            if self._is_class(scope):
                return None
        return None

    def _enclosing_class(self, node_id):
        if f"{node_id}.prototype" in self.nodes:
            # In a JavaScript function given a prototype, `this` is the
            # instance it constructs.
            return f"{node_id}.prototype"
        arrow = False
        for scope in self._scopes(node_id):
            metadata = self.nodes.get(scope, {}).get("metadata", {})
            # A non-arrow function's `this` is its own, which nothing
            # static can name.
            if (
                self.lang(scope) == "typescript"
                and self.nodes.get(scope, {}).get("type") == "Function"
                and not metadata.get("arrow")
            ):
                return None
            if self._is_class(scope) and not (
                arrow and metadata.get("kind") == "object"
            ):
                return scope
            # An arrow function's `this` is the one around it, which an
            # object literal it is a property of does not bind.
            arrow = bool(metadata.get("arrow"))
            # The body of a C/C++ macro definition is a member's of the type
            # that an argument of the macro names.
            words = self.nodes.get(scope, {}).get("metadata", {}).get("arguments", ())
            for word in words:
                named = self._c_type([], word, scope)
                if named and named != _EXTERNAL:
                    return named
        return None

    def _inherited(self, name, scope_id):
        """What the innermost scope enclosing ``scope_id`` that has a
        ``name`` has under it: a function local to a function, or a member
        of a type, its own or one of a base."""
        for scope in self._scopes(scope_id):
            if self._is_class(scope):
                hit = self._find_member(scope, name)[0]
            else:
                hit = self.members.get(scope, {}).get(name)
            if hit and (self._is_class(scope) or not self._is_class(hit)):
                return hit
        return None

    def takes(self, target, passed, callee="", origin=None):
        """Whether the method ``target`` can take what a call of it written
        ``callee``, of the origin path ``origin``, passes, where its parser
        counted what it takes (C#, Java, C/C++) and the call's: ``passed``
        arguments, and the value a C# extension method is called on, named
        or a literal's."""
        counts = self._entries(target)
        if "extends" in self.nodes[target]["metadata"] and isinstance(passed, int):
            owner = self._name(self._member_owner(target))
            receiver = _split_callee(callee)[0][-1:]
            passed += (
                receiver not in ([], [owner]) or not receiver and "." in (origin or "")
            )
        return counts is None or any(_admits(entry, passed) for entry in counts)

    def _entries(self, target):
        """``[least, most, types]`` of each overload of the C#, Java or C/C++
        method ``target``; for C/C++ the least that the defaults of all its
        declarations of the same parameter types allow, by position, and
        an overload its class declares that the project does not define."""
        metadata = self.nodes[target]["metadata"]
        counts = metadata.get("parameters")
        if counts is None or self.lang(target) != "clang":
            return counts
        owner = self._member_owner(target)
        owners = self._declarations(owner) if self._is_class(owner) else ()
        defined = [
            (types, node)
            for node in (self._overloads_in(owner, metadata["name"]) if owners else ())
            for _, _, types in self.nodes[node]["metadata"].get("parameters") or ()
            if types is not None
        ]
        overloads = [
            [least, most, types, target, set(range(least, len(types or ())))]
            for least, most, types in counts
        ]
        for positions, most, written, scope in self.builder.prototypes.get(
            target.partition("::")[2], ()
        ):
            same = [
                overload
                for overload in overloads
                if overload[2] is not None
                and self._c_same_types(written, scope, overload[2], overload[3])
            ]
            if same:
                same[0][4].update(positions)
            elif scope in owners and not any(
                self._c_same_types(written, scope, *one) for one in defined
            ):
                overloads.append([len(written), most, written, scope, set(positions)])
        entries = []
        for least, most, types, _, defaulted in overloads:
            while least - 1 in defaulted:
                least -= 1
            entries.append([least, most, types])
        return entries

    def _c_same_types(self, one, one_scope, other, other_scope):
        """Whether the C/C++ parameter types ``one``, written in ``one_scope``,
        and ``other``, written in ``other_scope``, are the same: alike once
        each name in them is taken for the project type it names there, and
        one that names none is read without the namespaces macros open."""

        def canonical(types, scope_id):
            def named(found):
                *qualifier, name = found[0].split("::")
                held = self._c_type([q for q in qualifier if q], name, scope_id)
                if held and held != _EXTERNAL:
                    return held
                return "::".join([*self._c_written(qualifier, (), scope_id), name])

            return [_C_TYPE_NAME_RE.sub(named, t) for t in types]

        return one == other or canonical(one, one_scope) == canonical(
            other, other_scope
        )

    def _overload_by_arity(self, hit, static=None):
        """The method that the call being resolved, which reached the C#,
        Java or C++ method ``hit``, is a call of: of the overloads of its
        name in the types `_lookup_types` gives, less those a type before
        overrides or hides and those the call cannot access, the one that
        takes as many arguments as the call passes and, with ``static``
        given, is static or not as that says. None, with them as the
        candidates, where several do. ``hit`` itself where it is no method.
        A Java or C++ node holds all the overloads of its name in its type."""
        if "parameters" not in self.nodes[hit]["metadata"]:
            return hit
        name = self.nodes[hit]["metadata"]["name"]
        overridden, fits, signatures, owners = set(), [], {}, {}
        # each type looked up, in order, with the signatures it writes; the
        # signature each fit was taken at
        order, raw = [], {}
        for owner, binding in self._lookup_types(hit, name):
            written = set()
            order.append((owner, written))
            for overload in self._overloads_in(owner, name):
                metadata = self.nodes[overload]["metadata"]
                for entry in self._entries(overload) or ():
                    types = entry[2]
                    signature = types is not None and tuple(
                        _substituted(t, binding) for t in types
                    )
                    written.add(signature)
                    if (
                        overload not in fits
                        and signature not in overridden
                        # A type before this one in the lookup overrides or
                        # hides the slot: its method replaces this one.
                        and self._accessible(overload)
                        and _admits(entry, self._arguments)
                        and self._lambdas_convert(types, overload)
                        and static in (None, metadata.get("static"))
                        and (
                            self.lang(overload) != "dotnet"
                            or self._typed_as_called(metadata, types)
                        )
                    ):
                        fits.append(overload)
                        signatures[overload] = types is not None and [
                            _moded(t, binding) for t in types
                        ]
                        owners[overload] = owner
                        raw[overload] = signature
            overridden |= written - {False}
        if len(fits) > 1 and self.lang(hit) == "dotnet":
            fits = self._most_derived_applicable(fits, signatures, owners, order, raw)
        if len(fits) > 1:
            self._candidates_now = sorted(fits)
        return fits[0] if len(fits) == 1 else None

    def _most_derived_applicable(self, fits, signatures, owners, order, raw):
        """Of the C# overloads ``fits``, those overload resolution keeps:
        the methods of the most derived type that declares one that
        applies, an override declared in the base its slot comes from. Of
        them, the one each argument's own type matches; one taking each
        argument as `object` applies whatever is passed. ``fits`` where it
        is not told whether the most derived type's methods apply."""
        types = [owner for owner, _ in order]

        def loose(signature):
            return signature and tuple(t.rstrip("?") for t in signature)

        def declared(fit):
            own = types.index(owners[fit])
            if not self.nodes[fit]["metadata"].get("override"):
                return own
            # Its slot is a base's, one outside the project where no base
            # here writes it; a `?` on a reference type is no other slot.
            return max(
                (
                    at
                    for at, (_, written) in enumerate(order)
                    if at > own and loose(raw[fit]) in {loose(w) for w in written}
                ),
                default=own + 1,
            )

        at = {fit: declared(fit) for fit in fits}
        for level in sorted(set(at.values())):
            held = [fit for fit in fits if at[fit] == level]
            applies = {fit: self._applies(fit, signatures[fit]) for fit in held}
            if all(verdict is False for verdict in applies.values()):
                # None of this type's applies: its bases' are looked at.
                continue
            maybe = [fit for fit in held if applies[fit] is not False]
            exact = self._exactly_typed(maybe, signatures) or self._taking_null(
                maybe, signatures
            )
            if exact:
                return exact
            return maybe if True in applies.values() else fits
        return fits

    def _applies(self, overload, signature):
        """Whether the C# overload ``overload`` of the parameters
        ``signature`` takes the arguments the call being resolved passes,
        each by value and by position: True, False, or None where an
        argument's type, or a conversion, is not told."""
        passed = self._passed
        if not signature or passed is None or len(passed) != len(signature):
            return None
        found = True
        for (mode, written), argument in zip(signature, passed):
            if isinstance(argument, str) and re.match(
                r"[\w@]+: |(?:ref|out|in) ", argument
            ):
                return None
            if mode:
                return None
            if isinstance(argument, str) and self._nullable_value(
                argument.removesuffix("()"), self._caller
            ):
                # A nullable value converts to its own nullable type only,
                # which the bare type it reads as does not tell.
                verdict = None
            else:
                verdict = self._converts(
                    self._dotnet_argument(argument), written, overload
                )
            if verdict is False:
                return False
            found = found and verdict
        return found

    def _converts(self, value, written, overload):
        """Whether C# converts an argument of ``value`` implicitly to the
        parameter type ``written`` of ``overload``, by the language's rules:
        identity, to `object` or a nullable of its type, a numeric widening,
        to a base or an interface of its type, or a user-defined implicit
        conversion the library declares. None where a type is not known, is
        generic, or is the project's and not related by inheritance."""
        if not isinstance(value, tuple):
            return None
        bare = written.rstrip("?")
        if bare in ("object", "Object", "System.Object"):
            return True
        if bare != written and self._nullable_value(written, overload):
            # `T?` of a value type takes a `T` by wrapping it; any other
            # conversion to it is not read here.
            held = self._returned_type(bare, overload)
            return True if held == value else None
        held = self._returned_type(bare, overload)
        if not isinstance(held, tuple):
            return None
        if held[:2] == value[:2]:
            return True if held[2:] == value[2:] else None
        if any(part[2:] not in ((), ((),)) for part in (held, value)):
            return None
        if value[0] == "self" and held[0] == "self":
            return True if self._name(held[1]) in self._ancestors(value[1]) else None
        if value[0] != "outside" or held[0] != "outside":
            return None
        source, target = value[1], held[1]
        data = self.standard["dotnet"].data
        return (
            target in _DOTNET_WIDENING.get(source, ())
            or source in data["implicit"].get(target, ())
            or target in data["types"].get(source, {}).get("bases", ())
            or any(
                target in data["types"].get(base, {}).get("bases", ())
                for base in data["types"].get(source, {}).get("bases", ())
            )
        )

    def _exactly_typed(self, fits, signatures):
        """The one C# overload of ``fits`` that takes each argument as of its
        own type, as a list; empty where the call's arguments are not all of
        a known type or not exactly one overload does. An identity
        conversion is better than any other, and of overloads that take the
        same types a non-generic one is better than a generic one, so that
        overload is the one C# calls."""
        passed = []
        for written in self._passed or ():
            name = mode = ""
            if isinstance(written, str):
                if named := re.match(r"([\w@]+): ", written):
                    name, written = named[1], written[named.end() :]
                if written.startswith(("ref ", "out ", "in ")):
                    mode, written = written.split(" ", 1)
                if self._nullable_value(written.removesuffix("()"), self._caller):
                    return []
            value = self._dotnet_argument(written)
            if not value:
                return []
            passed.append((name, mode, value))
        found = {f: self._identity(f, signatures[f], passed) for f in fits}
        exact = [f for f, generic in found.items() if generic is not None]
        if len(exact) > 1:
            exact = [f for f in exact if not found[f]]
        return exact if passed and len(exact) == 1 else []

    def _identity(self, overload, signature, passed):
        """Whether the C# overload ``overload``, of the parameters
        ``signature``, takes each of the arguments ``passed`` as of its own
        type, by position or by name: None where it does not, else whether
        one of its own type parameters stands for an argument's type."""
        metadata = self.nodes[overload]["metadata"]
        if not signature or len(signature) != len(passed):
            return None
        names = metadata.get("parameter_names") or ()
        placed = [None] * len(signature)
        for at, (name, mode, value) in enumerate(passed):
            if name:
                if name not in names:
                    return None
                at = names.index(name)
            if placed[at]:
                return None
            placed[at] = mode, value
        free = (
            () if metadata.get("constrained") else metadata.get("type_parameters", ())
        )
        inferred = {}
        for (mode, written), (given_mode, value) in zip(signature, placed):
            if mode != given_mode or self._nullable_value(written, overload):
                return None
            if not self._matches(written.rstrip("?"), value, overload, free, inferred):
                return None
        return bool(inferred)

    def _matches(self, written, value, overload, free, inferred):
        """Whether the C# type ``written`` in ``overload`` is that of
        ``value``, type arguments and all, each of its type parameters
        ``free`` standing for one type throughout, as ``inferred`` binds
        them."""
        if written in free:
            return (
                self._bindable(value) and inferred.setdefault(written, value) == value
            )
        held = self._returned_type(written, overload)
        if not isinstance(held, tuple) or held[:2] != value[:2]:
            return False
        _, generic, rest = written.partition("<")
        if not generic:
            return held[2:] == value[2:]
        parts = [p.strip() for p in _bracketed_parts(rest)]
        arguments = value[2] if len(value) > 2 else ()
        return len(parts) == len(arguments) and all(
            self._matches(p.rstrip("?"), a, overload, free, inferred)
            for p, a in zip(parts, arguments)
        )

    def _bindable(self, value):
        """Whether ``value`` is of a known C# type a type argument can be:
        the standard library's or the project's, but no `ref struct`."""
        if value[0] == "outside":
            data = self.standard["dotnet"].data
            return value[1] in data["types"] and value[1] not in data["ref_structs"]
        node = self.nodes.get(value[1])
        return bool(node) and not node["metadata"].get("ref")

    def _nullable_value(self, written, scope_id):
        """Whether the C# type ``written`` is a nullable value type: `T?` of
        a type not known to be a reference type, or `Nullable<T>`."""
        if written.endswith("?"):
            return self._kind_of(written[:-1], scope_id) != "reference"
        held = self._returned_type(written, scope_id)
        return isinstance(held, tuple) and held[1] == "System.Nullable`1"

    def _kind_of(self, written, scope_id):
        """Whether the C# type ``written`` is a "value" or a "reference"
        type, None where that is not known."""
        if written in ("string", "object"):
            return "reference"
        held = self._returned_type(written, scope_id)
        if not isinstance(held, tuple):
            return None
        if held[0] == "outside":
            # A library type the data has is a struct or enum by its bases,
            # else a class, interface or delegate.
            types = self.standard["dotnet"].data["types"]
            if held[1] not in types:
                return None
            bases = set(types[held[1]].get("bases", ()))
            return (
                "value" if {"System.ValueType", "System.Enum"} & bases else "reference"
            )
        kind = self.nodes.get(held[1], {}).get("metadata", {}).get("kind")
        if held[0] not in ("self", "def") or not kind:
            return None
        return "value" if kind in ("struct", "enum") else "reference"

    def _taking_null(self, fits, signatures):
        """The one C# overload of ``fits`` that may take a `null` where the
        call being resolved passes one, as a list: the others take a value
        type there. Empty where none or several may, or no `null` is passed."""
        nulls = [i for i, a in enumerate(self._passed or ()) if a == "null"]
        taking = [
            f
            for f in fits
            if nulls
            and not any(
                signatures[f]
                and i < len(signatures[f])
                and self._kind_of(signatures[f][i][1], f) == "value"
                and not self._nullable_value(signatures[f][i][1], f)
                for i in nulls
            )
        ]
        return taking if len(taking) == 1 else []

    def _dotnet_argument(self, origin):
        """The value of a C# argument of the call being resolved, given or
        by the origin the parser read it as; None where that is not known."""
        value = origin
        if isinstance(origin, str):
            value = self._value(origin, self._caller, set())
        return value if isinstance(value, tuple) and value[0] != "literal" else None

    def _overloads_in(self, owner_id, name):
        """The methods ``name`` of the type ``owner_id``, in each of its
        declarations, a C# overload as `<name>#<N>`."""
        members = [
            self.members.get(d, {}).get(name) for d in self._declarations(owner_id)
        ]
        for overload in self._defined([m for m in members if m]):
            declaration, count = self._member_owner(overload), 1
            while overload in self.nodes:
                yield overload
                count += 1
                overload = f"{declaration}::{name}#{count}"

    def _defined(self, members):
        """Of ``members``, one type's of a name, those a call can reach: a
        pure virtual function that is defined as well is called as its
        definition."""
        defined = [m for m in members if not self.nodes[m]["metadata"].get("pure")]
        return defined or members

    def _accessible(self, member):
        """Whether the call being resolved can reach the method ``member``:
        a private one only from the body of its type, in Java that of the
        top-level type around it; a protected one from the body of its type,
        or of a derived type when looked up from that type or one derived
        from it; an explicit C# interface implementation never by its name."""
        metadata = self.nodes[member]["metadata"]
        if metadata.get("explicit"):
            return False
        owner = self._member_owner(member)
        if metadata.get("protected"):
            start = self._looked_up and self._looked_up[0]
            return any(
                owner in self._declarations(scope)
                or self._name(owner) in self._ancestors(scope)
                and (
                    not start
                    or start in self._declarations(scope)
                    or self._name(scope) in self._ancestors(start)
                )
                for scope in self._scopes(self._caller)
                if self._is_class(scope)
            )
        if not metadata.get("private"):
            return True
        if self.lang(member) == "java":
            bodies = ["::".join(owner.split("::")[:2])]
        else:
            bodies = self._declarations(owner)
        return not set(bodies).isdisjoint(self._scopes(self._caller))

    def _lookup_types(self, hit, name):
        """``(type, {type parameter: its argument})`` of each type whose
        members ``name`` the lookup that reached the method ``hit`` finds:
        the one `_find_member` last started where that reaches ``hit``, else
        one from the type that defines ``hit``."""
        owner = self._member_owner(hit)
        if self._looked_up and self._looked_up[1] == name:
            found = self._lookup_order(self._looked_up[0], name)
            if any(owner in self._declarations(t) for t, _ in found):
                return found
        return self._lookup_order(owner, name)

    def _c_declares(self, declarations, name):
        """Whether the C++ class of ``declarations`` declares ``name``."""
        return any(
            name in self.members.get(d, {})
            or name in self.builder.fields.get(d, ())
            or name in self.builder.c_declared.get(d, ())
            for d in declarations
        )

    def _c_using(self, declarations, name):
        """The base a `using` declaration of ``name`` in the C++ class of
        ``declarations`` names: `using B::f` is `B`'s `f`, none of the other
        bases'."""
        return next(
            (
                re.sub(r"<.*", "", re.sub(r"\s*::\s*", "::", used))
                for d in declarations
                if (used := self.builder.c_using.get(d, {}).get(name))
            ),
            None,
        )

    @staticmethod
    def _c_used(used, base):
        """Whether ``used``, the base a `using` declaration names, names
        ``base``, the base as written."""
        parts, bare = _split_callee(base)
        path = "::".join([*parts, bare])
        return used == bare or path == used or path.endswith(f"::{used}")

    def _c_hides(self, declarations, name):
        """Whether the C++ class of ``declarations`` declares ``name``, which
        hides its bases' members of the name unless `using` brings them in."""
        return self._c_declares(declarations, name) and not any(
            name in self.builder.c_using.get(d, ()) for d in declarations
        )

    def _c_ambiguous(self, declarations, name):
        """Whether lookup of ``name`` in the C++ class of ``declarations``,
        which does not declare it, finds it through more than one of its
        bases, other than as one member of a base they share; the members
        found are then the candidates."""
        used = self._c_using(declarations, name)
        found = [
            self._find_member(t, name, seen=set())[0]
            for d in declarations
            for base in self.bases.get(d, ())
            if not used or self._c_used(used, base)
            for t in self._base_types(base, d) or ()
            if any(
                self._c_declares(self._declarations(x), name)
                for x, _ in self._lookup_order(t, name)
            )
        ]
        if len(found) < 2 or len(set(found)) == 1 and None not in found:
            return False
        self._candidates_now = sorted(set(found) - {None}) or None
        return True

    def _lookup_order(self, start, name):
        """``_lookup_types`` from the type ``start``: each type before its
        bases, and a class's base class before its interfaces, with the
        arguments of its type parameters as ``start`` writes them. C++
        looks no further than a type that declares ``name``, unless it
        brings in its bases' with `using`."""
        bindings, order = {start: {}}, []

        def visit(type_id):
            declarations = self._declarations(type_id)
            hides = self.lang(type_id) == "clang" and self._c_hides(declarations, name)
            bases = []
            for declaration in [] if hides else declarations:
                for base, written in zip(
                    self.bases.get(declaration, ()),
                    self.written_bases.get(declaration, ()),
                ):
                    _, opened, listed = written.partition("<")
                    given = [
                        _substituted(a, bindings[type_id])
                        for a in (_bracketed_parts(listed) if opened else ())
                    ]
                    for base_type in self._base_types(base, declaration) or ():
                        if not self._looks_into(declaration, base_type):
                            continue
                        params = self.nodes[base_type]["metadata"].get("generics", ())
                        # A parameter of a base written without arguments
                        # stands for no type a subtype writes.
                        filled = given + ["?"] * len(params)
                        bases.append((base_type, dict(zip(params, filled))))
            for base_type, binding in reversed(bases):
                if base_type not in bindings:
                    bindings[base_type] = binding
                    visit(base_type)
            order.append(type_id)

        visit(start)
        return [(t, bindings[t]) for t in reversed(order)]

    def _typed_as_called(self, metadata, types):
        """Whether a C# method of ``metadata`` whose overload takes
        parameters of ``types`` can be called as the call being resolved
        is: with type arguments written only if it has type parameters, and
        without them only if its parameters tell every one."""
        own = metadata.get("type_parameters", ())
        if self._typed_call:
            return bool(own)
        return all(any(re.search(rf"\b{t}\b", w) for w in types or ()) for t in own)

    def _member_owner(self, member_id):
        """The type that defines the member ``member_id``."""
        return _member_split(member_id, self.nodes[member_id]["metadata"]["name"])[0]

    def _extension_hits(self, ref, name, complete):
        """The C# extension methods ``name`` written for the type ``ref``,
        for a type it derives from, or for every type where ``complete``
        says it has no base from outside that may have the member itself,
        but for one an `#if` arm the call is not under alone defines."""
        written = {self._name(ref), *self._ancestors(ref)}
        if complete:
            written.add("object")
        return [
            e
            for e in self.extensions.get(name, ())
            if self.nodes[e]["metadata"]["extends"] in written
            and not self._guard_conflict(
                self.call_guards or (), self.nodes[e]["metadata"].get("guards", ())
            )
        ]

    def _extension_fits(self, hits, receiver):
        """The C# extension methods that the call being resolved, which
        reached the methods ``hits`` on a value ``receiver``, may be a call
        of: of each, the overload ``_overload_by_arity`` picks, else those
        it leaves open."""
        fits, passed, lambdas = set(), self._arguments, self._lambdas
        given = self._passed
        # The receiver is an extension method's first argument.
        self._arguments = passed + 1 if isinstance(passed, int) else passed
        self._lambdas = lambdas and [None, *lambdas]
        self._passed = None if given is None else [self._on or receiver, *given]
        for hit in hits:
            self._candidates_now = None
            one = self._overload_by_arity(hit, static=True)
            fits.update([one] if one else self._candidates_now or ())
        self._candidates_now, self._arguments, self._lambdas = None, passed, lambdas
        self._passed = given
        return sorted(fits)

    def _lambdas_convert(self, types, overload):
        """Whether each lambda the C# or Java call being resolved passes
        converts to the type of the parameter of ``types``, of the method
        ``overload``, it is passed at: a delegate type or a functional
        interface that takes as many parameters, or a type not known to be
        one."""
        for given, written in zip(self._lambdas or (), types or ()):
            takes = None if given is None else self._functional_takes(written, overload)
            if takes is not None and takes != given:
                return False
        return True

    def _functional_takes(self, written, scope_id):
        """How many parameters a lambda that converts to the type ``written``
        in ``scope_id`` declares: a C# delegate type or a Java functional
        interface, the project's or the standard library's. None for any
        other type, or one that takes any number."""
        library = self.standard[self.lang(scope_id)]
        written = re.sub(r"^(?:(?:this|params|ref|in|out|scoped)\s+)+", "", written)
        named, arguments, _ = library.parts(written)
        bare = re.sub(r"<.*", "", written)
        if library.lang == "java":
            project = self._java_type(bare, scope_id)
            return (
                self._java_functional(project)
                if project
                else library.data["functional"].get(library.key(named, scope_id))
            )
        qualifier, _, name = bare.rpartition(".")
        project = self._dotnet_type(
            qualifier.split(".") if qualifier else [], name, scope_id
        )
        if project not in (None, _EXTERNAL):
            # Of the types of the name, the one of as many type parameters
            for node in self._overloads_in(project.rpartition("::")[0], name):
                metadata = self.nodes[node]["metadata"]
                if len(metadata.get("generics", ())) == len(arguments):
                    delegate = metadata.get("kind") == "delegate"
                    return metadata["parameters"][0][1] if delegate else None
            return None
        key = project == _EXTERNAL and library.key(named, scope_id)
        return library.data["functional"].get(key)

    def _java_functional(self, type_id):
        """How many parameters the one abstract method of the Java interface
        ``type_id`` takes, of those `Object` does not have, where it extends
        no other; None for any other type."""
        declarations = self._declarations(type_id)
        if self.nodes[type_id]["metadata"].get("kind") != "interface" or any(
            set(self.bases.get(d, ())) - {_ROOT["java"]} for d in declarations
        ):
            return None
        root = self.standard["java"].data["types"]["java.lang.Object"]["methods"]
        abstract = [
            metadata["parameters"][0][1]
            for d in declarations
            for member in self.members.get(d, {}).values()
            if (metadata := self.nodes[member]["metadata"]).get("body") is False
            and not metadata.get("static")
            and metadata["name"] not in root
        ]
        return abstract[0] if len(abstract) == 1 else None

    def _shadows_import(self, name, source_id, lang):
        """Whether a declaration around ``source_id`` hides an import of
        ``name`` from a bare call of it. In TypeScript a local whose
        recorded binding holds at the call, which a phantom of the parse
        has none of; the file's own binding of it is the import itself. In
        Java a class around the call with a method of the name, its own or
        inherited, a library base's too: fields and types are of other
        namespaces."""
        if lang == "typescript":
            local = self._lexical(name, source_id, skip_classes=True)
            return bool(
                local
                and self._covers_call(local)
                and local.rpartition("::")[0] != source_id.split("::")[0]
            )
        for scope in filter(self._is_class, self._scopes(source_id)):
            hit, complete = self._find_member(scope, name)
            if hit and self.nodes[hit]["type"] in ("Method", "Function"):
                return True
            if not hit and not complete and self._std_inherited(scope, name):
                return True
        return False

    def _covers_call(self, local):
        """Whether ``local``, a local declaration, is in scope where the
        call being resolved is written: its recorded binding holds there,
        from before to after it, by line and column. One no scope is
        recorded for is no declaration a call reaches."""
        scope = self.nodes[local]["metadata"].get("scope")
        return bool(scope) and (
            self._line is None or tuple(scope[0]) <= self._line <= tuple(scope[1])
        )

    def _lexical(self, name, scope_id, skip_classes):
        """Definition of ``name`` in the innermost scope enclosing ``scope_id``.
        What a constant holds, and what an object literal holds, are no
        scopes: each is reached through the name it is written for."""
        for scope in self._scopes(scope_id):
            if skip_classes and self._is_class(scope):
                continue
            node = self.nodes.get(scope)
            if node and (
                node["type"] == "Constant" or node["metadata"].get("kind") == "object"
            ):
                continue
            hit = self.members.get(scope, {}).get(name)
            if hit and self.lang(scope) == "typescript":
                # A local given again, `name#2` and on, is one of its
                # name; of them, the one whose recorded binding holds at
                # the call is the one in scope.
                covering = [n for n in self._given_again(hit) if self._covers_call(n)]
                if covering:
                    hit = covering[-1]
                elif (
                    scope_id == self._caller
                    and self._is_function_scope(scope)
                    and self.nodes[hit]["metadata"].get("scope")
                ):
                    # A local of a function around the call whose block the
                    # call is past binds nothing there.
                    hit = None
            # A function's static members are no names in its body.
            if hit and not (
                self._is_function_scope(scope)
                and self.nodes[hit]["metadata"].get("static")
            ):
                return hit
            if (
                self.lang(scope) == "rust"
                and "::" in scope
                and self._rust_module(scope) == scope
                and any(map(self._is_function_scope, self._scopes(scope)))
            ):
                # A module a function declares sees no item of the function.
                return None
        return None

    def _types(self, lang, name):
        return [
            node_id
            for node_id in self.by_name.get((lang, name), ())
            if self.nodes[node_id]["type"] == "Class"
        ] + self.renamed_types.get((lang, name), [])

    def _types_in(self, lang, name, scope_id, alone=True):
        """The types of ``_types`` that ``name`` can stand for where it is
        written in ``scope_id``: in TypeScript those named there, a module's
        only in it where the name is written ``alone`` and not as a member,
        and of them the ones the innermost function or callback declares,
        whose declaration shadows one outside it."""
        types = self._types(lang, name)
        if lang != "typescript":
            return types
        file_id = scope_id.split("::")[0]
        types = [t for t in types if not alone or self._named_in(t, file_id)]
        depth = {
            t: sum(map(self._is_function_scope, self._scopes(t)))
            for t in self._named_from(types, scope_id)
        }
        return [t for t in depth if depth[t] == max(depth.values())]

    def _find_member(
        self, owner_id, name, inherited_only=False, seen=None, static=False
    ):
        """Look ``name`` up in a type or module and then in the type's bases;
        with ``static`` among a JavaScript class's static members, the
        constructor's, and those of the constructors it extends.

        Returns ``(node_id, complete)``; ``complete`` is False when a base is
        not a project type, so a miss may be a member defined outside, and
        `_SEVERAL` when the types a base may be have several members of the
        name, which the miss leaves as the candidates.
        """
        held = self.nodes[owner_id]["metadata"].get("kind") == "namespace" and (
            self._declared_type(owner_id)
        )
        # A value of a namespace's name has the members of its type the
        # namespace has not.
        if isinstance(held, tuple) and held[0] == "self":
            own = self.members.get(owner_id, {}).get(name)
            return (own, True) if own else self._find_member(held[1], name, seen=seen)
        if not self._is_class(owner_id) or (
            self.nodes[owner_id]["metadata"].get("kind") == "object"
        ):
            # A variable has the members of the type it is declared as, an
            # object those it is written with.
            held = self._declared_type(owner_id)
            if isinstance(held, tuple) and held[0] == "self":
                return self._find_member(held[1], name, seen=seen)
            if self._js_function(owner_id) and not static:
                # A function's instances take their members from its
                # prototype, which assigns none.
                return None, True
            given = self._js_function(owner_id) and self._assigned(owner_id, name)
            if given and len(given) > 1:
                # A member assigned again is any of its assignments.
                self._candidates_now = given
                return None, _SEVERAL
            return self._exported(owner_id, name, set()), True
        lang = self.lang(owner_id)
        applied = seen is not None
        if not applied:
            self._looked_up = None if inherited_only else (owner_id, name)
        seen = set() if seen is None else seen
        if owner_id in seen:
            return None, True
        declarations = self._declarations(owner_id)
        seen.update(declarations)
        # A Ruby class's singleton methods are `self.<name>`, those of the
        # modules it extends among them.
        single = lang == "ruby" and name.startswith("self.")
        if not inherited_only:
            hit = lang == "ruby" and not single and self._mixed(declarations, name)
            if hit:
                return hit, True
            if name in self.builder.py_class_assigned.get(owner_id, ()):
                # An attribute of the class, found before its bases' members;
                # `staticmethod` or `classmethod` called on its `def` is it.
                wrapper = self.builder.py_class_assigned[owner_id][name]
                value = wrapper and self._value(wrapper, owner_id, set())
                if value not in _PY_METHOD_WRAPPERS:
                    return None, True
            hits = [self.members.get(d, {}).get(name) for d in declarations]
            hits = [hit for hit in hits if hit and hit not in self.unmade]
            if lang == "typescript":
                given = [self._assigned(d, name, static) for d in declarations]
                if any(len(one) > 1 for one in given):
                    self._candidates_now = sorted({h for one in given for h in one})
                    return None, _SEVERAL
                hits = [one[0] for one in given if one]
            # Any two that are not a pure virtual function and its definition
            # are alternatives of the build.
            hits = self._defined(hits)
            if lang == "clang" and len(hits) > 1:
                return None, True
            if hits and lang == "rust":
                return self._rust_method(owner_id, hits, name)
            if hits:
                return hits[0], True
            if lang == "rust":
                # A method of an impl block for an alias of the type, which
                # holds where the type's arguments are the alias's: one of
                # several is not told.
                hits = sorted(
                    {
                        hit
                        for blocks in self._aliases_named(owner_id)
                        for block in blocks
                        if (hit := self.members.get(block, {}).get(name))
                    }
                )
                if len(hits) == 1:
                    return hits[0], True
                if hits:
                    self._candidates_now = hits
                    return None, _SEVERAL
            if lang == "clang" and self._c_hides(declarations, name):
                # Declared in the class, which hides the bases' members
                return None, True
            hit = single and self._mixed(declarations, name[5:], "extend")
            if hit:
                return hit, True
        if lang == "clang" and self._c_ambiguous(declarations, name):
            return None, _SEVERAL
        # A type the project only writes impl blocks for is declared outside.
        kinds = {self.nodes[d]["metadata"].get("kind") for d in declarations}
        complete = kinds != {"impl"}
        used = self._c_using(declarations, name) if lang == "clang" else None
        for declaration in declarations:
            for base in self.bases.get(declaration, ()):
                if used and not self._c_used(used, base):
                    continue
                bare = _split_callee(base)[1]
                found = self._base_types(base, declaration)
                if found is None:
                    # The base may be any type of its name, or of none: the
                    # members of those, or every member of the name, are
                    # candidates, not what the call reaches.
                    types = self._types(lang, bare)
                    hits = (
                        (
                            self._find_member(t, name, seen=set(seen), static=static)[0]
                            for t in types
                        )
                        if types
                        else (
                            c
                            for c in self.by_name.get((lang, name), ())
                            if self._is_class(c.rpartition("::")[0])
                        )
                    )
                    self._candidates_now = sorted(set(filter(None, hits))) or None
                    return None, _SEVERAL
                # A module's singleton methods are not its includers'.
                base_types = [
                    t
                    for t in found
                    if not single or self.nodes[t]["metadata"].get("kind") != "module"
                ]
                if not base_types:
                    # A standard trait gives only the members it has.
                    trait = self._standard_type(bare, declaration)
                    given = (trait, declaration, name, owner_id)
                    complete = (
                        complete and bool(trait) and not self._std_given(*given)[0]
                    )
                hits = set()
                for base_type in base_types:
                    reached = self._candidates_now
                    kind = self.nodes[base_type]["metadata"].get("kind")
                    if static and kind == "prototype":
                        # The constructor a class extends is the function.
                        base_type = base_type.removesuffix(".prototype")
                    hit, base_complete = self._find_member(
                        base_type, name, seen=seen, static=static
                    )
                    if not self._looks_into(declaration, base_type):
                        # Its members are not the type's, but what it derives
                        # from outside may have extension methods.
                        self._candidates_now = reached
                        complete = complete and bool(base_complete)
                        continue
                    if base_complete == _SEVERAL:
                        return None, _SEVERAL
                    hits.add(hit)
                    complete = complete and base_complete
                # Where the base may be any of several types, the member is
                # the one of those that has it; where several do, the call
                # reaches one of theirs.
                hits.discard(None)
                if len(hits) == 1:
                    return hits.pop(), True
                if hits:
                    self._candidates_now = sorted(hits)
                    return None, _SEVERAL
            for trait in self.builder.derived.get(declaration, ()):
                given = (trait, declaration, name, owner_id)
                complete = complete and not self._std_given(*given)[0]
        # A Dart mixin looked up as a base of the class it is applied to
        # leaves its `on` types to that class's superclass.
        hit = lang == "dart" and not applied and self._mixed(declarations, name, "on")
        if hit:
            return hit, True
        if lang == "rust" and not applied and not inherited_only:
            # After the type's own traits, a method of a blanket impl
            hits = self._rust_blankets(declarations, name)
            if len(hits) == 1:
                return hits[0], True
            if hits:
                self._candidates_now = hits
                return None, _SEVERAL
        return None, complete

    def _rust_blankets(self, declarations, name, trait=None):
        """The members ``name`` of the Rust blanket impls, of ``trait`` where
        given, whose every bound the type of ``declarations`` implements."""
        met = {_split_callee(b)[1] for d in declarations for b in self.bases.get(d, ())}
        return sorted(
            {
                hit
                for given in self.impl_types.get("", ())
                if set(self.nodes[given]["metadata"]["bounds"]) <= met
                if not trait
                or trait in {_split_callee(b)[1] for b in self.bases.get(given, ())}
                if (hit := self._blanket_member(given, name))
            }
        )

    def _blanket_member(self, given, name):
        """The member ``name`` of the Rust blanket impl ``given``."""
        key = given, name
        if key not in self.blanket_members:
            self.blanket_members[key] = self._find_member(given, name, seen=set())[0]
        return self.blanket_members[key]

    def _looks_into(self, declaration, base_type):
        """Whether a lookup of a member of ``declaration`` goes on into its
        base ``base_type``: in C# that of a class or a struct does not go
        into an interface, whose members only its own type reaches."""
        return not (
            self.lang(declaration) == "dotnet"
            and self.nodes[base_type]["metadata"].get("kind") == "interface"
            and self.nodes[declaration]["metadata"].get("kind") != "interface"
        )

    def _base_types(self, base, declaration):
        """The types the base ``base`` of ``declaration`` is: the type its
        name stands for where the subtype is declared, the types in reach
        the language's lookup cannot tell between, none for a type from
        outside, None when no type of the name is in reach and the base is
        not told to be outside."""
        if base in self.nodes:
            # A base the parser names by its node: a Java enum constant's body.
            return [base]
        named, found = self._named_base(base, declaration)
        if named == _EXTERNAL or (not named and self._standard_type(base, declaration)):
            # Where the subtype is declared the name is the library's.
            return []
        if named:
            return [named]
        lang, bare = self.lang(declaration), _split_callee(base)[1]
        if found or self._types(lang, bare):
            return found or None
        if lang == "python" and isinstance(
            self.builder.imports.get(declaration.split("::")[0], {}).get(bare), tuple
        ):
            # The base is one of several imports of its name.
            return None
        # A JavaScript base that an expression gives, or a value of the
        # project that is no type, is of a type not told.
        root = base.split(".")[0]
        untold = lang == "typescript" and (
            not re.fullmatch(r"[\w$]+(?:\.[\w$]+)*", base)
            or self._lexical(root, declaration, skip_classes=True)
            or isinstance(self._imported(root, declaration), list)
        )
        return None if untold else []

    def _mixed(self, declarations, name, how="prepend"):
        """The member ``name`` of the first type that ``declarations`` list
        under ``how`` in ``mixins`` that has one."""
        for declaration in declarations:
            for written in self.mixins.get((declaration, how), ()):
                reached = self._candidates_now
                module = self._type_named(written, declaration)
                self._candidates_now = reached
                hit = (
                    module not in (None, _EXTERNAL)
                    and (self._find_member(module, name)[0])
                )
                if hit:
                    return hit
        return None

    def _declarations(self, owner_id):
        """The type ``owner_id`` and its other declarations, in a language
        that spreads one type's members over several."""
        lang = self.lang(owner_id)
        if lang == "dart":
            return [owner_id, *self.impl_types.get(self._name(owner_id), ())]
        if lang not in _SPLIT_TYPE_LANGS:
            return [owner_id]
        unit = self._unit(owner_id, lang)
        # The impl blocks of a Rust alias are followed by those of its type.
        names = self._alias_chain(unit, self._name(owner_id))
        others = [
            t
            for name in names
            for t in self._types(lang, name)
            if t != owner_id and self._unit(t, lang) == unit
        ]
        if lang != "rust":
            return [owner_id, *others]
        own = self._name(owner_id)
        if self.nodes[owner_id]["metadata"].get("kind") == "impl":
            # An impl block is one declaration of the type it is for.
            types = [
                t
                for t in others
                if self._name(t) == own
                and self.nodes[t]["metadata"].get("kind") != "impl"
                and self._rust_impl_of(owner_id, t)
            ]
            if not types and owner_id in self.impl_of:
                # One for a type of another crate, or by a name a `use` binds
                types = [self.impl_of[owner_id]]
            if len(types) == 1:
                held = self._declarations(types[0])
                return [owner_id, *(d for d in held if d != owner_id)]
        return [
            owner_id,
            *(
                t
                for t in others
                if self._name(t) != own or self._rust_impl_of(t, owner_id)
            ),
            *(b for b in self.impls_for.get(owner_id, ()) if b not in others),
        ]

    def _rust_impl_of(self, block, owner_id):
        """Whether ``block``, a Rust type or impl block of the name of the
        type ``owner_id``, holds members of that type: an impl block whose
        header's path leads to it, else whose `use` of the name does, or,
        where none does, of the crate's only type of the name; another type
        is not."""
        metadata = self.nodes[block]["metadata"]
        if metadata.get("kind") != "impl":
            return False
        named = {
            self._type_named(path, module)
            for module, path in metadata.get("impl_paths", ())
        } - {None}
        if named:
            # The path each header of the name writes names the type from
            # the module it is written in.
            return owner_id in named
        if (block, owner_id) in self._impl_reading:
            return True
        self._impl_reading.add((block, owner_id))
        try:
            name = self._name(block)
            imported = self._imported(name, block.rpartition("::")[0])
        finally:
            self._impl_reading.discard((block, owner_id))
        if isinstance(imported, list) and imported:
            return owner_id in imported
        defined = [
            t
            for t in self._types("rust", name)
            if self.nodes[t]["metadata"].get("kind") != "impl"
            and self._unit(t, "rust") == self._unit(owner_id, "rust")
        ]
        return defined in ([], [owner_id])

    def _aliases_named(self, owner_id):
        """The impl blocks of its crate written for each alias of the Rust
        type ``owner_id``, or of the type it is an alias of, but itself."""
        unit, own = self._unit(owner_id, "rust"), self._name(owner_id)
        name = self._alias_chain(unit, own)[-1]
        if (unit, name) not in self.alias_blocks:
            self.alias_blocks[unit, name] = {
                alias: [
                    t
                    for t in self._types("rust", alias)
                    if self._unit(t, "rust") == unit
                ]
                for crate, alias in self.alias_targets
                if crate == unit and self._alias_chain(unit, alias)[-1] == name
            }
        return [
            blocks
            for alias, blocks in self.alias_blocks[unit, name].items()
            if alias != own
        ]

    def _alias_chain(self, unit, name):
        """``name`` and the names of what the Rust alias ``name`` of the crate
        ``unit`` stands for, an alias's alias after it."""
        chain = [name]
        while (target := self.alias_targets.get((unit, chain[-1]))) not in (
            None,
            *chain,
        ):
            chain.append(target)
        return chain

    def _build_alias_targets(self):
        """Fill ``alias_targets``: ``{(crate, alias name): name of the type it
        stands for}`` of the Rust aliases a crate declares, None for one it
        declares as several."""
        self.alias_targets = {}
        for (module, name), declared in self.builder.rust_alias_types.items():
            written = {
                _rust_type_parts(_RUST_BORROW_RE.sub("", h))[0]
                for _, h in _rust_declared(declared)
            }
            named = {path.rpartition("::")[2] for path in written}
            key = self._unit(module, "rust"), name
            target = next(iter(named)) if len(named) == 1 else None
            if self.alias_targets.get(key, target) != target:
                target = None
            self.alias_targets[key] = target

    def _std_given(self, trait, declaration, name, owner_id):
        """Look the member ``name`` up in ``trait``, a trait of the standard
        library's data that ``declaration`` implements or derives for the
        project type ``owner_id``: ``(True, the value it returns)`` where
        the trait has it, also as an attribute, which Python's data names
        behind a dot."""
        library = self.standard[self.lang(declaration)]
        tables = ("methods", name), ("statics", name), ("methods", f".{name}")
        for table, member in tables:
            receiver = ("self", owner_id)
            given = library.member(("outside", trait, ()), member, receiver, table)
            if given[0]:
                return given
        return False, None

    def _assigns(self, owner_id, name):
        """Whether the JavaScript, TypeScript or Python class ``owner_id``
        assigns the attribute ``name`` itself: `this.name = ...`, `self.name
        = ...`, or `name = ...` in a Python class body."""
        return any(
            name in self.builder.assigned.get(d, ())
            or name in self.builder.py_class_assigned.get(d, ())
            for d in self._declarations(owner_id)
        )

    def _instance_values(self, owner_id, name):
        """``(declaration, value)`` of each value that the methods of the
        Python, JavaScript or TypeScript class ``owner_id`` or a base assign
        to the attribute ``name`` of the instance, None for one not known."""
        return [
            (d, value)
            for t, _ in self._lookup_order(owner_id, name)
            for d in self._declarations(t)
            if name in self.builder.assigned.get(d, ())
            for value in self.builder.assigned[d][name] or [None]
        ]

    def _instance_member(self, owner_id, name):
        """`_find_member` on an instance of the Python, JavaScript or
        TypeScript class ``owner_id``: an attribute that the methods of the
        class or a base assign to the instance is found before a method of
        the name. The member is what the values assigned hold, a member
        read (`this.m`, `this.m.bind(this)`, `self.m`) the class's member
        of that name; where they do not tell one, None with it and the
        method as the candidates, and a field declared of the name. Read
        inside one of its values, the attribute is the class's member."""
        if self.lang(owner_id) not in ("python", "typescript"):
            return self._find_member(owner_id, name)
        hit, complete = self._find_member(owner_id, name, seen=set())
        if (owner_id, name) in self._reading:
            return hit, complete
        written = self._instance_values(owner_id, name)
        if not written:
            return hit, complete
        held = set()
        self._reading.add((owner_id, name))
        for declaration, value in written:
            read = value and re.fullmatch(
                r"(?:this|self)\.([\w$]+)(?:\.bind\([^()]*\))?", value
            )
            if read:
                held.add(self._find_member(owner_id, read[1], seen=set())[0])
            elif value and self.lang(declaration) == "python":
                found = self._value(value, declaration, set())
                held.add(found[1] if found and found[0] == "def" else None)
            else:
                held.add(None)
        self._reading.discard((owner_id, name))
        if len(held) == 1 and None not in held:
            return held.pop(), True
        if hit and self.nodes[hit]["type"] != "Method":
            return hit, complete
        fits = sorted((held | {hit}) - {None})
        if fits:
            self._candidates_now = fits
            return None, _SEVERAL
        return None, True

    def _std_inherited(self, owner_id, name):
        """Value the member ``name`` returns that the project type
        ``owner_id`` has from a standard trait, where the data says."""
        for declaration in self._declarations(owner_id):
            bases = self.bases.get(declaration, ())
            named = (
                self._standard_type(_split_callee(base)[1], declaration)
                for base in bases
            )
            derived = self.builder.derived.get(declaration, ())
            for trait in [*filter(None, named), *derived]:
                known, value = self._std_given(trait, declaration, name, owner_id)
                if known:
                    return value
        return None

    def _lineage(self, class_id, seen):
        """Yield the Python class ``class_id`` and the project classes it
        inherits from, each before its own bases."""
        if class_id in seen:
            return
        seen.add(class_id)
        yield class_id
        for base in self.bases.get(class_id, ()):
            named = self._type_named(base, class_id)
            if named not in (None, _EXTERNAL):
                yield from self._lineage(named, seen)

    def _unit(self, node_id, lang):
        """What bounds one split type: the scope that declares it in
        TypeScript, the package directory in Go, the crate in Rust, the
        enclosing namespace in Ruby, in C/C++ the file that defines the
        type and its qualified name."""
        path = node_id.split("::")[0][5:]
        if lang == "ruby":
            # A scope without a name of its own is one file's.
            scope = node_id.rpartition("::")[0]
            return scope if "::<" in node_id else scope.partition("::")[2]
        if lang == "typescript":
            return node_id.rpartition("::")[0]
        if lang == "dotnet":
            # A partial type is one type in every file of its namespace,
            # by its name and the number of its type parameters.
            metadata = self.nodes[node_id]["metadata"]
            if metadata.get("partial"):
                return self._dotnet_path(node_id), metadata["arity"]
            return node_id
        if lang == "clang":
            # The node that holds a type's out-of-class members stands for
            # the one definition of its qualified name that its file is
            # compiled with; one of another namespace is another type.
            qualified = node_id.partition("::")[2]
            defined = [
                t
                for t in self._types(lang, self._name(node_id))
                if "end_lineno" in self.nodes[t]["metadata"]
                and t.partition("::")[2] == qualified
                and self._in_reach(t, node_id)
            ]
            own = "end_lineno" in self.nodes[node_id]["metadata"]
            if not own and len(defined) == 1:
                path = defined[0].split("::")[0][5:]
            return path, qualified
        if lang == "go":
            return path.rpartition("/")[0]
        if lang == "rust":
            crate = next(
                (f for f in _ancestors(path) if f and f in self.crate_folders), ""
            )
            # Cargo builds what is under these as crates of their own.
            target = path[len(crate) :].lstrip("/").partition("/")[0]
            if target in ("examples", "tests", "benches"):
                return f"{crate}/{target}".lstrip("/")
            return crate
        return ""

    def _ancestors(self, owner_id, seen=None):
        """Names of every type ``owner_id``, in any of its declarations,
        inherits from."""
        seen = set() if seen is None else seen
        lang = self.lang(owner_id)
        for declaration in self._declarations(owner_id):
            for base in self.bases.get(declaration, ()):
                base = _split_callee(base)[1]
                if base not in seen:
                    seen.add(base)
                    for base_type in self._types(lang, base):
                        self._ancestors(base_type, seen)
        return seen

    def _decide(self, candidates, source_id):
        """The one candidate scope and build evidence leave of several
        that fit — None where nothing in the language decides between
        them. The survivors are remembered as the call's candidates."""
        fitting = self._fitting(candidates, source_id)
        if len(fitting) == 1 and not self._candidate_only(fitting[0], source_id):
            return fitting[0]
        # Those the rules did not rule out may be meant; where they ruled
        # all out, all stay possible but none is pinned.
        self._candidates_now = sorted(fitting or candidates)
        return None

    def _fitting(self, candidates, source_id):
        """The candidates the call's own file can mean: a definition it
        names per the language's scoping — its own declarations, its
        imports, the package or namespace it shares them with, what its
        `use`/`require`/`#include` reach — and, of those, the ones whose
        build can run with its (Go build constraints, C/C++ `#if` guards)."""
        lang = self.lang(source_id)
        # The file's own declaration of the name is what the name binds
        # to in it — only where the file declares none do its imports,
        # package, crate or includes decide.
        own = [c for c in candidates if c.split("::")[0] == source_id.split("::")[0]]
        if lang in ("clang", "dotnet"):
            # One an `#if` arm the call is not under alone builds binds it
            # only in the builds that pick the arm.
            guards = self._caller_guards(source_id)
            built = [
                c
                for c in own
                if set(self._effective_guards(c, source_id)) <= set(guards)
            ]
            own = built or own + [
                c for c in candidates if c not in own and self._in_scope(c, source_id)
            ]
        scoped = own or [c for c in candidates if self._in_scope(c, source_id)]
        if scoped:
            candidates = scoped
        if lang == "go":
            return self._go_fitting(candidates, source_id)
        if lang in ("clang", "dotnet"):
            return [
                c
                for c in candidates
                if not self._guard_conflict(
                    guards, self._effective_guards(c, source_id)
                )
            ]
        return list(candidates)

    def _candidate_only(self, candidate, source_id):
        """Whether the sole fitting definition still may not be pinned to
        the call: it is built only under an `#if` arm the call site does
        not select, and the standard library has the name for the builds
        that leave the arm out."""
        lang = self.lang(source_id)
        if lang not in ("clang", "dotnet"):
            return False
        caller = self._caller_guards(source_id)
        if caller or not self._effective_guards(candidate, source_id):
            return False
        if lang not in self._standard_names:
            data = self.standard[lang].data
            self._standard_names[lang] = (
                {
                    re.split(r"::|\.", key.partition("`")[0])[-1]
                    for key in (*data["types"], *data["functions"])
                },
                {
                    name
                    for held in data["types"].values()
                    for name in (*held["methods"], *held["statics"])
                },
            )
        names, members = self._standard_names[lang]
        name = self.nodes[candidate]["metadata"]["name"]
        owner = candidate.rpartition("::")[0]
        if self._is_class(owner):
            # A member: of a type the library names too, by a name its
            # types have.
            return self.nodes[owner]["metadata"]["name"] in names and name in members
        return name in names

    def _caller_guards(self, scope_id):
        """The signed `#if` atoms the call being resolved is under: its
        line's where told, else those of ``scope_id``."""
        return self.call_guards or self.nodes[scope_id]["metadata"].get("guards", ())

    def _effective_guards(self, candidate, source_id):
        """The guards a definition is built under: its own where it has
        them — or, for a C/C++ function, the arm a prototype in a header
        the caller includes declares it in."""
        guards = self.nodes[candidate]["metadata"].get("guards", ())
        if guards or self.lang(candidate) != "clang":
            return guards
        key = candidate, source_id.split("::")[0]
        if key not in self._guards:
            name = candidate.rpartition("::")[2]
            declared = [
                found[name]
                for file in self._include_reach(key[1])
                if name in (found := self.builder.declares.get(file, {}))
            ]
            missing = not declared or any(not arms for arms in declared)
            self._guards[key] = () if missing else frozenset().union(*declared)
        return self._guards[key]

    def _go_fitting(self, candidates, source_id):
        """The candidates in files a build with the caller's file holds:
        a `_test.go` definition is visible to the package's test files
        only, and platform constraints that contradict the caller's rule
        a definition out."""
        caller = source_id.split("::")[0]
        req, excl, caller_test = self._go_build(caller)
        kept = []
        for candidate in candidates:
            cfile = candidate.split("::")[0]
            c_req, c_excl, c_test = self._go_build(cfile)
            if c_test and not (
                caller_test and self._unit(cfile, "go") == self._unit(caller, "go")
            ):
                continue
            if not _go_builds_meet((req, excl), (c_req, c_excl)):
                continue
            kept.append(candidate)
        return kept

    def _go_build(self, file_id):
        """``(requires, excludes, is_test)`` of the Go file ``file_id``."""
        build = self.nodes.get(file_id, {}).get("metadata", {}).get("build", ())
        req = frozenset(build[0]) if build else frozenset()
        excl = frozenset(build[1]) if build else frozenset()
        test = bool(build and build[2])
        return req, excl, test

    @staticmethod
    def _guard_conflict(caller, candidate):
        """A signed `#if` atom the caller holds that the candidate's guards
        negate, or the other way around."""
        return any(_flip_sign(sign) in candidate for sign in caller)

    def _dotnet_extension(self, value, name):
        """The project extension methods a call under ``name`` on the
        outside-typed ``value`` may mean — written for the type or one it
        is of, per the library's data."""
        types = self.standard["dotnet"].data["types"]
        scope = self.standard["dotnet"].data["scope"]
        # The type and, through its bases, every type it is of.
        given, todo = set(), [value[1].split("<")[0]]
        while todo:
            seen = todo.pop()
            if seen in given:
                continue
            given.add(seen)
            todo += types.get(seen, {}).get("bases", ())
        hits = []
        for ext in self.extensions.get(name, ()):
            written = self.nodes[ext]["metadata"]["extends"]
            named, _, _ = _dotnet_type_parts(written)
            key = (
                scope.get(written)
                or scope.get(named)
                or self._dotnet_std_key(named, ext.split("::")[0])
                or next(
                    (
                        t
                        for t in given
                        if t.rpartition(".")[2].rpartition("`")[0]
                        == named.rpartition("`")[0]
                    ),
                    named,
                )
            )
            if key in given or (
                key == "System.Object" and value[1].split("<")[0] in types
            ):
                hits.append(ext)
        return hits

    def _include_reach(self, file_id):
        """The files ``file_id``'s `#include`s reach, transitively."""
        if file_id not in self.reaches:
            self.reaches[file_id] = _closure(self.includes, {file_id})
        return self.reaches[file_id]

    def _in_scope(self, candidate, source_id):
        """``candidate`` is a definition the file of ``source_id`` can mean
        for a name it writes: declared in that file, imported into it, or
        in the package, crate, module or namespace the language groups it
        with. What the file cannot name is not the call's."""
        c_file = candidate.split("::")[0]
        s_file = source_id.split("::")[0]
        if c_file == s_file:
            return True
        lang = self.lang(source_id)
        if lang != self.lang(candidate):
            return True
        owner = candidate.rpartition("::")[0]
        owner_name = owner.rpartition("::")[2] if owner != c_file else ""
        imports = self._uses(source_id)
        # The name an import binds is the type's for a member, the
        # definition's own for a type or a top-level one.
        named = owner_name or candidate.rpartition("::")[2]
        if named and named in imports:
            return True
        if lang == "go":
            if self._unit(candidate, "go") == self._unit(source_id, "go"):
                return True
            for target in imports.values():
                for folder, module in self.go_modules:
                    if target == module or target.startswith(f"{module}/"):
                        path = "/".join(
                            p for p in (folder, target[len(module) + 1 :]) if p
                        )
                        if c_file in self.go_packages.get(path, ()):
                            return True
            return False
        if lang == "rust":
            # Within a crate a name needs a `use`, or a module the module
            # tree puts at or beside the caller's or inside it; another
            # crate's names need a `use` too.
            if owner_name and owner_name in imports:
                return True
            own = self._rust_module(source_id)
            held = self._rust_module(candidate)
            return held == own or self._rust_parent(held) in (
                own,
                self._rust_parent(own) or own,
            )
        if lang == "java":
            own = self.nodes[c_file]["metadata"].get("java_package")
            here = self.nodes[s_file]["metadata"].get("java_package")
            if own and own == here:
                return True
            if owner_name and owner_name in imports:
                return True
            # `import pkg.*` makes every type of the package nameable.
            return any(
                key.endswith("*") and isinstance(target, str) and own == target
                for key, target in imports.items()
            )
        if lang == "dotnet":
            own = self.nodes[owner if owner.startswith("FILE:") else c_file][
                "metadata"
            ].get("namespace")
            here = self.nodes[s_file]["metadata"].get("namespace")
            if own and own == here:
                return True
            usings = self._dotnet_imports(s_file)
            return own in usings.values() or (owner_name and owner_name in usings)
        if lang == "php":
            own = self.nodes[c_file]["metadata"].get("namespace")
            here = self.nodes[s_file]["metadata"].get("namespace")
            return (own and own == here) or bool(owner_name and owner_name in imports)
        if lang == "python":
            if owner_name and owner_name in imports:
                return True
            return any(
                c_file in self.py_modules.get(target, {}).values()
                for target in imports.values()
                if isinstance(target, str)
            )
        if lang == "typescript":
            if owner_name and owner_name in imports:
                return True
            required = self.builder.required_names.get(s_file, ())
            return any(
                c_file in (self._js_module(target, s_file, name in required) or ())
                for name, target in imports.items()
                if isinstance(target, str)
            )
        if lang == "dart":
            if self.dart_parts.get(c_file, c_file) == self.dart_parts.get(
                s_file, s_file
            ):
                return True
            return owner_name and owner_name in imports
        if lang == "ruby":
            return c_file in self._loaded(s_file)
        if lang == "clang":
            if c_file in self._include_reach(s_file):
                return True
            # A prototype in an included header promises a definition
            # somewhere in the build: every file that defines the name
            # the caller can name.
            name = candidate.rpartition("::")[2]
            return any(
                name in self.builder.declares.get(file, ())
                for file in self._include_reach(s_file)
            )
        return True
