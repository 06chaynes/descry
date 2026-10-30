"""The Rust parser."""

import re
from bisect import bisect_right
from functools import cache

from descry import _stdlib
from descry._parsing import (
    _RUST_SPACE,
    _TYPE_ALIAS_RE,
    BaseParser,
    _bracketed_parts,
    _closing,
    _filled,
    _flat,
    _generic_metadata,
    _innermost,
    _named,
    _spaced,
    _strip_generics,
    blank_strings,
    source_lines,
    token_count,
)

# Where a pattern binds names: closure parameters, a `for` pattern, a `let`
# with a pattern rather than one name, and a match arm.
_RUST_BINDER_RES = (
    re.compile(r"\|([^|]*)\|"),
    re.compile(r"\bfor\s++(.+?)\s+in\b"),
    re.compile(r"\b(?:if|while)\s++let\s++(.+?)=(?!=)"),
    re.compile(r"\blet\s+(?:mut\s+)?([({\[A-Z].*?)=(?!=)"),
    # An arm's pattern, after the `match … {` that opens the arms on its line,
    # up to its guard
    re.compile(r"^\s*(?:(?:[^=]|=(?!>))*?\bmatch\b[^{]*\{)?\s*(.+?)(?:\s+if\b.*?)?=>"),
)


# `let name = vec![...];`, `let name = "...";` and `const NAME: Type = ...`,
# which is read in every function below where it is declared
_RUST_VEC_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*vec!\[[^\[\]]*\];"
)


_RUST_STRING_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*(?P<bytes>b)?r?#*\""
)


_RUST_CONST_RE = re.compile(
    r"\b(?:const|static)\s+(?:mut\s+)?(?P<name>[A-Z][A-Z0-9_]*)\s*:\s*"
    r"(?P<type>[^=;().]+?)\s*="
)


_RUST_INDEX_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*(?:&\s*(?:mut\s+)?)?(?P<of>\w+)\s*\["
)


# An array, a slice or a `vec!` literal, what it holds in group 1
_RUST_SEQUENCE_RE = re.compile(r"\s*(?:&\s*(?:mut\s+)?)?(?:vec!\s*)?\[(.*)\]\s*")


# Type arguments written after a path's `::`, nested to three levels
_RUST_TURBOFISH = r"::<(?:[^<>()]|<(?:[^<>()]|<[^<>()]*>)*>)*>"


# A struct literal, up to its `{`: a type written after modules alone, and
# the type arguments written after it
_RUST_STRUCT_LITERAL_RE = re.compile(
    rf"(?P<type>(?:[a-z_]\w*::)*[A-Z]\w*)(?P<turbofish>{_RUST_TURBOFISH})?\s*\{{"
)


_RUST_VALUE_PATH_RE = re.compile(r"(?:\w+::)*[A-Z]\w*")


# The traits a `#[derive(...)]` implements for the item below it
_RUST_DERIVE_RE = re.compile(r"#\[derive\(([^)]*)\)\]")


# What parentheses hold, a closure's or a struct literal's braces too
_RUST_PARENTHESIZED = r"[^()]*"


for _ in range(4):
    _RUST_PARENTHESIZED = rf"(?:[^()]|\({_RUST_PARENTHESIZED}\))*"


# The value in the head of a block, up to the `{` that opens it: the first
# outside parentheses, as no struct literal stands there bare
_RUST_HEAD_VALUE = rf"(?P<value>(?:[^(){{}}]|\({_RUST_PARENTHESIZED}\))*?)\s*\{{"


# Where a pattern is matched against a value: `for` and `if let` or `while
# let` with the block they open, `let` with a pattern, and `match`
_RUST_FOR_RE = re.compile(r"\bfor\s++(?P<pattern>.+?)\s+in\s+" + _RUST_HEAD_VALUE)


# A `for` up to its `in`, whose value goes on below
_RUST_FOR_HEAD_RE = re.compile(r"\bfor\s++(?P<pattern>.+?)\s+in\b(?!.*\{)")


_RUST_IF_LET_RE = re.compile(
    r"\b(?:if|while)\s++let\s++(?P<pattern>.+?)\s*=(?!=)\s*" + _RUST_HEAD_VALUE
)


_RUST_LET_RE = re.compile(
    r"\blet\s+(?P<pattern>[^=]*[(\[{][^=]*?)\s*=(?!=)\s*(?P<value>.+?)"
    r"(?:;|\s+else\s*\{(?:.*\}\s*;)?)\s*$"
)


def _rust_lets(code):
    """`_RUST_LET_RE`'s match on each `let` statement of ``code``, no `if
    let` or `while let`, each read up to the `;` it ends with where code
    goes on after it."""
    for head in re.finditer(r"\blet\s", code):
        if re.search(r"\b(?:if|while)\s*$", code[: head.start()]):
            continue
        end = _closing([code], 0, head.end(), ";")
        cut = code[: end[1] + 1] if end and end[2] == ";" else code
        if found := _RUST_LET_RE.match(cut, head.start()):
            yield found


def _rust_range(text):
    """The language item a range expression ``text`` makes, `a..b` a
    `Range`; None for any other expression."""
    flat = _flat(text).strip()
    start, dots, end = flat.partition("..")
    if not dots or "." in start[-1:] + end[:1].lstrip("="):
        return None
    if end.startswith("="):
        return "RangeInclusiveStruct" if start.strip() else "RangeToInclusive"
    if start.strip():
        return "Range" if end.strip() else "RangeFrom"
    return "RangeTo" if end.strip() else "RangeFull"


# A number with its type as a suffix, `0usize`, or a value cast, `i as u8`
_RUST_TYPED_VALUE_RE = re.compile(
    r"\d[\d_]*(?P<suffix>[iu](?:8|16|32|64|128|size))|.+\sas\s+(?P<cast>[A-Za-z_][\w:]*)"
)


# What a block's last statement diverges with
_RUST_DIVERGES_RE = re.compile(
    r"(?:return|continue|break)\b|(?:panic|unreachable|todo|unimplemented)!"
)


# A statement on a line below one that ends with a block's `}`, no `else`,
# method or operator that goes on with it
_RUST_STATEMENT_AFTER_RE = re.compile(r"[ \t]*\n\s*(?!else\b|as\b)['\w]")


# What opens a block that is an expression: `match x`, `if c`, `unsafe`,
# `loop`, a label
_RUST_BLOCK_HEAD_RE = re.compile(r"\b(?:match|if|unsafe|loop)\b|'\w+:")


# Where an expression, a bare block, starts: the line's start, or past `=`,
# `=>`, `(`, `,` or `return`
_RUST_EXPRESSION_START_RE = re.compile(
    r"(?:^|=>|(?<![=!<>])=(?![=>])|[(,]|\breturn)\s*$"
)


# A statement up to the type a struct pattern or literal names: no block
# follows a type, and `@` binds only in a pattern
_RUST_PATTERN_HEAD_RE = re.compile(
    r"(?:^\s*(?:for\s+)?|\blet\s+|(?<!\|)[(,|@]\s*&?\s*)(?:\w+::)*[A-Z]\w*\s*$"
)


# A control statement up to an expression it heads, where a brace opens its
# block: an `if` or a `while` past its `let`'s `=`, a `match`, a `for` past
# its `in`
_RUST_CONDITION_RE = re.compile(
    r"\s*(?:else\s+)?(?:(?:if|while)\b(?!\s+let\b[^=]*$)|match\b|for\b.*\bin\b)"
)


_RUST_MATCH_RE = re.compile(r"\bmatch\s++" + _RUST_HEAD_VALUE)


_RUST_ARM_RE = re.compile(r"^\s*+(?P<pattern>.+?)(?:\s+if\s.+?)?\s*=>")


# A bracket, a guard's `if`, or the `=` of a select!-style arm (`pattern =
# future =>`), which no pattern has (`..=` is a range's)
_RUST_ARM_HEAD_TOKEN_RE = re.compile(
    r"[(\[{]|[)\]}]|(?:^|[ \t])if\b|(?<![<>=!+\-*/%&|^.:])=(?![=>])"
)


def _arm_head_end(text, depth=0):
    """Where an arm's pattern written in ``text``, which starts ``depth``
    brackets in, ends before its `=>`: at a guard's `if` or a select!-style
    arm's `=` outside brackets, which an attribute's `#[cfg(feature =
    "x")]` is not. None where it does not."""
    for found in _RUST_ARM_HEAD_TOKEN_RE.finditer(text):
        token = found[0]
        if token in "([{":
            depth += 1
        elif token in ")]}":
            depth -= 1
        elif depth == 0:
            return found.start()
    return None


def _matches_arguments(code, start, depth, crossed=False):
    """``(depth, comma, close)`` of the arguments of a `matches!` read in
    ``code`` from ``start``, ``depth`` parentheses in: the depth where the
    reading stops, the first comma at the top level unless one is
    ``crossed`` already, and the offset past the `)` that closes them; None
    for one not on the line."""
    comma = close = None
    for j in range(start, len(code)):
        if code[j] == "(":
            depth += 1
        elif code[j] == ")":
            depth -= 1
            if depth == 0:
                close = j + 1
                break
        elif code[j] == "," and depth == 1 and comma is None and not crossed:
            comma = j
    return depth, comma, close


def _top_level_arrow(code, paren=0, bracket=0, brace=0):
    """``(offset, paren, bracket, brace)`` of the first `=>` in ``code``
    outside brackets, read from the depths given, and the depths there or
    at the line's end; offset None where there is none."""
    for j, ch in enumerate(code):
        paren += (ch == "(") - (ch == ")")
        bracket += (ch == "[") - (ch == "]")
        brace += (ch == "{") - (ch == "}")
        if ch == "=" and code[j + 1 : j + 2] == ">" and paren == bracket == brace == 0:
            return j, paren, bracket, brace
    return None, paren, bracket, brace


# The parameters of a closure passed to a call
_RUST_CLOSURE_RE = re.compile(r"(?:(?<=[(,])|^)\s*(?:move\s+)?\|(?P<params>[^|]*)\|")


# How a type is borrowed or abstracted over, before its name
_RUST_BORROW_RE = re.compile(r"^(?:[&\s]|'\w+\s|(?:mut|dyn|impl)\s)+")


@cache
def _rust_declarations():
    """The patterns ``scan_declarations`` reads a Rust file with. The
    primitive types are those the standard library's data names."""
    scope = _stdlib.load("rust")["scope"]
    primitives = "|".join(sorted(name for name, key in scope.items() if name == key))
    name = r"\b(?P<name>[a-z_]\w*)\s*:\s*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?"
    # Type arguments, nested three deep, that name no tuple or function type
    plain = r"[^<>().;={}]*"
    arguments = rf"<{plain}(?:<{plain}(?:<{plain}>{plain})*>{plain})*>"
    return (
        # `name: Type`, `name: &'a mut Type`, `name: &dyn Trait`, `name: Type<'a>`;
        # not `name: None`, a field of a struct literal. A type is read with
        # its type arguments, `Vec<Type>`, or marked generic where they are
        # more than those: unless the project defines it, its methods are
        # mostly those of what it wraps.
        re.compile(
            rf"{name}(?:dyn\s+|impl\s+)?(?P<type>(?!None\b)(?:[a-z_]\w*::)*[A-Z]\w*"
            rf"\b(?![:(!])(?:(?!<'\w+>){arguments})?)(?:<'\w+>|(?P<generic><))?"
        ),
        # `name: &[Type]`, `name: [u8; 4]`
        re.compile(
            rf"{name}(?P<type>\[\s*(?:&\s*)?(?:'\w+\s+)?(?:mut\s+)?"
            rf"(?:(?:[a-z_]\w*::)*[A-Z]|(?:{primitives})\b)[^\[\](),.={{}}]*\])"
        ),
        # `name: &str`, `name: usize`
        re.compile(rf"{name}(?P<type>{primitives})\b(?!\s*[:(!<])"),
        # A type parameter's bound, in `<T: Trait>` or a `where` clause
        re.compile(
            r"(?:[<,]|\bwhere\s|^)\s*(?P<param>[A-Z]\w*)\s*:\s*"
            r"(?P<type>[A-Z]\w*)\b(?![<:(!])"
        ),
    )


_RUST_GAP = rf"[{_RUST_SPACE}\n]"


def _unit(text, ends):
    """``{"unit": True}`` where ``text``, after the name of a struct or of a
    variant, ends it there, at what matches ``ends``: it holds nothing, and
    its name is its one value."""
    return {"unit": True} if re.match(rf"\s*(?:{ends})", text) else {}


def _rust_held(text):
    """``{position or field name: type}`` of what the `(A, B)` or the `{ a:
    A }` after the name of a struct or of a variant says it holds."""
    text = re.sub(r"^\s*<[^(){]*>", "", text).lstrip()
    if text[:1] not in ("(", "{"):
        return {}
    parts = [re.sub(rf"^{_RUST_VIS}", "", p) for p in _bracketed_parts(text[1:])]
    if text[0] == "(":
        return dict(enumerate(parts))
    fields = (part.partition(":") for part in parts)
    return {name.strip(): held.strip() for name, _, held in fields if held}


def _rust_bindings(pattern, steps=()):
    """Yield ``(name, steps)`` for each name the Rust pattern binds: the
    steps lead from the value matched to the part bound, `Some@0` for what
    `Some(..)` holds first, `@1` for the second of a tuple, `Point@x` for
    a field and `*` for an element of a slice."""
    # The first of `A | B`: every alternative binds the same names.
    first = _bracketed_parts(pattern.replace("|", ",") + ")")[:1]
    pattern = re.sub(r"^(?:&\s*|(?:mut|ref|box)\s+)+", "", "".join(first))
    name, at, rest = pattern.partition("@")
    if at:
        yield from _rust_bindings(name, steps)
        pattern = rest.strip()
    opened = re.match(r"((?:\w+::)*[A-Z]\w*)?\s*([(\[{])", pattern)
    if re.fullmatch(r"[a-z]\w*|_\w+", pattern):
        yield pattern, steps
    elif opened:
        head = (opened[1] or "").rpartition("::")[2]
        for position, part in enumerate(_bracketed_parts(pattern[opened.end() :])):
            if part.startswith(".."):
                break
            field, _, inner = part.partition(":")
            if opened[2] == "[":
                yield from _rust_bindings(part, (*steps, "*"))
            elif opened[2] == "(":
                yield from _rust_bindings(part, (*steps, f"{head}@{position}"))
            else:
                field = field.split()[-1]
                yield from _rust_bindings(inner or part, (*steps, f"{head}@{field}"))


def _rust_return_type(text):
    """The type the text of a Rust signature after its parameter list
    declares the function returns: up to its `where`, its body or its `;`
    outside brackets; `()` for none."""
    found = re.match(r"\s*->\s*", text)
    if not found:
        return "()"
    depth = 0
    for at in range(found.end(), len(text)):
        char = text[at]
        if char in "<([":
            depth += 1
        elif char in ")]" or char == ">" and text[at - 1] != "-":
            depth -= 1
        elif not depth and (
            char in "{;"
            or re.match(r"where\b", text[at:])
            and not re.match(r"\w", text[at - 1])
        ):
            return text[found.end() : at].strip()
    return text[found.end() :].strip()


def _angle_end(text, start):
    """Offset of the `>` that closes the `<` at ``start`` of ``text``, which
    the `>` of a `->` does not; None where none does."""
    depth = 0
    for at in range(start, len(text)):
        depth += (text[at] == "<") - (text[at] == ">" and text[at - 1] != "-")
        if not depth:
            return at
    return None


def _rust_type_declared(text):
    """``(name, type parameters as written, type)`` of the `type Name<T> =
    Type;` that ``text`` starts with, else None."""
    found = re.match(rf"\s*{_RUST_VIS}type\s+(\w+)\s*", text)
    if not found:
        return None
    rest, params = text[found.end() :], ""
    at = _angle_end(rest, 0) if rest.startswith("<") else None
    if at is not None:
        params, rest = rest[: at + 1], rest[at + 1 :]
    held = re.match(r"\s*=\s*([^;]+);", rest)
    return held and (found[1], params, held[1])


def _rust_borrow(written):
    """Whether the Rust expression ``written``, its string literals' text
    blanked to `_`, is a reference: True for `&x` and a string literal,
    False for `*x`, a number and a boolean, what it holds for one in
    parentheses alone, `(*x)`, None where its writing does not tell."""
    text = written.strip()
    if text.startswith("(") and len(_bracketed_parts(text[1:])) == 1:
        inner = _bracketed_parts(text[1:])[0]
        if text[1:].rstrip(")").strip() == inner:
            return _rust_borrow(inner)
    # A string's text and its quotes are blanked: two `_` at least.
    if text.startswith("&") or re.fullmatch(r"b?__+", text):
        return True
    if text.startswith("*") or re.fullmatch(r"-?\d[\w.]*|true|false", text):
        return False
    return None


def _use_leaves(tree: str, prefix: str = ""):
    """Yield ``(local name, full path)`` for each name a `use` tree binds:
    `a::{b, c::d as e, f::*}` binds b to `a::b` and e to `a::c::d`. The
    glob is yielded under its own path, `a::f::*`."""
    head, brace, group = tree.strip().partition("{")
    if not brace:
        path, _, alias = (prefix + head).partition(" as ")
        path = path.strip().removesuffix("::self")
        if path.endswith("::*"):
            yield path, path
        elif head.strip():
            yield alias.strip() or path.rpartition("::")[2], path
        return
    depth = 0
    start = 0
    group = group.rpartition("}")[0] + ","
    for column, char in enumerate(group):
        depth += (char == "{") - (char == "}")
        if char == "," and depth == 0:
            yield from _use_leaves(group[start:column], prefix + head.strip())
            start = column + 1


# An optional visibility: pub, pub(crate), pub(super), pub(in path)
_RUST_VIS = rf"(?:pub(?:\([^)]*\))?{_RUST_GAP}+)?"


# `extern crate name as alias;`
_RUST_EXTERN_CRATE_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?extern\s+crate\s+(\w+)\s+as\s+(\w+)\s*;",
    re.MULTILINE,
)


# A `use` declaration: `after` the `{` or `;` it follows on its line, its
# `vis`ibility and its `tree`, which may span lines
_RUST_USE_RE = re.compile(
    rf"(?:^|(?P<after>[{{;]))[{_RUST_SPACE}]*(?P<vis>{_RUST_VIS})use{_RUST_GAP}+"
    rf"(?P<tree>(?:[^;\s]|{_RUST_GAP})+);",
    re.MULTILINE,
)


def _rust_visibility(written):
    """The visibility ``written`` before an item: "pub", what `pub(...)`
    holds (`crate`, `super`, `self`, `in path`), "" for none."""
    held = re.match(r"\s*pub(?:\s*\(([^)]*)\))?", written)
    if not held:
        return ""
    return " ".join(held[1].split()) if held[1] else "pub"


# A type an item declares, and its name
_RUST_TYPE_DECLARATION_RE = re.compile(
    rf"^[{_RUST_SPACE}]*{_RUST_VIS}(?:struct|enum|union|trait|type)\s+(\w+)",
    re.MULTILINE,
)


# A closure trait a parameter is declared with: `impl Fn(A, B)`, `&dyn FnMut(A)`
_RUST_FN_TYPE_RE = re.compile(r"\b(?:impl|dyn)\s+(?:Fn|FnMut|FnOnce)\s*\(([^()]*)\)")


# A type parameter bound by a closure trait: `F: Fn(A) -> R`
_RUST_FN_BOUND_RE = re.compile(r"\b(\w+)\s*:\s*((?:Fn|FnMut|FnOnce)\s*\([^()]*\))")


# A parameter name in a `fn` signature
_RUST_PARAMETER_RE = re.compile(r"[(,]\s*(?:mut\s+)?([a-z_]\w*)\s*:")


# A `fn` with its body on the line of the `impl` block it is in
_RUST_INLINE_FN_RE = re.compile(
    r"\bfn\s+(?P<name>\w+)[^{;]*?(?:->(?P<returns>[^{;]+))?\{(?P<body>[^{}]*)\}"
)


# Where an `impl` block's header starts
_RUST_IMPL_RE = re.compile(r"^\s*(?:unsafe\s+)?impl(?=[<\s]|$)")


# A bracketed group that holds no other
_RUST_GROUP_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|<[^<>]*>")


# What an `impl` header is cut at, and the brackets that nest
_RUST_HEADER_RE = re.compile(r"->|[<(\[]|[>)\]]|\bfor\b|\bwhere\b|[{;]")


def _rust_named(text):
    """The path a type or trait is written with, behind `&`, `dyn` and the
    like; None where it is not written with a path, as a macro call is not."""
    text = _RUST_BORROW_RE.sub("", text)
    named = re.match(r"(?:\w*::)*(?!fn\b)\w+", text)
    called = named and text[named.end() :].lstrip().startswith("!")
    return named[0] if named and not called else None


def _last_segment(path):
    """The last segment of ``path``, None for None."""
    return path and path.rpartition("::")[2]


def _rust_impl(text):
    """What ``text``, an `impl` header read from after `impl`, says: the
    trait, the owner of the block's items, the bounds of a blanket `impl`,
    the path its type is written with, and the type arguments it is written
    with, None for a parameter of the block and `_` for one in another
    type: `impl W<Own>` gives `["Own"]`, `impl<T> W<T>` `[None]`, `impl<T>
    W<Vec<T>>` `["Vec<_>"]`.

    The trait is given by its path as written, the owner by the last
    segment of its type's: `<'de> de::Deserialize<'de> for Wrap<T> {` gives
    `("de::Deserialize", "Wrap", None)`. An inherent `impl` has no trait. A
    type source cannot name, as a tuple or a parameter of the block, gives
    an owner no source can name either: `<Trait for (A, B)>`. Where the type
    is a parameter of the block, or a reference to one, the block is for
    every type that meets the parameter's bounds, which are given by name;
    otherwise the bounds are None, and the path None for such a type."""
    # The text is cut where no bracket is open: after the block's own
    # parameters, at `for`, and at `where` or the `{` that ends the header.
    depth, cuts = 0, {}
    for found in _RUST_HEADER_RE.finditer(text):
        token = found[0]
        if token in "<([":
            depth += 1
        elif token in ">)]":
            depth -= 1
            if not depth:
                cuts.setdefault("<", found.end())
        elif token != "->" and not depth:
            if token in "{;":
                cuts["{"] = found.start()
                break
            cuts.setdefault(token, found.start())
    opened = cuts.get("<", 0) if text.lstrip().startswith("<") else 0
    end = cuts.get("{", len(text))
    constrained = cuts.get("where", end)
    named = text[opened:constrained]
    trait = None
    if "for" in cuts and opened <= cuts["for"] < constrained:
        trait = _rust_named(text[opened : cuts["for"]])
        named = text[cuts["for"] + 3 : constrained]
    # Each parameter and each `where` predicate, without what it nests.
    declared = f"{text[:opened].strip()[1:-1]},{text[constrained:end][5:]}"
    while (reduced := _RUST_GROUP_RE.sub("", declared)) != declared:
        declared = reduced
    parameters = dict(
        re.findall(r"(?:^|,)\s*(?:const\s+)?(\w+)\s*(?::([^,]*))?", declared)
    )
    path = _rust_named(named)
    owner = _last_segment(path)
    if owner not in parameters:
        bounds = None
    else:
        bounds = []
        for part in declared.split(","):
            name, _, bound = part.partition(":")
            if name.strip() == owner:
                # `for<'a> Trait<'a>` is a bound by `Trait`.
                traits = re.sub(r"\bfor\b", "", bound).split("+")
                bounds += filter(None, map(_last_segment, map(_rust_named, traits)))
    if not owner or bounds is not None:
        if not trait:
            return None, None, None, None, []
        written = re.sub(r"(?:\w*::)+", "", " ".join(named.split()))
        return trait, f"<{_last_segment(trait)} for {written}>", bounds, None, []
    given = named.strip()
    given = _bracketed_parts(given[given.index("<") + 1 :]) if "<" in given else []
    # A parameter of the block is any type: written alone it fixes none, in
    # another type it fixes that type, of any arguments.
    held = rf"\b(?:{'|'.join(parameters)})\b" if parameters else r"(?!)"
    arguments = [
        None if a in parameters else re.sub(held, "_", a) for a in given if a[:1] != "'"
    ]
    return trait, owner, bounds, path, arguments


# What a Rust item that holds others opens its body after, and an item a
# line may go on to after the end of another
_RUST_TYPE_HEAD_RE = re.compile(
    rf"\s*(?:{_RUST_VIS}\s*)?(?:unsafe\s+)?(?:impl|trait|mod)\b"
)


_RUST_RECEIVER_RE = re.compile(
    r"((?:&\s*(?:'\w+\s+)?)?(?:mut\s+)?self)(?:\s*:.*)?", re.DOTALL
)


_RUST_PATH_ARGUMENT_RE = re.compile(r"[A-Za-z_]\w*(?:::\w+)+")


_RUST_MACRO_RULES_RE = re.compile(r"\s*macro_rules!\s*(\w+)")


_RUST_DEFINITION_HEAD_RE = re.compile(
    rf"(?:{_RUST_VIS}\s*)?(?:(?:const|async|unsafe|extern(?:\s+\"[^\"]*\")?)\s+)*"
    r"(?:fn|impl|struct|enum|union|trait|mod|type|static|const|macro_rules!)\b"
)


class RustParser(BaseParser):
    language = "rust"
    head_scopes_else = False
    call_fields = True
    chained_values = True
    binding_re = re.compile(r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*=(?!=)")
    bindings = staticmethod(_rust_bindings)

    def _pattern_head(self, statement):
        """As the base's, and up to a type a struct pattern or literal
        names, whose braces open no block."""
        return super()._pattern_head(statement) or (
            _RUST_PATTERN_HEAD_RE.search(statement)
            and not _RUST_CONDITION_RE.match(statement)
        )

    def add_call(self, source_id, callee, lineno, since, start, paren, origin=None):
        """As the base's, and keep the call's edge by the offset in the file
        of the `)` that ends it, where the line does."""
        super().add_call(source_id, callee, lineno, since, start, paren, origin)
        if paren in self._pairs:
            offset = self._blocks[1][lineno - 1] + self._pairs[paren]
            self._called_at[offset] = len(self.builder.edges) - 1

    def _expression_origin(self, code, start, end, lineno, since, owner):
        """As ``_value_of`` reads it, at columns ``start`` to ``end`` of
        ``code``, line ``lineno``; None for a value with a call the line
        has yet to read, which ``_settle_calls`` reads once it has."""
        if lineno == self._line and not self._pairs and "(" in code[start:end]:
            return None
        line = self._blocks[1][lineno - 1]
        return self._value_of(line + start, line + end, since, owner)

    def _settled_origin(self, code, start, stop, lineno):
        """As ``_value_of`` reads the value, whatever it is."""
        owner = self._owning_type(self._scope)
        return self._expression_origin(code, start, stop, lineno, self._since, owner)

    def _span_origin(self, start, stop):
        """As ``_value_of`` reads it."""
        return self._value_of(start, stop, self._since, self._owning_type(self._scope))

    def _value_of(self, start, end, since, owner):
        """Origin path of the value of the Rust expression at offsets
        ``start`` to ``end`` of the file's code, read once the calls in it
        are: a name, a path or a field, a call or a chain of them, a struct
        literal, a block's last expression (`unsafe`, labelled, a `loop`'s
        by its `break`s), an `if` or a `match` by what the arms that do not
        diverge give, behind `&`, `*`, `?` or `as`; None where it does not
        tell."""
        text, starts, closer, _ = self._blocks
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        written = text[start:end]
        if not written:
            return None
        if written.endswith("?"):
            held = self._value_of(start, end - 1, since, owner)
            return held and f"{held}.?"
        cast = re.search(r"\sas\s+([A-Za-z_][\w:]*)$", written)
        if cast and " as " in _flat(written):
            return f"{cast[1]}()"
        lead = re.match(r"(?:&\s*(?:mut\s+)?|\*\s*)+|unsafe\s*(?=\{)", written)
        if lead:
            return self._value_of(start + lead.end(), end, since, owner)
        looped = re.match(r"(?:'(\w+):\s*)?loop\s*\{", written)
        if looped:
            opened = start + looped.end()
            spans = self._breaks(opened, end - 1, looped[1], True)
            return self._agreed(spans, since, owner)
        labelled = re.match(r"'(\w+):\s*(?=\{)", written)
        if labelled:
            opened = start + labelled.end()
            spans = [(opened, end), *self._breaks(opened, end, labelled[1], False)]
            return self._agreed(spans, since, owner)
        if written[0] == "{" and closer.get(start) == end - 1:
            return self._block_value(start, end, since, owner)
        if re.match(r"if\b", written):
            return self._agreed(self._if_arms(start, end), since, owner)
        if re.match(r"match\b", written):
            return self._agreed(self._match_arms(start, end), since, owner)
        fields = re.search(r"\)((?:\.[A-Za-z_]\w*)*)$", written)
        if fields:
            # A call, or a field of what one returns, by its edge
            edge = self._called_at.get(start + fields.start())
            return None if edge is None else f"#{edge}{fields[1]}"
        found = _RUST_STRUCT_LITERAL_RE.match(written)
        if found and closer.get(start + found.end() - 1) == end - 1:
            return self._literal_origin(start, found, end, since, owner)
        if "\n" in written:
            return None
        lineno = bisect_right(starts, start)
        code = text[starts[lineno - 1] :].split("\n", 1)[0]
        column = start - starts[lineno - 1]
        sequence = _RUST_SEQUENCE_RE.fullmatch(code, column, column + len(written))
        if sequence:
            return self._sequence_origin(code, sequence, lineno, since, owner)
        return super()._expression_origin(
            code, column, column + len(written), lineno, since, owner
        ) or self._item_origin(written, lineno, column, since)

    def _top_level(self, start, end, chars):
        """The offsets of ``chars`` between ``start`` and ``end`` of the
        file's code that no bracket opened there holds."""
        text, _, closer, _ = self._blocks
        at = start
        while at < end:
            if text[at] in chars:
                yield at
            at = closer.get(at, at) + 1

    def _block_value(self, start, end, since, owner):
        """A block's value at ``start`` to ``end``: its last expression's,
        past the last statement a `;` ends, or a block-like one (`if`,
        `match`, a loop) whose `}` ends its line; none where a `;` ends it."""
        text, _, closer, _ = self._blocks
        at = last = start + 1
        while at < end - 1:
            if text[at] == ";":
                last = at + 1
            elif at in closer:
                at = closer[at]
                if text[at] == "}" and _RUST_STATEMENT_AFTER_RE.match(text, at + 1):
                    last = at + 1
            at += 1
        return self._value_of(last, end - 1, since, owner)

    def _if_arms(self, start, end):
        """The spans of the blocks of the `if` at ``start`` to ``end``, its
        `else if`s and its `else`; None where it has no `else`, whose value
        is `()`."""
        text, _, closer, _ = self._blocks
        arms, at = [], start
        while True:
            opened = next(self._top_level(at, end, "{"), None)
            if opened is None:
                return None
            arms.append((opened, closer[opened] + 1))
            following = re.match(r"\s*else\b\s*", text[closer[opened] + 1 : end])
            if not following:
                return None
            at = closer[opened] + 1 + following.end()
            if text[at : at + 1] == "{":
                return [*arms, (at, closer[at] + 1)]
            if not re.match(r"if\b", text[at:end]):
                return None

    def _match_arms(self, start, end):
        """The spans of the bodies of the arms of the `match` at ``start``
        to ``end``."""
        text, _, closer, _ = self._blocks
        opened = next(self._top_level(start, end, "{"), None)
        if opened is None:
            return None
        bodies, at, closed = [], opened + 1, closer[opened]
        while True:
            arrow = next(
                (a for a in self._top_level(at, closed, "=") if text[a + 1] == ">"),
                None,
            )
            if arrow is None:
                return bodies
            body = arrow + 2
            while text[body].isspace():
                body += 1
            if text[body] == "{":
                stop = closer[body] + 1
            else:
                stop = next(self._top_level(body, closed, ","), closed)
            bodies.append((body, stop))
            at = stop + 1

    def _breaks(self, start, end, label, looping):
        """The spans of the values the `break`s between ``start`` and
        ``end`` give what is around them, labelled ``label``, a loop where
        ``looping``: one naming the label from any depth, one naming none
        where no loop inside holds it."""
        text, _, closer, _ = self._blocks
        inner = [
            (start + found.end() - 1, closer.get(start + found.end() - 1, start))
            for found in re.finditer(
                r"\b(?:loop\s*|while\b[^{;]*|for\b[^{;]*)\{", text[start:end]
            )
        ]
        spans = []
        for found in re.finditer(r"\bbreak\b(?:\s*'(\w+))?", text[start:end]):
            at = start + found.start()
            if (
                found[1] != label
                if found[1]
                else (not looping or any(o < at < c for o, c in inner))
            ):
                continue
            value = start + found.end()
            stop = next(self._top_level(value, end, ";}"), end)
            spans.append((value, stop))
        return spans

    def _diverges(self, start, end):
        """Whether the expression at ``start`` to ``end`` never gives a
        value: a `return`, `continue`, `break` or panicking macro, or a
        block whose last statement is one."""
        text, _, closer, _ = self._blocks
        written = text[start:end].strip()
        if written.startswith("{") and closer.get(start + text[start:end].index("{")):
            opened = start + text[start:end].index("{")
            ends = [opened, *self._top_level(opened + 1, end - 1, ";")]
            parts = [
                text[a + 1 : b].strip() for a, b in zip(ends, [*ends[1:], end - 1])
            ]
            written = next((part for part in reversed(parts) if part), "")
        return bool(_RUST_DIVERGES_RE.match(written))

    def _agreed(self, spans, since, owner):
        """The value the expressions at ``spans`` give, those that diverge
        aside: one origin where they agree, else `%N`, the entry of
        ``builder.literals`` that holds each, which the resolver reads as
        the value they all give; None where one is not known."""
        values = [
            self._value_of(s, e, since, owner)
            for s, e in spans or ()
            if not self._diverges(s, e)
        ]
        if not values or None in values:
            return None
        if len(set(values)) == 1:
            return values[0]
        key = f"%{self.builder.serial(self.builder.literals)}"
        self.builder.literals[key] = sorted(set(values))
        return key

    def _literal_origin(self, start, found, end, since, owner):
        """`Type`, `Type<A>` or `Type<{N}>` of the struct literal at
        ``start`` to ``end`` whose head ``found`` matches: `{N}` the entry of
        ``builder.literals`` that holds its fields' values by name; a struct
        update, `..base`, is of its base's type."""
        text, _, _, _ = self._blocks
        given = (found["turbofish"] or "").removeprefix("::")
        held, at = {}, start + found.end()
        for comma in [*self._top_level(at, end - 1, ","), end - 1]:
            part = text[at:comma]
            named = re.match(r"\s*(\w+)\s*:(?!:)", part)
            spread = re.match(r"\s*\.\.", part)
            value = at + (named or spread).end() if named or spread else at
            origin = self._value_of(value, comma, since, owner)
            key = ".." if spread else named[1] if named else part.strip()
            if origin and (spread or re.fullmatch(r"\w+", key)):
                held[key] = origin
            at = comma + 1
        if ".." in held:
            # A struct update is of its base's type.
            return held[".."]
        if held and not given:
            key = f"{{{self.builder.serial(self.builder.literals)}}}"
            self.builder.literals[key] = held
            given = f"<{key}>"
        return f"{found['type']}{given}()"

    def _closed_value(self, code):
        """The value whose `}` ends ``code``, the line being read, written
        from a line above."""
        ended = code.rstrip()
        closed = self._blocks[1][self._line - 1] + len(ended) - 1
        held = self._scope and ended.endswith("}") and self._value_closed_at(closed)
        return held and held[1] < self._blocks[1][self._line - 1] and held[0]

    def _value_closed_at(self, closed):
        """``(value, offset of its first line)`` of the expression whose
        last bracket is the `}` at offset ``closed``: the struct literal,
        the `match`, `if`, `unsafe`, `loop` or labelled block its `{` opens,
        or a bare block where an expression starts; None for any other, the
        last arm of an `if` among them."""
        text, _, closer, _ = self._blocks
        if self._openers is None:
            self._openers = {c: o for o, c in closer.items()}
        opened = self._openers.get(closed)
        if opened is None:
            return None
        line = text.rfind("\n", 0, opened) + 1
        before = text[line:opened]
        literal = re.search(
            rf"{_RUST_STRUCT_LITERAL_RE.pattern}$", text[line : opened + 1]
        )
        keyword = list(_RUST_BLOCK_HEAD_RE.finditer(before))
        if literal:
            start = line + literal.start()
        elif keyword:
            start = line + keyword[-1].start()
        elif _RUST_EXPRESSION_START_RE.search(before):
            start = opened
        else:
            return None
        owner = self._owning_type(self._scope)
        held = self._value_of(start, closed + 1, self._since, owner)
        return held and (held, line)

    def _path_origin(self, path, offset):
        """The origin of the value the path ``path``, written at ``offset``
        of the code, names: `!path`, its first name read through a `use` in
        a block around it that binds that name, the innermost first; None
        where such a block's glob `use` may bind it."""
        root, sep, rest = path.partition("::")
        blocks = sorted(self._block_uses, key=lambda used: used[0][1] - used[0][0])
        for (start, end), names in blocks:
            if not start <= offset <= end:
                continue
            if root in names:
                return f"!{names[root]}{sep}{rest}"
            if any(name.endswith("*") for name in names):
                return None
        return f"!{path}"

    def _element_origin(self, end, callee, lineno, since, source_id):
        """Origin path of a call on an element, `items[i].callee()`: of what
        indexing what `items` holds gives, what it holds itself for a range,
        `items[1..]`."""
        code, at = self._code, self._index_open(end)
        if at is None:
            return None
        before = code[:at].rstrip()
        called = len(before.rstrip("?"))
        if code[at - 1 : at] == "]":
            # An index of what an index gives
            held = self._element_origin(at, "x", lineno, since, source_id)
        elif (edge := self._closes.get(called - 1)) is not None:
            # An index of what a call returns, through its `?`s
            held = f"#{edge}{'.?' * (len(before) - called)}.x"
        elif not before and self._tail is not None:
            # An index alone on its line, of what the call the line before
            # ends with returns
            unwraps = ".?" * self._tail_unwraps
            held = f"#{self._tail}{self._tail_fields}{unwraps}.x"
        else:
            path = re.search(r"(?<![\w.])[A-Za-z_]\w*(?:\.\w+)*$", code[:at])
            path = path[0] if path else self._continued_path(code[:at])
            if not path and not code[:at].strip() and self._tail is None:
                # An index alone on its line, of the name the line before ends with
                path = self._tail_name
            owner = self._owning_type(source_id)
            held = path and self.call_origin(f"{path}.x", lineno, since, owner)
        # A range gives a slice of what is indexed, of its element still.
        step = "" if ".." in _flat(code[at + 1 : end - 1]) else ".[]"
        return held and f"{held[:-2]}{step}.{callee}"

    def _item_origin(self, text, lineno, start, since):
        """`!path` for a path to an item, `Kind::A` or `helper`; a name a
        binding in scope holds, of a type not known, is no item."""
        local = "::" not in text and self._seen(text, lineno, since)
        path = re.fullmatch(r"[A-Za-z_]\w*(?:::\w+)*", text)
        if path and text not in ("true", "false") and not local:
            return self._path_origin(text, self._blocks[1][lineno - 1] + start)
        return None

    def _receiver_origin(self, code, start, callee, lineno, since):
        """The origin of the call ``callee``, written at column ``start`` of
        line ``lineno``: of a method on a value written before its `.`, a
        path to a unit struct or variant, `Kind::A.tick`, or a struct
        literal, `Point { x: 1 }.norm`, a name alone read as names are; of a
        name bound to an item, `let make = Kind::Held; make(1)`, the item."""
        self._read_at = start
        called = re.fullmatch(r"\w+", callee) and not code[:start].rstrip().endswith(
            "."
        )
        held = called and self._declared(callee, lineno, since)
        self._read_at = None
        if held and held.startswith("!"):
            return held
        head, dot, name = callee.rpartition(".")
        offset = self._blocks[1][lineno - 1] + start
        if dot and "::" in head and _RUST_VALUE_PATH_RE.fullmatch(head):
            held = self._path_origin(head, offset)
            return held and f"{held}.{name}"
        before = code[:start].rstrip()
        if dot or not before.endswith("}."):
            return None
        held = self._value_closed_at(self._blocks[1][lineno - 1] + len(before) - 2)
        return held and f"{held[0]}.{callee}"

    def _operand(self, text):
        text = re.sub(r"^(?:[&*]\s*(?:mut\s+)?)+", "", text)
        plain = text.rstrip("?")
        return plain, len(text) - len(plain)

    def _item_reaches(self, name, declaration, lineno):
        """Whether an item declared in a function's body, a `const` or a
        `static`, is seen where line ``lineno`` is read: anywhere in that
        body, nested items too; one outside every function is the file's."""
        text, starts, _, spans = self._blocks
        held = _innermost(spans, starts[declaration[0] - 1] + (declaration[3] or 0))
        while held:
            start = text.rfind("\n", 0, held[0]) + 1
            if re.search(r"\bfn\b", text[start : held[0]]):
                read = starts[lineno - 1] + (self._read_at or 0)
                return held[0] <= read <= held[1]
            held = _innermost(spans, held[0])
        return True

    def _scope_end(self, name, declaration):
        """As the base's, but a closure's parameter is in scope to the end
        of the closure's body."""
        text, starts = self._blocks[:2]
        line = starts[declaration[0] - 1]
        at = line + self._column(name, declaration)
        for bars in _RUST_BINDER_RES[0].finditer(text, line, starts[declaration[0]]):
            if bars.start(1) <= at < bars.end(1):
                return self._statement_close(bars.end(), orelse=False)
        return super()._scope_end(name, declaration)

    def _pattern_origins(self, origin, pattern):
        """``{name: origin path}`` of each name ``pattern`` binds in a value
        of the origin ``origin``."""
        return {
            name: ".".join([origin, *steps]) for name, steps in self.bindings(pattern)
        }

    def _bind_arm(self, code, index):
        """Type the names that an arm of a `match` binds on line ``index``,
        from the value matched, up to the end of the arm."""
        indent = len(code) - len(code.lstrip())
        while self._matches and code.strip() and indent <= self._matches[-1][0]:
            self._matches.pop()
        found = self._matches and "=>" in code and _RUST_ARM_RE.search(code)
        if found and self._matches[-1][1]:
            bindings = self._pattern_origins(self._matches[-1][1], found["pattern"])
            at, block = found.start("pattern"), code.rstrip().endswith("{")
            self._bind(bindings, index, at, block)

    def _sequence_origin(self, code, sequence, lineno, since, owner):
        """Origin path of the literal ``sequence``, a match of
        ``_RUST_SEQUENCE_RE`` on ``code``: a `Vec` of what its elements hold,
        a path that is no variable the value it names, as `let` reads it."""
        key, at, held = (
            f"[{self.builder.serial(self.builder.literals)}]",
            sequence.start(1),
            [],
        )
        for part in _bracketed_parts(sequence[1] + "]", turbofish=True):
            # `[value; count]` holds the value
            element = part.partition(";")[0].strip()
            at = code.index(element, at)
            origin = self._expression_origin(
                code, at, at + len(element), lineno, since, owner
            )
            if not origin and _RUST_VALUE_PATH_RE.fullmatch(element):
                origin = self._path_origin(element, self._blocks[1][lineno - 1] + at)
            held.append(origin or None)
            at += len(element)
        self.builder.literals[key] = held
        return f"Vec<{key}>()"

    def _bind_patterns(self, code, index, since, owner):
        """Type the names that a `for`, an `if let`, a `while let` or a
        `let` with a pattern binds on line ``index``, from the value the
        pattern is matched against, and note the value of a `match`."""
        lineno = index + 1
        found = None
        if "for " in code:
            found = _RUST_FOR_RE.search(code)
            head = not found and _RUST_FOR_HEAD_RE.search(code)
            self._for_head = head and head["pattern"]
        elif (
            self._for_head and (head := _closing([code], 0, 0, "{")) and head[2] == "{"
        ):
            opened = head[1]
            # A head over several lines: its value ends where its block opens,
            # with the call the line before ended where nothing precedes.
            start = len(code) - len(code.lstrip())
            origin = (
                self._expression_origin(code, start, opened, lineno, since, owner)
                if code[start:opened].strip()
                else self._tail is not None and f"#{self._tail}"
            )
            bindings = (
                self._pattern_origins(f"{origin}.*", self._for_head) if origin else {}
            )
            self._bind(bindings, index, opened, True, opened)
            self._for_head = None
        if not found and "let " in code:
            found = _RUST_IF_LET_RE.search(code)
        if found:
            origin = self._expression_origin(
                code, *found.span("value"), lineno, since, owner
            )
            if origin and found.re is _RUST_FOR_RE:
                origin += ".*"
            bindings = self._pattern_origins(origin, found["pattern"]) if origin else {}
            # In the block the head's own `{` opens, past a closure's
            self._bind(bindings, index, found.start("pattern"), True, found.end() - 1)
        if "match " in code and (found := _RUST_MATCH_RE.search(code)):
            span = found.span("value")
            origin = self._expression_origin(code, *span, lineno, since, owner)
            indent = len(code) - len(code.lstrip())
            self._matches.append((indent, origin))
            # The arms the line goes on with, each binding from its pattern
            at = found.end()
            arms = _bracketed_parts(code[at:], expression=True, turbofish=True)
            for arm in arms if origin else ():
                at = code.index(arm, at)
                if written := _RUST_ARM_RE.match(arm):
                    bindings = self._pattern_origins(origin, written["pattern"])
                    self._bind(bindings, index, at + written.start("pattern"))
                at += len(arm)

    def _bind_let(self, code, found, lineno, since, owner):
        """Type the names that ``found``, a `let` of ``code``, line
        ``lineno``, with a pattern, binds from its value, past the pattern."""
        if re.match(r"(?:mut\s+)?\w+\s*:", found["pattern"]):
            return
        origin = self._expression_origin(
            code, *found.span("value"), lineno, since, owner
        )
        at = found.start("pattern")
        for name, bound in (
            self._pattern_origins(origin, found["pattern"]) if origin else {}
        ).items():
            self._rebind(name, lineno, self._named_from(name, at), bound)

    def _generic_path_origin(self, callee, lineno):
        """Origin of a path, read on line ``lineno``, that starts at a type
        parameter: `T::name` is a function of the bound of `T`,
        `T::Assoc::name` one of a type that only an implementation of the
        bound names."""
        root, *path = callee.split("::")
        bounds = path and self._bounds(root, lineno)
        if not bounds:
            return None
        if len(path) == 1 and len(bounds) == 1:
            return f"{bounds.pop()}.{path[0]}"
        return f"?.{path[-1]}"

    def _keep_visibility(self, node_id, line):
        """Keep the visibility of the item ``node_id`` declared on ``line``
        where it is less than `pub`."""
        held = _rust_visibility(line)
        if held != "pub":
            self.builder.rust_visibility[node_id] = held

    def parse(self, file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)
        self._called_at = {}  # offset of a call's `)` in the file -> its edge
        self._openers = None  # offset of a `}` in the file -> its `{`'s

        lines = source_lines(content)
        lines = self.split_source(
            lines,
            self.blank_lines(lines, ('"', "r")),
            _RUST_DEFINITION_HEAD_RE,
            _RUST_TYPE_HEAD_RE,
            statement=_STATEMENT_START_RE,
            attributes=True,
        )
        current_context = [file_id]
        context_indent = [0]  # Stack of indentation levels

        re_fn_start = re.compile(
            rf"^\s*{_RUST_VIS}(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
            r"(?:extern\s+\"[^\"]*\"\s+)?fn\s+([a-zA-Z0-9_]+)"
        )
        re_struct = re.compile(rf"^\s*{_RUST_VIS}(?:struct|trait)\s+([a-zA-Z0-9_]+)")
        re_enum = re.compile(rf"^\s*{_RUST_VIS}enum\s+([a-zA-Z0-9_]+)")
        # Enum variant patterns: Name, Name(Type), Name { field: Type }, Name = 1
        re_enum_variant = re.compile(r"^\s*([A-Z][a-zA-Z0-9_]*)\s*(?:[({\,=]|$)")
        re_mod = re.compile(rf"^\s*{_RUST_VIS}mod\s+([a-zA-Z0-9_]+)\s*\{{?")
        re_const = re.compile(
            r"^\s*(?:pub(?:\(.*?\))?\s+)?(?:const|static)\s+([A-Z0-9_]+)\s*:\s*.*?\s*=\s*(.*?);"
        )
        re_call = re.compile(rf"((?:{_RUST_TURBOFISH}|[a-zA-Z0-9_:.<>])+)\(")
        self.scan_declarations(lines, _rust_declarations(), file_id)
        # A name a pattern binds has, from there on, no declared type.
        for lineno, line in enumerate(lines, 1):
            for binder in _RUST_BINDER_RES:
                for pattern in binder.finditer(line):
                    names = re.compile(r"\b[a-z_]\w*\b").finditer(
                        line, *pattern.span(1)
                    )
                    for name in names:
                        self.declared.setdefault(name[0], []).append(
                            (lineno, None, False, name.start())
                        )
        aliases = _TYPE_ALIAS_RE.findall(content)
        self.builder.type_aliases.update(aliases)
        if renamed := {
            alias: name.replace("-", "_")
            for name, alias in _RUST_EXTERN_CRATE_RE.findall(content)
        }:
            self.builder.rust_externs[file_id] = renamed
        self.builder.rust_aliases[file_id] = aliases
        # The names each `use` binds, by the line of ``lines`` it starts on;
        # in ``uses_after`` one behind a `{` or `;` on its line
        uses_at, uses_after, at, counted = {}, {}, 1, 0
        # (line, name) -> the visibility of the `use` binding it there
        use_visibility = {}
        split = "\n".join(lines)
        for found in _RUST_USE_RE.finditer(split):
            at += split.count("\n", counted, found.start("tree"))
            counted = found.start("tree")
            leaves = dict(_use_leaves(_spaced(found["tree"], _RUST_GAP)))
            (uses_after if found["after"] else uses_at).setdefault(at, {}).update(
                leaves
            )
            for name in leaves:
                use_visibility[at, name] = _rust_visibility(found["vis"])

        def bind_uses(scope_id, uses, line):
            """Bind the names ``uses``, written on ``line``, imports in
            ``scope_id``; one bound already to another path is a `#[cfg]`
            alternative. One not `pub` keeps its visibility."""
            held = self.builder.imports.setdefault(scope_id, {})
            for name, path in uses.items():
                if use_visibility[line, name] != "pub":
                    self.builder.rust_use_visibility[scope_id, name] = use_visibility[
                        line, name
                    ]
                if held.get(name, path) != path:
                    paths = self.builder.rust_use_alternatives.setdefault(
                        (scope_id, name), set()
                    )
                    paths.update((held[name], path))
            held.update(uses)

        # The owner of the open `impl` block's items, and its type's name
        impl_id = current_impl_target = impl_bounds = impl_args = None
        trait_generic = False  # whether the open impl's trait takes type arguments
        impl_borrowed = False  # whether the open impl is for a reference, `for &T`
        current_trait_impl = None
        current_enum_id = None
        enum_depth = 0  # brackets open in the enum's body; a variant is at 0
        # id of each impl block's owner -> (name, lineno, bounds of a blanket one)
        impl_targets = {}
        fn_start = 0  # first line of the function being read
        fn_ids = set()  # functions whose body is a context; a `fn` in one is nested
        modules = {file_id}  # the file and its inline modules
        # (last line index, functions) of the `macro_rules!` being read
        macro = None
        self._matches = []  # (indentation, origin of the value) of each open `match`
        self._for_head = None  # the pattern of a `for` whose head goes on below
        # ("arm", paren, bracket, brace) or ("matches", paren): a pattern
        # that spans lines, carried from the line above.
        self._pattern_cont = None

        def opens_block(text):
            """True if ``text`` leaves a `{` open; `struct P { x: i32 }` does not."""
            return text.count("{") > text.count("}")

        # Each line without comments and literal text, and whether it lies
        # partly inside a comment or literal that spans lines.
        codes = []
        inside = None
        for lineno, raw in enumerate(lines, 1):
            code, ends_inside = blank_strings(raw, inside, ('"', "r"))
            codes.append((code, bool(inside or ends_inside)))
            # What a `let` binds to a literal, and a constant, have a type.
            value, item = None, False
            if match := _RUST_VEC_LET_RE.search(code):
                value = "Vec<>()"
            elif match := _RUST_STRING_LET_RE.search(raw):
                # The literal is all of the value where `;` ends it.
                if not inside and code[match.end() - 1 :].strip() == ";":
                    value = "[u8]()" if match["bytes"] else "str()"
            elif match := _RUST_CONST_RE.search(code):
                value, item = f"{_RUST_BORROW_RE.sub('', match['type'])}()", True
            elif match := _RUST_INDEX_LET_RE.search(code):
                # `let x = xs[i]` gives `x` what indexing `xs` gives, and a
                # range what `xs` holds itself.
                held = [
                    declared
                    for at, declared, _, _ in self.declared.get(match["of"], ())
                    if at <= lineno and declared
                ]
                closed = _closing([code], 0, match.end())
                index = code[match.end() : closed[1]] if closed else ""
                if held:
                    value = held[-1] if ".." in _flat(index) else f"{held[-1]}.[]"
            if value:
                self.declared.setdefault(match["name"], []).append(
                    (lineno, value, item, match.start("name"))
                )
            inside = ends_inside
        self.read_blocks([code for code, _ in codes])
        text, _, _, spans = self._blocks

        def module_of(at):
            """The ID of the module whose body holds offset ``at`` of the
            code, the file's outside any; None inside any other block."""
            path, held = [], _innermost(spans, at)
            while held:
                start = text.rfind("\n", 0, held[0]) + 1
                named = re_mod.match(text[start : held[0] + 1])
                if not named:
                    return None
                path.insert(0, named[1])
                held = _innermost(spans, held[0])
            return "::".join([file_id, *path])

        # The types each module of the file declares, wherever in it, and
        # the names it imports
        module_types, module_uses = {}, {}
        for found in _RUST_TYPE_DECLARATION_RE.finditer(text):
            module_types.setdefault(module_of(found.start()), set()).add(found[1])
        self._block_uses = []  # (block, names) of each `use` in a block
        for found in _RUST_USE_RE.finditer(text):
            module = module_of(found.start("tree"))
            leaves = dict(_use_leaves(_spaced(found["tree"], _RUST_GAP)))
            module_uses.setdefault(module, {}).update(leaves)
            if module is None:
                block = _innermost(spans, found.start("tree"))
                self._block_uses.append((block, leaves))

        def seen_type(module, name):
            """The ID of the type ``name`` names in ``module``: one a module of
            the file declares, which it is or it imports from by `self::` or
            `super::`; None for any other."""
            *steps, name = module_uses.get(module, {}).get(name, name).split("::")
            while steps[:1] in (["self"], ["super"]):
                if steps.pop(0) == "super":
                    module = module.rpartition("::")[0]
            owner = "::".join([module, *steps])
            return f"{owner}::{name}" if name in module_types.get(owner, ()) else None

        def attributes_from(index):
            """Index of the first line of the attributes above line ``index``."""
            while index and codes[index - 1][0].rstrip()[-1:] not in (";", "{", "}"):
                index -= 1
            return index

        def statement_at(index):
            """The code from line ``index`` on to the `;` that ends it."""
            text, end = codes[index][0], index
            while ";" not in text and end + 1 < len(codes):
                end += 1
                text += f" {codes[end][0]}"
            return text

        def add_variant(enum_id, text, lineno):
            """Add the variant of ``enum_id`` that ``text``, read on line
            ``lineno``, declares."""
            match = re_enum_variant.match(text)
            if not match:
                return
            variant_id = f"{enum_id}::{match[1]}"
            self.builder.add_node(
                variant_id,
                "Function",
                name=match[1],
                signature=f"{enum_id.split('::')[-1]}::{match[1]}",
                lineno=lineno,
                docstring="",
                **_unit(text[match.end(1) :], r"[,=}]|$"),
            )
            self.builder.add_edge(enum_id, variant_id, "DEFINES")
            if held := _rust_held(text[match.end(1) :]):
                self.builder.fields[variant_id] = held

        def add_derived(type_id, index):
            """Record the standard traits that a `#[derive(...)]` above line
            ``index`` implements for ``type_id``."""
            attributes = " ".join(
                code for code, _ in codes[attributes_from(index) : index]
            )
            scope = _stdlib.load("rust")["scope"]
            self.builder.derived[type_id] = [
                scope[trait]
                for derives in _RUST_DERIVE_RE.findall(attributes)
                for trait in re.findall(r"(\w+)\s*(?:,|$)", derives)
                if trait in scope
            ]

        def add_type(match, **unit):
            """Add the struct, trait or enum whose name ``match`` holds, on
            the line being read, with ``unit`` in its metadata. Returns its
            ID."""
            type_id = f"{parent_id}::{match[1]}"
            end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
            self.builder.add_node(
                type_id,
                "Class",
                name=match[1],
                lineno=lineno,
                end_lineno=end_lineno,
                token_count=token_count(lineno, end_lineno),
                docstring=self.get_leading_docstring(lines, i),
                **_generic_metadata(_rust_type_parameters(line[match.end() :])),
                **unit,
            )
            self._keep_visibility(type_id, line)
            self.builder.add_edge(parent_id, type_id, "DEFINES")
            add_derived(type_id, i)
            return type_id

        def impl_at(index, line):
            """What ``_rust_impl`` reads in the header of the `impl` block
            that starts with ``line``, line ``index``, the bounds the block
            gives its type parameters, its own or by `where`, whether its
            trait is written with type arguments, and whether its type is a
            reference, `impl T for &U`, the path its type is written with,
            and the type arguments it is written with; eight None where none
            does. The header runs to its `{`, on that line or a later one."""
            if not _RUST_IMPL_RE.match(line):
                return (None,) * 8
            header = []
            for code, _ in codes[index:]:
                header.append(code)
                if "{" in code or ";" in code:
                    break
            text = _RUST_IMPL_RE.sub("", " ".join(header), count=1)
            bounds = _rust_type_parameters(text)
            where = text.partition(" where ")[2]
            for name, bound in re.findall(r"\b(\w+)\s*:\s*([\w:]+)", where):
                bounds[name] = bounds.get(name) or bound
            trait, owner, blanket, path, arguments = _rust_impl(text)
            generic = trait and re.search(rf"\b{re.escape(trait)}\s*<\s*(?!')", text)
            borrowed = re.search(r"\bfor\s*&", text.partition(" where ")[0])
            return (
                trait,
                owner,
                blanket,
                {k: v for k, v in bounds.items() if v},
                generic,
                borrowed,
                path,
                arguments,
            )

        i = 0
        while i < len(lines):
            code, spanned = codes[i]
            # The text of a string or comment declares nothing.
            line = code if spanned else lines[i]
            lineno = i + 1
            indent = len(line) - len(line.lstrip())
            # The regex call fallback starts past a name declared on the line.
            call_scan_start = 0

            if "}" in line:
                while len(context_indent) > 1 and indent <= context_indent[-1]:
                    context_indent.pop()
                    popped_id = current_context.pop()
                    if current_enum_id and popped_id == current_enum_id:
                        current_enum_id = None
                    if popped_id == impl_id:
                        impl_id = current_impl_target = None
                        current_trait_impl = None

            parent_id = current_context[-1]
            in_body = current_enum_id and not enum_depth
            if current_enum_id:
                enum_depth += sum(map(code.count, "([{")) - sum(map(code.count, ")]}"))

            # A `use` binds in the module or the function it is in, one on
            # a function's head in its body.
            bind_uses(parent_id, uses_at.get(lineno, {}), lineno)
            if not re_fn_start.match(line):
                bind_uses(parent_id, uses_after.get(lineno, {}), lineno)
            if macro and i > macro[0]:
                macro = None
            if rules := _RUST_MACRO_RULES_RE.match(code):
                # The body runs to the bracket that closes its first.
                depth = opened = 0
                for end in range(i, len(codes)):
                    text = codes[end][0][rules.end() if end == i else 0 :]
                    opened = opened or any(c in text for c in "({[")
                    depth += sum(map(text.count, "({[")) - sum(map(text.count, ")}]"))
                    if opened and depth <= 0:
                        break
                macro = end, []
                self.builder.rust_macros.setdefault(rules[1], []).append(
                    (file_id, macro[1])
                )
            elif (
                impl_id
                and parent_id == impl_id
                and (called := re.match(r"\s*(\w+)!\s*[({\[]", code))
            ):
                # A macro called among an impl block's items writes items of it.
                self.builder.rust_macro_uses.setdefault(impl_id, set()).add(called[1])

            if match := re_const.search(line):
                name, val = match.groups()
                docstring = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                typed = _RUST_CONST_RE.search(line)
                self.builder.add_node(
                    cid,
                    "Constant",
                    name=name,
                    value=val,
                    lineno=lineno,
                    docstring=docstring,
                    **(
                        {"value_type": _RUST_BORROW_RE.sub("", typed["type"]).strip()}
                        if typed
                        else {}
                    ),
                )
                self._keep_visibility(cid, line)
                self.builder.add_edge(parent_id, cid, "DEFINES")

            if match := re_fn_start.search(line):
                name = match.group(1)
                # A signature holds no call: the line's calls start in its body.
                body = line.find("{", match.end())
                call_scan_start = body + 1 if body >= 0 else len(line)
                fn_start = lineno
                j, full_sig = self._consume_until_body(lines, i)

                param_types, params, ret = self._extract_rust_param_types(full_sig)

                sig = f"fn {name}(...) -> {ret}"
                docstring = self.get_leading_docstring(lines, i)

                if impl_id and parent_id not in fn_ids:
                    func_id = f"{impl_id}::{name}"
                    node_type = "Method"
                else:
                    func_id = f"{parent_id}::{name}"
                    node_type = "Function"
                # One given again, for another `#[cfg]` or trait argument
                # (`impl From<A>`, `impl From<B>`), is `name#2` and so on.
                func_id = self.free_id(func_id)

                end_lineno = (
                    self._find_block_end(lines, j) if "{" in lines[j] else lineno
                )

                node_kwargs = {
                    "name": name,
                    "signature": sig,
                    "lineno": lineno,
                    "end_lineno": end_lineno,
                    "token_count": token_count(lineno, end_lineno),
                    "docstring": docstring,
                    "return_type": ret,
                    "param_types": param_types,
                }
                if current_trait_impl and node_type == "Method":
                    node_kwargs["trait_impl"] = current_trait_impl
                    if trait_generic:
                        # Its trait takes type arguments: `impl From<A> for T`
                        node_kwargs["trait_generic"] = True
                    if impl_borrowed:
                        # Its block is for a reference: `impl Add for &T`
                        node_kwargs["impl_borrowed"] = True
                if impl_bounds and node_type == "Method":
                    # What its `impl` block bounds the type's parameters by
                    node_kwargs["impl_bounds"] = impl_bounds
                if any(impl_args or ()) and node_type == "Method":
                    # The types its `impl` block gives the type's parameters
                    node_kwargs["impl_args"] = impl_args

                self.builder.add_node(func_id, node_type, **node_kwargs)
                if node_type == "Function":
                    self._keep_visibility(func_id, line)
                self.builder.add_edge(parent_id, func_id, "DEFINES")
                if macro and node_type == "Function" and parent_id not in fn_ids:
                    macro[1].append(func_id)
                for parameter in _RUST_PARAMETER_RE.findall(full_sig):
                    self.add_local(func_id, parameter)
                for param in params:
                    self._bind_parameter(param, func_id, codes, i, j)

                if opens_block(lines[j]):
                    # The body ends at the `}` indented as its closing line.
                    closing = lines[end_lineno - 1]
                    current_context.append(func_id)
                    context_indent.append(len(closing) - len(closing.lstrip()))
                    fn_ids.add(func_id)
                if "{" in codes[j][0]:
                    # A body on the signature's last line: its calls are the
                    # function's.
                    parent_id = func_id
                    bind_uses(func_id, uses_after.get(j + 1, {}), j + 1)
                    if j > i:
                        lineno = j + 1
                        call_scan_start = codes[j][0].find("{") + 1
                i = j

            elif match := re_struct.search(line):
                call_scan_start = match.end()
                type_id = add_type(match, **_unit(line[match.end() :], ";"))
                if held := _rust_held(line[match.end() :]):
                    self.builder.fields[type_id] = held
                for field, written in held.items():
                    # Each field the struct's own line declares, `self.0` and
                    # `self.t`, as the scan files one on a line of its own
                    key, value = (match[1], str(field)), f"{written}()"
                    self.fields.setdefault(key, set()).add(value)
                    self._field_lines.setdefault(key, {})[lineno] = value
                if opens_block(line):
                    current_context.append(type_id)
                    context_indent.append(indent)

            elif match := re_enum.search(line):
                type_id = add_type(match)
                if opens_block(line) or "{" not in code:
                    # The body opens here or on a line below.
                    current_context.append(type_id)
                    context_indent.append(indent)
                    current_enum_id = type_id
                    # Its body is open, or opens with the next `{`.
                    enum_depth = 0 if "{" in code else -1
                elif (body := code.find("{", match.end())) >= 0:
                    # A body on the enum's line
                    for part in _bracketed_parts(code[body + 1 :]):
                        add_variant(type_id, part, lineno)

            elif in_body and (match := re_enum_variant.search(line)):
                call_scan_start = match.end(1)
                add_variant(current_enum_id, line, lineno)

            elif (named := impl_at(i, line))[1]:
                # The trait is tracked for method metadata.
                (
                    trait,
                    current_impl_target,
                    bounds,
                    impl_bounds,
                    generic,
                    borrowed,
                    written,
                    impl_args,
                ) = named
                current_trait_impl = _last_segment(trait)
                trait_generic = bool(generic)
                impl_borrowed = bool(borrowed)
                if written:
                    # Its first name read through the module's `use`s, so a
                    # block for `F` of `use crate::a::Foo as F;` is one for
                    # `Foo`; one for a type of another crate keeps its name.
                    head, sep, rest = written.partition("::")
                    bound = module_uses.get(parent_id, {}).get(head, head)
                    if bound.partition("::")[0] in ("crate", "self", "super"):
                        written = bound + sep + rest
                        current_impl_target = _last_segment(written)
                # The block is for the type its path names from its module.
                struct_id = seen_type(parent_id, written or current_impl_target)
                if struct_id is None:
                    # One a function declares before the block, else the file's
                    struct_id = f"{parent_id}::{current_impl_target}"
                    if struct_id not in self.builder.node_registry:
                        struct_id = f"{file_id}::{current_impl_target}"
                impl_id = struct_id
                # A type of another module is told by the path each block of
                # its name writes, read from the module it is written in.
                held = impl_targets.setdefault(
                    struct_id, [current_impl_target, lineno, bounds, set()]
                )
                if written:
                    held[3].add((parent_id, written))

                if trait:
                    # A path says where the trait is from.
                    self.builder.add_edge(
                        struct_id,
                        f"REF:{current_trait_impl}",
                        "INHERITS",
                        {"origin": trait} if "::" in trait else None,
                    )

                if opens_block(line) or "{" not in line:
                    # The body opens here or, after a `where` clause, below.
                    current_context.append(struct_id)
                    context_indent.append(indent)
                else:  # `impl Marker for Foo {}`
                    # Methods of a block written on one line.
                    body = line.index("{")
                    self.begin_line(code, lineno, raw=lines[i], since=lineno)
                    for method in _RUST_INLINE_FN_RE.finditer(code, body):
                        method_id = f"{struct_id}::{method['name']}"
                        self.builder.add_node(
                            method_id,
                            "Method",
                            name=method["name"],
                            signature=f"fn {method['name']}(...)",
                            lineno=lineno,
                            end_lineno=lineno,
                            token_count=10,
                            docstring="",
                            return_type=(method["returns"] or "()").strip(),
                        )
                        self.builder.add_edge(struct_id, method_id, "DEFINES")
                        for call_match in re_call.finditer(method["body"]):
                            callee = call_match.group(1).lstrip(".")
                            if callee:
                                end = method.start("body") + call_match.end()
                                self.add_call(
                                    method_id,
                                    callee,
                                    lineno,
                                    lineno,
                                    end - 1 - len(callee),
                                    end - 1,
                                )
                    call_scan_start = len(line)
                    impl_id = current_impl_target = None
                    current_trait_impl = None

            elif match := re_mod.search(line):
                name = match.group(1)
                mod_id = f"{parent_id}::{name}"
                if opens_block(line):
                    # The module is the source of its items' DEFINES edges.
                    self.builder.add_node(
                        mod_id,
                        "Class",
                        name=name,
                        kind="module",
                        lineno=lineno,
                        docstring=self.get_leading_docstring(lines, i),
                    )
                    self._keep_visibility(mod_id, line)
                    self.builder.add_edge(parent_id, mod_id, "DEFINES")
                    current_context.append(mod_id)
                    context_indent.append(indent)
                    modules.add(mod_id)
                else:
                    attributes = " ".join(lines[attributes_from(i) : i])
                    written = re.search(r'#\[path\s*=\s*"([^"]+)"', attributes)
                    self.builder.rust_mods.setdefault(file_id, []).append(
                        (mod_id, written and written[1])
                    )

            # Associated types an impl block declares, and type aliases a
            # module declares, up to the `;` that ends them
            elif (
                (parent_id == impl_id or parent_id in modules)
                and re.match(rf"\s*{_RUST_VIS}type\s", code)
                and (declared := _rust_type_declared(statement_at(i)))
            ):
                name, params, held = declared
                held = re.sub(r"\s+", " ", held).strip()
                if parent_id == impl_id:
                    key = impl_id, current_trait_impl, name
                    self.builder.rust_associated[key] = held
                else:
                    params = tuple(_rust_type_parameters(params))
                    written = self.builder.rust_alias_types.setdefault(
                        (parent_id, name), set()
                    )
                    written.add((params, held))

            elif match := _RUST_USE_RE.search(line):
                path = match["tree"].replace("::", ".")
                self.builder.add_edge(file_id, f"MODULE:{path}", "IMPORTS")

            if parent_id != file_id:
                code = codes[lineno - 1][0]
                owner = self._owning_type(parent_id)
                raw = lines[lineno - 1]
                self.begin_line(code, lineno, parent_id, raw, fn_start, owner)
                self._bind_arm(code, i)
                # The closures on the line whose parameters are still to be read
                self._closures = []
                if "|" in code:
                    self._closures = [
                        (found.start("params") - 1, found["params"], found.end())
                        for found in _RUST_CLOSURE_RE.finditer(code)
                    ]
                self._bodies = []  # (call, argument, columns of the body) of each
                # The block a `for`, an `if let` or a `match` opens, where its
                # body goes on on the line, or the end of a `let`'s statement:
                # the names the pattern binds hold from there.
                head = (
                    ("for " in code and _RUST_FOR_RE.search(code))
                    or ("let " in code and _RUST_IF_LET_RE.search(code))
                    or ("match " in code and _RUST_MATCH_RE.search(code))
                )
                body = head.end() if head else len(code)
                # Each `let` with a pattern, bound once its statement ends
                lets = list(_rust_lets(code)) if "let " in code else []
                patterns = [found.span("pattern") for found in lets]
                for found in (_RUST_FOR_RE.search(code), _RUST_IF_LET_RE.search(code)):
                    if found:
                        patterns.append(found.span("pattern"))
                # A closure's parameters are a pattern: `|Wrapper(x)|`. The
                # pipes around them follow an expression's start — `=`,
                # `(`, `,`, `=>`, `move` — never a value a bitwise or reads.
                for closure in re.finditer(r"(?<!\|)\|([^|]+)\|(?!\|)", code):
                    before = code[: closure.start()].rstrip()
                    if (
                        not before
                        or before[-1] in "=([{,;:&!>"
                        or re.search(r"(?:^|[^\w$.])move$", before)
                    ):
                        patterns.append(closure.span(1))

                # `matches!(value, Pattern)`: the pattern is the macro's
                # second argument, at its top-level comma, ending at the
                # closing `)` or a guard, whose condition stays a call.
                # Returns whether the guard starts.
                def held(start, end, code=code, patterns=patterns):
                    guard = re.search(r"(?:^|[ \t])if\b", code[start:end])
                    patterns.append((start, start + guard.start() if guard else end))
                    return bool(guard)

                opened = None
                for at in re.finditer(r"\bmatches!\s*\(", code):
                    depth, comma, close = _matches_arguments(code, at.end() - 1, 0)
                    if close is None:
                        # The macro's arguments continue past the line end.
                        guarded = comma is not None and held(comma + 1, len(code))
                        opened = depth, comma is not None, guarded
                    elif comma is not None:
                        held(comma + 1, close)

                # A match's arms: from the head's `{` (where the head is
                # on the line) or the line's start (a bare arm line), to
                # each arm's own `=>`, walking paren, bracket and brace
                # depth once. Only a comma at every depth zero separates
                # arms; a nested match's braces hide its `=>`s and commas;
                # a guard's condition stays a call. A pattern that spans
                # lines carries its depths, and an arm whose head is open
                # at the line's end carries itself, to the next line.
                def walk(
                    region,
                    paren=0,
                    bracket=0,
                    brace=0,
                    in_head=True,
                    code=code,
                    patterns=patterns,
                ):
                    """One pass over ``code`` from ``region``: arm spans into
                    ``patterns``, each arm's pattern held until the next
                    ends; ``in_head`` False starts in an arm's body. Returns
                    ``(paren, bracket, brace, carried)``: "arm" where the
                    line ends inside an arm's pattern, "guard" inside its
                    guard, else None."""
                    pstart, pending, last_close = region, None, None
                    # the brackets open where the pattern being read starts
                    opened = paren + bracket + brace
                    for j in range(region, len(code)):
                        ch = code[j]
                        if ch == "(":
                            paren += 1
                        elif ch == ")":
                            paren -= 1
                        elif ch == "[":
                            bracket += 1
                        elif ch == "]":
                            bracket -= 1
                        elif ch == "{":
                            brace += 1
                        elif ch == "}":
                            if brace == 0:
                                if pending is not None:
                                    patterns.append((pstart, pending))
                                return paren, bracket, brace, None
                            brace -= 1
                            if brace == 0:
                                last_close = j + 1
                        elif brace == 0 and paren == 0 and bracket == 0:
                            if ch == ",":
                                if pending is not None:
                                    patterns.append((pstart, pending))
                                pstart, pending, last_close = j + 1, None, None
                                in_head, opened = True, 0
                            elif ch == "=" and code[j + 1 : j + 2] == ">":
                                # A guard's condition stays a call.
                                in_head = False
                                pend = j
                                end = _arm_head_end(code[pstart:pend], opened)
                                if end is not None:
                                    pend = pstart + end
                                if pending is not None:
                                    # A block-bodied arm ends without a
                                    # comma: the next pattern starts past
                                    # its closing brace.
                                    patterns.append((pstart, pending))
                                    if last_close:
                                        pstart, opened = last_close, 0
                                pending = pend
                    if pending is not None:
                        patterns.append((pstart, pending))
                        # A block-bodied arm the line ends with: the next
                        # line starts an arm's head, as after a comma.
                        ended = last_close is not None and not code[last_close:].strip()
                        return paren, bracket, brace, "arm" if ended else None
                    if not in_head:
                        return paren, bracket, brace, None
                    # An arm's head the line ends in: its pattern up to a
                    # guard, or a select!-style arm's `=`, whose expression
                    # goes on as calls.
                    if (end := _arm_head_end(code[pstart:], opened)) is not None:
                        patterns.append((pstart, pstart + end))
                        return paren, bracket, brace, "guard"
                    patterns.append((pstart, len(code)))
                    return paren, bracket, brace, "arm"

                cont = self._pattern_cont
                self._pattern_cont = None
                if cont is not None:
                    if cont[0] == "matches":
                        # The macro's arguments continue: its value stays
                        # calls, and its pattern runs from the top-level
                        # comma to the closing `)` or its guard.
                        paren, crossed, guarded = cont[1]
                        paren, comma, end = _matches_arguments(code, 0, paren, crossed)
                        end = end or len(code)
                        crossed = crossed or comma is not None
                        if crossed and not guarded:
                            start = comma + 1 if comma is not None else 0
                            guarded = held(start, end)
                        if paren:
                            self._pattern_cont = ("matches", (paren, crossed, guarded))
                    elif cont[0] == "guard":
                        # The guard goes on to its `=>`, the arm's body
                        # after it.
                        arrow, paren, bracket, brace = _top_level_arrow(code, *cont[1:])
                        carried = cont[0]
                        if arrow is not None:
                            paren, bracket, brace, carried = walk(
                                arrow + 2, in_head=False
                            )
                        if carried and paren >= 0:
                            self._pattern_cont = (carried, paren, bracket, brace)
                    else:
                        _, paren, bracket, brace = cont
                        paren, bracket, brace, carried = walk(0, paren, bracket, brace)
                        if carried and paren >= 0:
                            self._pattern_cont = (carried, paren, bracket, brace)
                else:
                    starts = [m.end() for m in _RUST_MATCH_RE.finditer(code)]
                    # The line's first `=>` at no bracket depth: one inside
                    # a macro's parentheses or braces is not an arm.
                    first_arm = _top_level_arrow(code)[0]
                    if first_arm is not None and (not starts or first_arm < starts[0]):
                        # A bare arm line: the arms before the line's first
                        # head (a nested match in a value) are its own. An
                        # assignment before the `=>` is a select!-style arm
                        # (`name = expr =>`), not a match arm.
                        prefix = code[:first_arm]
                        if not re.search(r"(?<![<>=!+\-*/%&|^.:])=(?!=|>)", prefix):
                            starts.insert(0, 0)
                    for region in starts:
                        paren, bracket, brace, carried = walk(region)
                        if carried and paren >= 0:
                            self._pattern_cont = (carried, paren, bracket, brace)
                if opened:
                    self._pattern_cont = ("matches", opened)
                for call_match in re_call.finditer(code, call_scan_start):
                    # In `0..items.len()` the callee follows the range's `..`.
                    callee = call_match.group(1).rpartition("..")[2].lstrip(".")
                    start = call_match.end(1) - len(callee)
                    # `Boxed::<Dog>::new` is a function of the type written
                    # with its arguments.
                    typed = re.fullmatch(r"(\w+)::<(.+)>::(\w+)", callee)
                    typed = typed and f"{typed[1]}<{typed[2]}>.{typed[3]}"
                    # `parse::<T>` calls `parse`; the type arguments are the
                    # call site's, not part of the callee's name.
                    callee = _strip_generics(callee)
                    while lets and lets[0].end() <= start:
                        self._bind_let(code, lets.pop(0), lineno, fn_start, owner)
                    if start >= body:
                        self._bind_patterns(code, i, fn_start, owner)
                        body = len(code) + 1
                    # A pattern deconstructs; it is not a call.
                    if any(pstart <= start < pend for pstart, pend in patterns):
                        continue
                    # `pub(crate)` is a visibility, not a call.
                    if callee and callee != "pub":
                        self._bind_closures(code, i, lines, call_match.start(1))
                        self.add_call(
                            parent_id,
                            callee,
                            lineno,
                            fn_start,
                            start,
                            call_match.end() - 1,
                            self._generic_path_origin(callee, lineno)
                            or typed
                            or self._receiver_origin(
                                code, start, callee, lineno, fn_start
                            ),
                        )
                self._bind_closures(code, i, lines, len(code))
                self._name_results(code, lineno, fn_start, owner)
                # What each call closed on the line is passed, but in a
                # pattern
                opened = {closed: paren for paren, closed in self._pairs.items()}
                # A literal's text, blanked, is still an argument.
                filled = _filled(code, lines[i].ljust(len(code)), "_")
                for closed, index in self._closes.items():
                    if index is None or closed not in opened:
                        continue
                    at, passed, borrows = opened[closed] + 1, [], []
                    if any(start <= at < end for start, end in patterns):
                        continue
                    for part in _bracketed_parts(
                        filled[at:closed], expression=True, turbofish=True
                    ):
                        at = filled.index(part, at)
                        # A second dereference, `&**self`, is through a
                        # `Deref` impl, to another type.
                        origin = "**" not in part and (
                            self._expression_origin(
                                code, at, at + len(part), lineno, fn_start, owner
                            )
                        )
                        # A path to a value, `&Kind::Unit`, names it; any
                        # other path, `Self::helper`, is its own origin.
                        bare = re.sub(r"^&\s*(?:mut\s+)?", "", part)
                        if _RUST_VALUE_PATH_RE.fullmatch(bare):
                            offset = self._blocks[1][lineno - 1] + at
                            origin = origin or self._path_origin(bare, offset)
                        elif bare != part and _RUST_PATH_ARGUMENT_RE.fullmatch(bare):
                            # A borrowed path borrows a value, never a function.
                            offset = self._blocks[1][lineno - 1] + at
                            origin = origin or self._path_origin(bare, offset)
                        elif _RUST_PATH_ARGUMENT_RE.fullmatch(part):
                            origin = origin or part
                        elif re.fullmatch(r"\d[\d_]*", part.strip()):
                            # An integer literal without a suffix is of
                            # whichever integer type the call takes.
                            origin = "{integer}"
                        elif ranged := _rust_range(part):
                            origin = f"{{lang:{ranged}}}"
                        elif typed := _RUST_TYPED_VALUE_RE.fullmatch(part.strip()):
                            # One with a suffix, `0usize`, or a cast, `i as
                            # usize`, is of that type.
                            origin = origin or f"{typed['suffix'] or typed['cast']}()"
                        passed.append(origin or None)
                        borrows.append(_rust_borrow(part))
                        at += len(part)
                    site = self.builder.edges[index]["metadata"]
                    site.setdefault("site", {})["arguments"] = passed
                    site["site"]["borrows"] = borrows
                for found in lets:
                    self._bind_let(code, found, lineno, fn_start, owner)
                if body <= len(code):
                    self._bind_patterns(code, i, fn_start, owner)
            i += 1

        # A field's type, for each struct of that name the file declares.
        for owner, field in self.fields:
            for node in self.builder.nodes[self._first_node :]:
                held = node["id"].endswith(f"::{owner}") and self._field_types(
                    node, owner, field
                )
                if held and len(held) == 1:
                    fields = self.builder.fields.setdefault(node["id"], {})
                    value = next(iter(held))
                    written = self.builder.parameter_types.get(value, (value[:-2],))
                    # A tuple struct's position is kept as its number.
                    key = int(field) if field.isdigit() else field
                    fields.setdefault(key, written[0])

        # An impl block's edges are sourced at its type. A type declared in
        # another file has no node here, so give it one.
        for struct_id, (name, impl_lineno, bounds, paths) in impl_targets.items():
            if struct_id not in self.builder.node_registry:
                self.builder.add_node(
                    struct_id,
                    "Class",
                    name=name,
                    kind="impl",
                    lineno=impl_lineno,
                    docstring="",
                    # The traits a type must have for a blanket impl to be its.
                    **({} if bounds is None else {"bounds": bounds}),
                    **({"impl_paths": sorted(paths)} if paths else {}),
                )
                self.builder.add_edge(file_id, struct_id, "DEFINES")

        self.restore_source_lines()

    def _consume_until_body(self, lines, start_idx):
        sig_lines = [lines[start_idx]]
        j = start_idx
        open_p = lines[j].count("(")
        close_p = lines[j].count(")")
        while (
            open_p > close_p or ("{" not in lines[j] and ";" not in lines[j])
        ) and j < len(lines) - 1:
            j += 1
            sig_lines.append(lines[j])
            open_p += lines[j].count("(")
            close_p += lines[j].count(")")
        return j, " ".join([ln.strip() for ln in sig_lines])

    def _bind_parameter(self, param, function_id, codes, first, last):
        """Type the names that ``param``, a parameter of ``function_id`` on
        lines ``first`` to ``last`` (indices of ``codes``) written as a
        pattern, binds, from the type it is declared with: `Wrap(inner):
        Wrap<T>` binds `inner` to what `Wrap<T>` holds first."""
        found = re.fullmatch(r"(?s)(.*?[)\]}])\s*:(?!:)(.+)", param)
        for name, steps in self.bindings(found[1]) if found else ():
            held = _RUST_BORROW_RE.sub("", found[2].strip())
            # An element of a tuple is of the type the tuple's type writes there.
            while steps and re.fullmatch(r"@\d+", steps[0]) and held[:1] == "(":
                elements = _bracketed_parts(held[1:])
                if int(steps[0][1:]) >= len(elements):
                    break
                held = _RUST_BORROW_RE.sub("", elements[int(steps[0][1:])].strip())
                steps = steps[1:]
            origin = f"{held}()"
            if re.search(r"[().]", held):
                # A type no origin path can write, read as written
                origin = f"{{{self.builder.serial(self.builder.parameter_types)}}}"
                self.builder.parameter_types[origin] = held, function_id
            for index in range(first, last + 1):
                column = _named(codes[index][0], name, 0)
                if column is not None:
                    self._rebind(name, index + 1, column, ".".join([origin, *steps]))
                    break

    def _extract_rust_param_types(self, full_sig: str) -> tuple:
        """Extract parameter types from Rust function signature, with the
        parameters as written and the type it returns."""
        param_types, returns = [], "()"
        # The parameter list is the first `(` after the type parameters,
        # whose `Fn(...)` bounds type the parameters they name.
        named = re.search(r"\bfn\s+\w+\s*", full_sig)
        paren_start = full_sig.find("(", named.end() if named else 0)
        angle = re.match(r"\s*(?:pub(?:\([^)]*\))?\s+)?[\w\s]*?fn\s+\w+\s*<", full_sig)
        bounds, generics = {}, ""
        if angle:
            at = _angle_end(full_sig, angle.end() - 1)
            if at is not None:
                paren_start = full_sig.find("(", at)
                generics = full_sig[angle.end() : at]
            bounds = dict(_RUST_FN_BOUND_RE.findall(generics))
        bounds.update(_RUST_FN_BOUND_RE.findall(full_sig.partition(" where ")[2]))
        closed = paren_start >= 0 and _closing([full_sig], 0, paren_start + 1)
        if not closed:
            return [], [], returns
        paren_end = closed[1]

        returns = _rust_return_type(full_sig[paren_end + 1 :])
        params = _bracketed_parts(full_sig[paren_start + 1 : paren_end])

        named = {"Self", *re.findall(r"(?:^|,)\s*(?:const\s+)?(\w+)", generics)}
        for param in params:
            # The receiver: `self`, `mut self`, `&'a mut self`, `self: Box<Self>`
            if receiver := _RUST_RECEIVER_RE.fullmatch(param):
                param_types.append(receiver[1])
            elif ":" in param:
                type_part = param.split(":", 1)[1].strip()
                if type_part in bounds:
                    type_part = f"impl {bounds[type_part]}"
                # A closure parameter of a type the function's type
                # parameters or `Self` name is of no known type.
                type_part = _RUST_FN_TYPE_RE.sub(
                    lambda found: found[0].replace(
                        found[1],
                        ", ".join(
                            "_" if named & set(re.findall(r"\w+", part)) else part
                            for part in _bracketed_parts(found[1] + ")")
                        ),
                    ),
                    type_part,
                )
                if len(type_part) > 50:
                    type_part = type_part[:47] + "..."
                param_types.append(type_part)
            else:
                param_types.append("_")
        return param_types, params, returns


def _rust_type_parameters(text: str) -> dict[str, str]:
    """``{name: bound}`` of the type parameters ``text`` opens with,
    `<'a, T: 'a + Bound + Other = Default, const N: usize>`: no lifetime,
    and of a bound its first trait."""
    text = text.lstrip()
    found = {}
    for part in _bracketed_parts(text[1:]) if text.startswith("<") else ():
        head, _, bound = part.split("=")[0].partition(":")
        words = head.split()
        if words and not words[-1].startswith("'"):
            traits = [b.strip() for b in bound.split("+")]
            traits = [b for b in traits if b and b[0] not in "'?"]
            found[words[-1]] = traits[0] if traits and words[0] != "const" else ""
    return found


# What starts a statement after a block's `}` or a statement's `;` in the
# other brace languages: not a word that goes on with the statement before
_STATEMENT_START_RE = re.compile(
    r"(?!(?:else|elseif|catch|finally|while|on|in)(?![\w$]))[A-Za-z_$@\\[]"
)
