"""C / C++ regex-based baseline parser.

Covers ``#include`` directives, preprocessor macros (``#define``),
``namespace``, ``class`` / ``struct`` / ``union`` / ``enum``, function
definitions in and out of their class, and call sites. Nothing is
preprocessed: every arm of a conditional is read, and a macro is known only
by how it is written, in capitals. scip-clang fills in the hard cases when
a compile database is available.
"""

from __future__ import annotations

import re
from bisect import bisect_right

from descry.generate import (
    BaseParser,
    _bracketed_parts,
    _casts,
    _flip_sign,
    _strip_generics,
    blank_strings,
    is_generated_source,
)

# #include <foo.h>  /  #include "foo.h"
_RE_INCLUDE = re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]')

# #define MACRO value
_RE_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)")

_RE_OPENED_NAMESPACE = re.compile(r"\bnamespace\s+([A-Za-z_]\w*)\s*\{")
# In the text of a macro, a name that a `;` follows, with array bounds or a
# parameter list between them at most, or a name in capitals: a field, or a
# macro that declares more of them.
_RE_MACRO_FIELD = re.compile(
    r"([A-Za-z_]\w*)\s*(?:\)\s*\([^()]*\)\s*|\[[^\]]*\]\s*)*;"
    r"|\b([A-Z_][A-Z0-9_]{2,})\b(?!\s*[(;=\[])"
)
_RE_MACRO_NAME = re.compile(r"\s*#\s*define\s+\w+(?:\([^)]*\))?")
_RE_MACRO_HEAD = re.compile(r"\s*#\s*define\s+\w+\(([^)]*)\)")
_RE_DIRECTIVE = re.compile(r"\s*#\s*(\w*)")
_RE_IF_0 = re.compile(r"\s*#\s*if\s+0\s*$")
_RE_IFDEF = re.compile(r"\s*#\s*(?:el)?if(?:n?def)?\s+")
_RE_DEFINED = re.compile(r"^!?\s*defined\s*\(?\s*([A-Za-z_]\w*)\s*\)?$")
_RE_IF_ATOM = re.compile(r"^!?\s*([A-Za-z_]\w*)$")
_IF_KEYWORDS = frozenset({"defined", "and", "or", "not", "true", "false"})


def _if_signs(word, code):
    """The atoms a preprocessor conditional's arm holds, signed: `+X` where
    `X` holds, `-X` where it does not. A disjunction holds nothing for
    sure, and a comparison names no symbol."""
    if word == "else":
        return frozenset()
    if word == "ifdef":
        return frozenset({"+" + _RE_IFDEF.sub("", code).strip()})
    if word == "ifndef":
        return frozenset({"-" + _RE_IFDEF.sub("", code).strip()})
    expr = _RE_IFDEF.sub("", code).split("//")[0]
    if "||" in expr:
        return frozenset()
    signs = set()
    for term in expr.split("&&"):
        term = term.strip()
        match = _RE_DEFINED.match(term) or _RE_IF_ATOM.match(term)
        if match and match[1] not in _IF_KEYWORDS:
            signs.add(("-" if term.startswith("!") else "+") + match[1])
    return frozenset(signs)


def _arm_signs(conditions):
    """Signed symbols the current arm of a conditional holds: the earlier
    arms' conditions do not (a one-atom arm gives its negation), the
    current arm's does."""
    signs = {
        _flip_sign(sign)
        for prior in conditions[:-1]
        for sign in prior
        if len(prior) == 1
    }
    signs.update(conditions[-1])
    return frozenset(signs)


# What code ends with in the middle of a statement
_OPEN_STATEMENT = frozenset("(),=&|+-*/<>?")
_RE_ACCESS = re.compile(r"\b(?:public|protected|private)\s*:(?!:)")

# Where a statement at declaration level ends or a nested group starts.
_RE_BOUNDARY = re.compile(r"[;{}(]")
_RE_BRACKET = re.compile(r"[(){}]")
_RE_PAREN = re.compile(r"[()]")

# The symbol of an overloaded operator, which a head is read without.
_RE_OPERATOR = re.compile(
    r"\boperator\s*(?:\(\s*\)|\[\s*\]|->\*?|<=>|<<=?|>>=?|&&|\|\||\+\+|--"
    r"|[-+*/%^&|!=<>]=?|,|~)"
)
_RE_ATTRIBUTE = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_RE_TEMPLATE = re.compile(r"\s*(?:[A-Z_][A-Z0-9_]*\s+)*template\s*<")
_RE_TYPE_PARAMETER = re.compile(r"\b(?:typename|class)\s*(?:\.\.\.\s*)?([A-Za-z_]\w*)")

# namespace foo {  /  namespace a::b {  /  inline namespace v1 {  /  namespace {
_RE_NAMESPACE = re.compile(
    r"(?:\s*[A-Z_][A-Z0-9_]*\b(?:\s*\(\s*\))?)*\s*(?:inline\s+)?namespace"
    r"(?:\s+([\w:]+))?\s*"
)
# extern "C" {, its string blanked
_RE_LINKAGE = re.compile(r"\s*extern\s*")

# class Foo : public Bar  /  struct Foo  /  union Foo  /  enum [class] Foo,
# read from a head without its template arguments. Between the keyword and
# the name stand attributes: `alignas(...)`, `__attribute__((...))`, and
# export macros, which are names in capitals that another name follows.
_RE_TYPE = re.compile(
    r"\s*+(?P<typedef>typedef\b)?"
    r"(?:\s*(?:static|const|extern|[A-Z_][A-Z0-9_]*)\b(?:\s*\(\s*\))?)*\s*"
    r"(?P<kind>class|struct|union|enum)\b(?:\s+(?:class|struct)\b)?"
    r"(?:\s*(?:(?:alignas|__\w+)\s*\(\s*\)"
    r"|[A-Z_][A-Z0-9_]*\b(?=\s+(?!final\b)[A-Za-z_])))*"
    r"\s*(?P<name>(?:[A-Za-z_]\w*\s*::\s*)*[A-Za-z_]\w*)?\s*(?:final\b\s*)?"
    r"(?::(?!:)(?P<bases>.*))?",
    re.DOTALL,
)

# What may follow a parameter list: cv- and ref-qualifiers, `noexcept`,
# `override`, `final`, and attribute macros, each with arguments or none.
_AFTER_PARAMS = (
    r"(?:\s*(?:const|volatile|noexcept|override|final|throw|try|mutable|&&?"
    r"|[A-Z_][A-Z0-9_]*\b|__\w+)\s*(?:\(\s*\))?)*"
)
# The declarator of a function in a head whose parenthesised groups are
# blanked: `name( )`, `Type::name( ) const`, `~Type( )`, `operator+( )`,
# `operator bool( )`, `name<Args>( )`, and behind a declaration `= 0` or
# `= default`.
_RE_FUNCTION = re.compile(
    r"(?P<path>(?:[A-Za-z_]\w*\s*(?:<[^()]*>\s*)?::\s*)*)"
    r"(?P<name>~\s*[A-Za-z_]\w*|operator\x01+"
    r"|operator\s+[\w:*&<>,\s]*?[\w*&>]|[A-Za-z_]\w*)\s*(?:<[^()]*>\s*)?"
    r"(?P<params>\(\s*\))" + _AFTER_PARAMS + r"\s*(?P<body>=\s*\w+\s*)?$"
)
# The `:` that starts a constructor's member initialisers
_RE_INITIALISERS = re.compile(r"\)[^:;]*?(?<!:):(?!:)")
# A head is searched for a function this far back from its end.
_HEAD_LIMIT = 2000

# A statement made of macro invocations only: `NAME(args)` and names in
# capitals, in a row.
_RE_INVOCATIONS = re.compile(
    r"(?:\s*(?:[A-Za-z_]\w*\s*\(\s*\)|[A-Z_][A-Z0-9_]*\b))+\s*"
)
# What a statement goes on with behind a braced initialiser
_RE_GOES_ON = re.compile(r"\s*[;,=(\[.]")
# The name of an enumerator: what an enum's body or a `,` in it is followed
# by, unless that is a macro invocation.
_RE_ENUMERATOR = re.compile(r"(?:^|,)\s*([A-Za-z_]\w*)\b(?!\s*\()")
_RE_TYPE_KEYWORD = re.compile(r"\b(?:struct|class|union|enum)\s+$")
# The name a declarator gives: the last word of those it opens with
_RE_DECLARED_NAME = re.compile(r"[\s*]*(?:\w+[\s*]+)*([A-Za-z_]\w*)")
_RE_LOCAL_TYPE = re.compile(
    r"\b(?:struct|class)\s+([A-Za-z_]\w*)\s*(?:final\s*)?(?::(?!:)|\{)"
)
_RE_USING = re.compile(r"\s*using\s+(\w+)\s*=")
_RE_TYPEDEF = re.compile(r"\s*typedef\b")
_RE_ASSIGNMENT = re.compile(r"(?<![=!<>+\-*/%&|^])=(?!=)")

_SPECIFIER_WORDS = (
    r"const|static|constexpr|mutable|volatile|struct|class|enum|union"
    r"|typename|register|extern|inline|thread_local|unsigned|signed"
)
_SPECIFIERS = r"(?:(?:" + _SPECIFIER_WORDS + r")\s+)*"
_TYPE_NAME = (
    r"(?!(?:return|else|case|goto|delete|new|throw|typedef|using|namespace"
    r"|template|friend|operator|sizeof|auto|do|co_return|co_yield|co_await"
    r"|virtual|explicit|const|static|struct|class|enum|union|typename"
    r"|unsigned|signed|inline|extern|volatile|mutable|constexpr|register"
    r"|thread_local)\b)[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*"
)
# A type name with its template arguments: in `type` the name, and with it
# the member that the arguments are followed by, `Outer<Args>::Inner`
_TYPE = (
    r"(?P<type>"
    + _TYPE_NAME
    + r"(?:<[^;(){}=]*>)?(?:::[A-Za-z_]\w*)*)(?:<[^;(){}=]*>)?"
)
# A smart pointer, and in group 1 the type it points at
_POINTER_TYPE = r"(?:std::)?(?:unique_ptr|shared_ptr|optional|expected)\s*<[^;(){}=]*>"
_RE_POINTER = re.compile(
    r"((?:std::)?(?:unique_ptr|shared_ptr|optional|expected))\s*<\s*(?:const\s+)?([\w:]+)"
)
_DECLARATOR = r"(?:\s*[*&]+\s*(?:const\s+)?|\s+)(?P<name>[A-Za-z_]\w*)"

_DECLARATIONS = (
    # `std::unique_ptr<Type> name`, a local, a field or a parameter
    re.compile(
        r"(?:^|[;{}(,])\s*"
        + _SPECIFIERS
        + r"(?P<type>"
        + _POINTER_TYPE
        + r")"
        + _DECLARATOR
        + r"\s*(?=[=;,)(\[{])"
    ),
    # `Type name`, `Type* name`, `Type<Args>& name` where a statement starts
    re.compile(
        r"(?:^|[;{}])\s*" + _SPECIFIERS + _TYPE + _DECLARATOR + r"\s*(?=[;,\[{]|=(?!=))"
    ),
    # `Type first, name`
    re.compile(
        r"(?:^|[;{}])\s*"
        + _SPECIFIERS
        + _TYPE
        + r"(?:\s*[*&]+\s*|\s+)[A-Za-z_]\w*\s*,"
        + r"[\s*&]*(?P<name>[A-Za-z_]\w*)\s*(?=[;,\[]|=(?!=))"
    ),
    # The same in the head of a `for`, `if`, `while` or `switch`; before a
    # `:` the variable of a range-based `for`
    re.compile(
        r"\b(?:for|if|while|switch)\s*\(\s*"
        + _SPECIFIERS
        + _TYPE
        + _DECLARATOR
        + r"\s*(?==(?!=)|:(?!:))"
    ),
    # `for (auto& name : expr)` — `auto` reads no type; the range's does.
    re.compile(r"\bfor\s*\(\s*(?P<type>auto)\s*[*&\s]*(?P<name>[A-Za-z_]\w*)\s*:"),
)
# What stands before the name in `Type name(args);`, which declares `name`.
# A name in capitals there stands for a specifier macro.
_RE_DECLARES = re.compile(
    r"(?:^|[;{}])\s*(?:(?:"
    + _SPECIFIER_WORDS
    + r"|[A-Z_][A-Z0-9_]*)\s+)*(?P<type>auto|"
    + _TYPE_NAME
    + r"(?:<[^;(){}=]*>(?:::[A-Za-z_]\w*)+)?)(?:<[^;(){}=]*>)?"
    r"(?:\s*\*+\s*|\s*&\s*|\s+)$"
)
# `Type name`, `const Type& name`, `Type* const name`, `Type name[n]`
_RE_PARAMETER = re.compile(
    _SPECIFIERS
    + r"(?P<type>"
    + _TYPE_NAME
    + r"(?:\s*<.*>)?)"
    + r"(?:\s*[*&]+\s*(?:(?:const|restrict|__restrict)\s+)?|\s+)"
    + r"(?P<name>[A-Za-z_]\w*)\s*(?:\[.*\])?",
    re.DOTALL,
)
# The same for a field, with what may follow its name: array bounds, a
# bit-field width, attribute macros, a braced initialiser
_RE_FIELD = re.compile(
    _SPECIFIERS
    + r"(?P<type>"
    + _TYPE_NAME
    + r"(?:\s*<.*>)?)"
    + r"(?:\s*[*&]+\s*(?:const\s+)?|\s+)(?P<name>[A-Za-z_]\w*)"
    + r"\s*(?:\[[^\]]*\]\s*)*(?::\s*\w+\s*)?"
    + r"(?:[A-Z_][A-Z0-9_]*\s*(?:\(\s*\))?\s*)*(?:\{.*\}\s*)?",
    re.DOTALL,
)

# Call site: name(, ns::name(, ::name(, obj.method(, obj->method(,
# name<Args>(. Read from a line whose `->` are written `?.`.
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"((?:(?<![>:])::)?[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:(?:<[\w\s:,.*]*>)?::|\?\.|\.)[A-Za-z_][A-Za-z0-9_]*)*)"
    r"(?:<[\w\s:,.*<>]*>)?\s*\("
)
_RE_STEP = re.compile(r"(->|\.)")

_C_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "goto",
        "sizeof",
        "alignof",
        "typeid",
        "typedef",
        "using",
        "static_cast",
        "dynamic_cast",
        "const_cast",
        "reinterpret_cast",
        "new",
        "delete",
        "throw",
        "try",
        "catch",
        "noexcept",
        "decltype",
        "auto",
        "void",
        "nullptr",
        "true",
        "false",
        "this",
        "class",
        "struct",
        "enum",
        "union",
        "namespace",
        "template",
        "typename",
        "public",
        "protected",
        "private",
        "virtual",
        "override",
        "final",
        "inline",
        "constexpr",
        "consteval",
        "constinit",
        "static",
        "extern",
        "const",
        "volatile",
        "mutable",
        "register",
        "thread_local",
        "friend",
        "explicit",
        "operator",
        "co_await",
        "co_yield",
        "co_return",
        "requires",
        "concept",
        "alignas",
        "static_assert",
        "bool",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "signed",
        "unsigned",
    }
)


def _rewrites(text: str) -> list[list]:
    """How the replacement text of the `#define` ``text`` writes each
    parameter it does not leave as it stands, as ``[index, how]``: `::`
    behind the global qualifier, `.` behind another qualifier or a member
    access, `paste` joined to other text with `##`, and `NAME(n)` passed on
    as argument n of the macro NAME."""
    head = _RE_MACRO_HEAD.match(text)
    body = text[head.end() :] if head else ""
    found = []
    for index, param in enumerate(head[1].split(",") if head else ()):
        param = param.strip()
        if not param.isidentifier():
            continue
        if re.search(rf"##\s*{param}\b|\b{param}\s*##", body):
            found.append([index, "paste"])
        if re.search(rf"(?<![\w>])::\s*{param}\b", body):
            found.append([index, "::"])
        if re.search(rf"(?:->|\.|[\w>]\s*::)\s*{param}\b", body):
            found.append([index, "."])
        for call in re.finditer(r"([A-Za-z_]\w*)\s*\(", body):
            for position, part in enumerate(_bracketed_parts(body[call.end() :])):
                if part == param:
                    found.append([index, f"{call[1]}({position})"])
    return found


def _closers(text: str) -> dict[int, int]:
    """Offset of the `)` or `}` that closes each `(` and `{` of ``text``. A
    `(` still open where a block closes is dropped."""
    closers, stack = {}, []
    for match in _RE_BRACKET.finditer(text):
        char, at = match.group(), match.start()
        if char in "({":
            stack.append(at)
            continue
        while char == "}" and stack and text[stack[-1]] == "(":
            stack.pop()
        if stack and (text[stack[-1]] == "(") == (char == ")"):
            closers[stack.pop()] = at
    return closers


def _declarators(tail: str) -> list[str]:
    """The declarators in ``tail``, what follows the body of a typedef,
    each without `[[...]]` and without the calls in it: an attribute or a
    macro, or the parameters of a function type, none of which names the
    type."""
    flat, previous = _RE_ATTRIBUTE.sub("", tail), None
    while flat != previous:
        previous, flat = flat, re.sub(r"\((?:[^()]|\(\))+\)", "()", flat)
    return re.sub(r"\w+\s*\(\)", "", flat).split(",")


def _blank(match: re.Match) -> str:
    return " " * len(match.group())


def _held(written: str) -> str:
    """A written type as it is kept for a value: without its template
    arguments, unless it is a standard type — a smart pointer's, and a
    container's, name what the value's members return."""
    pointer = _RE_POINTER.match(written)
    if pointer:
        return f"{pointer[1]}<{pointer[2]}>"
    if written.strip().startswith("std::"):
        return written.strip()
    return _strip_generics(written).strip()


def _returned(written: str) -> str:
    """The type in the text before a function's name, or after its `->`:
    without specifiers, and without the macros written before it."""
    words = re.sub(r"\(\s*\)", " ", written)
    words = re.sub(
        r"\b(?:static|inline|virtual|explicit|constexpr|consteval|friend"
        r"|extern|typename|const|volatile|struct|class|enum|union)\b",
        " ",
        words,
    )
    words = re.sub(r"\b[A-Z_][A-Z0-9_]*\s+(?=[A-Za-z_])", "", words)
    held = _held(" ".join(words.split()))
    return "" if held == "auto" else re.sub(r"\s+(?=[*&])", "", held)


def _signature(flat: str) -> re.Match | None:
    """The function declared by a head that ``ClangParser._outline`` has
    blanked: a match with its `path` of scopes, its `name`, its `params`
    and, on a declaration that defines, its `body`. None for any other
    head."""
    declarator = flat.partition("->")[0]
    cut = _RE_INITIALISERS.search(declarator)
    if cut:
        declarator = declarator[: cut.end() - 1]
    found = _RE_FUNCTION.search(declarator, max(0, len(declarator) - _HEAD_LIMIT))
    while found and found["name"].isupper() and not declarator[: found.start()].strip():
        # Of macro invocations in a row the last is the head; those before
        # it are statements of their own.
        later = _RE_FUNCTION.search(declarator, found.end("params"))
        if not later:
            break
        declarator = " " * later.start() + declarator[later.start() :]
        found = _RE_FUNCTION.search(declarator)
    if (
        not found
        or "=" in declarator[: found.start()]
        or found["name"] in _C_CONTROL_KEYWORDS
    ):
        return None
    return found


def _before(found: re.Match) -> str:
    """The specifiers and the return type before the name that ``found``, a
    match of ``_signature``, has. A name in capitals after nothing but names
    in capitals has none: all of them are macros."""
    prefix = found.string[: found.start()].strip()
    macros = found["name"].isupper() and not re.search(r"[a-z*&]", prefix)
    return "" if macros else prefix


class ClangParser(BaseParser):
    """Brace-matching C/C++ parser."""

    language = "clang"

    binding_re = re.compile(r"\bauto\b[\s*&]*(?P<name>\w+)\s*=(?!=)")

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)
        self._source = content.splitlines()
        self._skip_calls = is_generated_source(content)
        self._macros = {}  # ID of each macro of the file -> its metadata
        self._guarded = {}  # line -> the conditionals that enclose it
        bodies = []  # (macro ID, first line, last line, parameters) of each `#define`
        self._lines = self._prepare(file_id, bodies)
        self._text = "\n".join(self._lines)
        self._starts = [0]  # offset in the text of each line
        for line in self._lines:
            self._starts.append(self._starts[-1] + len(line) + 1)
        self._closers = _closers(self._text)
        self._kinds = {}  # ID of each namespace and type of the file -> kind
        self._generic = frozenset()  # type parameters in scope
        self._internal = 0  # unnamed namespaces open
        self._member = False  # the calls being read are a member function's
        self._kr = None  # offsets of a function head with K&R parameters
        self.scan_declarations(self._lines, _DECLARATIONS)
        self._body(0, len(self._text), file_id, None)
        # A definition that sits inside an `#if` arm is built only in the
        # configurations that pick the arm: that is evidence about the
        # call sites that may reach it.
        for node in self.builder.nodes[self._first_node :]:
            line = node["metadata"].get("lineno")
            if line and (guards := self._guarded.get(line - 1)):
                node["metadata"]["guards"] = sorted(guards)
        # The calls in a macro's replacement text are the macro's; what its
        # parameters stand for is not known.
        for macro_id, first, last, parameters in () if self._skip_calls else bodies:
            nesting = []
            for parameter in parameters:
                self.declared.setdefault(parameter, []).append((first + 1, "?", False))
            for index in range(first, last + 1):
                code = blank_strings(self._source[index], None, ())[0]
                head = _RE_MACRO_NAME.match(code) if index == first else None
                code = " " * head.end() + code[head.end() :] if head else code
                # What a parameter names is not a function of that name,
                # nor is a definition's head a call.
                named = {
                    call.start()
                    for call in _RE_CALL.finditer(code.replace("->", "?."))
                    if re.split(r"::|\.", call[1])[-1] in parameters
                    or _RE_DECLARES.search(code[: call.start()])
                }
                self._line_calls(
                    macro_id, 0, index, code.rstrip("\\ "), nesting, named.__contains__
                )

    def _prepare(self, file_id, bodies):
        """The lines of the file as code: comments, the text of literals,
        preprocessor directives and access specifiers blanked, columns
        kept. Adds the includes and macros of the file, and to ``bodies``
        where each macro's replacement text is. Nothing of an
        `#if 0` is kept, and only the first of the alternatives of a
        conditional where that one leaves a block open or ends inside a
        statement: they are then alternatives of one piece of code."""
        lines = []
        inside = None
        continued = False  # the directive above goes on behind a backslash
        depth = 0
        last = ""  # the last character of code so far
        # [depth before, depth after the first alternative, skipped, the
        # directive's line, the signs each arm's condition holds] of each
        # open conditional
        groups = []
        closed = []  # (open line, arms, close line) of each ended `#if`
        defined_here = set()  # macros the file defines (include guards)
        for index, raw in enumerate(self._source):
            code, inside = blank_strings(raw, inside, ())
            directive = None if continued else _RE_DIRECTIVE.match(code)
            if not continued and not directive:
                if groups and groups[-1][2]:
                    code = ""
                if groups:
                    self._guarded[index] = frozenset().union(
                        *(_arm_signs(group[4]) for group in groups)
                    )
                code = _RE_ACCESS.sub(_blank, code)
                depth += code.count("{") - code.count("}")
                last = code.rstrip()[-1:] or last
                lines.append(code)
                continue
            continued = raw.rstrip().endswith("\\")
            lines.append("")
            if groups:
                self._guarded[index] = frozenset().union(
                    *(_arm_signs(group[4]) for group in groups)
                )
            if not directive:
                continue
            word = directive[1]
            if word in ("if", "ifdef", "ifndef"):
                skipped = bool(groups) and groups[-1][2]
                groups.append(
                    [
                        depth,
                        None,
                        skipped or bool(_RE_IF_0.match(code)),
                        index,
                        [_if_signs(word, code)],
                    ]
                )
            elif word in ("elif", "else") and groups:
                group = groups[-1]
                if group[1] is None:
                    group[1] = depth
                outer = len(groups) > 1 and groups[-2][2]
                group[2] = outer or group[1] != group[0] or last in _OPEN_STATEMENT
                group[4].append(_if_signs(word, code))
                depth = group[0]
            elif word == "endif" and groups:
                group = groups.pop()
                closed.append((group[3], group[4], index))
                if group[1] is not None:
                    depth = group[1]
            elif m_inc := _RE_INCLUDE.match(raw):
                self.builder.add_edge(file_id, f"MODULE:{m_inc.group(1)}", "IMPORTS")
            elif m_def := _RE_DEFINE.match(code):
                name = m_def.group(1)
                defined_here.add(name)
                macro_id = f"{file_id}::{name}"
                last = index
                while self._source[last].rstrip().endswith("\\"):
                    last += 1
                text = " ".join(self._source[index : last + 1])
                rewrites = _rewrites(text)
                head = _RE_MACRO_HEAD.match(text)
                parameters = re.findall(r"[A-Za-z_]\w*", head[1]) if head else []
                bodies.append((macro_id, index, last, parameters))
                if macro_id in self._macros:
                    # Another definition of the macro, for another build.
                    self._macros[macro_id].setdefault("rewrites", []).extend(rewrites)
                elif macro_id not in self.builder.node_registry:
                    # The namespaces its replacement text opens, and the
                    # fields and field macros it declares for a struct.
                    opens = _RE_OPENED_NAMESPACE.findall(text)
                    fields = _RE_MACRO_FIELD.findall(text.partition(name)[2])
                    self.builder.add_node(
                        macro_id,
                        "Constant",
                        name=name,
                        lineno=index + 1,
                        end_lineno=last + 1,
                        docstring="",
                        kind="macro",
                        **({"opens": opens} if opens else {}),
                        **({"rewrites": rewrites} if rewrites else {}),
                        **({"fields": fields} if fields else {}),
                    )
                    self.builder.add_edge(file_id, macro_id, "DEFINES")
                    self._macros[macro_id] = self.builder.nodes[-1]["metadata"]
        # A conditional that opens before all of the file's code, closes
        # after it and guards on a symbol the file itself defines — the
        # include-guard idiom — wraps the file in every build: its signs
        # are no evidence about which call site may reach what it holds.
        coded = [i for i, l in enumerate(lines) if l.strip()]
        if coded:
            ambient = frozenset()
            wrapping = [
                arms[0]
                for open_at, arms, close_at in closed
                if open_at < coded[0] and close_at >= coded[-1] and len(arms) == 1
            ] + [
                group[4][0]
                for group in groups
                if group[3] < coded[0] and len(group[4]) == 1
            ]
            ambient = frozenset().union(
                *(
                    arm
                    for arm in wrapping
                    if arm and {s[1:] for s in arm} <= defined_here
                )
            )
            if ambient:
                for index in list(self._guarded):
                    rest = self._guarded[index] - ambient
                    if rest:
                        self._guarded[index] = rest
                    else:
                        del self._guarded[index]
        return lines

    def _first_line(self, begin, offset):
        """Index of the line that the declaration around ``offset`` starts
        on: the first of the lines of code that lead up to it without a
        gap, no further back than the first of them from ``begin`` on."""
        begin = offset - len(self._text[begin:offset].lstrip())
        floor = bisect_right(self._starts, begin) - 1
        line = bisect_right(self._starts, offset) - 1
        while line > floor and self._lines[line - 1].strip():
            line -= 1
        return line

    def _outline(self, head):
        """``head``, the text of a declaration up to its body or its `;`,
        at the same length with what a declaration is not read from
        blanked: the text inside each pair of parentheses, attribute lists,
        `template <...>` prefixes, and the symbol of an overloaded
        operator, which is turned to U+0001s. Returns it with the offsets
        of each pair of parentheses and the type parameters the prefixes
        declare."""
        head = _RE_OPERATOR.sub(
            lambda match: "operator" + "\x01" * (len(match.group()) - 8), head
        )
        flat, spans, depth, kept, opened = [], [], 0, 0, 0
        for match in _RE_PAREN.finditer(head):
            if match.group() == "(":
                if not depth:
                    opened = match.start()
                    flat.append(head[kept : opened + 1])
                depth += 1
            elif depth:
                depth -= 1
                if not depth:
                    kept = match.start()
                    flat.append(" " * (kept - opened - 1))
                    spans.append((opened, kept))
        flat.append(" " * (len(head) - opened - 1) if depth else head[kept:])
        flat = _RE_ATTRIBUTE.sub(_blank, "".join(flat))
        parameters = set()
        while prefix := _RE_TEMPLATE.match(flat):
            end, depth = prefix.end(), 1
            while end < len(flat) and depth:
                depth += (flat[end] == "<") - (flat[end] == ">")
                end += 1
            parameters.update(_RE_TYPE_PARAMETER.findall(flat, prefix.end(), end))
            flat = " " * end + flat[end:]
        # What a type parameter stands for is not known.
        self.builder.type_aliases.update(parameters)
        return flat, spans, parameters

    def _body(self, start, end, parent_id, owner):
        """Read the declarations between offsets ``start`` and ``end``, the
        body of the file, namespace or type ``parent_id``. ``owner`` is the
        name of that type, None for a file or a namespace."""
        begin = at = start
        while match := _RE_BOUNDARY.search(self._text, at, end):
            at = match.start()
            if match.group() == "(":
                at = self._closers.get(at, end) + 1
            elif match.group() == "{":
                closer = self._closers.get(at, end)
                resume = self._block(begin, at, closer, end, parent_id, owner)
                if resume is not None:
                    begin = at = resume
                else:
                    # A block that is no declaration's body: an initialiser
                    # if its statement goes on behind it.
                    at = closer + 1
                    if not _RE_GOES_ON.match(self._text, at, end):
                        begin = at
            else:
                if match.group() == ";":
                    self._declaration(begin, at, parent_id, owner)
                begin = at = at + 1
        self._declaration(begin, end, parent_id, owner)

    def _block(self, begin, at, closer, end, parent_id, owner):
        """Read the block that the `{` at offset ``at`` opens and the `}`
        at ``closer`` ends, in a scope that ends at ``end``; its head
        starts at ``begin``. Returns the offset to read on from, None for a
        block that is no declaration's body."""
        head = self._text[begin:at]
        kr, self._kr = self._kr, None
        if kr and not head.strip():
            # The body of the function whose parameters the statements
            # before it declare.
            begin, head = kr[0], self._text[kr[0] : kr[1]]
        flat, spans, parameters = self._outline(head)
        space = _RE_NAMESPACE.fullmatch(flat)
        if space or _RE_LINKAGE.fullmatch(flat):
            # `extern "C"` and an unnamed namespace add no scope of their
            # own; what the latter holds is known to its file only.
            names = space[1].split("::") if space and space[1] else []
            lineno = bisect_right(self._starts, at)
            for name in filter(None, names):
                space_id = f"{parent_id}::{name}"
                if space_id not in self.builder.node_registry:
                    self.builder.add_node(
                        space_id,
                        "Class",
                        name=name,
                        kind="namespace",
                        lineno=lineno,
                        end_lineno=bisect_right(self._starts, closer),
                        docstring="",
                    )
                    self.builder.add_edge(parent_id, space_id, "DEFINES")
                self._kinds[space_id] = "namespace"
                parent_id, owner = space_id, None
            unnamed = bool(space) and not names
            self._internal += unnamed
            self._body(at + 1, closer, parent_id, owner)
            self._internal -= unnamed
            return closer + 1
        found = _signature(flat)
        if found:
            generic, self._generic = self._generic, self._generic | parameters
            function_id, since, member = self._function(
                found, flat, head, spans, begin, closer, parent_id, owner
            )
            self._member = member
            self._calls(function_id, since, begin + found.end("params"), at, True)
            # A type defined in the body is a member of the function, and
            # the calls in its own members are theirs.
            resume = at + 1
            for local in _RE_LOCAL_TYPE.finditer(self._text, at + 1, closer):
                opened = self._text.find("{", local.end() - 1, closer)
                if local.start() < resume or opened < 0:
                    continue
                self._calls(function_id, since, resume, local.start())
                ends = self._closers.get(opened, closer)
                resume = self._block(
                    local.start(), opened, ends, closer, function_id, None
                )
                resume = resume or local.start()
                self._member = member
            self._calls(function_id, since, resume, closer)
            self._member = False
            self._generic = generic
            return closer + 1
        declared = _RE_TYPE.fullmatch(_strip_generics(flat))
        if not declared:
            return None
        # The declaration ends at the `;` behind its body.
        stop = closer + 1
        while match := _RE_BOUNDARY.search(self._text, stop, end):
            stop = match.start()
            if match.group() in ";}":
                break
            stop = self._closers.get(stop, end) + 1
        stop = min(stop, end)
        tail = self._text[closer + 1 : stop]
        kind, name = declared["kind"], declared["name"]
        named = []
        if declared["typedef"]:
            named = [_RE_DECLARED_NAME.match(part) for part in _declarators(tail)]
        names = [found[1] for found in named if found]
        if name is None and named and named[0]:
            # `typedef struct { ... } Name;`, `typedef struct { ... } *Name;`
            name = names[0]
        if name is None:
            # An unnamed struct or union that declares no variable is a
            # member of the type around it, which has its fields, as the
            # scope around an unnamed enum has its enumerators.
            if kind in ("struct", "union") and not tail.strip():
                self._body(at + 1, closer, parent_id, owner)
            elif kind == "enum":
                self._enumerators(at + 1, closer, parent_id)
            return stop + 1
        first = self._first_line(begin, at)
        *path, name = re.split(r"\s*::\s*", name)
        parent_id, _ = self._qualified(parent_id, owner, path, first + 1)
        type_id = f"{parent_id}::{name}"
        end_lineno = bisect_right(self._starts, closer)
        if type_id not in self.builder.node_registry:
            self.builder.add_node(
                type_id,
                "Class",
                name=name,
                kind=kind,
                lineno=first + 1,
                end_lineno=end_lineno,
                token_count=(end_lineno - first) * 10,
                docstring=self.get_leading_docstring(self._source, first),
            )
            self.builder.add_edge(parent_id, type_id, "DEFINES")
        self._kinds[type_id] = kind
        for alias in names:
            self._alias(alias, name, parent_id)
        if kind == "enum":
            self._enumerators(at + 1, closer, type_id, name)
        else:
            generic, self._generic = self._generic, self._generic | parameters
            for base in _bracketed_parts(declared["bases"] or ""):
                base = base.split()[-1]
                # A type parameter, or a member of one, names no type.
                own = self._dependent(base)
                self.builder.add_edge(
                    type_id,
                    f"REF:{base}",
                    "INHERITS",
                    {"origin": f"={base}"} if own else None,
                )
            self._body(at + 1, closer, type_id, name)
            self._generic = generic
        return stop + 1

    def _dependent(self, written):
        """Whether ``written`` starts with a type parameter in scope: the
        parameter, a pack of it, a template it names or a member of one."""
        return re.match(r"\w*", written)[0] in self._generic

    def _opaque(self, written):
        """``written`` with `?` for each type parameter in scope that is one
        of its template arguments: what it stands for is not known."""
        if not self._generic:
            return written
        return re.sub(rf"(?<![\w:])(?:{'|'.join(self._generic)})\b", "?", written)

    def _enumerators(self, start, end, parent_id, enum=None):
        """Add the enumerators between offsets ``start`` and ``end`` to
        ``parent_id``: the enum named ``enum``, or the scope around an
        unnamed one."""
        flat = self._outline(self._text[start:end])[0]
        for found in _RE_ENUMERATOR.finditer(flat):
            constant_id = f"{parent_id}::{found[1]}"
            if constant_id not in self.builder.node_registry:
                self.builder.add_node(
                    constant_id,
                    "Constant",
                    name=found[1],
                    lineno=bisect_right(self._starts, start + found.start(1)),
                    docstring="",
                    **({"value_type": enum} if enum else {}),
                )
                self.builder.add_edge(parent_id, constant_id, "DEFINES")

    def _alias(self, name, written, scope_id):
        """Record that ``name`` is declared in ``scope_id`` as an alias of
        the type ``written``."""
        self.builder.type_aliases.add(name)
        held = _returned(written)
        if held and re.split(r"::|[*&\s]", held)[-1] != name:
            self.builder.c_aliases.setdefault(name, []).append((scope_id, held))

    def _named(self, scope_id, name):
        """ID of the namespace or type ``name`` of this file that is in
        scope in ``scope_id``, if there is one."""
        while f"{scope_id}::{name}" not in self._kinds:
            if "::" not in scope_id:
                return None
            scope_id = scope_id.rpartition("::")[0]
        return f"{scope_id}::{name}"

    def _qualified(self, scope_id, owner, path, lineno):
        """``(ID, type name)`` of the scope that ``path``, the names before
        the `::` of a definition written in ``scope_id`` on line
        ``lineno``, leads to; the scope itself and ``owner`` for no names.
        A type that this file does not declare gets a node to hold what is
        defined as its members."""
        for name in path:
            named = self._named(scope_id, name)
            if named is None:
                named = f"{scope_id}::{name}"
                self.builder.add_node(
                    named,
                    "Class",
                    name=name,
                    kind="class",
                    lineno=lineno,
                    docstring="",
                )
                self.builder.add_edge(scope_id, named, "DEFINES")
                self._kinds[named] = "class"
            scope_id = named
            owner = None if self._kinds[named] == "namespace" else name
        return scope_id, owner

    def _function(self, found, flat, head, spans, begin, last, parent_id, owner):
        """Add the function that ``found`` matched in ``flat``, the outline
        of the declaration ``head``, which starts at offset ``begin`` in
        the scope ``parent_id`` and ends at ``last``. Returns its ID, the
        line it starts on and whether it is a member function."""
        name = head[found.start("name") : found.end("name")]
        spaced = found["name"].startswith("operator") and "\x01" not in found["name"]
        name = re.sub(r"\s+", " " if spaced else "", name)
        prefix = _before(found)
        first = self._first_line(begin, begin + found.start("name"))
        path = re.findall(r"([A-Za-z_]\w*)\s*(?:<[^()]*>\s*)?::", found["path"])
        scope_id, owner = self._qualified(parent_id, owner, path, first + 1)
        if re.search(r"\bfriend\b", prefix):
            # A friend defined in a type is a function of the scope around it.
            owner = None
            while self._kinds.get(scope_id, "namespace") != "namespace":
                scope_id = scope_id.rpartition("::")[0]
        params = next(head[a + 1 : b] for a, b in spans if a == found.start("params"))
        declares = bool(prefix or found["path"] or name[0] == "~")
        words = []
        if declares or name == owner or name.startswith("operator"):
            parameters = _bracketed_parts(params)
            if name == owner:
                name = "<init>"
        else:
            # A macro invocation where a definition stands defines what its
            # body is: the function whose head it wraps, else a member of
            # the type one of its arguments names.
            inner, inner_spans, _ = self._outline(params)
            wrapped = _signature(inner)
            if wrapped and wrapped.string[: wrapped.start()].strip():
                begin += found.start("params") + 1
                return self._function(
                    wrapped, inner, params, inner_spans, begin, last, parent_id, owner
                )
            parameters = []
            words = re.findall(r"[A-Za-z_]\w*", params)
            name = f"{name}({', '.join(words)})"
            for word in words:
                named = self._named(parent_id, word)
                if named and self._kinds[named] in ("class", "struct"):
                    scope_id, owner, words = named, word, []
                    break
        member = owner is not None
        function_id = f"{scope_id}::{name}"
        end_lineno = bisect_right(self._starts, last)
        if function_id in self._macros:
            # A macro of the same name moves to `<file>::macro_<name>`.
            del self._macros[function_id]
            self.rename_node(function_id, f"{scope_id}::macro_{name}")
        # A `static` written under a conditional the name is not under
        # holds only where that arm compiles, so it is no linkage.
        inside = self._guarded.get(
            bisect_right(self._starts, begin + found.start("name")) - 1, frozenset()
        )
        static = any(
            self._guarded.get(
                bisect_right(self._starts, begin + word.start()) - 1, frozenset()
            )
            <= inside
            for word in re.finditer(r"\bstatic\b", flat[: found.start()])
        )
        if function_id not in self.builder.node_registry:
            extra = {}
            returns = _returned(flat.partition("->")[2] or prefix)
            if returns:
                extra["return_type"] = returns
            if name == "<init>":
                extra["is_constructor"] = True
            if found["body"] and "0" in found["body"]:
                extra["pure"] = True
            if words:
                # The type may be one of another file that an argument names.
                extra["arguments"] = words
            if inside:
                extra["guards"] = sorted(inside)
            if not member and (self._internal or static):
                extra["linkage"] = "internal"
            self.builder.add_node(
                function_id,
                "Method" if member else "Function",
                name=name,
                lineno=first + 1,
                end_lineno=end_lineno,
                token_count=(end_lineno - first) * 10,
                docstring=self.get_leading_docstring(self._source, first),
                **extra,
            )
            self.builder.add_edge(scope_id, function_id, "DEFINES")
        elif not member and not (self._internal or static):
            # A definition internal under one arm and external under
            # another is not internal.
            node = self._node(function_id)
            if node:
                node["metadata"].pop("linkage", None)
        for part in parameters:
            declared = _RE_PARAMETER.fullmatch(part.partition("=")[0].strip())
            if declared and declared["name"] not in _C_CONTROL_KEYWORDS:
                held = f"{_held(declared['type'])}()"
                self.declared.setdefault(declared["name"], []).append(
                    (first + 1, held, False)
                )
                self.add_local(function_id, declared["name"])
            elif pointer := re.search(r"\(\s*\*\s*(\w+)\s*\)", part):
                self.add_local(function_id, pointer[1])
        return function_id, first + 1, member or bool(words)

    def _declaration(self, begin, end, parent_id, owner):
        """Read the statement between offsets ``begin`` and ``end``, a
        declaration without a body: an alias, a member function that is
        pure or defaulted, a field of the type ``owner``, or a row of macro
        invocations. The calls of an initialiser are made by ``parent_id``."""
        head = self._text[begin:end]
        if not head.strip():
            return
        flat, spans, parameters = self._outline(head)
        if alias := _RE_USING.match(flat):
            self._alias(alias[1], flat[alias.end() :], parent_id)
            return
        after = spans[0][1] + 1 if spans else 0
        if (
            spans
            and _signature(flat[:after])
            and re.fullmatch(r"[\w\s,]+", head[spans[0][0] + 1 : after - 1])
            and _RE_FIELD.fullmatch(flat[after:].strip())
        ):
            # `name(a, b) int a;`: the head of a function whose parameters
            # are declared behind it, up to its body.
            self._kr = (begin, begin + after)
            return
        if alias := _RE_TYPEDEF.match(flat):
            # The alias is the last name, or the `(*name)` of a function
            # pointer's, which names no type that has members.
            pointer = spans and re.match(r"\s*\*\s*(\w+)", head[spans[0][0] + 1 :])
            names = re.finditer(r"[A-Za-z_]\w*", flat.partition("(")[0])
            *_, last = names
            if pointer or spans:
                self.builder.type_aliases.add(pointer[1] if pointer else last[0])
            else:
                self._alias(last[0], flat[alias.end() : last.start()], parent_id)
            return
        found = _signature(flat)
        if (
            found
            and owner
            and found["name"].isupper()
            and _RE_FIELD.fullmatch(flat.strip())
        ):
            # A field with an attribute macro behind its name.
            found = None
        if found:
            name = found["name"]
            prefix = _before(found)
            declares = prefix or found["path"] or name == owner or name[0] == "~"
            if not owner and not found["path"]:
                # A prototype at file or namespace level declares the
                # function wherever the file is included: the definition
                # it promises may sit in any file linked in, and the arm
                # it sits in is evidence about every build that has it.
                line = bisect_right(self._starts, end) - 1
                self.builder.declares.setdefault(
                    parent_id.split("::")[0], {}
                ).setdefault(name, self._guarded.get(line, ()))
            if declares or name.startswith("operator"):
                # A prototype defines nothing; `= 0` and `= default` do.
                defines = found["body"] and "delete" not in found["body"]
                if defines and (owner or found["path"]):
                    generic, self._generic = self._generic, self._generic | parameters
                    self._function(
                        found,
                        flat,
                        head,
                        spans,
                        begin,
                        begin + found.end(),
                        parent_id,
                        owner,
                    )
                    self._generic = generic
                return
        if spans and _RE_INVOCATIONS.fullmatch(flat):
            # A macro that wraps a prototype declares as the prototype does.
            wrapped = _signature(self._outline(head[spans[0][0] + 1 : spans[0][1]])[0])
            if not (wrapped and wrapped.string[: wrapped.start()].strip()):
                self._calls(parent_id, 0, begin, end)
            return
        if found:
            return
        assigned = _RE_ASSIGNMENT.search(flat)
        if owner:
            fields = self.builder.fields.setdefault(parent_id, {})
            # Macros in capitals before the declaration stand for fields of
            # their own.
            macros = re.match(r"(?:\s*[A-Z_][A-Z0-9_]*\b(?!\s*[*&(]))*", flat)
            for macro in macros[0].split():
                fields.setdefault(f"<{macro}>", "")
            flat = flat[macros.end() :]
            assigned = _RE_ASSIGNMENT.search(flat)
            parts = _bracketed_parts(flat[: assigned.start()] if assigned else flat)
            field = _RE_FIELD.fullmatch(parts[0]) if parts else None
            held = field and _held(field["type"])
            # `(*name)(params)` is a field that holds a function.
            pointer = spans and re.match(r"\s*\*\s*(\w+)", head[spans[0][0] + 1 :])
            if pointer:
                fields.setdefault(pointer[1], "")
            if field and not self._dependent(held):
                held = self._opaque(held)
                fields.setdefault(field["name"], held)
                for part in parts[1:]:
                    if more := re.match(r"[*&\s]*([A-Za-z_]\w*)", part):
                        fields.setdefault(more[1], held)
        if assigned:
            self._calls(parent_id, 0, end - len(flat) + assigned.end(), end)

    def _calls(self, source_id, since, start, end, initialisers=False):
        """Record the calls written between offsets ``start`` and ``end``
        as made by ``source_id``, whose definition starts on line ``since``
        (0 for a file, namespace or type). With ``initialisers`` the text is
        what stands between a parameter list and a body, where a name
        outside every parenthesis is a member being initialised."""
        if self._skip_calls:
            return
        nesting = []
        first = bisect_right(self._starts, start) - 1
        for index in range(first, bisect_right(self._starts, end - 1)):
            offset = self._starts[index]
            low = max(start - offset, 0)
            code = " " * low + self._lines[index][low : end - offset]
            skipped = initialisers and (
                lambda column, offset=offset: (
                    self._text.count("(", start, offset + column)
                    == self._text.count(")", start, offset + column)
                )
            )
            if code.strip():
                self._line_calls(source_id, since, index, code, nesting, skipped)

    def _line_calls(self, source_id, since, index, code, nesting, skipped=None):
        """Record the calls on ``code``, line ``index`` of the file without
        what is no code, as ``_calls`` does. ``nesting`` holds ``[callee or
        None, commas so far]`` for each `(` open before the line and is
        kept up; ``skipped`` tells whether the call at a column is none. A
        bare call written as an argument of another gets the origin
        `^callee^position` of that one: a macro may qualify it."""
        code = code.replace("->", "?.")
        self.begin_line(code, index + 1, source_id, self._source[index])
        matches = {match.start(1): match for match in _RE_CALL.finditer(code)}
        parens = {match.end() - 1: match[1] for match in matches.values()}
        for column, char in enumerate(code):
            if char == "(":
                nesting.append([parens.get(column), 0])
            elif char in "),":
                if nesting and char == ",":
                    nesting[-1][1] += 1
                elif nesting:
                    nesting.pop()
            match = matches.get(column)
            if not match:
                continue
            callee = match[1].replace("?.", "->")
            arguments = match[0][len(match[1]) : -1].strip()
            if arguments.startswith("<"):
                # `f<T>`: what a template call returns may be written with
                # its arguments.
                callee += arguments
            name = re.split(r"::|->|\.", callee)[-1]
            if name in _C_CONTROL_KEYWORDS or (skipped and skipped(match.start())):
                continue
            before = code[: match.start()]
            if _RE_TYPE_KEYWORD.search(before):
                # `struct name (*values)[n];` declares too.
                continue
            declared = name == callee and _RE_DECLARES.search(before)
            if declared:
                # `Type name(args);` declares `name`.
                self.declared.setdefault(name, []).append(
                    (index + 1, f"{declared['type']}()", False)
                )
                continue
            origin = None
            outer = nesting[-1] if nesting else (None, 0)
            if callee in self._generic:
                # A type parameter is the function's own.
                origin = f"={callee}"
            elif self._dependent(callee):
                # What a member of one is depends on its argument.
                origin = f"?.{name}"
            elif (
                name == callee
                and (outer[0] or "").isidentifier()
                and (source_id, name) not in self._locals
                and not before.rstrip().endswith(".")
            ):
                origin = f"^{outer[0]}^{outer[1]}"
            self.add_call(
                source_id,
                callee,
                index + 1,
                since,
                match.start(1),
                match.end() - 1,
                origin,
            )
            if index in self._guarded:
                self.builder.edges[-1]["metadata"]["guards"] = sorted(
                    self._guarded[index]
                )
        binding = self.binding_re.search(code)
        value = code[binding.end() :].lstrip() if binding else "["
        if value[:1] != "[" and not _casts(value):
            # `auto name = call...` binds `name` to what the last call to
            # close in its statement, outside any braces, returns; a lambda
            # and a cast bind nothing.
            depth, held = 0, None
            for column in range(binding.end(), len(code)):
                depth += (code[column] == "{") - (code[column] == "}")
                if code[column] == ";" and not depth:
                    break
                if not depth:
                    held = self._closes.get(column, held)
            if held is not None:
                self.declared.setdefault(binding["name"], []).append(
                    (index + 1, f"#{held}", False)
                )
                self.add_local(source_id, binding["name"])

    def call_origin(self, callee, lineno, since, owner=None):
        # `value.member`, `value->member` and longer paths. A value that
        # nothing declares is, in a member function, a field of its type;
        # `->` on a smart pointer reaches what it points at, `.` the
        # pointer itself.
        root, *steps = _RE_STEP.split(callee)
        if not steps or "::" in root:
            return None
        held = None if root == "this" else self._declared(root, lineno, since)
        if held and self._dependent(held):
            return None
        held = held and self._opaque(held)
        pointer = held and _RE_POINTER.match(held)
        if held == "?":
            origin = held
        elif held is None:
            if root == "this" and len(steps) == 2 or not self._member:
                return None
            origin = "self" if root == "this" else f"self.{root}"
        elif pointer:
            # `->` reaches what the pointer points at; `.` reaches the
            # pointer's own members — its type argument intact, so a
            # `get()`/`operator->` return is of it.
            origin = (
                f"{pointer[2]}()"
                if steps[0] == "->"
                else f"{pointer[1]}<{pointer[2]}>()"
            )
        else:
            origin = held if held[0] in "#=" else f"{_held(held[:-2])}()"
        for arrow, name in zip(steps[::2], steps[1::2]):
            name = name.split("<")[0]
            origin += f".?.{name}" if arrow == "->" else f".{name}"
        return origin
