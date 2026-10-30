"""The JavaScript and TypeScript parser."""

import re
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import accumulate, chain, islice
from operator import itemgetter

from descry import _stdlib
from descry._parsing import (
    _JS_BLOCK_OPENER_RE,
    _JS_CONTINUES_EXPRESSION,
    _JS_SPACE,
    _JSTS_CONTROL_KEYWORDS,
    _NON_CODE_RE,
    _NUMBER_DOT,
    _NUMBER_DOT_END_RE,
    _SPACES,
    _TYPE_ALIAS_RE,
    BaseParser,
    _blank,
    _bound_origin,
    _bracketed_parts,
    _generic_metadata,
    _innermost,
    _is_call_chain,
    _js_brackets,
    _js_opens_block,
    _named,
    _nested,
    _opens_body,
    _run_start,
    _spaced,
    _statement_end,
    _strip_generics,
    _strip_parens,
    source_lines,
    token_count,
)


def lines_after(lines, row):
    """The lines of ``lines`` after line ``row``, read as they are asked for,
    not copied."""
    return (lines[k] for k in range(row + 1, len(lines)))


# What a loop starts with
_LOOP_RE = re.compile(r"(?<![\w$.])(?:for|while|do)\b")


_JS_START = rf"(?:^|(?<=[;\r\u2028\u2029]))[{_JS_SPACE}]*"


_JS_WHITE = f"{_JS_SPACE}\n\r\u2028\u2029"


# What TypeScript's scanner takes as white space beyond ECMAScript's
_TS_ONLY_SPACE_RE = re.compile("[\x85\u200b]")


_JS_GAP = f"[{_JS_WHITE}]"


# A TypeScript alias written on one line: its name, its type parameters
# and the type it stands for
_TS_ALIAS_RE = re.compile(
    rf"{_JS_START}(?:export\s+)?(?:declare\s+)?type\s+([\w$]+)\s*"
    r"(?:<((?:[^<>]|<[^<>]*>)*)>)?\s*=\s*"
    # A function type's parameter list may go on over lines.
    rf"(\((?:[^()]|\([^()]*\))*\)\s*=>[^;\n]+?|[^;\n]+?);?[{_JS_SPACE}]*$",
    re.MULTILINE,
)


# Type arguments, nested three deep, that name no function type
_TS_PLAIN = r"[^<>();={}]*"


_TS_ARGUMENTS = (
    rf"<{_TS_PLAIN}(?:<{_TS_PLAIN}(?:<{_TS_PLAIN}>{_TS_PLAIN})*>{_TS_PLAIN})*>"
)


_TS_DECLARATIONS = (
    # `name: Type` on a parameter (alone on its line or not), variable or
    # field: a primitive, or a named type with its type arguments, either as
    # an array and whether or not it may be undefined or null instead
    re.compile(
        r"(?:^|[(,]|\b(?:const|let|var|private|public|protected|readonly)\s)\s*"
        r"(?:\.\.\.)?(?P<name>[A-Za-z_$][\w$]*)\??\s*:\s*(?P<type>(?:(?:[a-z_$][\w$]*\.)*"
        rf"[A-Z$][\w$.]*(?:{_TS_ARGUMENTS})?|string|number|boolean|bigint|symbol"
        r"|typeof\s+[\w$]+(?:\.[\w$]+)*)"
        r"(?:\[\])*)(?:\s*\|\s*(?:undefined|null))*(?=\s*(?:[,)=;]|$))"
    ),
    # `name = new Type(...)`, the construction the whole value
    re.compile(
        r"\b(?P<name>[A-Za-z_$][\w$]*)\s*=\s*new\s+"
        rf"(?P<type>[A-Za-z_$][\w$.]*(?:{_TS_ARGUMENTS})?)"
        r"(?=(?:\s*<[^;]*?>)?\s*\((?:[^()]|\([^()]*\))*\)\s*(?:[;,)]|$)"
        # or what goes on over the next lines, or none, `new Type;`
        r"|(?:\s*<[^;]*?>)?\s*\([^)]*$|\s*<[^;>]*$|\s*(?:[;,)]|$))"
    ),
    # `name = value as Type`, where the statement ends with the type and
    # no operator before it binds less tightly than `as`
    re.compile(
        r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=[^;?|&=]*\bas\s+"
        rf"(?P<type>(?:[a-z_$][\w$]*\.)*[A-Z$][\w$.]*(?:{_TS_ARGUMENTS})?(?:\[\])*)"
        r"\s*;?\s*$"
    ),
    # A type parameter's bound: `<T extends Type`
    re.compile(
        r"[<,]\s*(?P<param>[A-Z]\w*)\s+extends\s+(?P<type>[A-Z][\w$.]*)\b(?![<(])"
    ),
)


# Where JavaScript declares names: a variable or what a pattern takes apart,
# a function, a type, and what a `catch` is handed. A member's name, written
# after a `.` or a `#`, is read as one and binds nothing.
_JS_DECLARED_RE = re.compile(
    _NUMBER_DOT + r"|[.#]\s*[\w$]+"
    r"|\b(?:const|let|var|function\*?|class|interface|type|enum|namespace)\s+"
    r"([\w$]+|\{[^;]*?\}|\[[^;]*?\])|\bcatch\s*\(\s*([\w$]+|\{[^)]*\}|\[[^)]*\])"
)


# A parameter list, as a function's is followed by its body or its `=>`,
# and the one parameter written without a list
_JS_PARAMETERS_RE = re.compile(
    r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*(?::[^=;{()]*)?(?:=>|\{)"
    r"|(?<![\w$])([\w$]+)\s*=>"
)


def _js_member_name(lines, row, column):
    """Whether the word at column ``column`` of line ``row`` of ``lines`` is
    a member's name: written after a `.` or a `#`, on its line or one
    above, where the `.` ends no number."""
    line = lines[row]
    end = _run_start(line, column)
    while not end and row > 0:
        row -= 1
        line = lines[row]
        end = _run_start(line, len(line))
    digits = end - 1
    while digits > 0 and (line[digits - 1].isdecimal() or line[digits - 1] == "_"):
        digits -= 1
    return line[end - 1 : end] in (".", "#") and not _NUMBER_DOT_END_RE.match(
        line, digits, end
    )


def _js_bound(content):
    """The names the JavaScript source ``content`` declares or takes as
    parameters, in any of its scopes."""
    held = [
        name
        for found in _JS_DECLARED_RE.findall(content)
        for name, _ in _js_bindings("".join(found))
    ]
    for found in _JS_PARAMETERS_RE.finditer(content):
        if not _JS_CONTROL_HEAD_RE.search(
            content, max(0, found.start() - 12), found.start()
        ):
            # A parameter's name or pattern stands before its type and default.
            parts = _bracketed_parts((found[1] or "") + ")")
            held += [
                name
                for p in parts
                for name in _bracketed_parts(re.sub("[:=]", ",", p) + ")")[:1]
            ]
            held.append(found[2] or "")
    return set(re.findall(r"[A-Za-z_$][\w$]*", " ".join(held)))


# The name of a class member, `#` before a private one
_TS_MEMBER_NAME = r"#?[A-Za-z0-9_$]+"


# A member's name as the source writes it: a plain one, or a string literal
# with its quotes, a name no plain method has. A call's name is always a
# plain one: a quoted string is a value.
_TS_WRITTEN_MEMBER_NAME = r"#?(?:[A-Za-z0-9_$]+|\"[^\"\n]*\"|'[^'\n]*')"


_TS_VARIABLE_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)")


# The name of a property in an object literal: `name: value`, quoted or not
_TS_PROPERTY_RE = re.compile(rf"^\s*({_TS_WRITTEN_MEMBER_NAME})(?=\s*:)")


# A function's prototype assigned an object other than a literal,
# `F.prototype = value;`: the function's name and the value
_JS_PROTOTYPE_RE = re.compile(
    r"^\s*([\w$]+)\.prototype\s*=(?![=>])(?!\s*\{\s*$)\s*(.+?)[\s;]*$"
)


# `exports = module.exports = …`: the alias the line also gives the new
# exports, before the `module.exports` assignment
_CJS_ALIASED_RE = re.compile(r"\s*exports\s*=\s*(?=module\.exports\s*=(?![=>]))")


# `exports = …` at the start of a statement, or after `module.exports =`
_CJS_EXPORTS_RE = re.compile(r"\s*exports\s*=(?![=>])")


# A write of a CommonJS module's exports anywhere: `module.exports =`,
# `exports =`, `exports.name =`, `module.exports.name =`, each by any
# assignment operator, a member's of a computed name, `exports[key] =`, or
# a library function that puts members on the exports it is passed first
# (`Object.assign`, as the data's `writers` has them). Groups: `via` the
# `module.` it is written through, `member` the member's name, `unnamed`
# where the name is not written.
_JS_WRITERS = "|".join(
    r"\s*\.\s*".join(map(re.escape, writer.split(".")))
    for writer in sorted(_stdlib.section("typescript", "writers"))
)
_CJS_WRITE_RE = re.compile(
    rf"(?<![\w$.])(?:(?:{_JS_WRITERS})\s*\(\s*"
    r"(?P<unnamed>(?P<target>module\s*\.\s*)?exports)\b(?!\s*[.\[])"
    r"|(?P<via>module\s*\.\s*)?exports"
    r"(?:\s*\.\s*(?P<member>[\w$]+)|\s*(?P<key>\[)[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])?"
    r"\s*(?:\*\*|<<|>>>?|&&|\|\||\?\?|[-+*/%&|^])?=(?![=>]))"
)


# `require("m")`, a value that is the module's exports
_JS_REQUIRED_VALUE_RE = re.compile(r"""\s*require\(\s*['"]([^'"]+)['"]\s*\)\s*;?\s*""")


# A member assigned to a function's prototype, `F.prototype.name = value`,
# to the function, `F.name = value`, or to a CommonJS module's exports,
# `module.exports.name = value`: the member's name, what it is assigned to
# and `.prototype` for the first
_JS_ASSIGNED_MEMBER_RE = re.compile(
    r"^\s*(?=(?:module\.)?[\w$]+(?:\.prototype)?\.(?!prototype\b)([\w$]+)\s*=(?![=>]))"
    r"((?:module\.(?=exports\.))?[\w$]+)(\.prototype(?=\.))?"
)


# What a name declares a function or a class by
_JS_DECLARING_KINDS = frozenset({"function", "class", "var", "let", "const"})


def _prototype_base(value: str) -> str | None:
    """The base of a prototype assigned ``value``: `G` for `new G(...)`,
    `Object.create(G.prototype, ...)` or `G.prototype`, none for
    `Object.create(null)`, `null` or `undefined`, else the value as
    written, an expression whose type is not told."""
    found = re.fullmatch(
        r"new\s+([\w$.]+)\s*(?:\(.*\))?"
        r"|Object\.create\(\s*(?:([\w$.]+)\.prototype\b.*|null\s*)\)"
        r"|([\w$.]+)\.prototype|null|undefined",
        value,
    )
    return (found[1] or found[2] or found[3]) if found else value


def _ts_unquote(name: str) -> str:
    """A member's name as the source names it: a string-literal key's
    content, its quotes syntax only — `"~validate"` and `~validate` are
    one name, as `{"m": f}` and `{m: f}` are one property."""
    if len(name) > 1 and name[0] == name[-1] and name[0] in "\"'":
        return name[1:-1]
    return name


def _function_parameters(value: str) -> str:
    """The parameter list, without its parentheses, of the function
    ``value`` writes from its parameters on: `(a, b) => ...`, `a => ...` or
    `function (a) {`."""
    value = re.sub(
        r"\s*(?:async\s+)?(?:function\b\s*\*?\s*[\w$]*\s*)?", "", value, count=1
    )
    if not value.startswith("("):
        return value.partition("=>")[0].strip()
    depth = 0
    for at, char in enumerate(value):
        depth += (char == "(") - (char == ")")
        if not depth:
            return value[1:at]
    return value[1:]


def _path_written(written: str) -> str:
    """A TypeScript type as an origin path writes it: `(`/`)` as `<`/`>`,
    `.` as `::`."""
    return written.replace("(", "<").replace(")", ">").replace(".", "::")


def _value_named(statement: str) -> re.Match | None:
    """The match of the name ``statement`` gives a value, in group 1: a
    variable's, a property's or a member's assigned to a function or its
    prototype."""
    return (
        _TS_VARIABLE_RE.match(statement)
        or _TS_PROPERTY_RE.match(statement)
        or _JS_ASSIGNED_MEMBER_RE.match(statement)
    )


def _arrow_function(statement: str) -> tuple[str, str] | None:
    """Read `const name = (params) => ...`, with or without a type
    annotation on ``name``, `const name = function (params) ...`, or either
    function as the value of a property, `name: (params) => ...`, or of a
    prototype's member, `F.prototype.name = function (params) ...`. Returns
    the name and the function from its parameters on, else None."""
    match = _value_named(statement)
    if not match:
        return None
    # The initialiser follows the first `=` outside brackets that is not
    # the `=>` of a function type, the value of a property its `:`.
    assign = ":" if match.re is _TS_PROPERTY_RE else "="
    depth = 0
    for column in range(match.end(), len(statement)):
        char = statement[column]
        if char in "([{<":
            depth += 1
        elif char in ")]}>" and statement[column - 1] != "=":
            depth -= 1
        elif (
            char == assign and depth == 0 and statement[column + 1 : column + 2] != ">"
        ):
            break
    else:
        return None
    value = _function_value(statement[column + 1 :])
    return (_ts_unquote(match.group(1)), value) if value else None


_TS_OBJECT_PROPERTY_RE = re.compile(r"\s*(?:readonly\s+)?([\w$]+)\??\s*:\s*\{")


_TS_FUNCTION_PROPERTY_RE = re.compile(
    r"\s*(?:readonly\s+)?([\w$]+)\??\s*:\s*(<[^()=]*>)?\s*\("
)


def _function_value(text: str) -> str | None:
    """The function ``text``, the value given a name, is written as, from
    its parameters on, where it is an arrow function or a `function`
    expression; else None."""
    # `((params) => ...) as Type` holds the function in parentheses.
    value = re.sub(r"^\s*(?:\((?=\())?(?:async\b\s*)?", "", text)
    if value.startswith("<"):
        # Type parameters before the parameters: `<T,>(x: T) => ...`; the
        # `>` of a `=>` in one of their defaults closes nothing.
        depth = 0
        for column, char in enumerate(value):
            if char == "<":
                depth += 1
            elif char == ">" and value[column - 1] != "=":
                depth -= 1
                if depth == 0:
                    value = value[column + 1 :].lstrip()
                    break
    if re.match(r"function\b|[A-Za-z_$][\w$]*\s*=>", value):
        return value
    if not value.startswith("("):
        return None
    depth = 0
    for column, char in enumerate(value):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            # `=>` follows the parameters, after any return type.
            rest = value[column + 1 :].lstrip()
            is_arrow = rest.startswith("=>") or (rest.startswith(":") and "=>" in rest)
            return value if is_arrow else None
    return None


def _bounded_parameters(text: str) -> dict[str, str]:
    """``{name: bound}`` of the type parameters ``text`` opens with,
    `<const T extends Bound = Default, U>`, the bound "" for one of none."""
    text = text.lstrip()
    found = {}
    for part in _bracketed_parts(text[1:]) if text.startswith("<") else ():
        name, _, bound = re.split(r"\s*=(?!>)", part)[0].partition(" extends ")
        if name.split():
            found[name.split()[-1]] = bound.strip()
    return found


def _ts_type_member(lines: list[str], index: int) -> tuple[str, str] | None:
    """``(name, type)`` of a property that line ``index`` of ``lines``, code
    lines, declares in an interface or an object type: a function type as
    ``_ts_function_property`` writes it, or an object type whole, over the
    lines it goes on to. None for any other line."""
    if written := _ts_function_property(lines[index]):
        return written
    found = _TS_OBJECT_PROPERTY_RE.match(lines[index])
    if not found:
        return None
    text = "\n".join(lines[index : index + 40])
    close = _block_close(text, found.end() - 1)
    return close and (found[1], " ".join(text[found.end() - 1 : close + 1].split()))


def _ts_function_property(code: str) -> tuple[str, str] | None:
    """``(name, signature)`` of a property ``code`` declares with a
    function type, `name?: <T>(a: A) => R;`, the signature written as a
    method's, `name<T>(a: A): R`; None for any other line."""
    found = _TS_FUNCTION_PROPERTY_RE.match(code)
    if not found:
        return None
    depth = 0
    for column in range(found.end() - 1, len(code)):
        depth += (code[column] == "(") - (code[column] == ")")
        if not depth:
            returned = re.match(r"\s*=>\s*(.+?)\s*[;,]?\s*$", code[column + 1 :])
            params = code[found.end() - 1 : column + 1]
            if returned:
                return found[1], f"{found[1]}{found[2] or ''}{params}: {returned[1]}"
            return None
    return None


def _class_property(
    code: str, start: int = 0, text: str | None = None
) -> tuple[str, int, int] | None:
    """A property declared at column ``start`` of ``code`` with a value:
    its name, the column its name ends at, and the column its value starts
    at, past an annotation of any shape (`t: (a: A) => B = ...`). Where
    ``code`` has its literals blanked, ``text`` is the line with them, which
    a quoted name is read from."""
    named = _TS_CLASS_PROPERTY_RE.match(code, start)
    if not named:
        return None
    depth = 0
    for column in range(named.end(), len(code)):
        char = code[column]
        if char in "([{<":
            depth += 1
        elif char in ")]}>" and code[column - 1] != "=":
            depth -= 1
        elif char == ";" and depth == 0:
            return None
        elif (
            char == "="
            and depth == 0
            and code[column + 1 : column + 2] not in ("=", ">")
            and code[column - 1] not in "!<>="
        ):
            value = (
                column + 1 + len(code[column + 1 :]) - len(code[column + 1 :].lstrip())
            )
            written = (text or code)[named.start(1) : named.end(1)]
            return _ts_unquote(written), named.end(1), value
        if depth < 0:
            return None
    return None


def _member_written(
    line: str, in_class: bool, method_re, value: str | None = None
) -> tuple[str, int, int, bool] | None:
    """The member ``line`` opens: its name, the columns its name starts and
    ends at, and whether a value is assigned to it. `name(` and `name = (`
    as ``method_re`` reads them, and in a class a property holding a
    function whatever its annotation, on its own line or over the lines of
    its value, ``value`` being that value's text."""
    if match := method_re.search(line):
        return (
            _ts_unquote(match.group(1)),
            match.start(1),
            match.end(1),
            bool(match.group(2)),
        )
    if (
        in_class
        and (prop := _class_property(line))
        and _function_value(value if value is not None else line[prop[2] :])
    ):
        return prop[0], prop[1] - len(prop[0]), prop[1], True
    return None


# A reserved word that takes what follows it, after which a line break ends
# no statement
_JS_TAKES_NEXT_RE = re.compile(
    r"(?<![\w$.])(?:import|export|new|typeof|void|delete|in|instanceof|case|extends)$"
)


# A parameter, a name or a pattern that takes its value apart, and the
# type it is declared, a named, an object or a tuple type: `{ a, b }: Pair`
_TS_TYPED_PARAMETER_RE = re.compile(
    r"\s*([\[{].*[\]}]|[A-Za-z_$][\w$]*)\s*\??\s*:\s*"
    r"((?:[a-z_$][\w$]*\.)*[A-Z$][\w$.]*(?:<.*>)?(?:\[\])*|\{.*\}|(?:readonly\s+)?\[.*\])"
    r"\s*(?:=.*)?",
    re.DOTALL,
)


# A call that loads a module: `require(...)`, `import(...)`
_JS_LOADS_RE = re.compile(r"(?<![\w$.])(?:require|import)\s*\(")

# A directive of a prologue: a string literal alone as a statement
_JS_DIRECTIVE_RE = re.compile(
    r"\s*(?P<quote>[\"'])(?P<body>[^\"'\n]*)(?P=quote)\s*(?:;|(?=\n|\}))"
)

# A function or class declaration a statement starts with, and its name
_JS_DECLARATION_RE = re.compile(
    rf"(?:^|(?<=[;{{}}]))[{_JS_SPACE}]*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
    r"(?P<kind>function|class)\b\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


# An operator that gives what is on its left a value: `=`, or one that
# may, `&&=`, `||=` or `??=`. One that computes, as `+=`, leaves a string
# a string and a number a number.
_JS_ASSIGN_RE = re.compile(r"(?<![=!<>])(?:&&|\|\||\?\?)?=(?![=>])")


# Modifiers a class member is declared behind, in any order
_TS_MODIFIER_WORDS = (
    "private",
    "public",
    "protected",
    "static",
    "abstract",
    "override",
    "declare",
    "readonly",
    "async",
)


_TS_MEMBER_MODIFIERS = rf"(?:(?:{'|'.join(_TS_MODIFIER_WORDS)})\s+)*"


# The words a member's name follows: its modifiers, an accessor's and a
# statement's export
_JS_DECLARING_WORDS = frozenset(
    (*_TS_MODIFIER_WORDS, "get", "set", "accessor", "export", "default")
)


# The words a name after which starts no statement or member: those an
# expression follows, and those a statement that is their body follows
_JS_LEADING_WORDS = frozenset(
    {
        "return",
        "case",
        "await",
        "new",
        "typeof",
        "void",
        "yield",
        "throw",
        "in",
        "of",
        "instanceof",
        "delete",
        "extends",
        "implements",
        "keyof",
        "else",
        "do",
    }
)


# A name, a keyword or a number that ends where the search ends
_JS_WORD_END_RE = re.compile(r"[\w$#]+\Z")


# A decorator without arguments, or the name of one before its arguments
_JS_DECORATOR_END_RE = re.compile(r"@[\w$.]+\s*\Z")


# The label of a `switch`'s case
_JS_CASE_RE = re.compile(r"(?<![\w$.])(?:case\b|default\s*:)")


# A label's name and its `:`, which the code before a block ends with
_JS_LABEL_END_RE = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*:\Z")


# A `break` of a label, and the label
_JS_BREAK_RE = re.compile(r"(?<![\w$.])break\s+([A-Za-z_$][\w$]*)(?![\w$])")


# `null` or `undefined`, a value of no members
_JS_VOID_RE = re.compile(r"(?:null|undefined)\s*")


# The space after a declarator's name and the first character after it
_JS_FOLLOWS_RE = re.compile(r"(\s*)(.?)", re.DOTALL)


# A construction up to the `(` of its arguments: `new T(`, `new T<A>(`
_JS_CONSTRUCTION_RE = re.compile(r"\s*new\s+([A-Za-z_$][\w$.]*)\s*(?:<[^()]*>)?\s*\(")


# A `case` the code before a bracket ends with
_JS_CASE_END_RE = re.compile(r"(?<![\w$.])case\s*\Z")


# A head that makes the statement after it a body that may not run
_JS_BODY_HEAD_RE = re.compile(r"(?<![\w$.])(?:else|case|default)\s*\Z")


# What a statement holds before the `:` of a label: a `case` and its
# expression, `default` or the label's name
_JS_LABELLED_RE = re.compile(r"\s*(?:case\b.*|default|[A-Za-z_$][\w$]*)\s*", re.DOTALL)


# What a block opens after, but for a `)`, a `;` or a brace: a bare
# `catch`, a class's `static`, a namespace's or a module's name, or
# `declare global`
_JS_BLOCK_HEAD_RE = re.compile(
    r"(?<![\w$.])(?:catch|static|(?:namespace|module)\s+[\w$.\"' ]+?|declare\s+global)\s*\Z"
)


# The keyword of a statement whose head is in parentheses, before its `(`
_JS_CONTROL_HEAD_RE = re.compile(
    r"(?<![\w$.])(?:if|for(?:\s+await)?|while|switch|with|catch)\s*$"
)


# The origin path of a callback's parameter: `#edge|argument|position`
_CALLBACK_RE = re.compile(r"#\d+\|")


# The type an `as` or a `satisfies` casts to, with its type arguments
_TS_CAST_TYPE_RE = re.compile(
    r"(?<![\w$.])(?:as|satisfies)\s+[\w$.]+(?:<(?:[^<>()]|<(?:[^<>()]|<[^<>()]*>)*>)*>)?"
    r"(?:\[\])*"
)


# An `import` or an `export` statement, which outside a block makes a file a
# module, as `import.meta` does anywhere
_JS_MODULE_RE = re.compile(
    rf"{_JS_START}(?:import(?!{_JS_GAP}*[(.])|export)\b", re.MULTILINE
)


_JS_IMPORT_META_RE = re.compile(rf"(?<![\w$.])import{_JS_GAP}*\.{_JS_GAP}*meta\b")


# What makes a JavaScript file a CommonJS module, as TypeScript reads one: a
# `require` call of one argument, no definition of a `require`, or a value
# given to `module.exports` or to a property of it or of `exports`, by
# assignment or `Object.defineProperty`
_JS_COMMONJS_RE = re.compile(
    r"(?<![\w$.])(?:require\s*\(\s*(?=[^\s)])(?:[^,()\[\]{}]|\((?:[^()]|\([^()]*\))*\)"
    r"|\[[^\[\]]*\]|\{[^{}]*\})+\)(?!\s*\{)"
    r"|module\s*\.\s*exports\s*(?:(?:\.\s*[\w$]+|\[[^\]]*\])\s*)?="
    r"|exports\s*(?:\.\s*[\w$]+|\[[^\]]*\])\s*="
    r"|Object\s*\.\s*defineProperty\s*\(\s*(?:module\s*\.\s*)?exports\b)(?!=)"
)


# The one parameter an arrow function writes without parentheses
_JS_LONE_PARAM_RE = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*=>")


def _place(starts, offset):
    """``(line, column)`` of ``offset`` in a text whose lines start at the
    offsets ``starts``; lines count from 1."""
    row = bisect_right(starts, offset) - 1
    return row + 1, offset - starts[row]


class _JsScopes:
    """Where each name a JavaScript or TypeScript file binds is in scope,
    read from its code lines, literals blank: what a declaration, a
    parameter list, a `catch` or `for` head, a named function expression
    or an assignment binds. Each binding is ``(kind, at, written, span)``:
    the `(line, column)` it holds from, the first and last `(line, column)`
    of where its name and the value it is given are written, and the span
    the language gives it, the block it is in, for a `var` its function,
    for a parameter its function, for what a `catch` or a `for` head binds
    the statement the head leads, for an assignment that may not run, as
    in a branch, its own text. Lines count from 1."""

    def __init__(self, lines, raw=(), script=False):
        self.lines, self.raw, self.script = lines, raw, script
        self.text, self.starts, self.closer = _js_brackets(lines)
        self.opener = {close: opened for opened, close in self.closer.items()}
        self._brackets = _nested(sorted(self.closer.items()))
        self.params = self._param_regions()
        self.whole = (1, 0), (len(lines) + 1, 0)
        # The `{` of each function's body
        self._bodies = {
            self.opener.get(self.starts[r[3][0] - 1] + r[3][1])
            for held in self.params.values()
            for r in held
        }
        opens = sorted(at for at in self.closer if self.text[at] == "{")
        self._blocks = _nested([(at, self.closer[at]) for at in opens])
        self._classes = self._class_bodies()
        # A class's body is read when the class is made, as a function's is
        # when it is called.
        functions = {(r[0], r[3]) for held in self.params.values() for r in held}
        functions |= {
            (self.place(at), self.place(self.closer[at])) for at in self._classes
        }
        self._functions = _nested(
            sorted(functions, key=lambda f: (f[0], -f[1][0], -f[1][1]))
        )
        # Whether each `{` opens a block, read outer first
        self._opening = {}
        for at in opens:
            self._opening[at] = self._block(at)
        # What reading back from a token gives (see ``_walk``), whether a
        # block runs its statements (see ``_bare``), the labels of a
        # `switch` (see ``_cases``) and where an expression ends (see
        # ``_expression_end``)
        self._walks, self._bares, self._switches, self._ends = {}, {}, {}, {}
        # label -> the offset of each `break` of it
        self._breaks = {}
        for found in _JS_BREAK_RE.finditer(self.text):
            self._breaks.setdefault(found[1], []).append(found.start())
        self.bindings, self._added = {}, set()
        self._declarations()
        # Where each loop statement starts and just past where it ends
        self._loops = [
            (at, (end[0] + 1, end[1]))
            for found in _LOOP_RE.finditer(self.text)
            for at in [self.place(found.start())]
            for end in [self.past_statement(at[0] - 1, at[1])]
        ]
        # Their starts, and the furthest any of them up to each reaches
        self._loop_starts = [at for at, _ in self._loops]
        self._loop_reach = list(accumulate((end for _, end in self._loops), max))
        for held in self.params.values():
            for start, close, names, end, _ in held:
                for name in names:
                    self._add(name, "param", start, (start, close), (start, end))
        # Each name's bindings, indexed by their span for a read to find
        # only those around it, and by the lines they are written over.
        self._indexed, self._rows = {}, {}
        # name -> the last span's part of ``assignments`` and its functions,
        # the values the ``value`` reader last gave, and the state of the
        # last ``reassigned_values`` read: one entry per name keeps what is
        # retained linear in the input while repeat reads reuse it.
        self._acache, self._vcache, self._rstate = {}, {}, {}
        for name in self.bindings:
            self._index(name)

    def _index(self, name):
        """Index the bindings of ``name`` by position: their spans, nested
        as the scopes they are, each with the bindings of it in the order
        they hold from, and the lines each is written over, sorted."""
        by_span = {}
        for rank, binding in enumerate(self.bindings[name]):
            by_span.setdefault(binding[3], []).append((rank, binding))
        # Of each span, also where the run of assignments by one function
        # each is in starts, and which are declarations
        positions, runs, declared = {}, {}, {}
        for span, held in by_span.items():
            held.sort(key=lambda entry: entry[1][1])
            positions[span] = [entry[1][1] for entry in held]
            declared[span] = [i for i, (_, b) in enumerate(held) if b[0] != "="]
            run, key = [], None
            for i, (_, binding) in enumerate(held):
                by = self.function(binding[1]) if binding[0] == "=" else i
                run.append(run[-1] if run and by == key else i)
                key = by
            runs[span] = run
        nested = _nested(sorted(by_span, key=lambda s: (s[0], -s[1][0], -s[1][1])))
        self._indexed[name] = (
            nested,
            by_span,
            {entry[:2]: i for i, entry in enumerate(nested)},
            positions,
            runs,
            declared,
        )
        # The lines each binding is written over, sorted, and the last line
        # any of them up to each reaches
        rows = sorted(
            (binding[2][0][0], binding[2][1][0]) for binding in self.bindings[name]
        )
        self._rows[name] = rows, list(accumulate((row[1] for row in rows), max))

    def _param_regions(self):
        """``{line: [(start, close, names, end, written)]}``: each parameter
        list of a function or signature in the file on each line it is
        written over, with the `(line, column)` it starts and ends at, the
        names it binds, the `(line, column)` where what it is in scope in
        ends, its function's body or the signature itself, and the list as
        written."""
        text, starts, closer, lines = self.text, self.starts, self.closer, self.lines
        regions, arrows = {}, set()

        def add(opened, closed, declared):
            found = _js_function_end(text, closed + 1, closer, lines, starts, declared)
            if found is None or found[0] is None:
                return
            arrows.add(found[1])
            start, close, end = (
                (row + 1, at - starts[row])
                for at in (opened, closed, found[0])
                for row in [bisect_right(starts, at) - 1]
            )
            written = text[opened : closed + 1]
            names = _ts_params(written if written[0] == "(" else f"{written} =>")
            for line in range(start[0], close[0] + 1):
                regions.setdefault(line, []).append((start, close, names, end, written))

        for found in re.finditer(r"\(", text):
            opened = found.start()
            if (
                opened in closer
                and not _JS_CONTROL_HEAD_RE.search(text, max(opened - 12, 0), opened)
                and not self._case_label(opened)
            ):
                add(opened, closer[opened], _js_declares(text, opened, self.opener))
        # A name before `=>` is a parameter, but where it ends the return type
        # of a parameter list before it, `(a): T =>`.
        for found in _JS_LONE_PARAM_RE.finditer(text):
            if found.end() not in arrows:
                add(found.start(1), found.end(1) - 1, False)
        return regions

    def place(self, offset):
        return _place(self.starts, offset)

    def block(self, offset):
        """The span of the innermost `{`...`}` around ``offset``, else the
        whole file."""
        held = _innermost(self._blocks, offset)
        return (self.place(held[0]), self.place(held[1])) if held else self.whole

    def _strict(self, offset):
        """Whether the code at ``offset`` is strict: all of a module's, and
        in a script that in a class body, in a function or a script whose
        directive prologue has `"use strict"`."""
        if not self.script:
            return True
        around = [
            at
            for at in (*self._bodies, *self._classes)
            if at is not None and at < offset <= self.closer.get(at, -1)
        ]
        first = self.starts[1] if self.raw[:1] and self.raw[0].startswith("#!") else 0
        return self._prologue_strict(first) or any(
            at in self._classes or self._prologue_strict(at + 1) for at in around
        )

    def _prologue_strict(self, at):
        """Whether the directive prologue at offset ``at`` holds
        `"use strict"`, as the source writes it."""
        while found := _JS_DIRECTIVE_RE.match(self.text, at):
            row, column = self.place(found.start("quote"))
            size = found.end("body") - found.start("body")
            if self.raw[row - 1][column + 1 : column + 1 + size] == "use strict":
                return True
            at = found.end()
        return False

    def _lexical_between(self, name, at, outer):
        """Whether ``name`` is a parameter of the function span ``outer``, or
        a `let`, `const` or `class` of it is in scope at ``at`` from a block
        inside that span."""
        params = (r for held in self.params.values() for r in held)
        return any(name in r[2] and (r[0], r[3]) == outer for r in params) or any(
            kind in ("let", "const", "class")
            and span[0] <= at <= span[1]
            and outer[0] <= span[0]
            and span[1] <= outer[1]
            for kind, _, _, span in self.bindings.get(name, ())
        )

    def written(self, binding):
        """The code where ``binding``'s name and its value are written."""
        (first, start), (last, end) = binding[2]
        return self.text[
            self.starts[first - 1] + start : self.starts[last - 1] + end + 1
        ]

    def function(self, at):
        """The span of the innermost function or class body around ``at``,
        else None."""
        return _innermost(self._functions, at)

    def in_block(self, binding):
        """Whether ``binding`` holds in a block that is no function's body:
        a scope that no definition is."""
        span = binding[3]
        if span in (self.whole, self.function(binding[1])):
            return False
        return self.starts[span[0][0] - 1] + span[0][1] not in self._bodies

    def always_runs(self, offset):
        """Whether the statement at ``offset`` runs each time its file does:
        no function or class body is around it, and each block around it is
        a statement of its own."""
        if self.function(self.place(offset)):
            return False
        index = bisect_left(self._blocks, (offset,)) - 1
        while index >= 0:
            start, end, outer = self._blocks[index]
            if offset <= end and not self._bare((start, end)):
                return False
            index = outer
        return True

    def _class_bodies(self):
        """The offsets of the `{` of each class's body."""
        text, bodies = self.text, set()
        for found in re.finditer(r"(?<![\w$.])class\b(?!\s*[:=(),])", text):
            at, angles = found.end(), 0
            while at < len(text) and text[at] not in ";})":
                if text[at] in "([":
                    at = self.closer.get(at, at) + 1
                    continue
                angles += (text[at] == "<") - (text[at] == ">")
                if text[at] == "{" and not angles:
                    bodies.add(at)
                    break
                at += 1
        return bodies

    def _add(self, name, kind, at, lines, span):
        binding = (kind, at, lines, span)
        if (name, binding) not in self._added:
            self._added.add((name, binding))
            self.bindings.setdefault(name, []).append(binding)

    def _declarations(self):
        text, lines = self.text, self.lines
        # Where a declaration's own names and the parameters are written,
        # which no assignment is to
        bound = {
            (self.starts[s[0] - 1] + s[1], self.starts[c[0] - 1] + c[1])
            for held in self.params.values()
            for s, c, *_ in held
        }
        # line -> the offsets of the brackets of a declaration's pattern that
        # opens on an earlier line and closes on it
        self.patterns = {}
        # (name, `(line, column)`) of each declaration that gives its name
        # no value, which then holds `undefined`
        self.undefined = set()
        for found in _JS_BINDING_HEAD_RE.finditer(text):
            start = found.end()
            row, column = self.place(found.start())
            if _js_member_name(lines, row - 1, column):
                continue
            kind = found[1] or ("=" if found[2] == "for" else "catch")
            if text[start] in "[{(":
                closed = self.closer.get(start, start)
                names = [name for name, _ in _js_bindings(text[start : closed + 1])]
                after = closed + 1
            else:
                closed = after = start + len(re.match(r"[\w$]*", text[start:])[0])
                names = [text[start:closed]]
            if kind == "=" and not re.compile(r"\s+(?:of|in)\b").match(text, after):
                continue  # a `for` head that starts with an assignment
            bound.add((start, closed))
            head = kind in ("=", "catch") or _JS_FOR_HEAD_RE.search(
                text, max(found.start() - 80, 0), found.start()
            )
            if head:
                # The statement the head leads, from its `(`
                paren = text.rindex("(", 0, start)
                written = self.place(self.closer.get(paren, len(text)))
                opened = self.place(paren)
                row_after, column_after = self.past_statement(
                    *self._past(opened[0] - 1, opened[1])
                )
                span = opened, (row_after + 1, column_after)
            else:
                # The statement, which what its value is read on is
                last = _statement_end(lines, row - 1, column)
                written = self._past(row - 1, column, last)
                written = written[0] + 1, written[1]
                span = (
                    self.function(self.place(start)) or self.whole
                    if kind == "var"
                    else self.block(start)
                )
            at = self.place(start)
            for name in names:
                self._add(name, kind, at, (at, written), span)
            follows = _JS_FOLLOWS_RE.match(text, after)
            if (
                kind in ("let", "var")
                and text[start] not in "[{("
                and (
                    follows[2] in ";,}" or "\n" in follows[1] and follows[2] not in "=:"
                )
            ):
                self.undefined.add((names[0], at))
            if not head and at[0] < self.place(closed)[0]:
                self.patterns[self.place(closed)[0]] = start, closed
            if not head:
                at_row, at_column = self.place(after)
                for row, name, end, value in _declarators(lines, at_row - 1, at_column):
                    end += self.starts[row]
                    bound.add((end - len(name), end))
                    at = self.place(end - len(name))
                    self._add(name, kind, at, (at, (row + 1, len(lines[row]))), span)
                    if value is None and kind != "using":
                        self.undefined.add((name, at))
        declared = set()
        for found in _JS_DECLARATION_RE.finditer(text):
            declared.add(found.start("name"))
            at = self.place(found.start("name"))
            span = self.block(found.start())
            outer = self.function(at) or self.whole
            if (
                found["kind"] == "function"
                and not re.search(r"\basync\b|\*", found[0])
                and span != outer
                and not self._strict(found.start())
                and not self._lexical_between(found["name"], at, outer)
            ):
                # Outside strict code a plain function declared in a block
                # is its function's variable too, unless a parameter, or a
                # `let`, `const` or `class` of the name around it, would
                # clash (Annex B.3.3).
                span = outer
            self._add(found["name"], found["kind"], at, (at, at), span)
        for found in re.finditer(r"\bfunction\b\s*\*?\s*([\w$]+)\s*\(", text):
            if found.start(1) in declared:
                continue
            # A function expression's own name is in scope in it.
            opened = self.place(found.end() - 1)
            for region in self.params.get(opened[0], ()):
                if region[0] == opened:
                    self._add(
                        found[1],
                        "function",
                        opened,
                        (opened, opened),
                        (opened, region[3]),
                    )
        # A class expression's own name is in scope in its body.
        bodies = sorted(self._classes)
        for found in re.finditer(r"(?<![\w$.])class\s+([A-Za-z_$][\w$]*)", text):
            index = bisect_left(bodies, found.end())
            if found.start(1) in declared or index == len(bodies):
                continue
            body = bodies[index]
            span = self.place(body), self.place(self.closer[body])
            at = self.place(found.start(1))
            self._add(found[1], "class", at, (at, at), span)
        self._bound = _nested(sorted(bound, key=lambda b: (b[0], -b[1])))
        # line -> (name, at, start, end) of each assignment of a name, written
        # at `(line, column)` at, whose value is written on the line, from
        # column start to column end
        self.assigned = {}
        # (name, span) of each declaration
        self._declared = {
            (name, b[3])
            for name, held in self.bindings.items()
            for b in held
            if b[0] != "="
        }
        for found in _JS_ASSIGN_RE.finditer(text):
            end = _run_start(text, found.start())
            if text[end - 1 : end] in ("]", "}", ")") and end - 1 in self.opener:
                start = self.opener[end - 1]
                names = [name for name, _ in _js_bindings(text[start:end])]
            elif (
                word := _JS_WORD_END_RE.search(text, max(end - 100, 0), end)
            ) and word[0][0] not in "#0123456789":
                start, names = word.start(), [word[0]]
            else:
                continue
            row, column = self.place(start)
            if (
                not names
                or _js_member_name(lines, row - 1, column)
                or _innermost(self._bound, start + 1)
                or (span := self._reach(start, names)) is None
            ):
                continue
            # What it assigns holds from where its value ends, which reads
            # what the name held before. `null` and `undefined` have no
            # members a call through the name could reach.
            end = self._expression_end(found.end())
            value = min(_JS_FOLLOWS_RE.match(text, found.end()).end(1), end)
            if _JS_VOID_RE.fullmatch(text, value, end):
                continue
            written = (row, column), self.place(end)
            if not span or found[0] != "=":
                span = written
            for name in names:
                self._add(name, "=", written[1], written, span)
            # What the value gives each name, where it is written on its line:
            # the whole, or the element of an array literal at the name's
            # position in an array pattern.
            row = self.place(value)[0]
            given = {names[0]: (value, end)} if text[start] not in "[{" else {}
            if text[start] == "[" and self.place(self.closer.get(value, 0))[0] == row:
                elements, at = [], value + 1
                while at < self.closer[value]:
                    elements.append((at, self._expression_end(at)))
                    at = elements[-1][1] + 1
                given = {
                    name: elements[int(steps[0][1:])]
                    for name, steps in _js_bindings(text[start : found.start()])
                    if steps and len(steps) == 1 and int(steps[0][1:]) < len(elements)
                }
            for name, (first, last) in given.items():
                if row == written[1][0]:
                    columns = first - self.starts[row - 1], last - self.starts[row - 1]
                    self.assigned.setdefault(row, []).append(
                        (name, written[0], *columns)
                    )
        # A value that is itself an assignment gives what that one gives.
        for row, held in self.assigned.items():
            written = {}  # where each is written -> the first written there
            for index, (_, at, _, _) in enumerate(held):
                written.setdefault(at, index)
            for index in reversed(range(len(held))):
                name, at, first, last = held[index]
                if (inner := written.get((row, first))) is not None:
                    held[index] = name, at, *held[inner][2:]

    def _control(self, close):
        """The keyword of the control statement whose head the `)` at offset
        ``close`` closes, else None."""
        opened = self.opener.get(close)
        found = opened is not None and _JS_CONTROL_HEAD_RE.search(
            self.text, max(opened - 12, 0), opened
        )
        return found[0].split()[0] if found else None

    def _statements(self, held):
        """Whether the bracket span ``held``, None for the file, holds
        statements: the file, a block or a class's body."""
        return held is None or held[0] in self._classes or self._opening.get(held[0])

    def _block(self, at):
        """Whether the bracket at offset ``at`` opens a block: a function's
        body, or a `{` where a statement starts, as ``_js_opens_block``
        reads one, after a bare `catch`, a class's `static`, the name of a
        namespace or a module or `declare global`, or after the `:` of a
        label, a `case` or `default`."""
        text = self.text
        if text[at] != "{":
            return False
        if at in self._bodies or _js_opens_block(text, at):
            return True
        lead = _run_start(text, at)
        if _JS_BLOCK_HEAD_RE.search(text, max(lead - 100, 0), lead):
            return True
        held = _innermost(self._brackets, at)
        if text[lead - 1 : lead] != ":" or held and held[0] in self._classes:
            return False
        # The statement up to the `:`, past the brackets it closes
        start = lead - 1
        while start and text[start - 1] not in ";{}":
            start = self.opener.get(start - 1, start - 1)
        written = text[start : lead - 1]
        if self._statements(held) and any(
            _JS_LABELLED_RE.fullmatch(part)
            for part in (written, written.rpartition("\n")[2])
        ):
            return True
        # A label may prefix any statement: the name before the `:` follows
        # a controlled body's `)`, an `else`/`do`/`case`/`default`, a `,` or
        # another label's `:`.
        label = re.search(r"[A-Za-z_$][\w$]*\s*$", written)
        prefix = written[: label.start()].rstrip() if label else "x"
        return self._statements(held) and (
            not prefix
            or prefix[-1:] in "),:"
            or re.search(r"(?<![\w$.])(?:else|do|case|default)\s*$", prefix)
        )

    def _bare(self, held):
        """Whether the bracket span ``held`` is a block that is a statement
        of its own, or a label's that no `break` leaves: one that runs its
        statements each time the statement before it runs."""
        if held not in self._bares:
            self._bares[held] = self._runs(held)
        return self._bares[held]

    def _runs(self, held):
        """``_bare`` of the bracket span ``held``, read anew."""
        text = self.text
        lead = _run_start(text, held[0])
        if text[held[0]] != "{" or text[lead - 1 : lead] not in ";{}:":
            return False
        if text[lead - 1 : lead] != ":":
            return True
        label = _JS_LABEL_END_RE.search(text, max(lead - 100, 0), lead)
        if not label:
            return False
        # A label that a control's `)`, an `else`, a `case` or a `default`
        # prefixes is a body that may not run.
        lead = _run_start(text, label.start(1))
        if text[lead - 1 : lead] == ")" and self._control(lead - 1):
            return False
        if _JS_BODY_HEAD_RE.search(text, max(lead - 9, 0), lead):
            return False
        breaks = self._breaks.get(label[1], [])
        index = bisect_left(breaks, held[0])
        return not (index < len(breaks) and breaks[index] < held[1])

    def _switch(self, held):
        """Whether the bracket span ``held`` is a `switch`'s body."""
        text = self.text
        return bool(held) and self._control(_run_start(text, held[0]) - 1) == "switch"

    def _case_label(self, at):
        """Whether the bracket at offset ``at`` follows the `case` of one of a
        `switch`'s labels, and holds part of its expression: the token the
        bracket follows is `case`."""
        lead = _run_start(self.text, at)
        return bool(
            _JS_CASE_END_RE.search(self.text, max(lead - 4, 0), lead)
        ) and self._switch(_innermost(self._brackets, at))

    def _reach(self, start, names):
        """Where the left side of an assignment of ``names``, which starts at
        offset ``start``, is assigned each time the statement it is in runs:
        the span of the block the statement is in, or of the one around a
        block that is a statement of its own and declares none of them, of
        a `switch`'s body the statement's case, of the block around the
        `switch` for the first case's expression; False where it may not
        be, in a branch, an operand that may not be evaluated, a function's
        expression body or a loop's later clauses; None where it is no
        variable's: a declaration's type, a field, an enum member or a
        default in a pattern."""
        text = self.text
        decided, found = self._walk(start)
        if decided:
            return found
        at, held, direct = found
        end = _run_start(text, at)
        # A `do` runs its body at least once.
        while (opener := _JS_BLOCK_OPENER_RE.search(text, max(end - 8, 0), end)) and (
            opener[0] == "do"
        ):
            end = _run_start(text, opener.start())
        if opener or text[end - 1 : end] == ")" and self._control(end - 1):
            return False  # a statement's body without braces
        while held and self._bare(held):
            span = self.place(held[0]), self.place(held[1])
            if any((name, span) in self._declared for name in names):
                break
            held = _innermost(self._brackets, held[0])
        if not held:
            return self.whole
        if direct and held[0] in self._classes:
            return None
        end = held[1]
        if self._switch(held):
            # A `switch` runs the statements from the case it jumps to.
            starts = self._cases(held)[0]
            index = bisect_left(starts, start)
            end = starts[index] if index < len(starts) else end
        return self.place(held[0]), self.place(end)

    def _walk(self, start):
        """``(True, what _reach gives)`` where reading back from the left
        side of an assignment at offset ``start`` to where its statement
        starts decides it, else ``(False, (at, held, direct))``: the offset
        the statement starts at, the bracket span it is in, and whether the
        statement is the assignment. What is read back from each token is
        kept, so that no token is read back over twice alike."""
        text = self.text
        held = _innermost(self._brackets, start)
        # The token read is the first before the target; no bracket around
        # the target has been left; the statement is the assignment; a `:`
        # has been passed, as a label's.
        at, first, level, direct, labelled = start, True, True, True, False
        read = []  # the states read back from, past the first token
        while True:
            if not first:
                state = at, held, level, direct, labelled
                if state in self._walks:
                    found = self._walks[state]
                    break
                read.append(state)
            end = _run_start(text, at)
            if (
                "\n" in text[end:at]
                and self._statements(held)
                and not text.endswith(_JS_CONTINUES_EXPRESSION + (",",), 0, end)
                and not _JS_TAKES_NEXT_RE.search(text, max(end - 12, 0), end)
                and not text.startswith(_JS_CONTINUES_EXPRESSION, at)
            ):
                found = False, (at, held, direct)
                break
            if end == (held[0] + 1 if held else 0):
                if self._statements(held):
                    found = False, (at, held, direct)
                    break
                if first and text[held[0]] == "{" and text[held[0] - 1] != "$":
                    found = True, None  # a property's place, in a pattern or an enum
                    break
                outer = _innermost(self._brackets, held[0])
                if self._control(held[1]):
                    keyword = _JS_CONTROL_HEAD_RE.search(
                        text, max(held[0] - 12, 0), held[0]
                    )
                    found = False, (keyword.start(), outer, False)
                    break
                at, held, first, level, direct = held[0], outer, False, False, False
                continue
            char = text[end - 1]
            word = _JS_WORD_END_RE.search(text, max(end - 100, 0), end)
            if char in ")]}":
                opened = self.opener.get(end - 1)
                if self._opening.get(opened) and self._statements(held):
                    found = False, (at, held, direct)
                    break
                if char == ")" and self._control(end - 1):
                    found = True, False  # a statement's body without braces
                    break
                if first or opened is None:
                    found = True, None
                    break
                at = opened
            elif char == ";":
                # A `for` head's test or update
                found = (
                    (True, False)
                    if held and text[held[0]] == "("
                    else (
                        False,
                        (at, held, direct),
                    )
                )
                break
            elif char == "?" or text.startswith(("&&", "||", "=>"), end - 2):
                found = True, False
                break
            elif char == ":":
                # A declaration's type follows its name or pattern.
                named = _run_start(text, end - 1, "?!")
                if text[named - 1 : named] in ("]", "}"):
                    named = self.opener.get(named - 1, named)
                elif before := _JS_WORD_END_RE.search(text, max(named - 100, 0), named):
                    named = before.start()
                if _innermost(self._bound, named + 1):
                    found = True, None
                    break
                at, labelled = end - 1, True
            elif word:
                if word[0] == "else":
                    found = True, False
                    break
                if word[0] == "do":
                    found = False, (at, held, direct)  # a body that runs once
                    break
                if word[0] == "case" and not labelled and self._switch(held):
                    found = True, self._case_reach(held, word.start())
                    break
                if first and word[0] not in ("return", "yield", "throw"):
                    found = True, None
                    break
                at = word.start()
            elif char in "=,":
                direct = direct and char != "="
                at = end - 1
            elif first or char == "<" and level:
                found = True, None  # no assignment's place, or a type parameter's
                break
            else:
                at = end - 1
            first = False
        for state in read:
            self._walks[state] = found
        return found

    def _case_reach(self, held, label):
        """``_reach`` of the left side of an assignment in the expression of
        the label at offset ``label`` of the `switch` whose body is ``held``.
        The first case's expression runs with the `switch`, a later one's
        where no case before it is the one: in its own case, where none
        before it falls through."""
        starts, opening, exits = self._cases(held)
        if label == opening:
            return self.block(held[0])
        index = starts.index(label)
        end = starts[index + 1] if index + 1 < len(starts) else held[1]
        return exits[index] and (self.place(label), self.place(end))

    def _statement_span(self, at, end):
        """The span of the statement at offset ``at``: to its `;`, or past
        the `{` it opens, outside brackets."""
        text, pos = self.text, at
        while pos < end:
            char = text[pos]
            if char == "{":
                return at, self.closer.get(pos, pos) + 1
            if char in "([":
                pos = self.closer.get(pos, pos)
            elif char == ";":
                return at, pos + 1
            pos += 1
        return at, end

    def _section_exits(self, at, end):
        """Whether the section of a `switch` from offset ``at`` to ``end``
        leaves it: its last statement leaves unconditionally, a `break`,
        `continue`, `return` or `throw` of its own or an `if`/`else` whose
        branches all do; one inside a loop or a nested `switch` leaves
        that, not the section."""
        text, last = self.text, None
        pos, start = at, at
        while pos < end:
            char = text[pos]
            if char == ";":
                last = (start, pos + 1)
                start = pos + 1
            elif char in "([":
                pos = self.closer.get(pos, pos)
            elif char == "{":
                pos = self.closer.get(pos, pos)
                if re.match(r"\s*(?:else|catch|finally|while)\b", text[pos + 1 :]):
                    pos += 1
                    continue
                last = (start, pos + 1)
                start = pos + 1
            pos += 1
        if text[start:end].strip():
            last = (start, end)
        if last is None:
            return False
        written = text[last[0] : last[1]].lstrip()
        # The last statement's head, past the section's `case`/`default`
        # label and any before it.
        head = re.match(
            r"(?:(?:case\b[^:]*|default|[A-Za-z_$][\w$]*)\s*:\s*)*", written
        )
        written = written[head.end() :]
        if written.startswith("{"):
            # A bare block is no exit of the section's own.
            return False
        word = re.match(r"([A-Za-z_]\w*)", written)
        word = word[1] if word else ""
        if word in ("break", "continue", "return", "throw"):
            return True
        if word != "if":
            return False
        paren = text.find("(", last[0], last[1])
        if paren < 0 or paren not in self.closer:
            return False
        then = self._statement_span(self.closer[paren] + 1, last[1])
        rest = text[then[1] : last[1]]
        if not re.match(r"\s*else\b", rest):
            return False
        offset = then[1] + rest.index("else") + 4
        return self._block_exits(*then) and self._block_exits(
            *self._statement_span(offset, last[1])
        )

    def _block_exits(self, at, end):
        """Whether the statement that spans offsets ``at`` to ``end`` exits
        the `switch` section it is a branch of: into the block for one."""
        text = self.text
        written = text[at:end].lstrip()
        if written.startswith("{"):
            opened = at + len(text[at:end]) - len(written)
            return self._section_exits(opened + 1, self.closer.get(opened, end))
        return self._section_exits(at, end)

    def _cases(self, held):
        """``(starts, opening, exits)`` of the `switch` whose body is
        ``held``: the offset of each of its labels, that of its first
        `case`, and for each label whether every case before it ends with a
        statement that leaves it."""
        if held not in self._switches:
            text = self.text
            labels = [
                label
                for label in _JS_CASE_RE.finditer(text, held[0], held[1])
                if _innermost(self._brackets, label.start()) == held
            ]
            starts = [label.start() for label in labels]
            exits, every = [], True
            for index, at in enumerate(starts):
                exits.append(every)
                end = starts[index + 1] if index + 1 < len(starts) else at
                every = every and self._section_exits(at, end)
            opening = next((lb.start() for lb in labels if lb[0] == "case"), None)
            self._switches[held] = starts, opening, exits
        return self._switches[held]

    def _expression_end(self, at):
        """The offset where the expression that starts at offset ``at`` ends:
        at the `,`, the `;` or the closing bracket after it outside brackets,
        at a `:` that ends a branch of a conditional, or where its statement
        ends. Where it ends from each offset passed on its first line is
        kept, as where the statement ends is read only once the expression
        reaches its line's end."""
        text, lines, starts = self.text, self.lines, self.starts
        row, column = self.place(at)
        limit, known = starts[row - 1] + len(lines[row - 1]), False
        branches, passed = 0, []
        while True:
            if not known and not branches and at in self._ends:
                found = self._ends[at]
                break
            if not known and (
                at >= limit or text[at] in "([{" and at not in self.closer
            ):
                last = _statement_end(lines, row - 1, column)
                limit, known = starts[last] + len(lines[last]), True
            if at >= limit or text[at] in ",;)]}" or text[at] == ":" and not branches:
                found = min(at, limit)
                break
            if not known and not branches:
                passed.append(at)
            if text[at] in "([{":
                at = self.closer.get(at, limit)
            elif text.startswith(("?.", "??"), at):
                at += 1
            else:
                branches += (text[at] == "?") - (text[at] == ":")
            at += 1
        for at in passed:
            self._ends[at] = found
        return found

    def _code_at(self, row, column):
        """The line index and column of the first code at or after column
        ``column`` of line index ``row``."""
        lines = self.lines
        while row < len(lines) and not lines[row][column:].strip():
            row, column = row + 1, 0
        if row < len(lines):
            column = len(lines[row]) - len(lines[row][column:].lstrip())
        return row, column

    def _past(self, row, column, last=None):
        """Just past the brackets that open at column ``column`` of line
        index ``row`` or, for the statement there that ends by line index
        ``last``, past its `;`, or at the bracket that closes what holds it."""
        lines, depth = self.lines, 0
        for at_row in range(row, len(lines) if last is None else last + 1):
            for at in range(column, len(lines[at_row])):
                char = lines[at_row][at]
                depth += (char in "([{") - (char in ")]}")
                if depth < 0:
                    return at_row, at
                if depth == 0 and (last is None or char == ";"):
                    return at_row, at + 1
            column = 0
        return (len(lines) if last is None else last + 1), 0

    def past_statement(self, row, column):
        """The line index and column just past the statement that starts at
        or after column ``column`` of line index ``row``: a block, a
        statement a keyword starts with its head, its body and the clauses
        that go on with it, a labelled statement, or one statement."""
        lines = self.lines
        opened = []  # the keywords of the statements whose body is read
        while True:
            row, column = self._code_at(row, column)
            if row == len(lines):
                return row, 0
            compound = _JS_COMPOUND_RE.match(lines[row], column)
            if lines[row].startswith("{", column):
                row, column = self._past(row, column)
            elif compound and not (compound[1] or compound[2]):
                column = compound.end()
                continue
            elif compound:
                opened.append(compound[1] or compound[2])
                row, column = self._code_at(row, compound.end())
                if (
                    compound[1]
                    and row < len(lines)
                    and lines[row].startswith("(", column)
                ):
                    row, column = self._past(row, column)
                continue
            else:
                last = _statement_end(lines, row, column)
                row, column = self._past(row, column, last)
            # The statement read ends those whose body it is, but for the
            # clause that goes on with one.
            while opened:
                keyword = opened.pop()
                ahead, at = self._code_at(row, column)
                clause = (
                    ahead < len(lines)
                    and keyword in _JS_CLAUSE_RE
                    and _JS_CLAUSE_RE[keyword].match(lines[ahead], at)
                )
                if clause and keyword == "do":
                    row, column = self._past(*self._code_at(ahead, clause.end()))
                    ahead, at = self._code_at(row, column)
                    if ahead < len(lines) and lines[ahead].startswith(";", at):
                        row, column = ahead, at + 1
                elif clause:
                    row, column = ahead, at
                    break
            else:
                return row, column

    def typed(self, name, binding):
        """Whether the declaration ``binding`` of ``name`` writes its type."""
        kind, (row, column), written, _ = binding
        annotated = rf"{re.escape(name)}\s*[?!]?\s*:(?!:)"
        if kind == "param":
            start, end = (self.starts[r - 1] + c for r, c in written)
            return bool(re.search(rf"(?<![\w$.]){annotated}", self.text[start:end]))
        return kind != "=" and bool(re.match(annotated, self.lines[row - 1][column:]))

    def assignments(self, name, span):
        """The assignments of ``name`` in ``span``, of the variable a
        declaration of that span binds: those no declaration inside it
        binds. Only the last span read is kept per name: the reads of a
        name under one span come together, and what is held stays linear in
        the input where many nested spans hold the name."""
        cached = self._acache.get(name)
        if cached is not None and cached[0] == span:
            return cached[1]
        nested, by_span, at, *_ = self._indexed.get(name, ((), {}, {}))
        # Only the spans inside ``span`` are read: it and what it holds.
        inside = list(by_span.get(span, ()))
        at = at.get(span, 0)
        end = bisect_left(nested, ((span[1][0], span[1][1] + 1),))
        for entry in nested[at + 1 : end]:
            if entry[1] <= span[1]:
                inside += by_span[entry[:2]]
        declared = [b[3] for _, b in inside if b[0] != "=" and b[3] != span]
        inside = [
            b
            for _, b in inside
            if b[0] == "=" and not any(d[0] <= b[1] <= d[1] for d in declared)
        ]
        # Those of the blocks inside, in the order they hold from
        inner = sorted((b for b in inside if b[3] != span), key=itemgetter(1))
        self._acache[name] = (
            span,
            inside,
            frozenset(self.function(b[1]) for b in inside),
            inner,
            [b[1] for b in inner],
        )
        return inside

    def assignment_funcs(self, name, span):
        """The functions ``assignments`` of ``name`` in ``span`` are in."""
        self.assignments(name, span)
        return self._acache[name][2]

    def assigned_values(self, name, span, fingerprint, value):
        """The set ``value(b)`` of each assignment of ``name`` in ``span``,
        while what ``value`` reads is unchanged."""
        cached = self._vcache.get(name)
        if cached is not None and cached[:2] == (span, fingerprint):
            return cached[2]
        values = {
            value(b)
            for b in self.assignments(name, span)
            if (name, b[1]) not in self.undefined
        }
        self._vcache[name] = (span, fingerprint, values)
        return values

    def reassigned_values(self, name, span, after, read, changes, value):
        """The set ``value(b)`` over the assignments of ``name`` that
        blocks inside ``span`` hold, written after ``after`` and closed
        before ``read`` or in a loop that holds ``read`` and starts after
        ``after``, of the variable no declaration in those blocks binds:
        past its block one may have given what the variable holds, and in
        the loop's next run. The bindings are kept by the end of each one's
        span, so a moving ``read`` adds to or drops from one edge; only the
        name's last call is kept, so nothing held grows with the spans a
        name is read under. ``changes`` is the log of the lines the entries
        ``value`` reads are written on: what is held stays while no line
        written since is one a binding held is written on. Of the values
        held, two stand for all where there are more: what tells one value
        from several."""
        state = self._rstate.get(name)
        if (
            state is None
            or state[:2] != [span, after]
            or state[7] is not changes
            or any(changes[k] <= state[9] for k in range(state[8], len(changes)))
        ):
            self.assignments(name, span)
            inner, ats = self._acache[name][3:]
            bindings = sorted(
                (
                    b
                    for b in inner[bisect_right(ats, after) :]
                    if (name, b[1]) not in self.undefined
                ),
                key=lambda b: b[3][1],
            )
            written = sorted(bindings, key=lambda b: b[2][0])
            state = self._rstate[name] = [
                span,
                after,
                bindings,
                written,
                [b[2][0] for b in written],
                0,
                Counter(),
                changes,
                len(changes),
                0,
            ]
        bindings, written, ats, at, held = state[2:7]
        while at and bindings[at - 1][3][1] >= read:
            at -= 1
            given = value(bindings[at])
            if held[given] > 1:
                held[given] -= 1
            else:
                del held[given]
        while at < len(bindings) and bindings[at][3][1] < read:
            held[value(bindings[at])] += 1
            state[9] = max(state[9], bindings[at][2][1][0])
            at += 1
        state[5], state[8] = at, len(changes)
        values = set(islice(held, 2))
        first, k = (
            bisect_right(self._loop_starts, after),
            bisect_right(self._loop_starts, read),
        )
        while k > first and self._loop_reach[k - 1] >= read:
            k -= 1
            start, end = self._loops[k]
            if read > end:
                continue
            for b in written[bisect_left(ats, start) : bisect_right(ats, end)]:
                if b[2][1] <= end:
                    values.add(value(b))
        return values

    def binds_again(self, name, lineno):
        """Whether the lines more than one binding of ``name`` is written
        over hold line ``lineno``."""
        rows, reach = self._rows.get(name, ((), ()))
        at = bisect_right(rows, (lineno, 1 << 30))
        seen = 0
        # Back to where no binding written before reaches the line
        while at > 0 and seen < 2 and reach[at - 1] >= lineno:
            at -= 1
            seen += rows[at][1] >= lineno
        return seen > 1

    def _spans(self, name, read, last):
        """The spans of the bindings of ``name`` that can hold a read
        somewhere from ``read`` to ``last``, innermost first: those around
        where it is, and any that starts between it and the end of what it
        covers."""
        nested = self._indexed.get(name, ((),))[0]
        at = bisect_left(nested, (read,))
        spans = [
            s[:2] for s in nested[at : bisect_left(nested, ((last[0], last[1] + 1),))]
        ]
        index = at - 1
        while index >= 0:
            start, end, outer = nested[index]
            if read <= end:
                spans.append((start, end))
            index = outer
        return sorted(spans, key=lambda s: (s[0], (-s[1][0], -s[1][1])), reverse=True)

    def visible(self, name, read, last):
        """The bindings of ``name`` in scope where it is read, somewhere from
        ``read`` to ``last``, the innermost span's first and of one span the
        last written first: those whose span holds that, a declaration from
        the start of its span, an assignment written before the read or read
        from a function inside the one it is in."""
        reader = self.function(read)
        _, by_span, _, positions, runs, _ = self._indexed.get(name, (None,) * 6)

        def hidden(binding):
            """Whether ``binding``, written past ``last``, is an assignment
            of the reader's own function, which the read does not see."""
            return binding[0] == "=" and self.function(binding[1]) == reader

        for span in self._spans(name, read, last):
            entries, held, run = by_span[span], positions[span], runs[span]
            seen, k = bisect_right(held, last), len(entries)
            while k:
                if k > seen and hidden(entries[k - 1][1]):
                    # The run of the assignments of the reader's function
                    k = max(run[k - 1], seen)
                    continue
                # Of those that hold from one place, the first written first
                tie = bisect_left(held, held[k - 1], 0, k)
                yield from (
                    b
                    for i, (_, b) in enumerate(entries[tie:k], tie)
                    if i < seen or not hidden(b)
                )
                k = tie

    def declaration(self, name, read, last):
        """The first of ``visible`` that is no assignment, else None."""
        _, by_span, _, positions, _, declared = self._indexed.get(name, (None,) * 6)
        for span in self._spans(name, read, last):
            if found := declared[span]:
                held = positions[span]
                first = found[bisect_left(found, bisect_left(held, held[found[-1]]))]
                return by_span[span][first][1]
        return None

    def unset(self, binding, read):
        """Whether the declaration ``binding`` holds no value yet where it is
        read at ``read``: written after it in the same function, but for a
        function declaration, which is set at the start of its block."""
        return (
            read < binding[1]
            and binding[0] != "function"
            and self.function(binding[1]) == self.function(read)
        )


def _js_declares(text, at, opener):
    """Whether the `(` at offset ``at`` of ``text`` opens a declaration's
    parameters: after `function` and the function's name, or after a
    member's name, its modifiers and decorators, where a statement or a
    member starts. ``opener`` maps each closing bracket's offset to that of
    the one it closes."""

    def back(end):
        code = _run_start(text, end)
        return code, "\n" in text[code:end]

    def word(end):
        return _JS_WORD_END_RE.search(text, max(end - 100, 0), end)

    end = _run_start(text, at)
    if text[end - 1 : end] == ">" and text[end - 2 : end] != "=>":
        depth = 0  # the declaration's type parameters
        for opened in range(end - 1, -1, -1):
            depth += (text[opened] == ">") - (text[opened] == "<")
            if not depth:
                break
        end = _run_start(text, opened)
    if text[end - 1 : end] == "?" and not text[end - 2 : end - 1].isspace():
        end -= 1  # an optional member's
    if text[end - 1 : end] == "]":
        end = opener.get(end - 1, end - 1)
    elif text[end - 1 : end] in ("'", '"'):
        end = text.rfind(text[end - 1], 0, end - 1)
    elif found := word(end):
        if found[0] == "function":
            return True
        end = found.start()
    while True:
        end, broke = back(end)
        found = word(end)
        if found and found[0] == "function":
            return True
        if found and found[0] in _JS_DECLARING_WORDS:
            end = found.start()
        elif text[end - 1 : end] == "*":
            end -= 1
        elif decorator := _JS_DECORATOR_END_RE.search(
            text, max(end - 100, 0), opener.get(end - 1, end)
        ):
            end = decorator.start()
        else:
            break
    if not end or text[end - 1] in "{};,":
        return True
    # A line break ends the statement or member before.
    if found:
        return broke and found[0] not in _JS_LEADING_WORDS
    return broke and text[end - 1] in ")]>'\"`"


def _js_function_end(text, at, closer, lines, starts, declared):
    """``(offset, arrow)`` of the function whose parameter list ends before
    offset ``at`` of ``text``: where it ends, the `}` of its body, the end
    of its expression body, or of its return type where it has no body, and
    the offset just past its `=>`, None for no arrow. None where no `=>`,
    `{` or `:` follows the list, as after a call's arguments, and where no
    `=>` follows a list ``declared`` says no declaration writes."""
    head = re.compile(r"\s*(=>|\{|:)").match(text, at)
    if not head:
        return None
    end = _js_body_end(text, head, closer, lines, starts)
    return end if declared or end[1] is not None else None


def _js_body_end(text, head, closer, lines, starts):
    """``(offset, arrow)`` where the function ``_js_function_end`` reads
    from ``head``, the match of what follows its parameter list, ends, and
    the offset just past its `=>`, None for no arrow."""
    at, angles, asked = head.end(), 0, 0
    body = at if head[1] == "=>" else None  # where an arrow's body starts
    if head[1] == "{":
        return closer.get(at - 1), None
    while at < len(text):
        char = text[at]
        if body is None and not angles and text.startswith("=>", at):
            at = body = at + 2
            continue
        # A call's type arguments, `h<A, B>(`, and the type of a cast in an
        # expression body
        skipped = body is not None and (
            _JS_TYPE_ARGUMENTS_RE.match(text, at) or _TS_CAST_TYPE_RE.match(text, at)
        )
        if skipped:
            at = skipped.end()
            continue
        if (
            body is not None
            and char == "?"
            and text[at + 1 : at + 2] not in ".?"
            and text[at - 1 : at] != "?"
        ):
            asked += 1
        elif body is not None and char == ":":
            # The `:` of a conditional the body holds, else of one the
            # function is a branch of, which ends it
            if not asked:
                return at - 1, body
            asked -= 1
        if char == "{" and not angles:
            # A `{` right after an arrow's `=>` opens its block body; before
            # an arrow, one that starts no type opens the function's.
            if body is not None:
                opens = not text[body:at].strip()
            else:
                opens = not re.search(r"[:|&<,(]\s*$", text[max(at - 40, 0) : at])
            if opens:
                return closer.get(at), body
        if char in "([{":
            at = closer.get(at, at) + 1
            continue
        if body is None:
            angles = max(
                angles + (char == "<") - (char == ">" and text[at - 1] != "="), 0
            )
        if not angles and char in ",;)]}":
            return at - 1, body
        if not angles and char == "\n":
            row = bisect_right(starts, at) - 1
            # A body may open on the line after a return type.
            ahead = _SPACES.match(text, at).end()
            braced = body is None and text[ahead : ahead + 1] == "{"
            # An arrow's body may start on the line after its `=>`.
            begun = body is None or text[body:at].strip()
            if _statement_end(lines, row) == row and not braced and begun:
                return at - 1, body
        at += 1
    return len(text) - 1, body


def _own_line_span(code: str, name_end: int) -> tuple[int, int] | None:
    """The columns of ``code`` holding the parameters and the first line of
    the body of a definition whose name ends at ``name_end``: code that
    runs when the definition is called, its parameter defaults with it
    (ECMA-262 2026 10.2.11 FunctionDeclarationInstantiation), so a call on
    its own line is its, not its enclosing scope's.
    """
    if re.match(r"\s*:", code[name_end:]):
        # `const f: Type = value`: the value follows the annotation's `=`.
        depth = 0
        for at in range(name_end, len(code)):
            char, before = code[at], code[at - 1]
            depth += (char in "([{<") - (char in ")]}" or char == ">" and before != "=")
            if not depth and char == "=" and code[at + 1 : at + 2] not in ("=", ">"):
                name_end = at + 1
                break
    paren = code.find("(", name_end)
    arrow = code.find("=>", name_end)
    if arrow >= 0 and (paren < 0 or arrow < paren):
        # One parameter, unparenthesised (`x => g(x)`): the body is the
        # expression after the arrow, to the `,` or `;` that ends it.
        depth = 0
        for body in range(arrow + 2, len(code)):
            depth += (code[body] in "([{") - (code[body] in ")]}")
            if depth == 0 and code[body] in ",;":
                return (arrow, body)
        return (arrow, len(code))
    if paren < 0:
        return None
    depth = 0
    close = -1
    for col in range(paren, len(code)):
        depth += (code[col] == "(") - (code[col] == ")")
        if depth == 0:
            close = col
            break
    if close < 0:
        # The parameters span lines; the rest of this line is theirs.
        return (paren, len(code))
    # The body follows, past any return-type annotation.
    for offset, char in enumerate(code[close + 1 :]):
        if char == "{":
            # A block: where its braces balance on this line, else the
            # first line of a block that spans lines.
            body = _block_close(code, close + 1 + offset)
            return (paren, len(code) if body is None else body + 1)
        if char == "=" and code[close + 2 + offset : close + 3 + offset] == ">":
            # An expression: to the `,` or `;` that ends it, at bracket depth.
            depth = 0
            for body in range(close + 3 + offset, len(code)):
                char = code[body]
                depth += (char in "([{") - (char in ")]}")
                if depth == 0 and char in ",;":
                    return (paren, body)
            return (paren, len(code))
    return None


# The name, annotation and `=` of a declarator after a `,`, the `=` where
# it has a value
_TS_DECLARATOR_RE = re.compile(
    r"\s*([A-Za-z_$][\w$]*)\s*(?::[^=,;]*)?(?:(=)(?![=>])\s*|(?=[,;]|\s*$))"
)


def _value_end(code: str, start: int) -> int:
    """The column at which the declarator's value starting at column
    ``start`` of ``code`` ends: outside brackets, the `;` after it or the `,`
    before another declarator, else the end of the line."""
    depth = 0
    for column in range(start, len(code)):
        depth += (code[column] in "([{") - (code[column] in ")]}")
        if depth < 0 or (
            depth == 0
            and (
                code[column] == ";"
                or (
                    code[column] == ","
                    and (
                        not code[column + 1 :].strip()
                        or (found := _TS_DECLARATOR_RE.match(code, column + 1))
                        and found[2]
                    )
                )
            )
        ):
            return column
    return len(code)


def _passed_objects(
    code: str, brackets: list[bool], before: str, calls: set[int]
) -> tuple[list[int], str]:
    """The columns at which the object literals open that ``code`` passes
    to a call and leaves open: each `{` that starts an argument, after the
    `(` of a call or the `,` after another argument. ``brackets`` are the
    brackets open where the line starts, innermost last, True for the `(`
    of a call, and are left as the line leaves them; ``calls`` are the
    columns of the `(` of the calls on the line, and ``before`` is the
    last character of code above it. Returns the columns and the last
    character of code read."""
    if code[:1].strip() and code[0] not in ")}]":
        # An unindented line starts a top-level item, unless it opens
        # with what closes the one before.
        brackets.clear()
    opened = {}  # how deep each such object is, while open -> its column
    for column, char in enumerate(code):
        if char in "([{":
            if char == "{" and brackets[-1:] == [True] and before in ("(", ","):
                opened[len(brackets)] = column
            brackets.append(char == "(" and column in calls)
        elif char in ")]}" and brackets:
            brackets.pop()
            opened.pop(len(brackets), None)
        if not char.isspace():
            before = char
    return sorted(opened.values()), before


def _declarators(
    lines: list[str], index: int, after: int
) -> list[tuple[int, str, int, int | None]]:
    """Each declarator after the first of the `const`, `let` or `var`
    statement whose first binding, a name or a destructuring pattern, ends
    at column ``after`` of line ``index``: `b` in `const a = f, b = () =>
    g();` or on a line the statement goes on to, as the index of its line,
    its name, the column its name ends at and the column its value starts
    at, None for none. ``lines`` are the file's lines with the text of
    their literals blanked."""
    declarators = []
    depth = 0
    in_value = False  # past the declarator's `=`
    row, column = index, after
    while row < len(lines):
        code = lines[row]
        while column < len(code):
            char = code[column]
            if not in_value:
                # The annotation's `<>` hold commas of their own.
                depth += (char in "([{<") - (char in ")]}>" and code[column - 1] != "=")
                in_value = (
                    char == "="
                    and depth == 0
                    and code[column + 1 : column + 2] not in "=>"
                )
            else:
                depth += (char in "([{") - (char in ")]}")
            if depth < 0 or (depth == 0 and char == ";"):
                return declarators
            if depth == 0 and char == ",":
                rest, at, offset = code[column + 1 :], row, column + 1
                if not rest.strip() and row + 1 < len(lines):
                    rest, at, offset = lines[row + 1], row + 1, 0
                if named := _TS_DECLARATOR_RE.match(rest):
                    in_value = False
                    value = offset + named.end() if named[2] else None
                    declarators.append((at, named[1], offset + named.end(1), value))
            column += 1
        # The statement goes on where a bracket is open or the line ends
        # with the `,` before another declarator.
        if not (depth > 0 or code.rstrip().endswith(",")):
            return declarators
        row, column = row + 1, 0
    return declarators


def _class_body_open(code: str, start: int) -> int | None:
    """The column of the `{` that opens the body of the class whose name
    ends at column ``start`` of ``code``, where it is on that line: the
    first outside the brackets and type arguments of its heritage."""
    depth = 0
    for column in range(start, len(code)):
        char = code[column]
        if char == "{" and depth == 0:
            return column
        depth += (char in "([<") - (char in ")]>")
    return None


def _declared_static(code: str, name: str) -> bool:
    """Whether ``code``, a line of a class body, declares the member
    ``name`` `static`."""
    words = "|".join((*_TS_MODIFIER_WORDS, "get", "set", "accessor"))
    return bool(
        re.search(
            rf"(?<![\w$.])static\s+(?:(?:{words})\s+|\*\s*)*"
            rf"[\"']?{re.escape(name)}(?![\w$])",
            code,
        )
    )


def _member_filed(found: re.Match, text: str) -> str:
    """The name the member ``found``, a match of ``_TS_LINE_MEMBER_RE``, is
    filed under: the one ``text``, the line with its literals, writes
    there, after `get` or `set` for an accessor."""
    name = _ts_unquote(text[found.start(2) : found.end(2)])
    return " ".join(filter(None, (found.group(1), name)))


def _members_on_line(
    code: str, text: str, start: int
) -> tuple[list[tuple[str, int, str]], list[tuple[int, int]], int | None]:
    """The methods and accessors written on the line of a class, in the body
    whose `{` is at column ``start`` of ``code``, and the properties holding
    an arrow function, as on a later line: each as the name it is filed
    under (`get size` for an accessor), the column its name ends at, and its
    text. Also the columns of the value of each other field, and the column
    at which a member or field starts that the line leaves open, None where
    there is none. ``code`` is a line with the text of its literals
    blanked, ``text`` that line with it."""
    members = []
    fields = []
    column = start + 1
    while column < len(code):
        found = _TS_LINE_MEMBER_RE.match(code, column)
        end = None
        depth = 0
        for at in range(found.end() - 1 if found else column, len(code)):
            depth += (code[at] in "([{") - (code[at] in ")]}")
            if depth < 0:
                if found:
                    return members, fields, None
                end = at  # a field runs to the class's own `}`
                break
            if found and depth == 0 and code[at] == ")":
                # The parameters closed; the member ends with its body,
                # unless a `;` ends a body-less overload first.
                semi = brace = -1
                inner = 0
                for col in range(at + 1, len(code)):
                    ch = code[col]
                    if inner == 0 and ch == ";":
                        semi = col
                        break
                    if inner == 0 and ch == "{":
                        brace = col
                        break
                    inner += (ch in "([{") - (ch in ")]}")
                    if inner < 0:
                        break
                if semi >= 0:
                    end = semi + 1
                    break
                close = _block_close(code, brace) if brace >= 0 else None
                if close is None:
                    break
                end = close + 1
                break
            if not found and depth == 0 and code[at] in ";}":
                end = at + 1
                break
        if end is None:
            return members, fields, column if code[column:].strip() else None
        if end <= column:
            return members, fields, None
        if found:
            members.append(
                (_member_filed(found, text), found.end(2), code[column:end].strip())
            )
        elif prop := _class_property(code[:end], column, text):
            if _function_value(code[prop[2] : end]):
                members.append((prop[0], prop[1], code[prop[2] : end].strip()))
            else:
                fields.append((prop[2], end))
        column = end
    return members, fields, None


def _member_close_column(code: str) -> int | None:
    """The column of the `}` that closes the member whose next member is
    written on the same line, where ``code`` has it: the line's first `}`
    followed by a member declaration. None otherwise."""
    stripped = code.lstrip()
    if not stripped.startswith("}"):
        return None
    close = len(code) - len(stripped)
    rest = code[close + 1 :].lstrip()
    if not rest:
        return None
    at = close + 1 + len(code[close + 1 :]) - len(rest)
    # A keyword — `else`, `catch`, `for (`, `default:` — continues or
    # starts a statement; it names no member.
    if found := _TS_LINE_MEMBER_RE.match(code, at):
        name = found.group(2)
    elif prop := _class_property(code, at):
        name = prop[0]
    else:
        return None
    return None if name in _JSTS_CONTROL_KEYWORDS else close


def _closes_first(code: str) -> bool:
    """Whether the `}` of ``code`` that close blocks opened on earlier lines
    lead it: there is one, and before the last of them the line has nothing
    but closing brackets and separators."""
    depth = lowest = 0
    last = None
    for column, char in enumerate(code):
        depth += (char == "{") - (char == "}")
        if depth < lowest:
            lowest, last = depth, column
    return last is not None and not code[:last].strip("})];, \t")


def _braces_open(
    lines: list[str], first: int, start: int, index: int, column: int
) -> int:
    """How many of the `{` written from column ``start`` of line ``first``
    on are open past column ``column`` of the later line ``index``.
    ``lines`` are code lines with the text of their literals blanked."""
    written = "".join(
        (lines[first][start:], *lines[first + 1 : index], lines[index][: column + 1])
    )
    return written.count("{") - written.count("}")


def _block_close(code: str, opened: int) -> int | None:
    """The column of the `}` that closes the `{` at column ``opened`` of
    ``code``, where it is on that line."""
    depth = 0
    for column in range(opened, len(code)):
        depth += (code[column] == "{") - (code[column] == "}")
        if depth == 0:
            return column
    return None


def _definition_time_region(code: str) -> tuple[int, int] | None:
    """The columns of the decorators and computed keys that lead ``code``,
    each evaluated where the class or object is defined, not in the member
    they lead: a computed key by ECMA-262 2026 15.7.14
    ClassDefinitionEvaluation and 13.2.5.6 PropertyDefinitionEvaluation, a
    decorator, which ECMA-262 does not define, by TypeScript's decorator
    evaluation.
    """
    start = len(code) - len(code.lstrip())
    end = start
    while code[end : end + 1] in ("@", "["):
        opened = code[end]
        if opened == "@":
            named = re.match(r"@\s*[\w$.]+", code[end:])
            if named is None:
                break
            end += named.end()
            if code[end : end + 1] == "(":
                depth = 0
                for col in range(end, len(code)):
                    depth += (code[col] == "(") - (code[col] == ")")
                    if depth == 0:
                        end = col + 1
                        break
                else:
                    break
        else:
            depth = 0
            for col in range(end, len(code)):
                depth += (code[col] == "[") - (code[col] == "]")
                if depth == 0:
                    end = col + 1
                    break
            else:
                break
        end += len(code[end:]) - len(code[end:].lstrip())
    return (start, end) if end > start else None


def _ts_params(signature: str) -> set[str]:
    """The names that the parameters of ``signature`` bind: those in its
    first parameter list, or the one written alone before its `=>`, and
    those of an arrow function it returns (`a => (b) => ...`)."""
    names = set()
    depth = 0
    at_start = False
    found = list(re.finditer(r"[A-Za-z_$][\w$]*|\.\.\.|=>|\S", signature))
    tokens = [match[0] for match in found]
    arrow = tokens.index("=>") if "=>" in tokens else 0
    returned = ""
    if arrow and "(" not in tokens[:arrow]:
        names.add(tokens[arrow - 1])
        returned = signature[found[arrow].end() :]
    # The parameter list is the first `(` outside the type parameters.
    first, angles = len(tokens), 0
    for number, token in enumerate([] if returned else tokens):
        angles += (token == "<") - (token == ">")
        if token == "(" and not angles:
            first = number
            break
    for match, token in zip(found[first:], tokens[first:]):
        if token in ("(", "<", "[", "{"):
            if at_start and token in "[{":
                names.update(
                    name for name, _ in _js_bindings(signature[match.start() :])
                )
            depth += 1
            at_start = depth == 1
        elif token in (")", ">", "]", "}"):
            depth -= 1
            if depth == 0:
                body = re.match(r"\s*(?::[^=(){]+)?=>", signature[match.end() :])
                returned = body.string[body.end() :] if body else ""
                break
        elif depth == 1 and token == ",":
            at_start = True
        elif at_start and token not in _TS_PARAM_MODIFIERS:
            if token[0].isalpha() or token[0] in "_$":
                names.add(token)
            at_start = False
    if _TS_ARROW_HEAD_RE.match(returned):
        names |= _ts_params(returned)
    return names


# An arrow function from its parameters to its `=>`
_TS_ARROW_HEAD_RE = re.compile(
    r"\s*(?:async\s+)?(?:\([^()]*\)\s*(?::[^=(){]+)?|[\w$]+\s*)=>"
)


# What a line may hold before the name of an object literal's method
_TS_METHOD_MODIFIERS_RE = re.compile(r"\s*(?:async \*|async|static|get|set|\*)?\s*")


# The first declarator of a statement, without a type, up to its value
_TS_FIRST_DECLARATOR_RE = re.compile(
    r"(?:^|[;{}])\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=(?![=>])\s*"
)


# `const`, `let` or `var` and the first binding of its statement: a name, or
# a destructuring pattern without nested brackets
_TS_DECLARATION_HEAD_RE = re.compile(
    r"^\s*(?:export\s+)?(const|let|var)\s+"
    r"(?:[A-Za-z_$][\w$]*|\{[^{}]*\}|\[[^\[\]]*\])"
)


# A property a class declares, up to its annotation or value: `name`,
# `name?`, `name!`, plain- or quoted-keyed
_TS_CLASS_PROPERTY_RE = re.compile(
    rf"\s*{_TS_MEMBER_MODIFIERS}({_TS_WRITTEN_MEMBER_NAME})\s*[?!]?\s*(?=[:=])"
)


_TS_LINE_MEMBER_RE = re.compile(
    rf"\s*{_TS_MEMBER_MODIFIERS}(?:(get|set)\s+)?(?:\*\s*)?"
    rf"({_TS_WRITTEN_MEMBER_NAME})\s*(?:<[^>]*>)?\s*\("
)


# `const name = require("m")`, `const { a, b: c } = await import("m")`, a
# `/* */` comment before the module too
_TS_REQUIRE_RE = re.compile(
    rf"""\b(?:const|let|var|import){_JS_GAP}+([\w$]+|\{{(?:[^}}\s]|{_JS_GAP})*\}})"""
    rf"""{_JS_GAP}*={_JS_GAP}*(?:await{_JS_GAP}+)?(require|import)\({_JS_GAP}*"""
    rf"""(?:/\*(?:[^*]|\*(?!/))*\*/{_JS_GAP}*)*['"]([^'"]+)['"]{_JS_GAP}*\)"""
)


_TS_PARAM_MODIFIERS = ("...", "public", "private", "protected", "readonly")


# `export * from "<specifier>"`, `export { <clause> } from "<specifier>"` and,
# with no specifier, `export { <clause> }`, starting a line or past another
# statement's `;`
_TS_REEXPORT_RE = re.compile(
    rf"""{_JS_START}export{_JS_GAP}+(?:type{_JS_GAP}+)?(?:(\*)(?:{_JS_GAP}+as{_JS_GAP}+"""
    rf"""([\w$]+))?{_JS_GAP}+from{_JS_GAP}+|\{{((?:[^}}\s]|{_JS_GAP})*)\}}"""
    rf"""(?:{_JS_GAP}+from{_JS_GAP}+)?)"""
    r"""(?:['"]([^'"]+)['"])?""",
    re.MULTILINE,
)


# A declaration `export` gives the module under its name, one that makes a
# value when the module runs
_JS_EXPORTED_DECLARATION_RE = re.compile(
    rf"{_JS_START}export{_JS_GAP}+(?:(?:async{_JS_GAP}+)?function\b{_JS_GAP}*\*?"
    rf"|(?:abstract{_JS_GAP}+)?class\b|(?:const|let|var)\b(?!{_JS_GAP}+enum\b)|enum\b)"
    rf"{_JS_GAP}*([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


# What a module exports as its default, `export default <name>` or a
# declaration of a name after it, and whole, `export = <name>`, starting a
# line or past another statement's `;`; a name is the value where its
# statement ends after it (see ``_ends_after``)
_TS_DEFAULT_EXPORT_RE = re.compile(
    rf"{_JS_START}(?:export{_JS_GAP}+(default){_JS_GAP}+(?:"
    # `cond && function f`, called, is `f`: a function is never falsy.
    r"(?:[\w$.]+\s*&&\s*)*"
    rf"(?:(?:async{_JS_GAP}+)?function\b{_JS_GAP}*\*?|(?:abstract{_JS_GAP}+)?class\b"
    rf"|interface\b){_JS_GAP}*(?!extends\b|implements\b)([\w$]+)|([\w$]+)(?![\w$]))"
    rf"|export{_JS_GAP}*={_JS_GAP}*([\w$]+)(?![\w$]))",
    re.MULTILINE,
)


# What a line starts with where it goes on with the expression the line
# before ends with: a member, a call, an index, a template or a binary
# operator, `!=`, `in` and `instanceof` too; not `++` or `--`, before which
# the line break ends the statement, nor a number such as `.5`. A name that
# starts with `in` matches too; ``_ends_after`` tells it from the operator.
_JS_CONTINUES_RE = re.compile(
    r"(?!\+\+|--|\.\d)[.(\[`+\-*/%<>=&|^?:,]|!=|in(?:stanceof)?"
)


# What an ECMAScript name goes on with that a Python name does not: `$`, ZWNJ,
# ZWJ and the characters of ID_Continue that NFKC does not keep
_JS_NAME_ONLY = (
    "$\u200c\u200d\u037a\u309b\u309c\ufc5e\ufc5f\ufc60\ufc61\ufc62\ufc63\ufdfa"
    "\ufdfb\ufe70\ufe72\ufe74\ufe76\ufe78\ufe7a\ufe7c\ufe7e"
)


def _ends_after(code_lines, row, column):
    """Whether a JavaScript statement ends at ``column`` of line ``row`` of
    ``code_lines``, the lines blanked: a `;` or nothing follows on the line,
    and the next line holding code does not go on with the expression.
    U+2028 and U+2029 end a line too."""
    rest, *after = re.split("[\u2028\u2029]", code_lines[row][column:])
    if rest := rest.strip(_JS_WHITE):
        return rest.startswith(";")
    below = (
        line.strip(_JS_WHITE) for line in chain(after, lines_after(code_lines, row))
    )
    line = next((line for line in below if line), "")
    found = _JS_CONTINUES_RE.match(line)
    # `in` that a character of a name follows starts the name.
    tail = found and line[found.end() : found.end() + 1]
    named = tail and (tail in _JS_NAME_ONLY or f"a{tail}".isidentifier())
    return not found or found[0][0] == "i" and bool(named)


# `import <clause> from "<specifier>"`, the clause possibly over several
# lines, starting a line or past another statement's `;`
_TS_IMPORT_RE = re.compile(
    rf"""{_JS_START}import{_JS_GAP}++(?:type{_JS_GAP}++)?((?:[^;'"\s]|{_JS_GAP})*?)"""
    rf"""{_JS_GAP}+from{_JS_GAP}+['"]([^'"]+)['"]""",
    re.MULTILINE,
)


# A '...' literal closed on its line; a lone apostrophe, as in JSX text, is not one.
_JS_SINGLE_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'")


# A JSX element's opening tag up to its attributes: its name with the type
# arguments TypeScript gives it, or no name for a fragment's `<>`
_JSX_TAG_RE = re.compile(
    r"<(?:[A-Za-z_$][\w$.:-]*(?:\s*<[^<>]*(?:<[^<>]*>[^<>]*)*>)?(?=[\s/>{]|$)|(?=>))"
)


# Type parameters where a value can start, which a `.tsx` file writes as
# `<T,>`, `<T extends U>` or `<T = U>`, and as `<T>` before a parameter list
_TS_TYPE_PARAMETERS_RE = re.compile(
    r"<\s*(?:const\s+)?[\w$]+\s*(?:,|=|extends\b|>\s*\()"
)


# The files JSX can be written in
_JSX_EXTENSIONS = (".tsx", ".js", ".jsx", ".mjs", ".cjs")


# A /.../ literal with its flags, before what can follow a value. One starts
# where a value can: after what no value ends with, on its line or the one
# before.
_JS_REGEX_RE = re.compile(
    r"/(?:[^/\\\[*]|\\.|\[(?:[^\]\\]|\\.)*\])(?:[^/\\\[]|\\.|\[(?:[^\]\\]|\\.)*\])*/[a-z]*(?=\s*(?:[.,;:)\]}|&?/]|$))"
)


# A keyword that expects an operand where it ends the text before a `/`, so
# the `/` opens a regular expression (ECMAScript reads `/` for a literal
# where an operand is expected and for a division where a value just
# ended). `if`/`for`/`while`/`switch` take `(`, not a literal; `async`,
# `get` and `static` are names.
_JS_OPERAND_WORDS = {
    "return",
    "typeof",
    "instanceof",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "case",
    "throw",
    "yield",
    "await",
    "do",
    "else",
    "default",
    "break",
    "continue",
}


# A `break` or `continue` with the label it names on its line, which end a
# statement
_JS_LABEL_RE = re.compile(r"(?<![\w$.])(?:break|continue)[^\S\r\n\u2028\u2029]+[\w$]+$")


# The keyword whose parenthesised head a statement follows
_JS_STATEMENT_HEAD_RE = re.compile(r"(?<![\w$])(?:if|for(?:\s+await)?|while|with)\s*$")


def _js_regex_position(code, before=""):
    """Whether a ``/`` after ``code``, on a line that follows the code
    ``before``, can open a regular expression: true where no value precedes
    it, so at a line's start or after an operator, a bracket that opens, a
    spread's `...`, the `)` of an `if`, `for`, `while` or `with` head, a
    keyword that expects an operand, or the label of a `break` or
    `continue`. False after what ends a value: a word, a member's name, any
    other `)`, `]`, `}`, `.`, a postfix `++`/`--`, and a quote a blanked
    literal leaves."""
    lines, row = [before, code], 1
    end = _run_start(code, len(code))
    if not end:
        row, end = 0, _run_start(before, len(before))
    text = lines[row]
    if not end:
        return True
    if text.endswith(("++", "--"), 0, end):
        return False
    if text.endswith("...", 0, end):
        return True
    start = end
    while start and (text[start - 1].isalnum() or text[start - 1] in "_$"):
        start -= 1
    if start < end:
        if _JS_LABEL_RE.search(text, max(_run_start(text, start) - 8, 0), end):
            return True
        word = text[start:end]
        return word in _JS_OPERAND_WORDS and not _js_member_name(lines, row, start)
    if text[end - 1] == ")":
        depth, rows = 0, range(row, -1, -1)
        for row in rows:
            text = lines[row]
            for at in range(end - 1, -1, -1):
                depth += {")": 1, "(": -1}.get(text[at], 0)
                if not depth:
                    end = _run_start(text, at)
                    if not end and row:
                        row, text = 0, before
                        end = _run_start(text, len(text))
                    # The keyword ends at `end`: `for await` is the longest.
                    start = max(_run_start(text, max(end - 5, 0)) - 3, 0)
                    head = _JS_STATEMENT_HEAD_RE.search(text, start, end)
                    return bool(head) and not _js_member_name(lines, row, head.start())
            end = len(before)
        return False
    return text[end - 1] not in "]}.`'\""


# The end of a blanked /.../, "...", '...' or `...` literal a member is read of
_JS_LITERAL_RECEIVER_RE = re.compile(r"""([/"'`])\s*\.\s*$""")


# The type arguments a call is written with: `name<T, U>(`
_JS_TYPE_ARGUMENTS_RE = re.compile(
    r"(?<=[\w$])<(?:[^<>()]|<(?:[^<>()]|<[^<>()]*>)*>)*>(?=\s*\()"
)


# `const name: Type = `, up to where the value starts
_TS_TYPED_VALUE_RE = re.compile(
    r"\b(?:const|let|var)\s+[\w$]+\s*:\s*(?P<type>[^=;]+?)\s*=(?![=>])\s*"
)


# A `return` that is a statement of its own, up to its value, and the
# `new` that the value may start with
_JS_RETURN_RE = re.compile(
    r"(?:^|[;{}])\s*(?:(?:\}\s*)?else\s+|if\s*\([^;{}]*\)\s*|case\b[^:]*:\s*|default\s*:\s*)*"
    r"return\b\s*(?P<new>new\s+)?"
)


# A bracketed group of an object literal's text that holds no other
_JS_NESTED_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}")


def _nested_part(match):
    """A bracketed group of an object literal, as one marker of the length
    it is written: the text around it keeps the positions it has."""
    return "\0" + " " * (len(match[0]) - 1)


# A function without a name of its own whose body is a block that the line
# leaves open
_JS_BLOCK_FUNCTION_RE = re.compile(
    rf"(?:=>|\bfunction\b[^{{}}]*\)|\b[gs]et\s+{_TS_MEMBER_NAME}\s*\([^()]*\))"
    r"\s*(?::[^{}=]+)?\{\s*$"
)


# The `new` a constructed type is named after
_JS_NEW_RE = re.compile(r"\bnew\s+$")


# What follows a function's parameters up to its block body's `{`: a return
# type and an arrow's `=>`
_JS_BODY_OPENS_RE = re.compile(r"\s*(?::[^=;{}()]*)?(?:=>)?\s*\{")


_JS_FUNCTION_HEAD_RE = re.compile(r"\bfunction\s*\*?\s*$")


# What a function expression follows where it is a value
_JS_VALUE_START_RE = re.compile(r"(?:[(,=?:!|&]|\b(?:return|default|void))\s*$")


# A parameter that is passed a function, with that function's parameters
_TS_CALLABLE_RE = re.compile(r"[\w$]+\??\s*:\s*\(?\((?P<params>[^()]*)\)\s*=>")


# A variable whose value is one blanked literal, or an array
_JS_LITERAL_RE = re.compile(
    r"""\b(?:const|let|var)\s+(?P<name>[\w$]+)\s*=\s*"""
    r"""(?P<literal>/\s*|"\s*"|'\s*'|`[^`]*`|\[.*\]|\[$)\s*(?:as\s+const\s*)?;?\s*$"""
)


# The type of a literal, by the character it opens with: any other is a string
_JS_LITERAL_TYPES = {"/": "RegExp", "[": "Array"}


# A constructor parameter behind an accessibility, `override` or `readonly`,
# a field of the class as well, with what follows the modifiers
_TS_PARAMETER_PROPERTY_RE = re.compile(
    r"\s*(?:(?:public|private|protected|override|readonly)\s+)+(?P<rest>.*)", re.DOTALL
)
# `this.name =`, an assignment to a field of the object a method is of
_JS_THIS_FIELD_RE = re.compile(r"\bthis\.([\w$]+)\s*=(?![=>])")


# What opens a `for` head, and `const pattern of value` or `name in value`
# in it
_JS_FOR_START_RE = re.compile(r"\bfor\s*(?:await\s*)?\(")


_JS_FOR_PARTS_RE = re.compile(
    r"\s*(?:(?:const|let|var)\s+)?(?P<pattern>.+?)\s+(?P<kind>of|in)\s+(?P<value>.+?)\s*"
)


# The value a pattern's closing bracket is given, to the end of the line
_JS_PATTERN_VALUE_RE = re.compile(r"\s*=(?![=>])\s*(?P<value>.+?);?\s*$")


# A declaration that takes a value apart: `const { a, b } = value`
_JS_PATTERN_RE = re.compile(
    r"\b(?:const|let|var)\s+(?P<pattern>[\[{].*[\]}])\s*=(?!=)\s*(?P<value>.+?);?\s*$"
)


# A function passed to a call: its parameters, in a list or alone before the
# `=>`, and where its body starts
_JS_CLOSURE_RE = re.compile(
    r"(?:(?<=[(,=?!|&>])|(?<=[=?!|&>]\s)|(?<=return\s)|(?<=default\s)|(?<=void\s)"
    r"|(?<=:)(?=\s*(?:async\s+)?function\b)|^)\s*(?:async\s+)?"
    r"(?P<closure>(?:function\s*\*?\s*(?P<named>[\w$]+\s*)?)?"
    r"\((?P<params>[^()]*)\)(?:\s*:[^=(){]+)?\s*(?:=>|\{)|(?P<param>[\w$]+)\s*=>)"
)


# A function written as a property's value: `key: (params) =>`, `key: p =>`
# or `key: function (params) {`, or as a method, `key(params) {`
_JS_PROPERTY_CLOSURE_RE = re.compile(
    r"(?:^|[{,])\s*(?:[\w$]+\s*:\s*(?:async\s+)?(?P<closure>(?:function\s*\*?\s*[\w$]*\s*)?"
    r"\((?P<params>[^()]*)\)(?:\s*:[^=(){]+)?\s*(?:=>|\{)|(?P<param>[\w$]+)\s*=>)"
    r"|(?:async\s+)?\*?\s*[\w$]+\s*(?P<method>\((?P<method_params>[^()]*)\)"
    r"(?:\s*:[^=(){]+)?\s*\{))"
)


# A `for` head up to the declaration it starts with, or a `catch` head
# up to its binding
_JS_FOR_HEAD_RE = re.compile(r"(?<![\w$])(?:for\s*(?:await\s*)?|catch\s*)\(\s*$")


# What starts a statement of a keyword's: the keyword, of one that has a
# parenthesised head or of one that has none, or a label; and the clause
# that may go on with one: an `if`'s `else`, a `try`'s `catch` or
# `finally`, a `catch`'s `finally`, a `do`'s `while`
_JS_COMPOUND_RE = re.compile(
    r"(?:(if|for(?:\s+await)?|while|with|switch|catch)|(else|do|try|finally))\b"
    r"|[\w$]+\s*:"
)


_JS_CLAUSE_RE = {
    "if": re.compile(r"else\b"),
    "try": re.compile(r"(?:catch|finally)\b"),
    "catch": re.compile(r"finally\b"),
    "do": re.compile(r"while\b"),
}


# Where a variable's binding starts, a name or a pattern, and the keyword
# that declares it, `const`, `let`, `var` or `using`, or the clause whose
# head binds it: a `catch`, or a `for` whose head assigns a variable
_JS_BINDING_HEAD_RE = re.compile(
    r"(?<![\w$])(?:(const|let|var|using)\s+"
    r"|(catch|for)\s*(?:await\s*)?\(\s*(?!(?:const|let|var|using)\s))(?=[\w$\[{(])"
)


# The key of a property in an object pattern and the `:` after it, where
# the pattern binds another name than the key: `key: name`, `[computed]:`
_JS_PROPERTY_KEY_RE = re.compile(
    r"""(\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]|"[^"]*"|'[^']*'|[\w$]*)"""
    r"\s*(:?)"
)


def _js_bindings(pattern, steps=()):
    """Yield ``(name, steps)`` for each name the JavaScript pattern binds
    without declaring its type: the steps lead from the value taken apart
    to the part bound, the property's name for `{ name }` and `@N` for the
    element at position N of `[first, second]`, and are None for what
    `...rest` gathers, which is of no type the value gives; `&` for a rest
    parameter, which gathers the parameters from its own on."""
    pattern = pattern.strip()
    if pattern[:1] in ("[", "{"):
        for position, part in enumerate(_bracketed_parts(pattern[1:])):
            if part.startswith("..."):
                yield from ((name, None) for name, _ in _js_bindings(part[3:]))
            elif pattern[0] == "[":
                yield from _js_bindings(part, (*steps, f"@{position}"))
            else:
                key = _JS_PROPERTY_KEY_RE.match(part)
                bound = part[key.end() :] if key[2] else part
                yield from _js_bindings(bound, (*steps, key[1]))
    elif pattern.startswith("..."):
        if re.fullmatch(r"[A-Za-z_$][\w$]*", name := pattern[3:].strip()):
            yield name, (*steps, "&")
    elif re.fullmatch(
        r"[A-Za-z_$][\w$]*", name := pattern.partition("=")[0].strip().strip("()")
    ):
        yield name, steps  # an assignment's target may be parenthesized: `(x)`


def _blank_js_strings(
    line: str, stack: list | None = None, keep=False, jsx=False, before=""
) -> str:
    """Return ``line`` with the text of "...", '...', `...` and /.../
    literals and of ``/* */`` comments turned to spaces, and everything a
    ``//`` comment covers cut. The `${...}` expressions of a template
    literal are kept. ``stack`` holds the literals and comments open where
    the line starts and is left holding those open where it ends; a literal
    carried over from an earlier line is blanked together with the quote
    that closes it. ``keep`` leaves the text of the literals that open on
    the line where it is, for a reader of what a literal holds: an import's
    module, a constant's value. With ``jsx``, the text of a JSX element and
    the names of its attributes are turned to spaces too, and its tags and
    `{...}` expressions are kept; ``before`` is the code the line follows,
    which no line break ends before a `<` or a `/`."""
    out = ""
    # open literals and comments: a quote character, a `${`'s brace depth,
    # `/*` for a comment still open, `<` or `</` for a JSX tag and `>` for
    # an element's text
    stack = [] if stack is None else stack
    carried = len(stack)
    ended = None  # how much of `out` there was where a /.../ literal ended
    # What ``keep`` shows at a column of `out`, which holds the blanked line
    # either way, so that what is code is read alike with ``keep`` and without
    kept_at = {}
    i = 0
    while i < len(line):
        ch = line[i]
        top = stack[-1] if stack else None
        if top == "/*":
            end = line.find("*/", i)
            if end == -1:
                out += " " * (len(line) - i)
                break
            stack.pop()
            # A comment carried over has ended.
            carried = min(carried, len(stack))
            out += " " * (end + 2 - i)
            i = end + 2
            continue
        elif top == ">":
            # An element's text, up to a `{...}` or a tag in it
            if ch == "{":
                stack.append(0)
            elif line.startswith("</", i):
                stack.append("</")
            elif tag := _JSX_TAG_RE.match(line, i):
                stack.append("<")
                out += tag.group()
                i = tag.end()
                continue
            out += ch if ch in "{<" else " "
        elif top in ("<", "</"):
            # A tag: its name, its attributes, their "..." and {...} values
            if ch in "\"'":
                stack.append(ch)
            elif ch == "{":
                stack.append(0)
            elif ch == "/" and line[i + 1 : i + 2] == "/":
                break
            elif ch == "/" and line[i + 1 : i + 2] == "*":
                stack.append("/*")
                out += "  "
                i += 2
                continue
            elif ch == ">" and top == "<":
                stack[-1] = ">"
            elif ch == ">" or line.startswith("/>", i):
                # A closing tag, or a tag that closes itself, ends its element.
                del stack[-2 if top == "</" else -1 :]
                carried = min(carried, len(stack))
                if ch == "/":
                    out += "/"
                    i += 1
                ch = ">"
            elif top == "<" and (ch.isalnum() or ch in "_$-:"):
                # An attribute's name, which names no variable
                if keep:
                    kept_at[len(out)] = ch
                ch = " "
            out += ch
        elif isinstance(top, str):
            # The text of a literal that opened on this line.
            kept = keep and len(stack) > carried
            if ch == "\\":
                if kept:
                    kept_at[len(out)] = line[i : i + 2]
                out += "  "
                i += 2
                continue
            if top == "`" and line.startswith("${", i):
                stack.append(0)
                out += "${"
                i += 2
                continue
            if ch == top:
                stack.pop()
            if len(stack) < carried:
                # A literal carried over has ended.
                carried = len(stack)
                ch = " "
            if kept:
                kept_at[len(out)] = ch
            out += ch if ch == top else " "
        else:
            if ch in ('"', "`") or (ch == "'" and _JS_SINGLE_QUOTED_RE.match(line, i)):
                stack.append(ch)
            elif (
                jsx
                and ch == "<"
                and (tag := _JSX_TAG_RE.match(line, i))
                and not _TS_TYPE_PARAMETERS_RE.match(line, i)
                # An element starts where a value can.
                and _js_regex_position(out, before)
            ):
                stack.append("<")
                out += tag.group()
                i = tag.end()
                continue
            elif (
                ch == "/"
                and (regex := _JS_REGEX_RE.match(line, i))
                # Straight after a /.../ literal a `/` divides.
                and (ended is None or _run_start(out, len(out)) > ended)
                and _js_regex_position(out, before)
            ):
                if keep:
                    kept_at[len(out)] = regex.group()
                out += f"/{' ' * (len(regex.group()) - 1)}"
                ended = len(out)
                i = regex.end()
                continue
            elif ch == "/" and line[i + 1 : i + 2] == "/":
                break
            elif ch == "/" and line[i + 1 : i + 2] == "*":
                stack.append("/*")
                out += "  "
                i += 2
                continue
            elif ch == "{" and top is not None:
                stack[-1] += 1
            elif ch == "}" and top is not None:
                if top == 0:
                    stack.pop()
                else:
                    stack[-1] -= 1
            elif ch in _JS_SPACE:
                ch = " "
            out += ch
        i += 1
    # Only a template literal, a block comment, a JSX element and the value of
    # one of its tag's attributes go on over the next line.
    while stack[-1:] in (['"'], ["'"]) and stack[-2:-1] != ["<"]:
        stack.pop()
    shown, at = [], 0
    for column, kept in kept_at.items():
        shown += out[at:column], kept
        at = column + len(kept)
    return ("".join(shown) + out[at:])[: len(line)]


def _blank_js_lines(lines, jsx=False, code=None):
    """``_blank_js_strings`` of each of a file's ``lines``, read in turn.
    Given ``code``, what this returns for them without it, the text of the
    literals that open on a line is kept, and the code a line follows is
    read from ``code``."""
    stack, blanked, before = [], [], ""
    for row, line in enumerate(lines):
        blanked.append(_blank_js_strings(line, stack, code is not None, jsx, before))
        read = (code or blanked)[row]
        if read.strip():
            # A `}` that ends a line is read as a block's, after which a
            # statement starts.
            before = "" if read.rstrip().endswith("}") else read
    return blanked


# What a declaration ends with where a type is still to come, so that a `{`
# there starts an object type and a line break does not end the declaration
_TS_TYPE_EXPECTED_RE = re.compile(r"(?:[:|&?]|\b(?:extends|is|keyof|readonly))\s*$")


# What a line starts with where it carries on the type or opens the body of
# the declaration before it
_TS_TYPE_CONTINUES = ("|", "&", "?", ":", "{")


# A bare `=` that ends a line: a value is assigned that starts on the next
_JS_ASSIGNMENT_END_RE = re.compile(r"(?<![=<>!])=(?!=)\s*$")


# What a class's `extends` gives as its base, up to the body or `implements`
_TS_EXTENDS_RE = re.compile(r"\bextends\s+(.+?)\s*(?:\bimplements\b[^{]*)?\{?\s*$")


# What a class's or an interface's body, or that of an object a
# declaration, an arrow it declares, `export default`, a `return` or a
# property gives, opens after
_TS_BODY_HEAD_RE = re.compile(
    r"\s*(?:(?P<type>(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:(?:abstract\s+)?"
    r"(?:class|interface|(?:namespace|module)(?=\s+[\w$'\"]))\b[^{]*"
    r"|type\s+[\w$]+\s*(?:<[^{]*>)?\s*=)"
    r"|(?:extends|implements)\b[^{]*)"
    r"|(?:export\s+default|return"
    r"|(?:export\s+)?(?:const|let|var)\s+[\w$]+\s*(?::[^=]*)?=\s*"
    r"(?:\[|(?:async\s+)?(?:\([^()]*\)|[\w$]+)\s*(?::[^=]*?)?=>\s*\()?"
    rf"|(?P<nested>{_TS_WRITTEN_MEMBER_NAME})\s*:)\s*\Z)"
)


# A line that opens an object it ends with: `export default {`, a CommonJS
# module's `module.exports = {`, a property's `key: {`, a function's
# prototype, `F.prototype = {`, or an element's `{`
_TS_OBJECT_OPENS_RE = re.compile(
    rf"\s*(?:(?P<default>export\s+default\s*)|(?P<key>{_TS_WRITTEN_MEMBER_NAME})\s*:\s*"
    r"|(?P<commonjs>(?:exports\s*=\s*)?module\.exports\s*=\s*(?:exports\s*=\s*)?)"
    r"|(?P<prototype>[\w$]+)\.prototype\s*=\s*)?"
    r"\{\s*$"
)


# What starts a statement after a block's `}` or a statement's `;`: not a
# word that goes on with the statement before it, nor a template's `${`
_TS_STATEMENT_START_RE = re.compile(
    r"(?!(?:else|catch|finally|while|as|satisfies|instanceof|in|of|extends|implements"
    r"|from|is)"
    r"(?![\w$]))(?!\$\{)[A-Za-z_$]"
)


# A namespace written with its name, not a module a string names
_TS_NAMESPACE_RE = re.compile(
    r"\s*(?:export\s+)?(?:declare\s+)?(?:namespace|module)\s+([\w$]+(?:\.[\w$]+)*)\s*\{"
)


# What a line may go on to after the `}` or `;` that ends a definition:
# another declaration, or a member of a class or an object with its body
_TS_DEFINITION_HEAD_RE = re.compile(
    r"(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
    r"(?:function|class|interface|enum|const|let|var)\b"
    rf"|{_TS_MEMBER_MODIFIERS}(?:\*\s*)?(?:[gs]et\s+)?(?P<member>{_TS_MEMBER_NAME})\s*[(<]"
)


class TSParser(BaseParser):
    binds_ahead = False
    language = "typescript"

    def _property(self, lineno, line, column, declared, name):
        """`name: Type` that is no declarator of a `const`, `let` or `var`
        statement, outside the brackets of its declarators, and no parameter
        a list around it binds: a member of a type or a key of an object."""
        head = self._code_lines[lineno - 1][:column]
        opened = sum(map(head.count, "([{")) - sum(map(head.count, ")]}"))
        statement = re.match(
            r"\s*(?:export\s+)?(?:declare\s+)?(?:const|let|var)\b", line
        )
        return (
            ":" in declared
            and not re.match(r"\W*(?:const|let|var)\b", declared)
            and not (statement and not opened)
            and not any(
                r[0] < (lineno, column) < r[1] and name in r[2]
                for r in self._scopes.params.get(lineno, ())
            )
        )

    binding_re = re.compile(
        r"\b(?:const|let|var)\s+(?P<name>[\w$]+)\s*(?::[^;=]*?)?=(?![=>])\s*"
        r"(?P<unwrap>await\b)?"
    )
    bindings = staticmethod(_js_bindings)
    self_name = "this"
    chained_values = True
    # The lines braces are counted in have their literals blanked already.
    _non_code = None

    def _typed_origin(self, origin, declared):
        """A name declared with a type and bound to a call is read as what
        the call returns: the type reads the members that gives none."""
        return f"{origin}>{declared[:-2]}"

    def _binds_name(self, name):
        """A name the file imports is what the module gives it, whatever a
        declaration of that name in a narrower scope says."""
        return name not in self._imported

    def _bind(self, bindings, index, at, *_):
        """As the base's, where the binding is in scope is read from the
        source; each name is bound where the code first names it from
        column ``at`` of line ``index`` on."""
        self._bind_from(bindings, self._scopes.starts[index] + at)

    def _bind_from(self, bindings, offset):
        """Give each name of ``bindings`` its origin path where the code
        first names it from offset ``offset`` on."""
        for name, origin in bindings.items():
            found = _named(self._scopes.text, name, offset)
            line, column = self._scopes.place(offset if found is None else found)
            self._rebind(name, line, column, origin)

    def _read_as(self, origin):
        """The origin path a folded declaration reads a name as: what the
        call a name was bound to returns, read in the type it is declared
        with. Any other path is read as it is."""
        call, _, declared = (origin or "").partition(">")
        if not call.startswith("#") or not declared:
            return origin
        name, dot, member = declared.partition(".")
        return f"{name}(){dot}{member}"

    def _bind_closures(self, code, index, lines, before):
        """Type the parameters of an arrow function that opens before
        ``before`` as the value of an annotated declaration, an
        assignment or a `return`, from the function type it is given;
        arguments of calls go on to the shared path."""
        while self._closures and self._closures[0][0] < before:
            bar, params, _end = self._closures[0]
            if code[: _run_start(code, bar)].endswith("=>"):
                # What another function returns: a call passes it nothing.
                self._closures.pop(0)
                continue
            if self.bracket_around(bar)[1] == "(":
                # One in parentheses is a call's argument, or a function
                # type an annotation writes, which binds nothing.
                super()._bind_closures(code, index, lines, bar + 1)
                continue
            head = code[:bar]
            origin = None
            if re.search(r"\breturn\s*$", head):
                # `return (x) => ...`: the parameters the enclosing
                # function's declared return type takes.
                function_id = self.context_at(index + 1)[0]
                written = (self._signatures.get(function_id) or [{}])[-1].get("returns")
                if written:
                    origin = f">{_path_written(written)}"
            else:
                # The declarator the closure is the value of, after the
                # first in a statement too: `..., g: Fn =`
                declared = re.search(
                    r"(?:\b(?:const|let|var)\s+|,\s*)([\w$]+)"
                    r"(?:\s*:\s*((?:=>|<[^<>]*>|\((?:[^()]|\([^()]*\))*\)|[^;=,(<])*?))?\s*=\s*$",
                    head,
                )
                later = declared and declared[0].startswith(",")
                if later and not re.search(r"\b(?:const|let|var)\b", head):
                    declared = None
                assigned = re.search(r"(?<![\w$.])([\w$]+)\s*=(?![=>])\s*$", head)
                if assigned and head[: assigned.start()].rstrip().endswith("=>"):
                    # What follows `=>` is a function type's return, no target.
                    assigned = None
                if declared and declared[2]:
                    # The annotation is the closure's own type: read its
                    # parameters where they are written, not through the
                    # variable — a definition the value becomes does not
                    # hold it as a member. A declaration without a type
                    # gives the closure none.
                    origin = f">{_path_written(declared[2].strip())}"
                elif assigned and not declared:
                    origin = f"={assigned[1]}"
            self._closures.pop(0)
            if origin is None:
                # A function type an annotation writes outside parentheses
                continue
            bindings = self._param_bindings(f"{origin}.^", params)
            self._bind(bindings, index, bar)
        while self._held_closures and self._held_closures[0][0] < before:
            # A function given as a property of an object literal that a
            # call is passed takes what the call passes to that property.
            bar, params = self._held_closures.pop(0)
            if held := self._held_label(code, bar):
                bindings = self._param_bindings(f"{held}|", params)
                self._bind(bindings, index, bar)
        super()._bind_closures(code, index, lines, before)

    def _param_bindings(self, origin, params):
        """``{name: origin path}`` of the names the parameter list ``params``
        binds: ``origin`` and the position of the parameter, the steps into
        it of a pattern after."""
        return {
            name: _bound_origin(f"{origin}{number}", steps)
            for number, param in enumerate(_bracketed_parts(params + ")"))
            for name, steps in self.bindings(param)
        }

    def _element_origin(self, end, callee, lineno, since, source_id):
        """`items.*.callee` for a call on an element of an array,
        `items[i].callee()`: what iterating `items` gives, and
        `items.@N.callee` for a literal index N."""
        code, at = self._code, self._index_open(end)
        if at is None:
            return None
        if code[at - 1 : at] == ")":
            edge = self._closes.get(at - 1)
            held = None if edge is None else f"#{edge}"
        else:
            path = re.search(r"(?<![\w$.])(?:this\.)?[\w$]+(?:\.[\w$]+)*$", code[:at])
            path = path[0] if path else self._continued_path(code[:at])
            owner = self._owning_type(source_id)
            held = path and self.call_origin(f"{path}.x", lineno, since, owner)
            held = held and held[:-2]
        # A literal index reads a tuple's element at it.
        index = code[at + 1 : end - 1].strip()
        step = f"@{index}" if index.isdigit() else "*"
        return held and f"{held}.{step}.{callee}"

    def _held_label(self, code, column):
        """``#edge|argument`` for what starts at ``column`` of the line
        being read where it is an argument of a call on the line, and
        ``...:key`` where it is the value of the property `key` of an object
        literal so labelled, opened on the line or an earlier one; else
        None."""
        opened, bracket = self.bracket_around(column)
        if bracket == "(" and opened is not None:
            edge = self._closes.get(self._pairs.get(opened), self._opens.get(opened))
            parts = _bracketed_parts(code[opened + 1 : column] + "x", expression=True)
            if edge is None or parts[-1].strip() != "x":
                return None
            return f"#{edge}|{len(parts) - 1}"
        key = bracket == "{" and re.search(
            r"(?:^|[{,])\s*(?:(?:async\s+)?\*?\s*([\w$]+)|([\w$]+)\s*:\s*(?:async\s+)?)$",
            code[opened or 0 : column],
        )
        if not key:
            return None
        held = (
            self.held_around(column)
            if opened is None
            else self._held_label(code, opened)
        )
        return held and f"{held}:{key[1] or key[2]}"

    def _this_in_object(self, code, column):
        """``#edge|argument|this`` origin for a `this` at ``column`` that is
        inside an object literal given as an argument of a call on the
        line, else None."""
        opened = self.paren_around(column)
        if opened is None:
            return None
        # The object literal holds the `this`: a `{` inside the call's
        # arguments that is still open at the `this`.
        if code[opened:column].count("{") <= code[opened:column].count("}"):
            return None
        edge = self._closes.get(self._pairs.get(opened), self._opens.get(opened))
        if edge is None:
            return None
        brace = code.index("{", opened)
        position = (
            len(_bracketed_parts(code[opened + 1 : brace] + "x", expression=True)) - 1
        )
        return f"#{edge}|{position}|this"

    def _operand(self, text):
        operand = re.sub(r"^await\s+", "", text)
        return operand.rstrip("!"), len(operand) < len(text)

    def _constructed(self, text):
        """`new T(...)` constructs a `T`."""
        found = _JS_CONSTRUCTION_RE.match(text)
        if found and not _strip_parens(text[found.end() - 1 :]).strip():
            return f"{self._scanned_type({'type': found[1]})}()"
        return None

    def _bind_patterns(self, code, index, lines, since, owner):
        """Type the names that a `for` or a declaration with a pattern
        binds on line ``index``, from the value iterated or taken apart. A
        `for...in` gives the names of properties."""
        head = _JS_FOR_START_RE.search(code)
        close = head and self._pairs.get(head.end() - 1)
        found = close and _JS_FOR_PARTS_RE.fullmatch(code, head.end(), close)
        if not found:
            found = _JS_PATTERN_RE.search(code)
        pattern = found and found["pattern"]
        # The offset where the pattern is written
        written = found and self._scopes.starts[index] + found.start("pattern")
        kind = found and found.groupdict().get("kind")
        span = found and found.span("value")
        if not found and (over := self._for_head_over_lines(code, index, lines)):
            found, (pattern, kind, span, _, head) = True, over
            written = self._scopes.starts[head[0]] + head[1]
        if not found and (opened := self._scopes.patterns.get(index + 1)):
            # A pattern over lines that closes on this one
            written, closed = opened
            column = closed - self._scopes.starts[index] + 1
            if not (found := _JS_PATTERN_VALUE_RE.match(code, column)):
                return
            pattern = self._scopes.text[written : closed + 1]
            span = found.span("value")
        if not found:
            return
        lineno = index + 1
        origin = self._expression_origin(code, *span, lineno, since, owner)
        if origin is None and not kind:
            origin = self._opened_value(code, span[0], index, lines)
        if kind == "of" and code[span[0]] == "[" and code[span[1] - 1] == "]":
            key = f"[{self.builder.serial(self.builder.literals)}]"
            self.builder.literals[key] = self._elements(
                code, span[0] + 1, lineno, since, owner
            )
            origin = f"Array<{key}>()"
        if kind == "of":
            origin = origin and f"{origin}.*"
        elif kind:
            origin = "string()"
        bindings = {
            name: _bound_origin(origin, steps) for name, steps in self.bindings(pattern)
        }
        self._bind_from(bindings, written)

    @staticmethod
    def _for_head_over_lines(code, index, lines):
        """``(pattern, kind, value span, close, head)`` of a `for` head an
        earlier line opens whose value starts on line ``index``, of code
        ``code``: the span is the value's columns there, ``close`` the column
        of the `)` that closes the head where it is on the line, and
        ``head`` the line index and column the head starts at after its `(`.
        None for any other line."""
        for row in range(index - 1, max(index - 6, -1), -1):
            if heads := list(_JS_FOR_START_RE.finditer(lines[row])):
                break
        else:
            return None
        head = row, heads[-1].end()
        before = " ".join([lines[row][head[1] :], *lines[row + 1 : index]])
        depth = 1 + before.count("(") - before.count(")")
        if depth < 1:
            return None
        row, column = index, len(code)
        while depth > 0 and row < len(lines):
            for column, char in enumerate(lines[row]):
                depth += (char == "(") - (char == ")")
                if not depth:
                    break
            else:
                row, column = row + 1, len(code)
        if depth:
            return None
        current = code[: column if row == index else len(code)]
        parts = _JS_FOR_PARTS_RE.fullmatch(f"{before} {current}")
        shift = len(before) + 1
        if not parts or parts.start("value") < shift:
            return None
        span = parts.start("value") - shift, parts.end("value") - shift
        close = column if row == index else None
        return parts["pattern"], parts["kind"], span, close, head

    def _binds_file(self, name):
        """A line outside every definition that starts at the margin, as
        the body of a callback does not, binds for the module, unless the
        name is an import's. The margin is where the least indented line of
        the text being parsed starts: a `<script>` block may be indented as
        a whole."""
        margin = len(self._raw) - len(self._raw.lstrip()) == self._margin
        return self._scope is None and margin and name not in self._imported

    def call_origin(self, callee, lineno, since, owner=None):
        """As ``BaseParser.call_origin``, and for a bare callee that a
        pattern bound to a member of a value, `const { run } = make()`, the
        origin path of that member, and for one declared `typeof x`, `x`. A
        receiver a function or a block binds, of a value not known, is of a
        type not known, `?.member`, whatever the module defines of its name,
        unless the binding is itself a definition."""
        bound = "." not in callee and callee not in self._imported
        held = bound and self._declared(callee, lineno, since)
        if held and re.search(r"(?:^|\.)[\w$]+$", held):
            return held
        origin = super().call_origin(callee, lineno, since, owner)
        root, dot, rest = callee.partition(".")
        if origin is None and dot and root not in self._imported:
            inner = list(islice(self._bindings(root, lineno), 1))
            if inner and inner[0][3] != self._scopes.whole:
                return None if self._defines(root, inner[0][1][0]) else f"?.{rest}"
        return origin

    def _read(self, lineno):
        """Where a name on line ``lineno`` is read: from and to column
        ``_read_at`` where known, else anywhere on the line."""
        if self._read_at is None:
            return (lineno, 0), (lineno, 1 << 30)
        return (lineno, self._read_at), (lineno, self._read_at)

    def _bindings(self, name, lineno):
        """The bindings of ``name`` in scope where it is read on line
        ``lineno`` (see ``_read``): the innermost first, of those of one
        scope the last written."""
        return self._scopes.visible(name, *self._read(lineno))

    def _declaration(self, name, lineno):
        """The innermost of ``_bindings`` that declares ``name``, not an
        assignment; None for none."""
        return self._scopes.declaration(name, *self._read(lineno))

    def _local_at(self, source_id, name, lineno, column):
        """As the base's, where the binding of ``name`` there is a block's
        or a function's, not the file's."""
        bound = self._scopes.declaration(name, *[(lineno, column)] * 2)
        return (
            super()._local_at(source_id, name, lineno, column)
            and bound is not None
            and bound[3] != self._scopes.whole
        )

    def _binding_scope(self, name, lineno, column):
        """``(holds from, to)`` of the binding of ``name`` whose name is
        written at ``(lineno, column)``: where its scope starts and ends,
        the position a call it shadows an import around must be at. The
        scope is the binding's own span, which starts at its block or
        function: a hoisted or still-unwritten local is the binding all
        the same. None where the scope reader knows no binding of it."""
        # (this file's scopes, each binding's span by its name and where the
        # name is written)
        held = getattr(self, "_written_at", None)
        if held is None or held[0] is not self._scopes:
            self._written_at = (
                self._scopes,
                {
                    (held, bound[2][0]): bound[3]
                    for held, bounds in self._scopes.bindings.items()
                    for bound in bounds
                },
            )
        return self._written_at[1].get((name, (lineno, column)))

    def _declared(self, name, lineno, since, anywhere=False):
        """As the base's for a field; any other name is read as the
        innermost binding in scope declares or assigns it, where one of no
        value known, as an untyped parameter or an assignment of a value not
        read, hides those around it, and one read before it is set has no
        value. A function inside reads the binding written before it or any
        of its scope written after, and past an assignment that may not have
        run the variable holds what it gives or what it held: where they
        differ or one is not known the value is not known, but for a
        variable declared with a type."""
        if anywhere:
            return super()._declared(name, lineno, since, anywhere)
        entries = self.declared.get(name)
        # `value` re-reads the entries: a set of its results is stale from
        # the next write of one.
        fingerprint = entries and entries.version, len(self._members)

        def value(bound):
            """What the binding ``bound`` gives the name: the last of the
            declarations written in its text. A callback's parameter and a
            declaration on one line are told apart by what their value is."""
            first, last = bound[2]
            found = [
                d
                for d in self.declared.between(name, first[0], last[0])
                if (d[0], name) not in self._members and first <= (d[0], d[3]) <= last
            ]
            param = bound[0] == "param"
            kept = [d for d in found if bool(_CALLBACK_RE.match(d[1] or "")) == param]
            return (kept or found or [(0, None)])[-1][1]

        read = self._read(lineno)[1]
        bindings = self._bindings(name, lineno)
        declared = self._declaration(name, lineno)
        reader = self._scopes.function(read)
        # A function reads a variable from around it after any assignment
        # another function makes, or an earlier run of itself made.
        funcs = (
            self._scopes.assignment_funcs(name, declared[3])
            if declared and self._scopes.function(declared[1]) != reader
            else None
        )
        values = set()
        if funcs and funcs != {reader}:
            chosen = [declared]
            values.update(
                self._scopes.assigned_values(name, declared[3], fingerprint, value)
            )
        else:
            chosen = []
            for bound in bindings:
                if chosen and bound[3] != chosen[0][3]:
                    break
                if self._scopes.unset(bound, read):
                    return None
                chosen.append(bound)
                if bound[1] <= read:
                    values.update(
                        self._scopes.reassigned_values(
                            name,
                            bound[3],
                            bound[1],
                            read,
                            entries.changes if entries else (),
                            value,
                        )
                    )
                    if bound is declared and funcs is not None:
                        values.update(
                            self._scopes.assigned_values(
                                name, declared[3], fingerprint, value
                            )
                        )
                    break
        # `undefined` has no members a call through the name could reach.
        undefined = self._scopes.undefined
        values.update(value(b) for b in chosen if (name, b[1]) not in undefined)
        if len(values) == 1 and None not in values:
            return values.pop()
        # A variable declared with a type holds a value of that type.
        typed = declared and self._scopes.typed(name, declared)
        return value(declared) if typed else None

    def _assigned(self, code, name, at, start, end, lineno, since, owner):
        """Origin path of the value at columns ``start`` to ``end`` of
        ``code``, line ``lineno``, that an assignment of ``name`` written at
        ``(line, column)`` ``at`` gives it: of the call or the name it is.
        None where the line binds the name again or the scan gives it a
        type of its own, as `new T()` does, and for a variable declared
        with a type, which holds that type."""
        if self._scopes.binds_again(name, lineno) or any(
            d[3] == at[1] for d in self.declared.between(name, at[0], at[0])
        ):
            return None
        declared = self._read_from(start, self._declaration, name, lineno)
        if declared and self._scopes.typed(name, declared):
            return None
        held = self._expression_origin(code, start, end, lineno, since, owner)
        return None if held and re.search(r"(?:^|\.)[\w$]+$", held) else held

    def _object_held(self, lines, index, column, since, owner):
        """What each property of the object literal that opens at ``column``
        of line ``index`` holds, by name: the origin path of the name or
        member it is given as, of the call written for it, or of the call's
        result that a local it names was bound to; its own name where it is
        a method or a function written there, None where it is anything
        else. A property under a name that is no plain one, a spread among
        them, is left out. With it, the number of the line that closes the
        object. ``lines`` are the file's code lines."""
        written, depth = [], 0
        for line in chain([lines[index][column:]], lines_after(lines, index)):
            written.append(_blank_js_strings(line))
            depth += written[-1].count("{") - written[-1].count("}")
            if depth <= 0:
                break
        text = "\n".join(written)
        depth = 0
        for end, char in enumerate(text):
            depth += (char == "{") - (char == "}")
            if not depth:
                break
        body = text[1:end]
        # The text around a nested part keeps the column it is written at.
        while (reduced := _JS_NESTED_RE.sub(_nested_part, body)) != body:
            body = reduced
        held = {}
        at = 1  # where the part being read starts in `text`
        for piece in body.split(","):
            start = at + len(piece) - len(piece.lstrip())
            at += len(piece) + 1
            part = piece.strip()
            if not part:
                continue
            head, colon, tail = part.partition(":")
            name, value = head.strip(), tail.strip()
            # A function written in the object is a member of it, filed
            # under the object's own ID, where it is written below the line
            # the object opens on. That line is read as what opens it, so a
            # function on it gets no node.
            member = start > len(written[0])
            method = re.fullmatch(
                r"(?:async\s+|(get|set)\s+)?\*?\s*([\w$]+)\s*\0\s*\0", name
            )
            if method and not colon:
                # An accessor is filed as `get name` or `set name`.
                filed = " ".join(filter(None, method.groups()))
                held[filed] = f"={filed}" if member else filed
            elif not re.fullmatch(r"[\w$]+", name):
                continue
            elif not colon or re.fullmatch(r"[\w$]+(?:\.[\w$]+)*", value):
                held[name] = self._name_held(value or name, index + 1, since)
            else:
                made = "=>" in value or re.match(r"(?:async\s+)?function\b", value)
                value_at = start + len(head) + 1 + len(tail) - len(tail.lstrip())
                if made:
                    held[name] = f"={name}" if member else name
                else:
                    self._call_held(
                        held, name, text, index, column, value_at, at - 1, since, owner
                    )
        return held, index + 1 + text.count("\n", 0, end)

    def _returned_object_at(self, returned_objects, parent_id, lineno):
        """The object that the definition ``parent_id`` returns at the line
        being read, where the line is one of its lines: the innermost of
        those it returns that reaches it, as its ID and the line the names
        written in it are read from — the definition that returns it."""
        for parent, first, last, owner_id, scope in reversed(returned_objects):
            if parent == parent_id and first <= lineno <= last:
                return owner_id, scope
        return None, None

    def _name_held(self, value, lineno, since):
        """What a property of the object being read holds where the value it
        is given is a name: the origin path of that name where a local of
        the function was bound to a call's result, else the name itself."""
        bound = re.fullmatch(r"[\w$]+", value) and self._declared(value, lineno, since)
        return bound if bound and bound.startswith("#") else value

    def _call_held(self, held, name, text, index, column, start, end, since, owner):
        """Set ``held[name]`` to the origin path of the call that the
        property at ``start`` to ``end`` of the text of the object that
        opens at ``column`` of line ``index`` is given: now for one on that
        line, the line being read, else once its own line is read."""
        row = text.count("\n", 0, start)
        held[name] = None
        if row != text.count("\n", 0, end):
            return
        if row:
            line_start = text.rfind("\n", 0, start) + 1
            later = (held, name, start - line_start, end - line_start)
            self._held_later.setdefault(index + row, []).append(later)
            return
        held[name] = self._expression_origin(
            self._code, column + start, column + end, index + 1, since, owner
        )

    def _opened_value(self, code, start, index, lines):
        """Origin path of a value that is one call which ``code``, line
        ``index``, leaves open from column ``start`` and a later line
        closes, ending the statement. None for any other value."""
        opened = sorted(column for column in self._opens if column >= start)
        if not opened or not _is_call_chain(code[start : opened[0]]):
            return None
        depth = 0
        for line in chain([code[opened[0] :]], lines_after(lines, index)):
            for column, char in enumerate(line):
                depth += (char == "(") - (char == ")")
                if not depth:
                    # A `(` no call was read at, as `import(`'s, holds no
                    # call's result.
                    edge = self._opens[opened[0]]
                    ended = not line[column + 1 :].strip(" ;")
                    return f"#{edge}" if ended and edge is not None else None
        return None

    def _ends_value(self, code, edge):
        """Whether ``code``, the line being read, closes the call at edge
        index ``edge`` and has nothing after it but the end of the
        statement; None where it does not close it."""
        closed = [column for column, call in self._closes.items() if call == edge]
        return not code[max(closed) + 1 :].strip(" ;") if closed else None

    def _elements(self, code, start, lineno, since, owner):
        """The elements of an array literal or the arguments of a call that
        ``code``, the line being read, has from column ``start`` up to the
        bracket that closes it or to its end: the origin path of each, for
        an object literal those of its properties by name, None where it
        has none."""
        elements = []
        closed = self._pairs.get(start - 1)
        for part in _bracketed_parts(code[start:closed], expression=True):
            start = code.index(part, start)
            end = start + len(part)
            if part[0] + part[-1] != "{}":
                held = self._expression_origin(code, start, end, lineno, since, owner)
                elements.append(held)
                start = end
                continue
            held = {}
            for item in _bracketed_parts(part[1:], expression=True):
                start = code.index(item, start)
                name, colon, _ = item.partition(":")
                span = start + len(name) * bool(colon) + bool(colon), start + len(item)
                held[name.strip()] = self._expression_origin(
                    code, *span, lineno, since, owner
                )
                start = span[1]
            named = all(re.fullmatch(r"[\w$]+", name) for name in held)
            elements.append(held if named else None)
        return elements

    def parse(self, file_path, rel_path, content, line_offset=0):
        """Parse ``content``: the text of ``file_path``, or one ``<script>``
        block of it that starts ``line_offset`` lines in. Emitted line
        numbers are relative to ``content``."""
        file_id = self.add_file_node(rel_path, content)

        jsx = str(file_path).endswith(_JSX_EXTENSIONS)
        lines = source_lines(content)
        self.line_offset = line_offset
        # The source lines blanked, and the offset each starts at
        source_code = _blank_js_lines(lines, jsx=jsx)
        source_starts = [0]
        for line in lines:
            source_starts.append(source_starts[-1] + len(line) + 1)
        if rel_path.endswith((".ts", ".tsx", ".mts", ".cts")) and (
            spaces := list(_TS_ONLY_SPACE_RE.finditer(content))
        ):
            # A TypeScript file's code reads that white space as spaces.
            text = list(content)
            for found in spaces:
                row = bisect_right(source_starts, found.start()) - 1
                code = source_code[row][found.start() - source_starts[row] :]
                if code[:1] == found[0]:
                    text[found.start()] = " "
            content = "".join(text)
            lines = source_lines(content)
            source_code = _blank_js_lines(lines, jsx=jsx)

        def in_code(found):
            """Whether the statement ``found``, a match in ``content``,
            starts in code: not in a comment or a literal."""
            at = found.start() + len(found[0]) - len(found[0].lstrip(_JS_SPACE))
            row = bisect_right(source_starts, at) - 1
            return source_code[row][at - source_starts[row] :][:1] == content[at]

        # The lines read, a definition after another's end on a line on one
        # of its own, and the source line each is part of
        lines = self.split_source(
            lines,
            source_code,
            _TS_DEFINITION_HEAD_RE,
            _TS_BODY_HEAD_RE,
            ";,",
            _TS_STATEMENT_START_RE,
            objects=True,
            blocks=True,
        )

        written = (line for line in lines if line.strip())
        self._margin = min((len(l) - len(l.lstrip()) for l in written), default=0)
        self.enter_context(file_id, 0, len(lines))

        # A class at a line's margin, or a class expression: `const C =
        # class`, as `let` or `var`, `member = class`, and inside brackets
        # `new (class` or `call(a, class`, where `class` is no property's
        # name. Searched on the line with its literals blanked: an
        # expression is written anywhere on a line, and a string may hold
        # one.
        re_class = re.compile(
            r"(?P<expression>"
            r"(?:const|let|var)\s+(?P<declarator>[\w$]+)\s*(?::[^=,;]*)?=\s*"
            r"|^\s*[\w$][\w$.]*\s*=\s*"
            r"|(?P<enclosed>[(,]\s*|[\w$]\s*=\s*|\breturn\s+|=>\s*|[:?]\s*)"
            r"(?=class\s*(?:[\w$]+\s*)?(?:[{<]|(?:extends|implements)\b)))"
            r"class\b"
            r"|^\s*(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?class\b(?!\s*=)"
            r"(?:\s+(?!extends\b|implements\b)(?P<plain>[a-zA-Z0-9_$]+))?"
        )
        # `interface Name` and an object type, `type Name<T> = {`
        re_interface = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?"
            r"(?:interface\s+([\w$]+)|type\s+([\w$]+)\s*(?:<[^=]*>)?\s*=\s*(\{|$))"
        )
        # A function `export default cond &&` gives is a declaration too.
        re_func = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?(?:default\s+(?:[\w$.]+\s*&&\s*)*)?"
            r"(?:async\s+)?function\s+([a-zA-Z0-9_$]+)"
        )
        # `name(`, or `name = (` where a class property holds an arrow function,
        # either with type parameters before the `(`
        re_method = re.compile(
            rf"^\s*{_TS_MEMBER_MODIFIERS}"
            rf"(?:\*\s*)?({_TS_WRITTEN_MEMBER_NAME})\s*(=\s*(?:async\s+)?)?(?:[(]|<(?![\s=]))"
        )
        # Getter/setter methods: get name() or set name(value)
        re_getter_setter = re.compile(
            rf"^\s*{_TS_MEMBER_MODIFIERS}(get|set)\s+({_TS_WRITTEN_MEMBER_NAME})\s*[(]"
        )
        re_enum = re.compile(r"^\s*(?:export\s+)?enum\s+([a-zA-Z0-9_$]+)")
        # `const name: Type = value;`, the type kept where it is one named
        # type, or `const name` of a statement that goes on over the next lines
        re_const = re.compile(
            r"^\s*(?:export\s+)?(?:declare\s+)?const\s+([a-zA-Z0-9_$]+)"
            r"(?:\s*:\s*(\([^()]*\)\s*=>\s*[\w$.]+|[\w$.]+(?:<[^|&]*>)?)"
            r"(?=\s*(?:=(?!>)|;?\s*$)))?"
            r"(?:\s*(?::\s*.*?)?\s*=(?![=>])\s*(.*?);)?"
        )
        re_import = re.compile(r'import\s+.*?from\s+([\'"])(.*?)\1')
        re_call_candidate = re.compile(
            rf"({_TS_MEMBER_NAME}(?:!?\??\.{_TS_MEMBER_NAME})*)\s*!?(?:\?\.)?(?=[<(])"
        )
        # In an object literal: `...name`, `key: name` or `name` alone
        re_held = re.compile(r"^\s*(\.\.\.)?(?:([\w$]+)\s*:\s*)?([\w$]+)\s*,?\s*$")
        # Which characters are code is read once for the file. Each of
        # ``code_lines`` is a line without its comments and with the text of
        # its literals blanked, for the names it binds and the braces
        # counted on it.
        code_lines = self._code_lines = _blank_js_lines(lines, jsx=jsx)
        # A JavaScript file that is no module, by its syntax or its package,
        # is a script.
        script = rel_path.endswith((".js", ".jsx", ".cjs")) and not (
            self.builder.module or _JS_MODULE_RE.search("\n".join(code_lines))
        )
        self._scopes = _JsScopes(code_lines, lines, script)
        scopes = self._scopes
        # The blocks scope a type parameter.
        self.read_blocks(code_lines, (scopes.text, scopes.starts, scopes.closer))
        # What a module declares is its own: a file with an `import` or an
        # `export`, one its extension makes one, or a JavaScript file that
        # CommonJS does.
        if (
            any(
                scopes.block(found.end()) == scopes.whole
                for found in _JS_MODULE_RE.finditer(scopes.text)
            )
            or _JS_IMPORT_META_RE.search(scopes.text)
            or rel_path.endswith((".mjs", ".cjs", ".mts", ".cts"))
            or rel_path.endswith((".js", ".jsx"))
            and _JS_COMMONJS_RE.search(scopes.text)
            and self._declaration("require", len(code_lines)) is None
        ):
            self.builder.js_modules.add(file_id)
        self.scan_declarations(lines, _TS_DECLARATIONS)
        self.builder.type_aliases.update(_TYPE_ALIAS_RE.findall(content))
        for name, generics, text in _TS_ALIAS_RE.findall(content):
            parts = _bracketed_parts(f"{generics}>")
            params = tuple(part.split()[0] for part in parts if part.split())
            self.builder.ts_aliases[file_id, name] = params, text.strip()
        # What the file's `export`s give, ``reexports`` entries
        es = []
        for found in filter(in_code, _TS_REEXPORT_RE.finditer(content)):
            star, namespace, clause, specifier = found.groups("")
            clause = _spaced(clause, _JS_GAP)
            # `export * as ns` exports the module itself, under a name.
            names = [(namespace or "*", "*" if namespace else "")] if star else []
            for part in clause.split(","):
                # `name`, `type name`, `name as alias`
                original, _, alias = part.partition(" as ")
                if original.split():
                    names.append(
                        ((alias or original).split()[-1], original.split()[-1])
                    )
            es.extend(
                (exported, specifier, original, file_id) for exported, original in names
            )
        for found in filter(in_code, _TS_DEFAULT_EXPORT_RE.finditer(content)):
            default, *names = found.groups("")
            row = bisect_right(source_starts, found.end()) - 1
            if names[0] or _ends_after(
                source_code, row, found.end() - source_starts[row]
            ):
                es.append(("default" if default else "=", "", "".join(names), file_id))
        names = self.builder.import_names.setdefault(file_id, {})
        for found in filter(in_code, _TS_IMPORT_RE.finditer(content)):
            clause, specifier = found.groups("")
            clause = _spaced(clause, _JS_GAP)
            for part in re.finditer(r"[^{},]+", clause):
                # `name`, `type name`, `name as alias`, `* as alias`; before
                # a `{`, the module's default
                original, renamed, alias = part[0].partition(" as ")
                local = alias.split()[-1:] if renamed else original.split()[-1:]
                if local:
                    self.builder.imports.setdefault(file_id, {})[local[0]] = specifier
                    if original.split() == ["*"]:
                        names[local[0]] = "="
                    elif "{" not in clause[: part.start()]:
                        names[local[0]] = "default"
                    elif renamed and original.split():
                        names[local[0]] = original.split()[-1]
        # The lines a `require` or an `import()` binds names on
        required = set()
        for found in _TS_REQUIRE_RE.finditer(content):
            clause, call, specifier = found.groups()
            clause = _spaced(clause, _JS_GAP)
            line = self.read_line(bisect_right(source_starts, found.start()))
            required.add(line - 1)
            if "{" not in clause:
                names[clause] = "="
            for part in re.split(r"[{},]", clause):
                # `name`, or `name: local` in a destructuring pattern
                exported, renamed, local = part.rpartition(":")
                local = local.strip()
                if local:
                    self.builder.imports.setdefault(file_id, {})[local] = specifier
                    table = (
                        self.builder.required_names
                        if call == "require"
                        else self.builder.dynamic_names
                    )
                    table.setdefault(file_id, {})[local] = line
                    if renamed and exported.strip():
                        names[local] = exported.strip()
        imported = self._imported = self.builder.imports.get(file_id, ())
        bound = _js_bound("\n".join(code_lines))
        self.builder.js_bound[file_id] = bound.union(imported)
        # An imported name is what its module exports, whatever a field or
        # a parameter of that name is declared as.
        for name in imported:
            self.declared.pop(name, None)
        # A parameter pattern with a type binds the parts of a value of it,
        # and one of an object or a tuple type a value of that; the scan
        # reads a name of a named type.
        for row, regions in self._scopes.params.items():
            for start, close, *_, written in regions:
                listed = written[0] == "(" and start[0] == row
                # As written, its literal types too
                source = "\n".join(lines[start[0] - 1 : close[0]])
                source = source[
                    start[1] : len(source) - len(lines[close[0] - 1]) + close[1] + 1
                ]
                offset, at = scopes.starts[start[0] - 1] + start[1], 0
                for part in _bracketed_parts(source[1:]) if listed else ():
                    part_at = source.index(part, at)
                    at = part_at + len(part)
                    typed = _TS_TYPED_PARAMETER_RE.fullmatch(part)
                    written = typed and typed[2].removeprefix("readonly ").strip()
                    if not typed or typed[1][0] not in "[{" and written[0] not in "[{":
                        continue
                    held = f"{typed[2]}()"
                    if written[0] in "[{":
                        held = (
                            f"{{{self.builder.serial(self.builder.parameter_types)}}}"
                        )
                        self.builder.parameter_types[held] = typed[2], file_id
                    for name, steps in _js_bindings(typed[1]):
                        value = _bound_origin(held, steps)
                        named = scopes.place(
                            _named(scopes.text, name, offset + part_at)
                        )
                        self.declared.setdefault(name, []).append(
                            (named[0], value, False, named[1])
                        )
        # Interceptor, middleware and event-handler registrations, filed as
        # Configuration nodes
        re_interceptor = re.compile(
            r"^\s*([a-zA-Z0-9_]+)\.interceptors\.(request|response)\.use\s*\("
        )
        re_middleware = re.compile(
            r"^\s*(?:app|router|server)\.use\s*\(\s*(['\"]?)([^'\"(),]+)\1"
        )
        re_event_handler = re.compile(
            r"^\s*([a-zA-Z0-9_]+)\.on\s*\(\s*['\"]([a-zA-Z0-9_:]+)['\"]"
        )

        # The variable the line before declared, that line and its column
        untyped = ("", 0, None)
        # The call that starts the value of a variable declared with a type,
        # and that type, until the call is closed
        initialises = None
        array = None  # the array literal being read, and its open brackets
        # The constant whose value is an array over lines, and its last line
        elements_of = None
        # The call a `return` gives that is still open, with where it is
        # recorded, and the last line of the nameless functions each
        # definition being read has open: a `return` past it is that
        # function's, not the definition's.
        returning = None
        nameless = {}
        # Each object a definition returns: the definition, the lines the
        # object spans and the object's own ID. A function written on one of
        # those lines is a member of the object, not of the definition.
        returned_objects = []
        # Line index -> (what an object holds, property, columns of its
        # value) for each property given on that line of an object read before
        self._held_later = {}
        self._held_closures = []
        # Each callback given an owner, innermost last: its last line and how
        # many calls are open around it. A call left open inside it, not one
        # around it, makes a closure on a later line one of its arguments.
        callbacks = []
        callback_ids = set()  # the owners of those callbacks
        # The contexts that end on the line before the one their block closes
        # on: those of callbacks and of nameless functions
        closed_after = set()
        expression_ids = set()  # functions whose body is an expression
        # The definition, and its first and last line and column, of each
        # context: a column of None where it holds the whole line
        extents = []
        # The brackets open where the line being read starts, innermost
        # last, True for the `(` of a call, and the last character of code
        # before the line: an object that follows such a `(`, or a `,`
        # inside it, is an argument of the call.
        brackets = []
        before = ""
        self._signatures = {}  # node ID -> the signatures it is declared with
        class_ids = set()  # class/interface/enum nodes declared in this file
        namespaces = {}  # namespace ID -> the metadata of its node
        class_at = {}  # class ID -> the column its `class` ends at on its line
        # constants of the file or a callback that hold an object literal
        object_ids = set()
        # Each name the file's CommonJS exports object holds, in the order
        # the file writes it: `module.exports =` replaces it and orphans
        # `exports`, which writes the orphan from there on.
        cjs = {}
        cjs_orphan = False
        interface_ids = set()  # interfaces that have the ID `<scope>::<name>`
        type_bodies = set()  # interfaces and object types, whatever their ID
        value_ids = set()  # functions and variables, which always have it
        # What prototype assignments file members under: the prototypes,
        # `F.prototype`, made here, and classes
        prototype_ids = set()
        # The functions whose prototypes the file writes to
        prototyped = set(
            re.findall(rf"{_JS_START}([\w$]+)\.prototype\b", scopes.text, re.MULTILINE)
        )

        def declared(name, lineno):
            """``(ID, kind)`` of the function or the class that ``name``
            names where line ``lineno`` reads it, `function` or `class`:
            the binding in scope, declared in the innermost definition
            around the line that holds the declaration, the declaration's
            own aside. A function declaration is hoisted; a variable holds a
            function from where it is given one. None for anything else."""
            bound = self._declaration(name, lineno)
            if bound is None or bound[0] not in _JS_DECLARING_KINDS:
                return None
            row = bound[1][0]
            scope = next(c[0] for c in reversed(self._contexts) if c[1] < row <= c[2])
            owner_id = f"{scope}::{name}"
            if bound[0] == "class":
                return owner_id, "class"
            node = self._node(owner_id)
            value = node and node["metadata"].get("value")
            if (
                node
                and (node["type"] == "Function" or value and _function_value(value))
                or bound[0] == "function"
                and not node
                and row > lineno
            ):
                return owner_id, "function"
            return None

        def prototype(name, lineno, member=True):
            """The ID of what the instances of the function or class
            ``name``, as line ``lineno`` names it, take a member assigned to
            its prototype from: the class, or the function's prototype,
            `F.prototype`, made on first use. Without ``member`` the
            prototype itself is assigned, which a class's cannot be. None
            where ``name`` names no such function or class."""
            found = declared(name, lineno)
            if not found or found[1] == "class" and not member:
                return None
            if found[1] == "class":
                prototype_ids.add(found[0])
                return found[0]
            made = f"{found[0]}.prototype"
            if made not in prototype_ids:
                self.builder.add_node(
                    made,
                    "Class",
                    name=f"{name}.prototype",
                    kind="prototype",
                    lineno=lineno,
                )
                self.builder.add_edge(found[0].rpartition("::")[0], made, "DEFINES")
                prototype_ids.add(made)
            return made

        # The static members: those a class body declares `static`, and those
        # assigned to a function or a class
        static_ids = set()

        def free_key(owner_id, name):
            """The key a member ``name`` assigned to ``owner_id`` is filed
            under: ``name``, and given again `name#2` and on."""
            key, count = name, 1
            while self._node(f"{owner_id}::{key}"):
                count += 1
                key = f"{name}#{count}"
            return key

        def member_id(owner_id, name, lineno):
            """The ID of the member ``name`` that line ``lineno`` writes in
            the body of ``owner_id``: of a class, one declared `static` is
            the constructor's, and a static and an instance member of one
            name are two."""
            member = f"{owner_id}::{name}"
            row = code_lines[lineno - 1] if owner_id in class_ids else ""
            static = _declared_static(row, name.rpartition(" ")[2])
            if self._node(member) and static != (member in static_ids):
                member = f"{owner_id}::{free_key(owner_id, name)}"
            static_ids.update([member] if static else ())
            return member

        def entry_name(name, lineno):
            """``name`` as an export entry written on line ``lineno`` holds
            it, read in the definition around it: None where a block
            declares it."""
            bound = self._declaration(name, lineno)
            return None if bound and scopes.in_block(bound) else name

        def exported_object(lineno, last):
            """The object that `module.exports` is given on lines ``lineno``
            to ``last``, all a CommonJS module then exports: a name no source
            can write."""
            return self._nameless_owner(
                file_id, f"object@{self.source_line(lineno)}", lineno, last, "object"
            )

        def hold(owner_id, key, name, held, lineno):
            """File the member ``name`` of ``owner_id`` that holds what the
            name ``held`` does, under ``key``."""
            member_id = f"{owner_id}::{key}"
            self.builder.add_node(
                member_id, "Constant", name=name, value=held, lineno=lineno
            )
            self.builder.add_edge(owner_id, member_id, "DEFINES")

        def enter(node_id, lineno, end_lineno, signature="", span=None):
            """Make a definition that spans further lines the context, and
            record its ``span``: by default from the line after ``lineno``
            to the `}` that closes its body on line ``end_lineno``, or that
            whole line where its body is an expression."""
            params = _ts_params(signature)
            if end_lineno > lineno:
                self.enter_context(node_id, lineno, end_lineno, params)
                # An arrow function's signature starts with its parameters.
                arrow = re.match(r"\s*(?:\(|[\w$]+\s*=>)", signature)
                if arrow and signature.rpartition("=>")[2].strip() != "{":
                    expression_ids.add(node_id)
                closed = None
                if node_id not in expression_ids:
                    closed = body_end(lineno, end_lineno)
                extents.append(
                    (node_id, *(span or ((lineno - 1, None), (end_lineno - 1, closed))))
                )

        def body_end(lineno, end_lineno):
            """The column of the `}` on line ``end_lineno`` that closes the
            body of a definition written on line ``lineno``: the one whose
            `{` opens first there or after. None where there is none."""
            row = end_lineno - 1
            columns = braces_on[row] if row < len(braces_on) else ()
            closing = [
                (open_at[(row, column)], column)
                for column in columns
                if (row, column) in open_at and open_at[(row, column)][0] >= lineno - 1
            ]
            return min(closing)[1] if closing else None

        def owner_at(row, column):
            """The innermost definition whose span holds column ``column``
            of line ``row``, where a span starts or ends at a known column
            of it; else None, and the line is read whole. ``extents`` holds
            those that end on the line or after, outermost first."""
            if not any(
                start[0] == row
                and start[1] is not None
                or end[0] == row
                and end[1] is not None
                for _, start, end in extents
            ):
                return None
            for node_id, (first, opened), (last, closed) in reversed(extents):
                after = first < row or (
                    first == row and opened is not None and opened < column
                )
                before = last > row or closed is None or column < closed
                if after and before:
                    return node_id
            return file_id

        def invoked(row, column):
            """Whether the function written from column ``column`` of line
            ``row``, a 0-based index, is called where it is written:
            `(() => {...})()`, `(function () {...}())`, `!function () {...}()`.
            It is then no function but what it returns."""
            text, at = scopes.text, scopes.starts[row] + column
            opened = _run_start(text, at) - 1
            if text[opened : opened + 1] == "(" and opened in scopes.closer:
                after = _SPACES.match(text, scopes.closer[opened] + 1).end()
                if text[after : after + 1] == "(":
                    return True
            head = re.compile(r"(?:async\s*)?(?:function\b\s*\*?\s*[\w$]*\s*)?").match(
                text, at
            )
            params = head.end()
            params = (
                scopes.closer.get(params, params)
                if text[params : params + 1] == "("
                else params
            )
            body = _JS_BODY_OPENS_RE.match(text, params + 1)
            closed = body and scopes.closer.get(body.end() - 1)
            return bool(closed) and text[closed + 1 :].lstrip()[:1] == "("

        def value_start(row):
            """The column of line ``row`` that the value given on it starts
            at, inside the parentheses it may be written in; None where the
            line gives none."""
            code = code_lines[row]
            given = "=" if _TS_VARIABLE_RE.match(code) or "=" in code else ":"
            found = re.compile(
                rf"{given}(?![=>])\s*(?:\((?=\s*(?:\(|function\b|async\b))\s*)?"
            ).search(code)
            return found and found.end()

        def source_at(row, column, parent_id, spans):
            """The definition whose code column ``column`` of line ``row`` is:
            that of the innermost of ``spans`` that holds it, else the one
            ``owner_at`` gives, else ``parent_id``."""
            source = owner_at(row, column) or parent_id
            for span_start, span_end, node_id in spans:
                if span_start <= column < span_end:
                    source = node_id
            return source

        def value_of(name, lineno, since, owner, column):
            """The origin of a call of ``name`` at column ``column`` of line
            ``lineno`` that calls a value: `?` for a parameter, what one
            declared `typeof x` holds, and `=name` for a variable a function
            declares, where no definition there defines the name and the
            variable's origin is not known."""
            bound = self._read_from(column, self._declaration, name, lineno)
            # A name a `require` binds is the module's.
            if (
                bound is None
                or bound[0] in ("function", "class")
                or bound[1][0] - 1 in required
            ):
                return None
            if bound[0] == "param":
                held = self._declared(name, lineno, since)
                return held if held and not held.endswith("()") else "?"
            row, at = bound[1]
            given = re.match(
                r"[\w$]+\s*(?::[^=;]*)?=(?![=>])", code_lines[row - 1][at:]
            )
            if self._scopes.function(bound[1]) is None or (
                # One that holds a function is a definition of it.
                given and _function_value(code_lines[row - 1][at + given.end() :])
            ):
                return None
            # A definition of the name in a function around the call, or
            # one the binding's own line gives, is it; another of the module
            # the local name hides.
            for context in reaching(lineno):
                node = self._node(f"{context[0]}::{name}")
                if node and (
                    context[0] != file_id or node["metadata"].get("lineno") == row
                ):
                    return None
            held = self.call_origin(name, lineno, since, owner)
            return None if held else f"={name}"

        def reaching(at):
            """The contexts that reach line ``at``, innermost first."""
            return [
                context
                for context in reversed(self._contexts)
                if at <= context[2] + (context[:3] in closed_after)
            ]

        # Each of ``text_lines`` keeps the text of the literals that open on
        # it, and is the line the loop below reads.
        text_lines = _blank_js_lines(lines, jsx=jsx, code=code_lines)
        # The columns of each line's braces
        braces_on, braces = [], []
        # (line, column) of each `}` -> that of the `{` it closes, and back
        open_at, close_at = {}, {}
        for row, text in enumerate(code_lines):
            columns = []
            for at, char in enumerate(text):
                if char == "{":
                    braces.append((row, at))
                elif char == "}" and braces:
                    open_at[(row, at)] = braces[-1]
                    close_at[braces.pop()] = row, at
                else:
                    continue
                columns.append(at)
            braces_on.append(columns)

        # The last line holding code above each line, -1 for none
        code_above, last = [], -1
        for row, code in enumerate(code_lines):
            code_above.append(last)
            last = row if code.strip() else last

        i = 0
        signature_until = 0  # the last line of a definition's signature
        later_declarators = {}  # line index -> declarators a statement gives it
        constants = {}  # (line, column) of each constant's name -> its ID
        # (ID, scope) of each local of a name -> the node it is, and ID ->
        # how many scopes have one of it
        scoped, scopes_of = {}, {}

        def local_id(node_id, scope):
            """The ID of the declaration ``node_id`` whose binding holds over
            ``scope``: ``node_id``, or where one of the name in another scope
            of the same definition has it, `name#2` and on."""
            taken = self._node(node_id)
            held = taken and taken["metadata"].get("scope")
            if not (scope and held):
                return node_id
            scope, held = (tuple(map(tuple, s)) for s in (scope, held))
            scoped.setdefault((node_id, held), node_id)
            if (node_id, scope) not in scoped:
                scopes_of[node_id] = scopes_of.get(node_id, 1) + 1
                scoped[node_id, scope] = f"{node_id}#{scopes_of[node_id]}"
            return scoped[node_id, scope]

        while i < len(lines):
            line = text_lines[i]
            lineno = i + 1
            # Column of a function/method name declared on this line, which
            # the regex call fallback must not read as a call.
            declared_at = None
            valued = None  # the signature of the arrow function the line is
            arrow_id = None  # the node of the arrow function the line declares
            own_defs = []  # (node ID, end of its name) declared on the line
            holder = None  # the constant the line declares, and where its value starts
            field_spans = []  # (start, end, class ID) of a field's value on it
            nodes_before = len(self.builder.nodes)

            extents[:] = [extent for extent in extents if extent[2][0] >= i]
            parent_id = self.context_at(lineno)[0]
            # The line that the `}` closing a definition leads is read
            # outside it, as the line that closes a callback is. An
            # expression body ends with its statement, not at a `}`.
            if lineno > signature_until and _closes_first(code_lines[i]):
                unclosed = callback_ids | expression_ids
                while len(self._contexts) > 1 and (
                    self._contexts[-1][2] < lineno
                    or self._contexts[-1][2] == lineno
                    and self._contexts[-1][0] not in unclosed
                ):
                    self._contexts.pop()
                parent_id = self._contexts[-1][0]
            # The object the definition returns, where the line being read is
            # one of its lines: a function written there is a member of that
            # object, which no source can name, and reads the names the
            # definition that returns it does.
            object_id, object_scope = self._returned_object_at(
                returned_objects, parent_id, lineno
            )

            def claim(node_id):
                """Take ``node_id`` for a function or variable; an interface
                declared under it moves to `<scope>::interface_<name>`."""
                value_ids.add(node_id)
                if node_id in interface_ids:
                    interface_ids.discard(node_id)
                    scope, _, own = node_id.rpartition("::")
                    self.rename_node(node_id, f"{scope}::interface_{own}")

            def read_arrow(idx):
                """Read `const name = (params) => ...` from line ``idx`` to
                the line that closes its parameter list. Returns that line's
                index and what ``_arrow_function`` makes of the statement."""
                text = ""
                for j in range(idx, min(idx + 40, len(lines))):
                    text = f"{text} {text_lines[j].strip()}".lstrip()
                    arrow = _arrow_function(text)
                    code = _NON_CODE_RE.sub("", text).rstrip()
                    # The value may start on the line after the `=`, and an
                    # annotation's type arguments go on over lines.
                    angles = code.count("<") - code.count(">") + code.count("=>")
                    open_ended = (
                        code.endswith("=")
                        or code.count("(") > code.count(")")
                        or angles > 0
                    )
                    if arrow or not open_ended:
                        start = arrow and value_start(idx)
                        return j, None if start and invoked(idx, start) else arrow, text
                return idx, None, ""

            def consume_sig(idx, start=0):
                """Read the declaration that starts at column ``start`` of
                line ``idx`` up to the `{` that opens its body, or to the end
                of one that has none. Returns the index of that line, the
                declaration's text and the line its body ends on: its first
                line when it has none, and where it ends with an arrow's
                `=>`, the last line of the expression after it."""
                j, sig, end = read_sig(idx, start)
                if sig.rstrip().endswith("=>"):
                    end = _statement_end(code_lines, idx) + 1
                return j, sig, end

            def read_sig(idx, start):
                """What ``consume_sig`` returns, but for an arrow's
                expression body."""
                sig = ""
                stack = []  # brackets left open
                closed = False  # a parameter list has been read
                for j in range(idx, min(idx + 40, len(lines))):
                    skipped = start if j == idx else 0
                    text = " " * skipped + text_lines[j][skipped:]
                    code = " " * skipped + code_lines[j][skipped:]
                    for column, char in enumerate(code):
                        if (
                            char == "{"
                            and not stack
                            and not _TS_TYPE_EXPECTED_RE.search(sig + text[:column])
                        ):
                            end = self._find_block_end(code_lines, j, column)
                            return j, sig + text[: column + 1].lstrip(), end
                        # `a < b` compares; `Type<` opens type arguments.
                        if char in "([{" or (
                            char == "<"
                            and code[column + 1 : column + 2] not in (" ", "=")
                        ):
                            stack.append(char)
                        elif char == ">":
                            if stack[-1:] == ["<"] and code[column - 1 : column] != "=":
                                stack.pop()
                        elif char in ")]}":
                            while stack[-1:] == ["<"]:
                                stack.pop()
                            if not stack:
                                return j, (sig + text[:column]).strip(), j + 1
                            stack.pop()
                            closed = closed or (char == ")" and not stack)
                        elif char == ";" and not stack:
                            return j, sig + text[: column + 1].lstrip(), j + 1
                    sig += f"{text.strip()} "
                    following = code_lines[j + 1].lstrip() if j + 1 < len(lines) else ""
                    if (
                        closed
                        and not stack
                        and not _TS_TYPE_EXPECTED_RE.search(sig)
                        and not following.startswith(_TS_TYPE_CONTINUES)
                        and not _JS_ASSIGNMENT_END_RE.search(sig)
                    ):
                        return j, sig.strip(), j + 1
                return idx, code_lines[idx].strip(), idx + 1

            def members_on(class_id, idx, code, start, lineno):
                """Read the members written on ``code``, line ``idx``, from
                column ``start``, filed under ``class_id``, and make each a
                node. Returns the member the line leaves open, the spans of
                the values of the fields on it, the last line of a field
                the line leaves open, 0 where there is none, and each member
                as its ID and the column its name ends at."""
                raw = text_lines[idx]
                written, fields, open_at = _members_on_line(code, raw, start)
                spans = [(*field, class_id) for field in fields]
                written = [(*member, lineno) for member in written]
                sig_until = 0
                if open_at is not None:
                    member_end = _statement_end(code_lines, idx, open_at) + 1
                    found = _TS_LINE_MEMBER_RE.match(code, open_at)
                    prop = _class_property(code, open_at, raw)
                    if found:
                        filed = _member_filed(found, raw)
                        written.append(
                            (filed, found.end(2), code[open_at:], member_end)
                        )
                    elif prop and _function_value(raw[prop[2] :]):
                        written.append((prop[0], prop[1], raw[prop[2] :], member_end))
                    else:
                        sig_until = member_end
                        if prop:
                            spans.append((prop[2], len(code), class_id))
                open_member = None
                made = []
                for member, name_end, text, member_end in written:
                    mid = member_id(class_id, member, lineno)
                    return_type, param_types, typed = self._extract_ts_types(text)
                    self.builder.add_node(
                        mid,
                        "Method",
                        name=member,
                        signature=self._build_ts_signature(
                            member, param_types, return_type
                        ),
                        lineno=lineno,
                        end_lineno=member_end,
                        token_count=token_count(lineno, member_end),
                        docstring="",
                        return_type=return_type,
                        param_types=param_types,
                        signatures=self._overloads(mid, typed, member_end),
                    )
                    self.builder.add_edge(class_id, mid, "DEFINES")
                    made.append((mid, name_end))
                    if member_end > lineno:
                        open_member = (mid, lineno, member_end, text)
                return open_member, spans, sig_until, made

            def field_property(idx):
                """The class property written on line ``idx`` and the text of
                its value over the lines the statement spans, with the index
                of its last line; None where the line writes no property."""
                prop = _class_property(code_lines[idx])
                if prop is None:
                    return None
                end = _statement_end(code_lines, idx)
                value = " ".join(
                    [text_lines[idx][prop[2] :]]
                    + [t.strip() for t in text_lines[idx + 1 : min(end, idx + 40) + 1]]
                )
                return prop, value, end

            # A line of the signature of a definition above, whose calls are
            # its: it declares nothing of its own, opens no nameless function
            # and holds none of the definition's `return`s.
            continuing = lineno <= signature_until
            # A `for` or `catch` head the line above leaves open holds the
            # line's binding, and the line declares nothing else.
            above = code_above[i]
            for_head = above >= 0 and _JS_FOR_HEAD_RE.search(code_lines[above])
            held_open = for_head and not _js_member_name(
                code_lines, above, for_head.start()
            )
            # The class the line declares. An expression whose body closes
            # on the line is a const's value, read where the const is; one
            # that goes on over lines is the class's line, not the const's,
            # unless it is written inside brackets, where the line declares
            # what it does without it.
            cx = None if continuing else re_class.search(code_lines[i])
            # A member assigned to the prototype of a function or a class is
            # its instances', one assigned to a function, `F.name = value`,
            # the function's own, written `F.name` as source writes it: a
            # function, or what a name holds. A member assigned again is
            # `name#2` and on. A prototype assigned an object other than a
            # literal has the base the object is read through.
            # `exports = module.exports = v` assigns `module.exports`, and
            # `exports` with it.
            aliased = _CJS_ALIASED_RE.match(code_lines[i])
            view = code_lines[i][aliased.end() :] if aliased else code_lines[i]
            assignment = not continuing and _JS_ASSIGNED_MEMBER_RE.match(view)
            given = assignment and view[assignment.end() :].partition("=")[2]
            if given and (again := _CJS_EXPORTS_RE.match(given)):
                # `module.exports = exports = v`
                aliased, given = again, given[again.end() :]
            name_given = given and re.fullmatch(
                r"\s*(?!(?:true|false|null|undefined|this)\b)([A-Za-z_$][\w$]*)\s*;?\s*",
                given,
            )
            filed_as = None
            written = assignment and assignment[2]
            # A CommonJS module's exports, unless a binding of the name, a
            # `const` or a parameter's too, shadows it where the line is
            exported = written in ("exports", "module.exports", "module") and (
                self._declaration(written.partition(".")[0], lineno) is None
            )
            if exported and (written != "module" or assignment[1] == "exports"):
                self.builder.commonjs.add(file_id)
            holds_function = given and _function_value(given)
            immediate = holds_function and invoked(i, value_start(i) or 0)
            # `= require("m")`: what the module exports, read from the text
            loaded = given and _JS_REQUIRED_VALUE_RE.fullmatch(
                text_lines[i][len(code_lines[i]) - len(given) :]
            )
            # A CommonJS write takes its place among the others in a
            # statement the file's top level runs each time; one that may
            # not run, or runs at a time not known, leaves what it writes
            # unknown. One written past the line's start is such a write.
            indent = len(code_lines[i]) - len(code_lines[i].lstrip())
            definite = scopes.always_runs(scopes.starts[i] + indent)
            rebound = (
                not continuing
                and not assignment
                and _CJS_EXPORTS_RE.match(code_lines[i])
            )
            if rebound and self._declaration("exports", lineno) is None:
                # `exports = v` rebinds the alias: what it writes from there
                # on is no export.
                cjs_orphan = True
                self.builder.commonjs.add(file_id)
            # `module.exports = exports` gives the exports what the alias
            # holds: what they held, unless the alias was rebound.
            rejoined = (
                name_given
                and name_given[1] == "exports"
                and self._declaration("exports", lineno) is None
            )
            if (
                exported
                and written == "module"
                and assignment[1] == "exports"
                and not (rejoined and not cjs_orphan)
            ):
                # `module.exports =` gives the module's exports whole: it
                # replaces them — entries written before it, `=` and named,
                # bound the object it orphans — and where the new value is
                # not read they are unknown.
                cjs.clear()
                if definite:
                    cjs_orphan = not (aliased or rejoined)
                # `module.exports = function ...`: the module is the
                # function, filed as source writes it; `= <name>` is it,
                # read in the scope the line is written in.
                whole = (
                    "module.exports"
                    if holds_function and not immediate
                    else name_given and entry_name(name_given[1], lineno)
                )
                # `module.exports = { a, b: c }` on its line is the object
                # the module exports, which holds what the names do.
                listed = re.fullmatch(r"\s*\{(.*)\}\s*;?\s*", given or "")
                oid = listed and exported_object(lineno, lineno)
                # A value not read, or written where it may not run, leaves
                # the exports unknown.
                value = (loaded and "=") or (oid and oid.rpartition("::")[2]) or whole
                cjs["="] = (
                    ("=", loaded[1] if loaded else "", value or None, parent_id)
                    if definite
                    else ("=", "", None, parent_id)
                )
                if holds_function and not immediate:
                    object_id, object_scope, filed_as = (
                        file_id,
                        None,
                        "module.exports",
                    )
                for part in listed[1].split(",") if oid else ():
                    if held := re_held.match(part):
                        spread, key, name = held.groups()
                        self.builder.reexports.setdefault(oid, []).append(
                            (
                                "*" if spread else key or name,
                                "",
                                entry_name(name, lineno),
                                parent_id,
                            )
                        )
                        continue
                    method = re.match(
                        r"\s*(?:(?P<key>[\w$]+)\s*:\s*(?:function\b"
                        r"|(?:\([^)]*\)|[\w$]+)\s*=>)|(?P<short>[\w$]+)\s*\()",
                        part,
                    )
                    if method:
                        name = method["key"] or method["short"]
                        mid = f"{oid}::{free_key(oid, name)}"
                        self.builder.add_node(mid, "Function", name=name, lineno=lineno)
                        self.builder.add_edge(oid, mid, "DEFINES")
                        self.builder.reexports.setdefault(oid, []).append(
                            (name, "", name, parent_id)
                        )
            elif exported and written != "module":
                # A member of the module's exports, `exports.name = value`
                # or `module.exports.name = value`, its value read in the
                # scope the line is written in, unknown for one not read.
                # `exports` writes the orphan a `module.exports =` left,
                # `module.exports` the new exports object. A write that may
                # not run gives the member a value only where no other does.
                key = (
                    holds_function
                    and not immediate
                    and free_key(file_id, assignment[1])
                )
                value = (loaded and "=") or (
                    entry_name(name_given[1], lineno) if name_given else key
                )
                entry = (
                    assignment[1],
                    loaded[1] if loaded else "",
                    value or None,
                    parent_id,
                )
                if written != "exports" or not cjs_orphan:
                    held = cjs.get(assignment[1], entry)
                    cjs[assignment[1]] = (
                        entry
                        if definite or held == entry
                        else (assignment[1], "", None, parent_id)
                    )
                if key:
                    object_id, object_scope, filed_as = file_id, None, key
            elif immediate:
                pass
            elif name_given or holds_function:
                owner = (
                    prototype(written, lineno)
                    if assignment[3]
                    else (declared(written, lineno) or (None,))[0]
                )
                key = owner and free_key(owner, assignment[1])
                if not assignment[3] and owner:
                    static_ids.add(f"{owner}::{key}")
                if owner and name_given:
                    hold(owner, key, assignment[1], name_given[1], lineno)
                elif owner:
                    object_id, object_scope, filed_as = owner, None, key
            for found in _CJS_WRITE_RE.finditer(
                code_lines[i],
                rebound.end()
                if rebound
                else len(code_lines[i]) - len(given)
                if exported
                else 0,
            ):
                via = found["via"] or found["target"]
                member, unnamed = found["member"], found["unnamed"] or found["key"]
                if self._declaration("module" if via else "exports", lineno):
                    continue
                self.builder.commonjs.add(file_id)
                if unnamed:
                    # Any member may be the one written.
                    for name in cjs.keys() - {"="} if via or not cjs_orphan else ():
                        cjs[name] = (name, "", None, parent_id)
                elif member and (via or not cjs_orphan):
                    cjs[member] = (member, "", None, parent_id)
                elif via and not member:
                    cjs.clear()
                    cjs["="] = ("=", "", None, parent_id)
                elif not member:
                    cjs_orphan = True
            replaced = not continuing and _JS_PROTOTYPE_RE.match(code_lines[i])
            if replaced and (made := prototype(replaced[1], lineno, member=False)):
                base = _prototype_base(replaced[2])
                if base:
                    self.builder.add_edge(made, f"REF:{base}", "INHERITS")
            if continuing or held_open or (cx and cx.group("enclosed") is None):
                pass
            elif (match := re_const.search(line)) and (
                match.group(3) is not None or not read_arrow(i)[1]
            ):
                name, value_type, val = match.groups()
                if parent_id == file_id and line[:1].isspace():
                    # In a callback, where another may declare the name too.
                    value_type = None
                doc = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                claim(cid)
                scope = self._binding_scope(name, lineno, match.start(1))
                cid = local_id(cid, scope)
                holder = cid, match.end(1)
                if value_type and cid in namespaces:
                    # A namespace and a value of one name are one: the
                    # value's type gives the members the namespace has not.
                    namespaces[cid]["value_type"] = value_type
                if (
                    val
                    and (held := _arrow_function(line))
                    and not invoked(i, value_start(i) or 0)
                ):
                    valued = self._arrow_typed(held[1], line)
                    arrow_id = cid
                    own_defs.append((cid, match.end(1)))
                    enter(cid, lineno, _statement_end(code_lines, i) + 1, held[1])
                written = (
                    val
                    and line[match.start(3) : _value_end(code_lines[i], match.start(3))]
                )
                if written and re.fullmatch(r"[\w$]+", written.strip()):
                    # The constant the name is there, where its binding is
                    # one: a nearer one without a node, a parameter's, is not
                    bound = self._read_from(
                        match.start(3), self._declaration, written.strip(), lineno
                    )
                    if bound and bound[1] in constants:
                        self.builder.js_aliases[cid] = constants[bound[1]]
                constants[lineno, match.start(1)] = cid
                self.builder.add_node(
                    cid,
                    "Constant",
                    name=name,
                    value=written,
                    value_type=value_type,
                    lineno=lineno,
                    docstring=doc,
                    signatures=valued and self._overloads(cid, valued, lineno),
                    **({"scope": scope} if scope else {}),
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                below = next(
                    (row for row in lines_after(code_lines, i) if row.strip()), ""
                )
                opens_below = re.search(r"=\s*$", line) and below.lstrip()[:1] == "{"
                if parent_id not in class_ids and (
                    re.search(r"=\s*\{\s*$", line) or opens_below
                ):
                    # The functions it holds are its members, its `{` on its
                    # line or starting the next line of code.
                    object_ids.add(cid)
                    enter(cid, lineno, self._find_block_end(code_lines, i))
                elif parent_id not in class_ids and re.search(r"=\s*\[\s*$", line):
                    elements_of = cid, _statement_end(code_lines, i) + 1
                elif parent_id not in class_ids and re.search(r"=\s*\[\s*\{\s*$", line):
                    # Its first element, an object, owns the functions it holds.
                    last = self._find_block_end(code_lines, i)
                    element = self._nameless_owner(
                        cid,
                        f"object@{self.source_line(lineno)}",
                        lineno,
                        last,
                        "object",
                    )
                    object_ids.add(element)
                    enter(element, lineno, last)
            elif parent_id not in class_ids and (
                opened := _TS_OBJECT_OPENS_RE.match(code_lines[i])
            ):
                # An object `export default` gives, a property's, a
                # prototype or an element's: its functions are its members.
                # A property's is named as source reaches it, through its
                # key; the others no name source writes.
                last = self._find_block_end(code_lines, i)
                key = opened["key"] and parent_id in object_ids and opened["key"]
                if key:
                    oid = f"{parent_id}::{_ts_unquote(key)}"
                    self.builder.add_node(
                        oid, "Constant", name=_ts_unquote(key), lineno=lineno
                    )
                    self.builder.add_edge(parent_id, oid, "DEFINES")
                elif opened["prototype"]:
                    oid = prototype(opened["prototype"], lineno, member=False)
                elif opened["commonjs"] and exported:
                    oid = exported_object(lineno, last)
                    if definite:
                        cjs["="] = ("=", "", oid.rpartition("::")[2], parent_id)
                elif opened["key"] is None and (opened["default"] or array):
                    held = elements_of and lineno <= elements_of[1] and elements_of[0]
                    oid = self._nameless_owner(
                        held or parent_id,
                        f"object@{self.source_line(lineno)}",
                        lineno,
                        last,
                        "object",
                    )
                    if opened["default"]:
                        es.append(("default", "", oid.rpartition("::")[2], parent_id))
                else:
                    oid = None
                if oid:
                    object_ids.add(oid)
                    enter(oid, lineno, last)
            elif match := re_enum.search(line):
                name = match.group(1)
                doc = self.get_leading_docstring(lines, i)
                cid = f"{parent_id}::{name}"
                self.builder.add_node(
                    cid, "Class", name=name, lineno=lineno, docstring=doc
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                class_ids.add(cid)
                if "{" in line:
                    enter(cid, lineno, self._find_block_end(code_lines, i))
            elif match := _TS_NAMESPACE_RE.match(code_lines[i]):
                # A namespace owns what it declares, which source names
                # through it; `A.B` is `B` in `A`.
                last = self._find_block_end(code_lines, i)
                owner = parent_id
                for part in match[1].split("."):
                    nested = f"{owner}::{part}"
                    if nested not in self.builder.node_registry:
                        self.builder.add_node(
                            nested,
                            "Class",
                            name=part,
                            kind="namespace",
                            lineno=lineno,
                            end_lineno=last,
                            token_count=token_count(lineno, last),
                            docstring=self.get_leading_docstring(lines, i),
                        )
                        self.builder.add_edge(owner, nested, "DEFINES")
                        namespaces[nested] = self.builder.nodes[-1]["metadata"]
                    owner = nested
                enter(owner, lineno, last)
            elif (match := re_interface.search(line)) and (
                # An object type's `{` may open the next line of code.
                match[3] != ""
                or next(
                    (r for r in lines_after(code_lines, i) if r.strip()), ""
                ).lstrip()[:1]
                == "{"
            ):
                name = match.group(1) or match.group(2)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig, end_lineno = consume_sig(i)
                cid = f"{parent_id}::{name}"
                if cid in value_ids:
                    # The function or variable of that name has this ID.
                    cid = f"{parent_id}::interface_{name}"
                else:
                    interface_ids.add(cid)
                tokens = token_count(lineno, end_lineno)
                self.builder.add_node(
                    cid,
                    "Class",
                    name=name,
                    kind="interface",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=tokens,
                    docstring=docstring,
                    **_generic_metadata(
                        _bounded_parameters(line[match.end(1 if match[1] else 2) :])
                    ),
                )
                self.builder.add_edge(parent_id, cid, "DEFINES")
                class_ids.add(cid)
                type_bodies.add(cid)
                heritage = _strip_generics(full_sig)
                if "extends" in heritage:
                    bases = heritage.split("extends")[1].split("{")[0].split(",")
                    for b in bases:
                        self.builder.add_edge(cid, f"REF:{b.strip()}", "INHERITS")
                enter(cid, lineno, end_lineno)
                i = j
            elif match := re_func.search(line):
                name = match.group(1)
                declared_at = match.start(1)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig, end_lineno = consume_sig(i)
                held = self._binding_scope(name, lineno, declared_at)
                fid = local_id(f"{parent_id}::{name}", held)
                return_type, param_types, typed = self._extract_ts_types(full_sig)
                signature = self._build_ts_signature(name, param_types, return_type)
                tokens = token_count(lineno, end_lineno)
                claim(fid)
                self.builder.add_node(
                    fid,
                    "Function",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=tokens,
                    docstring=docstring,
                    return_type=return_type,
                    param_types=param_types,
                    signatures=self._overloads(fid, typed, end_lineno),
                    **({"scope": held} if held else {}),
                )
                self.builder.add_edge(parent_id, fid, "DEFINES")
                own_defs.append((fid, match.end(1)))
                enter(fid, lineno, end_lineno, full_sig)
                signature_until = j + 1  # its last line, as a line number
            elif (
                _TS_VARIABLE_RE.match(line) or parent_id in object_ids or object_id
            ) and (arrow := (statement := read_arrow(i))[1]):
                name, parameters = arrow
                j = statement[0]
                docstring = self.get_leading_docstring(lines, i)
                named = _value_named(line)
                held = self._binding_scope(
                    name,
                    lineno,
                    named.start(1)
                    if named and _ts_unquote(named.group(1)) == name
                    else line.find(name),
                )
                fid = local_id(f"{object_id or parent_id}::{filed_as or name}", held)
                # A block is left open where the statement stops; an
                # expression body ends with the statement.
                last = code_lines[j]
                end_lineno = (
                    self._find_block_end(code_lines, j)
                    if last.count("{") > last.count("}")
                    else _statement_end(code_lines, i) + 1
                )
                # Arrow functions don't have easily extractable signatures
                signature = f"const {name} = (...) => ..."
                tokens = token_count(lineno, end_lineno)
                claim(fid)
                typed = self._arrow_typed(parameters, line)
                if end_lineno == lineno or not _JS_BLOCK_FUNCTION_RE.search(parameters):
                    # A body that is an expression gives what it reads, on
                    # this line or one below; a block's `return`s are read
                    # where they are.
                    valued = typed
                self.builder.add_node(
                    fid,
                    # A prototype's function is its instances' method.
                    "Method"
                    if fid.rpartition("::")[0] in prototype_ids
                    else "Function",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=tokens,
                    docstring=docstring,
                    signatures=self._overloads(fid, typed, end_lineno),
                    **({"scope": held} if held else {}),
                    **({} if parameters.startswith("function") else {"arrow": True}),
                )
                self.builder.add_edge(object_id or parent_id, fid, "DEFINES")
                if named and _ts_unquote(named.group(1)) == name:
                    own_defs.append((fid, named.end(1)))
                enter(
                    fid,
                    object_scope if object_scope is not None else lineno,
                    end_lineno,
                    parameters,
                )
                head = statement[2][: len(statement[2]) - len(parameters)]
                annotated = re.search(r":\s*(.*?)\s*=\s*$", head)
                if j > i and annotated:
                    # Over lines, the arrow's parameters, the first list
                    # that closes on line j, read the function type it is
                    # declared as.
                    params = _function_parameters(parameters)
                    bindings = self._param_bindings(
                        f">{_path_written(annotated[1])}.^", params
                    )
                    opened = min(
                        (
                            r[0]
                            for r in scopes.params.get(j + 1, ())
                            if r[1][0] == j + 1
                        ),
                        default=(i + 1, 0),
                    )
                    self._bind(bindings, opened[0] - 1, opened[1])
                arrow_id = fid
                signature_until = j + 1  # its last line, as a line number
            elif parent_id in object_ids and (match := re_held.match(line)):
                # The object holds by its key what the name is, and spread
                # what that holds, read where the object is written.
                spread, key, name = match.groups()
                held = (
                    "*" if spread else key or name,
                    "",
                    entry_name(name, lineno),
                    self.enclosing_scope(parent_id),
                )
                if parent_id in prototype_ids and spread:
                    # A prototype spread an object has what that has.
                    self.builder.add_edge(parent_id, f"REF:{name}", "INHERITS")
                elif parent_id in prototype_ids:
                    member = key or name
                    hold(parent_id, free_key(parent_id, member), member, name, lineno)
                else:
                    self.builder.reexports.setdefault(parent_id, []).append(held)
            elif (
                parent_id in type_bodies
                and (written := _ts_type_member(code_lines, i))
                and _braces_open(
                    code_lines,
                    next(c[1] for c in reversed(self._contexts) if c[0] == parent_id)
                    - 1,
                    0,
                    i,
                    -1,
                )
                == 1
            ):
                # A property of a function type is a method the type
                # declares; one of an object type a field of that type.
                name, full_sig = written
                if full_sig.startswith("{"):
                    self.builder.fields.setdefault(parent_id, {})[name] = full_sig
                else:
                    mid = f"{parent_id}::{name}"
                    return_type, param_types, typed = self._extract_ts_types(full_sig)
                    self.builder.add_node(
                        mid,
                        "Method",
                        name=name,
                        signature=self._build_ts_signature(
                            name, param_types, return_type
                        ),
                        lineno=lineno,
                        end_lineno=lineno,
                        token_count=10,
                        docstring=self.get_leading_docstring(lines, i),
                        return_type=return_type,
                        param_types=param_types,
                        signatures=self._overloads(mid, typed, lineno),
                    )
                    self.builder.add_edge(parent_id, mid, "DEFINES")
            elif (
                parent_id in class_ids
                and not re_method.search(line)
                and (fv := field_property(i)) is not None
                and not _function_value(fv[1])
            ):
                # A field's value: the lines it goes on over declare nothing.
                # The field is assigned to each instance, as `this.name =`.
                signature_until = fv[2] + 1
                held = self.builder.assigned.setdefault(parent_id, {})
                held.setdefault(fv[0][0], set()).add(fv[1].strip().removesuffix(";"))
            elif (
                "::" in parent_id
                and parent_id != file_id
                and (
                    enclosing := (
                        parent_id
                        if parent_id in class_ids
                        else self.enclosing_scope(parent_id)
                    )
                )
                in class_ids
                and (close_col := _member_close_column(code_lines[i])) is not None
                # The `}` closes a member where it leaves the class's body
                # alone open, a block of a member's body where it does not.
                and _braces_open(
                    code_lines,
                    next(c[1] for c in reversed(self._contexts) if c[0] == enclosing)
                    - 1,
                    class_at.get(enclosing, 0),
                    i,
                    close_col,
                )
                == 1
            ):
                # A member written on the line that closes the member before
                # it is the class's, read as on its own line.
                open_member, field_spans, sig_until, made = members_on(
                    enclosing, i, code_lines[i], close_col, lineno
                )
                own_defs.extend(made)
                if sig_until:
                    signature_until = sig_until
                if open_member:
                    enter(*open_member)
            # A callback's body, as the file's, holds no members of its own: a
            # method written there is one of an object literal. Inside an
            # object that owns what it holds, a member is one of its own.
            elif (
                object_id
                or (
                    "::" in parent_id
                    and parent_id != file_id
                    and parent_id not in callback_ids
                )
            ) and (
                member := _member_written(
                    line,
                    parent_id in class_ids,
                    re_method,
                    (fv := field_property(i)) and fv[1],
                )
            ):
                name, name_start, name_end, assigned = member
                j, full_sig, end_lineno = consume_sig(i)
                # `name(` declares a method directly in a class body, or where
                # a block follows its parameter list; otherwise it is a call.
                # Outside a class a keyword names no member: the block after
                # `catch (` is that of a `try` statement, and `function (`
                # starts a function expression.
                arrow = parent_id in class_ids and re.search(
                    r"\)\s*(?::[^=;]+)?=>", full_sig
                )
                if name not in ["if", "for", "while", "switch"] and (
                    arrow
                    if assigned
                    else parent_id in class_ids
                    or (name not in _JSTS_CONTROL_KEYWORDS and _opens_body(full_sig))
                ):
                    declared_at = name_start
                    docstring = self.get_leading_docstring(lines, i)
                    mid = member_id(object_id or parent_id, name, lineno)
                    return_type, param_types, typed = self._extract_ts_types(full_sig)
                    signature = self._build_ts_signature(name, param_types, return_type)
                    tokens = token_count(lineno, end_lineno)
                    self.builder.add_node(
                        mid,
                        "Method",
                        name=name,
                        signature=signature,
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=tokens,
                        docstring=docstring,
                        return_type=return_type,
                        param_types=param_types,
                        signatures=self._overloads(mid, typed, end_lineno),
                    )
                    self.builder.add_edge(object_id or parent_id, mid, "DEFINES")
                    own_defs.append((mid, name_end))
                    if name == "constructor" and parent_id in class_ids:
                        self._parameter_properties(parent_id, typed["params"])
                    value = assigned and _class_property(full_sig)
                    enter(
                        mid,
                        object_scope if object_scope is not None else lineno,
                        end_lineno,
                        full_sig[value[2] :] if value else full_sig,
                    )
                    signature_until = j + 1  # its last line, as a line number
                elif assigned and parent_id in class_ids:
                    # A field whose parenthesised value is no function: its
                    # lines declare nothing, its calls the class's.
                    signature_until = _statement_end(code_lines, i) + 1
            # An accessor, `get name()` or `set name(value)`, of a Class, or
            # of an object that owns what it holds — one returned, passed to
            # a call, or a constant's
            elif (
                object_id
                or parent_id in object_ids
                or (
                    "::" in parent_id
                    and parent_id != file_id
                    and parent_id not in callback_ids
                    and (node := self._node(parent_id)) is not None
                    and node["type"] == "Class"
                )
            ) and (match := re_getter_setter.search(line)):
                accessor_type = match.group(1)
                name = _ts_unquote(match.group(2))
                declared_at = match.start(2)
                docstring = self.get_leading_docstring(lines, i)
                j, full_sig, end_lineno = consume_sig(i)
                # An accessor is filed as source writes it, `get name`, which
                # tells a pair apart and is no name a method can have.
                mid = member_id(
                    object_id or parent_id, f"{accessor_type} {name}", lineno
                )
                return_type, param_types, typed = self._extract_ts_types(full_sig)
                if accessor_type == "get":
                    signature = f"get {name}(): {return_type}"
                else:
                    param_str = ", ".join(param_types) if param_types else "value"
                    signature = f"set {name}({param_str})"
                tokens = token_count(lineno, end_lineno)
                self.builder.add_node(
                    mid,
                    "Method",
                    name=name,
                    signature=signature,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=tokens,
                    docstring=docstring,
                    return_type=return_type,
                    param_types=param_types,
                    accessor=accessor_type,
                    signatures=self._overloads(mid, typed, end_lineno),
                )
                self.builder.add_edge(object_id or parent_id, mid, "DEFINES")
                own_defs.append((mid, match.end(2)))
                enter(
                    mid,
                    object_scope if object_scope is not None else lineno,
                    end_lineno,
                    full_sig,
                )
                signature_until = j + 1  # its last line, as a line number
            elif match := re_import.search(line):
                self.builder.add_edge(file_id, f"MODULE:{match.group(2)}", "IMPORTS")

            if cx:
                # The declarator names the class it is given. An
                # expression's own name is visible only inside it: one given
                # to anything else has no name that source can write.
                name = (
                    cx.group("declarator")
                    or cx.group("plain")
                    or f"class@{self.source_line(lineno)}"
                )
                docstring = self.get_leading_docstring(lines, i)
                # The statement may start before the class and go on past it:
                # its heritage and its body are read from its keyword.
                j, full_sig, end_lineno = consume_sig(i, cx.end())
                body = _class_body_open(code_lines[i], cx.end())
                # What its lines are all in holds the class: a definition
                # the line declares, not one it closes.
                holders = reaching(end_lineno)
                held_by = holders[0][0]
                for node_id, name_end in own_defs:
                    span = _own_line_span(code_lines[i], name_end)
                    if span and span[0] <= cx.start() < span[1]:
                        held_by = node_id
                # A declared class's binding is scoped as a `let`'s.
                written = cx.group("plain") and cx.start("plain")
                scope = written and self._binding_scope(name, lineno, written)
                cid = local_id(f"{held_by}::{name}", scope)
                tokens = token_count(lineno, end_lineno)
                self.builder.add_node(
                    cid,
                    "Class",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=tokens,
                    docstring=docstring,
                    **_generic_metadata(_bounded_parameters(full_sig)),
                    **({"scope": scope} if scope else {}),
                )
                self.builder.add_edge(held_by, cid, "DEFINES")
                class_ids.add(cid)
                class_at[cid] = cx.end()
                # The methods and accessors written on the class's own line,
                # and the member or field the line leaves open, which ends
                # with its statement.
                # A field's value is evaluated with the class's instance or,
                # static, with the class: the class's.
                open_member = None
                # Its lines declare its members, whatever statement of the
                # line goes on over them.
                signature_until = min(signature_until, lineno)
                if body is not None:
                    open_member, field_spans, sig_until, made = members_on(
                        cid, i, code_lines[i], body, lineno
                    )
                    own_defs.extend(made)
                    if sig_until:
                        signature_until = sig_until
                # Generic arguments go first: `class A<T extends B> extends C<T>`
                # inherits from C.
                # The base is an expression: a name a type has, or a value
                # no name stands for, a call's or a cast's, as written.
                heritage = _strip_generics(full_sig)
                if extended := _TS_EXTENDS_RE.search(heritage):
                    base = " ".join(extended[1].split())
                    # A parameter of a definition around the class holds a
                    # value: its name is no type's.
                    passed = any(base.split(".")[0] in params for *_, params in holders)
                    self.builder.add_edge(
                        cid,
                        f"REF:{base}",
                        "INHERITS",
                        {"origin": f"={base}"} if passed else None,
                    )
                enter(cid, lineno, end_lineno)
                if open_member:
                    enter(*open_member)
                # A body that opens on this line has had what follows it
                # read; one on a later line is read from that line.
                if body is None:
                    i = j

            # The registration's name, signature, docstring, what opens its
            # callback, the lines that may open it and its least token count
            config = None
            if match := re_interceptor.search(line):
                client, kind = match.groups()
                config = (
                    f"{client}_{kind}_interceptor",
                    f"{client}.interceptors.{kind}.use(...)",
                    self.get_leading_docstring(lines, i)
                    or f"Configures {kind} interceptor for {client}",
                    ("{", "=>"),
                    5,
                    50,
                    {"config_type": "interceptor", "target": client},
                )
            elif (match := re_middleware.search(line)) and (
                middleware := match[2].strip()
            ):
                config = (
                    f"middleware_{middleware.replace('/', '_').replace('.', '_')}",
                    f"app.use({middleware})",
                    f"Registers middleware: {middleware}",
                    ("{",),
                    1,
                    30,
                    {"config_type": "middleware"},
                )
            elif match := re_event_handler.search(line):
                emitter, event = match.groups()
                config = (
                    f"{emitter}_on_{event.replace(':', '_')}",
                    f"{emitter}.on('{event}', ...)",
                    self.get_leading_docstring(lines, i)
                    or f"Handles '{event}' event on {emitter}",
                    ("{", "=>"),
                    1,
                    30,
                    {"config_type": "event_handler", "target": emitter, "event": event},
                )
            if config:
                config_name, signature, docstring, opens, ahead, least, extra = config
                end_lineno = next(
                    (
                        self._find_block_end(code_lines, j)
                        for j in range(i, min(i + ahead, len(lines)))
                        if any(mark in text_lines[j] for mark in opens)
                    ),
                    lineno,
                )
                self.add_config_node(
                    file_id,
                    f"{file_id}::{config_name}",
                    config_name,
                    signature=signature,
                    docstring=docstring,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=max(token_count(lineno, end_lineno), least),
                    **extra,
                )

            # Calls. Module-level calls are sourced at the file.
            code = code_lines[lineno - 1]
            # The type arguments of a call are no part of its chain.
            code = _JS_TYPE_ARGUMENTS_RE.sub(_blank, code)
            if _TS_VARIABLE_RE.match(code):
                # `const name = value as Type` casts nothing: the value
                # the line gives the name is the one written.
                code = _TS_CAST_TYPE_RE.sub(_blank, code)
            # A module's top level takes in the bodies of callbacks, whose
            # variables are not the module's.
            scope = None if parent_id == file_id else parent_id
            self.begin_line(code, lineno, scope, line)
            # A variable has what the line that declared it gave it:
            # nothing where that gave it no type, whatever an earlier
            # declaration of its name did.
            if name := untyped[0]:
                entries = self.declared.setdefault(name, [])
                for e in entries.between(untyped[1], untyped[1]):
                    if not e[1]:
                        entries.remove(e)
                if not entries.between(untyped[1], untyped[1]):
                    entries.append((untyped[1], None, False, untyped[2]))
            untyped = ("", 0, None)
            if literal := _JS_LITERAL_RE.search(code):
                builtin = _JS_LITERAL_TYPES.get(literal["literal"][0], "string")
                if builtin == "Array":
                    # Its elements are read once its calls are.
                    array = f"[{self.builder.serial(self.builder.literals)}]", 0
                    self.builder.literals[array[0]] = []
                    builtin = f"Array<{array[0]}>"
                self.declared.setdefault(literal["name"], []).append(
                    (
                        lineno,
                        f"{builtin}()",
                        not line[:1].isspace(),
                        literal.start("name"),
                    )
                )
            elif binding := self.binding_re.search(code):
                untyped = (binding["name"], lineno, binding.start("name"))
            since = self.context_at(lineno)[1]
            # The definition the line is in, before it enters those the
            # line opens
            scope = reaching(lineno)[0]
            owner = self._owning_type(parent_id)
            self._closures = [
                (
                    found.start("closure"),
                    found["params"] or found["param"] or "",
                    found.end(),
                )
                for found in _JS_CLOSURE_RE.finditer(code)
                # A function with a name is an expression where it is a
                # value, else a declaration.
                if not found["named"]
                or _JS_VALUE_START_RE.search(
                    code, max(_run_start(code, found.start()) - 7, 0), found.start()
                )
            ]
            self._held_closures = [
                (
                    found.start("closure" if found["closure"] else "method"),
                    found["params"] or found["param"] or found["method_params"] or "",
                )
                for found in _JS_PROPERTY_CLOSURE_RE.finditer(code)
            ]
            self._bodies = []
            # A nameless function passed as an argument, whose block body
            # opens here and goes on over later lines, is a scope of its
            # own: the locals it declares are its, not the enclosing
            # definition's, and each such function its own. One that
            # opens after another on the line is written inside it.
            within = parent_id
            # The definitions the line declares, by their own columns
            own_spans = [
                (node_id, span)
                for node_id, name_end in own_defs
                if (span := _own_line_span(code, name_end))
            ]
            for bar, _params, end in self._closures:
                # The innermost of those that holds it holds it.
                holding = [
                    node_id
                    for node_id, (start, stop) in own_spans
                    if start <= bar < stop
                ]
                if holding and not within.startswith(f"{holding[-1]}::"):
                    within = holding[-1]
                text = code[bar:end].rstrip()
                if text.endswith("{"):
                    opened = end - 1
                elif text.endswith("=>") and code[end:].lstrip().startswith("{"):
                    opened = end + len(code[end:]) - len(code[end:].lstrip())
                else:
                    continue  # an expression body, or one over later lines
                if _block_close(code, opened) is not None:
                    continue  # the block closes on this line
                while callbacks and callbacks[-1][0] < lineno:
                    callbacks.pop()
                outer = callbacks[-1][1] if callbacks else 0
                block_end = self._find_block_end(code_lines, i, opened)
                if block_end <= lineno:
                    continue
                if (
                    self.paren_around(bar) is None
                    and len(self._open_calls) <= outer
                    and not invoked(i, bar)
                ):
                    # Not an argument: the lines of its body are those
                    # of the definition around it, or of the constant
                    # whose value holds it, and read its parameters too.
                    context = (*reaching(block_end)[0][:2], block_end - 1)
                    if holder and ";" not in code[holder[1] : bar]:
                        context = (holder[0], lineno, block_end - 1)
                    self.enter_context(*context, _ts_params(code[bar : opened + 1]))
                else:
                    callback_id = within = self._nameless_owner(
                        within,
                        f"callback@{self.source_line(lineno)}",
                        lineno,
                        block_end,
                        "callback",
                    )
                    callbacks.append(
                        (block_end - 1, len(self._open_calls) + len(self._opens))
                    )
                    callback_ids.add(callback_id)
                    context = (callback_id, since, block_end - 1)
                    span = (
                        (i, opened),
                        close_at.get((i, opened), (block_end - 1, None)),
                    )
                    enter(*context, code[bar : opened + 1], span)
                closed_after.add(context)
            # The declarators of a statement after its first, on this line
            # or one the statement goes on to, are definitions of their
            # own, as its first is and where it is: a `const` a constant,
            # a `let` or `var` one holding a function that function. One
            # holding a function takes its calls.
            if not continuing and (
                head := _TS_DECLARATION_HEAD_RE.match(code_lines[i])
            ):
                for at, *declarator in _declarators(code_lines, i, head.end()):
                    if declarator[2] is None:
                        continue
                    top = self._binds_file(declarator[0])
                    later_declarators.setdefault(at, []).append(
                        (parent_id, head[1], top, *declarator)
                    )
            declared_here = later_declarators.pop(i, ())
            for first in _TS_FIRST_DECLARATOR_RE.finditer(code):
                # The first is typed by its value as a later one is, where no
                # call is it: what a call returns, the line's binding gives.
                value_end = _value_end(code_lines[i], first.end())
                value = code[first.end() : value_end].rstrip()
                if value.endswith(")") and _is_call_chain(value):
                    continue
                typed_as = self._expression_origin(
                    code, first.end(), value_end, lineno, since, owner
                )
                # A member read it holds is read through the name itself: a
                # call of a name a body declares calls what the name holds.
                if typed_as and not re.search(r"(?:^|\.)[\w$]+$", typed_as):
                    self._rebind(first[1], lineno, first.start(1), typed_as)
            for scope_id, kind, top, later, name_end, value_at in declared_here:
                # A later declarator is typed by its value as the first is.
                value_end = _value_end(code_lines[i], value_at)
                typed_as = self._expression_origin(
                    code, value_at, value_end, lineno, since, owner
                )
                if typed_as:
                    self.declared.setdefault(later, []).append(
                        (lineno, typed_as, top, name_end - len(later))
                    )
                if top and later in self.declared:
                    # The declaration scan reads an indented line as inside
                    # a definition.
                    self.declared[later] = [
                        (line, value, at_top or line == lineno, at)
                        for line, value, at_top, at in self.declared[later]
                    ]
                held = _function_value(line[value_at:])
                constant = kind == "const"
                if not (constant or held):
                    continue
                lid = f"{scope_id}::{later}"
                claim(lid)
                held_typed = held and self._arrow_typed(held, line)
                value = line[value_at : _value_end(code_lines[i], value_at)]
                self.builder.add_node(
                    lid,
                    "Constant" if constant else "Function",
                    name=later,
                    value=value.strip(),
                    value_type=None,
                    lineno=lineno,
                    docstring="",
                    signatures=held_typed and self._overloads(lid, held_typed, lineno),
                )
                self.builder.add_edge(scope_id, lid, "DEFINES")
                if held:
                    own_defs.append((lid, name_end))
            typed = _TS_TYPED_VALUE_RE.search(code)
            calls = []  # (edge index, column of its `(`) of each on the line
            # A definition on this line takes the calls its parameters
            # and the first line of its body enclose, which run when it
            # is called, where the node is its own: one that an earlier
            # definition of the name already holds is not. A decorator
            # or computed key leading the line runs where the class or
            # object is defined, not in the member it leads.
            added = {node["id"] for node in self.builder.nodes[nodes_before:]}
            spans = field_spans + [
                (*span, node_id)
                for node_id, name_end in own_defs
                if node_id in added and (span := _own_line_span(code, name_end))
            ]
            if (parent_id in class_ids or parent_id in object_ids) and (
                region := _definition_time_region(code)
            ):
                spans.append((*region, self.enclosing_scope(parent_id)))
            # A field a class's code assigns, `this.name = value`, is the
            # class's own; one the body of a function whose prototype the
            # file writes to assigns, its instance's.
            for assigned in _JS_THIS_FIELD_RE.finditer(code):
                writer = assigner = source_at(i, assigned.start(), parent_id, spans)
                while assigner and assigner not in class_ids | prototype_ids:
                    assigner = assigner.rpartition("::")[0]
                if not assigner and writer.rpartition("::")[2] in prototyped:
                    assigner = f"{writer}.prototype"
                if assigner:
                    value = code[assigned.end() :].partition(";")[0].strip()
                    held = self.builder.assigned.setdefault(assigner, {})
                    held.setdefault(assigned[1], set()).add(value)
            # A `for` head that closes on the line: the names it binds hold
            # in a body written after it there.
            head = _JS_FOR_START_RE.search(code)
            over = not head and self._for_head_over_lines(code, i, code_lines)
            body = self._pairs.get(head.end() - 1, len(code)) if head else len(code)
            body = over[3] if over and over[3] is not None else body
            for match in re_call_candidate.finditer(code):
                if match.start(1) > body:
                    self._bind_patterns(code, i, code_lines, since, owner)
                    body = len(code) + 1
                if match.start(1) == declared_at:
                    continue
                # An optional call (`a?.b?.()`) is a call, and `a!.b()` one
                # whose receiver is asserted not null; only the receiver may
                # be nullish.
                callee = re.sub(r"[!?]+\.", ".", match.group(1))
                # `function name(` names the function it defines.
                if _JS_FUNCTION_HEAD_RE.search(
                    code,
                    max(_run_start(code, match.start(1), "*") - 8, 0),
                    match.start(1),
                ):
                    continue
                # `name(params) {` at the start of a line defines a
                # method of an object literal.
                if (
                    _TS_METHOD_MODIFIERS_RE.fullmatch(code, 0, match.start(1))
                    and "." not in callee
                    and _opens_body(code[match.start(1) :])
                ):
                    continue
                # JS/TS control-flow keywords look like calls (`if(x)`,
                # `while (y)`). Reject them by name — language general,
                # not codebase specific. After a `.` one names a member.
                member = _js_member_name(code_lines, i, match.start(1))
                if callee in _JSTS_CONTROL_KEYWORDS and not member:
                    continue
                paren = match.end()
                if code[paren] == "<":
                    # Generic arguments: `name<T, U>(`
                    depth = 0
                    for offset, char in enumerate(code[paren:]):
                        depth += (char == "<") - (char == ">")
                        if depth == 0:
                            rest = code[paren + offset + 1 :]
                            paren += offset + 1 + len(rest) - len(rest.lstrip())
                            break
                if code[paren : paren + 1] == "(":
                    # A call of a value a parameter or a variable holds,
                    # unless the name is a member of an expression.
                    origin = None
                    if not member:
                        origin = value_of(callee, lineno, since, owner, match.start(1))
                    # A literal is a value of its built-in type.
                    quote = _JS_LITERAL_RECEIVER_RE.search(
                        code,
                        max(_run_start(code, match.start(1), ".") - 1, 0),
                        match.start(1),
                    )
                    if quote:
                        builtin = _JS_LITERAL_TYPES.get(quote[1], "string")
                        origin = f"{builtin}().{callee}"
                    self._bind_closures(code, i, lines, match.start(1))
                    # `a.b.name()` is called on the property `b` of `a`.
                    receiver, _, called = callee.rpartition(".")
                    start = match.start(1)
                    if "." in receiver and not member:
                        span = (start, start + len(receiver))
                        held = self._expression_origin(
                            code, *span, lineno, since, owner
                        )
                        origin = origin or (held and f"{held}.{called}")
                    if callee.startswith("this."):
                        # `this` in an object literal that is the
                        # argument of a call is the `ThisType` of the
                        # parameter that holds it.
                        held = self._this_in_object(code, match.start(1))
                        if held:
                            origin = f"{held}.{called}"
                    self.add_call(
                        source_at(i, match.start(1), parent_id, spans),
                        callee,
                        lineno,
                        since,
                        match.start(1),
                        paren,
                        origin,
                    )
                    # `new Type<arguments>(` constructs the type it names.
                    made = self.builder.edges[-1]["metadata"]
                    root = callee.partition(".")[0]
                    bound = self._scopes.declaration(root, *[(lineno, start)] * 2)
                    new = _JS_NEW_RE.search(
                        code, max(_run_start(code, start) - 3, 0), start
                    )
                    if (
                        bound
                        and not member
                        and bound[0] in ("let", "const", "class")
                        and (
                            bound[0] == "class"
                            and callee == root
                            and not new
                            or self._scopes.unset(bound, (lineno, start))
                        )
                    ):
                        # Read before its declaration runs, or a class
                        # called without `new`, it throws.
                        made["site"] = {"throws": True}
                    elif (
                        bound
                        and not member
                        and bound[3] != self._scopes.whole
                        and not _JS_LOADS_RE.search(self._scopes.written(bound))
                    ):
                        # A local of a block or function around binds it,
                        # not to what a module it loads exports.
                        made["site"] = {"shadowed": True}
                    written = line[start + len(callee) : paren].strip()
                    if "origin" not in made and new:
                        made["origin"] = f"new {callee}{written}"
                    elif written:
                        made.setdefault("site", {})["types"] = written
                    closed = self._pairs.get(paren)
                    if closed is not None and code[closed + 1 : closed + 2] == "(":
                        # What the call returns is called, `make()()`.
                        made.setdefault("site", {})["called"] = True
                    calls.append((len(self.builder.edges) - 1, paren, start))
                    if typed and start == typed.end():
                        initialises = (calls[-1][0], typed["type"])
            if initialises:
                # The type is that of the call's result where the value
                # ends with the call.
                ended = self._ends_value(code, initialises[0])
                if ended:
                    site = self.builder.edges[initialises[0]]["metadata"]
                    site.setdefault("site", {})["declared"] = initialises[1]
                if ended is not None:
                    initialises = None
            if returning and self._ends_value(code, returning[0]) is not None:
                if not self._ends_value(code, returning[0]):
                    returning[1][returning[2]] = None
                returning = None
            # What a function declared without a return type returns:
            # the value of each `return` of its body, or of an arrow
            # function the expression it is.
            returned = (self._signatures.get(parent_id) or [{}])[-1].get("returned")
            returns = (
                not continuing
                and lineno > nameless.get(parent_id, 0)
                and len(re.findall(r"\breturn\b", code))
            )
            found = None
            # Whether the value the line gives is an expression: an
            # arrow's `{` opens a block unless its body is parenthesised,
            # which an object expression must be.
            expression = returns == 1
            if valued and code.count("=>") == 1:
                returned = valued["returned"]
                found = re.search(r"=>\s*(?P<new>new\s+)?", code)
                expression = bool(found) and code[found.end() :].lstrip().startswith(
                    "("
                )
            elif returns == 1:
                found = _JS_RETURN_RE.search(code)
                if found:
                    # A `return` in a body written on the line is the
                    # innermost function's, which `parent_id` may not be.
                    text, starts = self._scopes.text, self._scopes.starts
                    absolute = self._blocks[1][lineno - 1] + found.start()
                    held = self._scopes.function(self._scopes.place(absolute))
                    lead = held and _run_start(
                        text, starts[held[0][0] - 1] + held[0][1]
                    )
                    named = held and _JS_WORD_END_RE.search(
                        text, max(lead - 100, 0), lead
                    )
                    name = named.group() if named else None
                    if name != parent_id.rpartition("::")[2] and name:
                        bound = self._declaration(name, lineno)
                        if bound and bound[0] == "function":
                            returned = self._signatures.get(
                                f"{parent_id}::{name}", [{}]
                            )[-1].get("returned")
            if returned is not None and (found or returns):
                value = code[found.end() :].partition(";")[0].strip() if found else ""
                first = [
                    index for index, _, start in calls if found and start == found.end()
                ]
                record = None
                if value == "this" and not found["new"]:
                    record = "", "self"
                elif first:
                    made = self.builder.edges[first[0]]
                    origin = made["metadata"].get("origin")
                    ended = self._ends_value(code, first[0])
                    # One call, whose receiver is no other call's result.
                    read = self._read_as(origin)
                    if "#" not in (read or "") and ended is not False:
                        record = made["target"][4:], read
                    if record and ended is None:
                        returning = first[0], returned, len(returned)
                elif re.fullmatch(r"[\w$]+", value):
                    held = self._read_as(
                        self.call_origin(f"{value}.x", lineno, since, owner)
                    )
                    bound = self._declaration(value, lineno)
                    if held and "#" not in held:
                        record = "", held[:-2]
                    elif bound and bound[0] in ("function", "class"):
                        # A function or a class the name declares
                        record = "", value
                elif value.lstrip("(").startswith("{") and expression:
                    # An object: its properties are read where the
                    # function is written, and the functions among them
                    # are members of the object itself.
                    opened = code.index("{", found.end())
                    key = f"[{self.builder.serial(self.builder.literals)}]"
                    # An arrow's own node holds the object its body is,
                    # which the arrow may declare on this line.
                    holder = arrow_id or parent_id
                    held, object_end = self._object_held(
                        code_lines, i, opened, since, owner
                    )
                    self.builder.literals[key] = [held]
                    self.builder.literal_scopes[key] = holder
                    record = "", key
                    # An object that holds a function written in it owns
                    # that function; one that holds none is no owner.
                    if any(str(value).startswith("=") for value in held.values()):
                        object_id = self._nameless_owner(
                            holder,
                            f"object@{self.source_line(lineno)}",
                            lineno,
                            object_end,
                            "object",
                        )
                        self.builder.literal_scopes[key] = object_id
                        returned_objects.append(
                            (holder, lineno, object_end, object_id, since)
                        )
                    # A `return` on a line of the object is one of a
                    # function written in it.
                    nameless[holder] = max(nameless.get(holder, 0), object_end)
                returned.append(record)
            # An object literal a call is passed, that its line leaves
            # open over later lines and that holds a function, owns the
            # functions written in it, a name no source can write. Its
            # properties are read as the ones of the object a function
            # returns are. One that opens after another on the line is
            # written inside it.
            passed, before = _passed_objects(
                code, brackets, before, {paren for _, paren, _ in calls}
            )
            for opened in passed:
                held, object_end = self._object_held(
                    code_lines, i, opened, since, owner
                )
                if not any(str(value).startswith("=") for value in held.values()):
                    continue
                opened_here = returned_objects and returned_objects[-1][1] == lineno
                # Its lines are those of the definition the next line is
                # in: one the line opens, or the one around one it closes.
                scope_id = self.context_at(lineno + 1)[0]
                returned_objects.append(
                    (
                        scope_id,
                        lineno,
                        object_end,
                        self._nameless_owner(
                            returned_objects[-1][3] if opened_here else scope_id,
                            f"object@{self.source_line(lineno)}",
                            lineno,
                            object_end,
                            "object",
                        ),
                        since,
                    )
                )
            if (
                not continuing
                and lineno > nameless.get(parent_id, 0)
                and _JS_BLOCK_FUNCTION_RE.search(code)
                and self.context_at(lineno + 1)[0] == parent_id
            ):
                nameless[parent_id] = self._find_block_end(code_lines, i)
            for held, name, start, end in self._held_later.pop(i, ()):
                # A property an object read on an earlier line is given
                # on this one holds what this line's calls give it.
                held[name] = self._expression_origin(
                    code, start, end, lineno, since, owner
                )
            self._bind_closures(code, i, code_lines, len(code))
            self._name_results(code, lineno, since, owner)
            for name, at, start, end in self._scopes.assigned.get(lineno, ()):
                held = self._assigned(code, name, at, start, end, lineno, since, owner)
                if held:
                    self.declared.setdefault(name, []).append(
                        (lineno, held, False, start)
                    )
            if body <= len(code):
                self._bind_patterns(code, i, code_lines, since, owner)
            for index, paren, _ in calls:
                # What a call closed on its line is passed, unless it
                # spreads a value over several arguments.
                closed = self._pairs.get(paren)
                if closed and "..." not in code[paren:closed]:
                    site = self.builder.edges[index]["metadata"]
                    site.setdefault("site", {})["arguments"] = self._elements(
                        code, paren + 1, lineno, since, owner
                    )
            if array:
                # One element to a line, or all on the line of the `[`;
                # one that goes on over lines leaves them unknown.
                start = literal.end("name") if literal and not array[1] else 0
                opened = code.index("[", start) + 1 if start else 0
                held = self.builder.literals[array[0]]
                if held is not None:
                    held += self._elements(code, opened, lineno, since, owner)
                depth = array[1] + sum(
                    (char in "([{") - (char in ")]}")
                    for char in code[max(opened - 1, 0) :]
                )
                if depth > 1:
                    self.builder.literals[array[0]] = None
                array = (array[0], depth) if depth > 0 else None
            i += 1

        if file_id in self.builder.commonjs:
            # A module that writes CommonJS exports runs as CommonJS, its
            # `export`s written to `exports` as TypeScript compiles them:
            # `export =` last, the others before a `module.exports =`, which
            # leaves none of them. A name both give is either.
            if any(entry[0] == "=" for entry in es):
                cjs = {}
            elif "=" in cjs:
                es = []
            else:
                es += [
                    (name, "", name, file_id)
                    for name in _JS_EXPORTED_DECLARATION_RE.findall(scopes.text)
                ]
            for name in {entry[0] for entry in es} & cjs.keys():
                cjs[name] = (name, "", None, file_id)
            es = [entry for entry in es if entry[0] not in cjs]
            es += cjs.values()
        if es:
            self.builder.reexports.setdefault(file_id, []).extend(es)
        for node in self.builder.nodes[self._first_node :]:
            if node["id"] in static_ids:
                node["metadata"]["static"] = True
        # What the resolver reads by the ID a node ends up with: its
        # signatures, and each field's type for each class or interface of
        # that name in the file.
        owned = {}
        for owner, field in self.fields:
            owned.setdefault(owner, []).append(field)
        for node in self.builder.nodes[self._first_node :]:
            if typed := node["metadata"].pop("signatures", None):
                self.builder.signatures[node["id"]] = typed
            name = node["metadata"].get("name")
            for field in owned.get(name, ()):
                held = self._field_types(node, name, field)
                if len(held) == 1:
                    fields = self.builder.fields.setdefault(node["id"], {})
                    fields.setdefault(field, next(iter(held))[:-2])

        self.restore_source_lines()

    def _parameter_properties(self, class_id, params):
        """Record the fields the constructor parameters ``params`` of the
        class ``class_id`` declare, of the type each is written with or, as
        TypeScript gives it, its default's."""
        owner = class_id.rpartition("::")[2]
        for param in params:
            if found := _TS_PARAMETER_PROPERTY_RE.fullmatch(param):
                for pattern in _TS_DECLARATIONS[:3]:
                    if declared := pattern.match(found["rest"]):
                        held = f"{self._scanned_type(declared.groupdict())}()"
                        self.fields.setdefault((owner, declared["name"]), set()).add(
                            held
                        )
                        break

    def _extract_ts_types(self, full_sig: str) -> tuple:
        """Extract return type and parameter types from TypeScript function signature.

        Returns: (return_type: str, param_types: list[str], typed: dict),
        the last the signature as the resolver reads it: the bound of each
        type parameter, the parameters and the return type as written, and
        the types of the parameters of each function the signature takes,
        by the position of the argument it is.

        Handles common patterns:
        - function foo(x: string): number
        - async function bar(): Promise<void>
        - (a: Type, b: Type) => ReturnType
        """
        return_type = "void"
        param_types = []
        typed = {"generics": {}, "params": [], "returns": "", "callbacks": {}}
        typed["body"] = full_sig.rstrip().endswith("{")
        # What each `return` of the body gives, as the call loop reads them;
        # None where what is returned is wrapped: in a promise or a generator
        typed["returned"] = [] if typed["body"] else None
        callbacks = typed["callbacks"]

        # The parameter list is the first `(...)` outside type
        # parameters. The return type follows it, up to the `{` or `;`
        # the signature ends with or the `=>` of an arrow function,
        # which unlike that of a function type follows no `)`.
        depth = 0
        opened = angle = None
        for closing, char in enumerate(full_sig):
            if char in "(<[{":
                if char == "(" and not depth:
                    opened = closing
                elif char == "<" and not depth:
                    angle = closing
                depth += 1
            elif char in ")]}" or (char == ">" and full_sig[closing - 1] != "="):
                depth -= 1
                if not depth and opened is not None:
                    break
        ret_match = opened is not None and re.match(
            r"\s*:\s*(.+?)(?:(?<![)\s])\s*=>.*|\s*[{;])?$", full_sig[closing + 1 :]
        )
        if ret_match:
            return_type = typed["returns"] = ret_match.group(1).strip()
            if len(return_type) > 50:
                return_type = return_type[:47] + "..."

        if opened is not None and re.search(r"\basync\b|\*", full_sig[:opened]):
            typed["returned"] = None

        if angle is not None and full_sig[:opened].rstrip().endswith(">"):
            typed["generics"].update(_bounded_parameters(full_sig[angle:opened]))

        if opened is not None:
            params_str = full_sig[opened + 1 : closing].strip()
            parts = _bracketed_parts(params_str + ")")
            # What the function is called on is no argument.
            this = bool(re.match(r"this\s*:", params_str))
            typed["params"] = parts[this:]
            for position, param in enumerate(typed["params"]):
                if found := _TS_CALLABLE_RE.match(param):
                    # A rest parameter's type is marked `...`.
                    callbacks[str(position)] = [
                        ("..." if part.startswith("...") else "")
                        + part.partition(":")[2].strip()
                        for part in _bracketed_parts(found["params"] + ")")
                        if not part.startswith("this")
                    ]
            for param in parts:
                if ":" not in param:
                    param_types.append("any")
                    continue
                # The type, without the parameter's default
                type_part = re.split(r"=(?!>)", param.partition(":")[2], maxsplit=1)[
                    0
                ].strip()
                if len(type_part) > 40:
                    type_part = type_part[:37] + "..."
                param_types.append(type_part)

        return return_type, param_types, typed

    def _arrow_typed(self, parameters, line):
        """The signature, as the resolver reads it, of the arrow function
        that ``line`` declares, ``parameters`` its text from its parameter
        list on."""
        typed = self._extract_ts_types(parameters)[2]
        if not parameters.startswith("("):
            typed = dict(typed, params=[], callbacks={})
        typed["returned"] = None if re.search(r"\basync\b", line) else []
        return typed

    def _overloads(self, node_id, typed, end):
        """The signatures that ``node_id`` is declared with up to this one,
        ``typed``, written to line ``end``: those of its overloads before
        it, and its own. The node's lines run to the last one's end."""
        if node_id not in self.builder.node_registry:
            self._signatures[node_id] = []
        elif node := self._node(node_id):
            held = node["metadata"]
            held["end_lineno"] = max(held.get("end_lineno") or 0, end)
        declared = self._signatures.setdefault(node_id, [])
        declared.append(typed)
        return declared

    def _build_ts_signature(
        self, name: str, param_types: list, return_type: str
    ) -> str:
        """Build a TypeScript-style signature string.

        Returns: "function name(p1: T1, p2: T2): ReturnType"
        """
        params = ", ".join(f"p{i}: {t}" for i, t in enumerate(param_types))
        return f"function {name}({params}): {return_type}"
