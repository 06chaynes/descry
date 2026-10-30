"""C# / VB.NET regex-based baseline parser.

Covers the common cases of .NET symbol extraction: namespaces (both
block and file-scoped), classes / interfaces / structs / records /
enums / delegates, methods (including async ``Task<T>``), constructors,
operators, local functions, properties, fields, ``using`` statements,
and call sites with the origin of their receiver. Of the branches of an
``#if`` only the first is read. Visual Basic source (.vb) goes through
the same pipeline with only file discovery — full VB parsing is out of
scope (scip-dotnet covers both via SCIP).
"""

from __future__ import annotations

import posixpath
import re

from descry._parsing import (
    BaseParser,
    _blank,
    _bracketed_parts,
    _closing,
    _flat,
    _generic_metadata,
    _innermost,
    _strip_generics,
    _text,
    call_arguments,
    is_generated_source,
    record_lambdas,
    source_lines,
    token_count,
)
from descry.clang_parser import _if_signs
from descry.rust_parser import _STATEMENT_START_RE

# namespace App.Foo  (block) or  namespace App.Foo;  (file-scoped, C# 10+)
_RE_NAMESPACE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.]*)")

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_PATH = rf"{_NAME}(?:\.{_NAME})*"
# Type arguments or type parameters, nested up to three deep
_GENERIC = r"<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>"

# using System;  /  using static System.Math;  /  using Foo = App.Bar;
_RE_USING = re.compile(
    r"^\s*(?P<global>global\s+)?using\s+(?P<static>static\s+)?"
    rf"(?:(?P<alias>{_NAME})\s*=\s*)?(?P<rooted>global::)?"
    rf"(?P<path>{_PATH})\s*(?:{_GENERIC})?\s*;"
)
# A type as written: a possibly qualified name with its type arguments, or a
# tuple, then any `?`, `*` and array ranks
_TYPE = (
    rf"(?:(?:global::)?{_NAME}(?:{_GENERIC})?(?:\.{_NAME}(?:{_GENERIC})?)*"
    r"|\((?:[^()]|\([^()]*\))*\))(?:\?|\*|\[[\s,]*\])*"
)

_TYPE_MODIFIERS = (
    r"(?:public|internal|private|protected|static|sealed|abstract|"
    r"partial|readonly|new|unsafe|ref|record|file)"
)
_RE_CLASS = re.compile(
    rf"^\s*(?:{_TYPE_MODIFIERS}\s+)*"
    r"(class|struct|record|interface|enum)\s+"
    # `record name = ...` is a field of a type named `record`.
    r"([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*=)"
)
# The type parameters and the opening `(` of a primary constructor, after
# the name of a type
_RE_PRIMARY_CONSTRUCTOR = re.compile(rf"\s*+(?:{_GENERIC})?\s*\(")

_METHOD_MODIFIERS = (
    r"(?:public|internal|private|protected|static|virtual|abstract|"
    r"override|sealed|async|extern|new|partial|unsafe|readonly|volatile|"
    r"required|const|event|ref|implicit|explicit)"
)
# public delegate TResult Handler<T>(T input);
_RE_DELEGATE = re.compile(
    rf"^\s*(?:{_METHOD_MODIFIERS}\s+)*delegate\s+{_TYPE}\s+({_NAME})"
)
# A member of a type, up to what follows its name: `(` for a method or a
# constructor, `[` for an indexer, anything else for a property, a field or
# an event. A constructor has no type; `IFoo.Name` implements a member of
# IFoo explicitly.
_RE_MEMBER = re.compile(
    rf"^\s*+(?P<modifiers>(?:{_METHOD_MODIFIERS}\s++)*)"
    rf"(?:(?P<type>{_TYPE})\s++)?(?:(?P<explicit>{_TYPE})\s*\.\s*)?"
    rf"(?P<name>~?{_NAME})\s*+(?P<generic>{_GENERIC})?\s*+(?P<next>[(\[{{;=,]|$)"
)
# public static Money operator +(Money a, Money b)  /  implicit operator Money(int n)
_RE_OPERATOR = re.compile(
    rf"^\s*(?P<modifiers>(?:{_METHOD_MODIFIERS}\s+)*)(?:(?P<type>{_TYPE})\s+)?"
    rf"operator\s*(?P<symbol>{_TYPE}|[^\s(]+)\s*\("
)
# A local function: `Type Name(` at the start of a statement
_RE_LOCAL_FUNCTION = re.compile(
    rf"^\s*(?:(?:static|async|unsafe|extern)\s+)*(?P<type>{_TYPE})\s+"
    rf"(?P<name>{_NAME})\s*(?P<generic>{_GENERIC})?\s*\("
)
# One parameter of a parameter list, behind its attributes
# A match never starts inside a run of whitespace, which the one starting
# where the run starts holds.
_RE_PARAMETER = re.compile(
    r"(?<!\s)\s*+(?:\[[^\]]*\]\s*+)*"
    r"(?P<modifiers>(?:(?:this|ref|out|in|params|scoped|readonly)\s++)*)"
    rf"(?P<type>{_TYPE})\s+(?P<name>{_NAME})\s*(?:=[^,]*)?(?:,|$)"
)
# A member of an enum, behind its attributes: `Name`, `Name = value`
_RE_ENUM_MEMBER = re.compile(
    rf"(?:^|(?<=[{{,]))\s*(?:\[[^\]]*\]\s*)*({_NAME})\s*(?=[=,}}]|$)"
)
# The name of an attribute in a section whose arguments are blanked:
# `[Name`, `, Name`, `[return: Name`, or a name that starts a line
_RE_ATTRIBUTE = re.compile(rf"(?:^|\[|,)\s*(?:\w+\s*:\s*)?({_NAME}(?:\.{_NAME})*)")

# Call site: identifier(, obj.method(, Class.static(, in code made plain
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*\("
)
# A keyword and the `.` of a member of the literal blanked behind it
_RE_KEYWORD_BEFORE_MEMBER = re.compile(
    r"(?:return|throw|in|is|as|case|else|do|await|out|ref|new|when|and|or|not"
    r"|true|false|null)\s*\.\s*"
)
# A literal that ends just before the `.` of a member call, written as C#
# allows: a character, `true`/`false`/`null`, an integer, its digits apart
# by `_` and its suffix one `u` and one `l` at most, or a real, its suffix
# one of `f`, `d` and `m`. `@name` is the identifier `name`, no literal.
_DIGITS = r"\d(?:[\d_]*\d)?"
_RE_LITERAL_TAIL = re.compile(
    r"(?<![\w.@])(?:'(?:[^'\\]|\\.)*'|true|false|null"
    r"|(?:0[xX](?:_*[0-9a-fA-F])+|0[bB](?:_*[01])+"
    rf"|{_DIGITS})(?:[uU][lL]?|[lL][uU]?)?"
    rf"|(?:{_DIGITS})?\.{_DIGITS}(?:[eE][+-]?{_DIGITS})?[fFdDmM]?"
    rf"|{_DIGITS}(?:[eE][+-]?{_DIGITS})?[fFdDmM]"
    rf"|{_DIGITS}[eE][+-]?{_DIGITS})\s*\.$"
)
# The integer types in the order C# gives an integer literal the first that
# holds its value, with the largest value of each
_INTEGERS = (
    ("int", 2**31 - 1),
    ("uint", 2**32 - 1),
    ("long", 2**63 - 1),
    ("ulong", 2**64 - 1),
)


def _negated_type(kind, literal):
    """The C# type of `-literal`, ``literal`` of type ``kind``: that of a
    signed number; `long` of a `uint`, but `int` for `-2147483648`; `long`
    for `-9223372036854775808`; None for any other."""
    if kind in ("int", "long", "float", "double", "decimal"):
        return kind
    value = literal.lower().replace("_", "")
    if kind == "uint":
        return "int" if value == "2147483648" else "long"
    return "long" if kind == "ulong" and value == "9223372036854775808" else None


def _literal_type(literal):
    """The C# type of ``literal``, None for `null`: a real literal's by its
    suffix, `double` without one; an integer literal's the first of `int`,
    `uint`, `long` and `ulong` its suffix allows that holds its value, none
    for one too large for all, which C# rejects."""
    low = literal.lower().replace("_", "")
    if low[0] == "'":
        return "char"
    if low in ("true", "false"):
        return "bool"
    if low == "null":
        return None
    based = low[:2] in ("0x", "0b")
    real = not based and {"f": "float", "d": "double", "m": "decimal"}.get(low[-1])
    if real or not based and ("." in low or "e" in low):
        return real or "double"
    digits = low.rstrip("ul")
    suffix = low[len(digits) :]
    significant = (digits[2:] if based else digits).lstrip("0")
    if len(significant) > 64:
        # More digits than 64 bits hold in any base
        return None
    value = int(significant or "0", {"0x": 16, "0b": 2}.get(low[:2], 10))
    allowed = {"": ("int", "uint", "long"), "u": ("uint",), "l": ("long",)}
    return next(
        (
            t
            for t, top in _INTEGERS
            if (t in allowed.get(suffix, ()) or t == "ulong") and value <= top
        ),
        None,
    )


def _literal_receiver(code_lines, lines, index, column):
    """The C# type of the literal that a member call written at ``column``
    of line ``index`` is made on, None for none. A character literal is read
    on the raw line; any other where comments are blank, past the line's
    start where the line starts with the call's `.`."""
    found = _RE_LITERAL_TAIL.search(lines[index][:column].rstrip())
    if found and found[0][0] == "'":
        return _literal_type(found[0].rstrip()[:-1].rstrip())
    row, text = index, code_lines[index][:column]
    if not text.rstrip().endswith("."):
        return None
    while row and not text.rstrip()[:-1].strip():
        row -= 1
        text = f"{code_lines[row]}\n{text}"
    found = _RE_LITERAL_TAIL.search(text.rstrip())
    if not found:
        return None
    # The line the literal starts on and its column there
    row += text.count("\n", 0, found.start())
    at = found.start() - text.rfind("\n", 0, found.start()) - 1
    if lines[row][at - 1 : at] == "@":
        return None
    return _literal_type(found[0].rstrip()[:-1].rstrip())


# The type arguments of a generic name, innermost first: what follows them
# tells them from two comparisons
_RE_TYPE_ARGUMENTS = re.compile(
    r"(?<=\w)<[\w\s,.?\[\]]*>(?=\s*(?:[(.{>,)\]]|$)|\s+[A-Za-z_])"
)
# An object created without its type next to its arguments, as what a
# variable, a property or a method is declared to be: `Type name = new(...)`,
# `Type Name { get; } = new(...)`, `Type Name(...) => new(...)`. Or without
# arguments: `new Type {`
_RE_CREATION = re.compile(
    rf"(?<![\w.])(?P<declared>{_PATH})\??\s+{_NAME}\s*"
    r"(?:\([^()]*\)\s*|\{[^{}]*\}\s*)?=>?\s*new\s*\("
    rf"|\bnew\s+(?P<created>{_PATH})\s*(?=\{{|$)"
)
# `((Type) value).` or `(value as Type).` before a member: the value is of
# that type
_RE_CAST = re.compile(
    rf"\(\(\s*({_PATH})\s*\??\s*\)\s*[\w.]+\s*\)\s*\.\s*$"
    rf"|\(\s*[\w.]+\s+as\s+({_PATH})\s*\??\s*\)\s*\.\s*$"
)
# A variable its declaration gives no type: `var name`, the names of `var
# (a, b)`, the parameters of a lambda, a range variable of a query
_RE_UNTYPED = re.compile(
    rf"\b(?:var|dynamic)\s+({_NAME})|\bvar\s*\(([^()]*)\)"
    rf"|(?<![\w.])({_NAME})\s*=>|\(\s*({_NAME}(?:\s*,\s*{_NAME})*)\s*\)\s*=>"
    rf"|\b(?:from|let|join|into)\s+({_NAME})\s*(?=in\b|=)"
)
# `foreach (var name in path)`, whose variable holds an element of the path
_RE_FOREACH_VAR = re.compile(rf"\bforeach\s*\(\s*var\s+({_NAME})\s+in\s+({_PATH})\s*\)")
# `var name = path[index];`, whose variable holds an element of the path
_RE_ELEMENT_VAR = re.compile(rf"\bvar\s+({_NAME})\s*=\s*({_PATH})\s*\[[^\[\];]*\]\s*;")
# What a variable an expression declares follows: `is` and a type, `out var`,
# whose `var` is group 1, or `out` and a type
_RE_EXPRESSION_VARIABLE = re.compile(
    r"\b(?:is(?:\s+not)?|out(\s+var)?)\s+[\w.<>,?\[\]\s]*$"
)
# What follows the parameter list of a method, a local function or a
# constructor
_RE_AFTER_PARAMETERS = re.compile(r"\s*(?:\{|=>|where\b|:)")
# The head of a `using` statement that declares its resource: `using (var`
_RE_RESOURCE = re.compile(r"\s*(?:await\s+)?using\s*\((?=\s*var\s)")
# A statement whose head scopes what an expression in it declares
_RE_SCOPING_HEAD = re.compile(
    r"\s*(?:await\s+)?(?:while|for|foreach|using|lock|fixed|switch|catch\b.*\bwhen)\s*$"
)

# What opens a comment or a literal: `//`, `/*`, a verbatim string, a raw
# string, or an ordinary string or character literal up to its end
_RE_LITERAL = re.compile(
    r"""//|/\*|(?:@\$?|\$@)"|"{3,}|"(?:[^"\\]|\\.)*"?|'(?:[^'\\]|\\.)*'?"""
)
# The rest of a verbatim string, which takes no backslash escapes: `""` is a
# quote in it.
_RE_VERBATIM_END = re.compile(r'(?:[^"]|"")*"(?!")')
_RE_DIRECTIVE = re.compile(r"\s*#\s*(\w*)\s?(.*)")

# The keywords that name a type
_BUILTIN_TYPES = frozenset(
    {
        "void",
        "dynamic",
        "object",
        "string",
        "bool",
        "byte",
        "sbyte",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "char",
        "float",
        "double",
        "decimal",
        "nint",
        "nuint",
    }
)

_DOTNET_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "foreach",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "yield",
        "throw",
        "try",
        "catch",
        "finally",
        "using",
        "lock",
        "fixed",
        "checked",
        "unchecked",
        "new",
        "typeof",
        "sizeof",
        "nameof",
        "is",
        "as",
        "when",
        "stackalloc",
        "async",
        "await",
        "true",
        "false",
        "null",
        "this",
        "base",
        "ref",
        "out",
        "in",
        "get",
        "set",
        "init",
        "add",
        "remove",
        "value",
        # Linq keywords (query syntax)
        "from",
        "where",
        "select",
        "group",
        "into",
        "orderby",
        "join",
        "let",
        "on",
        "equals",
        "by",
        "ascending",
        "descending",
        # Declaration keywords
        "var",
        "record",
        "struct",
        "class",
        "interface",
        "enum",
        "delegate",
        "namespace",
        "partial",
        "sealed",
        "abstract",
        "override",
        "virtual",
        "protected",
        "public",
        "private",
        "internal",
        "readonly",
        "static",
        "extern",
        "unsafe",
        "volatile",
        "params",
        "operator",
        "explicit",
        "implicit",
        "event",
        "goto",
        "notnull",
        "required",
        "file",
        "scoped",
        "global",
        "not",
        "and",
        "or",
        "with",
    }
    | _BUILTIN_TYPES
)

_NOT_TYPES = _DOTNET_CONTROL_KEYWORDS - _BUILTIN_TYPES

# A declared type: a possibly qualified name that is no keyword, and its
# type arguments
_DECLARED_TYPE = (
    rf"(?<![\w.])(?!(?:{'|'.join(sorted(_NOT_TYPES))}|dynamic)\b)"
    rf"(?P<type>{_PATH})(?P<generic>{_GENERIC})?(?P<nullable>\?)?"
)
# A declared name: no word that follows an expression
_DECLARED_NAME = (
    r"(?!(?:is|as|in|and|or|not|when|with|switch|ascending|descending)\b)"
    rf"(?P<name>{_NAME})"
)
_DECLARATIONS = (
    # `Type name` before `=`, `;`, `,`, `)`, `:`, `in` or `when`: a local, a
    # field, a parameter, a `foreach` variable or one a pattern binds
    re.compile(
        _DECLARED_TYPE + r"\s+" + _DECLARED_NAME + r"\s*(?=[=;,):]|(?:in|when)\b)"
    ),
    # `value is Type name`
    re.compile(
        r"\bis\s+(?:not\s+)?" + _DECLARED_TYPE + r"\s+" + _DECLARED_NAME + r"\b"
    ),
    # `var name = new Type(...);`, `using (var name = new Type(...))` /
    # `var name = new Type {`
    re.compile(
        rf"\bvar\s+(?P<name>{_NAME})\s*=\s*new\s+(?P<type>{_PATH})"
        rf"(?P<generic>{_GENERIC})?\s*"
        r"(?:\((?:[^()]|\([^()]*\))*\)\s*(?=[;{)])|(?=\{|$))"
    ),
    # `var name = (Type)value;` / `var name = value as Type;`
    re.compile(
        rf"\bvar\s+(?P<name>{_NAME})\s*=\s*\(\s*(?P<type>{_PATH})"
        rf"(?P<generic>{_GENERIC})?\s*\??\s*\)\s*[\w(@\"]"
    ),
    re.compile(
        rf"\bvar\s+(?P<name>{_NAME})\s*=[^;=]*\bas\s+(?P<type>{_PATH})"
        rf"(?P<generic>{_GENERIC})?\s*\??\s*;"
    ),
    # The first type a type parameter is constrained to: `where T : Type`
    _RE_WHERE := re.compile(
        rf"\bwhere\s+(?P<param>{_NAME})\s*:\s*"
        r"(?:(?:class|struct|notnull|unmanaged|default)\??\s*,\s*)*"
        rf"(?!(?:class|struct|notnull|unmanaged|default|new)\b)(?P<type>{_PATH})"
    ),
)
# An array type, which a declaration is read as `System.Array<Type>`
_RE_ARRAY_TYPE = re.compile(rf"({_PATH})(?:{_GENERIC})?\??(?:\[[\s,]*\])+")


# What a C# type's or namespace's body opens after, and a declaration a
# line may go on to after the end of another
_TYPE_KEYWORDS = r"(?:class|struct|interface|enum|record|namespace)\b"
_DECLARATION_MODIFIERS = (
    r"(?:(?:public|private|protected|internal|static|sealed|abstract|partial|"
    r"readonly|unsafe|file|new|ref)\s+)*"
)
_RE_TYPE_HEAD = re.compile(
    rf"\s*(?:\[[^\]]*\]\s*)*{_DECLARATION_MODIFIERS}{_TYPE_KEYWORDS}"
)
_RE_DEFINITION_HEAD = re.compile(
    rf"{_DECLARATION_MODIFIERS}(?:{_TYPE_KEYWORDS}|delegate\b)"
)


def _code_lines(lines):
    """``(lines as code, guarded lines)``: comments, preprocessor directives
    and string and character literals turned to spaces, and the lines of
    every `#elif` and `#else` branch emptied, so that one branch of each
    `#if` is read. ``guarded`` maps each line index that sits inside an
    `#if` to the signed symbols its arm holds (see ``clang_parser``)."""
    code_lines = []
    guarded = {}
    closer = None  # what ends the comment or string the line starts inside of
    inactive = []  # for each open `#if`, whether its `#else` was passed
    arms = []  # the signed symbols each open `#if`'s read arm holds
    for index, line in enumerate(lines):
        directive = None if closer else _RE_DIRECTIVE.match(line)
        if directive:
            if directive[1] == "if":
                inactive.append(False)
                arms.append(_if_signs("if", directive[2] or ""))
            elif directive[1] in ("elif", "else") and inactive:
                inactive[-1] = True
            elif directive[1] == "endif" and inactive:
                inactive.pop()
                arms.pop()
        if directive or any(inactive):
            code_lines.append("")
            continue
        if arms:
            guarded[index] = frozenset().union(*arms)
        code = []
        position = 0
        while position < len(line):
            start = position
            if closer is None:
                match = _RE_LITERAL.search(line, position)
                if not match or match.group() == "//":
                    code.append(line[position : match.start() if match else None])
                    break
                code.append(line[position : match.start()])
                start, position = match.span()
                opener = match.group()
                if _interpolated(line, match):
                    blanked, position, closer = _interpolation(line, match)
                    code.append(blanked)
                    continue
                if opener == "/*":
                    closer = "*/"
                elif opener.startswith('"""'):
                    closer = opener
                elif opener.endswith('"') and opener[0] in "@$":
                    closer = '"'
                else:
                    code.append(" " * (position - start))
                    continue
            if closer == '"':
                end = _RE_VERBATIM_END.match(line, position)
                end = end.end() if end else -1
            else:
                end = line.find(closer, position)
                end = end + len(closer) if end >= 0 else -1
            if end < 0:
                code.append(" " * (len(line) - start))
                break
            code.append(" " * (end - start))
            position, closer = end, None
        # `@name` is the identifier `name`.
        code_lines.append(re.sub(r"@(?=[A-Za-z_])", " ", "".join(code)))
    return code_lines, guarded


def _interpolated(line, match):
    """Whether the literal ``match`` finds in ``line`` is an interpolated
    string: `$"`, `$@"`, `@$"`, or a raw string after `$`."""
    return match.group() in ('$@"', '@$"') or (
        match.group()[0] == '"' and line[match.start() - 1 : match.start()] == "$"
    )


def _interpolation(line, match):
    """``(text, end, closer)`` of the interpolated string ``match`` finds in
    ``line``: the string turned to spaces but for the code of each `{...}`,
    whose own literals are blanked, the position after it, and what closes
    it on a later line where it goes on past this one. A raw string's code
    opens with as many braces as `$` it is written after; in any other `{{`
    is text."""
    opener = match.group()
    verbatim = opener in ('$@"', '@$"')
    quotes = opener if opener.startswith('"""') else '"'
    braces = len(line[: match.start()]) - len(line[: match.start()].rstrip("$"))
    braces = max(braces, 1)
    text = [" " * len(opener)] if quotes != '"' or verbatim else [" "]
    i = match.start() + len(text[0])
    while i < len(line):
        if quotes == '"' and not verbatim and line[i] == "\\":
            text.append(" " * len(line[i : i + 2]))
            i += 2
        elif verbatim and line.startswith('""', i):
            text.append("  ")
            i += 2
        elif line.startswith(quotes, i):
            text.append(" " * len(quotes))
            return "".join(text), i + len(quotes), None
        elif quotes == '"' and line.startswith("{{", i):
            text.append("  ")
            i += 2
        elif line.startswith("{" * braces, i):
            held, i = _hole(line, i + braces)
            text.append(" " * braces + held)
            closed = line[i : i + braces]
            text.append(" " * len(closed))
            i += len(closed)
        else:
            text.append(" ")
            i += 1
    return "".join(text), len(line), None if quotes == '"' and not verbatim else quotes


def _hole(line, position):
    """The code of a string's `{...}` from ``position`` up to the `}` that
    closes it, its literals blanked, and that brace's position, the end of
    ``line`` where it comes first."""
    held, depth, i = [], 0, position
    while i < len(line):
        match = _RE_LITERAL.search(line, i)
        stop = match.start() if match else len(line)
        for at in range(i, stop):
            depth += (line[at] == "{") - (line[at] == "}")
            if depth < 0:
                held.append(line[i:at])
                return "".join(held), at
        held.append(line[i:stop])
        if not match:
            break
        if _interpolated(line, match):
            blanked, i, _ = _interpolation(line, match)
        else:
            i = match.end()
            blanked = " " * (i - match.start())
        held.append(blanked)
    return "".join(held), len(line)


def _plain(code):
    """``code`` with the type arguments of every generic name, and the `?`
    or `!` before a `.`, turned to spaces: `Make<T>(x)?.Run()` reads as a
    call of `Make` and a call on what it returns."""
    previous = None
    while code != previous:
        previous = code
        code = _RE_TYPE_ARGUMENTS.sub(_blank, code)
    return re.sub(r"[?!](?=\.)", " ", code)


def _type_parameters(generic):
    """Names of the type parameters in ``generic``, a `<...>` list or None,
    in order."""
    listed = _strip_generics(generic[1:-1]) if generic else ""
    return tuple(re.findall(rf"({_NAME})\s*(?:,|$)", listed))


def _held_type(written, parameters):
    """The declared type ``written`` as the call resolver reads the type of
    a value: `System.Array<Type>` for an array, `System.ValueTuple<>` for a
    tuple, nothing for `dynamic`, a pointer or one of ``parameters``, the
    type parameters in scope."""
    written = re.sub(r",\s+", ",", (written or "").removeprefix("global::"))
    written = written.rstrip("?")
    if written.endswith("]"):
        return f"System.Array<{written[: written.index('[')].rstrip('?')}>"
    if written.startswith("("):
        return "System.ValueTuple<>"
    if written in parameters or written == "dynamic" or written.endswith("*"):
        return ""
    return written


def _parameter_counts(parameters):
    """``[[least, most, types]]``: the arguments that a call of a method
    with the parameter list ``parameters`` passes, ``most`` None for no
    limit, and the types of its parameters as written, which tell an
    override."""
    listed = [p for p in _flat(_strip_generics(parameters)).split(",") if p.strip()]
    optional = [p for p in listed if "=" in p or "params" in p.split()]
    most = None if any("params" in p.split() for p in listed) else len(listed)
    types = [
        " ".join(part.partition("=")[0].split()[:-1])
        for part in _bracketed_parts(parameters + ")")
    ]
    return [[len(listed) - len(optional), most, types]]


class DotnetParser(BaseParser):
    """C# parser (VB source contributes only file-level discovery)."""

    language = "dotnet"
    case_scopes_section = True

    def _member_id(self, parent_id, name):
        """ID for a member ``name`` of ``parent_id``: `<parent>::<name>`, or
        `<parent>::<name>#<n>` for the n-th overload of that name, a name no
        C# source can write."""
        member_id, count = f"{parent_id}::{name}", 1
        while member_id in self.builder.node_registry:
            count += 1
            member_id = f"{parent_id}::{name}#{count}"
        return member_id

    def _type_id(self, parent_id, name, arity, namespace=None):
        """ID for the type ``name`` with ``arity`` type parameters declared
        in ``parent_id`` and ``namespace``: that of its other declarations,
        another than that of a type with more or fewer, as `Pair<T>` is
        another type than `Pair<T, U>`, or of another namespace."""
        type_id = self._member_id(parent_id, name)
        return self._arities.setdefault((parent_id, namespace, name, arity), type_id)

    @staticmethod
    def _body_end(code_lines, opened):
        """Last line, 1-indexed, of the member whose header ends at
        ``opened`` (see ``_closing``): the `{` of its body, the `=` of its
        `=>` or of its initializer, or its `;`."""
        index, column, char = opened
        if char == ";":
            return index + 1
        if char == "{":
            closed = _closing(code_lines, index, column + 1)
            if closed is None:
                return len(code_lines)
            index, column = closed[:2]
            # A property may go on with an initializer: `{ get; } = value;`
            if not code_lines[index][column + 1 :].lstrip().startswith("="):
                return index + 1
        ended = _closing(code_lines, index, column + 1, ";")
        return ended[0] + 1 if ended else index + 1

    binding_re = re.compile(
        rf"\bvar\s+(?P<name>{_NAME})\s*=(?!=)\s*(?P<unwrap>await\b)?"
    )

    def _end_line(self):
        resource = _RE_RESOURCE.match(self._code)
        closed = resource and self._pairs.get(resource.end() - 1)
        if closed:
            # The resource is bound to what stands before the `)` that
            # closes the head, as a statement's value before its `;`.
            self._code = self._code[:closed].ljust(len(self._code))
        binding = self.binding_re.search(self._code)
        name = binding["name"] if binding else self._pending and self._pending[0]
        untyped = binding and (self._line, "?", False, binding.start("name"))
        # A `.Method(` that goes on a bound chain and is left open.
        opens = self._opens and self._code.lstrip().startswith(".")
        depth = len(self._open_calls)
        if super()._end_line() and untyped in self.declared[name].between(
            self._line, self._line
        ):
            # The call chain bound to the variable types it.
            self.declared[name].remove(untyped)
        if opens and self._pending and self._pending[2] is None:
            # The chain goes on behind the `)` that closes the call.
            self._pending = (*self._pending[:2], depth)

    def _scope_end(self, name, declaration):
        """As the base's, but a variable an expression declares, by `is` or
        as an `out` argument, is in scope to the end of the lambda body, of
        the loop, `using`, `lock`, `fixed`, `switch` or `catch` statement
        whose head declares it, else of the block around it."""
        text, _, _, spans = self._blocks
        at, head, held, written = self._declaration_site(name, declaration)
        before = text[head[0] : at] if head else written
        found = _RE_EXPRESSION_VARIABLE.search(before)
        if not found or (
            head
            and found[0].startswith("out")
            and not found[1]
            and _RE_AFTER_PARAMETERS.match(text, _innermost(spans, at)[1] + 1)
        ):
            return super()._scope_end(name, declaration)
        if "=>" in before:
            return self._statement_close(at - len(before) + before.rindex("=>") + 2)
        if head and _RE_SCOPING_HEAD.match(written):
            return self._statement_close(head[1] + 1)
        return held[1] if held else len(text)

    def _argument_origin(self, written, lineno, since):
        """The origin of what the argument ``written``, its literals'
        characters as `x`, passes: a literal's type, `null`, a variable's
        declared value, behind `ref`, `out` or `in` and the parameter's name
        as written; None for any other."""
        written = written.strip()
        if re.fullmatch(r'[$@]*"x*"', written):
            return "string()"
        if written == "null":
            return "null"
        if literal := re.fullmatch(r"(-\s*)?('x+'|true|false|[0-9][\w.]*)", written):
            kind = _literal_type(literal[2])
            kind = _negated_type(kind, literal[2]) if literal[1] else kind
            return f"{kind}()" if kind else None
        named = re.match(rf"({_NAME})\s*:(?!:)\s*", written)
        if named:
            held = self._argument_origin(written[named.end() :], lineno, since)
            return held and f"{named[1]}: {held}"
        passed = re.fullmatch(rf"(?:(ref|out|in)\s+)?({_NAME})", written)
        if passed:
            held = BaseParser.call_origin(self, f"{passed[2]}.", lineno, since)
            held = held and held.removesuffix(".")
            return f"{passed[1]} {held}" if held and passed[1] else held
        return None

    def call_origin(self, callee, lineno, since, owner=None):
        root, _, path = callee.partition(".")
        if not path:
            # A variable called by its name holds a delegate.
            return f"={callee}" if self._declared(root, lineno, since) else None
        if root == "this":
            return f"self.{path}"
        if root == "base":
            return f"super().{path}"
        held = super().call_origin(f"{root}.", lineno, since)
        # A name that is no variable is a member of an enclosing type, or a
        # type: the origin path is the callee itself. A member of `T?` is
        # read as one of `T`; only an overload tells the two apart.
        return f"{held.replace('?()', '()')}{path}" if held else callee

    def _scan_calls(self, source_id, code_lines, lines, index, start, since):
        """Record the calls on line ``index`` of ``code_lines``, plain code,
        from column ``start`` on, and the attributes read up to that line,
        as made by ``source_id``. ``since`` is the first line of the member
        the line is in."""
        lineno = index + 1
        pending = sorted({at for at, _, _ in self._attributes if at < index})
        for at in [*pending, index]:
            self.begin_line(code_lines[at], at + 1, raw=lines[at], since=since)
            for line, column, name in self._attributes:
                if line == at:
                    # `[Name]` applies the class NameAttribute.
                    if not name.endswith("Attribute"):
                        name += "Attribute"
                    origin = name.replace(".", "::")
                    self.add_call(source_id, name, at + 1, since, column, -1, origin)
        self._attributes = []
        code = code_lines[index]
        for untyped in _RE_UNTYPED.finditer(code, start):
            names = re.compile(_NAME).finditer(code, *untyped.span(untyped.lastindex))
            for name in names:
                self.declared.setdefault(name[0], []).append(
                    (lineno, "?", False, name.start())
                )
        for each in _RE_FOREACH_VAR.finditer(code, start):
            element = self.call_origin(f"{each[2]}.*", lineno, since)
            self.declared[each[1]][-1] = (lineno, element, False, each.start(1))
        for each in _RE_ELEMENT_VAR.finditer(code, start):
            # The variable holds an element of what it is read from, from the
            # statement's end on.
            entries = self.declared[each[1]]
            for entry in entries.between(lineno, lineno):
                if entry[1] == "?":
                    entries.remove(entry)
            element = self.call_origin(f"{each[2]}.*", lineno, since)
            entries.append((lineno, element, False, each.end() - 1))
        for created in _RE_CREATION.finditer(code):
            written = created[created.lastgroup]
            if created.end() > start and written not in _NOT_TYPES:
                self.add_call(
                    source_id,
                    written,
                    lineno,
                    since,
                    created.start(created.lastgroup),
                    created.end() - 1 if created["declared"] else -1,
                    written.replace(".", "::"),
                )
        for call_match in _RE_CALL.finditer(code, start):
            keyword = _RE_KEYWORD_BEFORE_MEMBER.match(call_match[1])
            first = call_match.start(1) + (keyword.end() if keyword else 0)
            callee = re.sub(r"\s", "", code[first : call_match.end(1)])
            if callee.rpartition(".")[2] in _DOTNET_CONTROL_KEYWORDS:
                continue
            before = code[:first]
            cast = _RE_CAST.search(before)
            created = re.search(r"\bnew\s+$", before)
            origin = None
            if created:
                # A constructor call is one of the type.
                origin = callee.replace(".", "::")
            elif cast:
                origin = f"{(cast[1] or cast[2]).replace('.', '::')}().{callee}"
            elif lines[index][: len(before)].rstrip().endswith('".'):
                origin = f"string().{callee}"
            elif kind := _literal_receiver(code_lines, lines, index, first):
                # `5.Val()`, `'c'.Ext()`, `true.Ext()`: the literal's own
                # type is the receiver's.
                origin = f"{kind}().{callee}"
            self.add_call(
                source_id,
                callee,
                lineno,
                since,
                call_match.end(1) - len(callee),
                call_match.end() - 1,
                origin,
            )
            generic = lines[index][call_match.end(1) :].lstrip().startswith("<")
            if generic and not created:
                # A method called with type arguments is `name<>`.
                self.builder.edges[-1]["target"] += "<>"
            closed = self._pairs.get(call_match.end() - 1)
            arguments = _flat(code[call_match.end() : closed or len(code)])
            passed = call_arguments(code_lines, lines, index, call_match.end())
            if passed is not None:
                self.builder.edges[-1]["metadata"]["args"] = len(passed)
                record_lambdas(self.builder.edges[-1]["metadata"], passed, "=>")
                # What each argument holds, for the overload that takes it
                written = call_arguments(
                    code_lines, lines, index, call_match.end(), quotes=True
                )
                site = self.builder.edges[-1]["metadata"].setdefault("site", {})
                site["arguments"] = [
                    self._argument_origin(a, lineno, since) for a in written
                ]
                # What a variable the call is made on holds, `T?` as such
                receiver = callee.rpartition(".")[0]
                if re.fullmatch(_NAME, receiver) and (
                    on := self._argument_origin(receiver, lineno, since)
                ):
                    site["on"] = on
            # `out var name` has the type of that `out` parameter of the
            # call it is an argument of: the result after those before it.
            passed = re.finditer(rf"\bout\s+(?:var\s+({_NAME}))?", arguments)
            for position, out in enumerate(passed, 1):
                if out[1]:
                    bound = f"#{len(self.builder.edges) - 1}.@{position}"
                    entries = self.declared[out[1]]
                    entries[-1] = (lineno, bound, False, entries[-1][3])

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        # VB files — scip-dotnet covers them; our regex parser only handles C#.
        if rel_path.endswith(".vb"):
            return

        lines = source_lines(content.removeprefix("\ufeff"))
        lines = self.split_source(
            lines,
            _code_lines(lines)[0],
            _RE_DEFINITION_HEAD,
            _RE_TYPE_HEAD,
            statement=_STATEMENT_START_RE,
        )
        skip_calls = is_generated_source(content)
        # Declarations and calls are read from `code_lines`, which hold no
        # comment and no text of a literal.
        code_lines, guarded = _code_lines(lines)

        namespace = next(
            (m[1] for m in map(_RE_NAMESPACE.match, code_lines[:60]) if m), None
        )
        if namespace:
            self.set_file_metadata(file_id, "dotnet_namespace", namespace)

        # A `global using` holds for the project of the file's directory;
        # each other `using`, an alias too, is recorded with the namespace
        # declaration it is written in, where it is resolved.
        directory = posixpath.dirname(rel_path)
        for code in code_lines:
            m = _RE_USING.match(code)
            if m:
                self.builder.add_edge(file_id, f"MODULE:{m['path']}", "IMPORTS")
                if m["global"] and m["static"]:
                    self.builder.dotnet_global_statics.setdefault(directory, []).append(
                        (m["path"], "", False, True, bool(m["rooted"]), None, None)
                    )
                elif m["global"]:
                    imports = self.builder.dotnet_global_usings.setdefault(
                        directory, {}
                    )
                    imports[m["alias"] or f"{m['path']}.*"] = m["path"]

        self.scan_declarations(
            [_RE_ARRAY_TYPE.sub(r"System.Array<\1>", code) for code in code_lines],
            _DECLARATIONS,
        )
        self.read_blocks(code_lines)
        plain_lines = [_plain(code) for code in code_lines]

        # The file and each type are the context of what is declared in
        # them; a member is the context, and the source, of the calls in it.
        self.enter_context(file_id, 0, len(lines))
        declaring = {file_id}
        since = {}  # member ID -> first line of the member it is, or is local to
        # ID -> the type parameters in scope that are a method's or a local
        # function's, of no type the resolver reads
        generics = {file_id: frozenset()}
        # (name, first line, last line) of each namespace declaration opened
        namespaces = []
        # Type ID -> (name, origin of its value) of each parameter of its
        # primary constructor
        captured = {}
        overloads = {}  # `<type ID>::<name>` -> metadata of each method of it
        # (line index, column, name) of each attribute read since the last
        # declaration, and the line the first is on
        self._attributes = []
        attributes_start = None
        self._arities = {}  # (parent ID, namespace, name, type parameter count) -> type ID
        # First and last line of the field initializer being read
        initializer = (0, 0)

        def declare(node_id, node_type, parent_id, index, end_lineno, **metadata):
            nonlocal attributes_start
            first = index if attributes_start is None else attributes_start
            attributes_start = None
            self.builder.add_node(
                node_id,
                node_type,
                lineno=index + 1,
                end_lineno=end_lineno,
                token_count=token_count(index + 1, end_lineno),
                docstring=self.get_leading_docstring(lines, first),
                **metadata,
            )
            self.builder.add_edge(parent_id, node_id, "DEFINES")

        def scan(source_id, index, start, first_line):
            if not skip_calls:
                self._scan_calls(
                    source_id, plain_lines, lines, index, start, first_line
                )

        def bind(type_id, first_line, written=None):
            """Declare, from the start of ``first_line`` on, the parameters
            of the primary constructor of ``type_id`` and, for an accessor
            of a property declared as ``written``, its `value`."""
            names = list(captured.get(type_id, ()))
            if written:
                held = re.fullmatch(_DECLARED_TYPE, written)
                held = (
                    f"{held['type']}{held['generic'] or ''}{held['nullable'] or ''}()"
                    if held
                    else "?"
                )
                names.append(("value", held.replace(".", "::")))
            for name, held in names:
                self.declared.setdefault(name, []).append((first_line, held, False, 0))

        i = 0
        n = len(lines)
        while i < n:
            code = code_lines[i]
            lineno = i + 1
            context_id, _ = self.context_at(lineno)

            if context_id not in declaring:
                # A statement of a member's body, or a local function.
                local = _RE_LOCAL_FUNCTION.match(code)
                close = None
                if (
                    local
                    and local["type"] not in _NOT_TYPES
                    and local["name"] not in _DOTNET_CONTROL_KEYWORDS
                ):
                    close = _closing(code_lines, i, local.end())
                opened = close and _closing(code_lines, close[0], close[1] + 1, "{;=")
                if opened and (
                    opened[2] == "{"
                    or code_lines[opened[0]].startswith("=>", opened[1])
                ):
                    function_id = self._member_id(context_id, local["name"])
                    end_lineno = self._body_end(code_lines, opened)
                    scope = generics[context_id] | set(
                        _type_parameters(local["generic"])
                    )
                    declare(
                        function_id,
                        "Function",
                        context_id,
                        i,
                        end_lineno,
                        name=local["name"],
                        return_type=_held_type(local["type"], scope),
                        static=False,
                        parameters=_parameter_counts(
                            _text(code_lines, i, local.end(), *close[:2])
                        ),
                    )
                    since[function_id] = since[context_id]
                    generics[function_id] = scope
                    self.enter_context(function_id, lineno, end_lineno)
                    scan(function_id, close[0], close[1] + 1, since[function_id])
                    i = close[0] + 1
                    continue
                scan(context_id, i, 0, since[context_id])
                i += 1
                continue

            if lineno <= initializer[1]:
                scan(context_id, i, 0, initializer[0])
                i += 1
                continue

            # Attribute sections, alone on their lines or before a declaration.
            while code.lstrip().startswith("["):
                close = _closing(code_lines, i, code.index("[") + 1)
                if close is None:
                    break
                if attributes_start is None:
                    attributes_start = i
                section = [code, *code_lines[i + 1 : close[0] + 1]]
                section[-1] = section[-1][: close[1] + 1]
                # The arguments of an attribute hold no attribute.
                flat, previous = "\n".join(section), None
                while flat != previous:
                    previous = flat
                    flat = re.sub(
                        r"\([^()]*\)", lambda m: re.sub(r"[^\n]", " ", m[0]), flat
                    )
                for at, text in enumerate(flat.split("\n"), i):
                    self._attributes += [
                        (at, m.start(1), m[1]) for m in _RE_ATTRIBUTE.finditer(text)
                    ]
                i = close[0]
                code = " " * (close[1] + 1) + code_lines[i][close[1] + 1 :]
            lineno = i + 1
            if not code.strip():
                i += 1
                continue

            m_namespace = context_id == file_id and _RE_NAMESPACE.match(code)
            if m_namespace:
                opened = _closing(code_lines, i, m_namespace.end(), "{;")
                last = n
                if opened and opened[2] == "{":
                    last = self._find_block_end(code_lines, *opened[:2])
                namespaces.append((m_namespace[1], lineno, last))
            m_using = context_id == file_id and _RE_USING.match(code)
            if m_using and not m_using["global"]:
                within = [d for d in namespaces if lineno <= d[2]]
                alias = m_using["alias"]
                if not within and not m_using["static"]:
                    self.builder.imports.setdefault(file_id, {})[
                        alias or f"{m_using['path']}.*"
                    ] = m_using["path"]
                self.builder.dotnet_usings.setdefault(file_id, []).append(
                    (
                        m_using["path"],
                        ".".join(name for name, _, _ in within),
                        i in guarded,
                        bool(m_using["static"]),
                        bool(m_using["rooted"]),
                        alias,
                        tuple(map(self.source_line, within[-1][1:]))
                        if within
                        else None,
                    )
                )

            m_type = _RE_CLASS.match(code)
            m_delegate = None if m_type else _RE_DELEGATE.match(code)
            operator = _RE_OPERATOR.match(code)
            member = _RE_MEMBER.match(code)
            owner = context_id.rpartition("::")[2]
            if m_type or m_delegate:
                declared = m_type or m_delegate
                opened = _closing(code_lines, i, declared.end(), "{;")
                opened = opened or (i, len(code), ";")
                generic = re.match(_GENERIC, code[declared.end() :])
                scope = _type_parameters(generic and generic[0])
                # What tells the type from others of its name: its
                # namespace, and for a partial type how many type parameters.
                metadata = {}
                within = [name for name, _, last in namespaces if lineno <= last]
                if context_id == file_id and within:
                    metadata["namespace"] = ".".join(within)
                # Its modifiers, those on lines of their own above included,
                # past comments, preprocessor lines and attribute lists
                head, above, open_lists = code[: declared.start(1)].split(), i - 1, 0
                while above >= 0:
                    line = code_lines[above]
                    # Read upwards, a `]` opens an attribute list, a `[` closes it.
                    open_lists += line.count("]") - line.count("[")
                    if open_lists > 0 or line.lstrip().startswith("["):
                        pass
                    elif re.fullmatch(rf"\s*(?:{_TYPE_MODIFIERS}\s*)+", line):
                        head += line.split()
                    elif not re.fullmatch(r"\s*(?:#.*)?", line):
                        break
                    above -= 1
                if "partial" in head:
                    metadata.update(partial=True, arity=len(scope))
                if "ref" in head:
                    # A `ref struct`, which no type argument can be
                    metadata["ref"] = True
            if m_type and opened[2] in "{;":
                kind, name = m_type.groups()
                type_id = self._type_id(
                    context_id, name, len(scope), metadata.get("namespace")
                )
                end_lineno = opened[0] + 1
                if opened[2] == "{":
                    end_lineno = self._find_block_end(code_lines, *opened[:2])
                declare(
                    type_id,
                    "Class",
                    context_id,
                    i,
                    end_lineno,
                    name=name,
                    kind=kind,
                    **metadata,
                )
                generics[type_id] = generics[context_id]
                primary = _RE_PRIMARY_CONSTRUCTOR.match(code, m_type.end())
                close = primary and _closing(code_lines, i, primary.end())
                if close:
                    declare(
                        f"{type_id}::<init>",
                        "Method",
                        type_id,
                        i,
                        close[0] + 1,
                        name="<init>",
                        is_constructor=True,
                    )
                    parameters = _text(code_lines, i, primary.end(), *close[:2])
                    parameters = list(_RE_PARAMETER.finditer(parameters))
                    # A member reads them as its own.
                    captured[type_id] = [
                        (p["name"], held)
                        for p in parameters
                        if (held := self._declared(p["name"], close[0] + 1, lineno))
                    ]
                    # The parameters of a record are its properties.
                    if "record" not in code[: m_type.end(2)].split():
                        parameters = []
                    for parameter in parameters:
                        declare(
                            f"{type_id}::{parameter['name']}",
                            "Constant",
                            type_id,
                            i,
                            close[0] + 1,
                            name=parameter["name"],
                            value_type=_held_type(parameter["type"], generics[type_id]),
                        )
                header = _text(code_lines, i, m_type.end(), *opened[:2])
                # A base is written with its type arguments, before the
                # constraints of the type parameters, which are none.
                bases = re.split(r"\bwhere\b", _flat(header))[0].partition(":")[2]
                for base in _bracketed_parts(bases) if kind != "enum" else ():
                    self.builder.add_edge(
                        type_id, f"REF:{' '.join(base.split())}", "INHERITS"
                    )
                bounds = {w["param"]: w["type"] for w in _RE_WHERE.finditer(header)}
                self.builder.nodes[-1]["metadata"].update(
                    _generic_metadata({name: bounds.get(name, "") for name in scope})
                )
                scan(type_id, i, len(code), lineno)
                if kind == "enum":
                    for at in range(opened[0], end_lineno):
                        members = code_lines[at]
                        if at == opened[0]:
                            members = members[opened[1] :]
                        for m in _RE_ENUM_MEMBER.finditer(members):
                            declare(
                                f"{type_id}::{m[1]}",
                                "Constant",
                                type_id,
                                at,
                                at + 1,
                                name=m[1],
                                value_type=name,
                            )
                    i = end_lineno - 1
                elif opened[2] == "{":
                    self.enter_context(type_id, lineno, end_lineno)
                    declaring.add(type_id)
                    # What follows the `{` on its line is read as a line.
                    i, column = opened[0], opened[1] + 1
                    code_lines[i] = " " * column + code_lines[i][column:]
                    self._attributes = []
                    continue
                else:
                    i = opened[0]
            elif m_delegate and opened[2] == ";":
                name = m_delegate[1]
                delegate_id = self._type_id(
                    context_id, name, len(scope), metadata.get("namespace")
                )
                paren = code.find("(", m_delegate.end())
                close = paren >= 0 and _closing(code_lines, i, paren + 1)
                listed = _text(code_lines, i, paren + 1, *close[:2]) if close else ""
                declare(
                    delegate_id,
                    "Class",
                    context_id,
                    i,
                    opened[0] + 1,
                    name=name,
                    kind="delegate",
                    parameters=_parameter_counts(listed),
                    **_generic_metadata(dict.fromkeys(scope, "")),
                    **metadata,
                )
                scan(delegate_id, i, len(code), lineno)
                i = opened[0]
            elif context_id == file_id or m_type or m_delegate:
                pass
            elif operator or (
                member
                and member["next"] == "("
                and (member["type"] or member["name"].lstrip("~") == owner)
            ):
                # A method, a constructor or an operator.
                scope, own = generics[context_id], []
                if operator:
                    name = f"operator {operator['symbol']}"
                    paren = operator.end() - 1
                    returns = operator["type"] or operator["symbol"]
                else:
                    name, returns = member["name"], member["type"]
                    paren = member.end() - 1
                    own = _type_parameters(member["generic"])
                    scope |= set(own)
                    if not member["type"] and name == owner:
                        static = "static" in member["modifiers"].split()
                        name = "<clinit>" if static else "<init>"
                close = _closing(code_lines, i, paren + 1)
                opened = close and _closing(code_lines, close[0], close[1] + 1, "{;=")
                if opened and opened[2] in "{;=":
                    method_id = self._member_id(context_id, name)
                    end_lineno = self._body_end(code_lines, opened)
                    listed = _text(code_lines, i, paren + 1, *close[:2])
                    read = list(_RE_PARAMETER.finditer(listed))
                    parameters = [
                        (p["modifiers"].split(), _held_type(p["type"], scope))
                        for p in read
                    ]
                    modifiers = (operator or member)["modifiers"].split()
                    extra = {
                        "static": "static" in modifiers,
                        # `private protected` is reachable from derived
                        # types, `protected internal` from anywhere.
                        "private": "private" in modifiers
                        and "protected" not in modifiers,
                        "protected": "protected" in modifiers
                        and "internal" not in modifiers,
                        "explicit": bool(member and member["explicit"]),
                        "parameters": _parameter_counts(listed),
                    }
                    if "override" in modifiers:
                        # Declared where its slot is, for overload resolution
                        extra["override"] = True
                    if len(read) == len(extra["parameters"][0][2]):
                        # Which a named argument is passed to
                        extra["parameter_names"] = [p["name"] for p in read]
                    if own:
                        extra["type_parameters"] = own
                        after = _text(code_lines, *close[:2], *opened[:2])
                        if re.search(r"\bwhere\b", after):
                            extra["constrained"] = True
                    if name in ("<init>", "<clinit>"):
                        extra["is_constructor"] = True
                    elif returns:
                        # What a call gives: its result, then what it
                        # assigns to each `out` argument.
                        out = [held for words, held in parameters if "out" in words]
                        extra["return_type"] = ", ".join(
                            [_held_type(returns, scope), *out]
                        )
                    if parameters and "this" in parameters[0][0]:
                        # An extension method, by the name of the type it
                        # is written for; `object` where for any.
                        written = parameters[0][1].partition("<")[0]
                        extra["extends"] = written.rpartition(".")[2] or "object"
                    declare(
                        method_id,
                        "Method",
                        context_id,
                        i,
                        end_lineno,
                        name=name,
                        **extra,
                    )
                    # Which of its overloads a call reaches is not always
                    # told, so what it returns is only where they all
                    # return the same.
                    group = overloads.setdefault(f"{context_id}::{name}", [])
                    group.append(self.builder.nodes[-1]["metadata"])
                    if len({m.get("return_type") for m in group}) > 1:
                        for metadata in group:
                            metadata["return_type"] = ""
                    since[method_id] = lineno
                    generics[method_id] = scope
                    bind(context_id, lineno)
                    self.enter_context(method_id, lineno, end_lineno)
                    scan(method_id, close[0], close[1] + 1, lineno)
                    i = close[0]
            elif member and member["type"]:
                # A property, an indexer, an event or a field.
                name, start = member["name"], member.end("name")
                if name == "this" and member["next"] == "[":
                    name, start = "this[]", code.index("]", member.end()) + 1
                opened = _closing(code_lines, i, start, "{;=")
                if opened and opened[2] in "{;=":
                    member_id = self._member_id(context_id, name)
                    end_lineno = self._body_end(code_lines, opened)
                    held = _held_type(member["type"], generics[context_id])
                    declare(
                        member_id,
                        "Constant",
                        context_id,
                        i,
                        end_lineno,
                        name=name,
                        value_type=held,
                    )
                    accessors = opened[2] == "{" or code_lines[opened[0]].startswith(
                        "=>", opened[1]
                    )
                    if accessors:
                        since[member_id] = lineno
                        generics[member_id] = generics[context_id]
                        bind(context_id, lineno, member["type"])
                        self.enter_context(member_id, lineno, end_lineno)
                        scan(member_id, opened[0], opened[1], lineno)
                    else:
                        initializer = (lineno, end_lineno)
                        bind(context_id, lineno)
                        scan(context_id, opened[0], opened[1], lineno)
                        # `int a = 1, b;` declares more than one.
                        flat = _text(code_lines, i, start, end_lineno - 1, None)
                        flat = _flat(_strip_generics(flat))
                        for more in re.findall(rf",\s*({_NAME})\s*(?=[=,;])", flat):
                            declare(
                                self._member_id(context_id, more),
                                "Constant",
                                context_id,
                                i,
                                end_lineno,
                                name=more,
                                value_type=held,
                            )
                    i = opened[0]
            elif "{" in code:
                # The block of something that is no declaration.
                i = self._find_block_end(code_lines, i) - 1
            self._attributes, attributes_start = [], None
            i += 1

        # A definition that sits inside an `#if` arm is built only in the
        # configurations that pick the arm: that is evidence about the call
        # sites that may reach it, as for C and C++.
        for node in self.builder.nodes[self._first_node :]:
            line = node["metadata"].get("lineno")
            if line and (arms := guarded.get(line - 1)):
                node["metadata"]["guards"] = sorted(arms)
        for edge in self.builder.edges[self._first_edge :]:
            line = edge.get("metadata", {}).get("lineno")
            if line and (arms := guarded.get(line - 1)):
                edge.setdefault("metadata", {})["guards"] = sorted(arms)
        self.restore_source_lines()
