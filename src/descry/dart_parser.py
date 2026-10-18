"""Dart regex-based baseline parser.

Covers the common case of Dart symbol extraction: library / part
directives, imports and exports (``package:`` / ``dart:`` / relative),
class / mixin / extension / enum / typedef declarations, functions,
methods, constructors, getters, setters, operators, fields, top-level
variables, and call sites. Mirrors JavaParser / GoParser in shape — walk
lines, give each definition the lines it spans, emit File →
Class/Function → Method → CALLS edges.

scip-dart augments resolution with type-aware cross-package
information when the binary is installed (see
``scip/adapters/dart.py``).
"""

from __future__ import annotations

import bisect
import re

from descry.generate import (
    _STATEMENT_START_RE,
    _TRAILING_PATH_RE,
    BaseParser,
    _bounded_parameters,
    _bracketed_parts,
    _generic_metadata,
    _innermost,
    _strip_generics,
    _strip_parens,
    blank_strings,
    is_generated_source,
    source_lines,
)

# import 'package:foo/bar.dart'; / export 'src/baz.dart'; / part 'foo.g.dart';
_RE_DIRECTIVE = re.compile(r"\s*(import|export|part)\s+(?!of\b)")
_RE_URI = re.compile(r"""['"]([^'"]+)['"]""")

# part of 'foo.dart'; / part of foo;
_RE_PART_OF = re.compile(r"""^\s*part\s+of\s+['"]?([^'";\s]+)""")

# library foo.bar;
_RE_LIBRARY = re.compile(r"^\s*library\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;")

# The patterns below read a declaration's header as ``_read_header`` returns
# it: what brackets enclose is `~`, so `<~*>` is a list of type arguments
# and `\(~*\)` one of parameters.

# @override / @Deprecated('...') / @Route.get('/path')
_RE_ANNOTATIONS = re.compile(r"(?:\s*@[\w$.]+(?:<~*>)?(?:\(~*\))?)*\s*")

# class Foo, mixin Foo, enum Foo, extension Foo on Bar, extension on Bar,
# extension type Foo(...), behind `abstract`, `base`, `sealed`, `final`,
# `interface` or `mixin`.
_RE_TYPE_DECL = re.compile(
    r"(?:(?:abstract|base|sealed|final|interface|mixin)\s+)*"
    r"(?P<kind>class|mixin|enum|extension\s+type|extension)\b\s*(?:const\s+)?"
    r"(?P<name>(?!on\b)[\w$]+)?(?P<generics><~*>)?"
)

# typedef Name<T> = ...; / typedef Returned Name(params);
_RE_TYPEDEF = re.compile(
    r"typedef\s+(?:.*\s)?(?P<name>[\w$]+)\s*(?P<generics><~*>)?\s*(?:\(~*\))?\s*"
)

# A type: `Foo`, `prefix.Foo<T>?`, a record `(int, String)`, or any of them
# before `Function(...)`.
_TYPE = (
    r"(?:Function(?:<~*>)?\(~*\)|[\w$.]+(?:<~*>)?|\(~*\))\??"
    r"(?:\s+Function(?:<~*>)?\(~*\)\??)*"
)

# A function, method, constructor, getter, setter, operator, field or
# variable: modifiers, a type, `get` or `set`, the name, type parameters,
# parameters, and for a field the names declared after the first.
_RE_MEMBER = re.compile(
    r"(?P<modifiers>(?:(?:external|static|abstract|covariant|late|final|const|var"
    r"|factory)\s+)*)"
    rf"(?:(?P<type>{_TYPE})\s+)??"
    r"(?:(?P<accessor>get|set)\s+)?"
    r"(?P<name>operator\s*[^\s\w$(]+|[\w$.]+)\s*"
    r"(?P<generics><~*>)?\s*(?P<params>\(~*\))?\s*(?:async\*?|sync\*)?\s*"
    r"(?P<more>(?:,\s*[\w$]+\s*)+)?"
)

# What a line that may declare a local function starts with: `Type name(`
# or `name(`.
_RE_LOCAL_FUNCTION = re.compile(r"\s*+(?:[\w$.<>?, ]+\s++)?[\w$]+\s*(?:<[^()=]*>)?\(")

# `=` where it is part of an operator's name: `operator ==`, `operator []=`
_RE_OPERATOR_NAME = re.compile(r"\boperator\s*[^\s\w$(]*$")

# What a `{` follows where it opens a map or set literal or the cases of a
# `switch` expression, not a body
_RE_VALUE_EXPECTED = re.compile(r"(?:[=(,:?>]|\bconst|\bswitch\s*\([^{}]*\))$")

# A declaration's type that ends its line, and the name the next line
# starts with: `late final Map<String, Entry>` / `entries;`
_RE_TYPE_ENDS = re.compile(r"[\w$>?]$")
_RE_NAME_STARTS = re.compile(r"\s*[\w$]+\s*(?:;|=(?![=>]))")

# The values of an enum: each name that starts the body or follows a `,`
_RE_ENUM_VALUE = re.compile(r"(?:^|,)\s*(?:@[\w$.]+(?:\(~*\))?\s*)*([\w$]+)")

# Call site: identifier(, or receiver.method(, or a?.method(
# Lookbehind guards against starting inside an identifier or a number.
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_$])"
    r"([A-Za-z_$][A-Za-z0-9_$]*(?:\??\.[A-Za-z_$][A-Za-z0-9_$]*)*)"
    r"\s*\("
)

# The type arguments of a call or of a named constructor: `Foo<T>(`,
# `Map<String, List<int>>(`, `Foo<T>.named(`
_RE_TYPE_ARGUMENTS = re.compile(
    r"(?<=[\w$])<[\w\s,.?$]*(?:<[\w\s,.?<>$]*>[\w\s,.?$]*)*>(?=\s*\(|\.[\w$])"
)

# A value that is one call, as ``_read_header`` returns it: `Type(~)`,
# `const Type<~>.named(~)`, `prefix.function(~)`
_RE_CALL_VALUE = re.compile(r"\s*(?:const\s+|new\s+)?([\w$.]+)(?:<~*>)?\(~*\)\s*")

# A value that is a literal, by the type it has: `<int>[`, `const {`, `1.5;`,
# `true;`, and with its text blanked a string
_RE_LITERAL = re.compile(
    r"\s*(?:const\s+)?(?:<[^()]*>\s*)?(?:(?P<List>\[)|(?P<Map>\{)"
    r"|(?P<num>\d[\w.]*\s*;)|(?P<bool>(?:true|false)\s*;)|(?P<String>;))"
)

# `name.path;`, a value that is a variable or a member read
_RE_COPIED = re.compile(r"\s*((?!null\b)[A-Za-z_$][\w$]*(?:\??\.[\w$]+)*)\s*;\s*")

# What follows the `)` of an object pattern in a `switch` expression
_RE_PATTERN_ENDS = re.compile(r"\s*(?:=>|when\b)")
# What an object pattern follows: `case Foo(`, `final Foo(:name) = value`,
# `case Foo() || Bar(`
_RE_PATTERN_STARTS = re.compile(
    r"\b(?:case|final|var)$|\bcase\b(?:(?!\bwhen\b).)*(?:\|\||&&)$"
)
# A field pattern of an object pattern: `name: Type(` of an object pattern
# in it, or a variable pattern without a type, `name: final x`, `:final x`
# or `:x`
_RE_FIELD_PATTERN = re.compile(
    r"\s*(?P<getter>[\w$]+)?\s*:\s*(?:(?P<object>(?:[a-z_$][\w$]*\.)?[_$]*[A-Z][\w$]*)"
    r"\s*\(|(?(getter)(?:var|final)\s+|(?:(?:var|final)\s+)?)(?P<name>[\w$]+)\s*[?!]?\s*$)"
)
# `for (var name in path)`, `for (final name in path)`
_RE_FOR_IN = re.compile(
    r"\bfor\s*\(\s*(?:var|final)\s+([\w$]+)\s+in\s+([\w$]+(?:\.[\w$]+)*)\s*\)"
)
# `switch (name) {`, a statement on a variable
_RE_SWITCH_ON = re.compile(r"\bswitch\s*\(\s*([a-z_$][\w$]*)\s*\)\s*\{\s*$")
# What a call that can only be a constructor's follows
_RE_CREATES = re.compile(r"\b(?:const|new)$")

# `(value as Type).` before a member: a cast gives the value that type
_RE_CAST_RECEIVER = re.compile(
    r"\bas\s+((?:[a-z_$][\w$]*\.)?_?[A-Z][\w$]*)(?:<[^()]*>)?\??\s*\)\s*\??\.$"
)

# The type a declaration gives a name: a class, possibly behind an import
# prefix, or a number or boolean
_DECLARED_TYPE = (
    r"(?P<type>(?:[a-z_$][\w$]*\.)?[_$]*[A-Z][\w$]*|int|double|num|bool)"
    r"(?P<generic><(?:[^<>;(){}=]|<(?:[^<>;(){}=]|<[^<>;(){}=]*>)*>)*>)?\??"
)

# The type a pattern tests for: an object pattern's, `Type(`, or a variable
# pattern's, `Type name`
_RE_PATTERN_TYPE = re.compile(
    rf"\s*(?:final\s+)?{_DECLARED_TYPE}\s*(?:\(|[a-z_$][\w$]*\s*(?::|$))"
)
_RE_CASE_STARTS = re.compile(r"\s*(?:case|default)\b")
# What ends an operand: an operator of lower precedence than `is`, a
# conditional's `?` or `:`, a separator or a bracket; `==` and the like
# are read past
_RE_OPERATOR = re.compile(r"&&|\|\||\?\?|=>|[=!<>]=|\?(?![.\[])|[=:,;()\[\]{}]")
# What may stand before an operand that starts a statement or a guard
_RE_OPERAND_HEAD = re.compile(r"(?s)\s*(?:(?:return|yield)\s+|.*\bwhen\s+)?")
# What ends the expression of an arm of a `switch` expression
_ARM_ENDS = ("", ",", ";", ")", "]", "}")

_DECLARATIONS = (
    # `Type name` before `=`, `;`, `,`, `:`, a closing bracket, `in` or
    # `when`: a local, field, parameter, loop or pattern variable
    re.compile(
        rf"(?<![\w$.])(?<!Function\(){_DECLARED_TYPE}\s+(?P<name>[a-z_$][\w$]*)\s*"
        r"(?=[=;,:)\]}]|\b(?:in|when)\b)"
    ),
    # `name is Type` promotes the name
    re.compile(
        rf"(?<![\w$.])(?P<name>[a-z_$][\w$]*)\s+is(?P<negated>!)?\s+{_DECLARED_TYPE}"
    ),
    # `on Type catch (name`
    re.compile(rf"\bon\s+{_DECLARED_TYPE}\s+catch\s*\(\s*(?P<name>[\w$]+)"),
    # `name = value as Type;`
    re.compile(rf"(?<![\w$.])(?P<name>[\w$]+)\s*=[^=;]*\bas\s+{_DECLARED_TYPE}\s*;"),
    # A type parameter's bound: `<T extends Type`
    re.compile(r"[<,]\s*(?P<param>[\w$]+)\s+extends\s+(?P<type>[\w$.]+)"),
)

# Names a line declares without a type or a value: `var name;`, the
# variable of `for (final name in`, a pattern's `var name`, `:name`,
# `:var name` and `:final name`, those of a `catch`, and one that holds a
# function: `Function(int) name`
_RE_UNTYPED = re.compile(
    r"\b(?:var|final|const|late)\s+([\w$]+)\s*(?=[;,):\]}?]|\s+in\b)"
    r"|\bcatch\s*\(\s*([\w$]+)(?:\s*,\s*([\w$]+))?"
    r"|[(,]\s*:\s*(?:(?:var|final)\s+)?([\w$]+)(?=\s*[,)?!]|\s+as\b)"
    r"|\bFunction\([^()]*\)\??\s+([\w$]+)\s*(?=[=;,)])"
)
# `var (a, b) = pair;` / `final [first, second] = list;`
_RE_DESTRUCTURED = re.compile(r"\b(?:var|final)\s*[(\[]([^()\[\]]*)[)\]]\s*=")
# The parameter list of a function expression: `(a, b) =>`, `(a) async {`
_RE_CLOSURE = re.compile(
    r"(?:^|[(,=:?\[{>]|\breturn)\s*\(([^()]*)\)\s*(?:async\*?\s*|sync\*\s*)?(?:=>|\{)",
    re.MULTILINE,
)

# Dart control-flow / keyword tokens that the call regex would otherwise
# capture. `new` is deprecated but still legal; `throw` and `assert` look
# like calls syntactically. `factory`, `late`, and type modifiers appear
# at declaration sites where the next `(` is part of a constructor, and
# `Function` where it is part of a function type.
_DART_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "return",
        "yield",
        "await",
        "async",
        "sync",
        "try",
        "catch",
        "throw",
        "rethrow",
        "assert",
        "new",
        "const",
        "final",
        "late",
        "factory",
        "this",
        "super",
        "is",
        "as",
        "in",
        "of",
        "with",
        "on",
        "void",
        "null",
        "true",
        "false",
        "break",
        "continue",
        "default",
        "print",
        "Function",
    }
)

# Reserved words a statement starts with or is named by; no declaration has
# one for its name or its type.
_STATEMENT_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "try",
        "catch",
        "return",
        "throw",
        "rethrow",
        "yield",
        "await",
        "assert",
        "new",
        "in",
        "is",
    }
)


# What a Dart type's body opens after, and a declaration a line may go on
# to after the end of another
_RE_TYPE_HEAD = re.compile(
    r"\s*(?:(?:abstract|base|final|interface|sealed|mixin)\s+)*"
    r"(?:class|mixin|enum|extension)\b"
)
_RE_DEFINITION_HEAD = re.compile(
    r"(?:(?:abstract|base|final|interface|sealed)\s+)*"
    r"(?:class|mixin|enum|extension|typedef)\b"
)


# What opens code in a string: `${`
_HOLES = dict.fromkeys(("'", '"', '"""'), re.compile(r"\$\{"))


def _code_lines(lines):
    """Each of ``lines`` without its comments and with its string literals
    blanked but for the code in them, in turn; `a!.b` reads as `a?.b`."""
    codes = []
    inside = None
    for line in lines:
        code, inside = blank_strings(
            line.replace("'''", '"""'), inside, ('"""', "'"), _HOLES
        )
        # Only a triple-quoted string goes on over the next line.
        inside = None if inside == "'" else inside
        codes.append(code.replace("!.", "?."))
    return codes


def _read_header(lines, index, column=0):
    """Read the declaration that starts at ``column`` of ``lines[index]``,
    lines without comments or literals, up to what ends its header: the `{`
    of a body, an `=`, the `:` of an initializer list, a `;`, or the bracket
    that closes what the declaration is in. Returns the header in one line,
    with what its brackets enclose turned to `~`, the offset in it at which
    each of its lines starts, and the line index and column of its end."""
    text = " " * column
    starts = []
    depth = angle = 0
    for at in range(index, len(lines)):
        line = lines[at]
        starts.append(0 if at == index else len(text))
        for col in range(column if at == index else 0, len(line)):
            char = line[col]
            if depth:
                depth += (char in "([{") - (char in ")]}")
                text += "~" if depth or angle else char
            elif char in "([" or (angle and char == "{"):
                depth = 1
                text += "~" if angle else char
            elif (
                char == ";"
                or (not angle and char in ")]}{:")
                or (char == "=" and not _RE_OPERATOR_NAME.search(text))
            ):
                return text, starts, at, col
            elif angle:
                angle += (char == "<") - (char == ">")
                text += "~" if angle else char
            else:
                # `Type<` opens type arguments; `a < b` compares.
                angle = int(char == "<" and (line[col - 1 : col] or " ") not in " <")
                text += char
        text += "~" if depth or angle else " "
    return text, starts, len(lines) - 1, len(lines[-1])


def _locate(starts, offset):
    """``(line, column)`` of ``offset`` in a header whose lines start at the
    offsets ``starts``, the line counted from the header's first."""
    line = bisect.bisect_right(starts, offset) - 1
    return line, offset - starts[line]


# A clause of a type's heritage, or a base it lists with its type arguments
_RE_BASE_WORD = re.compile(
    r"\b(?P<clause>extends|with|implements|on)\b|(?P<base>[\w$.]+(?:<~*>)?)"
)


def _type_parameters(generics: str) -> list[str]:
    """Names of the type parameters in `<T, U extends V<T>>`."""
    declared = _strip_generics(generics[1:-1]).split(",")
    return [param.split()[0] for param in declared if param.split()]


def _parameter_names(parameters: str) -> list[str]:
    """Names of the parameters in the text of a parameter list, but for
    those that initialise a field: `this.name`, `super.name`."""
    flat = _strip_parens(_strip_generics(parameters))
    names = []
    for item in re.sub(r"[{}\[\]]", " ", flat).split(","):
        named = re.search(
            r"(?<![\w$.])([A-Za-z_$][\w$]*)$", item.partition("=")[0].strip()
        )
        if named:
            names.append(named[1])
    return names


def _case_type(head):
    """The origin path of the type every alternative of the pattern in the
    case head ``head`` tests for, where they agree; None for any other."""
    pattern = re.match(r"\s*case\b((?:(?!\bwhen\b).)*)", head)
    types = set()
    for alternative in pattern[1].split("||") if pattern else [""]:
        tested = _RE_PATTERN_TYPE.match(alternative)
        held = tested and tested["type"].replace(".", "::")
        types.add(tested and f"{held}{tested['generic'] or ''}()")
    return types.pop() if len(types) == 1 else None


class DartParser(BaseParser):
    """Regex-driven Dart source parser."""

    language = "dart"
    head_scopes_else = False
    case_scopes_section = True
    binding_re = re.compile(
        r"\b(?:final|var|const)\s++(?:[\w$.<>?, ]+\s++)?(?P<name>[\w$]+)\s*"
        r"=(?![=>])\s*(?P<unwrap>await\b)?"
    )

    def __init__(self, builder):
        super().__init__(builder)
        self._type_params = set()  # names the file declares as type parameters
        self._since = 0  # first line of what the line being read belongs to
        # edge index of a cascaded call -> origin path of the cascade's target
        self._targets = {}
        # (name, brackets left open, awaited) of a binding whose value goes
        # on over the next lines
        self._unbound = None

    def _end_line(self):
        binding = self.binding_re.search(self._code)
        name = binding["name"] if binding else self._pending and self._pending[0]
        known = len(self.declared.get(name, ()))
        line, code, since = self._line, self._code, self._since
        super()._end_line()
        if len(self.declared.get(name, ())) > known:
            # A cascade has the value of its target.
            at, origin, top = self.declared[name][-1]
            target = origin[1:].isdigit() and self._targets.get(int(origin[1:]))
            if target:
                self.declared[name][-1] = (at, target, top)
        elif binding:
            # A copy has the type of what it copies, a literal its own; any
            # other value that is no call leaves the name without a known
            # type.
            copied = _RE_COPIED.fullmatch(code, binding.end())
            literal = _RE_LITERAL.match(code, binding.end())
            origin = "?"
            if literal and code.rstrip().endswith(("];", "};", literal[0])):
                origin = f"{literal.lastgroup}<>()"
            elif copied:
                path = copied[1].replace("?", "")
                origin = self._path_origin(f"{path}.", line, since)[:-1]
                if binding["unwrap"]:
                    origin += ".?"
            self.declared.setdefault(name, []).append((line, origin, False))
        statement = code.rstrip()
        if binding:
            self._unbound = (name, 0, bool(binding["unwrap"]))
            code = code[binding.end() :]
        if self._unbound:
            # The value is what the call that ends the statement returns,
            # where the last line goes on with or closes a call.
            name, depth, awaited = self._unbound
            depth += len(re.findall(r"[(\[{]", code)) - len(re.findall(r"[)\]}]", code))
            self._unbound = (name, depth, awaited)
            if statement.endswith(";") and depth <= 0:
                self._unbound = None
                if not binding:
                    edge = self._closes.get(len(statement[:-1].rstrip()) - 1)
                    chained = statement.lstrip()[0] in ".)}" and edge is not None
                    origin = self._targets.get(edge, f"#{edge}") if chained else "?"
                    self.declared.setdefault(name, []).append(
                        (line, origin + ".?" * (chained and awaited), False)
                    )

    def call_origin(self, callee, lineno, since, owner=None):
        root, dot, rest = callee.partition(".")
        held = dot and super().call_origin(f"{root}.", lineno, since)
        if not held:
            return None
        # A type parameter without a bound says nothing of the value.
        unbound = held.removesuffix("().") in self._type_params
        return ("?." if unbound else held) + rest

    def _path_origin(self, path, lineno, since):
        """Origin path of a call of ``path``, a dotted name: one through
        `this`, `super` or what its first name is declared as, else the path
        itself, whose first name is then looked up in scope."""
        root, _, rest = path.partition(".")
        if root == "this":
            return f"self.{rest}"
        if root == "super":
            return f"super().{rest}"
        return self.call_origin(path, lineno, since) or path

    def _target_origin(self, target, lineno, since):
        """Origin path of the value a cascade is made on: ``target``, the
        code before its `..`, or what the previous line ends with."""
        edge = self._closes.get(len(target) - 1) if target else self._tail
        if edge is not None:
            return self._targets.get(edge, f"#{edge}")
        named = _TRAILING_PATH_RE.search(target) if target else None
        name = named.group() if named else not target and self._tail_name
        return self._path_origin(f"{name}.", lineno, since)[:-1] if name else "?"

    def _bind_locals(self, code, lineno, scope):
        """Record the variables and parameters line ``lineno`` declares in
        ``scope``, those without a type as of no known one."""
        untyped = list(filter(None, sum(_RE_UNTYPED.findall(code), ())))
        closures = self._closure_params.get(lineno, ())
        for names in (*_RE_DESTRUCTURED.findall(code), *closures):
            untyped += _parameter_names(names)
        held_by = {}  # name -> what a pattern or a loop gives it
        for found in _RE_CALL.finditer(code):
            if self._object_pattern(code, found.start(1), found.end() - 1):
                self._bind_fields(found[1], code[found.end() :], held_by)
        for name, path in _RE_FOR_IN.findall(code):
            # An element of what the loop iterates
            held_by[name] = self.call_origin(f"{path}.*", lineno, self._since)
        for name in untyped:
            held = held_by.get(name) or "?"
            self.declared.setdefault(name, []).append((lineno, held, False))
        if scope:
            typed = (match["name"] for match in _DECLARATIONS[0].finditer(code))
            for name in (*untyped, *typed):
                self.add_local(scope, name)

    def _object_pattern(self, code, start, paren):
        """Whether ``code`` has an object pattern whose type starts at
        column ``start`` and whose `(` is at ``paren``."""
        closed = self._pairs.get(paren)
        return bool(
            _RE_PATTERN_STARTS.search(code[:start].rstrip())
            or (closed and _RE_PATTERN_ENDS.match(code, closed + 1))
        )

    def _bind_fields(self, pattern, inside, fields):
        """Add to ``fields`` the origin path of each variable that a field
        pattern of the object pattern of type ``pattern`` declares without a
        type, ``inside`` its code after the `(`: the getter it matches,
        `Type().getter`. A name two patterns bind is of no known type."""
        held = pattern.replace(".", "::")
        for part in _bracketed_parts(inside):
            field = _RE_FIELD_PATTERN.match(part)
            if field and field["object"]:
                self._bind_fields(field["object"], part[field.end() :], fields)
            elif field:
                name = field["name"]
                origin = f"{held}().{field['getter'] or name}"
                fields[name] = origin if fields.get(name, origin) == origin else "?"

    def _statement_end(self, lines, index, column, written=None):
        """Index of the line that ends the statement going on at ``column``
        of ``lines[index]``, and whether a `;` ends it there rather than the
        bracket that closes what it is in. ``written``, the lines with their
        literals, is given for a constructor's initializer list, which the
        block of its body ends too."""
        depth = 0
        for at in range(index, len(lines)):
            line = lines[at]
            for col in range(column if at == index else 0, len(line)):
                char = line[col]
                if (
                    written
                    and char == "{"
                    and not depth
                    and not _RE_VALUE_EXPECTED.search(written[at][:col].rstrip())
                ):
                    return self._find_block_end(lines, at, col) - 1, True
                depth += (char in "([{") - (char in ")]}")
                if depth < 0 or (char == ";" and not depth):
                    return at, char == ";"
        return len(lines) - 1, False

    def _scope_spans(self, line, name, held):
        """As the base's, but the declaration an `is` test on line ``line``
        makes is in scope only where the test promotes ``name``."""
        spans = self._promoted(line, name, held)
        return super()._scope_spans(line, name, held) if spans is None else spans

    def _operator(self, at, back=False):
        """``(token, offset)`` of what ends the operand going on at offset
        ``at`` (see `_RE_OPERATOR`), forward or ``back``, past brackets;
        an empty token at either end of the code."""
        text, _, closer, _ = self._blocks
        if not back:
            while found := _RE_OPERATOR.search(text, at):
                token, at = found[0], found.end()
                if token in "([{":
                    at = closer.get(found.start(), len(text)) + 1
                elif not re.fullmatch("[=!<>]=", token):
                    return token, found.start()
            return "", len(text)
        while at > 0:
            at -= 1
            if text[at] in ")]" and at in self._openers:
                at = self._openers[at]
            elif text[at - 1 : at + 1] in ("&&", "||", "??", "=>"):
                return text[at - 1 : at + 1], at - 1
            elif text[at - 1 : at + 1] in ("==", "!=", "<=", ">="):
                at -= 1
            elif text[at] in "?:,;=([{}" and text[at : at + 2] not in ("?.", "?["):
                return text[at], at
        return "", 0

    def _chain(self, start, end, joiner):
        """The operands that ``joiner``, `&&` or `||`, joins to the one at
        offsets ``start`` to ``end``: ``(start, end, after)`` of all of
        them, ``after`` the offset of the first joiner after the operand;
        None where it joins none."""
        token, at = self._operator(end)
        after = at if token == joiner else None
        while token == joiner:
            token, at = self._operator(at + 2)
        right = at if after is not None else end
        token, at = self._operator(start, back=True)
        joined = token == joiner
        while token == joiner:
            token, at = self._operator(at, back=True)
        left = at + len(token) if joined else start
        return (left, right, after) if joined or after is not None else None

    def _branches(self, question):
        """``(colon, end)``: the offsets of the `:` of the conditional whose
        `?` is at offset ``question`` and of the end of its else branch;
        None where it has no `:`."""
        depth, at, colon = 0, question + 1, None
        while True:
            token, at = self._operator(at)
            if token == "?":
                depth += 1
            elif token == ":" and depth:
                depth -= 1
            elif token == ":" and colon is None:
                colon = at
            elif token in ("", ",", ";", ")", "]", "}", ":"):
                return (colon, at) if colon is not None else None
            at += len(token)

    def _promoted(self, line, name, held):
        """The spans of the code where the `is` test of ``name`` on line
        ``line`` promotes it to ``held``, by Dart's flow analysis: the
        operands after it that an `&&` joins it to, `||` where it is
        negated, the branch of a conditional, an `if` or a `while` it is
        true in, and past an `if` whose branch it is false in exits. None
        where the line tests no `is` of the name for ``held``; no span where
        it tests more than one."""
        text, starts, closer, spans = self._blocks
        tests = [
            found
            for found in _DECLARATIONS[1].finditer(text, starts[line - 1], starts[line])
            if found["name"] == name
        ]
        if held not in {f"{self._scanned_type(t.groupdict())}()" for t in tests}:
            return None
        start, end = tests[0].span()
        left, opened = self._operator(start, back=True)
        right, closed = self._operator(end)
        # An operand of no operator binding closer than `&&`, `||` or `?`
        before = text[opened + len(left) : start]
        if (
            len(tests) > 1
            or text[end:closed].strip()
            or not _RE_OPERAND_HEAD.fullmatch(before)
        ):
            return []
        promoted, out = not tests[0]["negated"], []
        while True:
            for joiner, keeps in (("&&", True), ("||", False)):
                chained = self._chain(start, end, joiner)
                if not chained:
                    continue
                if promoted != keeps:
                    return out
                start, end, after = chained
                if after is not None:
                    out.append((after, end))
            left, opened = self._operator(start, back=True)
            right, closed = self._operator(end)
            if re.match(r"(?s).*\bwhen\s", text[opened + len(left) : start]):
                # A `when` guard promotes in the body of its case, or in
                # the expression of its arm of a `switch` expression.
                if promoted and right == ":":
                    block = _innermost(spans, closed)
                    out.append((closed, self._section_end(closed, block)))
                elif promoted and right == "=>":
                    arm = closed + 2
                    while (found := self._operator(arm))[0] not in _ARM_ENDS:
                        arm = found[1] + len(found[0])
                    out.append((closed, found[1]))
                return out
            if right == "?":
                branches = self._branches(closed)
                if branches:
                    colon, last = branches
                    out.append((closed, colon) if promoted else (colon, last))
                return out
            if left != "(" or right != ")" or closer.get(opened) != closed:
                return out
            head = text[:opened].rstrip()
            keyword = re.search(r"(?<![\w$.])(?:if|while)$", head)
            if re.search(r"(?<![\w$)\]])!$", head):
                start, end, promoted = len(head) - 1, closed + 1, not promoted
            elif keyword:
                break
            elif re.search(r"[\w$)\]>]$", head):
                return out
            else:
                start, end = opened, closed + 1
        if keyword[0] == "if":
            return out + self._branch_scope(closed, _innermost(spans, opened), promoted)
        if promoted:
            out.append((closed + 1, self._statement_close(closed + 1, orelse=False)))
        return out

    def _switch_cases(self, lines, index):
        """Where line ``index`` opens a `switch` statement on a variable Dart
        promotes, a local one or a private final field, declare it in each
        case body as the type that every pattern of the case heads sharing
        the body tests it for, and as it was declared before where they do
        not agree and from where the body assigns it."""
        found = _RE_SWITCH_ON.search(lines[index])
        if not found:
            return
        name, line = found[1], index + 1
        declared = [d for d in self.declared.get(name, ()) if d[0] <= line]
        promotable = [
            d
            for d in declared
            if d[0] >= self._since and self._in_scope(name, d[0], line, d[1])
        ] or [
            d
            for d in declared
            if name[0] == "_" and re.search(r"\bfinal\b", lines[d[0] - 1])
        ]
        if not promotable:
            return
        before = max(promotable, key=lambda d: d[0])[1:]
        _, starts, _, spans = self._blocks
        block = _innermost(spans, starts[line])
        assigns = re.compile(rf"(?<![\w$.]){re.escape(name)}\s*=(?![=>])")
        entries = self.declared[name]
        shared = []  # (line, type tested) of the heads of the body to come
        last, depth = self._find_block_end(lines, index) - 1, 0
        for at in range(line, last):
            code = lines[at]
            head = depth == 0 and _RE_CASE_STARTS.match(code)
            if head:
                shared.append((at + 1, _case_type(code)))
            if shared and code.strip() and not (head and code.rstrip()[-1] == ":"):
                # The body starts: the heads promote where they agree.
                types = {held for _, held in shared}
                held = types.pop() if len(types) == 1 else None
                entries.append((shared[-1][0], *((held, False) if held else before)))
                shared = []
            for assigned in [] if head else assigns.finditer(code):
                entries.append((at + 1, *before))
                # Out of a block in the body the variable stays as assigned.
                around, outer = _innermost(spans, starts[at] + assigned.start()), None
                while around and around != block:
                    around, outer = _innermost(spans, around[0]), around
                if outer:
                    end = self._statement_close(outer[0])
                    entries.append((bisect.bisect_right(starts, end), *before))
            depth += code.count("{") - code.count("}")

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        raw_lines = source_lines(content)
        raw_lines = self.split_source(
            raw_lines,
            _code_lines(raw_lines),
            _RE_DEFINITION_HEAD,
            _RE_TYPE_HEAD,
            statement=_STATEMENT_START_RE,
        )
        skip_calls = is_generated_source(content)
        lines = _code_lines(raw_lines)

        # A name is declared on its own line with the type the line before
        # ends with.
        joined = list(lines)
        for index in range(1, len(lines)):
            if _RE_NAME_STARTS.match(lines[index]) and _RE_TYPE_ENDS.search(
                lines[index - 1]
            ):
                joined[index] = f"{lines[index - 1].rstrip()} {lines[index].lstrip()}"
        self.scan_declarations(joined, _DECLARATIONS)
        self.read_blocks(lines)
        self._openers = {end: start for start, end in self._blocks[2].items()}
        # line -> the parameter lists of the function expressions opening there
        text, starts = self._blocks[:2]
        blanked = _RE_TYPE_ARGUMENTS.sub(lambda match: " " * len(match[0]), text)
        self._closure_params = {}
        for found in _RE_CLOSURE.finditer(blanked):
            line = bisect.bisect_right(starts, found.start(1))
            self._closure_params.setdefault(line, []).append(found[1])
        self.enter_context(file_id, 0, len(lines))
        types = {}  # ID of each class, mixin, extension and enum -> its name
        constructors = set()  # IDs of the named constructors
        # ID of each other member -> the ID it has beside such a constructor
        renamed = {}
        declared_at = {}  # line index -> columns of the names declared on it
        # Index of the first line that is not part of the header or the
        # value of the declaration read last, and that declaration's line.
        resume = statement = 0

        def define(node_type, parent_id, node_id, name, first, at, last, **metadata):
            """Add the node of the definition ``name`` that goes from line
            index ``first`` to ``last`` and is named on line index ``at``."""
            if node_id not in self.builder.node_registry:
                self.builder.add_node(
                    node_id,
                    node_type,
                    name=name,
                    lineno=at + 1,
                    end_lineno=last + 1,
                    token_count=(last - first + 1) * 10,
                    docstring=self.get_leading_docstring(raw_lines, first),
                    **metadata,
                )
                self.builder.add_edge(parent_id, node_id, "DEFINES")

        def declare_type(index, header, source, starts, at, col, begin):
            """Read the class, mixin, extension, enum or typedef whose
            ``header``, ``source`` as written, starts on line ``index`` and
            ends at ``col`` of line ``at``. Returns the line index to go on
            from, None where the header declares none."""
            match = _RE_TYPEDEF.fullmatch(header, begin)
            kind, heritage, aligned = "typedef", "", None
            if not match:
                match = _RE_TYPE_DECL.match(header, begin)
                if not match:
                    return None
                kind = " ".join(match["kind"].split())
                # Read where brackets hide their commas, written from source.
                heritage = re.sub(
                    r"\(~*\)", lambda m: " " * len(m[0]), header[match.end() :]
                )
                aligned = source[match.end() :]
            opened = lines[at][col : col + 1]
            if opened == "=":
                # `class Name = Base with Mixin;`, `typedef Name = Type;`
                heritage = "extends " + re.sub(
                    r"<~*>", "", _read_header(lines, at, col + 1)[0]
                )
                aligned = None
            if kind == "typedef" and (opened != "=" or "Function" in heritage):
                # A function type has the members of `Function`.
                heritage, aligned = "extends Function", None
            bases, clause = {}, None
            for word in _RE_BASE_WORD.finditer(heritage):
                if word["clause"]:
                    clause = word["clause"]
                    bases.setdefault(clause, [])
                elif clause:
                    base = (
                        aligned[slice(*word.span("base"))] if aligned else word["base"]
                    )
                    bases.setdefault(clause, []).append(base)
            if not match["name"] and "on" not in bases:
                return None
            # An extension without a name is known by the type it is on.
            written = " ".join(source[match.end() : len(header)].split())
            name = match["name"] or f"extension {written}"
            type_id = f"{file_id}::{name}"
            if opened == "{":
                last = self._find_block_end(lines, at, col) - 1
            else:
                last = self._statement_end(lines, at, col)[0]
            named = match.start("name") if match["name"] else begin
            named_at, column = _locate(starts, named)
            named_at += index
            written = match["generics"] and source[slice(*match.span("generics"))]
            define(
                "Class",
                file_id,
                type_id,
                name,
                index,
                named_at,
                last,
                kind=kind,
                **_generic_metadata(_bounded_parameters(written or "")),
            )
            # A mixin overrides the superclass and the mixins before it.
            # A mixin's `on` types are what a class it is applied to extends.
            mixin = kind == "mixin"
            for base in (
                *reversed(bases.get("with", ())),
                *bases.get("extends", ()),
                *bases.get("implements", ()),
                *(() if mixin else bases.get("on", ())),
            ):
                self.builder.add_edge(type_id, f"REF:{base}", "INHERITS")
            for base in bases.get("on", ()) if mixin else ():
                self.builder.add_edge(
                    type_id, f"REF:{base}", "INHERITS", {"mixin": "on"}
                )
            if kind in ("typedef", "extension type") and match["name"]:
                declared_at.setdefault(named_at, set()).add(column)
            if opened != "{":
                return last + 1
            types[type_id] = name
            self.enter_context(type_id, index + 1, last + 1)
            if kind == "extension type":
                # `extension type Name(Type name)` declares the field it wraps.
                held = re.match(
                    r"[^(]*\(\s*(?:final\s+)?(.*\S)\s+([\w$]+)\s*\)",
                    source[match.end("name") :],
                )
                if held:
                    self.builder.add_node(
                        f"{type_id}::{held[2]}",
                        "Constant",
                        name=held[2],
                        lineno=named_at + 1,
                        docstring="",
                        value_type=held[1],
                    )
                    self.builder.add_edge(type_id, f"{type_id}::{held[2]}", "DEFINES")
            if kind == "enum":
                values, value_starts, ended_at, _ = _read_header(lines, at, col + 1)
                for value in _RE_ENUM_VALUE.finditer(values):
                    line, column = _locate(value_starts, value.start(1))
                    value_id = f"{type_id}::{value[1]}"
                    self.builder.add_node(
                        value_id,
                        "Constant",
                        name=value[1],
                        lineno=at + line + 1,
                        docstring="",
                        value_type=name,
                    )
                    self.builder.add_edge(type_id, value_id, "DEFINES")
                    declared_at.setdefault(at + line, set()).add(column)
                return ended_at + 1
            return at + 1

        def declare_member(index, header, source, starts, at, col, begin, parent_id):
            """Read the function, method, constructor, accessor, operator,
            field or variable whose ``header``, ``source`` as written,
            starts on line ``index`` and ends at ``col`` of line ``at``, as
            a member of ``parent_id``. Returns the line index to go on from,
            None where the header declares none."""
            match = _RE_MEMBER.fullmatch(header, begin)
            if not match:
                return None
            name, accessor = match["name"], match["accessor"]
            ended = lines[at][col : col + 1]
            arrow = lines[at][col : col + 2] == "=>"
            local = parent_id != file_id and parent_id not in types
            owner = types.get(parent_id)
            if match["generics"]:
                self._type_params.update(
                    _type_parameters(source[slice(*match.span("generics"))])
                )
            declared_type = match["type"] and " ".join(
                source[slice(*match.span("type"))].split()
            )
            if declared_type and (
                re.match(r"[\w$]*", declared_type)[0] in self._type_params
            ):
                declared_type = None
            if {
                name,
                re.match(r"[\w$]*", match["type"] or "")[0],
            } & _STATEMENT_KEYWORDS:
                return None
            named_at, column = _locate(starts, match.start("name"))
            named_at += index
            if not match["params"] and accessor != "get":
                # A variable or a field, and those declared with it.
                valued = match["modifiers"] or match["type"]
                if local or accessor or not valued or arrow or ended not in ("=", ";"):
                    return None
                fields = [name, *re.findall(r"[\w$]+", match["more"] or "")]
                values = []  # the parts of the value that an `=` separates
                value_at, value_col = at, col
                while lines[value_at][value_col : value_col + 1] == "=":
                    value, _, value_at, value_col = _read_header(
                        lines, value_at, value_col + 1
                    )
                    values.append(value)
                    # `, name` before the next `=` declares one more.
                    fields += re.findall(r",\s*([\w$]+)\s*$", value)
                called = (
                    len(values) == 1
                    and not match["type"]
                    and lines[value_at][value_col : value_col + 1] == ";"
                    and _RE_CALL_VALUE.fullmatch(value)
                )
                for field in fields:
                    node_id = f"{parent_id}::{field}"
                    renamed[node_id] = f"{parent_id}::field {field}"
                    if node_id in constructors:
                        # A constructor of the same name keeps `<type>::<name>`.
                        node_id = renamed[node_id]
                    if node_id not in self.builder.node_registry:
                        self.builder.add_node(
                            node_id,
                            "Constant",
                            name=field,
                            lineno=named_at + 1,
                            docstring="",
                            value_type=declared_type,
                            value_call=called[1] if called else None,
                        )
                        self.builder.add_edge(parent_id, node_id, "DEFINES")
                return self._statement_end(lines, at, col)[0] + 1
            if match["more"] or (local and (accessor or ended not in ("{", "="))):
                return None
            if ended == "{":
                last = self._find_block_end(lines, at, col) - 1
            elif ended in ("=", ":"):
                last, closed = self._statement_end(
                    lines, at, col, raw_lines if ended == ":" else None
                )
                if local and not (arrow and closed):
                    return None
            else:
                last = at
            constructs = owner and re.fullmatch(
                rf"{re.escape(owner)}(?:\.[\w$]+)?", name
            )
            if constructs or "factory" in match["modifiers"]:
                # The constructor without a name of its own is `Type.new`.
                name = name.partition(".")[2] or "new"
                metadata = {"return_type": owner, "is_constructor": True}
            elif "." in name:
                return None
            elif accessor:
                metadata = {"accessor": accessor, "value_type": declared_type}
            else:
                name = "".join(name.split())
                metadata = {"return_type": declared_type}
            node_id = f"{parent_id}::{'set ' if accessor == 'set' else ''}{name}"
            if "is_constructor" in metadata:
                constructors.add(node_id)
                if node_id in renamed:
                    self.rename_node(node_id, renamed[node_id])
            else:
                renamed[node_id] = f"{parent_id}::{accessor or 'method'} {name}"
                if node_id in constructors:
                    # A constructor of the same name keeps `<type>::<name>`.
                    node_id = renamed[node_id]
            node_type = "Function" if parent_id == file_id or local else "Method"
            define(
                node_type, parent_id, node_id, name, index, named_at, last, **metadata
            )
            params = []
            if match["params"]:
                declared_at.setdefault(named_at, set()).add(column)
                params = _parameter_names(source[slice(*match.span("params"))][1:-1])
            for param in params:
                self.add_local(node_id, param)
                # One the line types, or tests with `is`, holds what that says.
                if (index + 1, param) not in self._declared_columns:
                    self.declared.setdefault(param, []).append((index + 1, "?", False))
            self.enter_context(node_id, index + 1, last + 1, frozenset(params))
            return at + 1

        for i, code in enumerate(lines):
            lineno = i + 1
            context_id = self.context_at(lineno)[0]
            at_definitions = context_id == file_id or context_id in types

            if i >= resume and context_id == file_id:
                m_lib = _RE_LIBRARY.match(code)
                if m_lib:
                    self.set_file_metadata(file_id, "dart_library", m_lib.group(1))
                    continue
                m_part = _RE_PART_OF.match(raw_lines[i])
                if m_part and code.lstrip().startswith("part"):
                    self.builder.add_edge(
                        file_id, f"MODULE:{m_part.group(1)}", "IMPORTS"
                    )
                    continue
                directive = _RE_DIRECTIVE.match(code)
                if directive:
                    resume = self._statement_end(lines, i, 0)[0] + 1
                    uris = _RE_URI.findall(" ".join(raw_lines[i:resume]))
                    names = {"show": [], "hide": []}
                    clauses = re.split(r"\b(show|hide)\b", " ".join(lines[i:resume]))
                    for clause, listed in zip(clauses[1::2], clauses[2::2]):
                        names[clause] += re.findall(r"[\w$]+", listed)
                    prefix = re.search(r"\bas\s+([\w$]+)", clauses[0])
                    self.builder.dart_directives.setdefault(file_id, []).append(
                        (
                            directive[1],
                            uris,
                            prefix and prefix[1],
                            names["show"],
                            names["hide"],
                        )
                    )
                    if directive[1] != "export":
                        for uri in uris:
                            self.builder.add_edge(file_id, f"MODULE:{uri}", "IMPORTS")
                    continue

            if (
                i >= resume
                and code.strip()
                and (at_definitions or _RE_LOCAL_FUNCTION.match(code))
            ):
                header, starts, at, col = _read_header(lines, i)
                source = " ".join(lines[i : at + 1])
                begin = _RE_ANNOTATIONS.match(header).end()
                declared = context_id == file_id and declare_type(
                    i, header, source, starts, at, col, begin
                )
                declared = declared or declare_member(
                    i, header, source, starts, at, col, begin, context_id
                )
                if declared:
                    resume, statement = declared, lineno
                    context_id = self.context_at(lineno)[0]
                    at_definitions = context_id == file_id or context_id in types

            if skip_calls:
                continue
            if at_definitions:
                # A variable's value is read from the line that declares it.
                scope, since = None, statement if i < resume else lineno
            else:
                scope = context_id
                since = next(
                    first
                    for node_id, first, *_ in self._contexts
                    if node_id != file_id and node_id not in types
                )
            written = code
            code = _RE_TYPE_ARGUMENTS.sub(lambda match: " " * len(match[0]), code)
            self.begin_line(code, lineno, scope, raw_lines[i])
            self._since = since
            self._bind_locals(code, lineno, scope)
            self._switch_cases(lines, i)
            pattern_end = 0
            for call_match in _RE_CALL.finditer(code):
                callee = call_match.group(1)
                start = call_match.start(1)
                paren = call_match.end() - 1
                path = callee.replace("?", "")
                root, _, rest = path.partition(".")
                if start in declared_at.get(i, ()) or start < pattern_end:
                    continue
                # After a `.` a keyword names a member.
                if root in _DART_CONTROL_KEYWORDS and (
                    not rest or root not in ("this", "super")
                ):
                    continue
                before = code[:start].rstrip()
                if self._object_pattern(code, start, paren):
                    # An object pattern, and the patterns inside it, call nothing.
                    pattern_end = self._pairs.get(paren) or len(code)
                    continue
                self._read_at = start  # `add_call` clears it
                if before.endswith("..") and not before.endswith("..."):
                    # A cascade (`..add(x)`) calls a method of its target.
                    target = self._target_origin(
                        before[:-2].rstrip("?").rstrip(), lineno, since
                    )
                    self._targets[len(self.builder.edges)] = target
                    origin = f"{target}.{path}"
                elif before.lstrip() == "." and self._tail is None and self._tail_name:
                    # The receiver is the name the line before ends with.
                    origin = self._path_origin(
                        f"{self._tail_name}.{path}", lineno, since
                    )
                elif before.endswith("."):
                    cast = _RE_CAST_RECEIVER.search(before)
                    named = _TRAILING_PATH_RE.search(before[:-1].rstrip("? "))
                    if cast:
                        origin = f"{cast[1].replace('.', '::')}().{path}"
                    elif raw_lines[i][: len(before) - 1].rstrip()[-1:] in ("'", '"'):
                        origin = f"String().{path}"
                    elif named:
                        # `Type<T>.named(`, its type arguments blanked
                        origin = self._path_origin(
                            f"{named.group()}.{path}", lineno, since
                        )
                    else:
                        origin = None
                elif rest or _RE_CREATES.search(before):
                    origin = self._path_origin(path, lineno, since)
                else:
                    # A variable that holds an object is called through the
                    # object's `call` method.
                    held = (context_id, path) in self._locals and self.call_origin(
                        f"{path}.call", lineno, since
                    )
                    origin = f"={held}" if held and held[0] != "?" else None
                self.add_call(context_id, callee, lineno, since, start, paren, origin)
                if given := written[start + len(callee) : paren].strip():
                    # `Box<Dog>(`: the type arguments it is called with
                    site = self.builder.edges[-1]["metadata"].setdefault("site", {})
                    site["types"] = given
        self.restore_source_lines()
