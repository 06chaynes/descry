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
    _TRAILING_PATH_RE,
    BaseParser,
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
# What an object pattern follows: `case Foo(`, `final Foo(:name) = value`
_RE_PATTERN_STARTS = re.compile(r"\b(?:case|final|var)$")
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
    r"(?P<generic><[^;(){}=]*>)?\??"
)

_DECLARATIONS = (
    # `Type name` before `=`, `;`, `,`, `:`, a closing bracket, `in` or
    # `when`: a local, field, parameter, loop or pattern variable
    re.compile(
        rf"(?<![\w$.])(?<!Function\(){_DECLARED_TYPE}\s+(?P<name>[a-z_$][\w$]*)\s*"
        r"(?=[=;,:)\]}]|\b(?:in|when)\b)"
    ),
    # `name is Type` promotes the name
    re.compile(rf"(?<![\w$.])(?P<name>[a-z_$][\w$]*)\s+is!?\s+{_DECLARED_TYPE}"),
    # `on Type catch (name`
    re.compile(rf"\bon\s+{_DECLARED_TYPE}\s+catch\s*\(\s*(?P<name>[\w$]+)"),
    # `name = value as Type;`
    re.compile(rf"(?<![\w$.])(?P<name>[\w$]+)\s*=[^=;]*\bas\s+{_DECLARED_TYPE}\s*;"),
    # A type parameter's bound: `<T extends Type`
    re.compile(r"[<,]\s*(?P<param>[\w$]+)\s+extends\s+(?P<type>[\w$.]+)"),
)

# Names a line declares without a type or a value: `var name;`, the
# variable of `for (final name in`, a pattern's `var name` and `:name`,
# those of a `catch`, and one that holds a function: `Function(int) name`
_RE_UNTYPED = re.compile(
    r"\b(?:var|final|const|late)\s+([\w$]+)\s*(?=[;,):\]}?]|\s+in\b)"
    r"|\bcatch\s*\(\s*([\w$]+)(?:\s*,\s*([\w$]+))?"
    r"|[(,]\s*:([\w$]+)"
    r"|\bFunction\([^()]*\)\??\s+([\w$]+)\s*(?=[=;,)])"
)
# `var (a, b) = pair;` / `final [first, second] = list;`
_RE_DESTRUCTURED = re.compile(r"\b(?:var|final)\s*[(\[]([^()\[\]]*)[)\]]\s*=")
# The parameter list of a function expression: `(a, b) =>`, `(a) async {`
_RE_CLOSURE = re.compile(
    r"(?:^|[(,=:?\[{>]|\breturn)\s*\(([^()]*)\)\s*(?:async\*?\s*|sync\*\s*)?(?:=>|\{)"
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


class DartParser(BaseParser):
    """Regex-driven Dart source parser."""

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
        for names in (*_RE_DESTRUCTURED.findall(code), *_RE_CLOSURE.findall(code)):
            untyped += _parameter_names(names)
        for name in untyped:
            self.declared.setdefault(name, []).append((lineno, "?", False))
        if scope:
            typed = (match["name"] for match in _DECLARATIONS[0].finditer(code))
            for name in (*untyped, *typed):
                self.add_local(scope, name)

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

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        raw_lines = source_lines(content)
        skip_calls = is_generated_source(content)

        # Each line without its comments and with its string literals
        # blanked; `a!.b` reads as `a?.b`.
        lines = []
        inside = None
        for line in raw_lines:
            code, inside = blank_strings(
                line.replace("'''", '"""').replace('r"""', ' """'),
                inside,
                ('"""', "'"),
            )
            # Only a triple-quoted string goes on over the next line.
            inside = None if inside == "'" else inside
            lines.append(code.replace("!.", "?."))

        # A name is declared on its own line with the type the line before
        # ends with.
        joined = list(lines)
        for index in range(1, len(lines)):
            if _RE_NAME_STARTS.match(lines[index]) and _RE_TYPE_ENDS.search(
                lines[index - 1]
            ):
                joined[index] = f"{lines[index - 1].rstrip()} {lines[index].lstrip()}"
        self.scan_declarations(joined, _DECLARATIONS)
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
            kind, heritage = "typedef", ""
            if not match:
                match = _RE_TYPE_DECL.match(header, begin)
                if not match:
                    return None
                kind = " ".join(match["kind"].split())
                heritage = re.sub(r"<~*>|\(~*\)", "", header[match.end() :])
            opened = lines[at][col : col + 1]
            if opened == "=":
                # `class Name = Base with Mixin;`, `typedef Name = Type;`
                heritage = "extends " + re.sub(
                    r"<~*>", "", _read_header(lines, at, col + 1)[0]
                )
            if kind == "typedef" and (opened != "=" or "Function" in heritage):
                # A function type has the members of `Function`.
                heritage = "extends Function"
            clauses = re.split(r"\b(extends|with|implements|on)\b", heritage)
            bases = {
                clause: [base.strip(" ?") for base in listed.split(",")]
                for clause, listed in zip(clauses[1::2], clauses[2::2])
            }
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
            define("Class", file_id, type_id, name, index, named_at, last, kind=kind)
            if match["generics"]:
                self._type_params.update(
                    _type_parameters(source[slice(*match.span("generics"))])
                )
            # A mixin overrides the superclass and the mixins before it.
            for base in (
                *reversed(bases.get("with", ())),
                *bases.get("extends", ()),
                *bases.get("implements", ()),
                *bases.get("on", ()),
            ):
                if re.fullmatch(r"[\w$.]+", base):
                    self.builder.add_edge(type_id, f"REF:{base}", "INHERITS")
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
            code = _RE_TYPE_ARGUMENTS.sub(lambda match: " " * len(match[0]), code)
            self.begin_line(code, lineno, scope, raw_lines[i])
            self._since = since
            self._bind_locals(code, lineno, scope)
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
                closed = self._pairs.get(paren)
                if _RE_PATTERN_STARTS.search(before) or (
                    closed and _RE_PATTERN_ENDS.match(code, closed + 1)
                ):
                    # An object pattern, and the patterns inside it, call nothing.
                    pattern_end = closed or len(code)
                    continue
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
