"""Go regex-based baseline parser.

Covers the common case of Go symbol extraction: packages, imports, type
definitions (structs, interfaces, type aliases), free functions, methods,
``const`` and ``var`` blocks, and call sites. Mirrors the RustParser /
JavaParser pattern: walk lines, track brace depth for context, emit
File → Package/Type → Function/Method → CALLS edges.

The parser is intentionally regex-based (no tree-sitter dependency). It
handles well-formatted single-line declarations and balanced braces.
Parenthesized ``const ( ... )`` and ``var ( ... )`` groups are parsed as
a block whose entries become Constant nodes. scip-go augments resolution
with type-aware cross-package information when the binary is installed.
"""

from __future__ import annotations

import re

from descry._comments import strip_block_comments, strip_line_comment
from descry.generate import (
    BaseParser,
    _bracketed_parts,
    _go_type_parts,
    blank_strings,
    go_constraints,
    is_generated_source,
    source_lines,
)

# Package declaration: `package foo`
_RE_PACKAGE = re.compile(r"^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)")

# Single import: `import "foo/bar"` or `import alias "foo/bar"` or `import _ "foo/bar"`
_RE_IMPORT_SINGLE = re.compile(
    r'^\s*import\s+(?:([A-Za-z_][A-Za-z0-9_]*)\s+|\.\s+)?"([^"]+)"'
)

# Grouped import block start: `import (`
_RE_IMPORT_BLOCK_START = re.compile(r"^\s*import\s*\(")

# Inside-block import line: `"foo/bar"` or `alias "foo/bar"`
_RE_IMPORT_BLOCK_LINE = re.compile(
    r'^\s*(?:([A-Za-z_][A-Za-z0-9_]*)\s+|\.\s+)?"([^"]+)"'
)

# What a declaration's name is followed by in brackets: its type parameters,
# `[K comparable, V any]`, or for a type the length of the array it is
# Type parameters, `[K comparable, V any]`: names, then a constraint, which
# an array's length, `[5]`, is not
_TYPE_PARAMS = (
    r"\[(?P<generics>\s*[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s+(?:[^][]|\[[^][]*\])+)\]"
)

# Type declaration: `type Name struct {`, `type Name interface {`, `type Name = AliasTarget`
_RE_TYPE_DECL = re.compile(
    rf"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*{_TYPE_PARAMS}\s*|\s+)"
    r"(struct|interface|func|chan|map|\[|\*|=|[A-Za-z])"
)

# What a Go type's body of fields or methods opens after, and a declaration
# a line may go on to after the end of another
_RE_TYPE_HEAD = re.compile(
    rf"\s*type\s+[A-Za-z_][A-Za-z0-9_]*(?:\s*{_TYPE_PARAMS})?\s+(?:struct|interface)\b"
)
_RE_DEFINITION_HEAD = re.compile(r"(?:func|type|var|const)\b")

# Function (free) declaration: `func Name(...) ...`
_RE_FUNC = re.compile(
    rf"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:{_TYPE_PARAMS})?\s*\("
)

# Method declaration: `func (receiver *Type) Name(...)` / `func (Type) Name(...)`
_RE_METHOD = re.compile(
    r"^\s*func\s+"
    r"\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)?\*?\s*([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[(?P<generics>[^]]*)\])?\s*\)\s+"  # type parameters: (s *Stack[T])
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^]]+\])?\s*\("
)

# Const single-line: `const Name = value`
_RE_CONST_SINGLE = re.compile(
    r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:[A-Za-z0-9_\[\]]*\s*)?="
)

# Package-level var: `var Name Type` / `var Name = value`
_RE_VAR_SINGLE = re.compile(r"^var\s+(?!_\b)([A-Za-z_][A-Za-z0-9_]*)\b")

# Grouped const/var block: `const (` / `var (`
_RE_CONST_BLOCK_START = re.compile(r"^\s*const\s*\(")
_RE_VAR_BLOCK_START = re.compile(r"^\s*var\s*\(")

# Inside a const/var block: `Name = value` or `Name Type = value`
_RE_BLOCK_ENTRY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z0-9_.\[\]*]*\s*=?")

# Call site: `identifier(` or `receiver.method(`
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*\("
)

# The type a declaration gives a name: `Type` or `Type[Argument]`, or a
# slice, an array, a map or a channel of it
_DECLARED_TYPE = (
    r"\**(?P<type>(?:(?:\[\w*\]|map\[[\w.*\[\]]+\]|(?:<-\s*)?chan(?:\s*<-)?\s+)\**)*"
    r"[A-Za-z_][\w.]*(?:\[[\w.*, ]+\])?)"
)

_DECLARATIONS = (
    # `name Type` / `name *Type` in a receiver or parameter list; the `(`
    # left before the match keeps it from being read as a file-level one
    re.compile(
        r"(?<=[(,])\s*(?P<name>[A-Za-z_]\w*)\s+\*?" + _DECLARED_TYPE + r"\s*(?=[,)])"
    ),
    # `name, other Type` in such a list: a name the next type is also for
    re.compile(
        r"(?<=[(,])\s*(?P<name>[A-Za-z_]\w*)(?=\s*,\s*(?:\w+\s*,\s*)*\w+\s+\*?"
        + _DECLARED_TYPE
        + r"\s*[,)])"
    ),
    # `var name Type`
    re.compile(r"\bvar\s+(?P<name>\w+)\s+\*?" + _DECLARED_TYPE),
    # `name := Type{` / `name = &Type{`
    re.compile(r"(?<![\w.])(?P<name>\w+)\s*:?=\s*&?" + _DECLARED_TYPE + r"\{"),
    # `name Type` / `name *Type`, alone on its line: a struct field, or with
    # a `,` a parameter or result of a list written one to a line
    re.compile(
        r"^\s++(?!(?:return|go|defer|goto|case|package|import|type|var|const)\b)"
        r"(?P<name>[A-Za-z_]\w*)\s++\*?"
        + _DECLARED_TYPE
        + r"\s*+,?\s*+(?:`[^`]*`\s*)?(?://.*)?$"
    ),
    # `name := new(Type)`
    re.compile(r"\b(?P<name>\w+)\s*:?=\s*new\(\s*(?P<type>[A-Za-z_][\w.]*)\s*\)"),
    # `name := value.(Type)` / `name, ok := value.(*Type)`, a type assertion
    # that ends its line or the initializer of an `if`
    re.compile(
        r"\b(?P<name>\w+)(?:\s*,\s*\w+)?\s*:?=[^=;]*\.\(\s*\*?"
        r"(?P<type>(?!type\b)[A-Za-z_][\w.]*)\s*\)\s*(?:;.*)?$"
    ),
    # `name := make([]Type, n)` / `name := make(map[Key]Type)`
    re.compile(
        r"(?<![\w.])(?P<name>\w+)\s*:?=\s*make\(\s*" + _DECLARED_TYPE + r"\s*[,)]"
    ),
)

# A line of a struct: `name Type`, `name, other Type`, or the embedded
# `Type` / `*pkg.Type`, a field named as its type is
_RE_FIELD = re.compile(
    r"\s+(?:(?P<names>\w+(?:\s*,\s*\w+)*)\s+\*?"
    + _DECLARED_TYPE
    + r"|\*?(?P<embedded>(?:\w+\.)?(?P<name>[A-Za-z_]\w*)))"
    r"\s*(?:`[^`]*`\s*)?(?://.*)?"
)

# `Name(` at the start of a line of an interface: a method it declares
_RE_INTERFACE_METHOD = re.compile(r"\s+([A-Za-z_]\w*)\s*\(")

# `name func(` / `name, other func(` on a line of a struct: fields that hold
# a function
_RE_FUNCTION_FIELD = re.compile(r"\s+(?P<names>\w+(?:\s*,\s*\w+)*)\s+func\s*\(")

# `var name, other Type`, or without `var` an entry of a `var (...)` block
_RE_VAR_GROUP = re.compile(
    r"\s*(?P<var>var\s+)?(?P<names>\w+(?:\s*,\s*\w+)+)\s+"
    + _DECLARED_TYPE
    + r"\s*(?:=.*)?"
)

# `for key, value := range over {`
_RE_RANGE = re.compile(
    r"\bfor\s+(?P<key>\w+)(?:\s*,\s*(?P<value>\w+))?\s*:=\s*range\s+"
    r"(?P<over>[^{]*)"
)

# `name := value` / `name, ok := value`, the value no call and no literal
_RE_COPY = re.compile(
    r"\s*(?P<name>\w+)(?:\s*,\s*\w+)?\s*:=\s*(?P<value>[^;(){}]+?)\s*$"
)

# `switch name := value.(type) {`
_RE_TYPE_SWITCH = re.compile(
    r"\s*switch\s+(?P<name>\w+)\s*:=\s*(?P<value>.+?)\.\(type\)\s*\{"
)
# `case Type:` / `case *Type:`, a clause of one type
_RE_CASE = re.compile(r"\s*case\s+\*?" + _DECLARED_TYPE + r"\s*:")

# The simple statement an `if`, `switch` or `for` header starts with, in
# group 2
_RE_INITIALIZER = re.compile(r"(\s*(?:\}\s*else\s+)?(?:if|switch|for)\s)([^;{]*);")

# `name := func(`, a variable that holds a function literal
_RE_LITERAL = re.compile(r"\s*(?P<name>\w+)\s*:?=\s*func\s*\(")

# `value[i].` or `new(Type).` before a callee: the receiver is an element
# of `value`, or a `Type`
_RE_RECEIVER = re.compile(
    r"(?:(?P<value>[\w.]+(?:\[[^\[\]]*\])+)|\bnew\(\s*(?P<new>[\w.]+)\s*\))\.\s*$"
)

# Go keywords and built-in functions that would spuriously match the call
# regex where they stand alone: a method may be named as a built-in is.
_GO_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "switch",
        "select",
        "return",
        "go",
        "defer",
        "func",
        "chan",
        "map",
        "range",
        "make",
        "new",
        "len",
        "cap",
        "append",
        "copy",
        "delete",
        "panic",
        "recover",
        "print",
        "println",
        "close",
    }
)


def _list_items(text: str) -> list[list[str]]:
    """The words of each item in the text of a parameter or result list."""
    segments, depth, start = [], 0, 0
    for column, char in enumerate(f"{text},"):
        depth += (char in "([{") - (char in ")]}")
        if char == "," and depth == 0:
            segments.append(text[start:column].split())
            start = column + 1
    return segments


def _type_parameters(written: str | None) -> dict[str, str | None] | None:
    """``{name: type}`` of each parameter of the type parameter list
    ``written``, the text in its brackets: its constraint where that is one
    named type, else None. None where ``written`` is an array's length,
    which no constraint follows."""
    parts = _bracketed_parts(f"{written}]")
    if not parts or not re.match(r"[A-Za-z_]\w*\s+[\w~\[]", parts[-1]):
        return None
    params, names = {}, []
    for part in parts:
        name, *constraint = part.split(None, 1)
        names.append(name)
        if constraint:
            named = re.fullmatch(r"[A-Za-z_][\w.]*", constraint[0].strip())
            bound = named and named[0] not in ("any", "comparable") and named[0]
            params.update(dict.fromkeys(names, bound or None))
            names = []
    return params


def _parameter_names(parameters: str) -> list[str]:
    """Names of the parameters in the text of a parameter list, which names
    all of them or none."""
    segments = _list_items(parameters)
    if not any(len(tokens) > 1 for tokens in segments):
        return []
    return [tokens[0] for tokens in segments if tokens and tokens[0].isidentifier()]


def _result_types(results: str) -> str:
    """The text of a result list, which names all of its results or none,
    without the names: `(T, error)` for `(value T, err error)`."""
    results = results.strip()
    if not results.startswith("("):
        return results
    segments = [tokens for tokens in _list_items(results[1:-1]) if tokens]
    named = any(len(tokens) > 1 and tokens[0].isidentifier() for tokens in segments)
    types = []
    for tokens in reversed(segments):
        if named:
            # `a, b T` gives `a` the type of `b`.
            tokens = tokens[1:] or types[-1:]
        types.append(" ".join(tokens))
    if len(types) == 1:
        return types[0]
    return f"({', '.join(reversed(types))})"


def _signature_tail(
    lines: list[str], index: int, line: str, params: int
) -> tuple[str, int, list[str], int | None]:
    """Read the `func` declaration that starts on ``line``, which is
    ``lines[index]`` without its comments, from column ``params``, just
    inside its parameter list, through the lines its lists run on to.
    Returns the types of its results (see ``_result_types``), the column its
    body starts at, the names of its parameters and the index of the line
    its body opens on, None for a declaration without a body."""
    depth, results, start = 1, None, params
    while True:
        for column in range(start, len(line)):
            char = line[column]
            if char in "()":
                depth += 1 if char == "(" else -1
                if depth == 0 and results is None:
                    results = column + 1
            elif char == "{" and depth == 0:
                if not line[:column].rstrip().endswith(("interface", "struct")):
                    names = _parameter_names(line[params : results - 1])
                    types = _result_types(line[results:column])
                    return types, column + 1, names, index
        if depth == 0 or index + 1 == len(lines):
            types = "" if results is None else _result_types(line[results:])
            return types, len(line), [], None
        index += 1
        start = len(line)
        line += " " + strip_line_comment(lines[index], backticks=True)


class GoParser(BaseParser):
    """Regex-driven Go source parser."""

    binding_re = re.compile(
        r"^\s*(?P<var>var\s+)?(?P<name>\w+)(?P<later>(?:\s*,\s*[\w.]+)*)\s*"
        r"(?P<short>:)?=(?!=)"
    )
    # What the binding being read has after its first name, and whether it
    # declares its names
    _later = ()
    _declares = False

    def _end_line(self):
        # The statement a header starts with binds as it does on a line of
        # its own.
        if header := _RE_INITIALIZER.match(self._code):
            self._code = " " * header.end(1) + header[2]
        binding = self.binding_re.search(self._code)
        if binding:
            self._later = re.findall(r"[\w.]+", binding["later"])
            self._declares = bool(binding["var"] or binding["short"])
        name = binding["name"] if binding else self._pending and self._pending[0]
        known = len(self.declared.get(name, ()))
        super()._end_line()
        if self._later and len(self.declared.get(name, ())) > known:
            # `a, b := f()` binds each name to the result of the call in its
            # position; an assignment leaves the later variables the types
            # they were declared with.
            line, call, _ = self.declared[name].pop()
            names = [name, *self._later] if self._declares else [name]
            for position, bound in enumerate(names):
                self.declared.setdefault(bound, []).append(
                    (line, f"{call}.@{position}", False)
                )

    def call_origin(self, callee, lineno, since, owner=None):
        return self._path_origin(callee, lineno, since) if "." in callee else None

    def _path_origin(self, path, lineno, since):
        """Origin path of the variable ``path`` or of `variable.field`, the
        fields to be read from what the variable is declared as. A local
        variable of no known type is `?`, not a package or a package-level
        variable of its name. None for any other name."""
        value, _, rest = path.partition(".")
        origin = self._declared(value, lineno, since)
        if parameter := re.fullmatch(r"(\w+)\(\)", origin or ""):
            # A value of a type parameter's type has its constraint's methods.
            for first, last, params in reversed(self._generics):
                if first <= lineno <= last and parameter[1] in params:
                    bound = params[parameter[1]]
                    origin = f"{bound.replace('.', '::')}()" if bound else "?"
                    break
        if origin is None and (self._scope, value) in self._locals:
            origin = "?"
        return origin and ".".join(filter(None, (origin, rest)))

    def _declared_on(self, name, lineno):
        """The type a declaration on line ``lineno`` gives ``name``, as it
        is written there; None where none does."""
        held = (held for at, held, _ in self.declared.get(name, ()) if at == lineno)
        held = next(held, "")
        return held[:-2].replace("::", ".") if held.endswith("()") else None

    def _held(self, expression, lineno, since):
        """Origin path of what ``expression`` holds at ``lineno``: a variable
        or a field of one, behind `*` or `&` or not, a variable of an
        imported package, an element `value[i]` or a part `value[i:j]` of
        one, what `<-value` receives, or the type a `[]Type{` or
        `map[Key]Type{` literal starts with. None where no declaration
        tells."""
        expression = expression.lstrip("*& ")
        # The first of what a channel holds is its element, the second of
        # what a slice or a map does.
        if expression.startswith("<-"):
            held = self._held(expression[2:], lineno, since)
            return held and f"{held}.@0"
        if indexed := re.fullmatch(r"(.+)\[([^\[\]]*)\]", expression):
            held = self._held(indexed[1], lineno, since)
            return held and (held if ":" in indexed[2] else f"{held}.@1")
        if re.fullmatch(r"\w+(?:\.\w+)*", expression):
            origin = self._path_origin(expression, lineno, since)
            package, _, member = expression.partition(".")
            imports = self.builder.imports.get(self._scope.split("::")[0], {})
            if origin is None and package in imports and member.isidentifier():
                # `pkg.Name`, a variable of an imported package
                origin = f"{package}::{member}"
            return origin
        if re.fullmatch(_DECLARED_TYPE, expression) and "[" in expression:
            return f"{expression.replace('.', '::')}()"
        return None

    def _scan_calls(self, source_id, code, lineno, since, start, raw):
        """Record the calls on ``code``, the line ``raw`` with its literals
        blanked, from column ``start`` on. ``since`` is the first line of
        the enclosing function."""
        self.begin_line(code, lineno, source_id, raw)
        if copied := _RE_COPY.fullmatch(
            header[2] if (header := _RE_INITIALIZER.match(code)) else code[start:]
        ):
            held = self._held(copied["value"], lineno, since)
            if held:
                self.declared.setdefault(copied["name"], []).append(
                    (lineno, held, False)
                )
        elif literal := _RE_LITERAL.match(code, start):
            # A call of the variable returns what the literal declares.
            returns = _signature_tail([code], 0, code, literal.end())[0]
            if returns:
                self.declared.setdefault(literal["name"], []).append(
                    (lineno, f"={literal['name']}>{returns}", False)
                )
        for call_match in _RE_CALL.finditer(code, start):
            callee = call_match.group(1)
            before = code[: call_match.start(1)].rstrip()
            member = "." in callee or before.endswith(".")
            if member or callee not in _GO_CONTROL_KEYWORDS:
                held = None
                if receiver := _RE_RECEIVER.search(code, 0, call_match.start(1)):
                    held = receiver["new"] and f"{receiver['new'].replace('.', '::')}()"
                    held = held or self._held(receiver["value"], lineno, since)
                origin = held and f"{held}.{callee}"
                if not member:
                    literal = self._declared(callee, lineno, since) or ""
                    origin = literal if literal.startswith("=") else None
                self.add_call(
                    source_id,
                    callee,
                    lineno,
                    since,
                    call_match.start(1),
                    call_match.end() - 1,
                    origin,
                )
        if ranged := _RE_RANGE.search(code, start):
            # The range variables are the first and the second of what the
            # value ranged over holds: its key or index and its element. It
            # is a value `_held` reads or what a call returned; any other
            # gives them no known type.
            over = ranged["over"].rstrip()
            held = self._held(over, lineno, since)
            call = self._closes.get(ranged.start("over") + len(over) - 1)
            if held is None and call is not None:
                held = f"#{call}"
            for position, part in enumerate(("key", "value")):
                if ranged[part]:
                    origin = f"{held}.@{position}" if held else "?"
                    self.declared.setdefault(ranged[part], []).append(
                        (lineno, origin, False)
                    )
        # In a clause of a type switch its variable has the one type the
        # clause names, else the type of the value switched on.
        indent = len(code) - len(code.lstrip())
        while self._switches and indent < self._switches[-1][0]:
            self._switches.pop()
        if switch := _RE_TYPE_SWITCH.match(code, start):
            held = self._held(switch["value"], lineno, since)
            self._switches.append((indent, switch["name"], held or "?"))
        elif self._switches and indent == self._switches[-1][0]:
            _, name, held = self._switches[-1]
            if code.lstrip().startswith("}"):
                self._switches.pop()
            elif code.lstrip().startswith(("case", "default")):
                if clause := _RE_CASE.match(code):
                    held = f"{clause['type'].replace('.', '::')}()"
                self.declared.setdefault(name, []).append((lineno, held, False))

    def _add_import(self, file_id, alias, path):
        self.builder.add_edge(file_id, f"MODULE:{path}", "IMPORTS")
        if alias is None:
            # The package name is by convention the path's last element,
            # before any major-version suffix (`yaml.v3`, `.../v2`).
            parts = [p for p in path.split("/") if not re.fullmatch(r"v\d+", p)]
            alias = parts[-1].split(".")[0] if parts else "_"
        if alias != "_":
            self.builder.imports.setdefault(file_id, {})[alias] = path

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)
        self._switches = []  # (indent, variable, origin of its value) of each
        # The first and last line of each declaration with type parameters,
        # and the type each parameter stands for (see `_type_parameters`)
        self._generics = []

        lines = source_lines(content)
        lines = self.split_source(
            lines, self.blank_lines(lines, ("`",)), _RE_DEFINITION_HEAD, _RE_TYPE_HEAD
        )

        # Autogenerated Go files (protoc-gen-go, mockgen, wire,
        # moq, etc.) follow the well-known "// Code generated ... DO
        # NOT EDIT" convention. Their call graph is boilerplate that
        # never resolves to user code; SCIP indexers cover them
        # anyway. Skip CALLS emission from generated files — still
        # walk the structure so cross-file references to generated
        # types/functions can resolve.
        skip_calls = is_generated_source(content)

        # Package declaration
        package = None
        for line in lines[:30]:
            m = _RE_PACKAGE.match(line)
            if m:
                package = m.group(1)
                break
        if package:
            self.set_file_metadata(file_id, "go_package", package)
        requires, excludes = go_constraints(rel_path, lines[:30])
        if requires or excludes or rel_path.endswith("_test.go"):
            self.set_file_metadata(
                file_id,
                "build",
                (sorted(requires), sorted(excludes), rel_path.endswith("_test.go")),
            )

        # Imports — single and grouped
        in_import_block = False
        in_const_block = False
        in_var_block = False
        # A variadic `...Type` parameter is a `[]Type` in its function.
        self.scan_declarations(
            [line.replace("...", "[]") for line in lines], _DECLARATIONS
        )
        self.enter_context(file_id, 0, len(lines))
        receivers: dict[str, tuple[str, int]] = {}  # type id -> (name, lineno)
        # The type parameters of each generic type the file declares, which
        # a method's receiver names in their order
        declared_generics = {
            found[1]: list(params.values())
            for found in map(_RE_TYPE_DECL.match, lines)
            if found and (params := _type_parameters(found["generics"]))
        }

        i = 0
        n = len(lines)
        in_block_comment = False
        in_raw_string = None
        while i < n:
            raw_line = lines[i]
            lineno = i + 1
            context_id, context_start = self.context_at(lineno)

            # Handle block comments (/* ... */), which may span lines.
            raw_line, in_block_comment = strip_block_comments(
                raw_line, in_block_comment
            )

            # Declarations are matched on `line`, which has no trailing
            # `// comment`; calls are read from `code`, which also has its
            # string literals blanked.
            line = strip_line_comment(raw_line, backticks=True)
            code, in_raw_string = blank_strings(raw_line, in_raw_string, ("`",))

            # Import block state machine
            if _RE_IMPORT_BLOCK_START.match(line):
                in_import_block = True
                i += 1
                continue
            if in_import_block:
                if line.strip().startswith(")"):
                    in_import_block = False
                    i += 1
                    continue
                m = _RE_IMPORT_BLOCK_LINE.match(line)
                if m:
                    self._add_import(file_id, *m.groups())
                i += 1
                continue

            m_imp = _RE_IMPORT_SINGLE.match(line)
            if m_imp:
                self._add_import(file_id, *m_imp.groups())
                i += 1
                continue

            # `var a, b Type` declares every name, as does such an entry of a
            # `var (...)` block.
            grouped = []
            group = _RE_VAR_GROUP.fullmatch(line)
            if group and (group["var"] or in_var_block):
                grouped = re.findall(r"\w+", group["names"])
                held = f"{group['type'].replace('.', '::')}()"
                for name in grouped:
                    self.declared.setdefault(name, []).append(
                        (lineno, held, context_id == file_id)
                    )

            # const / var block state machine
            if _RE_CONST_BLOCK_START.match(line):
                in_const_block = True
                i += 1
                continue
            if _RE_VAR_BLOCK_START.match(line):
                in_var_block = True
                i += 1
                continue
            if in_const_block or in_var_block:
                if line.strip().startswith(")"):
                    in_const_block = False
                    in_var_block = False
                    i += 1
                    continue
                m_entry = _RE_BLOCK_ENTRY.match(line)
                if m_entry and not line.strip().startswith("//"):
                    grouped = grouped or [m_entry.group(1)]
                for name in grouped:
                    parent_id = context_id
                    const_id = f"{parent_id}::{name}"
                    if const_id not in self.builder.node_registry:
                        self.builder.add_node(
                            const_id,
                            "Constant",
                            name=name,
                            lineno=lineno,
                            docstring="",
                            value_type=self._declared_on(name, lineno),
                        )
                        self.builder.add_edge(parent_id, const_id, "DEFINES")
                i += 1
                continue

            # Type declarations
            m_type = _RE_TYPE_DECL.match(line)
            if m_type:
                name, generics, kind = m_type.groups()
                parent_id = context_id
                type_id = f"{parent_id}::{name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                if params := _type_parameters(generics):
                    self._generics.append((lineno, end_lineno, params))
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    type_id,
                    "Class",
                    name=name,
                    kind=kind if kind in ("struct", "interface") else "type",
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                # field name -> its type, `>` and what it is declared to
                # return for one that holds a function, and under None the
                # embedded types
                fields = {}
                for body in lines[i + 1 : end_lineno - 1] if kind == "struct" else ():
                    body = strip_line_comment(body, backticks=True)
                    if function := _RE_FUNCTION_FIELD.match(body):
                        returns = _signature_tail([body], 0, body, function.end())[0]
                        names = re.findall(r"\w+", function["names"])
                        fields.update(dict.fromkeys(names, f">{returns}"))
                for field in map(_RE_FIELD.fullmatch, lines[i + 1 : end_lineno - 1]):
                    if field and kind in ("struct", "interface"):
                        held = (field["type"] or field["embedded"]).replace(".", "::")
                        if kind == "struct":
                            names = re.findall(r"\w+", field["names"] or field["name"])
                            fields.update(dict.fromkeys(names, held))
                        if field["embedded"]:
                            fields.setdefault(None, []).append(held)
                # A named slice, array, map or channel holds what that does.
                held, parts, _ = _go_type_parts(line[m_type.start(3) :].strip())
                if held in ("[]", "map", "chan"):
                    fields = dict(enumerate(parts))
                if fields:
                    self.builder.fields[type_id] = fields
                if end_lineno > lineno and kind == "struct":
                    self.enter_context(type_id, lineno, end_lineno)
                if end_lineno > lineno and kind == "interface":
                    # Each method the interface declares is a node of it;
                    # nothing in its body is a call.
                    for index in range(i + 1, end_lineno - 1):
                        declared = strip_line_comment(lines[index], backticks=True)
                        if method := _RE_INTERFACE_METHOD.match(declared):
                            method_id = f"{type_id}::{method[1]}"
                            self.builder.add_node(
                                method_id,
                                "Method",
                                name=method[1],
                                lineno=index + 1,
                                end_lineno=index + 1,
                                token_count=10,
                                docstring=self.get_leading_docstring(lines, index),
                                receiver=name,
                                return_type=_signature_tail(
                                    lines, index, declared, method.end()
                                )[0],
                            )
                            self.builder.add_edge(type_id, method_id, "DEFINES")
                    i = end_lineno - 1
                i += 1
                continue

            # Methods (receiver-bound)
            m_method = _RE_METHOD.match(line)
            if m_method:
                receiver_type, generics, method_name = m_method.groups()
                # Method lives under its receiver type.
                method_id = f"{file_id}::{receiver_type}::{method_name}"
                return_type, body, parameters, opens = _signature_tail(
                    lines, i, line, m_method.end()
                )
                end_lineno = lineno
                if opens is not None:
                    end_lineno = self._find_block_end(lines, opens)
                if generics:
                    bounds = declared_generics.get(receiver_type, [])
                    names = [name.strip() for name in generics.split(",")]
                    params = dict(zip(names, bounds + [None] * len(names)))
                    self._generics.append((lineno, end_lineno, params))
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    method_id,
                    "Method",
                    name=method_name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                    receiver=receiver_type,
                    return_type=return_type,
                )
                receiver_id = f"{file_id}::{receiver_type}"
                receivers.setdefault(receiver_id, (receiver_type, lineno))
                self.builder.add_edge(receiver_id, method_id, "DEFINES")
                for parameter in parameters:
                    self.add_local(method_id, parameter)
                # The calls in the method body are attributed to the method.
                if end_lineno > lineno:
                    self.enter_context(method_id, lineno, end_lineno)
                if not skip_calls:
                    self._scan_calls(method_id, code, lineno, lineno, body, raw_line)
                i += 1
                continue

            # Free functions
            m_func = _RE_FUNC.match(line)
            if m_func:
                name = m_func.group(1)
                parent_id = context_id
                func_id = f"{parent_id}::{name}"
                return_type, body, parameters, opens = _signature_tail(
                    lines, i, line, m_func.end()
                )
                end_lineno = lineno
                if opens is not None:
                    end_lineno = self._find_block_end(lines, opens)
                if params := _type_parameters(m_func["generics"]):
                    self._generics.append((lineno, end_lineno, params))
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    func_id,
                    "Function",
                    name=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                    return_type=return_type,
                )
                self.builder.add_edge(parent_id, func_id, "DEFINES")
                for parameter in parameters:
                    self.add_local(func_id, parameter)
                if end_lineno > lineno:
                    self.enter_context(func_id, lineno, end_lineno)
                if not skip_calls:
                    self._scan_calls(func_id, code, lineno, lineno, body, raw_line)
                i += 1
                continue

            # Single-line const / var
            m_const = _RE_CONST_SINGLE.match(line)
            if not m_const and context_id == file_id:
                m_const = _RE_VAR_SINGLE.match(line)
            if m_const:
                grouped = grouped or [m_const.group(1)]
            for name in grouped if context_id == file_id or m_const else ():
                parent_id = context_id
                const_id = f"{parent_id}::{name}"
                if const_id not in self.builder.node_registry:
                    self.builder.add_node(
                        const_id,
                        "Constant",
                        name=name,
                        lineno=lineno,
                        docstring="",
                        value_type=self._declared_on(name, lineno),
                    )
                    self.builder.add_edge(parent_id, const_id, "DEFINES")

            # Calls, none for an autogenerated file.
            if not skip_calls and context_id != file_id:
                self._scan_calls(context_id, code, lineno, context_start, 0, raw_line)

            i += 1

        # A method's DEFINES edge is sourced at its receiver type. A type
        # declared in another file of the package has no node here, so give
        # it one.
        for receiver_id, (name, method_lineno) in receivers.items():
            if receiver_id not in self.builder.node_registry:
                self.builder.add_node(
                    receiver_id,
                    "Class",
                    name=name,
                    kind="type",
                    lineno=method_lineno,
                    docstring="",
                )
                self.builder.add_edge(file_id, receiver_id, "DEFINES")
        self.restore_source_lines()
