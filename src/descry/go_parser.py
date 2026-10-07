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
from descry.generate import BaseParser, blank_strings, is_generated_source

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

# Type declaration: `type Name struct {`, `type Name interface {`, `type Name = AliasTarget`
_RE_TYPE_DECL = re.compile(
    r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(struct|interface|func|chan|map|\[\]|\*|=|[A-Za-z])"
)

# Function (free) declaration: `func Name(...) ...`
_RE_FUNC = re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^]]+\])?\s*\(")

# Method declaration: `func (receiver *Type) Name(...)`
_RE_METHOD = re.compile(
    r"^\s*func\s+"
    r"\(\s*[A-Za-z_][A-Za-z0-9_]*\s+\*?\s*([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[[^]]*\])?\s*\)\s+"  # optional type parameters: (s *Stack[T])
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^]]+\])?\s*\("
)

# Const single-line: `const Name = value`
_RE_CONST_SINGLE = re.compile(
    r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:[A-Za-z0-9_\[\]]*\s*)?="
)

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

_DECLARATIONS = (
    # `name Type` / `name *Type` in a receiver or parameter list
    re.compile(
        r"[(,]\s*(?P<name>[A-Za-z_]\w*)\s+\*?(?P<type>[A-Za-z_][\w.]*)\s*(?=[,)])"
    ),
    # `var name Type`
    re.compile(r"\bvar\s+(?P<name>\w+)\s+\*?(?P<type>[A-Za-z_][\w.]*)"),
    # `name := Type{` / `name = &Type{`
    re.compile(r"\b(?P<name>\w+)\s*:?=\s*&?(?P<type>[A-Za-z_][\w.]*)\{"),
)

# Go control-flow keywords that would spuriously match the call regex.
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


def _parameter_names(parameters: str) -> list[str]:
    """Names of the parameters in the text of a parameter list, which names
    all of them or none."""
    segments, depth, start = [], 0, 0
    for column, char in enumerate(f"{parameters},"):
        depth += (char in "([{") - (char in ")]}")
        if char == "," and depth == 0:
            segments.append(parameters[start:column].split())
            start = column + 1
    if not any(len(tokens) > 1 for tokens in segments):
        return []
    return [tokens[0] for tokens in segments if tokens and tokens[0].isidentifier()]


def _signature_tail(line: str, params: int) -> tuple[str, int, list[str]]:
    """Read a `func` line from column ``params``, just inside its parameter
    list. Returns the type of its first result (`T` for `(T, error)` or
    `(name T, err error)`), the column its body starts at and the names of
    its parameters."""
    depth = 1
    results = len(line)
    for column in range(params, len(line)):
        char = line[column]
        if char in "()":
            depth += 1 if char == "(" else -1
            if depth == 0 and results == len(line):
                results = column + 1
        elif char == "{" and depth == 0:
            if not line[:column].rstrip().endswith(("interface", "struct")):
                first = line[results:column].strip().lstrip("(")
                first = re.split(r"[,)]", first)[0].split()
                names = _parameter_names(line[params : results - 1])
                return (first[-1] if first else ""), column + 1, names
    return "", len(line), []


class GoParser(BaseParser):
    """Regex-driven Go source parser."""

    binding_re = re.compile(r"^\s*(?P<name>\w+)(?:\s*,\s*[\w.]+)*\s*:?=(?!=)")

    def _scan_calls(self, source_id, code, lineno, since, start, raw):
        """Record the calls on ``code``, the line ``raw`` with its literals
        blanked, from column ``start`` on. ``since`` is the first line of
        the enclosing function."""
        self.begin_line(code, lineno, source_id, raw)
        for call_match in _RE_CALL.finditer(code, start):
            callee = call_match.group(1)
            if callee.split(".")[-1] not in _GO_CONTROL_KEYWORDS:
                self.add_call(
                    source_id,
                    callee,
                    lineno,
                    since,
                    call_match.start(1),
                    call_match.end() - 1,
                )

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

        lines = content.splitlines()

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

        # Imports — single and grouped
        in_import_block = False
        in_const_block = False
        in_var_block = False
        self.scan_declarations(lines, _DECLARATIONS)
        self.enter_context(file_id, 0, len(lines))
        receivers: dict[str, tuple[str, int]] = {}  # type id -> (name, lineno)

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
                    name = m_entry.group(1)
                    parent_id = context_id
                    const_id = f"{parent_id}::{name}"
                    if const_id not in self.builder.node_registry:
                        self.builder.add_node(
                            const_id,
                            "Constant",
                            name=name,
                            lineno=lineno,
                            docstring="",
                        )
                        self.builder.add_edge(parent_id, const_id, "DEFINES")
                i += 1
                continue

            # Type declarations
            m_type = _RE_TYPE_DECL.match(line)
            if m_type:
                name, kind = m_type.groups()
                parent_id = context_id
                type_id = f"{parent_id}::{name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
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
                if end_lineno > lineno and kind in ("struct", "interface"):
                    self.enter_context(type_id, lineno, end_lineno)
                i += 1
                continue

            # Methods (receiver-bound)
            m_method = _RE_METHOD.match(line)
            if m_method:
                receiver_type, method_name = m_method.groups()
                # Method lives under its receiver type.
                method_id = f"{file_id}::{receiver_type}::{method_name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                docstring = self.get_leading_docstring(lines, i)
                return_type, body, parameters = _signature_tail(line, m_method.end())
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
            if m_func and "func(" not in line.replace(" ", ""):
                # Exclude `func(x int) ...` anonymous-style (the lookbehind is
                # approximate; the strict check is that `func` is followed by
                # an identifier before the `(`, which _RE_FUNC already enforces).
                name = m_func.group(1)
                parent_id = context_id
                func_id = f"{parent_id}::{name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
                docstring = self.get_leading_docstring(lines, i)
                return_type, body, parameters = _signature_tail(line, m_func.end())
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
            if m_const:
                name = m_const.group(1)
                parent_id = context_id
                const_id = f"{parent_id}::{name}"
                if const_id not in self.builder.node_registry:
                    self.builder.add_node(
                        const_id,
                        "Constant",
                        name=name,
                        lineno=lineno,
                        docstring="",
                    )
                    self.builder.add_edge(parent_id, const_id, "DEFINES")

            # Calls — regex path (no ast-grep Go backend today).
            # Skipped entirely for autogenerated files.
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
