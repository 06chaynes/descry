"""C# / VB.NET regex-based baseline parser.

Covers the common cases of .NET symbol extraction: namespaces (both
block and file-scoped), classes / interfaces / structs / records /
enums, methods (including async ``Task<T>``), properties, fields,
``using`` statements. Visual Basic source (.vb) goes through the same
pipeline with only file discovery — full VB parsing is out of scope
(scip-dotnet covers both via SCIP).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from descry.generate import BaseParser, is_generated_source, is_non_project_call

logger = logging.getLogger(__name__)


# namespace App.Foo  (block) or  namespace App.Foo;  (file-scoped, C# 10+)
_RE_NAMESPACE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.]*)")

# using System;  /  using static System.Math;  /  using Foo = App.Bar;
_RE_USING = re.compile(
    r"^\s*(?:global\s+)?using\s+(?:static\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?"
    r"([A-Za-z_][A-Za-z0-9_.]*)\s*;"
)

_TYPE_MODIFIERS = (
    r"(?:public|internal|private|protected|static|sealed|abstract|"
    r"partial|readonly|new|unsafe|ref|record)"
)
_RE_CLASS = re.compile(
    rf"^\s*(?:{_TYPE_MODIFIERS}\s+)*"
    r"(class|struct|record|interface|enum)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)

_METHOD_MODIFIERS = (
    r"(?:public|internal|private|protected|static|virtual|abstract|"
    r"override|sealed|async|extern|new|partial|unsafe|readonly)"
)
_RE_METHOD = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"(?:<[^>]+>\s+)?"  # optional generic params
    r"(?:(?:\??[A-Za-z_][A-Za-z0-9_.<>\[\],?\s]*)\s+)"  # return type
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]+>)?\s*\("
)
# Explicit / implicit operators etc. are edge cases we skip.

# public int Foo { get; set; }  /  public string Name { get; init; }
_RE_PROPERTY = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"(?:\??[A-Za-z_][A-Za-z0-9_.<>\[\],?\s]*?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*(?:get|set|init)"
)
# public readonly int foo;  /  private string _name;
_RE_FIELD = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"(?:\??[A-Za-z_][A-Za-z0-9_.<>\[\],?\s]*?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;)"
)
_RE_CONSTANT = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"const\s+[A-Za-z_][A-Za-z0-9_.]*\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*="
)

# Call site: identifier(, obj.method(, Class.static(, Class<T>.method(
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*(?:<[^>]+>)?\s*\("
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
        # Additional C# contextual / declaration keywords that the
        # regex call pattern (`ident(`) would otherwise scoop up. `var`
        # shows up tens of thousands of times in modern C# and is
        # strictly a declaration keyword.
        "var",
        "dynamic",
        "void",
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
    }
)


def _strip_line_comment(line: str) -> str:
    """Strip ``//...`` trailing comments while preserving strings."""
    in_single = False
    in_double = False
    in_verbatim = False
    escape = False
    i = 0
    while i < len(line):
        c = line[i]
        if escape:
            escape = False
        elif c == "\\" and (in_single or in_double) and not in_verbatim:
            escape = True
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == "'" and not in_double:
            in_single = not in_single
        elif (
            not in_single
            and not in_double
            and c == "/"
            and i + 1 < len(line)
            and line[i + 1] == "/"
        ):
            return line[:i]
        i += 1
    return line


class DotnetParser(BaseParser):
    """C# parser (VB source contributes only file-level discovery)."""

    def parse(self, file_path, rel_path, content):
        file_id = f"FILE:{rel_path}"
        self.builder.add_node(
            file_id,
            "File",
            path=rel_path,
            name=Path(rel_path).name,
            token_count=len(content) // 4,
        )

        # VB files — scip-dotnet covers them; our regex parser only handles C#.
        if rel_path.endswith(".vb"):
            return

        lines = content.splitlines()
        skip_calls = is_generated_source(content)

        # Namespace + using pass
        namespace = None
        for line in lines[:60]:
            stripped = _strip_line_comment(line)
            m = _RE_NAMESPACE.match(stripped)
            if m:
                namespace = m.group(1)
                break
        if namespace:
            for node in self.builder.nodes:
                if node.get("id") == file_id:
                    node.setdefault("metadata", {})["dotnet_namespace"] = namespace
                    break

        for line in lines:
            stripped = _strip_line_comment(line)
            m = _RE_USING.match(stripped)
            if m:
                self.builder.add_edge(file_id, f"MODULE:{m.group(1)}", "IMPORTS")

        current_context: list[str] = [file_id]
        context_enter_depth: list[int] = [0]
        class_name_stack: list[str] = []
        brace_depth = 0
        in_block_comment = False

        i = 0
        n = len(lines)
        while i < n:
            raw_line = lines[i]
            lineno = i + 1

            # Block comments
            stripped = raw_line.lstrip()
            if in_block_comment:
                if "*/" in raw_line:
                    in_block_comment = False
                i += 1
                continue
            if stripped.startswith(("/*", "/**")) and "*/" not in raw_line:
                in_block_comment = True
                i += 1
                continue
            # XML doc comments (///) are single-line — handled by _strip_line_comment

            line = _strip_line_comment(raw_line)

            # Type declarations
            m_type = _RE_CLASS.match(line)
            if m_type:
                kind, name = m_type.groups()
                parent_id = current_context[-1]
                type_id = f"{parent_id}::{name}"

                # Allman brace style — look ahead for the `{`.
                brace_line_idx = i
                if "{" not in line:
                    for j in range(i + 1, min(i + 10, n)):
                        if "{" in lines[j]:
                            brace_line_idx = j
                            break
                has_open_brace = "{" in lines[brace_line_idx]

                end_lineno = (
                    self._find_block_end(lines, brace_line_idx)
                    if has_open_brace
                    else lineno
                )
                docstring = self.get_leading_docstring(lines, i)
                self.builder.add_node(
                    type_id,
                    "Class",
                    name=name,
                    kind=kind,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if has_open_brace:
                    current_context.append(type_id)
                    context_enter_depth.append(brace_depth)
                    class_name_stack.append(name)
                    for k in range(i, brace_line_idx + 1):
                        brace_depth += lines[k].count("{") - lines[k].count("}")
                    while (
                        len(current_context) > 1
                        and brace_depth <= context_enter_depth[-1]
                    ):
                        current_context.pop()
                        context_enter_depth.pop()
                        if class_name_stack:
                            class_name_stack.pop()
                    i = brace_line_idx + 1
                else:
                    i += 1
                continue

            # Methods / properties / fields / constants (inside a type)
            if current_context[-1] != file_id:
                m_method = _RE_METHOD.match(line)
                if m_method:
                    name = m_method.group(1)
                    if name not in _DOTNET_CONTROL_KEYWORDS:
                        parent_id = current_context[-1]
                        method_id = f"{parent_id}::{name}"
                        end_lineno = (
                            self._find_block_end(lines, i) if "{" in line else lineno
                        )
                        docstring = self.get_leading_docstring(lines, i)
                        self.builder.add_node(
                            method_id,
                            "Method",
                            name=name,
                            lineno=lineno,
                            end_lineno=end_lineno,
                            token_count=(end_lineno - lineno + 1) * 10,
                            docstring=docstring,
                        )
                        self.builder.add_edge(parent_id, method_id, "DEFINES")

                if "{" not in line and ";" in line:
                    m_const = _RE_CONSTANT.match(line)
                    m_field = _RE_FIELD.match(line)
                    m_prop = None
                    if not m_const and not m_field:
                        m_prop = _RE_PROPERTY.match(line)
                    for match_obj in (m_const, m_field, m_prop):
                        if not match_obj:
                            continue
                        name = match_obj.group(1)
                        parent_id = current_context[-1]
                        sym_id = f"{parent_id}::{name}"
                        if sym_id not in self.builder.node_registry:
                            self.builder.add_node(
                                sym_id,
                                "Constant",
                                name=name,
                                lineno=lineno,
                                docstring="",
                            )
                            self.builder.add_edge(parent_id, sym_id, "DEFINES")
                        break

            # Calls
            if not skip_calls and current_context[-1] != file_id:
                parent_id = current_context[-1]
                for call_match in _RE_CALL.finditer(line):
                    callee = call_match.group(1)
                    simple = callee.split(".")[-1]
                    if simple in _DOTNET_CONTROL_KEYWORDS:
                        continue
                    if is_non_project_call(callee):
                        continue
                    self.builder.edges.append(
                        {
                            "source": parent_id,
                            "target": f"REF:{callee}",
                            "relation": "CALLS",
                            "metadata": {"lineno": lineno},
                        }
                    )

            brace_depth += line.count("{") - line.count("}")
            while len(current_context) > 1 and brace_depth <= context_enter_depth[-1]:
                current_context.pop()
                context_enter_depth.pop()
                if class_name_stack:
                    class_name_stack.pop()

            i += 1
