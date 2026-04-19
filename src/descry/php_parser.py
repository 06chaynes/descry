"""PHP regex-based baseline parser.

Covers PHP namespaces, classes (with ``extends`` / ``implements``),
interfaces, traits, enums (PHP 8.1+), methods, constructors,
properties, use-statements, and call sites. PHP syntax is close enough
to Java that the brace-tracking ``context_enter_depth`` pattern
transfers directly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from descry.generate import BaseParser, is_generated_source, is_non_project_call

logger = logging.getLogger(__name__)


# namespace App\\Foo;  (one-line) or  namespace App\\Foo { ... }  (block)
_RE_NAMESPACE = re.compile(r"^\s*namespace\s+([A-Za-z_\\][A-Za-z0-9_\\]*)\s*[;{]")

# use App\\Foo;  /  use App\\Foo as Bar;  /  use function foo;  /  use const X;
_RE_USE = re.compile(
    r"^\s*use\s+(?:function\s+|const\s+)?"
    r"([A-Za-z_\\][A-Za-z0-9_\\]*)"
    r"(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?\s*;"
)

# class Foo extends Bar implements Baz, Qux
_RE_CLASS = re.compile(
    r"^\s*(?:(?:abstract|final|readonly)\s+)*"
    r"class\s+([A-Za-z_][A-Za-z0-9_]*)"
)
# interface Foo
_RE_INTERFACE = re.compile(r"^\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)")
# trait Foo
_RE_TRAIT = re.compile(r"^\s*trait\s+([A-Za-z_][A-Za-z0-9_]*)")
# enum Status: string (PHP 8.1+)
_RE_ENUM = re.compile(r"^\s*enum\s+([A-Za-z_][A-Za-z0-9_]*)")

# public function foo( / private static function bar( / abstract protected function baz(
_METHOD_MODIFIERS = r"(?:public|protected|private|static|final|abstract|readonly)"
_RE_METHOD = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
# function top_level_fn( at file scope
_RE_FUNCTION = re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# public const FOO = 1;  /  private int $bar;  /  public readonly string $name;
_RE_PROPERTY = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"(?:(?:\?)?[A-Za-z_\\][A-Za-z0-9_\\|&]*\s+)?"
    r"\$([A-Za-z_][A-Za-z0-9_]*)\s*[;=]"
)
_RE_CONSTANT = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"const\s+([A-Z][A-Z0-9_]*)\s*="
)

# Call site: name(, $obj->method(, Class::static(, \\Fully\\Qualified(
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_$])"
    r"((?:\\?[A-Za-z_][A-Za-z0-9_\\]*"
    r"(?:->[A-Za-z_][A-Za-z0-9_]*|::[A-Za-z_][A-Za-z0-9_]*)*))"
    r"\s*\("
)

_PHP_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "elseif",
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
        "throw",
        "try",
        "catch",
        "finally",
        "match",
        "new",
        "instanceof",
        "isset",
        "unset",
        "empty",
        "list",
        "array",
        "echo",
        "print",
        "include",
        "include_once",
        "require",
        "require_once",
        "use",
        "yield",
        "fn",
        "function",
        "static",
        "self",
        "parent",
        "this",
        "true",
        "false",
        "null",
        "and",
        "or",
        "xor",
        "exit",
        "die",
    }
)


def _strip_line_comment(line: str) -> str:
    """Strip ``//...`` and ``#...`` trailing comments outside of strings."""
    in_single = False
    in_double = False
    escape = False
    i = 0
    while i < len(line):
        c = line[i]
        if escape:
            escape = False
        elif c == "\\" and (in_single or in_double):
            escape = True
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == "'" and not in_double:
            in_single = not in_single
        elif not in_single and not in_double:
            if c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                return line[:i]
            if c == "#":
                return line[:i]
        i += 1
    return line


class PhpParser(BaseParser):
    """Regex-driven PHP parser."""

    def parse(self, file_path, rel_path, content):
        file_id = f"FILE:{rel_path}"
        self.builder.add_node(
            file_id,
            "File",
            path=rel_path,
            name=Path(rel_path).name,
            token_count=len(content) // 4,
        )

        lines = content.splitlines()
        skip_calls = is_generated_source(content)

        # One-pass namespace + use
        namespace = None
        for line in lines[:60]:
            stripped = _strip_line_comment(line)
            m_ns = _RE_NAMESPACE.match(stripped)
            if m_ns:
                namespace = m_ns.group(1)
                break
        if namespace:
            for node in self.builder.nodes:
                if node.get("id") == file_id:
                    node.setdefault("metadata", {})["php_namespace"] = namespace
                    break

        for line in lines:
            stripped = _strip_line_comment(line)
            m = _RE_USE.match(stripped)
            if m:
                fqn = m.group(1).lstrip("\\")
                self.builder.add_edge(file_id, f"MODULE:{fqn}", "IMPORTS")

        # Walk with brace-depth context tracking (same pattern as Java).
        current_context: list[str] = [file_id]
        context_enter_depth: list[int] = [0]
        brace_depth = 0
        in_block_comment = False

        i = 0
        n = len(lines)
        while i < n:
            raw_line = lines[i]
            lineno = i + 1

            # Block comments /* ... */ (and /** ... */ Javadoc).
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

            line = _strip_line_comment(raw_line)

            # Type declarations: class / interface / trait / enum
            m_type = (
                _RE_CLASS.match(line)
                or _RE_INTERFACE.match(line)
                or _RE_TRAIT.match(line)
                or _RE_ENUM.match(line)
            )
            if m_type:
                name = m_type.group(1)
                parent_id = current_context[-1]
                type_id = f"{parent_id}::{name}"

                # PHP commonly uses Allman style — `{` often lands on the
                # next line after `class Foo implements X, Y, Z`. Look
                # ahead (up to 10 lines) to find the opening brace so we
                # can push a proper context.
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
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if has_open_brace:
                    current_context.append(type_id)
                    # Record the brace depth BEFORE the class's opening
                    # `{` is counted. Accumulate depth across any bridge
                    # lines between the class keyword and the `{`.
                    context_enter_depth.append(brace_depth)
                    # Advance past bridge lines so we don't re-scan them.
                    for k in range(i, brace_line_idx + 1):
                        brace_depth += lines[k].count("{") - lines[k].count("}")
                    while (
                        len(current_context) > 1
                        and brace_depth <= context_enter_depth[-1]
                    ):
                        current_context.pop()
                        context_enter_depth.pop()
                    i = brace_line_idx + 1
                else:
                    i += 1
                continue

            # Methods (inside a class) or top-level functions
            if current_context[-1] != file_id:
                m_method = _RE_METHOD.match(line)
                if m_method:
                    name = m_method.group(1)
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

                # Properties / constants
                if "{" not in line and ";" in line:
                    m_prop = _RE_PROPERTY.match(line)
                    if m_prop:
                        name = m_prop.group(1)
                        parent_id = current_context[-1]
                        prop_id = f"{parent_id}::{name}"
                        existing = {nd["id"] for nd in self.builder.nodes}
                        if prop_id not in existing:
                            self.builder.add_node(
                                prop_id,
                                "Constant",
                                name=name,
                                lineno=lineno,
                                docstring="",
                            )
                            self.builder.add_edge(parent_id, prop_id, "DEFINES")
                    m_const = _RE_CONSTANT.match(line)
                    if m_const:
                        name = m_const.group(1)
                        parent_id = current_context[-1]
                        const_id = f"{parent_id}::{name}"
                        existing = {nd["id"] for nd in self.builder.nodes}
                        if const_id not in existing:
                            self.builder.add_node(
                                const_id,
                                "Constant",
                                name=name,
                                lineno=lineno,
                                docstring="",
                            )
                            self.builder.add_edge(parent_id, const_id, "DEFINES")
            else:
                # File-scope function
                m_fn = _RE_FUNCTION.match(line)
                if m_fn:
                    name = m_fn.group(1)
                    func_id = f"{file_id}::{name}"
                    end_lineno = (
                        self._find_block_end(lines, i) if "{" in line else lineno
                    )
                    docstring = self.get_leading_docstring(lines, i)
                    self.builder.add_node(
                        func_id,
                        "Function",
                        name=name,
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=(end_lineno - lineno + 1) * 10,
                        docstring=docstring,
                    )
                    self.builder.add_edge(file_id, func_id, "DEFINES")

            # Calls (inside any context — methods, functions, etc.)
            if not skip_calls and current_context[-1] != file_id:
                parent_id = current_context[-1]
                for call_match in _RE_CALL.finditer(line):
                    callee = call_match.group(1)
                    simple_name = callee.split("::")[-1].split("->")[-1].split("\\")[-1]
                    if simple_name in _PHP_CONTROL_KEYWORDS:
                        continue
                    if is_non_project_call(callee.lstrip("\\")):
                        continue
                    self.builder.edges.append(
                        {
                            "source": parent_id,
                            "target": f"REF:{callee.lstrip(chr(92))}",
                            "relation": "CALLS",
                            "metadata": {"lineno": lineno},
                        }
                    )

            # Update brace depth + pop closed contexts
            brace_depth += line.count("{") - line.count("}")
            while len(current_context) > 1 and brace_depth <= context_enter_depth[-1]:
                current_context.pop()
                context_enter_depth.pop()

            i += 1
