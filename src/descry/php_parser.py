"""PHP regex-based baseline parser.

Covers PHP namespaces, classes (with ``extends`` / ``implements``),
interfaces, traits, enums (PHP 8.1+), methods, constructors,
properties, use-statements, and call sites. Contexts are tracked by
brace depth.
"""

from __future__ import annotations

import re

from descry._comments import strip_block_comments, strip_line_comment
from descry.generate import BaseParser, is_generated_source

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

_LINE_COMMENT_MARKERS = ("//", "#")


class PhpParser(BaseParser):
    """Regex-driven PHP parser."""

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        lines = content.splitlines()
        skip_calls = is_generated_source(content)

        # One-pass namespace + use
        namespace = None
        for line in lines[:60]:
            stripped = strip_line_comment(line, _LINE_COMMENT_MARKERS)
            m_ns = _RE_NAMESPACE.match(stripped)
            if m_ns:
                namespace = m_ns.group(1)
                break
        if namespace:
            self.set_file_metadata(file_id, "php_namespace", namespace)

        for line in lines:
            stripped = strip_line_comment(line, _LINE_COMMENT_MARKERS)
            m = _RE_USE.match(stripped)
            if m:
                fqn = m.group(1).lstrip("\\")
                self.builder.add_edge(file_id, f"MODULE:{fqn}", "IMPORTS")

        # Walk with brace-depth context tracking.
        current_context: list[str] = [file_id]
        context_enter_depth: list[int] = [0]
        brace_depth = 0
        in_block_comment = False

        i = 0
        n = len(lines)
        while i < n:
            raw_line = lines[i]
            lineno = i + 1
            # Call scanning starts past a name declared on the line.
            call_scan_start = 0

            # Block comments /* ... */ (and /** ... */ Javadoc), wherever
            # they open or close on the line.
            raw_line, in_block_comment = strip_block_comments(
                raw_line, in_block_comment, hash_comments=True
            )

            line = strip_line_comment(raw_line, _LINE_COMMENT_MARKERS)

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
                brace_line_idx = self.opening_brace_line(lines, i, line)
                has_open_brace = brace_line_idx is not None

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
                    brace_depth += line.count("{") - line.count("}")
                    for k in range(i + 1, brace_line_idx + 1):
                        brace_depth += lines[k].count("{") - lines[k].count("}")
                    self.leave_contexts(
                        current_context, context_enter_depth, brace_depth
                    )
                    i = brace_line_idx + 1
                else:
                    i += 1
                continue

            # Methods (inside a class) or top-level functions
            if current_context[-1] != file_id:
                m_method = _RE_METHOD.match(line)
                if m_method:
                    call_scan_start = m_method.end()
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
                        if prop_id not in self.builder.node_registry:
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
                        if const_id not in self.builder.node_registry:
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
                    call_scan_start = m_fn.end()
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
                for call_match in _RE_CALL.finditer(line, call_scan_start):
                    callee = call_match.group(1)
                    simple_name = callee.split("::")[-1].split("->")[-1].split("\\")[-1]
                    if simple_name in _PHP_CONTROL_KEYWORDS:
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
            self.leave_contexts(current_context, context_enter_depth, brace_depth)

            i += 1
