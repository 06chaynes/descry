"""Java regex-based baseline parser.

Covers the common case of Java symbol extraction: packages, top-level and
nested types (``class``, ``interface``, ``enum``, ``record``, ``@interface``),
methods, constructors, fields, imports, and call sites. Kotlin and Scala
sources are NOT parsed here — they rely on scip-java for symbol resolution
when the indexer binary is present. Without scip-java, ``.kt`` and
``.scala`` files contribute only to file discovery.

The parser is deliberately regex-based (mirroring the existing Rust/TS
parsers in ``generate.py``). It is not a full Java grammar; it handles
well-formatted single-line declarations and balanced braces. Multi-line
return types, deeply nested generics, anonymous inner classes, and lambda
expressions are approximated — scip-java fills those gaps when available.
"""

from __future__ import annotations

import re
from pathlib import Path

from descry.generate import BaseParser, is_generated_source, is_non_project_call


# Package declaration: `package com.foo.bar;`
_RE_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;")

# Import: `import foo.Bar;` / `import static foo.Bar.method;` / `import foo.*;`
_RE_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;")

# Type declaration: class / interface / enum / record / @interface
_TYPE_MODIFIERS = (
    r"(?:public|protected|private|abstract|final|static|sealed|non-sealed)"
)
_RE_TYPE_DECL = re.compile(
    rf"^\s*(?:{_TYPE_MODIFIERS}\s+)*"
    r"(class|interface|enum|record|@interface)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)"
)

# Method declaration. Requires at least one visibility/modifier so we don't
# match bare control-flow like `if (cond)`. Does NOT match constructors
# (those lack a return type; handled separately).
_METHOD_MODIFIERS = r"(?:public|protected|private|static|final|abstract|synchronized|default|native|strictfp)"
_RE_METHOD = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS})\s+)+"
    r"(?:<[^>]+>\s+)?"  # optional generic type params
    r"[\w<>\[\],?\s.]+?\s+"  # return type (lazy)
    r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\("  # name + open paren
)

# Constructor: visibility + CapitalizedName ( — no return type.
_RE_CONSTRUCTOR = re.compile(
    r"^\s*(?:(?:public|protected|private)\s+)"
    r"([A-Z][A-Za-z0-9_$]*)\s*\("
)

# Field declaration: modifiers + type + name + (= value)? ;
_RE_FIELD = re.compile(
    rf"^\s*(?:(?:{_METHOD_MODIFIERS}|volatile|transient)\s+)+"
    r"[\w<>\[\],?\s.]+?\s+"
    r"([a-z_$][A-Za-z0-9_$]*)\s*(?:=|;)"
)

# Call site: identifier(...) or receiver.method(...).
# The lookbehind excludes:
#   - Other identifier chars (so we don't match the tail of `foo.bar`)
#   - `@` (so annotations `@JsonProperty(...)` are NOT matched as calls)
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_$.@])"
    r"([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)"
    r"\s*\("
)

# Keywords that the call regex would spuriously match.
_JAVA_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "throw",
        "new",
        "super",
        "this",
        "do",
        "synchronized",
        "assert",
        "try",  # try-with-resources: `try (Resource r = ...)` matches the call regex
        "yield",  # switch expressions
    }
)


def _strip_line_comment(line: str) -> str:
    """Return `line` with any trailing ``//...`` comment removed.

    Tracks simple string state so ``"http://"`` is preserved. Block
    comments (``/* ... */``) are tracked by a separate in-block flag in
    the main parse loop.
    """
    in_string = False
    escape = False
    i = 0
    while i < len(line):
        c = line[i]
        if escape:
            escape = False
        elif c == "\\" and in_string:
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string and c == "/" and i + 1 < len(line) and line[i + 1] == "/":
            return line[:i]
        i += 1
    return line


class JavaParser(BaseParser):
    """Regex-driven Java source parser.

    Pattern mirrors ``RustParser``: walk lines, maintain a context stack
    keyed off matching braces, emit File → Class/Interface/Enum/Record →
    Method/Field → CALLS edges.
    """

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

        # Package declaration (informational only; Java resolution is
        # fully-qualified and SCIP handles cross-package resolution).
        package = None
        for line in lines[:50]:
            m = _RE_PACKAGE.match(line)
            if m:
                package = m.group(1)
                break
        if package:
            # Annotate the file node with its package for downstream queries.
            for node in self.builder.nodes:
                if node.get("id") == file_id:
                    node.setdefault("metadata", {})["java_package"] = package
                    break

        # Imports
        for line in lines:
            m = _RE_IMPORT.match(line)
            if m:
                path = m.group(1).rstrip(".*").rstrip(".")
                if path:
                    self.builder.add_edge(file_id, f"MODULE:{path}", "IMPORTS")

        # Walk for types, methods, fields, and calls using brace depth.
        # context_enter_depth tracks the brace depth at which each context
        # (class/interface/enum/record) was entered, so we only pop the
        # enclosing type when its matching `}` closes — method / constructor
        # bodies don't pop the type context prematurely.
        current_context: list[str] = [file_id]
        context_enter_depth: list[int] = [0]
        class_name_stack: list[str] = []
        brace_depth = 0

        i = 0
        n = len(lines)
        in_block_comment = False
        while i < n:
            raw_line = lines[i]
            lineno = i + 1

            # Block comments (/* ... */ or /** ... */). Spans longer than
            # one line are suppressed wholesale so annotations / method
            # signatures inside Javadoc don't leak into the graph.
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
            # Lines starting with ``*`` inside a Javadoc block are caught
            # by the in_block_comment flag; continuation lines with
            # leading ``*`` outside of a block are rare enough to skip.

            # Strip trailing `// comment` so patterns like `+string(x)` in
            # an inline comment don't look like a call.
            line = _strip_line_comment(raw_line)

            # 1. Type declarations (class/interface/enum/record/@interface)
            m_type = _RE_TYPE_DECL.match(line)
            if m_type:
                kind, name = m_type.groups()
                parent_id = current_context[-1]
                type_id = f"{parent_id}::{name}"
                end_lineno = self._find_block_end(lines, i) if "{" in line else lineno
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
                if "{" in line:
                    current_context.append(type_id)
                    context_enter_depth.append(brace_depth)
                    class_name_stack.append(name)
                # Account for this line's braces before advancing.
                brace_depth += line.count("{") - line.count("}")
                while (
                    len(current_context) > 1 and brace_depth <= context_enter_depth[-1]
                ):
                    current_context.pop()
                    context_enter_depth.pop()
                    if class_name_stack:
                        class_name_stack.pop()
                i += 1
                continue

            # 2. Methods and constructors (only inside a type context)
            if current_context[-1] != file_id:
                m_method = _RE_METHOD.match(line)
                if m_method:
                    name = m_method.group(1)
                    if name not in _JAVA_CONTROL_KEYWORDS:
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
                else:
                    m_ctor = _RE_CONSTRUCTOR.match(line)
                    if m_ctor and class_name_stack:
                        name = m_ctor.group(1)
                        if name == class_name_stack[-1]:
                            parent_id = current_context[-1]
                            ctor_id = f"{parent_id}::<init>"
                            end_lineno = (
                                self._find_block_end(lines, i)
                                if "{" in line
                                else lineno
                            )
                            self.builder.add_node(
                                ctor_id,
                                "Method",
                                name="<init>",
                                lineno=lineno,
                                end_lineno=end_lineno,
                                token_count=(end_lineno - lineno + 1) * 10,
                                docstring="",
                                is_constructor=True,
                            )
                            self.builder.add_edge(parent_id, ctor_id, "DEFINES")

                # 3. Fields (only inside types, not methods)
                # Skip if line ended in `{` — that's a method/type decl with inline body.
                if "{" not in line and ";" in line:
                    m_field = _RE_FIELD.match(line)
                    if m_field:
                        name = m_field.group(1)
                        # Filter common false positives (keywords and control flow
                        # would already be caught by _METHOD_MODIFIERS requirement).
                        parent_id = current_context[-1]
                        field_id = f"{parent_id}::{name}"
                        # Avoid overwriting a method with the same name — check if the
                        # node already exists as a Method.
                        existing_ids = {n["id"] for n in self.builder.nodes}
                        if field_id not in existing_ids:
                            self.builder.add_node(
                                field_id,
                                "Constant",
                                name=name,
                                lineno=lineno,
                                docstring="",
                            )
                            self.builder.add_edge(parent_id, field_id, "DEFINES")

            # 4. Calls. ast-grep has no Java backend today, so we always
            #    run the regex extractor (rather than gating on
            #    ``not USE_AST_GREP`` like RustParser/TSParser do).
            if not skip_calls and current_context[-1] != file_id:
                parent_id = current_context[-1]
                for call_match in _RE_CALL.finditer(line):
                    callee = call_match.group(1)
                    simple_name = callee.split(".")[-1]
                    if simple_name in _JAVA_CONTROL_KEYWORDS:
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

            # 5. Update brace depth and pop type contexts that have closed.
            brace_depth += line.count("{") - line.count("}")
            while len(current_context) > 1 and brace_depth <= context_enter_depth[-1]:
                current_context.pop()
                context_enter_depth.pop()
                if class_name_stack:
                    class_name_stack.pop()

            i += 1
