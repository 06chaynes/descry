"""Dart regex-based baseline parser.

Covers the common case of Dart symbol extraction: library / part
directives, imports (``package:`` / ``dart:`` / relative), class /
mixin / extension / enum declarations, top-level functions, methods,
top-level constants, and call sites. Mirrors JavaParser / GoParser in
shape — walk lines, track brace depth for context, emit File →
Class/Function → Method → CALLS edges.

scip-dart augments resolution with type-aware cross-package
information when the binary is installed (see
``scip/adapters/dart.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

from descry.generate import BaseParser, is_generated_source, is_non_project_call


# import 'package:foo/bar.dart'; / import 'dart:async'; / import '../baz.dart';
_RE_IMPORT = re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""")

# part 'foo.dart'; / part of foo;
_RE_PART = re.compile(r"""^\s*part\s+(?:of\s+)?['"]?([^'";\s]+)""")

# library foo.bar;
_RE_LIBRARY = re.compile(r"^\s*library\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;")

# class Foo [extends Bar] [with Mixin] [implements Iface] { ... }
# Also catches `abstract class`, `base class`, `sealed class`, `final
# class`, `interface class`, and `mixin class`. No end-of-line anchor —
# Dart 3 allows `{}` on one line and the regex only needs the name.
_RE_CLASS = re.compile(
    r"^\s*(?:abstract\s+|base\s+|sealed\s+|final\s+|interface\s+|mixin\s+)*"
    r"class\s+([A-Z][A-Za-z0-9_]*)"
)

# `extends ParentName` anywhere on the class-declaration line.
_RE_EXTENDS = re.compile(r"\bextends\s+([A-Za-z_][A-Za-z0-9_.]*)")

# mixin Foo on Bar { ... }
_RE_MIXIN = re.compile(r"^\s*(?:base\s+)?mixin\s+([A-Z][A-Za-z0-9_]*)")

# extension FooExt on SomeType { ... } (name is optional in Dart 3.1+,
# but we only index named extensions).
_RE_EXTENSION = re.compile(
    r"^\s*extension\s+([A-Z][A-Za-z0-9_]*)"
    r"(?:\s*<[^>]*>)?"
    r"\s+on\s+"
)

# enum Foo { a, b, c } — also covers enhanced enums with bodies
_RE_ENUM = re.compile(r"^\s*enum\s+([A-Z][A-Za-z0-9_]*)")

# Typedef: typedef Name = FnType; or typedef Name<T> = FnType<T>;
_RE_TYPEDEF = re.compile(r"^\s*typedef\s+([A-Z][A-Za-z0-9_]*)")

# Top-level constant: const Name = ... / final Name = ... / final Type Name = ...
_RE_TOP_CONST = re.compile(
    r"^\s*(?:const|final)\s+(?:[A-Za-z_][A-Za-z0-9_<>, ?]*\s+)?"
    r"([A-Z_][A-Za-z0-9_]*)\s*="
)

# Top-level function / method declaration. Captures the name after the
# return type. Handles: `void main()`, `Future<int> load()`, `Map<K,V>
# parse()`, `T foo<T>(...)`, `static void bar()`, `@override Widget
# build(...)`. The heuristic: skip optional modifiers and annotations,
# match a type expression (identifier optionally with generics / `?`),
# then a bareword name followed by `(`. We deliberately don't try to
# distinguish top-level from class-body method here; the caller's
# indent-depth context already scopes them correctly.
_RE_FUNC_OR_METHOD = re.compile(
    r"^\s*"
    r"(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s+)*"
    r"(?:static\s+|external\s+|abstract\s+|final\s+|const\s+)*"
    r"(?:(?:Future|Stream|Iterable|List|Map|Set|FutureOr)"
    r"\s*<[^>]*>\s+|"
    r"[A-Z_][A-Za-z0-9_]*(?:\s*<[^>]*>)?\s+|"
    r"void\s+|"
    r"int\s+|"
    r"double\s+|"
    r"num\s+|"
    r"bool\s+|"
    r"String\s+|"
    r"dynamic\s+|"
    r"Object\s+\??\s*|"
    r"var\s+)"
    r"(?P<name>[a-z_][A-Za-z0-9_]*)"
    r"\s*(?:<[^>]*>)?\s*\("
)

# Constructor declaration: `ClassName(...)` or `ClassName.named(...)`.
# Caller must already be inside a class context for this to fire.
_RE_CTOR = re.compile(
    r"^\s*"
    r"(?:const\s+|factory\s+)*"
    r"([A-Z][A-Za-z0-9_]*)"
    r"(?:\.([a-z_][A-Za-z0-9_]*))?"
    r"\s*\("
)

# Call site: identifier(, or receiver.method(, or a?.method(
# Lookbehind guards against numeric literals, ., and the call-chain dot.
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_.$])"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\??\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*\("
)

# Dart control-flow / keyword tokens that the call regex would otherwise
# capture. `new` is deprecated but still legal; `throw` and `assert` look
# like calls syntactically. `factory`, `late`, and type modifiers appear
# at declaration sites where the next `(` is part of a constructor.
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
    }
)


def _strip_line_comment(line: str) -> str:
    """Return `line` with any trailing ``//...`` comment removed,
    preserving content inside single- and double-quoted strings so that
    URL literals like ``'http://'`` survive intact.

    Does NOT handle block comments (``/* ... */``) — those spans are
    suppressed in the main loop via an `in_block_comment` flag.
    """
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
        elif c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
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


class DartParser(BaseParser):
    """Regex-driven Dart source parser."""

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

        current_context: list[str] = [file_id]
        # Depth at which each context frame was pushed, so we pop when the
        # matching `}` lands on the line that drops depth below that value.
        context_enter_depth: list[int] = [0]
        brace_depth = 0
        in_block_comment = False

        for i, raw_line in enumerate(lines):
            lineno = i + 1
            stripped = raw_line.lstrip()

            # Block comment suppression — any span inside /* ... */ is
            # opaque to the rest of the parser.
            if in_block_comment:
                if "*/" in raw_line:
                    in_block_comment = False
                continue
            if stripped.startswith("/*") and "*/" not in raw_line:
                in_block_comment = True
                continue
            # Dart doc comments `///` are one-line; `_strip_line_comment`
            # treats them like `//` which is the right behavior for the
            # call regex.
            line = _strip_line_comment(raw_line)

            # Library / part directives — file-level metadata, no nodes.
            m_lib = _RE_LIBRARY.match(line)
            if m_lib:
                for node in self.builder.nodes:
                    if node.get("id") == file_id:
                        node.setdefault("metadata", {})["dart_library"] = m_lib.group(1)
                        break
                brace_depth += line.count("{") - line.count("}")
                continue

            m_part = _RE_PART.match(line)
            if m_part:
                self.builder.add_edge(file_id, f"MODULE:{m_part.group(1)}", "IMPORTS")
                brace_depth += line.count("{") - line.count("}")
                continue

            # Imports
            m_imp = _RE_IMPORT.match(line)
            if m_imp:
                self.builder.add_edge(file_id, f"MODULE:{m_imp.group(1)}", "IMPORTS")
                brace_depth += line.count("{") - line.count("}")
                continue

            # Class / mixin / extension / enum / typedef declarations
            m_class = _RE_CLASS.match(line)
            m_mixin = _RE_MIXIN.match(line) if not m_class else None
            m_ext = _RE_EXTENSION.match(line) if not m_class and not m_mixin else None
            m_enum = (
                _RE_ENUM.match(line)
                if not m_class and not m_mixin and not m_ext
                else None
            )
            m_typedef = (
                _RE_TYPEDEF.match(line)
                if not m_class and not m_mixin and not m_ext and not m_enum
                else None
            )

            decl_match = m_class or m_mixin or m_ext or m_enum or m_typedef
            if decl_match:
                name = decl_match.group(1)
                parent_id = current_context[-1]
                type_id = f"{parent_id}::{name}"

                # Allman-style support: opening `{` may be on the next line.
                has_open_brace = "{" in line
                if not has_open_brace:
                    for la in lines[i + 1 : i + 10]:
                        las = la.strip()
                        if las.startswith("{"):
                            has_open_brace = True
                            break
                        if not las:
                            continue
                        if "{" in las:
                            has_open_brace = True
                            break
                        break

                end_lineno = (
                    self._find_block_end(lines, i) if has_open_brace else lineno
                )
                docstring = self.get_leading_docstring(lines, i)
                kind = "class"
                if m_mixin:
                    kind = "mixin"
                elif m_ext:
                    kind = "extension"
                elif m_enum:
                    kind = "enum"
                elif m_typedef:
                    kind = "typedef"

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

                # INHERITS edge for `extends` targets (classes only).
                if m_class:
                    m_ext = _RE_EXTENDS.search(line)
                    if m_ext:
                        base = m_ext.group(1).split(".")[-1].split("<")[0].strip()
                        if base:
                            self.builder.add_edge(type_id, f"REF:{base}", "INHERITS")

                if has_open_brace and kind != "typedef":
                    current_context.append(type_id)
                    context_enter_depth.append(brace_depth + 1)

                brace_depth += line.count("{") - line.count("}")
                continue

            # Top-level `const Foo = ...` / `final Foo = ...` at file scope
            if current_context[-1] == file_id:
                m_const = _RE_TOP_CONST.match(line)
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

            # Function / method declarations. Match at both file scope
            # (top-level fn) and inside a class / mixin / extension
            # (method). Push the body as a call-attribution context so
            # calls inside the body get attributed correctly instead of
            # bubbling up to the enclosing class/file node.
            m_fn = _RE_FUNC_OR_METHOD.match(line)
            ctor_name: str | None = None
            if not m_fn and current_context[-1] != file_id:
                m_ctor = _RE_CTOR.match(line)
                if m_ctor:
                    # Only match a constructor if its leading identifier
                    # matches the enclosing type name. Avoids treating
                    # `SomeType(...)` construction calls as declarations.
                    parent_short = current_context[-1].split("::")[-1]
                    if m_ctor.group(1) == parent_short:
                        ctor_name = (
                            f"{parent_short}.{m_ctor.group(2)}"
                            if m_ctor.group(2)
                            else parent_short
                        )

            if m_fn or ctor_name:
                name = ctor_name or m_fn.group("name")
                parent_id = current_context[-1]
                kind = "Method" if parent_id != file_id else "Function"
                fn_id = f"{parent_id}::{name}"

                # Handle Allman-style bodies where `{` is on the next
                # non-blank line — scan up to 10 lines ahead.
                has_open_brace = "{" in line or line.rstrip().endswith("=>")
                if not has_open_brace:
                    for la in lines[i + 1 : i + 10]:
                        las = la.strip()
                        if not las:
                            continue
                        if las.startswith("{") or las.startswith("=>"):
                            has_open_brace = True
                            break
                        if "{" in las:
                            has_open_brace = True
                            break
                        break

                # Skip if this is likely an abstract / interface method
                # signature (ends with `;` and has no body). Those are
                # real Methods but shouldn't be pushed as call contexts.
                is_abstract = line.rstrip().endswith(";") and not has_open_brace

                end_lineno = (
                    self._find_block_end(lines, i) if has_open_brace else lineno
                )
                docstring = self.get_leading_docstring(lines, i)

                if fn_id not in self.builder.node_registry:
                    self.builder.add_node(
                        fn_id,
                        kind,
                        name=name,
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=(end_lineno - lineno + 1) * 10,
                        docstring=docstring,
                    )
                    self.builder.add_edge(parent_id, fn_id, "DEFINES")

                if has_open_brace and not is_abstract:
                    current_context.append(fn_id)
                    context_enter_depth.append(brace_depth + 1)

            # Call extraction — emit for every context, including file
            # scope. Dart uses top-level functions and top-level variable
            # initializers heavily (unlike Java/C#), so gating calls on
            # non-file context would drop a large fraction of real calls.
            # Skipped entirely for autogenerated files.
            parent_id = current_context[-1]
            call_matches = iter(()) if skip_calls else _RE_CALL.finditer(line)
            for call_match in call_matches:
                callee = call_match.group(1)
                simple = callee.split(".")[-1].lstrip("?")
                if simple in _DART_CONTROL_KEYWORDS:
                    continue
                if callee.split(".")[0] in _DART_CONTROL_KEYWORDS:
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

            # Pop context frames whose brace has closed.
            while len(current_context) > 1 and brace_depth < context_enter_depth[-1]:
                current_context.pop()
                context_enter_depth.pop()
