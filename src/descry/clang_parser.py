"""C / C++ regex-based baseline parser.

Covers the common shapes of C/C++ symbol extraction: ``#include``
directives, ``namespace``, ``class`` / ``struct`` / ``union`` /
``enum``, top-level function definitions, member function definitions
(``Type::method``), preprocessor macros (``#define``), and call sites.
Template machinery (``template <...>``), forward declarations,
multi-line signatures, and operator overloads are best-effort — scip-
clang fills in the hard cases when a compile database is available.
"""

from __future__ import annotations

import re
from pathlib import Path

from descry.generate import BaseParser, is_generated_source, is_non_project_call


# #include <foo.h>  /  #include "foo.h"
_RE_INCLUDE = re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]')

# #define MACRO value
_RE_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)")

# namespace foo { ...  (including inline/nested namespaces)
_RE_NAMESPACE = re.compile(r"^\s*(?:inline\s+)?namespace\s+([A-Za-z_][A-Za-z0-9_:]*)")

# class Foo : public Bar  /  struct Foo  /  union Foo  /  enum [class] Foo
_RE_CLASS = re.compile(
    r"^\s*(?:template\s*<[^>]*>\s*)?"
    r"(class|struct|union|enum)(?:\s+class)?\s+"
    r"(?:alignas\([^)]*\)\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)

# Function / method definition detection uses a non-regex scanner to
# avoid the catastrophic backtracking that plagued earlier regex
# attempts on real C/C++ files (observed on Redis's dict.c: 2.3k
# lines, >20s timeout). See ``_extract_function_name`` below.
_IDENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _extract_function_name(line: str) -> str | None:
    """If ``line`` looks like a C/C++ function definition header,
    return the function name; otherwise return None.

    Heuristic: the line must end (ignoring trailing whitespace) with
    ``)`` or ``) {`` and must have a prefix before the last `(` that
    contains at least one non-identifier character (i.e. a type or
    qualifier) — that rules out plain call sites like ``foo();``.

    Runs in O(n) without regex so no backtracking is possible.
    """
    s = line.rstrip()
    if not s:
        return None
    # Trim trailing `{`, whitespace
    if s.endswith("{"):
        s = s[:-1].rstrip()
    if not s.endswith(")"):
        return None
    # Find the matching `(` — track depth since args may contain nested
    # parens (function pointers, casts).
    depth = 0
    open_idx = -1
    for i in range(len(s) - 1, -1, -1):
        c = s[i]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx < 0:
        return None
    # The identifier is the word immediately before open_idx.
    j = open_idx - 1
    while j >= 0 and s[j] == " ":
        j -= 1
    end = j + 1
    while j >= 0 and s[j] in _IDENT_CHARS:
        j -= 1
    name = s[j + 1 : end]
    if not name or name[0].isdigit():
        return None
    # Require some type/qualifier prefix before the name — otherwise
    # this is likely a plain call site on a statement line. A single
    # identifier prefix (`void zfree(...)`) is a valid definition; an
    # empty prefix (`zfree(...)` on its own) is not.
    prefix = s[: j + 1].strip()
    if not prefix:
        return None
    return name


# Call site: name(, ns::name(, obj.method(, obj->method(.
# Kept simple to avoid catastrophic backtracking on real C/C++ files
# (observed: the earlier composite pattern took 20+ seconds on
# Redis's dict.c). Template angle-bracket suffixes are accepted by
# the callsite but not captured.
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Za-z_][A-Za-z0-9_]*(?:(?:::|->|\.)[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*\("
)

_C_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "goto",
        "sizeof",
        "alignof",
        "typeid",
        "typedef",
        "using",
        "static_cast",
        "dynamic_cast",
        "const_cast",
        "reinterpret_cast",
        "new",
        "delete",
        "throw",
        "try",
        "catch",
        "noexcept",
        "decltype",
        "auto",
        "void",
        "nullptr",
        "true",
        "false",
        "this",
        "class",
        "struct",
        "enum",
        "union",
        "namespace",
        "template",
        "typename",
        "public",
        "protected",
        "private",
        "virtual",
        "override",
        "final",
        "inline",
        "constexpr",
        "consteval",
        "constinit",
        "static",
        "extern",
        "const",
        "volatile",
        "mutable",
        "register",
        "thread_local",
        "friend",
        "explicit",
        "operator",
        "co_await",
        "co_yield",
        "co_return",
        "requires",
        "concept",
    }
)


def _strip_line_comment(line: str) -> str:
    """Strip trailing ``//`` comments outside string/char literals."""
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


class ClangParser(BaseParser):
    """Regex + brace-depth C/C++ parser."""

    def parse(self, _file_path, rel_path, content):
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

        # Includes + defines — file-level
        for line in lines:
            m_inc = _RE_INCLUDE.match(line)
            if m_inc:
                self.builder.add_edge(file_id, f"MODULE:{m_inc.group(1)}", "IMPORTS")
            m_def = _RE_DEFINE.match(line)
            if m_def:
                name = m_def.group(1)
                macro_id = f"{file_id}::{name}"
                if macro_id not in self.builder.node_registry:
                    self.builder.add_node(
                        macro_id,
                        "Constant",
                        name=name,
                        lineno=lines.index(line) + 1,
                        docstring="",
                        kind="macro",
                    )
                    self.builder.add_edge(file_id, macro_id, "DEFINES")

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

            # Block comments /* ... */
            stripped_leading = raw_line.lstrip()
            if in_block_comment:
                if "*/" in raw_line:
                    in_block_comment = False
                i += 1
                continue
            if stripped_leading.startswith("/*") and "*/" not in raw_line:
                in_block_comment = True
                i += 1
                continue

            line = _strip_line_comment(raw_line)

            # Preprocessor lines — skip type/function detection (but still
            # contribute to brace counting so macros with braces don't
            # mess up nesting).
            if line.lstrip().startswith("#"):
                brace_depth += line.count("{") - line.count("}")
                i += 1
                continue

            # Namespace
            m_ns = _RE_NAMESPACE.match(line)
            if m_ns and "{" in line:
                name = m_ns.group(1)
                parent_id = current_context[-1]
                ns_id = f"{parent_id}::{name}"
                self.builder.add_node(
                    ns_id,
                    "Class",
                    name=name,
                    kind="namespace",
                    lineno=lineno,
                    docstring="",
                )
                self.builder.add_edge(parent_id, ns_id, "DEFINES")
                current_context.append(ns_id)
                context_enter_depth.append(brace_depth)
                class_name_stack.append(name)
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

            # class / struct / union / enum declarations
            m_class = _RE_CLASS.match(line)
            if m_class:
                kind, name = m_class.groups()
                parent_id = current_context[-1]
                type_id = f"{parent_id}::{name}"

                # Allman-style braces: `{` may be on the next line.
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

            # Function / method definitions (non-regex detector).
            fn_name = _extract_function_name(line)
            if fn_name and fn_name not in _C_CONTROL_KEYWORDS:
                parent_id = current_context[-1]
                # Use the containing type as parent if we're inside one,
                # else attach to the file.
                fn_id = f"{parent_id}::{fn_name}"
                if fn_id not in self.builder.node_registry:
                    # Handle out-of-class member def: `Type Class::method(...) {`
                    # Split the name on :: if present.
                    if "::" in fn_name:
                        head, _, tail = fn_name.rpartition("::")
                        class_name = head.split("::")[-1]
                        class_id = f"{parent_id}::{class_name}"
                        if class_id not in self.builder.node_registry:
                            self.builder.add_node(
                                class_id,
                                "Class",
                                name=class_name,
                                kind="class",
                                lineno=lineno,
                                docstring="",
                            )
                            self.builder.add_edge(parent_id, class_id, "DEFINES")
                        fn_id = f"{class_id}::{tail}"
                        node_type = "Method"
                        final_name = tail
                    else:
                        node_type = (
                            "Method" if current_context[-1] != file_id else "Function"
                        )
                        final_name = fn_name
                    end_lineno = (
                        self._find_block_end(lines, i) if "{" in line else lineno
                    )
                    self.builder.add_node(
                        fn_id,
                        node_type,
                        name=final_name,
                        lineno=lineno,
                        end_lineno=end_lineno,
                        token_count=(end_lineno - lineno + 1) * 10,
                        docstring=self.get_leading_docstring(lines, i),
                    )
                    # Attach to the closest appropriate parent.
                    attach_parent = (
                        fn_id.rsplit("::", 1)[0] if "::" in fn_id else file_id
                    )
                    self.builder.add_edge(attach_parent, fn_id, "DEFINES")

            # Calls — skipped for autogenerated sources.
            if skip_calls:
                call_matches = iter(())
            else:
                call_matches = _RE_CALL.finditer(line)
            parent_id = current_context[-1]
            for call_match in call_matches:
                callee = call_match.group(1)
                # Strip the final template angle-brackets to get the simple name.
                simple = callee.split("::")[-1].split(".")[-1].split("->")[-1]
                simple = simple.split("<")[0]
                if simple in _C_CONTROL_KEYWORDS:
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
