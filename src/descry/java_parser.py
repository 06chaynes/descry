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

from descry._comments import strip_block_comments, strip_line_comment
from descry.generate import BaseParser, blank_strings, is_generated_source

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
    r"(?P<returns>[\w<>\[\],?\s.]+?)\s+"  # return type (lazy)
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\("  # name + open paren
)

# Parameter list without nested parentheses, then any `throws` clause.
_PARAMS_AND_THROWS = r"\([^()]*\)\s*(?:throws\s+[\w.,\s]+)?"

# Method declared without a modifier: `Type name(params) {` or, in an
# interface, `Type name(params);`. A statement of that shape is never a call.
_RE_PACKAGE_METHOD = re.compile(
    rf"^\s*(?:<[^>]+>\s+)?(?!(?:return|new|throw|else|case|assert|yield|{_METHOD_MODIFIERS})\b)"
    r"(?P<returns>[\w<>\[\],?.]+)\s+"
    rf"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*{_PARAMS_AND_THROWS}[{{;]\s*$"
)

# Constructor: visibility + CapitalizedName ( — no return type.
_RE_CONSTRUCTOR = re.compile(
    r"^\s*(?:(?:public|protected|private)\s+)"
    r"([A-Z][A-Za-z0-9_$]*)\s*\("
)
# Constructor declared without a modifier: `Name(params) {`.
_RE_PACKAGE_CONSTRUCTOR = re.compile(
    rf"^\s*([A-Z][A-Za-z0-9_$]*)\s*{_PARAMS_AND_THROWS}\{{\s*$"
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
    r"(?<![A-Za-z0-9_$@])"
    r"([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)"
    r"\s*\("
)

_DECLARATIONS = (
    # `Type name` before `=`, `;`, `,`, `)` or `:` — a local, field,
    # parameter or for-each variable
    re.compile(
        r"\b(?P<type>[A-Z][\w.]*)(?:<[^;(){}=]*>)?(?:\[\])*(?:\.\.\.)?\s+"
        r"(?P<name>[A-Za-z_$][\w$]*)\s*(?=[=;,):])"
    ),
    # A type parameter's bound: `<T extends Type`
    re.compile(r"[<,]\s*(?P<param>[A-Z]\w*)\s+extends\s+(?P<type>[A-Z][\w.]*)"),
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


_RE_NEW_GENERIC = re.compile(r"\bnew\s+[A-Za-z_$][A-Za-z0-9_$.]*\s*<")
_RE_BODY_OPENS = re.compile(r"\)\s*\{\s*$")
_RE_NEW_ENDS = re.compile(r"\bnew\s+[A-Za-z_$][A-Za-z0-9_$.]*\s*$")
_RE_EMPTY_BODY = re.compile(r"(?<=\))\s*\{\s*\}(?=\s*\.)")


def _plain_instantiations(code: str) -> str:
    """Return ``code`` with the type arguments of every `new T<...>(` and
    every empty anonymous class body before a `.` turned to spaces, so that
    the instantiation reads as a call of `T` and a call on it as a chain."""
    for match in _RE_NEW_GENERIC.finditer(code):
        start, depth = match.end() - 1, 0
        for end in range(start, len(code)):
            char = code[end]
            depth += (char == "<") - (char == ">")
            if depth == 0:
                code = code[:start] + " " * (end + 1 - start) + code[end + 1 :]
                break
            if not (char.isalnum() or char in "_$.,?[]<> \t"):
                break
    return _RE_EMPTY_BODY.sub(lambda match: " " * len(match.group()), code)


class JavaParser(BaseParser):
    """Regex-driven Java source parser.

    Pattern mirrors ``RustParser``: walk lines, maintain a context stack
    keyed off matching braces, emit File → Class/Interface/Enum/Record →
    Method/Field → CALLS edges.
    """

    binding_re = re.compile(r"\bvar\s+(?P<name>\w+)\s*=(?!=)")

    def _node_type(self, node_id):
        """Type of the node of this file with ID ``node_id``, if there is one."""
        if node_id in self.builder.node_registry:
            for node in reversed(self.builder.nodes):
                if node["id"] == node_id:
                    return node["type"]
        return None

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

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
            self.set_file_metadata(file_id, "java_package", package)

        # Imports
        for line in lines:
            m = _RE_IMPORT.match(line)
            if m:
                path = m.group(1).rstrip(".*").rstrip(".")
                if path:
                    self.builder.add_edge(file_id, f"MODULE:{path}", "IMPORTS")
                    if not m.group(1).endswith("*"):
                        name = path.rpartition(".")[2]
                        self.builder.imports.setdefault(file_id, {})[name] = path

        self.scan_declarations(lines, _DECLARATIONS)

        # Walk for types, methods, fields, and calls. A type
        # (class/interface/enum/record) is the context for the definitions
        # up to its closing brace; a method or constructor is the source of
        # the calls in its body.
        self.enter_context(file_id, 0, len(lines))
        bodies: list[tuple[str, int, int]] = []  # open methods: ID, first, last line
        anonymous: dict[int, int] = {}  # last line of an anonymous body -> `new` edge

        i = 0
        n = len(lines)
        in_block_comment = False
        in_text_block = None
        while i < n:
            raw_line = lines[i]
            lineno = i + 1
            context_id, context_start = self.context_at(lineno)
            while bodies and lineno > bodies[-1][2]:
                bodies.pop()
            # Call scanning starts past a name declared on the line.
            call_scan_start = 0

            # Block comments (/* ... */ or /** ... */) are removed so
            # annotations / method signatures inside Javadoc don't leak into
            # the graph.
            raw_line, in_block_comment = strip_block_comments(
                raw_line, in_block_comment
            )

            # Declarations are matched on `line`, which has no trailing
            # `// comment`; calls are read from `code`, which also has its
            # string literals blanked.
            line = strip_line_comment(raw_line, backticks=True)
            was_in_text_block = in_text_block
            code, in_text_block = blank_strings(raw_line, in_text_block, ('"""',))
            if was_in_text_block or in_text_block:
                # The text of a text block declares nothing.
                line = code

            # 1. Type declarations (class/interface/enum/record/@interface)
            m_type = _RE_TYPE_DECL.match(line)
            if m_type:
                kind, name = m_type.groups()
                parent_id = context_id
                type_id = f"{parent_id}::{name}"

                # Allman brace style — look ahead for the `{`.
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
                    kind=kind,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    token_count=(end_lineno - lineno + 1) * 10,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if has_open_brace and end_lineno > brace_line_idx + 1:
                    self.enter_context(type_id, lineno, end_lineno)
                i = brace_line_idx + 1 if has_open_brace else i + 1
                continue

            # 2. Methods and constructors (only inside a type context)
            if context_id != file_id:
                m_method = _RE_METHOD.match(line) or _RE_PACKAGE_METHOD.match(line)
                if m_method:
                    name = m_method["name"]
                    if name not in _JAVA_CONTROL_KEYWORDS:
                        call_scan_start = m_method.end()
                        parent_id = context_id
                        method_id = f"{parent_id}::{name}"
                        end_lineno = (
                            self._find_block_end(lines, i) if "{" in line else lineno
                        )
                        docstring = self.get_leading_docstring(lines, i)
                        # A field of the same name moves to `<type>::field_<name>`.
                        if self._node_type(method_id) == "Constant":
                            self.rename_node(method_id, f"{parent_id}::field_{name}")
                        self.builder.add_node(
                            method_id,
                            "Method",
                            name=name,
                            lineno=lineno,
                            end_lineno=end_lineno,
                            token_count=(end_lineno - lineno + 1) * 10,
                            docstring=docstring,
                            return_type=m_method["returns"].strip(),
                        )
                        self.builder.add_edge(parent_id, method_id, "DEFINES")
                        if "{" in line:
                            bodies.append((method_id, lineno, end_lineno))
                else:
                    m_ctor = _RE_CONSTRUCTOR.match(line)
                    m_ctor = m_ctor or _RE_PACKAGE_CONSTRUCTOR.match(line)
                    if m_ctor:
                        name = m_ctor.group(1)
                        if name == context_id.rpartition("::")[2]:
                            call_scan_start = m_ctor.end()
                            parent_id = context_id
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
                            if "{" in line:
                                bodies.append((ctor_id, lineno, end_lineno))

                # 3. Fields (only inside types, not methods)
                # Skip if line ended in `{` — that's a method/type decl with inline body.
                if "{" not in line and ";" in line:
                    m_field = _RE_FIELD.match(line)
                    if m_field:
                        name = m_field.group(1)
                        # Filter common false positives (keywords and control flow
                        # would already be caught by _METHOD_MODIFIERS requirement).
                        parent_id = context_id
                        field_id = f"{parent_id}::{name}"
                        # A method of the same name keeps `<type>::<name>`.
                        if self._node_type(field_id) == "Method":
                            field_id = f"{parent_id}::field_{name}"
                        if field_id not in self.builder.node_registry:
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
            if not skip_calls and context_id != file_id:
                # The innermost of the enclosing type and the open method.
                parent_id = context_id
                if bodies and bodies[-1][1] >= context_start:
                    parent_id = bodies[-1][0]
                code = _plain_instantiations(code)
                self.begin_line(code, lineno, raw=raw_line)
                if lineno in anonymous and code.lstrip().startswith("}"):
                    self.continue_chain(code.index("}"), anonymous.pop(lineno))
                for call_match in _RE_CALL.finditer(code, call_scan_start):
                    callee = call_match.group(1)
                    simple_name = callee.split(".")[-1]
                    if simple_name in _JAVA_CONTROL_KEYWORDS:
                        continue
                    self.add_call(
                        parent_id,
                        callee,
                        lineno,
                        0,
                        call_match.start(1),
                        call_match.end() - 1,
                        # `Type.class` is a java.lang.Class.
                        f"Class().{simple_name}" if ".class." in callee else None,
                    )
                # `new T(...) {` opens the body of an anonymous subclass; a
                # call on its closing brace is made on the new object.
                body = _RE_BODY_OPENS.search(code)
                created = body and self.call_closed_at(body.start())
                if created and _RE_NEW_ENDS.search(code[: created[1]]):
                    anonymous[self._find_block_end(lines, i)] = created[0]

            i += 1
