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
from descry.generate import (
    _JAVA_BOXES,
    BaseParser,
    _bracketed_parts,
    _strip_generics,
    blank_strings,
    is_generated_source,
    source_lines,
)

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
# What a type's body opens after, and a type a line may go on to after the
# end of another
_TYPE_HEAD_RE = re.compile(
    rf"\s*(?:@[\w.]+\s+)*(?:{_TYPE_MODIFIERS}\s+)*(?:class|interface|enum|record|@interface)\b"
)
_DEFINITION_HEAD_RE = re.compile(
    rf"(?:{_TYPE_MODIFIERS}\s+)*(?:class|interface|enum|record|@interface)\b"
)

# Method declaration. Requires at least one visibility/modifier so we don't
# match bare control-flow like `if (cond)`. Does NOT match constructors
# (those lack a return type; handled separately).
_METHOD_MODIFIERS = r"(?:public|protected|private|static|final|abstract|synchronized|default|native|strictfp)"
# Type arguments or parameters, nested three deep
_PLAIN = r"[^<>;(){}=]*"
_TYPE_ARGUMENTS = rf"<{_PLAIN}(?:<{_PLAIN}(?:<{_PLAIN}>{_PLAIN})*>{_PLAIN})*>"
# Annotations written on the line of the declaration they are on
_ANNOTATIONS = r"(?:@[\w.]+(?:\([^()]*\))?\s+)*"
_RE_METHOD = re.compile(
    rf"^\s*{_ANNOTATIONS}(?:(?:{_METHOD_MODIFIERS})\s++)+"
    rf"(?:{_TYPE_ARGUMENTS}\s++)?"  # optional generic type params
    r"(?:@[\w.]+\s++)*"  # annotations of the return type
    # The return type, from a non-space to a non-space, shortest first
    r"(?P<returns>(?!<)[\w<>\[\],?.](?:[\w<>\[\],?\s.]*?[\w<>\[\],?.])??)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\("  # name + open paren
)

# Parameter list without nested parentheses, then any `throws` clause or,
# on an annotation's element, its `default` value.
_PARAMS_AND_THROWS = r"\([^()]*\)\s*(?:throws\s+[\w.,\s]+|default\s+[^;{]+)?"

# Method declared without a modifier: `Type name(params) {` or, in an
# interface, `Type name(params);`. A statement of that shape is never a call.
_STATEMENT_WORDS = (
    "return",
    "new",
    "throw",
    "else",
    "case",
    "when",
    "assert",
    "yield",
    "do",
)
_PACKAGE_METHOD = (
    rf"^\s*(?:{_TYPE_ARGUMENTS}\s+)?(?!(?:{'|'.join(_STATEMENT_WORDS)}|{_METHOD_MODIFIERS})\b)"
    rf"(?P<returns>[\w.]+(?:{_TYPE_ARGUMENTS})?(?:\[\])*)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
)
_RE_PACKAGE_METHOD = re.compile(rf"{_PACKAGE_METHOD}{_PARAMS_AND_THROWS}(?:\{{|;\s*$)")
_RE_PACKAGE_METHOD_OPENS = re.compile(rf"{_PACKAGE_METHOD}\(")

# Constructor: visibility + CapitalizedName ( — no return type.
_RE_CONSTRUCTOR = re.compile(
    rf"^\s*(?:(?:public|protected|private)\s+)(?:{_TYPE_ARGUMENTS}\s+)?"
    r"([A-Z][A-Za-z0-9_$]*)\s*\("
)
# Constructor declared without a modifier: `Name(params) {`.
_RE_PACKAGE_CONSTRUCTOR = re.compile(
    rf"^\s*([A-Z][A-Za-z0-9_$]*)\s*{_PARAMS_AND_THROWS}\{{"
)

# Field declaration: modifiers + type + name + (= value)? ;
_RE_FIELD = re.compile(
    rf"^\s*{_ANNOTATIONS}(?:(?:{_METHOD_MODIFIERS}|volatile|transient)\s++)+"
    rf"(?!{_METHOD_MODIFIERS}\b)(?P<type>[\w<>\[\],?.](?:[\w<>\[\],?\s.]*?[\w<>\[\],?.])??)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:=|;|$)"
)
# Field declared without a modifier: `Type name;` or `Type name = value`.
_RE_PACKAGE_FIELD = re.compile(
    r"^\s*(?!(?:return|throws?|new|else|case|assert|yield|default|package|import)\b)"
    rf"(?P<type>[\w.]+(?:{_TYPE_ARGUMENTS})?(?:\[\])*)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:=(?!=)|;)"
)
# An enum constant, alone on its line behind its annotations: `NAME,`,
# `NAME(args),`, `@Deprecated NAME {`
_RE_ENUM_CONSTANT = re.compile(
    r"^\s*(?:@[\w.]+(?:\([^)]*\))?\s+)*"
    r"([A-Z][A-Z0-9_]*)\s*+(?:\(.*\))?\s*+(?:\{.*\}\s*+)?[,;{]?\s*$"
)
# The word before a name: `Type name(` declares `name` where it is no
# statement's.
_RE_WORD_BEFORE = re.compile(r"([\w$]+)\s+$")
# `(Type) value` inside the parentheses before a member: a cast gives the
# value that type
_RE_CAST = re.compile(r"\(\s*([A-Z][\w.]*(?:<[^()]*>)?(?:\[\])*)\s*\)\s*[\w(\"]")

# Call site: identifier(...) or receiver.method(...).
# The lookbehind excludes:
#   - Other identifier chars (so we don't match the tail of `foo.bar`)
#   - `@` (so annotations `@JsonProperty(...)` are NOT matched as calls)
_RE_CALL = re.compile(
    r"(?<![A-Za-z0-9_$@])"
    r"([A-Za-z_$][A-Za-z0-9_$]*(?:\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*)"
    r"\s*\("
)

_PRIMITIVES = "boolean|byte|char|short|int|long|float|double"
_DECLARATIONS = (
    # `Type name` before `=`, `;`, `,`, `)`, `:`, `&&` or `->`: a local, a
    # field, a parameter, a for-each variable or what a pattern binds. The
    # type is read with its type arguments, and an array of any type.
    re.compile(
        rf"\b(?P<type>[A-Z][\w.]*(?:{_TYPE_ARGUMENTS})?(?:\[\])*|(?:{_PRIMITIVES})(?:\[\])+)"
        r"\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?=[=;,):]|&&|->)"
    ),
    # A type parameter's bound: `<T extends Type`
    re.compile(
        r"[<,]\s*(?P<param>[A-Z]\w*)\s+extends\s+"
        r"(?P<type>[A-Z][\w.]*(?:<(?:[^<>]|<[^<>]*>)*>)?)(?![<(])"
    ),
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


# The parameter of a lambda that has one, or the list of them, without types
_LAMBDA = r"(?<![\w$.])(?<!\bcase\s)([\w$]+)\s*->|\(([\w$\s,]+)\)\s*->"
_RE_LAMBDA = re.compile(_LAMBDA)
# The name a `var` binds, and those a lambda does
_RE_UNTYPED = re.compile(rf"\bvar\s+([\w$]+)|{_LAMBDA}")
# `receiver::name`, a reference to a method or with `new` to a constructor
_RE_METHOD_REFERENCE = re.compile(r"(?<![\w$.])([\w$.]+)\s*::\s*([A-Za-z_$][\w$]*)")
_RE_NEW_GENERIC = re.compile(r"(?:\bnew\s+[A-Za-z_$][A-Za-z0-9_$.]*|\.)\s*<")
_RE_BODY_OPENS = re.compile(r"\)\s*\{\s*$")


def _type_params(code, start=0):
    """The type parameter names a declaration writes in `<...>` at
    ``start``: `interface Function<F, T>` gives `("F", "T")`. A parameter's
    bound may itself be generic — `B extends Pair<T, B>` still names `B`."""
    lead = re.match(r"\s*<", code[start:])
    if not lead:
        return ()
    depth, end = 0, None
    for i, char in enumerate(code[start + lead.end() :], start + lead.end()):
        depth += char == "<"
        depth -= char == ">"
        if depth < 0:
            end = i
            break
    if end is None:
        return ()
    inner = code[start + lead.end() : end]
    return tuple(
        part.strip().split(" extends ")[0].split()[-1]
        for part in _bracketed_parts(inner)
        if part.strip()
    )


def _param_writings(code, start):
    """The type each parameter between the `(` at ``start`` and its `)`
    declares, in order."""
    depth = 0
    for end, char in enumerate(code[start:], start):
        depth += char == "("
        depth -= char == ")"
        if not depth:
            break
    else:
        return ()
    writings = []
    for part in _bracketed_parts(code[start + 1 : end] + ")"):
        part = re.sub(r"@\w+(\([^)]*\))?\s+", "", part.strip())
        part = re.sub(r"\bfinal\s+", "", part).rstrip()
        if " " in part:
            writings.append(part.rpartition(" ")[0])
        elif part:
            writings.append(part)
    return tuple(writings)


# `path[index].` before a member: an element of an array
_RE_ELEMENT = re.compile(r"(?<![\w$.])([\w$.]+)((?:\[[^\[\]]*\])+)\s*\.\s*$")
# `new T` or `outer.new T` before the `(` of an instantiation
_RE_NEW_ENDS = re.compile(
    r"(?:([\w$.]+)\s*\.\s*)?\bnew\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*$"
)
_RE_EMPTY_BODY = re.compile(r"(?<=\))\s*\{\s*\}(?=\s*\.)")


def _plain_instantiations(code: str) -> str:
    """Return ``code`` with the type arguments of every `new T<...>(` and
    of every `.<...>name(`, and every empty anonymous class body before a
    `.`, turned to spaces, so that the instantiation reads as a call of
    `T`, the call as one of `name`, and a call on either as a chain."""
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

    @staticmethod
    def bindings(param):
        """Yield ``(name, steps)`` for a lambda's parameter that is written
        without its type."""
        if re.fullmatch(r"\s*[\w$]+\s*", param):
            yield param.strip(), ()

    def _anonymous(self, parent_id, base, lineno, last):
        """Open the body, lines ``lineno`` to ``last``, of a class source
        cannot name: a subclass of ``base``, a node or a `REF:` to a name,
        made in ``parent_id``. Its members are its own."""
        anonymous_id = self._nameless_owner(parent_id, f"class@{lineno}", lineno, last)
        self.builder.add_edge(anonymous_id, base, "INHERITS")
        self.enter_context(anonymous_id, lineno + 1, last - 1)

    def call_origin(self, callee, lineno, since, owner=None):
        """Origin path of a callee that starts at a name. A local stands
        for its declared type or the call it was bound to, `?` where it has
        neither; `this` is `self`, `super` the supertype of the anonymous
        class the line is in, else `super()`, and `Outer.this` an `Outer`.
        Any other name is left to be read in scope."""
        root, _, path = callee.partition(".")
        if not path:
            return None
        if root == "this":
            return f"self.{path}"
        if root == "super":
            return f"{self._supers[-1][1] if self._supers else 'super'}().{path}"
        if path.startswith("this."):
            return f"{root}().{path[5:]}"
        typed = [d[0] for d in self.declared.get(root, ()) if since <= d[0] <= lineno]
        bound = [n for n in self._untyped.get(root, ()) if since <= n <= lineno]
        if max(bound, default=0) > max(typed, default=0):
            return f"?.{path}"
        if not typed:
            return f"{self._bound(root)}.{path}"
        head = super().call_origin(f"{root}.x", lineno, since)
        return f"{head[:-2] if head else '?'}.{path}"

    def _bound(self, written):
        """The type ``written``, or where the file bounds a type parameter
        of that name, its bound. A parameter bounded by its own bound's
        arguments — `B extends Builder<B>` — is a self-type: it stays
        written so the resolver can substitute the receiver's."""
        bound = self._declared(f"<{written}>", 0, 0, True)
        if bound and re.search(rf"<[^<>]*\b{written}\b", bound):
            return written
        return bound or written

    def _lambda_target(self, before):
        """The type a lambda at the end of ``before`` is assigned to or
        cast as, as a `>` origin root — None where it is an argument of a
        call, whose parameters the callee's give it."""
        cast = re.search(r"\(\s*([A-Z][\w.$]*)\s*(<[^<>;()]*>)?\s*\)\s*\(?\s*$", before)
        if cast:
            written = "".join((cast[2] or "").split())
            return f">{cast[1].replace('.', '::')}{written}"
        assigned = re.search(
            r"(?<![\w.])([A-Z][\w.$]*(?:<[^<>;()]*>)?)\s+[\w$]+\s*=\s*$", before
        )
        if assigned:
            return f">{assigned[1].replace('.', '::')}"
        return None

    def _bind_closures(self, code, index, lines, before):
        """Type the parameters of a lambda that opens before ``before``
        as the target of an assignment or a cast, from the functional
        interface it is for; arguments of calls go on to the shared path."""
        while self._closures and self._closures[0][0] < before:
            bar, params, _end = self._closures[0]
            origin = self._lambda_target(code[:bar])
            if origin is None:
                break
            self._closures.pop(0)
            bindings = {
                name: ".".join([f"{origin}.^{number}", *steps])
                for number, param in enumerate(_bracketed_parts(params + ")"))
                for name, steps in self.bindings(param)
            }
            self._bind(bindings, index, lines, code.rstrip().endswith("{"))
        super()._bind_closures(code, index, lines, before)

    def _has_body(self, lines, index, code):
        """Whether the method declared on ``lines[index]`` has a body: a
        `{` on that line, ``code`` without its comments, or below it where
        its signature goes on, before any `;`."""
        if "{" in code:
            return True
        brace = self.opening_brace_line(lines, index, code)
        return brace is not None and not any(
            ";" in strip_line_comment(text) for text in lines[index:brace]
        )

    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)

        lines = source_lines(content)
        lines = self.split_source(
            lines,
            self.blank_lines(lines, ('"""',)),
            _DEFINITION_HEAD_RE,
            _TYPE_HEAD_RE,
        )
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
                    # An on-demand import is kept under its own text.
                    name = path.rpartition(".")[2]
                    if m.group(1).endswith("*"):
                        name = m.group(1)
                    self.builder.imports.setdefault(file_id, {})[name] = path

        self.scan_declarations(
            [line.replace("...", "[]") for line in lines], _DECLARATIONS
        )
        # name -> the lines where a `var` or a lambda binds it without a
        # type
        self._untyped = {}
        for lineno, line in enumerate(lines, 1):
            for found in _RE_UNTYPED.finditer(line):
                for param in (found[1] or found[2] or found[3]).split(","):
                    for name in param.split()[-1:]:
                        self._untyped.setdefault(name, []).append(lineno)
        # (last line, type) of each anonymous class body the line is in
        self._supers = []

        # Walk for types, methods, fields, and calls. A type
        # (class/interface/enum/record) is the context for the definitions
        # up to its closing brace; a method or constructor is the source of
        # the calls in its body.
        self.enter_context(file_id, 0, len(lines))
        bodies: list[tuple[str, int, int]] = []  # open methods: ID, first, last line
        anonymous: dict[int, int] = {}  # last line of an anonymous body -> `new` edge
        enums: set[str] = set()  # IDs of the enum types

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
            while self._supers and lineno > self._supers[-1][0]:
                self._supers.pop()
            # The open bodies of the enclosing type, outermost first.
            inside = [body for body in bodies if body[1] >= context_start]
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
                    generics=_type_params(line, m_type.end()),
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                header = [line, *lines[i + 1 : (brace_line_idx or i) + 1]]
                header = _strip_generics(" ".join(header)).partition("{")[0]
                for clause in re.split(r"\b(?:extends|implements)\b", header)[1:]:
                    for base in clause.split(","):
                        if base.split():
                            self.builder.add_edge(
                                type_id, f"REF:{base.split()[0]}", "INHERITS"
                            )
                if kind == "enum":
                    enums.add(type_id)
                if has_open_brace and end_lineno > brace_line_idx + 1:
                    self.enter_context(type_id, lineno, end_lineno)
                i = brace_line_idx + 1 if has_open_brace else i + 1
                continue

            # 2. Methods and constructors (only inside a type context)
            if context_id != file_id:
                m_method = _RE_METHOD.match(line) or _RE_PACKAGE_METHOD.match(line)
                if not m_method and _RE_PACKAGE_METHOD_OPENS.match(line):
                    # `Type name(params)` whose parameters or whose `throws`
                    # clause go on over the next lines.
                    signature = line.rstrip()
                    for following in map(str.strip, lines[i + 1 : i + 10]):
                        listed = signature.count("(") <= signature.count(")")
                        if listed and not following.startswith(("throws ", "{")):
                            break
                        signature += f" {following}"
                    m_method = _RE_PACKAGE_METHOD.match(signature)
                if m_method:
                    name = m_method["name"]
                    if name not in _JAVA_CONTROL_KEYWORDS:
                        call_scan_start = m_method.end()
                        parent_id = context_id
                        method_id = f"{parent_id}::{name}"
                        has_body = self._has_body(lines, i, line)
                        end_lineno = (
                            self._find_block_end(lines, i) if has_body else lineno
                        )
                        docstring = self.get_leading_docstring(lines, i)
                        returns = self._bound(m_method["returns"].strip())
                        declared = self._node(method_id)
                        # A field of the same name moves to `<type>::field_<name>`.
                        if declared and declared["type"] == "Constant":
                            self.rename_node(method_id, f"{parent_id}::field_{name}")
                        elif declared and declared["type"] == "Method":
                            # Overloads that return something return the
                            # type they agree on, a primitive as its box,
                            # with its type arguments where those agree.
                            known = declared["metadata"].get("return_type") or ""
                            heads = (known.partition("<")[0], returns.partition("<")[0])
                            agreed = {_JAVA_BOXES.get(head, head) for head in heads}
                            if known == "void":
                                known = returns
                            elif returns not in (known, "void"):
                                known = agreed.pop() if len(agreed) == 1 else None
                            declared["metadata"]["return_type"] = known
                        head = m_method[0]
                        self.builder.add_node(
                            method_id,
                            "Method",
                            name=name,
                            lineno=lineno,
                            end_lineno=end_lineno,
                            token_count=(end_lineno - lineno + 1) * 10,
                            docstring=docstring,
                            return_type=returns,
                            params=_param_writings(
                                m_method.string,
                                m_method.string.index("(", m_method.end("name")),
                            ),
                            static=bool(re.search(r"\bstatic\b", head)),
                            default=bool(re.search(r"\bdefault\b", head)),
                            body=has_body,
                        )
                        self.builder.add_edge(parent_id, method_id, "DEFINES")
                        if has_body:
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
                            has_body = self._has_body(lines, i, line)
                            end_lineno = (
                                self._find_block_end(lines, i) if has_body else lineno
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
                            if has_body:
                                bodies.append((ctor_id, lineno, end_lineno))

                # 3. Fields (only inside types, not methods)
                if not inside:
                    m_field = _RE_FIELD.match(line) or _RE_PACKAGE_FIELD.match(line)
                    if m_field:
                        name = m_field["name"]
                        # Filter common false positives (keywords and control flow
                        # would already be caught by _METHOD_MODIFIERS requirement).
                        parent_id = context_id
                        field_id = f"{parent_id}::{name}"
                        # A method of the same name keeps `<type>::<name>`.
                        if (self._node(field_id) or {}).get("type") == "Method":
                            field_id = f"{parent_id}::field_{name}"
                        if field_id not in self.builder.node_registry:
                            self.builder.add_node(
                                field_id,
                                "Constant",
                                name=name,
                                lineno=lineno,
                                docstring="",
                                value_type=self._bound(m_field["type"].strip()),
                            )
                            self.builder.add_edge(parent_id, field_id, "DEFINES")
                m_constant = context_id in enums and _RE_ENUM_CONSTANT.match(code)
                if m_constant and not inside:
                    call_scan_start = m_constant.end(1)
                    constant_id = f"{context_id}::{m_constant[1]}"
                    self.builder.add_node(
                        constant_id,
                        "Constant",
                        name=m_constant[1],
                        lineno=lineno,
                        docstring="",
                        value_type=context_id.rpartition("::")[2],
                    )
                    self.builder.add_edge(context_id, constant_id, "DEFINES")
                    # A `{` that ends the line, or starts the next where
                    # nothing ends the constant, opens its body: a subclass
                    # of the enum.
                    opened = code.rstrip()[-1:]
                    below = lines[i + 1].lstrip() if i + 1 < n else ""
                    if opened == "{" or opened not in ",;}" and below.startswith("{"):
                        last = self._find_block_end(lines, i + (opened != "{"))
                        self._anonymous(constant_id, context_id, lineno, last)

            # 4. Calls. ast-grep has no Java backend today, so we always
            #    run the regex extractor (rather than gating on
            #    ``not USE_AST_GREP`` like RustParser/TSParser do).
            if not skip_calls and context_id != file_id:
                # The innermost of the enclosing type and the open method.
                parent_id = context_id
                if bodies and bodies[-1][1] >= context_start:
                    parent_id = bodies[-1][0]
                # A local is declared from the first line of the outermost
                # body the line is in, or on the line itself. A class written
                # in a method reads that method's locals too.
                since = bodies[0][1] if bodies else lineno
                written = code
                code = _plain_instantiations(code)
                self.begin_line(code, lineno, raw=raw_line)
                if lineno in anonymous and code.lstrip().startswith("}"):
                    self.continue_chain(code.index("}"), anonymous.pop(lineno))
                # The lambdas on the line whose parameters are still to be
                # read: those in the parentheses of a call, one the line
                # starts with, an argument of a call left open, and one
                # assigned or cast to a type.
                around = [*self._pairs.items(), *((o, len(code)) for o in self._opens)]
                self._closures = [
                    (found.start(), found[1] or found[2], found.end())
                    for found in _RE_LAMBDA.finditer(code if "->" in code else "")
                    if not code[: found.start()].strip()
                    or any(o < found.start() < c for o, c in around)
                    or self._lambda_target(code[: found.start()])
                ]
                self._bodies = []
                for call_match in _RE_CALL.finditer(code, call_scan_start):
                    # The callee without the type arguments blanked in it.
                    callee = "".join(call_match.group(1).split())
                    simple_name = callee.split(".")[-1]
                    if simple_name in _JAVA_CONTROL_KEYWORDS:
                        continue
                    before = code[: call_match.start(1)]
                    typed = _RE_WORD_BEFORE.search(before)
                    if typed and typed[1] not in _STATEMENT_WORDS:
                        # `Type name(...) {...}` in a body written on one line.
                        continue
                    self._bind_closures(code, i, lines, call_match.start(1))
                    paren = call_match.end() - 1
                    # The parentheses before the callee that end no call.
                    receiver = before.rstrip()
                    close = len(receiver) - 2 if receiver.endswith(").") else None
                    grouped = [o for o, c in self._pairs.items() if c == close]
                    cast = (
                        grouped
                        and not self.call_closed_at(close)
                        and _RE_CAST.match(code, grouped[0] + 1)
                    )
                    made = _RE_NEW_ENDS.search(code[:paren])
                    element = "]" in before and _RE_ELEMENT.search(before)
                    literal = raw_line[: len(before)].rstrip().endswith('".')
                    if ".class." in callee:
                        # `Type.class` is a java.lang.Class.
                        origin = f"Class().{simple_name}"
                    elif cast:
                        origin = f"{cast[1].replace('.', '::')}().{callee}"
                    elif literal:
                        origin = f"String().{callee}"
                    elif element:
                        # What an array holds is its type's first argument.
                        array = self.call_origin(f"{element[1]}.x", lineno, since)
                        held = ".@0" * element[2].count("[")
                        origin = f"{array[:-2]}{held}.{callee}"
                    elif made and made[1]:
                        # `outer.new T(...)` makes a member type of `outer`.
                        origin = self.call_origin(f"{made[1]}.{callee}", lineno, since)
                    elif made:
                        # `new T<A>(...)` is a call of the type `T<A>`.
                        arguments = written[call_match.end(1) : paren].strip()
                        origin = f"{callee}{arguments}".replace(".", "::")
                    elif callee != call_match.group(1):
                        origin = self.call_origin(callee, lineno, since)
                    else:
                        origin = None
                    self.add_call(
                        parent_id,
                        callee,
                        lineno,
                        since,
                        call_match.end(1) - len(callee),
                        paren,
                        origin,
                    )
                self._bind_closures(code, i, lines, len(code))
                references = "::" in code and _RE_METHOD_REFERENCE.finditer(code)
                for reference in references or ():
                    # `T::new` is a call of the type `T`, any other one of
                    # the method `receiver.name`.
                    receiver, name = reference.groups()
                    if name == "new":
                        callee, start = receiver, reference.start(1)
                        origin = receiver.replace(".", "::")
                    else:
                        callee = f"{receiver}.{name}"
                        start = reference.start(2) - len(receiver) - 1
                        origin = self.call_origin(callee, lineno, since)
                    self.add_call(parent_id, callee, lineno, since, start, -1, origin)
                # `new T(...) {` opens the body of an anonymous subclass; a
                # call on its closing brace is made on the new object.
                body = _RE_BODY_OPENS.search(code)
                created = body and self.call_closed_at(body.start())
                subclassed = created and _RE_NEW_ENDS.search(code[: created[1]])
                if subclassed:
                    last = self._find_block_end(lines, i)
                    anonymous[last] = created[0]
                    self._supers.append((last, subclassed[2].replace(".", "::")))
                    base = f"REF:{subclassed[2]}"
                    self._anonymous(parent_id, base, lineno, last)

            # A block opened outside every method is a body of the type's.
            opened = bodies and bodies[-1][1] == lineno
            if not inside and not opened and code.rstrip().endswith("{"):
                bodies.append((context_id, lineno, self._find_block_end(lines, i)))

            i += 1
        self.restore_source_lines()
