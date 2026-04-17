"""Optional tree-sitter-backed parser for TypeScript / TSX / JavaScript.

Activated when the ``descry-codegraph[ast]`` extra is installed (providing
``tree-sitter``, ``tree-sitter-typescript``, ``tree-sitter-javascript``) and
the user opts in via ``.descry.toml``::

    [features]
    use_tree_sitter_ts = true

When enabled, ``generate.TSParser`` routes file parsing through this module
and falls back to regex-only extraction for files the AST walker cannot
process. When disabled or when the extra is missing, behaviour is
indistinguishable from the regex-only baseline.

This parser extracts the symbol shapes needed for the call graph:

- ``class_declaration`` / ``interface_declaration`` / ``enum_declaration``
- ``function_declaration``
- ``method_definition`` (including ``get``/``set``) inside a class body
- Arrow / function-expression RHS of ``const foo = …``
- ``import_statement`` (ES) and ``call_expression`` (call edges)

Anything the grammar hits as ``ERROR`` is surfaced to the caller; the
BaseParser fallback can then re-run regex-only extraction on that file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import tree_sitter

logger = logging.getLogger(__name__)


# --- Optional dependency bootstrap -----------------------------------------

_TS_AVAILABLE: bool | None = None
_TS_LANGUAGE = None
_TSX_LANGUAGE = None
_JS_LANGUAGE = None


def tree_sitter_available() -> bool:
    """True iff tree-sitter + the TS/JS grammars are importable.

    Result is cached on first call.
    """
    global _TS_AVAILABLE, _TS_LANGUAGE, _TSX_LANGUAGE, _JS_LANGUAGE
    if _TS_AVAILABLE is not None:
        return _TS_AVAILABLE
    try:
        import tree_sitter as ts
        import tree_sitter_typescript as tsts
        import tree_sitter_javascript as tsjs

        _TS_LANGUAGE = ts.Language(tsts.language_typescript())
        _TSX_LANGUAGE = ts.Language(tsts.language_tsx())
        _JS_LANGUAGE = ts.Language(tsjs.language())
        _TS_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("tree-sitter unavailable: %s", exc)
        _TS_AVAILABLE = False
    return _TS_AVAILABLE


def _language_for(file_path: str | Path):
    """Return the tree-sitter Language for a given TS/TSX/JS path, or None."""
    if not tree_sitter_available():
        return None
    suffix = Path(file_path).suffix.lower()
    if suffix == ".tsx":
        return _TSX_LANGUAGE
    if suffix in (".ts", ".d.ts"):
        return _TS_LANGUAGE
    if suffix in (".js", ".jsx", ".mjs", ".cjs"):
        return _JS_LANGUAGE
    return None


# --- Symbol data classes ---------------------------------------------------


SymbolKind = Literal[
    "Class",
    "Interface",
    "Enum",
    "Function",
    "Method",
    "Const",
    "Import",
    "Call",
]


@dataclass
class ExtractedSymbol:
    kind: SymbolKind
    name: str
    lineno: int
    end_lineno: int
    parent_name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    is_async: bool = False
    is_static: bool = False
    accessor: Literal["get", "set", None] | None = None
    module: str | None = None  # populated for Import
    imported_names: list[str] = field(default_factory=list)  # for Import


@dataclass
class ParseResult:
    """Symbols + raw call sites extracted from a single file."""

    symbols: list[ExtractedSymbol] = field(default_factory=list)
    calls: list[tuple[int, str]] = field(default_factory=list)  # (lineno, callee)
    had_errors: bool = False


# --- Extractor -------------------------------------------------------------


def _node_text(node: "tree_sitter.Node") -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _first_child_of_type(
    node: "tree_sitter.Node", type_name: str
) -> "tree_sitter.Node | None":
    for child in node.named_children:
        if child.type == type_name:
            return child
    return None


def _line(node: "tree_sitter.Node") -> int:
    return node.start_point[0] + 1


def _end_line(node: "tree_sitter.Node") -> int:
    return node.end_point[0] + 1


def _walk(
    node: "tree_sitter.Node",
    on_enter,
    parent_name: str | None = None,
) -> None:
    """Depth-first traversal delegating per-node logic to on_enter(node, parent_name).

    on_enter returns the new `parent_name` to use for children (or None to
    propagate the incoming one).
    """
    next_parent = on_enter(node, parent_name)
    new_parent = next_parent if next_parent is not None else parent_name
    for child in node.named_children:
        _walk(child, on_enter, new_parent)


def parse_file(content: bytes, file_path: str | Path) -> ParseResult | None:
    """Parse a TS/TSX/JS source buffer and return extracted symbols + calls.

    Returns None when tree-sitter isn't available or the extension is not
    one we handle; the caller should fall back to regex in that case.
    """
    language = _language_for(file_path)
    if language is None:
        return None

    import tree_sitter as ts

    parser = ts.Parser(language)
    tree = parser.parse(content)
    root = tree.root_node

    result = ParseResult()
    result.had_errors = root.has_error

    def on_enter(node, parent_name):
        t = node.type
        if t == "class_declaration":
            # JS grammar uses `identifier` for class names; TS uses
            # `type_identifier`. Try both so the parser works on both
            # language variants without a grammar-specific branch.
            name_node = _first_child_of_type(
                node, "type_identifier"
            ) or _first_child_of_type(node, "identifier")
            name = _node_text(name_node) if name_node else "<anon>"
            result.symbols.append(
                ExtractedSymbol(
                    kind="Class",
                    name=name,
                    lineno=_line(node),
                    end_lineno=_end_line(node),
                    parent_name=parent_name,
                )
            )
            return name

        if t == "interface_declaration":
            name_node = _first_child_of_type(node, "type_identifier")
            name = _node_text(name_node) if name_node else "<anon>"
            result.symbols.append(
                ExtractedSymbol(
                    kind="Interface",
                    name=name,
                    lineno=_line(node),
                    end_lineno=_end_line(node),
                    parent_name=parent_name,
                )
            )
            return name

        if t == "enum_declaration":
            name_node = _first_child_of_type(node, "identifier")
            name = _node_text(name_node) if name_node else "<anon>"
            result.symbols.append(
                ExtractedSymbol(
                    kind="Enum",
                    name=name,
                    lineno=_line(node),
                    end_lineno=_end_line(node),
                    parent_name=parent_name,
                )
            )
            return None

        if t == "function_declaration":
            name_node = _first_child_of_type(node, "identifier")
            name = _node_text(name_node) if name_node else "<anon>"
            is_async = any(c.type == "async" for c in node.children)
            result.symbols.append(
                ExtractedSymbol(
                    kind="Function",
                    name=name,
                    lineno=_line(node),
                    end_lineno=_end_line(node),
                    parent_name=parent_name,
                    is_async=is_async,
                )
            )
            return name

        if t == "method_definition":
            name_node = _first_child_of_type(node, "property_identifier")
            name = _node_text(name_node) if name_node else "<anon>"
            # Detect get/set/static/async modifiers as leading sibling tokens.
            accessor = None
            is_async = False
            is_static = False
            for child in node.children:
                tt = child.type
                if tt == "get":
                    accessor = "get"
                elif tt == "set":
                    accessor = "set"
                elif tt == "async":
                    is_async = True
                elif tt == "static":
                    is_static = True
            result.symbols.append(
                ExtractedSymbol(
                    kind="Method",
                    name=name,
                    lineno=_line(node),
                    end_lineno=_end_line(node),
                    parent_name=parent_name,
                    is_async=is_async,
                    is_static=is_static,
                    accessor=accessor,
                )
            )
            return name

        if t == "lexical_declaration":
            # const foo = (...) => {...}  |  const foo = function(){}
            for decl in node.named_children:
                if decl.type != "variable_declarator":
                    continue
                name_node = _first_child_of_type(decl, "identifier")
                if not name_node:
                    continue
                rhs = decl.named_children[-1] if decl.named_children else None
                if rhs is None:
                    continue
                if rhs.type in ("arrow_function", "function_expression"):
                    name = _node_text(name_node)
                    result.symbols.append(
                        ExtractedSymbol(
                            kind="Function",
                            name=name,
                            lineno=_line(decl),
                            end_lineno=_end_line(decl),
                            parent_name=parent_name,
                            is_async=any(c.type == "async" for c in rhs.children),
                        )
                    )
            return None

        if t == "import_statement":
            source = _first_child_of_type(node, "string")
            module = None
            if source is not None:
                txt = _node_text(source)
                if len(txt) >= 2 and txt[0] in ("'", '"') and txt[-1] in ("'", '"'):
                    module = txt[1:-1]
            imported_names: list[str] = []
            clause = _first_child_of_type(node, "import_clause")
            if clause is not None:
                # default + named + namespace
                for child in clause.named_children:
                    if child.type == "identifier":
                        imported_names.append(_node_text(child))
                    elif child.type == "named_imports":
                        for spec in child.named_children:
                            if spec.type == "import_specifier":
                                ident = _first_child_of_type(spec, "identifier")
                                if ident is not None:
                                    imported_names.append(_node_text(ident))
                    elif child.type == "namespace_import":
                        ident = _first_child_of_type(child, "identifier")
                        if ident is not None:
                            imported_names.append(_node_text(ident))
            result.symbols.append(
                ExtractedSymbol(
                    kind="Import",
                    name=module or "<unknown>",
                    lineno=_line(node),
                    end_lineno=_end_line(node),
                    parent_name=parent_name,
                    module=module,
                    imported_names=imported_names,
                )
            )
            return None

        if t == "call_expression":
            callee_node = node.named_children[0] if node.named_children else None
            if callee_node is None:
                return None
            if callee_node.type == "identifier":
                result.calls.append((_line(node), _node_text(callee_node)))
            elif callee_node.type == "member_expression":
                # a.b.c(...) -> "a.b.c"
                result.calls.append((_line(node), _node_text(callee_node)))
            return None

        return None

    _walk(root, on_enter)
    return result
