"""Tests for the optional tree-sitter-based TS/TSX/JS extractor.

These only run when the ``descry-codegraph[ast]`` extra is installed; when
tree-sitter isn't importable, tests are skipped so the main test matrix
passes with or without the dep.
"""

from __future__ import annotations

import pytest

from descry.tree_sitter_parser import parse_file, tree_sitter_available


pytestmark = pytest.mark.skipif(
    not tree_sitter_available(), reason="tree-sitter not installed"
)


def _by_kind(result, kind):
    return [s for s in result.symbols if s.kind == kind]


class TestTypeScript:
    def test_class_and_method(self):
        src = b"""
        export class Thing {
          async bar(x: number) { baz(); }
          get name() { return this._n; }
        }
        """
        r = parse_file(src, "demo.ts")
        assert r is not None and not r.had_errors

        classes = _by_kind(r, "Class")
        assert [c.name for c in classes] == ["Thing"]

        methods = _by_kind(r, "Method")
        names = sorted((m.name, m.parent_name, m.accessor, m.is_async) for m in methods)
        assert ("bar", "Thing", None, True) in names
        assert ("name", "Thing", "get", False) in names

    def test_function_declaration(self):
        src = b"export function top() { return 1; }"
        r = parse_file(src, "demo.ts")
        fns = _by_kind(r, "Function")
        assert [f.name for f in fns] == ["top"]

    def test_arrow_function_const(self):
        src = b"const fly = async (x: number) => x + 1;"
        r = parse_file(src, "demo.ts")
        fns = _by_kind(r, "Function")
        assert fns and fns[0].name == "fly" and fns[0].is_async

    def test_interface_and_enum(self):
        src = b"export interface Spec { id: string; }\nexport enum Kind { A, B }"
        r = parse_file(src, "demo.ts")
        assert [i.name for i in _by_kind(r, "Interface")] == ["Spec"]
        assert [e.name for e in _by_kind(r, "Enum")] == ["Kind"]

    def test_imports_resolved(self):
        src = b"""
        import Default from "./x";
        import { A, B as RenamedB } from "./y";
        import * as NS from "./ns";
        """
        r = parse_file(src, "demo.ts")
        imports = _by_kind(r, "Import")
        mods = {i.module for i in imports}
        assert mods == {"./x", "./y", "./ns"}
        names = {n for i in imports for n in i.imported_names}
        assert {"Default", "A", "NS"}.issubset(names)

    def test_calls_extracted(self):
        src = b"""
        function top() {
            baz();
            obj.method();
            Namespace.helper.inner();
        }
        """
        r = parse_file(src, "demo.ts")
        callees = [c for _, c in r.calls]
        assert "baz" in callees
        assert "obj.method" in callees
        assert "Namespace.helper.inner" in callees


class TestTSX:
    def test_tsx_jsx_does_not_error(self):
        src = b"""
        const Card = ({title}: {title: string}) => <div>{title}</div>;
        function App() { return <Card title="hi" />; }
        """
        r = parse_file(src, "demo.tsx")
        assert r is not None
        # Should find both the arrow const and the function.
        fn_names = {f.name for f in r.symbols if f.kind == "Function"}
        assert {"Card", "App"}.issubset(fn_names)


class TestJavaScript:
    def test_plain_js(self):
        src = b"""
        function add(a, b) { return a + b; }
        class C { m() { add(1, 2); } }
        """
        r = parse_file(src, "demo.js")
        fns = {f.name for f in r.symbols if f.kind == "Function"}
        assert "add" in fns
        classes = {c.name for c in r.symbols if c.kind == "Class"}
        assert "C" in classes


class TestDegradation:
    def test_unknown_extension_returns_none(self):
        assert parse_file(b"x", "demo.rs") is None

    def test_syntax_errors_flagged_but_not_raising(self):
        # Missing closing brace; grammar surfaces ERROR but parser shouldn't
        # raise — caller uses this to fall back to regex extraction.
        src = b"function broken( { return 1;"
        r = parse_file(src, "demo.ts")
        assert r is not None
        assert r.had_errors is True
