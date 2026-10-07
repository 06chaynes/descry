"""Tests for descry.generate — CodeGraphBuilder, parsers, and filters."""

import json
import logging
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from descry import ast_grep, generate
from descry.generate import (
    BaseParser,
    CodeGraphBuilder,
    PythonParser,
    RustParser,
    TSParser,
)
from descry.generate import (
    is_non_project_call as is_stdlib_call,
)


class TestStdlibFilter:
    """Tests for stdlib/primitive filtering."""

    def test_filters_rust_primitives(self):
        """Should filter Rust primitives like Some, None, Ok, Err."""
        assert is_stdlib_call("Some")
        assert is_stdlib_call("None")
        assert is_stdlib_call("Ok")
        assert is_stdlib_call("Err")
        assert is_stdlib_call("unwrap")

    def test_filters_rust_prefixes(self):
        """Should filter stdlib prefixes like std::, tokio::."""
        assert is_stdlib_call("std::collections::HashMap")
        assert is_stdlib_call("tokio::spawn")
        assert is_stdlib_call("serde::Deserialize")

    def test_filters_method_suffix(self):
        """Should filter methods like foo.unwrap()."""
        assert is_stdlib_call("result.unwrap")
        assert is_stdlib_call("option.is_some")
        assert is_stdlib_call("vec.push")

    def test_allows_custom_functions(self):
        """Should allow custom function names."""
        assert not is_stdlib_call("validate_token")
        assert not is_stdlib_call("dispatch_deployment")
        assert not is_stdlib_call("CustomService::start")

    def test_filters_js_builtins(self):
        """Should filter JavaScript builtins."""
        assert is_stdlib_call("console")
        assert is_stdlib_call("Promise")
        assert is_stdlib_call("JSON.stringify")

    def test_filters_python_builtins(self):
        """Should filter Python builtins."""
        assert is_stdlib_call("print")
        assert is_stdlib_call("len")
        assert is_stdlib_call("isinstance")

    @pytest.mark.parametrize(
        "receiver",
        [
            "p",
            "req",
            "res",
            "resp",
            "request",
            "response",
            "config",
            "cfg",
            "result",
            "proc",
            "args",
            "ns",
            "parser",
            "sub",
            "subparsers",
            "fd",
            "fh",
            "gp",
            "path_obj",
            "match",
        ],
    )
    def test_receiver_name_alone_does_not_filter(self, receiver):
        """A method call is not dropped because of its receiver's name."""
        assert not is_stdlib_call(f"{receiver}.load_plugins")

    def test_diesel_heuristic_needs_plural_table(self):
        """`table::column.method` is DSL; `module::value.method` is a call."""
        assert is_stdlib_call("targets::id.eq")
        assert not is_stdlib_call("crate::helpers.load_all")
        assert not is_stdlib_call("self::state.reset_all")


class TestPythonParser:
    """Tests for Python parsing."""

    def test_parses_function_with_types(self):
        """Should extract function with type annotations."""
        content = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        builder = CodeGraphBuilder(".")
        parser = PythonParser(builder)
        parser.parse(Path("test.py"), "test.py", content)

        # Find the function node
        func_nodes = [n for n in builder.nodes if n["type"] == "Function"]
        assert len(func_nodes) == 1

        func = func_nodes[0]
        assert func["metadata"]["name"] == "add"
        assert func["metadata"]["return_type"] == "int"
        assert func["metadata"]["param_types"] == ["int", "int"]
        assert "Add two numbers" in func["metadata"]["docstring"]

    def test_parses_class_with_methods(self):
        """Should parse class definitions with methods."""
        content = '''
class Calculator:
    """A simple calculator."""

    def multiply(self, x: float, y: float) -> float:
        return x * y
'''
        builder = CodeGraphBuilder(".")
        parser = PythonParser(builder)
        parser.parse(Path("test.py"), "test.py", content)

        class_nodes = [n for n in builder.nodes if n["type"] == "Class"]
        assert len(class_nodes) == 1
        assert class_nodes[0]["metadata"]["name"] == "Calculator"

        method_nodes = [n for n in builder.nodes if n["type"] == "Method"]
        assert len(method_nodes) == 1
        assert method_nodes[0]["metadata"]["name"] == "multiply"

    def test_handles_union_types(self):
        """Should handle Python 3.10+ union types."""
        content = """
def process(data: str | bytes) -> int | None:
    pass
"""
        builder = CodeGraphBuilder(".")
        parser = PythonParser(builder)
        parser.parse(Path("test.py"), "test.py", content)

        func = next(n for n in builder.nodes if n["type"] == "Function")
        assert "str | bytes" in func["metadata"]["param_types"]
        assert func["metadata"]["return_type"] == "int | None"

    def test_definitions_under_compound_statements(self):
        """Definitions inside if/try/with/for/while blocks are indexed."""
        content = """
if TYPE_CHECKING:
    def in_if(): ...
try:
    def in_try(): ...
except ImportError:
    def in_except(): ...
with open("x") as f:
    def in_with(): ...
for i in range(3):
    def in_for(): ...

class K:
    if True:
        def cond_method(self): ...

def outer():
    if True:
        def cond_inner(): ...
"""
        builder = CodeGraphBuilder(".")
        PythonParser(builder).parse(Path("test.py"), "test.py", content)

        ids = {n["id"] for n in builder.nodes}
        for name in ("in_if", "in_try", "in_except", "in_with", "in_for"):
            assert f"FILE:test.py::{name}" in ids
        assert "FILE:test.py::outer::cond_inner" in ids
        method = next(n for n in builder.nodes if n["id"].endswith("::cond_method"))
        assert method["id"] == "FILE:test.py::K::cond_method"
        assert method["type"] == "Method"

    def test_imports_under_try_recorded(self):
        """Both branches of a try/except import produce IMPORTS edges."""
        content = """
try:
    import ujson
except ImportError:
    import json
"""
        builder = CodeGraphBuilder(".")
        PythonParser(builder).parse(Path("test.py"), "test.py", content)

        imports = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert imports == {"MODULE:ujson", "MODULE:json"}

    def test_nested_function_call_attributed_once(self):
        """A call in a nested function belongs to that function only."""
        content = """
def outer():
    def inner():
        helper()
    inner()
"""
        builder = CodeGraphBuilder(".")
        PythonParser(builder).parse(Path("test.py"), "test.py", content)

        helper_calls = [
            e["source"] for e in builder.edges if e["target"] == "REF:helper"
        ]
        assert helper_calls == ["FILE:test.py::outer::inner"]


class TestRustParser:
    """Tests for Rust parsing."""

    def test_parses_function(self):
        """Should parse Rust function definitions."""
        content = """
/// Validate a token
pub fn validate_token(token: &str) -> Result<Claims, AuthError> {
    // implementation
}
"""
        builder = CodeGraphBuilder(".")
        parser = RustParser(builder)
        parser.parse(Path("test.rs"), "test.rs", content)

        func_nodes = [n for n in builder.nodes if n["type"] == "Function"]
        assert len(func_nodes) == 1

        func = func_nodes[0]
        assert func["metadata"]["name"] == "validate_token"
        assert "Result<Claims, AuthError>" in func["metadata"]["return_type"]

    def test_parses_impl_methods(self):
        """Should parse impl block methods."""
        content = """
struct Server;

impl Server {
    pub fn start(&self) -> Result<(), Error> {
        Ok(())
    }
}
"""
        builder = CodeGraphBuilder(".")
        parser = RustParser(builder)
        parser.parse(Path("test.rs"), "test.rs", content)

        method_nodes = [n for n in builder.nodes if n["type"] == "Method"]
        assert len(method_nodes) == 1
        assert method_nodes[0]["metadata"]["name"] == "start"

    def _parse(self, content):
        builder = CodeGraphBuilder(".")
        builder.use_ast_grep = False
        RustParser(builder).parse(Path("test.rs"), "test.rs", content)
        return builder

    def test_restricted_visibility_items_indexed(self):
        """pub(crate), pub(super) and pub(in path) items produce nodes."""
        builder = self._parse("""
pub(crate) struct Config {
    name: String,
}

pub(super) enum Mode {
    Fast,
}

pub(in crate::app) fn load() {
    read_file();
}

pub(crate) use crate::app::settings;

impl Config {
    pub(crate) fn name(&self) -> &str {
        tidy_name();
    }
}
""")
        ids = {n["id"] for n in builder.nodes}
        assert {
            "FILE:test.rs::Config",
            "FILE:test.rs::Mode",
            "FILE:test.rs::load",
            "FILE:test.rs::Config::name",
        } <= ids
        calls = {
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        }
        assert ("FILE:test.rs::load", "REF:read_file") in calls
        assert ("FILE:test.rs::Config::name", "REF:tidy_name") in calls
        imports = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:crate.app.settings" in imports

    def test_const_and_extern_fn_indexed(self):
        """`const fn` and `extern "C" fn` are functions."""
        builder = self._parse("""
pub const fn limit() -> usize {
    64
}

pub unsafe extern "C" fn entry() {
}
""")
        ids = {n["id"] for n in builder.nodes}
        assert {"FILE:test.rs::limit", "FILE:test.rs::entry"} <= ids

    def test_every_edge_source_is_a_node(self):
        """impl blocks and inline modules source edges at nodes that exist."""
        builder = self._parse("""
impl Display for Remote {
    fn fmt(&self) {
    }
}

mod inner {
    fn deep() {
    }
}
""")
        ids = {n["id"] for n in builder.nodes}
        assert {"FILE:test.rs::Remote", "FILE:test.rs::inner"} <= ids
        assert {e["source"] for e in builder.edges} <= ids

    def test_impl_does_not_replace_struct_declared_below_it(self):
        """A struct declared after its impl keeps its own docstring."""
        builder = self._parse("""
impl Late {
    fn get(&self) {
    }
}

/// Declared after its impl.
struct Late;
""")
        late = [n for n in builder.nodes if n["id"] == "FILE:test.rs::Late"]
        assert len(late) == 1
        assert late[0]["metadata"]["docstring"] == "Declared after its impl."

    def test_one_line_block_does_not_capture_following_items(self):
        """Items after a one-line fn/struct/impl keep their own parent."""
        builder = self._parse("""
fn one() -> i32 { 1 }

struct Point { x: i32 }

impl Marker for Point {}

fn two() {
    helper();
}
""")
        two = next(n for n in builder.nodes if n["metadata"].get("name") == "two")
        assert two["id"] == "FILE:test.rs::two"
        assert two["type"] == "Function"

    def test_regex_fallback_skips_declarations(self):
        """Without ast-grep, declared names and `pub(...)` are not calls."""
        builder = self._parse("""
impl Server {
    pub(crate) fn start(&self) {
        open_listener();
    }
}

mod wire {
    pub(crate) struct Frame(u32);
}
""")
        calls = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert calls == {"REF:open_listener"}


class TestBaseParser:
    """Tests for the docstring and block-end helpers shared by the parsers."""

    def test_pointer_deref_is_not_a_docstring(self):
        """A statement starting with `*` above a declaration is code."""
        lines = ["    *out = 5;", "void next() {"]
        assert BaseParser(None).get_leading_docstring(lines, 1) == ""

    def test_block_comment_docstring(self):
        """A /** ... */ block above a declaration is its docstring."""
        lines = ["/**", " * Starts the server.", " */", "void start() {"]
        doc = BaseParser(None).get_leading_docstring(lines, 3)
        assert doc.strip() == "Starts the server."

    def test_licence_header_is_not_a_docstring(self):
        """A licence notice at the top of the file documents no symbol."""
        lines = ["// Copyright 2020 Example Corp.", "// MIT License", "fn main() {"]
        assert BaseParser(None).get_leading_docstring(lines, 2) == ""

    def test_block_end_ignores_braces_in_literals_and_comments(self):
        """Braces in strings, char literals and comments are not counted."""
        lines = [
            "fn render() {",
            '    let open = "{";',
            "    let close = '}';",
            "    // }",
            "    /* } */",
            '    let url = "http://example.com/{id}";',
            "}",
            "fn next() {",
            "}",
        ]
        assert BaseParser(None)._find_block_end(lines, 0) == 7

    def test_block_end_skips_brace_pairs_in_the_signature(self):
        """A `{ ... }` type inside the parameter list does not end the block."""
        lines = [
            "function run(opts: { fast?: boolean }) {",
            "  go();",
            "}",
        ]
        parser = BaseParser(None)
        assert parser._find_block_end(lines, 0) == 3
        assert parser._find_block_end(["struct P { x: i32 }", "fn f() {", "}"], 0) == 1


class TestTypeScriptParser:
    """Tests for TypeScript parsing."""

    def test_parses_function(self):
        """Should parse TypeScript function definitions."""
        content = """
export function fetchData(url: string): Promise<Response> {
    return fetch(url);
}
"""
        builder = CodeGraphBuilder(".")
        parser = TSParser(builder)
        parser.parse(Path("test.ts"), "test.ts", content)

        func_nodes = [n for n in builder.nodes if n["type"] == "Function"]
        assert len(func_nodes) == 1

        func = func_nodes[0]
        assert func["metadata"]["name"] == "fetchData"
        assert func["metadata"]["return_type"] == "Promise<Response>"

    def test_parses_class(self):
        """Should parse TypeScript class definitions."""
        content = """
export class ApiClient {
    async request(endpoint: string): Promise<any> {
        return null;
    }
}
"""
        builder = CodeGraphBuilder(".")
        parser = TSParser(builder)
        parser.parse(Path("test.ts"), "test.ts", content)

        class_nodes = [n for n in builder.nodes if n["type"] == "Class"]
        assert len(class_nodes) == 1
        assert class_nodes[0]["metadata"]["name"] == "ApiClient"

    def test_parses_generic_methods(self):
        """Should parse TypeScript methods with generic type parameters."""
        content = """
export class DataService {
    fetch<T = unknown>(url: string): Promise<T> {
        return null;
    }

    transform<T, U>(input: T): U {
        return null;
    }
}
"""
        builder = CodeGraphBuilder(".")
        parser = TSParser(builder)
        parser.parse(Path("test.ts"), "test.ts", content)

        method_nodes = [n for n in builder.nodes if n["type"] == "Method"]
        assert len(method_nodes) == 2

        method_names = {m["metadata"]["name"] for m in method_nodes}
        assert "fetch" in method_names
        assert "transform" in method_names

    def test_parses_getters_setters(self):
        """Should parse TypeScript getter and setter methods."""
        content = """
export class Counter {
    private _count = 0;

    get count(): number {
        return this._count;
    }

    set count(value: number) {
        this._count = value;
    }
}
"""
        builder = CodeGraphBuilder(".")
        parser = TSParser(builder)
        parser.parse(Path("test.ts"), "test.ts", content)

        method_nodes = [n for n in builder.nodes if n["type"] == "Method"]
        # Should have both getter and setter with unique IDs
        assert len(method_nodes) == 2

        # Check accessor metadata - both get and set should be present
        accessors = [m for m in method_nodes if m["metadata"].get("accessor")]
        assert len(accessors) == 2
        accessor_types = {m["metadata"]["accessor"] for m in accessors}
        assert "get" in accessor_types
        assert "set" in accessor_types

        # Check signatures are present
        for m in method_nodes:
            assert m["metadata"].get("signature") is not None

    def test_regex_fallback_skips_declarations(self):
        """Without ast-grep, declaring a method or function is not a call."""
        content = """
export class Service {
    constructor(private port: number) {
        this.bindPort();
    }
    stop(now: boolean): void {
        haltWorkers(now);
    }
}
function outer() {
    function inner() {
    }
}
"""
        builder = CodeGraphBuilder(".")
        builder.use_ast_grep = False
        TSParser(builder).parse(Path("test.ts"), "test.ts", content)

        calls = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert calls == {"REF:this.bindPort", "REF:haltWorkers"}


class TestCodeGraphBuilder:
    """Tests for the graph builder."""

    def test_adds_node(self):
        """Should add nodes to the graph."""
        builder = CodeGraphBuilder(".")
        builder.add_node("FILE:test.py", "File", path="test.py", name="test.py")

        assert len(builder.nodes) == 1
        assert builder.nodes[0]["id"] == "FILE:test.py"
        assert builder.nodes[0]["type"] == "File"

    def test_prevents_duplicate_nodes(self):
        """Should not add duplicate nodes."""
        builder = CodeGraphBuilder(".")
        builder.add_node("FILE:test.py", "File", path="test.py")
        builder.add_node("FILE:test.py", "File", path="test.py")

        assert len(builder.nodes) == 1

    def test_adds_edge(self):
        """Should add edges to the graph."""
        builder = CodeGraphBuilder(".")
        builder.add_edge("FILE:test.py", "FILE:test.py::func", "DEFINES")

        assert len(builder.edges) == 1
        assert builder.edges[0]["source"] == "FILE:test.py"
        assert builder.edges[0]["relation"] == "DEFINES"

    def test_resolves_references(self):
        """Should resolve REF: targets to actual nodes."""
        builder = CodeGraphBuilder(".")
        builder.add_node("FILE:test.py::validate", "Function", name="validate")
        builder.add_edge("FILE:test.py::main", "REF:validate", "CALLS")

        builder.resolve_references()

        # Edge should now point to resolved node
        assert builder.edges[0]["target"] == "FILE:test.py::validate"
        assert builder.edges[0]["relation"] == "CALLS_RESOLVED"


def _index(root, files):
    """Index ``files`` (relative path -> source) under ``root`` with the regex
    parsers and return the builder once references are resolved."""
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    builder = CodeGraphBuilder(root)
    builder.use_ast_grep = False
    builder.process_directory()
    builder.resolve_references()
    return builder


def _calls(builder, source):
    """Targets of the call edges that leave node ``source``, sorted."""
    return sorted(
        e["target"]
        for e in builder.edges
        if e["source"] == source and e["relation"].startswith("CALLS")
    )


class TestCallAccounting:
    """What resolve_references counts as resolved, unresolved and external."""

    def test_call_no_project_symbol_names_is_external(self, tmp_path):
        """A call on a local receiver to a name the project never defines
        is dropped, not kept as an unresolved project call."""
        builder = _index(
            tmp_path,
            {
                "cli.py": """
                def build(sub):
                    sub.add_argument("--flag")
                """
            },
        )
        assert _calls(builder, "FILE:cli.py::build") == []

    def test_ambiguous_call_stays_unresolved(self, tmp_path):
        """A method several unrelated classes define is not pinned to one
        of them when the receiver's type is unknown."""
        builder = _index(
            tmp_path,
            {
                "shapes.py": """
                class Circle:
                    def area(self): ...

                class Square:
                    def area(self): ...

                def total(shape):
                    return shape.area()
                """
            },
        )
        assert _calls(builder, "FILE:shapes.py::total") == ["REF:shape.area"]

    def test_summary_separates_external_from_unresolved(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="descry.generate"):
            _index(
                tmp_path,
                {
                    "app.py": """
                    class Circle:
                        def area(self): ...

                    class Square:
                        def area(self): ...

                    def helper(): ...

                    def main(shape, out):
                        helper()
                        shape.area()
                        out.write_everything()
                    """
                },
            )
        assert (
            "CALLS Resolution: 50% of 2 project calls "
            "(SCIP: 0/0%, Regex: 1/50%, Self: 0/0%, Unresolved: 1); "
            "1 external calls not counted"
        ) in caplog.text


class _ScipStub:
    """A SCIP index that knows references: ``(file, line)`` or ``(file,
    line, column)`` -> the ``(node_id, file, line)`` of the definition."""

    def __init__(self, references):
        self.references = references

    def definition_at(self, ref_name, source_file, line, column):
        return self.references.get(
            (source_file, line, column), self.references.get((source_file, line))
        )


class TestScipCallResolution:
    SOURCE = """
    fn probe() {}

    mod tests {
        fn probe() {}
    }

    fn run() {
        probe();
    }
    """

    def _resolve(self, tmp_path, definition):
        (tmp_path / "lib.rs").write_text(textwrap.dedent(self.SOURCE))
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = False
        builder.process_directory()
        builder.resolve_references(_ScipStub({("lib.rs", 9): definition}))
        return [e for e in builder.edges if e["source"] == "FILE:lib.rs::run"]

    def test_call_goes_to_the_node_defined_where_scip_says(self, tmp_path):
        """SCIP's definition line picks the node, however SCIP spells the
        symbol's path."""
        (edge,) = self._resolve(tmp_path, ("FILE:lib.rs::probe", "lib.rs", 5))
        assert edge["target"] == "FILE:lib.rs::tests::probe"
        assert edge["metadata"]["resolution_source"] == "scip"

    def test_scip_target_that_is_no_node_is_not_used(self, tmp_path):
        (edge,) = self._resolve(tmp_path, ("FILE:app/lib.rs::probe", "app/lib.rs", 2))
        assert edge["target"] == "FILE:lib.rs::probe"
        assert edge["metadata"]["resolution_source"] == "regex"

    def test_scip_is_asked_at_the_column_of_the_called_name(self, tmp_path):
        """Two calls of one name on a line are looked up at their own
        columns, and the column is not left in the graph."""
        source = """
        struct A;
        impl A {
            fn probe(&self) {}
        }
        struct B;
        impl B {
            fn probe(&self) {}
        }

        fn run(a: &Opaque, b: &Opaque) {
            a.probe(); b.probe();
        }
        """
        (tmp_path / "lib.rs").write_text(textwrap.dedent(source))
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = False
        builder.process_directory()
        stub = _ScipStub(
            {
                ("lib.rs", 12, 6): ("FILE:lib.rs::A::probe", "lib.rs", 4),
                ("lib.rs", 12, 17): ("FILE:lib.rs::B::probe", "lib.rs", 8),
            }
        )
        builder.resolve_references(stub)
        edges = [e for e in builder.edges if e["source"] == "FILE:lib.rs::run"]
        assert [e["target"] for e in edges] == [
            "FILE:lib.rs::A::probe",
            "FILE:lib.rs::B::probe",
        ]
        assert all("col" not in e["metadata"] for e in edges)

    def test_python_call_carries_its_column(self, tmp_path):
        """Two calls of one name on a Python line are looked up at their own
        columns, counted in every unit where the line is not ASCII."""
        source = """
        class A:
            def probe(self): ...

        class B:
            def probe(self): ...

        def run(a, b):
            a.probe(); b.probe()

        def accented(a, b):
            print("é"); a.probe(); b.probe()
        """
        (tmp_path / "lib.py").write_text(textwrap.dedent(source), encoding="utf-8")
        builder = CodeGraphBuilder(tmp_path)
        builder.process_directory()
        first = ("FILE:lib.py::A::probe", "lib.py", 3)
        second = ("FILE:lib.py::B::probe", "lib.py", 6)
        stub = _ScipStub(
            {
                ("lib.py", 9, 6): first,
                ("lib.py", 9, 17): second,
                ("lib.py", 12, (19, 18, 18)): first,
                ("lib.py", 12, (30, 29, 29)): second,
            }
        )
        builder.resolve_references(stub)
        for function in ("run", "accented"):
            edges = [
                e for e in builder.edges if e["source"] == f"FILE:lib.py::{function}"
            ]
            targets = [e["target"] for e in edges if e["target"].endswith("probe")]
            assert targets == [first[0], second[0]]

    def test_python_column_of_a_name_python_normalises(self, tmp_path):
        """A called name is located as written in the source, where the `fi`
        ligature makes it longer than the name Python reads it as."""
        source = "def finish(): ...\n\ndef run():\n    \ufb01nish()\n"
        (tmp_path / "lib.py").write_text(source, encoding="utf-8")
        builder = CodeGraphBuilder(tmp_path)
        builder.process_directory()
        stub = _ScipStub({("lib.py", 4, 4): ("FILE:lib.py::finish", "lib.py", 1)})
        builder.resolve_references(stub)
        (edge,) = [e for e in builder.edges if e["source"] == "FILE:lib.py::run"]
        assert edge["target"] == "FILE:lib.py::finish"
        assert edge["metadata"]["resolution_source"] == "scip"


class TestPytestFixtureParameters:
    """A parameter of a test or fixture has the value its fixture gives."""

    APP = """
    class Other:
        def settle(self): ...

    class Service:
        def settle(self): ...
    """

    def test_parameter_is_typed_by_the_returned_value_of_its_fixture(self, tmp_path):
        """`svc.settle()` reaches Service.settle through the fixture `svc`, though
        Other has a `settle` as well."""
        builder = _index(
            tmp_path,
            {
                "app.py": self.APP,
                "tests/test_app.py": """
                import pytest
                from app import Service

                @pytest.fixture
                def svc():
                    return Service()

                def test_run(svc):
                    svc.settle()
                """,
            },
        )
        assert _calls(builder, "FILE:tests/test_app.py::test_run") == [
            "FILE:app.py::Service::settle"
        ]

    def test_fixture_is_found_in_a_conftest_above_and_may_yield(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.py": self.APP,
                "tests/conftest.py": """
                import pytest
                import app

                @pytest.fixture(name="svc")
                def make_service():
                    service = app.Service()
                    yield service
                """,
                "tests/unit/test_app.py": """
                class TestRun:
                    def test_run(self, svc):
                        svc.settle()
                """,
            },
        )
        assert _calls(builder, "FILE:tests/unit/test_app.py::TestRun::test_run") == [
            "FILE:app.py::Service::settle"
        ]

    def test_fixture_that_returns_a_function_is_called_as_that_function(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "tests/test_app.py": """
                import pytest

                @pytest.fixture
                def writer(tmp_path):
                    def _write(text):
                        return tmp_path / text
                    return _write

                def test_write(writer):
                    writer("a")
                """,
            },
        )
        assert _calls(builder, "FILE:tests/test_app.py::test_write") == [
            "FILE:tests/test_app.py::writer::_write"
        ]

    def test_fixture_built_on_another_fixture(self, tmp_path):
        """A fixture's own parameters are fixtures; a return annotation is
        used where there is one."""
        builder = _index(
            tmp_path,
            {
                "app.py": self.APP,
                "tests/test_app.py": """
                import pytest
                from app import Service

                @pytest.fixture
                def base() -> Service:
                    return make()

                @pytest.fixture
                def svc(base):
                    base.settle()
                    return base

                def test_run(svc):
                    svc.settle()
                """,
            },
        )
        assert _calls(builder, "FILE:tests/test_app.py::svc") == [
            "FILE:app.py::Service::settle"
        ]
        assert _calls(builder, "FILE:tests/test_app.py::test_run") == [
            "FILE:app.py::Service::settle"
        ]

    def test_parametrized_argument_is_not_a_fixture(self, tmp_path):
        """`svc` comes from the `parametrize` mark, so the fixture of that
        name says nothing about it and `settle` stays ambiguous."""
        builder = _index(
            tmp_path,
            {
                "app.py": self.APP,
                "tests/test_app.py": """
                import pytest
                from app import Service

                @pytest.fixture
                def svc():
                    return Service()

                @pytest.mark.parametrize("svc, expected", CASES)
                def test_run(svc, expected):
                    svc.settle()
                """,
            },
        )
        assert _calls(builder, "FILE:tests/test_app.py::test_run") == ["REF:svc.settle"]

    def test_parameter_of_an_ordinary_function_is_not_a_fixture(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.py": self.APP,
                "tests/test_app.py": """
                import pytest
                from app import Service

                @pytest.fixture
                def svc():
                    return Service()

                def helper(svc):
                    svc.settle()
                """,
            },
        )
        assert _calls(builder, "FILE:tests/test_app.py::helper") == ["REF:svc.settle"]


class TestPythonCallResolution:
    def test_receiver_built_from_project_class_resolves_to_its_method(self, tmp_path):
        """`x = Cls(); x.m()` reaches Cls.m, whichever other class has an `m`
        and even when `m` is a common library method name."""
        builder = _index(
            tmp_path,
            {
                "svc.py": """
                class Other:
                    def index(self): ...

                class Service:
                    def index(self): ...

                def run():
                    svc = Service()
                    svc.index()
                """
            },
        )
        assert _calls(builder, "FILE:svc.py::run") == [
            "FILE:svc.py::Service",
            "FILE:svc.py::Service::index",
        ]

    def test_receiver_built_from_external_class_is_external(self, tmp_path):
        """A method call on an instance of an imported non-project class is
        external even when a project class has a method of that name."""
        builder = _index(
            tmp_path,
            {
                "cli.py": """
                import argparse

                class Registry:
                    def add_widget(self): ...

                def main():
                    parser = argparse.ArgumentParser()
                    sub = parser.add_subparsers()
                    sub.add_widget()
                """
            },
        )
        assert _calls(builder, "FILE:cli.py::main") == []

    def test_imported_name_resolves_to_the_imported_module(self, tmp_path):
        """A call through `from pkg.mod import name` reaches that module's
        definition, not a same-named one elsewhere."""
        builder = _index(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/a.py": "def helper(): ...\n",
                "pkg/b.py": "def helper(): ...\n",
                "pkg/main.py": """
                from pkg.b import helper
                from . import a

                def run():
                    helper()
                    a.helper()
                """,
            },
        )
        assert _calls(builder, "FILE:pkg/main.py::run") == [
            "FILE:pkg/a.py::helper",
            "FILE:pkg/b.py::helper",
        ]

    def test_call_through_non_project_import_is_external(self, tmp_path):
        """`lib.compute()` on an imported non-project module does not reach a
        project function called `compute`."""
        builder = _index(
            tmp_path,
            {
                "mathx.py": "def compute(): ...\n",
                "main.py": """
                import thirdparty as lib

                def run():
                    lib.compute()
                """,
            },
        )
        assert _calls(builder, "FILE:main.py::run") == []

    def test_reexported_name_is_followed(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "decoy.py": "def make(): ...\n",
                "pkg/__init__.py": "from pkg.impl import make\n",
                "pkg/impl.py": "def make(): ...\n",
                "main.py": """
                import pkg

                def run():
                    pkg.make()
                """,
            },
        )
        assert _calls(builder, "FILE:main.py::run") == ["FILE:pkg/impl.py::make"]

    def test_namespace_package_module_is_a_project_module(self, tmp_path):
        """A module in a directory without `__init__.py` is still reachable
        by its dotted path."""
        builder = _index(
            tmp_path,
            {
                "decoy.py": "def score(): ...\n",
                "tools/metrics.py": "def score(): ...\n",
                "run.py": """
                from tools.metrics import score

                def main():
                    score()
                """,
            },
        )
        assert _calls(builder, "FILE:run.py::main") == ["FILE:tools/metrics.py::score"]

    def test_self_call_resolves_through_base_class(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "base.py": """
                class Base:
                    def setup(self): ...
                """,
                "a_decoy.py": """
                class Unrelated:
                    def setup(self): ...
                """,
                "child.py": """
                from base import Base

                class Child(Base):
                    def run(self):
                        self.setup()
                """,
            },
        )
        assert _calls(builder, "FILE:child.py::Child::run") == [
            "FILE:base.py::Base::setup"
        ]

    def test_self_call_inherited_from_external_base_is_external(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "t.py": """
                import unittest

                class Helper:
                    def check(self): ...

                class Case(unittest.TestCase):
                    def test_it(self):
                        self.check()
                """
            },
        )
        assert _calls(builder, "FILE:t.py::Case::test_it") == []

    def test_return_annotation_types_the_result(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "svc.py": """
                class Service:
                    def start(self): ...

                class Engine:
                    def start(self): ...

                def make() -> Service:
                    return Service()

                def run():
                    make().start()
                """
            },
        )
        assert _calls(builder, "FILE:svc.py::run") == [
            "FILE:svc.py::Service::start",
            "FILE:svc.py::make",
        ]

    def test_annotated_parameter_types_the_receiver(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "svc.py": """
                class Service:
                    def start(self): ...

                class Engine:
                    def start(self): ...

                def run(engine: Engine):
                    engine.start()
                """
            },
        )
        assert _calls(builder, "FILE:svc.py::run") == ["FILE:svc.py::Engine::start"]

    def test_another_functions_local_is_not_visible(self, tmp_path):
        """A bare call does not reach a function nested in another function."""
        builder = _index(
            tmp_path,
            {
                "a.py": """
                def outer():
                    def helper(): ...
                    helper()

                def other():
                    helper()
                """
            },
        )
        assert _calls(builder, "FILE:a.py::outer") == ["FILE:a.py::outer::helper"]
        assert _calls(builder, "FILE:a.py::other") == []

    def test_function_local_import_is_followed(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "decoy.py": "def fenced(): ...\n",
                "lib.py": "def fenced(): ...\n",
                "test_lib.py": """
                def test_fenced():
                    from lib import fenced

                    fenced()
                """,
            },
        )
        assert _calls(builder, "FILE:test_lib.py::test_fenced") == [
            "FILE:lib.py::fenced"
        ]


class TestRustCallResolution:
    def test_path_through_non_project_type_is_external(self, tmp_path):
        """`BufReader::new` does not reach a project `new`."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Config;

                impl Config {
                    pub fn new() -> Config {
                        Config
                    }
                }

                pub fn open() {
                    let reader = BufReader::new(file);
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::open") == []

    def test_method_of_a_foreign_type_with_a_local_impl_is_external(self, tmp_path):
        """`Shared::new` stays external when the project implements a trait
        for the non-project type `Shared` and defines other `new` functions."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub trait Sink {
                    fn finish(&self);
                }

                impl<S: Sink> Sink for Shared<S> {
                    fn finish(&self) {}
                }

                pub struct Printer;

                impl Printer {
                    pub fn new() -> Printer {
                        Printer
                    }
                }

                pub fn boxed() {
                    let sink = Shared::new(Printer::new());
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::boxed") == [
            "FILE:src/lib.rs::Printer::new"
        ]

    def test_path_rooted_outside_the_project_reaches_only_local_impls(self, tmp_path):
        """`io::Error::new` is not the project's `Error::new`, while a trait
        method the project implements for `io::Error` is reached."""
        builder = _index(
            tmp_path,
            {
                "src/errors.rs": """
                pub struct Error;

                impl Error {
                    pub fn new() -> Error {
                        Error
                    }
                }
                """,
                "src/report.rs": """
                pub trait Report {
                    fn describe() -> Self;
                }

                impl Report for io::Error {
                    fn describe() -> io::Error {
                        io::Error::new(io::ErrorKind::Other, "x")
                    }
                }

                pub fn fail() {
                    let err = io::Error::describe();
                }
                """,
            },
        )
        method = "FILE:src/report.rs::Error::describe"
        assert _calls(builder, method) == []
        assert _calls(builder, "FILE:src/report.rs::fail") == [method]

    def test_qualified_call_prefers_the_nearest_type(self, tmp_path):
        """`Matcher::new` reaches the `Matcher` of the calling file, not one
        of the same name declared elsewhere."""
        builder = _index(
            tmp_path,
            {
                "a/src/util.rs": """
                pub struct Matcher;
                impl Matcher {
                    pub fn new() -> Matcher {
                        Matcher
                    }
                }
                """,
                "b/src/testutil.rs": """
                pub struct Matcher;
                impl Matcher {
                    pub fn new() -> Matcher {
                        Matcher
                    }
                }

                fn check() {
                    let m = Matcher::new();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:b/src/testutil.rs::check") == [
            "FILE:b/src/testutil.rs::Matcher::new"
        ]

    def test_free_call_and_method_call_are_told_apart(self, tmp_path):
        """`run(x)` is the free function; `self.run()` and `w.run()` are
        the method."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Worker;

                impl Worker {
                    fn run(&self) {
                        run(self);
                    }

                    fn again(&self) {
                        self.run();
                    }
                }

                fn run(worker: &Worker) {
                    worker.run();
                }
                """
            },
        )
        method, function = "FILE:src/lib.rs::Worker::run", "FILE:src/lib.rs::run"
        assert _calls(builder, method) == [function]
        assert _calls(builder, "FILE:src/lib.rs::Worker::again") == [method]
        assert _calls(builder, function) == [method]

    def test_declared_type_picks_the_method(self, tmp_path):
        """A parameter's declared type, a trait object or a bounded type
        parameter decides between same-named methods."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Glob;
                impl Glob {
                    pub fn matched(&self) {}
                }

                pub struct Ignore;
                impl Ignore {
                    pub fn matched(&self) {}
                }

                pub trait Flag {
                    fn name(&self);
                }

                pub struct Quiet;
                impl Flag for Quiet {
                    fn name(&self) {}
                }

                fn check(ignore: &Ignore<'_>, flag: &dyn Flag) {
                    ignore.matched();
                    flag.name();
                }

                fn generic<F: Flag>(flag: F) {
                    flag.name();
                }
                """
            },
        )
        trait_method = "FILE:src/lib.rs::Flag::name"
        assert _calls(builder, "FILE:src/lib.rs::check") == [
            trait_method,
            "FILE:src/lib.rs::Ignore::matched",
        ]
        assert _calls(builder, "FILE:src/lib.rs::generic") == [trait_method]

    def test_receiver_typed_by_an_alias_is_not_external(self, tmp_path):
        """A method call on a value whose declared type is a project alias
        stays a project call."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Match;
                impl Match {
                    pub fn span_start(&self) {}
                }

                pub struct Cursor;
                impl Cursor {
                    pub fn span_start(&self) {}
                }

                type Range = Match;

                fn span(range: &Range) {
                    range.span_start();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::span") == ["REF:range.span_start"]

    def test_another_functions_variable_does_not_type_the_receiver(self, tmp_path):
        """A parameter declared in one function says nothing about a
        same-named variable of another."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Reader;
                impl Reader {
                    pub fn buffer(&self) {}
                }

                pub struct Writer;
                impl Writer {
                    pub fn buffer(&self) {}
                }

                fn first(rdr: Reader) {}

                fn make() -> Box<dyn Buffered> {
                    todo!()
                }

                fn second() {
                    let rdr = make();
                    rdr.buffer();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::second") == [
            "FILE:src/lib.rs::make",
            "REF:rdr.buffer",
        ]

    def test_bare_call_reaches_another_file_only_through_a_use(self, tmp_path):
        """A closure called by name is not a same-named function of another
        file, of a function there, or of a module there; an imported
        function is reached."""
        builder = _index(
            tmp_path,
            {
                "src/a.rs": """
                pub fn enroll(glob: &str) {}
                pub fn strip(path: &str) {}

                fn commands() {
                    fn sift(glob: &str) {}
                }

                mod tests {
                    fn probe() {}
                }
                """,
                "src/b.rs": """
                use crate::a::{
                    strip,
                };

                fn patterns() {
                    let mut enroll = |pat: String| {};
                    enroll(String::new());
                    sift(1);
                    probe();
                    strip("x");
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/b.rs::patterns") == ["FILE:src/a.rs::strip"]

    def test_method_of_unknown_receiver_does_not_reach_a_free_function(self, tmp_path):
        """`.line_number()` in a call chain is not the free function
        `line_number`."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                fn line_number() {}

                fn build() {
                    make()
                        .line_number();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::build") == []

    def test_call_on_a_returned_value_is_typed_by_the_return_type(self, tmp_path):
        """Each link of a chain is a call on what the link before returns:
        `Self`, a named type, and what `?` or `unwrap()` takes out of a
        `Result`."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub struct Maker;
                impl Maker {
                    pub fn new() -> Self { Maker }
                    pub fn named(self, name: &str) -> Maker { self }
                    pub fn assemble(&self) -> Result<Product, Error> { todo!() }
                }
                pub struct Product;
                impl Product {
                    pub fn ship(&self) {}
                }
                pub struct Decoy;
                impl Decoy {
                    pub fn named(self, name: &str) {}
                    pub fn assemble(&self) {}
                    pub fn ship(&self) {}
                }

                fn chained() -> Result<(), Error> {
                    Maker::new().named("x").assemble()?.ship();
                    Ok(())
                }

                fn unwrapped() {
                    Maker::new()
                        .assemble()
                        .unwrap()
                        .ship();
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::chained") == [
            "FILE:lib.rs::Maker::assemble",
            "FILE:lib.rs::Maker::named",
            "FILE:lib.rs::Maker::new",
            "FILE:lib.rs::Product::ship",
        ]
        assert _calls(builder, "FILE:lib.rs::unwrapped") == [
            "FILE:lib.rs::Maker::assemble",
            "FILE:lib.rs::Maker::new",
            "FILE:lib.rs::Product::ship",
        ]

    def test_variable_bound_to_a_call_has_the_returned_type(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub struct Product;
                impl Product {
                    pub fn ship(&self) {}
                }
                pub struct Decoy;
                impl Decoy {
                    pub fn ship(&self) {}
                }
                fn make() -> Product { Product }
                fn load() -> Result<Product, Error> { todo!() }

                fn run() -> Result<(), Error> {
                    let made = make();
                    made.ship();
                    let loaded = load()?;
                    loaded.ship();
                    Ok(())
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::run") == [
            "FILE:lib.rs::Product::ship",
            "FILE:lib.rs::Product::ship",
            "FILE:lib.rs::load",
            "FILE:lib.rs::make",
        ]

    def test_use_path_names_the_crate_a_call_goes_to(self, tmp_path):
        """`use store_kit::probe` reaches the `probe` of the crate whose
        package is `store-kit`; a `use` of a crate that is not in the
        project reaches nothing."""
        builder = _index(
            tmp_path,
            {
                "crates/store/Cargo.toml": """
                [package]
                name = "store-kit"
                """,
                "crates/store/src/lib.rs": """
                pub fn probe() {}
                pub fn fetch() {}
                """,
                "crates/cache/Cargo.toml": """
                [package]
                name = "cache"
                """,
                "crates/cache/src/lib.rs": """
                pub fn probe() {}
                """,
                "crates/app/Cargo.toml": """
                [package]
                name = "app"
                """,
                "crates/app/src/main.rs": """
                use store_kit::probe;
                use reqwest::fetch;

                fn main() {
                    probe();
                    fetch();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:crates/app/src/main.rs::main") == [
            "FILE:crates/store/src/lib.rs::probe"
        ]

    def test_glob_use_reaches_only_the_crate_it_names(self, tmp_path):
        """`use crate::kinds::*` brings in nothing of another crate, while
        `use flags::*` brings in that crate's `probe`."""
        builder = _index(
            tmp_path,
            {
                "crates/flags/Cargo.toml": """
                [package]
                name = "flags"
                """,
                "crates/flags/src/lib.rs": """
                pub fn probe() {}
                """,
                "crates/app/Cargo.toml": """
                [package]
                name = "app"
                """,
                "crates/app/src/kinds.rs": """
                pub fn other() {}
                """,
                "crates/app/src/local.rs": """
                use crate::kinds::*;

                fn run() {
                    probe();
                    other();
                }
                """,
                "crates/app/src/main.rs": """
                use flags::*;

                fn main() {
                    probe();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:crates/app/src/local.rs::run") == [
            "FILE:crates/app/src/kinds.rs::other"
        ]
        assert _calls(builder, "FILE:crates/app/src/main.rs::main") == [
            "FILE:crates/flags/src/lib.rs::probe"
        ]

    def test_bare_call_does_not_reach_a_method_of_another_type(self, tmp_path):
        """`negate(..)` calls a local closure; no receiver or path leads to
        Filter::negate, the only definition of that name."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub struct Filter;
                impl Filter {
                    pub fn negate(&mut self, name: &str) {}
                }

                fn run() {
                    let negate = |name: &str| name.len();
                    negate("rust");
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::run") == []

    def test_call_of_a_local_value_is_not_a_function_of_that_name(self, tmp_path):
        """A closure held in a variable and a function passed as a parameter
        are called by name; neither call is one of the free function that
        has the same name in the file."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                fn helper(n: usize) -> usize { n }
                fn check() {}

                fn with_closure() {
                    let helper = |n: usize| n + 1;
                    helper(1);
                }

                fn with_parameter(check: impl Fn()) {
                    check();
                }

                fn direct() {
                    helper(2);
                    check();
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::with_closure") == []
        assert _calls(builder, "FILE:lib.rs::with_parameter") == []
        assert _calls(builder, "FILE:lib.rs::direct") == [
            "FILE:lib.rs::check",
            "FILE:lib.rs::helper",
        ]

    def test_path_through_a_type_parameter_goes_to_its_bound(self, tmp_path):
        """`S::begin` is a function of the bound of `S`. `S::Error` is an
        associated type, not the project's `Error`: `error_io` is the
        declaration every implementation of it shares."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub trait SinkError {
                    fn error_io(message: &str) -> Self;
                }

                pub trait Sink {
                    type Error: SinkError;
                    fn begin(&self) {}
                }

                pub struct Error;
                impl SinkError for Error {
                    fn error_io(message: &str) -> Self { Error }
                }

                pub struct Other;
                impl Other {
                    fn begin(&self) {}
                }

                fn run<S: Sink>(sink: &S) -> S::Error {
                    S::begin(sink);
                    S::Error::error_io("x")
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::run") == [
            "FILE:lib.rs::Sink::begin",
            "FILE:lib.rs::SinkError::error_io",
        ]

    def test_text_of_a_string_is_not_code(self, tmp_path):
        """Call text in a string, a string continued over lines or a raw
        string is no call, and a `fn` written inside one defines nothing."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                fn helper() {}
                fn other() {}

                fn run() {
                    let a = "then helper() runs";
                    let b = "first line
                        other() on the second
                fn ghost() {
                        helper()";
                    let c = r#"raw "quoted" other()"#;
                    other();
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::run") == ["FILE:lib.rs::other"]
        assert "FILE:lib.rs::ghost" not in builder.node_registry
        assert "FILE:lib.rs::run::ghost" not in builder.node_registry

    def test_methods_of_an_impl_written_on_one_line(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub struct Text;
                impl Text { fn doubled(&self) -> usize { twice(self.raw()) } fn raw(&self) -> usize { 1 } }

                fn twice(n: usize) -> usize { n }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::Text::doubled") == [
            "FILE:lib.rs::Text::raw",
            "FILE:lib.rs::twice",
        ]

    def test_function_nested_in_a_method_is_not_a_method(self, tmp_path):
        """A `fn` inside a method body belongs to that method and is
        reached from it by a bare call."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub struct Filter;
                impl Filter {
                    pub fn run(&self) {
                        fn inner() {}
                        inner();
                    }
                }
                """
            },
        )
        nodes = {n["id"]: n["type"] for n in builder.nodes}
        assert nodes["FILE:lib.rs::Filter::run::inner"] == "Function"
        assert "FILE:lib.rs::Filter::inner" not in nodes
        assert _calls(builder, "FILE:lib.rs::Filter::run") == [
            "FILE:lib.rs::Filter::run::inner"
        ]

    def test_field_of_a_generic_project_type_types_the_receiver(self, tmp_path):
        """`core: Core<'s, M>` is a Core, so `self.core.pos()` is Core::pos
        though Cursor has a `pos` too. `term: Option<Term>` is not a Term,
        and `term: None` in a struct literal declares no type at all."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                pub struct Core<'s, M> {
                    matcher: &'s M,
                }
                impl<'s, M> Core<'s, M> {
                    pub fn pos(&self) -> usize { 0 }
                }
                pub struct Cursor;
                impl Cursor {
                    pub fn pos(&self) -> usize { 0 }
                }
                pub struct Term;
                impl Term {
                    pub fn as_byte(&self) -> u8 { 0 }
                }

                pub struct Reader<'s, M> {
                    core: Core<'s, M>,
                    term: Option<Term>,
                }
                impl<'s, M> Reader<'s, M> {
                    fn new(core: Core<'s, M>) -> Reader<'s, M> {
                        Reader { core, term: None }
                    }

                    fn run(&self) -> u8 {
                        self.core.pos();
                        self.term.unwrap().as_byte()
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::Reader::run") == [
            "FILE:lib.rs::Core::pos",
            "FILE:lib.rs::Term::as_byte",
        ]

    def test_variable_named_like_a_module_is_not_the_module(self, tmp_path):
        """A module is named in a `::` path. `stats.total()` is a method
        call on a variable, and `0..rows.count()` one on `rows`."""
        builder = _index(
            tmp_path,
            {
                "src/stats.rs": """
                pub fn reset() {}
                """,
                "src/lib.rs": """
                pub struct Summary;
                impl Summary {
                    pub fn total(&self) -> usize { 0 }
                }
                pub struct Rows;
                impl Rows {
                    pub fn count(&self) -> usize { 0 }
                }
                pub struct Other;
                impl Other {
                    pub fn count(&self) -> usize { 0 }
                }

                fn run(rows: &Rows) {
                    let stats = gather();
                    stats.total();
                    for i in 0..rows.count() {}
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::run") == [
            "FILE:src/lib.rs::Rows::count",
            "FILE:src/lib.rs::Summary::total",
        ]

    def test_declarations_and_comments_are_not_calls(self, tmp_path):
        """An enum variant's declaration and commented-out code add no call
        edges."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                fn analysis(pattern: &str) {}

                pub enum Mode {
                    Search(u8),
                }

                fn scratch() {
                    // analysis("foo");
                }
                """
            },
        )
        assert [e for e in builder.edges if e["relation"].startswith("CALLS")] == []


class TestTypeScriptCallResolution:
    def test_call_statement_is_not_a_method_definition(self, tmp_path):
        """`name(arg, {` inside a function body is a call; only `name(...) {`
        declares an object-literal method."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function report(ctx, issue) {}

                function check(ctx) {
                  report(ctx, {
                    code: "bad",
                  });
                  return {
                    has(key) {
                      return true;
                    },
                  };
                }
                """
            },
        )
        ids = {n["id"] for n in builder.nodes}
        assert "FILE:a.ts::check::report" not in ids
        assert "FILE:a.ts::check::has" in ids
        assert _calls(builder, "FILE:a.ts::check") == ["FILE:a.ts::report"]

    def test_class_after_a_closed_class_is_not_nested_in_it(self, tmp_path):
        """A brace inside a string does not keep a class's scope open."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Opener {
                  open() {
                    return "{";
                  }
                }

                class Closer {
                  close() {}
                }
                """
            },
        )
        assert "FILE:a.ts::Closer::close" in {n["id"] for n in builder.nodes}

    def test_comments_are_not_calls(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function refine() {}

                function check() {
                  // refine() only answers yes or no
                  /* refine() */
                  return 1;
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::check") == []

    def test_declared_type_picks_the_method(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Doc {
                  typeset(line) {}
                }

                class Socket {
                  typeset(line) {}
                }

                function emit(doc: Doc) {
                  doc.typeset("x");
                }

                function send(peer) {
                  peer.typeset("x");
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::emit") == ["FILE:a.ts::Doc::typeset"]
        assert _calls(builder, "FILE:a.ts::send") == ["REF:peer.typeset"]

    def test_this_call_resolves_through_a_generic_base_class(self, tmp_path):
        """`class A<T extends B> extends C<T>` inherits from C, not B."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Decoy {
                  prepare() {}
                }

                class Base<T> {
                  prepare() {}
                }

                class Child<T extends Decoy> extends Base<T> {
                  run() {
                    this.prepare();
                  }
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::Child::run") == ["FILE:a.ts::Base::prepare"]

    def test_call_through_a_dependency_import_is_external(self, tmp_path):
        """A name imported from a declared dependency does not reach a
        project definition of that name; one imported from a project
        package does."""
        builder = _index(
            tmp_path,
            {
                "package.json": """
                {"name": "tool", "devDependencies": {"bundler": "^1.0.0"}}
                """,
                "scripts/make.ts": """
                export function build(depth) {}
                export function check(depth) {}
                """,
                "src/bundle.ts": """
                import * as path from "node:path";
                import {
                  build,
                } from "bundler";
                import { check } from "tool/scripts/make";

                export function bundle(entry) {
                  check(1);
                  path.check(entry);
                  return build({ entry });
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/bundle.ts::bundle") == [
            "FILE:scripts/make.ts::check"
        ]

    def test_call_through_a_project_import_reaches_that_module(self, tmp_path):
        """`ns.name()` on a namespace import reaches the imported file's
        definition, or the one it re-exports from next to it."""
        builder = _index(
            tmp_path,
            {
                "v1/util.ts": "export function issue(x) {}\n",
                "v2/util.ts": "export function issue(x) {}\n",
                "v2/schemas.ts": "export function text(x) {}\n",
                "v2/index.ts": 'export * from "./schemas.js";\n',
                "v3/schemas.ts": "export function text(x) {}\n",
                "v2/checks.ts": """
                import * as util from "./util.js";
                import * as z from "./index.js";

                export function refine(payload) {
                  util.issue(payload);
                  return z.text(payload);
                }
                """,
            },
        )
        assert _calls(builder, "FILE:v2/checks.ts::refine") == [
            "FILE:v2/schemas.ts::text",
            "FILE:v2/util.ts::issue",
        ]

    def test_parameter_on_its_own_line_types_the_receiver(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Socket {
                  typeset(line) {}
                }

                class Doc {
                  typeset(line) {}
                }

                function emit(
                  doc: Doc,
                  line,
                ) {
                  doc.typeset(line);
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::emit") == ["FILE:a.ts::Doc::typeset"]

    def test_call_of_a_parameter_is_not_a_same_named_definition(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export abstract class Shape {}

                export function make(Shape, options) {
                  return new Shape({
                    options,
                  });
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::make") == ["REF:Shape"]

    def test_dollar_names_are_definitions(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "core.ts": """
                export function $parse(input: string) {
                  return input;
                }

                export const $make = () => {
                  return $parse("a");
                };
                """
            },
        )
        assert _calls(builder, "FILE:core.ts::$make") == ["FILE:core.ts::$parse"]

    def test_annotated_arrow_constant_is_a_function(self, tmp_path):
        """A `const` holding an arrow function is a function whether or not
        it has a type annotation or parameters with defaults."""
        builder = _index(
            tmp_path,
            {
                "handlers.ts": """
                function helper() {}

                export const typed: Handler<Request> = async (req) => {
                  helper();
                };

                export const defaulted = (limit = 10, options = {}) => {
                  helper();
                };
                """
            },
        )
        assert _calls(builder, "FILE:handlers.ts::typed") == [
            "FILE:handlers.ts::helper"
        ]
        assert _calls(builder, "FILE:handlers.ts::defaulted") == [
            "FILE:handlers.ts::helper"
        ]

    def test_expression_bodied_arrow_ends_with_its_statement(self, tmp_path):
        """An arrow function whose expression body continues on the next
        line does not enclose the definitions that follow it."""
        builder = _index(
            tmp_path,
            {
                "fmt.ts": """
                const fmt = (n: number) =>
                  n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : n.toFixed(0);
                const wrap = (n: number) => ({
                  text: fmt(n),
                });

                function report(): void {
                  wrap(2);
                }
                """
            },
        )
        assert _calls(builder, "FILE:fmt.ts::wrap") == ["FILE:fmt.ts::fmt"]
        assert _calls(builder, "FILE:fmt.ts::report") == ["FILE:fmt.ts::wrap"]

    def test_inherited_method_comes_from_the_base_the_subclass_names(self, tmp_path):
        """Text extends the Schema of its own file, not the Schema another
        file declares."""
        builder = _index(
            tmp_path,
            {
                "legacy/types.ts": """
                export class Schema {
                  check() {}
                }
                """,
                "next/types.ts": """
                export class Schema {
                  check() {}
                }

                export class Text extends Schema {}

                export function run(text: Text) {
                  text.check();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:next/types.ts::run") == [
            "FILE:next/types.ts::Schema::check"
        ]

    def test_module_level_call_is_made_by_the_file(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "main.ts": """
                function setup() {}

                setup();
                """
            },
        )
        assert _calls(builder, "FILE:main.ts") == ["FILE:main.ts::setup"]

    def test_call_on_a_returned_value_is_typed_by_the_return_type(self, tmp_path):
        """`make().render()`, a variable bound to `make()`, the awaited
        value of a `Promise<Widget>` and `make()?.render()` all reach
        Widget.render, though Decoy has a `render` as well."""
        builder = _index(
            tmp_path,
            {
                "widgets.ts": """
                export class Widget {
                  render() {}
                  scaled(): this {
                    return this;
                  }
                }

                export class Decoy {
                  render() {}
                }

                function make(): Widget {
                  return new Widget();
                }

                async function load(): Promise<Widget> {
                  return new Widget();
                }

                function chained() {
                  make().scaled().render();
                }

                function bound() {
                  const widget = make();
                  widget.render();
                }

                async function awaited() {
                  const widget = await load();
                  widget.render();
                }

                function optional() {
                  make()?.render();
                }
                """
            },
        )
        render = "FILE:widgets.ts::Widget::render"
        make = "FILE:widgets.ts::make"
        assert _calls(builder, "FILE:widgets.ts::chained") == [
            render,
            "FILE:widgets.ts::Widget::scaled",
            make,
        ]
        assert _calls(builder, "FILE:widgets.ts::bound") == [render, make]
        assert _calls(builder, "FILE:widgets.ts::awaited") == [
            render,
            "FILE:widgets.ts::load",
        ]
        assert _calls(builder, "FILE:widgets.ts::optional") == [render, make]

    def test_package_import_follows_exports_and_reexports(self, tmp_path):
        """A subpath import goes to the file `exports` maps it to; a name
        the package's entry file re-exports goes to where it is defined."""
        builder = _index(
            tmp_path,
            {
                "packages/core/package.json": """
                {
                  "name": "@acme/core",
                  "exports": {
                    ".": {"types": "./src/index.ts", "default": "./src/index.ts"},
                    "./tools/*": "./src/tools/*.ts"
                  }
                }
                """,
                "packages/core/src/index.ts": """
                export { helper as assist } from "./tools/text";
                export * from "./tools/numbers";
                """,
                "packages/core/src/tools/text.ts": """
                export function helper() {}
                """,
                "packages/core/src/tools/numbers.ts": """
                export function total() {}
                """,
                "packages/app/src/decoys.ts": """
                export function helper() {}
                export function assist() {}
                export function total() {}
                """,
                "packages/app/src/main.ts": """
                import { assist, total } from "@acme/core";
                import { helper } from "@acme/core/tools/text";

                export function run() {
                  assist();
                  total();
                  helper();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:packages/app/src/main.ts::run") == [
            "FILE:packages/core/src/tools/numbers.ts::total",
            "FILE:packages/core/src/tools/text.ts::helper",
            "FILE:packages/core/src/tools/text.ts::helper",
        ]

    def test_call_of_a_local_variable_is_not_a_function_of_that_name(self, tmp_path):
        """A variable that holds a function is called by name. Where the
        variable is itself a definition, as an arrow function is, the call
        goes to it; otherwise it goes nowhere, and never to the module's
        function of the same name."""
        builder = _index(
            tmp_path,
            {
                "steps.ts": """
                function step() {}

                function format() {}

                function other() {}

                export function held() {
                  let step = other;
                  step();
                }

                export function defined() {
                  const format = (n: number) => {
                    return n;
                  };
                  format(1);
                }
                """
            },
        )
        assert _calls(builder, "FILE:steps.ts::held") == []
        assert _calls(builder, "FILE:steps.ts::defined") == [
            "FILE:steps.ts::defined::format"
        ]

    def test_object_literal_method_is_not_a_call(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "table.ts": """
                function build(spec: object) {
                  return spec;
                }

                function overwrite() {}

                function load() {}

                export const table = build({
                  overwrite(fn) {
                    return fn;
                  },
                  async load(id) {
                    return id;
                  },
                });
                """
            },
        )
        assert _calls(builder, "FILE:table.ts") == ["FILE:table.ts::build"]

    def test_arrow_constant_with_parameters_over_several_lines(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "multi.ts": """
                function helper(options: object) {}

                export const multi = (
                  first: string,
                  options = { deep: true },
                  second = "a;b",
                ) => {
                  helper(options);
                };

                export const below: (options: object) => void =
                  (options) => {
                    helper(options);
                  };
                """
            },
        )
        assert _calls(builder, "FILE:multi.ts::multi") == ["FILE:multi.ts::helper"]
        assert _calls(builder, "FILE:multi.ts::below") == ["FILE:multi.ts::helper"]

    def test_interface_and_function_of_one_name_are_two_nodes(self, tmp_path):
        """The function keeps `<scope>::<name>`; the interface declared
        under the same name is `<scope>::interface_<name>` and is still the
        type that name stands for. An interface alone keeps its ID."""
        builder = _index(
            tmp_path,
            {
                "widgets.ts": """
                export interface Widget {
                  render(): void;
                }

                export function Widget(): Widget {
                  return {} as Widget;
                }

                export interface Plain {
                  render(): void;
                }

                export function use(plain: Plain) {
                  Widget().render();
                  plain.render();
                }
                """
            },
        )
        nodes = {n["id"]: n["type"] for n in builder.nodes}
        assert nodes["FILE:widgets.ts::Widget"] == "Function"
        assert nodes["FILE:widgets.ts::interface_Widget"] == "Class"
        assert _calls(builder, "FILE:widgets.ts::use") == [
            "FILE:widgets.ts::Plain::render",
            "FILE:widgets.ts::Widget",
            "FILE:widgets.ts::interface_Widget::render",
        ]

    def test_import_through_a_tsconfig_path_alias(self, tmp_path):
        """`paths` and `baseUrl` are read from the nearest tsconfig, which
        may have comments and trailing commas and may extend another."""
        builder = _index(
            tmp_path,
            {
                "tsconfig.base.json": """
                {
                  // shared by every package
                  "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                      "@lib/*": ["libs/*/src/index.ts"], /* one per library */
                      "@lib/text/extra": ["libs/text/src/extra.ts"],
                    },
                  },
                }
                """,
                "app/tsconfig.json": """
                {
                  "extends": "../tsconfig.base",
                  "compilerOptions": {"strict": true}
                }
                """,
                "libs/text/src/index.ts": """
                export function trim() {}
                """,
                "libs/text/src/extra.ts": """
                export function pad() {}
                """,
                "libs/shared/util.ts": """
                export function join() {}
                """,
                "app/src/decoys.ts": """
                export function trim() {}
                export function pad() {}
                export function join() {}
                """,
                "app/src/main.ts": """
                import { trim } from "@lib/text";
                import { pad } from "@lib/text/extra";
                import { join } from "libs/shared/util";

                export function run() {
                  trim();
                  pad();
                  join();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app/src/main.ts::run") == [
            "FILE:libs/shared/util.ts::join",
            "FILE:libs/text/src/extra.ts::pad",
            "FILE:libs/text/src/index.ts::trim",
        ]

    def test_bare_and_method_calls_of_one_name_are_decided_apart(self, tmp_path):
        """A bare `settle()` can only be the function; `x.settle()` on an
        unknown receiver may be the function or Ledger's method, whichever
        of the two calls is resolved first."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export function settle() {}
                """,
                "b.ts": """
                export class Ledger {
                  settle() {}
                }
                """,
                "c.ts": """
                export function first() {
                  settle();
                }

                export function second(x) {
                  x.settle();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:c.ts::first") == ["FILE:a.ts::settle"]
        assert _calls(builder, "FILE:c.ts::second") == ["REF:x.settle"]

    def test_template_literal_text_is_not_a_call(self, tmp_path):
        """Text of a template literal adds no call; its `${...}` does."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function outline() {}
                function label() {}

                function emit(out, name) {
                  const head = `export const ${label()} = outline({`;
                  return head;
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::emit") == ["FILE:a.ts::label"]

    def test_single_quoted_text_is_not_a_call_but_an_apostrophe_hides_nothing(
        self, tmp_path
    ):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function outline() {}
                function label() {}

                function emit() {
                  const note = 'call outline() later';
                  // don't
                  return label();
                }

                function text() {
                  return <p>it's label() time</p>;
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::emit") == ["FILE:a.ts::label"]
        assert _calls(builder, "FILE:a.ts::text") == ["FILE:a.ts::label"]

    def test_same_named_type_in_another_file_is_another_type(self, tmp_path):
        """A declared type's methods are not looked up in a same-named type
        of another file."""
        builder = _index(
            tmp_path,
            {
                "v1/num.ts": """
                export class Num {
                  below(limit) {}
                }
                """,
                "v2/num.ts": """
                export interface Num {
                  above(limit): this;
                }

                export function clamp(value: Num) {
                  value.below(3);
                }
                """,
            },
        )
        assert _calls(builder, "FILE:v2/num.ts::clamp") == ["REF:value.below"]

    def test_chained_method_does_not_reach_a_local_of_that_name(self, tmp_path):
        """`.catchall()` in a call chain is not a variable called
        `catchall`."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function parse() {
                  const catchall = () => 1;
                  return catchall();
                }

                function build(z) {
                  return z.object({}).catchall(1);
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::build") == []


class TestGoCallResolution:
    def test_bare_call_is_the_function_not_a_method_of_that_name(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "cmd.go": """
                package cmd

                type Command struct{}

                func (c *Command) MarkRequired(name string) error {
                \treturn MarkRequired(c, name)
                }

                func MarkRequired(c *Command, name string) error {
                \treturn nil
                }
                """
            },
        )
        assert _calls(builder, "FILE:cmd.go::Command::MarkRequired") == [
            "FILE:cmd.go::MarkRequired"
        ]

    def test_receiver_type_resolves_a_common_method_name(self, tmp_path):
        """`c.Flags()` on a `*Command` receiver reaches Command.Flags, a name
        that is also a library method."""
        builder = _index(
            tmp_path,
            {
                "cmd.go": """
                package cmd

                type Command struct{}

                func (c *Command) Flags() int {
                \treturn 0
                }

                func (c *Command) Run() {
                \tc.Flags()
                }
                """
            },
        )
        assert _calls(builder, "FILE:cmd.go::Command::Run") == [
            "FILE:cmd.go::Command::Flags"
        ]

    def test_package_qualifier_follows_the_import(self, tmp_path):
        """`pkg.Name()` reaches the imported project package, or nothing when
        the import leaves the module."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/tool\n",
                "tool.go": """
                package tool

                type Job struct{}

                func (j *Job) Launch() {}

                func Launch() {}
                """,
                "doc/gen.go": """
                package doc

                import (
                \t"example.com/other/lib"
                \t"example.com/tool"
                )

                func Generate() {
                \ttool.Launch()
                \tlib.Launch()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:doc/gen.go::Generate") == ["FILE:tool.go::Launch"]

    def test_package_level_variable_types_the_receiver(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "cmd.go": """
                package cmd

                type Command struct{}

                func (c *Command) Flags() int {
                \treturn 0
                }

                var timesCmd = &Command{}

                func setup() {
                \ttimesCmd.Flags()
                }
                """
            },
        )
        assert _calls(builder, "FILE:cmd.go::setup") == ["FILE:cmd.go::Command::Flags"]

    def test_call_on_a_returned_value_is_typed_by_the_return_type(self, tmp_path):
        """Every link of a chain is recorded and typed by the first result
        of the call before it, as is a variable `:=` binds to a call."""
        builder = _index(
            tmp_path,
            {
                "maker.go": """
                package maker

                type Maker struct{}

                func NewMaker() *Maker {
                	return &Maker{}
                }

                func (m *Maker) Named(name string) *Maker {
                	return m
                }

                func (m *Maker) Assemble() (*Product, error) {
                	return &Product{}, nil
                }

                type Product struct{}

                func (p *Product) Ship() {}

                type Decoy struct{}

                func (d *Decoy) Named(name string) {}

                func (d *Decoy) Assemble() {}

                func (d *Decoy) Ship() {}

                func run() {
                	product, err := NewMaker().Named("x").Assemble()
                	product.Ship()
                }
                """
            },
        )
        assert _calls(builder, "FILE:maker.go::run") == [
            "FILE:maker.go::Maker::Assemble",
            "FILE:maker.go::Maker::Named",
            "FILE:maker.go::NewMaker",
            "FILE:maker.go::Product::Ship",
        ]

    def test_body_on_the_line_of_its_func_is_scanned(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                func helper() {}

                func short() { helper() }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::short") == ["FILE:app.go::helper"]

    def test_call_text_in_a_string_literal_is_not_a_call(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                func helper() {}

                func retry(n int) {}

                func quoted() string {
                \ta := "then retry(3) is called"
                \treturn a + `helper() too`
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::quoted") == []

    def test_raw_string_over_several_lines_is_not_code(self, tmp_path):
        """Text inside a raw string that spans lines is no call; a call
        between two such strings on one line is."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                func helper() string { return "" }

                func retry(n int) {}

                func usage() string {
                \treturn `first line
                then retry(3) // and more
                Simply type ` + helper() + ` help
                for retry(4) details.`
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::usage") == ["FILE:app.go::helper"]

    def test_call_of_a_local_value_is_not_a_function_of_that_name(self, tmp_path):
        """A function passed as a parameter or held in a variable is called
        by name; the call is not one of the package's function of that name."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                func check() {}

                func handler() {}

                func withParameter(name string, check func()) {
                \tcheck()
                }

                func withVariable() {
                \thandler := func() {}
                \thandler()
                }

                func direct() {
                \tcheck()
                \thandler()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::withParameter") == []
        assert _calls(builder, "FILE:app.go::withVariable") == []
        assert _calls(builder, "FILE:app.go::direct") == [
            "FILE:app.go::check",
            "FILE:app.go::handler",
        ]


class TestJavaCallResolution:
    def test_declared_type_picks_the_method(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "TreeReader.java": """
                public class TreeReader {
                    public int peek() { return 0; }
                }
                """,
                "StreamReader.java": """
                public class StreamReader {
                    public int peek() { return 0; }
                }
                """,
                "Adapter.java": """
                public class Adapter {
                    public int read(TreeReader in) {
                        return in.peek();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Adapter.java::Adapter::read") == [
            "FILE:TreeReader.java::TreeReader::peek"
        ]

    def test_call_is_made_by_the_method_it_is_in(self, tmp_path):
        """A call in a method or constructor body leaves that member; one in
        a field initialiser leaves the class."""
        builder = _index(
            tmp_path,
            {
                "Helper.java": """
                public class Helper {
                    static Helper create() { return null; }
                    static void prepare() {}
                    static void work() {}
                }
                """,
                "Client.java": """
                public class Client {
                    private final Helper helper = Helper.create();

                    Client() {
                        Helper.prepare();
                    }

                    void run() {
                        Helper.work();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client") == [
            "FILE:Helper.java::Helper::create"
        ]
        assert _calls(builder, "FILE:Client.java::Client::<init>") == [
            "FILE:Helper.java::Helper::prepare"
        ]
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Helper.java::Helper::work"
        ]

    def test_call_on_a_returned_value_is_typed_by_the_return_type(self, tmp_path):
        """Every link of a chain, on one line or continued on the next, is
        a call on what the link before returns; so is a call on a `var`
        bound to a call."""
        builder = _index(
            tmp_path,
            {
                "Maker.java": """
                public class Maker {
                    public Maker named(String name) { return this; }
                    public Product assemble() { return null; }
                }
                """,
                "Product.java": """
                public class Product {
                    public void ship() {}
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void named(String name) {}
                    public void assemble() {}
                    public void ship() {}
                }
                """,
                "Client.java": """
                public class Client {
                    void chained() {
                        new Maker().named("x").assemble().ship();
                    }

                    void continued() {
                        new Maker()
                            .named("x")
                            .assemble()
                            .ship();
                    }

                    void bound() {
                        var product = new Maker().assemble();
                        product.ship();
                    }
                }
                """,
            },
        )
        chain = [
            "FILE:Maker.java::Maker",
            "FILE:Maker.java::Maker::assemble",
            "FILE:Maker.java::Maker::named",
            "FILE:Product.java::Product::ship",
        ]
        assert _calls(builder, "FILE:Client.java::Client::chained") == chain
        assert _calls(builder, "FILE:Client.java::Client::continued") == chain
        assert _calls(builder, "FILE:Client.java::Client::bound") == [
            "FILE:Maker.java::Maker",
            "FILE:Maker.java::Maker::assemble",
            "FILE:Product.java::Product::ship",
        ]

    def test_generic_instantiation_is_a_call_of_the_type(self, tmp_path):
        """`new Box<String>(...)` instantiates Box, and a method called on
        it, or on an anonymous subclass with an empty body, is Box's."""
        builder = _index(
            tmp_path,
            {
                "Box.java": """
                public class Box<T> {
                    public T opened() { return null; }
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public Object opened() { return null; }
                }
                """,
                "Client.java": """
                import java.util.List;

                public class Client {
                    Object direct() {
                        return new Box<String>().opened();
                    }

                    Object anonymous() {
                        return new Box<List<String>>() {}.opened();
                    }
                }
                """,
            },
        )
        expected = ["FILE:Box.java::Box", "FILE:Box.java::Box::opened"]
        assert _calls(builder, "FILE:Client.java::Client::direct") == expected
        assert _calls(builder, "FILE:Client.java::Client::anonymous") == expected

    def test_call_on_what_an_external_call_returns_is_external(self, tmp_path):
        """`newBuilder` is no project method, so what it returns is not a
        project object and `settle` on it is not Ledger.settle, the only
        method of that name."""
        builder = _index(
            tmp_path,
            {
                "Ledger.java": """
                public class Ledger {
                    public void settle(int amount) {}
                }
                """,
                "Client.java": """
                public class Client {
                    void run(Message message) {
                        message.newBuilder().settle(100);
                        var built = message.newBuilder();
                        built.settle(200);
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == []

    def test_qualified_declared_type_is_read_whole(self, tmp_path):
        """`Wire.Builder` is the nested type of an imported class that the
        project does not define, not the project's own Builder."""
        builder = _index(
            tmp_path,
            {
                "Builder.java": """
                public class Builder {
                    public void build() {}
                }
                """,
                "Client.java": """
                import com.other.Wire;

                public class Client {
                    void run(Wire.Builder remote, Builder local) {
                        remote.build();
                        local.build();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Builder.java::Builder::build"
        ]

    def test_text_block_is_not_code(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "Client.java": """
                public class Client {
                    static String helper() {
                        return "";
                    }

                    static String usage() {
                        return \"\"\"
                            call helper() here // and
                            class Ghost { void nope() { helper(); } }
                            \"\"\" + other();
                    }

                    static String other() {
                        return "";
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::usage") == [
            "FILE:Client.java::Client::other"
        ]
        assert not [n for n in builder.node_registry if "Ghost" in n or "nope" in n]

    def test_call_on_an_anonymous_class_with_a_body(self, tmp_path):
        """A method called on the closing brace of `new Box<...>() { ... }`
        is Box's."""
        builder = _index(
            tmp_path,
            {
                "Box.java": """
                public class Box<T> {
                    public T opened() { return null; }
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public Object opened() { return null; }
                }
                """,
                "Client.java": """
                public class Client {
                    Object run() {
                        return new Box<String>() {
                            int extra = 1;
                        }.opened();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Box.java::Box",
            "FILE:Box.java::Box::opened",
        ]

    def test_receiver_of_non_project_type_is_external(self, tmp_path):
        """A method call on a variable declared with a type the project does
        not define, a static call on an imported non-project class, or a
        call on a `Type.class` object does not reach a project method of
        that name."""
        builder = _index(
            tmp_path,
            {
                "Hub.java": """
                public class Hub {
                    public void enroll(String name) {}
                }
                """,
                "Client.java": """
                import com.other.Widgets;

                public class Client {
                    public void run(Registry registry) {
                        registry.enroll("a");
                        Widgets.enroll("b");
                        Hub.class.enroll("c");
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == []


class TestProcessDirectory:
    """Tests for the directory walk and per-file dispatch."""

    @pytest.fixture
    def regex_builder(self, tmp_path):
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = False
        return builder

    @pytest.mark.parametrize("ext", [".jsx", ".mjs", ".cjs", ".mts", ".cts"])
    def test_js_family_extensions_parsed(self, regex_builder, tmp_path, ext):
        """Every JS/TS source extension reaches the TS parser."""
        (tmp_path / f"widget{ext}").write_text("export function render() {\n}\n")
        regex_builder.process_directory()

        ids = {n["id"] for n in regex_builder.nodes}
        assert f"FILE:widget{ext}::render" in ids
        assert regex_builder._get_language(f"FILE:widget{ext}") == "typescript"

    def test_every_indexed_extension_yields_a_file_node(self, regex_builder, tmp_path):
        """Each extension the walk accepts has a parser behind it."""
        for ext in generate._INDEXED_EXTENSIONS:
            (tmp_path / f"sample{ext}").write_text("<script>\n</script>\n")
        regex_builder.process_directory()

        ids = {n["id"] for n in regex_builder.nodes}
        # The Python source above is a syntax error, which yields no node.
        expected = {
            f"FILE:sample{ext}" for ext in generate._INDEXED_EXTENSIONS if ext != ".py"
        }
        assert expected <= ids

    def test_non_source_files_are_not_opened(
        self, regex_builder, tmp_path, monkeypatch
    ):
        """Files without a parser are skipped before being read."""
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "app.py").write_text("x = 1\n")
        opened = []
        real_open = os.open

        def recording_open(path, *args, **kwargs):
            opened.append(Path(path).name)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(generate.os, "open", recording_open)
        regex_builder.process_directory()

        assert opened == ["app.py"]

    def test_parser_failure_is_contained_to_its_file(
        self, regex_builder, tmp_path, monkeypatch
    ):
        """A parser raising on one file does not stop the walk."""
        (tmp_path / "a_bad.rs").write_text("fn broken() {}\n")
        (tmp_path / "b_good.py").write_text("def fine():\n    pass\n")

        def boom(self, file_path, rel_path, content):
            raise AttributeError("parser bug")

        monkeypatch.setattr(RustParser, "parse", boom)
        regex_builder.process_directory()

        assert "FILE:b_good.py::fine" in {n["id"] for n in regex_builder.nodes}

    def test_ast_grep_threshold_above_ten_times_default(self, tmp_path, monkeypatch):
        """A DESCRY_AST_GREP_MAX_FILES above 10x the default applies: every
        TS/JS file is counted."""
        monkeypatch.setattr(CodeGraphBuilder, "_AST_GREP_MAX_FILES_DEFAULT", 1)
        monkeypatch.setenv("DESCRY_AST_GREP_MAX_FILES", "12")
        monkeypatch.setattr(generate, "extract_calls_typescript", lambda path: [])
        monkeypatch.setattr(
            generate, "extract_imports_typescript", lambda path: {}, raising=False
        )
        for i in range(13):
            (tmp_path / f"m{i}.ts").write_text("")
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = True
        builder.process_directory()

        assert builder.use_ast_grep is False

    SVELTE = """<h1>Title</h1>
<script context="module">
function first() {
    const a = alpha();
}
</script>
<p>text</p>
<script>
function second() {
    const b = beta();
}
</script>
"""

    def test_svelte_call_edges_use_file_line_numbers(self, regex_builder, tmp_path):
        """Call edges from a <script> block carry lines of the .svelte file."""
        (tmp_path / "App.svelte").write_text(self.SVELTE)
        regex_builder.process_directory()

        linenos = {
            e["target"]: e["metadata"]["lineno"]
            for e in regex_builder.edges
            if e["relation"] == "CALLS"
        }
        assert linenos == {"REF:alpha": 4, "REF:beta": 10}

    def test_svelte_ast_grep_calls_attributed_per_block(self, tmp_path, monkeypatch):
        """ast-grep's whole-file line numbers map onto each <script> block."""
        (tmp_path / "App.svelte").write_text(self.SVELTE)
        monkeypatch.setattr(
            generate,
            "extract_calls_typescript",
            lambda path: [
                {"callee": "alpha", "lineno": 4},
                {"callee": "beta", "lineno": 10},
            ],
        )
        monkeypatch.setattr(
            generate, "extract_imports_typescript", lambda path: {}, raising=False
        )
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = True
        builder.process_directory()

        calls = [
            (e["source"], e["target"], e["metadata"]["lineno"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        assert calls == [
            ("FILE:App.svelte::first", "REF:alpha", 4),
            ("FILE:App.svelte::second", "REF:beta", 10),
        ]


class _SgResult:
    returncode = 0

    def __init__(self, stdout):
        self.stdout = stdout


class TestAstGrep:
    """Tests for the ast-grep subprocess wrappers."""

    def test_non_array_payload_yields_nothing(self, monkeypatch):
        """A JSON object where the match array should be is no matches."""
        monkeypatch.setattr(
            ast_grep.subprocess,
            "run",
            lambda *a, **kw: _SgResult('{"error": "unsupported language"}'),
        )
        assert list(ast_grep.extract_calls_rust("x.rs")) == []
        assert list(ast_grep.extract_calls_typescript("x.ts")) == []
        assert ast_grep.extract_imports_typescript("x.ts") == {
            "imports": {},
            "namespaces": {},
        }

    def test_unexecutable_binary_is_unavailable(self, monkeypatch):
        """An `sg` that exists but cannot be executed counts as absent."""

        def denied(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(ast_grep.subprocess, "run", denied)
        assert ast_grep.is_ast_grep_available() is False
        assert list(ast_grep.extract_calls_rust("x.rs")) == []
        assert list(ast_grep.extract_calls_typescript("x.ts")) == []
        assert ast_grep.extract_imports_typescript("x.ts")["imports"] == {}


class TestTypeScriptConfigurationNodes:
    """Tests for TypeScript configuration pattern detection (interceptors, middleware)."""

    @pytest.fixture
    def ts_config_content(self):
        """TypeScript file with interceptor and event handler patterns."""
        return """
import axios from 'axios';

const apiClient = axios.create({ baseURL: '/api/v1' });

// Request interceptor - adds auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = getAuthToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handles errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      await refreshToken();
    }
    return Promise.reject(error);
  }
);

export default apiClient;
"""

    def test_detects_request_interceptor(self, ts_config_content):
        """Should detect axios request interceptor as Configuration node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "client.ts"
            test_file.write_text(ts_config_content)

            builder = CodeGraphBuilder(tmpdir)
            parser = TSParser(builder)
            parser.parse(str(test_file), "client.ts", ts_config_content)

            config_nodes = [n for n in builder.nodes if n["type"] == "Configuration"]
            request_interceptors = [
                n
                for n in config_nodes
                if "request_interceptor" in n["metadata"]["name"]
            ]

            assert len(request_interceptors) == 1
            node = request_interceptors[0]
            assert node["metadata"]["config_type"] == "interceptor"
            assert "apiClient.interceptors.request.use" in node["metadata"]["signature"]

    def test_detects_response_interceptor(self, ts_config_content):
        """Should detect axios response interceptor as Configuration node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "client.ts"
            test_file.write_text(ts_config_content)

            builder = CodeGraphBuilder(tmpdir)
            parser = TSParser(builder)
            parser.parse(str(test_file), "client.ts", ts_config_content)

            config_nodes = [n for n in builder.nodes if n["type"] == "Configuration"]
            response_interceptors = [
                n
                for n in config_nodes
                if "response_interceptor" in n["metadata"]["name"]
            ]

            assert len(response_interceptors) == 1
            node = response_interceptors[0]
            assert node["metadata"]["config_type"] == "interceptor"
            assert (
                "apiClient.interceptors.response.use" in node["metadata"]["signature"]
            )

    def test_interceptor_linked_to_file(self, ts_config_content):
        """Configuration nodes should be linked to file via DEFINES edge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "client.ts"
            test_file.write_text(ts_config_content)

            builder = CodeGraphBuilder(tmpdir)
            parser = TSParser(builder)
            parser.parse(str(test_file), "client.ts", ts_config_content)

            file_id = "FILE:client.ts"
            defines_edges = [
                e
                for e in builder.edges
                if e["source"] == file_id and e["relation"] == "DEFINES"
            ]

            config_targets = [
                e["target"] for e in defines_edges if "interceptor" in e["target"]
            ]

            assert len(config_targets) == 2  # request + response interceptors

    def test_interceptor_has_line_numbers(self, ts_config_content):
        """Configuration nodes should have line number metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "client.ts"
            test_file.write_text(ts_config_content)

            builder = CodeGraphBuilder(tmpdir)
            parser = TSParser(builder)
            parser.parse(str(test_file), "client.ts", ts_config_content)

            config_nodes = [n for n in builder.nodes if n["type"] == "Configuration"]

            for node in config_nodes:
                assert "lineno" in node["metadata"]
                assert "end_lineno" in node["metadata"]
                assert node["metadata"]["lineno"] > 0
                assert node["metadata"]["end_lineno"] >= node["metadata"]["lineno"]


class TestTraitImplMetadata:
    """Tests for trait implementation metadata in Rust parsing."""

    def test_rust_parser_adds_trait_impl_metadata(self):
        """Should add trait_impl metadata when parsing impl Trait for Struct."""
        content = """
struct JwtAuth;

impl FromRequestParts<AppState> for JwtAuth {
    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, Self::Rejection> {
        Ok(JwtAuth)
    }
}
"""
        builder = CodeGraphBuilder(".")
        parser = RustParser(builder)
        parser.parse(Path("test.rs"), "test.rs", content)

        method_nodes = [n for n in builder.nodes if n["type"] == "Method"]
        assert len(method_nodes) == 1

        method = method_nodes[0]
        assert method["metadata"]["name"] == "from_request_parts"
        assert method["metadata"].get("trait_impl") == "FromRequestParts"

    def test_rust_parser_no_trait_impl_for_inherent_impl(self):
        """Should NOT add trait_impl metadata for inherent impl blocks."""
        content = """
struct Server;

impl Server {
    pub fn new() -> Self {
        Server
    }
}
"""
        builder = CodeGraphBuilder(".")
        parser = RustParser(builder)
        parser.parse(Path("test.rs"), "test.rs", content)

        method_nodes = [n for n in builder.nodes if n["type"] == "Method"]
        assert len(method_nodes) == 1

        method = method_nodes[0]
        assert method["metadata"]["name"] == "new"
        # Should not have trait_impl for inherent impl
        assert method["metadata"].get("trait_impl") is None


class TestNamingConventionNormalization:
    """Tests for naming convention normalization (camelCase/snake_case matching)."""

    def test_normalize_camel_to_snake(self):
        """Should convert camelCase to snake_case."""
        from descry.query import _normalize_name

        assert _normalize_name("getClient") == "get_client"
        assert _normalize_name("validateToken") == "validate_token"
        assert _normalize_name("createNewUser") == "create_new_user"

    def test_normalize_pascal_to_snake(self):
        """Should convert PascalCase to snake_case."""
        from descry.query import _normalize_name

        assert _normalize_name("GetClient") == "get_client"
        assert _normalize_name("ValidateToken") == "validate_token"
        assert _normalize_name("HTTPServer") == "http_server"

    def test_normalize_already_snake(self):
        """Should leave snake_case unchanged."""
        from descry.query import _normalize_name

        assert _normalize_name("get_client") == "get_client"
        assert _normalize_name("validate_token") == "validate_token"

    def test_get_variants(self):
        """Should return all naming convention variants."""
        from descry.query import _get_name_variants

        variants = _get_name_variants("getClient")
        assert "getClient" in variants
        assert "get_client" in variants


class TestCrateFilter:
    """Tests for the package/crate filter in search."""

    @pytest.fixture
    def multi_package_graph(self):
        """Create a graph with nodes from multiple packages (mirrors descry's own structure)."""
        graph = {
            "schema_version": 1,
            "nodes": [
                # src/descry package (Python)
                {
                    "id": "FILE:src/descry/handlers.py",
                    "type": "File",
                    "metadata": {
                        "path": "src/descry/handlers.py",
                        "name": "handlers.py",
                    },
                },
                {
                    "id": "FILE:src/descry/handlers.py::DescryService",
                    "type": "Class",
                    "metadata": {
                        "name": "DescryService",
                        "docstring": "Core service for graph queries",
                    },
                },
                # src/descry/scip subpackage (Python)
                {
                    "id": "FILE:src/descry/scip/parser.py",
                    "type": "File",
                    "metadata": {
                        "path": "src/descry/scip/parser.py",
                        "name": "parser.py",
                    },
                },
                {
                    "id": "FILE:src/descry/scip/parser.py::ScipIndex",
                    "type": "Class",
                    "metadata": {
                        "name": "ScipIndex",
                        "docstring": "Parses SCIP index for graph queries",
                    },
                },
                # src/native package (Rust)
                {
                    "id": "FILE:src/native/lib.rs",
                    "type": "File",
                    "metadata": {"path": "src/native/lib.rs", "name": "lib.rs"},
                },
                {
                    "id": "FILE:src/native/lib.rs::GraphStore",
                    "type": "Class",
                    "metadata": {
                        "name": "GraphStore",
                        "docstring": "Native store for graph queries",
                    },
                },
                # tests package (Python)
                {
                    "id": "FILE:tests/test_generate.py",
                    "type": "File",
                    "metadata": {
                        "path": "tests/test_generate.py",
                        "name": "test_generate.py",
                    },
                },
                {
                    "id": "FILE:tests/test_generate.py::TestStdlibFilter",
                    "type": "Class",
                    "metadata": {
                        "name": "TestStdlibFilter",
                        "docstring": "Tests for stdlib filter in graph queries",
                    },
                },
            ],
            "edges": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            f.flush()
            yield f.name

        os.unlink(f.name)

    def test_filter_by_package_src(self, multi_package_graph):
        """Should return only results from src/descry package."""
        from descry.query import GraphQuerier

        q = GraphQuerier(multi_package_graph)
        results = q.search_docs(["graph", "queries"], crate="src")

        assert len(results) > 0
        for r in results:
            assert r["id"].startswith("FILE:src/")

    def test_filter_by_package_includes_subpackages(self, multi_package_graph):
        """Filtering by top-level package should include subpackages (scip lives under src/)."""
        from descry.query import GraphQuerier

        q = GraphQuerier(multi_package_graph)
        results = q.search_docs(["graph", "queries"], crate="src")

        # Should find both handlers.py and scip/parser.py since both are under src/
        ids = {r["id"] for r in results}
        assert any("handlers" in i for i in ids)
        assert any("scip" in i for i in ids)

    def test_filter_by_tests(self, multi_package_graph):
        """Should return only results from tests package."""
        from descry.query import GraphQuerier

        q = GraphQuerier(multi_package_graph)
        results = q.search_docs(["graph", "queries"], crate="tests")

        assert len(results) > 0
        for r in results:
            assert r["id"].startswith("FILE:tests/")

    def test_no_filter_returns_all_packages(self, multi_package_graph):
        """Should return results from all packages when no filter."""
        from descry.query import GraphQuerier

        q = GraphQuerier(multi_package_graph)
        results = q.search_docs(["graph", "queries"])

        # Should find results from multiple packages
        paths = {r["id"] for r in results}
        has_src = any("src/descry" in p for p in paths)
        has_tests = any("tests/" in p for p in paths)
        assert has_src
        assert has_tests

    def test_nonexistent_crate_returns_empty(self, multi_package_graph):
        """Should return empty results for non-existent package."""
        from descry.query import GraphQuerier

        q = GraphQuerier(multi_package_graph)
        results = q.search_docs(["graph"], crate="nonexistent")

        assert len(results) == 0

    def test_combine_crate_and_lang_filters(self, multi_package_graph):
        """Should combine package and language filters."""
        from descry.query import GraphQuerier

        q = GraphQuerier(multi_package_graph)

        results = q.search_docs(["graph", "queries"], crate="src", lang="rust")
        assert [r["id"] for r in results] == ["FILE:src/native/lib.rs::GraphStore"]

        results = q.search_docs(["graph", "queries"], crate="src", lang="python")
        assert results
        assert all(".py::" in r["id"] and "FILE:src/" in r["id"] for r in results)


class TestExcludeTestsFilter:
    """Tests for the exclude_tests filter in search."""

    @pytest.fixture
    def graph_with_tests(self):
        """Create a graph with both production and test nodes."""
        graph = {
            "schema_version": 1,
            "nodes": [
                # Production files
                {
                    "id": "FILE:src/auth.rs",
                    "type": "File",
                    "metadata": {"path": "src/auth.rs", "name": "auth.rs"},
                },
                {
                    "id": "FILE:src/auth.rs::validate_token",
                    "type": "Function",
                    "metadata": {
                        "name": "validate_token",
                        "docstring": "Validate auth token",
                    },
                },
                # Test files
                {
                    "id": "FILE:tests/auth_test.rs",
                    "type": "File",
                    "metadata": {"path": "tests/auth_test.rs", "name": "auth_test.rs"},
                },
                {
                    "id": "FILE:tests/auth_test.rs::test_validate_token",
                    "type": "Function",
                    "metadata": {
                        "name": "test_validate_token",
                        "docstring": "Test validate token",
                    },
                },
                {
                    "id": "FILE:src/utils_test.py",
                    "type": "File",
                    "metadata": {"path": "src/utils_test.py", "name": "utils_test.py"},
                },
                {
                    "id": "FILE:src/utils_test.py::test_helper",
                    "type": "Function",
                    "metadata": {
                        "name": "test_helper",
                        "docstring": "Test helper function",
                    },
                },
            ],
            "edges": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            f.flush()
            yield f.name

        os.unlink(f.name)

    def test_exclude_tests_filters_test_directories(self, graph_with_tests):
        """Should filter out files in tests/ directory."""
        from descry.query import GraphQuerier

        q = GraphQuerier(graph_with_tests)
        results = q.search_docs(["validate"], exclude_tests=True)

        # Should find production code
        assert any("validate_token" in r["metadata"]["name"] for r in results)
        # Should NOT find test code in tests/ dir
        assert not any("test_validate_token" in r["metadata"]["name"] for r in results)

    def test_exclude_tests_filters_test_suffixes(self, graph_with_tests):
        """Should filter out files with _test.py suffix."""
        from descry.query import GraphQuerier

        q = GraphQuerier(graph_with_tests)
        results = q.search_docs(["helper"], exclude_tests=True)

        # Should NOT find test_helper from utils_test.py
        assert not any("test_helper" in r["metadata"]["name"] for r in results)

    def test_exclude_tests_filters_test_functions(self, graph_with_tests):
        """Should filter out functions starting with test_."""
        from descry.query import GraphQuerier

        q = GraphQuerier(graph_with_tests)
        results = q.search_docs(["token"], exclude_tests=True)

        # Should find validate_token
        names = [r["metadata"]["name"] for r in results]
        assert "validate_token" in names
        # Should NOT find test_validate_token
        assert "test_validate_token" not in names

    def test_exclude_tests_false_includes_all(self, graph_with_tests):
        """Should include test code when exclude_tests=False (default)."""
        from descry.query import GraphQuerier

        q = GraphQuerier(graph_with_tests)
        results = q.search_docs(["validate"], exclude_tests=False)

        names = [r["metadata"]["name"] for r in results]
        # Should find both production and test code
        assert "validate_token" in names
        assert "test_validate_token" in names


class TestRaisedThresholds:
    """Tests for the raised truncation thresholds (Phase 5).

    Note: Truncation only happens when BOTH conditions are met:
    - token_count >= threshold (1000 for primary, 1800 for secondary)
    - total_lines > 100
    """

    @pytest.fixture
    def large_function_graph(self, tmp_path):
        """Create a graph with a large function (>100 lines)."""
        # Create a source file with 150 lines (over 100 line threshold)
        source_file = tmp_path / "large.rs"
        lines = ["fn large_function() {\n"]
        for i in range(148):  # ~150 lines total
            lines.append(f"    let var{i} = {i};\n")
        lines.append("}\n")
        source_file.write_text("".join(lines))

        graph = {
            "schema_version": 1,
            "nodes": [
                {
                    "id": f"FILE:{source_file}::large_function",
                    "type": "Function",
                    "metadata": {
                        "name": "large_function",
                        "lineno": 1,
                        "end_lineno": 150,
                        "token_count": 1500,
                    },
                },
            ],
            "edges": [],
        }

        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(graph))
        return str(graph_file), str(source_file)

    def test_under_threshold_no_truncation(self, large_function_graph):
        """Functions under 1000 tokens should NOT be truncated regardless of line count."""
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        # 950 tokens is under the 1000 threshold, should show full source
        source = q.get_smart_source(source_file, 1, 150, 950, full=False)

        # Should NOT contain truncation marker (950 < 1000)
        assert "lines omitted" not in source
        assert source.count("let var") == 148

    def test_over_threshold_truncated(self, large_function_graph):
        """Functions over 1000 tokens AND over 100 lines should be truncated."""
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        # 1200 tokens is over the new 1000 threshold, 150 lines is over 100
        source = q.get_smart_source(source_file, 1, 150, 1200, full=False)

        # Should contain truncation marker
        assert "lines omitted" in source
        # Not all 148 lines should be present
        assert source.count("let var") < 148

    def test_secondary_threshold_60_15_treatment(self, large_function_graph):
        """Functions between 1000-1800 tokens get 60+15 line treatment."""
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        # 1200 tokens (between 1000-1800) should use 60+15 line treatment
        source = q.get_smart_source(source_file, 1, 150, 1200, full=False)

        # Should be truncated
        assert "lines omitted" in source
        # Head should have ~60 lines, tail ~15 lines = ~75 total
        # 150 - 75 = ~75 omitted
        line_count = source.count("let var")
        assert 70 <= line_count <= 80, f"Expected ~75 lines, got {line_count}"

    def test_very_large_threshold_50_15_treatment(self, large_function_graph):
        """Functions over 1800 tokens get 50+15 line treatment."""
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        # 2000 tokens (over 1800) should use 50+15 line treatment
        source = q.get_smart_source(source_file, 1, 150, 2000, full=False)

        # Should be truncated
        assert "lines omitted" in source
        # Head should have ~50 lines, tail ~15 lines = ~65 total
        line_count = source.count("let var")
        assert 60 <= line_count <= 70, f"Expected ~65 lines, got {line_count}"
