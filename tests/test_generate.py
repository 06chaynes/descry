"""Tests for descry.generate — CodeGraphBuilder, parsers, and filters."""

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from descry import ast_grep, generate
from descry.generate import (
    BaseParser,
    CallResolver,
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

    def test_nested_function_default_arguments_are_called_by_the_outer(self):
        """A default argument is evaluated where its function is defined."""
        content = """
def outer():
    def inner(a=make_default(), *, b=make_other(), c):
        helper()
    inner(c=1)
"""
        builder = CodeGraphBuilder(".")
        PythonParser(builder).parse(Path("test.py"), "test.py", content)

        calls = {
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        }
        assert calls == {
            ("FILE:test.py::outer", "REF:make_default"),
            ("FILE:test.py::outer", "REF:make_other"),
            ("FILE:test.py::outer", "REF:inner"),
            ("FILE:test.py::outer::inner", "REF:helper"),
        }


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

    def test_licence_notice_below_the_first_line_is_not_a_docstring(self):
        """A licence notice documents no symbol wherever the file has it; a
        comment that only mentions a licence still does."""
        lines = [
            "<?php",
            "// Copyright (c) Example Corp.",
            "// Licensed under the MIT License.",
            "function main() {",
            "}",
            "// Returns the license a package declares.",
            "function license() {",
        ]
        parser = BaseParser(None)
        assert parser.get_leading_docstring(lines, 3) == ""
        assert parser.get_leading_docstring(lines, 6) == (
            "Returns the license a package declares."
        )

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

    def test_class_property_holding_an_arrow_function_is_a_method(self):
        """`name = (params): Type => {` in a class body declares a method."""
        content = """
export class Shape {
    static create = <T>(side: T): Shape => {
        return new Shape();
    };
    side = (1 + 2) * 3;
}
"""
        builder = CodeGraphBuilder(".")
        builder.use_ast_grep = False
        TSParser(builder).parse(Path("shape.ts"), "shape.ts", content)

        members = {
            n["id"]: (n["type"], n["metadata"].get("return_type"))
            for n in builder.nodes
            if n["id"].startswith("FILE:shape.ts::Shape::")
        }
        assert members == {"FILE:shape.ts::Shape::create": ("Method", "Shape")}

    def test_signature_over_several_lines_is_read_to_its_end(self):
        """Parameters, type parameters and a return type may each run over
        lines and hold braces; the body opens at the `{` after them."""
        content = """
export function hash<Alg extends string>(
    alg: Alg,
    params?: {
        enc?: string; // one of { hex, base64 }
    } & Options
): Format<`${Alg}_hex`> {
    return build(alg);
}

export interface Schema {
    refine<Ch extends (arg: unknown) => unknown | Promise<unknown>>(
        check: Ch
    ): Refined<this>;
    shape(): {
        strict: boolean;
    };
    catch(fallback: unknown): Caught<this>;
    parse(data: unknown): unknown
    brand(): this
}
"""
        builder = CodeGraphBuilder(".")
        builder.use_ast_grep = False
        TSParser(builder).parse(Path("hash.ts"), "hash.ts", content)

        nodes = {
            n["id"]: (n["metadata"].get("return_type"), n["metadata"].get("end_lineno"))
            for n in builder.nodes
            if n["type"] in ("Function", "Method")
        }
        assert nodes == {
            "FILE:hash.ts::hash": ("Format<`${Alg}_hex`>", 9),
            "FILE:hash.ts::Schema::refine": ("Refined<this>", 12),
            "FILE:hash.ts::Schema::shape": ("{ strict: boolean; }", 15),
            "FILE:hash.ts::Schema::catch": ("Caught<this>", 18),
            "FILE:hash.ts::Schema::parse": ("unknown", 19),
            "FILE:hash.ts::Schema::brand": ("this", 20),
        }
        calls = [e for e in builder.edges if e["relation"] == "CALLS"]
        assert [(e["source"], e["target"]) for e in calls] == [
            ("FILE:hash.ts::hash", "REF:build")
        ]


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

    def test_library_named_method_the_project_defines_stays_unresolved(self, tmp_path):
        """`get` on a receiver of unknown type may be the project's `get` or
        a library's: the call is neither dropped nor pinned to the project's."""
        builder = _index(
            tmp_path,
            {
                "store.py": """
                class Store:
                    def get(self, key): ...

                def read(store, table):
                    store.get("a")
                    table.get("b")
                """
            },
        )
        assert _calls(builder, "FILE:store.py::read") == [
            "REF:store.get",
            "REF:table.get",
        ]

    def test_global_object_is_external_and_a_namesake_variable_is_not(self, tmp_path):
        """`JSON` and `console` are global objects; `time`, a variable
        bound to a call's result, is not the `time` module, so a call on
        it is neither dropped nor pinned to the one project definition."""
        builder = _index(
            tmp_path,
            {
                "schema.ts": """
                export class Schema {
                    validate(value) {
                        return value;
                    }
                    log(value) {
                        return value;
                    }
                }

                export function stringify(value) {
                    return String(value);
                }

                function make() {
                    return undefined;
                }

                export function check(value) {
                    const time = make();
                    time.validate(value);
                    console.log(value);
                    return JSON.stringify(value);
                }
                """
            },
        )
        assert _calls(builder, "FILE:schema.ts::check") == [
            "FILE:schema.ts::make",
            "REF:time.validate",
        ]

    def test_method_of_a_literal_is_external(self, tmp_path):
        """A regular expression or string literal has the methods of its
        built-in type; a receiver of unknown type may have the project's."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Suite {
                    test(name) {}
                    trim() {}
                }

                export function check(body, suite) {
                    suite.test(body);
                    return /^[A-Z]/.test(body) || `${body} `.trim() || "x".trim();
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::check") == ["REF:suite.test"]

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

    def test_built_in_method_name_is_not_pinned_to_the_one_project_method(
        self, tmp_path
    ):
        """`has` is a method of the language's own Map and Set, so a
        receiver of unknown type may not be the project's Registry."""
        builder = _index(
            tmp_path,
            {
                "registry.ts": """
                export class Registry {
                    has(key) {
                        return false;
                    }
                }

                export function seen(items, key) {
                    return items.has(key);
                }
                """,
                "app/Walk.java": """
                package app;

                class Excluder {
                    int modifiers() {
                        return 0;
                    }
                }

                class Walk {
                    void each(java.util.List<Object> all) {
                        all.forEach(r -> r.modifiers());
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:registry.ts::seen") == ["REF:items.has"]
        assert "REF:r.modifiers" in _calls(builder, "FILE:app/Walk.java::Walk::each")


class TestStdlibNamedBases:
    """Where an INHERITS edge to a base named like a stdlib type leads."""

    @staticmethod
    def _bases(builder):
        return {
            (e["source"].split("::", 1)[1], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        }

    def test_python_base_the_module_defines_or_imports(self, tmp_path):
        """`Error` is the project's class where the module defines or
        imports it, and the builtin's namesake nowhere else."""
        builder = _index(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/errors.py": """
                class Error(Exception): ...

                class ParseError(Error): ...
                """,
                "pkg/api.py": """
                from . import errors
                from .errors import Error

                class ApiError(Error): ...

                class Timeout(errors.Error): ...
                """,
                "other.py": """
                class Stray(Error): ...
                """,
            },
        )
        project_error = "FILE:pkg/errors.py::Error"
        assert self._bases(builder) == {
            ("ParseError", project_error),
            ("ApiError", project_error),
            ("Timeout", project_error),
        }

    def test_rust_trait_imported_from_the_project(self, tmp_path):
        """An imported project trait is the base; a std trait is not the
        project type that shares its name, nor the type implementing it."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub mod matcher;
                pub mod regex;
                """,
                "src/matcher.rs": """
                pub trait Matcher {}

                pub struct Debug;
                """,
                "src/regex.rs": """
                use std::fmt;
                use crate::matcher::Matcher;

                pub struct Error;

                impl std::error::Error for Error {}

                impl fmt::Debug for Error {}

                pub struct RegexMatcher;

                impl Matcher for RegexMatcher {}
                """,
            },
        )
        assert self._bases(builder) == {
            ("RegexMatcher", "FILE:src/matcher.rs::Matcher")
        }

    def test_typescript_base_defined_in_the_same_file(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Test {}

                class Subtest extends Test {}

                class Failure extends Error {}
                """
            },
        )
        assert self._bases(builder) == {("Subtest", "FILE:a.ts::Test")}


class _ScipStub:
    """A SCIP index that knows references: ``(file, line)`` or ``(file,
    line, column)`` -> the ``(node_id, file, line)`` of the definition."""

    def __init__(self, references, outside=()):
        self.references = references
        self.outside = outside

    def definition_at(self, ref_name, source_file, line, column):
        return self.references.get(
            (source_file, line, column), self.references.get((source_file, line))
        )

    def outside_at(self, ref_name, source_file, line):
        return (source_file, line) in self.outside


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

    def test_call_scip_saw_leave_the_project_is_external(self, tmp_path):
        """`probe()` has two project definitions to be one of, but SCIP
        recorded it as a call of code outside the project."""
        (tmp_path / "lib.rs").write_text(textwrap.dedent(self.SOURCE))
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = False
        builder.process_directory()
        builder.resolve_references(_ScipStub({}, outside={("lib.rs", 9)}))
        assert not [e for e in builder.edges if e["source"] == "FILE:lib.rs::run"]

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

    def test_fixture_that_yields_gives_what_it_yields(self, tmp_path):
        """A fixture annotated as the generator it is gives what it yields,
        also from another generator."""
        builder = _index(
            tmp_path,
            {
                "app.py": self.APP,
                "tests/test_app.py": """
                from typing import Iterator

                import pytest
                from app import Service

                def serve(service: Service) -> Iterator[Service]: ...

                @pytest.fixture
                def svc() -> Iterator[Service]:
                    yield from serve(Service())

                def test_run(svc):
                    svc.settle()
                """,
            },
        )
        assert _calls(builder, "FILE:tests/test_app.py::test_run") == [
            "FILE:app.py::Service::settle"
        ]


class TestPythonCallResolution:
    def test_library_named_call_on_unknown_receiver_is_not_pinned(self, tmp_path):
        """`requests.get` names a known library: on a receiver of unknown
        type the one project `get` is not pinned, nor is the call dropped."""
        builder = _index(
            tmp_path,
            {
                "api.py": """
                class Client:
                    def get(self): ...

                def fetch(client):
                    return client.get("/x"), requests.get("https://e")
                """
            },
        )
        calls = _calls(builder, "FILE:api.py::fetch")
        assert "FILE:api.py::Client::get" not in calls
        assert any(t.endswith("get") for t in calls)

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
                    def skipTest(self): ...

                class Case(unittest.TestCase):
                    def test_it(self):
                        self.skipTest()
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

    def test_star_import_reexports_the_public_names_of_a_module(self, tmp_path):
        """`pkg.Client` is the class `pkg` takes from `_impl` with `import *`;
        a private name is not taken."""
        builder = _index(
            tmp_path,
            {
                "decoy.py": """
                class Client:
                    def send(self): ...

                def _hidden(): ...
                """,
                "pkg/__init__.py": "from ._impl import *\n",
                "pkg/_impl.py": """
                class Client:
                    def send(self): ...

                def _hidden(): ...
                """,
                "main.py": """
                import pkg

                def run():
                    client = pkg.Client()
                    client.send()
                    pkg._hidden()
                """,
            },
        )
        assert _calls(builder, "FILE:main.py::run") == [
            "FILE:pkg/_impl.py::Client",
            "FILE:pkg/_impl.py::Client::send",
            "REF:pkg._hidden",
        ]

    def test_base_class_is_the_class_its_import_names(self, tmp_path):
        """A base imported under another name, and a base its subclass shares
        the name of, are the class the import names."""
        builder = _index(
            tmp_path,
            {
                "lib/__init__.py": "",
                "lib/core.py": """
                class View:
                    def render(self): ...
                """,
                "decoy.py": """
                class Page:
                    def render(self): ...
                """,
                "app.py": """
                import lib.core
                from lib.core import View as Base

                class Page(Base): ...

                class View(lib.core.View): ...

                def run():
                    Page().render()
                    View().render()
                """,
            },
        )
        assert _calls(builder, "FILE:app.py::run") == [
            "FILE:app.py::Page",
            "FILE:app.py::View",
            "FILE:lib/core.py::View::render",
            "FILE:lib/core.py::View::render",
        ]

    def test_base_written_with_type_arguments_is_a_base(self, tmp_path):
        """`Crate(Box[int])` has the methods of Box, and `Table(Mapping[str,
        int])` those the standard library's data gives a mapping."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                from typing import Generic, Mapping, TypeVar

                T = TypeVar("T")

                class Other:
                    def items(self): ...
                    def load(self): ...

                class Box(Generic[T]):
                    def load(self): ...

                class Crate(Box[int]): ...

                class Table(Mapping[str, int]): ...

                def run(crate: Crate, table: Table):
                    crate.load()
                    table.items()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == ["FILE:m.py::Box::load"]

    def test_relative_import_from_a_namespace_package_inside_a_package(self, tmp_path):
        """`pkg/sub` has no `__init__.py`: `..util` in it is still `pkg.util`."""
        builder = _index(
            tmp_path,
            {
                "util.py": "def helper(): ...\n",
                "pkg/__init__.py": "",
                "pkg/util.py": "def helper(): ...\n",
                "pkg/sub/mod.py": """
                from ..util import helper

                def run():
                    helper()
                """,
            },
        )
        assert _calls(builder, "FILE:pkg/sub/mod.py::run") == [
            "FILE:pkg/util.py::helper"
        ]

    def test_with_binds_what_enter_returns(self, tmp_path):
        """`__enter__` returns the type its `self` is annotated with, the type
        it names, or for `open` what the standard library's data says."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                from typing import TypeVar

                T = TypeVar("T", bound="Session")

                class Other:
                    def send(self): ...
                    def read(self): ...

                class Session:
                    def __enter__(self: T) -> T: ...
                    def send(self): ...

                class Pool:
                    def __enter__(self) -> Session: ...

                def run(path):
                    with Session() as session, Pool() as pooled:
                        session.send()
                        pooled.send()
                    with open(path) as handle:
                        handle.read()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Pool",
            "FILE:m.py::Session",
            "FILE:m.py::Session::send",
            "FILE:m.py::Session::send",
        ]

    def test_literal_is_a_value_of_its_builtin_type(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Url:
                    def join(self): ...
                    def encode(self): ...
                    def get(self): ...
                    def index(self): ...

                def run(parts, name):
                    ", ".join(parts)
                    f"{name}!".encode()
                    {"a": 1}.get(name)
                    [1, 2].index(name)
                    ("a" + name).encode()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == []

    def test_standard_container_holds_what_its_type_arguments_say(self, tmp_path):
        """A subscript, a method and a loop over a `list[Job]` or a `dict[str,
        Job]` give a Job, however the annotation is written."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                import typing
                from typing import Optional

                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...

                class Queue:
                    def __init__(self, jobs: list[Job], named: "dict[str, Job]"):
                        self.jobs = jobs
                        self.named: typing.Dict[str, "Job"] = named
                        self.last: Optional[Job] = None

                    def drain(self, name, extra: typing.Sequence[Job] | None):
                        self.jobs[0].run()
                        self.jobs.pop().run()
                        self.named[name].run()
                        self.named.get(name).run()
                        self.last.run()
                        extra[0].run()
                        self.jobs[1:].run()
                        for job in self.named.values():
                            job.run()
                        for key, job in self.named.items():
                            key.strip()
                            job.run()
                        [job.run() for job in self.jobs]
                """
            },
        )
        assert _calls(builder, "FILE:m.py::Queue::drain") == [
            *["FILE:m.py::Job::run"] * 9,
            "REF:run",
        ]

    def test_return_annotation_with_type_arguments_types_the_result(self, tmp_path):
        """`Optional[Job]` is a Job; a union of two classes is neither."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                from typing import Iterator, Optional, Union

                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...

                def jobs() -> list[Job]: ...
                def find() -> Optional[Job]: ...
                def each() -> Iterator[Job]: ...
                def either() -> Union[Job, Other]: ...

                def run():
                    jobs()[0].run()
                    if (job := find()) is not None:
                        job.run()
                    for job in each():
                        job.run()
                    either().run()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            *["FILE:m.py::Job::run"] * 3,
            "FILE:m.py::each",
            "FILE:m.py::either",
            "FILE:m.py::find",
            "FILE:m.py::jobs",
            "REF:run",
        ]

    def test_self_is_the_class_the_method_is_called_on(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "m.py": """
                from typing import Self

                class Other:
                    def build(self): ...

                class Base:
                    def copy(self) -> Self: ...

                class Query(Base):
                    def build(self): ...

                def run():
                    Query().copy().build()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Base::copy",
            "FILE:m.py::Query",
            "FILE:m.py::Query::build",
        ]

    def test_unpacked_tuple_item_has_the_type_at_its_position(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Other:
                    def run(self): ...
                    def stop(self): ...

                class Job:
                    def run(self): ...

                class Worker:
                    def stop(self): ...

                def pair() -> tuple[Job, Worker]: ...

                def run(pairs: list[tuple[Job, Worker]]):
                    job, worker = pair()
                    job.run()
                    worker.stop()
                    for first, second in pairs:
                        first.run()
                        second.stop()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Job::run",
            "FILE:m.py::Job::run",
            "FILE:m.py::Worker::stop",
            "FILE:m.py::Worker::stop",
            "FILE:m.py::pair",
        ]

    def test_comprehension_variable_is_bound_only_inside_it(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...

                def run(names):
                    job = Job()
                    [job.strip() for job in names]
                    job.run()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Job",
            "FILE:m.py::Job::run",
        ]

    def test_lambda_parameter_is_what_the_standard_function_passes(self, tmp_path):
        """`re.sub` passes its callable a match: `found.group` is no call of
        the one project `group`."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                import re

                class Token:
                    def group(self): ...

                def run(text):
                    re.sub("a", lambda found: found.group(0), text)
                    re.compile("a").sub(lambda found: found.group(0), text)
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == []

    def test_isinstance_and_class_pattern_narrow_the_name_they_test(self, tmp_path):
        """Inside the `if`, after the `assert` and in the `case` the name is
        an instance of the class, unless the block rebinds it."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...

                def run(item, other, late, source):
                    if isinstance(item, Job):
                        item.run()
                    assert isinstance(other, Job)
                    other.run()
                    if isinstance(late, Job):
                        late = source
                        late.run()
                    match source:
                        case Job():
                            source.run()
                        case Job() as job:
                            job.run()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            *["FILE:m.py::Job::run"] * 4,
            "REF:late.run",
        ]

    def test_attribute_has_the_type_it_is_declared_or_assigned(self, tmp_path):
        """A module's variable, an attribute a base class assigns, also
        beside `None`, a property and a class held in a class attribute."""
        builder = _index(
            tmp_path,
            {
                "state.py": """
                class Other:
                    def get(self): ...
                    def run(self): ...

                class Store:
                    def get(self): ...

                class Job:
                    def run(self): ...

                class Proxy: ...

                store: Store = Proxy()
                """,
                "app.py": """
                import state
                from state import Job, Store

                class Base:
                    def __init__(self):
                        self.job = Job()
                        self.late = None

                    def start(self):
                        self.late = Job()

                class Runner(Base):
                    store_class = Store

                    @property
                    def current(self) -> Job: ...

                def run(runner: Runner):
                    state.store.get()
                    runner.job.run()
                    runner.late.run()
                    runner.current.run()
                    runner.store_class().get()
                """,
            },
        )
        assert _calls(builder, "FILE:app.py::run") == [
            *["FILE:state.py::Job::run"] * 3,
            "FILE:state.py::Store",
            "FILE:state.py::Store::get",
            "FILE:state.py::Store::get",
        ]

    def test_none_says_nothing_of_what_a_name_holds(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "decoy.py": "def helper(): ...\n",
                "lib.py": "def helper(): ...\n",
                "main.py": """
                try:
                    from lib import helper
                except ImportError:
                    helper = None

                def run():
                    helper()
                """,
            },
        )
        assert _calls(builder, "FILE:main.py::run") == ["FILE:lib.py::helper"]

    def test_call_of_a_value_is_no_call_of_a_definition(self, tmp_path):
        """A parameter that is called calls what it holds, whatever
        definitions have its name; an instance that is called its `__call__`."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                from typing import Callable

                class Hooks:
                    def send(self): ...
                    def receive(self): ...

                class Handler:
                    def __call__(self): ...

                def run(send, receive: Callable[[], bytes], handler: Handler):
                    send()
                    receive()
                    handler()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == ["FILE:m.py::Handler::__call__"]

    def test_attribute_of_a_decorated_function_is_one_of_what_the_decorator_made(
        self, tmp_path
    ):
        """`cli` is what `click.group()` returned, so `cli.command` is no
        method of the project's."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                import click

                class Group:
                    def command(self): ...

                @click.group()
                def cli(): ...

                @cli.command()
                def sub(): ...
                """
            },
        )
        assert _calls(builder, "FILE:m.py::sub") == []

    def test_class_object_is_the_class(self, tmp_path):
        """A value annotated `type[Job]`, and `job.__class__`, are Job."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Other:
                    def run(self): ...

                    @classmethod
                    def create(cls): ...

                class Job:
                    def run(self): ...

                    @classmethod
                    def create(cls): ...

                def run(kind: type[Job], job: Job):
                    kind.create()
                    kind().run()
                    job.__class__.create()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Job",
            "FILE:m.py::Job::create",
            "FILE:m.py::Job::create",
            "FILE:m.py::Job::run",
        ]

    def test_standard_method_name_alone_does_not_pin_a_receiver(self, tmp_path):
        """`locate` is a method of `importlib.metadata.PackagePath` and
        `__html__` a protocol any class may implement: on a receiver of
        unknown type the one project definition is not the only one that
        fits."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Finder:
                    def locate(self): ...
                    def __html__(self): ...
                    def frobnicate(self): ...

                def run(thing):
                    thing.locate()
                    thing.__html__()
                    thing.frobnicate()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Finder::frobnicate",
            "REF:thing.__html__",
            "REF:thing.locate",
        ]

    def test_operator_and_variadic_parameters_are_standard_values(self, tmp_path):
        """`root / "x"` is a path, `args` a tuple, `kwargs` and `__dict__`
        dicts, and `__init__` without a base that defines it `object`'s."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                from pathlib import Path

                class Locator:
                    def resolve(self): ...
                    def setdefault(self): ...
                    def count(self): ...
                    def pop(self): ...

                class Plain:
                    def __init__(self, *args, **kwargs):
                        super().__init__()
                        kwargs.setdefault("a", 1)
                        args.count(1)
                        self.__dict__.pop("a")

                def run(root: Path):
                    (root / "x").resolve()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::Plain::__init__") == []
        assert _calls(builder, "FILE:m.py::run") == []

    def test_alias_and_union_of_standard_types(self, tmp_path):
        """A name bound to a standard type is that type, and a union of
        standard types some standard value; a union of two project classes
        is neither of them."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                import typing

                Scope = typing.MutableMapping[str, typing.Any]

                class Other:
                    def get(self): ...
                    def decode(self): ...
                    def run(self): ...

                class Job:
                    def run(self): ...

                def run(scope: Scope, raw: str | bytes, either: Job | Other):
                    scope.get("a")
                    raw.decode()
                    either.run()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == ["REF:either.run"]

    def test_standard_function_returns_what_the_data_says(self, tmp_path):
        """`re.compile(...).match(...)` is a match, whose `group` is the
        standard library's; what `copy.copy` returns the data cannot say, so
        `run` on it is not dropped."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                import copy
                import re

                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...
                    def group(self): ...

                def run(job):
                    copy.copy(job).run()
                    re.compile("a").match("a").group(0)
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == ["REF:run"]

    def test_unannotated_function_returns_what_its_body_does(self, tmp_path):
        """A function without a return annotation returns what its `return`
        statements agree on, `None` aside: an instance, `self`, an
        attribute. A property is read as that. A generator, and a function
        whose returns differ, return neither."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...

                    def named(self, name):
                        self.name = name
                        return self

                class Queue:
                    def __init__(self):
                        self.head = Job()

                    def first(self):
                        return self.head

                    @property
                    def last(self):
                        return self.head

                def make(flag):
                    if flag:
                        return None
                    return Job()

                def either(flag):
                    if flag:
                        return Job()
                    return Other()

                def each():
                    yield Job()

                def run(queue: Queue):
                    make(1).run()
                    Job().named("a").run()
                    queue.first().run()
                    queue.last.run()
                    either(1).run()
                    each().run()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Job",
            "FILE:m.py::Job::named",
            *["FILE:m.py::Job::run"] * 4,
            "FILE:m.py::Queue::first",
            "FILE:m.py::each",
            "FILE:m.py::either",
            "FILE:m.py::make",
            "REF:run",
            "REF:run",
        ]

    def test_context_manager_function_hands_the_block_what_it_yields(self, tmp_path):
        """Under `contextlib.contextmanager` or `asynccontextmanager`, `with
        f() as x` binds what the function yields or is annotated to."""
        builder = _index(
            tmp_path,
            {
                "m.py": """
                import contextlib
                from contextlib import asynccontextmanager
                from typing import AsyncIterator

                class Other:
                    def run(self): ...

                class Job:
                    def run(self): ...

                @contextlib.contextmanager
                def opened():
                    yield Job()

                @asynccontextmanager
                async def started() -> AsyncIterator[Job]:
                    yield source()

                async def run():
                    with opened() as job:
                        job.run()
                    async with started() as job:
                        job.run()
                """
            },
        )
        assert _calls(builder, "FILE:m.py::run") == [
            "FILE:m.py::Job::run",
            "FILE:m.py::Job::run",
            "FILE:m.py::opened",
            "FILE:m.py::started",
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

    def test_qualified_call_picks_the_type_the_file_declares(self, tmp_path):
        """`Matcher::new` where the file's own `Matcher` and another crate's
        both fit is the file's: a name declared here is not ambiguous."""
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

    def test_qualified_call_of_alike_types_names_its_candidates(self, tmp_path):
        """`Matcher::new` where two types of the name fit and neither is in
        scope stays unresolved, the edge naming the definitions it may
        reach."""
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
                """,
                "c/src/check.rs": """
                fn check() {
                    let m = Matcher::new();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:c/src/check.rs::check") == ["REF:Matcher::new"]
        (edge,) = (
            e
            for e in builder.edges
            if e["source"] == "FILE:c/src/check.rs::check" and e["relation"] == "CALLS"
        )
        assert edge["metadata"]["candidates"] == [
            "FILE:a/src/util.rs::Matcher::new",
            "FILE:b/src/testutil.rs::Matcher::new",
        ]

    def test_turbofish_call_reaches_the_function(self, tmp_path):
        """`parse::<T>()` is a call of `parse`: the turbofish's type
        arguments are the call's, not part of the callee's name."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub fn parse() -> i32 {
                    0
                }

                pub fn build() {
                    let x = parse::<i32>();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::build") == ["FILE:src/lib.rs::parse"]

    def test_turbofish_keeps_continuation_call_off_the_free_name(self, tmp_path):
        """`.line_number()` after a call is read on that call's result:
        the free `line_number` is not reached through it."""
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

    def test_free_call_with_type_arguments_resolves(self, tmp_path):
        """`Vec::<i32>::new()` records `Vec::new`: the path's turbofish
        does not hide the callee."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Vec2<T>(T);
                impl<T> Vec2<T> {
                    pub fn new() -> Self {
                        todo!()
                    }
                }

                pub fn build() {
                    let v = Vec2::<i32>::new();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::build") == [
            "FILE:src/lib.rs::Vec2::new"
        ]

    def test_a_name_only_hit_does_not_type_its_result(self, tmp_path):
        """`m.start()` where `m` is what `find_at` yields: `find_at` fits
        by name only on a receiver no project type owns, so what it
        returns is not read — `m` is of unknown type."""
        builder = _index(
            tmp_path,
            {
                "src/matcher.rs": """
                pub struct Match;
                impl Match {
                    pub fn start(&self) -> usize { 0 }
                    pub fn end(&self) -> usize { 0 }
                }
                pub struct Error;
                pub trait Matcher {
                    fn find_at(&self, at: usize) -> Result<Option<Match>, Error>;
                }
                impl<'a, M: Matcher> Matcher for &'a M {
                    fn find_at(&self, at: usize) -> Result<Option<Match>, Error> {
                        (**self).find_at(at)
                    }
                }
                """,
                "src/other.rs": """
                pub struct Regex;
                pub struct Engine { regex: Regex }
                impl Engine {
                    pub fn seek(&self) {
                        let m = self.regex.find_at(0);
                        m.start();
                        m.end();
                    }
                }
                """,
            },
        )
        calls = _calls(builder, "FILE:src/other.rs::Engine::seek")
        assert "REF:m.start" in calls and "REF:m.end" in calls

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

    def test_function_after_an_impl_in_a_module_is_no_method_of_it(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.rs": """
                mod tests {
                    struct Recording;
                    impl Service for Recording {
                        fn update(&self) -> u8 {
                            1
                        }
                    }
                    fn helper() -> u8 {
                        2
                    }
                    fn uses() {
                        helper();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:a.rs::tests::uses") == ["FILE:a.rs::tests::helper"]

    def test_impl_is_filed_under_the_type_its_header_is_for(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.rs": """
                struct User;
                struct Core;
                impl From<&User> for Core {
                    fn from(user: &User) -> Self {
                        Core
                    }
                }
                impl<'de> Deserialize<'de> for Core {
                    fn deserialize(input: &'de str) -> Self {
                        Core
                    }
                }
                impl<
                    T,
                > Mark<Box<dyn Fn(T) -> T>>
                    for &mut Core
                {
                    fn mark(&self) {}
                }
                unsafe impl Send for Core {}
                impl From<Core> for (u8, u8) {
                    fn from(core: Core) -> Self {
                        (0, 0)
                    }
                }
                impl<'a, M: Mark<u8>> Mark<u8> for &'a M {
                    fn marked(&self) {}
                }
                fn make(user: &User) {
                    Core::from(user).mark();
                }
                """,
            },
        )
        methods = sorted(
            node["id"] for node in builder.nodes if node["type"] == "Method"
        )
        assert methods == [
            "FILE:a.rs::<From for (u8, u8)>::from",
            "FILE:a.rs::<Mark for &'a M>::marked",
            "FILE:a.rs::Core::deserialize",
            "FILE:a.rs::Core::from",
            "FILE:a.rs::Core::mark",
        ]
        assert _calls(builder, "FILE:a.rs::make") == [
            "FILE:a.rs::Core::from",
            "FILE:a.rs::Core::mark",
        ]

    def test_types_a_base_name_may_stand_for_are_no_candidates(self, tmp_path):
        marked = """
        impl Mark for Outer {
            fn mark(&self) {}
        }
        """
        builder = _index(
            tmp_path,
            {
                "a.rs": marked,
                "b.rs": marked,
                "c.rs": """
                pub enum Mode {
                    On,
                }
                impl Outer for Mode {
                    fn fmt(&self) {}
                }
                pub struct Other;
                impl Other {
                    pub fn parse(x: u8) {}
                }
                """,
                "d.rs": """
                use crate::c::Mode;
                pub fn go() {
                    Mode::parse(1);
                }
                """,
            },
        )
        (call,) = (
            edge
            for edge in builder.edges
            if edge["source"] == "FILE:d.rs::go" and edge["relation"] == "CALLS"
        )
        assert call["target"] == "REF:Mode::parse"
        assert "candidates" not in call["metadata"]

    def test_trait_written_with_a_path_is_the_type_the_path_leads_to(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "Cargo.toml": '[package]\nname = "x"\n',
                "src/traits.rs": """
                pub trait Mark {
                    fn mark(&self);
                }
                """,
                "src/lib.rs": """
                pub mod traits;
                pub struct Index;
                pub struct Own;
                pub struct Table;
                impl crate::traits::Mark for Own {
                    fn mark(&self) {}
                }
                impl std::ops::Index<usize> for Table {
                    fn index(&self) {}
                }
                """,
                "src/aliased.rs": """
                use crate::traits as t;
                use std::fmt as f;
                pub struct Other;
                impl t::Mark for Other {
                    fn mark(&self) {}
                }
                impl f::Display for Other {
                    fn fmt(&self) {}
                }
                """,
            },
        )
        bases = {
            edge["source"]: edge["target"]
            for edge in builder.edges
            if edge["relation"] == "INHERITS"
        }
        assert bases == {
            "FILE:src/lib.rs::Own": "FILE:src/traits.rs::Mark",
            "FILE:src/aliased.rs::Other": "FILE:src/traits.rs::Mark",
        }

    def test_call_through_a_name_a_use_binds_a_module_under(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "Cargo.toml": '[package]\nname = "x"\n',
                "src/lib.rs": """
                pub mod shapes;
                pub mod tools;
                pub mod user;
                """,
                "src/shapes.rs": """
                pub struct Square;
                impl Square {
                    pub fn of() -> Square {
                        Square
                    }
                }
                pub fn make() {}
                """,
                "src/tools.rs": """
                pub fn sharpen() {}
                """,
                "src/user.rs": """
                use crate::shapes as s;
                use crate::shapes as tools;
                use std::fs as f;
                pub fn go() {
                    s::make();
                    s::Square::of();
                    f::read("x");
                    crate::tools::sharpen();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/user.rs::go") == [
            "FILE:src/shapes.rs::Square::of",
            "FILE:src/shapes.rs::make",
            "FILE:src/tools.rs::sharpen",
        ]

    def test_name_the_standard_library_has_does_not_pin_an_unknown_receiver(
        self, tmp_path
    ):
        builder = _index(
            tmp_path,
            {
                "a.rs": """
                pub struct Key;
                impl Key {
                    pub fn and(self, other: Key) -> Key {
                        self
                    }
                    pub fn only_here(self) -> Key {
                        self
                    }
                }
                pub fn go() {
                    let made = build!();
                    made.and(1);
                    made.only_here();
                }
                """,
            },
        )
        calls = {
            edge["target"]: edge["metadata"].get("candidates")
            for edge in builder.edges
            if edge["source"] == "FILE:a.rs::go"
            and edge["relation"].startswith("CALLS")
        }
        assert calls == {
            "REF:made.and": ["FILE:a.rs::Key::and"],
            "FILE:a.rs::Key::only_here": None,
        }

    def test_impl_whose_brace_follows_a_where_clause(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.rs": """
                struct Reader<R>(R);
                impl<R> Reader<R>
                where
                    R: std::io::Read,
                {
                    fn first(&self) {
                        self.second();
                    }
                    fn second(&self) {}
                }
                fn after() {}
                """,
            },
        )
        functions = {
            node["id"]: node["type"]
            for node in builder.nodes
            if node["type"] in ("Function", "Method")
        }
        assert functions == {
            "FILE:a.rs::Reader::first": "Method",
            "FILE:a.rs::Reader::second": "Method",
            "FILE:a.rs::after": "Function",
        }

    def test_body_on_the_signature_line(self, tmp_path):
        """A body written on the line of its `fn` makes its calls from
        that function, in a file and in an `impl` alike."""
        builder = _index(
            tmp_path,
            {
                "lib.rs": """
                fn one() {}
                fn both() { one(); Text.raw(); }

                pub struct Text;
                impl Text {
                    fn doubled(&self) -> usize { self.raw() }
                    fn raw(&self) -> usize { 1 }
                }
                """
            },
        )
        assert _calls(builder, "FILE:lib.rs::both") == [
            "FILE:lib.rs::Text::raw",
            "FILE:lib.rs::one",
        ]
        assert _calls(builder, "FILE:lib.rs::Text::doubled") == [
            "FILE:lib.rs::Text::raw"
        ]
        assert _calls(builder, "FILE:lib.rs::Text") == []

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

    def test_field_type_is_not_a_local_of_the_same_name(self, tmp_path):
        """A `let` binding of a call's result does not make a field of the
        same name ambiguous."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Glob;

                impl Glob {
                    pub fn hits(&self) -> bool { true }
                }

                pub struct Other;

                impl Other {
                    pub fn hits(&self) -> bool { true }
                }

                pub struct Matcher {
                    globs: Glob,
                }

                impl Matcher {
                    pub fn new() -> Matcher {
                        let globs = build();
                        Matcher { globs }
                    }

                    pub fn check(&self) -> bool {
                        self.globs.hits()
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::Matcher::check") == [
            "FILE:src/lib.rs::Glob::hits"
        ]

    def test_method_of_a_standard_type_is_external(self, tmp_path):
        """A receiver declared as a primitive or a standard generic type
        calls that type's method, whatever the project names its own."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Stack {
                    items: Vec<u32>,
                    name: Option<String>,
                }

                impl Stack {
                    pub fn len(&self) -> usize {
                        self.items.len()
                    }

                    pub fn is_none(&self) -> bool {
                        self.name.is_none()
                    }
                }

                pub fn total(text: &str, stack: &Stack) -> usize {
                    text.len() + stack.len()
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::Stack::len") == []
        assert _calls(builder, "FILE:src/lib.rs::Stack::is_none") == []
        assert _calls(builder, "FILE:src/lib.rs::total") == [
            "FILE:src/lib.rs::Stack::len"
        ]

    def test_project_trait_method_on_a_standard_type(self, tmp_path):
        """A trait the project implements for a standard type gives it a
        project method."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub trait Total {
                    fn len(&self) -> u32;
                }

                impl<T> Total for T {
                    fn len(&self) -> u32 { 0 }
                }

                pub struct Bag {
                    items: Vec<u32>,
                }

                impl Bag {
                    pub fn size(&self) -> u32 {
                        self.items.len()
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::Bag::size") != []

    def test_blanket_impl_is_for_the_types_that_meet_its_bounds(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub trait Sink {
                    fn context(&self) -> u8;
                }

                impl<'a, S: Sink> Sink for &'a mut S {
                    fn context(&self) -> u8 { 0 }
                }

                pub trait Loud {
                    fn shout(&self) -> u8;
                }

                impl<T: std::fmt::Display> Loud for T {
                    fn shout(&self) -> u8 { 0 }
                }

                pub struct Engine<S> {
                    sink: S,
                }

                impl<S: Sink> Engine<S> {
                    pub fn run(&self) -> u8 {
                        self.sink.context()
                    }
                }

                pub fn go(text: String) {
                    text.context();
                    text.shout();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::go") == [
            "FILE:src/lib.rs::<Loud for T>::shout"
        ]
        assert _calls(builder, "FILE:src/lib.rs::Engine::run") == [
            "FILE:src/lib.rs::Sink::context"
        ]
        blanket = {
            node["metadata"]["name"]: node["metadata"]["bounds"]
            for node in builder.nodes
            if "bounds" in node["metadata"]
        }
        assert blanket == {
            "<Sink for &'a mut S>": ["Sink"],
            "<Loud for T>": ["Display"],
        }

    def test_name_bound_by_a_pattern_has_no_declared_type(self, tmp_path):
        """A match arm that binds a name shadows the parameter of that name:
        it is what the variant matched holds."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Seq;

                impl Seq {
                    pub fn choose(&self) {}
                }

                pub struct Alt;

                impl Alt {
                    pub fn choose(&self) {}
                }

                pub fn pick(prev: Seq, next: Option<Alt>) {
                    match next {
                        Some(prev) => prev.choose(),
                        None => {}
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::pick") == [
            "FILE:src/lib.rs::Alt::choose"
        ]

    def test_variable_named_like_a_library_is_not_that_library(self, tmp_path):
        """`bytes` is a closure parameter, not Go's package of that name:
        its `len` is the project's."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Buf;

                impl Buf {
                    pub fn len(&self) -> usize { 0 }
                }

                pub fn sizes(all: Vec<Buf>) -> Vec<usize> {
                    all.iter().map(|bytes| bytes.len()).collect()
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::sizes") == [
            "FILE:src/lib.rs::Buf::len"
        ]

    def test_field_type_is_that_of_the_struct_declaring_it(self, tmp_path):
        """Two structs of one file may each have a field `inner`."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Glob;

                impl Glob {
                    pub fn hits(&self) {}
                }

                pub struct Other;

                impl Other {
                    pub fn hits(&self) {}
                }

                pub struct First {
                    inner: Glob,
                }

                pub struct Second {
                    inner: Other,
                }

                impl First {
                    pub fn go(&self) {
                        self.inner.hits()
                    }
                }

                impl Second {
                    pub fn go(&self) {
                        self.inner.hits()
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::First::go") == [
            "FILE:src/lib.rs::Glob::hits"
        ]
        assert _calls(builder, "FILE:src/lib.rs::Second::go") == [
            "FILE:src/lib.rs::Other::hits"
        ]

    def test_type_written_with_its_path(self, tmp_path):
        """A parameter typed `process::ChildStderr` is of a type from
        outside; a field typed `Box<dyn Visitor>` calls the trait's method."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::process;

                pub struct Reader;

                impl Reader {
                    pub fn read_to_end(&mut self) {}
                }

                pub trait Visitor {
                    fn visit(&mut self);
                }

                pub struct Counter;

                impl Counter {
                    pub fn visit(&mut self) {}
                }

                pub struct Worker {
                    visitor: Box<dyn Visitor>,
                }

                impl Worker {
                    pub fn run(&mut self) {
                        self.visitor.visit()
                    }
                }

                pub fn drain(stderr: &mut process::ChildStderr) {
                    stderr.read_to_end();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::Worker::run") == [
            "FILE:src/lib.rs::Visitor::visit"
        ]
        assert _calls(builder, "FILE:src/lib.rs::drain") == []

    def test_outside_wrapper_around_a_reference_to_a_project_type(self, tmp_path):
        """`anyhow::Result<&mut Builder>` unwraps to a Builder."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Builder;

                impl Builder {
                    pub fn pre(&mut self) -> anyhow::Result<&mut Builder> {
                        Ok(self)
                    }

                    pub fn globs(&mut self) {}
                }

                pub struct Other;

                impl Other {
                    pub fn globs(&mut self) {}
                }

                pub fn make(builder: &mut Builder) -> anyhow::Result<()> {
                    builder.pre()?.globs();
                    Ok(())
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::make") == [
            "FILE:src/lib.rs::Builder::globs",
            "FILE:src/lib.rs::Builder::pre",
        ]

    def test_example_target_is_a_crate_of_its_own(self, tmp_path):
        """`walker::Entry` is the library's type, not one an example of the
        same package declares under that name."""
        builder = _index(
            tmp_path,
            {
                "walker/Cargo.toml": """
                [package]
                name = "walker"
                """,
                "walker/examples/demo.rs": """
                enum Entry {
                    One,
                }

                impl Entry {
                    fn path(&self) {}
                }
                """,
                "walker/src/lib.rs": """
                pub struct Entry;

                impl Entry {
                    pub fn path(&self) {}
                }
                """,
                "app/Cargo.toml": """
                [package]
                name = "app"
                """,
                "app/src/main.rs": """
                struct Hay {
                    dent: walker::Entry,
                }

                impl Hay {
                    fn path(&self) {
                        self.dent.path()
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app/src/main.rs::Hay::path") == [
            "FILE:walker/src/lib.rs::Entry::path"
        ]

    def test_standard_method_returns_what_the_data_says(self, tmp_path):
        """`Option<T>::unwrap` gives a `T`, `Vec<T>::first` an `Option<&T>`
        and `HashMap<K, V>::get` an `Option<&V>`: the next call is one on
        that type, and a call on a standard type is the library's."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::collections::HashMap;

                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn len(&self) -> usize { 0 }
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                    pub fn len(&self) -> usize { 0 }
                }

                pub struct Store {
                    best: Option<Product>,
                    items: Vec<Product>,
                    named: HashMap<String, Product>,
                }

                impl Store {
                    pub fn best(&self) {
                        self.best.as_ref().unwrap().ship();
                    }

                    pub fn first(&self) {
                        self.items.first().unwrap().ship();
                    }

                    pub fn named(&self, name: &str) {
                        self.named.get(name).unwrap().ship();
                    }

                    pub fn size(&self) -> usize {
                        self.items.clone().len() + self.named.keys().len()
                    }
                }
                """
            },
        )
        ship = ["FILE:src/lib.rs::Product::ship"]
        assert _calls(builder, "FILE:src/lib.rs::Store::best") == ship
        assert _calls(builder, "FILE:src/lib.rs::Store::first") == ship
        assert _calls(builder, "FILE:src/lib.rs::Store::named") == ship
        assert _calls(builder, "FILE:src/lib.rs::Store::size") == []

    def test_iterator_keeps_the_type_of_its_items(self, tmp_path):
        """`filter` hands on the items of the iterator it is called on;
        what a block given to `map` as a closure returns is not known."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn count(&self) -> usize { 0 }
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                }

                pub fn kept(items: &[Product]) {
                    items.iter().filter(|p| true).next().unwrap().ship();
                }

                pub fn total(items: Vec<Product>) -> usize {
                    items.iter().rev().skip(1).count()
                }

                pub fn mapped(items: &[Product]) {
                    items.iter().map(|p| { p }).next().unwrap().ship();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::kept") == [
            "FILE:src/lib.rs::Product::ship"
        ]
        assert _calls(builder, "FILE:src/lib.rs::total") == []
        assert _calls(builder, "FILE:src/lib.rs::mapped") == ["REF:ship"]

    def test_question_mark_takes_the_value_out_of_a_standard_type(self, tmp_path):
        """`?` on an `io::Result<T>` or an `Option<T>` gives a `T`: each
        `?` of `??` in its turn, and one that ends the line before."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::io;

                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                }

                fn load() -> io::Result<Vec<Product>> { todo!() }

                fn find() -> Option<Option<Product>> { todo!() }

                pub fn run() -> io::Result<()> {
                    load()?.pop().unwrap().ship();
                    Ok(())
                }

                pub fn pick() -> Option<()> {
                    find()??.ship();
                    let found = find()??;
                    found.ship();
                    find()?
                        .unwrap()
                        .ship();
                    None
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::run") == [
            "FILE:src/lib.rs::Product::ship",
            "FILE:src/lib.rs::load",
        ]
        assert _calls(builder, "FILE:src/lib.rs::pick") == [
            "FILE:src/lib.rs::Product::ship",
            "FILE:src/lib.rs::Product::ship",
            "FILE:src/lib.rs::Product::ship",
            "FILE:src/lib.rs::find",
            "FILE:src/lib.rs::find",
            "FILE:src/lib.rs::find",
        ]

    def test_type_argument_of_unknown_type_is_not_guessed(self, tmp_path):
        """What a `Vec<T>` holds is not known inside a function generic
        over `T`."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                }

                pub fn all<T>(items: Vec<T>) -> usize {
                    items.first().unwrap().ship();
                    items.len()
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::all") == ["REF:ship"]

    def test_standard_pointer_has_its_own_methods_and_those_it_points_at(
        self, tmp_path
    ):
        """`Arc<T>::clone` is the pointer's, not the `clone` of what it
        points at; a method an `Rc` or a `RefCell` borrow has not is one of
        the value inside."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::cell::RefCell;
                use std::rc::Rc;
                use std::sync::Arc;

                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn clone(&self) -> Product { Product }
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                }

                pub fn share(shared: Arc<Product>) {
                    shared.clone().ship();
                }

                pub fn lend(cell: Rc<RefCell<Product>>) {
                    cell.borrow().ship();
                }
                """
            },
        )
        ship = ["FILE:src/lib.rs::Product::ship"]
        assert _calls(builder, "FILE:src/lib.rs::share") == ship
        assert _calls(builder, "FILE:src/lib.rs::lend") == ship

    def test_alias_hides_a_standard_type_only_in_its_own_file(self, tmp_path):
        """`type Result<T> = ...` in one file leaves `Result` the prelude's
        in another."""
        builder = _index(
            tmp_path,
            {
                "src/errors.rs": """
                pub type Result<T> = std::result::Result<T, String>;
                """,
                "src/lib.rs": """
                pub struct Product;

                impl Product {
                    pub fn unwrap(self) {}
                }

                fn load() -> Result<bool, String> { todo!() }

                pub fn run() -> bool {
                    load().unwrap()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::run") == ["FILE:src/lib.rs::load"]

    def test_function_of_a_standard_type_returns_what_the_data_says(self, tmp_path):
        """`Vec::new()` is a `Vec`, `u8::try_from(..)` a `Result`."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Queue;

                impl Queue {
                    pub fn push(&mut self, item: u8) {}
                    pub fn unwrap(self) {}
                }

                pub fn fill(size: usize) -> u8 {
                    let mut items = Vec::with_capacity(size);
                    items.push(1);
                    u8::try_from(size).unwrap()
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::fill") == []

    def test_literal_and_constant_have_a_type(self, tmp_path):
        """What a `let` binds to a `vec![..]` or a string literal, and a
        constant, call the methods of their type."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                const GREETING: &str = "hello";

                pub struct Queue;

                impl Queue {
                    pub fn push(&mut self, item: u8) {}
                    pub fn len(&self) -> usize { 0 }
                }

                pub fn fill() -> usize {
                    let mut items = vec![];
                    items.push(1);
                    let name = "queue";
                    name.len() + GREETING.len()
                }

                pub fn spelled(queue: &Queue) -> usize {
                    let name = "queue".to_string();
                    queue.len()
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::fill") == []
        assert _calls(builder, "FILE:src/lib.rs::spelled") == [
            "FILE:src/lib.rs::Queue::len"
        ]

    def test_trait_implemented_for_a_standard_type_is_that_type_s(self, tmp_path):
        """A method the project gives `Box<S>` is reached through a `Box`
        and does not make the same name on another standard type the
        project's."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::fmt;

                pub trait Sink {
                    fn finish(&self);
                }

                impl<S: Sink> Sink for Box<S> {
                    fn finish(&self) {}
                }

                pub struct Printer;

                impl Sink for Printer {
                    fn finish(&self) {}
                }

                pub fn close(boxed: Box<Printer>) {
                    boxed.finish();
                }

                pub fn show(f: &mut fmt::Formatter<'_>) {
                    f.debug_struct("Printer").finish();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::close") == [
            "FILE:src/lib.rs::Box::finish"
        ]
        assert _calls(builder, "FILE:src/lib.rs::show") == []

    def test_returned_trait_object_calls_the_trait(self, tmp_path):
        """A `Box<dyn Trait>` a function returns calls the trait's method."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub trait Visitor {
                    fn visit(&mut self);
                }

                pub struct Counter;

                impl Counter {
                    pub fn visit(&mut self) {}
                }

                fn build() -> Box<dyn Visitor + 'static> { todo!() }

                pub fn run() {
                    build().visit();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::run") == [
            "FILE:src/lib.rs::Visitor::visit",
            "FILE:src/lib.rs::build",
        ]

    def test_implementor_of_a_standard_trait_has_other_methods_too(self, tmp_path):
        """What is only known to implement `Write` calls the trait's
        `flush`; a method the trait has not may be the project's."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::io::Write;

                pub trait Tagged {
                    fn tag(&self);
                    fn flush(&self);
                }

                pub fn dump<W: Write + Tagged>(out: &mut W) {
                    out.tag();
                    out.flush();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::dump") == [
            "FILE:src/lib.rs::Tagged::tag"
        ]

    def test_derived_trait_gives_its_methods(self, tmp_path):
        """`#[derive(Clone, Default)]` makes `clone` and `default` the
        standard traits' and what they return the type itself; a type
        without the attribute has neither."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                /// Settings.
                #[derive(
                    Clone,
                    Debug, Default,
                )]
                pub struct Config {
                    depth: usize,
                }

                impl Config {
                    pub fn apply(&self) {}
                }

                /// Not `#[derive(Clone)]`.
                pub struct Decoy;

                impl Decoy {
                    pub fn apply(&self) {}
                }

                pub struct Other;

                impl Other {
                    pub fn clone(&self) -> Other { Other }
                    pub fn default() -> Other { Other }
                }

                pub fn copy(config: &Config) {
                    config.clone().apply();
                    Config::default().apply();
                }

                pub fn plain(decoy: &Decoy) {
                    decoy.clone().apply();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::copy") == [
            "FILE:src/lib.rs::Config::apply",
            "FILE:src/lib.rs::Config::apply",
        ]
        assert _calls(builder, "FILE:src/lib.rs::plain") == [
            "REF:apply",
            "REF:decoy.clone",
        ]

    def test_standard_trait_gives_only_the_members_it_has(self, tmp_path):
        """A type that implements `Deref` has `deref` from the standard
        library; another method it has not may be one of what it
        dereferences to."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::ops::Deref;

                pub struct Engine;

                impl Engine {
                    pub fn run(&self) {}
                }

                pub struct Handle(Engine);

                impl Deref for Handle {
                    type Target = Engine;

                    fn deref(&self) -> &Engine {
                        &self.0
                    }
                }

                pub struct Other;

                impl Other {
                    pub fn run(&self) {}
                }

                pub fn start(handle: &Handle) {
                    handle.run();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::start") == ["REF:handle.run"]

    def test_for_loop_binds_what_iterating_gives(self, tmp_path):
        """The name a `for` binds is an item of what is iterated: of a
        `Vec`, of a map with its key, of an `enumerate()` with its index."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                use std::collections::HashMap;

                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn label(&self) -> Label { Label }
                }

                pub struct Label;

                impl Label {
                    pub fn print(&self) {}
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn print(&self) {}
                }

                pub struct Store {
                    items: Vec<Product>,
                    named: HashMap<String, Product>,
                }

                impl Store {
                    pub fn all(&self) {
                        for product in &self.items {
                            product.ship();
                        }
                    }

                    pub fn named(&self) {
                        for (name, product) in &self.named {
                            product.ship();
                        }
                    }

                    pub fn counted(&self) {
                        for (index, product) in self.items.iter().enumerate() {
                            product.ship();
                        }
                        product.ship();
                    }
                }
                """
            },
        )
        ship = "FILE:src/lib.rs::Product::ship"
        assert _calls(builder, "FILE:src/lib.rs::Store::all") == [ship]
        assert _calls(builder, "FILE:src/lib.rs::Store::named") == [ship]
        assert _calls(builder, "FILE:src/lib.rs::Store::counted") == [
            ship,
            "REF:product.ship",
        ]

    def test_pattern_binds_what_the_value_holds(self, tmp_path):
        """A name in a pattern is what the variant, the struct or the tuple
        matched holds at its place: in `if let`, `while let`, `let` and the
        arms of a `match`, of standard and of project types."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn label(&self) -> Label { Label }
                }

                pub struct Label;

                impl Label {
                    pub fn print(&self) {}
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn print(&self) {}
                }

                pub struct Failure;

                impl Failure {
                    pub fn print(&self) {}
                }

                pub struct Wrapper(Product);

                pub struct Order {
                    product: Product,
                    count: usize,
                }

                pub enum Message {
                    Send(Product, usize),
                    Stop,
                }

                fn pair() -> (Product, usize) { todo!() }

                fn load() -> Result<Product, Failure> { todo!() }

                pub fn optional(best: Option<Product>, queue: Vec<Product>) {
                    if let Some(product) = best.as_ref() {
                        product.ship();
                    }
                    while let Some(product) = queue.pop() {
                        product.ship();
                    }
                }

                pub fn taken(wrapper: Wrapper, order: &Order) {
                    let (first, count) = pair();
                    first.ship();
                    let Wrapper(inner) = wrapper;
                    inner.ship();
                    let Order { product, .. } = order;
                    product.ship();
                }

                pub fn matched(message: Message) {
                    match message {
                        Message::Send(product, _) => product.ship(),
                        Message::Stop => {}
                    }
                    match load() {
                        Ok(product) => {
                            product.ship();
                        }
                        Err(failure) => failure.print(),
                    }
                }
                """
            },
        )
        ship = "FILE:src/lib.rs::Product::ship"
        assert _calls(builder, "FILE:src/lib.rs::optional") == [ship, ship]
        assert _calls(builder, "FILE:src/lib.rs::taken") == [
            ship,
            ship,
            ship,
            "FILE:src/lib.rs::Wrapper",
            "FILE:src/lib.rs::pair",
        ]
        assert _calls(builder, "FILE:src/lib.rs::matched") == [
            "FILE:src/lib.rs::Failure::print",
            "FILE:src/lib.rs::Message::Send",
            ship,
            ship,
            "FILE:src/lib.rs::load",
        ]

    def test_closure_parameter_has_the_type_the_method_passes(self, tmp_path):
        """What `for_each`, `filter` and `Option::map` pass to a closure is
        the item or the value held, on the line and in the closure's block;
        the name is no longer that after it."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn label(&self) -> Label { Label }
                }

                pub struct Label;

                impl Label {
                    pub fn print(&self) {}
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn print(&self) {}
                }

                pub fn each(items: &[Product], best: Option<Product>) -> usize {
                    items.iter().for_each(|product| product.ship());
                    best.as_ref().map(|product| product.ship());
                    items.iter().filter(|product| product.ready()).count()
                }

                pub fn block(items: Vec<Product>) {
                    items.iter().for_each(|product| {
                        product.ship();
                    });
                    product.ship();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::each") == [
            "FILE:src/lib.rs::Product::ready",
            "FILE:src/lib.rs::Product::ship",
            "FILE:src/lib.rs::Product::ship",
        ]
        assert _calls(builder, "FILE:src/lib.rs::block") == [
            "FILE:src/lib.rs::Product::ship",
            "REF:product.ship",
        ]

    def test_closure_result_types_what_the_method_returns(self, tmp_path):
        """`map` gives what its closure returns where that is one call."""
        builder = _index(
            tmp_path,
            {
                "src/lib.rs": """
                pub struct Product;

                impl Product {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn label(&self) -> Label { Label }
                }

                pub struct Label;

                impl Label {
                    pub fn print(&self) {}
                }

                pub struct Decoy;

                impl Decoy {
                    pub fn ship(&self) {}
                    pub fn ready(&self) -> bool { true }
                    pub fn print(&self) {}
                }

                pub fn labels(items: &[Product], best: Option<Product>) {
                    items.iter().map(|product| product.label()).next().unwrap().print();
                    best.map(|product| product.label()).unwrap().print();
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/lib.rs::labels") == [
            "FILE:src/lib.rs::Label::print",
            "FILE:src/lib.rs::Label::print",
            "FILE:src/lib.rs::Product::label",
            "FILE:src/lib.rs::Product::label",
        ]


class TestTypeScriptCallResolution:
    def test_this_in_an_object_literal_argument_reads_its_this_type(self, tmp_path):
        """`this` inside `{...}` passed to a parameter written
        `... & ThisType<Ctx>` is `Ctx`, not the enclosing class."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                interface Ctx {
                  tally(): number;
                }

                declare function configure(opts: { setup(): void } & ThisType<Ctx>): void;

                function run() {
                  configure({ setup() { this.tally(); } });
                }
                """
            },
        )
        assert "FILE:a.ts::Ctx::tally" in _calls(builder, "FILE:a.ts::run")

    def test_returned_arrow_reads_the_functions_return_type(self, tmp_path):
        """`return (x) => x.fetch()` in `(): (x: Dog) => Dog` types `x`."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Dog {
                  fetch(): number { return 1; }
                }

                function makeMaker(): (x: Dog) => Dog {
                  return (x) => x.fetch();
                }
                """
            },
        )
        assert "FILE:a.ts::Dog::fetch" in _calls(builder, "FILE:a.ts::makeMaker")

    def test_arrow_assigned_a_function_type_reads_its_params(self, tmp_path):
        """`const f: (x: Dog) => Dog = (y) => y.fetch()` types `y`; the call
        is the arrow's, on its own line."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Dog {
                  fetch(): number { return 1; }
                }

                const typed: (x: Dog) => Dog = (y) => y.fetch();
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::typed") == ["FILE:a.ts::Dog::fetch"]

    def test_union_member_calls_stay_alternatives(self, tmp_path):
        """`a.fetch()` on `Dog | Cat`, which both define, is genuinely
        ambiguous: the edge names the members it may reach."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                class Dog {
                  fetch(): number { return 1; }
                }
                class Cat {
                  fetch(): number { return 2; }
                  purr(): number { return 3; }
                }
                function makeA(): Dog | Cat { return new Dog(); }
                function run() {
                  const a = makeA();
                  a.fetch();
                  a.purr();
                }
                """
            },
        )
        calls = _calls(builder, "FILE:a.ts::run")
        assert "REF:a.fetch" in calls and "REF:a.purr" in calls
        edges = {
            e["target"]: e
            for e in builder.edges
            if e["source"] == "FILE:a.ts::run" and e["relation"] == "CALLS"
        }
        assert edges["REF:a.fetch"]["metadata"]["candidates"] == [
            "FILE:a.ts::Cat::fetch",
            "FILE:a.ts::Dog::fetch",
        ]
        assert edges["REF:a.purr"]["metadata"]["candidates"] == ["FILE:a.ts::Cat::purr"]

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
        # `has` is a method of the object `check` returns, whose owner no
        # source can name.
        assert "FILE:a.ts::check::object@8::has" in ids
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

    def test_module_reexported_under_a_name_is_a_namespace(self, tmp_path):
        """`export * as ns from` makes `lib.ns.name` a path into that module."""
        builder = _index(
            tmp_path,
            {
                "lib/coerce.ts": """
                export function text() {}
                """,
                "lib/other.ts": """
                export function text() {}
                """,
                "lib/index.ts": """
                export * as coerce from "./coerce";
                """,
                "app.ts": """
                import * as lib from "./lib/index";

                export function run() {
                    return lib.coerce.text();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == ["FILE:lib/coerce.ts::text"]

    def test_variable_bound_to_a_call_spread_over_lines(self, tmp_path):
        """A variable is bound to what a call returns although the call's
        arguments run over several lines."""
        builder = _index(
            tmp_path,
            {
                "box.ts": """
                export class Box {
                    open() {}
                }

                export class Crate {
                    open() {}
                }

                export function make(options: object): Box {
                    return new Box();
                }
                """,
                "app.ts": """
                import { make } from "./box";

                export function run() {
                    const box = make({
                        sealed: true,
                    });
                    box.open();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:box.ts::Box::open",
            "FILE:box.ts::make",
        ]

    def test_receiver_named_on_the_line_before(self, tmp_path):
        """A line that starts with `.name(` calls on the name the line
        before it ends with."""
        builder = _index(
            tmp_path,
            {
                "lib.ts": """
                export function start() {}

                export class Job {
                    start() {}
                }
                """,
                "app.ts": """
                import * as lib from "./lib";

                export function run() {
                    return lib
                        .start();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == ["FILE:lib.ts::start"]

    def test_return_type_named_through_an_import(self, tmp_path):
        """A declared return type is the type its file imports under that
        name, not another of the same name."""
        builder = _index(
            tmp_path,
            {
                "a/box.ts": """
                export class Box {
                    open() {}
                }
                """,
                "b/box.ts": """
                export class Box {
                    open() {}
                }
                """,
                "make.ts": """
                import { Box } from "./a/box";

                export function make(): Box {
                    return new Box();
                }
                """,
                "app.ts": """
                import { make } from "./make";

                export function run() {
                    const box = make();
                    box.open();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:a/box.ts::Box::open",
            "FILE:make.ts::make",
        ]

    def test_constant_naming_a_function_returns_what_it_returns(self, tmp_path):
        """Calling `const make = Shape.create` gives what `create` declares."""
        builder = _index(
            tmp_path,
            {
                "shape.ts": """
                export class Shape {
                    static create(): Shape {
                        return new Shape();
                    }
                    area() {}
                }

                export class Solid {
                    area() {}
                }

                export const make = Shape.create;
                """,
                "app.ts": """
                import { make } from "./shape";

                export function run() {
                    const shape = make();
                    shape.area();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:shape.ts::Shape::area",
            "FILE:shape.ts::make",
        ]

    def test_method_returning_this_returns_the_receivers_type(self, tmp_path):
        """`this` as a return type is the type the method was called on,
        which may be a subtype of the one that declares it."""
        builder = _index(
            tmp_path,
            {
                "schema.ts": """
                export interface Schema {
                    strict(): this;
                }

                export interface Text extends Schema {
                    upper(): Text;
                }

                export interface Count extends Schema {
                    upper(): Count;
                }

                export function text(): Text {
                    return build();
                }
                """,
                "app.ts": """
                import { text } from "./schema";

                export function run() {
                    return text().strict().upper();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:schema.ts::Schema::strict",
            "FILE:schema.ts::Text::upper",
            "FILE:schema.ts::text",
        ]

    def test_declarations_behind_modifiers_and_named_function_expressions(
        self, tmp_path
    ):
        """`abstract name(`, `override get name(` and `function name(`
        declare the name; none of them calls it."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export function run(data) {}

                export function size() {}
                """,
                "b.ts": """
                import { other } from "./c";

                export abstract class Base {
                    abstract run(data: unknown): void;
                    override get size() {
                        return 1;
                    }
                }

                export const api = {
                    start: function run(data) {
                        return data;
                    },
                };
                """,
                "c.ts": """
                export function other() {}
                """,
            },
        )
        calls = [e for e in builder.edges if e["relation"].startswith("CALLS")]
        assert calls == []

    def test_a_brace_in_a_regex_or_template_ends_no_block(self, tmp_path):
        """A `/` that opens a regular expression, a template literal that
        goes on over lines and a block comment do not open or close the
        block that `_find_block_end` reads them in."""
        builder = _index(
            tmp_path,
            {
                "f.ts": """
                export function a(s: string) {
                    if (/}/.test(s)) {
                        first();
                    }
                    second();
                }

                export function b(s: string) {
                    if (!/^https?:\\/\\//i.test(s)) { third(); }
                    fourth();
                }

                export function c(s: string) {
                    const t = `a {
                        ${s}`;
                    fifth();
                }

                export function d(s: string) {
                    const open = /[{]/;
                    sixth();
                }

                export class K {
                    m(s: string) {
                        return s.replace(/{/g, "(");
                    }
                    n() {
                        seventh();
                    }
                }

                function first() {}
                function second() {}
                function third() {}
                function fourth() {}
                function fifth() {}
                function sixth() {}
                function seventh() {}
                """,
            },
        )
        span = {
            n["id"].rpartition("::")[2]: (
                n["id"].count("::"),
                n["metadata"]["lineno"],
                n["metadata"].get("end_lineno"),
            )
            for n in builder.nodes
            if n["id"] != "FILE:f.ts"
        }
        assert span == {
            "a": (1, 2, 7),
            "b": (1, 9, 12),
            "c": (1, 14, 18),
            "t": (2, 15, None),
            "d": (1, 20, 23),
            "open": (2, 21, None),
            "K": (1, 25, 32),
            "m": (2, 26, 28),
            "n": (2, 29, 31),
            "first": (1, 34, 34),
            "second": (1, 35, 35),
            "third": (1, 36, 36),
            "fourth": (1, 37, 37),
            "fifth": (1, 38, 38),
            "sixth": (1, 39, 39),
            "seventh": (1, 40, 40),
        }
        assert _calls(builder, "FILE:f.ts::a") == [
            "FILE:f.ts::first",
            "FILE:f.ts::second",
        ]
        assert _calls(builder, "FILE:f.ts::b") == [
            "FILE:f.ts::fourth",
            "FILE:f.ts::third",
        ]
        assert _calls(builder, "FILE:f.ts::c") == ["FILE:f.ts::fifth"]
        assert _calls(builder, "FILE:f.ts::d") == ["FILE:f.ts::sixth"]
        assert _calls(builder, "FILE:f.ts::K::n") == ["FILE:f.ts::seventh"]

    def test_a_regex_holding_slashes_and_a_carried_template(self, tmp_path):
        """The shapes that beat a line-at-a-time reading: a `//` inside a
        regular expression is no comment, and a template literal opened
        with `${` on one line and closed on a later one keeps the lines
        between out of the block count. A `/` straight after a regular
        expression divides, so the comment after it is one."""
        builder = _index(
            tmp_path,
            {
                "g.ts": """
                function parseUrl(s: string) {
                    if (!/^https?:\\/\\//i.test(s)) {
                        return null;
                    }
                    return parse(s);
                }

                function describe(label: string) {
                    const body = `starts
                        ${label}
                        ends`;
                    return emit(body);
                }

                function ratio(s: string) {
                    const r = /a/ / /b/; // }
                    return parse(s);
                }

                function parse(s: string) {}
                function emit(s: string) {}
                """,
            },
        )
        span = {
            n["id"].rpartition("::")[2]: (
                n["metadata"]["lineno"],
                n["metadata"].get("end_lineno"),
            )
            for n in builder.nodes
            if n["id"] != "FILE:g.ts"
        }
        assert span["parseUrl"] == (2, 7)
        assert span["describe"] == (9, 14)
        assert span["ratio"] == (16, 19)
        assert _calls(builder, "FILE:g.ts::parseUrl") == ["FILE:g.ts::parse"]
        assert _calls(builder, "FILE:g.ts::describe") == ["FILE:g.ts::emit"]

    def test_a_literal_keeps_its_text_where_it_opens(self, tmp_path):
        """A constant's value and an import's module are read with the text
        of their literals; the lines a template literal goes on over are no
        code."""
        builder = _index(
            tmp_path,
            {
                "h.ts": """
                import { helper } from "./lib";
                export const API = "/api/v1/users";
                const slug = API.replace(/\\//g, "-");
                const sample = `
                    import { other } from "./nowhere";
                    const inner = "x";
                `;
                """,
                "lib.ts": "export function helper() {}",
            },
        )
        assert {
            node["metadata"]["name"]: node["metadata"]["value"]
            for node in builder.nodes
            if node["type"] == "Constant"
        } == {
            "API": '"/api/v1/users"',
            "slug": 'API.replace(/\\//g, "-")',
            "sample": None,
        }
        assert [
            edge["target"] for edge in builder.edges if edge["relation"] == "IMPORTS"
        ] == ["MODULE:./lib"]

    def test_bare_call_in_a_module_reaches_only_its_own_bindings(self, tmp_path):
        """A file that imports is a module: a bare name in it is no global
        of another file, and never a member of a class."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export function make() {}

                export class Box {
                    static seal() {}
                }
                """,
                "b.ts": """
                import { helper } from "./c";

                export function run() {
                    make();
                    seal();
                    return helper();
                }
                """,
                "c.ts": """
                export function helper() {}
                """,
            },
        )
        assert _calls(builder, "FILE:b.ts::run") == ["FILE:c.ts::helper"]

    def test_a_member_closing_the_object_on_its_line_ends_there(self, tmp_path):
        """A member written on one line, with the object's closing brace
        after it, ends on that line: what follows the object is no member."""
        builder = _index(
            tmp_path,
            {
                "a.js": """
                function make() {
                    return {
                        load() { return 1; },
                        save() { return 2; } };
                }
                function after() { return 3; }
                """,
            },
        )
        ids = {node["id"] for node in builder.nodes}
        assert "FILE:a.js::make::object@3::save" in ids
        assert "FILE:a.js::after" in ids

    def test_object_is_an_owner_where_it_holds_a_member_node(self, tmp_path):
        """An accessor is a member, and so is a function written after a
        brace in a property's value or in a comment. An object that holds no
        function below the line of its `return` gets no `object@` node, and
        a call of a function written on that line is no other definition's."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export function data(id: string) {
                    return {
                        id,
                        size: id.length,
                    };
                }

                export function oneLine(text: string) {
                    return { error: () => text };
                }

                export function accessors(items: string[]) {
                    return {
                        get size() {
                            return items.length;
                        },
                        set size(count) {
                            items.length = count;
                        },
                    };
                }

                export function firstLine(text: string) {
                    return { error: () => text,
                    };
                }

                export function pattern(text: string) {
                    return {
                        close: /}/,
                        error: () => text,
                    };
                }

                export function noted(text: string) {
                    return /* { */ {
                        error: () => text,
                    };
                }

                export function use() {
                    const { error } = oneLine("x");
                    error();
                    firstLine("x").error();
                    pattern("x").error();
                }
                """,
                "b.ts": """
                export class Log {
                    error() {}
                }
                """,
            },
        )
        assert _calls(builder, "FILE:a.ts::use") == [
            "FILE:a.ts::firstLine",
            "FILE:a.ts::oneLine",
            "FILE:a.ts::pattern",
            "FILE:a.ts::pattern::object@30::error",
        ]
        assert sorted(
            node["id"] for node in builder.nodes if "object@" in node["id"]
        ) == [
            "FILE:a.ts::accessors::object@14",
            "FILE:a.ts::accessors::object@14::get size",
            "FILE:a.ts::accessors::object@14::set size",
            "FILE:a.ts::noted::object@37",
            "FILE:a.ts::noted::object@37::error",
            "FILE:a.ts::pattern::object@30",
            "FILE:a.ts::pattern::object@30::error",
        ]

    def test_constant_bound_to_a_library_value_is_the_library_s(self, tmp_path):
        """A module-level name bound to what a library constructs is read
        as that type in a callback of a script and in the functions of an
        indented `<script>` block: its methods are no project method's."""
        builder = _index(
            tmp_path,
            {
                "reg.js": """
                export class Registry {
                    add(key) {}
                    delete(key) {}
                }
                """,
                "worker.js": """
                const ids = new Set()
                /**
                 * @type {Map<string, Set<number>>}
                 */
                const pending = new Map()

                addEventListener('message', async function (event) {
                    ids.delete(event.id)
                })
                """,
                "View.svelte": (
                    '<script lang="ts">\n'
                    "\tlet seen = new Set<string>();\n"
                    "\tfunction accept(key: string) {\n"
                    "\t\tseen.add(key);\n"
                    "\t}\n"
                    "</script>\n"
                ),
            },
        )
        assert _calls(builder, "FILE:worker.js") == []
        assert _calls(builder, "FILE:View.svelte::accept") == []

    def test_members_of_the_object_a_function_returns(self, tmp_path):
        """What a function returns as an object has the members the object
        names, read where the function is written: through the result, a
        name bound to it and a pattern that takes it apart, whatever return
        type the function declares."""
        builder = _index(
            tmp_path,
            {
                "make.ts": """
                export interface Store {
                    load(): void;
                    clear(): void;
                }

                export function plain(api: string) {
                    function load() {}
                    function fetchAll() {}
                    return { load, all: fetchAll, count: api.length };
                }

                export function typed(api: string): Store {
                    function load() {}
                    return {
                        load,
                        clear: () => api.trim(),
                    };
                }

                export function odd(api: string) {
                    function parse() {}
                    return {
                        'a-b': api, /* one, two */
                        get size() { return 1; },
                        ...api,
                        parse,
                    };
                }

                export class Registry {
                    has(key: string) { return true; }
                }
                """,
                "use.ts": """
                import { odd, plain, typed } from './make';

                const WORDS = new Set(
                    ['a']
                );
                const shared = plain('s');
                const { all: everything } = plain(
                    'p'
                );

                export function direct() {
                    plain('a').load();
                    const made = plain('b');
                    made.all();
                    const { load } = plain('c');
                    load();
                }

                export function fromModule() {
                    shared.load();
                    everything();
                }

                export function hidden(shared) {
                    shared.load();
                }

                export function declared() {
                    const store = typed('d');
                    store.load();
                    store.clear();
                }

                export function oddly(key: string) {
                    odd('o').parse();
                    return WORDS.has(key);
                }

                for (const looped of [plain('f')]) {
                    looped.load();
                }

                export function after() {
                    looped.load();
                }
                """,
                "Comp.svelte": (
                    '<script lang="ts">\n'
                    "\timport { plain } from './make';\n"
                    "\tconst { load } = plain('e');\n"
                    "\tfunction run() {\n"
                    "\t\tload();\n"
                    "\t}\n"
                    "</script>\n"
                ),
            },
        )
        assert _calls(builder, "FILE:Comp.svelte::run") == ["FILE:make.ts::plain::load"]
        made = ["FILE:make.ts::plain", "FILE:make.ts::plain::load"]
        assert _calls(builder, "FILE:use.ts::direct") == sorted(
            [*made, *made[:1], "FILE:make.ts::plain::fetchAll", *made]
        )
        assert _calls(builder, "FILE:use.ts::fromModule") == [
            "FILE:make.ts::plain::fetchAll",
            "FILE:make.ts::plain::load",
        ]
        assert _calls(builder, "FILE:use.ts::hidden") == ["REF:shared.load"]
        assert _calls(builder, "FILE:use.ts::declared") == [
            "FILE:make.ts::typed",
            "FILE:make.ts::typed::load",
            "FILE:make.ts::typed::object@15::clear",
        ]
        assert _calls(builder, "FILE:use.ts::oddly") == [
            "FILE:make.ts::odd",
            "FILE:make.ts::odd::parse",
        ]
        assert _calls(builder, "FILE:use.ts::after") == ["REF:looped.load"]

    def test_a_constant_declared_with_a_type_reads_what_its_value_returns(
        self, tmp_path
    ):
        """A constant declared with a type, whose value is a call returning
        an object, is read as that object's members and as the type for the
        ones the object names none of. A value that is no such call keeps
        the type alone, and a parameter of the name hides the constant."""
        builder = _index(
            tmp_path,
            {
                "make.ts": """
                export interface Store {
                    load(): void;
                    clear(): void;
                }

                export class Thing {
                    show() {}
                }

                export function createStore(name: string): Store {
                    function load() {}
                    return {
                        load,
                        clear: () => name.trim(),
                    };
                }

                export function openStore(name: string) {
                    function load() {}
                    return { load };
                }

                export function makeThing(): Thing {
                    return new Thing();
                }
                """,
                "use.ts": """
                import { createStore, openStore, Store, Thing } from './make';

                const base: Store = createStore('a');
                let mutable: Store = openStore('b');
                const cast = createStore('c') as Store;
                const made: Thing = makeThing();
                const words: Set<string> = new Set();
                const fromLib: Store = lib.make('d');
                const wrapped = (openStore as any)(createStore('e'));

                export function run() {
                    base.load();
                    base.clear();
                    mutable.load();
                    mutable.clear();
                    cast.load();
                    made.show();
                    words.has('a');
                    fromLib.load();
                }

                export function local() {
                    const one: Store = createStore('f');
                    one.load();
                    one.clear();
                }

                export function hidden(base) {
                    base.load();
                }
                """,
            },
        )
        # A cast is no part of the value: the calls written after one are
        # read, and the name is the call's alone.
        assert _calls(builder, "FILE:use.ts") == [
            "FILE:make.ts::createStore",
            "FILE:make.ts::createStore",
            "FILE:make.ts::createStore",
            "FILE:make.ts::openStore",
        ]
        # The object's `load`, the type's `clear`, the object's `clear`, the
        # object's `load` and the type's `load` where the call is a
        # library's, and the type `Thing` declares.
        assert _calls(builder, "FILE:use.ts::run") == [
            "FILE:make.ts::Store::clear",
            "FILE:make.ts::Store::load",
            "FILE:make.ts::Thing::show",
            "FILE:make.ts::createStore::load",
            "FILE:make.ts::createStore::load",
            "FILE:make.ts::createStore::object@13::clear",
            "FILE:make.ts::openStore::load",
        ]
        assert _calls(builder, "FILE:use.ts::local") == [
            "FILE:make.ts::createStore",
            "FILE:make.ts::createStore::load",
            "FILE:make.ts::createStore::object@13::clear",
        ]
        assert _calls(builder, "FILE:use.ts::hidden") == ["FILE:make.ts::Store::load"]

    def test_a_function_of_the_object_is_not_the_factorys_inner_one(self, tmp_path):
        """Two functions of one name — one the factory declares, one written
        in the object it returns — are two nodes. A name written in the
        object's function is read in the factory's scope."""
        builder = _index(
            tmp_path,
            {
                "make.ts": """
                export function wrap(name: string) {
                    function load(times: number) {
                        inner();
                    }
                    return {
                        load: () => {
                            outer();
                            load(1);
                        },
                    };
                }
                function inner() {}
                function outer() {}
                """,
                "use.ts": """
                import { wrap } from './make';
                export function run() {
                    wrap('a').load();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:make.ts::wrap::load") == ["FILE:make.ts::inner"]
        assert _calls(builder, "FILE:make.ts::wrap::object@6::load") == [
            "FILE:make.ts::outer",
            "FILE:make.ts::wrap::load",
        ]
        assert _calls(builder, "FILE:use.ts::run") == [
            "FILE:make.ts::wrap",
            "FILE:make.ts::wrap::object@6::load",
        ]

    def test_a_helper_returns_what_its_body_reaches(self, tmp_path):
        """A helper whose body hands back the result of a call that returns
        an object is read as returning that object, wherever the helper is
        written: in a callback, as an arrow over lines, through arguments
        passed on, and through a second helper."""
        builder = _index(
            tmp_path,
            {
                "thing.ts": """
                export interface ThingApi {
                    show(label: string): void;
                }

                export function useThing(options: { label: string }): ThingApi {
                    function show(label: string) {}
                    return { show };
                }
                """,
                "thing.test.ts": """
                import { useThing } from './thing';

                describe('thing', () => {
                    function make() {
                        return useThing({ label: 'solution' });
                    }

                    const open = () => useThing({
                        label: 'open',
                    });

                    function passed(options) {
                        return useThing(options);
                    }

                    it('shows', () => {
                        const detail = make();
                        detail.show('a');
                        const other = open();
                        other.show('b');
                        const third = passed({ label: 'c' });
                        third.show('c');
                    });
                });
                """,
                "deep.ts": """
                import { useThing } from './thing';

                function first() {
                    return useThing({ label: 'first' });
                }

                export function second() {
                    return first();
                }

                export function run() {
                    second().show('d');
                }
                """,
                "guards.ts": """
                import { useThing, other } from './thing';

                function disagree(flag) {
                    if (flag) {
                        return useThing({ label: 'a' });
                    }
                    return other();
                }

                function recursive(n) {
                    if (!n) {
                        return recursive(n - 1);
                    }
                    return useThing({ label: 'b' });
                }

                disagree(true).show('e');
                recursive(1).show('f');
                """,
            },
        )
        assert _calls(builder, "FILE:thing.test.ts") == [
            "FILE:thing.test.ts::make",
            "FILE:thing.test.ts::open",
            "FILE:thing.test.ts::passed",
            "FILE:thing.ts::useThing::show",
            "FILE:thing.ts::useThing::show",
            "FILE:thing.ts::useThing::show",
        ]
        # The arrow `open` calls `useThing` on its own first line.
        assert _calls(builder, "FILE:thing.test.ts::open") == [
            "FILE:thing.ts::useThing"
        ]
        assert _calls(builder, "FILE:deep.ts::run") == [
            "FILE:deep.ts::second",
            "FILE:thing.ts::useThing::show",
        ]
        # Ways out that disagree, and a helper that returns its own call,
        # leave the result unknown.
        assert _calls(builder, "FILE:guards.ts") == [
            "FILE:guards.ts::disagree",
            "FILE:guards.ts::recursive",
            "REF:show",
            "REF:show",
        ]

    def test_a_property_holds_what_its_value_holds(self, tmp_path):
        """A property of the object a function returns holds what the value
        written for it holds: a name a local of the function was bound to,
        the local's own origin path, or the call written there — on the
        line the property is on."""
        builder = _index(
            tmp_path,
            {
                "make.ts": """
                export function useSelection(keys: string[]) {
                    function toggle(key: string) {}
                    return { toggle };
                }

                export function named(keys: string[]) {
                    const api = useSelection(keys);
                    return { api, keys };
                }

                export function called(keys: string[]) {
                    return { api: useSelection(keys), keys };
                }

                export function wide(keys: string[]) {
                    const api = useSelection(keys);
                    return {
                        api,
                        keys,
                    };
                }

                export function taken(keys: string[]) {
                    const { toggle } = useSelection(keys);
                    return { toggle };
                }
                """,
                "use.ts": """
                import { called, named, taken, wide } from './make';

                export function run() {
                    named([]).api.toggle('a');
                    called([]).api.toggle('b');
                    wide([]).api.toggle('c');
                    taken([]).toggle('d');
                    const { api } = named(['e']);
                    api.toggle('e');
                }
                """,
            },
        )
        assert _calls(builder, "FILE:use.ts::run") == [
            "FILE:make.ts::called",
            "FILE:make.ts::named",
            "FILE:make.ts::named",
            "FILE:make.ts::taken",
            "FILE:make.ts::useSelection::toggle",
            "FILE:make.ts::useSelection::toggle",
            "FILE:make.ts::useSelection::toggle",
            "FILE:make.ts::useSelection::toggle",
            "FILE:make.ts::useSelection::toggle",
            "FILE:make.ts::wide",
        ]

    def test_a_function_written_in_the_object_a_function_returns(self, tmp_path):
        """A property of that object which holds a function written there is
        a member of the object, which no source names, and is what a call of
        the property reaches."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export function createOps(prefix: string) {
                    return {
                        create: (id: string) => save(prefix + id),
                        clear: () => {},
                        plain(id: string) { return save(id); },
                    };
                }

                function save(id: string) {}

                createOps('a').create('1');
                createOps('b').clear();
                createOps('c').plain('2');
                """,
            },
        )
        assert _calls(builder, "FILE:a.ts") == [
            "FILE:a.ts::createOps",
            "FILE:a.ts::createOps",
            "FILE:a.ts::createOps",
            "FILE:a.ts::createOps::object@3::clear",
            "FILE:a.ts::createOps::object@3::create",
            "FILE:a.ts::createOps::object@3::plain",
        ]

    def test_member_call_is_not_pinned_to_a_function_of_another_module(self, tmp_path):
        """`x.make()` reaches a module's `make` only through a binding of
        that module, which `require` and `import()` give as well."""
        builder = _index(
            tmp_path,
            {
                "a.js": """
                export function make() {}
                """,
                "b.js": """
                const lib = require("./a");

                export function direct() {
                    return lib.make();
                }

                export async function lazy() {
                    const { make } = await import("./a");
                    return make();
                }

                export function unknown(x) {
                    return x.make();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:b.js::direct") == ["FILE:a.js::make"]
        assert _calls(builder, "FILE:b.js::lazy") == ["FILE:a.js::make"]
        assert _calls(builder, "FILE:b.js::unknown") == ["REF:x.make"]

    def test_call_on_the_result_of_a_signature_over_several_lines(self, tmp_path):
        """The return type declared after a parameter list that runs over
        lines types the call made on the result, and the body that follows
        makes its own calls."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export interface Text {
                    upper(): Text;
                }

                export interface Count {
                    upper(): Count;
                }

                function build() {}

                export function text(
                    options?: {
                        trim?: boolean;
                    }
                ): Text {
                    return build();
                }

                export function run() {
                    return text({ trim: true }).upper();
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::text") == ["FILE:a.ts::build"]
        assert _calls(builder, "FILE:a.ts::run") == [
            "FILE:a.ts::Text::upper",
            "FILE:a.ts::text",
        ]

    def test_template_literal_over_several_lines_is_not_code(self, tmp_path):
        """Text of a template literal that spans lines adds no call and no
        definition; its `${...}` does, and code goes on after it."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function outline() {}
                function label() {}
                function close() {}

                function emit(name) {
                    const text = `
                        element(s) of ${label()} at http://host/outline()
                        function outline() {
                    `.trim();
                    return close();
                }
                """
            },
        )
        assert "FILE:a.ts::emit::outline" not in {n["id"] for n in builder.nodes}
        assert _calls(builder, "FILE:a.ts::emit") == [
            "FILE:a.ts::close",
            "FILE:a.ts::label",
        ]

    def test_regular_expression_text_is_not_a_call(self, tmp_path):
        """A quote or a `name(` inside a regular expression literal is text;
        the code after the literal is read."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                function outline() {}
                function label() {}

                function check(body) {
                    return /^["'`]outline(s)/.test(body) || label();
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::check") == ["FILE:a.ts::label"]

    def test_constant_declared_over_several_lines_is_a_definition(self, tmp_path):
        """A `const` whose annotation or value runs over lines is called
        through an import as one written on a line is."""
        builder = _index(
            tmp_path,
            {
                "lib/parse.ts": """
                export const parse: <T>(
                    schema: T,
                    value: unknown
                ) => T = make();

                export const table = make({
                    strict: true,
                });
                """,
                "lib/other.ts": """
                export class Reader {
                    parse(value) {}
                }
                """,
                "app.ts": """
                import * as lib from "./lib/parse";

                export function run(value) {
                    return lib.parse(lib.table, value);
                }
                """,
            },
        )
        nodes = {n["id"]: n["type"] for n in builder.nodes}
        assert nodes["FILE:lib/parse.ts::table"] == "Constant"
        assert _calls(builder, "FILE:app.ts::run") == ["FILE:lib/parse.ts::parse"]

    def test_export_under_another_name_is_found_through_an_import(self, tmp_path):
        """`export { _make as make }` is called as `lib.make()`, though no
        definition is named `make`."""
        builder = _index(
            tmp_path,
            {
                "lib.ts": """
                function _make() {}

                export { _make as make };
                """,
                "app.ts": """
                import * as lib from "./lib";

                export function run() {
                    return lib.make();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == ["FILE:lib.ts::_make"]

    def test_member_named_by_a_keyword_is_called(self, tmp_path):
        """After a `.` a keyword names a member: `lib.null()`, `.default()`
        and `.catch()` are calls, of what an interface declares under those
        names."""
        builder = _index(
            tmp_path,
            {
                "lib.ts": """
                export interface Schema {
                    default(value: unknown): Schema;
                    catch(value: unknown): Schema;
                    strict(): Schema;
                }

                function _null(): Schema {
                    return build();
                }

                export { _null as null };
                """,
                "app.ts": """
                import * as lib from "./lib";

                export function run() {
                    try {
                        return lib.null().default(1).catch(2).strict();
                    } catch (error) {
                        return null;
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:lib.ts::Schema::catch",
            "FILE:lib.ts::Schema::default",
            "FILE:lib.ts::Schema::strict",
            "FILE:lib.ts::_null",
        ]

    def test_constant_has_the_members_of_its_declared_type(self, tmp_path):
        """`const Widget: Factory<Widget> = ...` is called as a Factory,
        in its own file and through an import, though an interface of its
        name has a member called the same."""
        builder = _index(
            tmp_path,
            {
                "lib/factory.ts": """
                export interface Factory<T> {
                    init(instance: T): void;
                }
                """,
                "lib/widget.ts": """
                import { Factory } from "./factory";

                export interface Widget {
                    init(): void;
                }

                export const Widget: Factory<Widget> = build(
                    "Widget"
                );

                export function local(instance) {
                    Widget.init(instance);
                }
                """,
                "app.ts": """
                import * as lib from "./lib/widget";

                export function run(instance) {
                    lib.Widget.init(instance);
                }
                """,
            },
        )
        init = "FILE:lib/factory.ts::Factory::init"
        assert _calls(builder, "FILE:lib/widget.ts::local") == [init]
        assert _calls(builder, "FILE:app.ts::run") == [init]

    def test_function_held_by_an_object_literal_is_its_member(self, tmp_path):
        """`coerce.text()` reaches the function the constant `coerce` holds
        under `text`, as a property or as a method, not a function of that
        name."""
        builder = _index(
            tmp_path,
            {
                "lib.ts": """
                export function text() {}

                export function count() {}

                export const coerce = {
                    text: ((arg) => text({ ...arg, coerce: true })) as typeof text,
                    count(arg) {
                        return count(arg);
                    },
                };

                export function local() {
                    return coerce.text();
                }
                """,
                "app.ts": """
                import * as lib from "./lib";

                export function run() {
                    lib.coerce.text();
                    return lib.coerce.count();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib.ts::local") == ["FILE:lib.ts::coerce::text"]
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:lib.ts::coerce::count",
            "FILE:lib.ts::coerce::text",
        ]
        assert _calls(builder, "FILE:lib.ts::coerce::count") == ["FILE:lib.ts::count"]

    def test_object_literal_holds_the_modules_it_names(self, tmp_path):
        """`{ ...schemas, iso: dates }` holds what the module `schemas`
        exports, and the module `dates` under `iso`."""
        builder = _index(
            tmp_path,
            {
                "schemas.ts": """
                export function text() {}
                """,
                "dates.ts": """
                export function text() {}
                """,
                "app.ts": """
                import * as schemas from "./schemas";
                import * as dates from "./dates";

                const z = {
                    ...schemas,
                    iso: dates,
                };

                export function run() {
                    z.text();
                    return z.iso.text();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == [
            "FILE:dates.ts::text",
            "FILE:schemas.ts::text",
        ]

    def test_named_import_of_a_namespace_is_a_receiver(self, tmp_path):
        """`import { util }` binds what the module exports as `util`: the
        module `export * as util` names."""
        builder = _index(
            tmp_path,
            {
                "lib/util.ts": """
                export function pick() {}
                """,
                "lib/other.ts": """
                export function pick() {}
                """,
                "lib/index.ts": """
                export * as util from "./util";
                """,
                "app.ts": """
                import { util } from "./lib/index";

                export function run() {
                    return util.pick();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::run") == ["FILE:lib/util.ts::pick"]

    def test_property_that_holds_a_function_is_not_the_one_method(self, tmp_path):
        """`effect.refinement()` on an unknown receiver may call the
        function an Effect holds, so the one method of that name is not
        pinned."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export interface Effect {
                    refinement: (value: unknown) => void;
                }

                export class Schema {
                    refinement(check) {}
                    strict() {}
                }

                export function run(effect, schema) {
                    effect.refinement(1);
                    schema.strict();
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::run") == [
            "FILE:a.ts::Schema::strict",
            "REF:effect.refinement",
        ]

    def test_call_on_a_standard_value_is_external_and_typed(self, tmp_path):
        """A method of a standard type is the library's, whatever the
        project names its own, and what it returns is typed for the call
        after it: by the type arguments of an array or a map, a project
        type."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Schema {
                    run() {}
                    map(fn) {}
                    get(key) {}
                    split(by) {}
                    keys() {}
                }

                export class Decoy {
                    run() {}
                }

                export function chains(text: string, table: object) {
                    text.split(",").map((part) => part.trim()).filter(Boolean);
                    Object.keys(table).forEach((key) => key.trim());
                }

                export function held(all: Schema[], named = new Map<string, Schema>()) {
                    all.at(0).run();
                    named.get("a").run();
                }

                export async function fetched(url) {
                    const response = await fetch(url);
                    return response.json();
                }
                """
            },
        )
        run = "FILE:a.ts::Schema::run"
        assert _calls(builder, "FILE:a.ts::chains") == []
        assert _calls(builder, "FILE:a.ts::held") == [run, run]
        assert _calls(builder, "FILE:a.ts::fetched") == []

    def test_new_constructs_an_instance_and_a_call_returns(self, tmp_path):
        """`new Type<arguments>(...)` is an instance of the type with the
        arguments written; the same name called without `new` returns what
        its call signature says."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Schema {
                    run() {}
                    getTime() {}
                }

                export class Decoy {
                    run() {}
                    getTime() {}
                }

                export function built() {
                    new Date().getTime();
                    new Map<string, Schema>().get("a").run();
                    new Schema().run();
                }

                export function called() {
                    Date().getTime();
                }
                """
            },
        )
        run = "FILE:a.ts::Schema::run"
        assert _calls(builder, "FILE:a.ts::built") == [
            "FILE:a.ts::Schema",
            run,
            run,
        ]
        assert _calls(builder, "FILE:a.ts::called") == ["REF:getTime"]

    def test_type_parameter_is_what_the_call_site_says(self, tmp_path):
        """A parameter of a function passed to a generic function, declared
        as one of its type parameters, has the type the call is written
        with, else the one that the declared type of the variable the
        result initialises puts in its place, else the parameter's bound;
        with none of them it has no type."""
        builder = _index(
            tmp_path,
            {
                "core.ts": """
                export interface Maker<T> {
                    make(): T;
                }

                export function maker<T>(name: string, init: (inst: T) => void): Maker<T> {
                    return undefined as any;
                }

                export function each<T extends Bounded>(run: (item: T) => void): void {}

                export interface Bounded {
                    check(): void;
                }
                """,
                "a.ts": """
                import * as core from "./core";

                export interface Checked {
                    check(): void;
                }

                export interface Other {
                    check(): void;
                }

                export const One: core.Maker<Checked> = core.maker("One", (inst) => {
                    inst.check();
                });

                export const Two: core.Maker<Checked> = core.maker(
                    "Two",
                    (inst) => {
                        inst.check();
                    }
                );

                export const Three: core.Maker<Checked> = core.maker("Three", (inst) => {
                    inst.check();
                }).make();

                export function written() {
                    core.maker<Other>("Four", (inst) => inst.check());
                }

                export function bounded() {
                    core.each((item) => item.check());
                }

                export function unsaid() {
                    core.maker("Five", (inst) => inst.check());
                }
                """,
            },
        )
        maker = "FILE:core.ts::maker"
        checked = "FILE:a.ts::Checked::check"
        assert _calls(builder, "FILE:a.ts") == [
            checked,
            checked,
            "FILE:core.ts::Maker::make",
            maker,
            maker,
            maker,
            "REF:inst.check",
        ]
        assert _calls(builder, "FILE:a.ts::written") == [
            "FILE:a.ts::Other::check",
            maker,
        ]
        assert _calls(builder, "FILE:a.ts::bounded") == [
            "FILE:core.ts::Bounded::check",
            "FILE:core.ts::each",
        ]
        assert _calls(builder, "FILE:a.ts::unsaid") == [maker, "REF:inst.check"]

    def test_element_of_an_array_literal_is_any_of_its_elements(self, tmp_path):
        """What a `for...of` over an array literal binds is any of its
        elements, of an object literal the property named: a call on it is
        resolved where every element leads to the same definition."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Base {
                    run() {}
                }

                export class Left extends Base {
                    only() {}
                }

                export class Right extends Base {
                    only() {}
                }

                export class Decoy {
                    run() {}
                    only() {}
                }

                export function listed() {
                    const cases = [
                        { schema: new Left(), input: "a" },
                        { schema: new Right(), input: "b" },
                    ];
                    for (const { schema, input } of cases) {
                        schema.run();
                        schema.only();
                    }
                }

                export function inline() {
                    for (const each of [new Left(), new Right()]) {
                        each.run();
                    }
                }

                export function spread(more) {
                    const cases = [
                        { schema: new Left() },
                        ...more,
                    ];
                    for (const { schema } of cases) {
                        schema.run();
                    }
                }
                """
            },
        )
        run = "FILE:a.ts::Base::run"
        built = ["FILE:a.ts::Left", "FILE:a.ts::Right"]
        assert _calls(builder, "FILE:a.ts::listed") == [
            run,
            *built,
            "REF:schema.only",
        ]
        assert _calls(builder, "FILE:a.ts::inline") == [run, *built]
        assert _calls(builder, "FILE:a.ts::spread") == [
            "FILE:a.ts::Left",
            "REF:schema.run",
        ]

    def test_return_type_is_read_as_written(self, tmp_path):
        """A return type is read whole: a conditional type is the type its
        branches agree on, an intersection the operand derived from the
        others, a type parameter the type of the argument passed for it,
        and of overloads the one that takes as many arguments and type
        arguments, the first where all take the same types, else the type
        they all return."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Meta {
                    describe() {}
                }

                export class Base {
                    run() {}
                }

                export class Decoy {
                    run() {}
                    describe() {}
                    only() {}
                }

                export class Schema extends Base {
                    only() {}
                    refine<Ch extends (arg: unknown) => unknown>(
                        check: Ch
                    ): Ch extends (arg: any) => arg is infer R ? this & Base : this {
                        return this as any;
                    }
                    pick<K>(key: K): K extends string ? Schema : Meta {
                        return this as any;
                    }
                    meta(): Meta;
                    meta(data: object): this;
                    meta(data?: object): any {
                        return this;
                    }
                    with(value: string): Schema;
                    with(value: number): Schema;
                    with(value: any): any {
                        return this;
                    }
                    as(value: string): Schema;
                    as(value: number): Meta;
                    as(value: any): any {
                        return this;
                    }
                    same(value: string): Schema;
                    same<T>(value: string): Meta;
                    same(value: any): any {
                        return this;
                    }
                }

                export function compile<T>(schema: T): T {
                    return schema;
                }

                export function chained(schema: Schema) {
                    schema.refine((value) => true).only();
                    schema.meta({}).only();
                    schema.with(1).only();
                    compile(schema).only();
                    compile(new Schema()).only();
                    schema.same("a").only();
                }

                export function chosen(schema: Schema) {
                    schema.meta().describe();
                    schema.same<number>("a").describe();
                }

                export function undecided(schema: Schema) {
                    schema.pick("a").run();
                    schema.as(1).only();
                }
                """
            },
        )
        schema = "FILE:a.ts::Schema"
        assert _calls(builder, "FILE:a.ts::chained") == [
            schema,
            f"{schema}::meta",
            *[f"{schema}::only"] * 6,
            f"{schema}::refine",
            f"{schema}::same",
            f"{schema}::with",
            "FILE:a.ts::compile",
            "FILE:a.ts::compile",
        ]
        assert _calls(builder, "FILE:a.ts::chosen") == [
            "FILE:a.ts::Meta::describe",
            "FILE:a.ts::Meta::describe",
            f"{schema}::meta",
            f"{schema}::same",
        ]
        assert _calls(builder, "FILE:a.ts::undecided") == [
            f"{schema}::as",
            f"{schema}::pick",
            "REF:only",
            "REF:run",
        ]

    def test_function_without_a_return_type_returns_what_its_body_does(self, tmp_path):
        """A function that declares no return type returns what every
        `return` of its body gives: `this`, a construction, one call or a
        name of a declared type. One whose returns differ, or that returns
        anything else, returns what is not known."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Decoy {
                    run() {}
                }

                export class Other {
                    run() {}
                }

                export class Schema {
                    run() {}
                    self() {
                        return this;
                    }
                    made(flag) {
                        if (flag) return new Schema();
                        return new Schema({
                            flag,
                        });
                    }
                    called() {
                        return this.made(true);
                    }
                    given(other: Schema) {
                        return other;
                    }
                    mixed(flag) {
                        if (flag) {
                            return new Other();
                        }
                        return this;
                    }
                    wrapped() {
                        return [this].map((each) => {
                            return each;
                        });
                    }
                    chained() {
                        return this.made(true).self();
                    }
                    async later() {
                        return this;
                    }
                }

                const build = (flag) => new Schema().made(flag);
                const fresh = () => new Schema();

                export function decided(schema: Schema) {
                    schema.self().run();
                    schema.made(true).run();
                    schema.called().run();
                    schema.given(schema).run();
                    fresh().run();
                }

                export function undecided(schema: Schema) {
                    schema.mixed(true).run();
                    schema.wrapped().run();
                    schema.chained().run();
                    schema.later().run();
                    build(true).run();
                }
                """
            },
        )
        schema = "FILE:a.ts::Schema"
        assert _calls(builder, "FILE:a.ts::decided") == [
            f"{schema}::called",
            f"{schema}::given",
            f"{schema}::made",
            *[f"{schema}::run"] * 5,
            f"{schema}::self",
            "FILE:a.ts::fresh",
        ]
        assert _calls(builder, "FILE:a.ts::undecided") == [
            f"{schema}::chained",
            f"{schema}::later",
            f"{schema}::mixed",
            f"{schema}::wrapped",
            "FILE:a.ts::build",
            *["REF:run"] * 5,
        ]

    def test_variable_has_the_type_its_value_is_asserted_as(self, tmp_path):
        """`const name = value as Type` declares the variable as the type,
        unless an operator before the `as` takes only part of the value
        with it."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Schema {
                    run() {}
                }

                export class Decoy {
                    run() {}
                }

                export function asserted(event, other) {
                    const schema = event.target as Schema;
                    schema.run();
                    const twice = event.target as unknown as Schema;
                    twice.run();
                }

                export function partly(event, other) {
                    const either = other || event.target as Schema;
                    either.run();
                }
                """
            },
        )
        run = "FILE:a.ts::Schema::run"
        assert _calls(builder, "FILE:a.ts::asserted") == [run, run]
        assert _calls(builder, "FILE:a.ts::partly") == ["REF:either.run"]

    def test_member_a_standard_type_has_not_may_be_the_projects(self, tmp_path):
        """A value declared as a standard type may be of a project type
        that extends it, so a member the type has not is neither dropped
        nor pinned by its name."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Problem extends Error {
                    flatten() {}
                }

                export function report(error: Error) {
                    error.flatten();
                    return error.toString();
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::report") == ["REF:error.flatten"]

    def test_global_is_the_librarys_unless_its_file_binds_the_name(self, tmp_path):
        """`document` and `globalThis.console` are the browser's and the
        language's; a parameter called `document` is not."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Page {
                    typeset() {}
                    log() {}
                    querySelector() {}
                }

                export function setup() {}

                export function draw() {
                    document.querySelector("p").remove();
                    globalThis.console.log("drawn");
                    globalThis.setup();
                }
                """,
                "b.ts": """
                export function print(document) {
                    document.typeset();
                    document.querySelector("p");
                }
                """,
            },
        )
        assert _calls(builder, "FILE:a.ts::draw") == ["FILE:a.ts::setup"]
        assert _calls(builder, "FILE:b.ts::print") == [
            "FILE:a.ts::Page::typeset",
            "REF:document.querySelector",
        ]

    def test_function_passed_to_a_standard_method_is_typed_by_it(self, tmp_path):
        """What `map`, `forEach`, `find`, `reduce` and `then` hand to the
        function they are passed types its parameters, and what `map`
        returns is typed by what that function returns."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Schema {
                    child(): Child {
                        return new Child();
                    }
                }

                export class Child {
                    run() {}
                }

                export class Decoy {
                    child() {}
                    run() {}
                }

                export function each(all: Schema[], later: Promise<Schema>) {
                    all.forEach((schema) => schema.child());
                    all.find((schema) => schema.child()).child();
                    all.reduce((total, schema) => total + schema.child(), 0);
                    later.then((schema) => schema.child());
                }

                export function mapped(all: Array<Schema>) {
                    all.map((schema) => schema.child()).forEach((child) => {
                        child.run();
                    });
                }
                """
            },
        )
        child = "FILE:a.ts::Schema::child"
        assert _calls(builder, "FILE:a.ts::each") == [child] * 5
        assert _calls(builder, "FILE:a.ts::mapped") == ["FILE:a.ts::Child::run", child]

    def test_function_passed_to_a_project_function_is_typed_by_it(self, tmp_path):
        """The parameters of a function passed as an argument have the
        types the function called declares for them."""
        builder = _index(
            tmp_path,
            {
                "doc.ts": """
                export class Doc {
                    write(line: string) {}
                    indented(body: (inner: Doc, depth: number) => void) {}
                }

                export class Decoy {
                    write(line) {}
                }
                """,
                "app.ts": """
                import { Doc } from "./doc";

                export function emit(doc: Doc) {
                    doc.indented((inner, depth) => {
                        inner.write(depth.toFixed(0));
                    });
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.ts::emit") == [
            "FILE:doc.ts::Doc::indented",
            "FILE:doc.ts::Doc::write",
        ]

    def test_loop_variable_and_pattern_are_typed_by_their_value(self, tmp_path):
        """`for...of` gives the elements of what it iterates, `for...in`
        names; `const { a } = value` and `value.a.name()` read the property
        `a` as the type of `value` declares it."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Schema {
                    run() {}
                    trim() {}
                }

                export class Decoy {
                    run() {}
                }

                export interface Context {
                    schema: Schema;
                    seen: Map<string, Schema>;
                }

                export function loops(all: Schema[], table: object) {
                    for (const schema of all) {
                        schema.run();
                    }
                    for (const key in table) {
                        key.trim();
                    }
                }

                export function parts(context: Context, all: Schema[]) {
                    const { schema, seen } = context;
                    const [first] = all;
                    schema.run();
                    first.run();
                    seen.get("a").run();
                    context.schema.run();
                    context.seen.clear();
                }
                """
            },
        )
        run = "FILE:a.ts::Schema::run"
        assert _calls(builder, "FILE:a.ts::loops") == [run]
        assert _calls(builder, "FILE:a.ts::parts") == [run] * 4

    def test_variable_has_the_type_of_the_literal_it_is(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Suite {
                    test(name) {}
                    map(fn) {}
                    trim() {}
                }

                export function check(body, suite) {
                    const blank = /^\\s*$/;
                    const names = ["a", "b"];
                    const label = `check ${body}`;
                    suite.test(body);
                    return blank.test(body) || names.map(String) || label.trim();
                }
                """
            },
        )
        assert _calls(builder, "FILE:a.ts::check") == ["REF:suite.test"]

    def test_variable_declared_again_has_no_type_from_before(self, tmp_path):
        """A name declared a second time without a type is of none, not of
        the type an earlier declaration or a field of that name has."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Schema {
                    run() {}
                }

                export class Other {
                    run() {}
                }

                export interface Options {
                    memo?: Schema;
                }

                test("first", () => {
                    const schema: Schema = load();
                    schema.run();
                });

                test("second", () => {
                    const schema = flag ? left : right;
                    const memo = config.memo;
                    schema.run();
                    memo.run();
                });
                """
            },
        )
        assert _calls(builder, "FILE:a.ts") == [
            "FILE:a.ts::Schema::run",
            "REF:memo.run",
            "REF:schema.run",
        ]

    def test_value_of_a_chain_below_its_name_and_of_a_typed_call(self, tmp_path):
        """A variable is bound to the call its value ends with where the
        chain starts on the line below the name, and where the call is
        written with type arguments."""
        builder = _index(
            tmp_path,
            {
                "lib.ts": """
                export interface Schema {
                    strict(): Schema;
                    run(): void;
                }

                export interface Decoy {
                    run(): void;
                }

                export function make<T>(shape?: object): Schema {
                    return build(shape);
                }
                """,
                "app.ts": """
                import * as lib from "./lib";

                export function below() {
                    const schema = lib
                        .make({
                            deep: true,
                        })
                        .strict();
                    schema.run();
                }

                export function typed() {
                    const schema = lib.make<{ deep: boolean }>();
                    schema.run();
                }
                """,
            },
        )
        run = "FILE:lib.ts::Schema::run"
        assert run in _calls(builder, "FILE:app.ts::below")
        assert _calls(builder, "FILE:app.ts::typed") == [run, "FILE:lib.ts::make"]


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
                \thandler()
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
        assert _calls(builder, "FILE:app.go::withParameter") == ["FILE:app.go::handler"]
        assert _calls(builder, "FILE:app.go::withVariable") == []
        assert _calls(builder, "FILE:app.go::direct") == [
            "FILE:app.go::check",
            "FILE:app.go::handler",
        ]

    def test_signature_with_function_types_or_over_several_lines(self, tmp_path):
        """A function is read with its parameters and results where those
        have function types or are written one to a line."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct{}

                func (j *Job) Run() {}

                type Decoy struct{}

                func (d *Decoy) Run() {}

                func each(first *Job, visit func(*Job) error) func() error {
                \tfirst.Run()
                \treturn nil
                }

                func build(
                \tjob *Job,
                \twrap func(func() error) func() error,
                ) (
                \tmade *Job,
                \terr error,
                ) {
                \tjob.Run()
                \twrap()
                \treturn
                }

                func wrap() {}

                func use() {
                \tmade, _ := build(nil, nil)
                \tmade.Run()
                \teach(made, nil)
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::each") == ["FILE:app.go::Job::Run"]
        assert _calls(builder, "FILE:app.go::build") == ["FILE:app.go::Job::Run"]
        assert _calls(builder, "FILE:app.go::use") == [
            "FILE:app.go::Job::Run",
            "FILE:app.go::build",
            "FILE:app.go::each",
        ]

    def test_receiver_and_first_parameter_are_local_to_their_function(self, tmp_path):
        """A name another function has as its receiver or first parameter
        has no type from that."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct{}

                func (job *Job) Run() {}

                type Step struct{}

                func (s *Step) Run() {}

                func start(step *Step, count int) {
                \tstep.Run()
                }

                func (job *Job) each(items []interface{}) {
                \tjob.Run()
                \tstep := items[0]
                \tstep.Run()
                }

                func all(items []interface{}) {
                \tjob := items[0]
                \tjob.Run()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::start") == ["FILE:app.go::Step::Run"]
        assert _calls(builder, "FILE:app.go::Job::each") == [
            "FILE:app.go::Job::Run",
            "REF:step.Run",
        ]
        assert _calls(builder, "FILE:app.go::all") == ["REF:job.Run"]

    def test_variable_bound_to_a_field_or_variable_has_its_type(self, tmp_path):
        """`c := hs.c` gives `c` the type the field is declared with and
        `other := hs` that of `hs`; a field no declaration types gives none."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Conn struct{}

                func (c *Conn) Write() {}

                type Mock struct{}

                func (m *Mock) Write() {}

                type state struct {
                \tc *Conn
                }

                func (hs *state) send() {
                \tc := hs.c
                \tc.Write()
                }

                func (hs *state) copied() {
                \tother := hs
                \tconn := other.c
                \tconn.Write()
                }

                func (hs *state) untyped() {
                \tc := hs.spare
                \tc.Write()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::state::send") == [
            "FILE:app.go::Conn::Write"
        ]
        assert _calls(builder, "FILE:app.go::state::copied") == [
            "FILE:app.go::Conn::Write"
        ]
        assert _calls(builder, "FILE:app.go::state::untyped") == ["REF:c.Write"]

    def test_names_of_a_parameter_list_that_share_a_type(self, tmp_path):
        """`first, second *Conn` declares both names as a `Conn`."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Conn struct{}

                func (c *Conn) Write() {}

                type Mock struct{}

                func (m *Mock) Write() {}

                func pair(first, second, third *Conn, mock *Mock) (a, b *Mock) {
                \tfirst.Write()
                \tsecond.Write()
                \tthird.Write()
                \ta.Write()
                \treturn
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::pair") == [
            "FILE:app.go::Conn::Write",
            "FILE:app.go::Conn::Write",
            "FILE:app.go::Conn::Write",
            "FILE:app.go::Mock::Write",
        ]

    def test_type_assertion_in_an_if_initializer(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "net"

                type Conn struct{}

                func (c *Conn) Temporary() bool { return false }

                type Mock struct{}

                func (m *Mock) Write() {}

                func (c *Conn) Write() {}

                func check(err error, in interface{}) {
                \tif e, ok := err.(net.Error); ok && e.Temporary() {
                \t\treturn
                \t}
                \tif mock, ok := in.(*Mock); ok {
                \t\tmock.Write()
                \t}
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::check") == ["FILE:app.go::Mock::Write"]

    def test_entry_of_a_var_block_is_a_package_level_variable(self, tmp_path):
        """A name declared in a `var (...)` block has its declared type in
        every function of the file."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "crypto/elliptic"

                type Conn struct{}

                func (c *Conn) Write() {}

                func (c *Conn) Params() {}

                type Mock struct{}

                func (m *Mock) Write() {}

                var (
                \tcurve  elliptic.Curve
                \tconn   *Conn
                \tbackup = &Mock{}
                )

                func use() {
                \tcurve.Params()
                \tconn.Write()
                \tbackup.Write()
                }

                func local() {
                \tvar (
                \t\tinner *Mock
                \t)
                \tinner.Write()
                }

                func elsewhere(items []interface{}) {
                \tinner := items[0]
                \tinner.Write()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::use") == [
            "FILE:app.go::Conn::Write",
            "FILE:app.go::Mock::Write",
        ]
        assert _calls(builder, "FILE:app.go::local") == ["FILE:app.go::Mock::Write"]
        assert _calls(builder, "FILE:app.go::elsewhere") == ["REF:inner.Write"]

    def test_new_and_type_assertion_give_the_variable_its_type(self, tmp_path):
        """`new(T)` and `value.(*T)` yield a T: a project type's method
        resolves, an imported package's is external."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "bytes"

                type Command struct{}

                func (c *Command) Runnable() bool { return true }

                type Other struct{}

                func (o *Other) Runnable() bool { return false }

                func (o *Other) String() string { return "" }

                func usage(in interface{}) string {
                	c := in.(*Command)
                	buf := new(bytes.Buffer)
                	if c.Runnable() {
                		return buf.String()
                	}
                	return ""
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::usage") == [
            "FILE:app.go::Command::Runnable"
        ]

    def test_field_of_a_declared_value_has_its_declared_type(self, tmp_path):
        """`c.flags` is the FlagSet of an imported package and `c.parent`
        a Command, as the struct declares them."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import flag "github.com/spf13/pflag"

                type Command struct {
                	flags  *flag.FlagSet
                	parent *Command
                }

                func (c *Command) SetOutput() {}

                func (c *Command) run() {
                	c.flags.SetOutput()
                	c.parent.SetOutput()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::Command::run") == [
            "FILE:app.go::Command::SetOutput"
        ]

    def test_each_name_is_typed_by_the_result_in_its_position(self, tmp_path):
        """`a, b := f()` and `var a, b = f()` give every name the type `f`
        declares for that result, named or not."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct{}

                func (j *Job) Run() {}

                type Report struct{}

                func (r *Report) Run() {}

                func (r *Report) Send() {}

                type Decoy struct{}

                func (d *Decoy) Run() {}

                func (d *Decoy) Send() {}

                func start() (*Job, *Report, error) {
                \treturn nil, nil, nil
                }

                func named() (job, spare *Job, report *Report) {
                \treturn
                }

                func one() *Report {
                \treturn nil
                }

                func short() {
                \tjob, report, err := start()
                \tjob.Run()
                \treport.Send()
                }

                func blank() {
                \t_, report, _ := start()
                \treport.Run()
                }

                func grouped() {
                \tvar _, spare, report = named()
                \tspare.Run()
                \treport.Send()
                }

                func single() {
                \treport := one()
                \treport.Send()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::short") == [
            "FILE:app.go::Job::Run",
            "FILE:app.go::Report::Send",
            "FILE:app.go::start",
        ]
        assert _calls(builder, "FILE:app.go::blank") == [
            "FILE:app.go::Report::Run",
            "FILE:app.go::start",
        ]
        assert _calls(builder, "FILE:app.go::grouped") == [
            "FILE:app.go::Job::Run",
            "FILE:app.go::Report::Send",
            "FILE:app.go::named",
        ]
        assert _calls(builder, "FILE:app.go::single") == [
            "FILE:app.go::Report::Send",
            "FILE:app.go::one",
        ]

    def test_later_name_is_typed_only_by_a_declared_result(self, tmp_path):
        """A later name has no type where the call is of a method value or
        reaches no one definition; the last declaration of a name decides,
        and an assignment declares nothing."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Report struct{}

                func (r *Report) Send() {}

                type Decoy struct{}

                func (d *Decoy) Send() {}

                type Source struct{}

                func (s *Source) Fetch() (int, *Report) {
                \treturn 0, nil
                }

                type Mirror struct{}

                func (m *Mirror) Fetch() (int, *Decoy) {
                \treturn 0, nil
                }

                func methodValue(s *Source) {
                \tfetch := s.Fetch
                \t_, report := fetch()
                \treport.Send()
                }

                func unknownCallee(sources []interface{}) {
                \tsource := sources[0]
                \t_, report := source.Fetch()
                \treport.Send()
                }

                func shadowed(s *Source, m *Mirror) {
                \t_, report := s.Fetch()
                \tif m != nil {
                \t\t_, report := m.Fetch()
                \t\treport.Send()
                \t}
                }

                func assigned(sources []interface{}, report *Decoy) {
                \tsource := sources[0]
                \t_, report = source.Fetch()
                \treport.Send()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::methodValue") == ["REF:report.Send"]
        assert _calls(builder, "FILE:app.go::unknownCallee") == [
            "REF:report.Send",
            "REF:source.Fetch",
        ]
        assert _calls(builder, "FILE:app.go::shadowed") == [
            "FILE:app.go::Decoy::Send",
            "FILE:app.go::Mirror::Fetch",
            "FILE:app.go::Source::Fetch",
        ]
        assert _calls(builder, "FILE:app.go::assigned") == [
            "FILE:app.go::Decoy::Send",
            "REF:source.Fetch",
        ]

    def test_error_method_of_the_predeclared_error_is_external(self, tmp_path):
        """`Error()` on a value declared `error`, as a parameter, a named
        result or a result of a project function, is the language's. On a
        value of unknown type it may be a project type's."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type failure struct{}

                func (f *failure) Error() string { return "" }

                type refusal struct{}

                func (r *refusal) Error() string { return "" }

                func check() error {
                \treturn nil
                }

                func load() (int, error) {
                \treturn 0, nil
                }

                func fail() *failure {
                \treturn nil
                }

                func single() string {
                \terr := check()
                \treturn err.Error()
                }

                func later() string {
                \t_, err := load()
                \treturn err.Error()
                }

                func parameter(err error) string {
                \treturn err.Error()
                }

                func namedResult() (text string, err error) {
                \ttext = err.Error()
                \treturn
                }

                func own() string {
                \terr := fail()
                \treturn err.Error()
                }

                func unknown() string {
                \tproduce := check
                \terr := produce()
                \treturn err.Error()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::single") == ["FILE:app.go::check"]
        assert _calls(builder, "FILE:app.go::later") == ["FILE:app.go::load"]
        assert _calls(builder, "FILE:app.go::parameter") == []
        assert _calls(builder, "FILE:app.go::namedResult") == []
        assert _calls(builder, "FILE:app.go::own") == [
            "FILE:app.go::fail",
            "FILE:app.go::failure::Error",
        ]
        assert _calls(builder, "FILE:app.go::unknown") == ["REF:err.Error"]

    def test_project_type_named_error_is_not_the_predeclared_one(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type error struct{}

                func (e *error) Error() string { return "" }

                func check() *error {
                \treturn nil
                }

                func report() string {
                \terr := check()
                \treturn err.Error()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::report") == [
            "FILE:app.go::check",
            "FILE:app.go::error::Error",
        ]

    def test_range_variable_has_the_declared_element_type(self, tmp_path):
        """`for _, v := range xs` gives `v` the element type of the slice,
        array or map `xs` is declared as, or is a literal of, and
        `for k := range m` gives `k` the key type of a map."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Name string

                func (n Name) Run() {}

                type Step struct{}

                func (s *Step) Run() {}

                type Decoy struct{}

                func (d *Decoy) Run() {}

                type Job struct {
                \tsteps  []*Step
                \tbyName map[Name]*Step
                }

                func parameter(steps []*Step) {
                \tfor _, step := range steps {
                \t\tstep.Run()
                \t}
                }

                func variadic(steps ...*Step) {
                \tfor _, step := range steps {
                \t\tstep.Run()
                \t}
                }

                func field(j *Job) {
                \tfor name, step := range j.byName {
                \t\tname.Run()
                \t}
                \tfor _, step := range j.steps {
                \t\tstep.Run()
                \t}
                }

                func made() {
                \tindex := make(map[Name]Decoy)
                \tfor name := range index {
                \t\tname.Run()
                \t}
                }

                func array() {
                \tvar steps [4]Step
                \tfor _, step := range steps {
                \t\tstep.Run()
                \t}
                }

                func literal() {
                \tfor _, step := range []*Step{nil} {
                \t\tstep.Run()
                \t}
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::parameter") == ["FILE:app.go::Step::Run"]
        assert _calls(builder, "FILE:app.go::variadic") == ["FILE:app.go::Step::Run"]
        assert _calls(builder, "FILE:app.go::field") == [
            "FILE:app.go::Name::Run",
            "FILE:app.go::Step::Run",
        ]
        assert _calls(builder, "FILE:app.go::made") == ["FILE:app.go::Name::Run"]
        assert _calls(builder, "FILE:app.go::array") == ["FILE:app.go::Step::Run"]
        assert _calls(builder, "FILE:app.go::literal") == ["FILE:app.go::Step::Run"]

    def test_range_variable_of_no_known_element_type_is_untyped(self, tmp_path):
        """A range variable hides an earlier declaration of its name, and
        the single variable of a range over a slice is its index."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Step struct{}

                func (s *Step) Run() {}

                type Decoy struct{}

                func (d *Decoy) Run() {}

                func shadowed(step *Step, pending func() []*Decoy) {
                \tfor _, step := range pending() {
                \t\tstep.Run()
                \t}
                }

                func index(steps []*Step) {
                \tfor step := range steps {
                \t\tstep.Run()
                \t}
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::shadowed") == ["REF:step.Run"]
        assert _calls(builder, "FILE:app.go::index") == []

    def test_initializer_of_an_if_or_switch_binds_its_names(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Report struct{}

                func (r *Report) Send() {}

                type Decoy struct{}

                func (d *Decoy) Send() {}

                func one() *Report {
                \treturn nil
                }

                func two() (int, *Report) {
                \treturn 0, nil
                }

                func guarded(ready bool) {
                \tif report := one(); report != nil {
                \t\treport.Send()
                \t} else if _, other := two(); ready {
                \t\tother.Send()
                \t}
                \tswitch n, last := two(); n {
                \tcase 0:
                \t\tlast.Send()
                \t}
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::guarded") == [
            "FILE:app.go::Report::Send",
            "FILE:app.go::Report::Send",
            "FILE:app.go::Report::Send",
            "FILE:app.go::one",
            "FILE:app.go::two",
            "FILE:app.go::two",
        ]

    def test_value_from_another_module_has_no_project_methods(self, tmp_path):
        """Every result of a call into a package outside the module is from
        outside, as is a field of a value declared with a type of such a
        package."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import (
                \t"net/http"
                \t"os"

                \tflag "github.com/spf13/pflag"
                )

                type failure struct{}

                func (f *failure) Error() string { return "" }

                func (f *failure) Type() string { return "" }

                type refusal struct{}

                func (r *refusal) Error() string { return "" }

                func (r *refusal) Type() string { return "" }

                func temp() string {
                \t_, err := os.MkdirTemp("", "app")
                \treturn err.Error()
                }

                func kind(f *flag.Flag) string {
                \treturn f.Value.Type()
                }

                func fetch() string {
                \tresponse, _ := http.Get("")
                \treturn response.Body.Type()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::temp") == []
        assert _calls(builder, "FILE:app.go::kind") == []
        assert _calls(builder, "FILE:app.go::fetch") == []

    def test_results_of_the_standard_library_are_typed_by_its_data(self, tmp_path):
        """What a standard function or method returns has the type the
        data gives each result, so none of its methods is the project's and
        ranging over it yields its elements."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import (
                \t"bufio"
                \t"net/http"
                \t"os"
                )

                type Entry struct{}

                func (e *Entry) Name() string { return "" }

                type failure struct{}

                func (f *failure) Error() string { return "" }

                func (f *failure) Close() error { return nil }

                func (f *failure) Add(key, value string) {}

                type refusal struct{}

                func (r *refusal) Error() string { return "" }

                func (r *refusal) Close() error { return nil }

                func (r *refusal) Add(key, value string) {}

                func list(dir string) {
                \tentries, _ := os.ReadDir(dir)
                \tfor _, entry := range entries {
                \t\tentry.Name()
                \t}
                }

                func line(r *bufio.Reader) string {
                \t_, err := r.ReadString(10)
                \treturn err.Error()
                }

                func fetch(client *http.Client) {
                \trequest, _ := http.NewRequest("GET", "", nil)
                \trequest.Header.Add("a", "b")
                \tresponse, _ := client.Do(request)
                \tresponse.Body.Close()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::list") == []
        assert _calls(builder, "FILE:app.go::line") == []
        assert _calls(builder, "FILE:app.go::fetch") == []

    def test_standard_generic_type_hands_on_a_project_type(self, tmp_path):
        """`atomic.Pointer[Config]` loads a `Config`, as the data's `*$0`
        says."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "sync/atomic"

                type Config struct{}

                func (c *Config) Apply() {}

                type Decoy struct{}

                func (d *Decoy) Apply() {}

                type Stack[T any] struct{}

                func (s *Stack[T]) Apply() {}

                func reload(current *atomic.Pointer[Config], spare Stack[int]) {
                \tcurrent.Load().Apply()
                \tspare.Apply()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::reload") == [
            "FILE:app.go::Config::Apply",
            "FILE:app.go::Stack::Apply",
        ]

    def test_standard_method_name_is_not_pinned_by_name_alone(self, tmp_path):
        """On a receiver of unknown type a name that a type of the standard
        library also has as a method is not the project's one definition
        of it; any other name is."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct{}

                func (j *Job) Name() string { return "" }

                func (j *Job) Enqueue() {}

                func describe(jobs []interface{}) {
                \tjob := jobs[0]
                \tjob.Name()
                \tjob.Enqueue()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::describe") == [
            "FILE:app.go::Job::Enqueue",
            "REF:job.Name",
        ]

    def test_fields_are_read_from_the_struct_in_any_file(self, tmp_path):
        """A chain of fields is followed through the structs that declare
        them, in whatever file of the package, to a method, to a type of
        the standard library, or to a field that holds a function."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "conn.go": """
                package app

                import (
                \t"bytes"
                \t"net"
                )

                type Config struct {
                \tname string
                \tpick func() *Config
                }

                func (c *Config) Reset() {}

                type half struct{}

                func (h *half) Reset() {}

                type Conn struct {
                \tconn    net.Conn
                \tconfig  *Config
                \tin, out half
                \tlog     bytes.Buffer
                }

                func (c *Conn) Close() error { return nil }

                func (c *Conn) Reset() {}
                """,
                "state.go": """
                package app

                type state struct {
                \tc *Conn
                }

                func (s *state) run() {
                \ts.c.config.Reset()
                \ts.c.in.Reset()
                \ts.c.conn.Close()
                \ts.c.log.Reset()
                \ts.c.config.pick()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:state.go::state::run") == [
            "FILE:conn.go::Config::Reset",
            "FILE:conn.go::half::Reset",
        ]

    def test_embedded_type_gives_its_methods_and_its_field(self, tmp_path):
        """A struct has the methods of what it embeds and a field of that
        name: a project type's are the project's, `net.Conn`'s are not."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "net"

                type Conn struct{}

                func (c *Conn) Write(b []byte) {}

                func (c *Conn) Flush() {}

                type tokens struct{}

                func (t *tokens) Trim() {}

                type other struct{}

                func (o *other) Trim() {}

                func (o *other) Flush() {}

                type parser struct {
                \t*tokens
                \tConn
                }

                type wrapped struct {
                \tnet.Conn
                }

                func (p *parser) run() {
                \tp.Trim()
                \tp.tokens.Trim()
                \tp.Flush()
                }

                func (w *wrapped) run(b []byte) {
                \tw.Write(b)
                \tw.Conn.Write(b)
                \tw.SetDeadline()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::parser::run") == [
            "FILE:app.go::Conn::Flush",
            "FILE:app.go::tokens::Trim",
            "FILE:app.go::tokens::Trim",
        ]
        assert _calls(builder, "FILE:app.go::wrapped::run") == []

    def test_named_slice_and_element_receivers(self, tmp_path):
        """A named slice or map type holds what its definition says, and
        `value[i].method()` is called on an element of `value`."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct{}

                func (j *Job) Name() string { return "" }

                type Decoy struct{}

                func (d *Decoy) Name() string { return "" }

                type byName []*Job

                func (s byName) Less(i, j int) bool { return s[i].Name() < s[j].Name() }

                type registry map[string]*Decoy

                func first(jobs byName, known registry, grid [][]*Job) {
                \tfor _, job := range jobs {
                \t\tjob.Name()
                \t}
                \tknown["a"].Name()
                \tgrid[0][1].Name()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::byName::Less") == [
            "FILE:app.go::Job::Name",
            "FILE:app.go::Job::Name",
        ]
        assert _calls(builder, "FILE:app.go::first") == [
            "FILE:app.go::Decoy::Name",
            "FILE:app.go::Job::Name",
            "FILE:app.go::Job::Name",
        ]

    def test_range_over_a_call_a_map_and_a_channel(self, tmp_path):
        """The range variables are the key and the element of what a call
        returns, of a map and of a standard slice, and what a channel
        carries."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "os"

                type Job struct{}

                func (j *Job) Name() string { return "" }

                type Key struct{}

                func (k Key) Name() string { return "" }

                func pending() []*Job {
                \treturn nil
                }

                func index() (map[Key]*Job, error) {
                \treturn nil, nil
                }

                func all(queue chan *Job, files []os.DirEntry) {
                \tfor _, job := range pending() {
                \t\tjob.Name()
                \t}
                \tjobs, _ := index()
                \tfor key, job := range jobs {
                \t\tkey.Name()
                \t\tjob.Name()
                \t}
                \tfor job := range queue {
                \t\tjob.Name()
                \t}
                \tfor _, file := range files {
                \t\tfile.Name()
                \t}
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::all") == [
            "FILE:app.go::Job::Name",
            "FILE:app.go::Job::Name",
            "FILE:app.go::Job::Name",
            "FILE:app.go::Key::Name",
            "FILE:app.go::index",
            "FILE:app.go::pending",
        ]

    def test_type_switch_types_its_variable_in_each_clause(self, tmp_path):
        """`switch v := x.(type)` gives `v` the type a clause names, and in
        a clause of several types or none that of `x`."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "fmt"

                type Job struct{}

                func (j *Job) String() string { return "" }

                type Step struct{}

                func (s Step) String() string { return "" }

                type Decoy struct{}

                func (d *Decoy) String() string { return "" }

                func describe(in interface{}, fallback *Decoy) {
                \tswitch v := in.(type) {
                \tcase *Job:
                \t\tv.String()
                \tcase Step:
                \t\tv.String()
                \tcase fmt.Stringer:
                \t\tv.String()
                \t}
                \tswitch v := fallback.(type) {
                \tcase nil, *Decoy:
                \t\tv.String()
                \t}
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::describe") == [
            "FILE:app.go::Decoy::String",
            "FILE:app.go::Job::String",
            "FILE:app.go::Step::String",
        ]

    def test_variable_bound_to_a_part_of_a_typed_value(self, tmp_path):
        """`x := xs[i]`, `x, ok := m[k]`, `x := <-ch`, `x := xs[1:]`, `x :=
        *p` and the statement a `for` starts with give `x` the type the
        value's own says."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct {
                \tnext *Job
                }

                func (j *Job) Run() {}

                type Decoy struct{}

                func (d *Decoy) Run() {}

                func each(jobs []*Job, known map[string]*Job, queue chan *Job, p **Job) {
                \tfirst := jobs[0]
                \tfirst.Run()
                \tnamed, ok := known["a"]
                \tnamed.Run()
                \tsent := <-queue
                \tsent.Run()
                \trest := jobs[1:]
                \trest[0].Run()
                \tpointed := *p
                \tpointed.Run()
                \tfor job := first; job != nil; job = job.next {
                \t\tjob.Run()
                \t}
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::each") == ["FILE:app.go::Job::Run"] * 6

    def test_package_level_variable_of_another_file(self, tmp_path):
        """A package-level variable has its declared type in every file of
        the package; a local variable of its name and no known type has
        not, and a call of one that holds a function calls that value."""
        builder = _index(
            tmp_path,
            {
                "vars.go": """
                package app

                type Command struct{}

                func (c *Command) Name() string { return "" }

                type Decoy struct{}

                func (d *Decoy) Name() string { return "" }

                var rootCmd = &Command{}

                var (
                \tspare *Decoy
                )

                var hook func(*Command)
                """,
                "use.go": """
                package app

                func use() {
                \trootCmd.Name()
                \tspare.Name()
                \thook(rootCmd)
                }

                func local(items []interface{}) {
                \trootCmd := items[0]
                \trootCmd.Name()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:use.go::use") == [
            "FILE:vars.go::Command::Name",
            "FILE:vars.go::Decoy::Name",
        ]
        assert _calls(builder, "FILE:use.go::local") == ["REF:rootCmd.Name"]

    def test_receiver_made_with_new_and_method_named_as_a_built_in(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Job struct{}

                func (j *Job) close() {}

                type Decoy struct{}

                func (d *Decoy) close() {}

                func run(ch chan int) {
                \tnew(Job).close()
                \tclose(ch)
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::run") == ["FILE:app.go::Job::close"]

    def test_call_of_a_function_literal_returns_what_it_declares(self, tmp_path):
        """A variable that holds a function literal is no definition, but a
        call of it returns the results the literal declares."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Command struct{}

                func (c *Command) Execute() error { return nil }

                type Decoy struct{}

                func (d *Decoy) Execute() error { return nil }

                func (d *Decoy) Error() string { return "" }

                func run() string {
                \tgetCmd := func() *Command {
                \t\treturn &Command{}
                \t}
                \tc := getCmd()
                \terr := c.Execute()
                \treturn err.Error()
                }

                func both() {
                \tpair := func(name string) (*Decoy, *Command) { return nil, nil }
                \tbuild := pair
                \t_, c := build("a")
                \tc.Execute()
                }

                func untyped() {
                \tgetCmd := func() {}
                \tgetCmd()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::run") == ["FILE:app.go::Command::Execute"]
        assert _calls(builder, "FILE:app.go::both") == ["FILE:app.go::Command::Execute"]
        assert _calls(builder, "FILE:app.go::untyped") == []

    def test_call_on_an_interface_is_a_call_of_its_method(self, tmp_path):
        """A method an interface declares is a node of the interface, and
        a call on a receiver of that type is a call of it, whichever types
        implement it; what it returns is what it declares. A method the
        interface has from an embedded one is that one's."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "app.go": """
                package app

                import "io"

                type Report struct{}

                func (r *Report) Send() {}

                type Decoy struct{}

                func (d *Decoy) Send() {}

                // Fetcher fetches reports.
                type Fetcher interface {
                \t// Fetch returns the next report.
                \tFetch(name string) (*Report, error)
                \tclose()
                }

                type Source struct{}

                func (s *Source) Fetch(name string) (*Report, error) { return nil, nil }

                func (s *Source) close() {}

                type Mirror struct{}

                func (m *Mirror) Fetch(name string) (*Report, error) { return nil, nil }

                func (m *Mirror) close() {}

                type Stream interface {
                \tFetcher
                \tio.Reader
                }

                func (r *Report) Read(p []byte) (int, error) { return 0, nil }

                func (d *Decoy) Read(p []byte) (int, error) { return 0, nil }

                func drain(fetcher Fetcher, stream Stream, source *Source) string {
                \treport, err := fetcher.Fetch("a")
                \treport.Send()
                \tfetcher.close()
                \tstream.Fetch("b")
                \tstream.Read(nil)
                \tsource.Fetch("c")
                \treturn err.Error()
                }
                """,
            },
        )
        methods = {
            n["id"]: n["metadata"]["return_type"]
            for n in builder.nodes
            if n["id"].startswith("FILE:app.go::Fetcher::")
        }
        assert methods == {
            "FILE:app.go::Fetcher::Fetch": "(*Report, error)",
            "FILE:app.go::Fetcher::close": "",
        }
        assert _calls(builder, "FILE:app.go::Fetcher") == []
        assert _calls(builder, "FILE:app.go::drain") == [
            "FILE:app.go::Fetcher::Fetch",
            "FILE:app.go::Fetcher::Fetch",
            "FILE:app.go::Fetcher::close",
            "FILE:app.go::Report::Send",
            "FILE:app.go::Source::Fetch",
        ]

    def test_variable_bound_to_a_variable_of_another_package(self, tmp_path):
        """`out := os.Stdout` is from outside as `os` is; a variable of a
        project package has the type it is declared with there."""
        builder = _index(
            tmp_path,
            {
                "go.mod": "module example.com/app\n",
                "tool/tool.go": """
                package tool

                type Log struct{}

                func (l *Log) WriteString(s string) {}

                type Decoy struct{}

                func (d *Decoy) WriteString(s string) {}

                var Default = &Log{}
                """,
                "app.go": """
                package app

                import (
                \t"os"

                \t"example.com/app/tool"
                )

                func report() {
                \tout := os.Stdout
                \tout.WriteString("a")
                \tlog := tool.Default
                \tlog.WriteString("b")
                }
                """,
            },
        )
        assert _calls(builder, "FILE:app.go::report") == [
            "FILE:tool/tool.go::Log::WriteString"
        ]

    def test_call_of_a_function_valued_field_returns_what_it_declares(self, tmp_path):
        """A struct field that holds a function is no definition, but a
        call of it returns the results its type declares."""
        builder = _index(
            tmp_path,
            {
                "app.go": """
                package app

                type Agreement struct{}

                func (a *Agreement) Process() {}

                type Decoy struct{}

                func (d *Decoy) Process() {}

                type suite struct {
                \tka         func(version uint16) *Agreement // the key agreement
                \tpair, swap func() (*Decoy, *Agreement)
                \thook       func()
                }

                type state struct {
                \tsuite *suite
                }

                func (s *state) run() {
                \tagreement := s.suite.ka(1)
                \tagreement.Process()
                \t_, second := s.suite.swap()
                \tsecond.Process()
                \ts.suite.hook()
                }
                """
            },
        )
        assert _calls(builder, "FILE:app.go::state::run") == [
            "FILE:app.go::Agreement::Process",
            "FILE:app.go::Agreement::Process",
        ]

    def test_grouped_var_declares_every_name(self, tmp_path):
        """`var a, b *T` gives each name the type, in a function, at
        package level and in a `var (...)` block, for every file of the
        package."""
        builder = _index(
            tmp_path,
            {
                "vars.go": """
                package app

                type Job struct{}

                func (j *Job) Run() {}

                type Decoy struct{}

                func (d *Decoy) Run() {}

                var first, second *Job

                var (
                \tthird, fourth *Decoy
                )

                func local(items []interface{}) {
                \tvar a, b *Job = nil, nil
                \ta.Run()
                \tb.Run()
                \tsecond.Run()
                \tfourth.Run()
                }
                """,
                "use.go": """
                package app

                func use() {
                \tfirst.Run()
                \tsecond.Run()
                \tthird.Run()
                \tfourth.Run()
                }
                """,
            },
        )
        assert _calls(builder, "FILE:vars.go::local") == [
            "FILE:vars.go::Decoy::Run",
            "FILE:vars.go::Job::Run",
            "FILE:vars.go::Job::Run",
            "FILE:vars.go::Job::Run",
        ]
        assert _calls(builder, "FILE:use.go::use") == [
            "FILE:vars.go::Decoy::Run",
            "FILE:vars.go::Decoy::Run",
            "FILE:vars.go::Job::Run",
            "FILE:vars.go::Job::Run",
        ]

    def test_platform_files_leave_the_one_the_build_compiles(self, tmp_path):
        """`impl_linux.go` holds `Impl` in a build `call_linux.go` is of
        and `impl_windows.go` does not; nothing decides a plain caller's."""
        builder = _index(
            tmp_path,
            {
                "pkg/impl_linux.go": """
                package pkg
                func Impl() {}
                """,
                "pkg/impl_windows.go": """
                package pkg
                func Impl() {}
                """,
                "pkg/call_linux.go": """
                package pkg
                func Call() { Impl() }
                """,
                "pkg/plain.go": """
                package pkg
                func Plain() { Impl() }
                """,
            },
        )
        assert _calls(builder, "FILE:pkg/call_linux.go::Call") == [
            "FILE:pkg/impl_linux.go::Impl"
        ]
        assert _calls(builder, "FILE:pkg/plain.go::Plain") == ["REF:Impl"]

    def test_a_test_file_definition_is_no_call_of_plain_code(self, tmp_path):
        """A name only a `_test.go` file defines stays unresolved for a
        caller that is not a test file of the same package."""
        builder = _index(
            tmp_path,
            {
                "pkg/helper_test.go": """
                package pkg
                func helper() {}
                func exercise() { helper() }
                """,
                "pkg/plain.go": """
                package pkg
                func Plain() { helper() }
                """,
            },
        )
        assert _calls(builder, "FILE:pkg/helper_test.go::exercise") == [
            "FILE:pkg/helper_test.go::helper"
        ]
        assert _calls(builder, "FILE:pkg/plain.go::Plain") == ["REF:helper"]
        (edge,) = (
            e
            for e in builder.edges
            if e["source"] == "FILE:pkg/plain.go::Plain" and e["relation"] == "CALLS"
        )
        assert edge["metadata"]["candidates"] == ["FILE:pkg/helper_test.go::helper"]

    def test_a_go_build_tag_decides_between_definitions(self, tmp_path):
        """`//go:build linux` on the caller's file rules a `windows` build
        constraint on a definition's out."""
        builder = _index(
            tmp_path,
            {
                "pkg/on_linux.go": """
                //go:build linux
                package pkg
                func Impl() {}
                """,
                "pkg/on_windows.go": """
                //go:build windows
                package pkg
                func Impl() {}
                """,
                "pkg/call.go": """
                //go:build linux
                package pkg
                func Call() { Impl() }
                """,
            },
        )
        assert _calls(builder, "FILE:pkg/call.go::Call") == [
            "FILE:pkg/on_linux.go::Impl"
        ]


class TestOwnersSourceCannotName:
    """Members of a block that is for no type source can name are filed
    under an owner no source can name either, so no lookup of a type, a base
    or a qualified call lands on it."""

    @pytest.mark.parametrize(
        ("path", "source", "named"),
        [
            (
                "a.rs",
                """
                pub struct Core;
                pub trait Mark {
                    fn mark(&self);
                }
                impl Mark for (u8, u8) {
                    fn mark(&self) {}
                }
                impl<T: Mark> Mark for &T {
                    fn mark(&self) {}
                }
                impl<T> Mark for [T] {
                    fn mark(&self) {}
                }
                impl<T> Mark for T {
                    fn mark(&self) {}
                }
                impl Mark for Vec<u8> {
                    fn mark(&self) {}
                }
                impl Mark for made!() {
                    fn mark(&self) {}
                }
                macro_rules! marked {
                    ($name:ident) => {
                        impl Mark for $name {
                            fn mark(&self) {}
                        }
                    };
                }
                """,
                {"Core": [], "Mark": ["mark"], "Vec": ["mark"]},
            ),
            (
                "A.java",
                """
                class A {
                    void run() {}
                    Runnable make() {
                        return new Runnable() {
                            public void run() {}
                            void stop() {}
                        };
                    }
                }
                enum Mode {
                    ON {
                        void apply() {}
                        void more() {}
                    },
                    @Deprecated HALF(1) {
                        void half() {}
                    },
                    OFF
                    {
                        void off() {}
                    };
                    void apply() {}
                }
                """,
                {"A": ["make", "run"], "Mode": ["apply"]},
            ),
            (
                "a.php",
                """
                <?php
                function make() {
                    return new class implements Countable {
                        public function count(): int { return 0; }
                    };
                }
                """,
                {},
            ),
            (
                "a.dart",
                """
                extension on String {
                  int twice() => length * 2;
                }
                """,
                {},
            ),
            (
                "a.rb",
                """
                handler = Class.new(StandardError) do
                  def message; "x"; end
                end
                """,
                {},
            ),
        ],
    )
    def test_only_a_type_source_names_has_a_name_source_can_write(
        self, tmp_path, path, source, named
    ):
        builder = _index(tmp_path, {path: source})
        written = {
            node["id"]: []
            for node in builder.nodes
            if node["type"] == "Class" and node["metadata"]["name"].isidentifier()
        }
        for node in builder.nodes:
            owner, _, name = node["id"].rpartition("::")
            if owner in written and node["type"] in ("Method", "Function"):
                written[owner].append(name)
        assert {
            owner.rpartition("::")[2]: sorted(names) for owner, names in written.items()
        } == named

    def test_a_returned_object_owns_the_functions_written_in_it(self, tmp_path):
        """A factory's own members are the functions it declares; a function
        written in the object it returns is a member of that object, which
        no source can name. A name written in such a function is read in the
        factory's scope — its own members are reached through `this`."""
        builder = _index(
            tmp_path,
            {
                "a.js": """
                function wrap(name) {
                    function inner() {}
                    return {
                        member() {
                            inner();
                            this.sibling();
                            sibling();
                        },
                        sibling() {
                            inner();
                        },
                    };
                }
                """,
            },
        )
        assert {
            node["id"].rpartition("::")[2]
            for node in builder.nodes
            if node["id"].rpartition("::")[0] == "FILE:a.js::wrap"
        } == {"inner", "object@4"}
        assert _calls(builder, "FILE:a.js::wrap::object@4::member") == [
            "FILE:a.js::wrap::inner",
            "FILE:a.js::wrap::object@4::sibling",
        ]
        assert _calls(builder, "FILE:a.js::wrap::object@4::sibling") == [
            "FILE:a.js::wrap::inner"
        ]

    def test_an_accessor_has_a_name_no_method_can_have(self, tmp_path):
        """An accessor is filed as source writes it, `get size`, so a method
        whose name spells that with an underscore is a definition of its
        own."""
        builder = _index(
            tmp_path,
            {
                "a.ts": """
                export class Box {
                    get size() {
                        return 1;
                    }
                    set size(count) {}
                    get_size() {
                        return 2;
                    }
                }

                export function make() {
                    return {
                        get size() {
                            return 1;
                        },
                        get_size() {
                            return 2;
                        },
                    };
                }

                export function use(box: Box) {
                    box.get_size();
                    make().get_size();
                }
                """,
                "lib/a.dart": """
                class Box {
                  int get size => 1;
                  set size(int count) {}
                  int set_size() => 2;
                }
                """,
            },
        )
        members = {
            node["id"].partition("::")[2]
            for node in builder.nodes
            if node["type"] == "Method"
        }
        assert members == {
            "Box::get size",
            "Box::set size",
            "Box::get_size",
            "make::object@13::get size",
            "make::object@13::get_size",
            "Box::size",
            "Box::set_size",
        }
        assert _calls(builder, "FILE:a.ts::use") == [
            "FILE:a.ts::Box::get_size",
            "FILE:a.ts::make",
            "FILE:a.ts::make::object@13::get_size",
        ]


class TestJavaCallResolution:
    def test_lambda_parameter_reads_the_interfaces_abstract_method(self, tmp_path):
        """`x -> x.tally()` given for a `Mapper<Item, String>` parameter:
        `x` is what `Mapper.apply` takes, `Item`."""
        builder = _index(
            tmp_path,
            {
                "Mapper.java": """
                public interface Mapper<T, R> {
                    R apply(T input);
                    default int size() { return 0; }
                }
                """,
                "Item.java": """
                public class Item {
                    public String tally() { return "x"; }
                }
                """,
                "Use.java": """
                public class Use {
                    public String map(Mapper<Item, String> m) { return m.apply(new Item()); }
                    public void go() {
                        map(x -> x.tally());
                    }
                }
                """,
            },
        )
        assert "FILE:Item.java::Item::tally" in _calls(
            builder, "FILE:Use.java::Use::go"
        )

    def test_lambda_parameter_of_an_assignment_reads_the_type_written(self, tmp_path):
        """`Mapper<Item, String> f = y -> y.tally()`: `y` is the
        `apply` parameter's `Item` substitute."""
        builder = _index(
            tmp_path,
            {
                "Mapper.java": """
                public interface Mapper<T, R> {
                    R apply(T input);
                }
                """,
                "Item.java": """
                public class Item {
                    public String tally() { return "x"; }
                }
                """,
                "Other.java": """
                public class Other {
                    public String tally() { return "y"; }
                }
                """,
                "Use.java": """
                public class Use {
                    public void go() {
                        Mapper<Item, String> f = y -> y.tally();
                    }
                }
                """,
            },
        )
        assert "FILE:Item.java::Item::tally" in _calls(
            builder, "FILE:Use.java::Use::go"
        )

    def test_lambda_parameter_of_a_cast_reads_the_type_written(self, tmp_path):
        """`(Mapper<Item, String>) (z -> z.tally())` types `z` from the
        cast's functional interface."""
        builder = _index(
            tmp_path,
            {
                "Mapper.java": """
                public interface Mapper<T, R> {
                    R apply(T input);
                }
                """,
                "Item.java": """
                public class Item {
                    public String tally() { return "x"; }
                }
                """,
                "Other.java": """
                public class Other {
                    public String tally() { return "y"; }
                }
                """,
                "Use.java": """
                public class Use {
                    public void go() {
                        Object o = (Mapper<Item, String>) (z -> z.tally());
                    }
                }
                """,
            },
        )
        assert "FILE:Item.java::Item::tally" in _calls(
            builder, "FILE:Use.java::Use::go"
        )

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

    def test_method_of_a_returned_library_type_is_external(self, tmp_path):
        """What a project method declares it returns as `Set<String>` calls
        the library's `iterator`, not a project method of that name."""
        builder = _index(
            tmp_path,
            {
                "Registry.java": """
                import java.util.Set;

                public class Registry {
                    public Set<String> names() { return null; }
                }
                """,
                "Cursor.java": """
                public class Cursor {
                    public Object iterator() { return null; }
                }
                """,
                "Report.java": """
                public class Report {
                    public void print(Registry registry) {
                        registry.names().iterator();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Report.java::Report::print") == [
            "FILE:Registry.java::Registry::names"
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

    def test_platform_class_named_in_full_is_external(self, tmp_path):
        """`java.util.UUID.fromString` is the platform's, though a project
        class defines a `fromString` too."""
        builder = _index(
            tmp_path,
            {
                "app/Ids.java": """
                package app;

                class Token {
                    static Token fromString(String text) {
                        return null;
                    }
                }

                class Ids {
                    Object read(String text) {
                        return java.util.UUID.fromString(text);
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:app/Ids.java::Ids::read") == []

    def test_member_inherited_from_a_project_superclass(self, tmp_path):
        """A method a typed receiver's class does not declare is its
        superclass's; one of `java.lang.Object` is not the project's."""
        builder = _index(
            tmp_path,
            {
                "app/Shapes.java": """
                package app;

                class Element {
                    String describe() {
                        return "";
                    }
                }

                class Array extends Element implements Comparable<Array> {
                }

                class Other {
                    String describe() {
                        return "";
                    }

                    public String toString() {
                        return "";
                    }
                }

                class Use {
                    String both(Array array) {
                        return array.describe() + array.toString();
                    }
                }
                """
            },
        )
        inherits = {
            (e["source"].rpartition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        }
        assert ("Array", "FILE:app/Shapes.java::Element") in inherits
        assert _calls(builder, "FILE:app/Shapes.java::Use::both") == [
            "FILE:app/Shapes.java::Element::describe"
        ]

    def test_receiver_typed_by_a_cast_a_literal_or_a_static_field(self, tmp_path):
        """`((T) x).m()` is called on a T, `"s".m()` on a String, and
        `T.FIELD.m()` on what the field or enum constant is declared as."""
        builder = _index(
            tmp_path,
            {
                "app/Rules.java": """
                package app;

                enum Policy {
                    STRICT,
                    LOOSE;

                    int length() {
                        return 0;
                    }
                }

                class Holder {
                    static final Holder INSTANCE = new Holder();

                    int length() {
                        return 0;
                    }
                }

                class Use {
                    int all(Object value) {
                        return ((Holder) value).length()
                            + "text".length()
                            + Policy.STRICT.length()
                            + Holder.INSTANCE.length();
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:app/Rules.java::Use::all") == [
            "FILE:app/Rules.java::Holder::length",
            "FILE:app/Rules.java::Holder::length",
            "FILE:app/Rules.java::Policy::length",
        ]

    def test_declarations_that_are_not_calls(self, tmp_path):
        """A method declared without a body over two lines, an annotation
        element with a default, and an enum constant with arguments."""
        builder = _index(
            tmp_path,
            {
                "app/Kinds.java": """
                package app;

                interface Source {
                    void readInto(Object target, int index)
                        throws java.io.IOException;
                }

                @interface Expose {
                    boolean readInto() default true;
                }

                enum Color {
                    RED("red"),
                    BLUE("blue");

                    Color(String name) {
                    }
                }
                """
            },
        )
        declared = {
            n["id"].partition("::")[2]: n["type"]
            for n in builder.nodes
            if n["type"] in ("Method", "Constant")
        }
        assert declared == {
            "Source::readInto": "Method",
            "Expose::readInto": "Method",
            "Color::RED": "Constant",
            "Color::BLUE": "Constant",
            "Color::<init>": "Method",
        }
        calls = [e for e in builder.edges if e["relation"].startswith("CALLS")]
        assert calls == []

    def test_method_of_a_standard_type_is_typed_by_its_type_arguments(self, tmp_path):
        """A call on a standard value is the library's, and what it returns
        is typed by the type arguments of the value for the next call."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public void fit() {}
                    public String trim() { return ""; }
                    public int size() { return 0; }
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void fit() {}
                }
                """,
                "Client.java": """
                import java.util.ArrayList;
                import java.util.List;
                import java.util.Map;
                import java.util.Optional;

                public class Client {
                    void element(List<Part> parts) {
                        parts.get(0).fit();
                    }

                    void entry(Map<String, Part> byName) {
                        byName.entrySet().iterator().next().getValue().fit();
                    }

                    void held(Optional<Part> found) {
                        found.get().fit();
                    }

                    void made() {
                        new ArrayList<Part>().get(0).fit();
                        List<Part> parts = new ArrayList<>();
                        parts.get(0).fit();
                        var more = new ArrayList<Part>();
                        more.get(0).fit();
                    }

                    void library(int count, List<Part> parts) {
                        String.valueOf(count).trim();
                        parts.size();
                        new StringBuilder().append(count).toString().trim();
                    }
                }
                """,
            },
        )
        fit = "FILE:Part.java::Part::fit"
        assert _calls(builder, "FILE:Client.java::Client::element") == [fit]
        assert _calls(builder, "FILE:Client.java::Client::entry") == [fit]
        assert _calls(builder, "FILE:Client.java::Client::held") == [fit]
        assert _calls(builder, "FILE:Client.java::Client::made") == [fit, fit, fit]
        assert _calls(builder, "FILE:Client.java::Client::library") == []

    def test_standard_method_name_on_an_unknown_receiver_is_not_pinned(self, tmp_path):
        """`isInterrupted` is a method of `java.lang.Thread`, so the one
        project method of that name is not the only one that fits."""
        builder = _index(
            tmp_path,
            {
                "Monitor.java": """
                public class Monitor {
                    public boolean isInterrupted() { return false; }
                    public void report() {}
                }
                """,
                "Client.java": """
                public class Client {
                    void run() {
                        worker.isInterrupted();
                        worker.report();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Monitor.java::Monitor::report",
            "REF:worker.isInterrupted",
        ]

    def test_member_a_class_has_from_the_standard_library(self, tmp_path):
        """Every class has the methods of `Object`, an enum those of `Enum`,
        and a class those of the standard class it extends; a bare call of
        one is the library's, and typed by what the library returns."""
        builder = _index(
            tmp_path,
            {
                "app/Kinds.java": """
                package app;

                import java.util.ArrayList;

                enum Shade {
                    DARK,
                    LIGHT;
                }

                class Decoy {
                    public String name() { return ""; }
                    public String getSimpleName() { return ""; }
                    public void clear() {}
                }

                class Queue extends ArrayList<String> {
                    void drain() {
                        clear();
                    }
                }

                class Use {
                    String all(Shade shade) {
                        return shade.name() + getClass().getSimpleName();
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:app/Kinds.java::Queue::drain") == []
        assert _calls(builder, "FILE:app/Kinds.java::Use::all") == []

    def test_local_is_declared_in_its_method_only(self, tmp_path):
        """A parameter of one method does not type the same name in the
        next method, where it is a field; nor does a signature that goes on
        over several lines keep its body from being the method's."""
        builder = _index(
            tmp_path,
            {
                "Alpha.java": """
                public class Alpha {
                    public void run() {}
                }
                """,
                "Beta.java": """
                public class Beta {
                    public void run() {}
                }
                """,
                "Client.java": """
                public class Client {
                    Beta item;

                    void first(Alpha item) {
                        item.run();
                    }

                    void second() {
                        item.run();
                    }

                    void wrapped(
                            Alpha first,
                            Beta item) {
                        item.run();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::first") == [
            "FILE:Alpha.java::Alpha::run"
        ]
        assert _calls(builder, "FILE:Client.java::Client::second") == [
            "FILE:Beta.java::Beta::run"
        ]
        assert _calls(builder, "FILE:Client.java::Client::wrapped") == [
            "FILE:Beta.java::Beta::run"
        ]

    def test_field_is_read_through_the_types_that_declare_it(self, tmp_path):
        """A path of fields is followed through their declared types, from
        a local, from `this` or from a field the class inherits; a field of
        a standard type is the library's."""
        builder = _index(
            tmp_path,
            {
                "app/Parts.java": """
                package app;

                import java.util.List;

                class Part {
                    void fit() {}
                }

                class Decoy {
                    void fit() {}
                    void add(String name) {}
                }

                class Holder {
                    Part part = new Part();
                    List<String> names;
                }

                class Base {
                    protected Holder holder;
                }
                """,
                "app/Use.java": """
                package app;

                class Use extends Base {
                    void local(Holder other) {
                        other.part.fit();
                        other.names.add("x");
                    }

                    void inherited() {
                        holder.part.fit();
                        this.holder.part.fit();
                    }
                }
                """,
            },
        )
        fit = "FILE:app/Parts.java::Part::fit"
        assert _calls(builder, "FILE:app/Use.java::Use::local") == [fit]
        assert _calls(builder, "FILE:app/Use.java::Use::inherited") == [fit, fit]

    def test_super_and_qualified_this(self, tmp_path):
        """`super.m()` is a method of the supertype, of the anonymous
        class's where the call is in one, and `Outer.this.m()` one of
        `Outer`."""
        builder = _index(
            tmp_path,
            {
                "app/Shapes.java": """
                package app;

                class Shape {
                    void draw() {}
                }

                class Other {
                    void draw() {}
                }

                class Circle extends Shape {
                    void draw() {
                        super.draw();
                    }

                    class Edge extends Other {
                        void outer() {
                            Circle.this.draw();
                        }
                    }
                }

                class Use {
                    void draw() {}

                    Object anonymous() {
                        return new Shape() {
                            void draw() {
                                super.draw();
                            }
                        };
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:app/Shapes.java::Circle::draw") == [
            "FILE:app/Shapes.java::Shape::draw"
        ]
        assert _calls(builder, "FILE:app/Shapes.java::Circle::Edge::outer") == [
            "FILE:app/Shapes.java::Circle::draw"
        ]
        body = "FILE:app/Shapes.java::Use::anonymous::class@28"
        assert _calls(builder, f"{body}::draw") == ["FILE:app/Shapes.java::Shape::draw"]
        assert _calls(builder, "FILE:app/Shapes.java::Use::draw") == []

    def test_body_of_an_enum_constant_is_a_subclass_of_its_enum(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "A.java": """
                class A {
                    enum State {
                        ON {
                            boolean live() { return shared(); }
                        },
                        OFF;
                        boolean live() { return false; }
                        boolean shared() { return true; }
                    }
                }
                """,
                "B.java": """
                class B {
                    enum State { UP, DOWN }
                }
                """,
            },
        )
        body = "FILE:A.java::A::State::ON::class@4"
        bases = [
            edge["target"]
            for edge in builder.edges
            if edge["source"] == body and edge["relation"] == "INHERITS"
        ]
        assert bases == ["FILE:A.java::A::State"]
        assert _calls(builder, f"{body}::live") == ["FILE:A.java::A::State::shared"]
        assert _calls(builder, "FILE:A.java::A::State::live") == []

    def test_type_name_is_read_in_its_package(self, tmp_path):
        """A simple type name is the type of the file's own package, not a
        class nested in an unrelated one that sits nearer in the tree."""
        builder = _index(
            tmp_path,
            {
                "main/app/arch/Processor.java": """
                package app.arch;

                public class Processor {
                    public boolean is64Bit() { return true; }
                    public static Processor detect() { return null; }
                }
                """,
                "test/app/other/OtherTest.java": """
                package app.other;

                class OtherTest {
                    static class Processor {
                        boolean is64Bit() { return false; }
                        static Processor detect() { return null; }
                    }
                }
                """,
                "test/app/arch/ProcessorTest.java": """
                package app.arch;

                class ProcessorTest {
                    void typed(Processor processor) {
                        processor.is64Bit();
                    }

                    void named() {
                        Processor.detect();
                    }
                }
                """,
            },
        )
        source = "FILE:test/app/arch/ProcessorTest.java::ProcessorTest"
        target = "FILE:main/app/arch/Processor.java::Processor"
        assert _calls(builder, f"{source}::typed") == [f"{target}::is64Bit"]
        assert _calls(builder, f"{source}::named") == [f"{target}::detect"]

    def test_type_arguments_of_a_call_and_of_a_method(self, tmp_path):
        """`Type.<A>name()` is a call of `name`, and a method's own type
        parameters are no part of what it returns."""
        builder = _index(
            tmp_path,
            {
                "Maker.java": """
                public class Maker {
                    public static <T> Maker create() { return null; }
                    public static <A extends Comparable<A>> Check<A> is(A a) {
                        return null;
                    }
                    public Maker named(String name) { return this; }
                }
                """,
                "Check.java": """
                public class Check<A> {
                    public boolean holds() { return true; }
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void create() {}
                    public void named(String name) {}
                    public boolean holds() { return true; }
                }
                """,
                "Client.java": """
                public class Client {
                    void run() {
                        Maker.<String>create().named("x");
                        Maker.is("x").holds();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Check.java::Check::holds",
            "FILE:Maker.java::Maker::create",
            "FILE:Maker.java::Maker::is",
            "FILE:Maker.java::Maker::named",
        ]

    def test_element_of_an_array_and_cast_of_a_call(self, tmp_path):
        """`items[0].m()` is called on what the array holds, the array
        itself has the methods of `Object`, and `((T) make()).m()` is
        called on a T."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public void fit() {}
                    public Object clone() { return this; }
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void fit() {}
                }
                """,
                "Client.java": """
                public class Client {
                    Object make() { return null; }

                    void run(Part[] items, int[] sizes) {
                        items[0].fit();
                        items.clone();
                        sizes.clone();
                        ((Part) make()).fit();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Client.java::Client::make",
            "FILE:Part.java::Part::fit",
            "FILE:Part.java::Part::fit",
        ]

    def test_lambda_parameter_is_typed_by_the_method_it_is_passed_to(self, tmp_path):
        """A standard method that takes a functional interface passes the
        lambda what the interface's method takes."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public void fit() {}
                    public boolean ready() { return true; }
                    public Part copy() { return this; }
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void fit() {}
                    public boolean ready() { return true; }
                    public Decoy copy() { return this; }
                }
                """,
                "Client.java": """
                import java.util.List;
                import java.util.Map;

                public class Client {
                    void each(List<Part> parts) {
                        parts.forEach(part -> part.fit());
                    }

                    void stream(List<Part> parts) {
                        parts.stream().filter(part -> part.ready()).forEach(each -> {
                            each.fit();
                        });
                    }

                    void pairs(Map<Part, Part> pairs, Part key) {
                        pairs.forEach((left, right) -> right.fit());
                        pairs.computeIfAbsent(key, missing -> missing.copy());
                    }
                }
                """,
            },
        )
        part = "FILE:Part.java::Part"
        assert _calls(builder, "FILE:Client.java::Client::each") == [f"{part}::fit"]
        assert _calls(builder, "FILE:Client.java::Client::stream") == [
            f"{part}::fit",
            f"{part}::ready",
        ]
        assert _calls(builder, "FILE:Client.java::Client::pairs") == [
            f"{part}::copy",
            f"{part}::fit",
        ]

    def test_lambda_parameter_is_no_field(self, tmp_path):
        """A lambda's parameter is not the field of the same name."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public void fit() {}
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void fit() {}
                }
                """,
                "Client.java": """
                public class Client {
                    Part part;

                    void run(Registry registry) {
                        registry.visit(part -> part.fit());
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == ["REF:part.fit"]

    def test_method_reference_is_a_call(self, tmp_path):
        """`Type::name`, `value::name` and `this::name` call the method,
        `Type::new` the type."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public String getName() { return ""; }
                    public void fit() {}
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public String getName() { return ""; }
                    public void fit() {}
                    public void check(Object part) {}
                }
                """,
                "Client.java": """
                import java.util.Comparator;
                import java.util.List;
                import java.util.function.Supplier;

                public class Client {
                    void check(Part part) {}

                    void run(List<Part> parts, Part first) {
                        parts.sort(Comparator.comparing(Part::getName));
                        parts.forEach(this::check);
                        Runnable fit = first::fit;
                        Supplier<Part> maker = Part::new;
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Client.java::Client::check",
            "FILE:Part.java::Part",
            "FILE:Part.java::Part::fit",
            "FILE:Part.java::Part::getName",
        ]

    def test_pattern_binds_a_name_of_its_type(self, tmp_path):
        """What `instanceof` and a `case` label bind is of the type they
        name."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public void fit() {}
                }
                """,
                "Decoy.java": """
                public class Decoy {
                    public void fit() {}
                }
                """,
                "Client.java": """
                public class Client {
                    void tested(Object value) {
                        if (value instanceof Part part && part != null) {
                            part.fit();
                        }
                    }

                    void switched(Object value) {
                        switch (value) {
                            case Part part -> part.fit();
                            default -> {}
                        }
                    }
                }
                """,
            },
        )
        fit = "FILE:Part.java::Part::fit"
        assert _calls(builder, "FILE:Client.java::Client::tested") == [fit]
        assert _calls(builder, "FILE:Client.java::Client::switched") == [fit]

    def test_overloads_return_the_type_they_agree_on(self, tmp_path):
        """Overloads that return different types leave what a call of
        them returns unknown; a type parameter stands for its bound."""
        builder = _index(
            tmp_path,
            {
                "Part.java": """
                public class Part {
                    public void fit() {}
                }
                """,
                "Other.java": """
                public class Other {
                    public void fit() {}
                }
                """,
                "Store.java": """
                public class Store<T extends Part> {
                    public Part take(int index) { return null; }
                    public Other take(String name) { return null; }
                    public Part pick(int index) { return null; }
                    public void pick(String name) {}
                    public T first() { return null; }
                }
                """,
                "Client.java": """
                public class Client {
                    void run(Store<Part> store) {
                        store.take("x").fit();
                    }

                    void agreed(Store<Part> store) {
                        store.pick(0).fit();
                        store.first().fit();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.java::Client::run") == [
            "FILE:Store.java::Store::take",
            "REF:fit",
        ]
        assert _calls(builder, "FILE:Client.java::Client::agreed") == [
            "FILE:Part.java::Part::fit",
            "FILE:Part.java::Part::fit",
            "FILE:Store.java::Store::first",
            "FILE:Store.java::Store::pick",
        ]

    def test_declaration_without_a_modifier_is_no_call(self, tmp_path):
        """A constructor with its body on one line, a method that returns
        a type with arguments and one whose parameters go on over the next
        lines are declarations, and a member type is made through the
        instance it belongs to."""
        builder = _index(
            tmp_path,
            {
                "app/Tree.java": """
                package app;

                import java.util.List;
                import java.util.Map;

                class Tree {
                    class Node {
                        Node() {}
                    }

                    Tree() {}

                    Map<String, Node> index(List<Node> nodes) {
                        return null;
                    }

                    List<Node> collect(
                            Node first,
                            Node second) {
                        return null;
                    }

                    Node grow(Tree other) {
                        return other.new Node();
                    }
                }
                """
            },
        )
        methods = {
            n["id"].partition("::")[2] for n in builder.nodes if n["type"] == "Method"
        }
        assert methods == {
            "Tree::Node::<init>",
            "Tree::<init>",
            "Tree::index",
            "Tree::collect",
            "Tree::grow",
        }
        calls = [
            e["target"] for e in builder.edges if e["relation"].startswith("CALLS")
        ]
        assert calls == ["FILE:app/Tree.java::Tree::Node"]

    def test_lambda_of_a_method_declared_without_a_modifier(self, tmp_path):
        """`Op r = (w) -> w.spin()`: `w` is the parameter `run` declares."""
        builder = _index(
            tmp_path,
            {
                "Op.java": """
                interface Op {
                    void run(Widget w);
                }
                """,
                "Widget.java": """
                class Widget {
                    void spin() {}
                }
                """,
                "Gadget.java": """
                class Gadget {
                    void spin() {}
                }
                """,
                "Use.java": """
                class Use {
                    void go() {
                        Op r = (w) -> w.spin();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Use.java::Use::go") == [
            "FILE:Widget.java::Widget::spin"
        ]

    def test_method_without_a_modifier_whose_brace_is_on_the_next_line(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "Widget.java": """
                class Widget {
                    String spin() { return ""; }
                    String plain(int a)
                    {
                        return spin();
                    }
                    String wrapped(
                        int a)
                    {
                        return spin();
                    }
                    String noted() // not; a declaration
                    {
                        return spin();
                    }
                }
                """,
            },
        )
        spin = ["FILE:Widget.java::Widget::spin"]
        for name in ("plain", "wrapped", "noted"):
            assert _calls(builder, f"FILE:Widget.java::Widget::{name}") == spin
        assert _calls(builder, "FILE:Widget.java::Widget") == []

    def test_enum_constant_with_a_body_on_its_line(self, tmp_path):
        """`A { void f() {} },` declares the constant, and calls nothing."""
        builder = _index(
            tmp_path,
            {
                "E.java": """
                enum E {
                    A { void f() {} },
                    B;
                    void f() {}
                }
                """,
            },
        )
        assert "FILE:E.java::E::A" in builder.node_registry
        assert _calls(builder, "FILE:E.java::E") == []

    def test_variable_bound_to_a_cast_is_not_what_the_call_returns(self, tmp_path):
        """`var x = (Jar) pack();`: the call on `x` is no call on what
        `pack` returns."""
        builder = _index(
            tmp_path,
            {
                "A.java": """
                class Bag {
                    int get() { return 1; }
                }
                class Jar {
                    int get() { return 2; }
                }
                class U {
                    Object pack() { return new Bag(); }
                    void go() {
                        var x = (Jar) pack();
                        x.get();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:A.java::U::go") == [
            "FILE:A.java::U::pack",
            "REF:x.get",
        ]


class TestPhpCallResolution:
    def test_constructed_type_picks_the_method(self, tmp_path):
        """`new T(...)` is a call of the class T, and types the variable it
        is assigned to and a call made on it in parentheses."""
        builder = _index(
            tmp_path,
            {
                "src/Mailer.php": """
                <?php
                class Mailer {
                    public function send(): void {}
                }
                """,
                "src/Decoy.php": """
                <?php
                class Decoy {
                    public function send(): void {}
                }
                """,
                "src/Client.php": """
                <?php
                class Client {
                    public function run(): void
                    {
                        $mailer = new Mailer('smtp');
                        $mailer->send();
                    }

                    public function once(): void
                    {
                        (new Decoy())->send();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/Client.php::Client::run") == [
            "FILE:src/Mailer.php::Mailer",
            "FILE:src/Mailer.php::Mailer::send",
        ]
        assert _calls(builder, "FILE:src/Client.php::Client::once") == [
            "FILE:src/Decoy.php::Decoy",
            "FILE:src/Decoy.php::Decoy::send",
        ]

    def test_copy_has_the_type_of_what_it_copies(self, tmp_path):
        """A variable assigned another variable, a property or an enum case
        is of its type, and `new T` without an argument list is a call of
        T as `new T()` is."""
        builder = _index(
            tmp_path,
            {
                "src/Types.php": """
                <?php
                enum Level {
                    case Info;

                    public function label(): string { return ''; }
                }
                class Mailer {
                    public function send(): void {}
                }
                class Decoy {
                    public function send(): void {}
                    public function label(): string { return ''; }
                }
                class Client {
                    private Mailer $mailer;

                    public function run(): void
                    {
                        $mailer = $this->mailer;
                        $mailer->send();
                        $other = $mailer;
                        $other->send();
                        $level = Level::Info;
                        $level->label();
                        (new Mailer)->send();
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Types.php::Client::run") == [
            "FILE:src/Types.php::Level::label",
            "FILE:src/Types.php::Mailer",
            "FILE:src/Types.php::Mailer::send",
            "FILE:src/Types.php::Mailer::send",
            "FILE:src/Types.php::Mailer::send",
        ]

    def test_property_types_a_call_through_this(self, tmp_path):
        """`$this->name->m()` is called on what the property is declared
        as, in the class, as a promoted constructor parameter or in a
        superclass; `$this->m()` is the class's own method, though a
        property has that name."""
        builder = _index(
            tmp_path,
            {
                "src/Types.php": """
                <?php
                class Queue {
                    public function push(): void {}
                }
                class Clock {
                    public function push(): void {}
                }
                class Decoy {
                    public function push(): void {}
                }
                abstract class Base {
                    protected Clock $clock;
                }
                class Worker extends Base {
                    private Queue $queue;
                    private bool $push;

                    public function __construct(private readonly ?Decoy $spare)
                    {
                    }

                    public function run(): void
                    {
                        $this->queue->push();
                        $this->spare->push();
                        $this->clock->push();
                        $this->push();
                    }

                    private function push(): void {}
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Types.php::Worker::run") == [
            "FILE:src/Types.php::Clock::push",
            "FILE:src/Types.php::Decoy::push",
            "FILE:src/Types.php::Queue::push",
            "FILE:src/Types.php::Worker::push",
        ]

    def test_parameter_and_return_types_type_a_chain(self, tmp_path):
        """A call on a typed parameter, on what a call is declared to
        return (`static` is the class it was called on), and on a
        variable bound to a call."""
        builder = _index(
            tmp_path,
            {
                "src/Types.php": """
                <?php
                class Maker {
                    public function named(string $name): static { return $this; }
                    public function assemble(): ?Product { return null; }
                    public static function create(): self { return new self(); }
                }
                class Product {
                    public function ship(): void {}
                }
                class Decoy {
                    public function named(): void {}
                    public function assemble(): void {}
                    public function ship(): void {}
                }
                class Client {
                    public function chained(Maker $maker): void
                    {
                        $maker->named('x')->assemble()->ship();
                    }

                    public function continued(): void
                    {
                        Maker::create()
                            ->named('x')
                            ->assemble()
                            ->ship();
                    }

                    public function bound(Maker $maker): void
                    {
                        $product = $maker->assemble();
                        $product->ship();
                    }
                }
                """
            },
        )
        chain = [
            "FILE:src/Types.php::Maker::assemble",
            "FILE:src/Types.php::Maker::named",
            "FILE:src/Types.php::Product::ship",
        ]
        assert _calls(builder, "FILE:src/Types.php::Client::chained") == chain
        assert _calls(builder, "FILE:src/Types.php::Client::continued") == [
            *chain[:1],
            "FILE:src/Types.php::Maker::create",
            *chain[1:],
        ]
        assert _calls(builder, "FILE:src/Types.php::Client::bound") == [
            "FILE:src/Types.php::Maker::assemble",
            "FILE:src/Types.php::Product::ship",
        ]

    def test_parent_call_reaches_the_parent(self, tmp_path):
        """`parent::m()` is the superclass's `m`, not the caller's own;
        `self::m()` and `static::m()` are the class's."""
        builder = _index(
            tmp_path,
            {
                "src/Handlers.php": """
                <?php
                class Base {
                    public function __construct() {}
                    public function reset(): void {}
                }
                class Buffered extends Base {
                    public function __construct()
                    {
                        parent::__construct();
                    }

                    public function reset(): void
                    {
                        parent::reset();
                        self::clear();
                        static::flush();
                    }

                    private static function clear(): void {}
                    protected static function flush(): void {}
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Handlers.php::Buffered::__construct") == [
            "FILE:src/Handlers.php::Base::__construct"
        ]
        assert _calls(builder, "FILE:src/Handlers.php::Buffered::reset") == [
            "FILE:src/Handlers.php::Base::reset",
            "FILE:src/Handlers.php::Buffered::clear",
            "FILE:src/Handlers.php::Buffered::flush",
        ]

    def test_trait_gives_the_class_its_methods(self, tmp_path):
        """A method of a trait the class uses is the class's, ahead of an
        inherited one; a call in the trait to a method it leaves to the
        class using it is pinned to none."""
        builder = _index(
            tmp_path,
            {
                "src/Formats.php": """
                <?php
                trait Formats {
                    public function format(): string { return $this->render(); }
                }
                class Base {
                    public function format(): string { return ''; }
                    public function level(): int { return 0; }
                }
                class Line extends Base {
                    use Formats;

                    public function render(): string { return ''; }

                    public function run(Line $other): void
                    {
                        $this->format();
                        $other->level();
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Formats.php::Line::run") == [
            "FILE:src/Formats.php::Base::level",
            "FILE:src/Formats.php::Formats::format",
        ]
        assert _calls(builder, "FILE:src/Formats.php::Formats::format") == [
            "REF:$this->render"
        ]

    def test_namespace_and_imports_decide_the_class(self, tmp_path):
        """A name is the class its `use` import or its namespace says:
        `\\DateTimeImmutable` is PHP's, not the project's class of that
        name, and `extends \\Lib\\Record` is no project class either."""
        builder = _index(
            tmp_path,
            {
                "src/DateTimeImmutable.php": """
                <?php
                namespace App;

                class DateTimeImmutable extends \\DateTimeImmutable {
                    public function format(string $format): string { return ''; }
                }
                """,
                "src/Record.php": """
                <?php
                namespace App;

                class Record {
                    public function format(string $format): string { return ''; }
                }
                """,
                "src/Other/Record.php": """
                <?php
                namespace App\\Other;

                class Record extends \\Lib\\Record {
                    public function format(string $format): string { return ''; }
                }
                """,
                "src/Report.php": """
                <?php
                namespace App\\Reports;

                use App\\DateTimeImmutable as Stamp;
                use App\\Other;

                class Report {
                    public function run(\\App\\Record $record): void
                    {
                        (new \\DateTimeImmutable())->format('c');
                        (new Stamp())->format('c');
                        (new Other\\Record())->format('c');
                        $record->format('c');
                        (new Record())->format('c');
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/Report.php::Report::run") == [
            "FILE:src/DateTimeImmutable.php::DateTimeImmutable",
            "FILE:src/DateTimeImmutable.php::DateTimeImmutable::format",
            "FILE:src/Other/Record.php::Record",
            "FILE:src/Other/Record.php::Record::format",
            "FILE:src/Record.php::Record::format",
        ]
        inherits = [
            (e["source"].rpartition("/")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        ]
        assert inherits == [
            ("DateTimeImmutable.php::DateTimeImmutable", "REF:::DateTimeImmutable"),
            ("Record.php::Record", "REF:::Lib::Record"),
        ]

    def test_inherited_member_and_interface(self, tmp_path):
        """A method the receiver's class does not declare is its
        superclass's or its interface's; one that a base from outside the
        project may declare is not the project's."""
        builder = _index(
            tmp_path,
            {
                "src/Shapes.php": """
                <?php
                namespace App;

                interface Shape {
                    public function area(): float;
                }
                abstract class Figure implements Shape {
                    public function describe(): string { return ''; }
                }
                class Square extends Figure {
                    public function area(): float { return 1.0; }
                }
                class Failure extends \\RuntimeException {
                    public function describe(): string { return ''; }
                }
                class Decoy {
                    public function area(): float { return 0.0; }
                    public function describe(): string { return ''; }
                    public function getMessage(): string { return ''; }
                }
                class Client {
                    public function run(Shape $shape, Square $square, Failure $failure)
                    {
                        $shape->area();
                        $square->describe();
                        $failure->getMessage();
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Shapes.php::Client::run") == [
            "FILE:src/Shapes.php::Figure::describe",
            "FILE:src/Shapes.php::Shape::area",
        ]
        inherits = {
            (e["source"].rpartition("::")[2], e["target"].rpartition("::")[2])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        }
        assert inherits == {
            ("Figure", "Shape"),
            ("Square", "Figure"),
            ("Failure", "RuntimeException"),
        }

    def test_instanceof_and_catch_type_the_variable(self, tmp_path):
        """`instanceof` types a variable or a property inside the block it
        opens, and `catch` the exception it names."""
        builder = _index(
            tmp_path,
            {
                "src/Handlers.php": """
                <?php
                interface Handler {
                    public function handle(): void;
                }
                interface Resettable {
                    public function reset(): void;
                }
                class Failure extends \\Exception {
                    public function reset(): void {}
                }
                class Wrapper {
                    private Handler $handler;

                    public function reset(Handler $inner): void
                    {
                        if ($inner instanceof Resettable) {
                            $inner->reset();
                        }
                        $inner->reset();
                        if ($this->handler instanceof Resettable) {
                            $this->handler->reset();
                        }
                        if (!$inner instanceof Resettable) {
                            $inner->reset();
                        }
                    }

                    public function attempt(): void
                    {
                        try {
                            $this->handler->handle();
                        } catch (Failure $failure) {
                            $failure->reset();
                        }
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Handlers.php::Wrapper::reset") == [
            "FILE:src/Handlers.php::Resettable::reset",
            "FILE:src/Handlers.php::Resettable::reset",
            "REF:$inner->reset",
            "REF:$inner->reset",
        ]
        assert _calls(builder, "FILE:src/Handlers.php::Wrapper::attempt") == [
            "FILE:src/Handlers.php::Failure::reset",
            "FILE:src/Handlers.php::Handler::handle",
        ]

    def test_bare_call_reaches_a_function_only(self, tmp_path):
        """A `name(` without a receiver is the function of the caller's
        namespace, else the global one, else the one imported: never a
        method or a property of that name."""
        builder = _index(
            tmp_path,
            {
                "src/functions.php": """
                <?php
                namespace App\\Text;

                function trim(string $text): string { return $text; }
                """,
                "src/global.php": """
                <?php
                function describe($value): string { return ''; }
                function trim(string $text): string { return $text; }
                """,
                "src/Text/Cleaner.php": """
                <?php
                namespace App\\Text;

                class Cleaner {
                    private string $substr;

                    public function substr(string $text): string
                    {
                        return substr(trim($text), 1) . describe($text) . \\trim($text);
                    }
                }
                """,
                "src/Report.php": """
                <?php
                namespace App;

                use function App\\Text\\trim;

                class Report {
                    public function title(string $text): string
                    {
                        return trim($text) . Text\\trim($text) . describe();
                    }

                    public function describe(): string { return ''; }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/Text/Cleaner.php::Cleaner::substr") == [
            "FILE:src/functions.php::trim",
            "FILE:src/global.php::describe",
            "FILE:src/global.php::trim",
        ]
        assert _calls(builder, "FILE:src/Report.php::Report::title") == [
            "FILE:src/functions.php::trim",
            "FILE:src/functions.php::trim",
            "FILE:src/global.php::describe",
        ]

    def test_phpdoc_types_what_has_no_native_type(self, tmp_path):
        """`@var`, `@param` and `@return` type a property, a variable, a
        parameter and a result that the code leaves untyped, and a
        `@method` tag declares a method the class answers by `__call`."""
        builder = _index(
            tmp_path,
            {
                "src/Types.php": """
                <?php
                namespace App;

                /**
                 * @method Product latest()
                 */
                class Store {
                    /** @var Product|null */
                    private $featured;

                    /**
                     * @param Product $product
                     *
                     * @return Product
                     */
                    public function keep($product)
                    {
                        $product->ship();
                        $this->featured->ship();
                        return $product;
                    }

                    public function __call($name, $arguments) {}
                }
                class Product {
                    public function ship(): void {}
                }
                class Decoy {
                    public function ship(): void {}
                }
                class Client {
                    public function run(Store $store, array $items): void
                    {
                        $store->keep($items[0])->ship();
                        $store->latest()->ship();
                        /** @var Product $first */
                        $first = $items[0];
                        $first->ship();
                    }
                }
                """
            },
        )
        ship = "FILE:src/Types.php::Product::ship"
        assert _calls(builder, "FILE:src/Types.php::Store::keep") == [ship, ship]
        assert _calls(builder, "FILE:src/Types.php::Client::run") == [
            ship,
            ship,
            ship,
            "FILE:src/Types.php::Store::keep",
            "FILE:src/Types.php::Store::latest",
        ]

    def test_receiver_of_unknown_type_is_not_guessed(self, tmp_path):
        """A call on a value no declaration types, or on what a call into
        a library returns, stays unresolved though one project class alone
        has a method of that name, and is external where only a property
        has it; a call on a value of a type from outside the project is
        external."""
        builder = _index(
            tmp_path,
            {
                "src/Ledger.php": """
                <?php
                namespace App;

                use Lib\\Message;

                class Ledger {
                    public $total;

                    public function settle(int $amount): void {}
                }
                class Client {
                    public function run($ledger, Message $message, array $rows): void
                    {
                        $ledger->settle(1);
                        $rows[0]->settle(2);
                        $ledger->total();
                        $message->settle(3);
                        $message->reply()->settle(4);
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Ledger.php::Client::run") == [
            "REF:$ledger->settle",
            "REF:settle",
            "REF:settle",
        ]

    def test_built_in_types_and_functions_are_read_from_the_data(self, tmp_path):
        """A call on what a method or a function of PHP's own returns is
        made on that type: external where the type has the method, though
        a project class has one of that name. A project class has the
        methods of the built-in one it extends, and a method neither has
        stays unresolved."""
        builder = _index(
            tmp_path,
            {
                "src/Clock.php": """
                <?php
                namespace App;

                class Decoy {
                    public function format(string $format): string { return ''; }
                    public function getName(): string { return ''; }
                    public function getMessage(): string { return ''; }
                    public function explain(): string { return ''; }
                }
                class Failure extends \\RuntimeException {
                }
                class Clock {
                    public function run(\\DateTimeImmutable $now, Failure $failure): void
                    {
                        $now->modify('+1 day')->format('c');
                        $now->getTimezone()->getName();
                        date_create()->format('c');
                        \\DateTime::createFromFormat('c', '')->format('c');
                        $failure->getPrevious()->getMessage();
                        $failure->explain();
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:src/Clock.php::Clock::run") == [
            "REF:$failure->explain"
        ]

    def test_element_of_a_typed_array_or_iterable(self, tmp_path):
        """What `foreach` or a destructuring assignment takes from an array
        whose PHPDoc type names its elements, or from an iterator of the
        standard library, is of that element type."""
        builder = _index(
            tmp_path,
            {
                "src/Group.php": """
                <?php
                namespace App;

                class Handler {
                    public function handle(): void {}
                }
                class Decoy {
                    public function handle(): void {}
                }
                class Group {
                    /** @var Handler[] */
                    private array $handlers;

                    /**
                     * @param array<string, Handler> $named
                     * @param \\ArrayIterator<int, Handler> $queued
                     *
                     * @return list<Handler>
                     */
                    public function all(array $named, \\ArrayIterator $queued): array
                    {
                        foreach ($this->handlers as $handler) {
                            $handler->handle();
                        }
                        foreach ($named as $name => $handler) {
                            $handler->handle();
                        }
                        foreach ($queued as $handler) {
                            $handler->handle();
                        }
                        return $this->handlers;
                    }

                    public function first(array $untyped): void
                    {
                        [$first] = $this->all([], new \\ArrayIterator());
                        $first->handle();
                        foreach ($untyped as $handler) {
                            $handler->handle();
                        }
                    }
                }
                """
            },
        )
        handle = "FILE:src/Group.php::Handler::handle"
        assert _calls(builder, "FILE:src/Group.php::Group::all") == [handle] * 3
        assert _calls(builder, "FILE:src/Group.php::Group::first") == [
            "FILE:src/Group.php::Group::all",
            handle,
            "REF:$handler->handle",
        ]

    def test_return_type_read_from_the_body(self, tmp_path):
        """A method without a declared return type returns what each of its
        `return` statements gives, where that is `$this`, a `new` object or
        a typed property; with another `return` beside it, nothing known."""
        builder = _index(
            tmp_path,
            {
                "src/Builder.php": """
                <?php
                namespace App;

                class Product {
                    public function ship(): void {}
                }
                class Decoy {
                    public function ship(): void {}
                    public function named(): void {}
                }
                class Builder {
                    private Product $product;

                    public function named($name)
                    {
                        return $this;
                    }

                    public function build()
                    {
                        return new Product();
                    }

                    public function last()
                    {
                        return $this->product;
                    }

                    public function find($name)
                    {
                        if ($name === '') {
                            return null;
                        }
                        return new Product();
                    }
                }
                class Client {
                    public function run(Builder $builder): void
                    {
                        $builder->named('a')->named('b')->build()->ship();
                        $builder->last()->ship();
                        $builder->find('a')->ship();
                    }
                }
                """
            },
        )
        ship = "FILE:src/Builder.php::Product::ship"
        assert _calls(builder, "FILE:src/Builder.php::Client::run") == [
            "FILE:src/Builder.php::Builder::build",
            "FILE:src/Builder.php::Builder::find",
            "FILE:src/Builder.php::Builder::last",
            "FILE:src/Builder.php::Builder::named",
            "FILE:src/Builder.php::Builder::named",
            ship,
            ship,
            "REF:ship",
        ]

    def test_variable_named_parent_is_no_receiver_of_its_own_class(self, tmp_path):
        """`parent` names the superclass in PHP alone: elsewhere a variable
        of that name is of unknown type, and a method two classes define
        is not pinned to the caller's own."""
        builder = _index(
            tmp_path,
            {
                "tree.js": """
                class Node {
                    size() {
                        return 1;
                    }

                    total(parent) {
                        return parent.size();
                    }
                }

                class Leaf {
                    size() {
                        return 0;
                    }
                }
                """
            },
        )
        assert _calls(builder, "FILE:tree.js::Node::total") == ["REF:parent.size"]

    def test_return_on_the_line_of_the_opening_brace(self, tmp_path):
        """`function make() { return new Widget; }` returns a `Widget`."""
        builder = _index(
            tmp_path,
            {
                "a.php": """
                <?php
                class Widget {
                    function spin() {}
                }
                class Gadget {
                    function spin() {}
                }
                function make() { return new Widget; }
                function go() {
                    $a = make();
                    $a->spin();
                }
                """,
            },
        )
        assert "FILE:a.php::Widget::spin" in _calls(builder, "FILE:a.php::go")


class TestDotnetCallResolution:
    def test_declared_type_picks_the_method(self, tmp_path):
        """A parameter, a local, a field and a property are of the type they
        are declared as."""
        builder = _index(
            tmp_path,
            {
                "Readers.cs": """
                namespace App;

                public class TreeReader
                {
                    public int Peek() => 0;
                }

                public class StreamReader
                {
                    public int Peek() => 0;
                }
                """,
                "Adapter.cs": """
                namespace App;

                public class Adapter
                {
                    readonly StreamReader _stream;
                    TreeReader Tree { get; }

                    public int Read(TreeReader input)
                    {
                        StreamReader local = _stream;
                        return input.Peek() + local.Peek() + _stream.Peek() + Tree.Peek();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Adapter.cs::Adapter::Read") == [
            "FILE:Readers.cs::StreamReader::Peek",
            "FILE:Readers.cs::StreamReader::Peek",
            "FILE:Readers.cs::TreeReader::Peek",
            "FILE:Readers.cs::TreeReader::Peek",
        ]

    def test_receiver_of_non_project_type_is_external(self, tmp_path):
        """A call on a value declared with a type the project does not
        define, on a type or a namespace it does not define, or of a
        constructor of one, reaches no project member of that name."""
        builder = _index(
            tmp_path,
            {
                "Hub.cs": """
                public class Hub
                {
                    public void Flush() { }
                    public void Run() { }
                    public void TimeOnly(int hours) { }
                }
                """,
                "Client.cs": """
                using System.IO;

                public class Client : Hub
                {
                    readonly List<Hub> _hubs = new();

                    public void Go(TextWriter output, string text)
                    {
                        output.Flush();
                        text.Run();
                        BenchmarkRunner.Run<Client>();
                        System.Console.Out.Flush();
                        _hubs.Run();
                        var stamp = new TimeOnly(13);
                        stamp.Run();
                    }
                }
                """,
            },
        )
        calls = [
            e["target"]
            for e in builder.edges
            if e["relation"].startswith("CALLS") and "Client.cs" in e["source"]
        ]
        assert calls == []

    def test_call_is_made_by_the_member_it_is_in(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "Helper.cs": """
                public static class Helper
                {
                    public static Helper Create() => null;
                    public static void Prepare() { }
                    public static void Work() { }
                    public static int Measure() => 0;
                    public static void Visit(int item) { }
                }
                """,
                "Client.cs": """
                public class Client
                {
                    readonly object _helper = Helper.Create();

                    public Client()
                    {
                        Helper.Prepare();
                    }

                    public int Size
                    {
                        get { return Helper.Measure(); }
                    }

                    public void Run(int[] items)
                    {
                        void Each(int item) => Helper.Visit(item);
                        Array.ForEach(items, item =>
                        {
                            Helper.Work();
                        });
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.cs::Client") == [
            "FILE:Helper.cs::Helper::Create"
        ]
        assert _calls(builder, "FILE:Client.cs::Client::<init>") == [
            "FILE:Helper.cs::Helper::Prepare"
        ]
        assert _calls(builder, "FILE:Client.cs::Client::Size") == [
            "FILE:Helper.cs::Helper::Measure"
        ]
        assert _calls(builder, "FILE:Client.cs::Client::Run") == [
            "FILE:Helper.cs::Helper::Work"
        ]
        assert _calls(builder, "FILE:Client.cs::Client::Run::Each") == [
            "FILE:Helper.cs::Helper::Visit"
        ]

    def test_chain_is_typed_by_properties_and_return_types(self, tmp_path):
        """Every link of a chain, on one line or over several, is a call on
        what the link before returns or on what the property before is
        declared as; so is a call on a `var` bound to the chain."""
        builder = _index(
            tmp_path,
            {
                "Config.cs": """
                public class Config
                {
                    public SinkConfig WriteTo { get; }
                    public Logger Create() => new Logger();
                }

                public class SinkConfig
                {
                    public Config Sink(
                        object sink,
                        int level = 0) => null;
                }

                public class Logger
                {
                    public void Write(string text) { }
                }
                """,
                "Decoy.cs": """
                public class Decoy
                {
                    public void Sink(object sink) { }
                    public void Create() { }
                    public void Write(string text) { }
                }
                """,
                "Client.cs": """
                public class Client
                {
                    void Chained()
                    {
                        new Config().WriteTo.Sink(null).Create().Write("x");
                    }

                    void Continued()
                    {
                        var logger = new Config()
                            .WriteTo.Sink(
                                null,
                                level: 1)
                            .Create();
                        logger.Write("x");
                    }
                }
                """,
            },
        )
        chain = [
            "FILE:Config.cs::Config",
            "FILE:Config.cs::Config::Create",
            "FILE:Config.cs::Logger::Write",
            "FILE:Config.cs::SinkConfig::Sink",
        ]
        assert _calls(builder, "FILE:Client.cs::Client::Chained") == chain
        assert _calls(builder, "FILE:Client.cs::Client::Continued") == chain

    def test_instantiation_is_a_call_of_the_type(self, tmp_path):
        """`new Name(...)`, `Name value = new(...)` and `new Name { ... }`
        create a Name, though a member is called Name too."""
        builder = _index(
            tmp_path,
            {
                "Stamp.cs": """
                public class Stamp
                {
                    public int Hour;
                    public void Press() { }
                }
                """,
                "Tests.cs": """
                public class Tests
                {
                    public void Stamp() { }

                    public void All()
                    {
                        var first = new Stamp();
                        Stamp second = new();
                        var third = new Stamp { Hour = 1 };
                        first.Press();
                        second.Press();
                        third.Press();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Tests.cs::Tests::All") == [
            "FILE:Stamp.cs::Stamp",
            "FILE:Stamp.cs::Stamp",
            "FILE:Stamp.cs::Stamp",
            "FILE:Stamp.cs::Stamp::Press",
            "FILE:Stamp.cs::Stamp::Press",
            "FILE:Stamp.cs::Stamp::Press",
        ]

    def test_member_of_this_its_bases_and_interfaces(self, tmp_path):
        """A bare call, a call on `this` or `base` and a call on an
        inherited field reach the members of the base types; a name that is
        a property of the type is that property, not the type of its name."""
        builder = _index(
            tmp_path,
            {
                "Core.cs": """
                namespace App;

                public interface ILogger
                {
                    void Write(string text);
                    void Twice(string text)
                    {
                        Write(text);
                    }
                }

                public class Logger : ILogger
                {
                    public void Write(string text) { }
                    public void Close() { }
                }

                public abstract class Base
                {
                    protected ILogger Shared;
                    protected void Helper() { }
                    protected virtual void Step() { }
                }
                """,
                "Log.cs": """
                namespace App;

                public class Log : Base
                {
                    public static ILogger Logger { get; set; }

                    protected override void Step()
                    {
                        Helper();
                        this.Helper();
                        base.Step();
                        Shared.Write("a");
                        Logger.Write("b");
                    }
                }
                """,
            },
        )
        inherits = {
            (e["source"].rpartition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        }
        assert inherits == {
            ("Logger", "FILE:Core.cs::ILogger"),
            ("Log", "FILE:Core.cs::Base"),
        }
        assert _calls(builder, "FILE:Core.cs::ILogger::Twice") == [
            "FILE:Core.cs::ILogger::Write"
        ]
        assert _calls(builder, "FILE:Log.cs::Log::Step") == [
            "FILE:Core.cs::Base::Helper",
            "FILE:Core.cs::Base::Helper",
            "FILE:Core.cs::Base::Step",
            "FILE:Core.cs::ILogger::Write",
            "FILE:Core.cs::ILogger::Write",
        ]

    def test_bare_call_reaches_no_other_type(self, tmp_path):
        """A bare call is one of a member of an enclosing type, of a local
        function or of a type named by `using static`; a variable called by
        its name holds a delegate."""
        builder = _index(
            tmp_path,
            {
                "Tools.cs": """
                namespace App.Util;

                public static class Tools
                {
                    public static void Check(bool ok) { }
                }

                public class Other
                {
                    public void Emit() { }
                    public void Notify() { }
                }
                """,
                "Client.cs": """
                using static App.Util.Tools;

                public class Outer
                {
                    static void Shared() { }

                    class Client
                    {
                        void Run(Action Notify)
                        {
                            Check(true);
                            Shared();
                            Emit();
                            Notify();
                        }
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.cs::Outer::Client::Run") == [
            "FILE:Client.cs::Outer::Shared",
            "FILE:Tools.cs::Tools::Check",
        ]

    def test_receiver_typed_where_it_is_bound(self, tmp_path):
        """A cast, `as`, an `is` pattern, `catch`, `foreach`, `using` and an
        `out var` argument give a value the type they name; a lambda's
        parameter has none, so a call on it is of no one definition."""
        builder = _index(
            tmp_path,
            {
                "Model.cs": """
                public class Order : IDisposable
                {
                    public void Ship() { }
                    public void Dispose() { }
                }

                public class Failure : Exception
                {
                    public void Report() { }
                }

                public class Store
                {
                    public bool TryFind(string key, out Order found)
                    {
                        found = null;
                        return false;
                    }
                }

                public class Decoy
                {
                    public void Ship() { }
                    public void Report() { }
                }
                """,
                "Client.cs": """
                public class Client
                {
                    void Typed(object raw, Order[] orders, Store store)
                    {
                        ((Order)raw).Ship();
                        (raw as Order)?.Ship();
                        var cast = (Order)raw;
                        cast.Ship();
                        if (raw is Order order) order.Ship();
                        foreach (Order each in orders) each.Ship();
                        using var held = new Order();
                        held.Ship();
                        if (store.TryFind("k", out var found)) found.Ship();
                        try { } catch (Failure failure) { failure.Report(); }
                    }

                    void Untyped(Order[] orders)
                    {
                        Array.ForEach(orders, o => o.Ship());
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.cs::Client::Typed") == [
            "FILE:Model.cs::Failure::Report",
            "FILE:Model.cs::Order",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Order::Ship",
            "FILE:Model.cs::Store::TryFind",
        ]
        assert _calls(builder, "FILE:Client.cs::Client::Untyped") == ["REF:o.Ship"]

    def test_unknown_receiver_is_not_guessed(self, tmp_path):
        """One project method has the name, and the receiver's type is not
        known: the call is left unresolved."""
        builder = _index(
            tmp_path,
            {
                "Cursor.cs": """
                public class Cursor
                {
                    public bool MoveNext() => false;
                }
                """,
                "Client.cs": """
                public class Client
                {
                    void Run(IEnumerable<int> numbers)
                    {
                        foreach (var e in numbers.Select(n => n.GetEnumerator()))
                            e.MoveNext();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.cs::Client::Run") == ["REF:e.MoveNext"]

    def test_usings_and_namespaces_tell_types_apart(self, tmp_path):
        """A name is the type of an enclosing namespace, of a namespace a
        `using` names or of what an alias stands for; where the project's
        type of that name is in neither, which type is meant is not told."""
        builder = _index(
            tmp_path,
            {
                "src/Core/Timer.cs": """
                namespace App.Core
                {
                    public class Timer
                    {
                        public void Start() { }
                    }
                }
                """,
                "src/Test/Timer.cs": """
                namespace App.Test;

                public class Timer
                {
                    public void Start() { }
                }
                """,
                "src/Core/Jobs.cs": """
                namespace App.Core.Jobs;

                public class Job
                {
                    void Run(Timer timer) => timer.Start();
                }
                """,
                "src/Web/Page.cs": """
                using App.Test;
                using Clock = App.Core.Timer;

                namespace Web;

                public class Page
                {
                    void Load(Timer timer, Clock clock, App.Core.Timer named)
                    {
                        timer.Start();
                        clock.Start();
                        named.Start();
                    }
                }
                """,
                "src/Other/Tool.cs": """
                using System.Threading;

                namespace Other;

                public class Tool
                {
                    void Run(Timer timer) => timer.Start();
                }
                """,
            },
        )
        core = "FILE:src/Core/Timer.cs::Timer::Start"
        assert _calls(builder, "FILE:src/Core/Jobs.cs::Job::Run") == [core]
        assert _calls(builder, "FILE:src/Web/Page.cs::Page::Load") == [
            core,
            core,
            "FILE:src/Test/Timer.cs::Timer::Start",
        ]
        assert _calls(builder, "FILE:src/Other/Tool.cs::Tool::Run") == [
            "REF:timer.Start"
        ]

    def test_extension_method(self, tmp_path):
        """A method whose first parameter is `this T` is called on a T or
        on a type derived from it. One written for a type from outside is
        the call's target where the standard library's data has the type
        and it lacks the member; a receiver it cannot scope stays open."""
        builder = _index(
            tmp_path,
            {
                "Shapes.cs": """
                public interface IShape { }

                public class Circle : IShape
                {
                    public void Draw() { }
                }
                """,
                "Extensions.cs": """
                public static class Extensions
                {
                    public static void Describe(this IShape shape) { }
                    public static string Shout(this string text) => text;
                    public static int Total(this IEnumerable<int> numbers) => 0;
                }
                """,
                "Client.cs": """
                public class Client
                {
                    void Run(Circle circle, string text, List<int> numbers)
                    {
                        circle.Describe();
                        circle.Draw();
                        text.Shout();
                        numbers.Total();
                        numbers.Clear();
                        Extensions.Shout(text);
                        string.Shout(text);
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.cs::Client::Run") == [
            "FILE:Extensions.cs::Extensions::Describe",
            "FILE:Extensions.cs::Extensions::Shout",
            "FILE:Extensions.cs::Extensions::Shout",
            "FILE:Extensions.cs::Extensions::Shout",
            "FILE:Shapes.cs::Circle::Draw",
            "REF:numbers.Total",
        ]

    def test_attribute_is_a_call_of_its_class(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "TemplateAttribute.cs": """
                public sealed class TemplateAttribute : Attribute
                {
                    public TemplateAttribute(string name) { }
                }
                """,
                "Log.cs": """
                public class Log
                {
                    [Template("message")]
                    [Obsolete]
                    public void Write(string message)
                    {
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Log.cs::Log::Write") == [
            "FILE:TemplateAttribute.cs::TemplateAttribute"
        ]

    def test_partial_type_is_one_type(self, tmp_path):
        """The members of a partial type are those of all its declarations;
        another type of its name is not one of them."""
        builder = _index(
            tmp_path,
            {
                "Reader.cs": """
                namespace App;

                public partial class Reader
                {
                    public void Read() => Fill();
                }
                """,
                "Reader.Async.cs": """
                namespace App;

                public partial class Reader
                {
                    void Fill() { }
                }
                """,
                "Other/Reader.cs": """
                namespace Other;

                public class Reader
                {
                    void Fill() { }
                    public void Read() => Missing();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Reader.cs::Reader::Read") == [
            "FILE:Reader.Async.cs::Reader::Fill"
        ]
        assert _calls(builder, "FILE:Other/Reader.cs::Reader::Read") == []

    def test_overloads(self, tmp_path):
        """A call reaches the first overload that takes as many arguments
        as it passes and is called as it is, on the type or on a value;
        what the call returns is told only where every overload returns the
        same."""
        builder = _index(
            tmp_path,
            {
                "Token.cs": """
                public class Token
                {
                    public bool Matches(Token other) => true;
                    public static bool Matches(Token a, Token b) => true;
                    static string GetType(Token token) => "";
                    public Token Wrap() => this;
                    public Text Wrap(int depth) => null;
                    public Token Copy() => this;
                    public Token Copy(int depth, params int[] more) => this;
                    public bool Equals(Token other) => true;
                    public void Close() { }

                    bool Same(Token other) => Equals(this, other);
                }

                public class Text
                {
                    public void Close() { }
                }
                """,
                "Client.cs": """
                public class Client
                {
                    void Run(Token token)
                    {
                        Token.Matches(token, token);
                        token.Matches(token);
                        token.GetType();
                        token.Copy(1, 2, 3).Close();
                        token.Wrap(1).Close();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Token.cs::Token::Same") == []
        assert _calls(builder, "FILE:Client.cs::Client::Run") == [
            "FILE:Token.cs::Token::Close",
            "FILE:Token.cs::Token::Matches",
            "FILE:Token.cs::Token::overload2_Copy",
            "FILE:Token.cs::Token::overload2_Matches",
            "FILE:Token.cs::Token::overload2_Wrap",
            "REF:Close",
            "REF:token.GetType",
        ]

    def test_generic_type_from_outside(self, tmp_path):
        """A member of `List<Order>` is the list's; what it returns may be
        an Order, so a call on that is not dropped. A `Task<Order>` awaited
        is an Order; so may be what a generic method from outside returns."""
        builder = _index(
            tmp_path,
            {
                "Order.cs": """
                public class Order
                {
                    public void Add(int item) { }
                    public void Ship() { }
                }

                public class Orders
                {
                    public Task<Order> LoadAsync() => null;
                    public List<Order> All { get; }
                }

                public class Decoy
                {
                    public void Ship() { }
                }
                """,
                "Client.cs": """
                public class Client
                {
                    async Task Run(Orders orders, IServiceProvider services)
                    {
                        orders.All.Add(null);
                        orders.All.First().Ship();
                        var loaded = await orders.LoadAsync();
                        loaded.Ship();
                        services.GetService<Order>().Ship();
                    }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:Client.cs::Client::Run") == [
            "FILE:Order.cs::Order::Ship",
            "FILE:Order.cs::Orders::LoadAsync",
            "REF:Ship",
            "REF:Ship",
        ]

    def test_out_variable_of_a_call_that_closes_on_a_later_line(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.cs": """
                namespace N {
                class Widget {
                    public void Spin() {}
                }
                class Gadget {
                    public void Spin() {}
                }
                class U {
                    bool TryGet(int k, out Widget w, int z) { w = null; return true; }
                    void Go() {
                        TryGet(1, out var b,
                            2);
                        b.Spin();
                    }
                }
                }
                """,
            },
        )
        assert "FILE:a.cs::Widget::Spin" in _calls(builder, "FILE:a.cs::U::Go")

    def test_field_of_a_type_named_record(self, tmp_path):
        """`record r = new();` declares the field `r`, not a type."""
        builder = _index(
            tmp_path,
            {
                "a.cs": """
                namespace N {
                class U {
                    record r = new();
                    void Go() {}
                }
                }
                """,
            },
        )
        kinds = {node["id"]: node["type"] for node in builder.nodes}
        assert kinds["FILE:a.cs::U::r"] == "Constant"


class TestClangCallResolution:
    def test_declared_type_picks_the_method(self, tmp_path):
        """A local, a parameter, a `Type name(args)` variable and a
        range-`for` variable are of the type they are declared with."""
        builder = _index(
            tmp_path,
            {
                "tree.h": """
                class TreeReader {
                 public:
                  int Peek();
                };
                """,
                "tree.cc": """
                #include "tree.h"
                int TreeReader::Peek() { return 0; }
                """,
                "stream.h": """
                class StreamReader {
                 public:
                  int Peek() { return 0; }
                };
                """,
                "adapter.cc": """
                #include "stream.h"
                #include "tree.h"
                int Read(TreeReader* in, const StreamReader& other) {
                  TreeReader local;
                  StreamReader stream(1);
                  for (TreeReader* each : readers) {
                    each->Peek();
                  }
                  return in->Peek() + other.Peek() + local.Peek() + stream.Peek();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:adapter.cc::Read") == [
            "FILE:stream.h::StreamReader::Peek",
            "FILE:stream.h::StreamReader::Peek",
            "FILE:tree.cc::TreeReader::Peek",
            "FILE:tree.cc::TreeReader::Peek",
            "FILE:tree.cc::TreeReader::Peek",
        ]

    def test_member_declared_and_defined_is_one_node(self, tmp_path):
        """A member declared in its class and defined outside it is the
        definition; a pure virtual one is its declaration."""
        builder = _index(
            tmp_path,
            {
                "db.h": """
                namespace store {
                class DB {
                 public:
                  virtual Status Put(int key) = 0;
                  Status Get(int key);
                };
                }
                """,
                "db.cc": """
                #include "db.h"
                namespace store {
                Status DB::Get(int key) { return Status(); }
                }
                """,
                "client.cc": """
                #include "db.h"
                void Run(store::DB* db) {
                  db->Put(1);
                  db->Get(1);
                }
                """,
            },
        )
        methods = sorted(n["id"] for n in builder.nodes if n["type"] == "Method")
        assert methods == [
            "FILE:db.cc::store::DB::Get",
            "FILE:db.h::store::DB::Put",
        ]
        assert _calls(builder, "FILE:client.cc::Run") == methods

    def test_field_of_the_class_types_the_receiver(self, tmp_path):
        """A method defined outside its class uses the fields the class
        declares, in the header; `->` on a smart pointer reaches what it
        points at, and a field of a library type is from outside."""
        builder = _index(
            tmp_path,
            {
                "log.h": """
                class Writer {
                 public:
                  void Append() {}
                  void size() {}
                  void reset() {}
                };
                """,
                "db.h": """
                #include <memory>
                #include <string>
                class Writer;
                class DB {
                  void Write();
                  Writer* log_ GUARDED_BY(mutex_);
                  std::unique_ptr<Writer> owned_;
                  std::string name_;
                };
                """,
                "db.cc": """
                #include "db.h"
                #include "log.h"
                void DB::Write() {
                  log_->Append();
                  this->log_->Append();
                  owned_->Append();
                  owned_.reset();
                  name_.size();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:db.cc::DB::Write") == [
            "FILE:log.h::Writer::Append",
            "FILE:log.h::Writer::Append",
            "FILE:log.h::Writer::Append",
        ]

    def test_implicit_this_reaches_the_class_and_its_bases(self, tmp_path):
        """A bare call in a member function is to a member of the class or
        of a base, never to a method of an unrelated class."""
        builder = _index(
            tmp_path,
            {
                "shapes.h": """
                class Shape {
                 public:
                  int Area() { return 0; }
                };
                class Square : public Shape {
                 public:
                  int Side() { return 1; }
                  int Total();
                };
                class Other {
                 public:
                  int Volume() { return 0; }
                };
                """,
                "shapes.cc": """
                #include "shapes.h"
                int Square::Total() {
                  return Side() + Area() + this->Area() + Volume();
                }
                """,
            },
        )
        inherits = {
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        }
        assert inherits == {("FILE:shapes.h::Square", "FILE:shapes.h::Shape")}
        assert _calls(builder, "FILE:shapes.cc::Square::Total") == [
            "FILE:shapes.h::Shape::Area",
            "FILE:shapes.h::Shape::Area",
            "FILE:shapes.h::Square::Side",
        ]

    def test_body_of_a_macro_definition_is_a_member_of_its_fixture(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "store.h": """
                class Store {
                 public:
                  void Put() {}
                };
                """,
                "store_test.cc": """
                #include "store.h"
                class StoreTest : public testing::Test {
                 public:
                  void Put() { store_.Put(); }
                  Store store_;
                };
                TEST_F(StoreTest, Empty) {
                  Put();
                  store_.Put();
                }
                """,
            },
        )
        test = "FILE:store_test.cc::StoreTest::TEST_F(StoreTest, Empty)"
        assert _calls(builder, test) == [
            "FILE:store.h::Store::Put",
            "FILE:store_test.cc::StoreTest::Put",
        ]

    def test_qualified_call(self, tmp_path):
        """`Type::name` and `space::name` are members of that type or
        namespace; a qualifier the project does not declare is outside."""
        builder = _index(
            tmp_path,
            {
                "status.h": """
                namespace util {
                class Status {
                 public:
                  static Status OK() { return Status(); }
                };
                int move(int x) { return x; }
                }
                """,
                "client.cc": """
                #include "status.h"
                int Run() {
                  util::Status::OK();
                  util::move(1);
                  std::move(2);
                  return 0;
                }
                """,
            },
        )
        assert _calls(builder, "FILE:client.cc::Run") == [
            "FILE:status.h::util::Status::OK",
            "FILE:status.h::util::move",
        ]

    def test_call_on_a_returned_value_is_typed_by_the_return_type(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "db.h": """
                class Iterator {
                 public:
                  void Seek() {}
                };
                class DB {
                 public:
                  Iterator* NewIterator() { return nullptr; }
                  std::string Name() { return ""; }
                };
                class Decoy {
                 public:
                  void Seek() {}
                  void size() {}
                };
                """,
                "client.cc": """
                #include "db.h"
                void Run(DB* db) {
                  db->NewIterator()->Seek();
                  auto it = db->NewIterator();
                  it->Seek();
                  auto* made = new Iterator();
                  made->Seek();
                  db->Name().size();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:client.cc::Run") == [
            "FILE:db.h::DB::Name",
            "FILE:db.h::DB::NewIterator",
            "FILE:db.h::DB::NewIterator",
            "FILE:db.h::Iterator",
            "FILE:db.h::Iterator::Seek",
            "FILE:db.h::Iterator::Seek",
            "FILE:db.h::Iterator::Seek",
        ]

    def test_include_decides_which_type_is_meant(self, tmp_path):
        """A type is known where its file is included; one of a name the
        project does not declare is from outside."""
        builder = _index(
            tmp_path,
            {
                "a/handler.h": """
                class Handler {
                 public:
                  void Run() {}
                };
                """,
                "b/handler.h": """
                class Handler {
                 public:
                  void Run() {}
                };
                """,
                "main.cc": """
                #include "b/handler.h"
                void Start(Handler* handler, FILE* file) {
                  handler->Run();
                  file->Run();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:main.cc::Start") == [
            "FILE:b/handler.h::Handler::Run"
        ]

    def test_static_function_is_called_from_its_file_only(self, tmp_path):
        """A bare call reaches a function with external linkage anywhere,
        one with internal linkage and a macro only where their file is
        compiled in."""
        builder = _index(
            tmp_path,
            {
                "a.c": """
                static int helper(void) { return 0; }
                int shared(void) { return helper(); }
                """,
                "limits.h": """
                #define CLAMP(x) (x)
                """,
                "b.c": """
                #include "limits.h"
                int run(void) {
                  return helper() + shared() + CLAMP(1);
                }
                """,
                "c.c": """
                int other(void) { return CLAMP(2); }
                """,
            },
        )
        assert _calls(builder, "FILE:a.c::shared") == ["FILE:a.c::helper"]
        assert _calls(builder, "FILE:b.c::run") == [
            "FILE:a.c::shared",
            "FILE:limits.h::CLAMP",
        ]
        assert _calls(builder, "FILE:c.c::other") == []

    def test_definitions_that_are_alternatives_stay_unresolved(self, tmp_path):
        """Two definitions of one function, for two platforms or as
        overloads in two headers, are both what a call may reach."""
        builder = _index(
            tmp_path,
            {
                "src/unix/core.c": """
                int loop_run(void) { return 0; }
                """,
                "src/win/core.c": """
                int loop_run(void) { return 0; }
                """,
                "src/unix/tcp.c": """
                int tcp_open(void) { return loop_run(); }
                """,
                "test/run.c": """
                int main(void) { return loop_run(); }
                """,
                "include/a.h": """
                namespace fmt { int write(int x) { return x; } }
                """,
                "include/b.h": """
                #include "a.h"
                namespace fmt {
                int write(char c) { return c; }
                int both() { return write(1); }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:src/unix/tcp.c::tcp_open") == ["REF:loop_run"]
        (edge,) = (
            e
            for e in builder.edges
            if e["source"] == "FILE:src/unix/tcp.c::tcp_open"
            and e["relation"] == "CALLS"
        )
        assert edge["metadata"]["candidates"] == [
            "FILE:src/unix/core.c::loop_run",
            "FILE:src/win/core.c::loop_run",
        ]
        assert _calls(builder, "FILE:test/run.c::main") == ["REF:loop_run"]
        assert _calls(builder, "FILE:include/b.h::fmt::both") == ["REF:write"]

    def test_a_guarded_call_reaches_the_definition_of_its_arm(self, tmp_path):
        """A call written inside `#ifdef WIN32` reaches the definition the
        same arm gives; written outside a guard, it names both."""
        builder = _index(
            tmp_path,
            {
                "win.c": """
                #ifdef WIN32
                int config(void) { return 1; }
                #endif
                """,
                "nix.c": """
                #ifndef WIN32
                int config(void) { return 0; }
                #endif
                """,
                "use.c": """
                #ifdef WIN32
                int use(void) { return config(); }
                #endif
                """,
                "any.c": """
                int any(void) { return config(); }
                """,
            },
        )
        assert _calls(builder, "FILE:use.c::use") == ["FILE:win.c::config"]
        assert _calls(builder, "FILE:any.c::any") == ["REF:config"]
        (edge,) = (
            e
            for e in builder.edges
            if e["source"] == "FILE:any.c::any" and e["relation"] == "CALLS"
        )
        assert edge["metadata"]["candidates"] == [
            "FILE:nix.c::config",
            "FILE:win.c::config",
        ]

    def test_only_definition_under_a_directive_is_the_call(self, tmp_path):
        """A name whose one definition an `#if` arm builds is that
        definition's, unless the standard library has the name for the
        builds that leave the arm out."""
        builder = _index(
            tmp_path,
            {
                "compat.c": """
                #ifdef TRACING
                void trace(const char* text) { }
                #endif
                #ifndef HAVE_STRDUP
                char* strdup(const char* text) { return 0; }
                #endif
                """,
                "use.c": """
                void use(const char* text) {
                    trace(text);
                    strdup(text);
                }
                """,
            },
        )
        assert _calls(builder, "FILE:use.c::use") == [
            "FILE:compat.c::trace",
            "REF:strdup",
        ]

    def test_call_of_a_value_is_no_call_of_a_definition(self, tmp_path):
        """A parameter that holds a function, and a struct member that
        does, declared in the struct or by a macro that stands for fields,
        are not the functions of that name: each call is one of a value.
        The struct is reached through its typedef."""
        builder = _index(
            tmp_path,
            {
                "loop.h": """
                typedef struct handle_s handle_t;
                typedef void (*read_cb_t)(handle_t* h);
                #define HANDLE_FIELDS \\
                  read_cb_t read_cb;  \\
                  int flags;
                struct handle_s {
                  HANDLE_FIELDS
                  void (*close_cb)(handle_t* h);
                };
                """,
                "loop.c": """
                #include "loop.h"
                static void close_cb(handle_t* h) {}
                static void done(void) {}
                void finish(handle_t* h, struct handle_s* raw, void (*done)(void)) {
                  h->close_cb(h);
                  h->read_cb(h);
                  raw->close_cb(raw);
                  done();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:loop.c::finish") == []

    def test_macro_invocation_is_a_reference_to_the_macro(self, tmp_path):
        """Rows whose macro each includer defines its own way reach none
        of them alone; a namespace that a macro opens holds what is at the
        top of the files."""
        builder = _index(
            tmp_path,
            {
                "opcodes.h": """
                OP(LOAD, 1)
                OP(DUP, 2)
                """,
                "names.c": """
                #define OP(name, size) #name,
                static const char* names[] = {
                #include "opcodes.h"
                };
                """,
                "sizes.c": """
                #define OP(name, size) size,
                static const int sizes[] = {
                #include "opcodes.h"
                };
                """,
                "lib.h": """
                #define LIB_BEGIN namespace lib {
                #define LIB_END }
                LIB_BEGIN
                int format(int x) { return x; }
                LIB_END
                """,
                "use.cc": """
                #include "lib.h"
                struct Printer {
                  int format(int x) { return lib::format(x); }
                };
                """,
            },
        )
        assert _calls(builder, "FILE:opcodes.h") == ["REF:OP", "REF:OP"]
        assert _calls(builder, "FILE:use.cc::Printer::format") == ["FILE:lib.h::format"]

    def test_call_in_a_macro_argument_is_qualified_as_the_macro_writes_it(
        self, tmp_path
    ):
        """`SYS(close(fd))` is `::close(fd)` where the macro writes `::`
        before its parameter, itself or through a macro it passes it to:
        no call of the member of that name. A macro that writes the
        argument as it stands leaves the call a member's."""
        builder = _index(
            tmp_path,
            {
                "os.h": """
                #define SYS(call) ::call
                #define POSIX(call) SYS(call)
                #define CHECK(value) check(value)
                int check(int value);
                class File {
                 public:
                  int close();
                  int size();
                  int fd_;
                };
                """,
                "os.cc": """
                #include "os.h"
                int check(int value) { return value; }
                int File::size() { return 0; }
                int File::close() {
                  CHECK(size());
                  return POSIX(close(fd_));
                }
                """,
            },
        )
        assert _calls(builder, "FILE:os.cc::File::close") == [
            "FILE:os.cc::File::size",
            "FILE:os.h::CHECK",
            "FILE:os.h::POSIX",
        ]

    def test_alias_is_the_type_it_names(self, tmp_path):
        """A receiver, a field and a constructor call written with a
        `typedef` or `using` name are of the type the alias names where it
        is declared; an alias of a library type is from outside."""
        builder = _index(
            tmp_path,
            {
                "buffer.h": """
                #include <string>
                template <typename T> class basic_buffer {
                 public:
                  void append(T value) {}
                  int size() { return 0; }
                };
                using buffer = basic_buffer<char>;
                typedef std::string text;
                class Table {
                 public:
                  class Iterator {
                   public:
                    void Seek() {}
                  };
                };
                class Decoy {
                 public:
                  void Seek() {}
                  int size() { return 0; }
                };
                """,
                "use.cc": """
                #include "buffer.h"
                class Mem {
                  typedef Table Rows;
                  void Run(buffer& out, text& name);
                  buffer scratch_;
                };
                void Mem::Run(buffer& out, text& name) {
                  out.append('a');
                  scratch_.size();
                  name.size();
                  auto made = buffer();
                  made.size();
                  Rows::Iterator it;
                  it.Seek();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:use.cc::Mem::Run") == [
            "FILE:buffer.h::Table::Iterator::Seek",
            "FILE:buffer.h::basic_buffer",
            "FILE:buffer.h::basic_buffer::append",
            "FILE:buffer.h::basic_buffer::size",
            "FILE:buffer.h::basic_buffer::size",
        ]

    def test_fixture_of_a_macro_definition_declared_in_another_file(self, tmp_path):
        """The body is a member's of the type that an argument names where
        the file includes its declaration; a type defined in a body types
        what is declared with it."""
        builder = _index(
            tmp_path,
            {
                "fixture.h": """
                class Store {
                 public:
                  void Flush() {}
                };
                class StoreTest {
                 public:
                  void Fill() {}
                  Store* store_;
                };
                """,
                "store_test.cc": """
                #include "fixture.h"
                TEST_F(StoreTest, Empty) {
                  struct Probe {
                    void Check() {}
                  };
                  Probe probe;
                  probe.Check();
                  Fill();
                  store_->Flush();
                }
                """,
            },
        )
        test = "FILE:store_test.cc::TEST_F(StoreTest, Empty)"
        assert _calls(builder, test) == [
            "FILE:fixture.h::Store::Flush",
            "FILE:fixture.h::StoreTest::Fill",
            f"{test}::Probe::Check",
        ]

    def test_range_for_types_its_variable_as_the_iterated(self, tmp_path):
        """`for (auto& s : v)` binds `s` to the first type argument of the
        container `v` is declared as."""
        builder = _index(
            tmp_path,
            {
                "s.h": """
                struct S { int run() { return 0; } };
                """,
                "main.cc": """
                #include "s.h"
                #include <vector>
                int main() {
                  std::vector<S> v;
                  for (auto& s : v) { s.run(); }
                  for (auto other : v) other.run();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:main.cc::main") == [
            "FILE:s.h::S::run",
            "FILE:s.h::S::run",
        ]

    def test_smart_pointer_arrow_reaches_what_it_holds(self, tmp_path):
        """`p->member` calls the member on the pointed-to type while
        `p.member` calls the smart pointer's own — external where only the
        standard library defines it."""
        builder = _index(
            tmp_path,
            {
                "s.h": """
                struct S { int run() { return 0; } int tally() { return 0; } };
                """,
                "main.cc": """
                #include "s.h"
                #include <memory>
                int main() {
                  std::unique_ptr<S> p;
                  p->run();
                  p.get();
                  p.reset();
                }
                """,
            },
        )
        calls = builder.edges
        resolved = [
            e["target"]
            for e in calls
            if e["source"] == "FILE:main.cc::main" and e["relation"] == "CALLS_RESOLVED"
        ]
        assert resolved == ["FILE:s.h::S::run"]

    def test_optional_deref_types_the_value_it_holds(self, tmp_path):
        """`o->member` on a `std::optional<T>` is `T`'s member."""
        builder = _index(
            tmp_path,
            {
                "s.h": """
                struct S { int run() { return 0; } };
                """,
                "main.cc": """
                #include "s.h"
                #include <optional>
                int main() {
                  std::optional<S> o;
                  o->run();
                  o.value();
                }
                """,
            },
        )
        resolved = [
            e["target"]
            for e in builder.edges
            if e["source"] == "FILE:main.cc::main" and e["relation"] == "CALLS_RESOLVED"
        ]
        assert resolved == ["FILE:s.h::S::run"]

    def test_container_members_are_the_librarys(self, tmp_path):
        """A `std::` container's members are external calls the data types,
        not unresolved project calls."""
        builder = _index(
            tmp_path,
            {
                "s.h": """
                struct S { int run() { return 0; } };
                """,
                "main.cc": """
                #include "s.h"
                #include <vector>
                int main() {
                  std::vector<S> v;
                  v.front().run();
                  v.size();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:main.cc::main") == ["FILE:s.h::S::run"]

    def test_project_type_shadows_the_standard_name(self, tmp_path):
        """A project type of a standard name is what a declaration means."""
        builder = _index(
            tmp_path,
            {
                "string.h": """
                namespace std {
                class string {
                 public:
                  void own() {}
                };
                }
                """,
                "main.cc": """
                #include "string.h"
                int main() {
                  std::string s;
                  s.own();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:main.cc::main") == [
            "FILE:string.h::std::string::own"
        ]

    def test_standard_function_result_types_its_calls(self, tmp_path):
        """What `std::make_unique<T>` returns is `unique_ptr<T>`: `->` on it
        is `T`'s member."""
        builder = _index(
            tmp_path,
            {
                "s.h": """
                struct S { int run() { return 0; } };
                """,
                "main.cc": """
                #include "s.h"
                #include <memory>
                int main() {
                  auto p = std::make_unique<S>();
                  p->run();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:main.cc::main") == ["FILE:s.h::S::run"]

    def test_auto_binds_what_a_call_of_its_own_statement_returns(self, tmp_path):
        """`auto x = make(); other();`: `x` is what `make` returns."""
        builder = _index(
            tmp_path,
            {
                "a.cc": """
                struct Widget { void spin(); };
                struct Gadget { void spin(); };
                void Widget::spin() {}
                void Gadget::spin() {}
                Widget make() { return Widget(); }
                Gadget other() { return Gadget(); }
                void go() {
                  auto x = make(); other();
                  x.spin();
                }
                """,
            },
        )
        assert "FILE:a.cc::Widget::spin" in _calls(builder, "FILE:a.cc::go")
        assert "FILE:a.cc::Gadget::spin" not in _calls(builder, "FILE:a.cc::go")

    def test_auto_bound_to_a_lambda_is_not_what_its_body_calls(self, tmp_path):
        """`auto f = [&]() { pack(); };`: `f` is no `Bag`."""
        builder = _index(
            tmp_path,
            {
                "a.cc": """
                struct Bag { int get(); };
                struct Jar { int get(); };
                int Bag::get() { return 1; }
                int Jar::get() { return 2; }
                Bag pack() { return Bag(); }
                void go() {
                  auto f = [&]() { pack(); };
                  f.get();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:a.cc::go") == ["FILE:a.cc::pack", "REF:f.get"]

    def test_auto_bound_to_a_lambda_called_at_once_is_not_its_argument(self, tmp_path):
        """`auto x = [&](Bag b) { return jar(); }(pack());`: `x` is no `Bag`."""
        builder = _index(
            tmp_path,
            {
                "a.cc": """
                struct Bag { int get(); };
                struct Jar { int get(); };
                int Bag::get() { return 1; }
                int Jar::get() { return 2; }
                Bag pack() { return Bag(); }
                Jar jar() { return Jar(); }
                void go() {
                  auto x = [&](Bag b) { return jar(); }(pack());
                  x.get();
                }
                """,
            },
        )
        assert "FILE:a.cc::Bag::get" not in _calls(builder, "FILE:a.cc::go")

    def test_typedef_of_a_pointer_to_an_unnamed_struct(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.c": """
                typedef struct { int v; } *Ptr;
                typedef struct { int v; } *const Fixed;
                typedef struct { int v; } *__restrict Sole;
                typedef struct { int v; } __attribute__((unused)) *Marked;
                typedef struct { int v; } __attribute__((aligned((4)))) *Nested;
                typedef struct { int v; } __declspec(dllexport) *Exported;
                typedef struct { int v; } alignas(8) *Aligned;
                typedef struct { int v; } _Alignas(8) *Wide;
                typedef struct { int v; } API_EXPORT(x) *Api;
                typedef struct { int v; } Handler(int);
                typedef struct { int v; } [[deprecated]] Old;
                typedef struct { int v; } Packed __attribute__((packed));
                typedef struct { int v; } Plain, *PlainPtr;
                typedef struct { int v; } (*Callback)(int, int);
                """,
            },
        )
        types = {node["id"] for node in builder.nodes if node["type"] == "Class"}
        named = (
            "Ptr Fixed Sole Marked Nested Exported Aligned Wide Api Old Packed Plain"
        )
        assert types == {f"FILE:a.c::{name}" for name in named.split()}

    def test_auto_reference_binds_what_the_call_returns(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.cc": """
                struct Widget { void spin(); };
                struct Gadget { void spin(); };
                void Widget::spin() {}
                void Gadget::spin() {}
                Widget& make() { static Widget w; return w; }
                void go() {
                  auto& x = make();
                  x.spin();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:a.cc::go") == [
            "FILE:a.cc::Widget::spin",
            "FILE:a.cc::make",
        ]

    def test_auto_bound_to_a_cast_is_not_what_the_call_returns(self, tmp_path):
        """`auto x = (Jar)pack();`: `x` is a `Jar`, whatever `pack` returns."""
        builder = _index(
            tmp_path,
            {
                "a.cc": """
                struct Bag { int get(); };
                struct Jar { int get(); };
                int Bag::get() { return 1; }
                int Jar::get() { return 2; }
                Bag pack() { return Bag(); }
                void go() {
                  auto x = (Jar)pack();
                  x.get();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:a.cc::go") == ["FILE:a.cc::pack", "REF:x.get"]

    def test_parenthesis_left_open_in_a_block_ends_with_the_block(self, tmp_path):
        """A macro may open a `(` that the text never closes."""
        builder = _index(
            tmp_path,
            {
                "a.c": """
                void k(void) {}
                void f(void) { BEGIN( 1; }
                void h(void) { k(); }
                """,
            },
        )
        assert _calls(builder, "FILE:a.c::h") == ["FILE:a.c::k"]

    def test_macro_call_among_enumerators_is_no_enumerator(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.c": """
                enum Kind {
                  KIND_A,
                  KIND_LIST(X)
                };
                """,
            },
        )
        constants = {n["id"] for n in builder.nodes if n["type"] == "Constant"}
        assert constants == {"FILE:a.c::Kind::KIND_A"}

    def test_attribute_before_the_name_of_a_type(self, tmp_path):
        builder = _index(
            tmp_path,
            {"a.cc": "enum class [[deprecated]] Level : int { low = 1 };\n"},
        )
        assert "FILE:a.cc::Level" in builder.node_registry


class TestRubyCallResolution:
    def test_constructed_receiver_resolves_to_its_class_s_method(self, tmp_path):
        """A local or an instance variable assigned `Const.new` is a Const:
        a method called on it is Const's, among several of the name."""
        builder = _index(
            tmp_path,
            {
                "lib/parser.rb": """
                class Parser
                  def initialize(limit)
                  end

                  def parse(text)
                  end
                end
                """,
                "lib/decoy.rb": """
                class Decoy
                  def parse(text)
                  end
                end
                """,
                "lib/reader.rb": """
                class Reader
                  def initialize
                    @parser = Parser.new(10)
                  end

                  def read(text)
                    @parser.parse(text)
                  end

                  def scan(text)
                    parser = Parser.new(
                      20
                    )
                    parser.parse text
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/reader.rb::Reader::initialize") == [
            "FILE:lib/parser.rb::Parser"
        ]
        assert _calls(builder, "FILE:lib/reader.rb::Reader::read") == [
            "FILE:lib/parser.rb::Parser::parse"
        ]
        assert _calls(builder, "FILE:lib/reader.rb::Reader::scan") == [
            "FILE:lib/parser.rb::Parser",
            "FILE:lib/parser.rb::Parser::parse",
        ]

    def test_receiver_made_outside_the_project_is_external(self, tmp_path):
        """`StringIO.new` and `Time.now` are no project values: a method
        called on them is not the project's method of that name."""
        builder = _index(
            tmp_path,
            {
                "lib/log.rb": """
                class Log
                  def write(line)
                  end

                  def to_i
                  end
                end
                """,
                "lib/report.rb": """
                class Report
                  def print
                    out = StringIO.new
                    out.write("x")
                    stamp = Time.now
                    stamp.to_i
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/report.rb::Report::print") == []

    def test_receiver_of_unknown_type_is_not_pinned_by_name(self, tmp_path):
        """A parameter may be any object: the one project method of the
        name called on it stays a candidate."""
        builder = _index(
            tmp_path,
            {
                "lib/courier.rb": """
                class Courier
                  def deliver(parcel)
                  end
                end
                """,
                "lib/report.rb": """
                class Report
                  def print(out)
                    out.deliver("x")
                    @app.deliver("y")
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/report.rb::Report::print") == [
            "REF:@app.deliver",
            "REF:out.deliver",
        ]

    def test_constant_receiver_is_looked_up_from_where_it_is_written(self, tmp_path):
        """A constant names the definition of the innermost enclosing
        namespace that has one, in whichever file that namespace declares
        it; a module's method is called on the module."""
        builder = _index(
            tmp_path,
            {
                "lib/rack/utils.rb": """
                module Rack
                  module Utils
                    def self.escape(text)
                    end
                  end
                end
                """,
                "lib/rack/mock/utils.rb": """
                module Rack
                  module Mock
                    module Utils
                      def self.escape(text)
                      end
                    end
                  end
                end
                """,
                "lib/rack/request.rb": """
                module Rack
                  class Request
                    def path
                      Utils.escape("a")
                    end
                  end

                  module Mock
                    class Request
                      def path
                        Utils.escape("a")
                        Rack::Utils.escape "b"
                        ::JSON.escape("c")
                      end
                    end
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/rack/request.rb::Rack::Request::path") == [
            "FILE:lib/rack/utils.rb::Rack::Utils::escape"
        ]
        mock = "FILE:lib/rack/request.rb::Rack::Mock::Request::path"
        assert _calls(builder, mock) == [
            "FILE:lib/rack/mock/utils.rb::Rack::Mock::Utils::escape",
            "FILE:lib/rack/utils.rb::Rack::Utils::escape",
        ]

    def test_call_without_receiver_is_made_on_self(self, tmp_path):
        """It reaches a method of the enclosing class, of its superclass
        or of a module it includes, not one of an unrelated class or of
        the namespace around it."""
        builder = _index(
            tmp_path,
            {
                "lib/base.rb": """
                module App
                  def self.render
                  end

                  class Base
                    def halt(code)
                    end
                  end

                  module Helpers
                    def status
                    end
                  end

                  class Other
                    def render
                    end

                    def redirect(to)
                    end
                  end
                end
                """,
                "lib/page.rb": """
                module App
                  class Page < Base
                    include Helpers

                    def show
                      halt 404
                      status
                      log_request
                      self.log_request
                      render
                      redirect "/"
                    end

                    def log_request
                    end
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/page.rb::App::Page::show") == [
            "FILE:lib/base.rb::App::Base::halt",
            "FILE:lib/base.rb::App::Helpers::status",
            "FILE:lib/page.rb::App::Page::log_request",
            "FILE:lib/page.rb::App::Page::log_request",
            "REF:redirect",
            "REF:render",
        ]

    def test_call_in_a_module_reaches_the_types_that_include_it(self, tmp_path):
        """A method of the one class that includes the module, or of a
        module that class also includes, is the one called; where two
        classes include it and differ, neither is picked."""
        builder = _index(
            tmp_path,
            {
                "lib/request.rb": """
                class Request
                  include Env
                  include Helpers

                  def session
                  end
                end

                class Upload
                  include Helpers

                  def session
                  end
                end

                module Env
                  def get_header(name)
                  end
                end

                module Helpers
                  def host
                    get_header "HOST"
                    session
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/request.rb::Helpers::host") == [
            "FILE:lib/request.rb::Env::get_header",
            "REF:session",
        ]

    def test_super_calls_the_method_of_a_base(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "lib/shapes.rb": """
                class Shape
                  def area(scale)
                  end
                end

                class Other
                  def area(scale)
                  end
                end

                class Square < Shape
                  def area(scale)
                    super
                  end
                end

                class Circle < Struct
                  def area(scale)
                    super(scale)
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/shapes.rb::Square::area") == [
            "FILE:lib/shapes.rb::Shape::area"
        ]
        assert _calls(builder, "FILE:lib/shapes.rb::Circle::area") == []

    def test_class_and_instance_methods_of_one_name(self, tmp_path):
        """Called on the class, in the body of a class or in a singleton
        method, the name is the singleton method; called on an instance or
        in an instance method, the instance method. `new` is the class's
        own where it defines one, and a `Class.new(Base)` has the methods
        of Base."""
        builder = _index(
            tmp_path,
            {
                "lib/parser.rb": """
                class Parser
                  def self.parse(io)
                    new.parse(io)
                  end

                  def parse(io)
                  end
                end

                class Pool
                  def self.new(size)
                  end
                end

                class Strict < Parser
                  parse(nil)

                  def self.check(io)
                    parse(io)
                  end

                  def check(io)
                    parse(io)
                  end
                end
                """,
                "lib/main.rb": """
                def main(io)
                  Parser.parse(io)
                  Pool.new(2)
                end

                def derived(io)
                  base = Class.new(Parser)
                  base.parse(io)
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/main.rb::derived") == [
            "FILE:lib/parser.rb::Parser::self.parse"
        ]
        for source in ("Strict", "Strict::self.check"):
            assert _calls(builder, f"FILE:lib/parser.rb::{source}") == [
                "FILE:lib/parser.rb::Parser::self.parse"
            ]
        assert _calls(builder, "FILE:lib/parser.rb::Strict::check") == [
            "FILE:lib/parser.rb::Parser::parse"
        ]
        assert _calls(builder, "FILE:lib/parser.rb::Parser::self.parse") == [
            "FILE:lib/parser.rb::Parser",
            "FILE:lib/parser.rb::Parser::parse",
        ]
        assert _calls(builder, "FILE:lib/main.rb::main") == [
            "FILE:lib/parser.rb::Parser::self.parse",
            "FILE:lib/parser.rb::Pool::new",
        ]

    def test_top_level_method_is_reached_through_a_require(self, tmp_path):
        """A method defined outside any class is every object's once its
        file is loaded: the caller's own file, or one it requires."""
        builder = _index(
            tmp_path,
            {
                "test/helper.rb": """
                def fixture(name)
                end
                """,
                "test/other_spec.rb": """
                def sample
                end
                """,
                "test/parser_spec.rb": """
                require_relative 'helper'

                describe "parser" do
                  it "parses" do
                    fixture("a")
                    sample
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:test/parser_spec.rb") == [
            "FILE:test/helper.rb::fixture",
            "REF:sample",
        ]

    def test_chain_and_binding_follow_what_a_call_returns(self, tmp_path):
        """A call on `Const.new(...)`, on a variable bound to such a chain
        and on a reader of an instance variable is typed by what precedes
        it; `@return [Type]` and `@param [Type]` in YARD and a Sorbet `sig`
        state a type."""
        builder = _index(
            tmp_path,
            {
                "lib/mock.rb": """
                class Response
                  def ok?
                  end
                end

                class Decoy
                  def ok?
                  end

                  def get(path)
                  end
                end

                class Session
                  attr_reader :last

                  def initialize
                    @last = Response.new
                  end

                  # @return [Response]
                  def get(path)
                  end
                end
                """,
                "lib/check.rb": """
                class Check
                  def chained
                    Session.new.get("/").ok?
                  end

                  def bound
                    response = Session.new.get("/")
                    response.ok?
                  end

                  def read
                    Session.new.last.ok?
                  end

                  # @param session [Session]
                  def declared(session)
                    session.get "/"
                  end

                  sig { params(session: Session).returns(Response) }
                  def signed(session)
                    session.get "/"
                  end
                end
                """,
            },
        )
        chain = [
            "FILE:lib/mock.rb::Response::ok?",
            "FILE:lib/mock.rb::Session",
            "FILE:lib/mock.rb::Session::get",
        ]
        assert _calls(builder, "FILE:lib/check.rb::Check::chained") == chain
        assert _calls(builder, "FILE:lib/check.rb::Check::bound") == chain
        assert _calls(builder, "FILE:lib/check.rb::Check::read") == [
            "FILE:lib/mock.rb::Response::ok?",
            "FILE:lib/mock.rb::Session",
            "FILE:lib/mock.rb::Session::last",
        ]
        for method in ("declared", "signed"):
            assert _calls(builder, f"FILE:lib/check.rb::Check::{method}") == [
                "FILE:lib/mock.rb::Session::get"
            ]

    def test_definition_that_cannot_take_the_arguments_is_not_the_one_called(
        self, tmp_path
    ):
        """A block may be run with another `self`: a method of the class
        around it that takes no argument is not what `body "x"` calls.
        Keywords passed to a method that names none are one argument."""
        builder = _index(
            tmp_path,
            {
                "lib/page.rb": """
                class Page
                  def body
                  end

                  def title(text, size = 1)
                  end

                  def show
                    route { body "x" }
                    body
                    title "a"
                    title "a", bold: true
                    title
                    title "a", 2, 3
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/page.rb::Page::show") == [
            "FILE:lib/page.rb::Page::body",
            "FILE:lib/page.rb::Page::title",
            "FILE:lib/page.rb::Page::title",
            "REF:body",
            "REF:title",
            "REF:title",
        ]

    def test_method_of_a_block_is_reached_from_inside_that_block(self, tmp_path):
        """Each block that defines methods has its own: a call reaches the
        one of the block it is in, or of a block around that one."""
        builder = _index(
            tmp_path,
            {
                "spec/page_spec.rb": """
                describe "one" do
                  def magic
                  end

                  it "works" do
                    magic
                  end

                  context "inner" do
                    def other
                    end

                    it "works" do
                      magic
                      other
                    end
                  end
                end

                describe "two" do
                  def magic
                  end

                  it "works" do
                    magic
                  end
                end
                """,
            },
        )
        spec = "FILE:spec/page_spec.rb"
        assert _calls(builder, f"{spec}::<describe:2>") == [
            f"{spec}::<describe:2>::magic"
        ]
        assert _calls(builder, f"{spec}::<describe:2>::<context:10>") == [
            f"{spec}::<describe:2>::<context:10>::other",
            f"{spec}::<describe:2>::magic",
        ]
        assert _calls(builder, f"{spec}::<describe:21>") == [
            f"{spec}::<describe:21>::magic"
        ]

    def test_method_defined_in_a_block_run_in_another_object(self, tmp_path):
        """`Const.class_eval do` adds to the class Const names; a block
        run in an unknown object is no part of the class around it; one
        run in `self` is."""
        builder = _index(
            tmp_path,
            {
                "lib/suite.rb": """
                class Widget
                  def name
                  end
                end

                class Suite
                  Widget.class_eval do
                    def label
                      name
                    end
                  end

                  helpers do
                    def label
                    end
                  end

                  class_eval do
                    def own
                    end
                  end

                  def run
                    Widget.new.label
                    own
                    label
                  end
                end
                """,
            },
        )
        suite = "FILE:lib/suite.rb::Suite"
        assert _calls(builder, f"{suite}::<class_eval:8>::label") == [
            "FILE:lib/suite.rb::Widget::name"
        ]
        assert _calls(builder, f"{suite}::run") == [
            "FILE:lib/suite.rb::Suite::<class_eval:8>::label",
            "FILE:lib/suite.rb::Suite::own",
            "FILE:lib/suite.rb::Widget",
            "REF:label",
        ]

    def test_base_of_the_same_simple_name_gives_its_members(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "lib/rack/request.rb": """
                module Rack
                  class Request
                    def path
                    end
                  end
                end
                """,
                "lib/app/request.rb": """
                module App
                  class Request < Rack::Request
                    def show
                      path
                    end
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:lib/app/request.rb::App::Request::show") == [
            "FILE:lib/rack/request.rb::Rack::Request::path"
        ]

    def test_method_returns_what_its_body_ends_with(self, tmp_path):
        """A method without a declared type returns what its last
        statement and its `return`s all give: a new object, the result of
        one call, `self`, a typed instance variable; `nil` aside. Two
        different things give no type."""
        builder = _index(
            tmp_path,
            {
                "lib/client.rb": """
                class Response
                  def ok?
                  end
                end

                class Decoy
                  def ok?
                  end

                  def get(path)
                  end
                end

                class Client
                  def initialize
                    @last = Response.new
                  end

                  def get(path) request("GET", path) end

                  def request(verb, path)
                    log verb
                    Response.new verb, path
                  ensure
                    close
                  end

                  def last
                    @last
                  end

                  def cached
                    @cached ||= Response.new
                  end

                  def either(flag)
                    return nil unless flag
                    Response.new
                  end

                  def mixed(flag)
                    return "none" unless flag
                    Response.new
                  end

                  def rescued
                    Response.new
                  rescue IOError
                    false
                  end

                  def same
                    self
                  end

                  def self.build
                    new
                  end
                end
                """,
                "lib/check.rb": """
                class Check
                  def typed
                    client = Client.new
                    client.get("/").ok?
                    client.last.ok?
                    client.cached.ok?
                    client.either(true).ok?
                    client.same.last
                    Client.build.last
                  end

                  def untyped
                    client = Client.new
                    client.mixed(true).ok?
                    client.rescued.ok?
                  end
                end
                """,
            },
        )
        client, ok = "FILE:lib/client.rb::Client", "FILE:lib/client.rb::Response::ok?"
        assert _calls(builder, "FILE:lib/check.rb::Check::typed") == sorted(
            [client, f"{client}::get", ok, f"{client}::last", ok, f"{client}::cached"]
            + [ok, f"{client}::either", ok, f"{client}::same", f"{client}::last"]
            + [f"{client}::build", f"{client}::last"]
        )
        assert _calls(builder, "FILE:lib/check.rb::Check::untyped") == [
            client,
            f"{client}::mixed",
            f"{client}::rescued",
            "REF:ok?",
            "REF:ok?",
        ]

    def test_type_named_on_one_line_spans_the_body_that_reopens_it(self, tmp_path):
        padding = "# pad\n" * 30
        builder = _index(
            tmp_path,
            {
                "a.rb": padding
                + "module Empty; end\nmodule Empty\n  def a\n  end\nend\n",
            },
        )
        spans = {
            node["id"]: (node["metadata"]["lineno"], node["metadata"]["end_lineno"])
            for node in builder.nodes
            if node["type"] != "File"
        }
        assert spans["FILE:a.rb::Empty"] == (32, 35)

    def test_character_literal_is_no_call(self, tmp_path):
        """`x = ?a` reads the character `a`, and calls no method `a`."""
        builder = _index(
            tmp_path,
            {
                "a.rb": """
                class K
                  def a
                  end
                  def m
                    x = ?a
                    x ? a : nil
                  end
                end
                """,
            },
        )
        assert _calls(builder, "FILE:a.rb::K::m") == ["FILE:a.rb::K::a"]

    def test_do_on_the_line_after_its_loop_opens_no_block(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "a.rb": """
                class K
                  def m(x)
                    while x
                    do
                      x = nil
                    end
                  end
                  def after
                  end
                end
                """,
            },
        )
        spans = {
            node["id"]: (node["metadata"]["lineno"], node["metadata"]["end_lineno"])
            for node in builder.nodes
            if node["type"] != "File"
        }
        assert spans["FILE:a.rb::K::m"] == (3, 8)


class TestDartCallResolution:
    def test_bare_call_reaches_only_what_the_library_has_in_scope(self, tmp_path):
        """`expect` comes from a package outside the project: it is not the
        method of that name of a project class the file never extends. A
        function the library declares or imports is reached."""
        builder = _index(
            tmp_path,
            {
                "pubspec.yaml": "name: app\n",
                "lib/checker.dart": """
                class Checker {
                  void expect(Object value) {}
                }

                void verify(Object value) {}
                """,
                "test/app_test.dart": """
                import 'package:app/checker.dart';
                import 'package:test/test.dart';

                void helper() {}

                void main() {
                  expect(1, 1);
                  verify(1);
                  helper();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:test/app_test.dart::main") == [
            "FILE:lib/checker.dart::verify",
            "FILE:test/app_test.dart::helper",
        ]

    def test_bare_call_reaches_members_of_the_class_and_what_it_inherits(
        self, tmp_path
    ):
        """A member of the class, of its superclass, of a mixin or of the
        type an extension is on; a variable that holds a function is the
        definition its call reaches."""
        builder = _index(
            tmp_path,
            {
                "lib/shapes.dart": """
                class Base {
                  void fromBase() {}
                }

                mixin Mixed {
                  void fromMixin() {}
                }

                class Shape extends Base with Mixed {
                  final void Function() _callback;
                  Shape(this._callback);

                  void own() {}

                  void run() {
                    own();
                    fromBase();
                    fromMixin();
                    _callback();
                  }
                }

                extension Rounded on Shape {
                  void round() {
                    own();
                  }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/shapes.dart::Shape::run") == [
            "FILE:lib/shapes.dart::Base::fromBase",
            "FILE:lib/shapes.dart::Mixed::fromMixin",
            "FILE:lib/shapes.dart::Shape::_callback",
            "FILE:lib/shapes.dart::Shape::own",
        ]
        assert _calls(builder, "FILE:lib/shapes.dart::Rounded::round") == [
            "FILE:lib/shapes.dart::Shape::own"
        ]

    def test_imports_decide_which_definition_a_name_is(self, tmp_path):
        """A relative import, `show`, `hide`, an `export` and a prefix each
        say which library a name comes from; a prefix of a package outside
        the project names nothing in it."""
        builder = _index(
            tmp_path,
            {
                "pubspec.yaml": "name: app\n",
                "lib/a.dart": """
                void load() {}
                void save() {}
                """,
                "lib/b.dart": """
                void load() {}
                void save() {}
                void join() {}
                """,
                "lib/barrel.dart": """
                export 'b.dart' show join;
                """,
                "lib/main.dart": """
                import 'package:path/path.dart' as p;

                import 'a.dart' show load;
                import 'b.dart' hide load, join;
                import 'barrel.dart' as barrel;

                void main() {
                  load();
                  save();
                  p.join('a', 'b');
                  barrel.join();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/main.dart::main") == [
            "FILE:lib/a.dart::load",
            "FILE:lib/b.dart::join",
            "FILE:lib/b.dart::save",
        ]

    def test_part_shares_the_scope_of_its_library(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "pubspec.yaml": "name: app\n",
                "lib/tools.dart": """
                void sharpen() {}
                """,
                "lib/shop.dart": """
                import 'tools.dart' as tools;
                import 'tools.dart';

                part 'src/bench.dart';

                void _sweep() {}
                """,
                "lib/src/bench.dart": """
                part of '../shop.dart';

                void work() {
                  _sweep();
                  sharpen();
                  tools.sharpen();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/src/bench.dart::work") == [
            "FILE:lib/shop.dart::_sweep",
            "FILE:lib/tools.dart::sharpen",
            "FILE:lib/tools.dart::sharpen",
        ]

    def test_constructor_expression_types_what_it_is_bound_to(self, tmp_path):
        """`Maker(...)`, `const Maker(...)`, `Maker.named(...)` and
        `Maker<T>.named(...)` make a Maker; each link of a chain is called
        on what the link before is declared to return."""
        builder = _index(
            tmp_path,
            {
                "lib/maker.dart": """
                class Maker<T> {
                  const Maker();
                  Maker.named(String name);
                  factory Maker.cached() => Maker();

                  Maker<T> tuned(int level) => this;
                  Product assemble() => Product();
                }

                class Product {
                  void ship() {}
                }

                class Decoy {
                  void tuned(int level) {}
                  void assemble() {}
                  void ship() {}
                }
                """,
                "lib/client.dart": """
                import 'maker.dart';

                void direct() {
                  final maker = const Maker<int>();
                  maker.tuned(1);
                  Maker.named('x').tuned(2).assemble().ship();
                }

                void continued() {
                  var product = Maker<int>.cached()
                      .tuned(3)
                      .assemble();
                  product.ship();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/client.dart::direct") == [
            "FILE:lib/maker.dart::Maker",
            "FILE:lib/maker.dart::Maker::assemble",
            "FILE:lib/maker.dart::Maker::named",
            "FILE:lib/maker.dart::Maker::tuned",
            "FILE:lib/maker.dart::Maker::tuned",
            "FILE:lib/maker.dart::Product::ship",
        ]
        assert _calls(builder, "FILE:lib/client.dart::continued") == [
            "FILE:lib/maker.dart::Maker::assemble",
            "FILE:lib/maker.dart::Maker::cached",
            "FILE:lib/maker.dart::Maker::tuned",
            "FILE:lib/maker.dart::Product::ship",
        ]

    def test_value_over_several_lines_is_what_its_last_call_returns(self, tmp_path):
        """A chain that goes on past a closure with a block body binds the
        variable to what the call that ends the statement returns."""
        builder = _index(
            tmp_path,
            {
                "lib/pipeline.dart": """
                class Pipeline {
                  Pipeline add(void Function() step) => this;
                  Server build() => Server();
                  void start() {}
                }

                class Server {
                  void start() {}
                }

                void main() {
                  final server = Pipeline()
                      .add(() {
                        Pipeline().start();
                      })
                      .build();
                  server.start();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/pipeline.dart::main") == [
            "FILE:lib/pipeline.dart::Pipeline",
            "FILE:lib/pipeline.dart::Pipeline",
            "FILE:lib/pipeline.dart::Pipeline::add",
            "FILE:lib/pipeline.dart::Pipeline::build",
            "FILE:lib/pipeline.dart::Pipeline::start",
            "FILE:lib/pipeline.dart::Server::start",
        ]

    def test_declared_type_of_a_local_a_parameter_or_a_field(self, tmp_path):
        """A typed local, a positional, named or optional parameter, an
        initializing formal, a field the class declares or inherits, and a
        field without a type whose value is one constructor call."""
        builder = _index(
            tmp_path,
            {
                "lib/store.dart": """
                class Reader {
                  int peek() => 0;
                }

                class Writer {
                  int peek() => 0;
                }

                class Base {
                  final Reader inherited = Reader();
                }

                class Store extends Base {
                  final Reader _reader;
                  final _spare = Writer();
                  Reader? maybe;

                  Store(this._reader) {
                    _reader.peek();
                  }

                  int all(Writer first, {required Reader named, Writer? last}) {
                    Reader local = named;
                    return first.peek() +
                        named.peek() +
                        last!.peek() +
                        local.peek() +
                        _reader.peek() +
                        _spare.peek() +
                        maybe?.peek() +
                        inherited.peek();
                  }
                }
                """,
            },
        )
        reader = "FILE:lib/store.dart::Reader::peek"
        writer = "FILE:lib/store.dart::Writer::peek"
        assert _calls(builder, "FILE:lib/store.dart::Store::new") == [reader]
        assert _calls(builder, "FILE:lib/store.dart::Store::all") == [
            *[reader] * 5,
            *[writer] * 3,
        ]

    def test_receiver_of_a_type_from_outside_is_external(self, tmp_path):
        """A value declared with a type the library does not have from the
        project, one a constructor from outside makes, a literal, and a
        static member of such a type do not reach the project's method of
        that name."""
        builder = _index(
            tmp_path,
            {
                "pubspec.yaml": "name: app\n",
                "lib/pipe.dart": """
                class Pipe {
                  void close() {}
                  static void connect() {}
                }
                """,
                "lib/client.dart": """
                import 'dart:io';

                class Client {
                  final HttpServer _server;
                  final List<Socket> _sockets = [];
                  Client(this._server);

                  void stop(Socket socket) {
                    _server.close();
                    socket.close();
                    _sockets.close();
                    Socket.connect();
                    const Duration().close();
                    final queue = <Pipe>[];
                    final label = 'pipe';
                    queue.close();
                    label.close();
                  }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/client.dart::Client::stop") == []

    def test_variable_that_holds_an_object_is_called_through_call(self, tmp_path):
        """Calling a variable calls the `call` method of the object it
        holds; one that holds a function, or a value of no known type,
        calls no definition, whatever shares its name."""
        builder = _index(
            tmp_path,
            {
                "lib/handlers.dart": """
                typedef Handler = void Function(String path);

                class Recorder {
                  void call(String path) {}
                }

                void handler(String path) {}
                void plain(String path) {}

                void run(Handler handler, Object Function() make) {
                  final recorder = Recorder();
                  final plain = make();
                  recorder('/a');
                  handler('/b');
                  plain('/c');
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/handlers.dart::run") == [
            "FILE:lib/handlers.dart::Recorder",
            "FILE:lib/handlers.dart::Recorder::call",
        ]

    def test_cascade_is_called_on_its_target(self, tmp_path):
        """Every section of a cascade, on one line or on the next ones, is
        a call on the cascade's target, which is also its value."""
        builder = _index(
            tmp_path,
            {
                "lib/router.dart": """
                class Router {
                  Router get(String path) => this;
                  void mount(String path) {}
                  void serve() {}
                }

                class Decoy {
                  void get(String path) {}
                  void mount(String path) {}
                  void serve() {}
                }

                void build(Router given) {
                  final router = Router()
                    ..get('/a')
                    ..mount('/b');
                  router.serve();
                  given..mount('/c')..serve();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/router.dart::build") == [
            "FILE:lib/router.dart::Router",
            "FILE:lib/router.dart::Router::get",
            "FILE:lib/router.dart::Router::mount",
            "FILE:lib/router.dart::Router::mount",
            "FILE:lib/router.dart::Router::serve",
            "FILE:lib/router.dart::Router::serve",
        ]

    def test_this_super_and_the_order_of_mixins(self, tmp_path):
        """`this.m()` is the class's own or inherited `m`, `super.m()` the
        inherited one; of two that are inherited the last mixin's wins."""
        builder = _index(
            tmp_path,
            {
                "lib/animals.dart": """
                class Animal {
                  void move() {}
                  void rest() {}
                }

                mixin Swimmer on Animal {
                  void move() {}
                }

                mixin Flyer on Animal {
                  void move() {}
                }

                class Duck extends Animal with Swimmer, Flyer {
                  void rest() {
                    super.rest();
                    this.move();
                    this.rest();
                  }
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/animals.dart::Duck::rest") == [
            "FILE:lib/animals.dart::Animal::rest",
            "FILE:lib/animals.dart::Duck::rest",
            "FILE:lib/animals.dart::Flyer::move",
        ]

    def test_extension_method_on_a_project_or_an_outside_type(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "lib/extensions.dart": """
                import 'dart:convert';

                class Request {
                  void send() {}
                }

                extension RequestParams on Request {
                  String param(String name) => name;
                }

                extension on String {
                  String shout() => this;
                }

                class Decoy {
                  void param(String name) {}
                  void shout() {}
                }

                void use(Request request, String text) {
                  request.param('id');
                  text.shout();
                  'literal'.shout();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/extensions.dart::use") == [
            "FILE:lib/extensions.dart::RequestParams::param",
            "FILE:lib/extensions.dart::extension on String::shout",
            "FILE:lib/extensions.dart::extension on String::shout",
        ]

    def test_static_member_enum_value_and_getter(self, tmp_path):
        """`Type.m()` is a static member of Type; a member of an enum value
        or of what a getter is declared to return is that type's."""
        builder = _index(
            tmp_path,
            {
                "lib/levels.dart": """
                enum Level {
                  low,
                  high;

                  String label() => name;
                }

                class Config {
                  static Config load() => Config();
                  Level get level => Level.low;
                }

                class Decoy {
                  void label() {}
                  void load() {}
                }

                void show(Config config) {
                  Config.load();
                  Level.high.label();
                  config.level.label();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/levels.dart::show") == [
            "FILE:lib/levels.dart::Config::load",
            "FILE:lib/levels.dart::Level::label",
            "FILE:lib/levels.dart::Level::label",
        ]

    def test_promotion_cast_and_pattern_type_the_receiver(self, tmp_path):
        """`is T` promotes a variable, `as T` casts a value, a pattern
        declares its variable with the type written; an object pattern
        calls no constructor."""
        builder = _index(
            tmp_path,
            {
                "lib/shapes.dart": """
                sealed class Shape {}

                class Circle extends Shape {
                  final double radius;
                  Circle(this.radius);
                  double area() => radius;
                }

                class Square extends Shape {
                  double area() => 0;
                }

                double measure(Object value, Shape shape) {
                  if (value is Circle) {
                    return value.area();
                  }
                  final casted = value as Square;
                  casted.area();
                  switch (shape) {
                    case Circle(:final radius):
                      return radius;
                    case Square square:
                      return square.area();
                  }
                  return (value as Circle).area();
                }

                double describe(Shape shape) => switch (shape) {
                  Circle(radius: final r) => r,
                  Square() => 0,
                };
                """,
            },
        )
        assert _calls(builder, "FILE:lib/shapes.dart::measure") == [
            "FILE:lib/shapes.dart::Circle::area",
            "FILE:lib/shapes.dart::Circle::area",
            "FILE:lib/shapes.dart::Square::area",
            "FILE:lib/shapes.dart::Square::area",
        ]
        assert _calls(builder, "FILE:lib/shapes.dart::describe") == []

    def test_awaited_future_gives_what_it_holds(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "lib/fetch.dart": """
                class Page {
                  void render() {}
                }

                class Decoy {
                  void render() {}
                }

                Future<Page> fetch() async => Page();

                Future<void> show() async {
                  final page = await fetch();
                  page.render();
                }
                """,
            },
        )
        assert _calls(builder, "FILE:lib/fetch.dart::show") == [
            "FILE:lib/fetch.dart::Page::render",
            "FILE:lib/fetch.dart::fetch",
        ]

    def test_call_is_made_by_the_member_it_is_in(self, tmp_path):
        """A call in a closure passed as an argument, in a getter, in a
        constructor's initializer list or body, or in a nested function
        leaves that member; one in a field's value leaves the class and one
        in an annotation the member it annotates."""
        builder = _index(
            tmp_path,
            {
                "lib/jobs.dart": """
                class Tag {
                  const Tag.of(String name);
                }

                int count() => 0;
                int twice(int value) => value;
                void each(void Function(int) body) {}

                class Jobs {
                  final int started = count();
                  final int limit;

                  Jobs() : limit = count() {
                    twice(1);
                  }

                  int get size => twice(2);

                  @Tag.of('run')
                  void run() {
                    each((value) {
                      twice(value);
                    });
                    int nested(int value) {
                      return count();
                    }
                    nested(3);
                  }
                }
                """,
            },
        )
        count, twice = "FILE:lib/jobs.dart::count", "FILE:lib/jobs.dart::twice"
        assert _calls(builder, "FILE:lib/jobs.dart::Jobs") == [count]
        assert _calls(builder, "FILE:lib/jobs.dart::Jobs::new") == [count, twice]
        assert _calls(builder, "FILE:lib/jobs.dart::Jobs::size") == [twice]
        assert _calls(builder, "FILE:lib/jobs.dart::Jobs::run") == [
            "FILE:lib/jobs.dart::Jobs::run::nested",
            "FILE:lib/jobs.dart::Tag::of",
            "FILE:lib/jobs.dart::each",
            twice,
        ]
        assert _calls(builder, "FILE:lib/jobs.dart::Jobs::run::nested") == [count]

    def test_cascade_on_a_variable_of_a_declared_type(self, tmp_path):
        """`var y = p..add(1)`: `y` is `p`, whatever `add` returns."""
        builder = _index(
            tmp_path,
            {
                "a.dart": """
                class Widget {
                  void spin() {}
                  void add(int a) {}
                }
                class Gadget {
                  void spin() {}
                  void add(int a) {}
                }
                void go(Widget p) {
                  var y = p..add(1);
                  y.spin();
                }
                """,
            },
        )
        assert "FILE:a.dart::Widget::spin" in _calls(builder, "FILE:a.dart::go")


class TestModuleAliases:
    """A module specifier that is no relative path and no package."""

    STORE = """
    export const store = {
        list: (page?: number): Promise<void> => Promise.resolve(),
    };
    """
    PAGE = """
    import { store } from '%s';
    export async function load() {
        await store.list(1);
    }
    """

    def _load_calls(self, tmp_path, specifier, **more):
        files = {"ui/src/lib/store.ts": self.STORE, **more}
        files["ui/src/routes/page.ts"] = self.PAGE % specifier
        return _calls(_index(tmp_path, files), "FILE:ui/src/routes/page.ts::load")

    def test_sveltekit_lib_is_src_lib_beside_the_configuration(self, tmp_path):
        config = {"ui/svelte.config.js": "export default {};\n"}
        assert self._load_calls(tmp_path, "$lib/store", **config) == [
            "FILE:ui/src/lib/store.ts::store::list"
        ]

    @pytest.mark.parametrize("specifier", ["$lib/store", "@/lib/store", "#lib/store"])
    def test_alias_that_cannot_be_located_leaves_the_call_unresolved(
        self, tmp_path, specifier
    ):
        assert self._load_calls(tmp_path, specifier) == ["REF:store.list"]

    def test_package_that_is_not_declared_stays_outside(self, tmp_path):
        assert self._load_calls(tmp_path, "some-package/store") == []


class TestProcessDirectory:
    """Tests for the directory walk and per-file dispatch."""

    @pytest.fixture
    def regex_builder(self, tmp_path):
        builder = CodeGraphBuilder(tmp_path)
        builder.use_ast_grep = False
        return builder

    @staticmethod
    def _files(builder):
        builder.use_ast_grep = False
        builder.process_directory()
        return {node["id"] for node in builder.nodes if node["type"] == "File"}

    @pytest.fixture
    def ignoring_tree(self, tmp_path):
        """A git repository that ignores `out/` and two files, one of which
        it tracks all the same."""
        for name in ("src.py", "local.py", "tracked.py", "out/built.py"):
            (tmp_path / name).parent.mkdir(exist_ok=True)
            (tmp_path / name).write_text("def f():\n    pass\n")
        (tmp_path / ".gitignore").write_text("out/\nlocal.py\ntracked.py\n")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "add", "-f", "tracked.py"], cwd=tmp_path, check=True)
        return tmp_path

    def test_files_git_ignores_are_not_indexed(self, ignoring_tree):
        builder = CodeGraphBuilder(ignoring_tree, respect_gitignore=True)
        assert self._files(builder) == {"FILE:src.py", "FILE:tracked.py"}

    def test_files_git_ignores_are_indexed_on_request(self, ignoring_tree):
        assert self._files(CodeGraphBuilder(ignoring_tree)) == {
            "FILE:src.py",
            "FILE:tracked.py",
            "FILE:local.py",
            "FILE:out/built.py",
        }

    def test_tree_outside_a_repository_is_walked_whole(self, tmp_path):
        (tmp_path / "src.py").write_text("def f():\n    pass\n")
        (tmp_path / ".gitignore").write_text("src.py\n")
        builder = CodeGraphBuilder(tmp_path, respect_gitignore=True)
        assert self._files(builder) == {"FILE:src.py"}

    def test_file_over_its_parse_time_is_left_and_the_next_is_parsed(
        self, tmp_path, monkeypatch, caplog
    ):
        (tmp_path / "a_endless.py").write_text("def never():\n    pass\n")
        (tmp_path / "b.py").write_text("def reached():\n    pass\n")
        parse = generate.PythonParser.parse

        def stuck_on_the_first(self, file_path, rel_path, content):
            while rel_path == "a_endless.py":
                time.sleep(0.01)
            return parse(self, file_path, rel_path, content)

        monkeypatch.setattr(generate.PythonParser, "parse", stuck_on_the_first)
        builder = CodeGraphBuilder(tmp_path, max_parse_seconds=0.2)
        with caplog.at_level(logging.WARNING):
            builder.process_directory()
        assert "FILE:b.py::reached" in builder.node_registry
        assert "Stopped parsing a_endless.py" in caplog.text

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

    def test_walk_outside_the_id_base_is_refused(self, tmp_path):
        """Node IDs are relative to the project root; a file outside it gets
        none rather than an absolute one."""
        (tmp_path / "root").mkdir()
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "app.py").write_text("x = 1\n")
        builder = CodeGraphBuilder(tmp_path / "elsewhere", rel_base=tmp_path / "root")

        with pytest.raises(ValueError):
            builder.process_directory()
        assert builder.nodes == []

    @pytest.mark.parametrize(
        "search",
        [
            lambda resolver: resolver._js_alias("node:url", "/srv/app"),
            lambda resolver: resolver._fixture_value(
                "db", "FILE:/srv/app/test_a.py::test_x", set()
            ),
        ],
        ids=["tsconfig", "conftest"],
    )
    def test_upward_search_ends_at_the_filesystem_root(self, tmp_path, search):
        resolver = CallResolver(CodeGraphBuilder(tmp_path))
        worker = threading.Thread(target=search, args=(resolver,), daemon=True)
        worker.start()
        worker.join(5)

        assert not worker.is_alive()

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

    def test_svelte_ast_grep_call_stays_in_its_own_block(self, tmp_path, monkeypatch):
        """A call in a later block is not given to an earlier block's
        function whose line in the file equals the call's line in its block."""
        (tmp_path / "App.svelte").write_text(
            '<script context="module">\n'
            "function unused() {}\n"
            "function first() { alpha(); }\n"
            "</script>\n"
            "<script>\n"
            "function second() {\n"
            "    beta();\n"
            "}\n"
            "</script>\n"
        )
        monkeypatch.setattr(
            generate,
            "extract_calls_typescript",
            lambda path: [
                {"callee": "alpha", "lineno": 3},
                {"callee": "beta", "lineno": 7},
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
            ("FILE:App.svelte::first", "REF:alpha", 3),
            ("FILE:App.svelte::second", "REF:beta", 7),
        ]


def _run_indexer(*args, cwd, **env):
    """Run the regex-only indexer as a child process, as the service does.
    ``env`` sets environment variables; None removes one."""
    environ = {
        **os.environ,
        "DESCRY_NO_SCIP": "1",
        "DESCRY_NO_EMBEDDINGS": "1",
        **env,
    }
    return subprocess.run(
        [sys.executable, "-P", "-m", "descry.generate", *args],
        cwd=cwd,
        env={k: v for k, v in environ.items() if v is not None},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class TestCommandLine:
    """Tests for the descry-generate entry point."""

    @pytest.fixture
    def root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "app.py").write_text("def hello():\n    return 1\n")
        return root

    @pytest.fixture
    def elsewhere(self, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "other.ts").write_text(
            'import { fileURLToPath } from "node:url";\n'
            "export function away() {\n    return fileURLToPath(1);\n}\n"
        )
        return elsewhere

    @staticmethod
    def _ids(cache):
        graph = json.loads((cache / "codebase_graph.json").read_text())
        return {node["id"] for node in graph["nodes"]}

    def test_project_root_alone_is_what_gets_walked(self, root, elsewhere, tmp_path):
        """Without a path the project root is indexed, not the directory the
        command runs in."""
        cache = tmp_path / "cache"
        result = _run_indexer(
            "--project-root", str(root), cwd=elsewhere, DESCRY_CACHE_DIR=str(cache)
        )

        assert result.returncode == 0, result.stderr
        assert self._ids(cache) == {"FILE:app.py", "FILE:app.py::hello"}

    def test_path_outside_the_project_root_is_refused(self, root, elsewhere, tmp_path):
        cache = tmp_path / "cache"
        result = _run_indexer(
            "--project-root",
            str(root),
            str(elsewhere),
            cwd=root,
            DESCRY_CACHE_DIR=str(cache),
        )

        assert result.returncode == 2
        assert "outside the project root" in result.stderr
        assert not cache.exists()

    def test_log_level_comes_from_the_environment(self, root, tmp_path):
        cache = str(tmp_path / "cache")
        default = _run_indexer(
            str(root), cwd=root, DESCRY_CACHE_DIR=cache, DESCRY_LOG_LEVEL=None
        )
        quiet = _run_indexer(
            str(root), cwd=root, DESCRY_CACHE_DIR=cache, DESCRY_LOG_LEVEL="ERROR"
        )
        verbose = _run_indexer(
            "-v", str(root), cwd=root, DESCRY_CACHE_DIR=cache, DESCRY_LOG_LEVEL="ERROR"
        )

        assert "INFO: Graph exported" in default.stderr
        assert quiet.returncode == 0
        assert "INFO:" not in quiet.stderr
        assert "INFO: Graph exported" in verbose.stderr

    @pytest.mark.parametrize(
        ("max_file_mb", "indexed"), [("0.00001", False), ("0", True)]
    )
    def test_files_over_the_configured_size_are_skipped(
        self, root, tmp_path, max_file_mb, indexed
    ):
        (root / ".descry.toml").write_text(
            f"[code_files]\nmax_file_mb = {max_file_mb}\n"
        )
        cache = tmp_path / "cache"
        result = _run_indexer(str(root), cwd=root, DESCRY_CACHE_DIR=str(cache))

        assert result.returncode == 0, result.stderr
        assert ("FILE:app.py::hello" in self._ids(cache)) is indexed
        assert ("exceeds [code_files] max_file_mb" in result.stderr) is not indexed


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

    def test_array_elements_that_are_not_objects_are_skipped(self, monkeypatch):
        match = {
            "metaVariables": {"single": {"FUNC": {"text": "run"}}},
            "range": {"start": {"line": 2}},
        }
        monkeypatch.setattr(
            ast_grep.subprocess,
            "run",
            lambda *a, **kw: _SgResult(json.dumps([None, "text", 7, match])),
        )
        assert [c["callee"] for c in ast_grep.extract_calls_rust("x.rs")] == ["run"]
        assert [c["callee"] for c in ast_grep.extract_calls_typescript("x.ts")] == [
            "run"
        ]
        assert ast_grep.extract_imports_typescript("x.ts")["imports"] == {}

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
struct TokenAuth;

impl FromRequestParts<AppState> for TokenAuth {
    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, Self::Rejection> {
        Ok(TokenAuth)
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
            "schema_version": 2,
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
            "schema_version": 2,
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
            "schema_version": 2,
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

    def test_under_threshold_no_truncation(self, large_function_graph, tmp_path):
        """Functions under 1000 tokens should NOT be truncated regardless of line count."""
        from descry.handlers import DescryConfig
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # 950 tokens is under the 1000 threshold, should show full source
        source = q.get_smart_source(source_file, 1, 150, 950, full=False)

        # Should NOT contain truncation marker (950 < 1000)
        assert "lines omitted" not in source
        assert source.count("let var") == 148

    def test_over_threshold_truncated(self, large_function_graph, tmp_path):
        """Functions over 1000 tokens AND over 100 lines should be truncated."""
        from descry.handlers import DescryConfig
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # 1200 tokens is over the new 1000 threshold, 150 lines is over 100
        source = q.get_smart_source(source_file, 1, 150, 1200, full=False)

        # Should contain truncation marker
        assert "lines omitted" in source
        # Not all 148 lines should be present
        assert source.count("let var") < 148

    def test_secondary_threshold_60_15_treatment(self, large_function_graph, tmp_path):
        """Functions between 1000-1800 tokens get 60+15 line treatment."""
        from descry.handlers import DescryConfig
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # 1200 tokens (between 1000-1800) should use 60+15 line treatment
        source = q.get_smart_source(source_file, 1, 150, 1200, full=False)

        # Should be truncated
        assert "lines omitted" in source
        # Head should have ~60 lines, tail ~15 lines = ~75 total
        # 150 - 75 = ~75 omitted
        line_count = source.count("let var")
        assert 70 <= line_count <= 80, f"Expected ~75 lines, got {line_count}"

    def test_very_large_threshold_50_15_treatment(self, large_function_graph, tmp_path):
        """Functions over 1800 tokens get 50+15 line treatment."""
        from descry.handlers import DescryConfig
        from descry.query import GraphQuerier

        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # 2000 tokens (over 1800) should use 50+15 line treatment
        source = q.get_smart_source(source_file, 1, 150, 2000, full=False)

        # Should be truncated
        assert "lines omitted" in source
        # Head should have ~50 lines, tail ~15 lines = ~65 total
        line_count = source.count("let var")
        assert 60 <= line_count <= 70, f"Expected ~65 lines, got {line_count}"

    def test_opposite_preprocessor_arms_are_alternatives(self, tmp_path):
        """A macro stub and a function in opposite arms of one `#if` are the
        builds' alternatives: an unguarded call reaches one of them, a call
        under the matching arm reaches the function."""
        builder = _index(
            tmp_path,
            {
                "internal.h": """
                #if defined(__linux__)
                int iou_fs(const char* a, const char* b);
                #else
                #define iou_fs(a, b) 0
                #endif
                """,
                "linux.c": """
                #include "internal.h"
                int iou_fs(const char* a, const char* b) { return 1; }
                """,
                "fs.c": """
                #include "internal.h"
                void f(void) { iou_fs("a", "b"); }
                #if defined(__linux__)
                void g(void) { iou_fs("a", "b"); }
                #endif
                """,
            },
        )
        # f is compiled everywhere: the arm decides which `iou_fs` it means.
        (edge,) = (
            e
            for e in builder.edges
            if e["source"] == "FILE:fs.c::f" and e["relation"] == "CALLS"
        )
        assert edge["target"] == "REF:iou_fs"
        assert edge["metadata"]["candidates"] == [
            "FILE:internal.h::iou_fs",
            "FILE:linux.c::iou_fs",
        ]
        # g is under the arm that declares the function: it calls that one.
        assert _calls(builder, "FILE:fs.c::g") == ["FILE:linux.c::iou_fs"]

    def test_dotnet_extension_under_a_directive(self, tmp_path):
        """An extension method the library's data says the receiver's type
        has is external, whatever a project defines under `#if`; one the
        library lacks is the project's, also when an `#if` arm builds it."""
        builder = _index(
            tmp_path,
            {
                "LinqBridge.cs": """
                #if !HAVE_LINQ
                using System.Collections.Generic;
                namespace N {
                static class Enumerable {
                  public static IEnumerable<TSource> Union<TSource>(
                    this IEnumerable<TSource> first,
                    IEnumerable<TSource> second) { return first; }
                  public static string Shout(this IEnumerable<char> source) { return ""; }
                }
                }
                #endif
                """,
                "JavaScriptUtils.cs": """
                using System.Collections.Generic;
                namespace N {
                class JavaScriptUtils {
                  void EscapeText(char[] escapeChars) {
                    var x = escapeChars.Union(new[] { '"' });
                    var z = escapeChars.Shout();
                  }
                }
                }
                """,
            },
        )
        # `IEnumerable<char>` has `Union` in the data: the call is external.
        edges = {
            e["target"]: e
            for e in builder.edges
            if e["source"] == "FILE:JavaScriptUtils.cs::JavaScriptUtils::EscapeText"
            and e["relation"].startswith("CALLS")
        }
        assert "REF:escapeChars.Union" not in edges
        assert "FILE:LinqBridge.cs::Enumerable::Shout" in edges


class TestOneLineDefinitionCalls:
    """A call on a TypeScript/JavaScript definition's own line is the
    definition's: its parameters and the first line of its body run when
    it is called (ECMA-262 10.2.11 FunctionDeclarationInstantiation)."""

    def test_every_row_of_the_fixture(self, tmp_path):
        """The one-line shapes: a function, a method, a named arrow, the
        members of a returned object; the return-line member keeps the
        owner's decision and a later line's member takes its own."""
        builder = _index(
            tmp_path,
            {
                "s.ts": """
                export function helper(): number { return 1; }
                export function oneLine(): number { return helper(); }
                export class K {
                  m(): number { return helper(); }
                  n(): number {
                    return helper();
                  }
                }
                export const arrow = (): number => helper();
                export function make() {
                  return {
                    load() { return helper(); },
                    save() {
                      return helper();
                    },
                    run: () => helper(),
                  };
                }
                export function makeLine() {
                  return { load() { return helper(); },
                    save() { return helper(); },
                  };
                }
                """
            },
        )
        file = "FILE:s.ts"
        assert _calls(builder, f"{file}::oneLine") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::K::m") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::K::n") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::arrow") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::make::object@12::load") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::make::object@12::save") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::make::object@12::run") == [f"{file}::helper"]
        # The member on the return line has no node: the owner's decision
        # keeps its call with the function that returns the object.
        assert _calls(builder, f"{file}::makeLine") == [f"{file}::helper"]
        assert _calls(builder, f"{file}::makeLine::save") == [f"{file}::helper"]

    def test_an_arrow_with_one_unparenthesised_parameter(self, tmp_path):
        """`x => g(x)`: the call is the arrow's, not the call's `(` taken
        for its parameters."""
        builder = _index(
            tmp_path,
            {
                "u.ts": """
                export function helper(x: any): any { return x; }
                export const single = x => helper(x);
                export function make() {
                  return {
                    run: x => helper(x),
                  };
                }
                """
            },
        )
        assert _calls(builder, "FILE:u.ts::single") == ["FILE:u.ts::helper"]
        assert _calls(builder, "FILE:u.ts::make::object@5::run") == [
            "FILE:u.ts::helper"
        ]

    def test_a_parameter_default_is_the_functions(self, tmp_path):
        """`x = g()` runs at each call of the function, not where it is
        written (ECMA-262 10.2.11)."""
        builder = _index(
            tmp_path,
            {
                "p.ts": """
                export function helper(): number { return 1; }
                export function f(x = helper()): number { return x; }
                export function multi(
                  x = helper(),
                ): number { return x; }
                """
            },
        )
        assert _calls(builder, "FILE:p.ts::f") == ["FILE:p.ts::helper"]

    def test_a_decorator_runs_where_the_class_is_defined(self, tmp_path):
        """`@dec()` on a member is evaluated while the class is being
        defined, in the scope the class is declared in."""
        builder = _index(
            tmp_path,
            {
                "d.ts": """
                export function dec(): any { return null; }
                export function helper(): number { return 1; }
                export class Host {
                  @dec() m(): number { return helper(); }
                }
                """
            },
        )
        assert _calls(builder, "FILE:d.ts") == ["FILE:d.ts::dec"]
        # The decorated method yields no node; its call stays with the class.
        assert _calls(builder, "FILE:d.ts::Host") == ["FILE:d.ts::helper"]

    def test_an_extends_clause_runs_where_the_class_is_defined(self, tmp_path):
        """The heritage expression is evaluated at class definition
        (ECMA-262 15.7.14 ClassDefinitionEvaluation), outside the class."""
        builder = _index(
            tmp_path,
            {
                "e.ts": """
                export function base(): any { return null; }
                export class Ext extends base() { m(): number { return 1; } }
                """
            },
        )
        assert _calls(builder, "FILE:e.ts") == ["FILE:e.ts::base"]

    def test_a_computed_key_runs_where_the_definition_is_evaluated(self, tmp_path):
        """A computed key is evaluated where the class or object is defined
        (ECMA-262 15.7.14; 13.2.5.6 PropertyDefinitionEvaluation), not in the member."""
        builder = _index(
            tmp_path,
            {
                "k.ts": """
                export function helper(): any { return null; }
                export class Computed {
                  [helper() as string](): number { return 1; }
                }
                export const objLit = {
                  [helper() as string]: 1,
                };
                """
            },
        )
        assert _calls(builder, "FILE:k.ts") == [
            "FILE:k.ts::helper",
            "FILE:k.ts::helper",
        ]

    def test_a_field_initialiser_is_the_classs(self, tmp_path):
        """A field initialiser runs when an instance is constructed
        (ECMA-262 7.3.33 InitializeInstanceElements); with no field
        node, the class owns it."""
        builder = _index(
            tmp_path,
            {
                "i.ts": """
                export function helper(): number { return 1; }
                export class Fields {
                  field = helper();
                }
                """
            },
        )
        assert _calls(builder, "FILE:i.ts::Fields") == ["FILE:i.ts::helper"]

    def test_the_first_of_several_declarators_takes_its_call(self, tmp_path):
        """`const a = () => f(), b = 2;`: the arrow takes the call written in
        it; a later declarator of the line gets no node of its own."""
        builder = _index(
            tmp_path,
            {
                "v.ts": """
                export function helper(): number { return 1; }
                export const a = () => helper(), second = 2;
                """
            },
        )
        assert _calls(builder, "FILE:v.ts::a") == ["FILE:v.ts::helper"]

    def test_a_call_after_the_definition_closes_is_not_its(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "c.ts": """
                export function helper(): number { return 1; }
                export function ab() { helper(); } helper();
                export function go(fn: () => number) { return fn(); }
                export function wrapped() { go(() => helper()); helper(); }
                """
            },
        )
        assert _calls(builder, "FILE:c.ts::ab") == ["FILE:c.ts::helper"]
        assert _calls(builder, "FILE:c.ts") == ["FILE:c.ts::helper"]
        # A closure passed inline stays with the function that passes it.
        assert _calls(builder, "FILE:c.ts::wrapped") == [
            "FILE:c.ts::go",
            "FILE:c.ts::helper",
            "FILE:c.ts::helper",
        ]

    def test_a_named_arrow_inside_a_function_takes_its_own_call(self, tmp_path):
        builder = _index(
            tmp_path,
            {
                "n.ts": """
                export function helper(): number { return 1; }
                export function outer(): number {
                  const inner = () => helper();
                  return inner();
                }
                """
            },
        )
        assert _calls(builder, "FILE:n.ts::outer::inner") == ["FILE:n.ts::helper"]
        assert _calls(builder, "FILE:n.ts::outer") == ["FILE:n.ts::outer::inner"]
