"""Tests for descry.generate — CodeGraphBuilder, parsers, and filters."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from descry.generate import (
    CodeGraphBuilder,
    PythonParser,
    RustParser,
    TSParser,
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

        func = [n for n in builder.nodes if n["type"] == "Function"][0]
        assert "str | bytes" in func["metadata"]["param_types"]
        assert func["metadata"]["return_type"] == "int | None"


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


class TestTypeScriptConfigurationNodes:
    """Tests for TypeScript configuration pattern detection (interceptors, middleware)."""

    @pytest.fixture
    def ts_config_content(self):
        """TypeScript file with interceptor and event handler patterns."""
        return '''
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
'''

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
                n for n in config_nodes
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
                n for n in config_nodes
                if "response_interceptor" in n["metadata"]["name"]
            ]

            assert len(response_interceptors) == 1
            node = response_interceptors[0]
            assert node["metadata"]["config_type"] == "interceptor"
            assert "apiClient.interceptors.response.use" in node["metadata"]["signature"]

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
                e for e in builder.edges
                if e["source"] == file_id and e["relation"] == "DEFINES"
            ]

            config_targets = [
                e["target"] for e in defines_edges
                if "interceptor" in e["target"]
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
    """Tests for the crate filter in search."""

    @pytest.fixture
    def multi_crate_graph(self):
        """Create a graph with nodes from multiple crates."""
        graph = {
            "nodes": [
                # mandible crate
                {
                    "id": "FILE:mandible/src/routes/handlers.rs",
                    "type": "File",
                    "metadata": {"path": "mandible/src/routes/handlers.rs", "name": "handlers.rs"},
                },
                {
                    "id": "FILE:mandible/src/routes/handlers.rs::create_deployment",
                    "type": "Function",
                    "metadata": {"name": "create_deployment", "docstring": "Create a new deployment"},
                },
                # instinct crate
                {
                    "id": "FILE:instinct/src/models/deployment.rs",
                    "type": "File",
                    "metadata": {"path": "instinct/src/models/deployment.rs", "name": "deployment.rs"},
                },
                {
                    "id": "FILE:instinct/src/models/deployment.rs::Deployment",
                    "type": "Class",
                    "metadata": {"name": "Deployment", "docstring": "Deployment model"},
                },
                # lens crate (TypeScript)
                {
                    "id": "FILE:lens/src/lib/api/deployments.ts",
                    "type": "File",
                    "metadata": {"path": "lens/src/lib/api/deployments.ts", "name": "deployments.ts"},
                },
                {
                    "id": "FILE:lens/src/lib/api/deployments.ts::fetchDeployments",
                    "type": "Function",
                    "metadata": {"name": "fetchDeployments", "docstring": "Fetch deployments from API"},
                },
            ],
            "edges": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            f.flush()
            yield f.name

        os.unlink(f.name)

    def test_filter_by_crate_mandible(self, multi_crate_graph):
        """Should return only results from mandible crate."""
        from descry.query import GraphQuerier
        q = GraphQuerier(multi_crate_graph)
        results = q.search_docs(["deployment"], crate="mandible")

        # Should find create_deployment from mandible
        assert len(results) > 0
        for r in results:
            assert r["id"].startswith("FILE:mandible/")

    def test_filter_by_crate_instinct(self, multi_crate_graph):
        """Should return only results from instinct crate."""
        from descry.query import GraphQuerier
        q = GraphQuerier(multi_crate_graph)
        results = q.search_docs(["deployment"], crate="instinct")

        # Should find Deployment from instinct
        assert len(results) > 0
        for r in results:
            assert r["id"].startswith("FILE:instinct/")

    def test_filter_by_crate_lens(self, multi_crate_graph):
        """Should return only results from lens crate."""
        from descry.query import GraphQuerier
        q = GraphQuerier(multi_crate_graph)
        results = q.search_docs(["deployment"], crate="lens")

        # Should find fetchDeployments from lens
        assert len(results) > 0
        for r in results:
            assert r["id"].startswith("FILE:lens/")

    def test_no_filter_returns_all_crates(self, multi_crate_graph):
        """Should return results from all crates when no filter."""
        from descry.query import GraphQuerier
        q = GraphQuerier(multi_crate_graph)
        results = q.search_docs(["deployment"])

        # Should find results from all crates
        crates_found = {r["id"].split("/")[0].replace("FILE:", "") for r in results}
        assert "mandible" in crates_found
        assert "instinct" in crates_found
        assert "lens" in crates_found

    def test_nonexistent_crate_returns_empty(self, multi_crate_graph):
        """Should return empty results for non-existent crate."""
        from descry.query import GraphQuerier
        q = GraphQuerier(multi_crate_graph)
        results = q.search_docs(["deployment"], crate="nonexistent")

        assert len(results) == 0

    def test_combine_crate_and_lang_filters(self, multi_crate_graph):
        """Should combine crate and language filters."""
        from descry.query import GraphQuerier
        q = GraphQuerier(multi_crate_graph)

        # Filter by lens crate (TypeScript) - this is the only TS crate
        results = q.search_docs(["deployment"], crate="lens")
        assert all("lens" in r["id"] for r in results)


class TestExcludeTestsFilter:
    """Tests for the exclude_tests filter in search."""

    @pytest.fixture
    def graph_with_tests(self):
        """Create a graph with both production and test nodes."""
        graph = {
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
                    "metadata": {"name": "validate_token", "docstring": "Validate auth token"},
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
                    "metadata": {"name": "test_validate_token", "docstring": "Test validate token"},
                },
                {
                    "id": "FILE:src/utils_test.py",
                    "type": "File",
                    "metadata": {"path": "src/utils_test.py", "name": "utils_test.py"},
                },
                {
                    "id": "FILE:src/utils_test.py::test_helper",
                    "type": "Function",
                    "metadata": {"name": "test_helper", "docstring": "Test helper function"},
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
        yield str(graph_file), str(source_file)

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
