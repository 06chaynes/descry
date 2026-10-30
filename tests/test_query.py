"""Tests for descry.query — GraphQuerier and related query functions."""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from descry.handlers import _DEFAULT_SYNTAX_LANG_MAP, DescryConfig
from descry.query import (
    _DEFAULT_TEST_FILE_SUFFIXES,
    _DEFAULT_TEST_PATH_PATTERNS,
    MAX_PATH_DEPTH,
    GraphQuerier,
)


def _context(q, node_id, **kwargs):
    return q.context_markdown(q.context_data(node_id, **kwargs))


def _flow(q, start_id, **kwargs):
    return q.flow_markdown(q.flow_data(start_id, **kwargs))


def _flatten(q, class_id):
    return q.flatten_markdown(q.flatten_data(class_id))


class TestGraphQuerier:
    """Tests for graph querying."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:test.py",
                    "type": "File",
                    "metadata": {"path": "test.py", "name": "test.py"},
                },
                {
                    "id": "FILE:test.py::helper",
                    "type": "Function",
                    "metadata": {"name": "helper", "docstring": "A helper function"},
                },
                {
                    "id": "FILE:test.py::main",
                    "type": "Function",
                    "metadata": {"name": "main", "docstring": "Main entry point"},
                },
            ],
            "edges": [
                {
                    "source": "FILE:test.py",
                    "target": "FILE:test.py::helper",
                    "relation": "DEFINES",
                },
                {
                    "source": "FILE:test.py",
                    "target": "FILE:test.py::main",
                    "relation": "DEFINES",
                },
                {
                    "source": "FILE:test.py::main",
                    "target": "FILE:test.py::helper",
                    "relation": "CALLS_RESOLVED",
                    "metadata": {"lineno": 10},
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            f.flush()
            yield f.name

        os.unlink(f.name)

    def test_find_nodes_by_name(self, sample_graph):
        """Should find nodes by name."""
        q = GraphQuerier(sample_graph, DescryConfig())
        matches = q.find_nodes_by_name("helper")
        assert len(matches) == 1
        assert matches[0]["metadata"]["name"] == "helper"

    def test_test_detection_covers_prefixes_and_the_top_level_tests_dir(self, tmp_path):
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:tests/test_foo.py::first",
                    "type": "Function",
                    "metadata": {"name": "first"},
                },
                {
                    "id": "FILE:src/test_bar.py::second",
                    "type": "Function",
                    "metadata": {"name": "second"},
                },
                {
                    "id": "FILE:src/main.py::third",
                    "type": "Function",
                    "metadata": {"name": "third"},
                },
                {
                    "id": "FILE:tests/json.rs::fourth",
                    "type": "Function",
                    "metadata": {"name": "fourth"},
                },
                {
                    "id": "FILE:contest/x.rs::fifth",
                    "type": "Function",
                    "metadata": {"name": "fifth"},
                },
            ],
            "edges": [],
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(graph))
        q = GraphQuerier(str(path), DescryConfig())
        q._ensure_filter_indices()
        assert q._index_is_test == {
            "FILE:tests/test_foo.py::first",
            "FILE:src/test_bar.py::second",
            "FILE:tests/json.rs::fourth",
        }

    def test_search_ties_keep_one_order_filtered_or_not(self, tmp_path):
        nodes = [
            {
                "id": f"FILE:{name}.py::area",
                "type": "Function",
                "metadata": {"name": "area", "docstring": "the area"},
            }
            for name in ("z", "a", "m")
        ]
        q = _querier(tmp_path, nodes)
        ids = [n["id"] for n in q.search_docs(["area"])]
        assert ids == sorted(ids)
        assert [n["id"] for n in q.search_docs(["area"], lang="python")] == ids

    def test_a_related_test_in_another_file_is_not_the_symbol_s(self, tmp_path):
        """A test function named for the symbol counts where it is in the
        symbol's file, not in a file whose path holds that one's."""
        nodes = [
            {
                "id": f"FILE:{path}::{name}",
                "type": "Function",
                "metadata": {"name": name},
            }
            for path, name in (
                ("a.py", "load"),
                ("a.py", "test_load_twice"),
                ("data.py", "test_load_once"),
            )
        ]
        q = _querier(tmp_path, nodes)
        assert q._find_related_tests("a.py", "load") == ["FILE:a.py::test_load_twice"]

    def test_a_path_pattern_of_several_directories_matches_their_run(self, tmp_path):
        graph = {
            "schema_version": 2,
            "nodes": [
                {"id": f"FILE:{path}::f", "type": "Function", "metadata": {"name": "f"}}
                for path in ("pkg/src/integ/case.py", "src/x/integ/case.py")
            ],
            "edges": [],
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(graph))
        config = DescryConfig(project_root=tmp_path)
        config.test_path_patterns = ("/src/integ/",)
        q = GraphQuerier(str(path), config)
        q._ensure_filter_indices()
        assert q._index_is_test == {"FILE:pkg/src/integ/case.py::f"}

    def test_an_empty_suffix_list_marks_no_file_by_suffix(self, tmp_path):
        """`file_suffixes = []` is the user's list: `a_test.go` is source."""
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:pkg/a_test.go::f",
                    "type": "Function",
                    "metadata": {"name": "f"},
                }
            ],
            "edges": [],
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(graph))
        config = DescryConfig(project_root=tmp_path)
        config.test_file_suffixes = ()
        q = GraphQuerier(str(path), config)
        q._ensure_filter_indices()
        assert q._index_is_test == set()

    def test_get_callers(self, sample_graph):
        """Should find callers of a function."""
        q = GraphQuerier(sample_graph, DescryConfig())
        callers = q.get_callers("helper")
        assert len(callers) == 1
        assert "main" in callers[0]

    def test_get_callees(self, sample_graph):
        """Should find callees of a function."""
        q = GraphQuerier(sample_graph, DescryConfig())
        callees = q.get_callees("FILE:test.py::main")
        assert len(callees) == 1
        assert "helper" in callees[0]

    def test_callers_of_a_candidate_mark_the_call_as_one_of_several(self):
        """An unresolved edge's candidates give the call to each named
        definition, marked as one of N."""
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:a.py::run",
                    "type": "Function",
                    "metadata": {"name": "run"},
                },
                {
                    "id": "FILE:x.py::work",
                    "type": "Function",
                    "metadata": {"name": "work"},
                },
                {
                    "id": "FILE:y.py::work",
                    "type": "Function",
                    "metadata": {"name": "work"},
                },
            ],
            "edges": [
                {
                    "source": "FILE:a.py::run",
                    "target": "REF:work",
                    "relation": "CALLS",
                    "metadata": {
                        "lineno": 3,
                        "candidates": ["FILE:x.py::work", "FILE:y.py::work"],
                    },
                }
            ],
        }
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            path = f.name
        q = GraphQuerier(path, DescryConfig())
        assert q.get_callers("work") == ["FILE:a.py::run"]
        assert q.get_ambiguous_callers("work") == {"FILE:a.py::run": 2}
        assert q.get_callees("FILE:a.py::run") == ["REF:work"]
        assert q.get_ambiguous_callees("FILE:a.py::run") == {"REF:work": 2}

    def test_search_docs(self, sample_graph):
        """Should search by keywords."""
        q = GraphQuerier(sample_graph, DescryConfig())
        results = q.search_docs(["helper"])
        assert len(results) > 0
        assert any(r["metadata"]["name"] == "helper" for r in results)


class TestFindTraitImpls:
    """Tests for find_trait_impls() method."""

    @pytest.fixture
    def trait_impl_graph(self):
        """Create a graph with trait implementations."""
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:auth.rs::TokenAuth::from_request_parts",
                    "type": "Method",
                    "metadata": {
                        "name": "from_request_parts",
                        "trait_impl": "FromRequestParts",
                        "signature": "fn from_request_parts(...)",
                        "lineno": 10,
                    },
                },
                {
                    "id": "FILE:auth.rs::OptionalAuth::from_request_parts",
                    "type": "Method",
                    "metadata": {
                        "name": "from_request_parts",
                        "trait_impl": "FromRequestParts",
                        "signature": "fn from_request_parts(...)",
                        "lineno": 30,
                    },
                },
                {
                    "id": "FILE:pagination.rs::Pagination::from_request_parts",
                    "type": "Method",
                    "metadata": {
                        "name": "from_request_parts",
                        "trait_impl": "FromRequestParts",
                        "signature": "fn from_request_parts(...)",
                        "lineno": 5,
                    },
                },
                {
                    "id": "FILE:service.rs::MyService::call",
                    "type": "Method",
                    "metadata": {
                        "name": "call",
                        "trait_impl": "Service",
                        "signature": "fn call(...)",
                        "lineno": 20,
                    },
                },
                {
                    "id": "FILE:server.rs::Server::new",
                    "type": "Method",
                    "metadata": {
                        "name": "new",
                        # No trait_impl - inherent impl
                        "signature": "fn new() -> Self",
                        "lineno": 15,
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

    def test_find_all_impls_of_method(self, trait_impl_graph):
        """Should find all trait implementations of a method."""
        q = GraphQuerier(trait_impl_graph, DescryConfig())
        results = q.find_trait_impls("from_request_parts")

        assert len(results) == 3
        names = {r["id"].split("::")[-2] for r in results}
        assert names == {"TokenAuth", "OptionalAuth", "Pagination"}

    def test_find_impls_filtered_by_trait(self, trait_impl_graph):
        """Should filter by trait name."""
        q = GraphQuerier(trait_impl_graph, DescryConfig())
        results = q.find_trait_impls("call", "Service")

        assert len(results) == 1
        assert "MyService" in results[0]["id"]

    def test_find_impls_excludes_inherent_methods(self, trait_impl_graph):
        """Should not return inherent impl methods."""
        q = GraphQuerier(trait_impl_graph, DescryConfig())
        results = q.find_trait_impls("new")

        assert len(results) == 0  # "new" has no trait_impl

    def test_find_impls_no_matches(self, trait_impl_graph):
        """Should return empty list for non-existent methods."""
        q = GraphQuerier(trait_impl_graph, DescryConfig())
        results = q.find_trait_impls("nonexistent_method")

        assert len(results) == 0


HANDLE = "FILE:api.rs::handle_request"
PROCESS = "FILE:api.rs::process_data"
VALIDATE = "FILE:auth.rs::validate_token"


class TestFindCallPath:
    """Tests for find_call_path_result() method."""

    @pytest.fixture
    def call_path_graph(self):
        """Create a graph with call relationships."""
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:api.rs::handle_request",
                    "type": "Function",
                    "metadata": {"name": "handle_request", "lineno": 10},
                },
                {
                    "id": "FILE:api.rs::process_data",
                    "type": "Function",
                    "metadata": {"name": "process_data", "lineno": 30},
                },
                {
                    "id": "FILE:auth.rs::validate_token",
                    "type": "Function",
                    "metadata": {"name": "validate_token", "lineno": 5},
                },
                {
                    "id": "FILE:db.rs::save_result",
                    "type": "Function",
                    "metadata": {"name": "save_result", "lineno": 20},
                },
            ],
            "edges": [
                # handle_request -> process_data -> validate_token
                # handle_request -> process_data -> save_result
                {
                    "source": "FILE:api.rs::handle_request",
                    "target": "FILE:api.rs::process_data",
                    "relation": "CALLS_RESOLVED",
                    "metadata": {"lineno": 15},
                },
                {
                    "source": "FILE:api.rs::process_data",
                    "target": "FILE:auth.rs::validate_token",
                    "relation": "CALLS_RESOLVED",
                    "metadata": {"lineno": 35},
                },
                {
                    "source": "FILE:api.rs::process_data",
                    "target": "FILE:db.rs::save_result",
                    "relation": "CALLS_RESOLVED",
                    "metadata": {"lineno": 40},
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            f.flush()
            yield f.name

        os.unlink(f.name)

    def test_structured_flow_stops_at_the_target_and_reports_the_path(
        self, call_path_graph
    ):
        q = GraphQuerier(call_path_graph, DescryConfig())

        tree = q.flow_data(HANDLE, target="FILE:auth.rs::validate_token")

        assert tree["path_to_target"] == [
            "handle_request",
            "process_data",
            "validate_token",
        ]
        process_data = tree["root"]["children"][0]
        assert [c["name"] for c in process_data["children"]] == ["validate_token"]
        assert "path_to_target" not in q.flow_data(HANDLE)

    def test_find_direct_path(self, call_path_graph):
        """Should find direct call path."""
        q = GraphQuerier(call_path_graph, DescryConfig())
        path = q.find_call_path_result(HANDLE, PROCESS)["path"]

        assert len(path) == 1
        assert path[0]["caller_id"] == HANDLE
        assert path[0]["callee_id"] == PROCESS

    def test_find_two_hop_path(self, call_path_graph):
        """Should find multi-hop path."""
        q = GraphQuerier(call_path_graph, DescryConfig())
        path = q.find_call_path_result(HANDLE, VALIDATE)["path"]

        assert len(path) == 2
        assert path[0]["caller_id"] == HANDLE
        assert path[0]["callee_id"] == PROCESS
        assert path[1]["caller_id"] == PROCESS
        assert path[1]["callee_id"] == VALIDATE

    def test_path_includes_call_line(self, call_path_graph):
        """Should include call line information."""
        q = GraphQuerier(call_path_graph, DescryConfig())
        path = q.find_call_path_result(HANDLE, PROCESS)["path"]

        assert path[0]["call_line"] == 15

    def test_no_path_returns_empty(self, call_path_graph):
        """Should return empty list when no path exists."""
        q = GraphQuerier(call_path_graph, DescryConfig())
        path = q.find_call_path_result(VALIDATE, HANDLE)["path"]

        # No forward path from validate_token to handle_request
        assert len(path) == 0

    def test_backward_direction(self, call_path_graph):
        """Should find backward path (who calls me)."""
        q = GraphQuerier(call_path_graph, DescryConfig())
        path = q.find_call_path_result(VALIDATE, HANDLE, direction="backward")["path"]

        # Backward: validate_token is called by process_data is called by handle_request
        assert len(path) == 2
        # In backward direction, roles are swapped for hop info
        assert path[0]["callee_id"] == VALIDATE
        assert path[1]["callee_id"] == PROCESS

    def test_max_depth_limit(self, call_path_graph):
        """Should respect max_depth limit."""
        q = GraphQuerier(call_path_graph, DescryConfig())
        path = q.find_call_path_result(HANDLE, VALIDATE, max_depth=1)["path"]

        # Path requires 2 hops but max_depth is 1
        assert len(path) == 0


class TestExpandCallees:
    """Tests for expand_callees feature in descry context."""

    @pytest.fixture
    def graph_with_callees(self, tmp_path):
        """Create a graph with function that calls other functions, with actual source files."""
        # Create actual source file with all functions
        source_file = tmp_path / "auth.rs"
        source_content = """\
// File: auth.rs

/// Authenticates a request and returns the user.
fn authenticate(request: &Request) -> Result<User> {
    let token = extract_token(request)?;
    let claims = validate_token(&token)?;
    let user = fetch_user(&claims.user_id)?;
    Ok(user)
}

/// Validates JWT token and extracts claims.
fn validate_token(token: &str) -> Result<Claims> {
    let key = get_signing_key();
    let claims = decode_jwt(token, &key)?;
    if claims.is_expired() {
        return Err(AuthError::Expired);
    }
    Ok(claims)
}

/// Fetches user from database.
fn fetch_user(user_id: &str) -> Result<User> {
    let conn = get_db_connection()?;
    let user = conn.query_user(user_id)?;
    Ok(user)
}
"""
        source_file.write_text(source_content)

        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": f"FILE:{source_file}",
                    "type": "File",
                    "metadata": {"path": str(source_file), "name": "auth.rs"},
                },
                {
                    "id": f"FILE:{source_file}::authenticate",
                    "type": "Function",
                    "metadata": {
                        "name": "authenticate",
                        "signature": "fn authenticate(request: &Request) -> Result<User>",
                        "token_count": 50,
                        "lineno": 4,
                        "end_lineno": 9,
                    },
                },
                {
                    "id": f"FILE:{source_file}::validate_token",
                    "type": "Function",
                    "metadata": {
                        "name": "validate_token",
                        "signature": "fn validate_token(token: &str) -> Result<Claims>",
                        "token_count": 80,
                        "lineno": 12,
                        "end_lineno": 18,
                        "docstring": "Validates JWT token and extracts claims.",
                    },
                },
                {
                    "id": f"FILE:{source_file}::fetch_user",
                    "type": "Function",
                    "metadata": {
                        "name": "fetch_user",
                        "signature": "fn fetch_user(user_id: &str) -> Result<User>",
                        "token_count": 60,
                        "lineno": 21,
                        "end_lineno": 25,
                        "docstring": "Fetches user from database.",
                    },
                },
            ],
            "edges": [
                {
                    "source": f"FILE:{source_file}",
                    "target": f"FILE:{source_file}::authenticate",
                    "relation": "DEFINES",
                },
                {
                    "source": f"FILE:{source_file}",
                    "target": f"FILE:{source_file}::validate_token",
                    "relation": "DEFINES",
                },
                {
                    "source": f"FILE:{source_file}",
                    "target": f"FILE:{source_file}::fetch_user",
                    "relation": "DEFINES",
                },
                # authenticate calls validate_token and fetch_user
                {
                    "source": f"FILE:{source_file}::authenticate",
                    "target": f"FILE:{source_file}::validate_token",
                    "relation": "CALLS",
                },
                {
                    "source": f"FILE:{source_file}::authenticate",
                    "target": f"FILE:{source_file}::fetch_user",
                    "relation": "CALLS",
                },
            ],
        }

        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(graph))
        return str(graph_file), str(source_file)

    def test_expand_callees_shows_full_source(self, graph_with_callees, tmp_path):
        """expand_callees should show full source of direct callees."""
        graph_file, source_file = graph_with_callees
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        result = _context(
            q,
            f"FILE:{source_file}::authenticate",
            expand_callees=True,
            callee_budget=2000,
        )

        # Should include the context
        assert "Context for" in result
        assert "authenticate" in result

        # Should include expanded callees section with actual source
        assert "Expanded Callees" in result
        assert "validate_token" in result
        assert "fetch_user" in result
        # Should show actual source code from callees
        assert "decode_jwt" in result  # from validate_token
        assert "get_db_connection" in result  # from fetch_user

    def test_expand_callees_disabled_by_default(self, graph_with_callees):
        """expand_callees should be disabled by default."""
        graph_file, source_file = graph_with_callees
        q = GraphQuerier(graph_file, DescryConfig())

        result = _context(q, f"FILE:{source_file}::authenticate")

        # Should NOT have Expanded Callees section
        assert "Expanded Callees" not in result

    def test_expand_callees_respects_budget(self, graph_with_callees):
        """expand_callees should respect the token budget."""
        graph_file, source_file = graph_with_callees
        q = GraphQuerier(graph_file, DescryConfig())

        # With very small budget, should skip callees
        expanded = q._expand_callees_full(
            f"FILE:{source_file}::authenticate", budget=10
        )

        # Both callees (80 + 60 = 140 tokens) exceed budget of 10
        assert len(expanded) == 0


class TestContextByNodeId:
    @pytest.fixture
    def graph_with_symbols(self, tmp_path):
        """A graph of two functions whose names share a prefix."""
        # Create source file
        source_file = tmp_path / "auth.rs"
        source_content = """\
/// Validates a JWT token
fn validate_token(token: &str) -> Result<Claims> {
    decode_jwt(token)
}

/// Validates token format
fn validate_token_format(token: &str) -> bool {
    token.starts_with("Bearer ")
}
"""
        source_file.write_text(source_content)

        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": f"FILE:{source_file}",
                    "type": "File",
                    "metadata": {"path": str(source_file), "name": "auth.rs"},
                },
                {
                    "id": f"FILE:{source_file}::validate_token",
                    "type": "Function",
                    "metadata": {
                        "name": "validate_token",
                        "signature": "fn validate_token(token: &str) -> Result<Claims>",
                        "token_count": 30,
                        "lineno": 2,
                        "end_lineno": 4,
                        "docstring": "Validates a JWT token",
                    },
                },
                {
                    "id": f"FILE:{source_file}::validate_token_format",
                    "type": "Function",
                    "metadata": {
                        "name": "validate_token_format",
                        "signature": "fn validate_token_format(token: &str) -> bool",
                        "token_count": 25,
                        "lineno": 7,
                        "end_lineno": 9,
                        "docstring": "Validates token format",
                    },
                },
            ],
            "edges": [],
        }

        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(graph))
        return str(graph_file), str(source_file)

    def test_exact_node_id_works(self, graph_with_symbols, tmp_path):
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        result = _context(q, f"FILE:{source_file}::validate_token")

        assert "Context for" in result
        assert "validate_token" in result
        assert "Matched" not in result


class TestFullSourceBypass:
    """Tests for the full=True truncation bypass."""

    @pytest.fixture
    def large_function_graph(self, tmp_path):
        """Create a graph with a large function and corresponding source file."""
        # Create a large source file
        source_file = tmp_path / "large.rs"
        lines = ["fn large_function() {\n"]
        for i in range(200):
            lines.append(f"    let x{i} = {i};\n")
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
                        "end_lineno": 202,
                        "token_count": 2000,  # Large enough to trigger truncation
                    },
                },
            ],
            "edges": [],
        }

        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(graph))
        return str(graph_file), str(source_file)

    def test_full_false_truncates_large_function(self, large_function_graph, tmp_path):
        """Should truncate large functions when full=False."""
        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        source = q.get_smart_source(source_file, 1, 202, 2000, full=False)

        # Should contain truncation marker
        assert "lines omitted" in source
        # Should not contain all 200 lines
        assert source.count("let x") < 200

    def test_full_true_shows_complete_source(self, large_function_graph, tmp_path):
        """Should show complete source when full=True."""
        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        source = q.get_smart_source(source_file, 1, 202, 2000, full=True)

        # Should NOT contain truncation marker
        assert "lines omitted" not in source
        # Should contain all 200 lines
        assert source.count("let x") == 200

    def test_context_prompt_passes_full_parameter(self, large_function_graph, tmp_path):
        """Should pass full parameter through get_context_prompt."""
        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        node_id = f"FILE:{source_file}::large_function"

        # With full=False, should truncate
        context_truncated = _context(q, node_id, full=False)
        assert "lines omitted" in context_truncated

        # With full=True, should not truncate
        context_full = _context(q, node_id, full=True)
        assert "lines omitted" not in context_full


def _fn(node_id, **meta):
    return {
        "id": node_id,
        "type": "Function",
        "metadata": {"name": node_id.split("::")[-1], **meta},
    }


def _calls(source, target):
    return {"source": source, "target": target, "relation": "CALLS_RESOLVED"}


def _source_shown(q, node_id):
    """Everything the querier prints that can carry the node's source."""
    rel = node_id.split("::")[0].removeprefix("FILE:")
    return "\n".join(
        [
            _context(q, node_id),
            _flow(q, node_id.split("::")[-1], inline_threshold=100),
            q.get_source_segment(rel, 1, 2),
        ]
    )


def _querier(tmp_path, nodes, edges=(), **config):
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(
        json.dumps({"schema_version": 2, "nodes": nodes, "edges": list(edges)})
    )
    settings = {
        "max_depth": 3,
        "max_nodes": 100,
        "max_children_per_level": 10,
        "max_callers_shown": 15,
        "query_timeout_ms": 4000,
        "test_path_patterns": _DEFAULT_TEST_PATH_PATTERNS,
        "test_file_suffixes": _DEFAULT_TEST_FILE_SUFFIXES,
        "syntax_lang_map": _DEFAULT_SYNTAX_LANG_MAP,
        "project_root": tmp_path,
        "max_file_bytes": 0,
    }
    settings.update(config)
    return GraphQuerier(str(graph_file), config=SimpleNamespace(**settings))


class TestSourcePathsResolveAgainstProjectRoot:
    @pytest.fixture
    def querier(self, tmp_path):
        root = tmp_path / "project"
        (root / "src").mkdir(parents=True)
        (root / "src" / "foo.py").write_text(
            "def caller():\n    return helper()\n\n\ndef helper():\n    return 42\n"
        )
        (tmp_path / "elsewhere").mkdir()
        nodes = [
            _fn("FILE:src/foo.py::caller", lineno=1, end_lineno=2, token_count=10),
            _fn("FILE:src/foo.py::helper", lineno=5, end_lineno=6, token_count=10),
        ]
        edges = [
            {
                **_calls("FILE:src/foo.py::caller", "FILE:src/foo.py::helper"),
                "metadata": {"lineno": 2},
            }
        ]
        return _querier(tmp_path, nodes, edges, project_root=root)

    @pytest.mark.parametrize("cwd", ["elsewhere", "project/src"])
    def test_source_is_read_whatever_the_working_directory(
        self, querier, tmp_path, monkeypatch, cwd
    ):
        monkeypatch.chdir(tmp_path / cwd)

        context = _context(querier, "FILE:src/foo.py::caller", expand_callees=True)
        assert "return helper()" in context
        assert "return 42" in context

        flow = _flow(querier, "FILE:src/foo.py::caller", inline_threshold=100)
        assert "return 42" in flow
        # Inlined code is indented as its fence is, at every level
        fence = None
        for line in flow.splitlines():
            if line.lstrip().startswith("```"):
                fence = None if fence is not None else line[: line.index("`")]
            elif fence is not None:
                assert line.startswith(fence), line
        tree = querier.flow_data("FILE:src/foo.py::caller", inline_threshold=100)
        assert "return helper()" in tree["root"]["code"]

        hop = querier.find_call_path_result(
            "FILE:src/foo.py::caller", "FILE:src/foo.py::helper"
        )["path"][0]
        assert hop["file_path"] == "src/foo.py"
        assert "return helper()" in querier.call_snippet("src/foo.py", hop["call_line"])

    def test_a_file_over_the_size_limit_is_not_read(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / "big.py").write_text("def big():\n    return 'BODY'\n")
        nodes = [_fn("FILE:big.py::big", lineno=1, end_lineno=2, token_count=10)]

        within = _querier(tmp_path, nodes, project_root=root, max_file_bytes=1024)
        assert "BODY" in _context(within, "FILE:big.py::big")
        over = _querier(tmp_path, nodes, project_root=root, max_file_bytes=8)
        assert "BODY" not in _context(over, "FILE:big.py::big")
        assert "BODY" not in _flow(over, "FILE:big.py::big", inline_threshold=100)

    @pytest.mark.parametrize("link", [False, True])
    def test_a_node_path_leaving_the_project_is_not_read(self, tmp_path, link):
        root = tmp_path / "project"
        root.mkdir()
        (tmp_path / "secret.py").write_text("def leaked():\n    return 'TOP SECRET'\n")
        rel = "../secret.py"
        if link:
            (root / "inside.py").symlink_to(tmp_path / "secret.py")
            rel = "inside.py"
        node_id = f"FILE:{rel}::leaked"
        q = _querier(
            tmp_path,
            [_fn(node_id, lineno=1, end_lineno=2, token_count=10)],
            [],
            project_root=root,
        )

        assert "TOP SECRET" not in _context(q, node_id)
        assert "TOP SECRET" not in _flow(q, node_id, inline_threshold=100)
        assert "TOP SECRET" not in q.get_source_segment(rel, 1, 2)


class TestLanguageFilter:
    FILES = (
        "a/main.go",
        "a/App.java",
        "a/app.rb",
        "a/index.php",
        "a/Program.cs",
        "a/engine.cpp",
        "a/widget.dart",
        "a/view.ts",
        "a/tool.py",
        "a/lib.rs",
    )

    @pytest.fixture
    def querier(self, tmp_path):
        return _querier(tmp_path, [_fn(f"FILE:{f}::handler") for f in self.FILES])

    @pytest.mark.parametrize(
        ("lang", "expected"),
        [
            ("go", "a/main.go"),
            ("java", "a/App.java"),
            ("ruby", "a/app.rb"),
            ("php", "a/index.php"),
            ("cs", "a/Program.cs"),
            ("csharp", "a/Program.cs"),
            ("cpp", "a/engine.cpp"),
            ("dart", "a/widget.dart"),
            ("ts", "a/view.ts"),
            ("typescript", "a/view.ts"),
            ("py", "a/tool.py"),
            ("Python", "a/tool.py"),
            ("rust", "a/lib.rs"),
        ],
    )
    def test_lang_keeps_only_that_languages_files(self, querier, lang, expected):
        hits = querier.search_docs(["handler"], lang=lang)
        assert [n["id"] for n in hits] == [f"FILE:{expected}::handler"]

    @pytest.mark.parametrize("lang", ["kotlin", "klingon"])
    def test_lang_with_no_files_matches_nothing(self, querier, lang):
        assert querier.search_docs(["handler"], lang=lang) == []

    def test_none_does_not_filter(self, querier):
        assert len(querier.search_docs(["handler"], lang=None)) == len(self.FILES)

    @pytest.mark.parametrize(
        ("kind", "found"), [("function", len(FILES)), ("class", 0), ("struct", 0)]
    )
    def test_type_keeps_only_that_type(self, querier, kind, found):
        """A type the graph has none of, or no type descry has, matches
        nothing, as an unknown language does."""
        assert len(querier.search_docs(["handler"], symbol_type=kind)) == found


class TestCallPathDepthAndOutcome:
    def _chain(self, tmp_path, length, **config):
        nodes = [_fn(f"FILE:x.py::n{i}") for i in range(length)]
        nodes.append(_fn("FILE:x.py::island"))
        edges = [
            _calls(f"FILE:x.py::n{i}", f"FILE:x.py::n{i + 1}")
            for i in range(length - 1)
        ]
        return _querier(tmp_path, nodes, edges, **config)

    def test_max_depth_argument_bounds_the_search_not_the_tree_depth_config(
        self, tmp_path
    ):
        q = self._chain(tmp_path, 9, max_depth=3)
        assert (
            len(
                q.find_call_path_result("FILE:x.py::n0", "FILE:x.py::n8", max_depth=10)[
                    "path"
                ]
            )
            == 8
        )
        assert (
            q.find_call_path_result("FILE:x.py::n0", "FILE:x.py::n8", max_depth=7)[
                "path"
            ]
            == []
        )

    def test_max_depth_is_capped_at_the_hard_ceiling(self, tmp_path):
        q = self._chain(tmp_path, MAX_PATH_DEPTH + 5, max_nodes=10_000)
        at_ceiling = q.find_call_path_result(
            "FILE:x.py::n0", f"FILE:x.py::n{MAX_PATH_DEPTH}", max_depth=10**9
        )
        assert at_ceiling["status"] == "found"
        assert at_ceiling["max_depth"] == MAX_PATH_DEPTH
        beyond = q.find_call_path_result(
            "FILE:x.py::n0", f"FILE:x.py::n{MAX_PATH_DEPTH + 1}", max_depth=10**9
        )
        assert beyond["status"] == "not_found"

    def test_status_tells_a_miss_from_a_search_that_was_cut_off(self, tmp_path):
        q = self._chain(tmp_path, 30)
        assert (
            q.find_call_path_result("FILE:x.py::n0", "FILE:x.py::n5")["status"]
            == "found"
        )
        assert (
            q.find_call_path_result("FILE:x.py::n0", "FILE:x.py::island")["status"]
            == "not_found"
        )

        tight = self._chain(tmp_path, 30, max_nodes=5)
        assert (
            tight.find_call_path_result(
                "FILE:x.py::n0", "FILE:x.py::n29", max_depth=40
            )["status"]
            == "found"
        )

        slow = self._chain(tmp_path, 30, query_timeout_ms=-1)
        assert (
            slow.find_call_path_result("FILE:x.py::n0", "FILE:x.py::n29")["status"]
            == "timeout"
        )

    def test_a_limit_of_0_sets_none(self, tmp_path):
        """`query_ms = 0` and `max_nodes = 0` lift the limit, as in context,
        flow and path alike."""
        free = self._chain(tmp_path, 30, query_timeout_ms=0, max_nodes=0)
        assert free.find_call_path_result(
            "FILE:x.py::n0", "FILE:x.py::n29", max_depth=40
        )["status"] == ("found")
        flow = free.flow_data("FILE:x.py::n0", depth=40)
        assert flow.get("cut") is None


class TestCalleeExpansion:
    """Deep `context` opens each callee at the first level that reaches it,
    while the token budget covers it, and lists every callee it does not
    open."""

    @pytest.fixture
    def querier(self, tmp_path):
        # start -> x (120 toks), w; x -> y; w -> z -> y
        sizes = {"start": 10, "x": 120, "w": 20, "z": 20, "y": 20}
        (tmp_path / "m.py").write_text(
            "".join(f"def {n}():\n    pass\n" for n in sizes)
        )
        nodes = [
            _fn(
                f"FILE:m.py::{n}", lineno=2 * i + 1, end_lineno=2 * i + 2, token_count=t
            )
            for i, (n, t) in enumerate(sizes.items())
        ]
        pairs = [("start", "x"), ("start", "w"), ("x", "y"), ("w", "z"), ("z", "y")]
        edges = [_calls(f"FILE:m.py::{a}", f"FILE:m.py::{b}") for a, b in pairs]
        return _querier(tmp_path, nodes, edges, project_root=tmp_path)

    def expand(self, q, budget, inline_threshold=200):
        blocks, notes, _ = q._expand_callees(
            "FILE:m.py::start", 3, budget, inline_threshold
        )
        return "\n".join([*(line for block in blocks for line in block), *notes])

    def test_a_mark_points_at_an_entry(self, querier):
        """Under a budget that cannot open `x`, `y` opens under `z`, not as a
        mark of an entry no one drew."""
        text = self.expand(querier, 100)
        assert "- `x(...)` (120 toks)" in text
        assert "**y** (20 toks):" in text
        assert "(at level" not in text
        assert "*(token budget reached)*" in text

    def test_a_symbol_opens_once_at_its_level(self, querier):
        text = self.expand(querier, 1000)
        assert text.count("**y** (20 toks):") == 1
        assert "- `y` (at level 2)" in text

    def test_depth_one_holds_to_the_budget(self, querier):
        text = _context(querier, "FILE:m.py::start", max_tokens=5)
        assert "*(token budget reached)*" in text
        assert "- `w(...)` (20 toks)" in text

    def test_the_timeout_is_said(self, querier, monkeypatch):
        import itertools

        import descry.query

        ticks = itertools.count()
        monkeypatch.setattr(
            descry.query, "time", SimpleNamespace(time=lambda: next(ticks))
        )
        querier._timeout_ms = 1500
        text = self.expand(querier, 1000)
        assert "*(timeout reached)*" in text
        # The symbol's own callees are listed whatever the clock says
        assert "x" in text and "**w**" in text


class TestLangMapValueInTruncation:
    @pytest.mark.parametrize("lang", ["(", "c++", "[a-"])
    def test_source_block_is_truncated_whatever_the_fence_tag(self, tmp_path, lang):
        lines = [f"line {i}" for i in range(200)]
        lines[50] = "```{source_lang}"
        (tmp_path / "x.py").write_text("\n".join(lines) + "\n")
        node = _fn("FILE:x.py::f", lineno=1, end_lineno=200, token_count=2000)
        q = _querier(
            tmp_path, [node], project_root=tmp_path, syntax_lang_map={".py": lang}
        )
        out = _context(q, "FILE:x.py::f", full=True, max_output_tokens=50)
        assert f"```{lang}" in out
        assert "(170 lines omitted, max_output_tokens limit)" in out
        assert "line 199" not in out


class TestUnresolvedCallsInPathsAndFlows:
    """An unresolved call reaches one of the candidates its edge records,
    and is shown as such; a definition the edge does not name is not reached."""

    @pytest.fixture
    def querier(self, tmp_path):
        nodes = [
            _fn("FILE:a.py::start"),
            _fn("FILE:a.py::helper"),
            _fn("FILE:b.py::helper"),
            _fn("FILE:c.py::helper"),
            _fn("FILE:b.py::leaf"),
            _fn("FILE:a.py::bare"),
        ]
        edges = [
            {
                "source": "FILE:a.py::start",
                "target": "REF:helper",
                "relation": "CALLS",
                "metadata": {
                    "lineno": 2,
                    "candidates": ["FILE:b.py::helper", "FILE:c.py::helper"],
                },
            },
            _calls("FILE:b.py::helper", "FILE:b.py::leaf"),
            {"source": "FILE:a.py::bare", "target": "REF:leaf", "relation": "CALLS"},
        ]
        return _querier(tmp_path, nodes, edges, project_root=tmp_path)

    def test_context_marks_a_caller_that_may_reach_it(self, querier, tmp_path):
        (tmp_path / "b.py").write_text("def helper():\n    leaf()\n")
        context = _context(querier, "FILE:b.py::helper")
        assert "- `start` in a.py (0 toks) (one of 2 candidates)" in context

    @pytest.mark.parametrize("asked", ["Store::get", "Store.get"])
    def test_dotted_name_is_the_qualified_one(self, tmp_path, asked):
        nodes = [
            _fn("FILE:a.py::Store::get"),
            _fn("FILE:a.py::Cache::get"),
            _fn("FILE:a.py::one"),
            _fn("FILE:a.py::two"),
            _fn("FILE:a.py::three"),
        ]
        edges = [
            _calls("FILE:a.py::one", "FILE:a.py::Store::get"),
            _calls("FILE:a.py::two", "FILE:a.py::Cache::get"),
            {
                "source": "FILE:a.py::three",
                "target": "REF:Store.get",
                "relation": "CALLS",
                "metadata": {"lineno": 3},
            },
        ]
        querier = _querier(tmp_path, nodes, edges)
        # `three`'s call is left unresolved: a call of the name, not certain.
        assert querier.get_callers(asked) == ["FILE:a.py::one", "FILE:a.py::three"]
        assert querier.get_ambiguous_callers(asked) == {"FILE:a.py::three": 0}
        found = querier.find_nodes_by_name(asked)
        assert [node["id"] for node in found] == ["FILE:a.py::Store::get"]

    def test_qualified_name_is_no_part_of_a_longer_one(self, tmp_path):
        names = ("Foo::m", "Foo::mid", "BigFoo::m", "one", "two", "three")
        nodes = [_fn(f"FILE:a.py::{name}") for name in names]
        edges = [
            _calls("FILE:a.py::one", "FILE:a.py::Foo::m"),
            _calls("FILE:a.py::two", "FILE:a.py::Foo::mid"),
            _calls("FILE:a.py::three", "FILE:a.py::BigFoo::m"),
        ]
        querier = _querier(tmp_path, nodes, edges)
        assert querier.get_callers("Foo::m") == ["FILE:a.py::one"]
        assert querier.find_nodes_by_name("Foo::mi") == []
        found = querier.find_nodes_by_name("Foo::mi", fuzzy=True)
        assert [node["id"] for node in found] == ["FILE:a.py::Foo::mid"]

    def test_accessor_is_found_by_the_name_source_gives_it(self, tmp_path):
        """An accessor is filed as `get size` and named `size`: either
        finds it, alone or after its type."""
        nodes = [
            _fn("FILE:a.ts::Box::get size", name="size"),
            _fn("FILE:a.ts::Box::set size", name="size"),
            _fn("FILE:a.ts::Box::get_size"),
            _fn("FILE:a.ts::BigBox::get size", name="size"),
        ]
        querier = _querier(tmp_path, nodes, [])

        def found(name):
            return [
                node["id"].partition("::")[2]
                for node in querier.find_nodes_by_name(name)
            ]

        assert found("get size") == ["Box::get size", "BigBox::get size"]
        assert found("Box::get size") == ["Box::get size"]
        assert found("Box::size") == ["Box::get size", "Box::set size"]
        assert found("Box.size") == ["Box::get size", "Box::set size"]
        assert found("get_size") == ["Box::get_size"]

    def test_qualified_name_asks_for_that_method_alone(self, tmp_path):
        nodes = [
            _fn("FILE:a.py::Store::get"),
            _fn("FILE:a.py::Cache::get"),
            _fn("FILE:a.py::reads_store"),
            _fn("FILE:a.py::reads_cache"),
            _fn("FILE:a.py::reads_either"),
        ]
        edges = [
            _calls("FILE:a.py::reads_store", "FILE:a.py::Store::get"),
            _calls("FILE:a.py::reads_cache", "FILE:a.py::Cache::get"),
            {
                "source": "FILE:a.py::reads_either",
                "target": "REF:held.get",
                "relation": "CALLS",
                "metadata": {
                    "lineno": 9,
                    "candidates": ["FILE:a.py::Store::get", "FILE:a.py::Cache::get"],
                },
            },
        ]
        querier = _querier(tmp_path, nodes, edges)
        assert querier.get_callers("Store::get") == [
            "FILE:a.py::reads_either",
            "FILE:a.py::reads_store",
        ]
        assert querier.get_ambiguous_callers("Store::get") == {
            "FILE:a.py::reads_either": 2
        }
        assert len(querier.get_callers("get")) == 3

    def test_path_between_two_definitions_of_one_name(self, tmp_path):
        nodes = [_fn("FILE:a.py::run"), _fn("FILE:b.py::run")]
        edges = [_calls("FILE:a.py::run", "FILE:b.py::run")]
        found = _querier(tmp_path, nodes, edges).find_call_path_result(
            "FILE:a.py::run", "FILE:b.py::run"
        )
        assert [(hop["caller_id"], hop["callee_id"]) for hop in found["path"]] == [
            ("FILE:a.py::run", "FILE:b.py::run")
        ]

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_path_goes_through_a_candidate_and_says_so(self, querier, direction):
        ends = ("FILE:a.py::start", "FILE:b.py::leaf")
        if direction == "backward":
            ends = ends[::-1]
        found = querier.find_call_path_result(*ends, direction=direction)
        assert found["status"] == "found"
        hops = sorted(found["path"], key=lambda hop: hop["caller_id"])
        assert [(hop["caller_id"], hop["callee_id"]) for hop in hops] == [
            ("FILE:a.py::start", "FILE:b.py::helper"),
            ("FILE:b.py::helper", "FILE:b.py::leaf"),
        ]
        assert [hop.get("candidates") for hop in hops] == [2, None]

    def test_path_does_not_reach_a_definition_the_edge_leaves_out(self, querier):
        found = querier.find_call_path_result("FILE:a.py::start", "FILE:a.py::helper")
        assert found["status"] == "not_found"
        found = querier.find_call_path_result("FILE:a.py::bare", "FILE:b.py::leaf")
        assert found["status"] == "not_found"

    def test_backward_flow_marks_a_candidate_caller(self, querier):
        assert "**start** (0 toks) (one of 2 candidates)" in _flow(
            querier, "FILE:b.py::helper", direction="backward"
        )
        tree = querier.flow_data("FILE:b.py::helper", direction="backward")
        assert [(c["name"], c["candidates"]) for c in tree["root"]["children"]] == [
            ("start", 2)
        ]
        assert "**helper** (0 toks)\n" in _flow(
            querier, "FILE:b.py::leaf", direction="backward"
        )

    def test_flatten_follows_a_base_only_when_one_class_has_the_name(self, tmp_path):
        def cls(node_id):
            return {**_fn(node_id), "type": "Class"}

        nodes = [
            cls("FILE:a.py::Child"),
            cls("FILE:b.py::Base"),
            {**_fn("FILE:b.py::Base::run"), "type": "Method"},
            cls("FILE:c.py::Base"),
        ]
        edges = [
            {
                "source": "FILE:a.py::Child",
                "target": "REF:Base",
                "relation": "INHERITS",
            },
            {
                "source": "FILE:b.py::Base",
                "target": "FILE:b.py::Base::run",
                "relation": "DEFINES",
            },
        ]
        assert "run" not in _flatten(
            _querier(tmp_path, nodes, edges), "FILE:a.py::Child"
        )
        assert "run" in _flatten(
            _querier(tmp_path, nodes[:3], edges), "FILE:a.py::Child"
        )


class TestCallerOrder:
    def test_callers_come_back_sorted(self, tmp_path):
        callers = [f"FILE:m.py::caller_{i:02d}" for i in range(20)]
        q = _querier(
            tmp_path,
            [_fn("FILE:m.py::target"), *map(_fn, callers)],
            [_calls(c, "FILE:m.py::target") for c in callers],
        )
        assert q.get_callers("target") == callers


class TestCallersOfOneSymbol:
    """Of two symbols of one name, each has only its own callers."""

    @pytest.fixture
    def querier(self, tmp_path):
        from descry.generate import CodeGraphBuilder

        method = "    def parse(self):\n        return 1\n"
        (tmp_path / "a.py").write_text("class A:\n" + method)
        (tmp_path / "b.py").write_text("class B:\n" + method)
        (tmp_path / "use.py").write_text(
            "from a import A\nfrom b import B\n\n\n"
            "def only_a():\n    A().parse()\n\n\n"
            "def only_b():\n    B().parse()\n"
        )
        builder = CodeGraphBuilder(tmp_path)
        builder.process_directory()
        builder.export(str(tmp_path / "graph.json"))
        return GraphQuerier(
            str(tmp_path / "graph.json"), config=DescryConfig(project_root=tmp_path)
        )

    def test_the_count_is_of_its_callers(self, querier):
        meta = querier.nodes["FILE:a.py::A::parse"]["metadata"]
        assert meta["in_degree"] == 1
        brief = _context(querier, "FILE:a.py::A::parse", brief=True)
        assert "**Callers:** 1 " in brief

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_a_flow_of_depth_one_is_the_calls_one_level_away(self, querier, direction):
        start = (
            "FILE:use.py::only_a" if direction == "forward" else "FILE:a.py::A::parse"
        )
        tree = querier.flow_data(start, direction=direction, depth=1)
        names = [child["name"] for child in tree["root"]["children"]]
        assert ("parse" in names) if direction == "forward" else names == ["only_a"]
        assert all(not child["children"] for child in tree["root"]["children"])

    def test_context_and_backward_flow_list_its_callers(self, querier):
        for text in (
            _context(querier, "FILE:a.py::A::parse"),
            _flow(querier, "FILE:a.py::A::parse", direction="backward"),
        ):
            assert "only_a" in text
            assert "only_b" not in text


def test_a_call_with_one_candidate_is_not_one_of_several():
    assert GraphQuerier.one_of(1) == " (unresolved; the only candidate)"
    assert GraphQuerier.one_of(3) == " (one of 3 candidates)"
    assert GraphQuerier.one_of(None) == ""


class TestCertainBeforePossible:
    """A caller is certain only by a call resolved to the symbol; a list cut
    short keeps the certain ones."""

    def _graph(self, tmp_path, **config):
        nodes = [
            _fn("FILE:a.py::outer"),
            _fn("FILE:a.py::outer::imp"),
            _fn("FILE:b.py::imp"),
            _fn("FILE:c.py::sure", token_count=500),
            *(_fn(f"FILE:d.py::maybe{i}", token_count=1) for i in range(3)),
        ]
        for name, line in [
            ("a.py", "def outer():\n    pass\n"),
            ("b.py", "def imp():\n    pass\n"),
        ]:
            (tmp_path / name).write_text(line)
        unresolved = [
            {
                "source": source,
                "target": "REF:imp",
                "relation": "CALLS",
                "metadata": {"candidates": ["FILE:a.py::outer::imp", "FILE:b.py::imp"]},
            }
            for source in [
                "FILE:a.py::outer",
                *(f"FILE:d.py::maybe{i}" for i in range(3)),
            ]
        ]
        edges = [
            {
                "source": "FILE:a.py::outer",
                "target": "FILE:a.py::outer::imp",
                "relation": "DEFINES",
            },
            _calls("FILE:c.py::sure", "FILE:b.py::imp"),
            *unresolved,
        ]
        return _querier(tmp_path, nodes, edges, project_root=tmp_path, **config)

    def test_defining_a_symbol_is_not_calling_it(self, tmp_path):
        q = self._graph(tmp_path)
        assert q.get_ambiguous_callers("FILE:a.py::outer::imp")["FILE:a.py::outer"] == 2
        assert "**Callers:** 0 (+4 possible)" in _context(
            q, "FILE:a.py::outer::imp", brief=True
        )
        assert q.nodes["FILE:a.py::outer::imp"]["metadata"]["in_degree"] == 0

    def test_a_cut_list_keeps_the_certain_callers(self, tmp_path):
        q = self._graph(tmp_path, max_callers_shown=1, max_children_per_level=1)
        context = _context(q, "FILE:b.py::imp")
        assert "- `sure` in c.py" in context
        assert "(4 more possible callers)" in context
        flow = _flow(q, "FILE:b.py::imp", direction="backward", depth=1)
        assert "**sure**" in flow and "*(4 more possible callers)*" in flow
        tree = q.flow_data("FILE:b.py::imp", direction="backward", depth=1)
        assert [c["name"] for c in tree["root"]["children"]] == ["sure"]
        assert (tree["root"]["more"], tree["root"]["more_possible"]) == (4, 4)


def test_a_graph_is_counted_as_it_loads(tmp_path):
    """A graph an older descry wrote, with possible callers in its counts,
    is counted again."""
    nodes = [_fn("FILE:a.py::f", in_degree=7), _fn("FILE:a.py::g")]
    q = _querier(tmp_path, nodes, [_calls("FILE:a.py::g", "FILE:a.py::f")])
    assert q.nodes["FILE:a.py::f"]["metadata"]["in_degree"] == 1


def test_a_call_left_unresolved_is_one_of_the_name_only(tmp_path):
    """A call the resolver found no candidates for is listed by its name,
    marked apart from the callers resolved or narrowed to a definition; one
    with candidates reaches only those."""
    nodes = [
        _fn("FILE:a.py::list_templates"),
        _fn("FILE:c.py::Other::list_templates"),
        _fn("FILE:b.py::render"),
        _fn("FILE:b.py::narrowed"),
    ]
    edges = [
        {
            "source": "FILE:b.py::render",
            "target": "REF:loader.list_templates",
            "relation": "CALLS",
        },
        {
            "source": "FILE:b.py::narrowed",
            "target": "REF:held.list_templates",
            "relation": "CALLS",
            "metadata": {"candidates": ["FILE:c.py::Other::list_templates"]},
        },
    ]
    q = _querier(tmp_path, nodes, edges)
    callers, marks = q.ordered_callers("list_templates")
    assert callers == ["FILE:b.py::narrowed", "FILE:b.py::render"]
    assert marks == {"FILE:b.py::narrowed": 1, "FILE:b.py::render": 0}
    assert q.one_of(0) == " (unresolved; same name)"
    assert q.get_callers("FILE:a.py::list_templates") == []


def test_a_forward_flow_cut_short_says_what_it_leaves_out(tmp_path):
    nodes = [_fn("FILE:a.py::start"), *(_fn(f"FILE:a.py::sure{i}") for i in range(2))]
    nodes += [_fn("FILE:b.py::maybe"), _fn("FILE:c.py::maybe")]
    edges = [_calls("FILE:a.py::start", f"FILE:a.py::sure{i}") for i in range(2)]
    edges.append(
        {
            "source": "FILE:a.py::start",
            "target": "REF:x.maybe",
            "relation": "CALLS",
            "metadata": {"candidates": ["FILE:b.py::maybe", "FILE:c.py::maybe"]},
        }
    )
    q = _querier(tmp_path, nodes, edges, max_children_per_level=1)
    assert "*(1 more callees, 2 more possible callees)*" in _flow(
        q, "FILE:a.py::start", depth=1
    )
    root = q.flow_data("FILE:a.py::start", depth=1)["root"]
    assert (root["more"], root["more_possible"]) == (3, 2)


def test_a_target_past_the_per_level_limit_is_still_reached(tmp_path):
    nodes = [_fn("FILE:a.py::start"), _fn("FILE:a.py::end")]
    nodes += [_fn(f"FILE:a.py::mid{i}") for i in range(2)]
    edges = [_calls("FILE:a.py::start", f"FILE:a.py::mid{i}") for i in range(2)]
    edges.append(_calls("FILE:a.py::mid1", "FILE:a.py::end"))
    q = _querier(tmp_path, nodes, edges, max_children_per_level=1)
    flow = q.flow_data("FILE:a.py::start", target="FILE:a.py::end")
    assert flow["path_to_target"] == ["start", "mid1", "end"]
    limited = _querier(tmp_path, nodes, edges, max_children_per_level=1, max_nodes=1)
    flow = limited.flow_data("FILE:a.py::start", target="FILE:a.py::end")
    assert "path_to_target" not in flow and flow["cut"] == "node limit"


def test_calls_written_apart_are_callees_apart(tmp_path):
    """`a.add` and `b.add` end in one method name but are two calls: one
    the resolver narrowed, one it found nothing for."""
    nodes = [
        _fn("FILE:a.py::f"),
        _fn("FILE:b.py::Set::add"),
        _fn("FILE:c.py::List::add"),
    ]
    edges = [
        {
            "source": "FILE:a.py::f",
            "target": "REF:a.add",
            "relation": "CALLS",
            "metadata": {"candidates": ["FILE:b.py::Set::add", "FILE:c.py::List::add"]},
        },
        {"source": "FILE:a.py::f", "target": "REF:b.add", "relation": "CALLS"},
        {"source": "FILE:a.py::f", "target": "REF:b.add", "relation": "CALLS"},
    ]
    q = _querier(tmp_path, nodes, edges)
    callees = q.get_callees("FILE:a.py::f")
    assert callees == ["REF:a.add", "REF:b.add"]
    assert q.kinds(callees, q.get_ambiguous_callees("FILE:a.py::f")) == (0, 1, 1)


def test_an_edge_to_a_symbol_the_graph_lacks_does_not_break_context(tmp_path):
    """A graph is untrusted: a resolved edge may name a node it does not
    have."""
    (tmp_path / "a.py").write_text("def f():\n    gone()\n")
    nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
    q = _querier(
        tmp_path,
        nodes,
        [_calls("FILE:a.py::f", "FILE:a.py::gone")],
        project_root=tmp_path,
    )
    for depth, expand in ((1, False), (2, True)):
        assert "gone`" in _context(
            q, "FILE:a.py::f", depth=depth, expand_callees=expand
        )


def test_calls_written_alike_reach_what_any_of_them_may(tmp_path):
    """Calls of `put` narrowed to one definition each, then one to none:
    the callee `put` may reach either definition."""
    nodes = [
        _fn("FILE:a.py::f"),
        _fn("FILE:b.py::Facet::put"),
        _fn("FILE:c.py::Sink::put"),
    ]
    edges = [
        {
            "source": "FILE:a.py::f",
            "target": "REF:put",
            "relation": "CALLS",
            "metadata": {"candidates": ["FILE:b.py::Facet::put"]},
        },
        {
            "source": "FILE:a.py::f",
            "target": "REF:put",
            "relation": "CALLS",
            "metadata": {"candidates": ["FILE:c.py::Sink::put"]},
        },
        {"source": "FILE:a.py::f", "target": "REF:put", "relation": "CALLS"},
    ]
    assert _querier(tmp_path, nodes, edges).get_ambiguous_callees("FILE:a.py::f") == {
        "REF:put": 2
    }


def test_a_deep_context_says_what_it_leaves_out(tmp_path):
    """The symbol's own callees are all listed; a level below is cut to the
    per-level limit, with a note."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
    nodes += [
        _fn(f"FILE:a.py::g{i}", lineno=1, end_lineno=2, token_count=10)
        for i in range(3)
    ]
    nodes += [_fn(f"FILE:a.py::h{i}", token_count=500) for i in range(3)]
    edges = [_calls("FILE:a.py::f", f"FILE:a.py::g{i}") for i in range(3)]
    edges += [_calls("FILE:a.py::g0", f"FILE:a.py::h{i}") for i in range(3)]
    edges.append({"source": "FILE:a.py::g0", "target": "REF:x.h", "relation": "CALLS"})
    q = _querier(
        tmp_path, nodes, edges, project_root=tmp_path, max_children_per_level=2
    )
    text = _context(q, "FILE:a.py::f", depth=2)
    assert all(f"**g{i}**" in text for i in range(3))
    assert "*(1 more callees, 1 more unresolved callees)*" in text


class TestEachSymbolAtItsLevel:
    """`start` calls `a` and `b`, and `a` calls `b`: `b` is a direct callee,
    with its own callee `c` below it, and is met again under `a`."""

    @pytest.fixture
    def q(self, tmp_path):
        names = ["start", "a", "b", "c"]
        (tmp_path / "m.py").write_text(
            "".join(f"def {n}():\n    pass\n" for n in names)
        )
        nodes = [
            _fn(
                f"FILE:m.py::{n}",
                lineno=2 * i + 1,
                end_lineno=2 * i + 2,
                token_count=500,
            )
            for i, n in enumerate(names)
        ]
        pairs = [("start", "a"), ("start", "b"), ("a", "b"), ("b", "c")]
        edges = [_calls(f"FILE:m.py::{x}", f"FILE:m.py::{y}") for x, y in pairs]
        return _querier(tmp_path, nodes, edges, project_root=tmp_path)

    def test_forward(self, q):
        root = q.flow_data("FILE:m.py::start", depth=2)["root"]
        assert [(c["name"], c.get("seen", False)) for c in root["children"]] == [
            ("a", False),
            ("b", False),
        ]
        a, b = root["children"]
        assert [(c["name"], c.get("seen")) for c in a["children"]] == [("b", 1)]
        assert a["children"][0]["seen"] is not True
        assert [c["name"] for c in b["children"]] == ["c"]
        text = _flow(q, "FILE:m.py::start", depth=2)
        assert "    ├─[Function] **b** (at level 1)" in text
        assert "    ├─[Function] **c** (500 toks)" in text

    def test_backward(self, q):
        tree = q.flow_data("FILE:m.py::b", direction="backward", depth=2)
        a, start = tree["root"]["children"]
        assert (a["name"], start["name"], start.get("seen")) == ("a", "start", None)
        assert [(c["name"], c.get("seen")) for c in a["children"]] == [("start", 1)]
        assert a["children"][0]["seen"] is not True

    def test_deep_context_shows_a_callee_met_again(self, q):
        for node in q.nodes.values():
            node["metadata"]["token_count"] = 20
        text = _context(q, "FILE:m.py::start", depth=2)
        assert "- `b` (at level 1)" in text
        # `b` opens at level 1, so its callee is shown
        assert "**c**" in text


def test_usage_examples_are_calls_of_that_symbol(tmp_path):
    """Not calls left unresolved that end in its name; one it is a
    candidate of is marked."""
    (tmp_path / "a.py").write_text("def add():\n    pass\n")
    (tmp_path / "u.py").write_text(
        "def use():\n    add()\n    x.add()\n    vary.add('Cookie')\n"
    )
    nodes = [
        _fn("FILE:a.py::add", lineno=1, end_lineno=2, token_count=10),
        _fn("FILE:u.py::use"),
        _fn("FILE:b.py::add"),
    ]
    edges = [
        {
            "source": "FILE:u.py::use",
            "target": "FILE:a.py::add",
            "relation": "CALLS_RESOLVED",
            "metadata": {"lineno": 2},
        },
        {
            "source": "FILE:u.py::use",
            "target": "REF:x.add",
            "relation": "CALLS",
            "metadata": {
                "lineno": 3,
                "candidates": ["FILE:a.py::add", "FILE:b.py::add"],
            },
        },
        {
            "source": "FILE:u.py::use",
            "target": "REF:vary.add",
            "relation": "CALLS",
            "metadata": {"lineno": 4},
        },
    ]
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path)
    examples = q._get_usage_examples("FILE:a.py::add", limit=5)
    assert [(e["lineno"], e["candidates"]) for e in examples] == [(2, None), (3, 2)]
    assert "Line 3: (one of 2 candidates)" in _context(q, "FILE:a.py::add")


def test_a_node_id_names_that_node_alone(tmp_path):
    """Sinatra's instance `force_encoding` is not its `self.force_encoding`,
    though both are named `force_encoding` under `Base`."""
    nodes = [
        _fn("FILE:base.rb::Base::force_encoding", name="force_encoding"),
        _fn("FILE:base.rb::Base::self.force_encoding", name="force_encoding"),
        _fn("FILE:base.rb::Base::dispatch"),
        _fn("FILE:base.rb::Base::inline_templates="),
    ]
    edges = [
        _calls("FILE:base.rb::Base::dispatch", "FILE:base.rb::Base::force_encoding"),
        _calls(
            "FILE:base.rb::Base::inline_templates=",
            "FILE:base.rb::Base::self.force_encoding",
        ),
    ]
    q = _querier(tmp_path, nodes, edges)
    instance = "FILE:base.rb::Base::force_encoding"
    assert [n["id"] for n in q.find_nodes_by_name(instance)] == [instance]
    assert q.get_callers(instance) == ["FILE:base.rb::Base::dispatch"]
    assert len(q.get_callers("force_encoding")) == 2


def test_a_recursive_start_is_marked_at_level_zero(tmp_path):
    nodes = [_fn("FILE:a.py::fact"), _fn("FILE:a.py::helper")]
    edges = [
        _calls("FILE:a.py::fact", "FILE:a.py::fact"),
        _calls("FILE:a.py::fact", "FILE:a.py::helper"),
    ]
    q = _querier(tmp_path, nodes, edges)
    root = q.flow_data("FILE:a.py::fact", depth=2)["root"]
    assert [(c["name"], c.get("seen")) for c in root["children"]] == [
        ("fact", 0),
        ("helper", None),
    ]
    assert "**fact** (at level 0)" in _flow(q, "FILE:a.py::fact", depth=2)


def test_deep_context_lists_an_unresolved_call_met_twice(tmp_path):
    """Unresolved calls, and calls of a symbol the graph lacks, are not
    symbols to open: each is listed where it is made."""
    (tmp_path / "a.py").write_text(
        "".join(f"def {n}():\n    pass\n" for n in ("two", "c1", "c2"))
    )
    nodes = [
        _fn(f"FILE:a.py::{n}", lineno=2 * i + 1, end_lineno=2 * i + 2, token_count=20)
        for i, n in enumerate(("two", "c1", "c2"))
    ]
    edges = [
        _calls("FILE:a.py::two", "FILE:a.py::c1"),
        _calls("FILE:a.py::two", "FILE:a.py::c2"),
    ]
    for caller in ("c1", "c2"):
        edges.append(
            {
                "source": f"FILE:a.py::{caller}",
                "target": "REF:items.append",
                "relation": "CALLS",
            }
        )
        edges.append(_calls(f"FILE:a.py::{caller}", "FILE:a.py::gone"))
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path)
    text = _context(q, "FILE:a.py::two", depth=2)
    assert text.count("`append` (unresolved)") == 2
    assert text.count("gone`") == 2


def test_the_node_limit_lists_the_symbols_own_callees(tmp_path):
    """It cuts the levels below, counting what it did not reach."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
    nodes += [
        _fn(f"FILE:a.py::g{i}", lineno=1, end_lineno=2, token_count=10)
        for i in range(3)
    ]
    nodes += [_fn(f"FILE:a.py::h{i}", token_count=500) for i in range(3)]
    edges = [_calls("FILE:a.py::f", f"FILE:a.py::g{i}") for i in range(3)]
    edges += [_calls("FILE:a.py::g0", f"FILE:a.py::h{i}") for i in range(3)]
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path, max_nodes=2)
    text = _context(q, "FILE:a.py::f", depth=2)
    assert all(f"**g{i}**" in text for i in range(3))
    assert "*(3 more callees)*" in text
    assert "*(node limit reached)*" in text


def test_the_node_limit_counts_what_each_parent_leaves(tmp_path):
    """Past the limit, a parent opened at the level above lists its
    callees as left out, not as nothing."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    nodes = [
        _fn(f"FILE:a.py::{n}", lineno=1, end_lineno=2, token_count=10)
        for n in ("f", "g0", "g1", "g2")
    ]
    nodes += [_fn(f"FILE:a.py::{n}", token_count=500) for n in ("h0", "h1", "k0", "k1")]
    pairs = [
        ("f", "g0"),
        ("f", "g1"),
        ("f", "g2"),
        ("g0", "h0"),
        ("g0", "h1"),
        ("g1", "k0"),
        ("g1", "k1"),
    ]
    edges = [_calls(f"FILE:a.py::{a}", f"FILE:a.py::{b}") for a, b in pairs]
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path, max_nodes=2)
    text = _context(q, "FILE:a.py::f", depth=2)
    assert text.count("*(2 more callees)*") == 2


def test_the_node_limit_counts_below_what_it_opened(tmp_path):
    """A callee opened at the level the limit cuts says what it calls."""
    (tmp_path / "m.py").write_text("def f():\n    pass\n")
    small = ("f", "g0", "g1", "g2", "h0", "h1")
    nodes = [
        _fn(f"FILE:m.py::{n}", lineno=1, end_lineno=2, token_count=10) for n in small
    ]
    nodes.append(_fn("FILE:m.py::m0", token_count=500))
    pairs = [
        ("f", "g0"),
        ("f", "g1"),
        ("f", "g2"),
        ("g0", "h0"),
        ("g0", "h1"),
        ("h0", "m0"),
    ]
    edges = [_calls(f"FILE:m.py::{a}", f"FILE:m.py::{b}") for a, b in pairs]
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path, max_nodes=4)
    text = _context(q, "FILE:m.py::f", depth=3)
    # `h0` leaves out `m0`; `g0`, cut at the limit, leaves out `h1`
    assert text.count("*(1 more callees)*") == 2


class TestOutputCap:
    """`max_output_tokens` cuts the parts `context` is made of, not its text:
    code inside them is never read as structure."""

    @pytest.fixture
    def q(self, tmp_path):
        code = ["    - a bullet", "#### Linux:", "```", "    return 1"]
        source = ["def f():", "    pass"]
        for i in range(7):
            source += [f"def c{i}():", *code]
        for i in range(9):
            source += [f"def u{i}():", "    f()"]
        (tmp_path / "a.py").write_text("\n".join(source) + "\n")
        nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
        nodes += [
            _fn(
                f"FILE:a.py::c{i}",
                lineno=3 + 5 * i,
                end_lineno=7 + 5 * i,
                token_count=20,
            )
            for i in range(7)
        ]
        nodes += [
            _fn(
                f"FILE:a.py::u{i}",
                lineno=38 + 2 * i,
                end_lineno=39 + 2 * i,
                token_count=10,
            )
            for i in range(9)
        ]
        nodes.append(_fn("FILE:b.py::f"))
        edges = [_calls("FILE:a.py::f", f"FILE:a.py::c{i}") for i in range(7)]
        edges += [_calls(f"FILE:a.py::u{i}", "FILE:a.py::f") for i in range(7)]
        edges += [
            {
                "source": f"FILE:a.py::u{i}",
                "target": "REF:x.f",
                "relation": "CALLS",
                "metadata": {
                    "lineno": 39 + 2 * i,
                    "candidates": ["FILE:a.py::f", "FILE:b.py::f"],
                },
            }
            for i in (7, 8)
        ]
        return _querier(tmp_path, nodes, edges, project_root=tmp_path)

    def test_callees_go_whole(self, q):
        out = _context(q, "FILE:a.py::f", max_output_tokens=1)
        assert "*(2 more callees)*" in out
        assert "**c4**" in out and "**c5**" not in out
        assert out.count("#### Linux:") == 5

    def test_a_cut_caller_list_keeps_possible_callers_apart(self, q):
        out = _context(q, "FILE:a.py::f", max_output_tokens=1)
        assert "#### Callers (5 shown):" in out
        assert "*(2 more callers, 2 more possible callers)*" in out

    def test_a_dropped_section_is_said(self, q, tmp_path):
        q._find_related_tests = lambda *args: ["tests/test_a.py"]
        out = _context(q, "FILE:a.py::f", max_output_tokens=1)
        assert "#### Related Tests:\n*(omitted, max_output_tokens limit)*" in out


def test_the_source_cut_counts_from_the_symbols_lines(tmp_path):
    """Not from the source already shortened for display."""
    (tmp_path / "x.py").write_text("\n".join(f"line {i}" for i in range(300)) + "\n")
    node = _fn("FILE:x.py::f", lineno=1, end_lineno=300, token_count=3000)
    q = _querier(tmp_path, [node], project_root=tmp_path)
    out = _context(q, "FILE:x.py::f", max_output_tokens=50)
    assert "(270 lines omitted, max_output_tokens limit)" in out
    # Still over the cap with every step taken, it says so
    assert out.endswith("\n\n*[Output truncated to ~50 tokens]*")


def test_related_tests_are_ids_a_call_can_pass_back(tmp_path):
    """Each entry is a node ID: not a bare path, not prose after one."""
    nodes = [
        _fn("FILE:a.py::f"),
        _fn("FILE:a.py::test_f", lineno=1),
        {"id": "FILE:test_a.py", "type": "File", "metadata": {"path": "test_a.py"}},
    ]
    q = _querier(tmp_path, nodes)
    assert q._find_related_tests("a.py", "f") == [
        "FILE:test_a.py",
        "FILE:a.py::test_f",
    ]


def test_related_tests_of_an_acronym_name_are_its_snake_case_tests(tmp_path):
    nodes = [
        _fn("FILE:http.py::HTTPServer"),
        _fn("FILE:test_http.py::test_http_server"),
    ]
    q = _querier(tmp_path, nodes)
    assert q._find_related_tests("http.py", "HTTPServer") == [
        "FILE:test_http.py::test_http_server"
    ]


def test_related_test_files_are_named_by_the_configured_suffixes(tmp_path):
    nodes = [
        _fn("FILE:a.py::f"),
        *(
            {"id": f"FILE:{name}", "type": "File", "metadata": {"path": name}}
            for name in ("a_check.py", "a.spec.py")
        ),
    ]
    q = _querier(tmp_path, nodes, test_file_suffixes=("_check.py",))
    assert q._find_related_tests("a.py", "f") == ["FILE:a_check.py"]


def test_related_tests_are_test_code_as_test_detection_says(tmp_path):
    """A file named after the symbol's is among them only where
    `[test_detection]` makes it test code; the usage examples put tests
    first by the same rule."""
    nodes = [
        _fn("FILE:a.py::f"),
        {"id": "FILE:a.spec.py", "type": "File", "metadata": {"path": "a.spec.py"}},
        _fn("FILE:attest.py::use"),
        _fn("FILE:qa/check.py::use"),
    ]
    edges = [
        {
            "source": source,
            "target": "FILE:a.py::f",
            "relation": "CALLS",
            "metadata": {"lineno": 1},
        }
        for source in ("FILE:attest.py::use", "FILE:qa/check.py::use")
    ]
    for path in ("attest.py", "qa/check.py"):
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_text("f()\n")
    q = _querier(
        tmp_path, nodes, edges, test_path_patterns=("/qa/",), project_root=tmp_path
    )
    assert q._find_related_tests("a.py", "f") == []
    examples = q._get_usage_examples("FILE:a.py::f")
    assert [e["file"] for e in examples] == ["qa/check.py", "attest.py"]


@pytest.mark.parametrize("a0_calls_p", [False, True])
def test_a_cut_callee_list_keeps_kinds_and_marks_true(tmp_path, a0_calls_p):
    """Possible callees are counted apart, no mark points at a callee whose
    entry was cut, and a call drawn under a kept callee is not counted."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    small = [f"a{i}" for i in range(6)]
    nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
    nodes += [
        _fn(f"FILE:a.py::{n}", lineno=1, end_lineno=2, token_count=10) for n in small
    ]
    nodes += [_fn("FILE:b.py::p"), _fn("FILE:c.py::p")]
    edges = [_calls("FILE:a.py::f", f"FILE:a.py::{n}") for n in small]
    edges.append(_calls("FILE:a.py::a0", "FILE:a.py::a5"))
    for source in ["FILE:a.py::f"] + ["FILE:a.py::a0"] * a0_calls_p:
        edges.append(
            {
                "source": source,
                "target": "REF:x.p",
                "relation": "CALLS",
                "metadata": {"candidates": ["FILE:b.py::p", "FILE:c.py::p"]},
            }
        )
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path)
    out = _context(q, "FILE:a.py::f", depth=2, max_output_tokens=1)
    # `a5`, cut from f's list, is drawn under `a0`, and so is `x.p` when
    # `a0` makes the call
    assert ("more possible callees" in out) is not a0_calls_p
    assert "**a5**" in out
    assert "(at level" not in out


def test_a_self_call_past_the_callees_cut_is_counted(tmp_path):
    """The symbol is not drawn as one of its own callees."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    small = [f"a{i}" for i in range(6)]
    nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
    nodes += [
        _fn(f"FILE:a.py::{n}", lineno=1, end_lineno=2, token_count=10) for n in small
    ]
    edges = [_calls("FILE:a.py::f", f"FILE:a.py::{n}") for n in small]
    edges.append(_calls("FILE:a.py::f", "FILE:a.py::f"))
    q = _querier(tmp_path, nodes, edges, project_root=tmp_path)
    out = _context(q, "FILE:a.py::f", depth=2, max_output_tokens=1)
    assert "*(2 more callees)*" in out
