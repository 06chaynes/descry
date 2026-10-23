"""Tests for descry.query — GraphQuerier and related query functions."""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from descry.handlers import DescryConfig
from descry.query import MAX_PATH_DEPTH, GraphQuerier


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
        q = GraphQuerier(sample_graph)
        matches = q.find_nodes_by_name("helper")
        assert len(matches) == 1
        assert matches[0]["metadata"]["name"] == "helper"

    def test_get_callers(self, sample_graph):
        """Should find callers of a function."""
        q = GraphQuerier(sample_graph)
        callers = q.get_callers("helper")
        assert len(callers) == 1
        assert "main" in callers[0]

    def test_get_callees(self, sample_graph):
        """Should find callees of a function."""
        q = GraphQuerier(sample_graph)
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
        q = GraphQuerier(path)
        assert q.get_callers("work") == ["FILE:a.py::run"]
        assert q.get_ambiguous_callers("work") == {"FILE:a.py::run": 2}
        assert q.get_callees("FILE:a.py::run") == ["REF:work"]
        assert q.get_ambiguous_callees("FILE:a.py::run") == {"REF:work": 2}

    def test_search_docs(self, sample_graph):
        """Should search by keywords."""
        q = GraphQuerier(sample_graph)
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
        q = GraphQuerier(trait_impl_graph)
        results = q.find_trait_impls("from_request_parts")

        assert len(results) == 3
        names = {r["id"].split("::")[-2] for r in results}
        assert names == {"TokenAuth", "OptionalAuth", "Pagination"}

    def test_find_impls_filtered_by_trait(self, trait_impl_graph):
        """Should filter by trait name."""
        q = GraphQuerier(trait_impl_graph)
        results = q.find_trait_impls("call", "Service")

        assert len(results) == 1
        assert "MyService" in results[0]["id"]

    def test_find_impls_excludes_inherent_methods(self, trait_impl_graph):
        """Should not return inherent impl methods."""
        q = GraphQuerier(trait_impl_graph)
        results = q.find_trait_impls("new")

        assert len(results) == 0  # "new" has no trait_impl

    def test_find_impls_no_matches(self, trait_impl_graph):
        """Should return empty list for non-existent methods."""
        q = GraphQuerier(trait_impl_graph)
        results = q.find_trait_impls("nonexistent_method")

        assert len(results) == 0


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
        q = GraphQuerier(call_path_graph)

        tree = q.trace_flow_structured("handle_request", target="validate_token")

        assert tree["path_to_target"] == [
            "handle_request",
            "process_data",
            "validate_token",
        ]
        process_data = tree["root"]["children"][0]
        assert [c["name"] for c in process_data["children"]] == ["validate_token"]
        assert "path_to_target" not in q.trace_flow_structured("handle_request")

    def test_find_direct_path(self, call_path_graph):
        """Should find direct call path."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path_result("handle_request", "process_data")["path"]

        assert len(path) == 1
        assert path[0]["caller_name"] == "handle_request"
        assert path[0]["callee_name"] == "process_data"

    def test_find_two_hop_path(self, call_path_graph):
        """Should find multi-hop path."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path_result("handle_request", "validate_token")["path"]

        assert len(path) == 2
        assert path[0]["caller_name"] == "handle_request"
        assert path[0]["callee_name"] == "process_data"
        assert path[1]["caller_name"] == "process_data"
        assert path[1]["callee_name"] == "validate_token"

    def test_path_includes_call_line(self, call_path_graph):
        """Should include call line information."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path_result("handle_request", "process_data")["path"]

        assert path[0]["call_line"] == 15

    def test_no_path_returns_empty(self, call_path_graph):
        """Should return empty list when no path exists."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path_result("validate_token", "handle_request")["path"]

        # No forward path from validate_token to handle_request
        assert len(path) == 0

    def test_backward_direction(self, call_path_graph):
        """Should find backward path (who calls me)."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path_result(
            "validate_token", "handle_request", direction="backward"
        )["path"]

        # Backward: validate_token is called by process_data is called by handle_request
        assert len(path) == 2
        # In backward direction, roles are swapped for hop info
        assert path[0]["callee_name"] == "validate_token"
        assert path[1]["callee_name"] == "process_data"

    def test_max_depth_limit(self, call_path_graph):
        """Should respect max_depth limit."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path_result("handle_request", "validate_token", max_depth=1)[
            "path"
        ]

        # Path requires 2 hops but max_depth is 1
        assert len(path) == 0


class TestFlowDisambiguation:
    """Tests for trace_flow disambiguation when multiple matches exist."""

    @pytest.fixture
    def graph_with_duplicate_names(self):
        """Create a graph with multiple functions named 'new'."""
        # Create graph with two structs that each have a "new" method
        graph = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:service_a.rs",
                    "type": "File",
                    "metadata": {"path": "service_a.rs", "name": "service_a.rs"},
                },
                {
                    "id": "FILE:service_a.rs::ServiceA::new",
                    "type": "Method",
                    "metadata": {
                        "name": "new",
                        "signature": "pub fn new() -> Self",
                        "token_count": 20,
                        "lineno": 4,
                        "end_lineno": 6,
                    },
                },
                {
                    "id": "FILE:service_a.rs::ServiceA::process",
                    "type": "Method",
                    "metadata": {
                        "name": "process",
                        "signature": "pub fn process(&self)",
                        "token_count": 15,
                        "lineno": 8,
                        "end_lineno": 10,
                    },
                },
                {
                    "id": "FILE:service_b.rs",
                    "type": "File",
                    "metadata": {"path": "service_b.rs", "name": "service_b.rs"},
                },
                {
                    "id": "FILE:service_b.rs::ServiceB::new",
                    "type": "Method",
                    "metadata": {
                        "name": "new",
                        "signature": "pub fn new() -> Self",
                        "token_count": 20,
                        "lineno": 4,
                        "end_lineno": 6,
                    },
                },
                {
                    "id": "FILE:service_b.rs::ServiceB::process",
                    "type": "Method",
                    "metadata": {
                        "name": "process",
                        "signature": "pub fn process(&self)",
                        "token_count": 15,
                        "lineno": 8,
                        "end_lineno": 10,
                    },
                },
            ],
            "edges": [
                {
                    "source": "FILE:service_a.rs",
                    "target": "FILE:service_a.rs::ServiceA::new",
                    "relation": "DEFINES",
                },
                {
                    "source": "FILE:service_a.rs",
                    "target": "FILE:service_a.rs::ServiceA::process",
                    "relation": "DEFINES",
                },
                {
                    "source": "FILE:service_b.rs",
                    "target": "FILE:service_b.rs::ServiceB::new",
                    "relation": "DEFINES",
                },
                {
                    "source": "FILE:service_b.rs",
                    "target": "FILE:service_b.rs::ServiceB::process",
                    "relation": "DEFINES",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            f.flush()
            yield f.name

        os.unlink(f.name)

    def test_shows_disambiguation_for_multiple_matches(
        self, graph_with_duplicate_names
    ):
        """trace_flow should show which start point was selected when multiple exist."""
        q = GraphQuerier(graph_with_duplicate_names)

        result = q.trace_flow("new", direction="forward", depth=2)

        # Should contain disambiguation notice
        assert "Found 2 matches" in result
        assert "Using:" in result
        assert "Alternatives" in result

    def test_shows_alternatives_list(self, graph_with_duplicate_names):
        """trace_flow should list alternative starting points."""
        q = GraphQuerier(graph_with_duplicate_names)

        result = q.trace_flow("new", direction="forward", depth=2)

        # Should show the alternatives with FILE: prefix
        assert "FILE:" in result
        # Should mention full node ID for disambiguation
        assert "node ID" in result  # "use full node ID to select"

    def test_no_disambiguation_when_using_full_node_id(
        self, graph_with_duplicate_names
    ):
        """trace_flow should not show disambiguation when full node ID is used."""
        q = GraphQuerier(graph_with_duplicate_names)

        # Using full node ID should not trigger disambiguation
        result = q.trace_flow(
            "FILE:service_a.rs::ServiceA::new", direction="forward", depth=2
        )

        # When exact node ID is provided and matches exactly, no disambiguation needed
        # The result should contain the flow but not the alternatives message
        assert "Call Flow:" in result
        # Should not say "Found X matches" since we provided exact ID
        assert "Alternatives" not in result


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

        result = q.get_context_prompt(
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
        q = GraphQuerier(graph_file)

        result = q.get_context_prompt(f"FILE:{source_file}::authenticate")

        # Should NOT have Expanded Callees section
        assert "Expanded Callees" not in result

    def test_expand_callees_respects_budget(self, graph_with_callees):
        """expand_callees should respect the token budget."""
        graph_file, source_file = graph_with_callees
        q = GraphQuerier(graph_file)

        # With very small budget, should skip callees
        expanded = q._expand_callees_full(
            f"FILE:{source_file}::authenticate", budget=10
        )

        # Both callees (80 + 60 = 140 tokens) exceed budget of 10
        assert len(expanded) == 0


class TestFuzzyNodeMatching:
    """Tests for fuzzy node ID matching in descry context."""

    @pytest.fixture
    def graph_with_symbols(self, tmp_path):
        """Create a graph with various symbols for testing fuzzy matching."""
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
        """Exact node ID should work without fuzzy matching."""
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        result = q.get_context_prompt(f"FILE:{source_file}::validate_token")

        assert "Context for" in result
        assert "validate_token" in result
        assert "Matched" not in result  # No fuzzy match note

    def test_symbol_name_only_matches(self, graph_with_symbols, tmp_path):
        """Just symbol name should find the node via fuzzy matching."""
        graph_file, _ = graph_with_symbols
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # Just the function name, no FILE: prefix or path
        result = q.get_context_prompt("validate_token")

        assert "Context for" in result
        # Should have a fuzzy match note
        assert "Matched" in result
        assert "validate_token" in result

    def test_multiple_matches_shows_suggestions(self, graph_with_symbols):
        """When multiple nodes match, should show suggestions."""
        graph_file, _ = graph_with_symbols
        q = GraphQuerier(graph_file)

        # "validate" matches both functions partially
        resolved_id, msg = q._resolve_node_id("validate")

        # Should not auto-resolve (ambiguous)
        assert resolved_id is None
        assert "Did you mean" in msg
        assert "validate_token" in msg
        assert "validate_token_format" in msg

    def test_camel_case_finds_snake_case(self, graph_with_symbols, tmp_path):
        """camelCase query should find snake_case function."""
        graph_file, _ = graph_with_symbols
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # camelCase version
        result = q.get_context_prompt("validateToken")

        assert "Context for" in result
        # Should match validate_token
        assert "validate_token" in result

    def test_partial_path_matches(self, graph_with_symbols, tmp_path):
        """Partial path should help narrow down matches."""
        graph_file, _ = graph_with_symbols
        q = GraphQuerier(graph_file, config=DescryConfig(project_root=tmp_path))

        # Just filename::function
        result = q.get_context_prompt("auth.rs::validate_token")

        assert "Context for" in result
        assert "validate_token" in result


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
        context_truncated = q.get_context_prompt(node_id, full=False)
        assert "lines omitted" in context_truncated

        # With full=True, should not truncate
        context_full = q.get_context_prompt(node_id, full=True)
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
            q.get_context_prompt(node_id),
            q.trace_flow(node_id.split("::")[-1], inline_threshold=100),
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
        "test_path_patterns": None,
        "test_file_suffixes": None,
        "syntax_lang_map": None,
        "project_root": None,
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

        context = querier.get_context_prompt(
            "FILE:src/foo.py::caller", expand_callees=True
        )
        assert "return helper()" in context
        assert "return 42" in context

        assert "return 42" in querier.trace_flow("caller", inline_threshold=100)
        tree = querier.trace_flow_structured("caller", inline_threshold=100)
        assert "return helper()" in tree["root"]["code"]

        hop = querier.find_call_path_result("caller", "helper")["path"][0]
        assert "return helper()" in hop["call_snippet"]
        assert hop["file_path"] == "src/foo.py"

    def test_a_file_over_the_size_limit_is_not_read(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / "big.py").write_text("def big():\n    return 'BODY'\n")
        nodes = [_fn("FILE:big.py::big", lineno=1, end_lineno=2, token_count=10)]

        within = _querier(tmp_path, nodes, project_root=root, max_file_bytes=1024)
        assert "BODY" in within.get_context_prompt("FILE:big.py::big")
        over = _querier(tmp_path, nodes, project_root=root, max_file_bytes=8)
        assert "BODY" not in over.get_context_prompt("FILE:big.py::big")
        assert "BODY" not in over.trace_flow("big", inline_threshold=100)

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

        assert "TOP SECRET" not in q.get_context_prompt(node_id)
        assert "TOP SECRET" not in q.trace_flow("leaked", inline_threshold=100)
        assert "TOP SECRET" not in q.get_source_segment(rel, 1, 2)

    @pytest.mark.parametrize("absolute", [False, True])
    def test_no_source_is_read_without_a_project_root(
        self, tmp_path, monkeypatch, absolute
    ):
        source = tmp_path / "foo.py"
        source.write_text("def leaked():\n    return 'TOP SECRET'\n")
        monkeypatch.chdir(tmp_path)
        rel = str(source) if absolute else "foo.py"
        node_id = f"FILE:{rel}::leaked"
        nodes = [_fn(node_id, lineno=1, end_lineno=2, token_count=10)]
        rootless_config = _querier(tmp_path, nodes)
        no_config = GraphQuerier(str(tmp_path / "graph.json"))

        for q in (rootless_config, no_config):
            assert "TOP SECRET" not in _source_shown(q, node_id)


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
        from descry.handlers import _DEFAULT_SYNTAX_LANG_MAP

        return _querier(
            tmp_path,
            [_fn(f"FILE:{f}::handler") for f in self.FILES],
            syntax_lang_map=_DEFAULT_SYNTAX_LANG_MAP,
        )

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

    @pytest.mark.parametrize("lang", ["all", None])
    def test_all_and_none_do_not_filter(self, querier, lang):
        assert len(querier.search_docs(["handler"], lang=lang)) == len(self.FILES)


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
        assert len(q.find_call_path_result("n0", "n8", max_depth=10)["path"]) == 8
        assert q.find_call_path_result("n0", "n8", max_depth=7)["path"] == []

    def test_max_depth_is_capped_at_the_hard_ceiling(self, tmp_path):
        q = self._chain(tmp_path, MAX_PATH_DEPTH + 5, max_nodes=10_000)
        at_ceiling = q.find_call_path_result(
            "n0", f"n{MAX_PATH_DEPTH}", max_depth=10**9
        )
        assert at_ceiling["status"] == "found"
        assert at_ceiling["max_depth"] == MAX_PATH_DEPTH
        beyond = q.find_call_path_result(
            "n0", f"n{MAX_PATH_DEPTH + 1}", max_depth=10**9
        )
        assert beyond["status"] == "not_found"

    def test_status_tells_a_miss_from_a_search_that_was_cut_off(self, tmp_path):
        q = self._chain(tmp_path, 30)
        assert q.find_call_path_result("n0", "n5")["status"] == "found"
        assert q.find_call_path_result("n0", "island")["status"] == "not_found"
        assert q.find_call_path_result("nope", "n5")["status"] == "not_found"

        tight = self._chain(tmp_path, 30, max_nodes=5)
        cut_off = tight.find_call_path_result("n0", "n29", max_depth=40)
        assert cut_off == {"status": "node_budget", "max_depth": 40, "path": []}

        slow = self._chain(tmp_path, 30, query_timeout_ms=-1)
        assert slow.find_call_path_result("n0", "n29")["status"] == "timeout"


class TestExpansionMemo:
    @pytest.fixture
    def querier(self, tmp_path):
        (tmp_path / "m.py").write_text("".join(f"def f{i}(): pass\n" for i in range(4)))
        nodes = [
            _fn(f"FILE:m.py::{name}", lineno=i + 1, end_lineno=i + 1, token_count=10)
            for i, name in enumerate(("root", "mid", "leaf", "other"))
        ]
        edges = [
            _calls("FILE:m.py::root", "FILE:m.py::mid"),
            _calls("FILE:m.py::mid", "FILE:m.py::leaf"),
            _calls("FILE:m.py::root", "FILE:m.py::other"),
        ]
        return _querier(tmp_path, nodes, edges, project_root=tmp_path)

    def _expand(self, querier, expanded, budget=1000):
        return querier._expand_callees_recursive(
            "FILE:m.py::root", depth=3, budget=budget, expanded=expanded
        )

    def test_memoized_cost_includes_nested_callees(self, querier):
        first_lines, first_left = self._expand(querier, {"FILE:m.py::root"})
        again_lines, again_left = self._expand(querier, {"FILE:m.py::root"})
        # mid, leaf and other are inlined at 10 tokens each.
        assert first_left == again_left == 1000 - 30
        assert again_lines == first_lines

    def test_memo_hit_marks_the_same_nodes_expanded(self, querier):
        fresh, cached = {"FILE:m.py::root"}, {"FILE:m.py::root"}
        self._expand(querier, fresh)
        self._expand(querier, cached)
        assert cached == fresh
        assert "FILE:m.py::leaf" in cached

    def test_memo_is_keyed_on_the_already_expanded_nodes(self, querier):
        self._expand(querier, {"FILE:m.py::root"})
        lines, _ = self._expand(querier, {"FILE:m.py::root", "FILE:m.py::leaf"})
        assert not any("**leaf**" in line for line in lines)
        assert any("**mid**" in line for line in lines)

    def test_memo_is_keyed_on_the_budget(self, querier):
        tight, _ = self._expand(querier, {"FILE:m.py::root"}, budget=15)
        roomy, _ = self._expand(querier, {"FILE:m.py::root"}, budget=1000)
        inlined = [line for line in roomy if "toks):" in line]
        assert len(inlined) == 3
        assert roomy != tight

    def test_expansion_cut_short_by_the_timeout_is_not_memoized(
        self, querier, monkeypatch
    ):
        import itertools

        import descry.query

        # A clock that advances one second per reading runs out mid-expansion.
        ticks = itertools.count()
        with monkeypatch.context() as patched:
            patched.setattr(
                descry.query, "time", SimpleNamespace(time=lambda: next(ticks))
            )
            querier._timeout_ms = 2500
            cut_short, _ = self._expand(querier, {"FILE:m.py::root"})
        assert any("timeout" in line for line in cut_short)

        querier._timeout_ms = 4000
        complete, _ = self._expand(querier, {"FILE:m.py::root"})
        assert not any("timeout" in line for line in complete)


class TestLangMapValueInTruncation:
    @pytest.mark.parametrize("lang", ["(", "c++", "[a-"])
    def test_source_block_is_truncated_whatever_the_fence_tag(self, tmp_path, lang):
        q = _querier(tmp_path, [_fn("FILE:x.py::f")])
        body = "\n".join(f"line {i}" for i in range(200))
        text = f"### Context\n#### Source Code:\n```{lang}\n{body}\n```\n"
        out = q._truncate_output_progressively(text, max_tokens=50, source_lang=lang)
        assert "lines omitted, max_output_tokens limit" in out
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
        context = querier.get_context_prompt("FILE:b.py::helper")
        assert "- `start` in a.py (0 toks) (one of 2 candidates)" in context

    def test_path_starts_from_every_node_of_the_name(self, querier):
        found = querier.find_call_path_result("helper", "leaf")
        assert [(hop["caller_id"], hop["callee_id"]) for hop in found["path"]] == [
            ("FILE:b.py::helper", "FILE:b.py::leaf")
        ]

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
        found = _querier(tmp_path, nodes, edges).find_call_path_result("run", "run")
        assert [(hop["caller_id"], hop["callee_id"]) for hop in found["path"]] == [
            ("FILE:a.py::run", "FILE:b.py::run")
        ]

    def test_start_nodes_do_not_spend_the_node_budget(self, tmp_path):
        nodes = [_fn(f"FILE:f{i}.py::h") for i in range(120)]
        nodes.append(_fn("FILE:z.py::end"))
        edges = [_calls("FILE:f7.py::h", "FILE:z.py::end")]
        querier = _querier(tmp_path, nodes, edges)
        assert querier.find_call_path_result("h", "end")["status"] == "found"

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_path_goes_through_a_candidate_and_says_so(self, querier, direction):
        ends = ("start", "leaf") if direction == "forward" else ("leaf", "start")
        found = querier.find_call_path_result(*ends, direction=direction)
        assert found["status"] == "found"
        hops = sorted(found["path"], key=lambda hop: hop["caller_id"])
        assert [(hop["caller_id"], hop["callee_id"]) for hop in hops] == [
            ("FILE:a.py::start", "FILE:b.py::helper"),
            ("FILE:b.py::helper", "FILE:b.py::leaf"),
        ]
        assert [hop.get("candidates") for hop in hops] == [2, None]

    def test_path_does_not_reach_a_definition_the_edge_leaves_out(self, querier):
        found = querier.find_call_path_result("start", "FILE:a.py::helper")
        assert found["status"] == "not_found"
        assert querier.find_call_path_result("bare", "leaf")["status"] == "not_found"

    def test_backward_flow_marks_a_candidate_caller(self, querier):
        assert "**start** (0 toks) (one of 2 candidates)" in querier.trace_flow(
            "FILE:b.py::helper", direction="backward"
        )
        tree = querier.trace_flow_structured("FILE:b.py::helper", direction="backward")
        assert [(c["name"], c["candidates"]) for c in tree["root"]["children"]] == [
            ("start", 2)
        ]
        assert "**helper** (0 toks)\n" in querier.trace_flow(
            "leaf", direction="backward"
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
        assert "run" not in _querier(tmp_path, nodes, edges).flatten_class(
            "FILE:a.py::Child"
        )
        assert "run" in _querier(tmp_path, nodes[:3], edges).flatten_class(
            "FILE:a.py::Child"
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
        brief = querier.get_context_prompt("FILE:a.py::A::parse", brief=True)
        assert "**Callers:** 1 " in brief

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_a_flow_of_depth_one_is_the_calls_one_level_away(self, querier, direction):
        start = (
            "FILE:use.py::only_a" if direction == "forward" else "FILE:a.py::A::parse"
        )
        tree = querier.trace_flow_structured(start, direction=direction, depth=1)
        names = [child["name"] for child in tree["root"]["children"]]
        assert ("parse" in names) if direction == "forward" else names == ["only_a"]
        assert all(not child["children"] for child in tree["root"]["children"])

    def test_context_and_backward_flow_list_its_callers(self, querier):
        for text in (
            querier.get_context_prompt("FILE:a.py::A::parse"),
            querier.trace_flow("FILE:a.py::A::parse", direction="backward"),
        ):
            assert "only_a" in text
            assert "only_b" not in text


def test_a_call_with_one_candidate_is_not_one_of_several():
    assert GraphQuerier.one_of(1) == " (unresolved; the only candidate)"
    assert GraphQuerier.one_of(3) == " (one of 3 candidates)"
    assert GraphQuerier.one_of(None) == ""


def test_flatten_takes_a_class_name(tmp_path):
    widget = {
        "id": "FILE:w.py::Widget",
        "type": "Class",
        "metadata": {"name": "Widget"},
    }
    draw = {
        "id": "FILE:w.py::Widget::draw",
        "type": "Method",
        "metadata": {"name": "draw", "signature": "def draw(self)"},
    }
    defines = {"source": widget["id"], "target": draw["id"], "relation": "DEFINES"}
    flat = _querier(tmp_path, [widget, draw], [defines]).flatten_class("Widget")
    assert "def draw(self)" in flat
    assert "`FILE:w.py::Widget`" in flat


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
        assert "**Callers:** 0 (+4 possible)" in q.get_context_prompt(
            "FILE:a.py::outer::imp", brief=True
        )
        assert q.nodes["FILE:a.py::outer::imp"]["metadata"]["in_degree"] == 0

    def test_a_cut_list_keeps_the_certain_callers(self, tmp_path):
        q = self._graph(tmp_path, max_callers_shown=1, max_children_per_level=1)
        context = q.get_context_prompt("FILE:b.py::imp")
        assert "- `sure` in c.py" in context
        assert "(4 more possible callers)" in context
        flow = q.trace_flow("FILE:b.py::imp", direction="backward", depth=1)
        assert "**sure**" in flow and "*(4 more possible callers)*" in flow
        tree = q.trace_flow_structured("FILE:b.py::imp", direction="backward", depth=1)
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
    assert "*(1 more callees, 2 more possible callees)*" in q.trace_flow(
        "FILE:a.py::start", depth=1
    )
    root = q.trace_flow_structured("FILE:a.py::start", depth=1)["root"]
    assert (root["more"], root["more_possible"]) == (3, 2)


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
        assert "gone`" in q.get_context_prompt(
            "FILE:a.py::f", depth=depth, expand_callees=expand
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
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    nodes = [_fn("FILE:a.py::f", lineno=1, end_lineno=2, token_count=10)]
    nodes += [_fn(f"FILE:a.py::g{i}", token_count=500) for i in range(3)]
    edges = [_calls("FILE:a.py::f", f"FILE:a.py::g{i}") for i in range(3)]
    edges.append({"source": "FILE:a.py::f", "target": "REF:x.h", "relation": "CALLS"})
    q = _querier(
        tmp_path, nodes, edges, project_root=tmp_path, max_children_per_level=2
    )
    text = q.get_context_prompt("FILE:a.py::f", depth=2, expand_callees=True)
    assert "*(1 more callees, 1 more unresolved callees)*" in text
