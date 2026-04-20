"""Tests for descry.query — GraphQuerier and related query functions."""

import json
import os
import tempfile

import pytest

from descry.query import GraphQuerier


class TestGraphQuerier:
    """Tests for graph querying."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = {
            "schema_version": 1,
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
                    "target": "REF:helper",
                    "relation": "CALLS",
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
            "schema_version": 1,
            "nodes": [
                {
                    "id": "FILE:auth.rs::JwtAuth::from_request_parts",
                    "type": "Method",
                    "metadata": {
                        "name": "from_request_parts",
                        "trait_impl": "FromRequestParts",
                        "signature": "fn from_request_parts(...)",
                        "lineno": 10,
                    },
                },
                {
                    "id": "FILE:auth.rs::OptionalJwtAuth::from_request_parts",
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
        assert names == {"JwtAuth", "OptionalJwtAuth", "Pagination"}

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
    """Tests for find_call_path() method."""

    @pytest.fixture
    def call_path_graph(self):
        """Create a graph with call relationships."""
        graph = {
            "schema_version": 1,
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

    def test_find_direct_path(self, call_path_graph):
        """Should find direct call path."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path("handle_request", "process_data")

        assert len(path) == 1
        assert path[0]["caller_name"] == "handle_request"
        assert path[0]["callee_name"] == "process_data"

    def test_find_two_hop_path(self, call_path_graph):
        """Should find multi-hop path."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path("handle_request", "validate_token")

        assert len(path) == 2
        assert path[0]["caller_name"] == "handle_request"
        assert path[0]["callee_name"] == "process_data"
        assert path[1]["caller_name"] == "process_data"
        assert path[1]["callee_name"] == "validate_token"

    def test_path_includes_call_line(self, call_path_graph):
        """Should include call line information."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path("handle_request", "process_data")

        assert path[0]["call_line"] == 15

    def test_no_path_returns_empty(self, call_path_graph):
        """Should return empty list when no path exists."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path("validate_token", "handle_request")

        # No forward path from validate_token to handle_request
        assert len(path) == 0

    def test_backward_direction(self, call_path_graph):
        """Should find backward path (who calls me)."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path(
            "validate_token", "handle_request", direction="backward"
        )

        # Backward: validate_token is called by process_data is called by handle_request
        assert len(path) == 2
        # In backward direction, roles are swapped for hop info
        assert path[0]["callee_name"] == "validate_token"
        assert path[1]["callee_name"] == "process_data"

    def test_max_depth_limit(self, call_path_graph):
        """Should respect max_depth limit."""
        q = GraphQuerier(call_path_graph)
        path = q.find_call_path("handle_request", "validate_token", max_depth=1)

        # Path requires 2 hops but max_depth is 1
        assert len(path) == 0


class TestFlowDisambiguation:
    """Tests for trace_flow disambiguation when multiple matches exist."""

    @pytest.fixture
    def graph_with_duplicate_names(self):
        """Create a graph with multiple functions named 'new'."""
        # Create graph with two structs that each have a "new" method
        graph = {
            "schema_version": 1,
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
            "schema_version": 1,
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

    def test_expand_callees_shows_full_source(self, graph_with_callees):
        """expand_callees should show full source of direct callees."""
        graph_file, source_file = graph_with_callees
        q = GraphQuerier(graph_file)

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
            "schema_version": 1,
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

    def test_exact_node_id_works(self, graph_with_symbols):
        """Exact node ID should work without fuzzy matching."""
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file)

        result = q.get_context_prompt(f"FILE:{source_file}::validate_token")

        assert "Context for" in result
        assert "validate_token" in result
        assert "Matched" not in result  # No fuzzy match note

    def test_symbol_name_only_matches(self, graph_with_symbols):
        """Just symbol name should find the node via fuzzy matching."""
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file)

        # Just the function name, no FILE: prefix or path
        result = q.get_context_prompt("validate_token")

        assert "Context for" in result
        # Should have a fuzzy match note
        assert "Matched" in result
        assert "validate_token" in result

    def test_multiple_matches_shows_suggestions(self, graph_with_symbols):
        """When multiple nodes match, should show suggestions."""
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file)

        # "validate" matches both functions partially
        resolved_id, msg = q._resolve_node_id("validate")

        # Should not auto-resolve (ambiguous)
        assert resolved_id is None
        assert "Did you mean" in msg
        assert "validate_token" in msg
        assert "validate_token_format" in msg

    def test_camel_case_finds_snake_case(self, graph_with_symbols):
        """camelCase query should find snake_case function."""
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file)

        # camelCase version
        result = q.get_context_prompt("validateToken")

        assert "Context for" in result
        # Should match validate_token
        assert "validate_token" in result

    def test_partial_path_matches(self, graph_with_symbols):
        """Partial path should help narrow down matches."""
        graph_file, source_file = graph_with_symbols
        q = GraphQuerier(graph_file)

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
            "schema_version": 1,
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

    def test_full_false_truncates_large_function(self, large_function_graph):
        """Should truncate large functions when full=False."""
        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        source = q.get_smart_source(source_file, 1, 202, 2000, full=False)

        # Should contain truncation marker
        assert "lines omitted" in source
        # Should not contain all 200 lines
        assert source.count("let x") < 200

    def test_full_true_shows_complete_source(self, large_function_graph):
        """Should show complete source when full=True."""
        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        source = q.get_smart_source(source_file, 1, 202, 2000, full=True)

        # Should NOT contain truncation marker
        assert "lines omitted" not in source
        # Should contain all 200 lines
        assert source.count("let x") == 200

    def test_context_prompt_passes_full_parameter(self, large_function_graph):
        """Should pass full parameter through get_context_prompt."""
        graph_file, source_file = large_function_graph
        q = GraphQuerier(graph_file)

        node_id = f"FILE:{source_file}::large_function"

        # With full=False, should truncate
        context_truncated = q.get_context_prompt(node_id, full=False)
        assert "lines omitted" in context_truncated

        # With full=True, should not truncate
        context_full = q.get_context_prompt(node_id, full=True)
        assert "lines omitted" not in context_full
