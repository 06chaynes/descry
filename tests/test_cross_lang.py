"""Tests for descry.cross_lang — endpoint -> backend handler tracing."""

import json

import pytest

from descry.cross_lang import CrossLangTracer

SPEC = {
    "openapi": "3.0.0",
    "paths": {
        "/actions": {
            "get": {
                "operationId": "list_actions",
                "summary": "List actions",
                "tags": ["actions"],
            },
            "post": {"operationId": "create_action", "tags": ["actions"]},
            # Not an HTTP method: must not become an endpoint.
            "parameters": [{"name": "page", "in": "query"}],
        },
        "/actions/search": {
            "get": {"operationId": "search_actions", "tags": ["actions"]},
        },
        "/actions/{id}": {
            "get": {"operationId": "get_action", "tags": ["actions"]},
            "delete": {"operationId": "delete_action", "tags": ["actions"]},
        },
        "/files/{name}.json": {
            "get": {"operationId": "get_file_json", "tags": ["files"]},
        },
        "/health": {
            "get": {"operationId": "health_check", "tags": ["Ops"]},
            # No operationId: nothing to map to a handler.
            "put": {"summary": "anonymous"},
            # Outside the supported method set.
            "options": {"operationId": "health_options"},
        },
    },
}

GRAPH = {
    "schema_version": 2,
    "nodes": [
        {
            "id": "FILE:backend/src/routes/actions.rs::list_actions",
            "type": "Function",
            "metadata": {"name": "list_actions"},
        },
        {
            "id": "FILE:backend/src/routes/actions.rs::ActionRoutes::get_action",
            "type": "Method",
            "metadata": {"name": "get_action"},
        },
        {
            # Same name outside the handler directory, listed later so that
            # it wins when no pattern narrows the search.
            "id": "FILE:backend/tests/helpers.rs::list_actions",
            "type": "Function",
            "metadata": {"name": "list_actions"},
        },
        {
            # Right name, wrong node type.
            "id": "FILE:backend/src/routes/health.rs::health_check",
            "type": "Class",
            "metadata": {"name": "health_check"},
        },
    ],
    "edges": [],
}


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def spec_path(tmp_path):
    return _write_json(tmp_path / "openapi.json", SPEC)


@pytest.fixture
def graph_path(tmp_path):
    return _write_json(tmp_path / "codebase_graph.json", GRAPH)


@pytest.fixture
def tracer(spec_path):
    return CrossLangTracer(str(spec_path))


@pytest.fixture
def linked_tracer(spec_path, graph_path):
    return CrossLangTracer(
        str(spec_path),
        str(graph_path),
        backend_handler_patterns=["backend/src/routes"],
    )


class TestParseOpenapi:
    def test_loads_only_operations_with_an_id_and_a_supported_method(self, tracer):
        assert sorted(tracer.path_to_operation) == [
            ("DELETE", "/actions/{id}"),
            ("GET", "/actions"),
            ("GET", "/actions/search"),
            ("GET", "/actions/{id}"),
            ("GET", "/files/{name}.json"),
            ("GET", "/health"),
            ("POST", "/actions"),
        ]

    def test_operation_info_fields(self, tracer):
        assert tracer.path_to_operation[("GET", "/actions")] == {
            "operationId": "list_actions",
            "path": "/actions",
            "method": "GET",
            "summary": "List actions",
            "tags": ["actions"],
        }

    def test_missing_spec_yields_empty_tracer(self, tmp_path):
        tracer = CrossLangTracer(str(tmp_path / "absent.json"))
        assert tracer.path_to_operation == {}
        assert tracer.endpoint_to_handler("GET", "/actions") is None
        assert tracer.list_endpoints() == []

    @pytest.mark.parametrize(
        "spec",
        [[], "text", {}, {"paths": None}, {"paths": ["/actions"]}],
    )
    def test_spec_without_a_paths_object_yields_empty_tracer(self, tmp_path, spec):
        path = _write_json(tmp_path / "openapi.json", spec)
        tracer = CrossLangTracer(str(path))
        assert tracer.path_to_operation == {}
        assert tracer.endpoint_to_handler("GET", "/actions") is None

    def test_non_object_path_items_and_operations_are_skipped(self, tmp_path):
        spec = {
            "paths": {
                "/broken": None,
                "/listed": ["get"],
                "/mixed": {
                    "get": "list_mixed",
                    "post": None,
                    "delete": {"operationId": "delete_mixed"},
                },
            }
        }
        path = _write_json(tmp_path / "openapi.json", spec)
        tracer = CrossLangTracer(str(path))
        assert list(tracer.path_to_operation) == [("DELETE", "/mixed")]

    def test_malformed_spec_yields_empty_tracer(self, tmp_path):
        bad = tmp_path / "openapi.json"
        bad.write_text("{not json", encoding="utf-8")
        tracer = CrossLangTracer(str(bad))
        assert tracer.path_to_operation == {}
        assert tracer.get_stats()["total_endpoints"] == 0

    def test_undecodable_spec_yields_empty_tracer(self, tmp_path):
        bad = tmp_path / "openapi.json"
        bad.write_bytes(b'{"paths": "\xff"}')
        tracer = CrossLangTracer(str(bad))
        assert tracer.path_to_operation == {}


class TestEndpointToHandler:
    def test_exact_path(self, tracer):
        assert tracer.endpoint_to_handler("GET", "/actions") == "list_actions"
        assert tracer.endpoint_to_handler("POST", "/actions") == "create_action"

    def test_method_is_case_insensitive(self, tracer):
        assert tracer.endpoint_to_handler("get", "/actions") == "list_actions"

    def test_method_must_match(self, tracer):
        assert tracer.endpoint_to_handler("PATCH", "/actions") is None

    def test_path_parameter(self, tracer):
        assert tracer.endpoint_to_handler("GET", "/actions/abc123") == "get_action"
        assert tracer.endpoint_to_handler("DELETE", "/actions/7") == "delete_action"

    def test_path_parameter_does_not_span_segments(self, tracer):
        assert tracer.endpoint_to_handler("GET", "/actions/1/2") is None
        assert tracer.endpoint_to_handler("GET", "/actions/") is None

    def test_literal_path_beats_parameterised_one(self, tracer):
        assert tracer.endpoint_to_handler("GET", "/actions/search") == "search_actions"

    def test_regex_metacharacters_in_spec_path_are_literal(self, tracer):
        assert tracer.endpoint_to_handler("GET", "/files/a.json") == "get_file_json"
        # An unescaped "." would let any character stand in for the dot.
        assert tracer.endpoint_to_handler("GET", "/files/aXjson") is None

    def test_unknown_path(self, tracer):
        assert tracer.endpoint_to_handler("GET", "/nope") is None


class TestApiPrefixes:
    @pytest.mark.parametrize("prefix", ["/api/v1", "/api/v2", "/api"])
    def test_default_prefixes_are_stripped(self, tracer, prefix):
        assert tracer.endpoint_to_handler("GET", f"{prefix}/actions") == "list_actions"
        assert tracer.endpoint_to_handler("GET", f"{prefix}/actions/9") == "get_action"

    def test_prefix_stripping_applies_to_handler_info(self, tracer):
        info = tracer.get_handler_info("GET", "/api/v1/health")
        assert info["operationId"] == "health_check"
        assert info["path"] == "/health"

    def test_bare_prefix_maps_to_root(self, tmp_path):
        spec = _write_json(
            tmp_path / "root.json", {"paths": {"/": {"get": {"operationId": "index"}}}}
        )
        tracer = CrossLangTracer(str(spec))
        assert tracer.endpoint_to_handler("GET", "/api/v1") == "index"

    def test_custom_prefixes_replace_the_defaults(self, spec_path):
        tracer = CrossLangTracer(str(spec_path), api_prefixes=["/svc"])
        assert tracer.endpoint_to_handler("GET", "/svc/actions") == "list_actions"
        assert tracer.endpoint_to_handler("GET", "/api/v1/actions") is None

    def test_nested_prefixes_are_stripped_one_at_a_time(self, spec_path):
        tracer = CrossLangTracer(str(spec_path), api_prefixes=["/gateway", "/api"])
        assert (
            tracer.endpoint_to_handler("GET", "/gateway/api/actions") == "list_actions"
        )

    @pytest.mark.parametrize("prefixes", [[""], ["", "/api"], ["/"], ["/", ""]])
    @pytest.mark.parametrize("path", ["/nope", "/", "", "/api/nope"])
    def test_prefix_that_strips_nothing_does_not_recurse_forever(
        self, spec_path, prefixes, path
    ):
        tracer = CrossLangTracer(str(spec_path), api_prefixes=prefixes)
        assert tracer.endpoint_to_handler("GET", path) is None
        assert tracer.get_handler_info("GET", path) is None

    def test_empty_prefix_does_not_hide_the_ones_after_it(self, spec_path):
        tracer = CrossLangTracer(str(spec_path), api_prefixes=["", "/api"])
        assert tracer.endpoint_to_handler("GET", "/api/actions") == "list_actions"
        assert tracer.get_handler_info("GET", "/api/actions")["path"] == "/actions"


class TestGraphLinking:
    def test_handler_patterns_select_the_backend_node(self, linked_tracer):
        assert linked_tracer.operation_to_handler == {
            "list_actions": "FILE:backend/src/routes/actions.rs::list_actions",
            "get_action": (
                "FILE:backend/src/routes/actions.rs::ActionRoutes::get_action"
            ),
        }

    def test_name_that_two_functions_have_links_neither(self, spec_path, graph_path):
        tracer = CrossLangTracer(str(spec_path), str(graph_path))
        assert "list_actions" not in tracer.operation_to_handler
        assert "get_action" in tracer.operation_to_handler

    def test_function_outside_test_code_is_the_handler(self, spec_path, graph_path):
        tracer = CrossLangTracer(
            str(spec_path),
            str(graph_path),
            is_test=lambda node_id: "/tests/" in node_id,
        )
        assert tracer.operation_to_handler["list_actions"] == (
            "FILE:backend/src/routes/actions.rs::list_actions"
        )

    @pytest.mark.parametrize(
        "head",
        [
            '#[utoipa::path(\n    get,\n    path = "/api/v1/actions",\n)]\n\n',
            '#[get("/actions")]\n/// Lists them.\n',
            '@Get("/actions")\n/** Lists them. */\n',
        ],
    )
    def test_function_that_declares_the_path_is_the_handler(
        self, spec_path, tmp_path, head
    ):
        sources = {
            "client/api.rs": (
                'fn url() -> String {\n    api("/actions")\n}\n\n'
                '/// Lists what "/actions" answers.\n/// [see "/actions"]\n'
                '/** @see "/actions" */\nasync fn list_actions() {}\n'
            ),
            "server/routes.rs": f"{head}async fn list_actions() {{}}\n",
        }
        nodes = []
        for rel_path, text in sources.items():
            (tmp_path / rel_path).parent.mkdir()
            (tmp_path / rel_path).write_text(text)
            nodes.append(
                {
                    "id": f"FILE:{rel_path}::list_actions",
                    "type": "Function",
                    "metadata": {"name": "list_actions", "lineno": text.count("\n")},
                }
            )
        graph = tmp_path / "graph.json"
        graph.write_text(json.dumps({**GRAPH, "nodes": nodes}))
        tracer = CrossLangTracer(str(spec_path), str(graph), project_root=tmp_path)
        assert tracer.operation_to_handler["list_actions"] == (
            "FILE:server/routes.rs::list_actions"
        )
        unread = CrossLangTracer(
            str(spec_path), str(graph), project_root=tmp_path, max_file_bytes=8
        )
        assert "list_actions" not in unread.operation_to_handler

    def test_non_callable_nodes_are_not_handlers(self, spec_path, graph_path):
        tracer = CrossLangTracer(str(spec_path), str(graph_path))
        assert "health_check" not in tracer.operation_to_handler

    def test_handler_info_carries_node_id(self, linked_tracer):
        exact = linked_tracer.get_handler_info("GET", "/actions")
        assert exact["node_id"] == "FILE:backend/src/routes/actions.rs::list_actions"
        by_pattern = linked_tracer.get_handler_info("GET", "/api/v1/actions/42")
        assert by_pattern["operationId"] == "get_action"
        assert by_pattern["node_id"].endswith("::get_action")
        unlinked = linked_tracer.get_handler_info("POST", "/actions")
        assert unlinked["node_id"] is None

    def test_handler_info_is_a_copy(self, tracer):
        info = tracer.get_handler_info("GET", "/actions")
        info["operationId"] = "tampered"
        assert tracer.endpoint_to_handler("GET", "/actions") == "list_actions"
        assert "node_id" not in tracer.path_to_operation[("GET", "/actions")]

    def test_missing_graph_leaves_endpoints_unlinked(self, spec_path, tmp_path):
        tracer = CrossLangTracer(str(spec_path), str(tmp_path / "absent.json"))
        assert tracer.operation_to_handler == {}
        assert tracer.endpoint_to_handler("GET", "/actions") == "list_actions"

    def test_stale_schema_graph_is_rejected_not_linked(self, spec_path, tmp_path):
        stale = _write_json(tmp_path / "stale.json", {**GRAPH, "schema_version": 0})
        tracer = CrossLangTracer(str(spec_path), str(stale))
        assert tracer.operation_to_handler == {}
        assert tracer.get_stats()["total_endpoints"] == 7

    def test_undecodable_graph_leaves_endpoints_unlinked(self, spec_path, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_bytes(b'{"schema_version": 2, "nodes": [], "edges": [], "x": "\xff"}')
        tracer = CrossLangTracer(str(spec_path), str(bad))
        assert tracer.operation_to_handler == {}
        assert tracer.get_stats()["total_endpoints"] == 7


class TestListEndpointsAndStats:
    def test_sorted_by_path_then_method(self, tracer):
        listed = [(e["path"], e["method"]) for e in tracer.list_endpoints()]
        assert listed == sorted(listed)
        assert len(listed) == 7

    def test_tag_filter_is_case_insensitive(self, tracer):
        assert [e["operationId"] for e in tracer.list_endpoints(tag="ops")] == [
            "health_check"
        ]
        assert len(tracer.list_endpoints(tag="ACTIONS")) == 5
        assert tracer.list_endpoints(tag="missing") == []

    def test_entries_include_node_id(self, linked_tracer):
        by_op = {e["operationId"]: e["node_id"] for e in linked_tracer.list_endpoints()}
        assert by_op["list_actions"].startswith("FILE:backend/src/routes/")
        assert by_op["create_action"] is None

    def test_stats(self, linked_tracer, spec_path, graph_path):
        assert linked_tracer.get_stats() == {
            "total_endpoints": 7,
            "linked_to_graph": 2,
            "openapi_path": str(spec_path),
            "graph_path": str(graph_path),
        }

    def test_stats_without_graph(self, tracer):
        stats = tracer.get_stats()
        assert stats["linked_to_graph"] == 0
        assert stats["graph_path"] is None
