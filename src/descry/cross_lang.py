#!/usr/bin/env python3
"""
Cross-Language Tracing Module

Maps frontend API calls to backend handlers via OpenAPI specification.
Enables queries like "what backend handler does this frontend function call?"

Architecture:
- Frontend (TypeScript) uses apiClient.get/post/put/delete(endpoint)
- Backend (Rust) has handlers with operationId matching function names
- OpenAPI spec provides the bridge: path -> operationId -> handler function

Usage:
    config = DescryConfig(openapi_path="public/api/openapi.json", ...)
    tracer = CrossLangTracer(config)
    handler = tracer.endpoint_to_handler("GET", "/api/v1/actions")
    # Returns: "list_actions"
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CrossLangTracer:
    """Traces API calls across frontend/backend language boundaries."""

    def __init__(
        self,
        openapi_path: str,
        graph_path: Optional[str] = None,
        backend_handler_patterns: list[str] | None = None,
        frontend_api_patterns: list[str] | None = None,
        api_prefixes: list[str] | None = None,
    ):
        """Initialize the tracer with OpenAPI spec.

        Args:
            openapi_path: Path to OpenAPI JSON spec
            graph_path: Optional path to codebase graph for enriched lookups
            backend_handler_patterns: Path substrings identifying backend handler files
                (e.g., ["backend/src/routes"]). If empty, handler filtering is skipped.
            frontend_api_patterns: Path substrings identifying frontend API files
                (e.g., ["webapp/src/lib/api"]). If empty, frontend filtering is skipped.
            api_prefixes: API version prefixes to strip when matching endpoints
                (defaults to ["/api/v1", "/api/v2", "/api"]).
        """
        self.openapi_path = Path(openapi_path)
        self.graph_path = Path(graph_path) if graph_path else None
        self.backend_handler_patterns = backend_handler_patterns or []
        self.frontend_api_patterns = frontend_api_patterns or []
        self._API_PREFIXES = (
            tuple(api_prefixes) if api_prefixes else ("/api/v1", "/api/v2", "/api")
        )

        # Parsed mappings
        self.path_to_operation: dict[
            tuple[str, str], dict
        ] = {}  # (method, path) -> operation info
        self.operation_to_handler: dict[str, str] = {}  # operationId -> handler node_id
        self.endpoint_patterns: list[
            tuple[str, str, re.Pattern, dict]
        ] = []  # For path param matching

        self._parse_openapi()
        if self.graph_path and self.graph_path.exists():
            self._link_to_graph()

    def _parse_openapi(self):
        """Parse OpenAPI spec to extract endpoint -> operationId mappings."""
        if not self.openapi_path.exists():
            logger.warning(f"OpenAPI spec not found: {self.openapi_path}")
            return

        try:
            with open(self.openapi_path) as f:
                spec = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse OpenAPI spec: {e}")
            return

        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                if method not in ("get", "post", "put", "delete", "patch"):
                    continue

                operation_id = details.get("operationId", "")
                if not operation_id:
                    continue

                method_upper = method.upper()
                operation_info = {
                    "operationId": operation_id,
                    "path": path,
                    "method": method_upper,
                    "summary": details.get("summary", ""),
                    "tags": details.get("tags", []),
                }

                self.path_to_operation[(method_upper, path)] = operation_info

                # Create regex pattern for path with parameters.
                # Escape the path first so literal `.`, `+`, `(` in routes are
                # matched literally, then replace OpenAPI `{param}` markers
                # (which escape to `\{param\}`) with the `[^/]+` capture.
                escaped = re.escape(path)
                pattern_str = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", escaped)
                pattern = re.compile(f"^{pattern_str}$")
                self.endpoint_patterns.append(
                    (method_upper, path, pattern, operation_info)
                )

        logger.info(
            f"Loaded {len(self.path_to_operation)} API endpoints from OpenAPI spec"
        )

    def _link_to_graph(self):
        """Link operationIds to graph node IDs for backend handlers."""
        if not self.graph_path or not self.graph_path.exists():
            return

        try:
            from descry._graph import load_graph_with_schema

            graph = load_graph_with_schema(self.graph_path)
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return

        # Build index of function/method names to node IDs
        # Looking for handlers matching backend_handler_patterns
        handler_index = {}
        for node in graph.get("nodes", []):
            if node["type"] not in ("Function", "Method"):
                continue
            node_id = node["id"]
            # If patterns are configured, filter to matching files only
            if self.backend_handler_patterns:
                if not any(pat in node_id for pat in self.backend_handler_patterns):
                    continue
            name = node.get("metadata", {}).get("name", "")
            if name:
                handler_index[name] = node_id

        # Map operationIds to handlers
        for (method, path), info in self.path_to_operation.items():
            op_id = info["operationId"]
            if op_id in handler_index:
                self.operation_to_handler[op_id] = handler_index[op_id]

        logger.info(
            f"Linked {len(self.operation_to_handler)} operationIds to graph nodes"
        )

    def _strip_api_prefix(self, path: str) -> Optional[str]:
        """Strip known API version prefix from a path for matching against OpenAPI spec paths.

        Returns the stripped path, or None if no prefix matched.
        """
        for prefix in self._API_PREFIXES:
            if path.startswith(prefix):
                stripped = path[len(prefix) :]
                return stripped if stripped else "/"
        return None

    def endpoint_to_handler(self, method: str, path: str) -> Optional[str]:
        """Find the backend handler for an API endpoint.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: API path (e.g., "/api/v1/actions" or "/api/v1/actions/123")

        Returns:
            operationId (handler function name) or None if not found
        """
        method_upper = method.upper()

        # Try exact match first
        key = (method_upper, path)
        if key in self.path_to_operation:
            return self.path_to_operation[key]["operationId"]

        # Try pattern matching for paths with IDs
        for pattern_method, template_path, pattern, info in self.endpoint_patterns:
            if pattern_method == method_upper and pattern.match(path):
                return info["operationId"]

        # Fallback: strip API version prefix and retry.
        # Handles runtime paths (e.g., /api/v1/health) vs spec paths (e.g., /health).
        stripped = self._strip_api_prefix(path)
        if stripped is not None:
            return self.endpoint_to_handler(method, stripped)

        return None

    def endpoint_to_node_id(self, method: str, path: str) -> Optional[str]:
        """Find the graph node ID for a backend handler.

        Args:
            method: HTTP method
            path: API path

        Returns:
            Full node ID (e.g., "FILE:backend/src/routes/actions/handlers.rs::list_actions")
        """
        op_id = self.endpoint_to_handler(method, path)
        if op_id:
            return self.operation_to_handler.get(op_id)
        return None

    def get_handler_info(self, method: str, path: str) -> Optional[dict]:
        """Get full information about an endpoint's handler.

        Returns dict with operationId, path, method, summary, tags, node_id.
        """
        method_upper = method.upper()

        # Try exact match first
        key = (method_upper, path)
        if key in self.path_to_operation:
            info = self.path_to_operation[key].copy()
            info["node_id"] = self.operation_to_handler.get(info["operationId"])
            return info

        # Try pattern matching
        for pattern_method, template_path, pattern, info in self.endpoint_patterns:
            if pattern_method == method_upper and pattern.match(path):
                result = info.copy()
                result["node_id"] = self.operation_to_handler.get(info["operationId"])
                return result

        # Fallback: strip API version prefix and retry
        stripped = self._strip_api_prefix(path)
        if stripped is not None:
            return self.get_handler_info(method, stripped)

        return None

    def list_endpoints(self, tag: Optional[str] = None) -> list[dict]:
        """List all endpoints, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by (e.g., "actions", "deployments")

        Returns:
            List of endpoint info dicts
        """
        results = []
        for info in self.path_to_operation.values():
            if tag and tag.lower() not in [t.lower() for t in info.get("tags", [])]:
                continue
            result = info.copy()
            result["node_id"] = self.operation_to_handler.get(info["operationId"])
            results.append(result)
        return sorted(results, key=lambda x: (x["path"], x["method"]))

    def find_ts_api_calls(self, graph_data: dict) -> list[dict]:
        """Find frontend functions that call API endpoints.

        Looks for patterns like:
        - apiClient.get('/actions')
        - createCrudApi('/tags')
        - API_BASE_URL + '/api/v1/...'

        Args:
            graph_data: Parsed codebase graph

        Returns:
            List of dicts with ts_node_id, endpoint, method, backend_handler
        """
        # This would require parsing TypeScript source to find API calls
        # For now, we use heuristics based on file names and function names
        results = []

        for node in graph_data.get("nodes", []):
            if node["type"] not in ("Function", "Method"):
                continue
            node_id = node["id"]
            # If patterns are configured, filter to matching files only
            if self.frontend_api_patterns:
                if not any(pat in node_id for pat in self.frontend_api_patterns):
                    continue

            # Infer endpoint from file name
            # e.g., webapp/src/lib/api/actions.ts -> /actions
            parts = node_id.split("/")
            if "api" in parts:
                api_idx = parts.index("api")
                if api_idx + 1 < len(parts):
                    file_name = parts[api_idx + 1].split("::")[0]
                    endpoint_name = file_name.replace(".ts", "")

                    # Map function names to HTTP methods
                    func_name = node.get("metadata", {}).get("name", "")
                    method = self._infer_http_method(func_name)

                    # Try to find matching backend handler under any configured
                    # API prefix (E.3: previous hardcoded /api/v1/ removed).
                    handler = None
                    api_path = None
                    for prefix in self._API_PREFIXES:
                        candidate = f"{prefix}/{endpoint_name}".replace("//", "/")
                        h = self.endpoint_to_handler(method, candidate)
                        if h:
                            handler = h
                            api_path = candidate
                            break
                    if handler is None:
                        api_path = f"{self._API_PREFIXES[0] if self._API_PREFIXES else '/api'}/{endpoint_name}"

                    if handler:
                        results.append(
                            {
                                "ts_node_id": node_id,
                                "ts_function": func_name,
                                "inferred_endpoint": api_path,
                                "inferred_method": method,
                                "rust_handler": handler,
                                "rust_node_id": self.operation_to_handler.get(handler),
                            }
                        )

        return results

    def _infer_http_method(self, func_name: str) -> str:
        """Infer HTTP method from function name."""
        name_lower = func_name.lower()
        if name_lower.startswith(("get", "list", "fetch", "load")):
            return "GET"
        if name_lower.startswith(("create", "add", "post")):
            return "POST"
        if name_lower.startswith(("update", "put", "edit", "modify")):
            return "PUT"
        if name_lower.startswith(("delete", "remove")):
            return "DELETE"
        return "GET"  # Default

    def get_stats(self) -> dict:
        """Get statistics about the cross-language mapping."""
        return {
            "total_endpoints": len(self.path_to_operation),
            "linked_to_graph": len(self.operation_to_handler),
            "openapi_path": str(self.openapi_path),
            "graph_path": str(self.graph_path) if self.graph_path else None,
        }


def _create_cross_lang_edges(
    graph_data: dict,
    openapi_path: str,
) -> list[dict]:
    """Create CALLS_API edges linking frontend to backend handlers.

    This can be used during graph generation to add cross-language edges.

    Args:
        graph_data: Current graph data (nodes and edges)
        openapi_path: Path to OpenAPI spec

    Returns:
        List of new edges to add
    """
    tracer = CrossLangTracer(openapi_path)
    api_calls = tracer.find_ts_api_calls(graph_data)

    edges = []
    for call in api_calls:
        if call.get("rust_node_id"):
            edges.append(
                {
                    "source": call["ts_node_id"],
                    "target": call["rust_node_id"],
                    "relation": "CALLS_API",
                    "metadata": {
                        "endpoint": call["inferred_endpoint"],
                        "method": call["inferred_method"],
                        "handler": call["rust_handler"],
                    },
                }
            )

    return edges


if __name__ == "__main__":
    # Generic stats-only smoke test. Prints OpenAPI + graph linkage stats.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m descry.cross_lang <openapi_path> [graph_path]")
        sys.exit(1)

    openapi_path = sys.argv[1]
    graph_path = sys.argv[2] if len(sys.argv) > 2 else None

    tracer = CrossLangTracer(openapi_path, graph_path)
    stats = tracer.get_stats()
    print("CrossLangTracer stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
