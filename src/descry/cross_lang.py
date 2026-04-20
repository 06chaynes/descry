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
        for info in self.path_to_operation.values():
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
        for pattern_method, _, pattern, info in self.endpoint_patterns:
            if pattern_method == method_upper and pattern.match(path):
                return info["operationId"]

        # Fallback: strip API version prefix and retry.
        # Handles runtime paths (e.g., /api/v1/health) vs spec paths (e.g., /health).
        stripped = self._strip_api_prefix(path)
        if stripped is not None:
            return self.endpoint_to_handler(method, stripped)

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
        for pattern_method, _, pattern, info in self.endpoint_patterns:
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

    def get_stats(self) -> dict:
        """Get statistics about the cross-language mapping."""
        return {
            "total_endpoints": len(self.path_to_operation),
            "linked_to_graph": len(self.operation_to_handler),
            "openapi_path": str(self.openapi_path),
            "graph_path": str(self.graph_path) if self.graph_path else None,
        }


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
