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
from collections import defaultdict
from pathlib import Path

from descry._graph import GraphSchemaError, load_graph_with_schema
from descry.query import source_path

logger = logging.getLogger(__name__)

# A string literal on one line
_RE_STRING = re.compile(r"""["']([^"'\n]*)["']""")


# What a line of an attribute or a decorator starts with, and of a comment
_HEADS = ("@", "[", "#[")
_COMMENTS = ("#", "//", "/*", "*")


def _head(lines: list[str]) -> str:
    """What heads the definition below ``lines``: its attributes and
    decorators. Comment lines, and a blank line between the head and the
    definition, are passed over."""
    head, depth = [], 0
    for line in reversed(lines):
        text = line.strip()
        if not text:
            if head:
                break
            continue
        # Read upwards, a `)` or `]` opens what a `(` or `[` closes.
        opened = depth + sum(map(text.count, ")]")) - sum(map(text.count, "(["))
        if depth <= 0 and not text.startswith(_HEADS):
            if text.startswith(_COMMENTS):
                continue
            if opened <= 0:
                break
        head.append(text)
        depth = max(opened, 0)
    return "\n".join(head)


def _route(path: str | None) -> str | None:
    """``path`` with each parameter, written `{id}`, `:id` or `<id>`, as
    `{}`, so that a route reads the same in a specification and in code."""
    return path and re.sub(r"\{[^}/]*\}|:\w+|<[^>/]*>", "{}", path).rstrip("/")


class CrossLangTracer:
    """Traces API calls across frontend/backend language boundaries."""

    def __init__(
        self,
        openapi_path: str,
        graph_path: str | None = None,
        backend_handler_patterns: list[str] | None = None,
        api_prefixes: list[str] | None = None,
        is_test=None,
        project_root=None,
        max_file_bytes=0,
    ):
        """Initialize the tracer with OpenAPI spec.

        Args:
            openapi_path: Path to OpenAPI JSON spec
            graph_path: Optional path to codebase graph for enriched lookups
            backend_handler_patterns: Path substrings identifying backend handler files
                (e.g., ["backend/src/routes"]). If empty, handler filtering is skipped.
            api_prefixes: API version prefixes to strip when matching endpoints
                (defaults to ["/api/v1", "/api/v2", "/api"]).
            is_test: Tells whether a node ID is test code.
            project_root: Where the graph's files are, to read what heads a
                function that shares its name with another.
            max_file_bytes: A larger file is not read, when not 0.
        """
        self.is_test = is_test
        self.project_root = project_root
        self.max_file_bytes = max_file_bytes
        self.openapi_path = Path(openapi_path)
        self.graph_path = Path(graph_path) if graph_path else None
        self.backend_handler_patterns = backend_handler_patterns or []
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
            with open(self.openapi_path, encoding="utf-8") as f:
                spec = json.load(f)
        except (ValueError, OSError) as e:
            logger.error(f"Failed to parse OpenAPI spec: {e}")
            return

        paths = spec.get("paths") if isinstance(spec, dict) else None
        if not isinstance(paths, dict):
            return

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, details in methods.items():
                if method not in ("get", "post", "put", "delete", "patch"):
                    continue
                if not isinstance(details, dict):
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
        """Link each operation to the graph node of its handler, the function
        its operationId names. Where several functions have the name, the
        handler is the one whose attributes or decorators write the
        operation's path, else the one outside test code; while more than
        one is left, none is linked."""
        if not self.graph_path or not self.graph_path.exists():
            return

        try:
            graph = load_graph_with_schema(self.graph_path)
        except (OSError, ValueError, GraphSchemaError) as e:
            logger.error(f"Failed to load graph: {e}")
            return

        # The functions and methods of each name and of each file, in the
        # files that backend_handler_patterns allows.
        named, filed = defaultdict(list), defaultdict(list)
        for node in graph.get("nodes", []):
            if node["type"] not in ("Function", "Method"):
                continue
            if self.backend_handler_patterns and not any(
                pat in node["id"] for pat in self.backend_handler_patterns
            ):
                continue
            name = node.get("metadata", {}).get("name", "")
            if name:
                named[name].append(node)
                filed[node["id"].split("::")[0]].append(node)

        sources, declared, pending = {}, set(), []
        tested = self.is_test or (lambda _node_id: False)
        for info in self.path_to_operation.values():
            found = named.get(info["operationId"], [])
            declaring = [n for n in found if self._declares(n, info, sources)]
            if len(declaring) == 1:
                declared.add(declaring[0]["id"].split("::")[0])
            else:
                pending.append(info)
            for narrowed in (declaring, [n for n in found if not tested(n["id"])]):
                if len(found) > 1 and narrowed:
                    found = [n for n in found if n in narrowed]
            if len(found) == 1:
                self.operation_to_handler[info["operationId"]] = found[0]["id"]
        # A handler need not have its operation's name: one is looked for
        # among the functions of the files that hold the declared handlers.
        for info in pending:
            declaring = [
                node
                for file_id in declared
                for node in filed[file_id]
                if self._declares(node, info, sources, method=True)
            ]
            if len(declaring) == 1:
                self.operation_to_handler[info["operationId"]] = declaring[0]["id"]

        logger.info(
            f"Linked {len(self.operation_to_handler)} operationIds to graph nodes"
        )

    def _declares(self, node, info, sources, method=False) -> bool:
        """Whether what heads ``node``'s definition, its attributes or
        decorators, writes the path of the operation ``info`` in a string,
        with or without its API prefix, and for ``method`` names its HTTP
        method too. ``sources`` keeps the lines of the files read."""
        rel_path = node["id"][5:].split("::")[0]
        lineno = node.get("metadata", {}).get("lineno")
        if rel_path not in sources:
            located = self.project_root and source_path(
                self.project_root, rel_path, self.max_file_bytes
            )
            text = Path(located).read_text("utf-8", "ignore") if located else ""
            sources[rel_path] = text.split("\n")
        if not isinstance(lineno, int):
            return False
        head = _head(sources[rel_path][: lineno - 1])
        if method and info["method"].lower() not in head.lower():
            return False

        def routes(text):
            return {_route(text), _route(self._strip_api_prefix(text))} - {None}

        return any(
            routes(text) & routes(info["path"]) for text in _RE_STRING.findall(head)
        )

    def _strip_api_prefix(self, path: str) -> str | None:
        """Strip known API version prefix from a path for matching against OpenAPI spec paths.

        Returns the stripped path, or None if no prefix matched.
        """
        for prefix in self._API_PREFIXES:
            if path.startswith(prefix):
                stripped = path[len(prefix) :] or "/"
                # Callers retry the lookup with the result, so it must differ
                # from the input: an empty prefix, or "/" against "/", would
                # otherwise have them recurse on the same path forever.
                if stripped != path:
                    return stripped
        return None

    def endpoint_to_handler(self, method: str, path: str) -> str | None:
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

    def get_handler_info(self, method: str, path: str) -> dict | None:
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

    def list_endpoints(self, tag: str | None = None) -> list[dict]:
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
