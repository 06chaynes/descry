"""Graph schema versioning.

Centralized load helper for codebase_graph.json with schema_version check.
All callers that deserialize the graph should use load_graph_with_schema to
ensure they reject out-of-date graphs instead of silently producing wrong
results.
"""

import json
from pathlib import Path

CURRENT_SCHEMA = 2


class GraphSchemaError(Exception):
    """Raised when a graph file is not a graph this version can read.

    Either its schema_version does not match CURRENT_SCHEMA, or it is valid
    JSON of the wrong shape (hand-edited, truncated and re-closed, or not a
    graph at all).
    """


def _malformed(what: str) -> GraphSchemaError:
    return GraphSchemaError(
        f"Graph file is malformed ({what}); run 'descry index' to rebuild."
    )


# Node metadata that consumers do arithmetic on, and that they call string
# methods on. Each is optional.
_INT_FIELDS = ("lineno", "end_lineno", "token_count", "in_degree")
_STR_FIELDS = ("name", "signature", "docstring", "path")


def _check_shape(data: dict) -> None:
    """Verify the node/edge shape every consumer indexes into.

    Checked once here so consumers can use `node["id"]`, `node["type"]`,
    `node["metadata"]`, `edge["source"]`, `edge["target"]` and
    `edge["relation"]`, do arithmetic on `_INT_FIELDS` and on an edge's line
    number, and call string methods on `_STR_FIELDS`, without each guarding
    against a malformed file. A node with no metadata is given an empty one.
    """
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list):
        raise _malformed("'nodes' is not a list")
    if not isinstance(edges, list):
        raise _malformed("'edges' is not a list")

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise _malformed(f"node {i} is not an object")
        if not isinstance(node.get("id"), str) or not isinstance(node.get("type"), str):
            raise _malformed(f"node {i} lacks a string 'id' or 'type'")
        if not isinstance(node.setdefault("metadata", {}), dict):
            raise _malformed(f"node {i} has non-object 'metadata'")
        for key in _INT_FIELDS:
            if not isinstance(node["metadata"].get(key, 0), int):
                raise _malformed(f"node {i} has a non-integer '{key}'")
        for key in _STR_FIELDS:
            if not isinstance(node["metadata"].get(key, ""), str):
                raise _malformed(f"node {i} has a non-string '{key}'")

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise _malformed(f"edge {i} is not an object")
        if not all(
            isinstance(edge.get(key), str) for key in ("source", "target", "relation")
        ):
            raise _malformed(
                f"edge {i} lacks a string 'source', 'target' or 'relation'"
            )
        if not isinstance(edge.get("metadata", {}), dict):
            raise _malformed(f"edge {i} has non-object 'metadata'")
        if not isinstance(edge.get("metadata", {}).get("lineno", 0), int):
            raise _malformed(f"edge {i} has a non-integer 'lineno'")
        candidates = edge.get("metadata", {}).get("candidates", [])
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, str) for candidate in candidates
        ):
            raise _malformed(f"edge {i} has 'candidates' that is not a list of strings")


def load_graph_with_schema(path: Path | str) -> dict:
    """Load a codebase graph JSON file and verify its schema_version and shape.

    Raises:
        GraphSchemaError: if the file is not a JSON object, schema_version is
            missing or mismatched, or a node or edge is malformed.
        FileNotFoundError: if path does not exist.
        json.JSONDecodeError: if path is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise _malformed("top level is not an object")
    ver = data.get("schema_version")
    if ver != CURRENT_SCHEMA:
        raise GraphSchemaError(
            f"Graph schema version {ver!r} (expected {CURRENT_SCHEMA}); "
            f"run 'descry index' to rebuild."
        )
    _check_shape(data)
    return data
