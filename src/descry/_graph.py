"""Graph schema versioning.

Centralized load helper for codebase_graph.json with schema_version check.
All callers that deserialize the graph should use load_graph_with_schema to
ensure they reject out-of-date graphs instead of silently producing wrong
results.
"""

import json
from pathlib import Path


CURRENT_SCHEMA = 1


class GraphSchemaError(Exception):
    """Raised when a graph file's schema_version does not match CURRENT_SCHEMA."""


def load_graph_with_schema(path: Path | str) -> dict:
    """Load a codebase graph JSON file and verify its schema_version.

    Raises:
        GraphSchemaError: if schema_version is missing or mismatched.
        FileNotFoundError: if path does not exist.
        json.JSONDecodeError: if path is not valid JSON.
    """
    with open(path) as f:
        data = json.load(f)
    ver = data.get("schema_version")
    if ver != CURRENT_SCHEMA:
        raise GraphSchemaError(
            f"Graph schema version {ver!r} (expected {CURRENT_SCHEMA}); "
            f"run 'descry index' to rebuild."
        )
    return data
