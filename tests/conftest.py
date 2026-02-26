"""Shared fixtures for descry test suite."""

import json
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def sample_graph(tmp_path):
    """Create a minimal graph for testing."""
    graph = {
        "nodes": [
            {
                "id": "FILE:test.py::add",
                "type": "Function",
                "metadata": {
                    "name": "add",
                    "lineno": 1,
                    "token_count": 10,
                    "in_degree": 0,
                    "signature": "def add(a, b)",
                },
            }
        ],
        "edges": [],
    }
    graph_path = tmp_path / "codebase_graph.json"
    graph_path.write_text(json.dumps(graph))
    return graph_path
