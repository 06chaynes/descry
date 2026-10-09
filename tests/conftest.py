"""Shared fixtures for descry test suite."""

import contextlib
import json
import os
import signal
import sys
import time

import pytest


@pytest.fixture
def sample_graph(tmp_path):
    """Create a minimal graph for testing."""
    graph = {
        "schema_version": 2,
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


# A killed process can linger as a zombie, so its pid does not say whether it
# is gone; the lock it held does.
_INDEXER_TREE = """
import os, subprocess, sys, time
lock, ready = sys.argv[1:]
grandchild = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import fcntl, sys, time\\n"
        "held = open(sys.argv[1], 'w')\\n"
        "fcntl.flock(held, fcntl.LOCK_EX)\\n"
        "print('locked', flush=True)\\n"
        "time.sleep(120)",
        lock,
    ],
    stdout=subprocess.PIPE,
)
grandchild.stdout.readline()
with open(ready, "w") as f:
    f.write(f"{os.getpid()} {grandchild.pid}")
print("working", flush=True)
time.sleep(120)
"""


class IndexerTree:
    """Stands in for the indexer: a process that starts one of its own and hangs.

    `argv` runs it. It prints `working` once the process it started is up.
    """

    def __init__(self, tmp_path):
        self._lock = tmp_path / "indexer-tree.lock"
        self._ready = tmp_path / "indexer-tree.ready"
        self.argv = [
            sys.executable,
            "-c",
            _INDEXER_TREE,
            str(self._lock),
            str(self._ready),
        ]

    def started(self) -> bool:
        return self._ready.exists()

    def wait_started(self, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while not self.started():
            assert time.monotonic() < deadline, "the stand-in indexer never started"
            time.sleep(0.05)

    def grandchild_exited(self, timeout: float = 10) -> bool:
        """Whether the process the stand-in started is gone, waiting up to `timeout`."""
        import fcntl

        deadline = time.monotonic() + timeout
        with open(self._lock, "a") as held:
            while True:
                try:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        return False
                    time.sleep(0.05)

    def kill(self) -> None:
        if self.started():
            for pid in self._ready.read_text().split():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pid), signal.SIGKILL)


@pytest.fixture
def indexer_tree(tmp_path):
    pytest.importorskip("fcntl")
    tree = IndexerTree(tmp_path)
    yield tree
    tree.kill()
