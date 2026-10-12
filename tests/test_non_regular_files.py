"""A repository can put a FIFO where descry expects a file. Each reader below
leaves it alone instead of waiting on it."""

import asyncio
import os
import sys

import pytest

from descry import handlers
from descry._graph import GraphSchemaError, load_graph_with_schema
from descry.cross_lang import CrossLangTracer
from descry.generate import CodeGraphBuilder
from descry.handlers import DescryConfig
from descry.scip.adapters.dotnet import _global_json_satisfiable
from descry.scip.adapters.java import _runs_gradle
from descry.scip.adapters.php import _scip_php_incompatibility_reason
from descry.scip.adapters.rust import RustAdapter
from descry.scip.adapters.typescript import TypeScriptAdapter
from descry.scip.cache import ScipCacheManager
from descry.scip.parser import ScipIndex

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs"),
    pytest.mark.usefixtures("no_wait"),
]


def _fifo(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)
    return path


def test_walk_skips_a_fifo_source_file(tmp_path):
    (tmp_path / "ok.py").write_text("def f():\n    return 1\n")
    _fifo(tmp_path / "bad.py")
    builder = CodeGraphBuilder(tmp_path)
    builder.use_ast_grep = False

    builder.process_directory()

    files = {n["id"] for n in builder.nodes if n["type"] == "File"}
    assert files == {"FILE:ok.py"}


def test_fifo_config_is_ignored(tmp_path):
    _fifo(tmp_path / ".descry.toml")

    assert DescryConfig._load_toml(tmp_path) == {}


def test_fifo_graph_is_malformed(tmp_path):
    graph = _fifo(tmp_path / "codebase_graph.json")

    with pytest.raises(GraphSchemaError, match="not a regular file"):
        load_graph_with_schema(graph)


@pytest.mark.parametrize(
    ("adapter", "manifest"),
    [
        (TypeScriptAdapter(), "package.json"),
        (TypeScriptAdapter(), "pnpm-workspace.yaml"),
        (RustAdapter(), "Cargo.toml"),
    ],
)
def test_discovery_leaves_a_fifo_manifest(tmp_path, adapter, manifest):
    _fifo(tmp_path / manifest)

    adapter.discover(tmp_path, set())


def test_rust_discovery_leaves_a_fifo_member_manifest(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/a"]\n')
    (tmp_path / "crates" / "a" / "src").mkdir(parents=True)
    _fifo(tmp_path / "crates" / "a" / "Cargo.toml")
    adapter = RustAdapter()

    assert adapter.discover(tmp_path, set()) == []
    assert adapter._packages(tmp_path) == set()


def test_root_crate_is_named_from_its_own_manifest(tmp_path):
    """A subdirectory named like the root, holding a FIFO manifest, is not
    taken for the root crate."""
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "Cargo.toml").write_text('[package]\nname = "root"\n')
    (root / "proj" / "src").mkdir(parents=True)
    _fifo(root / "proj" / "Cargo.toml")

    assert RustAdapter()._packages(root) == {"root"}


def test_java_leaves_a_fifo_build_properties(tmp_path):
    (tmp_path / "settings.gradle").write_text("")
    _fifo(tmp_path / "project" / "build.properties")

    assert _runs_gradle(tmp_path) is True


def test_fifo_openapi_spec_is_left(tmp_path):
    tracer = CrossLangTracer(str(_fifo(tmp_path / "openapi.json")))

    assert tracer.operation_to_handler == {}


def test_php_and_dotnet_leave_a_fifo_manifest(tmp_path):
    _fifo(tmp_path / "composer.json")
    _fifo(tmp_path / "global.json")

    assert _scip_php_incompatibility_reason(tmp_path) is None
    assert _global_json_satisfiable(tmp_path, str(tmp_path)) is None


@pytest.mark.parametrize(
    ("project_type", "files"),
    [
        ("rust", ["Cargo.toml", "Cargo.lock", "src/lib.rs"]),
        ("typescript", ["package.json", "tsconfig.json", "src/a.ts"]),
        ("python", ["pyproject.toml", "pkg/a.py"]),
        ("go", ["a.go"]),
    ],
)
def test_hashers_leave_fifos(tmp_path, project_type, files):
    for rel in files:
        _fifo(tmp_path / rel)

    ScipCacheManager(tmp_path)._hash_project(tmp_path.name, project_type)


def test_fifo_checksums_and_index_are_left(tmp_path):
    mgr = ScipCacheManager(tmp_path)
    _fifo(mgr.checksums_file)

    assert mgr._load_checksums() == {}
    ScipIndex([_fifo(tmp_path / "x.scip")])


def test_indexer_child_reads_nothing_from_stdin():
    """Under MCP this process's stdin is the protocol stream."""

    async def child_stdin():
        argv = [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"]
        async with handlers._process_tree(argv, stdout=asyncio.subprocess.PIPE) as proc:
            out, _ = await proc.communicate()
        return out.decode().strip()

    read_end, write_end = os.pipe()
    os.write(write_end, b"protocol")
    os.close(write_end)
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        seen = asyncio.run(child_stdin())
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_end)

    assert seen == "''"
