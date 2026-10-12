"""The cache directory, and the project an indexer runs in, may come with the
repository. A symlink it ships where descry writes a file is never written
through: the file it points at is left as it was."""

import json
import os
import stat
import time

import pytest

from descry import _files
from descry._files import create_sibling
from descry._graph import CURRENT_SCHEMA
from descry.embeddings import _file_lock
from descry.generate import CodeGraphBuilder
from descry.scip.adapter import CommandSpec, DiscoveredProject
from descry.scip.cache import ScipCacheManager

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and FIFOs"),
    pytest.mark.usefixtures("no_wait"),
]


@pytest.fixture
def victim(tmp_path):
    path = tmp_path / "victim"
    path.write_text("precious\n")
    return path


def test_graph_temporary_is_not_written_through(tmp_path, victim):
    cache = tmp_path / "repo" / ".descry_cache"
    cache.mkdir(parents=True)
    (cache / "codebase_graph.json.tmp").symlink_to(victim)
    builder = CodeGraphBuilder(tmp_path / "repo")

    builder.export(cache / "codebase_graph.json")

    assert victim.read_text() == "precious\n"
    graph = json.loads((cache / "codebase_graph.json").read_text())
    assert graph["schema_version"] == CURRENT_SCHEMA


def test_sibling_is_new_and_has_the_mode_open_gives(tmp_path):
    target = tmp_path / "out.json"
    with open(tmp_path / "reference", "w"):
        pass

    fd, tmp = create_sibling(target)
    os.close(fd)

    assert tmp.parent == tmp_path and tmp.name.startswith(".out.json.")
    assert stat.S_IMODE(tmp.stat().st_mode) == stat.S_IMODE(
        (tmp_path / "reference").stat().st_mode
    )


def test_sibling_is_not_created_through_a_symlink_at_its_name(
    tmp_path, victim, monkeypatch
):
    monkeypatch.setattr(_files.secrets, "token_hex", lambda _n: "0" * 16)
    (tmp_path / ".out.json.0000000000000000.tmp").symlink_to(victim)

    with pytest.raises(FileExistsError):
        create_sibling(tmp_path / "out.json")

    assert victim.read_text() == "precious\n"


def test_orphaned_siblings_are_removed(tmp_path):
    target = tmp_path / "out.json"
    orphan = tmp_path / ".out.json.0123456789abcdef.tmp"
    live = tmp_path / ".out.json.fedcba9876543210.tmp"
    other = tmp_path / ".out.json.notatemporary.tmp"
    for path in (orphan, live, other):
        path.write_text("x")
    stale = time.time() - 2 * _files._ORPHANED_AFTER
    os.utime(orphan, (stale, stale))
    os.utime(other, (stale, stale))

    fd, tmp = create_sibling(target)
    os.close(fd)

    assert not orphan.exists()
    assert live.exists() and other.exists() and tmp.exists()


def test_lock_is_not_written_through_a_symlink(tmp_path, victim):
    lock = tmp_path / "embeddings.lock"
    lock.symlink_to(victim)

    with pytest.raises(OSError), _file_lock(lock, timeout=1):
        pass

    assert victim.read_text() == "precious\n"


def test_lock_refuses_a_fifo(tmp_path):
    lock = tmp_path / "embeddings.lock"
    os.mkfifo(lock)

    with pytest.raises(OSError), _file_lock(lock, timeout=1):
        pass


def test_lock_file_is_not_truncated(tmp_path):
    lock = tmp_path / "embeddings.lock"
    lock.write_text("kept\n")

    with _file_lock(lock, timeout=1):
        pass

    assert lock.read_text() == "kept\n"


class _Writer:
    """Adapter whose indexer writes `ran` to the output path, or to
    `index.scip` in its working directory in rename mode."""

    name = "writerlang"
    scheme = "scip-writer"
    binary = "sh"
    extensions = (".w",)

    def __init__(self, mode):
        self.mode = mode

    def discover(self, root, _excluded_dirs):
        return [DiscoveredProject(name="proj", root=root, language=self.name)]

    def build_command(self, project, out_path, _config):
        target = out_path if self.mode == "direct" else "index.scip"
        return CommandSpec(
            argv=["/bin/sh", "-c", f"printf ran > '{target}'"],
            cwd=project.root,
            output_mode=self.mode,
        )

    def parse_descriptors(self, _raw):
        return []


def test_scip_output_link_is_removed_not_written_through(tmp_path, victim):
    root = tmp_path / "repo"
    root.mkdir()
    mgr = ScipCacheManager(root)
    adapter = _Writer("direct")
    output = mgr._index_path(adapter.name, "proj")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(victim)
    project = DiscoveredProject(name="proj", root=root, language=adapter.name)

    assert mgr._generate_scip(adapter, project) is True

    assert victim.read_text() == "precious\n"
    assert not output.is_symlink() and output.read_text() == "ran"


def test_rename_mode_does_not_run_through_a_linked_index(tmp_path, victim):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "index.scip").symlink_to(victim)
    mgr = ScipCacheManager(root)
    adapter = _Writer("rename")
    project = DiscoveredProject(name="proj", root=root, language=adapter.name)

    assert mgr._generate_scip(adapter, project) is False

    assert victim.read_text() == "precious\n"
    assert (root / "index.scip").is_symlink()
