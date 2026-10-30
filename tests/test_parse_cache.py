"""Reusing each file's parse: the codec, reuse, invalidation, pruning, and
entries a repository could have shipped."""

import json
import os

import pytest

from descry._parse_cache import ParseCache, _decode, _encode, _key_path
from descry.generate import CodeGraphBuilder
from descry.handlers import DescryConfig


def _graph(root, cache_dir, *settings, key="key"):
    cache = ParseCache(cache_dir, *settings, key_path=cache_dir.parent / key)
    builder = CodeGraphBuilder(root, parse_cache=cache)
    builder.process_directory()
    builder.resolve_references()
    return builder


def _project(tmp_path, files):
    root = tmp_path / "p"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


FILES = {
    "a.py": "from b import B\n\ndef run():\n    B().go()\n",
    "b.py": "class B:\n    def go(self):\n        pass\n",
    "c.ts": "class C { m(): void {} }\nfunction f(c: C) { c.m(); }\n",
}


def test_values_round_trip_as_written():
    value = {
        "plain": [1, "x", None, True, 2.5],
        "t": (1, ("nested",)),
        "s": {"a", ("b", 1)},
        "f": frozenset({1}),
        ("key", 1): {"d": [1]},
        3: "int key",
    }
    held = _decode(json.loads(json.dumps(_encode(value))))
    assert held == value
    assert type(held["t"]) is tuple and type(held["f"]) is frozenset
    assert type(held[("key", 1)]["d"]) is list
    with pytest.raises(TypeError):
        _encode(object())


def test_a_second_index_reuses_every_parse_and_gives_the_same_graph(
    tmp_path, monkeypatch
):
    root = _project(tmp_path, FILES)
    first = _graph(root, tmp_path / "c")
    monkeypatch.setattr(
        CodeGraphBuilder, "_parse", lambda *_: pytest.fail("parsed again")
    )
    second = _graph(root, tmp_path / "c")
    assert second.nodes == first.nodes and second.edges == first.edges


def test_a_changed_file_is_parsed_again_and_its_old_entry_removed(tmp_path):
    root = _project(tmp_path, FILES)
    _graph(root, tmp_path / "c")
    before = set((tmp_path / "c").glob("*-*.json"))
    (root / "b.py").write_text("class B:\n    def go(self):\n        return 1\n")
    (root / "c.ts").unlink()
    cached = _graph(root, tmp_path / "c")
    fresh = _graph(root, tmp_path / "fresh")
    assert cached.nodes == fresh.nodes and cached.edges == fresh.edges
    after = set((tmp_path / "c").glob("*-*.json"))
    assert len(after) == 2 and len(before - after) == 2


@pytest.mark.parametrize(
    "shipped",
    [
        "[1, 2, 3]",
        '{"edges": {}}',
        '{"nodes": [{"type": "Function"}]}',
        json.dumps(
            {
                "serial_base": 0,
                "nodes": [
                    {"id": "FILE:evil.py::x", "type": "Function", "metadata": {}}
                ],
                "edges": [],
            }
        ),
    ],
)
def test_an_entry_descry_did_not_write_is_parsed_over(tmp_path, shipped):
    """A repository can ship its cache: an entry not signed with the user's
    key, or moved to another file's name, is read as none."""
    root = _project(tmp_path, FILES)
    first = _graph(root, tmp_path / "c")
    entries = sorted((tmp_path / "c").glob("*-*.json"))
    signed = entries[0].read_text()
    for entry in entries:
        entry.write_text(shipped)
    again = _graph(root, tmp_path / "c")
    assert again.nodes == first.nodes and again.edges == first.edges
    for entry in entries[1:]:
        entry.write_text(signed)
    assert _graph(root, tmp_path / "c").edges == first.edges
    foreign = _graph(root, tmp_path / "c", key="other")
    assert foreign.nodes == first.nodes and foreign.edges == first.edges


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs")
def test_a_shipped_entry_that_never_ends_is_parsed_over(tmp_path):
    """A FIFO, or a symlink to a device, where an entry or the resolution
    is kept would hold the index on a read that never ends."""
    root = _project(tmp_path, FILES)
    first = _graph(root, tmp_path / "c")
    kept = [
        *sorted((tmp_path / "c").glob("*-*.json")),
        tmp_path / "c" / "resolution.json",
    ]
    kept[0].unlink()
    os.mkfifo(kept[0])
    kept[-1].unlink()
    kept[-1].symlink_to("/dev/zero")
    again = _graph(root, tmp_path / "c")
    assert again.nodes == first.nodes and again.edges == first.edges


def test_a_shipped_cache_file_is_refused_not_crashed(tmp_path, monkeypatch, capsys):
    """A `.descry_cache` shipped as a file is refused with an error, not a
    FileExistsError."""
    import sys

    from descry.generate import main

    root = _project(tmp_path, FILES)
    cache = tmp_path / "cache"
    cache.write_text("shipped as a file")
    monkeypatch.setenv("DESCRY_CACHE_DIR", str(cache))
    monkeypatch.setattr(sys, "argv", ["descry-generate", str(root)])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "is a file, not a directory" in capsys.readouterr().err


def test_a_shipped_graph_directory_is_refused_not_crashed(
    tmp_path, monkeypatch, capsys
):
    """A `codebase_graph.json` shipped as a directory is refused with an
    error, not an IsADirectoryError."""
    import sys

    from descry.generate import main

    root = _project(tmp_path, FILES)
    cache = tmp_path / "cache"
    (cache / "codebase_graph.json").mkdir(parents=True)
    monkeypatch.setenv("DESCRY_CACHE_DIR", str(cache))
    monkeypatch.setattr(sys, "argv", ["descry-generate", str(root)])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "is a directory, not the graph file" in capsys.readouterr().err


def test_a_shipped_entry_directory_does_not_stop_an_index(tmp_path):
    """A shipped entry that is a directory is ignored, not unlinked."""
    root = _project(tmp_path, FILES)
    _graph(root, tmp_path / "c")
    entries = sorted((tmp_path / "c").glob("*-*.json"))
    (root / "a.py").write_text("def changed():\n    pass\n")
    entries[0].unlink()
    (tmp_path / "c" / entries[0].name).mkdir()
    again = _graph(root, tmp_path / "c")
    assert any(n["metadata"]["name"] == "changed" for n in again.nodes)


def test_a_setting_that_changes_a_parse_keys_its_entry(tmp_path, monkeypatch):
    root = _project(tmp_path, FILES)
    _graph(root, tmp_path / "c", 120)
    parsed = []
    original = CodeGraphBuilder._parse
    monkeypatch.setattr(
        CodeGraphBuilder,
        "_parse",
        lambda self, *a: parsed.append(a[2]) or original(self, *a),
    )
    _graph(root, tmp_path / "c", 0.5)
    assert sorted(parsed) == sorted(FILES)


def test_a_parse_json_cannot_write_is_not_kept(tmp_path, monkeypatch):
    root = _project(tmp_path, {"b.py": FILES["b.py"]})
    original = CodeGraphBuilder._state

    def unwritable(self):
        return {**original(self), "imports": {object(): 1}}

    monkeypatch.setattr(CodeGraphBuilder, "_state", unwritable)
    _graph(root, tmp_path / "c")
    assert not list((tmp_path / "c").glob("*-*.json"))


def test_a_shipped_parse_directory_outside_the_project_is_refused(tmp_path):
    root = _project(tmp_path, FILES)
    (root / ".git").mkdir()
    (root / ".descry_cache").mkdir()
    (root / ".descry_cache" / "parse").symlink_to(tmp_path)
    config = DescryConfig.from_env(root)
    assert "outside the project" in config.cache_dir_error()


def test_the_key_is_never_kept_where_the_project_could_ship_it(tmp_path, monkeypatch):
    """A relative state directory is ignored, as XDG has it, and a key that
    would be inside the project leaves the cache unused."""
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    assert _key_path().is_absolute()
    root = _project(tmp_path, FILES)
    inside = ParseCache(tmp_path / "c", key_path=root / "k", project=root)
    assert inside.key is None and not (root / "k").exists()
    builder = CodeGraphBuilder(root, parse_cache=inside)
    builder.process_directory()
    assert not (tmp_path / "c").exists()
