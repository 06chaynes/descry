"""Tests for ScipCacheManager index generation, naming and checksum state."""

import errno
import json
import os
import threading
import time
from pathlib import Path

import pytest

from descry.scip import adapter as adapter_mod
from descry.scip.adapter import ADAPTERS, CommandSpec, DiscoveredProject
from descry.scip.adapters.rust import RustAdapter
from descry.scip.cache import ScipCacheManager


class _StubAdapter:
    """Adapter whose indexer is `script`, run by /bin/sh in the project dir.

    ``{out}`` in the script is replaced with the requested output path.
    """

    scheme = "scip-stub"
    binary = "sh"
    extensions = (".stub",)

    def __init__(self, script, name="stublang", mode="direct"):
        self.script = script
        self.name = name
        self.mode = mode

    def discover(self, root, _excluded_dirs):
        return [DiscoveredProject(name="proj", root=root, language=self.name)]

    def build_command(self, project, out_path, _config):
        return CommandSpec(
            argv=["/bin/sh", "-c", self.script.format(out=out_path)],
            cwd=project.root,
            output_mode=self.mode,
        )

    def parse_descriptors(self, _raw):
        return []


@pytest.fixture
def stub_adapter(monkeypatch):
    """Factory for stub adapters registered for the duration of the test."""

    def _make(script, **kwargs):
        adapter = _StubAdapter(script, **kwargs)
        monkeypatch.setitem(ADAPTERS, adapter.name, adapter)
        return adapter

    return _make


def _project(root, adapter):
    return DiscoveredProject(name="proj", root=root, language=adapter.name)


def _seed_old_index(path):
    """Create an index file that looks like an earlier run wrote it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old index")
    os.utime(path, (1, 1))


class TestStaleIndex:
    def test_direct_mode_indexer_that_writes_nothing_fails(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        adapter = _StubAdapter("exit 1")
        _seed_old_index(mgr._index_path(adapter.name, "proj"))

        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is False

    def test_failed_run_leaves_project_needing_update(self, tmp_path, stub_adapter):
        mgr = ScipCacheManager(tmp_path)
        adapter = stub_adapter("exit 1")
        _seed_old_index(mgr._index_path(adapter.name, "proj"))

        mgr._update_changed_for_adapter(adapter, parallel=False)

        assert mgr.needs_update("proj", adapter.name) is True

    def test_direct_mode_rewrite_of_existing_index_succeeds(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        adapter = _StubAdapter("printf 'new index, longer' > {out}")
        index = mgr._index_path(adapter.name, "proj")
        _seed_old_index(index)

        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is True
        assert index.read_bytes() == b"new index, longer"

    def test_rename_mode_ignores_index_scip_it_did_not_write(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        adapter = _StubAdapter("exit 1", mode="rename")
        _seed_old_index(tmp_path / "index.scip")

        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is False
        assert (tmp_path / "index.scip").exists()

    def test_rename_mode_moves_index_across_filesystems(self, tmp_path, monkeypatch):
        def _cross_device(_src, _dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "rename", _cross_device)
        monkeypatch.setattr(os, "replace", _cross_device)
        mgr = ScipCacheManager(tmp_path)
        adapter = _StubAdapter("printf 'payload' > index.scip", mode="rename")

        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is True
        assert mgr._index_path(adapter.name, "proj").read_bytes() == b"payload"
        assert not (tmp_path / "index.scip").exists()


class TestProjectNameCollisions:
    """Two adapters each discover a project called ``proj`` in the same root."""

    def test_same_name_in_two_languages_keeps_separate_indexes(
        self, tmp_path, stub_adapter
    ):
        stub_adapter("printf 'from a' > {out}", name="lang_a")
        stub_adapter("printf 'from b' > {out}", name="lang_b")
        mgr = ScipCacheManager(tmp_path)

        paths = mgr.update_all()

        assert paths["lang_a:proj"].read_bytes() == b"from a"
        assert paths["lang_b:proj"].read_bytes() == b"from b"
        assert mgr.needs_update("proj", "lang_a") is False
        assert mgr.needs_update("proj", "lang_b") is False

    def test_indexes_are_counted_by_a_flat_scip_glob(self, tmp_path, stub_adapter):
        stub_adapter("printf 'from a' > {out}", name="lang_a")
        stub_adapter("printf 'from b' > {out}", name="lang_b")
        mgr = ScipCacheManager(tmp_path)

        mgr.update_all()

        assert len(list(mgr.cache_dir.glob("*.scip"))) == 2

    def test_name_with_slash_stays_directly_in_cache_dir(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        assert mgr._index_path("rust", "crates/util").parent == mgr.cache_dir

    def test_update_removes_only_indexes_without_an_adapter_prefix(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        mgr.cache_dir.mkdir(parents=True)
        (mgr.cache_dir / "app.scip").write_bytes(b"no prefix")
        (mgr.cache_dir / "rust--app.scip").write_bytes(b"adapter prefix")

        mgr.update_all()

        assert [p.name for p in mgr.cache_dir.glob("*.scip")] == ["rust--app.scip"]


class TestRustSingleCrateAtRoot:
    def _crate_arg(self, root):
        (project,) = RustAdapter().discover(root, set())
        config = ScipCacheManager(root)._adapter_config_for(RustAdapter())
        argv = RustAdapter().build_command(project, root / "out.scip", config).argv
        return Path(argv[argv.index("scip") + 1])

    def test_build_command_targets_the_root(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "mylib"\n')
        (tmp_path / "src").mkdir()

        assert self._crate_arg(tmp_path) == tmp_path

    def test_project_dirs_give_each_rust_index_its_crate_directory(self, tmp_path):
        """A member crate's documents are relative to its directory, a root
        crate's to the root; other adapters' indexes are left alone."""
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "mylib"\n')
        (tmp_path / "src").mkdir()
        member = tmp_path / "crates" / "cli"
        (member / "src").mkdir(parents=True)
        (member / "Cargo.toml").write_text('[package]\nname = "my-cli"\n')
        mgr = ScipCacheManager(tmp_path)
        paths = {
            f"rust:{tmp_path.name}": Path("root.scip"),
            "rust:crates/cli": Path("cli.scip"),
            "typescript:web": Path("web.scip"),
        }

        assert mgr.project_dirs(paths) == {
            Path("root.scip"): "",
            Path("cli.scip"): "crates/cli",
        }

    def test_root_package_beside_its_members_is_a_crate(self, tmp_path):
        """`rust-analyzer scip` indexes only the crate at the path it is
        given, so the root package is a project of its own."""
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "app"\n[workspace]\nmembers = ["crates/core"]\n'
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "crates" / "core" / "src").mkdir(parents=True)
        (tmp_path / "crates" / "core" / "Cargo.toml").write_text(
            '[package]\nname = "core"\n'
        )

        names = [p.name for p in RustAdapter().discover(tmp_path, set())]

        assert names == ["crates/core", tmp_path.name]

    def test_virtual_manifest_is_no_crate(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/core"]\n')
        (tmp_path / "src").mkdir()
        (tmp_path / "crates" / "core" / "src").mkdir(parents=True)
        (tmp_path / "crates" / "core" / "Cargo.toml").write_text(
            '[package]\nname = "core"\n'
        )

        names = [p.name for p in RustAdapter().discover(tmp_path, set())]

        assert names == ["crates/core"]

    def test_member_crate_named_like_the_root_targets_the_member(self, tmp_path):
        root = tmp_path / "mylib"
        (root / "mylib" / "src").mkdir(parents=True)
        (root / "Cargo.toml").write_text('[workspace]\nmembers = ["mylib"]\n')
        (root / "mylib" / "Cargo.toml").write_text('[package]\nname = "mylib"\n')

        assert self._crate_arg(root) == root / "mylib"

    def test_hash_tracks_source_edits(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "mylib"\n')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "lib.rs").write_text("pub fn f() {}\n")
        mgr = ScipCacheManager(tmp_path)

        first = mgr._hash_rust_crate(tmp_path.name)
        (tmp_path / "src" / "lib.rs").write_text("pub fn g() {}\n")

        assert mgr._hash_rust_crate(tmp_path.name) != first

    def test_typescript_root_package_hash_tracks_source_edits(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.ts").write_text("export const x = 1;\n")
        mgr = ScipCacheManager(tmp_path)

        first = mgr._hash_typescript_package(tmp_path.name)
        (tmp_path / "src" / "index.ts").write_text("export const x = 2;\n")

        assert mgr._hash_typescript_package(tmp_path.name) != first


class TestCacheDirParameter:
    def test_default_is_inside_the_project(self, tmp_path):
        assert ScipCacheManager(tmp_path).cache_dir == tmp_path / ".descry_cache/scip"

    def test_configured_cache_dir_receives_indexes_and_checksums(
        self, tmp_path, stub_adapter
    ):
        repo = tmp_path / "repo"
        repo.mkdir()
        mgr = ScipCacheManager(repo, cache_dir=tmp_path / "elsewhere")

        mgr._update_changed_for_adapter(
            stub_adapter("printf 'payload' > {out}"), parallel=False
        )

        assert sorted(p.name for p in (tmp_path / "elsewhere" / "scip").iterdir()) == [
            "checksums.json",
            "stublang--proj.scip",
        ]
        assert not (repo / ".descry_cache").exists()

    def test_relative_cache_dir_is_resolved_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mgr = ScipCacheManager(tmp_path / "repo", cache_dir=Path("relative-cache"))
        assert mgr.cache_dir == tmp_path / "relative-cache" / "scip"


class TestChecksumsFile:
    def test_non_dict_file_reads_as_empty(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        mgr.cache_dir.mkdir(parents=True)
        mgr.checksums_file.write_text("[]")

        assert mgr._load_checksums() == {}

    def test_failed_save_keeps_previous_contents(self, tmp_path, monkeypatch):
        mgr = ScipCacheManager(tmp_path)
        mgr._save_checksums({"rust:app": "abc"})

        def _fail(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", _fail)
        with pytest.raises(OSError, match="disk full"):
            mgr._save_checksums({"rust:app": "def"})

        assert json.loads(mgr.checksums_file.read_text()) == {"rust:app": "abc"}
        assert [p.name for p in mgr.cache_dir.iterdir()] == ["checksums.json"]

    def test_concurrent_adapters_keep_each_others_entries(
        self, tmp_path, monkeypatch, stub_adapter
    ):
        mgr = ScipCacheManager(tmp_path)
        real_load = mgr._load_checksums

        def _slow_load():
            loaded = real_load()
            # Hold the read-modify-write window open so the threads overlap.
            time.sleep(0.05)
            return loaded

        monkeypatch.setattr(mgr, "_load_checksums", _slow_load)
        adapters = [
            stub_adapter("printf 'payload' > {out}", name=f"lang{i}") for i in range(4)
        ]
        threads = [
            threading.Thread(target=mgr._update_changed_for_adapter, args=(a, False))
            for a in adapters
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert set(real_load()) == {f"lang{i}:proj" for i in range(4)}


class TestTimeout:
    @pytest.mark.parametrize("bad", ["x", "5", [5]])
    def test_non_numeric_timeout_is_ignored(self, tmp_path, monkeypatch, bad):
        monkeypatch.delenv("DESCRY_SCIP_TIMEOUT", raising=False)
        mgr = ScipCacheManager(tmp_path, scip_timeout_minutes=bad)
        adapter = _StubAdapter("printf 'payload' > {out}")

        assert mgr._get_timeout() is None
        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is True

    def test_numeric_timeout_is_minutes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DESCRY_SCIP_TIMEOUT", raising=False)
        assert ScipCacheManager(tmp_path, scip_timeout_minutes=10)._get_timeout() == 600
        assert ScipCacheManager(tmp_path, scip_timeout_minutes=0)._get_timeout() is None

    @pytest.mark.parametrize(("value", "seconds"), [("5", 300), ("none", None)])
    def test_env_var_overrides_the_configured_timeout(
        self, tmp_path, monkeypatch, value, seconds
    ):
        monkeypatch.setenv("DESCRY_SCIP_TIMEOUT", value)
        mgr = ScipCacheManager(tmp_path, scip_timeout_minutes=10)
        assert mgr._get_timeout() == seconds


class TestCommand:
    def test_empty_argv_is_refused(self, tmp_path):
        class _EmptyCommand(_StubAdapter):
            def build_command(self, project, _out_path, _config):
                return CommandSpec(argv=[], cwd=project.root)

        adapter = _EmptyCommand("")
        mgr = ScipCacheManager(tmp_path)

        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is False

    def test_probed_binary_runs_in_place_of_the_named_one(self, tmp_path, monkeypatch):
        """An indexer installed under another name runs by the path the
        probe found, where argv[0] names the adapter's binary."""
        installed = tmp_path / "bin" / "scip_alt"
        installed.parent.mkdir()
        installed.write_text('#!/bin/sh\n[ "$1" = --version ] || printf ran > "$1"\n')
        installed.chmod(0o755)

        class _Installed(_StubAdapter):
            binary = "scip-stub-not-installed"
            binary_alternatives = (str(installed),)

            def build_command(self, project, out_path, _config):
                return CommandSpec(argv=[self.binary, str(out_path)], cwd=project.root)

        monkeypatch.setattr(adapter_mod, "_binary_cache", {})
        adapter = _Installed("")
        mgr = ScipCacheManager(tmp_path)

        assert mgr._generate_scip(adapter, _project(tmp_path, adapter)) is True
        assert mgr._index_path(adapter.name, "proj").read_text() == "ran"
