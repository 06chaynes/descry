"""Tests for descry.scip — SCIP parsing, import resolution, and cache performance."""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from descry.generate import TypeScriptSymbolTable
from descry.scip.adapters.typescript import parse_backtick_descriptors
from descry.scip.cache import ADAPTERS, ScipCacheManager
from descry.scip.parser import ScipIndex


class TestTypescriptScipParsing:
    """Tests for the shared backtick-descriptor parser used by scip-typescript
    and scip-python. Routes through ``parse_backtick_descriptors`` — the
    canonical helper in ``descry.scip.adapters.typescript`` that both
    adapters delegate to.
    """

    def test_simple_function(self):
        """Should extract simple function name from TypeScript SCIP symbol."""
        descriptors = "src/lib/api/`client.ts`/getAuthToken()."
        result = parse_backtick_descriptors(descriptors)
        assert result == ["getAuthToken"], f"Expected ['getAuthToken'], got {result}"

    def test_class_method(self):
        """Should extract class and method from TypeScript SCIP symbol."""
        descriptors = "src/lib/stores/`users.ts`/UsersStore#fetchUsers()."
        result = parse_backtick_descriptors(descriptors)
        assert result == ["UsersStore", "fetchUsers"], (
            f"Expected ['UsersStore', 'fetchUsers'], got {result}"
        )

    def test_multiple_backticks(self):
        """Should handle multiple backtick segments in path."""
        descriptors = "src/lib/`stores`/`auth.ts`/AuthStore#login()."
        result = parse_backtick_descriptors(descriptors)
        assert result == ["AuthStore", "login"], (
            f"Expected ['AuthStore', 'login'], got {result}"
        )

    def test_nested_class_method(self):
        """Should handle nested type/method symbols."""
        descriptors = "src/`client.ts`/ApiClient#request()."
        result = parse_backtick_descriptors(descriptors)
        assert "ApiClient" in result
        assert "request" in result

    def test_export_function(self):
        """Should handle exported function symbols."""
        descriptors = "src/lib/`utils.ts`/formatDate()."
        result = parse_backtick_descriptors(descriptors)
        assert result == ["formatDate"], f"Expected ['formatDate'], got {result}"


class TestTypescriptImportResolution:
    """Tests for TypeScript import resolution."""

    def test_type_import_skipped(self):
        """Should identify type-only imports."""
        table = TypeScriptSymbolTable("/project/web/src/test.ts", "/project")
        table.imports = {
            "MyType": ("./types", "type"),
            "myFunc": ("./utils", "named"),
        }

        assert table.is_type_import("MyType") is True
        assert table.is_type_import("myFunc") is False

    def test_unknown_import(self):
        """Should return None for unknown imports."""
        table = TypeScriptSymbolTable("/project/web/src/test.ts", "/project")
        table.imports = {}

        assert table.is_type_import("unknownFunc") is False


class TestScipCachePerformance:
    """Tests for SCIP cache performance optimizations."""

    def test_get_max_workers_env_override(self):
        """Should respect DESCRY_SCIP_WORKERS env variable."""
        # Save original
        original = os.environ.get("DESCRY_SCIP_WORKERS")

        try:
            os.environ["DESCRY_SCIP_WORKERS"] = "1"
            manager = ScipCacheManager(Path("."))
            assert manager._get_max_workers(10) == 1

            os.environ["DESCRY_SCIP_WORKERS"] = "5"
            assert manager._get_max_workers(10) == 5
            # Should cap at num_items
            assert manager._get_max_workers(3) == 3
        finally:
            if original is None:
                os.environ.pop("DESCRY_SCIP_WORKERS", None)
            else:
                os.environ["DESCRY_SCIP_WORKERS"] = original

    def test_get_max_workers_defaults_to_reasonable_value(self):
        """Should default to reasonable worker count without env override."""
        # Ensure env var is not set
        original = os.environ.pop("DESCRY_SCIP_WORKERS", None)
        try:
            manager = ScipCacheManager(Path("."))
            workers = manager._get_max_workers(10)
            # Should return between 2 and 4 (reasonable defaults)
            assert 2 <= workers <= 4
        finally:
            if original is not None:
                os.environ["DESCRY_SCIP_WORKERS"] = original

    def test_get_prime_threads_env_override(self):
        """Should respect DESCRY_PRIME_THREADS env variable."""
        original = os.environ.get("DESCRY_PRIME_THREADS")
        try:
            os.environ["DESCRY_PRIME_THREADS"] = "8"
            manager = ScipCacheManager(Path("."))
            assert manager._get_prime_threads() == 8
        finally:
            if original is None:
                os.environ.pop("DESCRY_PRIME_THREADS", None)
            else:
                os.environ["DESCRY_PRIME_THREADS"] = original

    def test_get_prime_threads_defaults_based_on_cpu(self):
        """Should default based on CPU count."""
        original = os.environ.pop("DESCRY_PRIME_THREADS", None)
        try:
            manager = ScipCacheManager(Path("."))
            threads = manager._get_prime_threads()
            expected = max(2, multiprocessing.cpu_count() - 2)
            assert threads == expected
        finally:
            if original is not None:
                os.environ["DESCRY_PRIME_THREADS"] = original


def _discovered(manager, lang):
    """Names of the projects the ``lang`` adapter finds under the manager's
    root."""
    found = ADAPTERS[lang].discover(manager.project_root, manager.excluded_dirs)
    return [project.name for project in found]


class TestPythonSCIPDiscovery:
    """Discovery covers monorepo and single-package layouts."""

    def test_monorepo_layout_with_pyproject(self, tmp_path):
        for name in ("backend", "workers"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "pyproject.toml").write_text('[project]\nname = "' + name + '"\n')
            (pkg / "app.py").write_text("def main(): pass\n")
        # A frontend sibling with only JS should be excluded.
        (tmp_path / "frontend").mkdir()
        (tmp_path / "frontend" / "index.js").write_text("")

        manager = ScipCacheManager(tmp_path)
        assert _discovered(manager, "python") == ["backend", "workers"]

    def test_monorepo_respects_excluded_dirs(self, tmp_path):
        node_modules = tmp_path / "node_modules" / "some-pkg"
        node_modules.mkdir(parents=True)
        (node_modules / "pyproject.toml").write_text("[project]\nname='x'\n")
        (node_modules / "a.py").write_text("")

        manager = ScipCacheManager(tmp_path)
        assert _discovered(manager, "python") == []

    def test_single_package_layout_uses_root_name(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "standalone"\n')
        src = tmp_path / "src" / "standalone"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")
        (src / "app.py").write_text("def main(): pass\n")

        manager = ScipCacheManager(tmp_path)
        assert _discovered(manager, "python") == [tmp_path.name]

    def test_setup_py_recognized(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
        (tmp_path / "mod.py").write_text("x = 1\n")
        manager = ScipCacheManager(tmp_path)
        assert _discovered(manager, "python") == [tmp_path.name]

    def test_hash_python_changes_on_source_edit(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n')
        src = tmp_path / "a.py"
        src.write_text("x = 1\n")

        manager = ScipCacheManager(tmp_path)
        first = manager._hash_python_package(tmp_path.name)
        src.write_text("x = 2\n")
        second = manager._hash_python_package(tmp_path.name)
        assert first != second

    def test_hash_python_stable_without_edits(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n')
        (tmp_path / "a.py").write_text("x = 1\n")
        manager = ScipCacheManager(tmp_path)
        assert manager._hash_python_package(
            tmp_path.name
        ) == manager._hash_python_package(tmp_path.name)

    def test_needs_update_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n')
        (tmp_path / "a.py").write_text("x = 1\n")
        manager = ScipCacheManager(tmp_path)
        # No cached .scip → needs update.
        assert manager.needs_update(tmp_path.name, "python") is True


class TestPythonSCIPSymbolExtraction:
    """End-to-end name extraction for scip-python symbols.

    scip-python emits symbols in the same backtick-wrapped format as
    scip-typescript (``<scheme> <manager> <pkg> <ver> `module.path`/name().``),
    so both adapters delegate to ``parse_backtick_descriptors``.
    These tests pin that behaviour via ``ScipIndex._extract_name`` so a
    future refactor can't silently regress Python call resolution to 0%.
    """

    def test_extract_module_level_function(self):
        idx = ScipIndex([])
        sym = "scip-python python descry 0.1.0 `pi-extension.descry-cli`/main()."
        assert idx._extract_name(sym) == "main"

    def test_extract_class_method(self):
        idx = ScipIndex([])
        sym = "scip-python python descry 0.1.0 `descry.handlers`/DescryService#flow()."
        assert idx._extract_name(sym) == "flow"

    def test_extract_nested_qualified_module(self):
        idx = ScipIndex([])
        sym = "scip-python python descry 0.1.0 `src.descry.query`/GraphQuerier#find_call_path()."
        assert idx._extract_name(sym) == "find_call_path"

    def test_extract_init_method(self):
        idx = ScipIndex([])
        sym = "scip-python python descry 0.1.0 `descry.handlers`/DescryService#__init__()."
        assert idx._extract_name(sym) == "__init__"

    def test_local_symbol_returns_none(self):
        idx = ScipIndex([])
        assert idx._extract_name("local 42") is None

    def test_generic_parser_not_used_for_scip_python(self):
        """Regression: the generic parser tokenizes hyphens/dots in module
        paths into a jumble. Make sure we never accidentally route Python
        symbols through it.
        """
        idx = ScipIndex([])
        # If this regressed, _extract_name would return something like
        # 'cli' (last piece of the module path) instead of 'main'.
        sym = "scip-python python pkg 0.1.0 `pi-extension.descry-cli`/main()."
        assert idx._extract_name(sym) == "main"


class TestPythonSCIPAvailability:
    """support.py exposes scip-python status + per-adapter availability probes."""

    def test_scip_python_disabled_by_env(self, monkeypatch):
        from descry.scip import adapter as adapter_mod
        from descry.scip import support

        support.reset_scip_state()
        monkeypatch.setenv("DESCRY_NO_SCIP", "1")
        assert adapter_mod.indexer_available("python") is False

    def test_get_scip_status_reports_python_key(self, monkeypatch):
        from descry.scip import support

        support.reset_scip_state()
        monkeypatch.delenv("DESCRY_NO_SCIP", raising=False)
        status = support.get_scip_status()
        assert "scip-python" in status["indexers"]


class TestGenerateScipRenameMode:
    """_generate_scip handles CommandSpec(output_mode='rename') correctly.

    Exercises the scip-go contingency path: an indexer that writes to a
    fixed default filename (`index.scip`) in its working directory rather
    than accepting an --output flag. The shared runner should move the
    default output into the cache-dir location after success.
    """

    def _stub_adapter(self, mode):
        """Build a stub adapter that runs `true` and optionally pre-seeds
        the default-output file before the subprocess runs.
        """
        from descry.scip.adapter import CommandSpec, DiscoveredProject

        class _StubAdapter:
            name = "stublang"
            scheme = "scip-stub"
            binary = "true"
            extensions = (".stub",)

            def discover(self, root, _excluded_dirs):
                return [
                    DiscoveredProject(name="stubproj", root=root, language=self.name)
                ]

            def build_command(self, project, _out_path, _config):
                # Simulate the indexer by pre-creating index.scip in cwd
                # before the subprocess runs — a shell one-liner keeps the
                # test hermetic without needing a real SCIP binary.
                default_output = project.root / "index.scip"
                payload = b"stub-scip-payload"
                argv = [
                    "/bin/sh",
                    "-c",
                    f"printf '%s' {payload.decode()} > {default_output}",
                ]
                return CommandSpec(argv=argv, cwd=project.root, output_mode=mode)

            def parse_descriptors(self, _raw):
                return []

        return _StubAdapter()

    def test_rename_mode_moves_default_output_into_cache(self, tmp_path):
        from descry.scip.adapter import DiscoveredProject

        mgr = ScipCacheManager(tmp_path)
        mgr.cache_dir.mkdir(parents=True, exist_ok=True)
        adapter = self._stub_adapter(mode="rename")
        project = DiscoveredProject(name="stubproj", root=tmp_path, language="stublang")

        ok = mgr._generate_scip(adapter, project)

        assert ok is True
        expected = mgr._index_path("stublang", "stubproj")
        assert expected.exists()
        assert expected.read_bytes() == b"stub-scip-payload"
        assert not (tmp_path / "index.scip").exists()

    def test_direct_mode_leaves_no_stray_output(self, tmp_path):
        """Regression guard: output_mode='direct' must NOT move a stray
        index.scip sitting in cwd. Only 'rename' mode should do the move.
        """
        from descry.scip.adapter import CommandSpec, DiscoveredProject

        # Pre-seed a stray file so we can verify it's left in place.
        stray = tmp_path / "index.scip"
        stray.write_bytes(b"pre-existing stray file")

        class _DirectStubAdapter:
            name = "directlang"
            scheme = "scip-direct"
            binary = "true"
            extensions = (".direct",)

            def discover(self, _root, _excluded_dirs):
                return []

            def build_command(self, project, out_path, _config):
                argv = [
                    "/bin/sh",
                    "-c",
                    f"printf 'payload' > {out_path}",
                ]
                return CommandSpec(argv=argv, cwd=project.root, output_mode="direct")

            def parse_descriptors(self, _raw):
                return []

        mgr = ScipCacheManager(tmp_path)
        mgr.cache_dir.mkdir(parents=True, exist_ok=True)
        project = DiscoveredProject(
            name="directproj", root=tmp_path, language="directlang"
        )

        ok = mgr._generate_scip(_DirectStubAdapter(), project)

        assert ok is True
        assert mgr._index_path("directlang", "directproj").read_bytes() == b"payload"
        # Stray index.scip untouched by direct mode.
        assert stray.read_bytes() == b"pre-existing stray file"


def _write_scip(path, documents, project_root="", position_encoding=0):
    """Write a .scip file.

    ``documents`` maps a relative path to ``(zero_based_line, symbol,
    is_definition)`` occurrences, as an indexer would emit them; a fourth
    item is the occurrence's start column, 4 without one. ``project_root``
    is the URI the index states its document paths are relative to, and
    ``position_encoding`` the unit it counts columns in.
    """
    from descry.scip import pb2

    index = pb2.Index()
    index.metadata.project_root = project_root
    for relative_path, occurrences in documents.items():
        doc = index.documents.add()
        doc.relative_path = relative_path
        doc.position_encoding = position_encoding
        for line, symbol, is_definition, *column in occurrences:
            occ = doc.occurrences.add()
            start = column[0] if column else 4
            occ.range.extend([line, start, start + 3])
            occ.symbol = symbol
            occ.symbol_roles = 1 if is_definition else 0
            if is_definition:
                doc.symbols.add().symbol = symbol
    path.write_bytes(index.SerializeToString())
    return path


class TestScipIndexResolveLineSemantics:
    """resolve() takes 1-based descry linenos against 0-based SCIP ranges.

    Two types define ``run``. Name-only resolution returns the first one
    indexed (``Alpha``), so a result of ``Beta#run`` proves the exact
    (file, line) lookup matched.
    """

    ALPHA_RUN = "rust-analyzer cargo app 0.1.0 alpha/Alpha#run()."
    BETA_RUN = "rust-analyzer cargo app 0.1.0 beta/Beta#run()."
    BETA_NODE = "FILE:app/src/beta.rs::Beta::run"

    @pytest.fixture
    def index(self, tmp_path):
        documents = {
            "src/alpha.rs": [(0, self.ALPHA_RUN, True)],
            "src/beta.rs": [(0, self.BETA_RUN, True)],
            # Source line 5 references Beta#run, line 6 Alpha#run.
            "src/main.rs": [(4, self.BETA_RUN, False), (5, self.ALPHA_RUN, False)],
        }
        return ScipIndex([_write_scip(tmp_path / "app.scip", documents)])

    def test_reference_resolves_on_its_one_based_line(self, index):
        assert index.resolve("run", "app/src/main.rs", 5) == self.BETA_NODE

    def test_qualified_reference_matches_on_exact_line(self, index):
        assert index.resolve("Self::run", "app/src/main.rs", 5) == self.BETA_NODE
        assert index.resolve("worker.run", "app/src/main.rs", 5) == self.BETA_NODE

    def test_line_without_occurrence_falls_back_to_the_one_definition(
        self, index, tmp_path
    ):
        assert index.resolve("run", "app/src/main.rs", 40) is None
        documents = {"src/beta.rs": [(0, self.BETA_RUN, True)]}
        single = ScipIndex([_write_scip(tmp_path / "one.scip", documents)])
        assert single.resolve("run", "app/src/main.rs", 40) == self.BETA_NODE

    def test_qualified_name_resolves_to_the_one_type_of_that_name(
        self, index, tmp_path
    ):
        assert index.resolve("Alpha::run", "app/src/main.rs", 40) == (
            "FILE:app/src/alpha.rs::Alpha::run"
        )
        documents = {
            "src/one.rs": [
                (0, "rust-analyzer cargo app 0.1.0 one/Shared#run().", True)
            ],
            "src/two.rs": [
                (0, "rust-analyzer cargo app 0.1.0 two/Shared#run().", True)
            ],
        }
        twice = ScipIndex([_write_scip(tmp_path / "two.scip", documents)])
        assert twice.resolve("Shared::run", "app/src/main.rs", 40) is None

    @pytest.mark.parametrize(
        "other", ["shared/Util#run().", "state/SharedState#run()."]
    )
    def test_qualifier_is_a_type_not_a_part_of_a_path_or_a_longer_name(
        self, tmp_path, other
    ):
        documents = {
            "src/other.rs": [(0, f"rust-analyzer cargo app 0.1.0 {other}", True)],
            "src/single.rs": [
                (0, "rust-analyzer cargo app 0.1.0 core/Single#run().", True)
            ],
        }
        index = ScipIndex([_write_scip(tmp_path / "app.scip", documents)])
        assert index.resolve("Shared::run", "app/src/main.rs", 40) is None
        assert index.resolve("Single::run", "app/src/main.rs", 40) == (
            "FILE:app/src/single.rs::Single::run"
        )

    @pytest.mark.parametrize(
        ("called", "outside"),
        [
            # A method of a concrete type and a function of another crate.
            ("rust-analyzer cargo std 1.0 fs/impl#[DirEntry]path().", True),
            ("rust-analyzer cargo other 1.0 util/path().", True),
            # A method of a trait the project implements, and of one it does not.
            ("rust-analyzer cargo core 1.0 fmt/Display#path().", False),
            ("rust-analyzer cargo other 1.0 dsl/QueryDsl#path().", True),
            # No call, and a project symbol.
            ("rust-analyzer cargo std 1.0 fs/DirEntry#path.", False),
            ("rust-analyzer cargo app 0.1.0 alpha/impl#[Alpha]path().", False),
        ],
    )
    def test_call_of_an_outside_symbol_that_cannot_reach_the_project(
        self, tmp_path, called, outside
    ):
        own = "rust-analyzer cargo app 0.1.0 alpha/impl#[Alpha]path()."
        shown = "rust-analyzer cargo app 0.1.0 alpha/impl#[Alpha][Display]fmt()."
        documents = {
            "src/alpha.rs": [(0, own, True), (1, shown, True)],
            "src/main.rs": [(4, called, False)],
        }
        index = ScipIndex([_write_scip(tmp_path / "app.scip", documents)])
        assert index.outside_at("entry.path", "app/src/main.rs", 5) is outside
        assert index.outside_at("entry.path", "app/src/main.rs", 6) is False

    def test_definition_at_does_not_fall_back_to_name(self, index):
        assert index.definition_at("run", "app/src/main.rs", 5) == (
            self.BETA_NODE,
            "src/beta.rs",
            1,
        )
        assert index.definition_at("run", "app/src/main.rs", 40) is None

    def test_project_dirs_make_document_paths_root_relative(self, tmp_path):
        """An index of a crate at `crates/app` whose package is named
        differently is keyed, and yields node IDs, by repository paths."""
        scip_file = _write_scip(
            tmp_path / "app.scip",
            {
                "src/beta.rs": [(0, self.BETA_RUN, True)],
                "src/main.rs": [(4, self.BETA_RUN, False)],
            },
        )
        index = ScipIndex([scip_file], {scip_file: "crates/app-dir"})
        assert index.definition_at("run", "crates/app-dir/src/main.rs", 5) == (
            "FILE:crates/app-dir/src/beta.rs::Beta::run",
            "crates/app-dir/src/beta.rs",
            1,
        )

    def test_column_is_read_in_the_unit_the_document_states(self, tmp_path):
        """After non-ASCII text a column differs by unit. A document that
        counts UTF-8 code units is asked with that offset; one that does not
        say is matched by any of them."""
        documents = {
            "src/alpha.rs": [(0, self.ALPHA_RUN, True)],
            "src/beta.rs": [(0, self.BETA_RUN, True)],
            "src/main.rs": [
                (4, self.ALPHA_RUN, False, 9),
                (4, self.BETA_RUN, False, 18),
            ],
        }
        for encoding in (1, 0):
            scip_file = _write_scip(
                tmp_path / f"app{encoding}.scip", documents, position_encoding=encoding
            )
            index = ScipIndex([scip_file])
            # (UTF-8, UTF-16, UTF-32) offsets of a name after one `é`.
            found = index.definition_at("a.run", "app/src/main.rs", 5, (9, 8, 8))
            assert found[0] == "FILE:app/src/alpha.rs::Alpha::run"
            found = index.definition_at("b.run", "app/src/main.rs", 5, (18, 17, 17))
            assert found[0] == self.BETA_NODE
        # In the UTF-8 document the UTF-16 offset names nothing.
        utf8 = ScipIndex([tmp_path / "app1.scip"])
        assert utf8.definition_at("a.run", "app/src/main.rs", 5, (10, 9, 9)) is None

    def test_stated_project_root_makes_document_paths_root_relative(self, tmp_path):
        """An index that names `<root>/packages/app` as its project root is
        keyed, and yields node IDs, by repository paths."""
        documents = {
            "src/beta.rs": [(0, self.BETA_RUN, True)],
            "src/main.rs": [(4, self.BETA_RUN, False)],
        }
        scip_file = _write_scip(
            tmp_path / "app.scip",
            documents,
            (tmp_path / "packages" / "app").as_uri(),
        )
        index = ScipIndex([scip_file], root=tmp_path)
        assert index.definition_at("run", "packages/app/src/main.rs", 5) == (
            "FILE:packages/app/src/beta.rs::Beta::run",
            "packages/app/src/beta.rs",
            1,
        )

    def test_project_root_outside_the_repository_is_not_used(self, tmp_path):
        scip_file = _write_scip(
            tmp_path / "app.scip",
            {"src/beta.rs": [(0, self.BETA_RUN, True)]},
            (tmp_path / "elsewhere" / "app").as_uri(),
        )
        index = ScipIndex([scip_file], root=tmp_path / "repo")
        assert index.definitions[self.BETA_RUN] == ("src/beta.rs", 1)

    def test_column_tells_same_named_references_on_a_line_apart(self, tmp_path):
        """`a.run(); b.run();` on one line: each call gets the symbol whose
        occurrence starts at its own column."""
        documents = {
            "src/alpha.rs": [(0, self.ALPHA_RUN, True)],
            "src/beta.rs": [(0, self.BETA_RUN, True)],
            "src/main.rs": [
                (4, self.ALPHA_RUN, False, 6),
                (4, self.BETA_RUN, False, 15),
            ],
        }
        index = ScipIndex([_write_scip(tmp_path / "app.scip", documents)])
        first = index.definition_at("a.run", "app/src/main.rs", 5, 6)
        second = index.definition_at("b.run", "app/src/main.rs", 5, 15)
        assert (first[0], second[0]) == (
            "FILE:app/src/alpha.rs::Alpha::run",
            self.BETA_NODE,
        )
        assert index.definition_at("c.run", "app/src/main.rs", 5, 30) is None


def test_protobuf_runtime_mismatch_loads_nothing(tmp_path, monkeypatch):
    """A VersionError from importing the pb2 bindings is not raised to the caller."""
    import sys

    from google.protobuf import runtime_version

    import descry.scip

    scip_file = _write_scip(tmp_path / "app.scip", {"src/a.rs": []})

    def _mismatch(*_args, **_kwargs):
        raise runtime_version.VersionError("gencode is newer than runtime")

    # Drop the cached module so the import in _load_scip re-runs the check.
    monkeypatch.delitem(sys.modules, "descry.scip.pb2")
    monkeypatch.delattr(descry.scip, "pb2")
    monkeypatch.setattr(runtime_version, "ValidateProtobufRuntimeVersion", _mismatch)

    assert ScipIndex([scip_file]).definitions == {}


def test_typescript_workspace_entries_outside_root_are_dropped(tmp_path):
    from descry.scip.adapters.typescript import TypeScriptAdapter

    for pkg in (tmp_path / "outside", tmp_path / "repo" / "packages" / "a"):
        (pkg / "src").mkdir(parents=True)
        (pkg / "package.json").write_text("{}")
        (pkg / "src" / "index.ts").write_text("export const x = 1;\n")
    root = tmp_path / "repo"
    (root / "package.json").write_text(
        json.dumps({"workspaces": ["packages/*", "../outside", "../*"]})
    )

    projects = TypeScriptAdapter().discover(root, set())

    assert [p.name for p in projects] == ["packages/a"]


def test_typescript_absolute_glob_entry_is_skipped(tmp_path):
    from descry.scip.adapters.typescript import TypeScriptAdapter

    (tmp_path / "package.json").write_text(
        json.dumps({"workspaces": [f"{tmp_path}/*"]})
    )

    assert TypeScriptAdapter().discover(tmp_path, set()) == []


@pytest.mark.parametrize("member", ["../*", "{parent}/*"])
def test_rust_glob_members_outside_root_are_dropped(tmp_path, member):
    from descry.scip.adapters.rust import RustAdapter

    for crate in (tmp_path / "outside", tmp_path / "repo" / "inner"):
        (crate / "src").mkdir(parents=True)
        (crate / "Cargo.toml").write_text("[package]\n")
    root = tmp_path / "repo"
    members = ["inner", member.format(parent=tmp_path)]
    (root / "Cargo.toml").write_text(f"[workspace]\nmembers = {json.dumps(members)}\n")

    assert [p.name for p in RustAdapter().discover(root, set())] == ["inner"]


def test_rust_manifest_that_is_not_utf8_falls_back_to_crate_globbing(tmp_path):
    from descry.scip.adapters.rust import RustAdapter

    (tmp_path / "inner" / "src").mkdir(parents=True)
    (tmp_path / "inner" / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "Cargo.toml").write_bytes(b'[workspace]\nmembers = ["\xff"]\n')

    assert [p.name for p in RustAdapter().discover(tmp_path, set())] == ["inner"]


class TestNoScipEnvParsing:
    """DESCRY_NO_SCIP is read the same way as in DescryConfig.from_env()."""

    @pytest.fixture(autouse=True)
    def _every_indexer_installed(self, monkeypatch):
        from descry.scip import adapter as adapter_mod

        monkeypatch.setattr(
            adapter_mod, "_probe_binary", lambda _adapter: {"available": True}
        )

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_opt_out_values_disable_scip(self, monkeypatch, value):
        from descry.scip import support

        monkeypatch.setenv("DESCRY_NO_SCIP", value)

        assert support.scip_available() is False
        assert support.get_scip_status()["disabled_by_env"] is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_other_values_leave_scip_enabled(self, monkeypatch, value):
        from descry.handlers import DescryConfig
        from descry.scip import support

        monkeypatch.setenv("DESCRY_NO_SCIP", value)

        assert support.scip_available() is True
        assert support.get_scip_status()["disabled_by_env"] is False
        assert DescryConfig.from_env().enable_scip is True


@pytest.mark.parametrize(
    "lang",
    ["typescript", "python", "java", "go", "ruby", "php", "dotnet", "clang", "dart"],
)
def test_non_rust_adapters_ignore_adapter_config(tmp_path, lang):
    from descry.scip.adapter import ADAPTERS, AdapterConfig, DiscoveredProject

    adapter = ADAPTERS[lang]
    project = DiscoveredProject(name="app", root=tmp_path, language=lang)
    out = tmp_path / "app.scip"

    plain = adapter.build_command(project, out, AdapterConfig())
    loaded = adapter.build_command(
        project, out, AdapterConfig(toolchain="17", extra_args=("--output=/x",))
    )

    assert loaded.argv == plain.argv
    assert loaded.env_extras == plain.env_extras
