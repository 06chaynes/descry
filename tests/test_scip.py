"""Tests for descry.scip — SCIP parsing, import resolution, and cache performance."""

import multiprocessing
import os
from pathlib import Path


from descry.scip.parser import ScipIndex
from descry.scip.cache import ScipCacheManager
from descry.generate import TypeScriptSymbolTable


class TestTypescriptScipParsing:
    """Tests for TypeScript SCIP symbol parsing."""

    def test_simple_function(self):
        """Should extract simple function name from TypeScript SCIP symbol."""
        # Create a minimal index to test parsing
        index = ScipIndex([])

        # Test the internal descriptor parsing
        descriptors = "src/lib/api/`client.ts`/getAuthToken()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["getAuthToken"], f"Expected ['getAuthToken'], got {result}"

    def test_class_method(self):
        """Should extract class and method from TypeScript SCIP symbol."""
        index = ScipIndex([])

        descriptors = "src/lib/stores/`users.ts`/UsersStore#fetchUsers()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["UsersStore", "fetchUsers"], (
            f"Expected ['UsersStore', 'fetchUsers'], got {result}"
        )

    def test_multiple_backticks(self):
        """Should handle multiple backtick segments in path."""
        index = ScipIndex([])

        descriptors = "src/lib/`stores`/`auth.ts`/AuthStore#login()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["AuthStore", "login"], (
            f"Expected ['AuthStore', 'login'], got {result}"
        )

    def test_nested_class_method(self):
        """Should handle nested type/method symbols."""
        index = ScipIndex([])

        descriptors = "src/`client.ts`/ApiClient#request()."
        result = index._parse_typescript_descriptors(descriptors)
        assert "ApiClient" in result
        assert "request" in result

    def test_export_function(self):
        """Should handle exported function symbols."""
        index = ScipIndex([])

        descriptors = "src/lib/`utils.ts`/formatDate()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["formatDate"], f"Expected ['formatDate'], got {result}"


class TestTypescriptImportResolution:
    """Tests for TypeScript import resolution."""

    def test_symbol_table_basic(self):
        """Should track basic imports."""
        table = TypeScriptSymbolTable(
            "/project/lens/src/lib/stores/test.ts", "/project"
        )
        table.imports = {
            "fetchData": ("./api", "named"),
            "apiClient": ("$lib/api/client", "default"),
        }

        # Should be able to get import source
        source = table.get_import_source("fetchData")
        assert source == "./api"

        source = table.get_import_source("apiClient")
        assert source == "$lib/api/client"

    def test_namespace_import(self):
        """Should identify namespace imports."""
        table = TypeScriptSymbolTable(
            "/project/lens/src/lib/stores/test.ts", "/project"
        )
        table.namespaces = {"schedulesApi": "$lib/api/schedules"}
        table.imports = {"schedulesApi": ("$lib/api/schedules", "namespace")}

        # schedulesApi.list() should be identified as a namespace call
        assert table.is_namespace_call("schedulesApi.list") is True
        assert table.is_namespace_call("regularFunc") is False

        # Should get the import source
        source = table.get_import_source("schedulesApi")
        assert source == "$lib/api/schedules"

    def test_type_import_skipped(self):
        """Should identify type-only imports."""
        table = TypeScriptSymbolTable("/project/lens/src/test.ts", "/project")
        table.imports = {
            "MyType": ("./types", "type"),
            "myFunc": ("./utils", "named"),
        }

        assert table.is_type_import("MyType") is True
        assert table.is_type_import("myFunc") is False

    def test_unknown_import(self):
        """Should return None for unknown imports."""
        table = TypeScriptSymbolTable("/project/lens/src/test.ts", "/project")
        table.imports = {}

        assert table.get_import_source("unknownFunc") is None
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


class TestPythonSCIPDiscovery:
    """get_python_packages covers monorepo and single-package layouts."""

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
        assert manager.get_python_packages() == ["backend", "workers"]

    def test_monorepo_respects_excluded_dirs(self, tmp_path):
        node_modules = tmp_path / "node_modules" / "some-pkg"
        node_modules.mkdir(parents=True)
        (node_modules / "pyproject.toml").write_text("[project]\nname='x'\n")
        (node_modules / "a.py").write_text("")

        manager = ScipCacheManager(tmp_path)
        assert manager.get_python_packages() == []

    def test_single_package_layout_uses_root_name(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "standalone"\n')
        src = tmp_path / "src" / "standalone"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")
        (src / "app.py").write_text("def main(): pass\n")

        manager = ScipCacheManager(tmp_path)
        assert manager.get_python_packages() == [tmp_path.name]

    def test_setup_py_recognized(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
        (tmp_path / "mod.py").write_text("x = 1\n")
        manager = ScipCacheManager(tmp_path)
        assert manager.get_python_packages() == [tmp_path.name]

    def test_get_projects_includes_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n')
        (tmp_path / "m.py").write_text("")
        manager = ScipCacheManager(tmp_path)
        types = {t for _, t in manager.get_projects()}
        assert "python" in types

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
    so the parser routes both through ``_parse_typescript_descriptors``.
    These tests pin that behaviour so a future refactor can't silently
    regress Python call resolution to 0%.
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
    """support.py exposes scip_python_available and includes it in status."""

    def test_scip_python_disabled_by_env(self, monkeypatch):
        from descry.scip import support

        support.reset_scip_state()
        monkeypatch.setenv("DESCRY_NO_SCIP", "1")
        assert support.scip_python_available() is False

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

    def _stub_adapter(self, tmp_path, mode):
        """Build a stub adapter that runs `true` and optionally pre-seeds
        the default-output file before the subprocess runs.
        """
        from descry.scip.adapter import CommandSpec, DiscoveredProject

        class _StubAdapter:
            name = "stublang"
            scheme = "scip-stub"
            binary = "true"
            extensions = (".stub",)

            def discover(self, root, excluded_dirs):
                return [
                    DiscoveredProject(name="stubproj", root=root, language=self.name)
                ]

            def build_command(self, project, out_path, config):
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

            def parse_descriptors(self, raw):
                return []

        return _StubAdapter()

    def test_rename_mode_moves_default_output_into_cache(self, tmp_path):
        from descry.scip.adapter import DiscoveredProject

        mgr = ScipCacheManager(tmp_path)
        mgr.cache_dir.mkdir(parents=True, exist_ok=True)
        adapter = self._stub_adapter(tmp_path, mode="rename")
        project = DiscoveredProject(name="stubproj", root=tmp_path, language="stublang")

        ok = mgr._generate_scip(adapter, project)

        assert ok is True
        expected = mgr.cache_dir / "stubproj.scip"
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

            def discover(self, root, excluded_dirs):
                return []

            def build_command(self, project, out_path, config):
                argv = [
                    "/bin/sh",
                    "-c",
                    f"printf 'payload' > {out_path}",
                ]
                return CommandSpec(argv=argv, cwd=project.root, output_mode="direct")

            def parse_descriptors(self, raw):
                return []

        mgr = ScipCacheManager(tmp_path)
        mgr.cache_dir.mkdir(parents=True, exist_ok=True)
        project = DiscoveredProject(
            name="directproj", root=tmp_path, language="directlang"
        )

        ok = mgr._generate_scip(_DirectStubAdapter(), project)

        assert ok is True
        assert (mgr.cache_dir / "directproj.scip").read_bytes() == b"payload"
        # Stray index.scip untouched by direct mode.
        assert stray.read_bytes() == b"pre-existing stray file"
