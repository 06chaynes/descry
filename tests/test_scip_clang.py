"""Tests for the scip-clang adapter."""

from __future__ import annotations

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.clang import ClangAdapter
from descry.scip.parser import ScipIndex


class TestClangAdapterDiscovery:
    def test_cmake_at_root(self, tmp_path):
        (tmp_path / "CMakeLists.txt").write_text("project(test)")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.c").write_text("int main(){return 0;}")

        projects = ClangAdapter().discover(tmp_path, set())
        assert len(projects) == 1
        assert projects[0].language == "clang"

    def test_makefile_at_root(self, tmp_path):
        (tmp_path / "Makefile").write_text("all:")
        (tmp_path / "app.c").write_text("int main(){return 0;}")

        projects = ClangAdapter().discover(tmp_path, set())
        assert len(projects) == 1

    def test_monorepo(self, tmp_path):
        for name in ("alpha", "beta"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "CMakeLists.txt").write_text(f"project({name})")
            (pkg / "main.cpp").write_text("int main(){return 0;}")

        projects = ClangAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["alpha", "beta"]


class TestClangAdapterBuildCommand:
    def test_compdb_auto_detected_at_root(self, tmp_path):
        (tmp_path / "compile_commands.json").write_text("[]")
        project = DiscoveredProject(name="app", root=tmp_path, language="clang")
        spec = ClangAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert any(
            arg.startswith("--compdb-path=") and "compile_commands.json" in arg
            for arg in spec.argv
        )
        assert spec.output_mode == "direct"

    def test_compdb_in_build_dir(self, tmp_path):
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "compile_commands.json").write_text("[]")
        project = DiscoveredProject(name="app", root=tmp_path, language="clang")
        spec = ClangAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        compdb_arg = next(a for a in spec.argv if a.startswith("--compdb-path="))
        assert "build/compile_commands.json" in compdb_arg

    def test_compdb_override(self, tmp_path):
        (tmp_path / "custom").mkdir()
        (tmp_path / "custom" / "cc.json").write_text("[]")
        project = DiscoveredProject(name="app", root=tmp_path, language="clang")
        config = AdapterConfig(options={"compdb_path": "custom/cc.json"})
        spec = ClangAdapter().build_command(project, tmp_path / "app.scip", config)
        compdb_arg = next(a for a in spec.argv if a.startswith("--compdb-path="))
        assert "custom/cc.json" in compdb_arg

    def test_index_output_path_flag(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path, language="clang")
        spec = ClangAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert any(arg.startswith("--index-output-path=") for arg in spec.argv)


class TestScipClangHealthStatus:
    def test_clang_listed_in_scip_status(self):
        from descry.scip import support

        support.reset_scip_state()
        status = support.get_scip_status()
        assert "scip-clang" in status["indexers"]


class TestScipClangResolution:
    def test_clang_stats_bucket_exists(self):
        idx = ScipIndex([])
        assert "clang" in idx._resolution_stats

    def test_scheme_routes_cxx_symbols(self):
        """scip-clang emits scheme `cxx`, not `scip-clang`. Verify the
        adapter claims that scheme so ScipIndex dispatches through it."""
        from descry.scip.adapter import adapter_for_scheme

        adapter = adapter_for_scheme("cxx")
        assert adapter is not None
        assert adapter.name == "clang"

    def test_extract_name_for_cxx_function(self):
        idx = ScipIndex([])
        sym = "cxx . . $ listCreate(a153265b2bd52385)."
        # The scip-clang signature hash gets absorbed into the method
        # suffix; _extract_name returns the function name.
        assert idx._extract_name(sym) == "listCreate"
