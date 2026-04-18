"""Tests for the scip-dart adapter."""

from __future__ import annotations

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.dart import DartAdapter
from descry.scip.parser import ScipIndex


class TestDartAdapterDiscovery:
    def test_pubspec_at_root(self, tmp_path):
        (tmp_path / "pubspec.yaml").write_text("name: app\n")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "main.dart").write_text("void main() {}\n")

        projects = DartAdapter().discover(tmp_path, set())
        assert len(projects) == 1
        assert projects[0].language == "dart"
        assert projects[0].root == tmp_path

    def test_melos_workspace(self, tmp_path):
        packages = tmp_path / "packages"
        packages.mkdir()
        for name in ("alpha", "beta"):
            pkg = packages / name
            pkg.mkdir()
            (pkg / "pubspec.yaml").write_text(f"name: {name}\n")
            (pkg / "main.dart").write_text("void main() {}\n")

        projects = DartAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["alpha", "beta"]

    def test_subdir_packages(self, tmp_path):
        for name in ("alpha", "beta"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "pubspec.yaml").write_text(f"name: {name}\n")
            (pkg / "main.dart").write_text("void main() {}\n")

        projects = DartAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["alpha", "beta"]

    def test_excluded_dirs_respected(self, tmp_path):
        pkg = tmp_path / "example"
        pkg.mkdir()
        (pkg / "pubspec.yaml").write_text("name: example\n")
        (pkg / "main.dart").write_text("void main() {}\n")

        projects = DartAdapter().discover(tmp_path, {"example"})
        assert projects == []


class TestDartAdapterBuildCommand:
    def test_positional_dot_arg(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path, language="dart")
        spec = DartAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert spec.argv[0] == "scip-dart"
        assert "./" in spec.argv
        assert spec.cwd == tmp_path

    def test_output_mode_rename(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path, language="dart")
        spec = DartAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert spec.output_mode == "rename"

    def test_extra_args_appended(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path, language="dart")
        config = AdapterConfig(extra_args=("--foo", "bar"))
        spec = DartAdapter().build_command(project, tmp_path / "app.scip", config)
        assert "--foo" in spec.argv and "bar" in spec.argv


class TestScipDartHealthStatus:
    def test_dart_listed_in_scip_status(self):
        from descry.scip import support

        support.reset_scip_state()
        status = support.get_scip_status()
        assert "scip-dart" in status["indexers"]


class TestScipDartResolution:
    def test_dart_stats_bucket_exists(self):
        idx = ScipIndex([])
        assert "dart" in idx._resolution_stats

    def test_scheme_routes_scip_dart_symbols(self):
        from descry.scip.adapter import adapter_for_scheme

        adapter = adapter_for_scheme("scip-dart")
        assert adapter is not None
        assert adapter.name == "dart"
