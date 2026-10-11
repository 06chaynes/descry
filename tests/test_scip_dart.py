"""Tests for the scip-dart adapter."""

from __future__ import annotations

from pathlib import Path

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

    def test_pkgs_workspace(self, tmp_path):
        """shelf's layout keeps packages under `pkgs/` — a collection
        directory any name, one level above each package."""
        for name in ("shelf", "shelf_web_socket"):
            pkg = tmp_path / "pkgs" / name
            pkg.mkdir(parents=True)
            (pkg / "pubspec.yaml").write_text(f"name: {name}\n")
            (pkg / "lib").mkdir()
            (pkg / "lib" / "main.dart").write_text("void main() {}\n")

        projects = DartAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["shelf", "shelf_web_socket"]

    def test_binary_alternative_probes_scip_dart(self, tmp_path, monkeypatch):
        """`dart pub global activate scip_dart` installs `scip_dart`;
        the adapter probes both spellings."""
        import shutil

        fake = tmp_path / "scip_dart"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setattr(
            shutil, "which", lambda name: str(fake) if name == "scip_dart" else None
        )
        from descry.scip import adapter as adapter_mod

        adapter_mod.reset_registry_state()
        assert adapter_mod.resolved_binary(DartAdapter()) == str(fake)
        adapter_mod.reset_registry_state()

    def test_collection_directory_is_excluded(self, tmp_path):
        """The exclusion applies to the collection too: `build/` can hold a
        stale copy of a package."""
        pkg = tmp_path / "build" / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "pubspec.yaml").write_text("name: pkg\n")
        (pkg / "main.dart").write_text("void main() {}\n")

        assert DartAdapter().discover(tmp_path, {"build"}) == []

    def test_dot_collection_is_skipped(self, tmp_path):
        pkg = tmp_path / ".hidden" / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "pubspec.yaml").write_text("name: pkg\n")
        (pkg / "main.dart").write_text("void main() {}\n")

        assert DartAdapter().discover(tmp_path, set()) == []

    def test_pub_cache_is_honoured(self, tmp_path, monkeypatch):
        """`dart pub global` installs into `$PUB_CACHE/bin` when that
        variable is set, and the adapter probes it there."""
        monkeypatch.setenv("PUB_CACHE", str(tmp_path))

        assert str(tmp_path / "bin" / "scip_dart") in DartAdapter().binary_alternatives

    def test_pub_cache_default_is_the_home_directory(self, monkeypatch):
        monkeypatch.delenv("PUB_CACHE", raising=False)

        assert (
            str(Path.home() / ".pub-cache" / "bin" / "scip_dart")
            in DartAdapter().binary_alternatives
        )

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
