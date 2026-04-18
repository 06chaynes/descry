"""Tests for the scip-php adapter."""

from __future__ import annotations

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.php import PhpAdapter
from descry.scip.parser import ScipIndex


class TestPhpAdapterDiscovery:
    def test_single_composer_at_root(self, tmp_path):
        (tmp_path / "composer.json").write_text("{}")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "App.php").write_text("<?php class App {}")

        projects = PhpAdapter().discover(tmp_path, set())
        assert len(projects) == 1
        assert projects[0].name == tmp_path.name
        assert projects[0].language == "php"

    def test_monorepo(self, tmp_path):
        for name in ("pkg-a", "pkg-b"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "composer.json").write_text("{}")
            (pkg / "main.php").write_text("<?php class Main {}")

        projects = PhpAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["pkg-a", "pkg-b"]

    def test_no_php_sources_skipped(self, tmp_path):
        pkg = tmp_path / "empty"
        pkg.mkdir()
        (pkg / "composer.json").write_text("{}")
        # No .php files.

        assert PhpAdapter().discover(tmp_path, set()) == []


class TestPhpAdapterBuildCommand:
    def test_basic_command(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path, language="php")
        spec = PhpAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert spec.argv == ["scip-php"]
        assert spec.cwd == tmp_path
        assert spec.output_mode == "rename"

    def test_extra_args_forwarded(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path, language="php")
        spec = PhpAdapter().build_command(
            project,
            tmp_path / "app.scip",
            AdapterConfig(extra_args=("--memory-limit=2G",)),
        )
        assert spec.argv == ["scip-php", "--memory-limit=2G"]


class TestScipPhpHealthStatus:
    def test_php_listed_in_scip_status(self):
        from descry.scip import support

        support.reset_scip_state()
        status = support.get_scip_status()
        assert "scip-php" in status["indexers"]


class TestScipPhpResolution:
    def test_php_stats_bucket_exists(self):
        idx = ScipIndex([])
        assert "php" in idx._resolution_stats
