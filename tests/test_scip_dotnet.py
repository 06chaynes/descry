"""Tests for the scip-dotnet adapter."""

from __future__ import annotations

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.dotnet import DotnetAdapter


class TestDotnetAdapterDiscovery:
    def test_solution_at_root(self, tmp_path):
        (tmp_path / "App.sln").write_text("Microsoft Visual Studio Solution File")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "App.cs").write_text("class App {}")

        projects = DotnetAdapter().discover(tmp_path, set())
        assert len(projects) == 1
        assert projects[0].name == tmp_path.name

    def test_csproj_at_root(self, tmp_path):
        (tmp_path / "App.csproj").write_text("<Project/>")
        (tmp_path / "App.cs").write_text("class App {}")

        projects = DotnetAdapter().discover(tmp_path, set())
        assert len(projects) == 1

    def test_multi_project_no_solution(self, tmp_path):
        for name in ("svc-a", "svc-b"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / f"{name}.csproj").write_text("<Project/>")
            (pkg / "Main.cs").write_text("class M {}")

        projects = DotnetAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["svc-a", "svc-b"]

    def test_no_cs_sources_skipped(self, tmp_path):
        (tmp_path / "empty" / "nested").mkdir(parents=True)
        (tmp_path / "empty" / "App.csproj").write_text("<Project/>")
        # No .cs or .vb files.

        assert DotnetAdapter().discover(tmp_path, set()) == []


class TestDotnetAdapterBuildCommand:
    def test_basic_command(self, tmp_path):
        project = DiscoveredProject(name="app", root=tmp_path)
        spec = DotnetAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert spec.argv[0] == "scip-dotnet"
        assert spec.argv[1] == "index"
        assert "--output" in spec.argv
        assert str(tmp_path / "app.scip") in spec.argv
        assert spec.cwd == tmp_path
        assert spec.output_mode == "direct"
        # Roll-forward env so scip-dotnet (net9.0) runs on newer SDKs.
        assert spec.env_extras.get("DOTNET_ROLL_FORWARD") == "LatestMajor"

    def test_names_the_solution(self, tmp_path):
        """scip-dotnet does not find a `.slnx` itself."""
        (tmp_path / "App.slnx").write_text("<Solution/>")
        project = DiscoveredProject(name="app", root=tmp_path)
        spec = DotnetAdapter().build_command(
            project, tmp_path / "app.scip", AdapterConfig()
        )
        assert spec.argv[-1] == "App.slnx"

    def test_indexes_every_solution(self, tmp_path):
        """Every `.slnx` is named, and where a `.sln` or project file is
        there scip-dotnet is left to find each of them."""
        project = DiscoveredProject(name="app", root=tmp_path)
        out = tmp_path / "app.scip"
        for name in ("B.slnx", "A.slnx"):
            (tmp_path / name).write_text("<Solution/>")
        spec = DotnetAdapter().build_command(project, out, AdapterConfig())
        assert spec.argv[-3:] == [str(out), "A.slnx", "B.slnx"]
        for name in ("A.sln", "B.sln"):
            (tmp_path / name).write_text("")
        spec = DotnetAdapter().build_command(project, out, AdapterConfig())
        assert spec.argv[-1] == str(out)


class TestScipDotnetHealthStatus:
    def test_dotnet_listed_in_scip_status(self):
        from descry.scip import support

        status = support.get_scip_status()
        assert "scip-dotnet" in status["indexers"]
