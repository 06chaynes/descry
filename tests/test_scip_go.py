"""Tests for the scip-go adapter."""

from __future__ import annotations

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.go import GoAdapter
from descry.scip.parser import ScipIndex


class TestGoAdapterDiscovery:
    def test_single_module_at_root(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/thing\n\ngo 1.21\n")
        (tmp_path / "main.go").write_text("package main\nfunc main(){}")

        projects = GoAdapter().discover(tmp_path, set())

        assert len(projects) == 1
        assert projects[0].name == tmp_path.name

    def test_monorepo_modules(self, tmp_path):
        for name in ("svc-a", "svc-b"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "go.mod").write_text(f"module example.com/{name}\n\ngo 1.21\n")
            (pkg / "main.go").write_text("package main\nfunc main(){}")

        projects = GoAdapter().discover(tmp_path, set())

        names = sorted(p.name for p in projects)
        assert names == ["svc-a", "svc-b"]

    def test_vendor_ignored(self, tmp_path):
        pkg = tmp_path / "svc"
        pkg.mkdir()
        (pkg / "go.mod").write_text("module example.com/svc\n\ngo 1.21\n")
        # Only a vendored file — should NOT count as "has go sources".
        vendor = pkg / "vendor" / "foo"
        vendor.mkdir(parents=True)
        (vendor / "bar.go").write_text("package foo")

        projects = GoAdapter().discover(tmp_path, set())

        assert projects == []

    def test_excluded_dir_ignored(self, tmp_path):
        pkg = tmp_path / "build"
        pkg.mkdir()
        (pkg / "go.mod").write_text("module example.com/build\n\ngo 1.21\n")
        (pkg / "x.go").write_text("package build")

        projects = GoAdapter().discover(tmp_path, {"build"})

        assert projects == []


class TestGoAdapterBuildCommand:
    def test_basic_command(self, tmp_path):
        project = DiscoveredProject(name="svc", root=tmp_path)
        spec = GoAdapter().build_command(
            project, tmp_path / "svc.scip", AdapterConfig()
        )

        assert spec.argv[0] == "scip-go"
        assert "--output" in spec.argv
        assert str(tmp_path / "svc.scip") in spec.argv
        assert spec.cwd == tmp_path
        assert spec.output_mode == "direct"


class TestGoAdapterDescriptorParsing:
    def test_method_on_type(self):
        result = GoAdapter().parse_descriptors("pkg/http/Server#Handle().")
        assert result == ["Server", "Handle"]

    def test_nested_package_path(self):
        result = GoAdapter().parse_descriptors("pkg/kubelet/Kubelet#Run().")
        assert result == ["Kubelet", "Run"]

    def test_field_or_var(self):
        result = GoAdapter().parse_descriptors("pkg/types/Color#Red.")
        assert result == ["Color", "Red"]


class TestScipGoResolution:
    def test_extract_name_for_method(self):
        idx = ScipIndex([])
        sym = "scip-go gomod github.com/example/svc v1.0.0 pkg/http/Server#Handle()."
        assert idx._extract_name(sym) == "Handle"


class TestScipGoHealthStatus:
    def test_go_listed_in_scip_status(self):
        from descry.scip import support

        status = support.get_scip_status()
        assert "scip-go" in status["indexers"]
