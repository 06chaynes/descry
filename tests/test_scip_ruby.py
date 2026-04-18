"""Tests for the scip-ruby adapter."""

from __future__ import annotations

from descry.scip.adapter import AdapterConfig, DiscoveredProject
from descry.scip.adapters.ruby import RubyAdapter
from descry.scip.parser import ScipIndex


class TestRubyAdapterDiscovery:
    def test_single_gemfile_at_root(self, tmp_path):
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "app.rb").write_text("class App; end")

        projects = RubyAdapter().discover(tmp_path, set())

        assert len(projects) == 1
        assert projects[0].name == tmp_path.name
        assert projects[0].language == "ruby"

    def test_monorepo(self, tmp_path):
        for name in ("svc-a", "svc-b"):
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "Gemfile").write_text("source 'https://rubygems.org'")
            (pkg / "app.rb").write_text("class App; end")

        projects = RubyAdapter().discover(tmp_path, set())
        assert sorted(p.name for p in projects) == ["svc-a", "svc-b"]

    def test_no_ruby_sources_skipped(self, tmp_path):
        pkg = tmp_path / "empty"
        pkg.mkdir()
        (pkg / "Gemfile").write_text("source 'https://rubygems.org'")
        # No .rb file.

        assert RubyAdapter().discover(tmp_path, set()) == []


class TestRubyAdapterBuildCommand:
    def test_basic_command(self, tmp_path):
        project = DiscoveredProject(name="svc", root=tmp_path, language="ruby")
        spec = RubyAdapter().build_command(
            project, tmp_path / "svc.scip", AdapterConfig()
        )
        assert spec.argv == ["scip-ruby", "."]
        assert spec.cwd == tmp_path
        assert spec.output_mode == "rename"


class TestRubyAdapterDescriptorParsing:
    def test_backtick_descriptor(self):
        """scip-ruby descriptors should parse through the shared backtick
        helper (verified at smoke-test time; adjust if format differs)."""
        result = RubyAdapter().parse_descriptors("`app/models/user.rb`/User#save().")
        # Backtick helper strips the wrapped path and extracts symbols.
        assert "User" in result
        assert "save" in result


class TestScipRubyHealthStatus:
    def test_ruby_listed_in_scip_status(self):
        from descry.scip import support

        support.reset_scip_state()
        status = support.get_scip_status()
        assert "scip-ruby" in status["indexers"]


class TestScipRubyResolution:
    def test_ruby_stats_bucket_exists(self):
        idx = ScipIndex([])
        assert "ruby" in idx._resolution_stats
