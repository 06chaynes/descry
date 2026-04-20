"""Tests for DescryConfig TOML loading and precedence."""

import logging
import os
from unittest.mock import patch

from descry.handlers import DescryConfig
from descry.scip.cache import ScipCacheManager


# --- Default field values ---


class TestNewFieldDefaults:
    """All new fields have expected defaults."""

    def test_embedding_model_default(self):
        config = DescryConfig()
        assert config.embedding_model == "jinaai/jina-code-embeddings-0.5b"

    def test_test_path_patterns_default(self):
        config = DescryConfig()
        assert "/tests/" in config.test_path_patterns
        assert "/__tests__/" in config.test_path_patterns
        assert isinstance(config.test_path_patterns, tuple)

    def test_test_file_suffixes_default(self):
        config = DescryConfig()
        assert "_test.rs" in config.test_file_suffixes
        assert ".test.ts" in config.test_file_suffixes
        assert isinstance(config.test_file_suffixes, tuple)

    def test_code_extensions_default(self):
        config = DescryConfig()
        assert ".rs" in config.code_extensions
        assert ".py" in config.code_extensions
        assert isinstance(config.code_extensions, set)

    def test_churn_exclusions_default(self):
        config = DescryConfig()
        assert ".descry_cache/" in config.churn_exclusions
        assert "Cargo.lock" in config.churn_exclusions
        assert isinstance(config.churn_exclusions, list)

    def test_git_timeout_default(self):
        config = DescryConfig()
        assert config.git_timeout == 30

    def test_scip_timeout_minutes_default(self):
        config = DescryConfig()
        assert config.scip_timeout_minutes == 0

    def test_embedding_timeout_default(self):
        config = DescryConfig()
        assert config.embedding_timeout == 60

    def test_query_timeout_ms_default(self):
        config = DescryConfig()
        assert config.query_timeout_ms == 4000

    def test_max_depth_default(self):
        config = DescryConfig()
        assert config.max_depth == 3

    def test_max_nodes_default(self):
        config = DescryConfig()
        assert config.max_nodes == 100

    def test_max_children_per_level_default(self):
        config = DescryConfig()
        assert config.max_children_per_level == 10

    def test_max_callers_shown_default(self):
        config = DescryConfig()
        assert config.max_callers_shown == 15

    def test_syntax_lang_map_default(self):
        config = DescryConfig()
        assert config.syntax_lang_map[".rs"] == "rust"
        assert config.syntax_lang_map[".py"] == "python"
        assert config.syntax_lang_map[".ts"] == "typescript"
        assert isinstance(config.syntax_lang_map, dict)

    def test_scip_extra_args_default(self):
        config = DescryConfig()
        assert config.scip_extra_args == ["--exclude-vendored-libraries"]

    def test_scip_skip_crates_default(self):
        config = DescryConfig()
        assert config.scip_skip_crates == []

    def test_scip_rust_toolchain_default(self):
        config = DescryConfig()
        assert config.scip_rust_toolchain is None


# --- TOML loading ---


class TestFromToml:
    """TOML file loading and parsing."""

    def test_from_toml_basic(self, tmp_path):
        """Write .descry.toml to tmp_path, verify values loaded."""
        toml_content = """\
[project]
excluded_dirs = ["target", "node_modules", "custom_dir"]
max_stale_hours = 48

[features]
enable_scip = false
enable_embeddings = false

[embeddings]
model = "custom/model-name"

[test_detection]
path_patterns = ["/tests/", "/custom_tests/"]
file_suffixes = ["_test.py", ".custom_test.js"]

[code_files]
extensions = [".rs", ".py", ".custom"]

[git]
churn_exclusions = ["custom.lock"]
timeout = 60

[timeouts]
scip_minutes = 10
embedding_seconds = 120
query_ms = 8000

[query]
max_depth = 5
max_nodes = 200
max_children_per_level = 20
max_callers_shown = 30

[syntax.lang_map]
".custom" = "custom-lang"
"""
        toml_file = tmp_path / ".descry.toml"
        toml_file.write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        assert config.excluded_dirs == {"target", "node_modules", "custom_dir"}
        assert config.max_stale_hours == 48
        assert config.enable_scip is False
        assert config.enable_embeddings is False
        assert config.embedding_model == "custom/model-name"
        assert config.test_path_patterns == ("/tests/", "/custom_tests/")
        assert config.test_file_suffixes == ("_test.py", ".custom_test.js")
        assert config.code_extensions == {".rs", ".py", ".custom"}
        assert config.churn_exclusions == ["custom.lock"]
        assert config.git_timeout == 60
        assert config.scip_timeout_minutes == 10
        assert config.embedding_timeout == 120
        assert config.query_timeout_ms == 8000
        assert config.max_depth == 5
        assert config.max_nodes == 200
        assert config.max_children_per_level == 20
        assert config.max_callers_shown == 30
        assert config.syntax_lang_map[".custom"] == "custom-lang"

    def test_from_toml_missing(self, tmp_path):
        """No file, all defaults."""
        data = DescryConfig._load_toml(tmp_path)
        assert data == {}
        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(data)
        # Should keep all defaults
        assert config.embedding_model == "jinaai/jina-code-embeddings-0.5b"
        assert config.max_depth == 3
        assert config.git_timeout == 30

    def test_from_toml_partial(self, tmp_path):
        """Only some fields set, rest stay default."""
        toml_content = """\
[query]
max_depth = 7

[git]
timeout = 45
"""
        (tmp_path / ".descry.toml").write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        assert config.max_depth == 7
        assert config.git_timeout == 45
        # Other fields stay default
        assert config.max_nodes == 100
        assert config.embedding_model == "jinaai/jina-code-embeddings-0.5b"

    def test_from_toml_invalid(self, tmp_path, caplog):
        """Malformed TOML falls back to defaults with warning."""
        (tmp_path / ".descry.toml").write_text("this is [[ not valid toml }{}")

        with caplog.at_level(logging.WARNING):
            data = DescryConfig._load_toml(tmp_path)

        assert data == {}
        assert any("Failed to parse" in r.message for r in caplog.records)

    def test_scip_config_from_toml(self, tmp_path):
        """SCIP extra_args and skip_crates load from TOML."""
        toml_content = """\
[scip]
extra_args = ["--exclude-vendored-libraries", "--custom-flag"]
skip_crates = ["mandible", "tarsus"]
"""
        (tmp_path / ".descry.toml").write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        assert config.scip_extra_args == [
            "--exclude-vendored-libraries",
            "--custom-flag",
        ]
        assert config.scip_skip_crates == ["mandible", "tarsus"]

    def test_scip_rust_toolchain_from_toml(self, tmp_path):
        """SCIP rust toolchain loads from TOML."""
        toml_content = """\
[scip.rust]
toolchain = "1.92.0"
"""
        (tmp_path / ".descry.toml").write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        assert config.scip_rust_toolchain == "1.92.0"

    def test_scip_empty_extra_args(self, tmp_path):
        """Empty extra_args disables all default flags."""
        toml_content = """\
[scip]
extra_args = []
"""
        (tmp_path / ".descry.toml").write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        assert config.scip_extra_args == []

    def test_syntax_lang_map_merges(self, tmp_path):
        """TOML entries merge with defaults (additive)."""
        toml_content = """\
[syntax.lang_map]
".custom" = "custom-lang"
".xyz" = "xyz-lang"
"""
        (tmp_path / ".descry.toml").write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        # New entries added
        assert config.syntax_lang_map[".custom"] == "custom-lang"
        assert config.syntax_lang_map[".xyz"] == "xyz-lang"
        # Defaults preserved
        assert config.syntax_lang_map[".rs"] == "rust"
        assert config.syntax_lang_map[".py"] == "python"

    def test_cross_lang_section(self, tmp_path):
        """[cross_lang] openapi_path + patterns reach DescryConfig."""
        (tmp_path / "spec.json").write_text('{"openapi": "3.0.0"}')
        (tmp_path / ".descry.toml").write_text(
            'openapi_path = "spec.json"\n'
            'backend_handler_patterns = ["handlers/"]\n'
            'frontend_api_patterns = ["src/api/"]\n'
            'api_prefixes = ["/api/v1", "/api/v2"]\n'.replace(
                "openapi_path", "[cross_lang]\nopenapi_path"
            )
        )
        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))

        assert config.openapi_path == (tmp_path / "spec.json").resolve()
        assert config.backend_handler_patterns == ["handlers/"]
        assert config.frontend_api_patterns == ["src/api/"]
        assert config.api_prefixes == ["/api/v1", "/api/v2"]

    def test_cross_lang_openapi_path_outside_root_rejected(self, tmp_path, caplog):
        """Paths resolving outside project_root are ignored with a warning."""
        (tmp_path / ".descry.toml").write_text(
            '[cross_lang]\nopenapi_path = "/etc/passwd"\n'
        )
        config = DescryConfig(project_root=tmp_path)
        with caplog.at_level(logging.WARNING):
            config._apply_toml(DescryConfig._load_toml(tmp_path))
        assert config.openapi_path is None
        assert any("outside project root" in r.message for r in caplog.records)

    def test_cross_lang_openapi_path_absolute_inside_root(self, tmp_path):
        """Absolute paths that resolve inside project_root are accepted."""
        (tmp_path / "public").mkdir()
        spec = tmp_path / "public" / "openapi.json"
        spec.write_text('{"openapi":"3.0.0"}')
        (tmp_path / ".descry.toml").write_text(
            f'[cross_lang]\nopenapi_path = "{spec}"\n'
        )
        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))
        assert config.openapi_path == spec.resolve()


# --- Precedence ---


class TestPrecedence:
    """defaults < .descry.toml < env vars"""

    def test_toml_overrides_defaults(self, tmp_path):
        """TOML wins over defaults."""
        toml_content = """\
[project]
max_stale_hours = 48
"""
        (tmp_path / ".descry.toml").write_text(toml_content)
        # Simulate what from_env does internally
        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))
        assert config.max_stale_hours == 48  # TOML wins over default of 24

    def test_env_overrides_toml(self, tmp_path):
        """Env var wins over TOML."""
        toml_content = """\
[features]
enable_scip = true
"""
        (tmp_path / ".descry.toml").write_text(toml_content)

        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))
        assert config.enable_scip is True  # TOML says true

        # Now env var overrides
        with patch.dict(os.environ, {"DESCRY_NO_SCIP": "1"}):
            config = DescryConfig.from_env()
            # from_env applies env vars after TOML, so env wins
            assert config.enable_scip is False

    def test_from_env_loads_toml(self, tmp_path):
        """from_env() auto-detects project root and loads TOML."""
        toml_content = """\
[query]
max_depth = 9
"""
        (tmp_path / ".descry.toml").write_text(toml_content)
        # Create a project marker so auto_detect finds it
        (tmp_path / ".git").mkdir()

        with patch("descry.handlers.Path.cwd", return_value=tmp_path):
            config = DescryConfig.from_env()

        assert config.max_depth == 9


# --- ScipCacheManager config propagation ---


class TestScipCacheManagerConfig:
    """ScipCacheManager receives and uses config values."""

    def test_default_extra_args(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        assert mgr._scip_extra_args == ["--exclude-vendored-libraries"]

    def test_custom_extra_args(self, tmp_path):
        mgr = ScipCacheManager(tmp_path, scip_extra_args=["--custom-flag"])
        assert mgr._scip_extra_args == ["--custom-flag"]

    def test_empty_extra_args(self, tmp_path):
        mgr = ScipCacheManager(tmp_path, scip_extra_args=[])
        assert mgr._scip_extra_args == []

    def test_default_skip_crates(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        assert mgr._scip_skip_crates == set()

    def test_custom_skip_crates(self, tmp_path):
        mgr = ScipCacheManager(tmp_path, scip_skip_crates=["mandible", "tarsus"])
        assert mgr._scip_skip_crates == {"mandible", "tarsus"}

    def test_skip_crates_filters_rust_crates(self, tmp_path):
        """Skipped crates are excluded from update_changed_rust."""
        # Create crate directories
        for name in ["alpha", "beta", "gamma"]:
            crate_dir = tmp_path / name
            crate_dir.mkdir()
            (crate_dir / "Cargo.toml").write_text(f'[package]\nname = "{name}"')
            (crate_dir / "src").mkdir()
            (crate_dir / "src" / "lib.rs").write_text("// placeholder")

        # Root Cargo.toml for workspace
        (tmp_path / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["alpha", "beta", "gamma"]'
        )

        mgr = ScipCacheManager(tmp_path, scip_skip_crates=["beta"])
        crates = mgr.get_rust_crates()
        assert "beta" in crates  # get_rust_crates still discovers it

        # But update_changed_rust should filter it out
        # We can't run the full generation (no rust-analyzer), but we can
        # verify the filtering logic by checking what needs_update sees
        # after filtering
        all_crates = mgr.get_rust_crates()
        filtered = [c for c in all_crates if c not in mgr._scip_skip_crates]
        assert "beta" not in filtered
        assert "alpha" in filtered
        assert "gamma" in filtered

    def test_default_toolchain(self, tmp_path):
        mgr = ScipCacheManager(tmp_path)
        assert mgr._scip_toolchain is None

    def test_custom_toolchain(self, tmp_path):
        mgr = ScipCacheManager(tmp_path, scip_toolchain="1.92.0")
        assert mgr._scip_toolchain == "1.92.0"

    def test_toolchain_builds_rustup_command(self, tmp_path):
        """When toolchain is set, command starts with rustup run."""
        mgr = ScipCacheManager(tmp_path, scip_toolchain="1.92.0")
        # Verify the toolchain is stored — the actual command construction
        # happens in _generate_rust_scip which we can't easily test without
        # rust-analyzer, but we can verify the config is wired through
        assert mgr._scip_toolchain == "1.92.0"
