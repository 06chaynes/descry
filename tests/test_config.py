"""Tests for DescryConfig TOML loading and precedence."""

import logging
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from descry.handlers import DescryConfig


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
