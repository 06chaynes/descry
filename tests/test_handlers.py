"""Tests for descry.handlers — DescryConfig, DescryService, and format helpers."""

import pytest

from descry.handlers import (
    DescryConfig,
    DescryService,
    format_search_result,
    format_compact_result,
    is_natural_language_query,
)


class TestDescryConfig:
    def test_default_config(self):
        config = DescryConfig()
        assert config.max_stale_hours == 24
        assert config.enable_scip is True
        assert config.enable_embeddings is True
        assert ".git" in config.project_markers

    def test_from_env(self):
        config = DescryConfig.from_env()
        assert config.project_root.exists()


class TestFormatHelpers:
    def test_is_natural_language_query(self):
        assert is_natural_language_query(["how", "to", "authenticate"])
        assert not is_natural_language_query(["validate_token"])
