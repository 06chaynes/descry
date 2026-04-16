"""Tests for descry.handlers — DescryConfig, DescryService, and format helpers."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from descry.handlers import (
    DescryConfig,
    DescryService,
    format_search_result,
    format_compact_result,
    is_natural_language_query,
)


# --- DescryConfig ---


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

    def test_graph_path(self, tmp_path):
        config = DescryConfig(project_root=tmp_path)
        assert config.graph_path == tmp_path / ".descry_cache" / "codebase_graph.json"

    def test_custom_cache_dir(self, tmp_path):
        custom_cache = tmp_path / "my_cache"
        config = DescryConfig(project_root=tmp_path, cache_dir=custom_cache)
        assert config.cache_dir == custom_cache
        assert config.graph_path == custom_cache / "codebase_graph.json"

    def test_auto_detect_with_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        with patch("descry.handlers.Path.cwd", return_value=tmp_path):
            config = DescryConfig.auto_detect()
        assert config.project_root == tmp_path

    def test_auto_detect_with_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        with patch("descry.handlers.Path.cwd", return_value=tmp_path):
            config = DescryConfig.auto_detect()
        assert config.project_root == tmp_path

    def test_auto_detect_with_descry_toml(self, tmp_path):
        (tmp_path / ".descry.toml").write_text("[project]\nmax_stale_hours = 48")
        with patch("descry.handlers.Path.cwd", return_value=tmp_path):
            config = DescryConfig.auto_detect()
        assert config.project_root == tmp_path

    def test_index_timeout_default(self):
        config = DescryConfig()
        assert config.index_timeout_minutes == 30

    def test_index_timeout_from_toml(self, tmp_path):
        (tmp_path / ".descry.toml").write_text("[timeouts]\nindex_minutes = 60")
        (tmp_path / ".git").mkdir()
        config = DescryConfig(project_root=tmp_path)
        config._apply_toml(DescryConfig._load_toml(tmp_path))
        assert config.index_timeout_minutes == 60


# --- Format Helpers ---


class TestFormatSearchResult:
    def test_basic_result(self):
        node = {
            "id": "FILE:src/lib.rs::add",
            "type": "Function",
            "metadata": {
                "name": "add",
                "lineno": 10,
                "token_count": 50,
                "in_degree": 3,
                "signature": "fn add(a: i32, b: i32) -> i32",
            },
        }
        result = format_search_result(node, rank=1)
        assert "add" in result
        assert "src/lib.rs:10" in result
        assert "50 toks" in result
        assert "3 callers" in result
        assert "fn add" in result

    def test_result_with_docstring(self):
        node = {
            "id": "FILE:src/lib.rs::process",
            "type": "Function",
            "metadata": {
                "name": "process",
                "lineno": 20,
                "token_count": 100,
                "in_degree": 5,
                "docstring": "Process the incoming request and return a response.",
            },
        }
        result = format_search_result(node)
        assert "Process the incoming request" in result

    def test_result_with_score(self):
        node = {
            "id": "FILE:src/lib.rs::foo",
            "type": "Function",
            "metadata": {"name": "foo", "token_count": 0, "in_degree": 0},
        }
        result = format_search_result(node, show_score=True, score=0.95)
        assert "[0.95]" in result


class TestFormatCompactResult:
    def test_compact_format(self):
        node = {
            "id": "FILE:src/auth.rs::validate",
            "type": "Function",
            "metadata": {
                "name": "validate",
                "lineno": 5,
                "token_count": 80,
                "in_degree": 12,
            },
        }
        result = format_compact_result(node, rank=1)
        assert result.startswith("1.")
        assert "[Fun]" in result
        assert "validate" in result
        assert "80t" in result
        assert "12c" in result

    def test_compact_with_parent(self):
        node = {
            "id": "FILE:src/auth.rs::AuthService::validate",
            "type": "Method",
            "metadata": {
                "name": "validate",
                "parent_name": "AuthService",
                "lineno": 50,
                "token_count": 100,
                "in_degree": 8,
            },
        }
        result = format_compact_result(node, rank=2)
        assert "AuthService.validate" in result


class TestIsNaturalLanguageQuery:
    def test_natural_language_phrases(self):
        assert is_natural_language_query(["how", "to", "authenticate"])
        assert is_natural_language_query(["find", "the", "main", "entry", "point"])
        assert is_natural_language_query(["what", "is", "the", "auth", "flow"])
        assert is_natural_language_query(["where", "is", "the", "config", "loaded"])

    def test_code_identifiers(self):
        assert not is_natural_language_query(["validate_token"])
        assert not is_natural_language_query(["AuthService"])
        assert not is_natural_language_query(["std::collections::HashMap"])

    def test_short_queries_default_to_code(self):
        assert not is_natural_language_query(["foo"])
        assert not is_natural_language_query(["bar", "baz"])

    def test_longer_queries_default_to_natural(self):
        assert is_natural_language_query(["the", "user", "authentication"])


# --- DescryService ---


class TestDescryServiceHealth:
    @pytest.mark.asyncio
    async def test_health_no_graph(self, tmp_path):
        config = DescryConfig(project_root=tmp_path, enable_embeddings=False)
        svc = DescryService(config)
        result = await svc.health()
        assert '"status"' in result
        assert '"version"' in result
        assert '"no_graph"' in result

    @pytest.mark.asyncio
    async def test_health_with_graph(self, tmp_path):
        config = DescryConfig(project_root=tmp_path, enable_embeddings=False)
        graph = {
            "nodes": [{"id": "n1", "type": "Function", "metadata": {"name": "f"}}],
            "edges": [],
        }
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.graph_path.write_text(json.dumps(graph))
        svc = DescryService(config)
        result = await svc.health()
        assert '"nodes": 1' in result
        assert '"edges": 0' in result

    @pytest.mark.asyncio
    async def test_status_no_graph(self, tmp_path):
        config = DescryConfig(project_root=tmp_path, enable_embeddings=False)
        svc = DescryService(config)
        result = await svc.status()
        assert (
            "not found" in result.lower()
            or "no graph" in result.lower()
            or "does not exist" in result.lower()
        )


class TestDescryServiceSearch:
    @pytest.fixture
    def svc_with_graph(self, tmp_path):
        config = DescryConfig(
            project_root=tmp_path, enable_embeddings=False, enable_scip=False
        )
        graph_data = json.loads(
            (Path(__file__).parent / "fixtures" / "small_graph.json").read_text()
        )
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.graph_path.write_text(json.dumps(graph_data))
        return DescryService(config)

    @pytest.mark.asyncio
    async def test_search_finds_symbol(self, svc_with_graph):
        result = await svc_with_graph.search(["validate_token"])
        assert "validate_token" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, svc_with_graph):
        result = await svc_with_graph.search(["nonexistent_xyz_symbol"])
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_callers(self, svc_with_graph):
        result = await svc_with_graph.callers("validate_token")
        assert "handle_request" in result

    @pytest.mark.asyncio
    async def test_callees(self, svc_with_graph):
        result = await svc_with_graph.callees("main")
        assert "load_config" in result
        assert "run_server" in result

    @pytest.mark.asyncio
    async def test_quick(self, svc_with_graph):
        result = await svc_with_graph.quick("main")
        assert "main" in result
        assert "Application entry point" in result or "fn main()" in result
