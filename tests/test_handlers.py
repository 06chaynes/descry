"""Tests for descry.handlers — DescryConfig, DescryService, and format helpers."""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from descry.handlers import (
    DescryConfig,
    DescryService,
    ServiceError,
    format_compact_result,
    format_search_result,
    is_natural_language_query,
)

# --- DescryConfig ---


class TestDescryConfig:
    def test_default_config(self):
        config = DescryConfig()
        assert config.max_stale_hours == 24
        assert config.enable_scip is True
        assert config.enable_embeddings is True

    def test_from_env(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".descry.toml").write_text("[project]\nmax_stale_hours = 48\n")
        monkeypatch.chdir(tmp_path)
        for var in ("DESCRY_NO_SCIP", "DESCRY_NO_EMBEDDINGS", "DESCRY_LOG_LEVEL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DESCRY_CACHE_DIR", str(tmp_path / "elsewhere"))

        config = DescryConfig.from_env()

        assert config.project_root.resolve() == tmp_path.resolve()
        assert config.max_stale_hours == 48
        assert config.cache_dir == tmp_path / "elsewhere"

    def test_from_env_names_an_invalid_log_level(self, tmp_path, monkeypatch, caplog):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DESCRY_LOG_LEVEL", "LOUD")
        with caplog.at_level(logging.WARNING, logger="descry"):
            DescryConfig.from_env()
        assert any(
            "DESCRY_LOG_LEVEL" in r.message and "'LOUD'" in r.message
            for r in caplog.records
        )
        assert logging.getLogger("descry").level == logging.WARNING

    def test_importing_handlers_does_not_import_the_embedding_stack(self):
        code = (
            "import sys, descry.handlers; "
            "bad = [m for m in ('sentence_transformers', 'torch') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

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
        result = format_search_result(node)
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
            "schema_version": 1,
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


class TestDescryServiceIndex:
    """The child process embeds; the parent must not encode the graph again."""

    @pytest.fixture
    def indexed(self, tmp_path):
        config = DescryConfig(
            project_root=tmp_path, enable_embeddings=False, enable_scip=False
        )
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        config.graph_path.write_text('{"schema_version": 1, "nodes": [], "edges": []}')
        svc = DescryService(config)
        built = []

        class Searcher:
            nodes: tuple = ()

            def __init__(self, graph_path, **kwargs):
                self.kwargs = kwargs
                self.model_loads = 0
                built.append(self)

            def ensure_model(self):
                self.model_loads += 1

        svc._semantic_available = True
        svc._SemanticSearcher = Searcher
        return svc, built

    @staticmethod
    def _child_succeeded(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout="indexed", stderr="")

    @pytest.mark.asyncio
    async def test_index_loads_the_childs_embeddings_instead_of_rebuilding(
        self, indexed
    ):
        svc, built = indexed
        with patch("descry.handlers.subprocess.run", self._child_succeeded):
            result = await svc.index()
        assert "Index complete" in result
        assert len(built) == 1
        assert not built[0].kwargs.get("force_rebuild")
        assert built[0].model_loads == 0, "a one-shot index must not load the model"
        assert svc._semantic_cache["instance"] is built[0]

    @pytest.mark.asyncio
    async def test_every_searcher_is_given_the_project_root_as_source_root(
        self, indexed, tmp_path
    ):
        svc, built = indexed
        with patch("descry.handlers.subprocess.run", self._child_succeeded):
            await svc.index()
        svc._semantic_cache["instance"] = None
        await svc._get_semantic_searcher()
        svc._semantic_cache["instance"] = None
        await svc._load_embeddings_background()
        assert [s.kwargs["source_root"] for s in built] == [str(tmp_path)] * 3

    @pytest.mark.asyncio
    async def test_index_keeps_a_loaded_model_loaded(self, indexed):
        svc, built = indexed
        svc._semantic_cache["instance"] = object()
        with patch("descry.handlers.subprocess.run", self._child_succeeded):
            await svc.index()
        assert built[0].model_loads == 1


def _write_graph(config, nodes=(), edges=()):
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.graph_path.write_text(
        json.dumps({"schema_version": 1, "nodes": list(nodes), "edges": list(edges)})
    )


def _plain_config(root, **kwargs):
    return DescryConfig(
        project_root=root, enable_embeddings=False, enable_scip=False, **kwargs
    )


@pytest.fixture
def real_indexer_env(monkeypatch):
    """Let `index()` spawn the real indexer without models or SCIP tools."""
    monkeypatch.setenv("DESCRY_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("DESCRY_NO_SCIP", "1")
    monkeypatch.delenv("DESCRY_CACHE_DIR", raising=False)


class TestIndexChildIsolation:
    @pytest.mark.asyncio
    async def test_a_descry_package_in_the_indexed_repo_is_not_executed(
        self, tmp_path, real_indexer_env
    ):
        (tmp_path / ".git").mkdir()
        (tmp_path / "app.py").write_text("def hello():\n    return 1\n")
        planted = tmp_path / "descry"
        planted.mkdir()
        (planted / "__init__.py").write_text("")
        marker = tmp_path / "EXECUTED"
        (planted / "generate.py").write_text(
            f"open({str(marker)!r}, 'w').write('planted code ran')\n"
        )
        svc = DescryService(_plain_config(tmp_path))

        result = await svc.index()

        assert not marker.exists(), "the repository's descry/generate.py was run"
        assert not isinstance(result, ServiceError), result
        graph = json.loads(svc.config.graph_path.read_text())
        assert any(n["id"] == "FILE:app.py::hello" for n in graph["nodes"])

    @pytest.mark.asyncio
    async def test_child_argv_keeps_cwd_off_sys_path_and_env_is_sanitized(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("MY_API_TOKEN", "hunter2")
        monkeypatch.setenv("HF_TOKEN", "hf_x")
        config = _plain_config(tmp_path)
        _write_graph(config)
        seen = {}

        def fake_run(argv, **kwargs):
            seen["argv"], seen["kwargs"] = argv, kwargs
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

        with patch("descry.handlers.subprocess.run", fake_run):
            await DescryService(config).index()

        argv = seen["argv"]
        assert argv[:4] == [sys.executable, "-P", "-m", "descry.generate"]
        assert "MY_API_TOKEN" not in seen["kwargs"]["env"]
        assert seen["kwargs"]["env"]["HF_TOKEN"] == "hf_x"
        assert seen["kwargs"]["env"]["DESCRY_CACHE_DIR"] == str(
            config.cache_dir.resolve()
        )


class TestIndexPath:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".descry.toml").write_text(
            '[project]\nexcluded_dirs = ["skipme"]\n'
        )
        src = tmp_path / "src"
        (src / "skipme").mkdir(parents=True)
        (src / "app.py").write_text("def hello():\n    return 1\n")
        (src / "skipme" / "hidden.py").write_text("def hidden():\n    return 2\n")
        (src / ".descry.toml").write_text("[project]\nexcluded_dirs = []\n")
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "b.py").write_text("def elsewhere():\n    return 3\n")
        return tmp_path

    @pytest.mark.asyncio
    async def test_subdir_index_lands_in_the_services_cache_with_root_paths(
        self, project, real_indexer_env, monkeypatch
    ):
        cache = project / "custom-cache"
        config = _plain_config(project, cache_dir=cache)
        config._apply_toml(DescryConfig._load_toml(project))
        svc = DescryService(config)

        result = await svc.index(str(project / "src"))

        assert result.startswith("Index complete."), result
        assert config.graph_path.exists(), "graph is not where the service reads it"
        assert not (project / "src" / ".descry_cache").exists()
        assert not (project / ".descry_cache").exists()
        ids = {n["id"] for n in json.loads(config.graph_path.read_text())["nodes"]}
        assert "FILE:src/app.py::hello" in ids, "paths must be root-relative"
        assert not any("other/b.py" in i for i in ids), "walk was not narrowed"
        assert not any("skipme" in i for i in ids), "root .descry.toml was ignored"

        monkeypatch.chdir(project / "other")
        context = await svc.context("FILE:src/app.py::hello")
        assert "def hello():" in context

    @pytest.mark.asyncio
    @pytest.mark.parametrize("escape", ["/etc", "..", "../sibling", "link-out"])
    async def test_path_outside_the_root_is_refused_before_any_subprocess(
        self, tmp_path, escape
    ):
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "sibling").mkdir()
        (root / "link-out").symlink_to(tmp_path / "sibling")
        svc = DescryService(_plain_config(root))
        path = escape if os.path.isabs(escape) else str(root / escape)

        def must_not_run(*_a, **_k):
            raise AssertionError("indexer was spawned for an out-of-root path")

        with patch("descry.handlers.subprocess.run", must_not_run):
            result = await svc.index(path)

        assert isinstance(result, ServiceError)
        assert "outside project root" in result

    @pytest.mark.asyncio
    async def test_a_path_that_is_not_a_directory_does_not_replace_the_graph(
        self, tmp_path
    ):
        config = _plain_config(tmp_path)
        _write_graph(config)
        before = config.graph_path.read_text()
        svc = DescryService(config)

        def must_not_run(*_a, **_k):
            raise AssertionError("indexer was spawned for a missing directory")

        with patch("descry.handlers.subprocess.run", must_not_run):
            result = await svc.index(str(tmp_path / "no-such-dir"))

        assert isinstance(result, ServiceError)
        assert "is not a directory" in result
        assert config.graph_path.read_text() == before


class TestUnusableGraph:
    BAD_GRAPHS: ClassVar[dict[str, bytes]] = {
        "stale-schema": b'{"schema_version": 0, "nodes": [], "edges": []}',
        "truncated-json": b'{"schema_version": 1, "nodes": [',
        "invalid-utf8": b"\xff\xfe\x00garbage",
        "not-an-object": b"[1, 2, 3]",
        "null": b"null",
        "nodes-not-a-list": b'{"schema_version": 1, "nodes": {}, "edges": []}',
        "node-without-id": (
            b'{"schema_version": 1, "nodes": [{"type": "Function"}], "edges": []}'
        ),
        "edge-without-source": (
            b'{"schema_version": 1, "nodes": [], '
            b'"edges": [{"target": "a", "relation": "CALLS"}]}'
        ),
    }

    @pytest.fixture(params=sorted(BAD_GRAPHS))
    def svc(self, request, tmp_path):
        config = _plain_config(tmp_path)
        config.cache_dir.mkdir(parents=True)
        config.graph_path.write_bytes(self.BAD_GRAPHS[request.param])
        return DescryService(config)

    @pytest.mark.asyncio
    async def test_querier_is_none_instead_of_raising(self, svc):
        assert await svc._get_querier() is None
        assert svc._querier_cache["error"]

    @pytest.mark.asyncio
    async def test_tools_say_to_reindex(self, svc):
        for result in (
            await svc.callers("x"),
            await svc.callees("x"),
            await svc.search(["x"]),
            await svc.context("x"),
            await svc.flow("x"),
            await svc.path("a", "b"),
            await svc.structure("x.py"),
        ):
            assert isinstance(result, ServiceError)
            assert "descry index" in result
            assert "Graph not found" not in result

    @pytest.mark.asyncio
    async def test_health_and_status_report_it(self, svc):
        health = json.loads(await svc.health())
        assert health["status"] == "invalid_graph"
        assert health["graph"]["error"]
        assert (await svc.status()).startswith("Graph: UNUSABLE")

    @pytest.mark.asyncio
    async def test_ensure_rebuilds_it(self, svc):
        calls = []

        async def fake_index(path="."):
            calls.append(path)
            return "Index complete."

        svc.index = fake_index
        result = await svc.ensure()
        assert calls == ["."]
        assert result.startswith("Rebuilt unusable graph.")

    @pytest.mark.asyncio
    async def test_counts_of_the_graph_it_replaced_are_not_reported(self, tmp_path):
        config = _plain_config(tmp_path)
        _write_graph(
            config,
            nodes=[{"id": "n1", "type": "Function", "metadata": {"name": "f"}}],
        )
        svc = DescryService(config)
        assert json.loads(await svc.health())["graph"]["nodes"] == 1
        config.graph_path.write_text("{")
        later = time.time() + 5
        os.utime(config.graph_path, (later, later))
        assert json.loads(await svc.health())["graph"]["nodes"] == 0

    @pytest.mark.asyncio
    async def test_unreadable_graph_is_parsed_once_per_mtime(self, tmp_path):
        config = _plain_config(tmp_path)
        config.cache_dir.mkdir(parents=True)
        config.graph_path.write_text("{")
        svc = DescryService(config)
        loads = []

        def counting(path, config=None):
            loads.append(path)
            raise ValueError("bad")

        svc._GraphQuerier = counting
        await svc._get_querier()
        await svc._get_querier()
        assert len(loads) == 1


class TestEnsure:
    @pytest.fixture
    def svc(self, tmp_path):
        config = _plain_config(tmp_path)
        _write_graph(config)
        two_hours_ago = time.time() - 2 * 3600
        os.utime(config.graph_path, (two_hours_ago, two_hours_ago))
        svc = DescryService(config)
        svc.indexed = []

        async def fake_index(path="."):
            svc.indexed.append(path)
            return "Index complete."

        svc.index = fake_index
        return svc

    @pytest.mark.asyncio
    async def test_default_age_limit_is_the_configured_max_stale_hours(self, svc):
        svc.config.max_stale_hours = 1
        result = await svc.ensure()
        assert svc.indexed == ["."]
        assert result.startswith("Refreshed stale graph")

    @pytest.mark.asyncio
    async def test_default_age_limit_leaves_a_fresh_enough_graph_alone(self, svc):
        svc.config.max_stale_hours = 24
        assert (await svc.ensure()).startswith("Graph ready")
        assert svc.indexed == []

    @pytest.mark.asyncio
    async def test_explicit_age_limit_wins_over_config(self, svc):
        svc.config.max_stale_hours = 1
        assert (await svc.ensure(max_age_hours=100)).startswith("Graph ready")
        assert svc.indexed == []

    @pytest.mark.asyncio
    async def test_failed_index_is_not_dressed_as_a_generated_graph(self, tmp_path):
        svc = DescryService(_plain_config(tmp_path))

        async def failing_index(path="."):
            return ServiceError("Index failed:\nboom")

        svc.index = failing_index
        result = await svc.ensure()
        assert isinstance(result, ServiceError)
        assert "Generated new graph" not in result


class TestServiceErrors:
    @pytest.fixture
    def svc(self, tmp_path):
        config = _plain_config(tmp_path)
        _write_graph(config)
        return DescryService(config)

    @pytest.mark.asyncio
    async def test_index_child_failure(self, svc):
        failed = subprocess.CompletedProcess([], 3, stdout="", stderr="traceback")
        with patch("descry.handlers.subprocess.run", lambda *a, **k: failed):
            result = await svc.index()
        assert isinstance(result, ServiceError)
        assert result.startswith("Index failed:")

    @pytest.mark.asyncio
    async def test_index_timeout(self, svc):
        def timed_out(*_a, **_k):
            raise subprocess.TimeoutExpired("descry.generate", 1)

        with patch("descry.handlers.subprocess.run", timed_out):
            result = await svc.index()
        assert isinstance(result, ServiceError)
        assert result.startswith("Index timed out")

    @pytest.mark.asyncio
    async def test_index_spawn_error(self, svc):
        def broken(*_a, **_k):
            raise OSError("no such interpreter")

        with patch("descry.handlers.subprocess.run", broken):
            result = await svc.index()
        assert isinstance(result, ServiceError)
        assert result.startswith("Index error:")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool", "method", "args"),
        [
            ("churn", "get_churn", {}),
            ("evolution", "get_evolution", {"name": "f"}),
            ("changes", "get_changes", {}),
        ],
    )
    async def test_git_tool_failures(self, svc, tool, method, args):
        class Analyzer:
            def __getattr__(self, name):
                def boom(**_kwargs):
                    raise RuntimeError("git exploded")

                return boom

        async def analyzer():
            return Analyzer()

        svc._get_git_analyzer = analyzer
        result = await getattr(svc, tool)(**args)
        assert isinstance(result, ServiceError)
        assert result.startswith(f"Error analyzing {tool}")

    @pytest.mark.asyncio
    async def test_search_without_a_graph(self, tmp_path):
        svc = DescryService(_plain_config(tmp_path))
        result = await svc.search(["anything"])
        assert isinstance(result, ServiceError)
        assert "ERROR: Graph not found" in result

    @pytest.mark.asyncio
    async def test_a_miss_is_an_answer_not_a_failure(self, svc):
        for result in (await svc.search(["nothing_here"]), await svc.callers("nope")):
            assert not isinstance(result, ServiceError)

    @pytest.mark.asyncio
    async def test_openapi_path_that_is_a_directory(self, svc, tmp_path):
        svc.config.openapi_path = tmp_path
        result = await svc.cross_lang(mode="stats")
        assert isinstance(result, ServiceError)
        assert "OpenAPI spec not found" in result


class TestCallersCalleesUseOneQuerier:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("tool", "name"), [("callers", "b"), ("callees", "a")])
    async def test_graph_deleted_between_lookup_and_render(self, tmp_path, tool, name):
        config = _plain_config(tmp_path)
        _write_graph(
            config,
            nodes=[
                {"id": f"FILE:x.py::{n}", "type": "Function", "metadata": {"name": n}}
                for n in ("a", "b")
            ],
            edges=[
                {
                    "source": "FILE:x.py::a",
                    "target": "FILE:x.py::b",
                    "relation": "CALLS_RESOLVED",
                }
            ],
        )
        svc = DescryService(config)
        querier = await svc._get_querier()
        handed_out = []

        async def once_then_gone():
            handed_out.append(1)
            return querier if len(handed_out) == 1 else None

        svc._get_querier = once_then_gone
        result = await getattr(svc, tool)(name)
        assert "FILE:x.py::" in result
        assert len(handed_out) == 1


class TestSearchLanguageFilter:
    @pytest.fixture
    def svc(self, tmp_path):
        config = _plain_config(tmp_path)
        _write_graph(
            config,
            nodes=[
                {
                    "id": f"FILE:{path}::handler",
                    "type": "Function",
                    "metadata": {"name": "handler"},
                }
                for path in ("svc/main.go", "web/app.ts", "tools/run.py")
            ],
        )
        return DescryService(config)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("lang", "match"), [("go", "main.go"), ("ts", "app.ts"), ("py", "run.py")]
    )
    async def test_lang_narrows_the_results(self, svc, lang, match):
        result = await svc.search(["handler"], lang=lang)
        assert f"1 match(es) for 'handler' [keyword] [lang={lang}" in result
        assert match in result

    @pytest.mark.asyncio
    async def test_unknown_language_matches_nothing(self, svc):
        result = await svc.search(["handler"], lang="klingon")
        assert result.endswith("No matches for 'handler' [lang=klingon, compact].")

    @pytest.mark.asyncio
    async def test_all_does_not_filter(self, svc):
        result = await svc.search(["handler"], lang="all")
        assert "3 match(es)" in result
        assert "lang=" not in result


class TestStatusEmbeddingProbe:
    @pytest.mark.asyncio
    async def test_status_probes_the_configured_model_cache_dir_and_source_root(
        self, tmp_path
    ):
        config = _plain_config(tmp_path, cache_dir=tmp_path / "cache")
        config.embedding_model = "qwen3"
        _write_graph(config)
        svc = DescryService(config)
        seen = {}

        def probe(graph_path, **kwargs):
            seen["graph_path"], seen["kwargs"] = graph_path, kwargs
            return {
                "cached": True,
                "node_count": 7,
                "model": "Qwen/Qwen3-Embedding-0.6B",
            }

        svc._semantic_available = True
        svc._get_embeddings_status_fn = probe
        result = await svc.status()

        assert seen["kwargs"] == {
            "model_name": "qwen3",
            "cache_dir": str(tmp_path / "cache"),
            "source_root": str(tmp_path),
        }
        assert "Embeddings: Ready (7 nodes, Qwen/Qwen3-Embedding-0.6B)" in result


class TestPathMessages:
    def _svc(self, tmp_path, chain_len, **limits):
        config = _plain_config(tmp_path, **limits)
        _write_graph(
            config,
            nodes=[
                {
                    "id": f"FILE:x.py::n{i}",
                    "type": "Function",
                    "metadata": {"name": f"n{i}"},
                }
                for i in range(chain_len)
            ]
            + [
                {
                    "id": "FILE:x.py::island",
                    "type": "Function",
                    "metadata": {"name": "island"},
                }
            ],
            edges=[
                {
                    "source": f"FILE:x.py::n{i}",
                    "target": f"FILE:x.py::n{i + 1}",
                    "relation": "CALLS_RESOLVED",
                }
                for i in range(chain_len - 1)
            ],
        )
        return DescryService(config)

    @pytest.mark.asyncio
    async def test_path_longer_than_the_configured_tree_depth_is_found(self, tmp_path):
        svc = self._svc(tmp_path, 8)
        assert svc.config.max_depth == 3
        result = await svc.path("n0", "n7")
        assert "(7 hops)" in result

    @pytest.mark.asyncio
    async def test_depth_miss_suggests_a_larger_max_depth(self, tmp_path):
        svc = self._svc(tmp_path, 8)
        result = await svc.path("n0", "n7", max_depth=2)
        assert "within 2 hops" in result
        assert "Increase max_depth" in result

    @pytest.mark.asyncio
    async def test_miss_reports_the_depth_actually_searched(self, tmp_path):
        svc = self._svc(tmp_path, 4)
        result = await svc.path("n0", "island", max_depth=10**6)
        assert "within 50 hops" in result

    @pytest.mark.asyncio
    async def test_node_budget_is_not_reported_as_no_path(self, tmp_path):
        svc = self._svc(tmp_path, 30, max_nodes=5)
        result = await svc.path("n0", "n29", max_depth=40)
        assert "[query] max_nodes" in result
        assert "may still exist" in result
        assert "No path found" not in result

    @pytest.mark.asyncio
    async def test_timeout_is_not_reported_as_no_path(self, tmp_path):
        svc = self._svc(tmp_path, 30, query_timeout_ms=-1)
        result = await svc.path("n0", "n29", max_depth=40)
        assert "[timeouts] query_ms" in result
        assert "No path found" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("end", "limits", "status"),
        [
            ("n7", {}, "found"),
            ("island", {}, "not_found"),
            ("n29", {"max_nodes": 5}, "node_budget"),
            ("n29", {"query_timeout_ms": -1}, "timeout"),
        ],
    )
    async def test_data_says_how_the_search_ended(self, tmp_path, end, limits, status):
        svc = self._svc(tmp_path, 30, **limits)
        data = await svc.path_data("n0", end, max_depth=40)
        assert data["status"] == status
        if status == "found":
            assert len(data["path"]) == 7
            assert data["message"] is None
        else:
            assert data["path"] == []
            assert await svc.path("n0", end, max_depth=40) == data["message"]


class TestTextRendersFromData:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool", "args"),
        [
            ("flow", {"start": "f"}),
            ("structure", {"filename": "x.py"}),
            ("quick", {"name": "f"}),
            ("impls", {"method": "f"}),
            ("path", {"start": "a", "end": "b"}),
            ("semantic", {"query": "f"}),
            ("cross_lang", {"mode": "stats"}),
            ("churn", {}),
            ("evolution", {"name": "f"}),
            ("changes", {}),
        ],
    )
    async def test_data_method_failure_is_the_text_methods_error(
        self, tmp_path, tool, args
    ):
        svc = DescryService(_plain_config(tmp_path))
        calls = []

        async def failed(*a, **kw):
            calls.append((a, kw))
            return {"ok": False, "error": "boom", "reason": "failed"}

        setattr(svc, f"{tool}_data", failed)
        result = await getattr(svc, tool)(**args)

        assert isinstance(result, ServiceError)
        assert result.endswith("boom")
        assert len(calls) == 1


class TestSemanticText:
    NODE: ClassVar[dict] = {
        "id": "FILE:x.py::f",
        "type": "Function",
        "metadata": {"name": "f"},
    }

    def _svc(self, tmp_path, case):
        """A service whose semantic search is in the state named by `case`."""
        config = _plain_config(tmp_path)
        _write_graph(config, nodes=[self.NODE])
        svc = DescryService(config)
        load_error = {
            "load-fails": RuntimeError("model is gated"),
            "load-fails-silently": RuntimeError(),
        }.get(case)
        node = self.NODE

        class Searcher:
            def __init__(self, graph_path, **kwargs):
                if load_error is not None:
                    raise load_error

            def search(self, query, limit=10):
                if case == "search-fails":
                    raise ValueError("bad vector")
                return [(node, 0.9)]

        svc._semantic_available = case != "unavailable"
        svc._SemanticSearcher = Searcher
        svc._semantic_cache["loading"] = case == "loading"
        if case == "no-graph":
            config.graph_path.unlink()
        return svc

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("case", "reason", "result_type", "text"),
        [
            (
                "unavailable",
                "unavailable",
                ServiceError,
                (
                    "Semantic search not available. Install dependencies:\n"
                    "  pip install sentence-transformers numpy\n\n"
                    "Falling back to TF-IDF keyword search..."
                ),
            ),
            (
                "no-graph",
                "no_graph",
                str,
                "ERROR: Graph not found. Run descry ensure first.",
            ),
            (
                "loading",
                "loading",
                str,
                "Embeddings loading in background (~2-3s). Try again shortly.",
            ),
            (
                "load-fails",
                "failed",
                ServiceError,
                "Failed to initialize semantic search: model is gated",
            ),
            (
                "search-fails",
                "failed",
                ServiceError,
                "Semantic search error: bad vector",
            ),
        ],
        ids=["unavailable", "no-graph", "loading", "load-fails", "search-fails"],
    )
    async def test_each_failure_has_its_own_message(
        self, tmp_path, case, reason, result_type, text
    ):
        svc = self._svc(tmp_path, case)
        result = await svc.semantic("q")
        assert type(result) is result_type
        assert result == text
        assert (await svc.semantic_data("q"))["reason"] == reason

    @pytest.mark.asyncio
    async def test_load_failure_without_a_message_is_not_reported_as_loading(
        self, tmp_path
    ):
        svc = self._svc(tmp_path, "load-fails-silently")
        data = await svc.semantic_data("q")
        assert (data["ok"], data["reason"]) == (False, "failed")
        assert data["error"] == "Semantic search unavailable."
        result = await svc.semantic("q")
        assert isinstance(result, ServiceError)
        assert result == "Semantic search unavailable."

    @pytest.mark.asyncio
    async def test_hits_are_rendered_with_their_scores(self, tmp_path):
        svc = self._svc(tmp_path, "hits")
        result = await svc.semantic("q")
        assert not isinstance(result, ServiceError)
        assert "1 semantic match(es) for 'q':" in result
        assert "[0.90] [Fun] f | x.py" in result


class TestCrossLangSpecDiscovery:
    SPEC = '{"paths": {"/users": {"get": {"operationId": "list_users"}}}}'

    @pytest.fixture
    def svc(self, tmp_path):
        config = _plain_config(tmp_path)
        _write_graph(config)
        return DescryService(config)

    @staticmethod
    def _conventional(root, name):
        spec = root / "public" / "api" / name
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(TestCrossLangSpecDiscovery.SPEC)
        return spec

    @pytest.mark.asyncio
    async def test_conventional_location_needs_no_config(self, svc, tmp_path):
        spec = self._conventional(tmp_path, "openapi.json")

        data = await svc.cross_lang_data(mode="stats")
        assert data["ok"]
        assert data["result"]["openapi_path"] == str(spec)
        assert data["result"]["total_endpoints"] == 1
        assert svc.cross_lang_error() is None
        assert str(spec) in await svc.cross_lang(mode="stats")

    @pytest.mark.asyncio
    async def test_latest_json_is_preferred_to_openapi_json(self, svc, tmp_path):
        self._conventional(tmp_path, "openapi.json")
        latest = self._conventional(tmp_path, "latest.json")
        data = await svc.cross_lang_data(mode="stats")
        assert data["result"]["openapi_path"] == str(latest)

    @pytest.mark.asyncio
    async def test_configured_path_is_preferred_to_a_conventional_one(
        self, svc, tmp_path
    ):
        self._conventional(tmp_path, "openapi.json")
        svc.config.openapi_path = tmp_path / "custom.json"
        svc.config.openapi_path.write_text(self.SPEC)
        data = await svc.cross_lang_data(mode="stats")
        assert data["result"]["openapi_path"] == str(tmp_path / "custom.json")

    @pytest.mark.asyncio
    async def test_configured_path_that_is_missing_does_not_fall_back(
        self, svc, tmp_path
    ):
        self._conventional(tmp_path, "openapi.json")
        svc.config.openapi_path = tmp_path / "gone.json"
        data = await svc.cross_lang_data(mode="stats")
        assert (data["ok"], data["reason"]) == (False, "unavailable")
        assert "OpenAPI spec not found" in data["error"]

    @pytest.mark.asyncio
    async def test_conventional_spec_symlinked_out_of_the_project_is_ignored(
        self, tmp_path
    ):
        outside = tmp_path / "outside.json"
        outside.write_text(self.SPEC)
        root = tmp_path / "root"
        (root / "public" / "api").mkdir(parents=True)
        (root / "public" / "api" / "openapi.json").symlink_to(outside)
        svc = DescryService(_plain_config(root))

        data = await svc.cross_lang_data(mode="stats")

        assert (data["ok"], data["reason"]) == (False, "unavailable")
        assert "not configured" in data["error"]
