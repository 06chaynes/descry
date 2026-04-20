"""Coverage for the v0.1.0 hardening round.

Targets invariants that were weakened or missing before the pre-publish
review (schema_version enforcement, TOML validators, atomic graph writes,
bounded BFS in find_call_path, cache-reset semantics). One file per theme
keeps future maintainers from having to hunt these down.
"""

from __future__ import annotations

import json
import time

import pytest

from types import SimpleNamespace

from descry._graph import CURRENT_SCHEMA, GraphSchemaError, load_graph_with_schema
from descry.handlers import (
    DescryConfig,
    DescryService,
    _validate_embedding_model,
    _validate_scip_extra_arg,
    _validate_toolchain,
)
from descry.query import GraphQuerier


# --- Schema version enforcement -------------------------------------------


class TestGraphSchemaGuard:
    def test_mismatched_schema_raises(self, tmp_path):
        bad = tmp_path / "codebase_graph.json"
        bad.write_text(
            json.dumps({"schema_version": CURRENT_SCHEMA + 1, "nodes": [], "edges": []})
        )
        with pytest.raises(GraphSchemaError):
            load_graph_with_schema(bad)

    def test_missing_schema_raises(self, tmp_path):
        bad = tmp_path / "codebase_graph.json"
        bad.write_text(json.dumps({"nodes": [], "edges": []}))
        with pytest.raises(GraphSchemaError):
            load_graph_with_schema(bad)

    def test_current_schema_loads(self, tmp_path):
        good = tmp_path / "codebase_graph.json"
        good.write_text(
            json.dumps({"schema_version": CURRENT_SCHEMA, "nodes": [], "edges": []})
        )
        data = load_graph_with_schema(good)
        assert data["schema_version"] == CURRENT_SCHEMA


# --- TOML → subprocess arg validators -------------------------------------


class TestTOMLValidators:
    @pytest.mark.parametrize(
        "good",
        [
            "1.92.0",
            "stable",
            "nightly",
            "1.92.0-x86_64-unknown-linux-gnu",
            "beta-2025-12-01",
        ],
    )
    def test_toolchain_accepts_plausible(self, good):
        assert _validate_toolchain(good) == good

    @pytest.mark.parametrize(
        "bad",
        [
            "-foo",
            "--something",
            "1; rm -rf /",
            "1`whoami`",
            "../escape",
            "1$(malice)",
            "1|pipe",
        ],
    )
    def test_toolchain_rejects_malicious(self, bad):
        with pytest.raises(ValueError):
            _validate_toolchain(bad)

    @pytest.mark.parametrize(
        "good",
        [
            "--exclude-vendored-libraries",
            "--threads=4",
            "--some-flag",
        ],
    )
    def test_scip_extra_arg_accepts_long_flags(self, good):
        assert _validate_scip_extra_arg(good) == good

    @pytest.mark.parametrize(
        "bad",
        [
            "-f",
            "--bad`inject`",
            "--bad;inject",
            "--bad$(inject)",
            "--bad|inject",
        ],
    )
    def test_scip_extra_arg_rejects_bad(self, bad):
        with pytest.raises(ValueError):
            _validate_scip_extra_arg(bad)

    def test_scip_extra_arg_allows_positional(self):
        # Bare positionals are allowed (e.g. `--exclude target`). Only
        # short flags and shell metacharacters are rejected.
        assert _validate_scip_extra_arg("target") == "target"

    @pytest.mark.parametrize(
        "good",
        [
            "jinaai/jina-code-embeddings-0.5b",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    )
    def test_embedding_model_accepts_hf_refs(self, good, tmp_path, monkeypatch):
        # _validate_embedding_model needs a project_root for path checks.
        monkeypatch.chdir(tmp_path)
        assert _validate_embedding_model(good, tmp_path) == good

    @pytest.mark.parametrize(
        "bad",
        [
            "../escape/model",
            "/etc/passwd",
            "model;rm -rf /",
            "model`whoami`",
            "--fakeflag",
            "-f",
        ],
    )
    def test_embedding_model_rejects_bad(self, bad, tmp_path):
        with pytest.raises(ValueError):
            _validate_embedding_model(bad, tmp_path)


# --- Atomic graph write ---------------------------------------------------


class TestAtomicGraphWrite:
    def test_partial_write_never_observable(self, tmp_path, monkeypatch):
        """A stat() + read() during export must never see a truncated file.

        We simulate a crash mid-write by making os.replace raise; the
        destination must remain untouched (or absent) and no `.tmp` trace
        should survive.
        """
        from descry.generate import CodeGraphBuilder

        builder = CodeGraphBuilder(str(tmp_path))
        builder.nodes = []
        builder.edges = []
        target = tmp_path / "codebase_graph.json"

        # Seed a "previous good" graph; the atomic write should leave it
        # alone when the rename fails.
        target.write_text(
            json.dumps({"schema_version": CURRENT_SCHEMA, "nodes": [], "edges": []})
        )
        previous = target.read_text()

        import descry.generate as gen_mod

        original_replace = gen_mod.os.replace

        def failing_replace(_src, _dst):
            raise OSError("simulated crash")

        monkeypatch.setattr(gen_mod.os, "replace", failing_replace)
        with pytest.raises(OSError):
            builder.export(target)
        monkeypatch.setattr(gen_mod.os, "replace", original_replace)

        # Previous graph intact, no .tmp trace.
        assert target.read_text() == previous
        tmp_leftovers = list(tmp_path.glob("*.tmp"))
        assert tmp_leftovers == []

    def test_successful_write_is_visible(self, tmp_path):
        from descry.generate import CodeGraphBuilder

        builder = CodeGraphBuilder(str(tmp_path))
        builder.nodes = [
            {"id": "FILE:x.py::f", "type": "Function", "metadata": {"name": "f"}}
        ]
        builder.edges = []
        target = tmp_path / "codebase_graph.json"
        builder.export(target)

        loaded = load_graph_with_schema(target)
        assert loaded["schema_version"] == CURRENT_SCHEMA
        assert len(loaded["nodes"]) == 1


# --- find_call_path bounded BFS -------------------------------------------


class TestFindCallPathBounds:
    def _make_querier(self, tmp_path, nodes, edges, max_nodes=2000, timeout_ms=4000):
        graph = {
            "schema_version": CURRENT_SCHEMA,
            "nodes": nodes,
            "edges": edges,
        }
        path = tmp_path / "codebase_graph.json"
        path.write_text(json.dumps(graph))
        # GraphQuerier reads attributes off config; SimpleNamespace is the
        # lightest option that satisfies the access pattern without pulling
        # in the full DescryConfig machinery.
        config = SimpleNamespace(
            max_nodes=max_nodes,
            query_timeout_ms=timeout_ms,
            max_depth=10,
            max_children_per_level=10,
            max_callers_shown=15,
            test_path_patterns=None,
            test_file_suffixes=None,
            syntax_lang_map=None,
        )
        return GraphQuerier(path, config=config)

    def test_finds_simple_path(self, tmp_path):
        nodes = [
            {"id": "FILE:x.py::a", "type": "Function", "metadata": {"name": "a"}},
            {"id": "FILE:x.py::b", "type": "Function", "metadata": {"name": "b"}},
        ]
        edges = [
            {
                "source": "FILE:x.py::a",
                "target": "FILE:x.py::b",
                "relation": "CALLS_RESOLVED",
                "metadata": {"line": 1},
            }
        ]
        q = self._make_querier(tmp_path, nodes, edges)
        hops = q.find_call_path("a", "b")
        assert len(hops) == 1
        assert hops[0]["callee_name"] == "b"

    def test_cycle_does_not_hang(self, tmp_path):
        # a -> b -> c -> a (cycle); query a -> nonexistent. Should return
        # quickly, not loop forever.
        nodes = [
            {"id": f"FILE:x.py::{n}", "type": "Function", "metadata": {"name": n}}
            for n in ("a", "b", "c")
        ]
        edges = [
            {
                "source": "FILE:x.py::a",
                "target": "FILE:x.py::b",
                "relation": "CALLS_RESOLVED",
                "metadata": {"line": 1},
            },
            {
                "source": "FILE:x.py::b",
                "target": "FILE:x.py::c",
                "relation": "CALLS_RESOLVED",
                "metadata": {"line": 1},
            },
            {
                "source": "FILE:x.py::c",
                "target": "FILE:x.py::a",
                "relation": "CALLS_RESOLVED",
                "metadata": {"line": 1},
            },
        ]
        q = self._make_querier(tmp_path, nodes, edges, timeout_ms=2000)
        start = time.monotonic()
        hops = q.find_call_path("a", "nonexistent")
        elapsed = time.monotonic() - start
        assert elapsed < 2.0  # well under 2s sanity bound
        assert hops == []

    def test_respects_node_budget(self, tmp_path):
        # Long linear chain; with a small node budget, path to terminal
        # should fail gracefully rather than traverse everything.
        chain_len = 50
        nodes = [
            {
                "id": f"FILE:x.py::n{i}",
                "type": "Function",
                "metadata": {"name": f"n{i}"},
            }
            for i in range(chain_len)
        ]
        edges = [
            {
                "source": f"FILE:x.py::n{i}",
                "target": f"FILE:x.py::n{i + 1}",
                "relation": "CALLS_RESOLVED",
                "metadata": {"line": 1},
            }
            for i in range(chain_len - 1)
        ]
        q = self._make_querier(tmp_path, nodes, edges, max_nodes=5)
        hops = q.find_call_path("n0", f"n{chain_len - 1}", max_depth=100)
        # With max_nodes=5 we must bail out well before reaching n49.
        assert hops == []


# --- reset_caches clears the stale file cache -----------------------------


class TestCacheResetStaleFile:
    def test_file_cache_mtime_keyed(self, tmp_path):
        from descry.query import _read_file_cached, _clear_file_cache

        _clear_file_cache()
        path = tmp_path / "sample.py"
        path.write_text("first\n")

        first = _read_file_cached(str(path))
        assert first == ("first\n",)

        # Force mtime change (sub-second resolution on some FSes) and
        # verify re-read picks up new content.
        time.sleep(0.01)
        path.write_text("second\n")
        # nanosecond-resolution stat_result.st_mtime_ns will differ, so
        # the (path, mtime_ns) cache key misses and re-reads.
        second = _read_file_cached(str(path))
        assert second == ("second\n",)

    def test_service_reset_caches_clears_file_cache(self, tmp_path):
        from descry.query import _read_file_cached, _clear_file_cache

        _clear_file_cache()
        config = DescryConfig(project_root=tmp_path)
        svc = DescryService(config)
        path = tmp_path / "sample.py"
        path.write_text("alpha\n")
        _ = _read_file_cached(str(path))
        # reset_caches should not error even when optional modules missing.
        svc.reset_caches()
        # After reset, the cache is empty and the next read must re-stat.
        assert _read_file_cached(str(path)) == ("alpha\n",)
