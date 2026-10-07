"""Tests for descry.web.server — security hardening around the local HTTP API.

The web UI is local-only and unauthenticated by design, so the only realistic
attacker is (a) a malicious webpage open in the user's browser attempting a
cross-origin or DNS-rebinding attack against the loopback server, and (b) a
user mistakenly binding to a non-loopback interface. These tests verify the
mitigations from `src/descry/web/server.py`:

- No CORSMiddleware (same-origin requests need no CORS; absence denies by
  default on cross-origin fetches).
- TrustedHostMiddleware rejects requests whose Host header is not loopback.
- Cross-origin writes (a blind `no-cors` POST) are refused.
- `/api/source` refuses paths matching the secret-name blocklist even when
  the file exists and is inside project_root, and never follows a symlink
  out of it.
- `--host` argparse only accepts loopback literals.
- The page pins its third-party assets and ships a Content-Security-Policy.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from descry.web import server as web_server


@pytest.fixture
def web_project(tmp_path, monkeypatch):
    """Fresh project root + graph + configured web server state."""
    (tmp_path / ".git").mkdir()
    graph = {
        "schema_version": 1,
        "nodes": [
            {
                "id": "FILE:app.py::hello",
                "type": "Function",
                "metadata": {
                    "name": "hello",
                    "lineno": 1,
                    "token_count": 5,
                    "in_degree": 0,
                    "signature": "def hello()",
                },
            }
        ],
        "edges": [],
    }
    cache = tmp_path / ".descry_cache"
    cache.mkdir()
    (cache / "codebase_graph.json").write_text(json.dumps(graph))
    (tmp_path / "app.py").write_text("def hello():\n    return 'hi'\n")

    # Reset the module-global config + querier caches so each test sees
    # a fresh view pinned at tmp_path.
    web_server._config = None
    web_server._service = None
    web_server._querier = None
    web_server._querier_mtime = None
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client(web_project):
    """TestClient that defaults to a loopback Host header."""
    return TestClient(web_server.app, base_url="http://127.0.0.1")


# --- TrustedHostMiddleware / DNS-rebinding protection ---


class TestTrustedHost:
    def test_loopback_host_accepted(self, client):
        r = client.get("/api/health", headers={"host": "127.0.0.1:8787"})
        assert r.status_code == 200

    def test_localhost_host_accepted(self, client):
        r = client.get("/api/health", headers={"host": "localhost:8787"})
        assert r.status_code == 200

    def test_rebinding_attacker_host_rejected(self, client):
        r = client.get("/api/health", headers={"host": "evil.com"})
        assert r.status_code == 400

    def test_subdomain_of_loopback_rejected(self, client):
        # DNS-rebinding usually presents as attacker.com resolving to
        # 127.0.0.1. TrustedHostMiddleware matches on the Host header, so
        # a made-up "localhost.evil.com" should still be refused.
        r = client.get("/api/health", headers={"host": "localhost.evil.com"})
        assert r.status_code == 400

    @pytest.mark.parametrize("host", ["[::1]:8787", "[::1]"])
    def test_ipv6_loopback_host_accepted(self, client, host):
        """What a browser sends for http://[::1]:8787 when run with --host ::1."""
        r = client.get("/api/health", headers={"host": host})
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "host",
        ["[::1].evil.com", "[::1]:8787.evil.com", "[::2]:8787"],
    )
    def test_ipv6_lookalike_host_rejected(self, client, host):
        r = client.get("/api/health", headers={"host": host})
        assert r.status_code == 400


# --- No CORSMiddleware installed ---


class TestNoCORS:
    def test_no_access_control_allow_origin_header(self, client):
        r = client.get(
            "/api/health",
            headers={"host": "127.0.0.1", "origin": "http://evil.com"},
        )
        assert r.status_code == 200
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_preflight_not_handled(self, client):
        r = client.options(
            "/api/source?file=app.py",
            headers={
                "host": "127.0.0.1",
                "origin": "http://evil.com",
                "access-control-request-method": "GET",
            },
        )
        # Without CORSMiddleware there's no preflight handler → 405 Method
        # Not Allowed. The browser then refuses the real cross-origin fetch.
        assert r.status_code in (400, 405)


# --- /api/source secret-name blocklist ---


class TestSourceSecretBlocklist:
    @pytest.mark.parametrize(
        "sensitive_rel",
        [
            ".env",
            ".env.production",
            ".envrc",
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".aws/credentials",
            ".ssh/id_rsa",
            ".ssh/id_ed25519",
            ".gnupg/private-keys-v1.d",
            "secrets/prod.pem",
            "certs/server.key",
            "bundle.p12",
            "keystore.jks",
            ".git/config",
            # A directory named .env, and the gitfile a worktree or submodule
            # has in place of a .git directory.
            ".env/backup",
            "vendor/lib/.git",
        ],
    )
    def test_sensitive_paths_refused(self, client, web_project, sensitive_rel):
        path = web_project / sensitive_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("secret=hunter2\n")
        r = client.get(
            f"/api/source?file={sensitive_rel}", headers={"host": "127.0.0.1"}
        )
        assert r.status_code == 403, (
            f"{sensitive_rel} should be blocked, got {r.status_code}"
        )
        assert "sensitive" in r.json().get("error", "").lower()

    def test_benign_paths_still_work(self, client):
        r = client.get("/api/source?file=app.py", headers={"host": "127.0.0.1"})
        assert r.status_code == 200
        assert "def hello" in r.json()["content"]

    @pytest.mark.parametrize(
        "benign_rel",
        [".gitignore", ".github/workflows/ci.yml", ".environment"],
    )
    def test_lookalike_names_are_not_blocked(self, client, web_project, benign_rel):
        path = web_project / benign_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("plain\n")
        r = client.get(f"/api/source?file={benign_rel}")
        assert r.status_code == 200

    def test_symlink_to_secret_refused(self, client, web_project):
        target = web_project / "secret.pem"
        target.write_text("-----BEGIN PRIVATE KEY-----\n")
        link = web_project / "innocent.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this platform")
        # The resolved path ends in .pem; should be refused even though the
        # request used a benign-looking name.
        r = client.get("/api/source?file=innocent.txt", headers={"host": "127.0.0.1"})
        assert r.status_code == 403


# --- /api/source existing protections still hold ---


class TestSourceContainment:
    def test_path_traversal_refused(self, client):
        r = client.get(
            "/api/source?file=../../../etc/passwd", headers={"host": "127.0.0.1"}
        )
        assert r.status_code == 400

    def test_missing_file_404(self, client):
        r = client.get(
            "/api/source?file=does-not-exist.py", headers={"host": "127.0.0.1"}
        )
        assert r.status_code == 404

    def test_missing_file_param_400(self, client):
        r = client.get("/api/source", headers={"host": "127.0.0.1"})
        assert r.status_code == 400

    def test_nul_byte_in_file_param_400(self, client):
        r = client.get("/api/source?file=app.py%00.png")
        assert r.status_code == 400
        assert "Invalid path" in r.json()["error"]

    def test_oversize_file_413(self, client, web_project, monkeypatch):
        monkeypatch.setattr(web_server, "_MAX_SOURCE_BYTES", 16)
        (web_project / "big.txt").write_text("x" * 100)
        r = client.get("/api/source?file=big.txt")
        assert r.status_code == 413

    def test_binary_file_415(self, client, web_project):
        (web_project / "blob.bin").write_bytes(bytes(range(256)) * 8)
        r = client.get("/api/source?file=blob.bin")
        assert r.status_code == 415

    def test_directory_is_not_a_regular_file(self, client, web_project):
        (web_project / "pkg").mkdir()
        r = client.get("/api/source?file=pkg")
        assert r.status_code == 400
        assert "regular file" in r.json()["error"]

    def test_fifo_is_not_a_regular_file(self, client, web_project):
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs unsupported on this platform")
        os.mkfifo(web_project / "pipe")
        r = client.get("/api/source?file=pipe")
        assert r.status_code == 400
        assert "regular file" in r.json()["error"]


def _swap_after_path_checks(monkeypatch, swap):
    """Run `swap()` once, after api_source has resolved and contained the path.

    The blocklist check sits between path resolution and the stat/open, so
    hooking it reproduces a concurrent writer winning that race.
    """
    real_check = web_server._is_sensitive_path
    done = []

    def check(rel):
        if not done:
            done.append(True)
            swap()
        return real_check(rel)

    monkeypatch.setattr(web_server, "_is_sensitive_path", check)


@pytest.fixture
def outside_dir(tmp_path_factory):
    """A directory beside the project root, holding a file that must not leak."""
    outside = tmp_path_factory.mktemp("outside")
    (outside / "notes.txt").write_text("OUTSIDE SECRET\n")
    return outside


class TestSourceRaces:
    """A path that changes between the containment check and the read."""

    def test_file_swapped_for_symlink_is_not_followed(
        self, client, web_project, outside_dir, monkeypatch
    ):
        target = web_project / "notes.txt"
        target.write_text("inside\n")

        def swap():
            target.unlink()
            os.symlink(outside_dir / "notes.txt", target)

        _swap_after_path_checks(monkeypatch, swap)
        r = client.get("/api/source?file=notes.txt")
        assert r.status_code == 400
        assert "OUTSIDE SECRET" not in r.text

    def test_parent_directory_swapped_for_symlink_is_not_followed(
        self, client, web_project, outside_dir, monkeypatch
    ):
        docs = web_project / "docs"
        docs.mkdir()
        (docs / "notes.txt").write_text("inside\n")

        def swap():
            docs.rename(web_project / "docs.orig")
            os.symlink(outside_dir, docs)

        _swap_after_path_checks(monkeypatch, swap)
        r = client.get("/api/source?file=docs/notes.txt")
        assert r.status_code == 400
        assert "OUTSIDE SECRET" not in r.text

    def test_file_grown_after_size_check_is_not_read_past_the_cap(
        self, client, web_project, monkeypatch
    ):
        monkeypatch.setattr(web_server, "_MAX_SOURCE_BYTES", 64)
        target = web_project / "grow.txt"
        target.write_text("small\n")
        real_open = os.open

        def grow_then_open(path, flags, *args, **kwargs):
            if str(path).endswith("grow.txt"):
                target.write_text("x" * 1000)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", grow_then_open)
        r = client.get("/api/source?file=grow.txt")
        assert r.status_code == 413

    def test_file_swapped_for_fifo_after_stat_is_refused(
        self, client, web_project, monkeypatch
    ):
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs unsupported on this platform")
        target = web_project / "pipe.txt"
        target.write_text("text\n")
        real_open = os.open
        writers = []

        def swap_then_open(path, flags, *args, **kwargs):
            if str(path).endswith("pipe.txt") and not writers:
                target.unlink()
                os.mkfifo(target)
                # A writer with a full sample pending, so a blocking open and
                # read of the FIFO return instead of hanging the test.
                writers.append(real_open(target, os.O_RDWR))
                os.write(writers[0], b"x" * 16384)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", swap_then_open)
        try:
            r = client.get("/api/source?file=pipe.txt")
        finally:
            os.close(writers[0])
        assert r.status_code == 400
        assert "regular file" in r.json()["error"]

    def test_symlinks_inside_the_project_still_resolve(self, client, web_project):
        (web_project / "real").mkdir()
        (web_project / "real" / "mod.py").write_text("VALUE = 1\n")
        os.symlink(web_project / "real", web_project / "alias")
        r = client.get("/api/source?file=alias/mod.py")
        assert r.status_code == 200
        assert "VALUE = 1" in r.json()["content"]


class TestCrossOriginWrites:
    """A page on another site can send POSTs here even though it cannot read
    the replies; such a request must not start an index run."""

    @pytest.fixture
    def index_calls(self, web_project, monkeypatch):
        calls = []

        async def fake_index(path="."):
            calls.append(path)
            return "Index complete."

        monkeypatch.setattr(web_server._get_service(), "index", fake_index)
        return calls

    @pytest.mark.parametrize(
        "headers",
        [
            {"origin": "http://evil.com"},
            {"origin": "null"},
            {"sec-fetch-site": "cross-site"},
        ],
    )
    @pytest.mark.parametrize("path", ["/api/index", "/api/index/stream", "/api/ensure"])
    def test_cross_origin_post_refused(self, client, index_calls, path, headers):
        r = client.post(path, headers=headers)
        assert r.status_code == 403
        assert index_calls == []

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"origin": "http://127.0.0.1"},
            {"origin": "http://127.0.0.1", "sec-fetch-site": "same-origin"},
        ],
    )
    def test_same_origin_and_non_browser_post_allowed(
        self, client, index_calls, headers
    ):
        r = client.post("/api/index", headers=headers)
        assert r.status_code == 200
        assert index_calls == ["."]

    def test_origin_must_match_the_port_too(self, client, index_calls):
        r = client.post(
            "/api/index",
            headers={"host": "127.0.0.1:8787", "origin": "http://127.0.0.1:9999"},
        )
        assert r.status_code == 403
        assert index_calls == []


# --- --host argparse validation ---


class TestLoopbackHostFlag:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_accepted(self, host):
        assert web_server._loopback_host(host) == host

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "192.168.1.10", "example.com", "", "127.0.0.2"],
    )
    def test_non_loopback_rejected(self, host):
        with pytest.raises(argparse.ArgumentTypeError):
            web_server._loopback_host(host)


class TestServiceDelegation:
    """The web layer must not carry its own copy of DescryService's logic.

    CLAUDE.md: "DescryService owns all business logic. cli.py, mcp_server.py
    and web/server.py are thin wrappers."
    """

    def test_search_ranking_lives_only_in_the_service(self):
        """Referencing the ranking helpers here means the algorithm is being rebuilt."""
        src = Path(web_server.__file__).read_text(encoding="utf-8")
        assert "is_natural_language_query" not in src
        assert "reciprocal_rank_fusion" not in src

    def test_web_defines_no_function_that_handlers_also_defines(self):
        def top_level_funcs(rel):
            src = Path(rel).read_text(encoding="utf-8")
            return {
                n.name
                for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

        overlap = top_level_funcs("src/descry/web/server.py") & top_level_funcs(
            "src/descry/handlers.py"
        )
        assert overlap == set(), (
            f"web/server.py redefines handlers functions: {overlap}"
        )

    @pytest.mark.parametrize(
        ("endpoint", "method"),
        [
            ("api_search", "search_data"),
            ("api_semantic", "semantic_data"),
            ("api_callers", "callers_data"),
            ("api_callees", "callees_data"),
            ("api_structure", "structure_data"),
            ("api_impls", "impls_data"),
            ("api_quick", "quick_data"),
            ("api_flow", "flow_data"),
            ("api_path", "path_data"),
            ("api_cross_lang", "cross_lang_data"),
            ("api_churn", "churn_data"),
            ("api_evolution", "evolution_data"),
            ("api_changes", "changes_data"),
        ],
    )
    def test_endpoint_delegates_to_service(self, endpoint, method):
        """Each endpoint's source must call the service method, not rebuild it."""
        src = inspect.getsource(getattr(web_server, endpoint))
        calls = {
            n.func.attr
            for n in ast.walk(ast.parse(src.lstrip()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert method in calls, f"{endpoint} does not call {method}"

    def test_web_builds_no_tracer_or_git_analyzer_of_its_own(self):
        src = Path(web_server.__file__).read_text(encoding="utf-8")
        for name in ("CrossLangTracer", "GitHistoryAnalyzer", "_get_git_analyzer"):
            assert name not in src, f"web/server.py reaches for {name}"

    @pytest.mark.parametrize(
        "endpoint",
        [
            "api_structure",
            "api_impls",
            "api_quick",
            "api_flow",
            "api_path",
            "api_cross_lang",
            "api_churn",
            "api_evolution",
            "api_changes",
        ],
    )
    def test_endpoint_does_not_query_the_graph_itself(self, endpoint):
        assert "_get_querier" not in inspect.getsource(getattr(web_server, endpoint))

    def test_run_index_delegates_and_does_not_spawn_its_own_subprocess(self):
        src = inspect.getsource(web_server._run_index)
        assert "index()" in src
        assert "subprocess" not in src, "web must not run the indexer itself"


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


class TestServiceBackedEndpoints:
    @pytest.mark.parametrize(
        "url",
        [
            "/api/flow?start=hello&direction=sideways",
            "/api/flow?start=hello&direction=sideways&format=tree",
            "/api/path?start=hello&end=hello&direction=Backward",
        ],
    )
    def test_unknown_direction_rejected(self, client, url):
        r = client.get(url)
        assert r.status_code == 400
        assert "direction" in r.json()["error"]

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_known_directions_accepted(self, client, direction):
        r = client.get(f"/api/flow?start=hello&format=tree&direction={direction}")
        assert r.status_code == 200
        assert r.json()["tree"]["root"]["name"] == "hello"

    def test_ensure_without_a_body_uses_the_default_age(self, client):
        r = client.post("/api/ensure")
        assert r.status_code == 200
        assert r.json()["action"] == "ready"

    @pytest.mark.parametrize(
        "body", [b"not json", b"[1, 2]", b'{"max_age_hours": "soon"}']
    )
    def test_ensure_with_a_malformed_body_is_400(self, client, body):
        r = client.post("/api/ensure", content=body)
        assert r.status_code == 400

    def test_semantic_reports_why_the_model_did_not_load(self, client):
        class GatedModel:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("model is gated; run hf auth login")

        svc = web_server._get_service()
        svc._semantic_available = True
        svc._SemanticSearcher = GatedModel

        r = client.get("/api/semantic?query=greeting")
        assert r.status_code == 503
        assert "model is gated" in r.json()["error"]
        assert "Graph not found" not in r.json()["error"]

    def test_health_features_honor_the_enable_flags(self, web_project, monkeypatch):
        """A disabled feature is reported unavailable even when its indexer or
        model is installed."""
        import descry.scip.support as scip_support

        monkeypatch.delenv("DESCRY_NO_SCIP", raising=False)
        monkeypatch.delenv("DESCRY_NO_EMBEDDINGS", raising=False)
        monkeypatch.setattr(scip_support, "available_adapters", lambda: [object()])
        (web_project / ".descry.toml").write_text(
            "[features]\nenable_scip = false\nenable_embeddings = false\n"
        )
        client = TestClient(web_server.app, base_url="http://127.0.0.1")

        features = client.get("/api/health").json()["features"]
        assert features["scip"] is False
        assert features["embeddings"] is False

    def test_health_features_match_the_service(self, client):
        served = client.get("/api/health").json()["features"]
        own = json.loads(asyncio.run(web_server._get_service().health()))["features"]
        assert served == {**own, "cross_lang": False}

    def test_an_unusable_graph_is_reported_and_rebuilt(
        self, client, web_project, monkeypatch
    ):
        graph = web_project / ".descry_cache" / "codebase_graph.json"
        graph.write_text('{"schema_version": 0, "nodes": [], "edges": []}')
        calls = []

        async def fake_index(path="."):
            calls.append(path)
            return "Index complete."

        monkeypatch.setattr(web_server._get_service(), "index", fake_index)

        assert client.get("/api/health").json()["status"] == "invalid_graph"
        assert client.post("/api/ensure").json()["action"] == "generated"
        assert calls == ["."]

    def test_context_for_an_unknown_node_is_an_error_status(self, client):
        r = client.get("/api/context?node_id=FILE:missing.py::nothing_here_xyz")
        assert r.status_code == 404
        assert "not found" in r.json()["error"].lower()

    def test_context_for_a_known_node_renders(self, client):
        r = client.get("/api/context?node_id=FILE:app.py::hello")
        assert r.status_code == 200
        assert "hello" in r.json()["markdown"]

    def test_flatten_of_a_non_class_is_an_error_status(self, client):
        r = client.get("/api/flatten?class_node_id=FILE:app.py::hello")
        assert r.status_code == 404
        assert "not a Class" in r.json()["error"]

    def test_quick_for_a_symbol_whose_source_is_gone_is_an_error_status(
        self, client, web_project
    ):
        (web_project / "app.py").unlink()
        r = client.get("/api/quick?name=hello")
        assert r.status_code == 404
        assert "not found on disk" in r.json()["error"]

    def test_context_and_flatten_report_a_missing_graph_as_503(
        self, client, web_project
    ):
        (web_project / ".descry_cache" / "codebase_graph.json").unlink()
        for url in (
            "/api/context?node_id=FILE:app.py::hello",
            "/api/flatten?class_node_id=FILE:app.py::hello",
        ):
            r = client.get(url)
            assert r.status_code == 503, url
            assert "Graph not found" in r.json()["error"]

    def test_callers_and_callees_when_the_graph_vanishes_mid_request(
        self, client, web_project, monkeypatch
    ):
        graph_path = web_project / ".descry_cache" / "codebase_graph.json"
        graph = json.loads(graph_path.read_text())
        caller = json.loads(json.dumps(graph["nodes"][0]).replace("hello", "main"))
        graph["nodes"].append(caller)
        graph["edges"].append(
            {
                "source": caller["id"],
                "target": "FILE:app.py::hello",
                "relation": "CALLS",
            }
        )
        graph_path.write_text(json.dumps(graph))

        async def no_graph():
            return None

        monkeypatch.setattr(web_server, "_get_querier", no_graph)
        assert client.get("/api/callers?name=hello").status_code == 503
        assert client.get("/api/callees?name=main").status_code == 503

    def test_index_keeps_the_searcher_the_service_loaded(self, client, monkeypatch):
        svc = web_server._get_service()
        seeded = object()

        async def fake_index(path="."):
            svc.reset_caches()
            svc._semantic_cache["instance"] = seeded
            return "Index complete."

        monkeypatch.setattr(svc, "index", fake_index)
        r = client.post("/api/index")
        assert r.status_code == 200
        assert svc._semantic_cache["instance"] is seeded

    def test_stream_relays_the_service_index_stream(self, client, monkeypatch):
        async def fake_stream():
            yield {"type": "output", "line": "Parsed 1 file"}
            yield {"type": "done", "success": True, "message": "Index complete."}

        monkeypatch.setattr(web_server._get_service(), "index_stream", fake_stream)
        events = _sse_events(client.post("/api/index/stream").text)

        assert [e["type"] for e in events] == ["start", "output", "done"]
        assert events[1]["line"] == "Parsed 1 file"
        assert events[2]["success"] is True
        assert events[2]["graph"]["nodes"] == 1
        assert not web_server._index_lock.locked()


def _fn(name: str, file: str = "app.py", **meta) -> dict:
    return {
        "id": f"FILE:{file}::{name}",
        "type": "Function",
        "metadata": {"name": name, **meta},
    }


def _write_web_graph(root: Path, nodes, edges=()) -> None:
    graph = {"schema_version": 1, "nodes": list(nodes), "edges": list(edges)}
    (root / ".descry_cache" / "codebase_graph.json").write_text(json.dumps(graph))


class TestGraphEndpoints:
    @pytest.mark.parametrize(
        "url",
        [
            "/api/structure?filename=app.py",
            "/api/impls?method=area",
            "/api/quick?name=hello",
            "/api/flow?start=hello",
            "/api/flow?start=hello&format=tree",
            "/api/path?start=hello&end=hello",
        ],
    )
    def test_missing_graph_is_503_with_the_services_explanation(
        self, client, web_project, url
    ):
        (web_project / ".descry_cache" / "codebase_graph.json").unlink()
        r = client.get(url)
        assert r.status_code == 503
        assert r.json() == {"error": "Graph not found. Run descry ensure first."}

    def test_structure_groups_the_files_definitions_by_type(self, client, web_project):
        _write_web_graph(
            web_project,
            nodes=[
                {"id": "FILE:app.py", "type": "File", "metadata": {"name": "app.py"}},
                _fn("later", lineno=9, signature="def later()"),
                _fn("hello", lineno=1, signature="def hello()"),
                {
                    "id": "FILE:app.py::LIMIT",
                    "type": "Constant",
                    "metadata": {"name": "LIMIT", "lineno": 4},
                },
            ],
            edges=[
                {"source": "FILE:app.py", "target": target, "relation": relation}
                for target, relation in (
                    ("FILE:app.py::later", "DEFINES"),
                    ("FILE:app.py::hello", "DEFINES"),
                    ("FILE:app.py::LIMIT", "DEFINES"),
                    ("MODULE:sys", "IMPORTS"),
                    ("os.path", "IMPORTS"),
                )
            ],
        )

        r = client.get("/api/structure?filename=app.py")

        assert r.status_code == 200
        assert r.json() == {
            "file": "FILE:app.py",
            "imports": ["os.path", "sys"],
            "definitions": {
                "constant": [
                    {
                        "name": "LIMIT",
                        "lineno": 4,
                        "signature": "",
                        "id": "FILE:app.py::LIMIT",
                    }
                ],
                "function": [
                    {
                        "name": "hello",
                        "lineno": 1,
                        "signature": "def hello()",
                        "id": "FILE:app.py::hello",
                    },
                    {
                        "name": "later",
                        "lineno": 9,
                        "signature": "def later()",
                        "id": "FILE:app.py::later",
                    },
                ],
            },
        }
        text = asyncio.run(web_server._get_service().structure("app.py"))
        assert "Imports: os.path, sys" in text
        assert "Fn: hello, later" in text

    def test_structure_lists_a_definition_without_a_name(self, client, web_project):
        _write_web_graph(
            web_project,
            nodes=[
                {"id": "FILE:app.py", "type": "File", "metadata": {"name": "app.py"}},
                {"id": "FILE:app.py::anon", "type": "Function", "metadata": {}},
            ],
            edges=[
                {
                    "source": "FILE:app.py",
                    "target": "FILE:app.py::anon",
                    "relation": "DEFINES",
                }
            ],
        )

        r = client.get("/api/structure?filename=app.py")

        assert r.status_code == 200
        assert r.json()["definitions"]["function"][0]["name"] == ""

    def test_structure_of_an_unknown_file_is_404(self, client):
        r = client.get("/api/structure?filename=nope.py")
        assert r.status_code == 404
        assert r.json() == {"error": "File 'nope.py' not found"}

    def test_impls_lists_each_implementation_with_its_type_and_trait(
        self, client, web_project
    ):
        _write_web_graph(
            web_project,
            nodes=[
                {
                    "id": "FILE:shapes.rs::Circle::area",
                    "type": "Method",
                    "metadata": {
                        "name": "area",
                        "trait_impl": "Shape",
                        "lineno": 3,
                        "signature": "fn area(&self) -> f64",
                    },
                },
                {
                    "id": "FILE:shapes.rs::Blob::area",
                    "type": "Method",
                    "metadata": {"name": "area", "trait_impl": "Measure"},
                },
                _fn("area", file="shapes.rs"),
            ],
        )

        filtered = client.get("/api/impls?method=area&trait_name=Shape")

        assert filtered.status_code == 200
        assert filtered.json() == {
            "method": "area",
            "trait_filter": "Shape",
            "total": 1,
            "implementations": [
                {
                    "id": "FILE:shapes.rs::Circle::area",
                    "struct_name": "Circle",
                    "trait_name": "Shape",
                    "file_path": "shapes.rs",
                    "lineno": 3,
                    "signature": "fn area(&self) -> f64",
                }
            ],
        }
        unfiltered = client.get("/api/impls?method=area").json()
        assert unfiltered["trait_filter"] is None
        assert [i["struct_name"] for i in unfiltered["implementations"]] == [
            "Circle",
            "Blob",
        ]

    def test_quick_returns_the_best_ranked_match_and_counts_the_rest(
        self, client, web_project
    ):
        _write_web_graph(
            web_project,
            nodes=[
                {
                    "id": "FILE:models.py::hello",
                    "type": "Class",
                    "metadata": {"name": "hello", "lineno": 1},
                },
                _fn("hello", lineno=1, signature="def hello()"),
            ],
        )

        r = client.get("/api/quick?name=hello&brief=true")

        assert r.status_code == 200
        body = r.json()
        assert body["symbol"]["id"] == "FILE:app.py::hello"
        assert body["other_matches"] == 1
        assert body["markdown"].startswith("### Brief: `FILE:app.py::hello`")

    def test_quick_for_an_unknown_symbol_is_404(self, client):
        r = client.get("/api/quick?name=nothing_here_xyz")
        assert r.status_code == 404
        assert r.json() == {"error": "No symbol found for 'nothing_here_xyz'"}

    def test_flow_markdown_is_the_services_trace_without_the_graph_header(self, client):
        r = client.get("/api/flow?start=hello&depth=2")

        assert r.status_code == 200
        body = r.json()
        assert (body["start"], body["direction"], body["depth"]) == (
            "hello",
            "forward",
            2,
        )
        assert body["markdown"].startswith("### Call Flow: `hello` (forward)")
        text = asyncio.run(web_server._get_service().flow("hello", depth=2))
        assert text == f"[1n/0e | 0.0h ago]\n\n{body['markdown']}"


class TestPathEndpoint:
    @pytest.fixture
    def chain(self, web_project):
        """A call chain n0 -> ... -> n29, and an `island` nothing calls."""
        _write_web_graph(
            web_project,
            nodes=[_fn(f"n{i}") for i in range(30)] + [_fn("island")],
            edges=[
                {
                    "source": f"FILE:app.py::n{i}",
                    "target": f"FILE:app.py::n{i + 1}",
                    "relation": "CALLS_RESOLVED",
                    "metadata": {"lineno": 1},
                }
                for i in range(29)
            ],
        )

    def test_found_path_lists_each_hop(self, client, chain):
        r = client.get("/api/path?start=n0&end=n2")

        assert r.status_code == 200
        body = r.json()
        assert (body["start"], body["end"], body["status"], body["hops"]) == (
            "n0",
            "n2",
            "found",
            2,
        )
        assert body["path"] == [
            {
                "step": 1,
                "caller": "n0",
                "callee": "n1",
                "file_path": "app.py",
                "line": 1,
                "snippet": "def hello():\n    return 'hi'",
            },
            {
                "step": 2,
                "caller": "n1",
                "callee": "n2",
                "file_path": "app.py",
                "line": 1,
                "snippet": "def hello():\n    return 'hi'",
            },
        ]
        assert body["markdown"].startswith("### Call Path: `n0` -> `n2` (2 hops)\n")

    def test_unreachable_target_is_not_found(self, client, chain):
        r = client.get("/api/path?start=n0&end=island")

        assert r.status_code == 200
        body = r.json()
        assert (body["status"], body["hops"], body["path"]) == ("not_found", 0, [])
        assert body["markdown"].startswith("No path found from 'n0' to 'island'")

    @pytest.mark.parametrize(
        ("limit", "value", "status", "setting"),
        [
            ("max_nodes", 5, "node_budget", "[query] max_nodes"),
            ("query_timeout_ms", -1, "timeout", "[timeouts] query_ms"),
        ],
    )
    def test_search_cut_short_is_not_reported_as_no_path(
        self, client, chain, limit, value, status, setting
    ):
        setattr(web_server._get_config(), limit, value)

        r = client.get("/api/path?start=n0&end=n29&max_depth=40")

        assert r.status_code == 200
        body = r.json()
        assert (body["status"], body["hops"], body["path"]) == (status, 0, [])
        assert setting in body["markdown"]
        assert "may still exist" in body["markdown"]
        assert "No path found" not in body["markdown"]


_OPENAPI_SPEC = {
    "paths": {
        "/users": {
            "get": {
                "operationId": "list_users",
                "tags": ["users"],
                "summary": "List users",
            }
        }
    }
}


class TestCrossLangEndpoint:
    @pytest.fixture
    def graph(self, web_project):
        _write_web_graph(web_project, nodes=[_fn("list_users", file="routes.py")])

    @pytest.fixture
    def spec(self, web_project, graph):
        """A spec named by `[cross_lang] openapi_path`."""
        (web_project / "spec.json").write_text(json.dumps(_OPENAPI_SPEC))
        (web_project / ".descry.toml").write_text(
            '[cross_lang]\nopenapi_path = "spec.json"\n'
        )

    def test_stats_counts_endpoints_and_graph_links(self, client, web_project, spec):
        r = client.get("/api/cross-lang?mode=stats")
        assert r.status_code == 200
        assert r.json() == {
            "total_endpoints": 1,
            "linked_to_graph": 1,
            "openapi_path": str((web_project / "spec.json").resolve()),
            "graph_path": str(
                (web_project / ".descry_cache" / "codebase_graph.json").resolve()
            ),
        }

    def test_list_and_endpoint_return_the_handler_info(self, client, spec):
        info = {
            "operationId": "list_users",
            "path": "/users",
            "method": "GET",
            "summary": "List users",
            "tags": ["users"],
            "node_id": "FILE:routes.py::list_users",
        }

        listed = client.get("/api/cross-lang?mode=list&tag=users")
        assert listed.status_code == 200
        assert listed.json() == {"tag": "users", "total": 1, "endpoints": [info]}

        found = client.get("/api/cross-lang?method=get&path=/api/v1/users")
        assert found.status_code == 200
        assert found.json() == info

    def test_endpoint_without_a_handler_is_404(self, client, spec):
        r = client.get("/api/cross-lang?mode=endpoint&method=delete&path=/users")
        assert r.status_code == 404
        assert r.json() == {"error": "No handler for DELETE /users"}

    @pytest.mark.parametrize(
        ("query", "error"),
        [
            ("mode=endpoint&method=get", "Endpoint mode requires 'method' and 'path'"),
            ("mode=bogus", "Unknown mode 'bogus'"),
        ],
    )
    def test_arguments_the_service_rejects_are_400(self, client, spec, query, error):
        r = client.get(f"/api/cross-lang?{query}")
        assert r.status_code == 400
        assert r.json()["error"].startswith(error)

    def test_project_without_a_spec_is_reported_not_configured(self, client):
        r = client.get("/api/cross-lang?mode=stats")
        assert r.status_code == 200
        assert r.json() == {
            "not_configured": True,
            "message": "Cross-language tracing not configured. "
            "Set openapi_path in config.",
        }
        assert client.get("/api/health").json()["features"]["cross_lang"] is False

    @pytest.mark.parametrize("name", ["latest.json", "openapi.json"])
    def test_spec_at_a_conventional_location_needs_no_config(
        self, client, web_project, graph, name
    ):
        (web_project / "public" / "api").mkdir(parents=True)
        (web_project / "public" / "api" / name).write_text(json.dumps(_OPENAPI_SPEC))

        r = client.get("/api/cross-lang?mode=stats")

        assert r.status_code == 200
        assert r.json()["total_endpoints"] == 1
        assert client.get("/api/health").json()["features"]["cross_lang"] is True

    def test_openapi_path_naming_a_directory_is_reported_not_configured(
        self, client, web_project, graph
    ):
        (web_project / "specs").mkdir()
        (web_project / ".descry.toml").write_text(
            '[cross_lang]\nopenapi_path = "specs"\n'
        )

        r = client.get("/api/cross-lang?mode=stats")

        assert r.status_code == 200
        assert r.json()["not_configured"] is True
        assert "OpenAPI spec not found" in r.json()["message"]
        assert client.get("/api/health").json()["features"]["cross_lang"] is False


class _RecordingAnalyzer:
    """Stands in for GitHistoryAnalyzer: records each call, then answers or raises."""

    def __init__(self):
        self.calls = []
        self.error = None

    def __getattr__(self, name):
        def call(**kwargs):
            self.calls.append((name, kwargs))
            if self.error is not None:
                raise self.error
            return {"from": name} if name.endswith("_structured") else f"# {name}"

        return call


class TestGitEndpoints:
    @pytest.fixture
    def analyzer(self, web_project, monkeypatch):
        recording = _RecordingAnalyzer()

        async def get_analyzer():
            return recording

        monkeypatch.setattr(
            web_server._get_service(), "_get_git_analyzer", get_analyzer
        )
        return recording

    @pytest.mark.parametrize(
        ("url", "body", "call"),
        [
            (
                "/api/churn?mode=files&limit=7&path_filter=src/&exclude_generated=false",
                {"mode": "files", "markdown": "# get_churn"},
                (
                    "get_churn",
                    {
                        "time_range": None,
                        "path_filter": "src/",
                        "limit": 7,
                        "mode": "files",
                        "exclude_generated": False,
                    },
                ),
            ),
            (
                "/api/churn?format=structured",
                {"data": {"from": "get_churn_structured"}},
                (
                    "get_churn_structured",
                    {
                        "time_range": None,
                        "path_filter": None,
                        "limit": 20,
                        "mode": "symbols",
                        "exclude_generated": True,
                    },
                ),
            ),
            (
                "/api/evolution?name=hello&show_diff=true&crate=core",
                {"name": "hello", "markdown": "# get_evolution"},
                (
                    "get_evolution",
                    {
                        "name": "hello",
                        "time_range": None,
                        "limit": 10,
                        "show_diff": True,
                        "crate": "core",
                    },
                ),
            ),
            (
                "/api/changes?commit_range=HEAD~2..HEAD&show_callers=false",
                {"markdown": "# get_changes"},
                (
                    "get_changes",
                    {
                        "commit_range": "HEAD~2..HEAD",
                        "time_range": None,
                        "path_filter": None,
                        "show_callers": False,
                        "limit": 50,
                    },
                ),
            ),
            (
                "/api/changes?format=structured&time_range=last+7+days",
                {"data": {"from": "get_changes_structured"}},
                (
                    "get_changes_structured",
                    {
                        "commit_range": None,
                        "time_range": "last 7 days",
                        "path_filter": None,
                        "show_callers": True,
                        "limit": 50,
                    },
                ),
            ),
        ],
    )
    def test_endpoint_returns_the_services_analysis(
        self, client, analyzer, url, body, call
    ):
        r = client.get(url)
        assert r.status_code == 200
        assert r.json() == body
        assert analyzer.calls == [call]

    @pytest.mark.parametrize(
        "url", ["/api/churn", "/api/evolution?name=hello", "/api/changes"]
    )
    def test_git_failure_is_500_with_gits_message(self, client, analyzer, url):
        from descry.git_history import GitError

        analyzer.error = GitError("Not inside a git repository.")
        r = client.get(url)
        assert r.status_code == 500
        assert r.json() == {"error": "Not inside a git repository."}

    @pytest.mark.parametrize(
        ("url", "tool"),
        [
            ("/api/churn?format=structured", "churn"),
            ("/api/evolution?name=hello", "evolution"),
            ("/api/changes", "changes"),
        ],
    )
    def test_unexpected_failure_is_500_naming_the_analysis(
        self, client, analyzer, url, tool
    ):
        analyzer.error = RuntimeError("boom")
        r = client.get(url)
        assert r.status_code == 500
        assert r.json() == {"error": f"Error analyzing {tool}: boom"}

    @pytest.mark.parametrize(
        "url", ["/api/churn", "/api/evolution?name=hello", "/api/changes"]
    )
    def test_missing_git_history_module_is_503(self, client, analyzer, url):
        web_server._get_service()._git_available = False
        r = client.get(url)
        assert r.status_code == 503
        assert "Git history tools not available" in r.json()["error"]
        assert analyzer.calls == []


@pytest.fixture
def fake_indexer(monkeypatch):
    """Swap the indexer child for a short script.

    Returns a function taking the script source; it returns the list that
    collects `(argv, kwargs, process)` for each spawn the service makes.
    """
    real_spawn = asyncio.create_subprocess_exec
    spawned = []

    def install(script: str) -> list:
        async def spawn(*argv, **kwargs):
            proc = await real_spawn(sys.executable, "-c", script, **kwargs)
            spawned.append((argv, kwargs, proc))
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        return spawned

    yield install
    for _argv, _kwargs, proc in spawned:
        if proc.returncode is None:
            proc.kill()


_STUCK_INDEXER = "import time; print('working', flush=True); time.sleep(120)"


class TestIndexRuns:
    @pytest.mark.asyncio
    async def test_closing_the_stream_kills_the_indexer(
        self, web_project, fake_indexer
    ):
        spawned = fake_indexer(_STUCK_INDEXER)
        response = await web_server.api_index_stream(None)
        body = response.body_iterator
        async for chunk in body:
            if "working" in chunk:
                break
        await body.aclose()

        assert await asyncio.wait_for(spawned[0][2].wait(), timeout=10) != 0
        assert not web_server._index_lock.locked()

    @pytest.mark.asyncio
    async def test_cancelling_the_response_kills_the_indexer(
        self, web_project, fake_indexer
    ):
        spawned = fake_indexer(_STUCK_INDEXER)
        response = await web_server.api_index_stream(None)
        working = asyncio.Event()

        async def read_body():
            async for chunk in response.body_iterator:
                if "working" in chunk:
                    working.set()

        reader = asyncio.create_task(read_body())
        await asyncio.wait_for(working.wait(), timeout=10)
        reader.cancel()

        assert await asyncio.wait_for(spawned[0][2].wait(), timeout=10) != 0
        assert not web_server._index_lock.locked()

    @pytest.mark.asyncio
    async def test_stream_stops_at_the_configured_index_timeout(
        self, web_project, fake_indexer
    ):
        spawned = fake_indexer(_STUCK_INDEXER)
        svc = web_server._get_service()
        svc.config.index_timeout_minutes = 0.01

        events = [event async for event in svc.index_stream()]

        assert events[0] == {"type": "output", "line": "working"}
        assert events[-1]["success"] is False
        assert "timed out" in events[-1]["message"]
        assert await asyncio.wait_for(spawned[0][2].wait(), timeout=10) != 0

    @pytest.mark.asyncio
    async def test_stream_spawns_the_indexer_safely_and_reloads_the_caches(
        self, web_project, fake_indexer, monkeypatch
    ):
        monkeypatch.setenv("DESCRY_TEST_API_TOKEN", "hunter2")
        spawned = fake_indexer("print('Parsed 1 file')")
        svc = web_server._get_service()
        svc._graph_cache["mtime"] = -1
        reloaded = []

        class Searcher:
            nodes = (object(),)

            def ensure_model(self):
                reloaded.append("model")

        async def fake_searcher():
            reloaded.append("vectors")
            return Searcher()

        monkeypatch.setattr(svc, "_get_semantic_searcher", fake_searcher)
        svc._semantic_cache["instance"] = object()

        events = [event async for event in svc.index_stream()]

        assert events == [
            {"type": "output", "line": "Parsed 1 file"},
            {"type": "output", "line": "Embeddings: 1 nodes indexed"},
            {"type": "done", "success": True, "message": "Index complete."},
        ]
        assert svc._graph_cache["mtime"] == 0
        assert reloaded == ["vectors", "model"]
        argv, kwargs, _proc = spawned[0]
        assert argv == tuple(svc._indexer_child(str(svc.config.project_root))[0])
        assert argv[:4] == (sys.executable, "-P", "-m", "descry.generate")
        assert "DESCRY_TEST_API_TOKEN" not in kwargs["env"]

    @pytest.mark.asyncio
    async def test_stream_ends_normally_when_the_model_cannot_be_reloaded(
        self, web_project, fake_indexer, monkeypatch
    ):
        fake_indexer("print('Parsed 1 file')")
        svc = web_server._get_service()

        class Searcher:
            nodes = ()

            def ensure_model(self):
                raise RuntimeError("out of memory")

        async def fake_searcher():
            return Searcher()

        monkeypatch.setattr(svc, "_get_semantic_searcher", fake_searcher)
        svc._semantic_cache["instance"] = object()

        events = [event async for event in svc.index_stream()]

        assert events[-2] == {
            "type": "output",
            "line": "Embeddings: Failed (out of memory)",
        }
        assert events[-1]["success"] is True

    @pytest.mark.asyncio
    async def test_stream_reports_a_failed_indexer(self, web_project, fake_indexer):
        fake_indexer("import sys; print('boom'); sys.exit(3)")

        events = [event async for event in web_server._get_service().index_stream()]

        assert events[-1]["success"] is False
        assert "code 3" in events[-1]["message"]

    def test_index_requests_refused_while_a_run_is_in_progress(
        self, client, web_project
    ):
        (web_project / ".descry_cache" / "codebase_graph.json").unlink()
        assert web_server._index_lock.acquire(blocking=False)
        try:
            index = client.post("/api/index")
            ensure = client.post("/api/ensure")
            stream = _sse_events(client.post("/api/index/stream").text)
        finally:
            web_server._index_lock.release()

        assert index.status_code == ensure.status_code == 409
        assert "in progress" in index.json()["error"]
        assert stream == [
            {
                "type": "done",
                "success": False,
                "message": "An index run is already in progress",
            }
        ]

    @pytest.mark.asyncio
    async def test_concurrent_index_requests_start_one_run(
        self, web_project, monkeypatch
    ):
        runs = 0

        async def slow_index(path="."):
            nonlocal runs
            runs += 1
            await asyncio.sleep(0.05)
            return "Index complete."

        monkeypatch.setattr(web_server._get_service(), "index", slow_index)
        responses = await asyncio.gather(
            *(web_server.api_index(None) for _ in range(3))
        )

        assert sorted(r.status_code for r in responses) == [200, 409, 409]
        assert runs == 1


INDEX_HTML = Path(web_server.WEB_DIR) / "index.html"


def _flow_renderer_source() -> str:
    """The flow-tree rendering methods of the page's Alpine component."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    start = html.index("        renderFlowTree(node, depth) {")
    end = html.index("        // Event delegation handler for flow tree clicks")
    return html[start:end]


def _remote_asset_tags() -> list[str]:
    html = INDEX_HTML.read_text(encoding="utf-8")
    tags = re.findall(r"<(?:script|link)\b[^>]*>", html)
    return [t for t in tags if re.search(r'(?:src|href)="https?://', t)]


class TestFlowTreeEscaping:
    """The flow tree is an HTML string built from repository-derived values
    (file names, symbol names) and inserted with x-html."""

    def test_repository_strings_are_escaped_before_interpolation(self):
        body = _flow_renderer_source()
        for raw in ("${node.name}", "${node.type}", "${node.lang"):
            assert raw not in body
        assert "const type = this._esc(node.type);" in body
        assert "const typeClass = type.toLowerCase();" in body
        assert "const fileLoc = this._esc(" in body

    def test_hostile_node_renders_as_text(self):
        node_bin = shutil.which("node")
        if node_bin is None:
            pytest.skip("node is not installed")
        payload = "<img src=x onerror=alert(1)>"
        hostile = {
            "id": 'FILE:a".py::f',
            "name": payload,
            "type": f'"{payload}',
            "file": f"FILE:{payload}.py",
            "line": 3,
            "tokens": 12,
            "lang": f'py"{payload}',
            "code": f"x = '{payload}'",
            "children": [],
        }
        script = (
            f"const component = {{\n{_flow_renderer_source()}\n}};\n"
            f"process.stdout.write(component.renderFlowTree({json.dumps(hostile)}, 0));"
        )
        out = subprocess.run(
            [node_bin, "-e", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout

        assert "&lt;img src=x onerror=alert(1)&gt;" in out
        # Quoted attribute values may hold "<"; what matters is the markup.
        markup = re.sub(r'="[^"]*"', '=""', out)
        assert set(re.findall(r"<(\w+)", markup)) == {"div", "span", "pre", "code"}


class TestPage:
    @pytest.mark.parametrize("tag", _remote_asset_tags())
    def test_remote_asset_is_pinned_and_integrity_checked(self, tag):
        url = re.search(r'(?:src|href)="([^"]+)"', tag).group(1)
        assert re.search(r"[@/]\d+\.\d+\.\d+/", url), f"not an exact version: {url}"
        integrity = re.search(r'integrity="sha384-([A-Za-z0-9+/]+=*)"', tag)
        assert integrity, f"no sha384 integrity on {url}"
        assert len(base64.b64decode(integrity.group(1))) == 48
        assert 'crossorigin="anonymous"' in tag

    def test_policy_allows_only_the_cdns_and_the_pages_own_scripts(self, client):
        html = INDEX_HTML.read_text(encoding="utf-8")
        hashes = [
            "'sha256-"
            + base64.b64encode(hashlib.sha256(block.encode()).digest()).decode()
            + "'"
            for block in re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        ]
        assert len(hashes) == 2

        header = client.get("/").headers["content-security-policy"]
        policy = {d.split()[0]: d.split()[1:] for d in header.split("; ")}

        assert policy["default-src"] == ["'none'"]
        assert policy["script-src"] == [
            "https://cdn.jsdelivr.net",
            "https://cdnjs.cloudflare.com",
            "'unsafe-eval'",
            *hashes,
        ]
        assert policy["connect-src"] == ["'self'"]
        assert policy["frame-ancestors"] == ["'none'"]
        for tag in _remote_asset_tags():
            origin = re.search(r'"(https://[^/]+)/', tag).group(1)
            assert origin in policy["script-src"] + policy["style-src"]

    def test_policy_also_covers_the_static_copy_of_the_page(self, client):
        assert (
            client.get("/static/index.html").headers["content-security-policy"]
            == client.get("/").headers["content-security-policy"]
        )

    def test_churn_truncation_note_is_rendered_as_text(self):
        html = INDEX_HTML.read_text(encoding="utf-8")
        (tag,) = re.findall(r"<[^>]*truncation_note[^>]*>", html)
        assert 'x-show="results.churn?.data?.truncation_note"' in tag
        assert 'x-text="results.churn?.data?.truncation_note"' in tag
        assert "x-html" not in tag


class TestStartup:
    @pytest.mark.parametrize("missing", ["uvicorn", "starlette"])
    def test_missing_web_extra_is_one_line_naming_the_extra(self, missing):
        code = f"import sys; sys.modules[{missing!r}] = None; import descry.web.server"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(Path(web_server.__file__).parents[2])},
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert len(result.stderr.strip().splitlines()) == 1
        assert "pip install 'descry-codegraph[web]'" in result.stderr

    def test_unrecognised_log_level_falls_back_to_warning_like_the_service(self):
        tree = ast.parse(Path(web_server.__file__).read_text(encoding="utf-8"))
        (level,) = (
            keyword.value
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "logging.basicConfig"
            for keyword in node.value.keywords
            if keyword.arg == "level"
        )
        code = compile(ast.Expression(level), "<level>", "eval")

        def resolve(name):
            return eval(code, {"logging": logging, "_log_level": name})

        assert resolve("verbose") == logging.WARNING
        assert resolve("debug") == logging.DEBUG
