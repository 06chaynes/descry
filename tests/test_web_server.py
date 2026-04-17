"""Tests for descry.web.server — security hardening around the local HTTP API.

The web UI is local-only and unauthenticated by design, so the only realistic
attacker is (a) a malicious webpage open in the user's browser attempting a
cross-origin or DNS-rebinding attack against the loopback server, and (b) a
user mistakenly binding to a non-loopback interface. These tests verify the
mitigations from `src/descry/web/server.py`:

- No CORSMiddleware (same-origin requests need no CORS; absence denies by
  default on cross-origin fetches).
- TrustedHostMiddleware rejects requests whose Host header is not loopback.
- `/api/source` refuses paths matching the secret-name blocklist even when
  the file exists and is inside project_root.
- `--host` argparse only accepts loopback literals.
"""

from __future__ import annotations

import argparse
import json
import os

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
