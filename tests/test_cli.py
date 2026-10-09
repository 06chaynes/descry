"""Tests for descry.cli — command-line interface."""

import json
import os
import signal
import subprocess
import sys
from typing import ClassVar

import pytest

from descry import cli
from descry.handlers import DescryService, ServiceError


class TestCliHelp:
    """All subcommands have working --help."""

    COMMANDS: ClassVar[list[list[str]]] = [
        [],
        ["health"],
        ["status"],
        ["ensure"],
        ["index"],
        ["search"],
        ["callers"],
        ["callees"],
        ["context"],
        ["flow"],
        ["quick"],
        ["structure"],
        ["flatten"],
        ["semantic"],
        ["embedding-models"],
        ["impls"],
        ["path"],
        ["cross-lang"],
        ["churn"],
        ["evolution"],
        ["changes"],
    ]

    @pytest.mark.parametrize("args", COMMANDS, ids=lambda a: a[0] if a else "root")
    def test_help(self, args):
        cmd = [sys.executable, "-m", "descry.cli", *args, "--help"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
        assert result.returncode == 0
        # Each subcommand prints its own usage line, not the root parser's.
        assert f"usage: {' '.join(['descry', *args])} " in result.stdout


class TestCliNoCommand:
    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "descry.cli"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 1


class TestCliHealth:
    def test_health_returns_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "descry.cli", "health"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
        assert '"version"' in result.stdout
        assert '"status"' in result.stdout


class TestCliExitStatus:
    @staticmethod
    async def _returns(value):
        return value

    @pytest.mark.parametrize(
        "failure",
        [
            "Index path '/etc' outside project root; must be relative to /repo.",
            "Index failed:\nTraceback ...",
            "Index timed out after 30 minutes.",
            "Index error: [Errno 2] No such file or directory",
            "Error analyzing churn: boom",
            "[Graph: NOT FOUND]\n\nERROR: Graph not found. Run descry ensure first.",
        ],
    )
    def test_service_error_exits_nonzero_whatever_its_text(self, failure, capsys):
        with pytest.raises(SystemExit) as exit_info:
            cli._print_result(self._returns(ServiceError(failure)))
        assert exit_info.value.code == 2
        assert failure in capsys.readouterr().out

    def test_plain_result_exits_zero(self, capsys):
        cli._print_result(self._returns("No matches for 'x'."))
        assert "No matches" in capsys.readouterr().out

    def test_error_prefix_exits_nonzero(self):
        with pytest.raises(SystemExit) as exit_info:
            cli._print_result(self._returns("ERROR: Node 'x' not found in graph."))
        assert exit_info.value.code == 2

    def test_index_outside_the_root_exits_nonzero(self, tmp_path):
        (tmp_path / ".git").mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "descry.cli", "index", "/etc"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            cwd=tmp_path,
        )
        assert result.returncode == 2
        assert "outside project root" in result.stdout

    def test_ensure_leaves_the_age_limit_to_the_service(self, monkeypatch):
        seen = {}

        class Service:
            async def ensure(self, max_age_hours):
                seen["max_age_hours"] = max_age_hours
                return "Graph ready"

        monkeypatch.setattr(cli, "_make_service", Service)
        monkeypatch.setattr(sys, "argv", ["descry", "ensure"])
        cli.main()
        assert seen["max_age_hours"] is None

    @pytest.mark.parametrize("age", ["nan", "inf", "-1"])
    def test_ensure_refuses_an_age_limit_that_is_not_an_age(
        self, tmp_path, monkeypatch, capsys, age
    ):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DESCRY_CACHE_DIR", raising=False)
        monkeypatch.setenv("DESCRY_NO_EMBEDDINGS", "1")
        indexed = []

        async def fake_index(self, path="."):
            indexed.append(path)
            return "Index complete."

        monkeypatch.setattr(DescryService, "index", fake_index)
        monkeypatch.setattr(sys, "argv", ["descry", "ensure", "--max-age-hours", age])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 2
        assert "max_age_hours must be a finite" in capsys.readouterr().out
        assert indexed == []

    def test_negative_limit_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DESCRY_CACHE_DIR", raising=False)
        monkeypatch.setenv("DESCRY_NO_EMBEDDINGS", "1")
        monkeypatch.setattr(sys, "argv", ["descry", "callers", "f", "--limit", "-1"])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 2
        assert "limit must not be negative" in capsys.readouterr().out


# `descry index` in a project whose indexer is the stand-in named by argv[1].
_INDEX_WITH = """
import json, os, sys
from descry import cli
from descry.handlers import DescryService

argv = json.loads(sys.argv[1])
DescryService._indexer_child = lambda self, _path: (argv, dict(os.environ))
sys.argv = ["descry", "index"]
cli.main()
"""


class TestCliStopped:
    @pytest.mark.parametrize("stop", ["SIGINT", "SIGTERM", "SIGHUP"])
    def test_stopping_an_index_run_stops_what_the_indexer_started(
        self, tmp_path, indexer_tree, stop
    ):
        """The indexer has a session of its own, so neither Ctrl-C nor a
        signal sent to the CLI's process group reaches it."""
        (tmp_path / ".git").mkdir()
        env = {**os.environ, "DESCRY_NO_EMBEDDINGS": "1"}
        env.pop("DESCRY_CACHE_DIR", None)
        index = subprocess.Popen(
            [sys.executable, "-c", _INDEX_WITH, json.dumps(indexer_tree.argv)],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            indexer_tree.wait_started()
            index.send_signal(getattr(signal, stop))
            assert index.wait(timeout=30) == -getattr(signal, stop)
        finally:
            index.kill()
            index.wait()

        assert indexer_tree.grandchild_exited()
