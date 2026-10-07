"""Tests for descry.cli — command-line interface."""

import subprocess
import sys
from typing import ClassVar

import pytest

from descry import cli
from descry.handlers import ServiceError


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
