"""Tests for descry.cli — command-line interface."""

import subprocess
import sys

import pytest


class TestCliHelp:
    """All subcommands have working --help."""

    COMMANDS = [
        [],
        ["health"],
        ["status"],
        ["ensure"],
        ["index"],
        ["search", "--help"],
        ["callers", "--help"],
        ["callees", "--help"],
        ["context", "--help"],
        ["flow", "--help"],
        ["quick", "--help"],
        ["structure", "--help"],
        ["flatten", "--help"],
        ["semantic", "--help"],
        ["impls", "--help"],
        ["path", "--help"],
        ["cross-lang", "--help"],
        ["churn", "--help"],
        ["evolution", "--help"],
        ["changes", "--help"],
    ]

    @pytest.mark.parametrize("args", COMMANDS, ids=lambda a: a[0] if a else "root")
    def test_help(self, args):
        cmd = [sys.executable, "-m", "descry.cli"] + (
            ["--help"] if len(args) <= 1 else args
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "descry" in result.stdout.lower() or "usage" in result.stdout.lower()


class TestCliNoCommand:
    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "descry.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1


class TestCliHealth:
    def test_health_returns_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "descry.cli", "health"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert '"version"' in result.stdout
        assert '"status"' in result.stdout
