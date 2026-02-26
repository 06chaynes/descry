"""Tests for descry.cli — command-line interface."""

import subprocess


def test_cli_help():
    result = subprocess.run(
        ["python", "-m", "descry.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "descry" in result.stdout.lower()
