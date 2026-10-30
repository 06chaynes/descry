"""Tests for descry.cli — command-line interface."""

import argparse
import json
import os
import signal
import subprocess
import sys
from typing import ClassVar

import pytest

from descry import cli
from descry._answers import fail
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


class TestCliArguments:
    @pytest.mark.parametrize(
        "args", [["callers", "x", "--limit", "ten"], ["search", "x", "--frob", "1"]]
    )
    def test_an_argument_it_cannot_read_answers_invalid_with_json(self, args):
        """With `--json` an argument the parser cannot read is answered as
        every failure is; without it, the usage text says what is wrong."""
        command = [sys.executable, "-m", "descry.cli", *args]
        found = subprocess.run(
            [*command, "--json"], capture_output=True, text=True, check=False
        )
        assert found.returncode == 2
        assert json.loads(found.stdout)["reason"] == "invalid"
        plain = subprocess.run(command, capture_output=True, text=True, check=False)
        assert plain.returncode == 2
        assert "usage:" in plain.stderr


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

    def test_health_takes_json_as_every_tool_does(self):
        assert cli.build_parser().parse_args(["health", "--json"]).json

    @pytest.mark.parametrize(
        ("argv", "flag"),
        [(["search", "x"], "compact"), (["changes"], "show_callers")],
    )
    def test_on_by_default_flags_turn_off_with_no(self, argv, flag):
        parse = cli.build_parser().parse_args
        option = "--no-" + flag.replace("_", "-")
        assert getattr(parse(argv), flag) is True
        assert getattr(parse([*argv, option]), flag) is False


class TestCliExitStatus:
    @staticmethod
    def _print(answer, text, as_json=False):
        """`_print_answer` of ``answer``, whose text is ``text``."""

        async def given():
            return answer

        async def render(_answer):
            return text

        cli._print_answer(argparse.Namespace(json=as_json), given(), render)

    @pytest.mark.parametrize("as_json", [False, True])
    def test_a_failed_answer_exits_nonzero(self, as_json, capsys):
        failure = "Index timed out after 30 minutes."
        with pytest.raises(SystemExit) as exit_info:
            self._print(fail("failed", failure), ServiceError(failure), as_json)
        assert exit_info.value.code == 2
        assert failure in capsys.readouterr().out

    @pytest.mark.parametrize("as_json", [False, True])
    def test_an_answer_exits_zero(self, as_json, capsys):
        self._print(
            {"ok": True, "message": "No matches"}, "No matches for 'x'.", as_json
        )
        assert "No matches" in capsys.readouterr().out

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
            async def ensure_answer(self, max_age_hours):
                seen["max_age_hours"] = max_age_hours
                return {"ok": True, "summary": "Graph ready", "_report": None}

            async def ensure_text(self, answer):
                return answer["summary"]

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
        monkeypatch.setenv("DESCRY_NO_EMBEDDINGS", "1")
        monkeypatch.setattr(sys, "argv", ["descry", "callers", "f", "--limit", "-1"])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 2
        assert "limit must be at least 1" in capsys.readouterr().out


# `descry index` in a project whose indexer is the stand-in named by argv[1].
# `descry index` with the stand-in indexer `sys.argv[1]` gives, its signals
# handled as a foreground shell leaves them but for those named after it,
# which are ignored.
_INDEX_WITH = """
import json, os, signal, sys
from descry import cli
from descry.handlers import DescryService

argv = json.loads(sys.argv[1])
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGHUP, signal.SIG_DFL)
for name in sys.argv[2:]:
    signal.signal(getattr(signal, name), signal.SIG_IGN)
DescryService._indexer_child = lambda self, _path: (argv, dict(os.environ))
sys.argv = ["descry", "index"]
cli.main()
"""


class TestCliStopped:
    @pytest.fixture
    def index(self, tmp_path, indexer_tree, request):
        """`descry index` running the stand-in indexer, ignoring the signals
        the test is parametrized with."""
        (tmp_path / ".git").mkdir()
        env = {**os.environ, "DESCRY_NO_EMBEDDINGS": "1"}
        script = [sys.executable, "-c", _INDEX_WITH, json.dumps(indexer_tree.argv)]
        index = subprocess.Popen(
            [*script, *getattr(request, "param", ())],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        indexer_tree.wait_started()
        yield index
        index.kill()
        index.wait()

    @pytest.mark.parametrize("stop", ["SIGINT", "SIGTERM", "SIGHUP"])
    def test_stopping_an_index_run_stops_what_the_indexer_started(
        self, index, indexer_tree, stop
    ):
        """The indexer has a session of its own, so neither Ctrl-C nor a
        signal sent to the CLI's process group reaches it."""
        index.send_signal(getattr(signal, stop))
        assert index.wait(timeout=30) == -getattr(signal, stop)
        assert indexer_tree.grandchild_exited()

    @pytest.mark.parametrize("index", [("SIGHUP",)], indirect=True)
    def test_a_signal_the_run_ignores_does_not_stop_it(self, index, indexer_tree):
        """Under `nohup` a hangup leaves the indexer running."""
        index.send_signal(signal.SIGHUP)
        assert not indexer_tree.grandchild_exited(timeout=1)
        assert index.poll() is None


class TestSearchType:
    """`--type` takes the types descry has, as every interface writes them,
    and refuses others."""

    @pytest.mark.parametrize("written", ["struct", "Function"])
    def test_an_unknown_type_is_refused(self, monkeypatch, capsys, written):
        monkeypatch.setattr(sys, "argv", ["descry", "search", "x", "--type", written])
        with pytest.raises(SystemExit) as exit_:
            cli.main()
        assert exit_.value.code == 2
        assert f"invalid choice: '{written}'" in capsys.readouterr().err
