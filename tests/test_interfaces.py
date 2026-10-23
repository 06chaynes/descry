"""Each interface, run as a user runs it, answers questions whose answers a
small project fixes: the CLI and the MCP server as processes, the web UI
through its HTTP app."""

import asyncio
import json
import os
import subprocess
import sys

import pytest

from tests.test_git_history import _commit, _init_repo

A = "FILE:a.py::A::parse"
ENV = {**os.environ, "DESCRY_NO_SCIP": "1", "DESCRY_NO_EMBEDDINGS": "1"}


def _a(body):
    return (
        f"class A:\n    def parse(self, text, strict=False):\n        return {body}\n"
    )


def _b(body):
    return f"class B:\n    def parse(self, text):\n        return {body}\n"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Two methods named `parse`, each with one caller and one that may be
    either's, changed by the last commit; and a class that inherits a
    method."""
    repo = _init_repo(tmp_path_factory.mktemp("interfaces") / "repo")
    _commit(
        repo,
        "one",
        **{
            "a.py": _a("text"),
            "b.py": _b(1),
            "base.py": "class Base:\n    def inherited(self):\n        pass\n\n\n"
            "class Derived(Base):\n    def own(self):\n        pass\n",
            "use.py": "from a import A\nfrom b import B\n\n\n"
            "def only_a():\n    A().parse('x')\n\n\n"
            "def only_b():\n    B().parse('y')\n\n\n"
            "def guess(x):\n    x.parse('z')\n",
        },
    )
    _commit(repo, "two", **{"a.py": _a("text.strip()"), "b.py": _b(2)})
    subprocess.run(
        [sys.executable, "-m", "descry.cli", "index"],
        cwd=repo,
        env=ENV,
        check=True,
        capture_output=True,
    )
    return repo


def _cli(project, command, *args):
    return subprocess.run(
        [sys.executable, "-m", "descry.cli", command, *args],
        cwd=project,
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def _mcp(project, tool, **arguments):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def call():
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "descry.mcp_server"],
            cwd=project,
            env=ENV,
        )
        async with (
            stdio_client(server) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(f"descry_{tool}", arguments)
            return "\n".join(c.text for c in result.content)

    return asyncio.run(call())


@pytest.fixture
def web(project, monkeypatch):
    from starlette.testclient import TestClient

    from descry.web import server

    for name in ("_config", "_service"):
        monkeypatch.setattr(server, name, None)
    monkeypatch.chdir(project)
    with TestClient(server.app, base_url="http://127.0.0.1") as client:
        yield lambda path, **params: client.get(path, params=params).text


# (question, CLI arguments, MCP tool and arguments, web path and parameters)
QUESTIONS = {
    "context": (
        ("context", A),
        ("context", {"node_id": A}),
        ("/api/context", {"node_id": A}),
    ),
    "brief": (
        ("context", A, "--brief"),
        ("context", {"node_id": A, "brief": True}),
        ("/api/context", {"node_id": A, "brief": "true"}),
    ),
    "upstream": (
        ("flow", A, "--direction", "backward", "--depth", "1"),
        ("flow", {"start": A, "direction": "backward", "depth": 1}),
        ("/api/flow", {"start": A, "direction": "backward", "depth": 1}),
    ),
    "changes": (
        ("changes", "--commit-range", "HEAD~1..HEAD"),
        ("changes", {"commit_range": "HEAD~1..HEAD"}),
        ("/api/changes", {"commit_range": "HEAD~1..HEAD"}),
    ),
    "signature": (
        ("quick", A),
        ("quick", {"name": A}),
        ("/api/quick", {"name": A}),
    ),
    "flatten": (
        ("flatten", "Derived"),
        ("flatten", {"class_node_id": "Derived"}),
        ("/api/flatten", {"class_node_id": "Derived"}),
    ),
}

# What each answer says, and what it must not
ANSWERS = {
    "context": (
        ["only_a", "`guess` in use.py:13 (20 toks) (one of 2 candidates)"],
        ["only_b"],
    ),
    "brief": (["**Callers:** 1 (+1 possible)"], []),
    "upstream": (["only_a", "guess", "(one of 2 candidates)"], ["only_b"]),
    "changes": (
        [
            "Callers: only_a, guess (one of 2 candidates)",
            "Callers: only_b, guess (one of 2 candidates)",
        ],
        ["only_a, only_b"],
    ),
    "signature": (["def parse(self, text, strict=False)"], ["-> None"]),
    "flatten": (["inherited", "own"], ["ERROR"]),
}


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("interface", ["cli", "mcp", "web"])
def test_each_interface_answers_what_the_project_says(
    project, web, interface, question
):
    cli_args, (tool, arguments), (path, params) = QUESTIONS[question]
    answer = {
        "cli": lambda: _cli(project, *cli_args),
        "mcp": lambda: _mcp(project, tool, **arguments),
        "web": lambda: web(path, **params),
    }[interface]()
    says, never = ANSWERS[question]
    for text in says:
        assert text in answer
    for text in never:
        assert text not in answer


def test_the_page_is_given_the_graphs_languages(web):
    assert json.loads(web("/api/health"))["graph"]["languages"] == ["python"]


@pytest.mark.parametrize("interface", ["cli", "mcp"])
def test_impls_says_where_it_applies(project, interface):
    answer = (
        _cli(project, "impls", "parse")
        if interface == "cli"
        else _mcp(project, "impls", method="parse")
    )
    assert "recorded for Rust code only" in answer


@pytest.mark.parametrize("interface", ["cli", "mcp"])
def test_callees_say_what_they_leave_out_and_which_they_answer(project, interface):
    ask = {
        "cli": lambda *a: _cli(project, "callees", a[0], "--limit", str(a[1])),
        "mcp": lambda *a: _mcp(project, "callees", name=a[0], limit=a[1]),
    }[interface]
    assert "(1 more callees, limit 1)" in ask("only_a", 1)
    assert "1 other definition(s) of 'parse'" in ask("parse", 5)


def test_the_web_gives_the_callees_it_leaves_out(web):
    answer = json.loads(web("/api/callees", name="only_a", limit=1))
    assert (answer["total"], len(answer["callees"])) == (2, 1)


def test_the_count_is_of_the_callers_resolved_to_it(project, web):
    """`guess` may call it but is not counted."""
    assert "(20 toks, 1 caller)" in _cli(project, "quick", A)
    assert "(20 toks, 1 caller)" in _mcp(project, "quick", name=A)
    assert json.loads(web("/api/quick", name=A))["symbol"]["in_degree"] == 1


def test_callers_say_how_many_are_possible(project, web):
    said = f"1 caller(s) of '{A}' (+1 possible):"
    assert said in _cli(project, "callers", A)
    assert said in _mcp(project, "callers", name=A)
    answer = json.loads(web("/api/callers", name=A))
    assert (answer["total"], answer["possible"]) == (2, 1)


def test_a_change_marks_its_possible_callers_on_the_page(web):
    """The page renders the structured form: each caller carries its
    candidates, and the change how many of its callers only may be."""
    data = json.loads(
        web("/api/changes", commit_range="HEAD~1..HEAD", format="structured")
    )
    symbols = {s["id"]: s for s in data["data"]["symbols"]}
    callers = {c["name"]: c["candidates"] for c in symbols[A]["callers"]}
    assert callers == {"only_a": None, "guess": 2}
    assert symbols[A]["callers_possible"] == 1


@pytest.mark.parametrize("interface", ["cli", "mcp", "web"])
def test_a_depth_below_one_is_refused(project, web, interface):
    answer = {
        "cli": lambda: (
            _cli(project, "flow", A, "--depth", "0")
            + _cli(project, "path", "only_a", A, "--max-depth", "0")
        ),
        "mcp": lambda: (
            _mcp(project, "flow", start=A, depth=0)
            + _mcp(project, "path", start="only_a", end=A, max_depth=0)
        ),
        "web": lambda: (
            web("/api/flow", start=A, depth=0)
            + web("/api/path", start="only_a", end=A, max_depth=0)
        ),
    }[interface]()
    assert "depth must be at least 1, got 0." in answer
    assert "max_depth must be at least 1, got 0." in answer


def test_a_depth_past_the_ceiling_says_so(project, web):
    assert "is past `[query] max_depth`; traced to 3." in _mcp(
        project, "flow", start=A, direction="backward", depth=9
    )
    assert json.loads(web("/api/flow", start=A, depth=9))["depth"] == 3


def test_what_is_left_out_and_what_only_may_be_is_said(project):
    assert "(1 more possible callers, limit 1)" in _mcp(
        project, "callers", name=A, limit=1
    )
    assert "**Callees:** 0 (+1 possible)" in _mcp(
        project, "context", node_id="FILE:use.py::guess", brief=True
    )


@pytest.mark.parametrize("interface", ["cli", "mcp", "web"])
def test_context_depth_is_bounded_as_flow_is(project, web, interface):
    refused, capped = {
        "cli": lambda: (
            _cli(project, "context", A, "--depth", "0"),
            _cli(project, "context", A, "--depth", "9"),
        ),
        "mcp": lambda: (
            _mcp(project, "context", node_id=A, depth=0),
            _mcp(project, "context", node_id=A, depth=9),
        ),
        "web": lambda: (
            web("/api/context", node_id=A, depth=0),
            web("/api/context", node_id=A, depth=9),
        ),
    }[interface]()
    assert "depth must be at least 1, got 0." in refused
    assert "[depth=3] [depth 9 is past [query] max_depth]" in capped
