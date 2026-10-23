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
    return f"class B:\n    def parse(self):\n        return {body}\n"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Two methods named `parse`, each with one caller, changed by the last
    commit; and a class that inherits a method."""
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
            "def only_b():\n    B().parse()\n",
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
        ("flow", A, "--direction", "backward"),
        ("flow", {"start": A, "direction": "backward"}),
        ("/api/flow", {"start": A, "direction": "backward"}),
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
    "context": (["only_a"], ["only_b"]),
    "brief": (["**Callers:** 1 "], []),
    "upstream": (["only_a"], ["only_b"]),
    "changes": (["Callers: only_a", "Callers: only_b"], ["only_a, only_b"]),
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
