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


def _mcp_result(project, tool, **arguments):
    """The CallToolResult of one MCP tool call, over stdio."""
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
            return await session.call_tool(f"descry_{tool}", arguments)

    return asyncio.run(call())


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
        ("context", {"symbol": A}),
        ("/api/context", {"symbol": A}),
    ),
    "brief": (
        ("context", A, "--brief"),
        ("context", {"symbol": A, "brief": True}),
        ("/api/context", {"symbol": A, "brief": "true"}),
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
        ("quick", {"symbol": A}),
        ("/api/quick", {"symbol": A}),
    ),
    "flatten": (
        ("flatten", "Derived"),
        ("flatten", {"symbol": "Derived"}),
        ("/api/flatten", {"symbol": "Derived"}),
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
def test_callees_say_what_they_leave_out_and_refuse_a_shared_name(project, interface):
    ask = {
        "cli": lambda *a: _cli(project, "callees", a[0], "--limit", str(a[1])),
        "mcp": lambda *a: _mcp(project, "callees", symbol=a[0], limit=a[1]),
    }[interface]
    assert "(1 more callees, limit 1)" in ask("only_a", 1)
    refused = ask("parse", 5)
    assert "'parse' names 2 definitions; give one by node ID:" in refused
    assert "FILE:a.py::A::parse" in refused and "FILE:b.py::B::parse" in refused


def test_the_web_gives_the_callees_it_leaves_out(web):
    answer = json.loads(web("/api/callees", symbol="only_a", limit=1))
    assert (answer["total"], len(answer["callees"])) == (2, 1)


def test_the_count_is_of_the_callers_resolved_to_it(project, web):
    """`guess` may call it but is not counted."""
    assert "(20 toks, 1 caller)" in _cli(project, "quick", A)
    assert "(20 toks, 1 caller)" in _mcp(project, "quick", symbol=A)
    assert json.loads(web("/api/quick", symbol=A))["symbol"]["callers"] == 1


def test_callers_say_how_many_are_possible(project, web):
    said = f"1 caller(s) of '{A}' (+1 possible):"
    assert said in _cli(project, "callers", A)
    assert said in _mcp(project, "callers", symbol=A)
    answer = json.loads(web("/api/callers", symbol=A, limit=1))
    assert (answer["total"], answer["certain"], answer["possible"]) == (2, 1, 1)
    assert answer["more"] == {"certain": 0, "possible": 1, "unresolved": 0}


def test_a_change_marks_its_possible_callers_on_the_page(web):
    """The page renders the structured form: each caller carries its
    candidates, and the change how many of its callers only may be."""
    data = json.loads(web("/api/changes", commit_range="HEAD~1..HEAD"))
    symbols = {s["id"]: s for s in data["symbols"]}
    callers = {c["name"]: c.get("candidates") for c in symbols[A]["callers"]}
    assert callers == {"only_a": None, "guess": 2}
    assert all(c["file"] and c["line"] for c in symbols[A]["callers"])
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
        project, "callers", symbol=A, limit=1
    )
    assert "**Callees:** 0 (+1 possible)" in _mcp(
        project, "context", symbol="FILE:use.py::guess", brief=True
    )


@pytest.mark.parametrize("interface", ["cli", "mcp", "web"])
def test_context_depth_is_bounded_as_flow_is(project, web, interface):
    refused, capped = {
        "cli": lambda: (
            _cli(project, "context", A, "--depth", "0"),
            _cli(project, "context", A, "--depth", "9"),
        ),
        "mcp": lambda: (
            _mcp(project, "context", symbol=A, depth=0),
            _mcp(project, "context", symbol=A, depth=9),
        ),
        "web": lambda: (
            web("/api/context", symbol=A, depth=0),
            web("/api/context", symbol=A, depth=9),
        ),
    }[interface]()
    assert "depth must be at least 1, got 0." in refused
    assert "[depth=3] [depth 9 is past [query] max_depth]" in capped


def test_a_possible_callee_is_named_and_followed(project):
    flow = _mcp(project, "flow", start="FILE:use.py::guess", depth=1)
    assert flow.count("(one of 2 candidates)") == 2
    context = _mcp(project, "context", symbol="FILE:use.py::guess")
    assert "`parse` (one of 2 candidates)" in context
    assert "external" not in context.lower()


@pytest.mark.parametrize(
    ("cli_args", "mcp_call", "web_call"),
    [
        (
            ("callers", "parse"),
            ("callers", {"symbol": "parse"}),
            ("/api/callers", {"symbol": "parse"}),
        ),
        (
            ("callees", "parse"),
            ("callees", {"symbol": "parse"}),
            ("/api/callees", {"symbol": "parse"}),
        ),
        (
            ("context", "parse"),
            ("context", {"symbol": "parse"}),
            ("/api/context", {"symbol": "parse"}),
        ),
        (
            ("quick", "parse"),
            ("quick", {"symbol": "parse"}),
            ("/api/quick", {"symbol": "parse"}),
        ),
        (
            ("flow", "parse"),
            ("flow", {"start": "parse"}),
            ("/api/flow", {"start": "parse"}),
        ),
        (
            ("path", "only_a", "parse"),
            ("path", {"start": "only_a", "end": "parse"}),
            ("/api/path", {"start": "only_a", "end": "parse"}),
        ),
    ],
)
def test_a_name_two_definitions_share_is_refused_with_both(
    project, web, cli_args, mcp_call, web_call
):
    for answer in (
        _cli(project, *cli_args),
        _mcp(project, mcp_call[0], **mcp_call[1]),
        web(*web_call[:1], **web_call[1]),
    ):
        assert "'parse' names 2 definitions; give one by node ID:" in answer
        assert "FILE:a.py::A::parse" in answer and "FILE:b.py::B::parse" in answer


def test_mcp_gives_callers_as_structure_without_prose(project):
    """The structured answer has IDs, places, kinds and counts; nothing a
    repository wrote reaches the client outside the fence."""
    result = _mcp_result(project, "callers", symbol=A)
    answer = result.structured_content
    assert (answer["certain"], answer["possible"], answer["unresolved"]) == (1, 1, 0)
    assert [(c["name"], c["kind"]) for c in answer["callers"]] == [
        ("only_a", "resolved"),
        ("guess", "candidate"),
    ]
    assert set(answer["callers"][0]) <= {
        "id",
        "name",
        "file",
        "line",
        "kind",
        "candidates",
    }
    assert result.content[0].text.startswith("<descry:repo_content")


def test_mcp_gives_a_path_as_structure_without_prose(project):
    """Each hop is the caller, the callee and the call's place; the code at
    the call is only in the fenced text."""
    result = _mcp_result(project, "path", start="only_a", end=A)
    answer = result.structured_content
    assert answer["status"] == "found"
    assert [set(hop) for hop in answer["hops"]] == [
        {"caller", "callee", "file", "line"}
    ]
    assert answer["hops"][0]["callee"]["id"] == A
    assert "```" in result.content[0].text


@pytest.mark.parametrize(
    ("tool", "arguments", "key"),
    [
        ("search", {"terms": ["parse"]}, "results"),
        ("quick", {"symbol": A}, "symbol"),
    ],
)
def test_mcp_gives_definitions_as_structure_without_prose(
    project, tool, arguments, key
):
    found = _mcp_result(project, tool, **arguments).structured_content[key]
    for ref in found if isinstance(found, list) else [found]:
        assert set(ref) == {
            "id",
            "type",
            "name",
            "file",
            "line",
            "tokens",
            "callers",
        }


@pytest.mark.parametrize(
    ("cli_args", "tool", "arguments"),
    [
        (("callers", A), "callers", {"symbol": A}),
        (("path", "only_a", A), "path", {"start": "only_a", "end": A}),
        (("search", "parse"), "search", {"terms": ["parse"]}),
        (("context", A, "--depth", "2"), "context", {"symbol": A, "depth": 2}),
    ],
)
def test_the_cli_json_is_the_mcp_structured_answer(project, cli_args, tool, arguments):
    cli = json.loads(_cli(project, *cli_args, "--json"))
    assert cli == _mcp_result(project, tool, **arguments).structured_content


def test_mcp_gives_a_context_as_structure_without_source(project):
    result = _mcp_result(project, "context", symbol=A)
    answer = result.structured_content
    assert answer["symbol"]["id"] == A
    assert [(c["name"], c["kind"]) for c in answer["callers"]] == [
        ("only_a", "resolved"),
        ("guess", "candidate"),
    ]
    assert [e["file"] for e in answer["examples"]] == ["use.py", "use.py"]
    assert "strip" not in json.dumps(answer)
    assert "strip" in result.content[0].text


def test_mcp_gives_a_flow_tree_with_kinds_and_no_code(project):
    result = _mcp_result(project, "flow", start=A, direction="backward", depth=1)
    root = result.structured_content["root"]
    assert {(c["name"], c["kind"]) for c in root["children"]} == {
        ("only_a", "resolved"),
        ("guess", "candidate"),
    }
    assert "code" not in json.dumps(result.structured_content)


# What each tool's structured answer holds (CONTRACT.md); `None` marks a
# failure here: semantic search and cross-language tracing are off.
CONTRACT = {
    "callers": (
        {"symbol": A},
        "symbol total certain possible unresolved more callers",
    ),
    "callees": (
        {"symbol": "only_a"},
        "symbol total certain possible unresolved more callees",
    ),
    "context": (
        {"symbol": A},
        "symbol brief depth callers callees callers_more callees_more examples members tests configs omitted truncated",
    ),
    "flow": ({"start": "only_a"}, "start direction asked depth target root"),
    "path": (
        {"start": "only_a", "end": A},
        "status start end direction max_depth hops",
    ),
    "quick": ({"symbol": "only_a"}, "symbol"),
    "search": ({"terms": ["parse"]}, "method query total results"),
    "semantic": ({"query": "parse"}, None),
    "structure": ({"filename": "a.py"}, "found file imports definitions"),
    "flatten": ({"symbol": "Derived"}, "class name note chain methods"),
    "impls": ({"method": "parse"}, "method trait total implementations"),
    "cross_lang": ({"mode": "stats"}, None),
    "churn": ({}, "mode time_range path_filter symbols"),
    "changes": (
        {"commit_range": "HEAD~1..HEAD"},
        "range commits files_count symbols_count symbols",
    ),
    "evolution": ({"symbol": "only_a"}, "node_id mode authors commits"),
    "status": (
        {},
        "exists age_str age_hours nodes edges languages stale error scip embeddings",
    ),
    "health": (
        {},
        "status version project_root cache_dir warm max_depth features graph",
    ),
    "ensure": ({}, "action summary graph"),
}


# The answers of the tools the MCP test environment cannot answer (no model,
# no spec), where another environment can
ANSWERED_ELSEWHERE = {
    "semantic": "query total results",
    "cross_lang": "mode total_endpoints linked_to_graph",
}

# What the web adds for its page: a tool's text, the code a flow tree inlines,
# and the spec's summaries of the endpoints cross_lang names
WEB_ADDITIONS = {"markdown", "tree", "summaries"}


@pytest.fixture
def web_answer(project, monkeypatch):
    from starlette.testclient import TestClient

    from descry.web import server

    for name in ("_config", "_service"):
        monkeypatch.setattr(server, name, None)
    monkeypatch.chdir(project)

    def answer(tool, arguments):
        path = f"/api/{tool.replace('_', '-')}"
        if tool in ("ensure", "index"):
            return client.post(path, json=arguments).json()
        params = {
            k: ",".join(v)
            if isinstance(v, list)
            else str(v).lower()
            if isinstance(v, bool)
            else v
            for k, v in arguments.items()
        }
        return client.get(path, params=params).json()

    with TestClient(server.app, base_url="http://127.0.0.1") as client:
        yield answer


@pytest.mark.parametrize("tool", sorted(CONTRACT))
def test_each_web_answer_holds_what_the_contract_lists(web_answer, tool):
    """The web answers each tool as the contract does, with only what its
    page shows besides."""
    arguments, fields = CONTRACT[tool]
    body = web_answer(tool, arguments)
    if fields is None and "error" in body:
        assert set(body) == {"error", "reason"}
    else:
        expected = fields or ANSWERED_ELSEWHERE[tool]
        assert set(body) - WEB_ADDITIONS == set(expected.split())


@pytest.mark.parametrize("tool", sorted(CONTRACT))
def test_each_mcp_answer_holds_what_the_contract_lists(project, tool):
    arguments, fields = CONTRACT[tool]
    result = _mcp_result(project, tool, **arguments)
    if fields is None:
        assert result.is_error
        assert set(result.structured_content) == {"error", "reason"}
    else:
        assert not result.is_error
        assert set(result.structured_content) == set(fields.split())


def test_a_refusal_has_the_contracts_shape(project):
    result = _mcp_result(project, "callers", symbol="parse")
    assert result.is_error
    answer = result.structured_content
    assert (answer["reason"], set(answer)) == (
        "ambiguous",
        {"error", "reason", "choices"},
    )
    assert set(answer["choices"]) == {A, "FILE:b.py::B::parse"}
