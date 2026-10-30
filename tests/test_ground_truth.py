"""Answers written down by hand, per language.

Each directory under `tests/ground_truth/` is a small project and its
`expected.json`: the callers and callees of symbols by kind, calls that
leave the project, the methods a class flattens to, trait implementations,
call paths, and names several definitions share. The project is indexed as
a user would index it, and every answer is checked; counts, `limit` and
`flow` are checked against the same callers and callees.
"""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from descry.handlers import DescryConfig, DescryService

ROOT = Path(__file__).parent / "ground_truth"
LANGUAGES = sorted(p.name for p in ROOT.iterdir() if p.is_dir())


def _expected(language):
    return json.loads((ROOT / language / "expected.json").read_text())


def _cases(section):
    return [
        pytest.param(language, item, id=f"{language}:{item}")
        for language in LANGUAGES
        for item in (
            _expected(language).get(section, {})
            if section != "path"
            else range(len(_expected(language).get("path", [])))
        )
    ]


@pytest.fixture(scope="module")
def services(tmp_path_factory):
    """An indexed service per language, built once."""
    built = {}

    def get(language):
        if language not in built:
            root = tmp_path_factory.mktemp(language)
            shutil.copytree(ROOT / language, root, dirs_exist_ok=True)
            (root / "expected.json").unlink()
            svc = DescryService(
                DescryConfig(
                    project_root=root, enable_scip=False, enable_embeddings=False
                )
            )
            asyncio.run(svc.index())
            built[language] = svc
        return built[language]

    return get


@pytest.mark.parametrize(("language", "symbol"), _cases("callers"))
def test_callers(services, language, symbol):
    answer = asyncio.run(services(language).callers_answer(symbol, limit=100))
    expected = _expected(language)["callers"][symbol]
    found = {
        kind: sorted(c["id"] for c in answer["callers"] if c["kind"] == kind)
        for kind in expected
    }
    assert found == {kind: sorted(ids) for kind, ids in expected.items()}
    assert (answer["certain"], answer["possible"]) == (
        len(expected["resolved"]),
        len(expected.get("candidate", [])),
    )
    # A limit of one says how many it leaves out
    cut = asyncio.run(services(language).callers_answer(symbol, limit=1))
    assert sum(cut["more"].values()) == max(answer["total"] - 1, 0)


@pytest.mark.parametrize(("language", "symbol"), _cases("callees"))
def test_callees(services, language, symbol):
    answer = asyncio.run(services(language).callees_answer(symbol, limit=100))
    expected = _expected(language)["callees"][symbol]
    # Resolved callees by node ID; the others by the name the call is written with
    found = {
        kind: sorted(
            c["id"] if kind == "resolved" else c["name"]
            for c in answer["callees"]
            if c["kind"] == kind
        )
        for kind in expected
    }
    assert found == {kind: sorted(names) for kind, names in expected.items()}
    counts = {
        "resolved": "certain",
        "candidate": "possible",
        "unresolved": "unresolved",
    }
    for kind, names in expected.items():
        assert answer[counts[kind]] == len(names), kind
    assert answer["total"] == sum(len(names) for names in expected.values())


@pytest.mark.parametrize(("language", "symbol"), _cases("callees"))
def test_flow_reaches_the_callees(services, language, symbol):
    answer = asyncio.run(services(language).flow_answer(symbol, depth=1))
    reached = {child["id"] for child in answer["root"]["children"]}
    assert set(_expected(language)["callees"][symbol].get("resolved", [])) <= reached


@pytest.mark.parametrize(("language", "symbol"), _cases("external"))
def test_external_calls_are_not_callees(services, language, symbol):
    answer = asyncio.run(services(language).callees_answer(symbol, limit=100))
    names = {c["name"] for c in answer["callees"]} | {
        c["id"] for c in answer["callees"]
    }
    assert not names & set(_expected(language)["external"][symbol])


@pytest.mark.parametrize(("language", "method"), _cases("impls"))
def test_impls(services, language, method):
    answer = asyncio.run(services(language).impls_answer(method))
    assert sorted(i["id"] for i in answer["implementations"]) == sorted(
        _expected(language)["impls"][method]
    )


@pytest.mark.parametrize(("language", "symbol"), _cases("flatten"))
def test_flatten(services, language, symbol):
    answer = asyncio.run(services(language).flatten_answer(symbol))
    assert [m["name"] for m in answer["methods"]] == _expected(language)["flatten"][
        symbol
    ]


@pytest.mark.parametrize(("language", "index"), _cases("path"))
def test_path(services, language, index):
    start, end, hops = _expected(language)["path"][index]
    answer = asyncio.run(services(language).path_answer(start, end))
    assert answer["status"] == "found"
    assert [hop["callee"]["id"] for hop in answer["hops"]] == hops


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_shared_name_is_refused(services, language):
    for name in _expected(language).get("refused", []):
        answer = asyncio.run(services(language).callers_answer(name))
        assert answer.get("reason") == "ambiguous", name


@pytest.mark.parametrize("language", LANGUAGES)
def test_bounds(services, language):
    """Every tool that takes a limit or a depth refuses 0, and a depth past
    the ceiling is traced to the ceiling."""
    svc = services(language)
    symbol = next(iter(_expected(language)["callers"]))
    for answer in (
        svc.callers_answer(symbol, limit=0),
        svc.callees_answer(symbol, limit=0),
        svc.search_answer(["parse"], limit=0),
        svc.semantic_answer("parse", limit=0),
        svc.flow_answer(symbol, depth=0),
        svc.path_answer(symbol, symbol, max_depth=0),
        svc.context_answer(symbol, depth=0),
        svc.churn_answer(limit=0),
        svc.changes_answer(limit=0),
        svc.evolution_answer(symbol, limit=0),
    ):
        assert asyncio.run(answer)["reason"] == "invalid"
    ceiling = svc.config.max_depth
    assert asyncio.run(svc.flow_answer(symbol, depth=ceiling + 5))["depth"] == ceiling
