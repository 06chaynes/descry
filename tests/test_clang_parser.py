"""Unit tests for the C/C++ regex parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.clang_parser import ClangParser
from descry.generate import CodeGraphBuilder


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="src/foo.cpp"):
    ClangParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestIncludesAndDefines:
    def test_include_angle(self, builder):
        src = """
#include <stdio.h>
#include <vector>
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:stdio.h" in targets
        assert "MODULE:vector" in targets

    def test_include_quoted(self, builder):
        src = """
#include "myheader.h"
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:myheader.h" in targets

    def test_define_macro(self, builder):
        src = """
#define MAX_CONN 100
#define VERSION "1.0"
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/foo.cpp::MAX_CONN" in ids
        assert "FILE:src/foo.cpp::VERSION" in ids


class TestTypesAndNamespaces:
    def test_namespace(self, builder):
        src = """
namespace app {
    int x = 0;
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::app" in _ids(builder)

    def test_class_allman(self, builder):
        src = """
class Server
{
public:
    void start();
};
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Server" in _ids(builder)

    def test_struct(self, builder):
        src = """
struct Point { int x; int y; };
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Point" in _ids(builder)

    def test_enum_class(self, builder):
        src = """
enum class Color { Red, Green, Blue };
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Color" in _ids(builder)


class TestFunctions:
    # Function / method extraction is intentionally deferred to scip-clang
    # (see ClangParser module docstring for why). Only calls are extracted.

    def test_control_flow_not_a_function(self, builder):
        src = """
int go() {
    if (ready) { tick(); }
    while (running) { work(); }
    return 0;
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:while" not in targets
        assert "REF:tick" in targets
        assert "REF:work" in targets


class TestCallsAndComments:
    def test_call_emitted(self, builder):
        src = """
void run() {
    doStuff();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:doStuff" in targets

    def test_method_call(self, builder):
        src = """
void run(Server* srv) {
    srv->start();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert any("start" in t for t in targets)

    def test_line_comment_stripped(self, builder):
        src = """
void run() {
    // fake(x);
    real();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets

    def test_block_comment_stripped(self, builder):
        src = """
/*
 * fake(x) inside a block comment
 */
void run() {
    real();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets
