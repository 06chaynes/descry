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
        # Use a name that's not in STDLIB_FILTER ('start' matches Python's
        # re.Match.start() and is intentionally filtered cross-language).
        src = """
void run(Server* srv) {
    srv->launchWorker();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert any("launchWorker" in t for t in targets)

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


class TestDeclarationsAndInlineComments:
    def test_definition_is_not_a_call(self, builder):
        """Defining a function does not call it."""
        src = """
int compute(int x) {
    return scale(x);
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:scale"}

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
void run() {
    stepOne(); /* hidden(1);
    also_hidden(2); */ stepTwo();
    /* inline(3); */ stepThree();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:stepOne", "REF:stepTwo", "REF:stepThree"}

    def test_macro_line_numbers(self, builder):
        src = """
#define FIRST 1
int x;
#define SECOND 2
""".strip()
        _parse(builder, src)
        linenos = {
            n["metadata"]["name"]: n["metadata"]["lineno"]
            for n in builder.nodes
            if n["metadata"].get("kind") == "macro"
        }
        assert linenos == {"FIRST": 1, "SECOND": 3}


def _types(builder):
    return {n["id"]: n["type"] for n in builder.nodes}


class TestTypeNamesThatDefineNothing:
    def test_forward_declaration_gets_no_node_and_no_body(self, builder):
        """`class Widget;` ends before any `{`: the function after it keeps
        its own node and the call in its body."""
        src = """
class Widget;
struct Opaque;
void run() {
    start();
}
""".strip()
        _parse(builder, src)
        assert _types(builder) == {
            "FILE:src/foo.cpp": "File",
            "FILE:src/foo.cpp::run": "Function",
        }
        calls = [
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        assert calls == [("FILE:src/foo.cpp", "REF:start")]

    def test_definition_after_its_forward_declaration_keeps_its_span(self, builder):
        src = """
class Widget;

class Widget {
    void draw();
};
""".strip()
        _parse(builder, src)
        (widget,) = [n["metadata"] for n in builder.nodes if n["type"] == "Class"]
        assert (widget["lineno"], widget["end_lineno"]) == (3, 5)

    def test_struct_typed_variable_is_not_a_class(self, builder):
        """A variable declared with `struct` is a use of the type; the calls
        around it are still recorded."""
        src = """
int size(void) {
    struct stat st;
    if (probe(&st)) {
        return 1;
    }
    struct point origin = {0, 0};
    return finish(origin);
}
""".strip()
        _parse(builder, src)
        assert _types(builder) == {
            "FILE:src/foo.cpp": "File",
            "FILE:src/foo.cpp::size": "Function",
        }
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:probe", "REF:finish"}

    def test_struct_return_type_and_parameter_are_not_classes(self, builder):
        src = """
struct node *find(struct list *items) {
    return lookup(items);
}

int scan(int fd,
         struct stat *st) {
    return check(fd, st);
}
""".strip()
        _parse(builder, src)
        types = _types(builder)
        assert "Class" not in types.values()
        assert types["FILE:src/foo.cpp::find"] == "Function"

    def test_definition_heads_are_classes(self, builder):
        src = """
class Shape final : public Base {
};
template <> struct hash<Key *> {
};
struct Point { int x; int y; };
enum class Color : int { Red };
""".strip()
        _parse(builder, src)
        names = [n["metadata"]["name"] for n in builder.nodes if n["type"] == "Class"]
        assert names == ["Shape", "hash", "Point", "Color"]

    def test_brace_in_a_comment_does_not_open_the_class(self, builder):
        """The class closes at its own `}`; without a `{` of its own it has
        no body."""
        src = """
class Gap // {
{
    void m();
};
class Next {
};
class Tail // {
""".strip()
        _parse(builder, src)
        spans = {
            n["id"]: (n["metadata"]["lineno"], n["metadata"]["end_lineno"])
            for n in builder.nodes
            if n["type"] == "Class"
        }
        assert spans == {
            "FILE:src/foo.cpp::Gap": (1, 4),
            "FILE:src/foo.cpp::Next": (5, 6),
            "FILE:src/foo.cpp::Tail": (7, 7),
        }
