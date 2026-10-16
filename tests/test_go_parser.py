"""Unit tests for the Go regex-based baseline parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.generate import CodeGraphBuilder
from descry.go_parser import GoParser


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="pkg/app.go"):
    GoParser(builder).parse(Path(rel), rel, source)
    return builder


def _node_ids(builder):
    return [n["id"] for n in builder.nodes]


class TestPackageAndImports:
    def test_package_metadata(self, builder):
        src = """
package myapp

func main() {}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:pkg/app.go")
        assert file_node.get("metadata", {}).get("go_package") == "myapp"

    def test_single_import(self, builder):
        src = """
package myapp
import "fmt"
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:fmt" in targets

    def test_grouped_imports(self, builder):
        src = """
package myapp

import (
    "fmt"
    "net/http"
    "github.com/example/lib"
)
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:fmt" in targets
        assert "MODULE:net/http" in targets
        assert "MODULE:github.com/example/lib" in targets

    def test_aliased_import(self, builder):
        src = """
package myapp
import alias "github.com/example/lib"
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:github.com/example/lib" in targets

    def test_grouped_aliased_and_underscore(self, builder):
        src = """
package myapp

import (
    _ "net/http/pprof"
    myfmt "fmt"
)
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:net/http/pprof" in targets
        assert "MODULE:fmt" in targets


class TestTypeAndFuncExtraction:
    def test_struct(self, builder):
        src = """
package app

type User struct {
    Name string
    Age  int
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::User" in _node_ids(builder)

    def test_interface(self, builder):
        src = """
package app

type Handler interface {
    Handle()
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::Handler" in _node_ids(builder)

    def test_type_alias(self, builder):
        src = """
package app

type ID = string
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::ID" in _node_ids(builder)

    def test_free_function(self, builder):
        src = """
package app

func DoStuff(x int) int {
    return x + 1
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::DoStuff" in _node_ids(builder)

    def test_method_on_pointer_receiver(self, builder):
        src = """
package app

type Server struct{}

func (s *Server) Start() error {
    return nil
}
""".strip()
        _parse(builder, src)
        ids = _node_ids(builder)
        assert "FILE:pkg/app.go::Server" in ids
        assert "FILE:pkg/app.go::Server::Start" in ids

    def test_method_on_value_receiver(self, builder):
        src = """
package app

type Point struct{ X, Y int }

func (p Point) Sum() int {
    return p.X + p.Y
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::Point::Sum" in _node_ids(builder)

    def test_generic_function(self, builder):
        src = """
package app

func Identity[T any](v T) T {
    return v
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::Identity" in _node_ids(builder)

    def test_generic_declarations_with_bracketed_constraints(self, builder):
        """A type parameter list may hold brackets of its own; a generic
        type is a type. A length in brackets is an array's."""
        src = """
package app

func Map[S ~[]E, E any](s S) S {
    return s
}

type Set[K comparable] struct{}

type Pair[A, B any] struct{}

type Grid [4]int
""".strip()
        _parse(builder, src)
        ids = _node_ids(builder)
        for name in ("Map", "Set", "Pair", "Grid"):
            assert f"FILE:pkg/app.go::{name}" in ids

    def test_return_type_lists_every_result(self, builder):
        src = """
package app

func One() *Server { return nil }

func Two() (*Server, error) { return nil, nil }

func Named() (first, second *Server, err error) { return }

func Nested() (Pair[string, int], map[string]Server) { return nil, nil }

func None() {}
""".strip()
        _parse(builder, src)
        returns = {
            n["id"].rpartition("::")[2]: n["metadata"]["return_type"]
            for n in builder.nodes
            if n["type"] == "Function"
        }
        assert returns == {
            "One": "*Server",
            "Two": "(*Server, error)",
            "Named": "(*Server, *Server, error)",
            "Nested": "(Pair[string, int], map[string]Server)",
            "None": "",
        }

    def test_function_typed_and_multi_line_signatures(self, builder):
        src = """
package app

func Walk(root string, visit func(path string) error) error {
    return visit(root)
}

func Chain(steps ...func() error) func() error {
    return nil
}

func Nested(wrap func(func(int) error) func() error) (func(a, b int) error, error) {
    return nil, nil
}

func Spread(
    root string,
    visit func(string) error,
) (
    count int,
    err error,
) {
    helper()
    return
}

func (s *Server) Handle(
    fn func(*Server),
) *Server {
    helper()
    return s
}

func helper() {}
""".strip()
        _parse(builder, src)
        returns = {
            n["id"].rpartition("::")[2]: n["metadata"]["return_type"]
            for n in builder.nodes
            if n["type"] in ("Function", "Method")
        }
        assert returns == {
            "Walk": "error",
            "Chain": "func() error",
            "Nested": "(func(a, b int) error, error)",
            "Spread": "(int, error)",
            "Handle": "*Server",
            "helper": "",
        }
        callers = {e["source"] for e in builder.edges if e["target"] == "REF:helper"}
        assert callers == {
            "FILE:pkg/app.go::Spread",
            "FILE:pkg/app.go::Server::Handle",
        }


class TestConstantsAndVars:
    def test_single_const(self, builder):
        src = """
package app

const MaxConns = 100
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::MaxConns" in _node_ids(builder)

    def test_grouped_const_block(self, builder):
        src = """
package app

const (
    RED = 0
    GREEN = 1
    BLUE = 2
)
""".strip()
        _parse(builder, src)
        ids = _node_ids(builder)
        assert "FILE:pkg/app.go::RED" in ids
        assert "FILE:pkg/app.go::GREEN" in ids
        assert "FILE:pkg/app.go::BLUE" in ids


class TestCallExtraction:
    def test_project_call_emitted(self, builder):
        src = """
package app

type Svc struct{}

func (s *Svc) Go() {
    myProjectFunc()
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:myProjectFunc" in targets

    def test_stdlib_call_filtered(self, builder):
        src = """
package app

type Svc struct{}

func (s *Svc) Go() {
    fmt.Println("hi")
}
""".strip()
        _parse(builder, src)
        builder.resolve_references()
        assert [e for e in builder.edges if e["relation"].startswith("CALLS")] == []

    def test_one_line_function_does_not_enclose_what_follows(self, builder):
        src = """
package app

func noop(x int) {}

func real() {
    helper()
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::real" in _node_ids(builder)

    def test_call_after_a_nested_block_keeps_its_function(self, builder):
        src = """
package app

func first() {
    if ready {
        prepare()
    }
    finish()
}

func second() {
    cleanup()
}
""".strip()
        _parse(builder, src)
        sources = {
            e["target"]: e["source"] for e in builder.edges if e["relation"] == "CALLS"
        }
        assert sources == {
            "REF:prepare": "FILE:pkg/app.go::first",
            "REF:finish": "FILE:pkg/app.go::first",
            "REF:cleanup": "FILE:pkg/app.go::second",
        }

    def test_control_flow_not_a_call(self, builder):
        src = """
package app

type Svc struct{}

func (s *Svc) Go() {
    for running {
        tick()
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:for" not in targets
        assert "REF:tick" in targets

    def test_builtin_filtered(self, builder):
        src = """
package app

type Svc struct{}

func (s *Svc) Go() {
    m := make(map[string]int)
    _ = append(nil, 1)
    _ = len(m)
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        # `make`, `append`, `len` are Go builtins — should not become CALLS edges.
        assert "REF:make" not in targets
        assert "REF:append" not in targets
        assert "REF:len" not in targets


class TestUnexportedAndReceivers:
    def test_unexported_type_and_const(self, builder):
        """Lowercase types and consts are indexed like exported ones."""
        src = """
package app

type worker struct {
    id int
}

const maxRetries = 3

func (w *worker) Run() {
    dispatch(w)
}
""".strip()
        _parse(builder, src)
        ids = _node_ids(builder)
        assert "FILE:pkg/app.go::worker" in ids
        assert "FILE:pkg/app.go::maxRetries" in ids
        assert "FILE:pkg/app.go::worker::Run" in ids

    def test_generic_receiver_method(self, builder):
        """A method on a generic type is indexed under that type."""
        src = """
package app

type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(v T) {
    grow(s)
}
""".strip()
        _parse(builder, src)
        assert "FILE:pkg/app.go::Stack::Push" in _node_ids(builder)
        calls = {
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        }
        assert ("FILE:pkg/app.go::Stack::Push", "REF:grow") in calls

    def test_method_with_an_unnamed_receiver(self, builder):
        src = """
package app

type timeoutError struct{}

func (timeoutError) Error() string { return describe() }

func (*timeoutError) Timeout() bool {
    return check()
}

func (Stack[T]) Empty() bool { return true }
""".strip()
        _parse(builder, src)
        methods = {
            n["id"]: (n["metadata"]["receiver"], n["metadata"]["return_type"])
            for n in builder.nodes
            if n["type"] == "Method"
        }
        assert methods == {
            "FILE:pkg/app.go::timeoutError::Error": ("timeoutError", "string"),
            "FILE:pkg/app.go::timeoutError::Timeout": ("timeoutError", "bool"),
            "FILE:pkg/app.go::Stack::Empty": ("Stack", "bool"),
        }
        calls = {
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        }
        assert calls == {
            ("FILE:pkg/app.go::timeoutError::Error", "REF:describe"),
            ("FILE:pkg/app.go::timeoutError::Timeout", "REF:check"),
        }

    def test_receiver_declared_in_another_file_gets_node(self, builder):
        """Every edge source exists even when the receiver type is elsewhere."""
        src = """
package app

func (r *Remote) Do() {
}
""".strip()
        _parse(builder, src)
        ids = set(_node_ids(builder))
        assert "FILE:pkg/app.go::Remote" in ids
        assert {e["source"] for e in builder.edges} <= ids

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
package app

func run() {
    stepOne() /* hidden(1)
    alsoHidden(2) */
    stepTwo()
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:stepOne", "REF:stepTwo"}
