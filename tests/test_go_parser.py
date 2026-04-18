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
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fmt.Println" not in targets

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
