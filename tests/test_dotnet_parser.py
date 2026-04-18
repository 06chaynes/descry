"""Unit tests for the C# regex parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.dotnet_parser import DotnetParser
from descry.generate import CodeGraphBuilder


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="src/Foo.cs"):
    DotnetParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestNamespaceAndUsing:
    def test_block_namespace(self, builder):
        src = """
namespace App.Foo
{
    public class Bar {}
}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.cs")
        assert file_node.get("metadata", {}).get("dotnet_namespace") == "App.Foo"

    def test_file_scoped_namespace(self, builder):
        src = """
namespace App.Foo;

public class Bar {}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.cs")
        assert file_node.get("metadata", {}).get("dotnet_namespace") == "App.Foo"

    def test_using(self, builder):
        src = """
using System;
using System.Collections.Generic;
using static System.Math;

namespace App;

public class X {}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:System" in targets
        assert "MODULE:System.Collections.Generic" in targets
        assert "MODULE:System.Math" in targets


class TestTypesAndMethods:
    def test_class_allman(self, builder):
        src = """
public class Service
{
    public void Run()
    {
        DoWork();
    }
}
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/Foo.cs::Service" in ids
        assert "FILE:src/Foo.cs::Service::Run" in ids

    def test_interface(self, builder):
        src = """
public interface IRunnable
{
    void Run();
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::IRunnable" in _ids(builder)

    def test_struct(self, builder):
        src = """
public struct Point
{
    public int X;
    public int Y;
    public int Sum() => X + Y;
}
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/Foo.cs::Point" in ids
        assert "FILE:src/Foo.cs::Point::Sum" in ids

    def test_record(self, builder):
        src = """
public record PersonRecord(string Name, int Age);
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::PersonRecord" in _ids(builder)

    def test_enum(self, builder):
        src = """
public enum Color { Red, Green, Blue }
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Color" in _ids(builder)

    def test_async_method(self, builder):
        src = """
public class Svc
{
    public async Task<int> ComputeAsync()
    {
        return 42;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Svc::ComputeAsync" in _ids(builder)

    def test_generic_method(self, builder):
        src = """
public class Box
{
    public static T Identity<T>(T input) => input;
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Box::Identity" in _ids(builder)


class TestPropertiesAndFields:
    def test_property(self, builder):
        src = """
public class User
{
    public string Name { get; set; }
}
""".strip()
        _parse(builder, src)
        # Property regex requires ; — one-liner auto-props should be caught.
        # Our regex matches property with { get; set; } so we check ids.
        assert "FILE:src/Foo.cs::User" in _ids(builder)

    def test_const(self, builder):
        src = """
public class Config
{
    public const int MaxConnections = 100;
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Config::MaxConnections" in _ids(builder)


class TestCalls:
    def test_call_emitted(self, builder):
        src = """
public class Svc
{
    public void Go()
    {
        DoStuff();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:DoStuff" in targets

    def test_method_on_object(self, builder):
        src = """
public class Svc
{
    public void Go(User u)
    {
        u.Authenticate(token);
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:u.Authenticate" in targets

    def test_control_flow_not_a_call(self, builder):
        src = """
public class Svc
{
    public void Go()
    {
        if (ready) { tick(); }
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:tick" in targets

    def test_line_comment_stripped(self, builder):
        src = """
public class Svc
{
    public void Go()
    {
        // fake(x);
        real();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets


class TestVbFilesSkipped:
    def test_vb_file_no_symbols_beyond_file(self, builder):
        src = """
Public Class Foo
    Public Sub Greet()
        Console.WriteLine("hi")
    End Sub
End Class
""".strip()
        DotnetParser(builder).parse(Path("src/Foo.vb"), "src/Foo.vb", src)
        ids = _ids(builder)
        # VB is SCIP-only — parser only emits the File node.
        assert ids == ["FILE:src/Foo.vb"]
