"""Unit tests for the Java regex-based baseline parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.generate import CodeGraphBuilder
from descry.java_parser import JavaParser


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder: CodeGraphBuilder, source: str, rel: str = "src/Foo.java"):
    JavaParser(builder).parse(Path(rel), rel, source)
    return builder


def _node_names(builder: CodeGraphBuilder) -> list[str]:
    return [n["id"] for n in builder.nodes]


class TestClassExtraction:
    def test_simple_public_class(self, builder):
        src = """
package com.example;

public class Foo {
    public void bar() {
        System.out.println("hi");
    }
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java" in ids
        assert "FILE:src/Foo.java::Foo" in ids
        assert "FILE:src/Foo.java::Foo::bar" in ids

    def test_interface(self, builder):
        src = """
public interface MyApi {
    public void doThing();
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::MyApi" in ids
        assert "FILE:src/Foo.java::MyApi::doThing" in ids

    def test_enum_with_method(self, builder):
        src = """
public enum Color {
    RED, GREEN, BLUE;

    public String hex() {
        return "#000";
    }
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::Color" in ids
        assert "FILE:src/Foo.java::Color::hex" in ids

    def test_record(self, builder):
        src = """
public record Point(int x, int y) {
    public int sum() {
        return x + y;
    }
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        # The record declaration registers as a Class-kind symbol.
        assert "FILE:src/Foo.java::Point" in ids
        # The method inside the record body is extracted.
        assert "FILE:src/Foo.java::Point::sum" in ids

    def test_annotation_interface(self, builder):
        src = """
public @interface MyAnnotation {
    String value() default "";
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::MyAnnotation" in ids


class TestMethodExtraction:
    def test_static_method(self, builder):
        src = """
public class Utils {
    public static int add(int a, int b) {
        return a + b;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.java::Utils::add" in _node_names(builder)

    def test_private_final_method(self, builder):
        src = """
public class Thing {
    private final String describe() {
        return "thing";
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.java::Thing::describe" in _node_names(builder)

    def test_generic_method(self, builder):
        src = """
public class Box {
    public static <T> T identity(T input) {
        return input;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.java::Box::identity" in _node_names(builder)

    def test_constructor(self, builder):
        src = """
public class Widget {
    public Widget(int n) {
        this.n = n;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.java::Widget::<init>" in _node_names(builder)

    def test_if_not_mistaken_for_method(self, builder):
        """Bare `if (cond)` must not be extracted as a method named 'if'."""
        src = """
public class Guard {
    public void work() {
        if (ok) {
            doStuff();
        }
    }
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::Guard::if" not in ids
        assert "FILE:src/Foo.java::Guard::work" in ids


class TestImports:
    def test_plain_import(self, builder):
        src = """
package com.example;
import java.util.List;
import java.util.Map;

public class Foo {}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:java.util.List" in targets
        assert "MODULE:java.util.Map" in targets

    def test_static_import(self, builder):
        src = """
import static java.util.Arrays.asList;

public class X {}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:java.util.Arrays.asList" in targets

    def test_wildcard_import(self, builder):
        src = """
import java.util.*;

public class X {}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:java.util" in targets


class TestCallExtraction:
    def test_project_call_emitted(self, builder):
        src = """
public class Caller {
    public void go() {
        doProjectThing();
    }
}
""".strip()
        _parse(builder, src)
        calls = [e for e in builder.edges if e["relation"] == "CALLS"]
        targets = {e["target"] for e in calls}
        assert "REF:doProjectThing" in targets

    def test_stdlib_call_filtered(self, builder):
        src = """
public class X {
    public void go() {
        java.lang.System.out.println("hi");
    }
}
""".strip()
        _parse(builder, src)
        call_targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        # java.lang.System.out.println should not produce a CALLS edge.
        assert "REF:java.lang.System.out.println" not in call_targets

    def test_control_flow_not_a_call(self, builder):
        src = """
public class X {
    public void go() {
        while (running) { tick(); }
    }
}
""".strip()
        _parse(builder, src)
        call_targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:while" not in call_targets
        assert "REF:tick" in call_targets


class TestPackageMetadata:
    def test_package_attached_to_file_node(self, builder):
        src = """
package com.example.app;
public class Foo {}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.java")
        assert file_node.get("metadata", {}).get("java_package") == "com.example.app"
