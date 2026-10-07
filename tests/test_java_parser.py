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


class TestTypeScope:
    def test_brace_in_char_literal_does_not_end_the_class(self, builder):
        src = """
public class Reader {
    public int peek() {
        switch (c) {
            case '}':
                return 1;
        }
        return 0;
    }

    public int next() {
        return 2;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.java::Reader::next" in _node_names(builder)


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
        builder.resolve_references()
        assert [e for e in builder.edges if e["relation"].startswith("CALLS")] == []

    def test_constructor_without_modifier_is_a_declaration_not_a_call(self, builder):
        src = """
class Wrapper {
    Wrapper(Instrument instrument) {
        this.instrument = instrument;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.java::Wrapper::<init>" in _node_names(builder)
        assert [e for e in builder.edges if e["relation"] == "CALLS"] == []

    def test_string_literal_text_is_not_a_call(self, builder):
        src = """
public class X {
    public String hint() {
        return "Use reader.setLenient(true) to accept this";
    }
}
""".strip()
        _parse(builder, src)
        assert [e for e in builder.edges if e["relation"] == "CALLS"] == []

    def test_method_without_modifier_is_a_declaration_not_a_call(self, builder):
        src = """
interface Strategy {
    Number readNumber(Reader in) throws IOException;
}

class Impl implements Strategy {
    long[] values(Instant instant) {
        return null;
    }
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::Strategy::readNumber" in ids
        assert "FILE:src/Foo.java::Impl::values" in ids
        assert [e for e in builder.edges if e["relation"] == "CALLS"] == []

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


class TestFieldAndMethodOfOneName:
    def test_field_gets_its_own_node_and_the_method_keeps_its_id(self, builder):
        """A field declared before or after a method of its name is
        `<type>::field_<name>`; other fields keep `<type>::<name>`."""
        src = """
public class Counter {
    private int size;
    private int limit;

    public int size() {
        return size;
    }

    public int total() {
        return total;
    }

    private int total;
}
""".strip()
        _parse(builder, src)
        types = {n["id"]: n["type"] for n in builder.nodes}
        owner = "FILE:src/Foo.java::Counter"
        assert types[f"{owner}::size"] == "Method"
        assert types[f"{owner}::field_size"] == "Constant"
        assert types[f"{owner}::total"] == "Method"
        assert types[f"{owner}::field_total"] == "Constant"
        assert types[f"{owner}::limit"] == "Constant"
        assert f"{owner}::field_limit" not in types


class TestPackageMetadata:
    def test_package_attached_to_file_node(self, builder):
        src = """
package com.example.app;
public class Foo {}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.java")
        assert file_node.get("metadata", {}).get("java_package") == "com.example.app"


class TestAllmanBraces:
    def test_class_members_indexed(self, builder):
        """A class whose `{` is on a later line still gets its members."""
        src = """
public class Foo
    extends Base
{
    private int count;

    public void bar()
    {
        helper();
    }
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::Foo::bar" in ids
        assert "FILE:src/Foo.java::Foo::count" in ids
        calls = {
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        }
        assert ("FILE:src/Foo.java::Foo", "REF:helper") in calls

    def test_context_closes_with_class(self, builder):
        """Members after an Allman class belong to the next class."""
        src = """
class First
{
    public void a() {}
}

class Second {
    public void b() {}
}
""".strip()
        _parse(builder, src)
        ids = _node_names(builder)
        assert "FILE:src/Foo.java::First::a" in ids
        assert "FILE:src/Foo.java::Second::b" in ids


class TestDeclarationsAndComments:
    def test_declaration_is_not_a_call(self, builder):
        """Declaring a method or constructor does not call it."""
        src = """
public class Foo {
    public Foo(int size) {
        applySize(size);
    }

    public void bar(int x) {
        helper(x);
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:applySize", "REF:helper"}

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
public class Foo {
    public void bar() {
        stepOne(); /* hidden(1);
        alsoHidden(2); */ stepTwo();
        /* inline(3); */ stepThree();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:stepOne", "REF:stepTwo", "REF:stepThree"}
