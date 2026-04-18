"""Unit tests for the Dart regex parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.dart_parser import DartParser
from descry.generate import CodeGraphBuilder


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="lib/foo.dart"):
    DartParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestImportsAndParts:
    def test_package_import(self, builder):
        src = """
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:package:flutter/material.dart" in targets
        assert "MODULE:package:http/http.dart" in targets

    def test_dart_sdk_import(self, builder):
        src = """
import 'dart:async';
import 'dart:io';
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:dart:async" in targets
        assert "MODULE:dart:io" in targets

    def test_relative_import(self, builder):
        src = """
import 'other.dart';
import '../lib/shared.dart';
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:other.dart" in targets
        assert "MODULE:../lib/shared.dart" in targets


class TestClassesAndInheritance:
    def test_simple_class(self, builder):
        src = """
class Widget {
  void build() {}
}
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::Widget" in _ids(builder)

    def test_class_extends_emits_inherits(self, builder):
        src = """
class MyWidget extends StatelessWidget {
  void build() {}
}
""".strip()
        _parse(builder, src)
        inherits = [e for e in builder.edges if e["relation"] == "INHERITS"]
        assert any(
            e["source"] == "FILE:lib/foo.dart::MyWidget"
            and e["target"] == "REF:StatelessWidget"
            for e in inherits
        )

    def test_abstract_class(self, builder):
        src = """
abstract class Base {
  void run();
}
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::Base" in _ids(builder)

    def test_sealed_class(self, builder):
        src = """
sealed class Shape {}
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::Shape" in _ids(builder)

    def test_mixin(self, builder):
        src = """
mixin Lockable {
  void lock() {}
}
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::Lockable" in _ids(builder)

    def test_extension(self, builder):
        src = """
extension StringX on String {
  String reversed() => split('').reversed.join();
}
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::StringX" in _ids(builder)

    def test_enum(self, builder):
        src = """
enum Status { idle, running, done }
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::Status" in _ids(builder)


class TestCallsAndComments:
    def test_call_emitted(self, builder):
        src = """
class Foo {
  void activate() {
    doStuff();
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:doStuff" in targets

    def test_method_call(self, builder):
        # Use a distinctive method name — common ones like `start` / `run`
        # / `get` get filtered as stdlib last-part matches (shared filter
        # across all languages). Dart bare-method calls follow the same
        # trade-off as Python/Ruby.
        src = """
class Foo {
  void activate(Bar bar) {
    bar.performWidgetAction();
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert any("performWidgetAction" in t for t in targets)

    def test_control_flow_not_a_call(self, builder):
        src = """
class Foo {
  void activate(bool ready) {
    if (ready) { tick(); }
    while (running) { work(); }
    return;
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:while" not in targets
        assert "REF:return" not in targets
        assert "REF:tick" in targets
        assert "REF:work" in targets

    def test_line_comment_stripped(self, builder):
        src = """
class Foo {
  void activate() {
    // fake();
    real();
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets

    def test_block_comment_stripped(self, builder):
        src = """
/*
 * fake() inside a block comment
 */
class Foo {
  void activate() {
    real();
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets

    def test_url_in_string_preserved(self, builder):
        # "//" inside a string literal shouldn't be mistaken for a comment —
        # if it were, `projectHandler` on the following line would be
        # swallowed as comment text.
        src = """
class Foo {
  void activate() {
    final url = 'https://example.com';
    projectHandler(url);
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert any("projectHandler" in t for t in targets)
