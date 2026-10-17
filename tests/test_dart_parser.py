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


class TestCallChains:
    def test_every_call_of_a_chain_is_recorded(self, builder):
        src = """
class Foo {
  void go(Maker maker) {
    maker.named('x').assemble()
        .ship();
  }
}
""".strip()
        _parse(builder, src)
        calls = [e["target"] for e in builder.edges if e["relation"] == "CALLS"]
        assert calls == ["REF:maker.named", "REF:assemble", "REF:ship"]

    def test_call_with_type_arguments_is_recorded(self, builder):
        src = """
class Foo {
  void go() {
    final box = Box<String>('x');
    final pairs = makePairs<String, List<int>>(box);
  }
}
""".strip()
        _parse(builder, src)
        calls = [e["target"] for e in builder.edges if e["relation"] == "CALLS"]
        assert calls == ["REF:Box", "REF:makePairs"]

    def test_cascade_is_a_method_call_on_its_target(self, builder):
        """`..add(1)` is recorded as a call on a receiver, so it cannot be
        taken for a call of a top-level `add`."""
        src = """
class Foo {
  void go(List<int> items) {
    items
      ..add(1);
  }
}
""".strip()
        _parse(builder, src)
        (edge,) = [e for e in builder.edges if e["relation"] == "CALLS"]
        assert (edge["target"], edge["metadata"]["origin"]) == (
            "REF:add",
            "List<int>().add",
        )


class TestDeclarationsAndInlineComments:
    def test_code_in_a_string_is_read(self, builder):
        """The `${...}` of a string is code, over lines too, and a brace in
        a string in it is text; an escaped `\\$` and a raw string hold
        none."""
        src = """
class Foo {
  String show() {
    final a = 'x ${name()} \\${not()} ${true ? "}" : 'y'}';
    final b = r'${raw()}' + r\"\"\"${raw()}\"\"\";
    return \"\"\"
${switch (a) {
      _ => fallback(),
    }}\"\"\";
  }

  void after() {
    done();
  }
}
""".strip()
        _parse(builder, src)
        calls = sorted(
            (e["source"].partition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        )
        assert calls == [
            ("Foo::after", "REF:done"),
            ("Foo::show", "REF:fallback"),
            ("Foo::show", "REF:name"),
        ]

    def test_declaration_is_not_a_call(self, builder):
        """Declaring a function, method or constructor does not call it."""
        src = """
class Foo {
  Foo(int y) {
    wireUp(y);
  }
  void bar(int x) {
    helper(x);
  }
}
void top(int a) {
  helper(a);
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:wireUp", "REF:helper"}

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
class Foo {
  void bar() {
    stepOne(); /* hidden(1);
    alsoHidden(2); */ stepTwo();
  }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:stepOne", "REF:stepTwo"}

    def test_string_text_is_neither_braces_nor_calls(self, builder):
        """A brace or a `name(` inside a string literal is text."""
        src = """
class Svc {
  void open() {
    var a = "}}";
    var b = 'see http://example.com/{id} or call hidden(1)';
    real(a);
  }
  void next() {
  }
}
""".strip()
        _parse(builder, src)
        assert "FILE:lib/foo.dart::Svc::next" in _ids(builder)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:real"}


def _defined(builder):
    """Each definition's path below its file -> (type, first line, last line)."""
    return {
        n["id"].partition("::")[2]: (
            n["type"],
            n["metadata"].get("lineno"),
            n["metadata"].get("end_lineno"),
        )
        for n in builder.nodes
        if n["type"] != "File"
    }


class TestDefinitions:
    def test_fields_accessors_and_variables(self, builder):
        src = """
final Map<String, List<int>> cache = {};
var counter = 0, total = 0;
int get size => cache.length;
set size(int value) {}

class Box {
  static const limit = 3;
  late final Stream<List<int>> _stream;
  final void Function(String) onDone;
  int? width, height;

  int get area => 0;
  set area(int value) {
    width = value;
  }
}
""".strip()
        _parse(builder, src)
        assert _defined(builder) == {
            "cache": ("Constant", 1, None),
            "counter": ("Constant", 2, None),
            "total": ("Constant", 2, None),
            "size": ("Function", 3, 3),
            "set size": ("Function", 4, 4),
            "Box": ("Class", 6, 16),
            "Box::limit": ("Constant", 7, None),
            "Box::_stream": ("Constant", 8, None),
            "Box::onDone": ("Constant", 9, None),
            "Box::width": ("Constant", 10, None),
            "Box::height": ("Constant", 10, None),
            "Box::area": ("Method", 12, 12),
            "Box::set area": ("Method", 13, 15),
        }
        types = {
            n["id"].rpartition("::")[2]: n["metadata"]["value_type"]
            for n in builder.nodes
            if n["metadata"].get("value_type")
        }
        assert types == {
            "cache": "Map<String, List<int>>",
            "size": "int",
            "_stream": "Stream<List<int>>",
            "onDone": "void Function(String)",
            "width": "int?",
            "height": "int?",
            "area": "int",
        }

    def test_every_kind_of_type(self, builder):
        src = """
class _Private {}
sealed class Shape {}
abstract interface class Reader {}
base mixin class Mixed {}
final class Circle extends Shape with Mixed implements Reader {}
class Alias = Shape with Mixed;
mixin _Lockable on Shape {}
extension on Shape {}
extension type Id(int value) implements Object {}
enum Level implements Reader {
  low,
  high(2);

  const Level([this.weight = 1]);
  final int weight;
}
typedef _Callback = void Function(int);
typedef int Compare(int a, int b);
""".strip()
        _parse(builder, src)
        kinds = {
            n["id"].partition("::")[2]: n["metadata"].get("kind")
            for n in builder.nodes
            if n["type"] == "Class"
        }
        assert kinds == {
            "_Private": "class",
            "Shape": "class",
            "Reader": "class",
            "Mixed": "class",
            "Circle": "class",
            "Alias": "class",
            "_Lockable": "mixin",
            "extension on Shape": "extension",
            "Id": "extension type",
            "Level": "enum",
            "_Callback": "typedef",
            "Compare": "typedef",
        }
        members = {k for k in _defined(builder) if "::" in k}
        assert members == {
            "Id::value",
            "Level::low",
            "Level::high",
            "Level::new",
            "Level::weight",
        }
        inherits = [
            (e["source"].rpartition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        ]
        assert inherits == [
            ("Circle", "REF:Mixed"),
            ("Circle", "REF:Shape"),
            ("Circle", "REF:Reader"),
            ("Alias", "REF:Mixed"),
            ("Alias", "REF:Shape"),
            ("_Lockable", "REF:Shape"),
            ("extension on Shape", "REF:Shape"),
            ("Id", "REF:Object"),
            ("Level", "REF:Reader"),
            ("_Callback", "REF:Function"),
            ("Compare", "REF:Function"),
        ]
        assert not [e for e in builder.edges if e["relation"] == "CALLS"]

    def test_constructors_are_named_as_they_are_called(self, builder):
        """`Response.ok(...)` calls the member `ok` of Response; the
        constructor without a name is `Response.new`. A field or a getter
        that shares a constructor's name gets an ID of its own."""
        src = """
class Response {
  final int status;
  final bool ok;
  Response(this.status);
  Response.ok() : this(200);
  const Response._(this.status);
  factory Response.notFound() => Response(404);
  factory Response.from(Response other) {
    return other;
  }
  Response get from => this;
}
""".strip()
        _parse(builder, src)
        assert {"Response::field ok", "Response::get from"} < set(_defined(builder))
        constructors = {
            n["id"].partition("::")[2]: n["metadata"]["return_type"]
            for n in builder.nodes
            if n["metadata"].get("is_constructor")
        }
        assert constructors == dict.fromkeys(
            (
                "Response::new",
                "Response::ok",
                "Response::_",
                "Response::notFound",
                "Response::from",
            ),
            "Response",
        )
        calls = [e["target"] for e in builder.edges if e["relation"] == "CALLS"]
        assert calls == ["REF:Response"]

    def test_signature_is_read_to_the_body(self, builder):
        """A return type with nested type arguments or a prefix, a signature
        over several lines, and a block of named parameters, which is not
        the body."""
        src = """
Stream<List<int>> read() async* {
  yield [];
}

shelf.Handler mockHandler(
  String path, {
  bool strict = false,
}) {
  return fallback;
}

Future<Map<String, List<int>>> load({required String path}) => parse(path);

class Server {
  static Future<Server> bind(
      Object address, int port,
      {int backlog = 0}) async {
    return Server._(address);
  }

  Server._(this.address)
      : port = 0,
        options = const {} {
    start();
  }

  bool operator ==(Object other) => other is Server;
  external int get hashCode;
  T convert<T extends Object>(T Function(Server) body);
}
""".strip()
        _parse(builder, src)
        assert _defined(builder) == {
            "read": ("Function", 1, 3),
            "mockHandler": ("Function", 5, 10),
            "load": ("Function", 12, 12),
            "Server": ("Class", 14, 30),
            "Server::bind": ("Method", 15, 19),
            "Server::_": ("Method", 21, 25),
            "Server::operator==": ("Method", 27, 27),
            "Server::hashCode": ("Method", 28, 28),
            "Server::convert": ("Method", 29, 29),
        }
        returns = {
            n["id"].rpartition("::")[2]: n["metadata"].get("return_type")
            for n in builder.nodes
            if n["type"] in ("Function", "Method")
        }
        assert returns == {
            "read": "Stream<List<int>>",
            "mockHandler": "shelf.Handler",
            "load": "Future<Map<String, List<int>>>",
            "bind": "Future<Server>",
            "_": "Server",
            "operator==": "bool",
            "hashCode": None,
            "convert": None,
        }
        calls = [
            (e["source"].rpartition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        assert calls == [
            ("load", "REF:parse"),
            ("bind", "REF:Server._"),
            ("_", "REF:start"),
        ]

    def test_multi_line_string_is_not_code(self, builder):
        src = """
const usage = '''
class Ghost {
  void nope() { hidden(); }
}
''';

void real() {}
""".strip()
        _parse(builder, src)
        assert _defined(builder) == {
            "usage": ("Constant", 1, None),
            "real": ("Function", 7, 7),
        }
        assert not [e for e in builder.edges if e["relation"] == "CALLS"]

    def test_annotation_calls_a_constructor(self, builder):
        """`@Route.get('/')` calls the constructor `Route.get`, from the
        member it annotates; `@override` calls nothing."""
        src = """
class Api {
  @Route.get('/users')
  @override
  Future<Response> users(Request request) async {
    return list();
  }
}
""".strip()
        _parse(builder, src)
        calls = [
            (e["source"].rpartition("::")[2], e["target"], e["metadata"].get("origin"))
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        assert calls == [
            ("users", "REF:Route.get", "Route.get"),
            ("users", "REF:list", None),
        ]

    def test_member_named_like_a_keyword_is_called_through_its_receiver(self, builder):
        src = """
void run(Theme theme, AsyncValue value) {
  Theme.of(theme);
  value.when(data: show);
  this.start();
}
""".strip()
        _parse(builder, src)
        calls = [e["target"] for e in builder.edges if e["relation"] == "CALLS"]
        assert calls == ["REF:Theme.of", "REF:value.when", "REF:this.start"]
