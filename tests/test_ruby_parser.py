"""Unit tests for the Ruby regex/indent-based baseline parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.generate import CodeGraphBuilder
from descry.ruby_parser import RubyParser


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="app/foo.rb"):
    RubyParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestClassesAndModules:
    def test_class(self, builder):
        src = """
class Foo
  def bar
    42
  end
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb" in ids
        assert "FILE:app/foo.rb::Foo" in ids
        assert "FILE:app/foo.rb::Foo::bar" in ids

    def test_class_with_superclass(self, builder):
        src = """
class Widget < ApplicationRecord
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb::Widget" in ids
        edges = {(e["source"], e["target"], e["relation"]) for e in builder.edges}
        assert (
            "FILE:app/foo.rb::Widget",
            "REF:ApplicationRecord",
            "INHERITS",
        ) in edges

    def test_module(self, builder):
        src = """
module Helpers
  def greet
    "hi"
  end
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb::Helpers" in ids
        assert "FILE:app/foo.rb::Helpers::greet" in ids

    def test_nested_module_class(self, builder):
        src = """
module Admin
  class Panel
    def render
    end
  end
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb::Admin" in ids
        assert "FILE:app/foo.rb::Admin::Panel" in ids
        assert "FILE:app/foo.rb::Admin::Panel::render" in ids


class TestMethods:
    def test_instance_method(self, builder):
        src = """
class Foo
  def speak
  end
end
""".strip()
        _parse(builder, src)
        assert "FILE:app/foo.rb::Foo::speak" in _ids(builder)

    def test_self_method(self, builder):
        src = """
class Foo
  def self.factory
  end
end
""".strip()
        _parse(builder, src)
        assert "FILE:app/foo.rb::Foo::factory" in _ids(builder)

    def test_predicate_method(self, builder):
        src = """
class User
  def admin?
    true
  end
end
""".strip()
        _parse(builder, src)
        assert "FILE:app/foo.rb::User::admin?" in _ids(builder)

    def test_bang_method(self, builder):
        src = """
class User
  def save!
  end
end
""".strip()
        _parse(builder, src)
        assert "FILE:app/foo.rb::User::save!" in _ids(builder)


class TestAccessors:
    def test_attr_reader(self, builder):
        src = """
class User
  attr_reader :name, :email
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb::User::name" in ids
        assert "FILE:app/foo.rb::User::email" in ids

    def test_attr_accessor(self, builder):
        src = """
class User
  attr_accessor :id
end
""".strip()
        _parse(builder, src)
        assert "FILE:app/foo.rb::User::id" in _ids(builder)


class TestImports:
    def test_require(self, builder):
        src = """
require 'json'
require_relative 'util/helpers'
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:json" in targets
        assert "MODULE:util/helpers" in targets


class TestConstants:
    def test_top_level_constant(self, builder):
        src = """
MAX_CONNECTIONS = 100

class Foo
  DEFAULT = 42
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb::MAX_CONNECTIONS" in ids
        assert "FILE:app/foo.rb::Foo::DEFAULT" in ids


class TestCalls:
    def test_paren_call_emitted(self, builder):
        src = """
class Svc
  def run
    do_work()
  end
end
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:do_work" in targets

    def test_receiver_call(self, builder):
        src = """
class Svc
  def run(user, token)
    user.authenticate(token)
  end
end
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:user.authenticate" in targets

    def test_keyword_not_a_call(self, builder):
        src = """
class Svc
  def run
    if ready? then tick() end
  end
end
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:tick" in targets

    def test_hash_comment_stripped(self, builder):
        src = """
class Svc
  def run
    # call_never_happens(x)
    real_call()
  end
end
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:call_never_happens" not in targets
        assert "REF:real_call" in targets


class TestOneLineBodiesAndDocs:
    def test_one_line_class_does_not_capture_siblings(self, builder):
        """`class X; end` closes on its line; what follows is its sibling."""
        src = """
module Billing
  class Error < StandardError; end
  LIMIT = 3

  def charge
    submit(1)
  end
end
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:app/foo.rb::Billing::LIMIT" in ids
        assert "FILE:app/foo.rb::Billing::charge" in ids

    def test_one_line_module_does_not_capture_siblings(self, builder):
        """`module X; end` closes on its line too."""
        src = """
module Billing
  module Marker; end
  LIMIT = 3
end
""".strip()
        _parse(builder, src)
        assert "FILE:app/foo.rb::Billing::LIMIT" in _ids(builder)

    def test_hash_comments_become_docstrings(self, builder):
        """`#` lines directly above a class, module or method document it."""
        src = """
# Billing helpers.
module Billing
  # Raised when a charge fails.
  class Error
  end

  # Charges the card.
  # Retries once.
  def charge
  end
end
""".strip()
        _parse(builder, src)
        docs = {n["id"]: n["metadata"].get("docstring") for n in builder.nodes}
        assert docs["FILE:app/foo.rb::Billing"] == "Billing helpers."
        assert docs["FILE:app/foo.rb::Billing::Error"] == "Raised when a charge fails."
        assert (
            docs["FILE:app/foo.rb::Billing::charge"]
            == "Charges the card.\nRetries once."
        )


def _nodes(builder):
    """``(type, line, end line)`` of each node, by its ID below the file."""
    return {
        n["id"].partition("::")[2]: (
            n["type"],
            n["metadata"].get("lineno"),
            n["metadata"].get("end_lineno"),
        )
        for n in builder.nodes
        if n["type"] != "File"
    }


def _call_edges(builder):
    """``(source below the file, callee)`` of each call, in source order."""
    return [
        (e["source"].partition("::")[2], e["target"].removeprefix("REF:"))
        for e in builder.edges
        if e["relation"] == "CALLS"
    ]


class TestBlockStructure:
    def test_definition_after_a_block_belongs_to_the_class(self, builder):
        """A `do ... end`, an `if` and a `case` inside a method close with
        their own `end`: what follows the method is the class's."""
        src = """
class Report
  def rows(items)
    items.each do |item|
      if item.ready?
        yield item
      end
    end
    case items.size
    when 0 then nil
    end
  end

  attr_reader :title
  LIMIT = 3

  def title?
    !title.nil?
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Report": ("Class", 1, 19),
            "Report::rows": ("Method", 2, 11),
            "Report::title": ("Method", 13, 13),
            "Report::LIMIT": ("Constant", 14, 14),
            "Report::title?": ("Method", 16, 18),
        }

    def test_modifiers_and_one_line_methods_open_no_block(self, builder):
        """A statement modifier has no `end`; an endless method and a
        method written on one line end on their line; an `if` that is a
        value has an `end` of its own."""
        src = """
class Gate
  def open?(user) = user.admin?
  def reset; @count = 0; end

  def enter(user)
    return unless open?(user)
    @count += 1 while @count < 3
    label = if user.guest? then "guest" else "member" end
    @log << if label == "guest"
              "g"
            else
              "m"
            end
  end

  def leave
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Gate": ("Class", 1, 18),
            "Gate::open?": ("Method", 2, 2),
            "Gate::reset": ("Method", 3, 3),
            "Gate::enter": ("Method", 5, 14),
            "Gate::leave": ("Method", 16, 17),
        }

    def test_definition_in_a_block_belongs_to_the_block(self, builder):
        """A method of a spec block is not a method of the method defined
        before it, nor of the file: the block is a scope named after the
        method it is passed to."""
        src = """
def helper
  1
end

describe Widget do
  def build(name)
    Widget.new(name)
  end

  it "has a name" do
    build("a").name
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "helper": ("Method", 1, 3),
            "<describe:5>": ("Class", 5, 13),
            "<describe:5>::build": ("Method", 6, 8),
        }

    def test_text_of_literals_and_comments_is_not_code(self, builder):
        """Strings, heredocs, regexps, %-literals and `=begin` comments
        define nothing, call nothing and close no block; code interpolated
        into a string is code."""
        src = """
class Page
  SCRIPT = <<~JS
    function end() { render(1) }
    class Fake
  JS
  WORDS = %w[def class end]
  PATTERN = /
    def\\s+(\\w+) # not a method
  /x

=begin
  def hidden
  end
=end

  def title(name)
    "def shown(#{escape(name)}) end"
  end

  def slash(path)
    path.sub(/\\/end$/, 'if (x)')
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Page": ("Class", 1, 23),
            "Page::SCRIPT": ("Constant", 2, 2),
            "Page::WORDS": ("Constant", 6, 6),
            "Page::PATTERN": ("Constant", 7, 7),
            "Page::title": ("Method", 16, 18),
            "Page::slash": ("Method", 20, 22),
        }
        assert _call_edges(builder) == [
            ("Page::title", "escape"),
            ("Page::slash", "path.sub"),
        ]

    def test_hash_key_is_no_constant(self, builder):
        src = """
module Codes
  NAMES = {
    OK => "ok",
    MISSING => "missing",
  }
  OK == 200
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Codes": ("Class", 1, 7),
            "Codes::NAMES": ("Constant", 2, 2),
        }


class TestSingletonMethods:
    def test_singleton_methods_are_the_class_s(self, builder):
        """`def self.name`, an operator on `self` and a method in `class <<
        self` are methods of the class; none makes a method named `self`."""
        src = """
class Registry
  def self.[](key)
    lookup(key)
  end

  class << self
    def lookup(key)
    end
  end

  def Registry.clear
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Registry": ("Class", 1, 13),
            "Registry::[]": ("Method", 2, 4),
            "Registry::lookup": ("Method", 7, 8),
            "Registry::clear": ("Method", 11, 12),
        }

    def test_singleton_method_beside_an_instance_method_of_its_name(self, builder):
        """Both are nodes: the singleton method is `self.name`."""
        src = """
class Parser
  def self.parse(io)
    new.parse(io)
  end

  def parse(io)
  end

  def call(env)
  end

  def self.call(env)
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Parser": ("Class", 1, 14),
            "Parser::self.parse": ("Method", 2, 4),
            "Parser::parse": ("Method", 6, 7),
            "Parser::call": ("Method", 9, 10),
            "Parser::self.call": ("Method", 12, 13),
        }
        names = {n["id"]: n["metadata"]["name"] for n in builder.nodes}
        assert names["FILE:app/foo.rb::Parser::self.parse"] == "parse"

    def test_method_of_one_object_is_named_with_the_object(self, builder):
        """`def object.name` and a method in `class << object` belong to
        that object, not to the class or file around them."""
        src = """
def stub
  reader = Object.new
  def reader.rewind; end
  class << reader
    def read
      nil
    end
  end
  reader
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "stub": ("Method", 1, 10),
            "stub::reader.rewind": ("Method", 3, 3),
            "stub::reader.read": ("Method", 5, 7),
        }


class TestDefinitions:
    def test_methods_defined_other_than_by_a_plain_def(self, builder):
        src = """
class Box
  private def seal(tape)
  end

  alias close seal
  alias_method :shut, :seal
  define_method(:open) { |force| force }
  attr_writer :label
  attr_accessor :weight,
                :size

  def ==(other)
  end

  def label=(text)
  end

  def <=>(other)
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Box": ("Class", 1, 20),
            "Box::seal": ("Method", 2, 3),
            "Box::close": ("Method", 5, 5),
            "Box::shut": ("Method", 6, 6),
            "Box::open": ("Method", 7, 7),
            "Box::label=": ("Method", 8, 8),
            "Box::weight": ("Method", 9, 9),
            "Box::weight=": ("Method", 9, 9),
            "Box::size": ("Method", 9, 9),
            "Box::size=": ("Method", 9, 9),
            "Box::==": ("Method", 12, 13),
            "Box::<=>": ("Method", 18, 19),
        }

    def test_struct_and_class_made_by_a_call(self, builder):
        """`Struct.new` gives its members as accessors; the block of a
        `Struct.new` or `Class.new` is the body of the type, named by the
        constant it is assigned to or by the line it is made on."""
        src = """
Point = Struct.new(:x, :y) do
  def norm
    x + y
  end
end
Failure = Class.new(StandardError)

def app
  Class.new(Base) do
    def call(env)
    end
  end
end

class Pair < Struct.new(:left)
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Point": ("Class", 1, 5),
            "Point::x": ("Method", 1, 1),
            "Point::x=": ("Method", 1, 1),
            "Point::y": ("Method", 1, 1),
            "Point::y=": ("Method", 1, 1),
            "Point::norm": ("Method", 2, 4),
            "Failure": ("Class", 6, 6),
            "app": ("Method", 8, 13),
            "app::<Class:9>": ("Class", 9, 12),
            "app::<Class:9>::call": ("Method", 10, 11),
            "Pair": ("Class", 15, 16),
            "Pair::left": ("Method", 15, 15),
            "Pair::left=": ("Method", 15, 15),
        }
        inherits = {
            (e["source"].partition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        }
        assert inherits == {
            ("Failure", "REF:StandardError"),
            ("app::<Class:9>", "REF:Base"),
            ("Pair", "REF:Struct"),
        }

    def test_type_named_on_one_line_spans_its_first_body(self, builder):
        """`module Rack; end` only names the namespace: the node spans
        the body that reopens it, and has what every reopening defines."""
        src = """
module Rack; end

# The toolkit.
module Rack
  VERSION = "1"
end

module Rack
  def self.release
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Rack": ("Class", 4, 6),
            "Rack::VERSION": ("Constant", 5, 5),
            "Rack::release": ("Method", 9, 10),
        }
        docs = {n["id"]: n["metadata"].get("docstring") for n in builder.nodes}
        assert docs["FILE:app/foo.rb::Rack"] == "The toolkit."

    def test_block_that_defines_methods_is_a_scope(self, builder):
        """Each such block is a node named after the method it is passed
        to; a constant written in it belongs to the class around it."""
        src = """
class Suite
  helpers do
    LIMIT = 3
    alias_method :h, :escape
  end

  items.each do |item|
    def visit
    end
  end
end
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Suite": ("Class", 1, 11),
            "Suite::<helpers:2>": ("Class", 2, 5),
            "Suite::LIMIT": ("Constant", 3, 3),
            "Suite::<helpers:2>::h": ("Method", 4, 4),
            "Suite::<each:7>": ("Class", 7, 10),
            "Suite::<each:7>::visit": ("Method", 8, 9),
        }

    def test_mixins_are_bases(self, builder):
        src = """
class Request
  include Env, Helpers
  extend Rack::Builder
  prepend Checks
end
""".strip()
        _parse(builder, src)
        bases = [e["target"] for e in builder.edges if e["relation"] == "INHERITS"]
        assert bases == ["REF:Env", "REF:Helpers", "REF:Rack::Builder", "REF:Checks"]


class TestCallSites:
    def test_call_belongs_to_the_method_it_is_in(self, builder):
        """A call in a block is the enclosing method's, one in a class body
        the class's and one outside both the file's."""
        src = """
setup(1)

class Job
  register(self)

  def run(items)
    items.each do |item|
      process(item)
    end
  end

  def stop
    halt(1)
  end
end
""".strip()
        _parse(builder, src)
        assert _call_edges(builder) == [
            ("", "setup"),
            ("Job", "register"),
            ("Job::run", "items.each"),
            ("Job::run", "process"),
            ("Job::stop", "halt"),
        ]

    def test_calls_without_parentheses(self, builder):
        """A method named behind a `.` or `&.` is called, with arguments,
        with a block or with neither, as is one assigned to; a name read
        without a receiver is called unless it is a local variable."""
        src = """
class Mailer
  def deliver(message, list)
    message.validate
    message.send_to list
    message&.archive
    message.subject = "sent"
    list.each { |entry| log entry }
    count = total
    notify count
    footer unless !sealed? || !list.empty?
  end
end
""".strip()
        _parse(builder, src)
        assert [callee for _, callee in _call_edges(builder)] == [
            "message.validate",
            "message.send_to",
            "message.archive",
            "message.subject=",
            "list.each",
            "log",
            "total",
            "notify",
            "footer",
            "sealed?",
            "list.empty?",
        ]

    def test_local_variables_are_not_calls(self, builder):
        """A parameter, a block parameter and a name assigned earlier are
        read, not called; a name assigned in a block is no variable after
        it, nor in the next method."""
        src = """
class Tally
  def add(amount, *rest, by: 1, &block)
    sum = amount + by
    rest.each do |more|
      inner = more
      sum += inner
    end
    first, second = rest
    block.call(sum, first, second)
    inner
  rescue Failure => error
    error.report
  end

  def reset
    sum
  end
end
""".strip()
        _parse(builder, src)
        assert _call_edges(builder) == [
            ("Tally::add", "rest.each"),
            ("Tally::add", "block.call"),
            ("Tally::add", "inner"),
            ("Tally::add", "error.report"),
            ("Tally::reset", "sum"),
        ]

    def test_variable_assigned_a_block_s_result_outlives_the_block(self, builder):
        """The variable is one of the scope the statement is in, a
        parameter of the block one of the block."""
        src = """
def build(items)
  app = lambda do |env|
    env
  end
  names = items.map { |item| item.name }
  app.call(names)
  item
end
""".strip()
        _parse(builder, src)
        assert _call_edges(builder) == [
            ("build", "items.map"),
            ("build", "item.name"),
            ("build", "app.call"),
            ("build", "item"),
        ]

    def test_chain_of_calls(self, builder):
        """Each link is a call on what the link before returns, on one
        line or continued after or before a `.`."""
        src = """
class Client
  def fetch(url)
    session.get(url).body.strip
    session.
      post(url).
      status
    session
      .head(url)
  end
end
""".strip()
        _parse(builder, src)
        calls = [
            (e["target"], e["metadata"].get("origin"))
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        first = next(i for i, e in enumerate(builder.edges) if e["relation"] == "CALLS")
        assert calls == [
            ("REF:session", None),
            ("REF:get", f"#{first}.get"),
            ("REF:body", f"#{first + 1}.body"),
            ("REF:strip", f"#{first + 2}.strip"),
            ("REF:session", None),
            ("REF:post", f"#{first + 4}.post"),
            ("REF:status", f"#{first + 5}.status"),
            ("REF:session", None),
            ("REF:head", f"#{first + 7}.head"),
        ]
