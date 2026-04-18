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
  def run
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
