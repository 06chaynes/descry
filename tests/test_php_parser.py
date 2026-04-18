"""Unit tests for the PHP regex parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.generate import CodeGraphBuilder
from descry.php_parser import PhpParser


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="src/Foo.php"):
    PhpParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestNamespacesAndUse:
    def test_namespace(self, builder):
        src = """<?php
namespace App\\Foo;

class Bar {}
"""
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.php")
        assert file_node.get("metadata", {}).get("php_namespace") == "App\\Foo"

    def test_use(self, builder):
        src = """<?php
use App\\Foo\\Bar;
use App\\Foo\\Baz as Qux;

class X {}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:App\\Foo\\Bar" in targets
        assert "MODULE:App\\Foo\\Baz" in targets


class TestClassesAndMethods:
    def test_class(self, builder):
        src = """<?php
class Service {
    public function run(): void {
        doWork();
    }
}
"""
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/Foo.php::Service" in ids
        assert "FILE:src/Foo.php::Service::run" in ids

    def test_interface(self, builder):
        src = """<?php
interface Runnable {
    public function run(): void;
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Runnable" in _ids(builder)
        assert "FILE:src/Foo.php::Runnable::run" in _ids(builder)

    def test_trait(self, builder):
        src = """<?php
trait Loggable {
    public function log(string $msg): void {}
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Loggable" in _ids(builder)
        assert "FILE:src/Foo.php::Loggable::log" in _ids(builder)

    def test_enum(self, builder):
        src = """<?php
enum Status: string {
    case Active = 'active';
    case Inactive = 'inactive';
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Status" in _ids(builder)

    def test_static_method(self, builder):
        src = """<?php
class Utils {
    public static function hash(string $s): string {
        return md5($s);
    }
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Utils::hash" in _ids(builder)

    def test_abstract_method(self, builder):
        src = """<?php
abstract class Handler {
    abstract public function handle(): void;
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Handler::handle" in _ids(builder)

    def test_top_level_function(self, builder):
        src = """<?php
function helper(string $x): string {
    return strtoupper($x);
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::helper" in _ids(builder)


class TestPropertiesAndConstants:
    def test_property(self, builder):
        src = """<?php
class Config {
    public string $name = 'default';
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Config::name" in _ids(builder)

    def test_constant(self, builder):
        src = """<?php
class Config {
    public const TIMEOUT = 30;
}
"""
        _parse(builder, src)
        assert "FILE:src/Foo.php::Config::TIMEOUT" in _ids(builder)


class TestCallExtraction:
    def test_function_call(self, builder):
        src = """<?php
class Svc {
    public function go(): void {
        doProjectWork();
    }
}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:doProjectWork" in targets

    def test_method_call_on_obj(self, builder):
        src = """<?php
class Svc {
    public function go($user): void {
        $user->authenticate($token);
    }
}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        # $user->authenticate — the call regex grabs from the identifier forward.
        assert any("authenticate" in t for t in targets)

    def test_static_call(self, builder):
        src = """<?php
class Svc {
    public function go(): void {
        Helper::compute($x);
    }
}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:Helper::compute" in targets

    def test_control_flow_not_a_call(self, builder):
        src = """<?php
class Svc {
    public function go(): void {
        if (ready()) {
            tick();
        }
    }
}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:ready" in targets
        assert "REF:tick" in targets

    def test_line_comment_stripped(self, builder):
        src = """<?php
class Svc {
    public function go(): void {
        // fake(arg);
        real();
    }
}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets

    def test_block_comment_stripped(self, builder):
        src = """<?php
class Svc {
    /**
     * @param string $x fake($x) inside a docblock
     */
    public function go(string $x): void {
        real();
    }
}
"""
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets
