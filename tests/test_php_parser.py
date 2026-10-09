"""Unit tests for the PHP regex parser."""

from __future__ import annotations

import re
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


class TestDeclarationsAndComments:
    def test_declaration_is_not_a_call(self, builder):
        """Declaring a method or function does not call it."""
        src = """
<?php
class Svc {
    public function syncOrders($x) {
        compute($x);
    }
}
function syncAll($y) {
    compute($y);
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:compute"}

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
<?php
class Svc {
    public function run() {
        stepOne(); /* hidden(1);
        also_hidden(2); */ stepTwo();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:stepOne", "REF:stepTwo"}

    def test_brace_in_a_comment_does_not_open_the_class(self, builder):
        """The class closes at its own `}`; without a `{` of its own it has
        no body."""
        src = """
<?php
class Gap // {
{
    public function m() { a(); }
}
class Next
{
}
class Tail # {
""".strip()
        _parse(builder, src)
        spans = {
            n["id"]: (n["metadata"]["lineno"], n["metadata"]["end_lineno"])
            for n in builder.nodes
            if n["type"] == "Class"
        }
        assert spans == {
            "FILE:src/Foo.php::Gap": (2, 5),
            "FILE:src/Foo.php::Next": (6, 8),
            "FILE:src/Foo.php::Tail": (9, 9),
        }

    def test_string_text_is_neither_braces_nor_calls(self, builder):
        """A brace or a `name(` inside a string literal is text."""
        src = """
<?php
class Svc {
    public function open() {
        $a = "}}";
        $b = 'see http://example.com/{id} or call hidden(1)';
        real($a);
    }
    public function next() {
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.php::Svc::next" in _ids(builder)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:real"}


def _nodes(builder):
    """``{ID without the file: (type, first line, last line)}``."""
    return {
        n["id"].partition("::")[2]: (
            n["type"],
            n["metadata"].get("lineno"),
            n["metadata"].get("end_lineno"),
        )
        for n in builder.nodes
        if n["type"] != "File"
    }


def _calls(builder):
    """``(source without the file, callee, origin)`` of each call edge. An
    origin names the call it starts at by its callee: `#Registry::get`."""

    def callee(edge):
        return edge["target"].removeprefix("REF:")

    return [
        (
            e["source"].partition("::")[2],
            callee(e),
            e["metadata"].get("origin")
            and re.sub(
                r"#(\d+)",
                lambda call: f"#{callee(builder.edges[int(call[1])])}",
                e["metadata"]["origin"],
            ),
        )
        for e in builder.edges
        if e["relation"] == "CALLS"
    ]


class TestDefinitions:
    def test_method_and_property_of_one_name_are_two_nodes(self, builder):
        src = """
<?php
class Chat {
    private bool $split;
    public function __construct(private int $retries) {
    }
    public function split(): self {
    }
    public function retries(): int {
    }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Chat": ("Class", 2, 10),
            "Chat::field_split": ("Constant", 3, None),
            "Chat::__construct": ("Method", 4, 5),
            "Chat::field_retries": ("Constant", 4, None),
            "Chat::split": ("Method", 6, 7),
            "Chat::retries": ("Method", 8, 9),
        }

    def test_members_a_modifier_or_one_line_does_not_bound(self, builder):
        """A constant without a modifier, an initialiser that runs on over
        lines, the cases of an enum, a function that returns a reference, a
        method without a modifier and a promoted constructor parameter."""
        src = """
<?php
class Table {
    const RFC5424e = 2;
    protected const SPEC = [
        0 => ['pipe', 'r'],
    ];
    private array $names = [
        'a',
    ];
    public function __construct(
        private readonly Store $store,
        int $size,
    ) {
    }
    function &rows(): array {
        static $cache = [];
    }
}
enum Level: int {
    case Debug = 100;
    case Info = 200;
    const DEFAULT = self::Debug;
}
interface Source {
    function read(int $bytes): string;
}
""".strip()
        _parse(builder, src)
        assert {name: node[0] for name, node in _nodes(builder).items()} == {
            "Table": "Class",
            "Table::RFC5424e": "Constant",
            "Table::SPEC": "Constant",
            "Table::names": "Constant",
            "Table::__construct": "Method",
            "Table::store": "Constant",
            "Table::rows": "Method",
            "Level": "Class",
            "Level::Debug": "Constant",
            "Level::Info": "Constant",
            "Level::DEFAULT": "Constant",
            "Source": "Class",
            "Source::read": "Method",
        }

    def test_block_ends_at_its_own_brace(self, builder):
        """A body whose `{` is on the line after the header ends at its
        `}`, however long it is; a declaration without a body ends with
        its header."""
        body = "\n".join("        step();" for _ in range(150))
        src = f"""
<?php
abstract class Job
{{
    public function run(
        int $times
    ): void
    {{
{body}
    }}

    abstract protected function prepare(
        array $options
    ): void;
}}
""".strip()
        _parse(builder, src)
        nodes = _nodes(builder)
        assert nodes["Job"] == ("Class", 2, 163)
        assert nodes["Job::run"] == ("Method", 4, 158)
        assert nodes["Job::prepare"] == ("Method", 160, 162)

    def test_text_that_runs_on_over_lines_is_not_code(self, builder):
        """A brace or a `name(` in a string, a heredoc or a nowdoc that
        spans lines neither closes the class nor is a call."""
        src = """
<?php
class Fixture {
    public function json() {
        return '{
            "note": "it\\'s } hidden(1)"
        }}}' . real(1);
    }
    public function script() {
        return <<<JS
            c.debug("}") // }
        JS . real(2);
    }
    public function raw() {
        $text = <<<'EOT'
            }} hidden(2) # not a comment
            EOT;
        return real(3);
    }
    # a comment's quote isn't a string
    public function last() {
    }
}
""".strip()
        _parse(builder, src)
        assert {name for name, node in _nodes(builder).items()} == {
            "Fixture",
            "Fixture::json",
            "Fixture::script",
            "Fixture::raw",
            "Fixture::last",
        }
        assert [(source, callee) for source, callee, _ in _calls(builder)] == [
            ("Fixture::json", "real"),
            ("Fixture::script", "real"),
            ("Fixture::raw", "real"),
        ]

    def test_anonymous_class_has_its_own_members(self, builder):
        src = """
<?php
class Factory {
    public function make(): Handler {
        return new class($this->level()) extends Handler {
            public function handle(): bool {
                return $this->accepts();
            }
        };
    }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Factory": ("Class", 2, 10),
            "Factory::make": ("Method", 3, 9),
            "Factory::make::class@4": ("Class", 4, 8),
            "Factory::make::class@4::handle": ("Method", 5, 7),
        }
        assert _calls(builder) == [
            ("Factory::make", "class@4", "=class@4"),
            ("Factory::make", "$this->level", "self.level"),
            ("Factory::make::class@4::handle", "$this->accepts", "self.accepts"),
        ]

    def test_use_imports_at_file_level_and_gives_a_trait_in_a_class(self, builder):
        src = """
<?php
namespace App;

use Lib\\{Clock, Json as J};
use function Lib\\now;

class Job extends Base implements J, \\Countable {
    use Logs, Retries {
        Logs::write as protected;
    }
}
""".strip()
        _parse(builder, src)
        edges = [
            (e["relation"], e["target"])
            for e in builder.edges
            if e["relation"] in ("IMPORTS", "INHERITS")
        ]
        assert edges == [
            ("IMPORTS", "MODULE:Lib\\Clock"),
            ("IMPORTS", "MODULE:Lib\\Json"),
            ("IMPORTS", "MODULE:Lib\\now"),
            ("INHERITS", "REF:::App::Logs"),
            ("INHERITS", "REF:::App::Retries"),
            ("INHERITS", "REF:::App::Base"),
            ("INHERITS", "REF:::Lib::Json"),
            ("INHERITS", "REF:::Countable"),
        ]


class TestCallsAndReceivers:
    def test_call_is_made_by_the_member_it_is_in(self, builder):
        """A call in a method or function body leaves that member; one in
        a property's initial value leaves the class."""
        src = """
<?php
class Client {
    private $helper = [Helper::KEY => Helper::create()];

    public function run(): void
    {
        Helper::work();
    }
}
function boot(): void
{
    Helper::prepare();
}
""".strip()
        _parse(builder, src)
        assert [(source, callee) for source, callee, _ in _calls(builder)] == [
            ("Client", "Helper::create"),
            ("Client::run", "Helper::work"),
            ("boot", "Helper::prepare"),
        ]

    def test_receiver_is_kept(self, builder):
        """Each call is recorded with what it is made on, and a function
        with the namespace it is looked up in first."""
        src = """
<?php
namespace App;

use function Lib\\now;

class Svc extends Base {
    public function go($user, Mailer $mailer) {
        $user->authenticate();
        $user?->profile->refresh();
        $mailer->send();
        $this->queue->push(now());
        self::log();
        static::$instance->flush();
        parent::go($user);
        Registry::get('x')->close();
        (new Report($user))->render();
        $this->builder
            ->from('a')
            ->build();
        trim(\\strtolower(Util\\name()));
        $callback($user);
        $this->$method();
    }
}
""".strip()
        _parse(builder, src)
        assert [(callee, origin) for _, callee, origin in _calls(builder)] == [
            ("$user->authenticate", "?.authenticate"),
            ("$user?->profile->refresh", "?.refresh"),
            ("$mailer->send", "::App::Mailer().send"),
            ("$this->queue->push", "self.queue.push"),
            ("\\Lib\\now", None),
            ("self::log", "self.log"),
            ("static::$instance->flush", "self.instance.flush"),
            ("parent::go", "super().go"),
            ("Registry::get", "::App::Registry.get"),
            ("close", "#Registry::get.close"),
            ("Report", "::App::Report"),
            ("render", "#Report.render"),
            ("from", "self.builder.from"),
            ("build", "#from.build"),
            ("App\\trim", None),
            ("\\strtolower", None),
            ("\\App\\Util\\name", None),
        ]

    def test_declarations_type_a_variable_until_it_is_bound_again(self, builder):
        src = """
<?php
class Svc {
    public function go(array $items) {
        $job = new Job();
        $job->run();
        $job = $items[0];
        $job->run();
        foreach ($items as $job) {
            $job->run();
        }
        if ($job instanceof Timed) {
            $job->elapsed();
        }
        $job->run();
        try {
            $job->run();
        } catch (Failure | Timeout $job) {
            $job->report();
        } catch (\\Throwable $job) {
            $job->report();
        }
    }
}
""".strip()
        _parse(builder, src)
        assert [origin for _, _, origin in _calls(builder)] == [
            "::Job",
            "::Job().run",
            "?.run",
            "array().*.run",
            "::Timed().elapsed",
            "array().*.run",
            "array().*.run",
            "?.report",
            "::Throwable().report",
        ]
