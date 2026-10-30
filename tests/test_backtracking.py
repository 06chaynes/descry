"""A long run of whitespace is taken once by each parser pattern that meets
it, not split between the parts that can take it, which made the match time
cubic in the run's length; and a long file of one line repeated is read in
time linear in its length."""

import time

import pytest

from descry import (
    dotnet_parser,
    go_parser,
    java_parser,
    ruby_parser,
    rust_parser,
    ts_parser,
)
from descry.clang_parser import _RE_TYPE
from descry.dart_parser import _RE_CLOSURE, _RE_LOCAL_FUNCTION, DartParser
from descry.generate import CodeGraphBuilder

_GO_FIELD = next(r for r in go_parser._DECLARATIONS if "(?!(?:return" in r.pattern)


def _seconds(run):
    """The least CPU time of three calls of `run`."""

    def once():
        start = time.process_time()
        run()
        return time.process_time() - start

    return min(once() for _ in range(3))


def assert_linear(run, n):
    """`run(n)` takes under ten times the CPU time of `run(n // 4)`: time
    linear in `n` gives four times, quadratic sixteen."""
    quarter = _seconds(lambda: run(n // 4))
    assert _seconds(lambda: run(n)) < 10 * max(quarter, 0.05)


# 3,000 characters took seconds to minutes where the time was cubic, and
# 20,000 seconds where it was quadratic.
@pytest.mark.parametrize(
    ("find", "text", "length"),
    [
        (_RE_LOCAL_FUNCTION.match, lambda n: " " * n + "!", 3000),
        (DartParser.binding_re.search, lambda n: "const" + " " * n, 3000),
        (java_parser._RE_METHOD.match, lambda n: "final" + " " * n, 3000),
        (java_parser._RE_FIELD.match, lambda n: "final" + " " * n, 3000),
        (ts_parser._TS_IMPORT_RE.findall, lambda n: "import" + " " * n + "x;\n", 3000),
        (rust_parser._RUST_ARM_RE.search, lambda n: " " * n + "!", 3000),
        (java_parser._RE_METHOD.match, lambda n: "public <T>" + " " * n, 3000),
        (java_parser._RE_METHOD.match, lambda n: "public @A" + " " * n, 3000),
        (java_parser._RE_ENUM_CONSTANT.match, lambda n: "A" + " " * n + ")", 3000),
        (rust_parser._RUST_BINDER_RES[1].findall, lambda n: "for" + " " * n, 3000),
        (rust_parser._RUST_FOR_RE.search, lambda n: "    for" + " " * n + "x", 3000),
        (rust_parser._RUST_IF_LET_RE.search, lambda n: "while let" + " " * n, 3000),
        (rust_parser._RUST_MATCH_RE.search, lambda n: "match" + " " * n + "a b", 3000),
        (ruby_parser._RE_INLINE_BODY.match, lambda n: "x" + "\r\n" * n + ")", 1500),
        (
            dotnet_parser._RE_MEMBER.match,
            lambda n: "partial unsafe" + " " * n + ")",
            20000,
        ),
        (
            dotnet_parser._RE_PARAMETER.search,
            lambda n: "scoped readonly" + " " * n + "x;",
            20000,
        ),
        (_GO_FIELD.search, lambda n: "  chan a" + " " * n + "x(", 20000),
        (ruby_parser._RE_ALIAS.search, lambda n: "alias" + " " * n + ")", 20000),
        (ruby_parser._RE_MIXIN.match, lambda n: "  prepend" + " " * n + "=", 20000),
        (_RE_TYPE.fullmatch, lambda n: " " * n + ")", 20000),
        (
            java_parser._RE_METHOD.match,
            lambda n: "private final a" + " " * n + "b",
            20000,
        ),
        (
            java_parser._RE_FIELD.match,
            lambda n: "public static a" + " " * n + "b(",
            20000,
        ),
        (ruby_parser._RE_ENDLESS.match, lambda n: " " * n + "x", 60000),
        (ruby_parser._RE_INLINE_BODY.match, lambda n: "end" + "\n" * n + "x", 60000),
        (dotnet_parser._RE_PRIMARY_CONSTRUCTOR.match, lambda n: " " * n + "x", 60000),
        (
            dotnet_parser._RE_PARAMETER.findall,
            lambda n: "readonly" + "\t\n " * n + ")",
            20000,
        ),
        (_RE_CLOSURE.findall, lambda n: "\n" * n, 60000),
        (ts_parser._TS_IMPORT_RE.findall, lambda n: "\n" * n, 60000),
        (ts_parser._TS_DEFAULT_EXPORT_RE.findall, lambda n: "\u2028" * n, 60000),
        (rust_parser._RUST_USE_RE.findall, lambda n: "\n" * n, 60000),
    ],
    ids=[
        "dart-local-function",
        "dart-binding",
        "java-method",
        "java-field",
        "ts-import",
        "rust-arm",
        "java-method-type-parameters",
        "java-method-annotation",
        "java-enum-constant",
        "rust-for-binder",
        "rust-for",
        "rust-if-let",
        "rust-match",
        "ruby-inline-body",
        "dotnet-member",
        "dotnet-parameter",
        "go-field",
        "ruby-alias",
        "ruby-mixin",
        "clang-type",
        "java-return-type",
        "java-field-type",
        "ruby-endless",
        "ruby-inline-body-newlines",
        "dotnet-primary-constructor",
        "dotnet-parameter-mixed-run",
        "dart-closure-blank-lines",
        "ts-import-blank-lines",
        "ts-default-export-line-separators",
        "rust-use-blank-lines",
    ],
)
def test_a_whitespace_run_is_taken_once(find, text, length):
    assert_linear(lambda n: find(text(n)), length)


@pytest.mark.parametrize(
    ("name", "source", "count"),
    [
        ("m.js", lambda n: "const b = { a: g() };\n" * n, 10000),
        ("m.js", lambda n: "let b = 1;\n" + "b = g();\nb.x();\n" * n, 5000),
        (
            "m.js",
            lambda n: (
                "function f(xs) {\n" + "  for (const x of xs) { x.a(); }\n" * n + "}\n"
            ),
            7500,
        ),
        ("m.rb", lambda n: "b = 2\n" * n, 10000),
        ("m.rb", lambda n: "a = []\n" + "a <<-b\n" * n, 30000),
        ("m.dart", lambda n: "void main() {\n" + "  var b = 2;\n" * n + "}\n", 10000),
        ("m.py", lambda n: "def f():\n" + "    b = B()\n    b.x()\n" * n, 10000),
        (
            "m.py",
            lambda n: "def f(xs):\n" + "    for x in xs:\n        x.a()\n" * n,
            5000,
        ),
        (
            "M.java",
            lambda n: (
                "class M {\n  void m() {\n"
                + "    B b = new B();\n    b.x();\n" * n
                + "  }\n}\n"
            ),
            5000,
        ),
        (
            "M.cs",
            lambda n: (
                "class M {\n  void N() {\n"
                + "    var b = new B();\n    b.X();\n" * n
                + "  }\n}\n"
            ),
            4000,
        ),
        (
            "M.cs",
            lambda n: "".join(
                f"class C{i} {{ int M() {{ return {i}; }} }}\n" for i in range(n)
            ),
            1500,
        ),
        (
            "m.go",
            lambda n: (
                "package m\nfunc f() error {\n"
                + "\tx, err := g()\n\tif err != nil {\n\t\treturn err\n\t}\n\tx.A()\n"
                * n
                + "\treturn nil\n}\n"
            ),
            5000,
        ),
        (
            "m.php",
            lambda n: (
                "<?php\nfunction f($xs) {\n"
                + "    foreach ($xs as $x) { $x->a(); }\n" * n
                + "}\n"
            ),
            4000,
        ),
        (
            "src/lib.rs",
            lambda n: "".join(
                f"fn f{i}<T: Clone>(a: T) -> T {{ a.clone() }}\n" for i in range(n)
            ),
            6000,
        ),
        (
            "m.js",
            lambda n: (
                "function f(v) {\n  let b;\n"
                + "  switch (v) { case 1: b = g(); break; default: b = h(); }\n  b.x();\n"
                * n
                + "}\n"
            ),
            2500,
        ),
        (
            "src/lib.rs",
            lambda n: (
                "fn f(v: Option<B>) {\n"
                + "    match v { Some(b) => b.a(), None => () }\n" * n
                + "}\n"
            ),
            5000,
        ),
    ],
    ids=[
        "js-rebinds",
        "js-reassignments",
        "js-for-of",
        "ruby-rebinds",
        "ruby-shifts",
        "dart-rebinds",
        "python-rebinds",
        "python-loops",
        "java-rebinds",
        "dotnet-rebinds",
        "dotnet-types",
        "go-errors",
        "php-loops",
        "rust-generics",
        "js-switch-reassignments",
        "rust-matches",
    ],
)
def test_a_repeated_line_is_read_in_linear_time(name, source, count, tmp_path):
    """A line that binds a name again, loops over it, matches it, or
    declares a type or a generic function costs the same time however many
    came before it, where four times the lines took sixteen times as long,
    minutes in all."""

    def reading(lines):
        root = tmp_path / str(lines)
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(source(lines))
        builder = CodeGraphBuilder(root)
        builder.process_directory()
        builder.resolve_references()

    assert_linear(reading, count)


def test_the_patterns_still_match_source():
    binding = DartParser.binding_re.search("  final Map<String, int> counts = {};")
    method = java_parser._RE_METHOD.match("  public static   int  size(")
    field = java_parser._RE_FIELD.match("  private   final int  count;")
    arm = rust_parser._RUST_ARM_RE.search("        Some(x) if x > 1 => x,")

    assert _RE_LOCAL_FUNCTION.match("  Future<int> load(")
    assert binding["name"] == "counts"
    assert method["name"] == "size"
    assert field["name"] == "count"
    assert ts_parser._TS_IMPORT_RE.findall("import  { a }  from 'm';") == [
        ("{ a }", "m")
    ]
    assert arm["pattern"] == "Some(x)"
