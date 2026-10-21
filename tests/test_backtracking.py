"""A long run of whitespace is taken once by each parser pattern that meets
it, not split between the parts that can take it, which made the match time
cubic in the run's length; and a long file of one line repeated is read in
time linear in its length."""

import time

import pytest

from descry import dotnet_parser, generate, go_parser, java_parser, ruby_parser
from descry.clang_parser import _RE_TYPE
from descry.dart_parser import _RE_CLOSURE, _RE_LOCAL_FUNCTION, DartParser
from descry.generate import CodeGraphBuilder

_GO_FIELD = next(r for r in go_parser._DECLARATIONS if "(?!(?:return" in r.pattern)

# 3,000 characters took seconds to minutes where the time was cubic, and
# 20,000 seconds where it was quadratic; each now takes milliseconds.
_RUN = " " * 3000
_LONG_RUN = " " * 20000


@pytest.mark.parametrize(
    ("find", "text"),
    [
        (_RE_LOCAL_FUNCTION.match, _RUN + "!"),
        (DartParser.binding_re.search, "const" + _RUN),
        (java_parser._RE_METHOD.match, "final" + _RUN),
        (java_parser._RE_FIELD.match, "final" + _RUN),
        (generate._TS_IMPORT_RE.findall, "import" + _RUN + "x;\n"),
        (generate._RUST_ARM_RE.search, _RUN + "!"),
        (java_parser._RE_METHOD.match, "public <T>" + _RUN),
        (java_parser._RE_METHOD.match, "public @A" + _RUN),
        (java_parser._RE_ENUM_CONSTANT.match, "A" + _RUN + ")"),
        (generate._RUST_BINDER_RES[1].findall, "for" + _RUN),
        (generate._RUST_FOR_RE.search, "    for" + _RUN + "x"),
        (generate._RUST_IF_LET_RE.search, "while let" + _RUN),
        (generate._RUST_MATCH_RE.search, "match" + _RUN + "a b"),
        (ruby_parser._RE_INLINE_BODY.match, "x" + "\r\n" * 1500 + ")"),
        (dotnet_parser._RE_MEMBER.match, "partial unsafe" + _LONG_RUN + ")"),
        (dotnet_parser._RE_PARAMETER.search, "scoped readonly" + _LONG_RUN + "x;"),
        (_GO_FIELD.search, "  chan a" + _LONG_RUN + "x("),
        (ruby_parser._RE_ALIAS.search, "alias" + _LONG_RUN + ")"),
        (ruby_parser._RE_MIXIN.match, "  prepend" + _LONG_RUN + "="),
        (_RE_TYPE.fullmatch, _LONG_RUN + ")"),
        (java_parser._RE_METHOD.match, "private final a" + _LONG_RUN + "b"),
        (java_parser._RE_FIELD.match, "public static a" + _LONG_RUN + "b("),
        (ruby_parser._RE_ENDLESS.match, _LONG_RUN * 3 + "x"),
        (ruby_parser._RE_INLINE_BODY.match, "end" + "\n" * 60000 + "x"),
        (dotnet_parser._RE_PRIMARY_CONSTRUCTOR.match, _LONG_RUN * 3 + "x"),
        (dotnet_parser._RE_PARAMETER.findall, "readonly" + "\t\n " * 20000 + ")"),
        (_RE_CLOSURE.findall, "\n" * 60000),
        (generate._TS_IMPORT_RE.findall, "\n" * 60000),
        (generate._TS_DEFAULT_EXPORT_RE.findall, "\u2028" * 60000),
        (generate._RUST_USE_RE.findall, "\n" * 60000),
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
def test_a_whitespace_run_is_taken_once(find, text):
    start = time.perf_counter()

    find(text)

    assert time.perf_counter() - start < 1


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("m.js", "const b = { a: g() };\n" * 20000),
        ("m.js", "let b = 1;\n" + "b = g();\nb.x();\n" * 10000),
        (
            "m.js",
            "function f(xs) {\n" + "  for (const x of xs) { x.a(); }\n" * 15000 + "}\n",
        ),
        ("m.rb", "b = 2\n" * 20000),
        ("m.rb", "a = []\n" + "a <<-b\n" * 60000),
        ("m.dart", "void main() {\n" + "  var b = 2;\n" * 20000 + "}\n"),
        ("m.py", "def f():\n" + "    b = B()\n    b.x()\n" * 20000),
        ("m.py", "def f(xs):\n" + "    for x in xs:\n        x.a()\n" * 10000),
        (
            "M.java",
            "class M {\n  void m() {\n"
            + "    B b = new B();\n    b.x();\n" * 10000
            + "  }\n}\n",
        ),
        (
            "M.cs",
            "class M {\n  void N() {\n"
            + "    var b = new B();\n    b.X();\n" * 8000
            + "  }\n}\n",
        ),
        (
            "M.cs",
            "".join(
                f"class C{i} {{ int M() {{ return {i}; }} }}\n" for i in range(3000)
            ),
        ),
        (
            "m.go",
            "package m\nfunc f() error {\n"
            + "\tx, err := g()\n\tif err != nil {\n\t\treturn err\n\t}\n\tx.A()\n"
            * 10000
            + "\treturn nil\n}\n",
        ),
        (
            "m.php",
            "<?php\nfunction f($xs) {\n"
            + "    foreach ($xs as $x) { $x->a(); }\n" * 8000
            + "}\n",
        ),
        (
            "src/lib.rs",
            "".join(
                f"fn f{i}<T: Clone>(a: T) -> T {{ a.clone() }}\n" for i in range(12000)
            ),
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
    ],
)
def test_a_repeated_line_is_read_in_linear_time(name, source, tmp_path):
    """A line that binds a name again, loops over it, declares a type or a
    generic function costs the same time however many came before it:
    seconds in all, where it took minutes."""
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text(source)
    start = time.perf_counter()

    builder = CodeGraphBuilder(tmp_path)
    builder.process_directory()
    builder.resolve_references()

    assert time.perf_counter() - start < 10


def test_the_patterns_still_match_source():
    binding = DartParser.binding_re.search("  final Map<String, int> counts = {};")
    method = java_parser._RE_METHOD.match("  public static   int  size(")
    field = java_parser._RE_FIELD.match("  private   final int  count;")
    arm = generate._RUST_ARM_RE.search("        Some(x) if x > 1 => x,")

    assert _RE_LOCAL_FUNCTION.match("  Future<int> load(")
    assert binding["name"] == "counts"
    assert method["name"] == "size"
    assert field["name"] == "count"
    assert generate._TS_IMPORT_RE.findall("import  { a }  from 'm';") == [
        ("{ a }", "m")
    ]
    assert arm["pattern"] == "Some(x)"
