"""A long run of whitespace is taken once by each parser pattern that meets
it, not split between the parts that can take it, which made the match time
cubic in the run's length."""

import time

import pytest

from descry import dotnet_parser, generate, go_parser, java_parser, ruby_parser
from descry.clang_parser import _RE_TYPE
from descry.dart_parser import _RE_LOCAL_FUNCTION, DartParser

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
    ],
)
def test_a_whitespace_run_is_taken_once(find, text):
    start = time.perf_counter()

    find(text)

    assert time.perf_counter() - start < 1


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
