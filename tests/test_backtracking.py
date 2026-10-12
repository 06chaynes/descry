"""A long run of whitespace is taken once by each parser pattern that meets
it, not split between the parts that can take it, which made the match time
cubic in the run's length."""

import time

import pytest

from descry import generate, java_parser
from descry.dart_parser import _RE_LOCAL_FUNCTION, DartParser

# 3,000 characters took seconds to minutes before; each now takes milliseconds.
_RUN = " " * 3000


@pytest.mark.parametrize(
    ("find", "text"),
    [
        (_RE_LOCAL_FUNCTION.match, _RUN + "!"),
        (DartParser.binding_re.search, "const" + _RUN),
        (java_parser._RE_METHOD.match, "final" + _RUN),
        (java_parser._RE_FIELD.match, "final" + _RUN),
        (generate._TS_IMPORT_RE.findall, "import" + _RUN + "x;\n"),
        (generate._RUST_ARM_RE.search, _RUN + "!"),
    ],
    ids=[
        "dart-local-function",
        "dart-binding",
        "java-method",
        "java-field",
        "ts-import",
        "rust-arm",
    ],
)
def test_a_whitespace_run_is_taken_once(find, text):
    start = time.perf_counter()

    assert not find(text)

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
