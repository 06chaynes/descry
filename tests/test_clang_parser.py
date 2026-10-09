"""Unit tests for the C/C++ regex parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.clang_parser import ClangParser
from descry.generate import CodeGraphBuilder


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="src/foo.cpp"):
    ClangParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestIncludesAndDefines:
    def test_include_angle(self, builder):
        src = """
#include <stdio.h>
#include <vector>
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:stdio.h" in targets
        assert "MODULE:vector" in targets

    def test_include_quoted(self, builder):
        src = """
#include "myheader.h"
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:myheader.h" in targets

    def test_define_macro(self, builder):
        src = """
#define MAX_CONN 100
#define VERSION "1.0"
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/foo.cpp::MAX_CONN" in ids
        assert "FILE:src/foo.cpp::VERSION" in ids


class TestTypesAndNamespaces:
    def test_namespace(self, builder):
        src = """
namespace app {
    int x = 0;
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::app" in _ids(builder)

    def test_class_allman(self, builder):
        src = """
class Server
{
public:
    void start();
};
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Server" in _ids(builder)

    def test_struct(self, builder):
        src = """
struct Point { int x; int y; };
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Point" in _ids(builder)

    def test_enum_class(self, builder):
        src = """
enum class Color { Red, Green, Blue };
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Color" in _ids(builder)


class TestFunctions:
    # Function / method extraction is intentionally deferred to scip-clang
    # (see ClangParser module docstring for why). Only calls are extracted.

    def test_control_flow_not_a_function(self, builder):
        src = """
int go() {
    if (ready) { tick(); }
    while (running) { work(); }
    return 0;
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:while" not in targets
        assert "REF:tick" in targets
        assert "REF:work" in targets


class TestCallsAndComments:
    def test_call_emitted(self, builder):
        src = """
void run() {
    doStuff();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:doStuff" in targets

    def test_method_call(self, builder):
        # Use a name that's not in STDLIB_FILTER ('start' matches Python's
        # re.Match.start() and is intentionally filtered cross-language).
        src = """
void run(Server* srv) {
    srv->launchWorker();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert any("launchWorker" in t for t in targets)

    def test_line_comment_stripped(self, builder):
        src = """
void run() {
    // fake(x);
    real();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets

    def test_block_comment_stripped(self, builder):
        src = """
/*
 * fake(x) inside a block comment
 */
void run() {
    real();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets


class TestDeclarationsAndInlineComments:
    def test_definition_is_not_a_call(self, builder):
        """Defining a function does not call it."""
        src = """
int compute(int x) {
    return scale(x);
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:scale"}

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
void run() {
    stepOne(); /* hidden(1);
    also_hidden(2); */ stepTwo();
    /* inline(3); */ stepThree();
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:stepOne", "REF:stepTwo", "REF:stepThree"}

    def test_macro_line_numbers(self, builder):
        src = """
#define FIRST 1
int x;
#define SECOND 2
""".strip()
        _parse(builder, src)
        linenos = {
            n["metadata"]["name"]: n["metadata"]["lineno"]
            for n in builder.nodes
            if n["metadata"].get("kind") == "macro"
        }
        assert linenos == {"FIRST": 1, "SECOND": 3}


def _types(builder):
    return {n["id"]: n["type"] for n in builder.nodes}


class TestTypeNamesThatDefineNothing:
    def test_forward_declaration_gets_no_node_and_no_body(self, builder):
        """`class Widget;` ends before any `{`: the function after it keeps
        its own node and the call in its body."""
        src = """
class Widget;
struct Opaque;
void run() {
    start();
}
""".strip()
        _parse(builder, src)
        assert _types(builder) == {
            "FILE:src/foo.cpp": "File",
            "FILE:src/foo.cpp::run": "Function",
        }
        calls = [
            (e["source"], e["target"])
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        assert calls == [("FILE:src/foo.cpp::run", "REF:start")]

    def test_definition_after_its_forward_declaration_keeps_its_span(self, builder):
        src = """
class Widget;

class Widget {
    void draw();
};
""".strip()
        _parse(builder, src)
        (widget,) = [n["metadata"] for n in builder.nodes if n["type"] == "Class"]
        assert (widget["lineno"], widget["end_lineno"]) == (3, 5)

    def test_struct_typed_variable_is_not_a_class(self, builder):
        """A variable declared with `struct` is a use of the type; the calls
        around it are still recorded."""
        src = """
int size(void) {
    struct stat st;
    if (probe(&st)) {
        return 1;
    }
    struct point origin = {0, 0};
    return finish(origin);
}
""".strip()
        _parse(builder, src)
        assert _types(builder) == {
            "FILE:src/foo.cpp": "File",
            "FILE:src/foo.cpp::size": "Function",
        }
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:probe", "REF:finish"}

    def test_struct_return_type_and_parameter_are_not_classes(self, builder):
        src = """
struct node *find(struct list *items) {
    return lookup(items);
}

int scan(int fd,
         struct stat *st) {
    return check(fd, st);
}
""".strip()
        _parse(builder, src)
        types = _types(builder)
        assert "Class" not in types.values()
        assert types["FILE:src/foo.cpp::find"] == "Function"

    def test_definition_heads_are_classes(self, builder):
        src = """
class Shape final : public Base {
};
template <> struct hash<Key *> {
};
struct Point { int x; int y; };
enum class Color : int { Red };
""".strip()
        _parse(builder, src)
        names = [n["metadata"]["name"] for n in builder.nodes if n["type"] == "Class"]
        assert names == ["Shape", "hash", "Point", "Color"]

    def test_brace_in_a_comment_does_not_open_the_class(self, builder):
        """The class closes at its own `}`; without a `{` of its own it
        defines nothing."""
        src = """
class Gap // {
{
    void m();
};
class Next {
};
class Tail // {
""".strip()
        _parse(builder, src)
        spans = {
            n["id"]: (n["metadata"]["lineno"], n["metadata"]["end_lineno"])
            for n in builder.nodes
            if n["type"] == "Class"
        }
        assert spans == {
            "FILE:src/foo.cpp::Gap": (1, 4),
            "FILE:src/foo.cpp::Next": (5, 6),
        }

    def test_string_text_is_neither_braces_nor_calls(self, builder):
        """A brace or a `name(` inside a string literal is text."""
        src = """
class Svc {
    void open() {
        const char *a = "}}";
        const char *b = "see http://example.com/{id} or call hidden(1)";
        real(a);
    }
    void next() {
    }
};
""".strip()
        _parse(builder, src)
        assert "FILE:src/foo.cpp::Svc::next" in _ids(builder)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:real"}


def _spans(builder, node_type):
    return {
        n["id"].partition("::")[2]: (
            n["metadata"]["lineno"],
            n["metadata"]["end_lineno"],
        )
        for n in builder.nodes
        if n["type"] == node_type
    }


def _call_pairs(builder):
    return [
        (e["source"].partition("::")[2], e["target"][4:])
        for e in builder.edges
        if e["relation"] == "CALLS"
    ]


class TestFunctionDefinitions:
    def test_layouts_of_a_definition(self, builder):
        """A body on the header's line, a signature over several lines, a
        return type on a line of its own and a brace on the next line."""
        src = """
static int is_valid(jv x) { return kind(x) != 0; }

int scan(int fd,
         struct stat *st) {
    return check(fd, st);
}

static Bigint *
Balloc(struct dtoa_context* C, int k)
{
    return 0;
}
""".strip()
        _parse(builder, src, "src/foo.c")
        assert _spans(builder, "Function") == {
            "is_valid": (1, 1),
            "scan": (3, 6),
            "Balloc": (8, 12),
        }
        assert _call_pairs(builder) == [("is_valid", "kind"), ("scan", "check")]

    def test_what_follows_the_parameters(self, builder):
        """Qualifiers, attribute macros, a trailing return type and member
        initialisers stand between the parameter list and the body."""
        src = """
class Reader : public Source {
 public:
  Reader(File* file, int size) : file_(file), pos_(0), end_(Limit(size)) {}
  ~Reader() override = default;
  int Peek() const noexcept override { return Fill(); }
  auto Next() -> Token final { return Token(); }
  void Flush() EXCLUSIVE_LOCKS_REQUIRED(mu_) { Sync(); }
  virtual void Close() = 0;
  void Reset();
  bool operator==(const Reader& other) const { return Same(other); }
  operator bool() const { return Valid(); }
};
""".strip()
        _parse(builder, src)
        assert _spans(builder, "Method") == {
            "Reader::<init>": (3, 3),
            "Reader::~Reader": (4, 4),
            "Reader::Peek": (5, 5),
            "Reader::Next": (6, 6),
            "Reader::Flush": (7, 7),
            "Reader::Close": (8, 8),
            "Reader::operator==": (10, 10),
            "Reader::operator bool": (11, 11),
        }
        assert _call_pairs(builder) == [
            ("Reader::<init>", "Limit"),
            ("Reader::Peek", "Fill"),
            ("Reader::Next", "Token"),
            ("Reader::Flush", "Sync"),
            ("Reader::operator==", "Same"),
            ("Reader::operator bool", "Valid"),
        ]

    def test_out_of_class_definition_keeps_its_class(self, builder):
        """`Type::name` is a member of `Type`: of the class the file
        declares, else of a node that holds what is defined for it."""
        src = """
namespace db {
class Table {
  int Size();
};
int Table::Size() { return Count(); }
Status Writer::Sync() {
  return Flush();
}
Writer::Writer(int fd) : fd_(fd) {}
Writer::~Writer() { Close(); }
template <typename Key>
int List<Key>::Node::Height() const { return 1; }
}
""".strip()
        _parse(builder, src)
        assert _spans(builder, "Method") == {
            "db::Table::Size": (5, 5),
            "db::Writer::Sync": (6, 8),
            "db::Writer::<init>": (9, 9),
            "db::Writer::~Writer": (10, 10),
            "db::List::Node::Height": (11, 12),
        }
        kinds = {
            n["id"].partition("::")[2]: n["metadata"].get("end_lineno")
            for n in builder.nodes
            if n["type"] == "Class"
        }
        assert kinds == {
            "db": 13,
            "db::Table": 4,
            "db::Writer": None,
            "db::List": None,
            "db::List::Node": None,
        }
        assert _call_pairs(builder) == [
            ("db::Table::Size", "Count"),
            ("db::Writer::Sync", "Flush"),
            ("db::Writer::~Writer", "Close"),
        ]

    def test_function_of_a_namespace_is_a_function(self, builder):
        """One in an unnamed namespace, like a `static` one, is known to
        its file only; `extern "C"` adds no scope."""
        src = """
namespace a::b {
int Twice(int x) { return 2 * x; }
namespace {
int Hidden() { return 0; }
}
}
extern "C" {
static int local(void) { return 0; }
int exported(void) { return local(); }
}
""".strip()
        _parse(builder, src)
        linkage = {
            n["id"].partition("::")[2]: n["metadata"].get("linkage")
            for n in builder.nodes
            if n["type"] == "Function"
        }
        assert linkage == {
            "a::b::Twice": None,
            "a::b::Hidden": "internal",
            "local": "internal",
            "exported": None,
        }
        assert not _spans(builder, "Method")

    def test_macro_invocation_with_a_body_defines(self, builder):
        """Its node is named as it is written, and it is a member of the
        type an argument names; one that wraps the head of a function
        defines that function."""
        src = """
class StoreTest : public testing::Test {
 public:
  void Fill() {}
};
TEST_F(StoreTest, Empty) {
  Fill();
}
TEST(Format, Number) { Check(1); }
HEAP_EXPORT(void heap_init(struct heap* heap)) { clear(heap); }
""".strip()
        _parse(builder, src)
        assert _spans(builder, "Method") == {
            "StoreTest::Fill": (3, 3),
            "StoreTest::TEST_F(StoreTest, Empty)": (5, 7),
        }
        assert _spans(builder, "Function") == {
            "TEST(Format, Number)": (8, 8),
            "heap_init": (9, 9),
        }
        assert _call_pairs(builder) == [
            ("StoreTest::TEST_F(StoreTest, Empty)", "Fill"),
            ("TEST(Format, Number)", "Check"),
            ("heap_init", "clear"),
        ]


class TestTypeDefinitions:
    def test_export_macro_is_not_the_name(self, builder):
        src = """
class LEVELDB_EXPORT Slice {
};
struct __attribute__((packed)) Header {
};
class FMT_API Writer final : public Sink, private detail::Base<Writer> {
};
""".strip()
        _parse(builder, src)
        assert list(_spans(builder, "Class")) == ["Slice", "Header", "Writer"]
        bases = [e["target"] for e in builder.edges if e["relation"] == "INHERITS"]
        assert bases == ["REF:Sink", "REF:detail::Base"]

    def test_typedef_and_unnamed_types(self, builder):
        """A typedef names an unnamed struct; an unnamed union in a struct
        gives the struct its fields; an unnamed enum has no node."""
        src = """
typedef struct {
  int kind;
  union {
    double number;
    struct node* ptr;
  };
} value;
typedef struct list_s {
  struct list_s* next;
} list_t;
enum { RED, GREEN };
enum class Mode : int { Fast, Slow };
static struct { int a; } table[] = {{1}, {2}};
""".strip()
        _parse(builder, src, "src/foo.c")
        assert _spans(builder, "Class") == {
            "value": (1, 7),
            "list_s": (8, 10),
            "Mode": (12, 12),
        }
        assert builder.fields["FILE:src/foo.c::value"] == {
            "kind": "int",
            "number": "double",
            "ptr": "node",
        }
        assert "list_t" in builder.type_aliases


class TestWhatIsNotACall:
    def test_declarations_are_not_calls(self, builder):
        """A prototype, a member declaration, a variable declared with
        arguments and a cast call nothing."""
        src = """
block gen_const(block expr);
HEAP_EXPORT(void heap_init(struct heap* heap));
class Env {
  virtual Status NewFile(const std::string& name) = 0;
  Status Lock(int fd);
  Env(const Env&) = delete;
};
int run(int n) {
  Slice key(buffer, n);
  int size = int(n) + sizeof(key);
  extern int helper(int);
  return work(key);
}
""".strip()
        _parse(builder, src)
        assert _call_pairs(builder) == [("run", "work")]
        assert _spans(builder, "Method") == {"Env::NewFile": (4, 4)}

    def test_lambda_body_calls_from_its_function(self, builder):
        src = """
void each(Items* items) {
  auto visit = [&](Item* item) {
    record(item);
  };
  walk(items, visit);
}
""".strip()
        _parse(builder, src)
        assert _call_pairs(builder) == [("each", "record"), ("each", "walk")]


class TestPreprocessor:
    def test_alternatives_of_one_definition(self, builder):
        """Of the heads of one body only the first is read, as is nothing
        of an `#if 0`; a definition under each of two conditions is one
        node."""
        src = """
#ifdef _WIN32
int open_file(HANDLE h) {
#else
int open_file(int fd) {
#endif
  return probe();
}
#if 0
int dead(void) { return gone(); }
#endif
#ifdef A
int pick(void) { return first(); }
#else
int pick(void) { return second(); }
#endif
""".strip()
        _parse(builder, src, "src/foo.c")
        assert _spans(builder, "Function") == {"open_file": (2, 7), "pick": (12, 12)}
        assert _call_pairs(builder) == [
            ("open_file", "probe"),
            ("pick", "first"),
            ("pick", "second"),
        ]

    def test_macro_named_like_a_function(self, builder):
        """The function keeps the name; the macro moves aside."""
        src = """
#define yyrestart jq_yyrestart
void yyrestart(FILE* input) {
}
""".strip()
        _parse(builder, src, "src/foo.c")
        assert _types(builder) == {
            "FILE:src/foo.c": "File",
            "FILE:src/foo.c::macro_yyrestart": "Constant",
            "FILE:src/foo.c::yyrestart": "Function",
        }

    def test_macro_that_opens_a_namespace(self, builder):
        src = """
#define LIB_BEGIN_NAMESPACE \\
  namespace lib {           \\
  inline namespace v2 {
#define LIB_END_NAMESPACE }}
""".strip()
        _parse(builder, src)
        opens = {
            n["metadata"]["name"]: (
                n["metadata"].get("opens"),
                n["metadata"]["end_lineno"],
            )
            for n in builder.nodes
            if n["type"] == "Constant"
        }
        assert opens == {
            "LIB_BEGIN_NAMESPACE": (["lib", "v2"], 3),
            "LIB_END_NAMESPACE": (None, 4),
        }

    def test_generated_scanner_records_no_calls(self, builder):
        src = """
#line 2 "src/lexer.c"
/* A lexical scanner generated by flex */
int yylex_init(yyscan_t* scanner) {
  return yy_init_globals(*scanner);
}
""".strip()
        _parse(builder, src, "src/lexer.c")
        assert _spans(builder, "Function") == {"yylex_init": (3, 5)}
        assert _call_pairs(builder) == []

    def test_static_written_under_a_condition(self, builder):
        """A `static` only one preprocessor arm writes confers no linkage:
        the definition is external where the arm does not compile."""
        src = """
#ifndef WIN32
static
#endif
void fini(void) {}
static int always_local(void) { return 0; }
#ifdef A
static int pick(void) { return 1; }
#else
int pick(void) { return 2; }
#endif
#ifdef A
static int both(void) { return 1; }
#else
static int both(void) { return 2; }
#endif
""".strip()
        _parse(builder, src, "src/foo.c")
        linkage = {
            n["metadata"]["name"]: n["metadata"].get("linkage")
            for n in builder.nodes
            if n["type"] == "Function"
        }
        assert linkage == {
            "fini": None,
            "always_local": "internal",
            "pick": None,
            "both": "internal",
        }

    def test_a_function_head_in_a_macro_is_no_call(self, builder):
        """A definition's head written in a replacement text declares; it
        is not a call the macro makes."""
        src = """
#define FORMAT_AS(Type, Base)          \\
  struct formatter : Base {            \\
    FMT_CONSTEXPR auto format(Type v,  \\
        int ctx) const -> int {        \\
      return Base::format(v, ctx);     \\
    }                                  \\
  };
#define WRAP(name)                     \\
  static int wrap_##name(void) {       \\
    return name();                     \\
  }
""".strip()
        _parse(builder, src, "src/foo.h")
        calls = {
            (e["source"].partition("::")[2], e["target"][4:])
            for e in builder.edges
            if e["relation"] == "CALLS"
        }
        assert calls == {("FORMAT_AS", "Base::format")}

    def test_calls_in_a_macro_are_the_macros(self, builder):
        """The calls of a replacement text leave the macro's node. A call
        of a parameter, or of a member that a parameter names, is none."""
        src = """
#define RELEASE(handle, method, done) \\
  do {                                \\
    unref(handle);                    \\
    (handle)->method(handle);         \\
    done(handle);                     \\
  } while (0)
#define LIMIT compute(8)
""".strip()
        _parse(builder, src, "src/foo.h")
        assert _call_pairs(builder) == [("RELEASE", "unref"), ("LIMIT", "compute")]


class TestScopes:
    def test_type_defined_in_a_function(self, builder):
        """It is a member of the function, and the calls in its methods are
        theirs."""
        src = """
int Recover(Log* log) {
  struct Reporter : public Listener {
    int dropped;
    void Corruption(int bytes) { Count(bytes); }
  };
  struct stat st;
  Reporter reporter;
  return Read(log, &reporter);
}
""".strip()
        _parse(builder, src)
        assert _types(builder) == {
            "FILE:src/foo.cpp": "File",
            "FILE:src/foo.cpp::Recover": "Function",
            "FILE:src/foo.cpp::Recover::Reporter": "Class",
            "FILE:src/foo.cpp::Recover::Reporter::Corruption": "Method",
        }
        assert _call_pairs(builder) == [
            ("Recover::Reporter::Corruption", "Count"),
            ("Recover", "Read"),
        ]

    def test_parameters_declared_behind_the_head(self, builder):
        src = """
int add(a, b)
  int a;
  char *b;
{
  return twice(a) + *b;
}
""".strip()
        _parse(builder, src, "src/foo.c")
        assert _spans(builder, "Function") == {"add": (1, 6)}
        assert _call_pairs(builder) == [("add", "twice")]

    def test_enumerators(self, builder):
        """Those of a named enum are its members, those of an unnamed one
        members of the scope around it."""
        src = """
enum class Kind { kValue = 1, kDeletion = Next(kValue), kLast };
typedef enum { JV_NULL, JV_TRUE } jv_kind;
enum { MAX_DEPTH = 256, ERRNO_MAP(XX) };
""".strip()
        _parse(builder, src)
        constants = {
            n["id"].partition("::")[2]: (
                n["metadata"]["lineno"],
                n["metadata"].get("value_type"),
            )
            for n in builder.nodes
            if n["type"] == "Constant"
        }
        assert constants == {
            "Kind::kValue": (1, "Kind"),
            "Kind::kDeletion": (1, "Kind"),
            "Kind::kLast": (1, "Kind"),
            "jv_kind::JV_NULL": (2, "jv_kind"),
            "jv_kind::JV_TRUE": (2, "jv_kind"),
            "MAX_DEPTH": (3, None),
        }
