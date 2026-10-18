"""Unit tests for the C# regex parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from descry.dotnet_parser import DotnetParser
from descry.generate import CodeGraphBuilder


@pytest.fixture
def builder(tmp_path):
    return CodeGraphBuilder(tmp_path)


def _parse(builder, source, rel="src/Foo.cs"):
    DotnetParser(builder).parse(Path(rel), rel, source)
    return builder


def _ids(builder):
    return [n["id"] for n in builder.nodes]


class TestNamespaceAndUsing:
    def test_block_namespace(self, builder):
        src = """
namespace App.Foo
{
    public class Bar {}
}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.cs")
        assert file_node.get("metadata", {}).get("dotnet_namespace") == "App.Foo"

    def test_file_scoped_namespace(self, builder):
        src = """
namespace App.Foo;

public class Bar {}
""".strip()
        _parse(builder, src)
        file_node = next(n for n in builder.nodes if n["id"] == "FILE:src/Foo.cs")
        assert file_node.get("metadata", {}).get("dotnet_namespace") == "App.Foo"

    def test_using(self, builder):
        src = """
using System;
using System.Collections.Generic;
using static System.Math;

namespace App;

public class X {}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "IMPORTS"}
        assert "MODULE:System" in targets
        assert "MODULE:System.Collections.Generic" in targets
        assert "MODULE:System.Math" in targets


class TestTypesAndMethods:
    def test_class_allman(self, builder):
        src = """
public class Service
{
    public void Run()
    {
        DoWork();
    }
}
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/Foo.cs::Service" in ids
        assert "FILE:src/Foo.cs::Service::Run" in ids

    def test_interface(self, builder):
        src = """
public interface IRunnable
{
    void Run();
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::IRunnable" in _ids(builder)

    def test_struct(self, builder):
        src = """
public struct Point
{
    public int X;
    public int Y;
    public int Sum() => X + Y;
}
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/Foo.cs::Point" in ids
        assert "FILE:src/Foo.cs::Point::Sum" in ids

    def test_record(self, builder):
        src = """
public record PersonRecord(string Name, int Age);
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::PersonRecord" in _ids(builder)

    def test_enum(self, builder):
        src = """
public enum Color { Red, Green, Blue }
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Color" in _ids(builder)

    def test_async_method(self, builder):
        src = """
public class Svc
{
    public async Task<int> ComputeAsync()
    {
        return 42;
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Svc::ComputeAsync" in _ids(builder)

    def test_generic_method(self, builder):
        src = """
public class Box
{
    public static T Identity<T>(T input) => input;
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Box::Identity" in _ids(builder)


class TestPropertiesAndFields:
    def test_property(self, builder):
        src = """
public class User
{
    public string Name { get; set; }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::User::Name" in _ids(builder)

    def test_property_with_initializer_and_field(self, builder):
        src = """
public class User
{
    public int Age { get; private set; } = 3;
    private string _nick;
}
""".strip()
        _parse(builder, src)
        ids = _ids(builder)
        assert "FILE:src/Foo.cs::User::Age" in ids
        assert "FILE:src/Foo.cs::User::_nick" in ids

    def test_const(self, builder):
        src = """
public class Config
{
    public const int MaxConnections = 100;
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Config::MaxConnections" in _ids(builder)


class TestCalls:
    def test_call_emitted(self, builder):
        src = """
public class Svc
{
    public void Go()
    {
        DoStuff();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:DoStuff" in targets

    def test_method_on_object(self, builder):
        src = """
public class Svc
{
    public void Go(User u)
    {
        u.Authenticate(token);
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:u.Authenticate" in targets

    def test_control_flow_not_a_call(self, builder):
        src = """
public class Svc
{
    public void Go()
    {
        if (ready) { tick(); }
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:if" not in targets
        assert "REF:tick" in targets

    def test_line_comment_stripped(self, builder):
        src = """
public class Svc
{
    public void Go()
    {
        // fake(x);
        real();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert "REF:fake" not in targets
        assert "REF:real" in targets


class TestCallChains:
    def test_every_call_of_a_chain_is_recorded(self, builder):
        src = """
public class Svc
{
    public void Go(Maker maker)
    {
        maker.Named("x").Assemble()
            .Ship();
    }
}
""".strip()
        _parse(builder, src)
        calls = [e["target"] for e in builder.edges if e["relation"] == "CALLS"]
        assert calls == ["REF:maker.Named", "REF:Assemble", "REF:Ship"]


class TestVbFilesSkipped:
    def test_vb_file_no_symbols_beyond_file(self, builder):
        src = """
Public Class Foo
    Public Sub Greet()
        Console.WriteLine("hi")
    End Sub
End Class
""".strip()
        DotnetParser(builder).parse(Path("src/Foo.vb"), "src/Foo.vb", src)
        ids = _ids(builder)
        # VB is SCIP-only — parser only emits the File node.
        assert ids == ["FILE:src/Foo.vb"]


class TestDeclarationsAndStrings:
    def test_code_in_an_interpolated_string_is_read(self, builder):
        """The `{...}` of `$"..."`, `$@"..."` and a raw string after `$`
        are code, a string in one is a string, `{{` is text and so is any
        `{...}` of another string or of a character literal."""
        src = '''
public class Foo
{
    public string Bar(char value)
    {
        Log($"a {Name()} b {(true ? "x}" : "y")} c", value == '$');
        Log($@"p {{q()}} {Path()} ""r}"" s", "plain {Not()}");
        return $$"""raw {Text()} {{Raw()}} t""";
    }

    public void Baz()
    {
        Done();
    }
}
'''.strip()
        _parse(builder, src)
        assert _calls(builder) == [
            ("Foo::Bar", "Log"),
            ("Foo::Bar", "Name"),
            ("Foo::Bar", "Log"),
            ("Foo::Bar", "Path"),
            ("Foo::Bar", "Raw"),
            ("Foo::Baz", "Done"),
        ]

    def test_declaration_is_not_a_call(self, builder):
        """Declaring a method does not call it."""
        src = """
public class Foo
{
    public void Bar(int x) {
        Compute(x);
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:Compute"}

    def test_comment_after_verbatim_string_stripped(self, builder):
        """A backslash does not escape the closing quote of @"..."."""
        src = """
public class Foo
{
    public void Bar() {
        var dir = @"C:\\temp\\"; // Hidden(dir)
        Compute(dir);
    }
}
""".strip().replace("\\\\", "\\")
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:Compute"}

    def test_mid_line_block_comment_stripped(self, builder):
        """Code inside a block comment that shares a line with code is skipped."""
        src = """
public class Foo
{
    public void Bar() {
        StepOne(); /* Hidden(1);
        AlsoHidden(2); */ StepTwo();
    }
}
""".strip()
        _parse(builder, src)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:StepOne", "REF:StepTwo"}

    def test_brace_in_a_comment_does_not_open_the_class(self, builder):
        """The class closes at its own `}`; without a `{` of its own it has
        no body."""
        src = """
public class Gap // {
{
    public void M() { A(); }
}
public class Next
{
}
public class Tail // {
""".strip()
        _parse(builder, src)
        spans = {
            n["id"]: (n["metadata"]["lineno"], n["metadata"]["end_lineno"])
            for n in builder.nodes
            if n["type"] == "Class"
        }
        assert spans == {
            "FILE:src/Foo.cs::Gap": (1, 4),
            "FILE:src/Foo.cs::Next": (5, 7),
            "FILE:src/Foo.cs::Tail": (8, 8),
        }

    def test_string_text_is_neither_braces_nor_calls(self, builder):
        """A brace or a `name(` inside a string literal is text, and a
        backslash ends no verbatim string early."""
        src = """
public class Svc
{
    public void Open() {
        var a = "}}";
        var b = "see http://example.com/{id} or call Hidden(1)";
        if (a == @"C:\\") { Real(a); }
    }
    public void Next() {
    }
}
""".strip()
        _parse(builder, src)
        assert "FILE:src/Foo.cs::Svc::Next" in _ids(builder)
        targets = {e["target"] for e in builder.edges if e["relation"] == "CALLS"}
        assert targets == {"REF:Real"}


def _nodes(builder):
    """``{ID below the file: (type, first line, last line)}`` of the nodes."""
    return {
        n["id"].partition("::")[2]: (
            n["type"],
            n["metadata"]["lineno"],
            n["metadata"]["end_lineno"],
        )
        for n in builder.nodes
        if n["type"] != "File"
    }


def _calls(builder):
    """``(source below the file, callee)`` of each call edge, in order."""
    return [
        (e["source"].partition("::")[2], e["target"].removeprefix("REF:"))
        for e in builder.edges
        if e["relation"] == "CALLS"
    ]


class TestMembers:
    def test_each_overload_is_a_node(self, builder):
        src = """
public class Log
{
    public void Write(string text)
    {
    }

    public void Write(string text, int level) => Write(text);

    public void Write<T>(T value)
    {
    }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Log": ("Class", 1, 12),
            "Log::Write": ("Method", 3, 5),
            "Log::overload2_Write": ("Method", 7, 7),
            "Log::overload3_Write": ("Method", 9, 11),
        }
        assert _calls(builder) == [("Log::overload2_Write", "Write")]

    def test_members_without_a_modifier(self, builder):
        """An interface member and a private member have no modifier; a
        declaration is no call of its own name."""
        src = """
public interface ISink
{
    void Emit(Event e);
    int Count { get; }
    ISink Next(
        int skip);
}

class Sink : ISink
{
    int _count;
    bool TryEmit(Event e, [NotNullWhen(true)] out string? error)
    {
        error = null;
        return true;
    }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "ISink": ("Class", 1, 7),
            "ISink::Emit": ("Method", 3, 3),
            "ISink::Count": ("Constant", 4, 4),
            "ISink::Next": ("Method", 5, 6),
            "Sink": ("Class", 9, 17),
            "Sink::_count": ("Constant", 11, 11),
            "Sink::TryEmit": ("Method", 12, 16),
        }
        assert _calls(builder) == []

    def test_constructors_operators_and_indexers(self, builder):
        src = """
public class Money
{
    static Money() { Zero = new Money(0); }
    public Money(int amount) : base()
    {
    }
    public Money(int amount, string unit) : this(amount) { }
    ~Money() { }
    public static Money operator +(Money a, Money b) => new Money(a.Amount);
    public static implicit operator int(Money m) => m.Amount;
    public int this[int index] => index;
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Money": ("Class", 1, 12),
            "Money::<clinit>": ("Method", 3, 3),
            "Money::<init>": ("Method", 4, 6),
            "Money::overload2_<init>": ("Method", 7, 7),
            "Money::~Money": ("Method", 8, 8),
            "Money::operator +": ("Method", 9, 9),
            "Money::operator int": ("Method", 10, 10),
            "Money::this[]": ("Constant", 11, 11),
        }
        assert _calls(builder) == [
            ("Money::<clinit>", "Money"),
            ("Money::operator +", "Money"),
        ]

    def test_type_without_a_body(self, builder):
        """A type that ends in `;` takes no brace of what follows it, and
        the parameters of a record are its properties."""
        src = """
public class Listener
{
    public readonly record struct Failure(
        string Message,
        Exception? Error);

    public List<Failure> Failures { get; } = [];
}
class Empty;
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Listener": ("Class", 1, 8),
            "Listener::Failure": ("Class", 3, 5),
            "Listener::Failure::<init>": ("Method", 3, 5),
            "Listener::Failure::Message": ("Constant", 3, 5),
            "Listener::Failure::Error": ("Constant", 3, 5),
            "Listener::Failures": ("Constant", 7, 7),
            "Empty": ("Class", 9, 9),
        }

    def test_bodies_end_where_they_close(self, builder):
        """A body whose `{` is on the next line, an expression body that
        runs over lines, and a property with accessors and an initializer."""
        src = """
public class Shape
{
    public int Area()
    {
        return Width * Height;
    }

    public int Twice() =>
        Area()
        + Area();

    public int Width
    {
        get { return Read(); }
        set { Store(value); }
    }

    public List<int> Sides { get; } = new List<int>
    {
        1,
    };

    static readonly Dictionary<string, int> Names = new()
    {
        ["a"] = Rank("a"),
    };
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Shape": ("Class", 1, 27),
            "Shape::Area": ("Method", 3, 6),
            "Shape::Twice": ("Method", 8, 10),
            "Shape::Width": ("Constant", 12, 16),
            "Shape::Sides": ("Constant", 18, 21),
            "Shape::Names": ("Constant", 23, 26),
        }
        assert _calls(builder) == [
            ("Shape::Twice", "Area"),
            ("Shape::Twice", "Area"),
            ("Shape::Width", "Read"),
            ("Shape::Width", "Store"),
            ("Shape::Sides", "List"),
            ("Shape", "Dictionary"),
            ("Shape", "Rank"),
        ]

    def test_local_function(self, builder):
        src = """
public class Tests
{
    public IEnumerable<object[]> Cases()
    {
        static object[] Case(int level, bool expected)
        {
            return Pack(level, expected);
        }
        int Twice(int n) => Double(n);

        yield return Case(Twice(1), true);
    }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Tests": ("Class", 1, 13),
            "Tests::Cases": ("Method", 3, 12),
            "Tests::Cases::Case": ("Function", 5, 8),
            "Tests::Cases::Twice": ("Function", 9, 9),
        }
        assert _calls(builder) == [
            ("Tests::Cases::Case", "Pack"),
            ("Tests::Cases::Twice", "Double"),
            ("Tests::Cases", "Case"),
            ("Tests::Cases", "Twice"),
        ]

    def test_nested_partial_and_generic_types(self, builder):
        """A nested type is a member of its type; a partial type declared
        again is the same type; `Pair<T>` is another type than `Pair`."""
        src = """
namespace App.Model
{
    public partial class Outer
    {
        enum Kind { One, Two = 2 }
        public class Inner { public void Run() { } }
        public delegate void Handler(object sender);
    }
    public partial class Outer
    {
        public void More() { }
    }
    public class Pair { }
    public class Pair<T> : Pair { }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Outer": ("Class", 3, 8),
            "Outer::Kind": ("Class", 5, 5),
            "Outer::Kind::One": ("Constant", 5, 5),
            "Outer::Kind::Two": ("Constant", 5, 5),
            "Outer::Inner": ("Class", 6, 6),
            "Outer::Inner::Run": ("Method", 6, 6),
            "Outer::Handler": ("Class", 7, 7),
            "Outer::More": ("Method", 11, 11),
            "Pair": ("Class", 13, 13),
            "overload2_Pair": ("Class", 14, 14),
        }
        outer = next(n for n in builder.nodes if n["id"].endswith("::Outer"))
        assert outer["metadata"]["namespace"] == "App.Model"
        assert outer["metadata"]["partial"]
        inherits = [
            (e["source"].partition("::")[2], e["target"])
            for e in builder.edges
            if e["relation"] == "INHERITS"
        ]
        assert inherits == [("overload2_Pair", "REF:Pair")]

    def test_fields_declared_together(self, builder):
        src = """
class Buffers
{
    object[] _zero = null!, _one = Make(1), _two;
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Buffers": ("Class", 1, 4),
            "Buffers::_zero": ("Constant", 3, 3),
            "Buffers::_one": ("Constant", 3, 3),
            "Buffers::_two": ("Constant", 3, 3),
        }


class TestAttributes:
    def test_attribute_applies_its_class(self, builder):
        """`[Name(...)]` on its own line is a call of NameAttribute by what
        it is applied to, and the doc comment above it is the member's."""
        src = """
[Serializable]
public class Store
{
    /// <summary>Saves.</summary>
    [Obsolete("old"), Audit.Tracked]
    [return: NotNull]
    public string Save([CallerMemberName] string caller = "") => Write(caller);
}
""".strip()
        _parse(builder, src)
        assert _calls(builder) == [
            ("Store", "SerializableAttribute"),
            ("Store::Save", "ObsoleteAttribute"),
            ("Store::Save", "Audit.TrackedAttribute"),
            ("Store::Save", "NotNullAttribute"),
            ("Store::Save", "Write"),
        ]
        save = next(n for n in builder.nodes if n["id"].endswith("::Save"))
        assert save["metadata"]["docstring"] == "<summary>Saves.</summary>"
        assert save["metadata"]["lineno"] == 7


class TestCallers:
    def test_call_is_made_by_the_member_it_is_in(self, builder):
        """A lambda's calls are its member's; a field initializer's are the
        type's."""
        src = """
public class Client
{
    readonly Helper _helper = Helper.Create();

    public Client()
    {
        Prepare();
    }

    public int Size => Measure();

    public void Run()
    {
        Each(item =>
        {
            Visit(item);
        });
    }
}
""".strip()
        _parse(builder, src)
        assert _calls(builder) == [
            ("Client", "Helper.Create"),
            ("Client::<init>", "Prepare"),
            ("Client::Size", "Measure"),
            ("Client::Run", "Each"),
            ("Client::Run", "Visit"),
        ]


class TestLiteralsAndDirectives:
    def test_string_over_lines_is_not_code(self, builder):
        src = '''
public class Texts
{
    const string Json = @"{
        ""call"": Hidden(1) }
    ";
    const string Raw = """
        class Ghost { void Nope() { Hidden(2); } }
        """;
    string Mail() => "user@" + Host() + "";
}
'''.strip()
        _parse(builder, src)
        assert sorted(_nodes(builder)) == [
            "Texts",
            "Texts::Json",
            "Texts::Mail",
            "Texts::Raw",
        ]
        assert _calls(builder) == [("Texts::Mail", "Host")]

    def test_first_branch_of_a_directive_is_read(self, builder):
        src = """
public class Writer
{
#if FEATURE_SPAN
    public void Write(Span<char> text)
#else
    public void Write(char[] text)
#endif
    {
        Flush();
    }
}
""".strip()
        _parse(builder, src)
        assert _nodes(builder) == {
            "Writer": ("Class", 1, 11),
            "Writer::Write": ("Method", 4, 10),
        }
        assert _calls(builder) == [("Writer::Write", "Flush")]

    def test_verbatim_identifier(self, builder):
        src = """
public class Bus
{
    public void Send(Event @event) => @event.Publish();
}
""".strip()
        _parse(builder, src)
        edge = next(e for e in builder.edges if e["relation"] == "CALLS")
        assert edge["target"] == "REF:event.Publish"
        assert edge["metadata"]["origin"] == "Event().Publish"


class TestDeclaredTypes:
    def test_member_metadata(self, builder):
        """What the resolver reads: the type of a field or property, what a
        method returns and assigns to its `out` arguments, what type an
        extension method is written for."""
        src = """
namespace App;

public static class Lookup
{
    public static Dictionary<string, Entry>? Index { get; }
    static Entry[] _all;
    public static bool TryFind(string key, out Entry? found, out int at) => false;
    public static T First<T>(this IEnumerable<T> items) => default;
    public static string Shout(this Entry entry) => "";
    public Task<Entry> LoadAsync() => null;
    public Entry Load() => null;
    public Text Load(int id) => null;
}
""".strip()
        _parse(builder, src)
        metadata = {
            n["id"].rpartition("::")[2]: n["metadata"]
            for n in builder.nodes
            if n["type"] != "File"
        }
        assert metadata["Lookup"]["namespace"] == "App"
        assert metadata["Index"]["value_type"] == "Dictionary<string,Entry>"
        assert metadata["_all"]["value_type"] == "System.Array<Entry>"
        assert metadata["TryFind"]["return_type"] == "bool, Entry, int"
        assert metadata["First"]["return_type"] == ""
        assert metadata["First"]["extends"] == "IEnumerable"
        assert metadata["Shout"]["extends"] == "Entry"
        assert metadata["LoadAsync"]["return_type"] == "Task<Entry>"
        # Overloads that return different types tell no type.
        assert metadata["Load"]["return_type"] == ""
        assert metadata["overload2_Load"]["return_type"] == ""
        assert metadata["Load"]["static"] is False
        assert metadata["Shout"]["static"] is True
        assert metadata["TryFind"]["parameters"] == [
            [3, 3, ["string", "out Entry?", "out int"]]
        ]

    def test_receiver_origins(self, builder):
        src = """
public class Svc(Repo repo)
{
    Cache _cache;

    public void Go(User user, object raw, Item[] items)
    {
        user.Login();
        var made = new Maker<int>(1);
        Order order = new();
        var box = new Box { Size = 1 };
        var text = (Text)raw;
        this.Step();
        base.Step();
        _cache.Clear();
        repo.Load();
        ((Admin)user).Grant();
        (raw as Admin)?.Revoke();
        made?.Run<int>();
        "x".Trim();
        items.Sort();
        foreach (var each in items) each.Use();
        if (raw is Admin admin && Find("k", out var hit)) admin.Grant();
        Act(it => it.Fire());
        return "y".Pad();
    }
}
""".strip()
        _parse(builder, src)
        origins = [
            (e["target"].removeprefix("REF:"), e["metadata"].get("origin"))
            for e in builder.edges
            if e["relation"] == "CALLS"
        ]
        assert origins == [
            ("user.Login", "User().Login"),
            ("Maker", "Maker"),
            ("Order", "Order"),
            ("Box", "Box"),
            ("this.Step", "self.Step"),
            ("base.Step", "super().Step"),
            ("_cache.Clear", "_cache.Clear"),
            ("repo.Load", "Repo().Load"),
            ("Grant", "Admin().Grant"),
            ("Revoke", "Admin().Revoke"),
            ("made.Run<>", "Maker<int>().Run"),
            ("Trim", "string().Trim"),
            ("items.Sort", "System::Array<Item>().Sort"),
            ("each.Use", "System::Array<Item>().*.Use"),
            ("Find", None),
            ("admin.Grant", "Admin().Grant"),
            ("Act", None),
            ("it.Fire", "?.Fire"),
            ("Pad", "string().Pad"),
        ]
