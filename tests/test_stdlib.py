"""Tests for the generated standard-library data and its generators."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from functools import reduce
from importlib import resources
from pathlib import Path

import pytest

from descry._stdlib import load

LANGUAGES = (
    "typescript",
    "python",
    "go",
    "java",
    "rust",
    "php",
    "dotnet",
    "ruby",
    "dart",
)
GENERATORS = Path(__file__).parent.parent / "tools" / "stdlib"
# Where tools/stdlib/README.md puts each download, under the directory
# $DESCRY_STDLIB_SOURCES names; Java is read from the JDK at $JAVA_HOME.
DOWNLOADS = {
    "typescript": "typescript/typescript-6.0.2.tgz",
    "python": "python",
    "go": "go/go1.26.0",
    "rust": "rust/nightly-2026-09-09/json",
    "php": "phpstorm-stubs/phpstorm-stubs-e4f5f6c3de39f3bab3e9f3fca4b8cdb8b061e681.tar.gz",
    "ruby": "rbs",
    "dart": "dart-sdk",
    "node": "types-node/node-26.6.5.tgz",
}

# (language, path into the loaded data, what is there)
FACTS = [
    # A value-returning overload, its implementation typedef followed
    (
        "cpp",
        ("types", "std::basic_ostringstream`3", "methods", "str"),
        "basic_string<_CharT, _Traits, _Alloc>",
    ),
    # A member template defined outside its class is no function of `std`.
    ("cpp", ("scope", "string"), "std::basic_string`3"),
    # A class's template parameter, not an alias of that name elsewhere
    (
        "cpp",
        ("types", "std::basic_string`3", "methods", "front"),
        "_Traits::char_type",
    ),
    ("python", ("scope", "IOError"), "builtins.IOError"),
    (
        "python",
        ("context_managers",),
        ["contextlib.asynccontextmanager", "contextlib.contextmanager"],
    ),
    # A module function (`def self?.require`) is a method and a static.
    ("ruby", ("types", "Kernel", "methods", "require"), "bool"),
    ("ruby", ("types", "Kernel", "statics", "require"), "bool"),
    (
        "ruby",
        ("evaluators",),
        [
            "class_eval",
            "class_exec",
            "define_singleton_method",
            "instance_eval",
            "instance_exec",
            "module_eval",
            "module_exec",
        ],
    ),
    (
        "dotnet",
        ("types", "System.Collections.Generic.IEnumerable`1", "methods", "Where"),
        "System.Collections.Generic.IEnumerable`1<T>",
    ),
    (
        "dotnet",
        (
            "types",
            "System.Collections.Generic.IEnumerable`1",
            "methods",
            "GetEnumerator",
        ),
        "System.Collections.Generic.IEnumerator`1<T>",
    ),
    (
        "dotnet",
        ("types", "System.Collections.Generic.List`1", "params"),
        ["T"],
    ),
    ("dotnet", ("types", "System.String", "methods", "Contains"), "System.Boolean"),
    ("dotnet", ("types", "System.DateTime", "statics", "Now"), "System.DateTime"),
    ("dotnet", ("scope", "int"), "System.Int32"),
    ("dotnet", ("scope", "string"), "System.String"),
    (
        "dotnet",
        (
            "callbacks",
            "System.Collections.Generic.IEnumerable`1",
            "Select",
            "0",
            "params",
        ),
        ["T", "System.Int32"],
    ),
    (
        "dotnet",
        (
            "callbacks",
            "System.Collections.Generic.IEnumerable`1",
            "Select",
            "0",
            "returns",
        ),
        "TResult",
    ),
    (
        "dotnet",
        ("extensions", "System.Collections.Generic.IEnumerable`1", "Skip"),
        ["System.Linq"],
    ),
    ("dotnet", ("functional", "System.Func`3"), 2),
    ("java", ("functional", "java.util.function.BiFunction"), 2),
    ("typescript", ("types", "Array", "params"), ["T"]),
    ("typescript", ("types", "Array", "methods", "map"), "U[]"),
    ("typescript", ("types", "Array", "methods", "push"), "number"),
    ("typescript", ("types", "Array", "statics", "isArray"), "arg is any[]"),
    ("typescript", ("types", "Map", "methods", "has"), "boolean"),
    ("typescript", ("types", "Map", "methods", "clear"), None),
    ("typescript", ("types", "Document", "bases", 0), "Node"),
    (
        "typescript",
        ("types", "ArrayIterator", "bases"),
        ["IteratorObject<T, BuiltinIteratorReturn, unknown>"],
    ),
    ("typescript", ("types", "IteratorObject", "params"), ["T", "TReturn", "TNext"]),
    ("typescript", ("types", "IteratorObject", "methods", "toArray"), "T[]"),
    ("typescript", ("types", "Intl.NumberFormat", "methods", "format"), "string"),
    ("typescript", ("types", "Math", "statics", "max"), "number"),
    ("typescript", ("types", "TypeError", "statics", "isError"), "error is Error"),
    ("typescript", ("functions", "parseInt"), "number"),
    ("typescript", ("functions", "String"), "string"),
    ("typescript", ("scope", "console"), "Console"),
    ("typescript", ("scope", "Math"), "Math"),
    ("typescript", ("scope", "Array"), "Array"),
    ("typescript", ("scope", "JSON"), "JSON"),
    ("typescript", ("types", "string", "bases"), ["String"]),
    ("typescript", ("types", "Array", "methods", "filter"), "T[]"),
    ("typescript", ("types", "Array", "methods", "find"), "T | undefined"),
    ("typescript", ("types", "Array", "statics", "from"), "unknown[]"),
    ("typescript", ("types", "Promise", "statics", "all"), "Promise<unknown>"),
    (
        "typescript",
        ("callbacks", "Array", "map", "0"),
        {"params": ["T", "number", "T[]"], "returns": "U"},
    ),
    (
        "typescript",
        ("callbacks", "Array", "reduce", "0", "params"),
        ["unknown", "T", "number", "T[]"],
    ),
    (
        "typescript",
        ("callbacks", "Promise", "then", "0"),
        {"params": ["T"], "returns": "TResult1"},
    ),
    (
        "typescript",
        ("callbacks", "Map", "forEach", "0"),
        {"params": ["V", "K", "Map<K, V>"], "returns": None},
    ),
    ("python", ("types", "builtins.str", "methods", "split"), "list[str]"),
    ("python", ("types", "builtins.dict", "params"), ["_KT", "_VT"]),
    ("python", ("types", "builtins.dict", "methods", "get"), "_VT | None"),
    ("python", ("types", "builtins.dict", "methods", "copy"), "dict[_KT, _VT]"),
    ("python", ("types", "builtins.list", "methods", "__getitem__"), "_T"),
    ("python", ("types", "pathlib.PurePath", "methods", ".parent"), "typing.Self"),
    ("python", ("types", "re.Match", "methods", ".string"), "AnyStr"),
    ("python", ("types", "typing.List", "bases"), ["builtins.list[_T]"]),
    (
        "python",
        ("types", "builtins.dict", "bases"),
        ["typing.MutableMapping[_KT, _VT]"],
    ),
    ("python", ("types", "builtins.str", "bases"), ["typing.Sequence[str]"]),
    ("python", ("types", "typing.Sequence", "params"), ["_T_co"]),
    (
        "python",
        ("types", "typing.Sequence", "methods", "__iter__"),
        "typing.Iterator[_T_co]",
    ),
    ("python", ("types", "pathlib.Path", "methods", "resolve"), "typing.Self"),
    ("python", ("types", "asyncio.Queue", "bases"), ["asyncio.queues.Queue[_T]"]),
    ("python", ("functions", "json.loads"), "typing.Any"),
    ("python", ("functions", "os.path.exists"), "bool"),
    ("python", ("functions", "os.path.join"), "typing.Sequence[typing.Any]"),
    ("python", ("functions", "os.fspath"), None),
    ("python", ("functions", "builtins.open"), "typing.IO[typing.Any]"),
    ("python", ("functions", "builtins.sorted"), "list[typing.Any]"),
    ("python", ("functions", "re.match"), "re.Match[typing.Any] | None"),
    ("python", ("functions", ".os.environ"), "os._Environ[str]"),
    ("python", ("functions", ".sys.stdout"), "typing.TextIO | typing.Any"),
    (
        "python",
        ("callbacks", "re", "sub", "1"),
        {"params": ["re.Match[typing.Any]"], "returns": "typing.Any"},
    ),
    ("python", ("callbacks", "builtins.list", "sort", "key", "params"), ["_T"]),
    ("python", ("functions", "builtins.len"), "int"),
    ("python", ("scope", "len"), "builtins.len"),
    ("python", ("scope", "str"), "builtins.str"),
    ("go", ("functions", "strings.Split"), "[]string"),
    ("go", ("functions", "net/http.Get"), "(*net/http.Response, error)"),
    ("go", ("functions", "math/rand.New"), "*math/rand.Rand"),
    ("go", ("functions", "len"), "int"),
    ("go", ("functions", "recover"), "any"),
    ("go", ("functions", "append"), None),
    ("go", ("types", "byte", "bases"), ["uint8"]),
    ("go", ("scope", "string"), "string"),
    ("go", ("types", "bytes.Buffer", "methods", "String"), "string"),
    ("go", ("types", "bytes.Buffer", "methods", "Reset"), None),
    ("go", ("types", "error", "methods", "Error"), "string"),
    ("go", ("types", "sync/atomic.Pointer", "params"), ["$0"]),
    ("go", ("types", "sync/atomic.Pointer", "methods", "Load"), "*$0"),
    ("go", ("types", "bufio.ReadWriter", "bases"), ["bufio.Reader", "bufio.Writer"]),
    ("go", ("scope", "error"), "error"),
    ("go", ("types", "[]", "methods"), {}),
    ("go", ("types", "()", "params"), []),
    ("java", ("types", "java.util.Map", "params"), ["K", "V"]),
    (
        "java",
        ("types", "java.util.Map", "methods", "entrySet"),
        "java.util.Set<java.util.Map.Entry<K, V>>",
    ),
    ("java", ("types", "java.util.Map", "statics", "of"), "java.util.Map<K, V>"),
    (
        "java",
        ("types", "java.util.Properties", "bases"),
        ["java.util.Hashtable<Object, Object>"],
    ),
    ("java", ("types", "java.util.Hashtable", "params"), ["K", "V"]),
    ("java", ("types", "java.util.Hashtable", "methods", "get"), "V"),
    ("java", ("types", "java.lang.String", "bases", 1), "java.lang.Comparable<String>"),
    ("java", ("types", "java.lang.Object", "methods", "toString"), "String"),
    ("java", ("types", "java.lang.Object", "methods", "notify"), None),
    (
        "java",
        ("types", "java.lang.StringBuilder", "methods", "append"),
        "StringBuilder",
    ),
    (
        "java",
        ("types", "java.lang.StringBuilder", "bases", 0),
        "java.lang.AbstractStringBuilder",
    ),
    ("java", ("types", "java.lang.String", "statics", "valueOf"), "String"),
    ("java", ("scope", "String"), "java.lang.String"),
    (
        "java",
        ("callbacks", "java.util.Optional", "map"),
        {"0": {"params": ["T"], "returns": "U"}},
    ),
    (
        "java",
        ("callbacks", "java.util.Map", "computeIfAbsent"),
        {"1": {"params": ["K"], "returns": None}},
    ),
    ("java", ("callbacks", "java.util.Map", "forEach", "0", "params"), ["K", "V"]),
    ("java", ("callbacks", "java.util.List", "sort", "0", "params"), ["E", "E"]),
    ("java", ("callbacks", "java.util.List", "replaceAll", "0", "params"), ["E"]),
    (
        "java",
        ("callbacks", "java.util.Comparator", "thenComparing"),
        {"1": {"params": ["U", "U"], "returns": None}},
    ),
    ("rust", ("types", "std::option::Option", "params"), ["T"]),
    # What `[T]::get` returns is decided by its argument's type.
    ("rust", ("by_argument", "[T]", "get", "0", "usize"), "Option<&T>"),
    ("rust", ("by_argument", "[T]", "get", "0", "{integer}"), "Option<&T>"),
    ("rust", ("by_argument", "[T]", "get", "0", "std::ops::Range"), "Option<&[T]>"),
    ("rust", ("lang", "Range"), "std::ops::Range"),
    ("rust", ("lang", "RangeInclusiveStruct"), "std::ops::RangeInclusive"),
    ("rust", ("same", "std::os::fd::RawFd"), "std::os::raw::c_int"),
    ("rust", ("types", "std::option::Option", "methods", "unwrap"), "T"),
    ("rust", ("types", "std::option::Option", "methods", "as_ref"), "Option<&T>"),
    ("rust", ("types", "std::option::Option", "methods", "ok_or"), "Result<T, _>"),
    ("rust", ("types", "std::option::Option", "methods", "is_none"), "bool"),
    ("rust", ("types", "std::result::Result", "params"), ["T", "E"]),
    ("rust", ("types", "std::result::Result", "methods", "ok"), "Option<T>"),
    ("rust", ("types", "std::result::Result", "methods", "unwrap_err"), "E"),
    ("rust", ("types", "std::vec::Vec", "methods", "len"), "usize"),
    ("rust", ("types", "std::vec::Vec", "methods", "push"), None),
    ("rust", ("types", "std::vec::Vec", "statics", "new"), "Self"),
    ("rust", ("types", "[T]", "methods", "iter"), "std::slice::Iter<T>"),
    ("rust", ("types", "[T]", "methods", "first"), "Option<&T>"),
    ("rust", ("types", "str", "methods", "as_bytes"), "&[u8]"),
    ("rust", ("types", "str", "methods", "to_lowercase"), "String"),
    (
        "rust",
        ("types", "std::iter::Iterator", "methods", "next"),
        "Option<Self::Item>",
    ),
    (
        "rust",
        ("types", "std::iter::Iterator", "methods", "filter"),
        "std::iter::Filter<Self, P>",
    ),
    ("rust", ("types", "std::iter::Iterator", "methods", "collect"), "_"),
    ("rust", ("types", "std::clone::Clone", "methods", "clone"), "Self"),
    (
        "rust",
        ("types", "std::convert::TryFrom", "statics", "try_from"),
        "Result<Self, Self::Error>",
    ),
    ("rust", ("types", "std::collections::HashMap", "params"), ["K", "V", "S", "A"]),
    ("rust", ("types", "std::collections::HashMap", "methods", "get"), "Option<&V>"),
    ("rust", ("types", "std::io::Result", "bases"), ["Result<T, std::io::Error>"]),
    ("rust", ("types", "core::option::Option", "bases"), ["Option<T>"]),
    (
        "rust",
        ("types", "std::collections::hash_map::HashMap", "bases"),
        ["std::collections::HashMap<K, V, S, A>"],
    ),
    ("rust", ("functions", "std::fs::read_to_string"), "std::io::Result<String>"),
    ("rust", ("functions", "std::mem::take"), "_"),
    ("rust", ("scope", "Vec"), "std::vec::Vec"),
    ("rust", ("scope", "Iterator"), "std::iter::Iterator"),
    ("rust", ("scope", "str"), "str"),
    ("rust", ("scope", "Debug"), "std::fmt::Debug"),
    ("rust", ("variants", "std::option::Option"), {"Some": ["T"]}),
    ("rust", ("variants", "std::result::Result", "Err"), ["E"]),
    ("rust", ("types", "std::option::Option", "methods", "map"), "Option<F>"),
    (
        "rust",
        ("callbacks", "std::option::Option", "map"),
        {"0": {"params": ["T"], "returns": "F"}},
    ),
    (
        "rust",
        ("callbacks", "std::iter::Iterator", "fold", "1", "params"),
        ["F", "Self::Item"],
    ),
    ("php", ("types", "DateTimeImmutable", "bases"), ["DateTimeInterface"]),
    ("php", ("types", "DateTimeImmutable", "methods", "add"), "DateTimeImmutable"),
    ("php", ("types", "DateTimeInterface", "methods", "format"), "string"),
    (
        "php",
        ("types", "DateTimeImmutable", "statics", "createFromFormat"),
        "DateTimeImmutable|false",
    ),
    ("php", ("types", "Exception", "bases"), ["Throwable"]),
    ("php", ("types", "Throwable", "methods", "getPrevious"), "Throwable|null"),
    ("php", ("types", "Iterator", "params"), ["TKey", "TValue"]),
    ("php", ("types", "Iterator", "methods", "current"), "TValue"),
    ("php", ("types", "Iterator", "methods", "next"), None),
    ("php", ("types", "Traversable", "bases"), ["iterable<TKey, TValue>"]),
    ("php", ("types", "Generator", "bases"), ["Iterator<TKey, TYield>"]),
    ("php", ("types", "Generator", "methods", "throw"), "TYield"),
    ("php", ("types", "array", "bases"), ["iterable<TKey, TValue>"]),
    ("php", ("types", "Closure", "statics", "fromCallable"), "Closure"),
    ("php", ("types", "ReflectionClass", "methods", "newInstance"), "T"),
    ("php", ("types", "Random\\Randomizer", "methods", "getInt"), "int"),
    ("php", ("functions", "date_create"), "DateTime|false"),
    ("php", ("functions", "json_decode"), "mixed"),
    ("php", ("functions", "fopen"), "resource|false"),
    ("php", ("functions", "array_map"), "array"),
    ("php", ("functions", "error_log"), "bool"),
    ("php", ("scope",), {}),
    ("ruby", ("types", "String", "methods", "length"), "Integer"),
    ("ruby", ("types", "Integer", "methods", "to_s"), "String"),
    ("dart", ("types", "Iterable", "methods", "map"), "Iterable<T>"),
    ("dart", ("types", "Iterable", "methods", "first"), "E"),
    ("dart", ("types", "String", "methods", "length"), "int"),
    ("dart", ("types", "AsciiDecoder", "statics", "new"), "AsciiDecoder"),
    ("dart", ("types", "CustomEvent", "methods", "detail"), None),
    ("dart", ("types", "DateTime", "statics", "now"), "DateTime"),
    ("dart", ("types", "Stopwatch", "statics", "new"), "Stopwatch"),
    ("dart", ("types", "Map", "statics", "from"), "Map"),
    ("dart", ("types", "Duration", "statics", "millisecondsPerSecond"), "int"),
    ("dart", ("types", "DateTime", "methods", "operator =="), "bool"),
    ("dart", ("types", "UriData", "methods", "mimeType"), "String"),
    ("dart", ("types", "Element", "methods", "id"), "String"),
    ("dart", ("functions", "runZonedGuarded"), "R"),
    (
        "dart",
        ("callbacks", "Iterable", "forEach", "0", "params"),
        ["E"],
    ),
]

# (language, type, one of its bases)
BASES = [
    (
        "dotnet",
        "System.Collections.Generic.List`1",
        "System.Collections.Generic.IEnumerable`1<T>",
    ),
    ("rust", "std::slice::Iter", "Iterator<Item = &T>"),
    ("rust", "std::iter::Peekable", "Iterator<Item = I::Item>"),
    ("rust", "std::iter::Map", "Iterator<Item = F>"),
    (
        "rust",
        "std::option::Option",
        "std::ops::Try<Output = T, Residual = Option<!>>",
    ),
    (
        "rust",
        "std::vec::Vec",
        "IntoIterator<IntoIter = std::vec::IntoIter<T, A>, Item = T>",
    ),
    ("rust", "std::rc::Rc", "AsRef<T>"),
    ("rust", "std::string::String", "std::ops::Deref<Target = str>"),
    ("rust", "std::string::String", "str"),
    ("rust", "std::vec::Vec", "[T]"),
    ("rust", "std::boxed::Box", "T"),
    ("rust", "[T; N]", "[T]"),
    ("rust", "std::fmt::Display", "ToString"),
    ("rust", "std::clone::Clone", "ToOwned<Owned = Self>"),
    ("ruby", "Array", "Enumerable[Elem]"),
    ("ruby", "String", "Comparable"),
    ("dart", "List", "Iterable<E>"),
    ("dart", "StreamSink", "EventSink<S>"),
    ("dart", "TypeError", "Error"),
]


def _file(lang):
    return resources.files("descry") / "stdlib" / f"{lang}.json"


def _base_name(base):
    """What a base names: the type before its type arguments, without how
    Rust borrows it, or a slice, an array or a pointer as Rust's file keys
    them."""
    base = re.sub(r"^&(mut )?", "", base)
    if base.startswith("["):
        return "[T; N]" if ";" in base else "[T]"
    if base.startswith("*"):
        return "*const T"
    return re.split(r"[<\[]", base)[0]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_data_has_the_documented_shape(lang):
    data = load(lang)
    optional = {
        "variants",
        "callbacks",
        "by_argument",
        "lang",
        "same",
        "extensions",
        "functional",
        "ref_structs",
        "qualifiers",
        "evaluators",
        "attributes",
        "context_managers",
        "pointers",
        "writers",
        "makers",
        "awaited",
        "implicit",
        "implicit_usings",
        "platforms",
    }
    assert set(data) - optional == {"source", "types", "functions", "scope"}
    assert set(data.get("ref_structs", ())) <= set(data["types"])
    assert data["source"]
    assert data["types"]
    tables = [data["functions"]]
    for kind in data["types"].values():
        assert set(kind) == {"params", "bases", "methods", "statics"}
        assert all(isinstance(name, str) for name in kind["params"] + kind["bases"])
        tables += [kind["methods"], kind["statics"]]
    returns = [returns for table in tables for returns in table.values()]
    assert all(it is None or isinstance(it, str) and it for it in returns)
    for key, methods in data.get("by_argument", {}).items():
        assert set(methods) <= set(data["types"][key]["methods"]), key
        given = [r for p in methods.values() for t in p.values() for r in t.values()]
        assert all(isinstance(it, str) and it for it in given)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_bases_and_scope_name_what_the_file_holds(lang):
    """A base is a type of the file under its key or its name in `scope`,
    or in Rust a parameter of the type or an associated type of one; what
    `scope` names is a type, a function, or an enum's variant."""
    data = load(lang)
    types, scope = data["types"], data["scope"]
    for kind in types.values():
        for base in map(_base_name, kind["bases"]):
            parameter = lang == "rust" and base.partition("::")[0] in kind["params"]
            assert scope.get(base, base) in types or parameter, base
    variants = data.get("variants", {})
    assert all(
        name in types
        or name in data["functions"]
        or name.rpartition("::")[0] in variants
        for name in scope.values()
    )


@pytest.mark.parametrize("lang", LANGUAGES)
def test_bases_form_no_cycle(lang):
    data = load(lang)
    types, scope = data["types"], data["scope"]
    done = set()

    def walk(name, above):
        assert name not in above, (*above, name)
        if name in done:
            return
        done.add(name)
        for base in map(_base_name, types[name]["bases"]):
            base = scope.get(base, base)
            if base in types:
                walk(base, (*above, name))

    for name in types:
        walk(name, ())


@pytest.mark.parametrize(("lang", "name", "base"), BASES)
def test_type_has_the_base(lang, name, base):
    assert base in load(lang)["types"][name]["bases"]


def test_a_cpp_specialization_is_its_template_s():
    """A partial or explicit specialization of a class template is no type
    of its own: `vector<bool, _Alloc>` is no `std::vector` of one parameter
    beside the template of two."""
    types = load("cpp")["types"]
    assert [k for k in types if k.startswith("std::vector`")] == ["std::vector`2"]


def test_node_s_modules_are_named_as_an_import_writes_them():
    """A built-in module is named bare and with `node:`, but one Node
    provides only with `node:` is named only so: a bare `test` is a
    package's."""
    modules = load("node")["modules"]
    assert {"fs", "node:fs", "fs/promises", "path", "node:test"} <= set(modules)
    assert "test" not in modules


@pytest.mark.parametrize("lang", (*LANGUAGES, "node"))
def test_file_is_as_the_generator_writes_it(lang):
    text = _file(lang).read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=1, sort_keys=True) + "\n"


@pytest.mark.parametrize(("lang", "path", "expected"), FACTS)
def test_fact(lang, path, expected):
    assert reduce(lambda node, key: node[key], path, load(lang)) == expected


@pytest.mark.parametrize("lang", (*LANGUAGES, "c", "cpp", "node"))
def test_generator_reproduces_the_file(lang, tmp_path):
    if lang == "java":
        source = Path(os.environ.get("JAVA_HOME", ""))
        release = source / "release"
        version = load("java")["source"].split()[1]
        if not release.is_file() or f'"{version}"' not in release.read_text("utf-8"):
            pytest.skip(f"$JAVA_HOME is not the JDK {version} the data was read from")
    elif lang == "dotnet":
        shell = os.environ.get("PWSH") or shutil.which("pwsh") or ""
        source = Path(shell).parent if shell else None
        if not source or not (source / "System.Private.CoreLib.dll").is_file():
            pytest.skip("pwsh is not installed to reflect the .NET runtime with")
        spec = importlib.util.spec_from_file_location(
            lang, GENERATORS / "gen_dotnet.py"
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        runtime = load(lang)["source"].split()[2].split("-")[0]
        if generator.sdk_folder(source, runtime) is None:
            pytest.skip(f"no .NET SDK bundling the {runtime} runtime is installed")
    elif lang in ("c", "cpp"):
        compiler = shutil.which("clang" if lang == "c" else "clang++")
        roots = sorted(Path("/usr/include/c++").glob("*")) if lang == "cpp" else []
        source = roots[-1] if roots else Path("/usr/include")
        if not compiler or not source.is_dir():
            pytest.skip(f"Clang and {source} are not installed to read headers from")
        # Headers of another toolchain may lack ones the data was read from.
        spec = importlib.util.spec_from_file_location(
            lang, GENERATORS / f"gen_{lang}.py"
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        # Only the C++ toolchain is named from the include directory.
        given = (source,) if lang == "cpp" else ()
        toolchain = generator._version(Path(compiler).name, *given)
        if toolchain != load(lang)["source"]:
            pytest.skip(
                f"this machine's toolchain is {toolchain}, not {load(lang)['source']}"
            )
    else:
        source = Path(os.environ.get("DESCRY_STDLIB_SOURCES", ""), DOWNLOADS[lang])
        if not source.exists():
            pytest.skip(f"{DOWNLOADS[lang]} is not under $DESCRY_STDLIB_SOURCES")
    out = tmp_path / f"{lang}.json"
    generator = GENERATORS / f"gen_{lang}.py"
    subprocess.run([sys.executable, "-I", generator, source, out], check=True)
    # The runtime pwsh runs on is named as the generator reads it.
    read_from = load(lang)["source"]
    if lang == "dotnet" and json.loads(out.read_text("utf-8"))["source"] != read_from:
        pytest.skip(f"this machine's toolchain is not {read_from}")
    assert out.read_bytes() == _file(lang).read_bytes()


def _generator(lang):
    """The module of `tools/stdlib/gen_<lang>.py`."""
    sys.path.insert(0, str(GENERATORS))
    try:
        spec = importlib.util.spec_from_file_location(
            f"gen_{lang}", GENERATORS / f"gen_{lang}.py"
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
    finally:
        sys.path.remove(str(GENERATORS))
    return generator


def test_ruby_reads_the_arguments_of_a_type_written_from_the_root():
    """`::Array[Integer]` is `Array[Integer]`, as `Array[Integer]` is."""
    assert _generator("ruby")._written("::Array[Integer]") == "Array[Integer]"


def test_dart_reads_a_libraries_directory_given_itself(tmp_path):
    """The generator reads the `sdk/lib` directory named on its own, not
    only a checkout holding it."""
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "widget.dart").write_text("class Widget {\n}\n")
    assert "Widget" in _generator("dart").build(tmp_path)["types"]


def test_dotnet_sdk_is_found_beside_dotnet_exe(tmp_path, monkeypatch):
    """The SDK that bundles a runtime is found beside a `dotnet.exe` on the
    path, as Windows names it."""
    spec = importlib.util.spec_from_file_location(
        "dotnet", GENERATORS / "gen_dotnet.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    sdk = tmp_path / "bin" / "sdk" / "10.0.100"
    sdk.mkdir(parents=True)
    (tmp_path / "bin" / "dotnet.exe").write_text("")
    (sdk / "Microsoft.NETCoreSdk.BundledVersions.props").write_text(
        "<BundledNETCoreAppPackageVersion>10.0.1<"
    )
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert generator.sdk_folder(tmp_path / "a" / "b" / "c", "10.0.1") == sdk / "Sdks"
