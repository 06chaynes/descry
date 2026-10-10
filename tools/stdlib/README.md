# Standard-library data

`src/descry/stdlib/<lang>.json` tells the call resolver which types and
functions a language's standard library has and what each method returns. The
files are written by the generators in this directory and never edited by
hand. `src/descry/_stdlib.py` documents the format and loads the files.

Every generator takes its source and the file to write, reads the source
without running or unpacking it, and writes the same bytes for the same
source:

    python -I tools/stdlib/gen_<lang>.py SOURCE src/descry/stdlib/<lang>.json

`tests/test_stdlib.py` regenerates each file and compares it when the source is
there: the downloads under the directory `$DESCRY_STDLIB_SOURCES` names, at the
paths used below, and the JDK at `$JAVA_HOME`.

## What the files hold

- A type name inside a return type is a key of `types`, a name of `scope`, a
  type parameter, or one of the language's own keywords and primitive types.
- A base is a type the file holds, followed by the type arguments the source
  gives it, written as return types are (`java.util.Hashtable<Object, Object>`,
  `typing.Sequence[str]`, `Iterator<Item = &T>`); the type is what stands
  before the first `<` or `[`. A base the file does not hold is left out, and
  so is the root every type has without saying so (`java.lang.Object`,
  `builtins.object`).
- A class that a module hands on under its own name (`asyncio.Queue`,
  `os.FileInfo`) is a type whose one base is the class it is.
- `variants` and `callbacks` are written only by a generator that has them.

## TypeScript and JavaScript

Source: the `lib.*.d.ts` files of the `typescript` npm package, version 6.0.2:
`lib.esnext.d.ts` and `lib.dom.d.ts` with every library they reference.

    mkdir -p "$DESCRY_STDLIB_SOURCES/typescript"
    curl -fLo "$DESCRY_STDLIB_SOURCES/typescript/typescript-6.0.2.tgz" \
        https://registry.npmjs.org/typescript/-/typescript-6.0.2.tgz
    python -I tools/stdlib/gen_typescript.py \
        "$DESCRY_STDLIB_SOURCES/typescript/typescript-6.0.2.tgz" \
        src/descry/stdlib/typescript.json

sha256 `0ae5c188a2f5db22df72fe5e74dcbc122afb52031a86dbac33e78a86db39c65e`.

The declarations are read by the generator's own reader for the subset of the
language these files use; the package's code is not run. Interfaces of one
name are merged. A namespace is a type whose `statics` are its functions. For
`declare var X: T`, when a type `X` exists the methods of `T` are the `statics`
of `X`, a call signature of `T` is the function `X`, and `scope` maps `X` to
itself (`Array`, `Math`); otherwise `scope` maps `X` to `T` (`console` to
`Console`). The keyword of a primitive type is a type whose one base is the
interface that has its methods (`string` has `String`).

A type parameter of a signature that extends one type is written as that type
(`filter<S extends T>` returns `T[]`). Signatures of one name that return the
same type with other type arguments leave that type, with `unknown` for each
argument they differ in (`Array.from` returns `unknown[]`). `callbacks` has,
for a method that takes a function, the argument it is, the types of its
parameters, `unknown` where the signatures differ, and the type parameter of
the signature that the function returns, where it returns one (`U` for
`Array.map`); a `this` parameter is left out.

## Python

Source: the `stdlib` stubs of typeshed at commit
`c4b5c1cfaba3da11df6915b12a45c6b3468195ae`.

    mkdir -p "$DESCRY_STDLIB_SOURCES/typeshed"
    curl -fLo "$DESCRY_STDLIB_SOURCES/typeshed/typeshed-c4b5c1cfaba3da11df6915b12a45c6b3468195ae.tar.gz" \
        https://github.com/python/typeshed/archive/c4b5c1cfaba3da11df6915b12a45c6b3468195ae.tar.gz
    python -I tools/stdlib/gen_python.py \
        "$DESCRY_STDLIB_SOURCES/typeshed/typeshed-c4b5c1cfaba3da11df6915b12a45c6b3468195ae.tar.gz" \
        src/descry/stdlib/python.json

sha256 `e0925dd090c5258856dcb42b5f9e2eb033bc9e9080a5e9d198ff06e9e3cdff17`.

The stubs are read with `ast` as the newest Python sees them
(`sys.version_info` tests hold, modules `VERSIONS` gives an end are left out)
and on every platform at once. `scope` is the public classes and functions of
`builtins`. Class and static methods are `statics`. An `async def` returns
`typing.Coroutine[typing.Any, typing.Any, T]`, a function without an
annotation `typing.Any`. Type aliases are written out unless they take
arguments, `LiteralString` is written `str`, and a name both `typing` and
`typing_extensions` define is `typing`'s. The names `typing` has for classes
of other modules (`typing.List`, PEP 585), which the stubs only declare, are
types whose one base is the class.

What is read rather than called is written like the method or function of
that name behind a dot, with its type: a property or an annotated attribute
of a class in its `methods` (`.parent`), a module's annotated variable in
`functions` (`.sys.stdout`). The type is what `Final` or `ClassVar` holds; a
variable without one is left out.

Overloads that disagree return what they share: the type they all return
once `None` and a type variable of their own are left out of a union
(`dict.get` returns `_VT | None`); failing that the one type they all name,
with `typing.Any` for the arguments they give it differently (`re.match`
returns `re.Match[typing.Any] | None`); failing that the nearest type all of
them are, `object` at the least (`open` returns `typing.IO[typing.Any]`);
None where one of them returns a type variable or nothing. A union of
several types is merged like as many overloads. The overload of
`__getitem__` that takes a slice is left out: the method is what an index
gives.

`callbacks` has, for a parameter annotated `Callable[[A, ...], R]`, the types
`A, ...` under its position after the receiver and, unless it is
positional-only, under its name; `returns` is `R` as written. Those of a
function are under its module (`re`, `sub`).

## Go

Source: `api/go1.txt`, `api/go1.1.txt` to `api/go1.26.txt` and
`src/builtin/builtin.go` of the Go repository at tag `go1.26.0`.

    mkdir -p "$DESCRY_STDLIB_SOURCES/go/go1.26.0/api" \
        "$DESCRY_STDLIB_SOURCES/go/go1.26.0/src/builtin"
    for file in src/builtin/builtin.go api/go1.txt $(seq -f 'api/go1.%g.txt' 1 26); do
        curl -fLo "$DESCRY_STDLIB_SOURCES/go/go1.26.0/$file" \
            "https://raw.githubusercontent.com/golang/go/go1.26.0/$file"
    done
    python -I tools/stdlib/gen_go.py "$DESCRY_STDLIB_SOURCES/go/go1.26.0" \
        src/descry/stdlib/go.json

`LC_ALL=C sh -c 'sha256sum api/go1*.txt src/builtin/builtin.go | sha256sum'` in
`go1.26.0` gives
`d83016686a77e86fca1ef48ee4bcee35a44be1ea8123670867ec270a62cc6674`.

A type is named by its package's import path (`net/http.Request`), also inside
return types, which are otherwise as the listing writes them: `uint8` for
`byte`, `$0` for a type's first parameter, several results in parentheses. The
lines of every platform are read. An embedded field and the target of a type
alias are `bases`. The predeclared identifiers, which the listing does not
have, are those `builtin.go` declares: its types and functions are `scope`. A
built-in function whose result `builtin.go` writes with one of its stand-in
types (`Type`, `IntegerType`) or a type parameter has no return type. The types
Go writes without a name are `[]`, a slice or an array, `map`, `chan` and
`()`, the several results of a function: each without members.

## Java

Source: the exported packages of the `java.base` module of Eclipse Temurin
JDK 21.0.12.1+1-LTS, read with that JDK's `jimage` and `javap`.

    python -I tools/stdlib/gen_java.py "$JAVA_HOME" src/descry/stdlib/java.json

The file holds the public and protected methods of every public class, without
the bridge methods the compiler adds, and the classes those extend whatever
their access. `scope` is the top-level classes of `java.lang`, which return
types name without their package.

`callbacks` has, for an instance method, each argument that is a functional
interface, one whose only abstract method besides those `Object` implements is
its own or a base's: that method's parameter types with the type arguments the
argument is written with, a wildcard as its bound, and as `returns` the type
parameter of the taking method that stands for the result, null where none
does. A position where overloads take different callables is left out.

## Rust

Source: `core.json`, `alloc.json` and `std.json` of the `rust-docs-json`
component of the toolchain `nightly-2026-09-09` (rustc 1.100.0-nightly,
4aa1fbcf4) for `x86_64-unknown-linux-gnu`, rustdoc's JSON in format 61.

    rustup toolchain install nightly-2026-09-09 --component rust-docs-json
    mkdir -p "$DESCRY_STDLIB_SOURCES/rust/nightly-2026-09-09"
    cp -r "$(rustup run nightly-2026-09-09 rustc --print sysroot)/share/doc/rust/json" \
        "$DESCRY_STDLIB_SOURCES/rust/nightly-2026-09-09/json"
    python -I tools/stdlib/gen_rust.py \
        "$DESCRY_STDLIB_SOURCES/rust/nightly-2026-09-09/json" \
        src/descry/stdlib/rust.json

sha256 of `core.json`
`79ffb263f2d2117c40dd3b5c0ec187ea6ac3f974ce71e9348a4021b4e56f1ff3`, of
`alloc.json`
`f6dba1bc4d4a7b33f3221f0444c66640cfee9344307fafdeff4cfca42f215f89`, of
`std.json`
`7c1d752c23efa761e796564d9a1b23376cd9c63710d809169d47230e810e077a`.

The file holds every type, trait, alias and function a path from the root of
one of the three crates reaches, stable or not, but for the functions of
`core::arch` and `std::arch`, one for each instruction of each processor. A
type is keyed by its shortest path in `std`, failing that in `alloc`, failing
that in `core`; a primitive by its name (`str`), or by how it is written where
it has none (`[T]`, `[T; N]`, `()`). Every other path a type can be imported
by, and every `type` alias, is a type whose one base is what it stands for
(`core::option::Option` has `Option<T>`, `std::io::Result` has
`Result<T, std::io::Error>`). `scope` is the prelude of the newest edition and
the primitives that have a name; a type in it is written by that name, any
other by its key.

`params` are the type and const parameters; lifetimes are left out everywhere.
A parameter of an impl or of a method that no parameter of the type stands for
is written `_` (`Option::ok_or` returns `Result<T, _>`). An impl for `&T`
counts as one for `T` where `T` has none of that trait.

`bases` are the traits a type implements, with their arguments and the
associated types the impl sets (`Iterator<Item = &T>`, `_` where the impls of
one trait differ), and the type it dereferences to, which may be one of its
parameters (`Box<T>` has `T`); an array has the methods of a slice. They are
in the order a method is looked for: the traits of the prelude, what the type
dereferences to, then the traits that are in scope only after a `use`. A
blanket impl is a base of every type when it bounds nothing, and a base of the
trait it is bounded by otherwise (`Display` has `ToString`). A trait without
methods is no base, nor is one a pointer implements only where what it points
at does (`Box<I>: Iterator`): dereferencing finds those methods.

A type parameter bounded as a callable, `F: FnMut(A) -> B`, stands for what
the callable returns, and so does `B`: `Option::map` returns `Option<F>` and
`Map<I, F>` is an `Iterator<Item = F>`. `callbacks` has, for a method that
takes such a callable, the argument it is and `A`. `variants` has what the
variants of an enum hold (`Some` of `Option` holds `T`).

The source has no inherent impl of `alloc` for `[T]`, so `to_vec`, `sort`,
`join` and `concat` of a slice are missing.

## PHP

Source: JetBrains' phpstorm-stubs at commit
`e4f5f6c3de39f3bab3e9f3fca4b8cdb8b061e681`.

    mkdir -p "$DESCRY_STDLIB_SOURCES/phpstorm-stubs"
    curl -fLo "$DESCRY_STDLIB_SOURCES/phpstorm-stubs/phpstorm-stubs-e4f5f6c3de39f3bab3e9f3fca4b8cdb8b061e681.tar.gz" \
        https://github.com/JetBrains/phpstorm-stubs/archive/e4f5f6c3de39f3bab3e9f3fca4b8cdb8b061e681.tar.gz
    python -I tools/stdlib/gen_php.py \
        "$DESCRY_STDLIB_SOURCES/phpstorm-stubs/phpstorm-stubs-e4f5f6c3de39f3bab3e9f3fca4b8cdb8b061e681.tar.gz" \
        src/descry/stdlib/php.json

sha256 `dde0ca3a4d6ebfa346fb34d0b7eeb14ecafe96a0ea2489eb697cd2ff1ec1148a`.

The stubs are read by the generator's own reader of the declarations they
consist of. The file holds the directories of the extensions that the PHP
source tree itself has, the list `BUNDLED` in the generator (`Core`,
`standard`, `SPL`, `date`, `Reflection`, `json`, `curl`, `intl` and so on):
367 types and 2286 functions. The extensions distributed apart from PHP
(`redis`, `mongodb`, `swoole`, `imagick`, `xdebug` and the like) are left out;
with them the file would hold 1533 types and 5038 functions and be four times
the size. Constants, properties and enum cases are not read: the format has no
place for them.

A class is keyed by its name with its namespace (`Random\Randomizer`), a
function likewise; `scope` is empty, since PHP names a class in full or by an
import. `statics` are the static methods, `methods` the others, public and
protected. The stubs are read as the newest PHP has them: a declaration that
`PhpStormStubsElementAvailable` gives an end (`to:`) or whose comment says
`@removed` is left out, and of the types `LanguageLevelTypeAware` lists for a
return, the one of the newest version is taken. A method the stubs declare as
`PS_UNRESERVE_PREFIX_throw` is `throw`.

A return type is the declared one, as PHP writes it (`DateTime|false`,
`null|Closure`, `static`), with `self` written as the class it stands in and
`?T` as `null|T`. Where the `@return` tag says more, a type parameter or type
arguments, it is taken instead (`Iterator::current` returns `TValue`), and
where nothing is declared the tag is all there is. `void` and `never` are no
return type. `T[]` is written `array<int, T>`.

`params` are the `@template` parameters of a class, and a base carries the
type arguments that `@template-extends`, `@template-implements`, `@extends` or
`@implements` give it (`ArrayIterator` has `SeekableIterator<TKey, TValue>`).
`array` and `iterable`, which are no classes, are types without members whose
parameters are `TKey` and `TValue`: `array` has the base `iterable<TKey,
TValue>`, as the stubs give `Traversable`, so that the second argument of the
`iterable` a type reaches through its bases is what a `foreach` over it gives.

The stubs give no parameter types for the callables that functions such as
`array_map` and `usort` take, so the file has no `callbacks`.

## C# (.NET)

Source: the .NET shared runtime — the directory that holds
`System.Private.CoreLib.dll` and the `System.*.dll` beside it. PowerShell
7.6.6 ships .NET 10.0.12; the generator takes that directory:

    python -I tools/stdlib/gen_dotnet.py \
        "$(dirname "$(command -v pwsh)")" src/descry/stdlib/dotnet.json

The public and protected types and members of the `System.*` and
`netstandard` assemblies are read by reflection in `pwsh`; the type it is
read from is never instantiated. A type's key is its full name with the
generic arity (`List`1`), nested types under a `.`. `scope` is the language
aliases (`int` is `System.Int32`). An extension method of `this T`, where
every argument `this` is written with is a parameter of the method, is a
method of `T` — `Enumerable.Where` is a method of `IEnumerable`1` —
written with the parameter names `T` declares; one written with a
concrete type (`IEnumerable<XElement>`) is left out, for it does not hold
of every value of the type. `callbacks` has, for a method that takes a
delegate, the signature its `Invoke` gives the argument, with the
parameter that stands for its result where one does (`TResult` for
`Select`'s selector).


## Ruby

Source: the `rbs` gem's repository at tag `v3.9.0` — `core/*.rbs` for the
classes and modules every Ruby program has, `stdlib/*/0/*.rbs` for those a
`require` brings:

    mkdir -p "$DESCRY_STDLIB_SOURCES/rbs"
    curl -fLo "$DESCRY_STDLIB_SOURCES/rbs.tar.gz" \
        https://github.com/ruby/rbs/archive/refs/tags/v3.9.0.tar.gz
    tar -xzf "$DESCRY_STDLIB_SOURCES/rbs.tar.gz" --strip-components=1 \
        -C "$DESCRY_STDLIB_SOURCES/rbs"
    python -I tools/stdlib/gen_ruby.py "$DESCRY_STDLIB_SOURCES/rbs" \
        src/descry/stdlib/ruby.json

Declarations are read by a reader for the RBS syntax they use; the
repository's code is not run. A `module`/`class`/`struct` is a type; an
`interface` is not, for it is only what a class mixes in. `include`,
`extend` and `prepend` name `bases`. `self.` methods are `statics`. A
method's return type is the declared one; overloads that disagree leave
None. A `def` whose block parameter is written `(&block: T -> S)` records
`callbacks` for argument 0.

## Dart

Source: the `sdk/lib` libraries of `dart-lang/sdk` at the `3.7.0` tag —
`core`, `async`, `collection`, `convert`, `io`, `math`, `typed_data` and
the rest of the `dart:` libraries:

    mkdir -p "$DESCRY_STDLIB_SOURCES/dart-sdk"
    curl -fLo "$DESCRY_STDLIB_SOURCES/dart-sdk.tar.gz" \
        https://github.com/dart-lang/sdk/archive/refs/tags/3.7.0.tar.gz
    tar -xzf "$DESCRY_STDLIB_SOURCES/dart-sdk.tar.gz" --strip-components=1 \
        -C "$DESCRY_STDLIB_SOURCES/dart-sdk"
    python -I tools/stdlib/gen_dart.py "$DESCRY_STDLIB_SOURCES/dart-sdk" \
        src/descry/stdlib/dart.json

Sources under `_internal`, `internal`, `native`, `js`, `wasm` and `web`
are not read, nor are members declared in a type's implementation
(class-body lines only), nor private names. A parameter list and the
header of a type are read as one line each. Classes, mixins, enums,
extension types and named extensions are types; an unnamed `extension on
T` is `T`'s members. Types of one name in two libraries are one type,
and a member they type differently has no type. `extends`, `with`,
`implements` and `on` name `bases` — one the file does not hold is left
out. `static` and `factory` members are `statics`, as are constructors:
the unnamed one as `new`, a named one under its name. Top-level
functions are `functions`. `get` members and `set` members share a name:
the getter's return type is the member's. A parameter written as a function signature — `T toElement(E
e)`, `void action(E element)` — records `callbacks` for argument 0, its
return a type parameter where the signature writes one.
