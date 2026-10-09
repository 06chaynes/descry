# Third-party notices

descry is MIT-licensed (see `LICENSE`). The files in `src/descry/stdlib/` are
derived from the sources below by the generators in `tools/stdlib/`. Each file
holds the names of a standard library's types, functions and methods and the
types they take and return. No implementation code and no documentation text
is copied. `tools/stdlib/README.md` records how each file is produced.

| File | Derived from | Copyright | Licence |
|---|---|---|---|
| `rust.json` | rustdoc JSON of `core`, `alloc` and `std`, Rust 1.100.0-nightly (4aa1fbcf4) | The Rust Project Contributors | MIT OR Apache-2.0 |
| `typescript.json` | `lib.*.d.ts` of the `typescript` npm package 6.0.2 | Microsoft Corporation | Apache-2.0 |
| `python.json` | the `stdlib` stubs of typeshed at `c4b5c1cfaba3da11df6915b12a45c6b3468195ae` | the typeshed contributors | Apache-2.0 (parts MIT) |
| `go.json` | `api/go1*.txt` and `src/builtin/builtin.go` of Go 1.26.0 | The Go Authors | BSD-3-Clause |
| `java.json` | the exported packages of `java.base`, Eclipse Temurin JDK 21.0.12 | Oracle and/or its affiliates | GPL-2.0-only WITH Classpath-exception-2.0 |
| `php.json` | JetBrains phpstorm-stubs at `e4f5f6c3de39f3bab3e9f3fca4b8cdb8b061e681` | JetBrains s.r.o. | Apache-2.0 |
| `dotnet.json` | the .NET 10.0.12 shared runtime, by reflection | .NET Foundation and Contributors | MIT |
| `ruby.json` | `core` and `stdlib` declarations of rbs 3.9.0 | Soutaro Matsumoto and the rbs contributors | BSD-2-Clause OR Ruby |
| `dart.json` | `sdk/lib` of the Dart SDK 3.7.0 | the Dart project authors | BSD-3-Clause |
| `c.json` | the C library headers of glibc 2.44, read with clang 22.1.8 | Free Software Foundation, Inc. | LGPL-2.1-or-later |
| `cpp.json` | the headers of libstdc++ 16, read with clang++ 22.1.8 | Free Software Foundation, Inc. | GPL-3.0-or-later WITH GCC-exception-3.1 |

The full licence texts are published by each project at the version named.
