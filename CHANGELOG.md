# Changelog

All notable changes to Descry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to pre-1.0 semver (minor-version bumps may include breaking changes; see README versioning stance).

## [Unreleased]

## [0.2.0] — 2026-04-20

A large feature + resolution-quality release. Seven new SCIP language
adapters land (Java/Kotlin/Scala, Go, Ruby, PHP, C#/VB, C/C++, Dart)
alongside a language-general STDLIB_FILTER sweep that brought CALLS
resolution above 90% on 20 of 26 measured corpora. The release also
folds in a full session of convergence-audit hardening: user-input
parsing no longer 500s on garbage query strings, subprocess output
tolerates non-UTF-8 bytes, and the documented security invariants match
what the code actually does.

Graph schema stays at v1 — existing `.descry_cache/codebase_graph.json`
files remain readable. CLI subcommands and MCP tool names are unchanged
(19 of each, 1:1 parity).

### Added

- **Dart / Flutter support** via `scip-dart` (Milestone D). DartAdapter
  discovers projects by `pubspec.yaml` at the root or under
  `packages/*` (melos / workspace layout); runs `scip-dart ./` from the
  package root and uses `output_mode="rename"` to move the emitted
  `index.scip` into the cache. DartParser handles imports
  (`package:` / `dart:` / relative), classes (including `abstract`,
  `base`, `sealed`, `final`, `interface class`, `mixin class`), mixins,
  extensions, enums, typedefs, top-level constants, class inheritance
  (`extends` → INHERITS edge), and call sites with null-safe receiver
  chains (`?.method()`). Dart has heavy top-level code (top-level
  functions, top-level variable initializers), so the parser emits
  calls at file scope too — unlike Java/C# parsers which gate calls on
  non-file context. pub smoke test hit **71%** resolution
  (10,279 / 14,490 CALLS edges) — below the 91.2% Rust bar, accepted
  as the Dart ceiling for test-heavy codebases where pub's internal
  test-helper DSL (`servePackages`, `d.file`, `runPub`, etc.) contributes
  the bulk of unresolved calls.
- **Java / Kotlin / Scala support** via `scip-java` (Milestone J of Wave 2).
  JavaParser extracts classes, interfaces, enums, records, methods,
  constructors, fields, imports, and call sites. JavaAdapter ships a
  Gradle init-script that strips `-Werror` so scip-java works out of the
  box on Apache projects (Kafka, etc.) whose builds treat warnings as
  fatal. Kafka smoke test hit **92.7%** CALLS resolution.
- **Go support** via `scip-go` (Milestone G). GoParser covers packages,
  grouped imports, type declarations, free functions, methods with
  receivers, const/var blocks, and call sites. Kubernetes smoke test
  hit **98.3%** resolution.
- **Ruby support** via `scip-ruby` (Milestone R). RubyParser uses
  indent-based context tracking (Ruby uses `end` not `}`); extracts
  classes with INHERITS edges, modules, methods (including `self.foo`,
  `foo?`, `foo!`), `attr_reader/writer/accessor`, `require` /
  `require_relative`, and top-level constants. Rails smoke test hit
  **87.8%** — below the 91.2% Rust bar, accepted as the Ruby-without-
  Sorbet ceiling (scip-ruby falls back to `# typed: false` heuristics
  when Sorbet annotations are absent).
- **PHP support** via `scip-php` (Milestone P, third-party indexer by
  davidrjenni). PhpParser handles namespaces, classes / interfaces /
  traits / enums, `public/protected/private function`, properties,
  constants, and method / static / instance calls; Allman-brace lookahead
  (scan up to 10 lines for the opening `{`) was needed for Laravel's
  style. Laravel smoke test hit **88.6%**.
- **C# / VB.NET support** via `scip-dotnet` (Milestone N). DotnetAdapter
  sets `DOTNET_ROLL_FORWARD=LatestMajor` so scip-dotnet's net9 target
  runs on systems with only net10 installed. Serilog smoke test hit
  **83.7%**.
- **C / C++ support** via `scip-clang` (Milestone C). ClangAdapter emits
  scheme `cxx` (verified from real indexes; not `scip-clang` as a name
  might suggest). Discovery gives top priority to root-level
  `compile_commands.json` so Bear-backed Makefile builds and top-level
  CMake projects work as a single unit. ClangParser avoids regex
  catastrophic backtracking (which hit >20s on Redis `src/dict.c` in
  an early draft) via a hand-rolled `_extract_function_name` that scans
  right-to-left through the argument list. Redis smoke test hit
  **79.2%**; headers lag .c files (63.3% vs 81.2%) due to scip-clang
  compdb-coverage limits on transitively-included headers.

### Changed

- **CALLS resolution rose above the 90% bar on 20 of 26 measured
  corpora** (up from ~10 pre-sweep) via a focused language-general
  filter pass — zero codebase-specific entries. The sweep expanded
  `STDLIB_FILTER` / `STDLIB_PREFIXES` across every supported language:
  - **TypeScript / JavaScript**: `_JSTS_CONTROL_KEYWORDS` filter in
    the `TSParser` regex fallback (rejects `if(`, `while(`, `for(`,
    `switch(` captures — removes ~37k spurious unresolved on the
    TypeScript compiler alone), tslib emit helpers (`__awaiter`,
    `__generator`, `__rest`, `__spreadArray`, `__classPrivateFieldGet`,
    `__addDisposableResource`, …), DOM APIs (`textContent`,
    `getAttribute`, `classList`, `appendChild`, …), Array/Object
    statics (`isArray`, `fromEntries`, `defineProperty`, …),
    Playwright (`page.*`), Vitest/Jest todo / only / each / snapshot
    modifiers, React core hooks + TanStack Query + Node `assert`
    strict variants + Lingui / nanoid / uuid.* / date-fns / RN-Expo
    surface. ast-grep now rejects `$FUNC` captures that include parens
    or non-identifier characters (stops curried-call shape like
    `test.runIf(isBuild)(…)` from polluting CALLS targets).
  - **C#**: `var` added to `_DOTNET_CONTROL_KEYWORDS` (fixes ~1k/file
    spurious `var (` captures on modern C# codebases) along with the
    full modern declaration vocabulary (`record`, `partial`, `sealed`,
    `override`, `static`, `params`, `required`, `scoped`, `file`,
    `global`) and every primitive alias (`object`, `string`, `bool`,
    `int`, `long`, `byte`, `char`, `float`, `double`, `decimal`,
    `nint`, `nuint`). Moq (`It.IsAny`, `Mock.Of`, `Setup`, `Verify`,
    `ReturnsAsync`), FakeItEasy (`A.CallTo`, `A.Fake`, `A.Dummy`),
    ASP.NET Core DI (`Add*`, `Use*`, `Map*`, `GetRequiredService`),
    .NET Reflection + Span/Memory, `System.String` statics, Selenium
    WebDriver.
  - **Java**: AssertJ (full `assertThat*` surface, `hasCauseInstanceOf`,
    `isThrownBy`, `satisfies`), BDDMockito (`given`, `willReturn`,
    `willThrow`), Project Reactor (`StepVerifier`, `expectNext`,
    `flatMap`, `switchIfEmpty`, `subscribeOn`), Mockito static
    (`mockStatic`, `mockConstruction`, `invocation.getArgument`),
    Hamcrest matcher vocabulary, Apache Lucene types, ANTLR-generated
    runtime, `java.time` / `java.util.Properties` / `java.nio.Buffer`
    / `java.io.File`, `java.sql.*` + `java.util.Locale.*`,
    `jakarta.servlet` + `javax.xml.sax`, reflection extras.
  - **Go**: `encoding/binary` (`BigEndian/LittleEndian.Uint*/PutUint*`),
    `sync` lock methods (`Lock`, `Unlock`, `RLock`, `RUnlock`,
    `Add`/`Done`/`Wait`), `bytes.Buffer` + `strings.Builder` common
    methods, `golang.org/x/sync/errgroup`, `go.uber.org/goleak`,
    AWS SDK v2 helpers (`aws.Int32`, `aws.String`, …), ULID,
    `klog.`/`ginkgo.`/`Gomega.`/`testify.` prefixes.
  - **Rust**: clap CLI (`value_name`, `help_heading`, `get_one`,
    `get_flag`, `long_about`, subcommand builders), snapbox test
    (`stdout_eq`, `stderr_eq`, `subset_matches`), integer arithmetic
    (`wrapping_*`, `saturating_*`, `checked_*`, `overflowing_*`,
    `rotate_*`, `count_ones`/`_zeros`), atomic compare-exchange /
    fetch variants, pointer `as_ptr`/`as_mut_ptr`/`copy_nonoverlapping`/
    `read_volatile`, slice `is_ascii_*`/`split_whitespace`/
    `char_indices`, fd `from_raw_fd`/`as_raw_fd`/`borrow_raw`.
  - **C / C++**: TCL C API (`Tcl_*` prefix — sqlite bindings),
    LLVM C API (`LLVM*` — postgres JIT), CPython C API (`Py_*`,
    `PyObject_*`, `PyDict_*`, …), Win32 C API (`CreateFile`,
    `GetLastError`, `LoadLibrary`, registry, `HeapAlloc`, …), POSIX
    pthread full surface (`pthread_mutex_*`, `pthread_cond_*`,
    `pthread_rwlock_*`, `pthread_spin_*`, `pthread_barrier_*`),
    stdarg (`va_start`, `va_end`, `va_arg`, `va_copy`), C11
    stdatomic, BSD-specific strings (`strlcpy`, `strlcat`, `arc4random_*`,
    `reallocarray`), BSD socket byte-order (`ntohs`, `htonl`, …) +
    full socket API, OpenSSL (`BN_`, `HMAC_`, `EC_KEY_`, `ENGINE_`),
    jemalloc (public `mallocx`/`xallocx`/`rallocx`/`sallocx`/
    `dallocx`/`nallocx` + internals `edata_`, `emap_`, `sz_`, `pa_`,
    `hpa_`, `prof_`, `tsd_`, `tsdn_`, `atomic_*_zu`).
  - **PHP**: PHPUnit `assertIs*` full surface + PHPUnit:: qualified
    variants, Mockery extras (`makePartial`, `withArgs`), Carbon +
    PHP `DateTime` full method set, `version_compare`, binary /
    encoding / output stdlib (`pack`, `unpack`, `bin2hex`,
    `curl_setopt_array`, `ob_start`/`ob_end_flush`, `strspn`, `strtr`,
    `addslashes`), PSR-7 HTTP helpers (`getStatusCode`, `withHeader`,
    `withBody`, `getUri`).
  - **Ruby**: Full `String` / `Enumerable` / `Hash` / `IO` method
    surface (`gsub`, `scan`, `end_with?`, `is_a?`, `key?`, `sysread`,
    `each_with_object`, `tally`, …), `SecureRandom`/`Addrinfo`/
    `Socket`/`TCP*`/`UDP*`/`UNIX*`/`IPAddr`/`Resolv`/`Zlib`/`CGI`/
    `ERB`/`Liquid::` prefixes, JRuby Java-side API (`RubyString.`,
    `RubyArray.`, `RubyHash.`, `ByteList.`, `ThreadContext.`, …) +
    bare `newString`, `newFixnum`, `callMethod`, `getRuntime`.
  - **Dart**: `package:test_descriptor` (`d.file`, `d.dir`,
    `d.nothing`), `package:test_process` (`shouldExit`), `dart:typed_data`
    fromList constructors, `dart:convert` (`JsonEncoder`, `Utf8Decoder`,
    `LineSplitter`), `package:shelf` Response helpers, `package:args`
    CLI builders (`addFlag`, `addOption`, `addSubcommand`,
    `wasParsed`), `package:yaml` loaders, Dart SDK error types
    (`StateError`, `UnsupportedError`, `FormatException`, …),
    pub_semver (`Version`, `VersionConstraint`).
- **`is_non_project_call` now splits on `->` in addition to `.` and
  `::`** so PHP `$obj->method()` and C `struct_ptr->method()` resolve
  their bare method name against `STDLIB_FILTER` — previously
  `headers->set`, `response->getStatusCode` never matched. Laravel /
  Symfony resolution climbed 88% → 91%.

### Fixed

- **SCIP incremental re-indexing** no longer silently degrades to zero
  for adapters outside rust / typescript / python. `_hash_project` now
  falls back to `_hash_generic_adapter` (walks the adapter's declared
  extensions + hashes paths + bytes) for java / go / ruby / php /
  dotnet / clang / dart, instead of raising `ValueError: Unknown project
  type`. Before this fix, every second+ index produced a `.scip`-less
  graph that looked successful in logs but had no SCIP-resolved CALLS
  edges.
- **SCIP Strategy 1 name match.** `ScipIndex.resolve`'s exact
  (file, line) lookup used to fall back to the first candidate in the
  definitions map when no name matched. Downstream, the cross-crate
  name check in `generate.py` would reject those mismatched targets,
  but by then Strategy 2 (fuzzy resolve by name) had already been
  skipped. Tighten Strategy 1 to require a name match; non-matching
  line lookups now fall through to fuzzy. Measured: pub Dart 52% →
  71%, Redis C 79.2% → 80.4%.
- **`changes` no longer crashes on non-UTF-8 git diff output.**
  `GitHistoryAnalyzer._run_git` previously used `subprocess.run(text=True)`
  which raises `UnicodeDecodeError` the moment `git diff` emits a byte
  that isn't valid UTF-8 (observed on c-postgres — byte 0x92 — and
  rust-coreutils — byte 0xfd, both carrying Latin-1 content in older
  source files). All subprocess callers that capture stdout/stderr
  (`git_history.py`, `scip/cache.py`, `scip/adapter.py`, `ast_grep.py`)
  now decode with `encoding="utf-8", errors="replace"` so a stray
  non-UTF-8 byte becomes `U+FFFD` instead of aborting the tool.
  Regression: `tests/test_git_history.py::TestGitHistoryNonUtf8Output`
  seeds a throwaway repo with 0x92 / 0xfd bytes and asserts
  `_run_git` + `get_changes` stay clean.
- **User-input numeric parsing is crash-safe.** The web API's
  `?limit=foo` / `?depth=bar` inputs and the `DESCRY_SCIP_WORKERS` /
  `DESCRY_PRIME_THREADS` / `DESCRY_SCIP_TIMEOUT` env-var overrides
  previously called `int(raw)` directly; a non-integer value raised
  `ValueError` and surfaced as a 500 (web) or a hard crash during
  SCIP cache init (env var). Web handlers now go through the new
  `_int_param(request, name, default)` helper that returns the
  default on parse failure; env-var paths catch `ValueError`, log a
  warning, and fall through to the auto-tuned default.
- **Exception chains preserved on re-raise.** Three sites in
  `git_history.py` and one in `embeddings.py` re-raised exceptions
  without `from err` / `from None`, which hid the underlying cause in
  tracebacks. Now `FileNotFoundError` → `GitError("git not found")` /
  `TimeoutExpired` → `GitError("timed out")` / `FileExistsError` →
  `TimeoutError("embeddings lock")` all use `from None` (cause
  already captured in the message), and `_verify_git`'s internal
  re-raise uses `from err` to preserve the inner git failure.
- **`descry.__version__` matches `pyproject.toml`.** Since the 0.1.1
  tag, the package version string was `"0.1.0"` in `__init__.py` but
  `"0.1.1"` in `pyproject.toml`, so `descry health` reported the
  wrong version. Bumped `__init__.py` and added this to the release
  gate so a future mismatch is caught before tagging.
- **Documented security invariants now match the code.** `CLAUDE.md`
  said `descry-web` "allows CORS `*`"; in reality the code omits
  `CORSMiddleware` entirely so browsers enforce same-origin by
  default (with `TrustedHostMiddleware` defeating DNS-rebinding).
  The subprocess-sanitization list also named only `scip-typescript`
  as the representative SCIP indexer — now enumerates all ten.
- **Implicit-`Optional` annotations in `query.py` made explicit.**
  11 signatures across `flow` / `trace_flow_structured` /
  `get_context_prompt` / `find_trait_impls` used `x: int = None`
  (which PEP 484 prohibits); now `int | None = None`.
- **Pure-JavaScript repos no longer drop every CALLS edge.** ast-grep
  has separate `typescript` and `javascript` grammars; the
  `extract_calls_typescript` / `extract_imports_typescript` helpers
  hardcoded `-l typescript` for the subprocess, which returns no
  matches when fed `.js` / `.jsx` / `.mjs` / `.cjs`. Caught by the
  pre-publish corpus sweep against express (100% .js): scip-typescript
  emitted 9,702 references but the graph had **0 CALLS edges**. New
  `_ast_grep_lang_for(file_path)` routes to `javascript` for JS-family
  extensions; express now resolves 58% of CALLS (graph went 1,353 →
  6,117 edges). 0.1.x users on pure-JS repos were silently affected.

### Changed

- **Configuration docs now cover every parsed field.** README's
  `[cross_lang]` TOML section (with `openapi_path`,
  `backend_handler_patterns`, `frontend_api_patterns`, `api_prefixes`)
  and `[timeouts] index_minutes` were parsed at runtime but undocumented;
  added. Four `DESCRY_*` env vars (`SCIP_WORKERS`, `SCIP_TIMEOUT`,
  `PRIME_THREADS`, `AST_GREP_MAX_FILES`) were honoured by the code but
  missing from the environment-variables table; added.

### Known limitations (surfaced by the pre-publish corpus sweep)

These aren't regressions in 0.2.0 — they're scenarios the new corpus
coverage made visible. Documenting so users know what to expect.

- **Kotlin codebases need scip-java to be useful.** Descry has no
  `.kt` / `.scala` parser baseline; Kotlin and Scala source files are
  walked by the file walker but produce no nodes when scip-java
  doesn't run. scip-java in turn fails on Kotlin-DSL Gradle projects
  that use precompiled-script-plugins (observed on both `ktorio/ktor`
  and `Kotlin/kotlinx.coroutines` — same upstream task-ordering
  error). Practical effect: pure-Kotlin repos with that build
  pattern will produce a near-empty graph. Mixed Java+Kotlin repos
  index the Java side only. The `JavaAdapter` now pre-flights for
  this scenario and skips scip-java with a clear log message
  ("scip-java incompatibility: kotlin-dsl precompiled-script-plugins
  in <path>") instead of a confusing 1100-byte stderr dump from the
  failed Gradle invocation. Detection covers `buildSrc/` and any
  `includeBuild()` participants in `settings.gradle{,.kts}`.
- **C++ resolution ceiling without scip-clang.** scip-clang requires
  a working `compile_commands.json` with all transitive deps
  available (Boost, gflags, glog, fmt, etc. for `facebook/folly`).
  When deps are missing scip-clang is skipped and the regex
  `ClangParser` baseline reaches ~46–54% on modern template-heavy
  C++ (folly 46%, abseil-cpp 54%). With a working compdb scip-clang
  contributes ~40% of resolved CALLS; the rest is regex fallback.
  Plain C codebases (redis, sqlite, postgres, curl) sit at 72–80%
  on regex baseline alone.
- **scip-java does not support Ant.** Maven, Gradle, sbt, and Mill
  are supported; Ant projects (e.g. `apache/cassandra`) fall back
  to the regex `JavaParser`, which still reaches ~86% on its own.
- **scip-dart needs `dart pub get` first.** scip-dart aborts with
  "Unable to locate packageConfig" when the project hasn't been
  bootstrapped. Run `dart pub get` (or `flutter pub get` for Flutter
  packages) before `descry index` to get type-aware Dart resolution.
- **scip-dotnet honours `global.json` SDK pins.** When a project
  pins an SDK not installed on the host (e.g. `PowerShell/PowerShell`
  pins .NET 11 preview), scip-dotnet's adapter detects the
  incompatibility and falls back to the regex `DotnetParser`. The
  regex baseline alone reached 90% on PowerShell, so the fallback
  is usually fine.
- **Pinned Rust toolchains need `rust-analyzer` installed.** A
  `rust-toolchain.toml` pin (e.g. `helix` pins `1.90.0`) means
  rust-analyzer must be installed for that specific toolchain
  via `rustup component add rust-analyzer --toolchain <version>`.
  Otherwise scip-rust emits per-crate warnings and the regex
  `RustParser` baseline carries the project (~80%).

## [0.1.1] — 2026-04-17

Patch release focused on making cross-language tracing actually configurable,
plus a round of dead-code cleanup. No breaking changes.

### Fixed

- **Cross-language tracing is now configurable from `.descry.toml`.** The
  `DescryConfig.openapi_path` field existed in 0.1.0 but had no TOML loader
  and the web `/api/cross-lang` handler hardcoded `public/api/latest.json`,
  so custom spec locations silently did nothing. Added a `[cross_lang]`
  section with `openapi_path`, `backend_handler_patterns`,
  `frontend_api_patterns`, and `api_prefixes` keys. Web and CLI/MCP now
  both honour the config and pass all four through to `CrossLangTracer`.
- `[cross_lang] openapi_path` is containment-checked against the project
  root; a crafted `.descry.toml` cannot point the indexer at files outside
  the configured project.

### Removed

Dead code with no call sites anywhere in the tree was pruned:

- `DescryConfig.project_markers` (auto-detect used a module constant).
- `DescryConfig.use_tree_sitter_ts` (scaffolding field with no consumer).
- `descry.query.MAX_INLINE_THRESHOLD` constant.
- `CrossLangTracer.endpoint_to_node_id()` and module-level
  `_create_cross_lang_edges()` helper.
- `SemanticSearcher._find_similar()` (distinct from
  `GraphQuerier._find_similar_nodes`, which remains).
- `descry.ast_grep.extract_imports_typescript_batch()`.
- `ScipIndex.get_definition_location()` and `get_symbol_info()`.
- `TypeScriptSymbolTable.file_dir` attribute.
- `DescryService._clear_dedup_cache()` (superseded by `reset_caches()` in
  the 0.1.0 hardening sweep).

A `vulture --min-confidence 80` sweep on `src/descry/` now reports zero
findings outside generated protobuf code.

## [0.1.0] — 2026-04-16

Initial public PyPI release. Descry is a polyglot codebase knowledge graph toolkit with three interfaces: CLI (`descry`), MCP server (`descry-mcp`), and local web UI (`descry-web`).

### Features

- **Indexer**: Parses Rust, Python, TypeScript, JavaScript, and Svelte into a cached knowledge graph of symbols (functions, classes, constants) and edges (calls, imports, defines).
- **SCIP integration**: Optional type-aware call resolution via `rust-analyzer` and `scip-typescript`; regex fallback otherwise.
- **Semantic search**: Optional embeddings via sentence-transformers (Jina code embeddings by default; model pinned by revision).
- **19 MCP tools**: search, callers, callees, context, flow, path, impls, structure, flatten, cross-lang (preview), churn, evolution, changes, semantic, quick, index, status, ensure, health.
- **Web UI**: Starlette + Alpine.js; 20+ UI panels for browsing the graph visually.
- **Configuration**: `.descry.toml` + env-var overrides for cache dir, timeouts, embedding model, SCIP toolchain, excluded dirs.

### Security

- Git argument injection hardening on all user-controlled inputs (commit ranges, symbol names, file paths, pathspecs).
- Safe embedding-cache storage (JSON sidecar + safe-mode numpy load) with atomic writes and content-addressed cache keys.
- Default embedding model revision pinned for supply-chain integrity; `trust_remote_code` defaults to `False` for user-supplied models.
- TOML-sourced subprocess args (scip toolchain, extra args, embedding model path) are validated.
- Subprocess env sanitized against known credential patterns.
- Web UI path traversal containment on `/api/source` + regular-file / size / text-file checks.
- MCP `descry_index(path=...)` restricted to project root.
- Graph JSON carries `schema_version`; mismatched graphs are rejected with an actionable error.

