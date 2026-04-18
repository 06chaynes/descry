# Changelog

All notable changes to Descry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to pre-1.0 semver (minor-version bumps may include breaking changes; see README versioning stance).

## [Unreleased]

### Added

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

### Fixed

- **SCIP incremental re-indexing** no longer silently degrades to zero
  for adapters outside rust / typescript / python. `_hash_project` now
  falls back to `_hash_generic_adapter` (walks the adapter's declared
  extensions + hashes paths + bytes) for java / go / ruby / php /
  dotnet / clang, instead of raising `ValueError: Unknown project type`.
  Before this fix, every second+ index produced a `.scip`-less graph
  that looked successful in logs but had no SCIP-resolved CALLS edges.

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
- **18 MCP tools**: search, callers, callees, context, flow, path, impls, structure, flatten, cross-lang (preview), churn, evolution, changes, semantic, quick, index, status, ensure, health.
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

