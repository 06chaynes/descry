# Descry

Polyglot codebase knowledge graph with call-graph analysis, semantic search, and SCIP integration. Built for AI coding agents (MCP), with CLI and Web UI interfaces.

Descry indexes your codebase into a knowledge graph of symbols (functions, classes, constants) and their relationships (calls, imports, defines). SCIP-backed type-aware resolution is available for Rust, TypeScript, JavaScript, Svelte, Python, Java (+ Kotlin / Scala), Go, Ruby, PHP, C# (+ VB.NET), C / C++, and Dart. The `.js` / `.jsx` / `.svelte` paths run through `scip-typescript`; pure Kotlin / Scala source needs `scip-java` (which doesn't always work on Kotlin-DSL Gradle projects — see [CHANGELOG](CHANGELOG.md#020--2026-04-20) known limitations).

> ### ⚠️ Disclaimer — please read
>
> **Descry was built with heavy assistance from AI coding agents.** While
> the codebase has been through an internal security and correctness
> review, AI-authored code can still contain subtle defects, missed edge
> cases, or security assumptions that weren't obvious to the reviewer.
> Treat this tool as experimental pre-1.0 software: **do not rely on it
> for safety-critical analysis, and do not point it at code or
> repositories you don't trust.**
>
> **This tool is designed to run locally on your own machine.** The web
> UI (`descry-web`) binds to `127.0.0.1` and is unauthenticated by design —
> the threat model assumes a single trusted user on the host.
>
> - **Do not expose `descry-web` to a network.** No reverse proxy, no
>   public tunnel, no `--host 0.0.0.0`. If you need remote access, put
>   it behind your own authenticated reverse proxy and understand that
>   any browser tab on the host machine can still reach it via
>   `localhost`.
> - **Do not run Descry on untrusted repositories.** Indexing a
>   repository executes configuration from its `.descry.toml`, walks
>   its file tree, and feeds its docstrings/source to your LLM (if you
>   use the MCP server). The same trust boundary that applies to
>   `cargo build`, `npm install`, and opening the repo in your IDE
>   applies here.
> - **Report security issues privately** — see [SECURITY.md](SECURITY.md).

## Quick Start

```bash
# Install with all optional features
pip install descry-codegraph[all]

# Index your project
cd your-project
descry index

# Search for symbols
descry search authenticate

# Find callers of a function
descry callers validate_token

# One-step lookup (search + full context)
descry quick handle_request
```

## Interfaces

| Interface | Command | Use Case |
|-----------|---------|----------|
| **CLI** | `descry <command>` | Interactive terminal use |
| **MCP Server** | `descry-mcp` | AI coding agents (Claude Code, etc.) |
| **Web UI** | `descry-web` | Visual exploration at `http://127.0.0.1:8787` |
| **Pi Extension** | See `pi-extension/` | Pi coding agent integration |
| **Claude Code Skill** | See `claude-skill/` | Teaches Claude when/how to reach for descry |

## MCP Setup

### Claude Code

Add to your Claude Code MCP settings (`.claude/settings.json` or global):

```json
{
  "mcpServers": {
    "descry": {
      "command": "descry-mcp",
      "args": []
    }
  }
}
```

Or with a specific Python path:

```json
{
  "mcpServers": {
    "descry": {
      "command": "/path/to/venv/bin/descry-mcp",
      "args": []
    }
  }
}
```

### Other MCP Hosts

Descry uses the standard MCP stdio transport. Any MCP-compatible host can spawn `descry-mcp` as a subprocess.

## Tools

Descry provides 19 tools, available through all interfaces:

| Tool | Description |
|------|-------------|
| `health` | Diagnostic check — version, graph status, feature availability |
| `status` | Graph existence and freshness |
| `ensure` | Ensure graph exists and is fresh (regenerates if stale) |
| `index` | Regenerate graph, SCIP indices, and embeddings |
| `search` | Search symbol names and docstrings (keyword + semantic) |
| `semantic` | Pure semantic search using embeddings only |
| `quick` | Find symbol and show full context in one step |
| `callers` | Find all callers of a symbol |
| `callees` | Find what a symbol calls |
| `context` | Full context for a symbol — source, callers, callees, tests |
| `flow` | Trace call flow from a starting symbol (forward/backward) |
| `path` | Find shortest call path between two symbols |
| `structure` | Show file structure — imports, classes, functions |
| `flatten` | Show effective API of a class including inherited methods |
| `impls` | Find all implementations of a trait/interface method |
| `cross-lang` | Trace frontend API calls to backend handlers via OpenAPI |
| `churn` | Find code churn hotspots (symbols, files, or co-change pairs) |
| `evolution` | Track how a symbol has changed over time |
| `changes` | Analyze change impact for a commit range |

## Configuration

Descry works zero-config by auto-detecting your project root. The walker looks for any of: `.git`, `.descry.toml`, `Cargo.toml`, `package.json`, `pyproject.toml`, `setup.py`, `go.mod`, `Gemfile`, `composer.json`, `build.gradle{,.kts}`, `settings.gradle{,.kts}`, `pom.xml`, `build.sbt`, `global.json`, `pubspec.yaml`, `CMakeLists.txt`, or `compile_commands.json`.

For customization, add a `.descry.toml` to your project root.

> **Heads-up on list-typed fields.** Several settings below are **lists that REPLACE descry's defaults** when set, not merge. These include `[project] excluded_dirs`, `[code_files] extensions`, `[test_detection] path_patterns` / `file_suffixes`, and `[git] churn_exclusions`. Leave them out unless you have a specific reason — descry's defaults already cover a wide language matrix. If you do set them, copy the full default list (see `_DEFAULT_*` constants in `src/descry/handlers.py`) and modify from there. `[syntax.lang_map]` is the one list-shaped field that *merges* with defaults.

```toml
[project]
# OPTIONAL — REPLACES the 25-element default. Leave unset to keep defaults
# (.git, .gradle, .next, .svelte-kit, .venv, node_modules, target, build, …).
# excluded_dirs = ["target", "node_modules", "dist", "build", "vendor"]
max_stale_hours = 24

[features]
enable_scip = true         # Type-aware resolution (auto-detects which indexers are on PATH)
enable_embeddings = true   # Semantic search (requires sentence-transformers)

[embeddings]
model = "jinaai/jina-code-embeddings-0.5b"

[test_detection]
# OPTIONAL — REPLACES defaults. Defaults cover Rust/Python/TS/Go/Ruby/Java/Kotlin/
# Scala/PHP/C#/Dart/C/C++ test conventions.
# path_patterns = ["/tests/", "/test/", "/__tests__/"]
# file_suffixes = ["_test.rs", ".test.ts", ".spec.ts", "_test.py"]

[code_files]
# OPTIONAL — REPLACES the 30-element default (.rs/.py/.ts/.tsx/.js/.jsx/.svelte/
# .go/.java/.kt/.scala/.rb/.rake/.gemspec/.php/.cs/.vb/.c/.cc/.cpp/.cxx/.cu/.h/
# .hh/.hpp/.hxx/.dart/.css/.scss/.html). Leave unset to keep defaults.
# extensions = [".rs", ".py", ".ts", ".tsx", ".go", ".java"]

[git]
# OPTIONAL — REPLACES defaults (.descry_cache/, .beads/, Cargo.lock,
# package-lock.json, yarn.lock, pnpm-lock.yaml).
# churn_exclusions = [".descry_cache/", "Cargo.lock", "package-lock.json"]
timeout = 30

[timeouts]
scip_minutes = 0       # 0 = unlimited
embedding_seconds = 60
query_ms = 4000
index_minutes = 30     # Timeout for `descry index` subprocess

[query]
max_depth = 3
max_nodes = 100
max_children_per_level = 10
max_callers_shown = 15

[scip]
extra_args = ["--exclude-vendored-libraries"]
skip_crates = []         # Crate names to skip during SCIP indexing

[scip.rust]
toolchain = "1.92.0"     # Pin rust-analyzer version via rustup

[syntax.lang_map]
# MERGES with defaults (additive). Maps file extensions to syntax-highlight names.
".svelte" = "svelte"
".proto" = "protobuf"

[cross_lang]
# Frontend -> backend handler tracing via OpenAPI spec.
openapi_path = "public/api/openapi.json"   # must resolve inside project_root
backend_handler_patterns = ["backend/src/routes"]  # path substrings; empty = no filter
frontend_api_patterns = ["webapp/src/lib/api"]     # same
api_prefixes = ["/api/v1", "/api/v2", "/api"]      # stripped when matching spec paths
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DESCRY_LOG_LEVEL` | `WARNING` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `DESCRY_CACHE_DIR` | `.descry_cache/` | Override cache directory location |
| `DESCRY_NO_SCIP` | `false` | Disable SCIP indexing |
| `DESCRY_NO_EMBEDDINGS` | `false` | Disable semantic search |
| `DESCRY_SCIP_WORKERS` | auto | Max parallel SCIP indexer workers (overrides the memory-based default) |
| `DESCRY_SCIP_TIMEOUT` | auto | Per-project SCIP indexer timeout in minutes (`0` / `none` disables) |
| `DESCRY_PRIME_THREADS` | auto | Threads used when pre-warming `rust-analyzer` cache |
| `DESCRY_AST_GREP_MAX_FILES` | `5000` | Skip ast-grep per-file invocation on repos larger than this many target-language files |

Configuration precedence: defaults < `.descry.toml` < environment variables.

## Language Support

| Language | Parsing | SCIP (Type-Aware) | Requirements |
|----------|---------|-------------------|--------------|
| Rust | Regex (+ ast-grep when `sg` on PATH) | Yes | `rust-analyzer` via rustup |
| TypeScript | Regex (+ ast-grep when `sg` on PATH) | Yes | `scip-typescript` via npm |
| JavaScript | Regex (+ ast-grep when `sg` on PATH) | Yes (via `scip-typescript`) | `scip-typescript` via npm |
| Svelte | Regex | Yes (via `scip-typescript`) | `scip-typescript` via npm |
| Python | Regex + Python `ast` module | Yes | `scip-python` via npm |
| Java / Kotlin / Scala | Regex (Java only — `.kt` / `.scala` produce no nodes when scip-java doesn't run) | Yes | `scip-java` (`coursier install scip-java`); see [CHANGELOG](CHANGELOG.md#020--2026-04-20) for the Kotlin-DSL Gradle limitation |
| Go | Regex | Yes | `scip-go` (`go install github.com/sourcegraph/scip-go/cmd/scip-go@latest`) |
| Ruby | Regex | Yes | `scip-ruby` gem or direct binary |
| PHP | Regex | Yes | `scip-php` via `composer require --dev davidrjenni/scip-php` (third-party indexer) |
| C# / VB.NET | Regex (C# only) | Yes | `scip-dotnet` via `dotnet tool install --global scip-dotnet` |
| C / C++ | Regex | Yes | `scip-clang` binary + `compile_commands.json` (CMake `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, Meson+Ninja, or `bear -- make`) |
| Dart / Flutter | Regex | Yes | `scip-dart` via `dart pub global activate scip_dart`; requires `dart pub get` at project root |

`ast-grep` is the `sg` Rust binary (install via Homebrew or `cargo install ast-grep`) — when present on `PATH`, descry uses it for higher-fidelity call extraction in Rust, TypeScript, and JavaScript files. It's auto-disabled on TS/JS corpora over `DESCRY_AST_GREP_MAX_FILES` (default 5,000) where per-file subprocess overhead would dominate. The `descry-codegraph[ast]` Python extra installs `tree-sitter*` packages that are scaffolding for a future AST-driven parser at `src/descry/tree_sitter_parser.py` — currently not wired into the active pipeline.

SCIP provides precise call-graph resolution (resolving which specific function is called through traits, generics, etc.). Without SCIP, Descry falls back to regex-based name matching which handles most cases but may produce false positives on overloaded names.

## Installation

### Minimal (graph + CLI only)

```bash
pip install descry-codegraph
```

### With MCP server

```bash
pip install descry-codegraph[mcp]
```

### With Web UI

```bash
pip install descry-codegraph[web]
```

### With semantic search

```bash
pip install descry-codegraph[embeddings]
```

### Everything

```bash
pip install descry-codegraph[all]
```

### Development

```bash
git clone https://github.com/06chaynes/descry.git
cd descry
just install    # Creates venv and installs with dev deps
just test       # Run tests
just lint       # Ruff linting
just fmt        # Ruff formatting
```

Requires [uv](https://github.com/astral-sh/uv) and [just](https://github.com/casey/just).

## How It Works

1. **Index** — Descry walks your codebase, parses source files into an AST-like representation, and builds a graph of symbols and their relationships. If SCIP is available, it overlays type-aware call resolution.

2. **Cache** — The graph is cached as JSON in `.descry_cache/codebase_graph.json`. Embeddings are cached separately. SCIP indices are cached per-crate/package.

3. **Query** — All tools query the cached graph. Keyword search uses TF-IDF scoring. Semantic search uses sentence-transformer embeddings. Call-graph traversal follows edges in the graph.

4. **Freshness** — `ensure` regenerates the graph when its age exceeds the `--max-age-hours` flag (CLI) / `max_age_hours` argument (MCP), defaulting to 24 hours. The `[project] max_stale_hours` config drives the "STALE" badge in `descry status` output but does not by itself trigger regeneration. The MCP server pre-warms the graph on startup.

## Design Notes

### Web UI is local-only — same-origin + DNS-rebind defenses

`descry-web` is designed as a single-user local development tool. It binds to `127.0.0.1` by default, has no authentication, and **deliberately omits `CORSMiddleware`** so browsers enforce same-origin by default — a tab on `evil.com` cannot read `/api/source` or trigger `/api/index`. This is deliberate:

- The UI is served by the same process that reads your repository; any authentication layer would be a shared-secret between your browser and your own terminal.
- `TrustedHostMiddleware` rejects requests whose `Host` header isn't loopback (`127.0.0.1`, `localhost`, `::1`), defeating DNS-rebinding attacks that would otherwise bypass the 127.0.0.1 bind.
- The `--host` flag is validated by `_loopback_host` and rejects non-loopback values with a clear error pointing at the reverse-proxy guidance.
- Path-traversal hardening on `/api/source` is independent of any CORS posture: it enforces project-root containment, rejects non-regular files, caps size at 10 MiB, refuses non-text content, and uses `O_NOFOLLOW` on the final open to defeat symlink swaps.
- The reindex endpoints (`/api/index`, `/api/index/stream`) accept no path parameter; they always index the configured project root.

**Do not expose `descry-web` to an untrusted network.** If you need remote access, put it behind your own authenticated reverse proxy.

## Versioning

Descry is pre-1.0. Minor version bumps (`0.1.x` → `0.2.x`) may include breaking changes to the library API, graph schema, CLI, or MCP tool signatures. Patch releases (`0.1.0` → `0.1.1`) will not introduce breaking changes. Once we reach `1.0.0`, the project will follow [semver](https://semver.org/) strictly.

The only stable public API in pre-1.0 releases is `descry.__version__`. Submodules (`descry.handlers`, `descry.query`, etc.) are not considered stable API yet — refactor freely.

## Further Reading

- [CHANGELOG](CHANGELOG.md) — release notes.
- [SECURITY](SECURITY.md) — security policy and disclosure.

## License

MIT
