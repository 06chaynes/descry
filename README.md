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

## Embedding models

Semantic search embeds one text per symbol: its name, signature, docstring and
the first 1,500 characters of its source. With no model configured, descry uses
`embeddinggemma`; if HuggingFace refuses that download because the account has
not accepted its licence, it warns and uses `f2llm-v2-0.6b` instead.

```toml
[embeddings]
model = "f2llm-v2-0.6b"       # alias, full HuggingFace repo id, or a local path
```

```bash
descry embedding-models       # what is available, and what is selected
```

| alias | dim | licence | notes |
|---|---|---|---|
| `embeddinggemma` (default) | 768 | Gemma | gated on HuggingFace; 2K context |
| `f2llm-v2-0.6b` (fallback) | 1024 | Apache-2.0 | not gated; index is a third larger |
| `qwen3` | 1024 | Apache-2.0 | not gated |
| `embeddinggemma-2` | 768 | Apache-2.0 | not gated; needs the `embeddinggemma-2` extra; 8K context |
| `jina-code` | 896 | CC-BY-NC-4.0 | not gated; non-commercial licence |

To use the default model, accept its licence once and authenticate:

```bash
# 1. accept the licence at https://huggingface.co/google/embeddinggemma-300m
hf auth login                 # 2. or export HF_TOKEN=...
descry index                  # 3. each index tries the default first
```

Searches use whichever model built the current index, so the switch happens at
the next `descry index`. A model named in `.descry.toml` is used as given: it
never falls back, and naming `f2llm-v2-0.6b` skips the attempt and the warning.

Measured with `tests/eval/` — docstring→symbol queries, leave-one-out leakage
control, paired bootstrap:

**descry** (3,014 nodes, 1,377 queries, Python)

| model | R@1 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| `embeddinggemma` | **0.560** | **0.853** | **0.665** | **0.706** |
| `f2llm-v2-0.6b` | 0.430 | 0.800 | 0.561 | 0.613 |
| `jina-code` | 0.413 | 0.759 | 0.533 | 0.582 |
| `qwen3` | 0.379 | 0.781 | 0.520 | 0.577 |
| `embeddinggemma-2` | 0.373 | 0.698 | 0.488 | 0.533 |

**ripgrep** at `3fce3b5` (3,820 nodes, 691 queries, Rust)

| model | R@1 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| `embeddinggemma` | **0.569** | **0.903** | **0.694** | **0.743** |
| `f2llm-v2-0.6b` | 0.420 | 0.857 | 0.576 | 0.639 |
| `qwen3` | 0.411 | 0.845 | 0.561 | 0.626 |
| `jina-code` | 0.369 | 0.831 | 0.531 | 0.598 |
| `embeddinggemma-2` | 0.356 | 0.810 | 0.515 | 0.582 |

Rerun on your own project — the ranking is measured, not universal:

```bash
python tests/eval/compare.py /path/to/.descry_cache/codebase_graph.json
```

Each entry carries its own pinned revision and task prompts, both of which feed
the cache key — changing model re-embeds rather than silently reusing vectors
from a different embedding space. **descry never loads a model with
`trust_remote_code`** — not for registry entries, and not for a model you name
yourself, which is also loaded unpinned.

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
# OPTIONAL — in a git repository, files git ignores are not indexed.
# Default true; false indexes them too.
# respect_gitignore = true
max_stale_hours = 24

[features]
enable_scip = true         # Type-aware resolution (auto-detects which indexers are on PATH)
enable_embeddings = true   # Semantic search (requires sentence-transformers)

[embeddings]
model = "embeddinggemma"   # see "Embedding models"; `descry embedding-models` lists them

[test_detection]
# OPTIONAL — REPLACES defaults. Defaults cover Rust/Python/TS/Go/Ruby/Java/Kotlin/
# Scala/PHP/C#/Dart/C/C++ test conventions.
# path_patterns = ["/tests/", "/test/", "/__tests__/"]
# file_suffixes = ["_test.rs", ".test.ts", ".spec.ts", "_test.py"]

[code_files]
# OPTIONAL — REPLACES the 35-element default (.rs/.py/.ts/.tsx/.mts/.cts/.js/
# .jsx/.mjs/.cjs/.svelte/.proto/.go/.java/.kt/.scala/.rb/.rake/.gemspec/.php/
# .cs/.vb/.c/.cc/.cpp/.cxx/.cu/.h/.hh/.hpp/.hxx/.dart/.css/.scss/.html). Leave
# unset to keep defaults.
# extensions = [".rs", ".py", ".ts", ".tsx", ".go", ".java"]
# OPTIONAL — larger files are skipped by the indexer (it logs each one) and
# refused by the web source view. Default 10; 0 = no limit.
# max_file_mb = 10
# OPTIONAL — a file whose parsing takes longer is left as far as it got (the
# indexer logs it) and the next one is read. Default 120; 0 = no limit. Not
# enforced on Windows.
# max_parse_seconds = 120

[git]
# OPTIONAL — REPLACES defaults (.descry_cache/, .beads/, Cargo.lock,
# package-lock.json, yarn.lock, pnpm-lock.yaml).
# churn_exclusions = [".descry_cache/", "Cargo.lock", "package-lock.json"]
timeout = 30

[timeouts]
scip_minutes = 30      # per indexer; 0 = unlimited
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
# Frontend -> backend handler tracing via OpenAPI spec. An operation's handler
# is the function its operationId names; where several functions share the
# name, the one whose attributes or decorators write the operation's path
# (`#[utoipa::path(.., path = "/x")]`, `@app.get("/x")`), else the one outside
# test code. A handler named otherwise is found by the path it declares.
openapi_path = "public/api/openapi.json"   # must resolve inside project_root
# OPTIONAL — only functions in files whose path holds one of these are
# handlers. Needed only where the rules above leave a name ambiguous.
# backend_handler_patterns = ["backend/src/routes"]
api_prefixes = ["/api/v1", "/api/v2", "/api"]      # stripped when matching spec paths
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DESCRY_LOG_LEVEL` | `WARNING` (`INFO` for the indexer) | Log level (DEBUG, INFO, WARNING, ERROR) |
| `DESCRY_CACHE_DIR` | `.descry_cache/` | Override cache directory location |
| `DESCRY_NO_SCIP` | `false` | Disable SCIP indexing |
| `DESCRY_NO_EMBEDDINGS` | `false` | Disable semantic search |
| `DESCRY_SCIP_WORKERS` | auto | Max parallel SCIP indexer workers (overrides the memory-based default) |
| `DESCRY_SCIP_TIMEOUT` | auto | Per-project SCIP indexer timeout in minutes (`0` / `none` disables) |
| `DESCRY_PRIME_THREADS` | auto | Threads used when pre-warming `rust-analyzer` cache |

Configuration precedence: defaults < `.descry.toml` < environment variables.

## Language Support

| Language | Parsing | SCIP (Type-Aware) | Requirements |
|----------|---------|-------------------|--------------|
| Rust | Regex | Yes | `rust-analyzer` via rustup |
| TypeScript | Regex | Yes | `scip-typescript` via npm |
| JavaScript | Regex | Yes (via `scip-typescript`) | `scip-typescript` via npm |
| Svelte | Regex | Yes (via `scip-typescript`) | `scip-typescript` via npm |
| Python | Regex + Python `ast` module | Yes | `scip-python` via npm; writes no index without a `pip` in the project's environment (a `uv venv` has none) |
| Java / Kotlin / Scala | Regex (Java only — `.kt` / `.scala` produce no nodes when scip-java doesn't run) | Yes | `scip-java` 0.12.3 (`coursier bootstrap --standalone -o scip-java com.sourcegraph:scip-java_2.13:0.12.3 --main com.sourcegraph.scip_java.ScipJava`), with a JDK and Maven or Gradle on `PATH`; see [CHANGELOG](CHANGELOG.md#020--2026-04-20) for the Kotlin-DSL Gradle limitation |
| Go | Regex | Yes | `scip-go` (`go install github.com/scip-code/scip-go/cmd/scip-go@latest`) |
| Ruby | Regex | Yes | `scip-ruby` gem or direct binary |
| PHP | Regex | Yes | `scip-php` via `composer require --dev davidrjenni/scip-php` (third-party indexer) |
| C# / VB.NET | Regex (C# only) | Yes | `scip-dotnet` via `dotnet tool install --global scip-dotnet` |
| C / C++ | Regex | Yes | `scip-clang` binary + `compile_commands.json` (CMake `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, Meson+Ninja, or `bear -- make`) |
| Dart / Flutter | Regex | Yes | `scip-dart` via `dart pub global activate scip_dart`; requires `dart pub get` at project root |

JavaScript and TypeScript cover `.js`, `.jsx`, `.mjs`, `.cjs`, `.ts`, `.tsx`, `.mts` and `.cts`.

Each call is resolved from evidence: imports (including `go.mod`, `Cargo.toml`, `package.json` and `tsconfig.json`), declared and returned types, lexical scope, the class hierarchy and the language's visibility rules. Names bound by loops, patterns, destructuring and closure parameters are typed from declarations, and each language's standard library is described by generated data (`src/descry/stdlib/`), so a chain through library types keeps its type. A call ends as *resolved* to one definition, *unresolved* when nothing in the source decides — the definitions that fit are kept on the edge as `candidates` — or *external* when it leaves the project; the index summary reports all three. Nothing is matched by name alone when more than one definition fits, nothing by file path, and a receiver whose type is not known is never guessed. SCIP adds type-aware resolution through traits and generics, and is verified for Rust workspaces; the other indexers' output is used only where it names a node in the graph.

Share of project calls resolved without SCIP, measured on public repositories at the commits named:

| Language | Repository | Resolved |
|---|---|---|
| Rust | ripgrep `3fce3b5` | 84% (95% with rust-analyzer) |
| TypeScript | zod `0b216ef` | 93% |
| Go | cobra `adbc881` | 100% |
| Java | gson `845664b`, commons-lang `682a8ff` | 100%, 99% |
| Python | httpx `b5addb6`, flask `d086db8` | 97%, 96% |
| PHP | monolog `3bed304`, guzzle `9393947` | 95%, 93% |
| C# | serilog `bebc771`, Newtonsoft.Json `52fa3ae` (`Src/Newtonsoft.Json`) | 94%, 91% |
| C | jq `a19efdd`, libuv `954c1d1` | 99%, 79% |
| C++ | leveldb `7ee830d`, fmt `c4927e6` | 97%, 86% |
| Dart | shelf `b2cb503`, riverpod `1efba1c` (`packages/riverpod`) | 93%, 91% |
| Ruby | rack `a9833c8`, sinatra `cb22afd` | 70%, 39% |

The remainder is unresolved, not wrong: of 450 resolved edges drawn at random from these repositories and read against the source, every one reaches the definition the call reaches. Ruby is low because the language declares no types and calls inside DSL blocks run with another `self`; libuv because most of its functions exist once for Unix and once for Windows, and the build files, not the source, choose between them.

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

To select `embeddinggemma-2`, install its extra instead (adds Pillow and
torchvision, and requires sentence-transformers 6.1 or newer):

```bash
pip install "descry-codegraph[embeddinggemma-2]"
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

4. **Freshness** — `ensure` regenerates the graph when its age exceeds `[project] max_stale_hours` (default 24), or the `--max-age-hours` flag (CLI) / `max_age_hours` argument (MCP) when one is given. The same setting drives the "STALE" badge in `descry status`. The MCP server pre-warms the graph on startup.

## Design Notes

### Web UI is local-only — same-origin + DNS-rebind defenses

`descry-web` is designed as a single-user local development tool. It binds to `127.0.0.1` by default, has no authentication, and **deliberately omits `CORSMiddleware`** so browsers enforce same-origin by default — a tab on `evil.com` cannot read `/api/source`. Requests that change state (`/api/index`, `/api/index/stream`, `/api/ensure`) are refused unless they come from the page's own origin, so it cannot trigger a reindex either. This is deliberate:

- The UI is served by the same process that reads your repository; any authentication layer would be a shared-secret between your browser and your own terminal.
- `TrustedHostMiddleware` rejects requests whose `Host` header isn't loopback (`127.0.0.1`, `localhost`, `::1`), defeating DNS-rebinding attacks that would otherwise bypass the 127.0.0.1 bind.
- The `--host` flag is validated by `_loopback_host` and rejects non-loopback values with a clear error pointing at the reverse-proxy guidance.
- Path-traversal hardening on `/api/source` is independent of any CORS posture: it enforces project-root containment, rejects non-regular files, caps size at `[code_files] max_file_mb`, refuses non-text content, and opens every path component with `O_NOFOLLOW` to defeat symlink swaps.
- The page's scripts are pinned with Subresource Integrity hashes and a Content-Security-Policy is sent with every response; rendered repository markdown cannot load remote images.
- The reindex endpoints (`/api/index`, `/api/index/stream`) accept no path parameter; they always index the configured project root, one run at a time (a second request gets `409`).

**Do not expose `descry-web` to an untrusted network.** If you need remote access, put it behind your own authenticated reverse proxy.

## Versioning

Descry is pre-1.0. Minor version bumps (`0.1.x` → `0.2.x`) may include breaking changes to the library API, graph schema, CLI, or MCP tool signatures. Patch releases (`0.1.0` → `0.1.1`) will not introduce breaking changes. Once we reach `1.0.0`, the project will follow [semver](https://semver.org/) strictly.

The only stable public API in pre-1.0 releases is `descry.__version__`. Submodules (`descry.handlers`, `descry.query`, etc.) are not considered stable API yet — refactor freely.

## Further Reading

- [CHANGELOG](CHANGELOG.md) — release notes.
- [SECURITY](SECURITY.md) — security policy and disclosure.

## License

MIT. The bundled standard-library data is derived from other projects; see
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES.md).
