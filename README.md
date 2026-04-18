# Descry

Polyglot codebase knowledge graph with call-graph analysis, semantic search, and SCIP integration. Built for AI coding agents (MCP), with CLI and Web UI interfaces.

Descry indexes your codebase into a knowledge graph of symbols (functions, classes, constants) and their relationships (calls, imports, defines). It supports Rust, Python, TypeScript, JavaScript, Svelte, Go, Java, and more — with type-aware resolution via SCIP for Rust and TypeScript.

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

Descry provides 18 tools, available through all interfaces:

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

Descry works zero-config by auto-detecting your project root (looks for `.git`, `Cargo.toml`, `package.json`, `pyproject.toml`). For customization, add a `.descry.toml` to your project root:

```toml
[project]
excluded_dirs = ["target", "node_modules", "dist", ".git", "__pycache__", "build", "vendor"]
max_stale_hours = 48

[features]
enable_scip = true        # Type-aware resolution (requires rust-analyzer or scip-typescript)
enable_embeddings = true   # Semantic search (requires sentence-transformers)

[embeddings]
model = "jinaai/jina-code-embeddings-0.5b"

[test_detection]
path_patterns = ["/tests/", "/test/", "/__tests__/"]
file_suffixes = ["_test.rs", ".test.ts", ".spec.ts", "_test.py"]

[code_files]
extensions = [".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".svelte", ".go", ".java"]

[git]
churn_exclusions = [".descry_cache/", "Cargo.lock", "package-lock.json"]
timeout = 30

[timeouts]
scip_minutes = 0       # 0 = unlimited
embedding_seconds = 60
query_ms = 4000

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
".svelte" = "svelte"
".proto" = "protobuf"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DESCRY_LOG_LEVEL` | `WARNING` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `DESCRY_CACHE_DIR` | `.descry_cache/` | Override cache directory location |
| `DESCRY_NO_SCIP` | `false` | Disable SCIP indexing |
| `DESCRY_NO_EMBEDDINGS` | `false` | Disable semantic search |

Configuration precedence: defaults < `.descry.toml` < environment variables.

## Language Support

| Language | Parsing | SCIP (Type-Aware) | Requirements |
|----------|---------|-------------------|--------------|
| Rust | Regex + AST | Yes | `rust-analyzer` via rustup |
| TypeScript | Regex (+ Tree-sitter opt-in) | Yes | `scip-typescript` via npm; `descry-codegraph[ast]` for tree-sitter |
| Python | Regex + AST | Yes | `scip-python` via npm |
| Java / Kotlin / Scala | Regex (Java) | Yes | `scip-java` (`coursier install scip-java`) |
| Go | Regex | Yes | `scip-go` (`go install github.com/sourcegraph/scip-go/cmd/scip-go@latest`) |
| Ruby | Regex | Yes | `scip-ruby` gem or direct binary |
| PHP | Regex | Yes | `scip-php` via `composer require --dev davidrjenni/scip-php` (third-party indexer) |
| C# / VB.NET | Regex (C# only) | Yes | `scip-dotnet` via `dotnet tool install --global scip-dotnet` |
| C / C++ | Regex | Yes | `scip-clang` binary + `compile_commands.json` (CMake `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, Meson+Ninja, or `bear -- make`) |
| Dart / Flutter | Regex | Yes | `scip-dart` via `dart pub global activate scip_dart`; requires `dart pub get` at project root |
| JavaScript | Regex (+ Tree-sitter opt-in) | — | `descry-codegraph[ast]` for tree-sitter |
| Svelte | Regex | — | — |

The tree-sitter TS/TSX/JS parser is currently extractor-only (symbol discovery) and ships behind the `ast` extra + `[features] use_tree_sitter_ts = true` in `.descry.toml`. It runs alongside the regex parser and is a stepping stone toward full AST-driven extraction in a future release.

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

4. **Freshness** — `ensure` checks graph age against `max_stale_hours` and regenerates if needed. The MCP server pre-warms the graph on startup.

## Design Notes

### Web UI is local-only (CORS + auth)

`descry-web` is designed as a single-user local development tool. It binds to `127.0.0.1` by default, allows cross-origin requests (`allow_origins=["*"]`), and does not require authentication. This is deliberate:

- The UI is served by the same process that reads your repository; any authentication layer would be a shared-secret between your browser and your own terminal.
- Path traversal and file-serving endpoints are hardened independently of CORS: `/api/source` enforces project-root containment, rejects non-regular files, caps size at 10 MiB, and refuses non-text content (with `O_NOFOLLOW` on the final open to defeat symlink swaps).
- The reindex endpoints accept no path parameter; they always index the configured project root.

**Do not expose `descry-web` to an untrusted network.** If you need remote access, put it behind your own authenticated reverse proxy.

## Versioning

Descry is pre-1.0. Minor version bumps (`0.1.x` → `0.2.x`) may include breaking changes to the library API, graph schema, CLI, or MCP tool signatures. Patch releases (`0.1.0` → `0.1.1`) will not introduce breaking changes. Once we reach `1.0.0`, the project will follow [semver](https://semver.org/) strictly.

The public library API in v0.1 is limited to `descry.__version__`. Submodules (`descry.handlers`, `descry.query`, etc.) are not considered stable API yet.

## Further Reading

- [CHANGELOG](CHANGELOG.md) — release notes.
- [SECURITY](SECURITY.md) — security policy and disclosure.

## License

MIT
