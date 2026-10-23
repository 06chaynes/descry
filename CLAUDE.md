# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Development uses `uv` + `just`. The `justfile` wraps all common tasks — prefer it over raw commands:

```bash
just install          # uv venv .venv + uv pip install -e ".[dev]"
just test             # pytest tests/
just test-v           # pytest tests/ -v
just lint             # ruff check src/ tests/ tools/
just fmt              # ruff format src/ tests/ tools/
just clean            # remove .descry_cache, dist, caches, __pycache__
```

Run a single test or filter:

```bash
just test tests/test_handlers.py                 # one file
just test tests/test_handlers.py::test_index     # one test
just test -k "cross_lang and not slow"           # by expression
```

Python 3.11+ required (enforced by `pyproject.toml`). Pre-commit gate: `just lint && just fmt && just test` all pass (2787 tests currently — a drop is a regression signal).

Invoke the tool while developing:

```bash
just index            # descry index on current project
just health           # descry health
just web              # descry-web, http://127.0.0.1:8787
just mcp              # descry-mcp (stdio transport)
just search <query>
```

Packaging: `pyproject.toml` defines four console scripts — `descry`, `descry-mcp`, `descry-generate`, `descry-web` — each a thin entry point into `src/descry/`.

## Architecture

**One service, three interfaces.** `DescryService` in `src/descry/handlers.py` owns all business logic. `cli.py`, `mcp_server.py`, and `web/server.py` are thin wrappers that parse their protocol's input, call a service method, and format the output. When adding a feature, implement it as a service method first — the interfaces then expose it with minimal glue.

**Two graph phases:**

1. **Generate** (`generate.py`, invoked via `descry index` or `descry-generate`) — walks the project, parses source files with regex + AST, resolves calls (`CallResolver`) from imports, declared and returned types, scope and the class hierarchy, optionally runs SCIP indexers (rust-analyzer, scip-typescript) for type-aware call resolution, optionally builds embeddings. Writes `.descry_cache/codebase_graph.json`. A call is resolved, left unresolved with the definitions that fit recorded as `candidates`, or dropped as external. Never resolve by name alone when more than one definition fits, never by file path, and never guess the type of a receiver: an unknown receiver stays unresolved.
2. **Query** (`query.py` → `GraphQuerier`) — loads the cached graph and answers callers/callees/flow/path/context queries. All service methods lazy-load the querier; freshness is tracked by graph file mtime.

**Graph schema is versioned.** `src/descry/_graph.py` exports `CURRENT_SCHEMA` (currently `2`) and `load_graph_with_schema()`. Every graph-consuming code path must load via this helper — it raises `GraphSchemaError` on mismatch so stale graphs are rejected rather than silently producing wrong results. If you change the graph shape in a breaking way, bump `CURRENT_SCHEMA`.

**Standard-library knowledge is generated.** `src/descry/stdlib/<lang>.json` (format in `src/descry/_stdlib.py`, loaded lazily per language) is written by `tools/stdlib/gen_<lang>.py` from the language's own machine-readable description and never edited by hand. `class StandardLibrary` in `generate.py` is the one lookup every language plugs into, by supplying how it writes a type and how a name in source maps to the data. Do not add lists of library types, functions or members to a parser or the resolver — extend the generator. `tests/test_stdlib.py` regenerates each file when its source is present (`$DESCRY_STDLIB_SOURCES`, `$JAVA_HOME`).

**Owners are named as source names them.** A parser files members under the type that source names. A block for no type source can name (an anonymous class, a Rust `impl` for a tuple or a type parameter, an unnamed extension) gets an owner whose name no source can write (`class@12`, `object@12`, `<Trait for (A, B)>`, `extension on String`), never the name of a trait, a base or the enclosing type, so no lookup of a type lands on it. An accessor is filed as source writes it (`get size`), a name no method can have. `TestOwnersSourceCannotName` in `tests/test_generate.py` holds a fixture per language; a new parser adds one.

**Resolution changes need numbers.** Measure a resolver or parser change on public repositories before and after (`DESCRY_NO_SCIP=1 DESCRY_NO_EMBEDDINGS=1`, cache outside the repository), compare the graphs edge by edge, and read a random sample of the moved edges against the source. A wrong edge is worse than an unresolved one; a rule must be the language's, never tuned to one repository.

**Config is layered.** `DescryConfig.from_env()` builds config in three ordered steps: `auto_detect()` walks up from cwd looking for project markers (`.git`, `Cargo.toml`, `package.json`, `pyproject.toml`, `.descry.toml`) → `_apply_toml()` merges `.descry.toml` → env vars (`DESCRY_CACHE_DIR`, `DESCRY_NO_SCIP`, `DESCRY_NO_EMBEDDINGS`, `DESCRY_LOG_LEVEL`) override. The CLI, MCP and web entry points construct config this way; `descry-generate` applies the same TOML and env layers to the root it is given instead of auto-detecting one. Don't add an alternate path.

**Optional modules degrade gracefully.** `scip/`, `embeddings.py`, `cross_lang.py`, and `git_history.py` are all imported via `_try_import_*` helpers in `handlers.py`. Missing optional deps (e.g. `sentence-transformers`, `scip-typescript` binary) must never crash the service — the feature becomes unavailable instead. Preserve this when touching imports.

**MCP transport uses stdout.** `mcp_server.py` logs to `stderr` only; never `print()` or log to stdout from anything reachable by the MCP server, or you will corrupt the protocol stream. The `_fenced()` helper wraps verbatim repo content in a nonce-tagged XML envelope to defend against prompt-injection via indexed docstrings — preserve this behavior when returning source content from new tools.

## Security invariants

These are load-bearing — see the `### Security` sections of `CHANGELOG.md`. Do not weaken without a review:

- **The indexer child is isolated from the repository it indexes.** It is spawned with `python -P`, so a `descry/` package inside the repository cannot shadow the installed one.
- **Git runs with the repository's config neutralised.** `_run_git` in `git_history.py` disables hooks, fsmonitor, signature display, external diff, textconv and filter drivers, never lazily fetches (`GIT_NO_LAZY_FETCH`), forces the plain uncoloured `a/` `b/` diff format, and frames log output with NUL bytes. Arguments from callers go through the `_validate_git_*` helpers. New git calls go through `_run_git`.
- **Source is only read from inside the project root.** Graph node paths are untrusted (a graph file can be hand-edited or shipped by a repository); every read goes through `source_path()` in `query.py`, which refuses anything that resolves outside the root or is not a regular file.
- **The default cache directory stays inside the project.** A repository can ship `.descry_cache`, or its `scip` subdirectory, as a symlink to another directory; `DescryConfig.cache_dir_error()` refuses indexing and embedding there, and only `DESCRY_CACHE_DIR` names a cache outside the project.
- **TOML-sourced subprocess args are validated.** `_validate_toolchain`, `_validate_scip_extra_arg`, `_validate_embedding_model` in `handlers.py` reject short flags, shell metacharacters, and out-of-root local paths. New TOML → subprocess plumbing must validate.
- **Web UI is local-only by design.** `descry-web` binds `127.0.0.1`, has no auth, and deliberately omits `CORSMiddleware` so browsers enforce same-origin — a tab on `evil.com` cannot read `/api/source`. State-changing requests are refused unless same-origin (`Origin` / `Sec-Fetch-Site`), so it cannot trigger `/api/index` either. The host middleware rejects non-loopback `Host` headers to defeat DNS-rebinding that would otherwise bypass the 127.0.0.1 bind. `/api/source` enforces project-root containment, rejects non-regular files / non-text content / >10 MiB, and opens every path component with `O_NOFOLLOW`. `/api/index` (and `/api/index/stream`) take no path parameter (always indexes the configured root). `DescryService` runs one index at a time for every interface and stops the indexer's whole process tree when a run ends early. The page's remote scripts carry SRI hashes and every response carries the Content-Security-Policy that `web/server.py` builds from the page's own tags, allowing exactly those asset URLs; a new remote script needs an SRI hash. Don't add a `--host 0.0.0.0` flag, don't add `CORSMiddleware`, and don't expose network listeners.
- **MCP `index(path=...)`** is restricted to the configured project root; a path below it narrows the walk but still writes the root's graph.
- **Embedding cache** uses numpy safe-mode load + JSON sidecar + atomic writes + content-addressed keys. `MODEL_REGISTRY` in `embeddings.py` is the single source of truth for which models descry ships pinned configurations for — each entry carries its revision and its task prompts. **`trust_remote_code` is always `False`**: there is no registry field, loader parameter or config key that enables it, so descry never executes code from a model repository — do not add one. A model named in `.descry.toml` that is not registered is loaded unpinned. Adding a model means adding a registry entry, not editing the loader.
- **The cache key is the embedding recipe's identity.** `_recipe_fingerprint()` folds in the resolved repo id, the pinned revision, `RECIPE_VERSION`, `DOCSTRING_CHAR_LIMIT`, the prompt names, the entry's loading options, and whether source code is embedded (`CODE_CHAR_LIMIT`). Anything that changes the vectors without changing the graph must be added there and `RECIPE_VERSION` bumped, or a stale cache is served against freshly-encoded queries — comparing two different embedding spaces.
- **`node_text()` and `code_excerpt()` are shared with the eval harness.** `tests/eval/corpus.doc_text` delegates to them so the harness's leave-one-out substitute cannot drift from the index rows it replaces. Change the text assembly there, not in a copy, and bump `RECIPE_VERSION`.
- **The default model has a fallback.** With no model configured, `SemanticSearcher` uses `DEFAULT_MODEL_ALIAS` and, when HuggingFace refuses it as gated, builds with `FALLBACK_MODEL_ALIAS` and warns. Searches use whichever model built the index; only a rebuild retries the default. A model named explicitly never falls back.
- **Retrieval changes need numbers.** `tests/eval/` scores candidate models on a project's own graph (docstring→symbol queries, leave-one-out leakage control, paired bootstrap). Run `python tests/eval/compare.py` before changing the default model, the text assembly, or the prompts. Uncontrolled, docstring leakage roughly doubles R@1, so do not evaluate without the leave-one-out swap.

## Versioning

Pre-1.0. The only stable public API is `descry.__version__`. Submodules (`descry.handlers`, `descry.query`, etc.) are **not** stable — refactor freely. Minor bumps (`0.1.x` → `0.2.x`) may break the library API, graph schema, CLI, or MCP tool signatures; patch bumps won't.
