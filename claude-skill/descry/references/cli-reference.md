# descry CLI — full reference

All subcommands read the graph at `.descry_cache/codebase_graph.json` under
the detected project root. Freshness is managed via `descry status` /
`descry ensure` / `descry index`.

## Global discovery

```bash
descry health               # version, graph status, SCIP/embeddings/git state
descry status               # just the graph: exists? nodes? edges? stale?
descry ensure               # regenerate only if missing or stale
descry index [path]         # force full rebuild (SCIP + embeddings)
```

`--max-age-hours N` on `ensure` overrides the config's `max_stale_hours`.

## Search

```bash
descry search <terms...>
    [--compact | --no-compact]   # default compact
    [--limit N]                  # default 10
    [--lang rust|ts|py|...]
    [--crate NAME]
    [--type Function|Class|Method|...]
    [--exclude-tests]

descry semantic <query>
    [--limit N]                  # default 10

descry quick <name>
    [--full]                     # no truncation
    [--brief]                    # minimal; signature + location + counts
```

- `search` is keyword + TF-IDF; fast, deterministic.
- `semantic` is embedding-based; best for behavior-worded queries.
- `quick` is search + `context` in one call. Prefer it when you know the name.

## Call graph

```bash
descry callers <name> [--limit N]           # default 20
descry callees <name> [--limit N]           # default 20

descry flow <start>
    [--direction forward|backward]   # default forward
    [--depth N]                       # default 3, capped by [query] max_depth
    [--target NAME]                   # stop early if reached
    [--inline-threshold N]            # inline callee source below N tokens

descry path <start> <end>
    [--max-depth N]                   # default 10
    [--direction forward|backward]    # default forward
```

- `flow` = full tree; `path` = shortest chain. Both are bounded by
  `[query] max_nodes` / `timeout_ms` from `.descry.toml`.

## Context and structure

```bash
descry context <node_id>
    [--brief | --full]
    [--expand-callees]               # inline small callees
    [--deduplicate]                  # avoid re-emitting same symbol
    [--depth N]                       # default 1
    [--max-tokens N]                  # default 2000
    [--callee-budget N]               # token budget for expanded callees
    [--head-lines N]                  # truncate source head
    [--max-output-tokens N]           # overall cap

descry structure <filename>
descry flatten <class_node_id>       # class API incl. inherited methods
descry impls <method> [--trait-name X]
```

- Node IDs come from `search` / `quick` output (format:
  `FILE:path/to/file.ext::Symbol`).
- `flatten` is especially useful for OOP hierarchies.
- `impls` is the right tool when the user asks "where is this trait / interface
  implemented".

## Change impact and history

Git-backed. Requires a `.git` directory.

```bash
descry churn
    [--time-range "30 days" | "2 weeks" | "since v1.0"]
    [--path-filter "src/"]
    [--limit N]                      # default 20
    [--mode symbols|files|co-change] # default symbols
    [--include-generated]            # default excludes lockfiles etc.

descry evolution <name>
    [--time-range ...]
    [--limit N]                      # default 10
    [--show-diff]
    [--crate NAME]

descry changes
    [--commit-range HEAD~5..HEAD | main...HEAD]
    [--time-range ...]
    [--path-filter ...]
    [--show-callers | --no-show-callers]   # default show
    [--limit N]                            # default 50
```

- `churn --mode co-change` surfaces symbol / file pairs that change together
  across commits — useful for uncovering hidden coupling.
- `changes --commit-range main...HEAD` is the standing-PR pattern.

## Cross-language (frontend → backend)

```bash
descry cross-lang
    [--mode endpoint|list|stats]  # default endpoint
    [--method GET|POST|PUT|PATCH|DELETE]
    [--path /api/v1/...]
    [--tag TAG]
```

Needs an OpenAPI spec. Configure in `.descry.toml`:

```toml
[cross_lang]
openapi_path             = "public/api/openapi.json"
backend_handler_patterns = ["handlers/", "routes/"]
frontend_api_patterns    = ["src/api/"]
api_prefixes             = ["/api/v1", "/api/v2", "/api"]
```

All four are optional — `api_prefixes` defaults to `["/api/v1", "/api/v2",
"/api"]`; the pattern lists default to empty (no filtering).

## Exit codes

- `0` on success.
- `2` when the service returns `ERROR: ...` (most commonly: graph missing).
  Useful in shell chains — `descry callers X && next_step` no longer silently
  advances on a missing graph.

## Environment variables

| Var | Default | Effect |
|-----|---------|--------|
| `DESCRY_LOG_LEVEL` | `WARNING` | Logging verbosity |
| `DESCRY_CACHE_DIR` | `.descry_cache/` | Override cache location |
| `DESCRY_NO_SCIP` | unset | Disable SCIP indexing even if binaries exist |
| `DESCRY_NO_EMBEDDINGS` | unset | Disable semantic search |

Precedence: defaults < `.descry.toml` < environment variables.
