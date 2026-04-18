---
name: descry
description: >
  Use descry (a polyglot codebase knowledge graph) for call-graph navigation,
  cross-file code exploration, and change-impact analysis on Rust / TypeScript
  / Python / JavaScript / Svelte / Go / Java projects. Reach for this skill
  whenever the user asks who calls a function, what a function calls, how one
  symbol eventually reaches another, what changed in a commit range and who it
  affects, where a trait or interface is implemented, or wants to find a
  symbol by meaning rather than exact name. Also apply when Claude is about to
  do more than ~3 Grep/Read calls to answer a structural question, when
  exploring an unfamiliar codebase, or when the user mentions "callers",
  "callees", "call flow", "call graph", "impact", "hotspots", "churn", "impls",
  or "who uses this". Prefer descry over grep+read whenever the question spans
  multiple files or requires understanding relationships — even if the user
  doesn't say "descry".
license: MIT
version: 1.0.0
---

# Descry — codebase knowledge graph navigation

Descry indexes a project into a graph of symbols (functions, classes, methods,
constants) and edges (CALLS, IMPORTS, DEFINES). It ships three identical
interfaces over the same service — a CLI (`descry`), a stdio MCP server
(`descry-mcp`), and a local web UI (`descry-web`) — so the right command to
reach for depends only on what's in the session, not on what's being asked.

Optional SCIP resolvers (rust-analyzer, scip-typescript, scip-python) give
type-aware call resolution; optional sentence-transformer embeddings give
semantic search. Without either the graph still works; it just falls back to
regex matching.

This skill's job is to make sure the model reaches for descry instead of
thrashing with Grep+Read whenever the question is about relationships rather
than about the contents of one known file.

## When to use descry

Descry is the right tool when the question is structural — when answering it
requires knowing how pieces of code fit together, not just what a single file
contains.

Reach for it when:

- The user asks "who calls X", "what does X call", or any variation
  ("who uses this", "where is this used", "what depends on this").
- The user wants a path: "how does A eventually reach B", "trace from X".
- The user wants change impact: "what changed in this PR", "what callers are
  affected by this commit range", "what are the hotspots".
- The user wants implementations: "where is this trait implemented",
  "where is this interface implemented".
- The user wants to explore an unfamiliar codebase at any level ("how does
  auth work here", "show me the shape of this file").
- The user describes something by behavior, not name: "find the thing that
  decodes JWTs", "find where we talk to the database" — this is the semantic
  search case.
- Claude is about to run 3+ Grep/Read calls to trace a structural question.
  Each Grep/Read is one round trip; a single `descry callers` / `descry flow`
  answers the same question in one round trip with call-graph-aware ranking
  and inlined source context.
- The codebase is large (hundreds of files) or polyglot (backend + frontend).

## When *not* to use descry

- The user points at a single known file and wants to read or edit it.
  `Read` is the right tool.
- The search is textual and narrow (a specific string in a specific path).
  `Grep` is faster and doesn't need a graph.
- There's no graph yet and the user hasn't asked to index. See "Preconditions"
  below — indexing a large repo can take minutes.
- The question is about runtime behaviour (logs, traces, production
  metrics). Descry is static; it sees source code, not runtime.

When in doubt: if the question is answerable by reading one file, use `Read`;
if it requires cross-file relationship knowledge, use descry.

## Preconditions: is the graph there and fresh?

Every descry query needs a graph cached in `.descry_cache/codebase_graph.json`
under the project root. Before the first query in a session:

```bash
descry status   # Is there a graph? How old is it?
descry ensure   # Regenerate if missing or stale (respects max_stale_hours)
```

`descry ensure` is safe to call unconditionally — it no-ops if the graph is
fresh. For a full rebuild, `descry index` regenerates from scratch (this also
refreshes SCIP indices and embeddings, which can take minutes on large
Rust workspaces). If the user hasn't asked for indexing and no graph exists,
prompt them before running `descry index`.

Graph freshness is tracked by file mtime. After the user edits code, a single
`descry ensure` (or `descry index` for an authoritative rebuild) brings
everything current; cached queriers, semantic searchers, and dedup caches all
invalidate by mtime.

## Commands — grouped by intent

### Search and discover

| Goal | Command |
|------|---------|
| Find a symbol by name | `descry search <terms>` |
| Find by meaning, not exact name | `descry semantic <natural language query>` |
| Get everything about a symbol in one shot | `descry quick <name>` |

`descry quick` combines search + context into one call — the fastest way to
go from "I know the name" to "I see the source, callers, callees, and tests".

### Navigate the call graph

| Goal | Command |
|------|---------|
| Who calls this function? | `descry callers <name>` |
| What does this call? | `descry callees <name>` |
| Trace forward from a function | `descry flow <name>` |
| Trace backward from a function | `descry flow <name> --direction backward` |
| Find the shortest call path between two symbols | `descry path <start> <end>` |

`flow` shows the full tree (good for exploring); `path` is focused (good when
you already know both endpoints and want the connection).

### Understand a symbol or file

| Goal | Command |
|------|---------|
| Full context for one symbol (source, callers, callees, tests) | `descry context <node_id>` |
| Structure of a file (imports, classes, functions) | `descry structure <filename>` |
| Effective API of a class, inherited included | `descry flatten <class_node_id>` |
| All implementations of a trait/interface method | `descry impls <method> [--trait-name X]` |

Node IDs look like `FILE:src/auth.rs::validate_token`. `descry search` prints
them; `descry quick` accepts a bare name. Pass `--full` to `context` /
`quick` to skip any token-budget truncation.

### Change impact and history

| Goal | Command |
|------|---------|
| Hotspots (most-changed symbols) | `descry churn [--time-range "30 days"]` |
| How one symbol has changed over time | `descry evolution <name>` |
| Impact of a commit range | `descry changes --commit-range HEAD~5..HEAD` |

These require git history; they're no-ops in repos without a `.git`.

### Cross-language (frontend → backend)

| Goal | Command |
|------|---------|
| Trace frontend API calls to backend handlers | `descry cross-lang --mode endpoint --method POST --path /api/v1/auth/login` |
| List all endpoints | `descry cross-lang --mode list` |
| Summary stats | `descry cross-lang --mode stats` |

Requires an OpenAPI spec. Configure via `.descry.toml`:

```toml
[cross_lang]
openapi_path = "public/api/openapi.json"   # or your export location
```

### Configuration

The project's `.descry.toml` controls excluded directories, SCIP toolchain,
query limits, timeouts, and the cross-language section. If it's absent, the
defaults work fine on most projects.

## Recipes

### "Who uses this function?"

```bash
descry callers validate_token
```

If the user asks the question as a behavior ("who verifies tokens"), start
with semantic search:

```bash
descry semantic "verify jwt token"
```

then run `callers` on the top hit.

### "How does the login flow work?"

```bash
descry quick login_handler        # see it + its callees + tests
descry flow login_handler         # full forward tree, inlined source
```

If the user names two specific endpoints ("how does the auth route reach the
database"), `path` is more focused than `flow`:

```bash
descry path auth_handler query_users
```

### "What's the blast radius of this change?"

```bash
descry changes --commit-range HEAD~3..HEAD --show-callers
```

For a standing PR:

```bash
descry changes --commit-range main...HEAD
```

### "Where's the hot code?"

```bash
descry churn --time-range "30 days" --mode symbols
descry churn --mode files          # file-level aggregation
descry churn --mode co-change      # pairs that change together
```

### "I need to understand this unfamiliar repo"

```bash
descry health                      # what languages, SCIP status, graph size
descry structure src/main.rs       # top-level layout of the entry file
descry semantic "http server setup"
descry semantic "database connection"
```

Follow each semantic hit with `descry quick <name>` for full context.

## MCP variant

When the session has the descry MCP server available, the tools are named
`descry_<command>` and return the same content — the only difference is that
outputs are wrapped in a nonce-tagged `<descry:repo_content>` envelope so the
model can tell indexed repo content apart from LLM-generated text. Prefer the
MCP tools in MCP-enabled sessions; they're faster (no shell round-trip) and
safer to feed into downstream reasoning.

The 19 tools map 1:1 to the CLI subcommands. See
`references/mcp-tools.md` for the full list and argument conventions.

## Outputs and further reading

- `references/cli-reference.md` — full CLI flag reference for every command.
- `references/mcp-tools.md` — MCP tool list with argument notes.
- `references/recipes.md` — extended query patterns for common questions.

If an answer is going to be large, prefer `--brief` (on `context` / `quick`)
to keep the output tight, then follow up with `--full` only on the specific
symbol the user actually wants to read.
