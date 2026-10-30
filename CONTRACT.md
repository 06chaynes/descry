# Interface contract

What every descry tool takes and answers, the same through the MCP server,
the web API and the CLI. `tests/test_interfaces.py` holds the three
interfaces to it.

## Rules

- **One name per concept.** The symbol a tool asks about is `symbol`, a
  node ID or a name; `start` and `end` name the two ends of `flow` and
  `path`. A limit is `limit`, a depth `depth`.
- **A name several definitions share is refused** with the node IDs to
  choose from, by every tool but `search` and `impls`, which look names
  up; `flow`'s `target` is refused the same way, as `start` is. A name no
  definition has is `not_found`, its text naming close matches when the
  lookup finds them; `callers` of such a name lists the calls written
  with it that the graph keeps unresolved, not those it drops as calls
  out of the project.
- **One answer per tool.** The MCP server returns it as structured content
  beside the text, the web API as JSON, and the CLI with `--json`. The
  text is rendered from the same answer.
- **Structure, not prose.** An answer carries IDs, names, places, counts
  and kinds. Source, signatures, docstrings, commit messages, OpenAPI
  summaries and indexer output are only in the text, which the MCP server
  fences.
- **Certainty is a field.** Every caller and callee entry has `kind`:
  `resolved`, `candidate` (with `candidates`, how many definitions the
  call may reach) or `unresolved`. Counts are split the same way.

## Failures

A failure is `{error, reason}`, and `choices` for a shared name.

| reason | meaning | web status |
|---|---|---|
| `invalid` | an argument is out of range or malformed | 400 |
| `ambiguous` | the name fits several definitions (`choices`) | 400 |
| `not_found` | the symbol or file is not there | 404 |
| `busy` | an index run is in progress | 409 |
| `no_graph` | there is no graph that loads | 503 |
| `unavailable` | the feature's dependency or configuration is missing | 503 |
| `loading` | embeddings are still loading | 503 |
| `failed` | the work itself failed, as git does on a ref or range it cannot read | 500 |

MCP sets `is_error` on a failure; the CLI exits 2; the web answers the
failure as JSON with its status. An answer with nothing to list is not a
failure: git tools say why in `message`. An argument the web cannot read
as its type (`limit=ten`) is `invalid` there too.

## Entries

| entry | fields |
|---|---|
| reference | `id`, `name`, `file`, `line`, `kind`, `candidates`? |
| definition | `id`, `type`, `name`, `file`, `line`, `tokens`, `callers` (certain) |
| counts | `certain`, `possible`, `unresolved` |

## Tools

| tool | parameters | answer |
|---|---|---|
| `callers`, `callees` | `symbol`, `limit`=20 | `symbol`, `total`, counts, `more` (counts), `callers` / `callees` (references) |
| `context` | `symbol`, `brief`, `full`, `expand_callees`, `deduplicate`, `depth`=1, `max_tokens`=2000, `callee_budget`=2000, `head_lines`, `max_output_tokens` | `symbol` (definition), `brief`; brief: `callers`, `callees` (counts); else `depth`, `callers`, `callees` (references), `callers_more`, `callees_more` (counts), `examples` (`file`, `line`, `candidates`), `members` (`id`, `name`, `line`, `tokens`), `tests`, `configs` (`id`, `line`, `type`), `omitted`, `truncated`; with `deduplicate`, a symbol shown before answers `symbol` and `repeated` only |
| `flow` | `start`, `direction`=forward, `depth`=3, `target`, `inline_threshold`=100 | `start`, `direction`, `asked`, `depth`, `target`, `root` (tree: `id`, `name`, `type`, `tokens`, `file`, `line`, `kind`, `candidates`, `children`, `more`, `more_possible`, `seen`), `path_to_target`?, `cut`? |
| `path` | `start`, `end`, `max_depth`=10, `direction`=forward | `status` (`found`, `not_found`, `timeout`), `start`, `end`, `direction`, `max_depth`, `hops` (`caller`, `callee` references, `file`, `line`) |
| `quick` | `symbol`, `full`, `brief` | `symbol` (definition) |
| `search` | `terms`, `compact` (the text only), `limit`=10, `lang`, `crate`, `type`, `exclude_tests` | `method` (`keyword`, `hybrid`), `query`, `total`, `results` (definitions) |
| `semantic` | `query`, `limit`=10 | `query`, `total`, `results` (definitions with `score`) |
| `structure` | `filename` | `found`, `file`, `imports`, `definitions` (by kind: `id`, `name`, `line`) |
| `flatten` | `symbol` | `class`, `name`, `note`, `chain` (`id`, `name`, `external`), `methods` (`id`, `name`, `file`, `line`) |
| `impls` | `method`, `trait_name` | `method`, `trait`, `total`, `implementations` (`id`, `type`, `trait`, `file`, `line`) |
| `cross_lang` | `mode`=endpoint, `method`, `path`, `tag` | `mode`; stats: `total_endpoints`, `linked_to_graph`; list: `tag`, `total`, `endpoints`; endpoint: `endpoint` (`method`, `path`, `operation_id`, `tags`, `node_id`), null where no handler is |
| `churn` | `time_range`, `path_filter`, `limit`=20, `mode`=symbols, `exclude_generated`=true | `mode`, `time_range`, `path_filter`, `truncation_note`?, and `symbols`, `files`, `pairs` (with `file_level`?) or `message` |
| `changes` | `commit_range`, `time_range`, `path_filter`, `show_callers`=true, `limit`=50 | `range`, `commits`, `files_count`, `symbols_count`, and `symbols` (with `callers`) or `changed_files`; or `message` |
| `evolution` | `symbol`, `time_range`, `limit`=10, `show_diff`, `crate` | `node_id`, `mode` (`lines`, `file`), `truncation_note`?, `authors`, `commits` (`hash`, `date`, and in `lines` mode `added`, `removed`); or `message` |
| `status` | — | `exists`, `age_str`, `age_hours`, `nodes`, `edges`, `languages`, `stale`, `error`, `scip` (`state`, `indexers`, `cached`), `embeddings` (`state`, `nodes`, `model`) |
| `health` | — | `status`, `version`, `project_root`, `cache_dir`, `warm`, `max_depth` (the deepest a `flow` or `context` goes), `features`, `graph` |
| `ensure` | `max_age_hours` | `action` (`ready`, `refreshed`, `generated`), `summary`, `graph` (status) |
| `index` | `path`=. (not on the web, which always indexes the project root) | the graph's status after the run |

The MCP tools are `descry_<tool>`; the web routes are `/api/<tool>`
(`/api/cross-lang`), with `ensure` and `index` taking POST; the CLI
commands are `descry <tool>` (`descry cross-lang`). The web API answers
each tool as above and adds, for its page, only: `markdown`, the text
(for `ensure` and `index` the indexer's report); `tree`, the `flow` tree
with the code it inlines, asked for with `format=tree`; `summaries`, the
spec's summary of each operation `cross-lang` names; and `signature` and
`docstring` on each `search` and `semantic` result. Three routes serve the
page and no tool: `/api/source` (a project file's text, within the root),
`/api/examples` (sample queries for the project) and `/api/index/stream`
(`index` as server-sent progress events). The CLI's `embedding-models`
lists the embedding models descry ships pinned configurations for; it is
no tool of the other interfaces.
