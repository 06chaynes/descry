# Upgrading from 0.3 to 1.0

What changes for a user of descry 0.3: the interfaces, the answers and
the index.

## MCP tools
- `descry_callers`, `descry_callees`, `descry_quick`, `descry_evolution`:
  `name` → `symbol`.
- `descry_context`: `node_id` → `symbol`.
- `descry_flatten`: `class_node_id` → `symbol`.
- `symbol` takes a node ID or a name in each. `flow` and `path` keep
  `start` and `end`.

## Web API
- `/api/callers`, `/api/callees`, `/api/quick`, `/api/evolution`:
  `?name=` → `?symbol=`; `/api/context`: `?node_id=` → `?symbol=`;
  `/api/flatten`: `?class_node_id=` → `?symbol=`.
- Each route answers its tool as CONTRACT.md writes it, the same answer
  as MCP's structured content and `--json`, with only `markdown`, `tree`,
  `summaries` and the result cards' `signature` and `docstring` besides.
  So: a failure is `{error, reason}` (and `choices`), under the status
  CONTRACT.md gives its reason; `/api/health` gives `cache_dir`, `warm`
  and `max_depth`, its `graph` being `/api/status`; `/api/ensure` and
  `/api/index` give the indexer's report as `markdown` (not `result`);
  `/api/structure` answers an unknown file with `found: false` (not 404);
  `/api/flow` gives `asked` and `target` and the resolved `start`, each
  tree node's `type` the node type ("Function", was "Fun");
  `/api/cross-lang` gives `mode` and `endpoint` (null for no handler, not
  404) with endpoints as `operation_id`, the spec's summaries under
  `summaries`; `/api/context`, `/api/flatten` and `/api/evolution` no
  longer echo `symbol` beyond what the answer holds.
- A query parameter that is not of its type (`?limit=ten`) is a 400
  `invalid`, where it used to fall back to the default.
- `/api/context` takes `deduplicate`, `callee_budget`, `head_lines` and
  `max_output_tokens`.

## CLI
- The positional argument of `callers`, `callees`, `context`, `flatten`,
  `quick` and `evolution` is named `symbol` (usage text only; invocations
  are unchanged).
- `health` takes `--json`, as every other tool does.

## pi extension
- Tools take `symbol`; `descry-cli.py` takes `--symbol` for every
  symbol-taking command (was `--node-id`, `--name`).

## A name several definitions share
- `callers`, `callees`, `context`, `quick`, `flatten`, `flow` (`start`),
  `path` (`start`, `end`), `evolution` and `structure` (a file name
  several files share) refuse it, listing every node ID to choose from.
  They used to pick one (`callees`, `quick`, `flow`, `evolution`,
  `structure`) or answer for all together (`callers`, `path`).
- Those tools refuse a name no definition has as `not_found`, naming in
  the text close matches when the lookup finds them; they used to answer
  for the one such definition without saying so (`callers` and `callees`,
  whose `fuzzy` field is removed).
- Web: 400 with `{"error", "reason", "choices"}`; a symbol that names
  nothing is 404 with `{"error", "reason"}`.
- `callees`' `others` and `quick`'s `other_matches` are removed.
- `evolution --crate` narrows a shared name to one where it can.
- `search` and `impls` still take names that many definitions share.

## C# overload IDs
- The second and later overloads of a C# method are `Type::Name#2`,
  `Type::Name#3`, ...; they were `Type::overload2_Name`. A node ID held
  from a 0.3 graph names nothing in a 1.0 one.

## The same arguments everywhere
- `search --type` is lower case in every interface: the CLI no longer
  lowercases it (`Function` is refused as MCP and the web refuse it).
- MCP `descry_search` takes the language aliases (`ts`, `py`, ...) the CLI
  and the web take; `descry_cross_lang` takes `method` in any case, as the
  others do, and every interface refuses one that is no HTTP method.
- An empty search, query, file or method name is `invalid` in every
  interface; a `time_range` git reads no date in is `invalid`.
- `descry embedding-models` takes `--json`.

## Structured answers (callers, callees, path, impls, structure, search, semantic, quick, churn, changes, evolution, flatten, cross_lang, flow, context: every tool)
- MCP `descry_callers` and `descry_callees` return structured content
  beside the fenced markdown: `symbol`, `total`, `certain`,
  `possible`, `unresolved`, `more` ({certain, possible, unresolved}) and
  the list, each entry `{id, name, file, line, kind, candidates?}` with
  `kind` one of `resolved`, `candidate`, `unresolved`. No source text.
- Web `/api/callers` and `/api/callees` return the same object. Their
  entries lose `location`, `file_path`, `lineno`, `type` and `signature`
  (use `file` and `line`); `candidates` is set only for `candidate`.
- MCP `descry_path` returns structured content beside the fenced markdown:
  `status`, `start` and `end` (node IDs), `direction`, `max_depth` and
  `hops`, each `{caller, callee, file, line}` with `caller` and `callee`
  entries as above (`callee.kind` is `candidate` on a hop through an
  unresolved call). The code at each call is only in the markdown.
- Web `/api/path` returns that object plus `markdown`. `hops` is the list
  (was a count; `path` is gone), `start`/`end` are node IDs, and each hop
  loses `step`, `file_path` (now `file`), `snippet` and `candidates` (now
  on `callee`).
- MCP `descry_impls` returns structured content beside the fenced
  markdown: `method`, `trait` (the filter), `total` and `implementations`,
  each `{id, type, trait, file, line}`. Signatures are only in the markdown.
- Web `/api/impls` returns that object: `trait_filter` → `trait`; each
  entry's `struct_name` → `type`, `trait_name` → `trait`, `file_path` →
  `file`, `lineno` → `line`, and `signature` is gone. `trait` is null where
  it was "unknown".
- MCP `descry_structure` returns structured content beside the fenced
  markdown: `found`, `file` (node ID), `imports` and `definitions`, a map
  from `constant`, `class`, `function` and `configuration` to entries
  `{id, name, line}` in line order. Signatures are only in the markdown.
- Web `/api/structure` returns `file`, `imports` and `definitions` as
  above: each entry's `lineno` → `line`, and `signature` is gone.
- MCP `descry_search`, `descry_semantic` and `descry_quick` return
  structured content beside the fenced markdown. A definition there is
  `{id, type, name, file, line, tokens, callers}` (`callers`: how
  many are certain). `search`: `method`, `query`, `total`, `results`;
  `semantic`: `query`, `total`, `results` (each with `score`); `quick`:
  `symbol`. No signatures or docstrings.
- Web `/api/search`, `/api/semantic` and `/api/quick` use those fields:
  `parent_name` is gone (no parser set it; the ID names the owner),
  `file_path` → `file`, `lineno` → `line`,
  `token_count` → `tokens`, `in_degree` → `callers`; `location` is gone.
  Search and semantic results keep `signature` and `docstring` for the
  page's cards; `/api/quick`'s `symbol` has neither.
- `quick` of a symbol whose source file is gone is an error in every
  interface (it was a 404 on the web only; the CLI and MCP printed the
  header and an `ERROR:` line).
- MCP `descry_churn` returns structured content beside the fenced
  markdown: `mode`, `time_range`, `path_filter`, `truncation_note?` and
  one of `symbols` ({id, type, name, file, line, commits, added,
  removed}), `files` ({file, commits}) or `pairs` ({a_id, a_name, a_file,
  b_id, b_name, b_file, shared_commits}, `file_level?`); `message` says
  why there is nothing to list.
- Web `/api/churn` returns that object plus `markdown` at the top level
  (was `{"data": ...}` with `format=structured`, else `{mode, markdown}`);
  `format` is gone. A symbol's `type` is the node type ("Function", was
  "Fun"), null for a file. A file-level pair's `a_id`/`b_id` are null (was
  ""). The no-symbols message no longer carries the truncation note; it is
  in `truncation_note`.
- MCP `descry_changes` returns structured content beside the fenced
  markdown: `range`, `commits`, `files_count`, `symbols_count` and either
  `symbols` ({id, type, name, file, line, added, removed, callers?,
  callers_total?, callers_possible?}, each caller {id, name, file, line,
  kind, candidates?})
  or `changed_files` ({file, added, removed}); `message` says why there
  is nothing to report.
- Web `/api/changes` returns that object plus `markdown` at the top level
  (was `{"data": ...}` with `format=structured`, else `{markdown}`);
  `format` is gone. A symbol's `type` is the node type, null for a file.
- MCP `descry_evolution` returns structured content beside the fenced
  markdown: `node_id`, `mode` ("lines" or "file"), `authors`
  (a count) and `commits`, each {hash, date, added?, removed?}; `message`
  says why there is no history. Authors' names, subjects and diffs are
  only in the markdown.
- Web `/api/evolution` returns that object and `markdown`. Web markdown never carries the graph header line (the page
  shows the graph's state itself); CLI and MCP text does.
- Any key of a service answer starting with `_` is for the text only; MCP
  and the web API leave it out.
- MCP `descry_flatten` returns structured content beside the fenced
  markdown: `class` (ID), `name`, `note`, `chain` ({id, name, external},
  root to leaf) and `methods` ({id, name, file, line}, by name). Web
  `/api/flatten` returns that object and `markdown`.
- MCP `descry_cross_lang` returns structured content beside the fenced
  markdown: `mode` and, for "stats", `total_endpoints` and
  `linked_to_graph`; for "list", `tag`, `total` and `endpoints`; for
  "endpoint", `endpoint` (null when none matches). An endpoint is
  `{method, path, operation_id, tags, node_id}`; the spec's summaries are
  only in the markdown. Web `/api/cross-lang` gives that object plus the
  matched operations' `summaries` (an unmatched endpoint is `endpoint:
  null`, not 404).

## CLI `--json`
- `callers`, `callees`, `search`, `semantic`, `quick`, `impls`,
  `structure`, `flatten`, `path`, `cross-lang`, `churn`, `evolution` and
  `changes` take `--json`: they print the structured answer MCP gives
  (without `ok`), and exit 2 when it is a refusal or a failure.

## A node ID the graph lacks
- `callers` of a `FILE:` node ID the graph does not have is refused as
  not found (it answered "No callers"); a bare name no definition has
  still lists the calls written with it that the graph keeps unresolved.

## Operational tools
- `status` gives `scip` ({state, indexers, cached}) and `embeddings`
  ({state, nodes, model}) as data; web `/api/status` and the `graph` of
  `/api/health`, `/api/ensure` and `/api/index` carry them too.
- MCP `descry_health` and `descry_status` return their structure beside
  the same unfenced text; `descry_ensure` (`action`, `summary`, `graph`)
  and `descry_index` (the graph's status after the run) beside the fenced
  text. The indexer's report is only in the text.
- CLI `status`, `ensure` and `index` take `--json`.

## Flow
- MCP `descry_flow` returns structured content beside the fenced markdown:
  `start` (ID), `direction`, `asked`, `depth`, `target`, `root` (the tree:
  id, name, type, tokens, file, line, candidates, children, more,
  more_possible, seen), and `path_to_target`, `cut` where they apply. The
  code a node inlines is only in the markdown.
- Web `/api/flow?format=tree` keeps the code in the tree for the page; the
  tree also carries `start`, `name`, `direction`, `asked`, `depth` and
  `target`. CLI `flow` takes `--json`.

## Context
- MCP `descry_context` returns structured content beside the fenced
  markdown: `symbol` (a definition entry), `brief`; brief: `callers`
  ({certain, possible, unresolved}) and `callees` ({certain, possible, unresolved});
  else `depth`, `callers` and `callees` (entries with `kind`),
  `callers_more` and `callees_more` (by kind), `examples` ({file, line,
  candidates}), `members` ({id, name, line, tokens}), `tests`, `configs`
  ({id, line, type}), `omitted` and `truncated`; `repeated` when
  `deduplicate` finds it shown before. Source, signatures, docstrings and
  the callee expansion are only in the markdown.
- Web `/api/context` returns that object (`symbol` is the definition; it
  echoed the string asked for) and `markdown`. Web `/api/quick` returns
  the answer and `markdown`. CLI `context` takes `--json`.

## Certainty as data
- Every caller and callee entry carries `kind` ("resolved", "candidate",
  "unresolved"): `callers`, `callees`, `path` hops, `context`, `flow` tree
  nodes and `changes` callers.

## Failures
- Every failure is `{error, reason}` (and `choices` for a shared name), with
  `reason` one of `invalid`, `ambiguous`, `not_found`, `busy`, `no_graph`,
  `unavailable`, `loading`, `failed` (table and web statuses in
  CONTRACT.md). A missing graph is now `no_graph` (it had no reason).
- MCP failures carry that structure as structured content, with
  `is_error` set (they carried none).
- `churn`, `changes` and `evolution`: an empty result says why in
  `message` (it was `error`); an unknown churn mode or a bad commit range
  or path filter is an `invalid` failure, a symbol git cannot find
  `not_found` (web 404), a failed git call `failed` (they were answers
  whose `error` said so). The page reads `message`.
- Successful answers no longer carry `error: null`.
- An `index` path outside the project, and `flatten` of a symbol that is
  no class, are `invalid` (they were `failed` and `not_found`). A
  semantic model that will not load is `unavailable`.
- `status` gives `languages`, `stale` and `error` with no graph too; brief
  `context` counts the callers' `unresolved` as it does the callees';
  `changes` gives each caller's `file` and `line`.

## Contract
- `CONTRACT.md` lists every tool's parameters, answer fields and failure
  reasons; `tests/test_interfaces.py` checks each MCP answer against it.

## Text
- `callers` and `callees` name the symbol by its node ID in their first
  line (`3 caller(s) of 'FILE:a.py::A::parse':`, `'FILE:a.py::main' calls
  3 function(s):`); they named it as asked.
- `churn`, `changes` and `evolution` refuse a malformed time range or
  path filter as `invalid` (web 400); they were failures (web 500) in
  `churn` and `evolution`.

## Path search
- `path` no longer stops after `[query] max_nodes` symbols (that setting
  bounds what `context` and `flow` print). It searches within `max_depth`
  and the time limit; its `status` is `found`, `not_found` or `timeout`
  (`node_budget` is gone).

## Evolution
- `evolution` counts the commits that changed a symbol's own lines, those
  no definition inside it holds, as `churn` does; it read `git log -L`,
  which also counted a nested function's changes and skipped merges. A
  merge that changed the symbol against every parent is listed.
  `truncation_note` says when the time budget cut the history short.

## C# overloads
- A C# call to an overloaded method resolves to the overload whose
  parameters are each exactly of its argument's known type, when that
  overload belongs to the most derived type with an overload of that arity.
  Calls that were candidates between such overloads now have one target.
- Protected methods are a call's candidates only where C# can reach them;
  explicit interface implementations (`void IBag.Add(…)`) never are by
  name; `private protected` methods are reachable from derived types.
- `int?` and a struct's `T?` are other types than `int` and `T`; a
  negative literal has its type; `null` goes to the one overload that
  may take it; `ref`, `out`, `in` and named arguments are matched; an
  extension method's receiver counts as its first argument; an
  unconstrained type parameter stands for an argument's type (never a
  `ref struct`), a non-generic overload of the same types winning.

## Fields a constructor declares
- A TypeScript parameter property (`constructor(private client: Client)`,
  or `= new Client()` without a type) is a field of the class with that
  type, so calls through `this.client` resolve.
- A TypeScript field declared `typeof C` without a value holds the class:
  `new this.f()` constructs a `C`.
- A C# one-line member's calls on a primary constructor parameter
  (`class A(Client client) { void Run() { client.Send(); } }`) are no
  longer dropped.

## Incremental indexing
- An index keeps what parsing each file gave in `<cache>/parse` and parses
  again only the files that changed (or all of them after descry itself
  changes). It keeps each file's resolved calls with the keys of the
  tables their resolution read, and resolves again only the files whose
  parse or read keys changed, so the graph is the one a full build gives.
  Without SCIP, an unchanged aws-lc tree indexes in 7 s instead of 42 s,
  and one changed C file in 13 s.
- `.descry_cache/parse` shipped as a symlink outside the project is
  refused, as `.descry_cache/scip` is.
- Entries are signed with a key descry makes in the user's state
  directory (`~/.local/state/descry/parse.key`); an entry it did not sign
  is ignored. A different `max_parse_seconds` or Python keeps its own
  entries.
