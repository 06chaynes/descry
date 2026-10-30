# descry — recipes

Concrete workflows for common requests. Each recipe assumes the graph is
fresh (`descry ensure` has been run at least once in the session).

## Recipe: "Who uses this function?"

Direct case — you have the name:

```bash
descry callers validate_token
```

Descriptive case — user names it by behavior:

```bash
descry semantic "verify jwt token" --limit 5
# pick the best match, then:
descry callers <that_name>
```

If the user wants the full callers-of-callers tree, use `flow` backward:

```bash
descry flow validate_token --direction backward --depth 3
```

## Recipe: "How does feature X work?"

Start wide, then narrow:

```bash
descry semantic "<behavior description>" --limit 10
descry quick <top_hit>               # see source + callers + callees
descry flow <top_hit> --depth 3      # full forward tree with inlined callees
```

If the user names two concrete endpoints ("how does the login route reach
the user store"), `path` is more focused than `flow`:

```bash
descry path login_handler user_store_query
```

## Recipe: "What's the blast radius of this change?"

Committed changes:

```bash
descry changes --commit-range HEAD~3..HEAD --show-callers
```

Standing PR against main:

```bash
descry changes --commit-range main...HEAD --show-callers
```

Path-scoped:

```bash
descry changes --commit-range main...HEAD --path-filter src/auth/
```

## Recipe: "Where are the hot spots?"

```bash
descry churn --time-range "30 days"               # symbol-level
descry churn --mode files --time-range "30 days"  # file-level
descry churn --mode co-change                      # coupled pairs
```

Filter by area:

```bash
descry churn --path-filter src/auth/ --time-range "90 days"
```

## Recipe: "How has this function evolved?"

```bash
descry evolution login_handler --time-range "90 days"
descry evolution login_handler --show-diff        # include diffs
```

Useful when debugging regressions — see when behavior changed.

## Recipe: "Where is this trait / interface implemented?"

```bash
descry impls render                          # all implementors of render()
descry impls render --trait-name Component   # only under Component
```

Trait implementations are recorded for Rust: `impls` lists what Rust
`impl` blocks define, not implementations of interfaces in other
languages.

## Recipe: "Explore an unfamiliar repo"

1. `descry health` — what does descry see? which SCIP indexers are active?
2. `descry search main --type Function` — find entry points.
3. `descry structure src/main.rs` (or the entry file) — top-level layout.
4. `descry semantic "<top-level concept>"` — find the relevant subsystems.
5. For each subsystem of interest: `descry quick <symbol>`.

Do *not* jump straight to `descry flow` on an unknown symbol — unbounded
trees on a large graph waste budget. Narrow with search/semantic first.

## Recipe: "Find the call site"

User asks: "where is `parse_config` actually called, and with what args?"

```bash
descry callers parse_config --limit 20
# then for a specific caller:
descry context "FILE:src/cli.rs::main" --expand-callees
```

`--expand-callees` inlines small callee source at each call site, so you see
the actual invocation, not just the caller's file/line.

## Recipe: "Frontend → backend trace" (requires OpenAPI spec)

```bash
descry cross-lang --mode list --tag auth                       # all auth endpoints
descry cross-lang --mode endpoint --method POST --path /api/v1/auth/login
# result includes the backend handler node_id, which you can then:
descry context <symbol>
```

Requires an OpenAPI spec: set `[cross_lang] openapi_path = "..."` in
`.descry.toml`, or keep it at `public/api/latest.json` or
`public/api/openapi.json`.

## Anti-patterns

- **Don't** run `descry flow` with default depth on huge graphs without a
  target — use `--target`, reduce `--depth`, or start with `path`.
- **Don't** use `descry search` for behavior-worded queries; `descry semantic`
  is built for that and will do better.
- **Don't** bypass `status` on a stale graph. Out-of-date graphs
  produce plausible-looking wrong answers. After edits that change
  structure, `descry index` (not `ensure`, which rebuilds only a graph
  older than `max_stale_hours`) brings the graph current.
- **Don't** shell out to grep when the question is structural. Even one
  descry call beats three Greps for "who calls X".

## Pairing with Read / Grep / Edit

Descry is a *locator* — it tells you which file/line is relevant. Once it
names a symbol, use `Read` to see the full source, `Edit` to change it, and
`Grep` for narrow textual follow-ups. The typical loop:

1. `descry search` / `descry semantic` / `descry callers` — locate.
2. `Read` — see the actual code.
3. `Edit` — change it.
4. `descry index` — if the change affects structure (new symbols, renames),
   so subsequent queries see the new layout.
