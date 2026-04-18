# Descry Claude Code Skill

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)
that teaches Claude when and how to reach for
[descry](https://pypi.org/project/descry-codegraph/) instead of raw
`grep` / `read` when navigating a codebase.

## Why

Without a skill, Claude defaults to Grep+Read for code navigation. That's
fine for small repos and file-local questions, but burns round-trips on
structural questions like "who calls this?" or "how does A reach B?" —
questions descry answers in a single call with call-graph-aware ranking
and inlined source.

This skill tells Claude:

- **When** descry beats grep/read (structural questions, polyglot repos,
  3+ file exploration).
- **When** it's the wrong tool (single-file reads, narrow text search).
- **Which command** to reach for per intent (search / callers / callees /
  flow / path / context / churn / changes / impls / cross-lang).
- **How** to pair descry with Read/Edit in a sane loop.

It bumps descry's trigger probability on the specific phrases users
actually use ("who calls", "impact", "hotspots", "where is this used")
without forcing it on simple file-read tasks.

## Prerequisites

- Python 3.11+.
- `descry-codegraph` installed: `pip install descry-codegraph[all]`.
- Optional but recommended: the MCP server registered (see below).
- Optional: `rust-analyzer` (Rust SCIP), `scip-typescript` (TS SCIP),
  `scip-python` (Python SCIP) for type-aware call resolution.

## Install

### As a personal skill

```bash
cp -r ~/Documents/descry/claude-skill/descry ~/.claude/skills/
```

Claude Code will pick it up on the next session start. Verify with:

```
/skills
```

`descry` should appear in the list.

### As a project-local skill

```bash
mkdir -p <your-project>/.claude/skills
cp -r ~/Documents/descry/claude-skill/descry <your-project>/.claude/skills/
```

Project skills only apply in that project; personal skills apply
everywhere.

## Recommended: also register the MCP server

The skill works with either the CLI or the MCP server, but the MCP path
is faster (no subprocess per call) and its outputs are fenced for
prompt-injection safety. Add to `.claude/settings.json`:

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

Or add to your global config for all projects.

## What's in the skill

```
claude-skill/descry/
├── SKILL.md                      # triggers + decision heuristics + top commands
└── references/
    ├── cli-reference.md          # every CLI subcommand with flags
    ├── mcp-tools.md              # the 19 MCP tools, argument notes, fencing
    └── recipes.md                # extended workflows for common requests
```

Claude loads `SKILL.md` when the triggering description matches, and
pulls references in on demand.

## Verify the skill is active

Start a Claude Code session in a repo with a descry graph
(`.descry_cache/codebase_graph.json` present), then ask something
structural like:

> Who calls `validate_token` in this codebase?

Claude should reach for `descry callers` rather than running multiple
`Grep` calls. If it doesn't, run `/skills` to confirm the skill is loaded,
and check that `SKILL.md` is at
`~/.claude/skills/descry/SKILL.md` (or `.claude/skills/descry/SKILL.md`
in the current project).

## License

MIT — same as descry.
