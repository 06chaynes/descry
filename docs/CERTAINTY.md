# What an answer is certain of

descry reads source without compiling it. Every call it records ends in
one of three states, and every answer says which, so a caller can tell
what the source decides from what it leaves open.

## A call's three states

- **Resolved.** The source decides the one definition the call reaches:
  by its imports, the declared or returned types of its receiver and
  arguments, lexical scope, the class hierarchy, the language's
  visibility rules, and, where an indexer ran, SCIP.
- **Unresolved, with candidates.** Several definitions fit and nothing in
  the source chooses between them, or the receiver's type is not known.
  The definitions that fit are kept as `candidates`; descry does not pick
  one.
- **External.** The call leaves the project, into the standard library or
  a dependency. It is not an edge of the graph.

Three rules hold for every language: a call is never resolved by its name
alone when more than one definition fits, never by file path, and a
receiver whose type is not known is never guessed. A wrong edge is worse
than an unresolved one, so where the language's rule is not modelled the
call stays unresolved. How often each state is reached, and how resolved
calls compare with SCIP's, is in the README's
[Language Support](../README.md#language-support).

## In an answer

- Each entry of `callers` and `callees` has `kind`: `resolved`,
  `candidate` (the call may reach this definition; `candidates` says
  among how many) or `unresolved` (a call whose target is not known).
- Counts are split the same way: `certain`, `possible`, `unresolved`. A
  definition with ten certain callers and three possible ones has ten
  callers the source proves and three it might have.
- `flow` and `context` mark each step with its `kind` too, and say what a
  cut left out: `more` and `more_possible` on a node, `omitted` and
  `truncated` on a context.
- `path` looks for a path through certain calls first and, failing one,
  through unresolved calls to their candidates; each hop has its `kind`.
  Its `status` is `found`, `not_found` within `max_depth`, or `timeout`,
  which says nothing about whether a path exists.

## Names

A symbol is a node ID (`FILE:app/io.py::Reader::parse`) or a name. A name
that several definitions share is refused, with every node ID to choose
from (`reason: ambiguous`), by every tool but `search` and `impls`, which
look names up. descry never answers for one of them in the name of all.
A name no definition has is not found, and the refusal names close
matches when the lookup finds them; `callers` of such a name lists the
calls written with it that the graph keeps unresolved, which no definition
answers; a call dropped as one out of the project is not among them.

## Keeping up with the source

An index is a snapshot. `status` says how old it is; `ensure` builds it
again when it is older than `[project] max_stale_hours`. An index after a
change parses only the files that changed, and resolves again only the
calls of files whose resolution read something that changed, so its
answers are those a full index would give.
