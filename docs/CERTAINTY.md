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

A scope that defines one name more than once (a function under `if` and
again under `else`) has one node for it, with the signature and line of
the first definition. A call of the name reaches that node, the calls each
definition makes are the node's, and in Python it returns what either
definition returns; where the later one always replaces the first, what
that one returns.

In Python a local name is followed through the statements of its function,
and a call on it resolves only where every way to the call gives it one
type; a read in a nested function, a lambda or a generator expression holds
what the name may when that runs. A name bound elsewhere too is not told:
one a function declares `nonlocal` and binds, and one of a module that any
function of its file declares `global`, however the module binds it; a
call on either is unresolved, an `isinstance` test of it tells nothing. A
name a loop binds from its own earlier value (`node = node.parent`) holds,
where the loop's third pass still changes it, a value descry does not
know. A function reads any other name of its module as the module leaves
it bound, so a call made while the module still runs, before a later
statement binds the name again, is read as made on the later binding.

An annotated assignment (`x: Base = make()`) binds the name to an instance
of the class it is annotated with, whatever the value; a later plain
assignment binds it to what that gives. A name a function reads is read in
that function, and what a method assigns to an attribute in that method, but
for its annotations, which are evaluated where the function is defined.
Where the function's body or the class body defining it binds a name a
parameter's annotation writes, or a function enclosing it binds one to a
value, the parameter holds a value descry does not know; where the body
binds a name the return annotation writes, or an enclosing function binds
one to a value, the function returns what its `return` statements give, and
a generator a value descry does not know. An `isinstance` test, an `assert`
of one or a class pattern tells each value the name may hold apart; one of a
class written as a name that the function, or the nearest function enclosing
it that binds the name, binds to a value tells nothing. A value descry
knows, of a class that inherits the one tested for, stays of its own class,
as does a class, function or module of the project tested for a class
outside it. An instance of a class that neither inherits nor is inherited by
the project class tested for may be of either where the class tested for, or
one it inherits, has a metaclass or a base outside the project, whose test
may pass it (a runtime-checkable `Protocol`, an ABC's `register`, a
metaclass's `__instancecheck__`); else it is read as an instance of the
class tested for. A value descry does not know is read as an instance of the
class tested for, so a call on it resolves to that class's method, which a
subclass may override. Such values include a parameter's, an attribute's,
what a call from outside the project or one descry does not follow gives,
what an operator gives on such a value, an element of an unpacked tuple or
of what a loop iterates, what a pattern captures below the top of its case,
the name `except … as` binds, a copy of a name bound elsewhere too, a name a
way to the read leaves unbound, and what a name holds past the loop's third
pass or as a step on from a name of several values. Where the values part,
the call is unresolved. Where a way that passed the test meets one that did
not bind the name, or each way only tested it, the value tested is again the
value it was tested on, so past the test's block the name is read as it was
before the test, told what earlier tests told.

## Keeping up with the source

An index is a snapshot. `status` says how old it is and how many of its
files changed since; `ensure` builds it again when it is older than
`[project] max_stale_hours` or files it holds changed or were removed. An
index after a change parses only the files that changed, and resolves again
only the calls of files whose resolution read something that changed, so its
answers are those a full index would give.
