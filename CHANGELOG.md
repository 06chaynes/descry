# Changelog

All notable changes to Descry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to pre-1.0 semver (minor-version bumps may include breaking changes; see README versioning stance).

## [Unreleased]

## [0.3.0] — 2026-10-07

Call resolution is rebuilt for every supported language, semantic search
and the SCIP integration are reworked, the MCP server is ported to the MCP
Python SDK v2, and a full-tree review's findings are fixed.

**Upgrading**

- The first `descry index` re-embeds every symbol and regenerates every
  SCIP index. Embedding builds take longer than before, because each
  symbol's source is now embedded.
- The graph schema is v2. An existing graph is rejected with a message to
  run `descry index`; node IDs and node types change in most languages.
- The CALLS resolution percentage is measured differently and is not
  comparable with earlier releases (see *Changed*).
- The `mcp` extra requires `mcp>=2`; the `embeddings` extra requires
  `transformers>=4.51`.
- The CLI exits `2` on every service failure, not only on `ERROR:` text.
- The web API answers `500` for a failed index run, `404` for callees of
  an unknown function and `400` for a negative `limit`; these were `200`.
- `[timeouts] scip_minutes` defaults to 30 (was unlimited; `0` still means
  unlimited).
- Removed: the `ast` extra, `[cross_lang] frontend_api_patterns`, and the
  per-adapter SCIP options of every adapter except Rust.
- Subprocesses inherit the environment unfiltered, and `/api/source` no
  longer refuses files by name (see *Removed*).

### Security

- **Web UI.** Symbol and file names are escaped in the flow tree. Remote
  scripts carry Subresource Integrity hashes and every response carries a
  Content-Security-Policy that allows exactly those asset URLs.
  State-changing requests are refused unless same-origin. `/api/source`
  opens every path component with `O_NOFOLLOW` and caps the read.
- **Index runs.** One runs at a time per service, whichever interface
  asks. The indexer and everything it started are stopped on timeout,
  cancellation, client disconnect, and `SIGINT`/`SIGTERM`/`SIGHUP`.
- **Indexer child process.** Spawned with `python -P`, so a `descry/`
  package inside the indexed repository cannot shadow the installed one,
  and with its stdin on `/dev/null`, so neither it nor an indexer it starts
  reads the terminal or the stream descry was started with.
- **Non-regular files.** A FIFO or a link to a device where descry
  expects a file — a source file, `.descry.toml`, the graph, a SCIP
  manifest, index or checksum file, an embedding sidecar, the OpenAPI
  spec — is left alone rather than read, which would wait or never end;
  a graph that is not a regular file is reported as malformed.
- **Backtracking.** A long run of whitespace, after a keyword such as
  `final`, `public <T>`, `for`, `match` or `import`, made patterns in the
  Dart, Java, TypeScript, Rust, Ruby, C#, Go and C/C++ parsers take time
  cubic or quadratic in its length: 3,200 spaces took a minute on a Dart
  line and half a minute after a Java `public <T>`. `max_parse_seconds`
  (default 120 s) stopped such a file cut short on the main thread of a
  platform with interval timers, and nothing stopped it elsewhere. Each
  run is now taken once. What the patterns match changes only where the
  old one read whitespace as a Java type, a Rust pattern or a TypeScript
  import clause, none of them valid source, and for a TypeScript default
  import named `type` (`import type  from "m"`), now read as `type`.
  Dart type arguments left unclosed, `x<a<b>a<b>…(`, took time
  exponential in their length; they are now read in one pass, as are
  JavaScript lines of thousands of chained assignments or nested labelled
  blocks, which took tens of seconds or exhausted the stack, a name
  assigned thousands of times, whose every read went over all of them,
  thousands of branches in a Python, Ruby or PHP function, each of whose
  merges went over every name, and thousands of nested blocks, whose ends
  were each counted from their start. A run of blank lines in a
  JavaScript, TypeScript, Rust or Dart file took time quadratic in its
  length, each line looking back over the run or a pattern over the file
  starting at each of them: 30,000 took 21 s. So did, in every language
  but C/C++, a name bound again and again or looped over, each binding,
  read and merge of branches going over all its bindings (10,000 took up
  to a minute and a half, 10,000 JavaScript `switch`es assigning one
  over three minutes), Rust's generic functions and field reads,
  TypeScript interfaces and their fields, functions that read a
  parameter, JavaScript's `require`s and object literals, each going over
  those before it or the rest of the file, Ruby's `a <<-b`, and a C# file
  of many types, its braces counted again from its start for each (10,000
  one-line classes took over five minutes). Each is now read once.
- **Writes into the cache.** The cache directory sits in the indexed
  repository by default, which can ship it. The graph and the embedding
  cache are written through new, randomly named temporaries, so a symlink
  at a guessable name no longer has its target overwritten, and one a
  killed write leaves is removed once nothing has written to it for an
  hour; the embedding
  lock is opened without following a symlink and never truncated; a
  symlink at a SCIP index path is removed rather than written through, and
  an indexer that writes `index.scip` in the project does not run through
  one there. Indexing refuses a default cache directory, or a `scip`
  subdirectory, that resolves outside the project, where descry would
  replace another directory's files; `DESCRY_CACHE_DIR` names any other.
- **Git.** Hooks, fsmonitor, signature display, external diff, textconv
  and filter drivers are disabled; partial clones are never lazily
  fetched; diffs are forced to the plain `a/` `b/` format; log output is
  NUL-framed, so commit subjects and file names cannot forge a commit;
  argument validators match the whole value.
- **Source reads.** Paths in the graph are confined to the project root in
  the query engine and the embedder, so a hand-edited graph cannot make a
  tool return files from outside it.
- **Embedding models.** No model is loaded with `trust_remote_code`; the
  option and every way of enabling it are removed.
- **SCIP cache.** The checksum file is written atomically, and Rust and
  TypeScript workspace entries that resolve outside the project are
  dropped.
- **MCP.** `descry_ensure` output is fenced like `descry_index`, and a
  closing fence tag inside repository content is neutralised.
- **Config.** `[syntax.lang_map]`, `[timeouts]`, `[query]` and
  `[cross_lang] api_prefixes` values are validated on load.
- **Release workflow.** A manual publish to PyPI must be dispatched from
  the tag that matches the package version.

### Changed

- **Call resolution is decided from evidence.** Imports (including
  `go.mod`, `Cargo.toml`, `package.json`, `tsconfig.json` and SvelteKit's
  `$lib`), declared and
  returned types, lexical scope, the class hierarchy and each language's
  visibility rules decide each call. Names bound by loops, patterns,
  destructuring and closure or lambda parameters are typed from
  declarations. A call is *resolved* to one definition, left *unresolved*
  when nothing in the source decides, or dropped as *external*. Previously
  a call resolved to the first symbol of that name, a name list discarded
  most call sites before counting, and ties were broken by file path. The
  summary line is now
  `CALLS Resolution: P% of N project calls (...); E external calls not counted`.
- **A block for no type source can name has an owner source cannot name.**
  A Rust `impl` for a tuple, a slice or a type parameter is filed under
  `<Trait for (A, B)>`, and the members of a Java anonymous class or enum
  constant body under `class@<line>`, as PHP, Dart and Ruby already did; they were filed under
  the trait's name, the parameter's or the enclosing class. A blanket
  `impl` gives its methods to a type only where the type can meet its
  bounds, and a Rust trait written with a path is looked up by that path.
  So is the object a JavaScript or TypeScript function returns, under
  `object@<line>`: a function written in it is a member of the object, not
  of the function, and two functions of one name are two nodes.
- **An accessor is filed as source writes it.** A JavaScript or TypeScript
  `get size()` is `Type::get size` and a Dart setter `Type::set size`,
  where they were `get_size` and `set_size` and shared a node with a method
  of that name. A Dart member renamed beside a constructor of its name is
  `field name`, `get name` or `method name`.
- **The object a JavaScript or TypeScript function returns is followed.**
  `make().load()`, `const h = make(); h.load()` and `const { load } =
  make(); load()` reach the function that `make` returns under that name,
  whatever return type it declares, also where the pattern's call goes on
  over lines. A `const` at a module's margin is in scope in the module's
  functions unless a parameter or a local hides it. A constant declared
  with a type is read as what its value's call returns, and as the type
  for the members the object names none of. A helper that hands back a
  call's result returns that object as well. A property of the object
  holds what the value written for it holds: a name a local was bound to a
  call's result, or the call itself. A function written there, whether a
  method, an arrow or a function expression, is a member of the object,
  which no source can name. A name written in one of them is read in the
  returning function's scope; the object's other members are reached
  through `this`. A function the factory declares and one the object
  writes under the same name are two nodes.
- **A receiver of unknown type is never resolved by name.** In every
  language a member call whose receiver the source does not type is left
  unresolved with the definitions that fit as its candidates; it was
  resolved to the one definition of the member's name — and a name the
  standard library also has was left unresolved while a name only the
  project defined was resolved. C#, PHP and Ruby already behaved this
  way. A test's mock of the same name no longer captures such a call.
- **A base type is found as a call's target is.** In every language an
  `INHERITS` edge reaches the type the base's name stands for where the
  subtype is declared — through its import, its qualifier, its namespace
  or its library. It went to the first definition of that name, in the
  same file or in any, a property or a constant among them. A base
  outside the project, and one that several types fit with nothing to
  tell which, stay unresolved, and no type inherits from itself. A C++
  template's type parameter names no type of the project, whether
  written alone, as a pack (`Ts...`), with template arguments
  (`T<int>`), before a member type (`typename T::type`) or a member
  template (`T::template rebind<U>`), as the argument of a standard type
  (`std::unique_ptr<T>`) or of a standard template's call
  (`std::make_unique<T>()`), or as a function's return type: a template
  that inherits from one has no member of a class of that name, and a
  member called through one (`T::make()`, `p->run()`, `get().run()`)
  stays unresolved. Nor does a parameter of the function a TypeScript or
  JavaScript class is written in — a method, an accessor, a callback or
  a function without a name, an arrow function another returns, a name a
  destructuring pattern binds, `...rest` among them, one after type
  parameters that hold brackets: `class extends Base` there extends the
  value passed, also where the class closes on the function's last
  line, and a call of such a name calls no function of the project. A
  call of a variable a function declares without a value, takes from a
  pattern, loops over or catches calls the value it holds, as one of a
  variable given a value did, while the variable is in scope: a `var` to
  the end of its function, the binding of a `for` or a `catch` to the end
  of its body however the head and the body are written, any other to the
  end of its block; one declared in a function a line opens and closes is
  in scope on that line alone. A function or class expression's name is
  no variable of the function around it. A `:` in a pattern's default
  (`{ a = c ? x : y }`) is no key.
- **Unresolved calls name their candidates.** A `CALLS` edge that could
  reach several definitions carries them in `metadata.candidates`.
  `callers` and backward `flow` mark such a caller "one of N candidates"
  and `callees` lists the call once. `path` goes through such a call only
  when no path of resolved calls exists, and marks the hop. Forward `flow`
  and `context` follow resolved edges only.
- **Standard libraries are generated data.**
  `src/descry/stdlib/<lang>.json` says which types and members a
  language's standard library has and what they return, for Rust,
  TypeScript and JavaScript, Python, Go, Java, PHP, C#, Ruby, Dart, C and
  C++. A call on a standard value is external, and its result is typed for
  the next call in a chain. The files are written by `tools/stdlib/` from
  each language's own machine-readable description (rustdoc JSON,
  TypeScript's `lib.*.d.ts`, typeshed, Go's `api/*.txt`, `javap`,
  phpstorm-stubs, reflection over the .NET shared runtime, RBS, the Dart
  SDK, clang's AST) and replace the hand-written name lists for those
  languages. A C++ partial or explicit specialization is its template's,
  not a type of its own beside it.
- **C#, PHP, Ruby, Dart, C and C++ parsers are rewritten.** Every
  definition is a node under its real parent with its line range
  (overloads, constructors, properties, accessors, enum members, local
  functions, out-of-class C++ members, Ruby singleton and generated
  methods); a call belongs to the member it is written in; receivers are
  typed from declarations. Ruby calls written without parentheses are
  recorded. Each C++ parameter has its own type where several take
  template arguments, and a name in capitals right before a field's name
  is the field's type, not a macro. A type written from the global scope
  (`::std::unique_ptr<T>`) is the type written without the `::`.
- **Conditional compilation.** In C, C++ and C#, a definition under `#if`
  is chosen for a call only when the call sits under the same condition,
  or when nothing else, in the project or the standard library, could be
  meant. Go build constraints and `_test.go` visibility decide between
  platform variants.
- **Go, Java, Python, Rust, TypeScript.** Go interface methods, struct
  fields, package-level variables and multi-value results; Java
  inheritance edges, enum constants, lambdas and method references; Python
  star re-exports, class attributes, `with`, properties and return types
  read from bodies; Rust `#[derive]`, `?`, patterns and turbofish;
  TypeScript signatures over several lines, overloads, type parameters and
  re-exports.
- **SCIP.** Rust workspace indexes map onto graph nodes (they did not
  before, and every SCIP-labelled edge fell back to name matching).
  Occurrences are matched by line and column. A call that a Rust index
  records as one of a function, of a concrete type's method or of a trait
  the project implements nowhere, outside the project, is external, where
  it was kept as one of the project definitions of its name. The
  definition an index records is the node whose lines hold its line, a
  template, decorator or annotation before the name included, and whose
  name it has, where the node's ID spelled the name otherwise (a C#
  overload, a Ruby singleton method) and the call went to the first
  definition of the name; one recorded at each overload signature is the
  node that holds them all, and of two definitions of a name on one line
  the index's own symbol tells which. Where the index records two symbols
  at the call's column, as of a declaration merged with another, defines
  the symbol at lines of a file no one node holds (alternatives of the
  build) or only in files other than the call's, names a method that
  cannot take what the call passes (an index that did not bind the call; a
  C# extension method takes the value it is called on, named or a
  literal), or records the name on the call's line only elsewhere, the
  call is resolved from source; one recorded at the macro whose
  parentheses hold the call is the call's. Index files are named
  `<adapter>--<project>.scip`. The SCIP cache follows `DESCRY_CACHE_DIR`,
  and `DESCRY_SCIP_TIMEOUT` overrides the configured timeout.
- **Semantic search embeds one text per symbol**: name, signature,
  docstring and up to 1,500 characters of its source, through the model's
  task prompts. The cache key covers everything that shapes the vectors.
- **Embedding models are a registry** (`descry embedding-models`), selected
  by alias, repo id or local path. The default is `embeddinggemma`
  (`google/embeddinggemma-300m`, gated on HuggingFace). With no model
  configured, a gated refusal falls back to `f2llm-v2-0.6b` with a warning;
  each `descry index` tries the default again. A model named explicitly
  never falls back.
- **MCP server targets the MCP Python SDK v2** (`MCPServer`) and reports
  descry's version in its handshake.
- **The web UI delegates every operation to `DescryService`.** `/api/path`
  reports whether a search found a path, found none, timed out or ran out
  of budget. `/api/health`, `/api/status`, `/api/ensure` and
  `/api/examples` follow the service's view of the graph.
- **Files git ignores are not indexed.** In a git repository the indexer
  leaves out what `.gitignore` and git's other exclude files ignore: build
  output, vendored bundles, caches. A tracked file is always indexed.
  `[project] respect_gitignore = false` indexes ignored files as before;
  outside a repository nothing changes.
- **Cross-language tracing links an endpoint by what the code declares.**
  Where several functions have an operation's name, its handler is the one
  whose attributes or decorators write the operation's path, else the one
  outside test code; with more than one left, none is linked. It was the
  last function of that name in the graph. A handler whose name is not the
  operationId is found by the path it declares.
- **CLI and tools.** `ensure` uses `[project] max_stale_hours` unless given
  an age. `index <subdir>` narrows the walk and writes the root's graph.
  `--lang` covers every indexed language and accepts `ts`, `js`, `py`,
  `rs`, `rb`, `cs`. `path --max-depth` is honoured up to 50. An OpenAPI
  spec at `public/api/latest.json` or `public/api/openapi.json` is found
  without configuration by all three interfaces. `descry --help` no longer
  imports the embedding stack.
- **Git history.** Symbol churn reads at most the 100 most recent commits
  per file within a 30-second budget and says when results are partial; a
  file's first commit is counted.
- **ruff is pinned** to `ruff==0.16.4` in the dev extra and in CI.

### Fixed

- **Scopes are read as the language gives them.** A JavaScript or
  TypeScript name is bound by a declaration, `using` among them, a
  parameter list, a `catch` or `for` head or a named function expression:
  a `var` holds in its function, a `let`, `const` or `class` in its block
  and a parameter in its function, each hiding a name outside from the
  start of its span and holding no value before it is written but for a
  function declaration; a field is no variable of the methods, and a
  field's value, as a function's body, sees what is written after it. A
  parenthesis opens parameters after `function`, before `=>` or after a
  member's name where a member starts, not a call's arguments before a
  conditional's `:` or a `case`'s; a name before `=>` is a parameter
  unless it ends a return type. A local of no known value is of no known
  type, whatever the module defines of its name. A `{` that starts a
  statement is a block wherever one may start, after a label's, a `case`'s
  or a `default`'s `:`, a bare `catch`, a class's `static` or a
  `namespace`, and a block that is no branch runs what it holds; a label
  may prefix any statement, and a labelled body of an `if`, an `else` or a
  loop may not run. An assignment is read wherever it is written, in a
  statement or an expression, a `case` label, a template's hole, a
  destructuring pattern, a parenthesized target, `&&=`, `||=` and `??=` or
  a `for (x of …)` head, the first `case` label running where the switch
  does, a later one in its own section where every section before it
  leaves the switch, as its last statement does or an `if`/`else` all of
  whose branches do, and a `do` body at least once, and from the end of
  its value the variable holds what it assigns, a call's or a name's value
  too, each name of a chain `x = y = v` and of an array pattern its own,
  or no type known for a value descry does not read, and `let x;` holds
  nothing that gives a type; past an assignment that may not run, in a
  later case, in a loop that comes round to it, or in a function that
  reads a variable from around it that other code assigns, the variable
  holds one of the values written, and where they differ its value is not
  known, unless it is declared with a type. Two bindings of a name on one
  line hold what each is given, a conditional's value has the type both
  its branches give or construct, a member's assignment types no local,
  and a type parameter's bound holds only in the declaration it is written
  on, in Java, C#, Dart and Rust too. A file with an `import` or an
  `export` outside every block, after a BOM or indentation too, or with
  `import.meta`, a `.mjs`, `.cjs`, `.mts` or `.cts` file, and one that calls
  `require` or assigns `module.exports` or `exports` is a module: what it
  declares is named in it alone, where it was any file's of the name, and
  an interface or object type is no value a call reaches. A specifier
  naming a directory, the project's root among them, is the module the
  directory's package.json names, else its index; a CommonJS module
  exports what Node gives it, `module.exports = { … }` that object's
  properties, `module.exports = function …` the function and `exports.name
  = …` each name, on one line or over several, unless a declaration of
  `exports` or `module` shadows it, `exports` the alias of
  `module.exports` until a `module.exports =` replaces the exports, an
  invoked function's value too, or `exports =` rebinds it, `exports =
  module.exports = v` keeping it, the last write of a name the one that
  counts, `module.name =` no export, a write that may not run, in a
  branch, a function or past a statement's start, leaving what it writes
  unknown, one of a name not written, `exports[key] =` or
  `Object.assign(module.exports, …)`, leaving each member unknown, and one
  by any assignment operator a write, the module's `export`s written to
  `exports` as TypeScript compiles them (`export =` last, none past a
  `module.exports =`, a name both give either), and `require` giving the
  exports object alone, its names read where they are written, one a
  block declares no value; exports are unknown where the value given them
  is not read,
  `module.exports = require(m)` exports what `m` does, two modules that
  export each other whole export nothing so and a module is no callee,
  and `require` and a destructuring pattern of it bind them; a name an
  import binds to a project module that cannot be located is no global,
  and a call on it, or on `this` of a class it is the base of, stays open,
  as does one of a member a class lacks whose base is a value or an
  expression. JavaScript members are where the language puts them: a
  class's `static` members, and one assigned to a class or a function,
  `F.m = …`, are the constructor's, reached through it, a subclass and
  `this` or `super` in a static member, never an instance; a function
  assigned to `F.prototype.name` or listed in `F.prototype = { … }` is a
  method of `F`'s instances, of `this` in `F` and of a class that `extends
  F`, `Object.create(G.prototype)` or `new G()` giving it G's; a member
  assigned a name holds what the name does, one assigned again is each
  assignment, a call of it open with all of them; a function declared
  inside `F` is no property of it, a class that extends a function reaches
  its statics, and a function called where it is written is no function,
  its constant holding what it returns; a call of what a call returned
  reaches the function it returns by name, a `return` in a one-line body
  too. An attribute a method assigns to the instance, `this.name`, or
  `self.name` in Python, is found before a method of its name, what the
  class and its bases assign counting and a subclass's never, as a name a
  Python class body binds last by an assignment is before a base's and its
  own `def`, but for `staticmethod` or `classmethod` of that `def`, and a
  call of it reaches what it holds, a member it reads, bound or not, or a
  definition, for a field the class declares as for one it does not, or
  else stays open, as does a call of a Python variable or parameter of no
  known value. After an `if`, a loop, a `try`, a `case` or `switch` and a
  statement an `if` or `unless` modifies, a Python, Ruby or PHP name holds
  what every way through it that goes on gives, where they agree, a way of
  an `if` not seeing what another assigns, a `break` skipping a loop's
  `else`, from a `match` case too, a Ruby block perhaps not running, PHP's
  `if: … elseif: … endif;` syntax too, and a branch ending on the line of
  the read counting; `||=`, `&&=` and `??=` assign only where the name may
  be null, or may hold an object, a way that leaves it null counting
  whatever the others hold; a Ruby assignment holds from where its value
  ends, which reads the name's earlier value; and `None`, `nil` or `null`
  hold nothing a call is made on; a PHP function and a Ruby `def` read
  none of the file's variables, a PHP closure its own and those it takes,
  by `use`, one by reference holding past it what it held or what the
  closure binds, and past a rebinding what that gives or what the closure
  binds that variable to, an arrow function each, a `@var` tag types its
  variable as a declaration does, a `match`'s `default` is no label of a
  `switch:` around it, and an anonymous class on one line has its
  methods, the rest of its statement past its body being the member's.
  In Go, Rust, C, C++, Dart, Java and C#, a declaration in a block, a
  statement's head or a parameter list is in scope to the end of that
  block or statement, a Java or C# `case` pattern to the end of its switch
  section, and no longer types its name after it. Each declaration is in a
  scope of its own from where it is written, another of its name on the
  line too, and hides its name from where its statement ends, of no type
  known until its value is read, its initializer still reading the outer
  one; one whose value is a name, a field or a borrow holds what that
  holds, and one whose value starts on the next line that value; each name
  of a Go `x, y := a, b` holds the value in its position, one bound to a
  call holds its result for a read later on the line, a C# `using`
  resource included, and a Rust literal's elements give what iterating it
  gives. A `use path::item as name` binds the item under the name, one
  another module's `pub use` re-exports too, and a function declared in a
  function hides it there. Past
  an `if let`, a `for` or a match arm a name its pattern binds
  is what it was before, an arm on the line that opens the `match` binding
  its pattern's names alone, a Rust closure's parameter is in scope to the end
  of the closure's body, a C++ `catch` parameter has its type in its
  handler, and a Go statement ends at a line end only where Go would
  insert a semicolon. A Java `instanceof` variable holds its type only
  where the test is known to have matched, after a negated test whose
  branch exits too, an `else` being the nearest `if`'s without one, also
  after a branch of one statement without braces; a C# `is` or `out`
  variable to the end of its block, lambda or loop, a C# `using` variable
  has its initializer's type and a `foreach` variable what iterating
  gives, and a Go assignment no longer hides a package-level variable.
- **Overloads, mixins and `#if` arms.** A call that several overloads take
  by the number of arguments it passes is left open with them as
  candidates: C# took the first that took as many, and Java and C++ took a
  subtype's method of the name whatever a base declares. The overloads
  that member lookup finds count: in Java those of every supertype, a base
  class's before an interface's; in C# those of every part of a partial
  type, and of no interface for a class's value; in C++ only those of the
  first class that declares the name, defined or not, a field among them,
  ambiguous where two bases find it but as one member of a base they share
  and `using B::f` takes `B`'s, and its bases' where `using` brings them
  in. An overload that a type before it declares with the same parameters,
  the base's type parameters read as the subtype's arguments, does not
  count, nor does a private one called from outside its type, so a call
  only a base's overload takes reaches it. C/C++, Java and C# count a
  call's arguments from its `(` to the `)` that closes it, over lines,
  braced initialisers and template arguments, and record none where it is
  not found. A C/C++ function takes as few arguments as its declarations
  allow, those of the same parameter types however each writes them,
  through an alias, a using-declaration, a qualifier or a namespace a
  macro opens, their defaults adding up by position, and a member function
  the class declares and the project does not define is an overload of its
  name; a member's defaults are no free function's. A C# or Java lambda
  converts only to a delegate type or functional interface whose method
  takes as many parameters as it declares (`dotnet.json` and `java.json`
  gain `functional`, which lists no annotation interface). A C# extension
  method's receiver, named or a literal's, of the literal's own type (`5`
  an `int`, `2.5` or `.5` a `double`, `'c'` a `char`, a suffix's, read
  past a comment or a line break; `@true` an identifier and a literal C#
  rejects none), is its first argument, and one a standard type has only
  from extension methods is looked up from the namespace around the call
  outward, with the `using` directives written at each level, each naming
  the first of its path under the namespaces around it, `global::` the
  path alone and `using static` a type's members, `global using static` in
  every file below its directory, so a project extension can be the one
  called (`dotnet.json` gains `extensions`); a method of the type and an
  extension that both take a call leave it open, but for one that passes
  nothing to a method that takes nothing; a generic method whose type
  argument no argument tells is called with it written; a type nested in
  the enclosing class is meant before one of the namespace, and of a type
  declared in parts, the caller's file's part. Ruby looks in a prepended
  module before the class, in an included one after it and before the
  superclass, the last included first and the first of one `include A, B`
  nearest, and in a module a class extends for its singleton methods; an
  `alias_method` inside `class << self` is a singleton method. A Dart
  mixin applied to a class leaves its `on` types to the class's
  superclass. A Dart `is` test, `switch` case or arm and if-case promote
  only where Dart's flow analysis does: a case whose heads, and the
  alternatives of an `||` pattern, test one type, a local or a private
  final field, until an assignment reaches it, a parenthesized target, a
  pattern's variable, a `for` loop's variable, a compound assignment,
  `++`, `--` and a write in an arrow body among them, but not one to a
  name declared inside, never in another branch or case nor past one that
  exits, and not where every assignment reaching it gives a value whose
  type derives from the promoted one, a variable's, a getter's, an
  operator's result or a pattern's part; past a `while (!(x is T))` only
  its condition ends; the operands after an `&&` (an `||` for `is!`); the
  branch of a conditional, `if` or `while` it is true in; past an `if`
  whose other branch exits. A Dart `when` guard promotes in its case's
  body or its switch-expression arm, a standard type's members are read
  from the library's data whether or not the project extends it, a case's
  variables are in scope through it, and a parameter gives way to an `is`
  test on its line. No Dart pattern of a `switch` head or arm, a record or
  multi-line case head, an alternative of `||` or the left of `=` among
  them, is a constructor call, a guard's call before `=>` is one, `:final
  x` binds `x`, a field pattern's variable has the type of the getter it
  matches, `for (var x in xs)` what iterating `xs` gives, a standard
  generic type keeps its type arguments, an extension on a standard type
  applies to types derived from it and one on a project type to no library
  type of its name, and a function expression's parameters written over
  lines are its own. A Go variable is of the type it is declared with
  whatever is assigned to it later, an entry of a `var (...)` block
  declares, `x := (B{})` is a `B`, a literal that goes on to a method call
  types nothing, and a method called on a composite literal, `T{}.M()`, is
  `T`'s. A C/C++ definition an `#if` arm the call is not under alone
  builds is a candidate beside one every build has, where the unguarded
  one was taken; one defined in several arms is built under what they
  share, where it had the first arm's; `#ifndef X` that opens with
  `#define X` anywhere in a file, as an amalgamation holds headers, is no
  arm. A C++ specialization is a type of its own, named with its
  arguments, `template <> void Box<int>::put()` a member of the template,
  and an instance of a template is of the most specialized partial or
  explicit specialization its arguments, completed by default template
  arguments, match; where a macro, an alias or a type parameter leaves
  that undecided, the call stays open with each class it may be, unless
  all reach one member through a base. An override whose function-pointer
  parameters are named otherwise is an override. An alias declared in a
  block, class or namespace around a call is found before a definition of
  its name in a namespace that is not, an alias of a standard type gives
  its library value, `->` and subscripts on a raw pointer reach what it
  points at, and a call of an alias whose type is not told stays open. Raw
  strings are blanked over all their lines, and `T (*name)(…)` declares
  `name`. A macro invocation without `;` alone on its lines ends its
  statement, so the declaration after it is read; what a macro stringizes
  is no call; a member type `X<A>::m` is that member, the one the
  library's data writes for an instance or a project template's own, never
  `X<A>`; a name whose template arguments close a list opened before it is
  not called by the `(` after them; default template arguments of every
  declaration of a template are merged; and a quoted include that is no
  file beside the includer is read from a directory of the include path it
  may climb out of. A C++ caller's function has an overload where a header
  both files are compiled with declares one, and the call is the one of
  them that takes it; a conversion operator, `operator fmt::string_view`,
  is no definition of the name after its `::`. A Rust `impl` in a module
  is for the type the module declares, before or after it, or imports with
  `self::`/`super::`; an item after attributes on its line is read; `let b
  = B;` makes `b` an instance of a unit struct `B`; a path from `crate::`,
  `self::` or `super::` is followed through the module tree `mod`
  declarations lay out, `#[path]` included, and the module's `use`
  bindings, where its calls were dropped and its types taken by name; a
  `use` binds only in the module or function it is written in, a module
  that globs `use super::*` seeing its parent's; and a block a line opens
  and splits the statements of ends at its `}`, where what followed a
  one-line function was filed under it.
- **Parsers.** Rust `pub(crate)`/`pub(super)` items, `const fn`,
  `extern fn`, one-line blocks, calls in a body on the `fn` line, `impl`
  edges, functions after an `impl` inside a module, methods of an
  `impl` whose brace follows a `where` clause, and `impl` headers with
  references, lifetimes, `unsafe` or wrapped lines, which were filed under
  the trait; Python definitions under
  `if`/`try`/`with`/`for` and calls counted twice in nested functions;
  Java Allman braces, class scope ended by a `'}'` literal, calls
  attributed to the class instead of the method, members of an anonymous
  class filed under the enclosing one; Go unexported types,
  generic receivers, one-line functions, generic types and functions with
  brackets in their type parameters, which had no node, and a method of
  a type parameter's value, dropped as external instead of read through
  its constraint, a generic struct's fields, read as an array's elements
  from its type parameters, and `type A [5]T`, whose `[5]` was read as
  type parameters; Ruby one-line classes and modules,
  `#` docstrings; C# properties and verbatim strings; TypeScript and
  JavaScript `.jsx`/`.mjs`/`.cjs`/`.mts`/`.cts` files, `$` identifiers,
  typed arrow constants, file-level calls, a definition after the `}` or
  `;` that ends another on its line, which had no node, and a call on the
  line a scope opens or closes on, filed by the line instead of the
  column, a class written as a value (returned, an arrow's body, a
  property's value, a default) or as `export default class` or
  `declare class`, which had no node or left its methods to the
  function around it, a JSX `class` attribute that starts a line, read
  as a class, a name a renamed import binds (`import { A as B }`,
  `const { A: B } = require(...)`), looked up in its module by the new
  name, so a call through it was dropped or reached a definition of that
  name, a base written as an expression, `extends (A as any)`,
  recorded by its first word, the functions and accessors of an object
  `export default` gives, of a property's or an element's object, and
  of one written on its constant's line, which had no owner, a function
  after `?`, `||`, `&&`, `!` or `void`, read as none, and a class whose
  body opens on a later line with members on it, a namespace, `namespace
  A.B` or `declare namespace`, whose members had no owner, and a
  `declare const`, which was no constant, so a namespace that is a value
  too had neither's members, and a template literal that goes on to the
  next line with another inside a `${...}`, which ended the block around
  it. A JavaScript or TypeScript import, and `export default`, one past
  another statement on its line too and none in a comment or a literal,
  its words parted by any of JavaScript's white space and line breaks, a
  BOM and a no-break space too, around `as` as well, and by nothing
  JavaScript does not read as white space, a TypeScript file's by
  TypeScript's, which takes U+0085 and U+200B too, a line break after
  `import` ending no statement, its name the value where the statement
  ends after it, as JavaScript
  ends one at a line break, binds what its module exports under its kind:
  a default import the module's default (`export default f`, a
  declaration or an object after it), whatever it is named, where
  it was looked up by its own name and its calls were dropped; a named
  import the export of its name, whose members were read from the module
  itself, so `a.run()` reached the module's `run`; a namespace import or
  a `require`, `import x = require(...)` too, which was read as a call of
  `require`, the module whole; a module that assigns its exports,
  `export =` or `module.exports =`, exports that value's members. An
  object holds what `{ f }`, `{ key: f }` and `{ ...o }` name, and a
  constant whose value names another constant, as declared there and not
  hidden by a nearer binding, holds that object; a call on a
  receiver of no known value whose member only objects define is left
  unresolved with them, where it was dropped as external; a built-in
  module `@types/node` declares, `fs` or `node:fs`, is outside the project,
  ahead of a project package of its name, as Node resolves it, where a bare
  one was read as not known; a module
  is located in a declaration file where no other is at its path, a
  package's conditional export is the first that applies to a `require`
  or an import, not to either, and `./index.d.cts` is that file, not
  `index.d.ts`. What is declared `typeof x`, a variable, a parameter or
  a field, holds `x`. A function written on the line of a `return {`
  or of a declared arrow's `=> ({`, which had no node, its calls the
  returning function's, is the object's member as on a later line, and
  a property of a returned object given a call on a later line holds
  what the call returns, where it held nothing. A property of a function
  type in an interface, `run: (c: Dog) => void`, which was no member, so
  `b.run()` reached nothing, is a method the interface declares, and an
  object type, `type T = {...}`, which had no node, owns its members as
  an interface does. A function written as a property of an object
  literal passed to a call, `use({ run: (c) => c.bark() })`, on the
  call's line or a later one, takes the parameters that property's type
  declares in the call's parameter; `items[i]` is an element of the
  array `items` is, and a rest parameter, declared or gathering those of
  a function type, is an array of its type, where each was untyped.
  In C++, `items[i]` is what the container's `operator[]` returns and a
  C array's element is of its declared type, a range-based `for` over a
  parameter or a field gives the element, `T const x` and `NS ::Name`
  declare their types, and a lambda's parameters have theirs in its body;
  in Rust, a closure takes the parameters of the `Fn` bound the callee
  declares, `impl Fn(A)`, `F: Fn(A)` or in a `where` clause, `a < b,
  c > (d)` is two comparisons where it was read as type arguments, and
  the `Fn(...)` of a signature is no call. A Python module path more
  than one directory gives, `pkg.models` below `src/` and `lib/`, is the
  one below the importing file's own, not the first found, and one below
  two others is unresolved. A TypeScript generic type, a class, an
  interface or an object type, holds its type arguments: `Box<Dog>`'s
  `T` is a dog through a field, a method's return, `this` and a subclass
  that writes its base's arguments, and inside the type a parameter is its
  bound, where each was untyped; overloads that return one type with
  other arguments return that type, where they returned none. So does a
  Java generic type, whose type parameters were each read as the bound
  any class of the file gave one of their name, so `Box<Cat>`'s `T` was
  a `Dog` where `Bounded<T extends Dog>` was written beside it; a base is
  kept with its type arguments. So does a C# generic type, whose members
  of a type parameter's type had none, and a class's `where T : Dog`, read
  as a base of the class, bounds `T` inside it. So does a Dart generic
  type, whose members of a type parameter's type had none, and one its
  constructor is called with, `Box<Dog>(...)`; a method called through a
  field declared a generic type, `Box<Cat> box`, reads the field's
  arguments, where it read none. So does a C++ class template, whose
  members of a parameter's type had none, read through a field, a
  pointer, a local, a construction and an alias; a variable of an alias
  template's type is read as a field of it is, and a project template
  that names several specializations wraps nothing. `->` on a raw pointer
  to a smart pointer reached what the smart pointer holds: `p->reset()`
  on a `std::unique_ptr<Cat>*` called `Cat::reset`. A call of a member a
  C++ type declares without defining it has the definitions of the types
  derived from it as candidates. So does a Rust generic struct or enum,
  through a field, a method's return, `Self`, a field declared
  `Boxed<Cat>` and a turbofish path, `Boxed::<Dog>::new` or
  `Vec::<Dog>::new`, whose type arguments were read off the callee, and
  an `impl` block's bounds hold for its own items. A call through a field
  of a variable, `b.held.meow()`, is read from the variable's type in
  every language, where it was read by name; a Rust field read is the
  field where a method shares its name; and a `for`, `if let`, `while
  let` or `match` whose body goes on on its line binds its names there.
  A TypeScript alias whose name holds a `$` was read as a type from
  outside. A type parameter an argument is passed for is that argument's
  type, its bound only limiting it, and an instance's argument stands
  for its parameter where it is not known: the bound was taken for
  either, and members of the type passed were missing from it. So does a
  Go generic type, `Box[Cat]`, through a field, a method's return, a
  receiver that names its parameters as it likes, a composite literal and
  a generic function called with its type arguments, `NewBox[Dog](...)`,
  which was no call at all; a constraint bounds a parameter inside its
  type. A variable bound to a call chain with an index or type arguments
  in it, `x := a[i].b()`, holds what its last call returns. A TypeScript
  arrow a union of function types is declared as took the first
  alternative's parameter types, `x` of `((x: Dog) => void) | ((x: Cat) =>
  void)` being a `Dog`, and one a generic function type is declared as
  took the type parameter's name for a type from outside; their
  parameters are of no known type. One an intersection is declared as
  takes its one function type's, and its body on its line is its own. A
  name a pattern binds with a default, or a `catch` pattern, is no global;
  a function another returns, `() => () => {`, is a scope of its own; and
  a parameter written after a `(` on an unindented line was read as a name
  of the whole file. It is in scope in the definition and the functions it
  holds, also where the definition is on one line, whose parameters were
  read in the definitions after it, `g = (x: B) => ...` beside `f = (x: A)
  => ...` taking `x` as an `A`, and after the definition's end on its
  line, `function j() { const k = (w: A) => w.use(); w.use(); }` or a
  later declarator; a one-line arrow's calls after the first `;` of its
  body were the file's. A destructured parameter, `({ a }: P)`, takes its
  parts from its type, where a name `a` the file declared elsewhere was
  read, and a member a type declares, `a: A;`, was a name in scope in
  every function of the file. A parameter of an object type, `(props: {
  scene: Scene })` or a function type an arrow is declared as, has the
  properties it writes; an arrow whose annotation's type arguments go on
  over lines is a definition whose parameters it types; a method written
  in an object argument, `withH({ run(c) { ... } })`, takes its parameters
  from the property the call's parameter type declares, and a callback's
  rest parameter an array of what the callback's type gathers. A first
  declarator, `let a = d, b = d`, is typed by its value as a later one
  is, and `typeof d` by what
  `d` is declared as; a one-line `namespace` body and an object whose `{`
  opens on the next line own their members; a one-line `for (const x of
  y)` binds `x`; an arrow's own parameters read its annotation over lines,
  and its rest parameter an array. `typeof p` of a parameter is what `p`
  is declared as, `new Box;` makes a `Box`, a declarator on a line of its
  own is of the scope its statement starts in, an object a `=` gives after
  a blank line, or with members on its `{` line, owns them, a `for` head
  binds its names wherever its lines break, a callback whose body starts
  on the line after its `=>` binds its parameters there, and iterating a
  `Map` gives its key and value by position, as a parameter of a tuple
  type gives its elements through a pattern or a literal index. An alias's
  object type may open on the line after its `=`, a later declarator's
  annotation types the arrow it holds, and `typeof` follows names to a
  type. A `readonly` member, an intersection of object types, a cast in an
  expression body, a method an object type declares, called, a `readonly`
  or aliased tuple, and an indexed access type, `Type["key"]`, are read as
  written. `new Box().open()` typed the name it was bound to as a `Box`. A
  field a JavaScript class assigns, `this.cb = ...`, is its own, not a
  member of an outside base, and `export default cond && function f`, on
  one line or over two, is `f`. A type parameter whose bound only names
  it, `T extends Base<T>`, is its bound, where it was read as the
  receiver's own type, so `b.get().self()` reached `B::self`. A Dart raw
  string over lines held code on its later lines. In Java, C#, Dart, PHP,
  Rust and Go a field or statement after another on its line, `A a; B b;`,
  was not read, and a Java field behind an annotation and the calls of a
  PHP function on one line were not either. In C++ a spaced qualifier, `ns
  ::T`, of a parameter or field kept the space; a qualified name is looked
  up as written where it names, from the call's scope or a namespace
  around it, a namespace that holds the definition, though a macro opens
  one of that name elsewhere, and finds the type it names in full there,
  where it found any in a scope of its last name; a base list reads `ns
  ::Base` as `ns::Base`; a macro in a qualifier, in a return type, a
  parameter, a local or a base, is read as it expands, a namespace it
  stands for as that and one that stands for nothing or an attribute as
  nothing; of two scopes a qualifier names the innermost is the one; a
  lambda with a return type binds its parameters; `typename std::...`
  declared a type parameter `std`; a macro before `::std::string` was read
  as its type; and `ns::Box<int>()` reached the file's own specialization
  where the template is in a namespace a macro opens; calls in comments,
  string
  literals and JSX text; the character after a JavaScript or TypeScript
  `/* */` comment, which was dropped; a `require` or dynamic `import()`
  with a comment before its module or written over lines; a call on a
  parameter of an arrow constant or of an arrow assigned to a variable,
  dropped as external where it is left unresolved; a parameter of a
  function type that is itself a function, and a `this` parameter,
  typing an arrow's parameter; a `<` or `/` that goes on with a
  comparison or a division from the line before, read as a JSX element
  that hid the rest of the file or as a regular expression; a JSX
  element or regular expression after the head of an `if`, `for`,
  `while` or `with`, a spread's `...`, `default`, or a `break` or
  `continue` and its label, read as a comparison or a division; chained
  calls; declarations recorded as calls; braces and calls inside string
  literals in C#, PHP, Dart and C/C++, and the code a string holds, the
  `{...}` of a C# `$"..."`, `$@"..."` or raw string after `$`, Dart's
  `${...}`, over lines too, and PHP's `{$...}` in a "..." string or a
  heredoc, whose calls were dropped and whose quoted braces could end
  the body around it; in Java, C#, Dart, PHP, Rust and
  Go, the members of a type written on one line with it, which had no
  node, and a definition after the end of another on its line; a licence notice
  inside a file read as the next symbol's docstring; inheritance from a
  project type named like a standard one; calls in a second Svelte
  `<script>` block; calls in a nested Python function's default arguments;
  a `{` or `}` in a JavaScript or TypeScript regular expression, template
  literal or block comment ending or opening a block the definitions
  after it are filed under, and a `//` inside a regular expression cut
  as a comment, taking the calls after it on the line with it. A block
  written on one line whose line goes on to close what encloses it
  (`save() { ... } };`) ends on that line, rather than taking in the
  definitions after it. A call on a TypeScript or JavaScript definition's
  own line is now the definition's — its parameter defaults and the
  first line of its body run when it is called — so a one-line
  function's, method's or arrow's call is no longer filed under the
  enclosing file or class; a decorator or computed key on the line runs
  where the class or object is defined, and an arrow constant written
  with a function-type annotation resolves its body calls through the
  annotation, not through a local of its name the arrow declares; a
  Java interface no longer types a TypeScript arrow's parameters where
  it shares the name of the type the arrow is given, and a union or an
  intersection of function types no longer types them as its first.
  A generic arrow (`<T>(x: T) => ...`) is read as a function,
  each declarator of `const a = ..., b = ...` as a constant of its own
  with its own value, and each of `let` or `var` holding a function as
  that function, after a destructured first one and on the lines the
  statement goes on to, each where the statement is written, not in a
  definition one of its lines closes; a method, accessor, constructor or
  arrow-valued property, typed or not, as a member, on its class's line
  too, and a
  private member (`#m`) as a member that `this.#m()` reaches. A call on a
  later line of a signature or a parameter default, an optional call
  (`f?.()`, `o.m?.()`), and a call in an arrow's expression body on the
  lines after its `=>`, are the function's, where they were not read or
  were filed under the enclosing scope. A member that starts on its
  class's line and goes on over the next ones is a member to the end of
  its statement, and the lines a field's value goes on over declare
  nothing, where a call on one was taken for a method of the class; a
  field's value on the class's line is the class's, as on a later line.
  A member whose type parameters run over lines ends with its statement,
  not inside them, and owns the call its expression body makes; a member
  written on the line of the `}` that closes the member before it is a
  member of the class, not a call taken for one; what a line goes on to
  after the `}` that closes a class, a function, a method or an object —
  a call, or a callback it opens — is the enclosing scope's, not the
  closed definition's; the calls in the body of a class's accessor are
  the accessor's, not the class's, and a local it declares is no member
  of the class; a body-less overload on
  the class's line no longer swallows the member after it; a typed class
  property whose arrow value starts on the next line is a member; a field
  whose parenthesised value runs over lines is no member, its calls the
  class's; and each callback a call is passed, a named function
  expression among them, is a scope of its own, so a local of one
  callback is never filed under a local of the same name in a sibling,
  and `function (` names no member. A local that
  holds a function is the definition it holds when a function is passed
  to it as well. An object a call is passed, as any of its arguments,
  whose lines go on past its first and which holds a function, owns the
  functions written in it — a method, an arrow or an accessor — under
  `object@<line>`, a name no source can write, where they were the
  enclosing scope's; and a constant of a function's body holding an
  object owns the functions written in it, as at the top of a file and in
  a callback, and a member call on its value reaches them, where the
  objects of one function were one member of it. A method or property a
  string literal names is a member or property, filed as the name the
  source names — the literal's content,
  which no plain declaration can write for a `"~validate"` and which
  `{"m"}` and `{m}` share — where its body was read as the class's and
  made a phantom member the real method of that name shared a node with.
  A class expression — a declarator's value, a member's value, or one a
  `new` constructs or a call is passed — is a class of the scope it is
  written in: the `const`, `let` or `var` that declares it names it, its
  own name is visible only inside it, and one no declarator names is
  `class@<line>`, a name no source can write. Its members are its, where
  they were the enclosing scope's, also where its body opens on a later
  line or other code comes before it on its line, and the definition its
  lines are all in holds it; an object a string-literal key gives a
  function to is the property the key names, so a class written in one is
  named by the function that holds it.
- **A class declared inside a function or a callback is named only inside
  it.** For TypeScript and JavaScript a type, a call of that name or a
  member call on it written elsewhere — in a sibling function of the file
  or in another file — is the language's global, and the class and its
  members are not candidates a call there can reach. The class a test's
  callback in another file declares no longer captures the name. Inside,
  it shadows a class of its name written outside, the two share no
  member, and no class written elsewhere extends it.
- **Indexer.** A parser failure or an unusable `sg` binary no longer
  aborts the whole index, nor does resolving a member of a union one of
  whose alternatives is a standard type, a value a dynamic `import()`
  gives over several lines, or a Dart variable whose value is a call of a
  standard type's method. A PHP `switch` that is empty, or whose labels
  are class constants (`case self::A:`), no longer drops its file, nor
  does a Ruby `begin … rescue … else` or a `??=` or `||=` of a name no
  scope binds, and a C# `using` in a namespace beside the call's no longer
  aborts the index; a statement a branch reader cannot read merges
  nothing, with a warning. A line holding a form feed, a vertical tab or
  another character Python's `str.splitlines` ends a line at is one line,
  as an editor and the source view show it, so the definitions and calls
  after it are no longer filed a line or more below where they are
  written. A JavaScript or TypeScript parameter list that names nothing,
  `(:)`, no longer fails its file. A call after a `.` or a `->` that ends
  the line before is a member of what that line ends with, no longer a
  call of the function of its name. `index` no longer encodes every symbol
  twice. `descry-generate` honours `DESCRY_LOG_LEVEL`, walks the project
  root when given `--project-root` alone, and refuses a path outside it.
  scip-java no longer passes the Gradle init-script on Maven or sbt
  projects, where `mvn`/`sbt` rejected it and no index was written; its
  Kotlin-DSL pre-flight likewise runs only for a Gradle build, which is
  now recognised by every marker scip-java's own detector accepts
  (`settings.gradle` and `gradlew` beside the two build files). A Gradle
  multi-project root is indexed as one build without a build file of its
  own: by `settings.gradle`, or by `settings.gradle.kts` beside `gradlew`.
  Where scip-java also detects another build tool at such a root (Maven,
  sbt, Mill, Bazel or a SCIP config), with or without a build file of its
  own, it refuses the root, and the root's subprojects are indexed
  instead. Every JVM project is discovered on the same terms: a directory
  where scip-java detects exactly one build tool, by the regular files its
  own detectors read, so it is not run where it would answer "No build
  tool detected" or "Multiple build tools detected", nor where a
  `project/build.properties` holds a `\u` escape without four hex digits,
  on which it aborts. A Rust root package beside its workspace members, or
  beside nested crates, is indexed too: `rust-analyzer scip` indexes only
  the crate at the path it is given, so the root package had no index; it
  is named `.` where a member holds the root directory's name. scip-dart
  is run on `.`; given `./` it took the package for a nested one, skipped
  it and wrote an empty index. scip-dart is found under either spelling —
  `scip-dart` or the `scip_dart` `dart pub global` installs, read from
  `$PUB_CACHE` when it is set — and its package discovery reaches
  `<collection>/<name>/pubspec.yaml` (`pkgs/`, `apps/`, …), not only
  `packages/`, while a package under an excluded or dot-directory is left
  alone. An index file that parses but carries no symbols is logged, since
  that usually means the indexer's own environment was missing rather than
  a project with no symbols. A root with a build of its own is one project
  to every indexer, where its subdirectories were indexed apart and the
  root's own sources had none: a CMake or Make root, a Maven root that
  lists `<modules>`, a Gradle root with a settings file, a `package.json`
  (and a workspace root whose `src/` holds sources, beside its members),
  `pyproject.toml`, `setup.py` or `setup.cfg`, a `Gemfile`, a
  `composer.json`; a Maven root aggregates where its POM lists modules
  outside a profile. scip-php is probed with `--help`, since anything else
  indexes the directory; scip-dotnet is left to index every `.sln` and
  project file at a root and given every `.slnx` where there are none;
  scip-clang runs only with a compile database that exists, the
  subdirectories' under a root build file that has none. A JavaScript or
  TypeScript package without a `tsconfig.json` is indexed with one written
  beside the index, which reads JavaScript too with the current library:
  `--infer-tsconfig` wrote one into the indexed package.
- **Service and query.** Source is read relative to the project root
  whatever the working directory. A missing, stale or corrupt graph no
  longer stops the MCP server from starting, and is reported by `health`
  and rebuilt by `ensure`. A malformed graph raises a schema error instead
  of a traceback, including for metadata of the wrong type. An empty
  `api_prefixes` entry no longer recurses forever. A graph file removed
  between two looks is reported as missing. A streamed index run always
  ends with a `done` event. `ensure` refuses an age limit that is not a
  finite, non-negative number, and a negative `limit` is an error in every
  interface. Callers come back in a stable order, certain ones before
  candidates, and `callers Type::name` lists the callers of that method
  instead of every `name`; `Type.name` is read the same way, and a
  qualified name no longer matches inside a longer one. `path` no longer follows
  an unresolved call to the first definition of its name, and searches from
  every definition its start name matches; `flatten` follows an unresolved
  base only when one class has the name.
- **Embeddings.** The file lock now excludes; cache cleanup cannot delete
  another process's temporary; concurrent first searches load the model
  once; `min_score` applies after re-ranking; a stale or mismatched cache
  is detected; NaN scores and empty graphs are handled; a gated model
  fails with the licence page and login command. The MCP pre-warm waits
  for the load it starts, and a search that has waited `embedding_seconds`
  for the load it starts is told the embeddings are loading while the
  load goes on to cache them; both reported a timeout while the load went
  on, and a search started a second one. A search during a reindex's load
  is told the embeddings are loading rather than starting another, a load
  begun before a reindex no longer caches the old graph's embeddings as
  the new one's, and a failed load is reported to every search until one
  succeeds.
- **Git history.** Changes over a range reaching the root commit, the
  default range in a one-commit repository, non-ASCII paths, SHA-256
  repositories and shallow-clone detection in worktrees.
- **SCIP.** The single-crate Rust path, a stale index counted as fresh,
  projects of one name in two languages sharing an index, and
  `DESCRY_NO_SCIP` parsed two ways.
- **Web.** `/api/ensure` without a body, unvalidated `direction`, errors
  returned as `200`, IPv6 loopback `Host`, a NUL byte in `?file=`, an
  `Origin` header that is not a URL, text files in other encodings refused
  by `/api/source`, no syntax language for most extensions in the
  source view, and a file whose lines end in a carriage return alone shown
  there as one line.
- **Packaging.** `descry-mcp` and `descry-web` name the extra to install
  when it is missing. `pi-extension` uses one interpreter name and a
  Windows-safe script path. Licence metadata uses the SPDX form.

### Added

- `embeddinggemma-2` (`google/embeddinggemma-2`, text-only) with its
  dependencies in the `embeddinggemma-2` extra, and `f2llm-v2-0.6b`
  (`codefuse-ai/F2LLM-v2-0.6B`).
- `tests/eval/` — a retrieval-evaluation harness scoring models on a
  project's own graph (docstring→symbol queries, leave-one-out leakage
  control, paired bootstrap, per-language breakdown).
- `descry-generate --project-root`.
- `[code_files] max_file_mb` (default 10, `0` for no limit): the size above
  which a file is skipped by the indexer, not read for source snippets and
  refused by `/api/source`.
- `[code_files] max_parse_seconds` (default 120, `0` for no limit): a file
  whose parsing takes longer is left as far as it got and logged, so one
  file cannot keep an index run from finishing. Not enforced on Windows.
- `tools/stdlib/` — the generators of the standard-library data, one per
  language, each reproducible from a pinned source.
- `THIRD-PARTY-NOTICES.md`, shipped in the wheel and the sdist: the source,
  copyright holder and licence of each standard-library data file.
- `descry_status` in the Pi extension.
- `graph.error` in `health`, `invalid_graph` in `status`, and
  `truncation_note` in structured churn.

### Removed

- The `ast` extra and the unwired `tree_sitter_parser` module.
- The ast-grep call extraction and `DESCRY_AST_GREP_MAX_FILES`. With `sg`
  on `PATH`, ast-grep replaced the Rust, TypeScript and JavaScript
  readers' calls with ones that carry no receiver's type and are filed by
  line alone: commander resolved 30% of its calls with it and 86%
  without, ripgrep 57% and 84%. The readers' own calls are used always.
- Subprocess environment filtering (`safe_env`). Git, the SCIP indexers
  and the indexing child inherit the environment as it is.
- The credential-filename blocklist on `/api/source`. Reads stay confined
  to regular text files inside the project root.
- `[cross_lang] frontend_api_patterns`, which had no effect.
- `GraphQuerier.find_call_path` (use `find_call_path_result`) and the
  `python -m descry.embeddings` entry point.

## [0.2.0] — 2026-04-20

A large feature + resolution-quality release. Seven new SCIP language
adapters land (Java/Kotlin/Scala, Go, Ruby, PHP, C#/VB, C/C++, Dart)
alongside a language-general STDLIB_FILTER sweep that brought CALLS
resolution above 90% on 20 of 26 measured corpora. The release also
folds in a full session of convergence-audit hardening: user-input
parsing no longer 500s on garbage query strings, subprocess output
tolerates non-UTF-8 bytes, and the documented security invariants match
what the code actually does.

Graph schema stays at v1 — existing `.descry_cache/codebase_graph.json`
files remain readable. CLI subcommands and MCP tool names are unchanged
(19 of each, 1:1 parity).

### Added

- **Dart / Flutter support** via `scip-dart` (Milestone D). DartAdapter
  discovers projects by `pubspec.yaml` at the root or under
  `packages/*` (melos / workspace layout); runs `scip-dart ./` from the
  package root and uses `output_mode="rename"` to move the emitted
  `index.scip` into the cache. DartParser handles imports
  (`package:` / `dart:` / relative), classes (including `abstract`,
  `base`, `sealed`, `final`, `interface class`, `mixin class`), mixins,
  extensions, enums, typedefs, top-level constants, class inheritance
  (`extends` → INHERITS edge), and call sites with null-safe receiver
  chains (`?.method()`). Dart has heavy top-level code (top-level
  functions, top-level variable initializers), so the parser emits
  calls at file scope too — unlike Java/C# parsers which gate calls on
  non-file context. pub smoke test hit **71%** resolution
  (10,279 / 14,490 CALLS edges) — below the 91.2% Rust bar, accepted
  as the Dart ceiling for test-heavy codebases where pub's internal
  test-helper DSL (`servePackages`, `d.file`, `runPub`, etc.) contributes
  the bulk of unresolved calls.
- **Java / Kotlin / Scala support** via `scip-java` (Milestone J of Wave 2).
  JavaParser extracts classes, interfaces, enums, records, methods,
  constructors, fields, imports, and call sites. JavaAdapter ships a
  Gradle init-script that strips `-Werror` so scip-java works out of the
  box on Apache projects (Kafka, etc.) whose builds treat warnings as
  fatal. Kafka smoke test hit **92.7%** CALLS resolution.
- **Go support** via `scip-go` (Milestone G). GoParser covers packages,
  grouped imports, type declarations, free functions, methods with
  receivers, const/var blocks, and call sites. Kubernetes smoke test
  hit **98.3%** resolution.
- **Ruby support** via `scip-ruby` (Milestone R). RubyParser uses
  indent-based context tracking (Ruby uses `end` not `}`); extracts
  classes with INHERITS edges, modules, methods (including `self.foo`,
  `foo?`, `foo!`), `attr_reader/writer/accessor`, `require` /
  `require_relative`, and top-level constants. Rails smoke test hit
  **87.8%** — below the 91.2% Rust bar, accepted as the Ruby-without-
  Sorbet ceiling (scip-ruby falls back to `# typed: false` heuristics
  when Sorbet annotations are absent).
- **PHP support** via `scip-php` (Milestone P, third-party indexer by
  davidrjenni). PhpParser handles namespaces, classes / interfaces /
  traits / enums, `public/protected/private function`, properties,
  constants, and method / static / instance calls; Allman-brace lookahead
  (scan up to 10 lines for the opening `{`) was needed for Laravel's
  style. Laravel smoke test hit **88.6%**.
- **C# / VB.NET support** via `scip-dotnet` (Milestone N). DotnetAdapter
  sets `DOTNET_ROLL_FORWARD=LatestMajor` so scip-dotnet's net9 target
  runs on systems with only net10 installed. Serilog smoke test hit
  **83.7%**.
- **C / C++ support** via `scip-clang` (Milestone C). ClangAdapter emits
  scheme `cxx` (verified from real indexes; not `scip-clang` as a name
  might suggest). Discovery gives top priority to root-level
  `compile_commands.json` so Bear-backed Makefile builds and top-level
  CMake projects work as a single unit. ClangParser avoids regex
  catastrophic backtracking (which hit >20s on Redis `src/dict.c` in
  an early draft) via a hand-rolled `_extract_function_name` that scans
  right-to-left through the argument list. Redis smoke test hit
  **79.2%**; headers lag .c files (63.3% vs 81.2%) due to scip-clang
  compdb-coverage limits on transitively-included headers.

### Changed

- **CALLS resolution rose above the 90% bar on 20 of 26 measured
  corpora** (up from ~10 pre-sweep) via a focused language-general
  filter pass — zero codebase-specific entries. The sweep expanded
  `STDLIB_FILTER` / `STDLIB_PREFIXES` across every supported language:
  - **TypeScript / JavaScript**: `_JSTS_CONTROL_KEYWORDS` filter in
    the `TSParser` regex fallback (rejects `if(`, `while(`, `for(`,
    `switch(` captures — removes ~37k spurious unresolved on the
    TypeScript compiler alone), tslib emit helpers (`__awaiter`,
    `__generator`, `__rest`, `__spreadArray`, `__classPrivateFieldGet`,
    `__addDisposableResource`, …), DOM APIs (`textContent`,
    `getAttribute`, `classList`, `appendChild`, …), Array/Object
    statics (`isArray`, `fromEntries`, `defineProperty`, …),
    Playwright (`page.*`), Vitest/Jest todo / only / each / snapshot
    modifiers, React core hooks + TanStack Query + Node `assert`
    strict variants + Lingui / nanoid / uuid.* / date-fns / RN-Expo
    surface. ast-grep now rejects `$FUNC` captures that include parens
    or non-identifier characters (stops curried-call shape like
    `test.runIf(isBuild)(…)` from polluting CALLS targets).
  - **C#**: `var` added to `_DOTNET_CONTROL_KEYWORDS` (fixes ~1k/file
    spurious `var (` captures on modern C# codebases) along with the
    full modern declaration vocabulary (`record`, `partial`, `sealed`,
    `override`, `static`, `params`, `required`, `scoped`, `file`,
    `global`) and every primitive alias (`object`, `string`, `bool`,
    `int`, `long`, `byte`, `char`, `float`, `double`, `decimal`,
    `nint`, `nuint`). Moq (`It.IsAny`, `Mock.Of`, `Setup`, `Verify`,
    `ReturnsAsync`), FakeItEasy (`A.CallTo`, `A.Fake`, `A.Dummy`),
    ASP.NET Core DI (`Add*`, `Use*`, `Map*`, `GetRequiredService`),
    .NET Reflection + Span/Memory, `System.String` statics, Selenium
    WebDriver.
  - **Java**: AssertJ (full `assertThat*` surface, `hasCauseInstanceOf`,
    `isThrownBy`, `satisfies`), BDDMockito (`given`, `willReturn`,
    `willThrow`), Project Reactor (`StepVerifier`, `expectNext`,
    `flatMap`, `switchIfEmpty`, `subscribeOn`), Mockito static
    (`mockStatic`, `mockConstruction`, `invocation.getArgument`),
    Hamcrest matcher vocabulary, Apache Lucene types, ANTLR-generated
    runtime, `java.time` / `java.util.Properties` / `java.nio.Buffer`
    / `java.io.File`, `java.sql.*` + `java.util.Locale.*`,
    `jakarta.servlet` + `javax.xml.sax`, reflection extras.
  - **Go**: `encoding/binary` (`BigEndian/LittleEndian.Uint*/PutUint*`),
    `sync` lock methods (`Lock`, `Unlock`, `RLock`, `RUnlock`,
    `Add`/`Done`/`Wait`), `bytes.Buffer` + `strings.Builder` common
    methods, `golang.org/x/sync/errgroup`, `go.uber.org/goleak`,
    AWS SDK v2 helpers (`aws.Int32`, `aws.String`, …), ULID,
    `klog.`/`ginkgo.`/`Gomega.`/`testify.` prefixes.
  - **Rust**: clap CLI (`value_name`, `help_heading`, `get_one`,
    `get_flag`, `long_about`, subcommand builders), snapbox test
    (`stdout_eq`, `stderr_eq`, `subset_matches`), integer arithmetic
    (`wrapping_*`, `saturating_*`, `checked_*`, `overflowing_*`,
    `rotate_*`, `count_ones`/`_zeros`), atomic compare-exchange /
    fetch variants, pointer `as_ptr`/`as_mut_ptr`/`copy_nonoverlapping`/
    `read_volatile`, slice `is_ascii_*`/`split_whitespace`/
    `char_indices`, fd `from_raw_fd`/`as_raw_fd`/`borrow_raw`.
  - **C / C++**: TCL C API (`Tcl_*` prefix — sqlite bindings),
    LLVM C API (`LLVM*` — postgres JIT), CPython C API (`Py_*`,
    `PyObject_*`, `PyDict_*`, …), Win32 C API (`CreateFile`,
    `GetLastError`, `LoadLibrary`, registry, `HeapAlloc`, …), POSIX
    pthread full surface (`pthread_mutex_*`, `pthread_cond_*`,
    `pthread_rwlock_*`, `pthread_spin_*`, `pthread_barrier_*`),
    stdarg (`va_start`, `va_end`, `va_arg`, `va_copy`), C11
    stdatomic, BSD-specific strings (`strlcpy`, `strlcat`, `arc4random_*`,
    `reallocarray`), BSD socket byte-order (`ntohs`, `htonl`, …) +
    full socket API, OpenSSL (`BN_`, `HMAC_`, `EC_KEY_`, `ENGINE_`),
    jemalloc (public `mallocx`/`xallocx`/`rallocx`/`sallocx`/
    `dallocx`/`nallocx` + internals `edata_`, `emap_`, `sz_`, `pa_`,
    `hpa_`, `prof_`, `tsd_`, `tsdn_`, `atomic_*_zu`).
  - **PHP**: PHPUnit `assertIs*` full surface + PHPUnit:: qualified
    variants, Mockery extras (`makePartial`, `withArgs`), Carbon +
    PHP `DateTime` full method set, `version_compare`, binary /
    encoding / output stdlib (`pack`, `unpack`, `bin2hex`,
    `curl_setopt_array`, `ob_start`/`ob_end_flush`, `strspn`, `strtr`,
    `addslashes`), PSR-7 HTTP helpers (`getStatusCode`, `withHeader`,
    `withBody`, `getUri`).
  - **Ruby**: Full `String` / `Enumerable` / `Hash` / `IO` method
    surface (`gsub`, `scan`, `end_with?`, `is_a?`, `key?`, `sysread`,
    `each_with_object`, `tally`, …), `SecureRandom`/`Addrinfo`/
    `Socket`/`TCP*`/`UDP*`/`UNIX*`/`IPAddr`/`Resolv`/`Zlib`/`CGI`/
    `ERB`/`Liquid::` prefixes, JRuby Java-side API (`RubyString.`,
    `RubyArray.`, `RubyHash.`, `ByteList.`, `ThreadContext.`, …) +
    bare `newString`, `newFixnum`, `callMethod`, `getRuntime`.
  - **Dart**: `package:test_descriptor` (`d.file`, `d.dir`,
    `d.nothing`), `package:test_process` (`shouldExit`), `dart:typed_data`
    fromList constructors, `dart:convert` (`JsonEncoder`, `Utf8Decoder`,
    `LineSplitter`), `package:shelf` Response helpers, `package:args`
    CLI builders (`addFlag`, `addOption`, `addSubcommand`,
    `wasParsed`), `package:yaml` loaders, Dart SDK error types
    (`StateError`, `UnsupportedError`, `FormatException`, …),
    pub_semver (`Version`, `VersionConstraint`).
- **`is_non_project_call` now splits on `->` in addition to `.` and
  `::`** so PHP `$obj->method()` and C `struct_ptr->method()` resolve
  their bare method name against `STDLIB_FILTER` — previously
  `headers->set`, `response->getStatusCode` never matched. Laravel /
  Symfony resolution climbed 88% → 91%.

### Fixed

- **SCIP incremental re-indexing** no longer silently degrades to zero
  for adapters outside rust / typescript / python. `_hash_project` now
  falls back to `_hash_generic_adapter` (walks the adapter's declared
  extensions + hashes paths + bytes) for java / go / ruby / php /
  dotnet / clang / dart, instead of raising `ValueError: Unknown project
  type`. Before this fix, every second+ index produced a `.scip`-less
  graph that looked successful in logs but had no SCIP-resolved CALLS
  edges.
- **SCIP Strategy 1 name match.** `ScipIndex.resolve`'s exact
  (file, line) lookup used to fall back to the first candidate in the
  definitions map when no name matched. Downstream, the cross-crate
  name check in `generate.py` would reject those mismatched targets,
  but by then Strategy 2 (fuzzy resolve by name) had already been
  skipped. Tighten Strategy 1 to require a name match; non-matching
  line lookups now fall through to fuzzy. Measured: pub Dart 52% →
  71%, Redis C 79.2% → 80.4%.
- **`changes` no longer crashes on non-UTF-8 git diff output.**
  `GitHistoryAnalyzer._run_git` previously used `subprocess.run(text=True)`
  which raises `UnicodeDecodeError` the moment `git diff` emits a byte
  that isn't valid UTF-8 (observed on c-postgres — byte 0x92 — and
  rust-coreutils — byte 0xfd, both carrying Latin-1 content in older
  source files). All subprocess callers that capture stdout/stderr
  (`git_history.py`, `scip/cache.py`, `scip/adapter.py`, `ast_grep.py`)
  now decode with `encoding="utf-8", errors="replace"` so a stray
  non-UTF-8 byte becomes `U+FFFD` instead of aborting the tool.
  Regression: `tests/test_git_history.py::TestGitHistoryNonUtf8Output`
  seeds a throwaway repo with 0x92 / 0xfd bytes and asserts
  `_run_git` + `get_changes` stay clean.
- **User-input numeric parsing is crash-safe.** The web API's
  `?limit=foo` / `?depth=bar` inputs and the `DESCRY_SCIP_WORKERS` /
  `DESCRY_PRIME_THREADS` / `DESCRY_SCIP_TIMEOUT` env-var overrides
  previously called `int(raw)` directly; a non-integer value raised
  `ValueError` and surfaced as a 500 (web) or a hard crash during
  SCIP cache init (env var). Web handlers now go through the new
  `_int_param(request, name, default)` helper that returns the
  default on parse failure; env-var paths catch `ValueError`, log a
  warning, and fall through to the auto-tuned default.
- **Exception chains preserved on re-raise.** Three sites in
  `git_history.py` and one in `embeddings.py` re-raised exceptions
  without `from err` / `from None`, which hid the underlying cause in
  tracebacks. Now `FileNotFoundError` → `GitError("git not found")` /
  `TimeoutExpired` → `GitError("timed out")` / `FileExistsError` →
  `TimeoutError("embeddings lock")` all use `from None` (cause
  already captured in the message), and `_verify_git`'s internal
  re-raise uses `from err` to preserve the inner git failure.
- **`descry.__version__` matches `pyproject.toml`.** Since the 0.1.1
  tag, the package version string was `"0.1.0"` in `__init__.py` but
  `"0.1.1"` in `pyproject.toml`, so `descry health` reported the
  wrong version. Bumped `__init__.py` and added this to the release
  gate so a future mismatch is caught before tagging.
- **Documented security invariants now match the code.** `CLAUDE.md`
  said `descry-web` "allows CORS `*`"; in reality the code omits
  `CORSMiddleware` entirely so browsers enforce same-origin by
  default (with `TrustedHostMiddleware` defeating DNS-rebinding).
  The subprocess-sanitization list also named only `scip-typescript`
  as the representative SCIP indexer — now enumerates all ten.
- **Implicit-`Optional` annotations in `query.py` made explicit.**
  11 signatures across `flow` / `trace_flow_structured` /
  `get_context_prompt` / `find_trait_impls` used `x: int = None`
  (which PEP 484 prohibits); now `int | None = None`.
- **Pure-JavaScript repos no longer drop every CALLS edge.** ast-grep
  has separate `typescript` and `javascript` grammars; the
  `extract_calls_typescript` / `extract_imports_typescript` helpers
  hardcoded `-l typescript` for the subprocess, which returns no
  matches when fed `.js` / `.jsx` / `.mjs` / `.cjs`. Caught by the
  pre-publish corpus sweep against express (100% .js): scip-typescript
  emitted 9,702 references but the graph had **0 CALLS edges**. New
  `_ast_grep_lang_for(file_path)` routes to `javascript` for JS-family
  extensions; express now resolves 58% of CALLS (graph went 1,353 →
  6,117 edges). 0.1.x users on pure-JS repos were silently affected.

### Changed

- **Configuration docs now cover every parsed field.** README's
  `[cross_lang]` TOML section (with `openapi_path`,
  `backend_handler_patterns`, `frontend_api_patterns`, `api_prefixes`)
  and `[timeouts] index_minutes` were parsed at runtime but undocumented;
  added. Four `DESCRY_*` env vars (`SCIP_WORKERS`, `SCIP_TIMEOUT`,
  `PRIME_THREADS`, `AST_GREP_MAX_FILES`) were honoured by the code but
  missing from the environment-variables table; added.

### Known limitations (surfaced by the pre-publish corpus sweep)

These aren't regressions in 0.2.0 — they're scenarios the new corpus
coverage made visible. Documenting so users know what to expect.

- **Kotlin codebases need scip-java to be useful.** Descry has no
  `.kt` / `.scala` parser baseline; Kotlin and Scala source files are
  walked by the file walker but produce no nodes when scip-java
  doesn't run. scip-java in turn fails on Kotlin-DSL Gradle projects
  that use precompiled-script-plugins (observed on both `ktorio/ktor`
  and `Kotlin/kotlinx.coroutines` — same upstream task-ordering
  error). Practical effect: pure-Kotlin repos with that build
  pattern will produce a near-empty graph. Mixed Java+Kotlin repos
  index the Java side only. The `JavaAdapter` now pre-flights for
  this scenario and skips scip-java with a clear log message
  ("scip-java incompatibility: kotlin-dsl precompiled-script-plugins
  in <path>") instead of a confusing 1100-byte stderr dump from the
  failed Gradle invocation. Detection covers `buildSrc/` and any
  `includeBuild()` participants in `settings.gradle{,.kts}`.
- **C++ resolution ceiling without scip-clang.** scip-clang requires
  a working `compile_commands.json` with all transitive deps
  available (Boost, gflags, glog, fmt, etc. for `facebook/folly`).
  When deps are missing scip-clang is skipped and the regex
  `ClangParser` baseline reaches ~46–54% on modern template-heavy
  C++ (folly 46%, abseil-cpp 54%). With a working compdb scip-clang
  contributes ~40% of resolved CALLS; the rest is regex fallback.
  Plain C codebases (redis, sqlite, postgres, curl) sit at 72–80%
  on regex baseline alone.
- **scip-java does not support Ant.** Maven, Gradle, sbt, and Mill
  are supported; Ant projects (e.g. `apache/cassandra`) fall back
  to the regex `JavaParser`, which still reaches ~86% on its own.
- **scip-dart needs `dart pub get` first.** scip-dart aborts with
  "Unable to locate packageConfig" when the project hasn't been
  bootstrapped. Run `dart pub get` (or `flutter pub get` for Flutter
  packages) before `descry index` to get type-aware Dart resolution.
- **scip-dotnet honours `global.json` SDK pins.** When a project
  pins an SDK not installed on the host (e.g. `PowerShell/PowerShell`
  pins .NET 11 preview), scip-dotnet's adapter detects the
  incompatibility and falls back to the regex `DotnetParser`. The
  regex baseline alone reached 90% on PowerShell, so the fallback
  is usually fine.
- **Pinned Rust toolchains need `rust-analyzer` installed.** A
  `rust-toolchain.toml` pin (e.g. `helix` pins `1.90.0`) means
  rust-analyzer must be installed for that specific toolchain
  via `rustup component add rust-analyzer --toolchain <version>`.
  Otherwise scip-rust emits per-crate warnings and the regex
  `RustParser` baseline carries the project (~80%).

## [0.1.1] — 2026-04-17

Patch release focused on making cross-language tracing actually configurable,
plus a round of dead-code cleanup. No breaking changes.

### Fixed

- **Cross-language tracing is now configurable from `.descry.toml`.** The
  `DescryConfig.openapi_path` field existed in 0.1.0 but had no TOML loader
  and the web `/api/cross-lang` handler hardcoded `public/api/latest.json`,
  so custom spec locations silently did nothing. Added a `[cross_lang]`
  section with `openapi_path`, `backend_handler_patterns`,
  `frontend_api_patterns`, and `api_prefixes` keys. Web and CLI/MCP now
  both honour the config and pass all four through to `CrossLangTracer`.
- `[cross_lang] openapi_path` is containment-checked against the project
  root; a crafted `.descry.toml` cannot point the indexer at files outside
  the configured project.

### Removed

Dead code with no call sites anywhere in the tree was pruned:

- `DescryConfig.project_markers` (auto-detect used a module constant).
- `DescryConfig.use_tree_sitter_ts` (scaffolding field with no consumer).
- `descry.query.MAX_INLINE_THRESHOLD` constant.
- `CrossLangTracer.endpoint_to_node_id()` and module-level
  `_create_cross_lang_edges()` helper.
- `SemanticSearcher._find_similar()` (distinct from
  `GraphQuerier._find_similar_nodes`, which remains).
- `descry.ast_grep.extract_imports_typescript_batch()`.
- `ScipIndex.get_definition_location()` and `get_symbol_info()`.
- `TypeScriptSymbolTable.file_dir` attribute.
- `DescryService._clear_dedup_cache()` (superseded by `reset_caches()` in
  the 0.1.0 hardening sweep).

A `vulture --min-confidence 80` sweep on `src/descry/` now reports zero
findings outside generated protobuf code.

## [0.1.0] — 2026-04-16

Initial public PyPI release. Descry is a polyglot codebase knowledge graph toolkit with three interfaces: CLI (`descry`), MCP server (`descry-mcp`), and local web UI (`descry-web`).

### Features

- **Indexer**: Parses Rust, Python, TypeScript, JavaScript, and Svelte into a cached knowledge graph of symbols (functions, classes, constants) and edges (calls, imports, defines).
- **SCIP integration**: Optional type-aware call resolution via `rust-analyzer` and `scip-typescript`; regex fallback otherwise.
- **Semantic search**: Optional embeddings via sentence-transformers (Jina code embeddings by default; model pinned by revision).
- **19 MCP tools**: search, callers, callees, context, flow, path, impls, structure, flatten, cross-lang (preview), churn, evolution, changes, semantic, quick, index, status, ensure, health.
- **Web UI**: Starlette + Alpine.js; 20+ UI panels for browsing the graph visually.
- **Configuration**: `.descry.toml` + env-var overrides for cache dir, timeouts, embedding model, SCIP toolchain, excluded dirs.

### Security

- Git argument injection hardening on all user-controlled inputs (commit ranges, symbol names, file paths, pathspecs).
- Safe embedding-cache storage (JSON sidecar + safe-mode numpy load) with atomic writes and content-addressed cache keys.
- Default embedding model revision pinned for supply-chain integrity; `trust_remote_code` defaults to `False` for user-supplied models.
- TOML-sourced subprocess args (scip toolchain, extra args, embedding model path) are validated.
- Subprocess env sanitized against known credential patterns.
- Web UI path traversal containment on `/api/source` + regular-file / size / text-file checks.
- MCP `descry_index(path=...)` restricted to project root.
- Graph JSON carries `schema_version`; mismatched graphs are rejected with an actionable error.

