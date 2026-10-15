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
- **Standard libraries are generated data.** `src/descry/stdlib/<lang>.json`
  says which types and members a language's standard library has and what
  they return, for Rust, TypeScript and JavaScript, Python, Go, Java, PHP,
  C#, Ruby, Dart, C and C++. A call on a standard value is external, and
  its result is typed for the next call in a chain. The files are written
  by `tools/stdlib/` from each language's own machine-readable description
  (rustdoc JSON, TypeScript's `lib.*.d.ts`, typeshed, Go's `api/*.txt`,
  `javap`, phpstorm-stubs, .NET reflection, RBS, the Dart SDK, clang's AST)
  and replace the hand-written name lists for those languages.
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
  it was kept as one of the project definitions of its name. Index files
  are named
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
  generic receivers, one-line functions; Ruby one-line classes and modules,
  `#` docstrings; C# properties and verbatim strings; TypeScript and
  JavaScript `.jsx`/`.mjs`/`.cjs`/`.mts`/`.cts` files, `$` identifiers,
  typed arrow constants, file-level calls; calls in comments, string
  literals and JSX text; the character after a JavaScript or TypeScript
  `/* */` comment, which was dropped; a `require` or dynamic `import()`
  with a comment before its module or written over lines; a call on a
  parameter of an arrow constant or of an arrow assigned to a variable,
  dropped as external where it is left unresolved; a `<` or `/` that
  goes on with a comparison or a division from the line before, read as
  a JSX element that hid the rest of the file or as a regular
  expression; a JSX element or regular expression after the head of an
  `if`, `for`, `while` or `with`, a spread's `...` or `default`, read as
  a comparison or a division; chained calls; declarations recorded as
  calls; braces and calls inside string literals in C#, PHP, Dart and
  C/C++; a licence notice
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
  it shares the name of the type the arrow is given. A generic arrow (`<T>(x: T) => ...`) is read as a function,
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
  gives over several lines, or a Dart variable whose value is a call of
  a standard type's method. `index` no longer encodes every symbol twice.
  `descry-generate` honours `DESCRY_LOG_LEVEL`, walks the project root
  when given `--project-root` alone, and refuses a path outside it.
  scip-java no longer passes the Gradle init-script on Maven or sbt
  projects, where `mvn`/`sbt` rejected it and no index was written; its
  Kotlin-DSL pre-flight likewise runs only for a Gradle build, which is
  now recognised by every marker scip-java's own detector accepts
  (`settings.gradle` and `gradlew` beside the two build files). A Gradle
  multi-project root is indexed as one build without a build file of its
  own: by `settings.gradle`, or by `settings.gradle.kts` beside `gradlew`.
  Where scip-java also detects another build tool at such a root (Maven,
  sbt, Mill, Bazel or a SCIP config), with or without a build file of its
  own, it refuses the root, and the root's subprojects are indexed instead.
  Every JVM project is discovered on the same terms: a directory where
  scip-java detects exactly one build tool, by the regular files its own
  detectors read, so it is not run where it would answer "No build tool
  detected" or "Multiple build tools detected". A Rust root package beside
  its workspace members, or beside nested crates, is indexed too:
  `rust-analyzer scip` indexes only the crate at the path it is given, so
  the root package had no index; it is named `.` where a member holds the
  root directory's name. scip-dart is run on `.`; given `./` it
  took the package for a nested one, skipped it and wrote an empty index.
  scip-dart is found under either spelling — `scip-dart` or the
  `scip_dart` `dart pub global` installs, read from `$PUB_CACHE` when
  it is set — and its package discovery reaches
  `<collection>/<name>/pubspec.yaml` (`pkgs/`, `apps/`, …), not only
  `packages/`, while a package under an excluded or dot-directory is left
  alone. An index file that parses but carries no symbols is logged, since
  that usually means the indexer's own environment was missing rather than
  a project with no symbols.
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
  by `/api/source`, and no syntax language for most extensions in the
  source view.
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
- Subprocess environment filtering (`safe_env`). Git, ast-grep, the SCIP
  indexers and the indexing child inherit the environment as it is.
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

