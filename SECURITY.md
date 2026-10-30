# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Descry, please report it privately via email to **06chaynes@gmail.com** with the subject line `[SECURITY] <short description>`.

Please do **not** file a public GitHub issue for security concerns.

Include in your report:
- A description of the issue and its impact.
- Steps to reproduce (minimal example or proof of concept).
- Affected version(s) and platform.
- Any suggested remediation, if you have one.

## Response Timeline

- Acknowledgement of your report within **7 days**.
- Initial triage and severity assessment within **14 days**.
- Coordinated disclosure target: **90 days** from acknowledgement, or sooner if a fix can be shipped quickly. If an issue requires longer, we will discuss a revised timeline with you.

## Security Model

descry reads repositories it does not trust: the code it indexes, a
`.descry.toml` or cache a repository ships, and the text of its files
are data, never instructions.

- **descry's own indexing runs nothing from the repository.** The indexer
  is a child Python started with `-P`, so a `descry/` package in the
  repository cannot shadow descry's own. Git runs with the repository's
  hooks, filters, external diffs and other configured commands turned off.
- **SCIP indexers are the languages' own tools, and some run the
  project's code:** rust-analyzer runs build scripts and procedural
  macros, scip-java runs the Gradle or Maven build, scip-dotnet runs
  MSBuild, scip-clang runs the compiler the repository's
  `compile_commands.json` names. Index a repository you do not trust with SCIP off
  (`DESCRY_NO_SCIP=1`, or `[features] enable_scip = false`).
- **Files are read only inside the project root.** A path in a graph, or
  a symlink in the tree, that leads outside it is refused. With SCIP on,
  the indexers, and descry's checks of what they will index, read the
  tree as the languages' own tools do.
- **The cache stays inside the project** unless `DESCRY_CACHE_DIR` names
  another place; a shipped `.descry_cache` that resolves elsewhere is
  refused. A file's kept parse, and a node's kept embedding vector, are
  reused only from what descry signed with a key in the user's state
  directory, so a shipped cache is parsed and encoded over.
- **Settings that reach a subprocess are checked.** Toolchains, indexer
  arguments and model names from `.descry.toml` are refused if they carry
  shell syntax, short flags or paths outside the project.
- **No model code runs.** Embedding models load with
  `trust_remote_code=False`, at pinned revisions for the models descry
  knows.
- **The web UI is for this machine only.** It listens on 127.0.0.1 and
  rejects other `Host` headers and cross-origin writes.
- **The MCP server fences repository text.** Source, docstrings and other
  text from the repository reach the agent inside a tagged envelope, and
  structured answers carry none of it but the OpenAPI tags, paths and
  operation IDs a `cross-lang` answer lists, the import lines a
  `structure` answer names, and the file paths its IDs are.

## Supported Versions

Only the latest release on PyPI receives security fixes. Users on older versions are encouraged to upgrade.

## Credits

Researchers who report verified vulnerabilities are credited in the release notes unless they request anonymity.
