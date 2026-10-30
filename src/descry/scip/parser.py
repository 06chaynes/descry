"""Parse SCIP index files and provide symbol resolution.

SCIP (Source Code Index Protocol) provides type-aware symbol information
from the language's SCIP indexer that enables more accurate call resolution
than regex-based approaches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import descry.scip.adapters  # noqa: F401 — side-effect: populate ADAPTERS registry
from descry._files import open_regular
from descry.scip.adapter import adapter_for_scheme, descriptor_names

logger = logging.getLogger(__name__)

# Symbol role bitflags (from SCIP spec)
SYMBOL_ROLE_DEFINITION = 0x1


class ScipIndex:
    """Merged SCIP index for symbol resolution.

    Loads multiple SCIP files and provides lookup capabilities for:
    - Finding where symbols are defined
    - Resolving references to their definitions
    - Converting between SCIP symbol IDs and descry node IDs
    """

    def __init__(
        self,
        scip_files: list[Path],
        project_dirs: dict[Path, str] | None = None,
        root: Path | None = None,
    ):
        """Initialize the index from SCIP files.

        Args:
            scip_files: List of paths to .scip files to load
            project_dirs: For an index whose document paths are relative to
                its project rather than to the repository root, that
                project's directory relative to the root ("" for the root)
            root: The repository root. An index not in ``project_dirs`` that
                names a project root at or below it gets its project
                directory from that.
        """
        self._project_dirs = project_dirs or {}
        self._root = root
        # Document paths made relative to the repository root on load.
        self._rooted_files: set[str] = set()

        # symbol_id -> file_path -> the lines defining it there
        self.definitions: dict[str, dict[str, set[int]]] = {}

        # (file_path, line) -> list of symbol_ids (may be multiple if several
        # crates contribute the same relative file). At resolve time we prefer
        # a candidate whose extracted name matches ref_name, then fall back to
        # the first.
        self.references: dict[tuple[str, int], list[str]] = {}

        # (file_path, line) -> (start column, symbol_id) of each occurrence
        # there
        self._spans: dict[tuple[str, int], list[tuple[int, str]]] = {}
        # file_path -> the SCIP position encoding its columns are counted in
        self._position_encodings: dict[str, int] = {}

        # Simple name -> list of symbol_ids (for fuzzy matching)
        self.name_to_symbols: dict[str, list[str]] = {}

        # scheme -> test of whether an outside symbol's call leaves the project
        self._leaving: dict[str, object] = {}

        # The symbols a symbol of the indexes implements: a base a class
        # derives from, a method one overrides
        self.implemented: set[str] = set()

        for scip_file in scip_files:
            if scip_file.is_file():
                self._load_scip(scip_file)

    def _load_scip(self, scip_file: Path):
        """Load a SCIP index file.

        Args:
            scip_file: Path to the .scip file
        """
        try:
            # Import here to avoid import errors if protobuf not installed
            from google.protobuf.message import DecodeError
            from google.protobuf.runtime_version import VersionError

            from descry.scip import pb2 as scip_pb2
        except ImportError:
            logger.warning("SCIP: protobuf bindings not available")
            return
        except VersionError as e:
            # Raised by pb2 at import when the protobuf runtime is older
            # than the generated code.
            logger.warning(f"SCIP: protobuf bindings unusable: {e}")
            return

        try:
            index = scip_pb2.Index()
            with open_regular(scip_file) as f:
                index.ParseFromString(f.read())
        except (OSError, DecodeError) as e:
            logger.warning(f"SCIP: Failed to parse {scip_file}: {e}")
            return

        if not any(d.occurrences for d in index.documents):
            # An indexer that ran but found nothing usually means its own
            # environment was missing (a `pip`/`java` on PATH, a `dart pub
            # get`), not that the project holds no symbols — the regex
            # rates that follow look correct while SCIP contributes 0%.
            logger.warning(
                f"SCIP: {scip_file} carries no symbols — "
                f"check the indexer's output and its own requirements"
            )

        project_dir = self._project_dirs.get(scip_file)
        if project_dir is None:
            project_dir = self._stated_project_dir(index.metadata.project_root)
        for doc in index.documents:
            file_path = doc.relative_path
            if project_dir is not None:
                file_path = f"{project_dir}/{file_path}" if project_dir else file_path
                self._rooted_files.add(file_path)
            self._position_encodings[file_path] = doc.position_encoding
            self.implemented.update(
                relation.symbol
                for info in doc.symbols
                for relation in info.relationships
                if relation.is_implementation
            )

            # Process occurrences (references and definitions)
            for occ in doc.occurrences:
                symbol_id = occ.symbol
                if not symbol_id:
                    continue

                # Extract line number from range (range is [startLine, startChar, ...])
                # SCIP lines are 0-based; descry edge linenos are 1-based.
                line = occ.range[0] + 1 if occ.range else 0

                # Check if this is a definition (symbol_roles & 0x1)
                if occ.symbol_roles & SYMBOL_ROLE_DEFINITION:
                    self.definitions.setdefault(symbol_id, {}).setdefault(
                        file_path, set()
                    ).add(line)

                # Store all occurrences for reverse lookup. Append to a list
                # so multi-crate workspaces that share relative filenames
                # don't silently overwrite each other.
                key = (file_path, line)
                if len(occ.range) > 1:
                    self._spans.setdefault(key, []).append((occ.range[1], symbol_id))
                bucket = self.references.get(key)
                if bucket is None:
                    self.references[key] = [symbol_id]
                elif symbol_id not in bucket:
                    bucket.append(symbol_id)

            # Build name-to-symbol index for fuzzy matching
            for sym in doc.symbols:
                name = self._extract_name(sym.symbol)
                if name:
                    known = self.name_to_symbols.setdefault(name, [])
                    if sym.symbol not in known:
                        known.append(sym.symbol)

    def _extract_name(self, symbol_id: str) -> str | None:
        """Extract the simple name from a SCIP symbol ID.

        SCIP symbol format:
        rust-analyzer cargo <crate> <version> <descriptors...>
        scip-typescript npm <package> <version> <descriptors...>

        Examples:
        - "rust-analyzer cargo mydb 0.1.0 database/migrations/run_migrations()."
          -> "run_migrations"
        - "rust-analyzer cargo backend 0.1.0 state/AppState#new()."
          -> "new"
        - "scip-typescript npm webapp 0.1.0 src/lib/api/`client.ts`/getAuthToken()."
          -> "getAuthToken"

        Returns the last identifier (e.g., "run_migrations"); None for a
        local symbol (``local <id>``) or one without descriptors.
        """
        parts = symbol_id.split()
        if len(parts) < 5:
            return None
        name_parts = self._name_parts(parts)
        return name_parts[-1] if name_parts else None

    @staticmethod
    def _name_parts(parts: list[str]) -> list[str]:
        """The names of the descriptors of the symbol split into ``parts``
        (scheme, manager, package, version, descriptors), as the adapter of
        its scheme reads them; Rust's rule where no adapter owns it."""
        descriptors = " ".join(parts[4:])
        adapter = adapter_for_scheme(parts[0])
        if adapter is not None:
            return adapter.parse_descriptors(descriptors)
        return descriptor_names(descriptors, frozenset({"impl"}))

    def resolve(
        self,
        ref_name: str,
        source_file: str,
        line: int,
        column: int | None = None,
    ) -> str | None:
        """Resolve a reference to its definition node ID.

        Attempts resolution in order:
        1. The occurrence SCIP recorded at the position (``column``),
           whatever name it carries — an aliased base is recorded under
           the definition's name, an unresolved one as ``local N``. Its
           symbol without a project definition names an external one.
        2. Exact location lookup (file + line, by name)
        3. Try with package-relative path (for TypeScript: webapp/src/... -> src/...)
        4. Symbol name matching

        Args:
            ref_name: The reference name to resolve (e.g., "validate_token")
            source_file: Path to the file containing the reference
            line: 1-based line number of the reference (a descry edge `lineno`)
            column: 0-based column of the reference, when the parser
                recorded it

        Returns:
            Descry node ID if resolved, None otherwise
        """
        if column is not None:
            path, recorded = self._recorded_at(source_file, line, column)
            if len(recorded) == 1:
                symbol = recorded[0]
                # A `local N` symbol has no name to build a node from.
                if symbol in self.definitions and self._extract_name(symbol):
                    return self._to_node_id(symbol, self._defining_file(symbol, path))
                # Recorded, no project definition: external or unresolved.
                return None
            if recorded:
                # Several symbols at the position: SCIP does not tell
                # which; a name match could take the wrong one.
                return None
        path, chosen = self._symbol_at(ref_name, source_file, line)
        if chosen is not None:
            return self._to_node_id(chosen, self._defining_file(chosen, path))

        # Any occurrence at the position, whatever its name, means the
        # reference's identity is known: never the name fallback.
        if self._recorded_at(source_file, line, None)[1]:
            return None

        # Strategy 2: Try symbol name matching
        return self._fuzzy_resolve(ref_name)

    def definition_at(
        self,
        ref_name: str,
        source_file: str,
        line: int,
        column: int | None = None,
        alone: bool = True,
        within=(),
    ) -> tuple[str, str, tuple[int, ...]] | None:
        """Find the definition of the symbol SCIP recorded at a reference.

        Unlike ``resolve`` this never falls back to matching by name. With
        ``column`` (0-based) given, the symbol of ``ref_name``'s name whose
        occurrence on the line starts there is taken, else, where the call
        is ``alone`` of its name on the line, the line's one recorded where
        a call whose parentheses hold it starts, at one of the columns
        ``within``: an indexer records it at the macro whose expansion
        holds the call. A column is one number or ``(UTF-8, UTF-16,
        UTF-32)`` code-unit offsets.

        A symbol defined in several files, as an indexer names C `static`
        functions of one name, is the one of the reference's own file;
        where none is, SCIP does not tell which.

        Returns:
            ``(node_id, file, lines)`` of the definition, the lines 1-based
            and sorted, several where the file defines the symbol at each,
            as overload signatures or alternatives of the build; or None
            when no occurrence at ``source_file``:``line`` carries
            ``ref_name``'s name, ``column`` does not tell which symbol of the
            name the call is, or no file tells which definition it is
        """
        path, chosen = self._symbol_at(
            ref_name, source_file, line, column, alone, within
        )
        if chosen is None:
            return None
        def_file = self._defining_file(chosen, path)
        lines = self.definitions[chosen].get(def_file)
        if not lines:
            return None
        return self._to_node_id(chosen, def_file), def_file, tuple(sorted(lines))

    def _defining_file(self, symbol: str, path: str) -> str:
        """The file of the definition of ``symbol`` a reference in ``path``
        reaches: its only one, else ``path``'s own."""
        defined = self.definitions[symbol]
        return path if len(defined) > 1 else next(iter(defined))

    def _stated_project_dir(self, project_root: str) -> str | None:
        """The directory, relative to the repository root, that an index
        names in its metadata as the base of its document paths."""
        if self._root is None or not project_root.startswith("file://"):
            return None
        stated = Path(url2pathname(urlsplit(project_root).path))
        try:
            relative = stated.relative_to(self._root).as_posix()
        except ValueError:
            return None
        return "" if relative == "." else relative

    def outside_at(self, ref_name: str, source_file: str, line: int) -> bool:
        """Whether every symbol SCIP recorded at ``source_file``:``line``
        under ``ref_name``'s last segment is one outside the project whose
        call cannot reach it, as the adapter of its scheme tells."""
        named = self._named_at(ref_name, source_file, line)[1]
        return bool(named) and all(self._leaves(symbol) for symbol in named)

    def _leaves(self, symbol: str) -> bool:
        if symbol in self.definitions:
            return False
        scheme = symbol.partition(" ")[0]
        if scheme not in self._leaving:
            adapter = adapter_for_scheme(scheme)
            defined = [
                known for known in self.definitions if known.startswith(f"{scheme} ")
            ]
            # An adapter that cannot tell says nothing leaves.
            build = getattr(adapter, "leaving", None)
            self._leaving[scheme] = (
                build(defined, self._root, self.implemented)
                if build
                else (lambda _: False)
            )
        return self._leaving[scheme](symbol)

    def _symbol_at(
        self,
        ref_name: str,
        source_file: str,
        line: int,
        column: int | None = None,
        alone: bool = True,
        within=(),
    ) -> tuple[str, str | None]:
        """The path SCIP has ``source_file`` under, and the defined symbol
        referenced at ``line`` of it under ``ref_name``'s last segment, if
        any, as ``definition_at`` tells it."""
        path, named = self._named_at(ref_name, source_file, line)
        named = [cid for cid in named if cid in self.definitions]
        if column is not None and named:
            spans = self._spans.get((path, line), ())
            columns = self._columns(path, column)
            exact = set(named) & {s for at, s in spans if at in columns}
            if exact or len(set(named)) > 1 or not alone:
                return path, exact.pop() if len(exact) == 1 else None
            # The line's one symbol of the name, recorded where a call that
            # holds this one starts: at a macro whose expansion holds it.
            macros = set().union(*(self._columns(path, c) for c in within))
            held = any(s == named[0] and at in macros for at, s in spans)
            return path, named[0] if held else None
        return path, named[0] if named else None

    def _columns(self, path: str, column) -> set[int]:
        """The offsets ``column`` stands for in ``path``'s document: a column
        given in several units is read in the document's, and where the
        index does not say which, any of them may match."""
        if isinstance(column, int):
            return {column}
        unit = self._position_encodings.get(path)
        return {column[unit - 1]} if unit else set(column)

    def _recorded_at(
        self, source_file: str, line: int, column: int | None
    ) -> tuple[str, list]:
        """The path SCIP has ``source_file`` under, and the symbols it
        recorded at ``line`` of it — every name, at the exact ``column``
        when one is given: the reference's own identity."""
        path, bucket = self._bucket(source_file, line)
        if not bucket or column is None:
            return path, bucket
        at = {
            symbol
            for start, symbol in self._spans.get((path, line), ())
            if start in self._columns(path, column)
        }
        return path, list(at)

    def _named_at(self, ref_name: str, source_file: str, line: int):
        """The path SCIP has ``source_file`` under, and the symbols it
        recorded on ``line`` of it under ``ref_name``'s last segment."""
        path, candidates = self._bucket(source_file, line)

        # Only candidates whose extracted name matches count: one
        # (file, line) can hold several occurrences (type prefix, method
        # call, closing punctuation). SCIP names are unqualified, so
        # compare against the last segment of ref_name.
        simple_name = self._simple_name(ref_name)
        return path, [c for c in candidates if self._extract_name(c) == simple_name]

    def _bucket(self, source_file: str, line: int) -> tuple[str, list[str]]:
        """The path SCIP has ``source_file`` under, and the symbols it
        recorded on ``line`` of it: the path itself, else, where SCIP has
        nothing there, the path without its first component, as an index
        relative to its package has it (``webapp/src/auth.ts`` as
        ``src/auth.ts``)."""
        bucket = self.references.get((source_file, line))
        if not bucket and "/" in source_file:
            path = source_file.split("/", 1)[1]
            return path, self.references.get((path, line)) or []
        return source_file, bucket or []

    @staticmethod
    def _simple_name(ref_name: str) -> str:
        """Last segment of a qualified reference name."""
        return ref_name.split("::")[-1].split(".")[-1]

    def _fuzzy_resolve(self, ref_name: str) -> str | None:
        """Attempt fuzzy resolution by name.

        Args:
            ref_name: Reference name to look up

        Returns:
            Descry node ID if found, None otherwise
        """
        # Extract simple name from qualified references
        # e.g., "Server::start" -> "start"
        simple_name = self._simple_name(ref_name)

        if simple_name in self.name_to_symbols:
            defined = {
                self._to_node_id(sym_id, next(iter(self.definitions[sym_id])))
                for sym_id in self.name_to_symbols[simple_name]
                if sym_id in self.definitions
            }
            if "::" in ref_name:
                # Those of the type the name is qualified with, if any is.
                ending = "::" + "::".join(ref_name.split("::")[-2:])
                typed = {node_id for node_id in defined if node_id.endswith(ending)}
                defined = typed or defined
            # The one definition left, where there is one.
            if len(defined) == 1:
                return defined.pop()

        return None

    def _to_node_id(self, symbol_id: str, file_path: str) -> str:
        """Convert SCIP symbol to descry node ID.

        SCIP symbol formats (first token = scheme, third token = package):
        - Rust: "rust-analyzer cargo backend 0.1.0 state/AppState#new()."
        - TypeScript: "scip-typescript npm webapp 0.1.0 src/lib/api/`client.ts`/getAuthToken()."
        - Java/scip-java: "semanticdb maven maven/org.apache.kafka/kafka-clients 3.6 org/apache/kafka/common/Uuid#randomUuid()."

        Descry node ID format:
        "FILE:backend/src/state.rs::AppState::new"

        The key transformations:
        1. Prepend package name to file path when scip emits module-relative
           paths (Rust, TypeScript): "src/state.rs" -> "backend/src/state.rs".
           Skip the prepend when the package token looks like a Maven/workspace
           coordinate (contains "/"): scip-java emits workspace-relative paths
           already, so prepending the maven coord produces a non-matching
           node id like "FILE:maven/org.apache.kafka/kafka-clients/clients/..."
        2. Extract only type/method names, not module paths
           (state/AppState#new -> AppState::new)

        Args:
            symbol_id: SCIP symbol identifier
            file_path: Path to the file where the symbol is defined

        Returns:
            Descry-style node ID
        """
        parts = symbol_id.split()
        if len(parts) < 4:
            return f"FILE:{file_path}"

        # Extract crate/package name (e.g., "backend", "mydb", "webapp")
        package_name = parts[2]

        # Build full file path with package prefix. Skip the prepend when:
        # - the package token contains a "/" (Maven / workspace coord),
        # - the package token is "." (scip-dotnet's "no package name"
        #   sentinel — emits the .scip with workspace-relative paths).
        # Both classes of indexer emit file_path already workspace-relative,
        # so prepending would produce a mismatched node id. So would
        # prepending to a path made root-relative through `project_dirs`.
        if (
            "/" in package_name
            or package_name == "."
            or file_path in self._rooted_files
        ):
            full_path = file_path
        elif not file_path.startswith(package_name + "/"):
            full_path = f"{package_name}/{file_path}"
        else:
            full_path = file_path

        name_parts = self._name_parts(parts)
        if name_parts:
            return f"FILE:{full_path}::{'::'.join(name_parts)}"
        return f"FILE:{full_path}"

    def get_stats(self) -> dict:
        """The number of symbols the index defines, and of their names."""
        return {
            "definitions": len(self.definitions),
            "unique_names": len(self.name_to_symbols),
        }
