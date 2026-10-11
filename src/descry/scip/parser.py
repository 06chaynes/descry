# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "protobuf>=4.0.0",
# ]
# ///
"""Parse SCIP index files and provide symbol resolution.

SCIP (Source Code Index Protocol) provides type-aware symbol information
from rust-analyzer that enables more accurate call resolution than
regex-based approaches.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import descry.scip.adapters  # noqa: F401 — side-effect: populate ADAPTERS registry
from descry.scip.adapter import (
    ADAPTERS,
    adapter_for_extension,
    adapter_for_scheme,
)

logger = logging.getLogger(__name__)

# Symbol role bitflags (from SCIP spec)
SYMBOL_ROLE_DEFINITION = 0x1
SYMBOL_ROLE_IMPORT = 0x2
SYMBOL_ROLE_WRITE_ACCESS = 0x4
SYMBOL_ROLE_READ_ACCESS = 0x8


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

        # symbol_id -> (file_path, line_number)
        self.definitions: dict[str, tuple[str, int]] = {}

        # (file_path, line) -> list of symbol_ids (may be multiple if several
        # crates contribute the same relative file). At resolve time we prefer
        # a candidate whose extracted name matches ref_name, then fall back to
        # the first.
        self.references: dict[tuple[str, int], list[str]] = {}

        # (file_path, line, start column) -> symbol_id of the occurrence there
        self._columns: dict[tuple[str, int, int], str] = {}
        # file_path -> the SCIP position encoding its columns are counted in
        self._position_encodings: dict[str, int] = {}

        # symbol_id -> parsed symbol metadata
        self.symbols: dict[str, dict] = {}

        # Simple name -> list of symbol_ids (for fuzzy matching)
        self.name_to_symbols: dict[str, list[str]] = {}

        # scheme -> test of whether an outside symbol's call leaves the project
        self._leaving: dict[str, object] = {}

        # Resolution statistics by language, driven by the adapter registry
        # so new SCIP languages automatically get a stats bucket.
        self._resolution_stats: dict[str, dict[str, int]] = {
            adapter.name: {"attempted": 0, "resolved": 0}
            for adapter in ADAPTERS.values()
        }

        for scip_file in scip_files:
            if scip_file.exists():
                self._load_scip(scip_file)

        logger.info(
            f"SCIP: Loaded {len(self.definitions)} definitions, "
            f"{len(self.references)} references from {len(scip_files)} files"
        )

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
            index.ParseFromString(scip_file.read_bytes())
        except (OSError, DecodeError) as e:
            logger.warning(f"SCIP: Failed to parse {scip_file}: {e}")
            return

        project_dir = self._project_dirs.get(scip_file)
        if project_dir is None:
            project_dir = self._stated_project_dir(index.metadata.project_root)
        for doc in index.documents:
            file_path = doc.relative_path
            if project_dir is not None:
                file_path = f"{project_dir}/{file_path}" if project_dir else file_path
                self._rooted_files.add(file_path)
            self._position_encodings[file_path] = doc.position_encoding

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
                    self.definitions[symbol_id] = (file_path, line)

                # Store all occurrences for reverse lookup. Append to a list
                # so multi-crate workspaces that share relative filenames
                # don't silently overwrite each other.
                key = (file_path, line)
                if len(occ.range) > 1:
                    self._columns[(*key, occ.range[1])] = symbol_id
                bucket = self.references.get(key)
                if bucket is None:
                    self.references[key] = [symbol_id]
                elif symbol_id not in bucket:
                    bucket.append(symbol_id)

            # Process symbol information
            for sym in doc.symbols:
                self.symbols[sym.symbol] = {
                    "kind": sym.kind,
                    "display_name": sym.display_name,
                    "documentation": list(sym.documentation),
                }

                # Build name-to-symbol index for fuzzy matching
                name = self._extract_name(sym.symbol)
                if name:
                    if name not in self.name_to_symbols:
                        self.name_to_symbols[name] = []
                    self.name_to_symbols[name].append(sym.symbol)

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

        Returns the last identifier (e.g., "run_migrations").
        """
        if not symbol_id:
            return None

        # Local symbols have format "local <id>"
        if symbol_id.startswith("local "):
            return None

        # Split on spaces to get parts
        parts = symbol_id.split()
        if len(parts) < 4:
            return None

        scheme = parts[0]

        # The descriptors are after scheme, manager, package, version
        # Example: ["rust-analyzer", "cargo", "mydb", "0.1.0", "database/migrations/run_migrations()."]
        descriptors = " ".join(parts[4:]) if len(parts) > 4 else ""
        if not descriptors:
            return None

        # Delegate descriptor parsing to the adapter that owns this scheme.
        # Falls back to the Rust-style parser if no adapter claims the
        # scheme (keeps behavior safe for unknown/future schemes).
        adapter = adapter_for_scheme(scheme)
        if adapter is not None:
            name_parts = adapter.parse_descriptors(descriptors)
        else:
            name_parts = self._parse_descriptors(descriptors)

        return name_parts[-1] if name_parts else None

    def resolve(self, ref_name: str, source_file: str, line: int) -> str | None:
        """Resolve a reference to its definition node ID.

        Attempts resolution in order:
        1. Exact location lookup (file + line)
        2. Try with package-relative path (for TypeScript: webapp/src/... -> src/...)
        3. Symbol name matching

        Args:
            ref_name: The reference name to resolve (e.g., "validate_token")
            source_file: Path to the file containing the reference
            line: 1-based line number of the reference (a descry edge `lineno`)

        Returns:
            Descry node ID if resolved, None otherwise
        """
        lang = self._count_attempt(source_file)
        chosen = self._symbol_at(ref_name, source_file, line)
        if chosen is not None:
            def_file, _ = self.definitions[chosen]
            if lang in self._resolution_stats:
                self._resolution_stats[lang]["resolved"] += 1
            return self._to_node_id(chosen, def_file)

        # Strategy 2: Try symbol name matching
        result = self._fuzzy_resolve(ref_name)
        if result and lang in self._resolution_stats:
            self._resolution_stats[lang]["resolved"] += 1
        return result

    def definition_at(
        self, ref_name: str, source_file: str, line: int, column: int | None = None
    ) -> tuple[str, str, int] | None:
        """Find the definition of the symbol SCIP recorded at a reference.

        Unlike ``resolve`` this never falls back to matching by name. Where
        the line references several symbols of ``ref_name``'s name, the one
        whose occurrence starts at ``column`` (0-based) is taken. ``column``
        is one number or ``(UTF-8, UTF-16, UTF-32)`` code-unit offsets.

        Returns:
            ``(node_id, file, line)`` of the definition, the line 1-based,
            or None when no occurrence at ``source_file``:``line`` carries
            ``ref_name``'s name, or ``column`` is given and does not tell
            several such occurrences apart
        """
        lang = self._count_attempt(source_file)
        chosen = self._symbol_at(ref_name, source_file, line, column)
        if chosen is None:
            return None
        if lang in self._resolution_stats:
            self._resolution_stats[lang]["resolved"] += 1
        def_file, def_line = self.definitions[chosen]
        return self._to_node_id(chosen, def_file), def_file, def_line

    def _stated_project_dir(self, project_root: str) -> str | None:
        """The directory, relative to the repository root, that an index
        names in its metadata as the base of its document paths."""
        if self._root is None or not project_root.startswith("file://"):
            return None
        stated = Path(unquote(urlsplit(project_root).path))
        try:
            relative = stated.relative_to(self._root).as_posix()
        except ValueError:
            return None
        return "" if relative == "." else relative

    def _count_attempt(self, source_file: str) -> str:
        """Count a resolution attempt for ``source_file``'s language and
        return that language."""
        # Determine language for stats tracking via the adapter registry.
        # Extension → adapter.name; unknown extensions fall back to "rust" to
        # preserve the pre-refactor behavior (rust-analyzer is the broadest
        # resolver in practice).
        ext_adapter = None
        dot_idx = source_file.rfind(".")
        if dot_idx >= 0:
            ext_adapter = adapter_for_extension(source_file[dot_idx:])
        lang = ext_adapter.name if ext_adapter is not None else "rust"
        if lang in self._resolution_stats:
            self._resolution_stats[lang]["attempted"] += 1
        return lang

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
                build(defined, self._root) if build else (lambda _: False)
            )
        return self._leaving[scheme](symbol)

    def _symbol_at(
        self, ref_name: str, source_file: str, line: int, column: int | None = None
    ) -> str | None:
        """The defined symbol referenced at ``source_file``:``line`` under
        ``ref_name``'s last segment, if any."""
        path, named = self._named_at(ref_name, source_file, line)
        named = [cid for cid in named if cid in self.definitions]
        if column is not None and len(named) > 1:
            # A column given in several units is read in the document's;
            # where the index does not say which, any of them may match.
            if isinstance(column, int):
                columns = {column}
            else:
                unit = self._position_encodings.get(path)
                columns = {column[unit - 1]} if unit else set(column)
            exact = {self._columns.get((path, line, c)) for c in columns}
            exact &= set(named)
            return exact.pop() if len(exact) == 1 else None
        return named[0] if named else None

    def _named_at(self, ref_name: str, source_file: str, line: int):
        """The path SCIP has ``source_file`` under, and the symbols it
        recorded on ``line`` of it under ``ref_name``'s last segment."""
        # Strategy 1: Try exact location lookup with full path
        candidates = self.references.get((source_file, line)) or []
        path = source_file

        # Strategy 1b: Try with package-relative path
        # Descry uses: webapp/src/lib/stores/auth.ts
        # SCIP indexes:   src/lib/stores/auth.ts
        if not candidates and "/" in source_file:
            # Try stripping the first path component (package name)
            parts = source_file.split("/", 1)
            if len(parts) == 2:
                path = parts[1]
                candidates = self.references.get((path, line)) or []

        # Only candidates whose extracted name matches count: one
        # (file, line) can hold several occurrences (type prefix, method
        # call, closing punctuation). SCIP names are unqualified, so
        # compare against the last segment of ref_name.
        simple_name = self._simple_name(ref_name)
        return path, [c for c in candidates if self._extract_name(c) == simple_name]

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
                self._to_node_id(sym_id, self.definitions[sym_id][0])
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

        scheme = parts[0]

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

        # Extract descriptors (everything after scheme, manager, package, version)
        descriptors = " ".join(parts[4:]) if len(parts) > 4 else ""

        # Delegate descriptor parsing to the adapter that owns this scheme.
        adapter = adapter_for_scheme(scheme)
        if adapter is not None:
            name_parts = adapter.parse_descriptors(descriptors)
        else:
            name_parts = self._parse_descriptors(descriptors)

        if name_parts:
            return f"FILE:{full_path}::{'::'.join(name_parts)}"
        return f"FILE:{full_path}"

    def _parse_descriptors(self, descriptors: str) -> list[str]:
        """Parse SCIP descriptors into name components for descry node IDs.

        SCIP descriptors use suffixes to indicate type:
        - / for namespaces/modules (SKIP these - they're in file path)
        - # for types (struct, enum, trait) (INCLUDE)
        - . for terms (constants, statics) (INCLUDE)
        - () for methods (INCLUDE)
        - [] for type parameters (SKIP)
        - [impl] for impl blocks (extract struct name)

        Example:
        - "state/AppState#new()." -> ["AppState", "new"]
        - "database/migrations/run_migrations()." -> ["run_migrations"]
        - "impl#[AppState]new()." -> ["AppState", "new"]

        Args:
            descriptors: Raw descriptor string from SCIP symbol

        Returns:
            List of name components (types and methods only)
        """
        names = []

        # Pattern to match descriptors with their suffixes
        # Group 1: name, Group 2: suffix
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)(\([^)]*\)|[#./\[\]])?"

        for match in re.finditer(pattern, descriptors):
            name = match.group(1)
            suffix = match.group(2) or ""

            if not name:
                continue

            # Skip modules/namespaces (suffix /)
            if suffix == "/":
                continue

            # Skip impl keyword but include the type name in brackets
            if name == "impl":
                continue

            # Include types (#), terms (.), and methods (())
            names.append(name)

        return names

    def get_stats(self) -> dict:
        """Get statistics about the loaded index.

        Returns:
            Dictionary with counts of definitions, references, resolution rates, etc.
        """
        # Calculate resolution rates by language
        resolution_rates = {}
        for lang, stats in self._resolution_stats.items():
            attempted = stats["attempted"]
            resolved = stats["resolved"]
            if attempted > 0:
                rate = 100 * resolved / attempted
                resolution_rates[lang] = {
                    "attempted": attempted,
                    "resolved": resolved,
                    "rate_percent": round(rate, 1),
                }

        return {
            "definitions": len(self.definitions),
            "references": len(self.references),
            "symbols": len(self.symbols),
            "unique_names": len(self.name_to_symbols),
            "resolution_by_language": resolution_rates,
        }
