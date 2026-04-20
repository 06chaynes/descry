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
from typing import TYPE_CHECKING

import descry.scip.adapters  # noqa: F401 — side-effect: populate ADAPTERS registry
from descry.scip.adapter import (
    ADAPTERS,
    adapter_for_extension,
    adapter_for_scheme,
)

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

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

    def __init__(self, scip_files: List[Path]):
        """Initialize the index from SCIP files.

        Args:
            scip_files: List of paths to .scip files to load
        """
        # symbol_id -> (file_path, line_number)
        self.definitions: Dict[str, Tuple[str, int]] = {}

        # (file_path, line) -> list of symbol_ids (may be multiple if several
        # crates contribute the same relative file). At resolve time we prefer
        # a candidate whose extracted name matches ref_name, then fall back to
        # the first.
        self.references: Dict[Tuple[str, int], List[str]] = {}

        # symbol_id -> parsed symbol metadata
        self.symbols: Dict[str, dict] = {}

        # Simple name -> list of symbol_ids (for fuzzy matching)
        self.name_to_symbols: Dict[str, List[str]] = {}

        # Resolution statistics by language, driven by the adapter registry
        # so new SCIP languages automatically get a stats bucket.
        self._resolution_stats: Dict[str, Dict[str, int]] = {
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
            from descry.scip import pb2 as scip_pb2
        except ImportError:
            logger.warning("SCIP: protobuf bindings not available")
            return

        try:
            index = scip_pb2.Index()
            index.ParseFromString(scip_file.read_bytes())
        except Exception as e:
            logger.warning(f"SCIP: Failed to parse {scip_file}: {e}")
            return

        for doc in index.documents:
            file_path = doc.relative_path

            # Process occurrences (references and definitions)
            for occ in doc.occurrences:
                symbol_id = occ.symbol
                if not symbol_id:
                    continue

                # Extract line number from range (range is [startLine, startChar, ...])
                line = occ.range[0] if occ.range else 0

                # Check if this is a definition (symbol_roles & 0x1)
                if occ.symbol_roles & SYMBOL_ROLE_DEFINITION:
                    self.definitions[symbol_id] = (file_path, line)

                # Store all occurrences for reverse lookup. Append to a list
                # so multi-crate workspaces that share relative filenames
                # don't silently overwrite each other.
                key = (file_path, line)
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

    def _extract_name(self, symbol_id: str) -> Optional[str]:
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

    def resolve(self, ref_name: str, source_file: str, line: int) -> Optional[str]:
        """Resolve a reference to its definition node ID.

        Attempts resolution in order:
        1. Exact location lookup (file + line)
        2. Try with package-relative path (for TypeScript: webapp/src/... -> src/...)
        3. Symbol name matching

        Args:
            ref_name: The reference name to resolve (e.g., "validate_token")
            source_file: Path to the file containing the reference
            line: Line number of the reference

        Returns:
            Descry node ID if resolved, None otherwise
        """
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

        # Strategy 1: Try exact location lookup with full path
        candidates = self.references.get((source_file, line)) or []

        # Strategy 1b: Try with package-relative path
        # Descry uses: webapp/src/lib/stores/auth.ts
        # SCIP indexes:   src/lib/stores/auth.ts
        if not candidates and "/" in source_file:
            # Try stripping the first path component (package name)
            parts = source_file.split("/", 1)
            if len(parts) == 2:
                relative_path = parts[1]
                candidates = self.references.get((relative_path, line)) or []

        if candidates:
            # Require an extracted-name match on the candidate — the
            # (file, line) tuple can contain multiple occurrences (type
            # prefix, method call, closing punctuation), and the
            # no-name-match fallback used to pick the first candidate in
            # `definitions`, which routinely yielded wrong targets that
            # later got rejected by the cross-crate name check in
            # generate.py without ever trying Strategy 2. Require name
            # match here so non-matching line lookups fall through to
            # Strategy 2's fuzzy resolve.
            chosen = None
            for cid in candidates:
                if cid in self.definitions and self._extract_name(cid) == ref_name:
                    chosen = cid
                    break
            if chosen is not None:
                def_file, def_line = self.definitions[chosen]
                if lang in self._resolution_stats:
                    self._resolution_stats[lang]["resolved"] += 1
                return self._to_node_id(chosen, def_file)

        # Strategy 2: Try symbol name matching
        result = self._fuzzy_resolve(ref_name)
        if result and lang in self._resolution_stats:
            self._resolution_stats[lang]["resolved"] += 1
        return result

    def _fuzzy_resolve(self, ref_name: str) -> Optional[str]:
        """Attempt fuzzy resolution by name.

        Args:
            ref_name: Reference name to look up

        Returns:
            Descry node ID if found, None otherwise
        """
        # Extract simple name from qualified references
        # e.g., "ThoraxServer::start" -> "start"
        simple_name = ref_name.split("::")[-1].split(".")[-1]

        if simple_name in self.name_to_symbols:
            symbol_ids = self.name_to_symbols[simple_name]
            # Prefer symbols that match more of the qualified name
            if "::" in ref_name:
                type_name = ref_name.split("::")[-2] if "::" in ref_name else ""
                for sym_id in symbol_ids:
                    if type_name and type_name.lower() in sym_id.lower():
                        if sym_id in self.definitions:
                            def_file, _ = self.definitions[sym_id]
                            return self._to_node_id(sym_id, def_file)

            # Fall back to first match
            for sym_id in symbol_ids:
                if sym_id in self.definitions:
                    def_file, _ = self.definitions[sym_id]
                    return self._to_node_id(sym_id, def_file)

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
        # so prepending would produce a mismatched node id.
        if "/" in package_name or package_name == ".":
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

    def _parse_descriptors(self, descriptors: str) -> List[str]:
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

            # Skip 'tests' module in test paths
            if name == "tests" and suffix == "/":
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
