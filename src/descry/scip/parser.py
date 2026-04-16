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

        # (file_path, line) -> symbol_id
        self.references: Dict[Tuple[str, int], str] = {}

        # symbol_id -> parsed symbol metadata
        self.symbols: Dict[str, dict] = {}

        # Simple name -> list of symbol_ids (for fuzzy matching)
        self.name_to_symbols: Dict[str, List[str]] = {}

        # Resolution statistics by language
        self._resolution_stats: Dict[str, Dict[str, int]] = {
            "rust": {"attempted": 0, "resolved": 0},
            "typescript": {"attempted": 0, "resolved": 0},
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
            try:
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

                # Store all occurrences for reverse lookup
                self.references[(file_path, line)] = symbol_id

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

        # Use language-specific parsing
        if scheme == "scip-typescript":
            name_parts = self._parse_typescript_descriptors(descriptors)
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
        # Determine language for statistics tracking
        lang = (
            "typescript"
            if source_file.endswith((".ts", ".tsx", ".js", ".jsx", ".svelte"))
            else "rust"
        )
        if lang in self._resolution_stats:
            self._resolution_stats[lang]["attempted"] += 1

        # Strategy 1: Try exact location lookup with full path
        symbol_id = self.references.get((source_file, line))

        # Strategy 1b: Try with package-relative path
        # Descry uses: webapp/src/lib/stores/auth.ts
        # SCIP indexes:   src/lib/stores/auth.ts
        if not symbol_id and "/" in source_file:
            # Try stripping the first path component (package name)
            parts = source_file.split("/", 1)
            if len(parts) == 2:
                relative_path = parts[1]
                symbol_id = self.references.get((relative_path, line))

        if symbol_id and symbol_id in self.definitions:
            def_file, def_line = self.definitions[symbol_id]
            if lang in self._resolution_stats:
                self._resolution_stats[lang]["resolved"] += 1
            return self._to_node_id(symbol_id, def_file)

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

        SCIP symbol formats:
        - Rust: "rust-analyzer cargo backend 0.1.0 state/AppState#new()."
        - TypeScript: "scip-typescript npm webapp 0.1.0 src/lib/api/`client.ts`/getAuthToken()."

        Descry node ID format:
        "FILE:backend/src/state.rs::AppState::new"

        The key transformations:
        1. Prepend crate/package name to file path (src/state.rs -> backend/src/state.rs)
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

        # Build full file path with package prefix
        # SCIP gives: src/state.rs, descry expects: backend/src/state.rs
        if not file_path.startswith(package_name + "/"):
            full_path = f"{package_name}/{file_path}"
        else:
            full_path = file_path

        # Extract descriptors (everything after scheme, manager, package, version)
        descriptors = " ".join(parts[4:]) if len(parts) > 4 else ""

        # Parse descriptors using language-specific parser
        if scheme == "scip-typescript":
            name_parts = self._parse_typescript_descriptors(descriptors)
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

    def _parse_typescript_descriptors(self, descriptors: str) -> List[str]:
        """Parse TypeScript SCIP descriptors into name components for descry node IDs.

        TypeScript SCIP descriptors use backticks to wrap filenames:
        - src/lib/api/`client.ts`/getAuthToken(). -> ["getAuthToken"]
        - src/lib/stores/`users.ts`/UsersStore#fetchUsers(). -> ["UsersStore", "fetchUsers"]
        - src/lib/`stores`/`auth.ts`/AuthStore#login(). -> ["AuthStore", "login"]

        The key insight is that everything before and inside backtick segments
        is file path information (which we already have from file_path), so we
        need to extract the symbol path after the LAST backtick segment.

        Suffix meanings (same as Rust):
        - # for types (class, interface, type alias)
        - . for terms/exports
        - () for methods/functions
        - [] for type parameters (SKIP)

        Args:
            descriptors: Raw descriptor string from SCIP symbol

        Returns:
            List of name components (types and methods only)
        """
        names = []

        # Find the last backtick segment to skip past file path components
        # Pattern: `filename.ts` or `dirname`
        last_backtick_end = -1
        backtick_positions = []
        i = 0
        while i < len(descriptors):
            if descriptors[i] == "`":
                start = i
                i += 1
                # Find closing backtick
                while i < len(descriptors) and descriptors[i] != "`":
                    i += 1
                if i < len(descriptors):
                    backtick_positions.append((start, i))
                    last_backtick_end = i
            i += 1

        # Extract the symbol portion after the last backtick segment
        if last_backtick_end >= 0 and last_backtick_end < len(descriptors) - 1:
            symbol_portion = descriptors[last_backtick_end + 1 :]
        else:
            # No backticks found, use entire string (fallback)
            symbol_portion = descriptors

        # Strip leading path separator if present
        symbol_portion = symbol_portion.lstrip("/")

        # Pattern to match TypeScript symbol descriptors with their suffixes
        # Handles: name#, name., name(), name[T]
        # Group 1: name, Group 2: suffix
        pattern = r"([a-zA-Z_$][a-zA-Z0-9_$]*)(\([^)]*\)|[#./\[\]])?"

        for match in re.finditer(pattern, symbol_portion):
            name = match.group(1)
            suffix = match.group(2) or ""

            if not name:
                continue

            # Skip path separators - these indicate nested paths, not symbol names
            if suffix == "/":
                continue

            # Skip type parameters like [T] or [T, U]
            if suffix.startswith("["):
                continue

            # Skip common TypeScript keywords that aren't real symbols
            if name in (
                "export",
                "default",
                "async",
                "function",
                "class",
                "interface",
                "type",
                "const",
                "let",
                "var",
            ):
                continue

            # Include types (#), terms (.), and methods/functions (())
            names.append(name)

        return names

    def get_definition_location(self, symbol_id: str) -> Optional[Tuple[str, int]]:
        """Get the definition location for a symbol.

        Args:
            symbol_id: SCIP symbol identifier

        Returns:
            Tuple of (file_path, line_number) or None if not found
        """
        return self.definitions.get(symbol_id)

    def get_symbol_info(self, symbol_id: str) -> Optional[dict]:
        """Get metadata for a symbol.

        Args:
            symbol_id: SCIP symbol identifier

        Returns:
            Dictionary with kind, display_name, documentation, or None
        """
        return self.symbols.get(symbol_id)

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
