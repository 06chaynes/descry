import logging
import math
import os
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from descry._graph import CALL_RELATIONS, load_graph_with_schema

logger = logging.getLogger(__name__)

# --- Module-level Constants ---
# These provide consistent limits across all operations

# Depth and recursion limits
MAX_DEPTH = 3  # Maximum recursion depth for callees/flow
MAX_NODES_PER_OPERATION = 100  # Safety limit for recursive operations
MAX_CHILDREN_PER_LEVEL = 10  # Branching limit per level

# Token budgets and thresholds
DEFAULT_TOKEN_BUDGET = 2000  # Default token budget for expanded callees
CALLEE_INLINE_THRESHOLD = 150  # Inline callees smaller than this (was 100)
MAX_CALLERS_SHOWN = 15  # Maximum callers to show in context

# Timeout (milliseconds)
TIMEOUT_MS = 4000  # Wall-clock timeout for recursive operations

# Hard ceiling on find_call_path_result's max_depth argument. The node budget and
# the timeout bound the search itself.
MAX_PATH_DEPTH = 50

# Short spellings accepted by the `lang` search filter.
_LANG_ALIASES = {
    "ts": "typescript",
    "js": "javascript",
    "py": "python",
    "rs": "rust",
    "rb": "ruby",
    "cs": "csharp",
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation: 4 chars ≈ 1 token)."""
    return len(text) // 4


# --- File Content LRU Cache ---
# Reduces I/O from ~150 to ~30 reads for deep context operations


@lru_cache(maxsize=128)
def _read_file_cached_inner(
    file_path: str,
    mtime_ns: int,
) -> tuple[str, ...]:
    """Inner cache keyed on (path, mtime_ns) so edits invalidate the entry.

    The mtime is read once by the outer wrapper and threaded through as a
    cache key — callers get tuple-of-lines without worrying about staleness.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return tuple(f.readlines())
    except OSError:
        return ()


def _read_file_cached(file_path: str) -> tuple[str, ...]:
    """Read file content with mtime-aware LRU caching.

    Returns tuple of lines (tuple for hashability in cache). Stats the file
    to capture mtime_ns and uses (path, mtime_ns) as the cache key, so an
    edit on disk invalidates the cached content automatically.
    """
    try:
        mtime_ns = os.stat(file_path).st_mtime_ns
    except OSError:
        return ()
    return _read_file_cached_inner(file_path, mtime_ns)


def source_path(root, rel_path: str, max_bytes: int = 0) -> str:
    """Where a graph-relative source file lives under `root`.

    Empty unless it is a regular file inside `root`: a graph may be
    hand-edited, so its paths are not trusted to stay in the project.
    Empty too for a file larger than `max_bytes`, when that is not 0.
    """
    root = Path(root).resolve()
    try:
        path = (root / rel_path).resolve()
        if (
            path.is_relative_to(root)
            and path.is_file()
            and not (max_bytes and path.stat().st_size > max_bytes)
        ):
            return str(path)
    except (OSError, RuntimeError, ValueError):
        pass
    return ""


def _clear_file_cache():
    """Clear the file content cache (useful after indexing)."""
    _read_file_cached_inner.cache_clear()


def _get_syntax_lang(file_path: str, lang_map: dict[str, str] | None = None) -> str:
    """Determine syntax highlighting language from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if lang_map:
        return lang_map.get(ext, "")
    from descry.handlers import _DEFAULT_SYNTAX_LANG_MAP

    return _DEFAULT_SYNTAX_LANG_MAP.get(ext, "")


def _normalize_name(name: str) -> str:
    """Normalize a name to snake_case for case-insensitive matching.

    Handles:
    - camelCase -> camel_case
    - PascalCase -> pascal_case
    - already_snake_case -> already_snake_case
    - SCREAMING_CASE -> screaming_case

    This allows searching for 'getClient' to find 'get_client'.
    """
    # Insert underscore before uppercase letters that follow lowercase
    # e.g., getClient -> get_Client
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Insert underscore before uppercase letters that precede lowercase (for acronyms)
    # e.g., HTTPServer -> HTTP_Server -> http_server
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()


def _get_name_variants(name: str) -> list[str]:
    """Get all naming convention variants of a name.

    Returns a list of possible names in different conventions:
    - Original
    - snake_case normalized
    - With common prefixes/suffixes

    This allows 'getClient' to match 'get_client' and vice versa.
    """
    variants = [name]

    # Add normalized snake_case version
    normalized = _normalize_name(name)
    if normalized != name.lower():
        variants.append(normalized)

    # Also try the lowercased original (for case-insensitive exact match)
    if name.lower() != name:
        variants.append(name.lower())

    return list(set(variants))  # Deduplicate


def _ends_at_segment(node_id: str, name: str) -> bool:
    """Whether ``node_id`` ends with ``name`` where one of its segments
    starts: `Foo::m` ends `a.py::Foo::m` and not `a.py::BigFoo::m`."""
    before = node_id[: -len(name)][-1:]
    return node_id.endswith(name) and not (before.isalnum() or before == "_")


def _clean_ref_name(ref_name: str, max_len: int = 60) -> str:
    """Clean up an unresolved reference name for display.

    - Takes only the first line (removes multi-line expressions)
    - Extracts the meaningful part (function/method name)
    - Truncates if too long
    """
    # Take first line only
    first_line = ref_name.split("\n")[0].strip()

    # If it's a qualified call like Type::method(args), extract Type::method
    if "::" in first_line:
        # Match Type::method or Type::Variant pattern
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)", first_line
        )
        if match:
            first_line = match.group(1)

    # If it's a method chain like foo.bar or foo.bar().baz, extract the last method
    elif "." in first_line:
        # Split by dots and extract the last meaningful method name
        # Handles both: foo.bar.method and foo.bar().method()
        parts = first_line.split(".")
        for part in reversed(parts):
            # Extract method name (strip parentheses and args)
            method_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)", part)
            if method_match:
                first_line = method_match.group(1)
                break

    # Truncate if still too long
    if len(first_line) > max_len:
        first_line = first_line[: max_len - 3] + "..."

    return first_line


class GraphQuerier:
    def __init__(self, graph_file, config=None):
        self.data = load_graph_with_schema(graph_file)

        # Paths in node IDs are relative to the project root.
        self._project_root = config.project_root if config else None
        self._max_file_bytes = config.max_file_bytes if config else 0

        self.nodes = {n["id"]: n for n in self.data["nodes"]}
        self.edges = self.data["edges"]

        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        # candidate node id -> the unresolved edges that may reach it
        self.candidate_incoming = defaultdict(list)

        for edge in self.edges:
            self.outgoing[edge["source"]].append(edge)
            self.incoming[edge["target"]].append(edge)
            for candidate in edge.get("metadata", {}).get("candidates", ()):
                self.candidate_incoming[candidate].append(edge)

        # Config-driven limits (fall back to module-level constants)
        self._max_depth = config.max_depth if config else MAX_DEPTH
        self._max_nodes = config.max_nodes if config else MAX_NODES_PER_OPERATION
        self._max_children_per_level = (
            config.max_children_per_level if config else MAX_CHILDREN_PER_LEVEL
        )
        self._max_callers_shown = (
            config.max_callers_shown if config else MAX_CALLERS_SHOWN
        )
        self._timeout_ms = config.query_timeout_ms if config else TIMEOUT_MS
        self._test_path_patterns = config.test_path_patterns if config else None
        self._test_file_suffixes = config.test_file_suffixes if config else None
        self._syntax_lang_map = config.syntax_lang_map if config else None

        # Build IDF cache for TF-IDF search
        self._idf_cache = self._build_idf_cache()

        # Lazy filter indices - built on first use
        self._index_by_crate = None
        self._index_by_lang = None
        self._index_by_type = None
        self._index_is_test = None  # Test files and test functions
        self._indices_built = False

    def _disk_path(self, file_path):
        """Where a graph-relative source path lives on disk; see source_path.

        Empty when the querier has no project root to resolve it against,
        and for a file over `[code_files] max_file_mb`.
        """
        if self._project_root:
            return source_path(self._project_root, file_path, self._max_file_bytes)
        return ""

    def get_source_segment(self, file_path, start_line, end_line):
        """Get source code segment using cached file content."""
        try:
            lines = _read_file_cached(self._disk_path(file_path))
            if not lines:
                return "<Error: file not found or empty>"
            # line numbers are 1-based
            start = max(0, start_line - 1)
            end = min(len(lines), end_line)
            return "".join(lines[start:end])
        except OSError as e:
            return f"<Error reading file: {e}>"

    def is_test(self, node_id):
        """Whether ``node_id`` is test code: in a test file, or a test
        function by its name."""
        self._ensure_filter_indices()
        return node_id in self._index_is_test

    def _ensure_filter_indices(self):
        """Build filter indices lazily on first use.

        Creates indices for:
        - Crate/module filtering (e.g., 'backend', 'webapp')
        - Language filtering (rust, typescript, python, svelte)
        - Type filtering (Function, Class, Method, Constant, File)
        - Test detection (test files and test functions)

        Uses sets of node IDs for fast intersection operations.
        """
        if self._indices_built:
            return

        self._index_by_crate = defaultdict(set)
        self._index_by_lang = defaultdict(set)
        self._index_by_type = defaultdict(set)
        self._index_is_test = set()

        # Test detection patterns
        test_path_patterns = self._test_path_patterns or (
            "/tests/",
            "/test/",
            "/_test/",
            "/spec/",
            "/testing/",
            "/fixtures/",
            "/mocks/",
            "/__tests__/",
        )
        test_file_suffixes = self._test_file_suffixes or (
            "_test.rs",
            ".test.ts",
            ".spec.ts",
            "_test.py",
            ".test.js",
            ".spec.js",
            ".test.tsx",
            ".spec.tsx",
        )
        test_function_pattern = re.compile(r"^(test_|it_|describe_|spec_)")

        for node in self.data["nodes"]:
            node_id = node["id"]
            node_type = node.get("type", "")
            node_name = node.get("metadata", {}).get("name", "")

            # Index by type
            self._index_by_type[node_type].add(node_id)

            # Index by crate and language (extract from file path)
            if node_id.startswith("FILE:"):
                path = node_id.replace("FILE:", "").split("::")[0]

                # Extract crate (first directory component)
                parts = path.split("/")
                if parts:
                    crate = parts[0]
                    self._index_by_crate[crate].add(node_id)

                # Extract language from extension
                lang = _get_syntax_lang(path, self._syntax_lang_map)
                if lang:
                    self._index_by_lang[lang].add(node_id)

                # Test detection: check path patterns and file suffixes
                is_test_file = any(p in path for p in test_path_patterns) or any(
                    path.endswith(s) for s in test_file_suffixes
                )
                if is_test_file:
                    self._index_is_test.add(node_id)

            # Test detection: check function/method names
            if node_type in ("Function", "Method") and test_function_pattern.match(
                node_name
            ):
                self._index_is_test.add(node_id)

        self._indices_built = True
        logger.debug(
            f"Filter indices built: {len(self._index_by_crate)} crates, "
            f"{len(self._index_by_lang)} langs, {len(self._index_by_type)} types, "
            f"{len(self._index_is_test)} test nodes"
        )

    def get_smart_source(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        token_count: int,
        full: bool = False,
        head_lines_only: int | None = None,
    ) -> str:
        """Get source code with smart truncation based on token count.

        Args:
            file_path: Path to the source file
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)
            token_count: Estimated token count for the function
            full: If True, bypass truncation and return complete source
            head_lines_only: If set, show only first N lines (for quick orientation)

        Strategy (when full=False and head_lines_only=None):
        - < 1000 tokens: Show full source (most functions fit here)
        - 1000-1800 tokens: Show first 60 lines + last 15 lines (medium-large function)
        - > 1800 tokens: Show first 50 lines + last 15 lines with clear truncation

        This preserves the signature, docstring, key logic, and return statement.
        Threshold of 1000 tokens (~100 lines) covers most middleware/interceptor patterns.
        """
        try:
            all_lines = _read_file_cached(self._disk_path(file_path))
            if not all_lines:
                return "<Error: file not found or empty>"

            # line numbers are 1-based
            start = max(0, start_line - 1)
            end = min(len(all_lines), end_line)
            source_lines = list(all_lines[start:end])
            total_lines = len(source_lines)

            # head_lines_only: show just the first N lines for quick orientation
            if head_lines_only is not None and head_lines_only > 0:
                if total_lines <= head_lines_only:
                    return "".join(source_lines)
                head = source_lines[:head_lines_only]
                remaining = total_lines - head_lines_only
                return (
                    "".join(head)
                    + f"\n    // ... ({remaining} more lines, use full=true to see all) ...\n"
                )

            # Bypass truncation when full=True
            if full:
                return "".join(source_lines)

            # Most functions: show everything (raised from 800 to 1000)
            if token_count < 1000 or total_lines <= 100:
                return "".join(source_lines)

            # Medium-large functions: show more context
            if token_count < 1800:
                head_lines = 60
                tail_lines = 15
            else:
                # Very large functions: still show substantial context
                head_lines = 50
                tail_lines = 15

            if total_lines <= head_lines + tail_lines + 5:
                # Not worth truncating
                return "".join(source_lines)

            # Build truncated output
            head = source_lines[:head_lines]
            tail = source_lines[-tail_lines:]
            omitted = total_lines - head_lines - tail_lines

            return (
                "".join(head)
                + f"\n    // ... ({omitted} lines omitted, use full=true to see complete source) ...\n\n"
                + "".join(tail)
            )

        except OSError as e:
            return f"<Error reading file: {e}>"

    def get_context_prompt(
        self,
        node_id,
        depth: int = 1,
        max_tokens: int | None = None,
        inline_threshold: int | None = None,
        full: bool = False,
        head_lines: int | None = None,
        expand_callees: bool = False,
        callee_budget: int = 2000,
        brief: bool = False,
        max_output_tokens: int | None = None,
    ):
        """Get comprehensive context for a symbol.

        Args:
            node_id: Full node ID to get context for
            depth: Recursion depth for callee expansion (1-3, default: 1)
            max_tokens: Token budget for expanded callees (default: DEFAULT_TOKEN_BUDGET)
            inline_threshold: Token threshold for inlining callees (default: CALLEE_INLINE_THRESHOLD)
            full: If True, show full source without truncation (default: False)
            head_lines: If set, show only first N lines of source (for quick orientation)
            expand_callees: If True, inline full source of direct callees (default: False)
            callee_budget: Token budget for expanded callees (default: 2000)
            brief: If True, return minimal output (~50 tokens): signature, location, counts only
            max_output_tokens: Hard cap on total output tokens. Applies progressive truncation.
        """
        if max_tokens is None:
            max_tokens = DEFAULT_TOKEN_BUDGET
        if inline_threshold is None:
            inline_threshold = CALLEE_INLINE_THRESHOLD
        depth = self.flow_depth(depth)

        # Resolve node ID with fuzzy matching
        resolved_id, resolve_msg = self._resolve_node_id(node_id)
        if resolved_id is None:
            return f"ERROR: {resolve_msg}"

        node = self.nodes.get(resolved_id)
        if not node:
            return f"ERROR: Node '{node_id}' not found in graph."

        # Use resolved ID from here on
        node_id = resolved_id
        fuzzy_match_note = f"\n\n> **Note:** {resolve_msg}\n" if resolve_msg else ""

        # 1. Identify File
        file_path_rel = node["id"].split("::")[0].replace("FILE:", "")
        if not os.path.exists(self._disk_path(file_path_rel)):
            return f"ERROR: Source file '{file_path_rel}' not found on disk."

        # Determine syntax highlighting language
        source_lang = _get_syntax_lang(file_path_rel, self._syntax_lang_map)

        # 2. Get Source Code with smart truncation (or full if requested)
        meta = node["metadata"]
        start = meta.get("lineno", 1)
        end = meta.get("end_lineno")
        token_count = meta.get("token_count", 0)

        # BRIEF MODE: Return minimal output (~50 tokens) for quick verification
        if brief:
            signature = meta.get("signature", f"{meta.get('name', '?')}(...)")
            docstring = meta.get("docstring", "")
            doc_preview = ""
            if docstring:
                doc_lines = [
                    line.strip() for line in docstring.split("\n") if line.strip()
                ]
                if doc_lines:
                    doc_preview = doc_lines[0][:120]

            # Callers whose calls are resolved to it, then those that may be
            possible = len(self.get_ambiguous_callers(node_id))
            certain = len(self.get_callers(node_id)) - possible
            callers_count = f"{certain}{self.possible(possible)}"
            certain, possible, unresolved = self.kinds(
                self.get_callees(node_id), self.get_ambiguous_callees(node_id)
            )
            callees_count = f"{certain}{self.possible(possible, unresolved)}"

            output = [
                f"### Brief: `{node_id}`",
                f"**Location:** {file_path_rel}:{start} | **Tokens:** {token_count} | **Callers:** {callers_count} | **Callees:** {callees_count}",
                "",
                f"```{source_lang}",
                signature,
                "```",
            ]
            if doc_preview:
                output.append("")
                output.append(doc_preview)
            if fuzzy_match_note:
                output.append(fuzzy_match_note)

            return "\n".join(output)

        # If end_lineno not set, estimate from token count or use generous default
        if end is None:
            if token_count > 0:
                # Estimate ~10 tokens per line
                estimated_lines = max(20, token_count // 10)
                end = start + estimated_lines
            else:
                # Generous default for unknown sizes
                end = start + 100

        source_code = self.get_smart_source(
            file_path_rel,
            start,
            end,
            token_count,
            full=full,
            head_lines_only=head_lines,
        )

        # 2b. For Classes/Structs, include impl block methods
        impl_methods = []
        if node["type"] == "Class":
            # Find all methods that belong to this struct (children in the graph)
            for child_id, child_node in self.nodes.items():
                if child_node["type"] == "Method" and child_id.startswith(
                    node_id + "::"
                ):
                    child_meta = child_node["metadata"]
                    child_sig = child_meta.get("signature", child_meta.get("name", ""))
                    child_doc = child_meta.get("docstring", "").split("\n")[0][:80]
                    child_tokens = child_meta.get("token_count", 0)
                    impl_methods.append(
                        {
                            "name": child_meta.get("name", ""),
                            "signature": child_sig,
                            "docstring": child_doc,
                            "tokens": child_tokens,
                            "lineno": child_meta.get("lineno", 0),
                            "id": child_id,
                        }
                    )
            # Sort by line number
            impl_methods.sort(key=lambda m: m["lineno"])

        # 3. Find Dependencies (Callees) with Smart Inlining - deduplicated
        # Use recursive expansion if depth > 1
        callee_info = self._expand_callees(node_id, depth, max_tokens, inline_threshold)

        # 4. Find Usage Examples (Callers) - grouped by file
        usage_examples = self._get_usage_examples(node_id, limit=4)

        # 4b. Get callers summary for explicit caller list
        callers_summary = self._get_callers_summary(node_id)

        # 5. Find Related Tests
        related_tests = self._find_related_tests(file_path_rel, meta.get("name"))

        # 5b. Find Related Configuration nodes in same file
        file_id = f"FILE:{file_path_rel}"
        related_configs = self._find_related_configs(file_id, node_id)

        # 6. Format Output
        output = []
        output.append(f"### Context for `{node['id']}`")
        if fuzzy_match_note:
            output.append(fuzzy_match_note)
        output.append(
            f"**Tokens:** {meta.get('token_count', '?')} | **In-Degree:** {meta.get('in_degree', 0)}\n"
        )
        output.append("#### Source Code:")
        output.append(f"```{source_lang}\n{source_code}\n```\n")

        # For Classes/Structs, show impl methods
        if impl_methods:
            output.append(f"#### Implementation Methods ({len(impl_methods)} methods):")
            for method in impl_methods:
                sig = method["signature"]
                doc = method["docstring"]
                tokens = method["tokens"]
                # Show signature and brief doc
                output.append(f"- `{sig}` ({tokens} toks)")
                if doc:
                    output.append(f"  > {doc}")
            output.append("")

        if usage_examples:
            output.append("#### Usage Examples:")
            # Group examples by file for better readability
            by_file = defaultdict(list)
            for ex in usage_examples:
                by_file[ex["file"]].append(ex)

            for file_path, examples in by_file.items():
                ex_lang = _get_syntax_lang(file_path, self._syntax_lang_map)
                output.append(f"\n**From `{file_path}`:**")
                for ex in examples:
                    output.append(
                        f"Line {ex['lineno']}:{self.one_of(ex['candidates'])}"
                    )
                    output.append(f"```{ex_lang}\n{ex['code']}\n```")
            output.append("")

        # Add explicit callers section
        if callers_summary:
            callers, ambiguous = self.ordered_callers(node_id)
            output.append(f"#### Callers ({len(callers_summary)} shown):")
            for caller in callers_summary:
                loc = (
                    f"{caller['file']}:{caller['lineno']}"
                    if caller["lineno"]
                    else caller["file"]
                )
                output.append(
                    f"- `{caller['name']}` in {loc} ({caller['tokens']} toks)"
                    f"{self.one_of(caller['candidates'])}"
                )
            if len(callers) > len(callers_summary):
                shown = {caller["id"] for caller in callers_summary}
                more = [caller for caller in callers if caller not in shown]
                possible = sum(caller in ambiguous for caller in more)
                output.append(
                    f"  *({self.more_callers(len(more) - possible, possible)})*"
                )
            output.append("")

        if callee_info:
            output.append("#### Dependencies (Callees):")
            output.append("\n".join(callee_info))
            output.append("")

        # Expanded callees - inline full source of direct callees
        if expand_callees:
            expanded_callees = self._expand_callees_full(node_id, callee_budget)
            if expanded_callees:
                output.append("#### Expanded Callees (Full Source):")
                for exp in expanded_callees:
                    output.append(
                        f"\n**{exp['name']}** ({exp['file']}:{exp['lineno']}, {exp['tokens']} toks):"
                    )
                    output.append(f"```{exp['lang']}\n{exp['source']}\n```")
                output.append("")

        if related_tests:
            output.append("#### Related Tests:")
            for t in related_tests:
                output.append(f"- `{t}`")
            output.append("")

        if related_configs:
            output.append("#### Related Configuration (same file):")
            for cfg in related_configs:
                cfg_type = f" [{cfg['config_type']}]" if cfg["config_type"] else ""
                output.append(f"- L{cfg['lineno']}: `{cfg['signature']}`{cfg_type}")
                output.append(f"  Node: `{cfg['id']}`")
            output.append("")

        # Apply max_output_tokens progressive truncation if specified
        result = "\n".join(output)
        if (
            max_output_tokens is not None
            and _estimate_tokens(result) > max_output_tokens
        ):
            result = self._truncate_output_progressively(
                result,
                max_output_tokens,
                source_lang,
            )

        return result

    def _find_related_tests(self, source_file, symbol_name):
        """
        Heuristic to find test files/functions related to the source.
        """
        filename, ext = os.path.splitext(os.path.basename(source_file))
        base_name = filename

        possible_test_files = [
            f"test_{base_name}{ext}",
            f"{base_name}_test{ext}",
            f"{base_name}.test{ext}",  # TS
            f"{base_name}.spec{ext}",  # TS
        ]

        found_tests = []

        # 1. Scan all File nodes to find matches
        for node in self.nodes.values():
            if node["type"] == "File":
                fpath = node.get("metadata", {}).get("path", "")
                fbase = os.path.basename(fpath)
                if fbase in possible_test_files:
                    found_tests.append(fpath)

        # 2. Look for test functions
        # Try both exact name and snake_case conversion
        candidates = []
        if symbol_name:
            candidates.append(f"test_{symbol_name}")
            # CamelToSnake
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", symbol_name).lower()
            candidates.append(f"test_{snake}")

        if candidates:
            for node in self.nodes.values():
                if node["type"] in ("Function", "Method"):
                    name = node["metadata"].get("name", "")
                    # Check exact match or verify it starts with test_ and contains symbol
                    if name in candidates:
                        found_tests.append(f"{node['id']} (Direct match)")
                    elif (
                        name.startswith("test_")
                        and snake in name
                        and source_file in node["id"]
                    ):
                        # Heuristic: Test in same file containing the snake_case symbol name
                        found_tests.append(f"{node['id']} (In-file match)")

        return found_tests[:5]  # Limit results

    def _find_related_configs(self, file_id: str, exclude_node_id: str) -> list[dict]:
        """Find Configuration nodes in the same file (interceptors, middleware, event handlers).

        This helps users understand the full setup when looking at related functions.
        For example, when viewing set_token_getter, show the request interceptor that uses it.

        Args:
            file_id: The file ID (e.g., "FILE:webapp/src/lib/api/client.ts")
            exclude_node_id: The current node to exclude from results

        Returns:
            List of dicts with config node info (name, signature, lineno, id)
        """
        configs = []

        # Find DEFINES edges from this file to Configuration nodes
        for edge in self.edges:
            if edge["source"] == file_id and edge["relation"] == "DEFINES":
                target_id = edge["target"]
                if target_id == exclude_node_id:
                    continue

                target_node = self.nodes.get(target_id)
                if target_node and target_node.get("type") == "Configuration":
                    meta = target_node.get("metadata", {})
                    configs.append(
                        {
                            "name": meta.get("name", "?"),
                            "signature": meta.get("signature", "?"),
                            "lineno": meta.get("lineno", 0),
                            "id": target_id,
                            "config_type": meta.get("config_type", ""),
                        }
                    )

        # Sort by line number
        configs.sort(key=lambda c: c["lineno"])
        return configs[:5]  # Limit to 5 config nodes

    def _truncate_output_progressively(
        self,
        result: str,
        max_tokens: int,
        source_lang: str,
    ) -> str:
        """Apply progressive truncation to fit within token budget.

        Truncation priority (remove in order):
        1. Expanded Callees (Full Source) section
        2. Usage Examples - reduce to 2
        3. Related Tests section
        4. Related Configuration section
        5. Callers - reduce to 5
        6. Dependencies (Callees) - reduce to 5
        7. Source code - apply head_lines truncation

        Args:
            result: Full output text
            max_tokens: Target token budget
            source_lang: Language for the Source-Code fenced block match.
        """
        current_tokens = _estimate_tokens(result)

        # Priority 1: Remove Expanded Callees section
        if current_tokens > max_tokens:
            result = re.sub(
                r"#### Expanded Callees \(Full Source\):.*?(?=####|\Z)",
                "",
                result,
                flags=re.DOTALL,
            )
            result = re.sub(r"\n{3,}", "\n\n", result)
            current_tokens = _estimate_tokens(result)

        # Priority 2: Reduce Usage Examples to 2
        if current_tokens > max_tokens:
            # Find and truncate usage examples section
            match = re.search(
                r"(#### Usage Examples:.*?)(?=####|\Z)", result, re.DOTALL
            )
            if match:
                examples_section = match.group(1)
                # Keep header and first 2 code blocks
                code_blocks = list(
                    re.finditer(r"```.*?```", examples_section, re.DOTALL)
                )
                if len(code_blocks) > 2:
                    truncated_examples = examples_section[: code_blocks[1].end()]
                    truncated_examples += (
                        f"\n\n*({len(code_blocks) - 2} more examples omitted)*\n\n"
                    )
                    result = (
                        result[: match.start()]
                        + truncated_examples
                        + result[match.end() :]
                    )
                    current_tokens = _estimate_tokens(result)

        # Priority 3: Remove Related Tests section
        if current_tokens > max_tokens:
            result = re.sub(
                r"#### Related Tests:.*?(?=####|\Z)", "", result, flags=re.DOTALL
            )
            result = re.sub(r"\n{3,}", "\n\n", result)
            current_tokens = _estimate_tokens(result)

        # Priority 4: Remove Related Configuration section
        if current_tokens > max_tokens:
            result = re.sub(
                r"#### Related Configuration.*?(?=####|\Z)", "", result, flags=re.DOTALL
            )
            result = re.sub(r"\n{3,}", "\n\n", result)
            current_tokens = _estimate_tokens(result)

        # Priority 5: Reduce Callers to 5
        if current_tokens > max_tokens:
            match = re.search(
                r"(#### Callers \(\d+ shown\):)(.*?)(?=####|\Z)", result, re.DOTALL
            )
            if match:
                callers_content = match.group(2)
                lines = callers_content.strip().split("\n")
                caller_lines = [line for line in lines if line.startswith("- `")]
                if len(caller_lines) > 5:
                    # Keep first 5 callers
                    new_content = "\n".join(caller_lines[:5])
                    new_content += (
                        f"\n  *({len(caller_lines) - 5} more callers omitted)*\n"
                    )
                    result = (
                        result[: match.start()]
                        + match.group(1)
                        + "\n"
                        + new_content
                        + result[match.end() :]
                    )
                    current_tokens = _estimate_tokens(result)

        # Priority 6: Reduce Dependencies to 5
        if current_tokens > max_tokens:
            match = re.search(
                r"(#### Dependencies \(Callees\):)(.*?)(?=####|\Z)", result, re.DOTALL
            )
            if match:
                # Each top-level dependency with what is nested under it:
                # its code and its own callees go with it or not at all.
                entries = []
                for line in match.group(2).strip().split("\n"):
                    if line.startswith(("- ", "*(")) or not entries:
                        entries.append([line])
                    else:
                        entries[-1].append(line)
                if len(entries) > 5:
                    dropped = sum(
                        sum(line.lstrip().startswith("- ") for line in entry)
                        for entry in entries[5:]
                    )
                    new_content = "\n".join("\n".join(entry) for entry in entries[:5])
                    new_content += f"\n*({dropped} more dependencies omitted)*\n"
                    result = (
                        result[: match.start()]
                        + match.group(1)
                        + "\n"
                        + new_content
                        + result[match.end() :]
                    )
                    current_tokens = _estimate_tokens(result)

        # Priority 7: Truncate source code (last resort)
        if current_tokens > max_tokens:
            # Apply aggressive head_lines truncation
            match = re.search(
                r"(#### Source Code:\n```" + re.escape(source_lang) + r"\n)(.*?)(```)",
                result,
                re.DOTALL,
            )
            if match:
                source = match.group(2)
                lines = source.split("\n")
                if len(lines) > 30:
                    truncated = "\n".join(lines[:30])
                    truncated += f"\n// ... ({len(lines) - 30} lines omitted, max_output_tokens limit) ...\n"
                    result = (
                        result[: match.start()]
                        + match.group(1)
                        + truncated
                        + match.group(3)
                        + result[match.end() :]
                    )

        # Add truncation note if we applied truncation
        if current_tokens > max_tokens:
            result += f"\n\n*[Output truncated to ~{max_tokens} tokens]*"

        return result

    def _expand_callees_full(self, node_id: str, budget: int = 2000) -> list[dict]:
        """Expand direct callees with their full source code.

        This inlines the complete implementation of functions called by node_id,
        allowing users to understand the full flow without separate queries.

        Args:
            node_id: The node ID to find callees for
            budget: Maximum total tokens to include (default: 2000)

        Returns:
            List of dicts with expanded callee info (name, file, lineno, tokens, lang, source)
        """
        callees = self.get_callees(node_id)
        expanded = []
        budget_remaining = budget

        for callee_id in callees:
            # Skip unresolved REFs
            if callee_id.startswith("REF:"):
                continue

            callee_node = self.nodes.get(callee_id)
            if not callee_node:
                continue

            meta = callee_node.get("metadata", {})
            tokens = meta.get("token_count", 0)

            # Skip if over budget or no token info
            if tokens <= 0 or tokens > budget_remaining:
                continue

            # Get file path and check it exists
            file_path = callee_id.split("::")[0].replace("FILE:", "")
            if not os.path.exists(self._disk_path(file_path)):
                continue

            # Get source code
            start_line = meta.get("lineno", 1)
            end_line = meta.get("end_lineno", start_line + 50)
            source = self.get_source_segment(file_path, start_line, end_line)

            if source:
                expanded.append(
                    {
                        "name": meta.get("name", "?"),
                        "file": file_path,
                        "lineno": start_line,
                        "tokens": tokens,
                        "lang": _get_syntax_lang(file_path, self._syntax_lang_map),
                        "source": source.rstrip(),
                    }
                )
                budget_remaining -= tokens

            # Stop if budget exhausted
            if budget_remaining <= 0:
                break

        return expanded

    def _find_similar_nodes(self, query: str, limit: int = 5) -> list[dict]:
        """Find nodes similar to the given query when exact match fails.

        Handles partial matches for:
        - Symbol name only (e.g., "authenticate" finds "FILE:auth.rs::authenticate")
        - Class::method format (e.g., "TokenAuth::from_request_parts")
        - Partial paths (e.g., "auth.rs::validate_token")
        - Naming convention variants (camelCase vs snake_case)

        Returns list of candidate nodes with match quality scores.
        """
        # Skip when query is empty/whitespace (would "match" all nodes).
        if not query or not query.strip():
            return []
        candidates = []

        # Extract the symbol part from the query (last component after ::)
        query_parts = query.replace("FILE:", "").split("::")
        query_symbol = query_parts[-1] if query_parts else query
        query_normalized = _normalize_name(query_symbol)

        # Also check for Class::method pattern
        query_class = query_parts[-2] if len(query_parts) >= 2 else None
        query_class_normalized = _normalize_name(query_class) if query_class else None

        for node_id, node in self.nodes.items():
            if node["type"] == "File":
                continue  # Skip file nodes

            meta = node.get("metadata", {})
            node_name = meta.get("name", "")
            node_name_normalized = _normalize_name(node_name)

            # Extract class/struct name from node_id
            id_parts = node_id.replace("FILE:", "").split("::")
            node_class = (
                id_parts[-2]
                if len(id_parts) >= 2 and node["type"] == "Method"
                else None
            )

            score = 0

            # Exact symbol name match (highest priority)
            if node_name == query_symbol:
                score += 100
            elif node_name_normalized == query_normalized:
                score += 80  # Naming convention variant

            # Partial symbol match (require >= 2 chars to avoid
            # `"" in anything` being always True)
            elif len(query_normalized) >= 2 and (
                query_normalized in node_name_normalized
                or node_name_normalized in query_normalized
            ):
                score += 40

            # Class::method match
            if query_class and node_class:
                node_class_normalized = _normalize_name(node_class)
                if node_class == query_class:
                    score += 50
                elif node_class_normalized == query_class_normalized:
                    score += 40

            # Path match (if query contains path components)
            if len(query_parts) > 1:
                node_path = node_id.replace("FILE:", "")
                if query.replace("FILE:", "") in node_path:
                    score += 30

            if score > 0:
                candidates.append(
                    {
                        "id": node_id,
                        "name": node_name,
                        "type": node["type"],
                        "score": score,
                        "file": id_parts[0] if id_parts else "",
                        "lineno": meta.get("lineno", 0),
                    }
                )

        # Sort by score descending, then by name
        candidates.sort(key=lambda x: (-x["score"], x["name"]))
        return candidates[:limit]

    def _resolve_node_id(self, query: str) -> tuple[str | None, str | None]:
        """Resolve a query to an exact node ID, with fuzzy matching fallback.

        Returns:
            (node_id, message) - node_id if found, message for user feedback
        """
        # Try exact match first
        if query in self.nodes:
            return query, None

        # Try with FILE: prefix if not present
        if not query.startswith("FILE:") and f"FILE:{query}" in self.nodes:
            return f"FILE:{query}", None

        # Fuzzy matching fallback
        similar = self._find_similar_nodes(query, limit=5)
        if not similar:
            return None, f"Node '{query}' not found and no similar matches."

        # Auto-select if:
        # - Single match, OR
        # - High-confidence match (score >= 80 for naming variant, >= 100 for exact) that's clearly ahead
        best = similar[0]
        second_score = similar[1]["score"] if len(similar) > 1 else 0

        if len(similar) == 1 or (
            best["score"] >= 80 and best["score"] > second_score + 20
        ):
            note = f"(Matched '{query}' → `{best['id']}`)"
            return best["id"], note

        # Multiple candidates - return suggestions
        suggestions = []
        for s in similar[:5]:
            suggestions.append(
                f"  - `{s['id']}` ({s['type']}, {s['file']}:{s['lineno']})"
            )

        msg = f"Node '{query}' not found. Did you mean:\n" + "\n".join(suggestions)
        return None, msg

    def _get_usage_examples(self, target_node_id, limit=2):
        """Snippets of the calls of ``target_node_id``: those resolved to it
        first, then those it is one of the candidates of, each with
        ``candidates`` (None when resolved); examples and tests first within
        each."""
        if target_node_id not in self.nodes:
            return []
        calls = [
            edge
            for index in (self.incoming, self.candidate_incoming)
            for edge in index.get(target_node_id, [])
            if edge["relation"] in CALL_RELATIONS
        ]

        def order(edge):
            src = edge["source"]
            place = 0 if "example" in src else 1 if "test" in src else 2
            return edge["target"] != target_node_id, place

        examples = []
        for edge in sorted(calls, key=order):
            if len(examples) == limit:
                break
            file_path = edge["source"].split("::")[0].replace("FILE:", "")
            lineno = edge.get("metadata", {}).get("lineno")
            if lineno and os.path.exists(self._disk_path(file_path)):
                code = self.get_source_segment(file_path, lineno - 1, lineno + 2)
                candidates = edge.get("metadata", {}).get("candidates")
                examples.append(
                    {
                        "file": file_path,
                        "lineno": lineno,
                        "code": code.strip(),
                        "candidates": len(candidates) if candidates else None,
                    }
                )
        return examples

    def _get_callers_summary(self, target_node_id, limit=None):
        """Get summary of callers: name, file, lineno, tokens, and the
        number of candidates of a call that may reach the target.

        Args:
            target_node_id: The node ID to find callers for
            limit: Maximum callers to return (defaults to MAX_CALLERS_SHOWN)

        Returns:
            List of dicts with caller info, sorted by token count (simpler first)
        """
        if limit is None:
            limit = self._max_callers_shown

        if target_node_id not in self.nodes:
            return []
        caller_ids = self.get_callers(target_node_id)
        ambiguous = self.get_ambiguous_callers(target_node_id)

        callers = []
        for caller_id in caller_ids:
            caller_node = self.nodes.get(caller_id)
            if caller_node:
                meta = caller_node.get("metadata", {})
                file_path = caller_id.split("::")[0].replace("FILE:", "")
                callers.append(
                    {
                        "id": caller_id,
                        "name": meta.get("name", "?"),
                        "file": file_path,
                        "lineno": meta.get("lineno", 0),
                        "tokens": meta.get("token_count", 0),
                        "type": caller_node.get("type", "?"),
                        "candidates": ambiguous.get(caller_id),
                    }
                )

        # Those resolved to it first, then simpler callers, then by name
        callers.sort(key=lambda c: (bool(c["candidates"]), c["tokens"], c["name"]))
        return callers[:limit]

    def _expand_callees(self, node_id, depth, budget, inline_threshold):
        """Lines listing the callees of ``node_id``, ``depth`` levels deep.

        Planned level by level: the symbol's own callees are all listed, and
        each level below is cut to the per-level limit, and at the node
        limit, with a note. A symbol
        is opened (its code inlined, its own callees listed) at the first
        level that reaches it, when it is small
        enough to inline and the token ``budget`` still covers it; a symbol
        not opened is listed by its signature, one met again as "(at level
        N)", and an unresolved call where it is made. A note says what cut
        the expansion short: the budget, the node limit or the timeout.
        """
        start_time = time.time()
        placed = {node_id: 0}
        rows = {}  # parent -> ([(kind, callee)], callees left out, marks)
        cuts = []
        frontier = [node_id]
        for level in range(1, depth + 1):
            opened = []
            for parent in frontier:
                callees = self.get_callees(parent)
                marks = self.get_ambiguous_callees(parent)
                # The symbol's own callees are all listed; below them, each
                # level is cut to the per-level limit.
                limit = len(callees) if level == 1 else self._max_children_per_level
                listed = []
                rows[parent] = (listed, callees[limit:], marks)
                for index, callee in enumerate(callees[:limit]):
                    if (time.time() - start_time) * 1000 > self._timeout_ms:
                        cuts.append("timeout")
                        break
                    if callee not in self.nodes:
                        listed.append(("call", callee))
                        continue
                    if callee in placed:
                        listed.append(("seen", callee))
                        continue
                    if level > 1 and self._max_nodes and len(placed) > self._max_nodes:
                        # Those not reached are counted with those left out
                        rows[parent] = (listed, callees[index:], marks)
                        cuts.append("node limit")
                        break
                    placed[callee] = level
                    meta = self.nodes[callee]["metadata"]
                    tokens = meta.get("token_count", 999)
                    on_disk = os.path.exists(
                        self._disk_path(callee.split("::")[0].replace("FILE:", ""))
                    )
                    if tokens <= inline_threshold and on_disk:
                        if tokens <= budget:
                            budget -= tokens
                            listed.append(("open", callee))
                            opened.append(callee)
                            continue
                        if "token budget" not in cuts:
                            cuts.append("token budget")
                    listed.append(("summary", callee))
                if cuts and cuts[-1] != "token budget":
                    break
            if cuts and cuts[-1] != "token budget":
                break
            frontier = opened

        lines = []

        def render(parent, indent):
            listed, left_out, marks = rows.get(parent, ([], [], {}))
            pad = "  " * indent
            for kind, callee in listed:
                meta = self.nodes.get(callee, {}).get("metadata", {})
                name = meta.get("name", "?")
                if kind == "call":
                    clean = _clean_ref_name(callee.replace("REF:", ""))
                    lines.append(
                        f"{pad}- `{clean}`{self.callee_note(marks.get(callee))}"
                    )
                elif kind == "seen":
                    lines.append(f"{pad}- `{name}` (at level {placed[callee]})")
                elif kind == "summary":
                    sig = meta.get("signature", f"{name}(...)")
                    lines.append(
                        f"{pad}- `{sig}` ({meta.get('token_count', 999)} toks)"
                    )
                    doc = (meta.get("docstring", "") or "").split("\n")[0][:80]
                    if doc:
                        lines.append(f"{pad}  > {doc}")
                else:
                    file_path = callee.split("::")[0].replace("FILE:", "")
                    start_line = meta.get("lineno", 1)
                    code = self.get_source_segment(
                        file_path, start_line, meta.get("end_lineno", start_line + 10)
                    )
                    lang = _get_syntax_lang(file_path, self._syntax_lang_map)
                    lines.append(
                        f"{pad}- **{name}** ({meta.get('token_count', 999)} toks):"
                    )
                    lines.append(f"{pad}  ```{lang}")
                    lines.extend(f"{pad}  {line}" for line in code.rstrip().split("\n"))
                    lines.append(f"{pad}  ```")
                    render(callee, indent + 1)
            if left_out:
                certain, possible, none = self.kinds(left_out, marks)
                more = self.more_callers(certain, possible, "callees", none)
                lines.append(f"{pad}*({more})*")

        render(node_id, 0)
        lines.extend(f"*({cut} reached)*" for cut in cuts)
        return lines

    @property
    def max_flow_depth(self):
        """The most levels of calls a flow or context goes: the project's
        `[query] max_depth`, or 5 when unset."""
        return self._max_depth or 5

    def flow_depth(self, depth):
        """The depth a flow of ``depth`` is traced to."""
        return min(depth, self.max_flow_depth)

    def _resolve_flow_start(self, start_name, depth):
        """Clamp depth to the configured ceiling and resolve the start symbol.

        Returns ``(func_nodes, depth)``; ``func_nodes`` is empty when nothing
        matched, leaving each caller to raise its own error shape.
        """
        depth = self.flow_depth(depth)

        start_nodes = self.find_nodes_by_name(start_name)
        if not start_nodes:
            start_nodes = self.find_nodes_by_name(start_name, fuzzy=True)

        func_nodes = [n for n in start_nodes if n["type"] in ("Function", "Method")]
        if not func_nodes and start_nodes:
            # Fall back to the first match even when it is not a function.
            func_nodes = start_nodes[:1]
        return func_nodes, depth

    def _flow_next(self, node_id, direction):
        """The next level of a flow from ``node_id``: its callees, or its
        callers, those resolved first; and ``{id: number of candidates}`` of
        those that only may be."""
        if direction == "forward":
            # Those resolved, then each definition an unresolved call may reach
            callees = [c for c in self.get_callees(node_id) if c in self.nodes]
            possible = {}
            for edge in self.outgoing[node_id]:
                candidates = edge.get("metadata", {}).get("candidates")
                if edge["relation"] == "CALLS" and candidates:
                    for candidate in candidates:
                        if candidate not in callees:
                            possible.setdefault(candidate, len(candidates))
            return callees + list(possible), possible
        return self.ordered_callers(node_id)

    def _flow_tree(
        self, start_id, direction, depth, target, inline_threshold, timeout_ms
    ):
        """The flow from ``start_id``, level by level: each symbol is placed
        once, at the first level that reaches it, and met again as `seen`,
        the level of that entry (the start is level 0).
        Returns ``(root, path, cut)``: the tree, the node IDs from the start
        to the first node named ``target``, and why it stopped early
        ("timeout", "node limit") or None."""
        start_time = time.time()

        def tree_node(node_id, candidates):
            node = self.nodes[node_id]
            meta = node.get("metadata", {})
            tokens = meta.get("token_count", 0)
            file_path = node_id.split("::")[0].replace("FILE:", "")
            code = None
            if 0 < tokens <= inline_threshold and os.path.exists(
                self._disk_path(file_path)
            ):
                start_line = meta.get("lineno", 1)
                end_line = meta.get("end_lineno", start_line + 10)
                code = self.get_source_segment(file_path, start_line, end_line)
            return {
                "id": node_id,
                "name": meta.get("name", "?"),
                "type": node.get("type", "?")[:3],
                "tokens": tokens,
                "file": file_path,
                "line": meta.get("lineno"),
                "code": code,
                "lang": _get_syntax_lang(file_path, self._syntax_lang_map),
                "candidates": candidates,
                "children": [],
                # Those of the next level past the per-level limit, and how
                # many of them only may be
                "more": 0,
                "more_possible": 0,
            }

        root = tree_node(start_id, None)
        paths = {start_id: [start_id]}
        if target and root["name"] == target:
            return root, paths[start_id], None
        level = [root]
        for _ in range(depth):
            next_level = []
            for parent in level:
                next_ids, marks = self._flow_next(parent["id"], direction)
                left_out = next_ids[self._max_children_per_level :]
                parent["more"] = len(left_out)
                parent["more_possible"] = sum(i in marks for i in left_out)
                for child_id in next_ids[: self._max_children_per_level]:
                    if (time.time() - start_time) * 1000 > timeout_ms:
                        return root, None, "timeout"
                    if child_id not in self.nodes:
                        continue
                    if child_id in paths:
                        seen = tree_node(child_id, marks.get(child_id))
                        seen.update(code=None, seen=len(paths[child_id]) - 1)
                        parent["children"].append(seen)
                        continue
                    if len(paths) > self._max_nodes:
                        return root, None, "node limit"
                    child = tree_node(child_id, marks.get(child_id))
                    paths[child_id] = [*paths[parent["id"]], child_id]
                    parent["children"].append(child)
                    if target and child["name"] == target:
                        return root, paths[child_id], None
                    next_level.append(child)
            level = next_level
        return root, None, None

    def trace_flow(
        self,
        start_name: str,
        direction: str = "forward",
        depth: int = 3,
        target: str | None = None,
        inline_threshold: int = 100,
        timeout_ms: int | None = None,
    ) -> str:
        """Trace call flow showing path with inline code.

        Args:
            start_name: Starting symbol name or node ID
            direction: 'forward' traces callees, 'backward' traces callers
            depth: Levels of calls below the start (default: 3), at most
                `flow_depth` allows
            target: Optional target to stop when reached
            inline_threshold: Inline code below this token count (default: 100)
            timeout_ms: Timeout in milliseconds (default: TIMEOUT_MS)

        Returns:
            Formatted flow trace as markdown string
        """
        if timeout_ms is None:
            timeout_ms = self._timeout_ms
        asked = depth
        func_nodes, depth = self._resolve_flow_start(start_name, depth)
        if not func_nodes:
            return f"No function '{start_name}' found."

        start_node = func_nodes[0]
        start_id = start_node["id"]
        start_meta = start_node.get("metadata", {})

        output = [
            f"### Call Flow: `{start_meta.get('name', start_name)}` ({direction})"
        ]
        output.append("")
        if depth < asked:
            output.append(
                f"**Note:** depth {asked} is past `[query] max_depth`; traced to {depth}."
            )
            output.append("")

        # Show disambiguation info if multiple matches
        if len(func_nodes) > 1:
            start_file = start_id.split("::")[0].replace("FILE:", "")
            output.append(
                f"**Note:** Found {len(func_nodes)} matches for `{start_name}`. Using:"
            )
            output.append(f"  `{start_id}`")
            output.append(f"  ({start_file})")
            output.append("")
            output.append("**Alternatives** (use full node ID to select):")
            for alt in func_nodes[1:5]:  # Show up to 4 alternatives
                output.append(f"  - `{alt['id']}`")
            if len(func_nodes) > 5:
                output.append(f"  - ... and {len(func_nodes) - 5} more")
            output.append("")

        root, path, cut = self._flow_tree(
            start_id, direction, depth, target, inline_threshold, timeout_ms
        )
        of = "callees" if direction == "forward" else "callers"

        def render(node, level):
            indent = "  " * level
            arrow = "├─" if level else ""
            head = f"{indent}{arrow}[{node['type']}] **{node['name']}**"
            if node.get("seen") is not None:
                output.append(
                    f"{head}{self.one_of(node['candidates'])} (at level {node['seen']})"
                )
                return
            output.append(
                f"{head} ({node['tokens']} toks){self.one_of(node['candidates'])}"
            )
            if node["code"]:
                output.append(f"{indent}  ```{node['lang']}")
                output.append(node["code"].rstrip())
                output.append(f"{indent}  ```")
            for child in node["children"]:
                render(child, level + 1)
            if node["more"]:
                certain = node["more"] - node["more_possible"]
                more = self.more_callers(certain, node["more_possible"], of)
                output.append(f"{indent}  *({more})*")

        render(root, 0)
        if cut:
            output.append(f"*({cut})*")
        if target and path:
            output.append("")
            output.append(f"#### Path to `{target}`:")
            output.append(" -> ".join(p.split("::")[-1] for p in path))

        return "\n".join(output)

    def trace_flow_structured(
        self,
        start_name: str,
        direction: str = "forward",
        depth: int = 3,
        target: str | None = None,
        inline_threshold: int = 100,
        timeout_ms: int | None = None,
    ) -> dict:
        """Trace call flow, returning a structured tree for UI rendering.

        Returns dict with keys: root, alternatives (optional), note (optional),
        path_to_target (symbol names from the start to `target`, when reached),
        cut (why the trace stopped early, when it did). Each tree node has: id,
        name, type, tokens, file, line, code, lang, candidates, children,
        more, more_possible, and `seen`, the level its entry is at, where it
        is placed there already. The trace stops at the first node named `target`.
        """
        if timeout_ms is None:
            timeout_ms = self._timeout_ms
        func_nodes, depth = self._resolve_flow_start(start_name, depth)
        if not func_nodes:
            return {"error": f"No function '{start_name}' found."}

        start_node = func_nodes[0]
        start_id = start_node["id"]

        # Build alternatives list for disambiguation
        alternatives = []
        note = None
        if len(func_nodes) > 1:
            start_file = start_id.split("::")[0].replace("FILE:", "")
            note = (
                f'Found {len(func_nodes)} matches for "{start_name}". '
                f"Using: {start_id} ({start_file})"
            )
            for alt in func_nodes[1:6]:
                alt_meta = alt.get("metadata", {})
                alt_file = alt["id"].split("::")[0].replace("FILE:", "")
                alternatives.append(
                    {
                        "id": alt["id"],
                        "name": alt_meta.get("name", "?"),
                        "file": alt_file,
                    }
                )

        root, path, cut = self._flow_tree(
            start_id, direction, depth, target, inline_threshold, timeout_ms
        )
        result = {"root": root}
        if path:
            result["path_to_target"] = [
                self.nodes[p]["metadata"].get("name", "?") for p in path
            ]
        if cut:
            result["cut"] = cut
        if note:
            result["note"] = note
        if alternatives:
            result["alternatives"] = alternatives

        return result

    def flatten_class(self, class_node_id):
        """
        Traverses INHERITS edges to build a full list of methods for a class.
        """
        class_node_id, resolve_msg = self._resolve_node_id(class_node_id)
        if class_node_id is None:
            return f"ERROR: {resolve_msg}"
        class_node = self.nodes[class_node_id]
        if class_node["type"] != "Class":
            return f"ERROR: '{class_node_id}' is a {class_node['type']}, not a Class."

        # 1. Collect Inheritance Chain
        chain = []  # List of class nodes, starting from root to leaf
        visited = set()

        def traverse_up(current_id):
            if current_id in visited:
                return
            visited.add(current_id)

            # Find definition node if current_id is a REF
            if current_id.startswith("REF:"):
                # Try to find the Class node definition
                name = current_id.replace("REF:", "")
                matches = self.find_nodes_by_name(name)
                class_matches = [m for m in matches if m["type"] == "Class"]
                if len(class_matches) == 1:
                    current_node = class_matches[0]
                else:
                    # External, not found, or one of several classes of the name
                    chain.insert(
                        0, {"id": current_id, "name": name, "type": "External"}
                    )
                    return
            else:
                current_node = self.nodes.get(current_id)

            if current_node:
                chain.insert(0, current_node)
                # Find parents
                for edge in self.outgoing[current_node["id"]]:
                    if edge["relation"] == "INHERITS":
                        traverse_up(edge["target"])

        traverse_up(class_node_id)

        # 2. Merge Methods
        methods = {}  # Name -> Node/Signature

        output = []
        output.append(
            f"### Flattened View: `{class_node['metadata'].get('name', class_node_id)}`"
        )
        if resolve_msg:
            output.append(f"> **Note:** {resolve_msg}")
        output.append("#### Inheritance Chain:")
        output.append(
            " -> ".join(
                [
                    c.get("metadata", {}).get("name") or c.get("name") or c["id"]
                    for c in chain
                ]
            )
        )
        output.append("\n#### Effective API:")

        for cls in chain:
            if cls.get("type") == "External":
                continue

            # Find methods defined in this class
            # (Look at edges outgoing from class node with 'DEFINES' relation)
            for edge in self.outgoing[cls["id"]]:
                if edge["relation"] == "DEFINES":
                    child = self.nodes.get(edge["target"])
                    if child and child["type"] == "Method":
                        name = child["metadata"].get("name", "")
                        methods[name] = child

        # Output sorted methods
        for name in sorted(methods.keys()):
            m = methods[name]
            meta = m["metadata"]
            sig = meta.get("signature", name)
            defined_in = m["id"].split("::")[0].replace("FILE:", "")
            output.append(f"- `{sig}`")
            output.append(f"  *Defined in: {defined_in}*")
            if meta.get("docstring"):
                output.append(f"  > {meta['docstring'].splitlines()[0]}")

        return "\n".join(output)

    def _build_idf_cache(self):
        """Build IDF (Inverse Document Frequency) cache for TF-IDF search.

        IDF = log(N / df) where:
        - N = total number of documents (nodes)
        - df = document frequency (number of docs containing the term)
        """
        doc_freq = defaultdict(int)  # term -> count of docs containing it
        total_docs = len(self.data["nodes"])

        for node in self.data["nodes"]:
            meta = node.get("metadata", {})
            # Tokenize name and docstring
            text = f"{meta.get('name', '')} {meta.get('docstring', '')}"
            # Simple tokenization: split on non-alphanumeric, lowercase
            tokens = set(re.findall(r"[^\W\d]\w*", text.lower(), flags=re.UNICODE))
            for token in tokens:
                doc_freq[token] += 1

        # Compute IDF with smoothing to avoid log(0)
        idf = {}
        for term, df in doc_freq.items():
            # Add 1 to df for smoothing (avoids division issues)
            idf[term] = math.log((total_docs + 1) / (df + 1)) + 1
        return idf

    def search_docs(
        self, query_terms, lang=None, crate=None, symbol_type=None, exclude_tests=False
    ):
        """
        Searches docstrings and names for keywords using TF-IDF scoring.
        Returns matches sorted by relevance.

        Args:
            query_terms: List of search terms
            lang: Filter by language ('rust', 'typescript', 'python', 'svelte', 'all')
            crate: Filter by crate/module (e.g., 'backend', 'webapp')
            symbol_type: Filter by type ('function', 'class', 'method', 'constant', 'file', 'all')
            exclude_tests: If True, filter out test files and test functions (default: False)

        Scoring:
        - Name matches: TF-IDF * 10 (high weight for name match)
        - Docstring matches: TF-IDF * 1 (standard weight)
        - Exact name match: +50 bonus
        - All terms present (phrase): +30 bonus per additional term
        - Adjacent terms (phrase proximity): +20 bonus
        """
        results = []
        # Drop empty/whitespace-only terms before scoring — `"" in s` is
        # always True and would match every node.
        terms = [t.lower() for t in query_terms if t and t.strip()]
        num_terms = len(terms)
        if num_terms == 0:
            return []

        # Apply filters if any specified
        if lang or crate or symbol_type or exclude_tests:
            self._ensure_filter_indices()
            candidate_ids = set(self.nodes.keys())

            # Filter out test files and functions if requested
            if exclude_tests and self._index_is_test:
                candidate_ids -= self._index_is_test

            if crate:
                # If crate filter specified but doesn't exist, return empty results
                if crate not in self._index_by_crate:
                    return []
                candidate_ids &= self._index_by_crate[crate]
            if lang and lang != "all":
                lang = _LANG_ALIASES.get(lang.lower(), lang.lower())
                candidate_ids &= self._index_by_lang.get(lang, set())
            if symbol_type and symbol_type != "all":
                type_map = {
                    "function": "Function",
                    "class": "Class",
                    "method": "Method",
                    "constant": "Constant",
                    "file": "File",
                }
                type_key = type_map.get(symbol_type.lower())
                if type_key and type_key in self._index_by_type:
                    candidate_ids &= self._index_by_type[type_key]

            candidates = [self.nodes[nid] for nid in candidate_ids]
        else:
            candidates = self.data["nodes"]

        for node in candidates:
            score = 0.0
            meta = node.get("metadata", {})
            name = meta.get("name", "").lower()
            doc = meta.get("docstring", "").lower()
            sig = meta.get("signature", "").lower()

            # Combine searchable text
            full_text = f"{name} {doc} {sig}"

            # Tokenize for TF calculation
            name_tokens = re.findall(r"[^\W\d]\w*", name, flags=re.UNICODE)
            doc_tokens = re.findall(r"[^\W\d]\w*", doc, flags=re.UNICODE)
            full_tokens = re.findall(r"[^\W\d]\w*", full_text, flags=re.UNICODE)

            terms_found = 0
            for term in terms:
                idf = self._idf_cache.get(term, 1.0)  # Default IDF if term not in cache

                # Term frequency in name (normalized by name length)
                name_tf = name_tokens.count(term) / max(len(name_tokens), 1)
                if name_tf > 0:
                    score += name_tf * idf * 10  # Higher weight for name match
                    terms_found += 1

                # Term frequency in docstring (normalized)
                doc_tf = doc_tokens.count(term) / max(len(doc_tokens), 1)
                if doc_tf > 0:
                    score += doc_tf * idf * 1
                    if name_tf == 0:  # Only count once
                        terms_found += 1

                # Check in signature too
                if term in sig:
                    score += idf * 2
                    if name_tf == 0 and doc_tf == 0:
                        terms_found += 1

                # Bonus for exact name match
                if name == term:
                    score += 50

                # Bonus for substring match in name (partial matching)
                elif term in name:
                    score += idf * 5

            # Multi-term query handling - use tier-based scoring
            # Full matches ALWAYS rank above partial matches
            if num_terms > 1:
                if terms_found == num_terms:
                    # ALL TERMS PRESENT - boost to tier 2 (1000+ base)
                    # This ensures full matches always outrank partial matches
                    score = 1000 + score * 2  # Base tier + amplified TF-IDF

                    # Check for adjacent terms (phrase proximity)
                    phrase = "_".join(terms)  # e.g., "order_dispatch"
                    phrase_space = " ".join(terms)  # e.g., "deployment dispatch"

                    if phrase in name or phrase in full_text:
                        score += 500  # Strong bonus for exact phrase as compound word
                    elif phrase_space in full_text:
                        score += 200  # Bonus for adjacent terms

                    # Check for terms appearing within 3 tokens of each other
                    for i, token in enumerate(full_tokens):
                        if token == terms[0]:
                            window = full_tokens[i : i + num_terms + 2]
                            if all(t in window for t in terms):
                                score += 100  # Proximity bonus
                                break
                elif terms_found > 0:
                    # PARTIAL MATCH - stays in tier 1 (0-999)
                    # Apply harsh penalty: partial matches are deprioritized
                    match_ratio = terms_found / num_terms
                    score *= match_ratio * match_ratio  # Quadratic penalty
                    # Cap partial matches below the full-match tier
                    score = min(score, 999)

            if score > 0:
                # Apply in-degree boost (symbols with more callers are more important)
                # Use logarithmic scaling to prevent extremely popular symbols from dominating
                in_degree = node.get("metadata", {}).get("in_degree", 0)
                if in_degree > 0:
                    # Log scaling: in_degree of 1 -> 0, 10 -> 2.3, 100 -> 4.6
                    in_degree_boost = math.log(1 + in_degree) * 5
                    score += in_degree_boost

                # Type preference: Functions/Classes ranked above Constants for same score
                # Adds small tie-breaker that doesn't overwhelm TF-IDF relevance
                node_type = node.get("type", "")
                type_boost = {
                    "Function": 0.5,
                    "Method": 0.5,
                    "Class": 0.4,
                    "Constant": 0.1,
                    "File": 0.0,
                }.get(node_type, 0.2)
                score += type_boost

                results.append((score, node))

        # Sort by score desc
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results]

    def get_node_info(self, node_id):
        return self.nodes.get(node_id)

    def find_nodes_by_name(self, name, fuzzy=False):
        """Find nodes by name, supporting multiple matching strategies.

        Supports:
        - Exact name match: "start" matches any node with name="start"
        - Naming convention variants: "getClient" matches "get_client" (camelCase/snake_case)
        - Qualified name: "Server::start", also written "Server.start", matches node IDs
          ending with that pattern
        - File path suffix: "server.rs" matches files ending with that path
        - Partial qualified (when fuzzy=True): "Server::sta" matches IDs containing that pattern
        - Fuzzy/prefix matching (when fuzzy=True): "validate_token" matches "validate_token_format"

        Args:
            name: Name or pattern to search for
            fuzzy: If True, also matches names that start with or contain the search term
        """
        # Empty/whitespace name would match every node (`"" in s` is True).
        if not name or not name.strip():
            return []
        # A node ID names that node and no other.
        if name in self.nodes:
            return [self.nodes[name]]
        exact_matches = []
        variant_matches = []
        fuzzy_matches = []

        # Get naming convention variants (e.g., getClient -> [getClient, get_client])
        name_variants = _get_name_variants(name)

        for node in self.data["nodes"]:
            meta = node.get("metadata", {})
            node_id = node["id"]
            node_name = meta.get("name", "")
            node_name_normalized = _normalize_name(node_name) if node_name else ""
            owner, _, filed = node_id.rpartition("::")

            # Exact name match: the name source gives it, or the one it is
            # filed under where that is another (an accessor's `get size`).
            if name in (node_name, filed):
                exact_matches.append(node)
                continue

            # Naming convention variant match (e.g., getClient matches get_client)
            if node_name and (
                node_name_normalized in name_variants
                or node_name.lower() in [v.lower() for v in name_variants]
            ):
                variant_matches.append(node)
                continue

            # File path suffix match
            if str(meta.get("path", "")).endswith(name):
                exact_matches.append(node)
                continue

            # Qualified name match (e.g., "Server::start"); an ID that only
            # contains the name is a fuzzy match.
            if "::" in name:
                lowered = node_id.lower()
                if _ends_at_segment(node_id, name) or _ends_at_segment(
                    f"{owner}::{node_name}", name
                ):
                    exact_matches.append(node)
                elif any(_ends_at_segment(lowered, v) for v in name_variants):
                    variant_matches.append(node)
                elif fuzzy and name.lower() in lowered:
                    fuzzy_matches.append(node)

            # Fuzzy matching: prefix or contains (for simple names)
            if fuzzy and "::" not in name and len(name) >= 3:
                # Prefix match on node name (e.g., "validate_token" matches "validate_token_format")
                if node_name.startswith(name):
                    fuzzy_matches.append(node)
                    continue
                # Contains match in node ID (for cases like searching "dispatch" finding "dispatch_order")
                if f"::{name}" in node_id.lower() or node_id.lower().endswith(
                    name.lower()
                ):
                    fuzzy_matches.append(node)
                    continue

        # Return exact matches first, then variant matches, then fuzzy if no others
        found = exact_matches or variant_matches or fuzzy_matches
        if not found and "." in name:
            # `Type.name` that matches nothing as written is `Type::name`.
            return self.find_nodes_by_name(name.replace(".", "::"), fuzzy)
        return found

    def find_trait_impls(self, method_name: str, trait_name: str | None = None) -> list:
        """Find all implementations of a trait method across the codebase.

        When you know a trait method name (e.g., 'from_request_parts') but need
        to find which structs implement it, this discovers all implementations.

        Args:
            method_name: The method name to search for (e.g., 'from_request_parts')
            trait_name: Optional filter by specific trait (e.g., 'FromRequestParts')

        Returns:
            List of nodes that are trait implementations matching the criteria.
            Each node includes metadata with 'trait_impl' field showing which trait
            it implements.

        Example:
            find_trait_impls("from_request_parts")
            # Returns the extractor types: an auth guard, a pagination query, etc.

            find_trait_impls("from_request_parts", "FromRequestParts")
            # Returns only FromRequestParts implementations
        """
        results = []

        for node in self.data["nodes"]:
            # Only consider Method and Function types
            if node["type"] not in ("Method", "Function"):
                continue

            meta = node.get("metadata", {})
            node_name = meta.get("name", "")

            # Check if method name matches
            if node_name != method_name:
                continue

            # Check if this is a trait implementation
            impl_trait = meta.get("trait_impl")
            if impl_trait and (trait_name is None or impl_trait == trait_name):
                results.append(node)

        return results

    def get_callers(self, func_name, fuzzy=False):
        """Find all callers of a function or users of a struct/class.

        OPTIMIZED: Uses incoming index for O(in-degree) instead of O(E) scan.

        Supports:
        - Simple name: "start" matches calls to any function named start
        - Qualified name: "Server::start" matches that specific method
        - Full node ID: "FILE:path::Class::method" for exact matching
        - Struct names: includes INSTANTIATES edges for struct usage tracking
        - Fuzzy matching (when fuzzy=True): "validate_token" matches "validate_token_format"

        Args:
            func_name: Function/method name to find callers for
            fuzzy: If True, also matches names that start with the search term
        """
        return sorted(self._caller_marks(func_name, fuzzy))

    @staticmethod
    def possible(count, unresolved=0):
        """The note on a count of callers or callees of those that only may
        be: ``count`` of them, and ``unresolved`` calls of the name the
        resolver found no candidates for; empty for none."""
        extra = [f"+{count} possible"] if count else []
        if unresolved:
            extra.append(f"+{unresolved} unresolved")
        return f" ({', '.join(extra)})" if extra else ""

    @staticmethod
    def more_callers(certain, possible, of="callers", unresolved=0):
        """The note on the callers, or callees, a list leaves out: ``certain``
        of them, ``possible`` that only may be, and ``unresolved`` calls of
        the name."""
        parts = [f"{certain} more {of}"] if certain else []
        if possible:
            parts.append(f"{possible} more possible {of}")
        if unresolved:
            parts.append(f"{unresolved} more unresolved {of}")
        return ", ".join(parts)

    @staticmethod
    def kinds(ids, marks):
        """``(certain, possible, unresolved)`` of ``ids``, by the ``marks`` of
        `get_ambiguous_callers`."""
        possible = sum(bool(marks.get(i)) for i in ids)
        unresolved = sum(marks.get(i) == 0 for i in ids)
        return len(ids) - possible - unresolved, possible, unresolved

    @staticmethod
    def one_of(candidates):
        """The note on a call that is one of ``candidates`` definitions an
        unresolved edge names, or of the name only (0); empty for a call
        that is certain (None)."""
        if candidates == 0:
            return " (unresolved; same name)"
        if candidates == 1:
            return " (unresolved; the only candidate)"
        return f" (one of {candidates} candidates)" if candidates else ""

    def ordered_callers(self, func_name, fuzzy=False):
        """The callers of a symbol matching ``func_name``: those its calls are
        resolved to, then those that may be, then unresolved calls of the
        name; and the `get_ambiguous_callers` marks of all but the first."""
        ambiguous = self.get_ambiguous_callers(func_name, fuzzy)
        callers = self.get_callers(func_name, fuzzy)
        order = {caller: (True, mark == 0) for caller, mark in ambiguous.items()}
        return sorted(callers, key=lambda c: order.get(c, (False, False))), ambiguous

    def get_ambiguous_callers(self, func_name, fuzzy=False):
        """``{caller: candidates}`` of the callers of a symbol matching
        ``func_name`` that are not certain: the number of candidates of an
        unresolved call it is one of, or 0 for an unresolved call of its name
        with no candidates at all."""
        marks = self._caller_marks(func_name, fuzzy)
        return {caller: mark for caller, mark in marks.items() if mark is not None}

    def _caller_marks(self, func_name, fuzzy=False):
        """``{caller: mark}`` of the callers of a symbol matching
        ``func_name``: None for a call resolved to it, the number of
        candidates for an unresolved call it is one of, 0 for an unresolved
        call of the name the resolver found no candidates for."""
        target_ids = self._resolve_target_ids(func_name, fuzzy)
        defined = [t for t in target_ids if not t.startswith("REF:")]
        marks = {}
        for target_id in defined:
            for edge in self.incoming.get(target_id, []):
                if edge["relation"] in CALL_RELATIONS:
                    marks[edge["source"]] = None
        for target_id in defined:
            for edge in self.candidate_incoming.get(target_id, []):
                if edge["relation"] in CALL_RELATIONS:
                    marks.setdefault(
                        edge["source"], len(edge["metadata"]["candidates"])
                    )
        for target_id in target_ids - set(defined):
            for edge in self.incoming.get(target_id, []):
                if edge["relation"] in CALL_RELATIONS and not edge.get(
                    "metadata", {}
                ).get("candidates"):
                    marks.setdefault(edge["source"], 0)
        return marks

    def _resolve_target_ids(self, func_name: str, fuzzy: bool = False) -> set:
        """Resolve a function name to all possible target IDs in the graph.

        Returns node IDs and REF: IDs that match the given name.
        """
        target_ids = set()
        # `Type::name`, also written `Type.name`, asks for that one; a bare
        # name for every `name`.
        names = {func_name, func_name.replace(".", "::"), func_name.replace("::", ".")}

        # Direct matches - these are the most common
        target_ids.add(func_name)
        target_ids.update(f"REF:{name}" for name in names)

        # Find resolved nodes that match the name
        matches = self.find_nodes_by_name(func_name, fuzzy=fuzzy)
        for match in matches:
            target_ids.add(match["id"])

        # Build additional targets from existing nodes/edges that end with our name
        # This catches cases like REF:Foo::Bar when searching for "Bar"
        for node_id in self.nodes:
            if node_id.endswith(tuple(f"::{name}" for name in names)):
                target_ids.add(node_id)

        # Also check for REF targets in incoming edges (for unresolved calls)
        # This is still needed because REF: targets may not be nodes themselves
        for target_id in list(self.incoming.keys()):
            if target_id.startswith("REF:"):
                ref_name = target_id.replace("REF:", "")
                # Exact or suffix match
                endings = tuple(
                    f"{mark}{name}" for name in names for mark in ("::", ".")
                )
                if ref_name in names or ref_name.endswith(endings):
                    target_ids.add(target_id)
                # Fuzzy prefix match
                elif fuzzy and len(func_name) >= 3:
                    clean_ref = ref_name.split("::")[-1].split(".")[-1]
                    if clean_ref.startswith(func_name):
                        target_ids.add(target_id)

        return target_ids

    def get_callees(self, func_node_id):
        """The callees of a function: each definition a call is resolved to,
        and each unresolved call by what it is written as, once. Two calls
        written apart (`a.add`, `b.add`) are two callees, whatever method
        name they end in."""
        callees = []
        for edge in self.outgoing[func_node_id]:
            relation, target = edge["relation"], edge["target"]
            if relation in ("CALLS", "CALLS_RESOLVED") and target not in callees:
                callees.append(target)
        return callees

    def get_ambiguous_callees(self, func_node_id):
        """``{REF: target: number of candidates}`` of the calls of
        ``func_node_id`` the resolver left unresolved, by what they are
        written as: the definitions any of them may reach, or none it found
        for any (0)."""
        reach = {}
        for edge in self.outgoing[func_node_id]:
            if edge["relation"] == "CALLS" and edge["target"].startswith("REF:"):
                candidates = edge.get("metadata", {}).get("candidates") or ()
                reach.setdefault(edge["target"], set()).update(candidates)
        return {target: len(candidates) for target, candidates in reach.items()}

    @classmethod
    def callee_note(cls, candidates):
        """The note on a callee: one of ``candidates`` definitions an
        unresolved call names, unresolved with none (0), or certain (None)."""
        return " (unresolved)" if candidates == 0 else cls.one_of(candidates)

    def find_call_path_result(
        self,
        start_name: str,
        end_name: str,
        max_depth: int = 10,
        direction: str = "forward",
    ) -> dict:
        """Find shortest call path between two symbols using BFS.

        Much more focused than trace_flow which shows entire call trees.
        Use for "how does X reach Y" questions.

        Args:
            start_name: Starting symbol name (e.g., 'handle_request')
            end_name: Target symbol name (e.g., 'validate_token')
            max_depth: Maximum path length (default: 10, at most MAX_PATH_DEPTH)
            direction: 'forward' (start calls end) or 'backward' (end calls start)

        Returns:
            Dict with:
            - status: 'found'; 'not_found' when the search completed without
              reaching the target; 'timeout' or 'node_budget' when it was cut
              off first, so a path may still exist
            - max_depth: the depth actually searched
            - path: list of hops (empty unless found), each containing:
            - caller_id: Full node ID of the caller
            - caller_name: Short name
            - callee_id: Full node ID of the callee
            - callee_name: Short name
            - call_line: Line number where call occurs
            - call_snippet: Code snippet around the call (3 lines)
            - file_path: File containing the call
            - candidates: on a hop through an unresolved call, how many
              definitions the call may reach; absent on a resolved call

        Example:
            find_call_path_result("create_order", "dispatch_order")
            # Returns the direct path with code snippets at each hop
        """
        from collections import deque

        # Cap visited-node expansion and wall time so a pathological graph
        # shape cannot stall the service however deep the caller asks to look.
        max_depth = min(max_depth, MAX_PATH_DEPTH)
        node_budget = self._max_nodes if self._max_nodes else 2000
        deadline = time.time() + (self._timeout_ms / 1000.0)

        def result(status, path=()):
            return {"status": status, "max_depth": max_depth, "path": list(path)}

        # Resolve start and end to node IDs
        start_nodes = self.find_nodes_by_name(start_name)
        end_nodes = self.find_nodes_by_name(end_name)

        if not start_nodes:
            return result("not_found")
        if not end_nodes:
            return result("not_found")

        start_ids = [n["id"] for n in start_nodes]
        end_ids = {n["id"] for n in end_nodes}

        # BFS with path tracking: first over the calls that are certain, then
        # also through unresolved calls, each reaching one of its candidates.
        passes = (False, True) if self.candidate_incoming else (False,)
        for through_candidates in passes:
            # (node_id, path_so_far)
            queue = deque((start_id, []) for start_id in start_ids)
            visited = set(start_ids)

            while queue:
                if time.time() > deadline:
                    return result("timeout")
                if len(visited) - len(start_ids) >= node_budget:
                    return result("node_budget")

                current_id, path = queue.popleft()

                # Get edges based on direction
                if direction == "forward":
                    edges = [
                        e
                        for e in self.outgoing.get(current_id, [])
                        if e["relation"] in ("CALLS", "CALLS_RESOLVED")
                    ]
                else:
                    incoming = self.incoming.get(current_id, [])
                    if through_candidates:
                        incoming = incoming + self.candidate_incoming.get(
                            current_id, []
                        )
                    edges = [
                        e
                        for e in incoming
                        if e["relation"] in ("CALLS", "CALLS_RESOLVED")
                    ]

                for edge in edges:
                    if direction == "backward":
                        reached = (edge["source"],)
                    elif not edge["target"].startswith("REF:"):
                        reached = (edge["target"],)
                    elif through_candidates:
                        # An unresolved call reaches one of its candidates.
                        reached = edge.get("metadata", {}).get("candidates", ())
                    else:
                        reached = ()

                    for next_id in reached:
                        if next_id in visited and next_id not in end_ids:
                            continue
                        visited.add(next_id)

                        # Build hop info
                        hop = self._build_hop_info(edge, current_id, next_id, direction)
                        new_path = path + [hop]

                        # Check if we reached the target
                        if next_id in end_ids:
                            return result("found", new_path)

                        if len(new_path) < max_depth:
                            queue.append((next_id, new_path))

        return result("not_found")

    def _build_hop_info(
        self, edge: dict, caller_id: str, callee_id: str, direction: str
    ) -> dict:
        """Build hop info dictionary with code snippet.

        Args:
            edge: The edge connecting caller to callee
            caller_id: Full node ID of the caller
            callee_id: Full node ID of the callee
            direction: 'forward' or 'backward' (affects which node is source)

        Returns:
            Dict with hop details including call site and snippet
        """
        # When direction is backward, the roles are reversed
        if direction == "backward":
            caller_id, callee_id = callee_id, caller_id

        caller_node = self.nodes.get(caller_id, {})
        callee_node = self.nodes.get(callee_id, {})

        caller_meta = caller_node.get("metadata", {})
        callee_meta = callee_node.get("metadata", {})

        call_line = edge.get("metadata", {}).get("lineno")
        file_path = caller_id.split("::")[0].replace("FILE:", "")

        # Get code snippet around call site (3 lines: before, call, after)
        snippet = ""
        if call_line and os.path.exists(self._disk_path(file_path)):
            snippet = self.get_source_segment(file_path, call_line - 1, call_line + 2)

        hop = {
            "caller_id": caller_id,
            "caller_name": caller_meta.get("name", caller_id.split("::")[-1]),
            "callee_id": callee_id,
            "callee_name": callee_meta.get("name", callee_id.split("::")[-1]),
            "call_line": call_line,
            "call_snippet": snippet.strip() if snippet else "",
            "file_path": file_path,
        }
        if edge["target"].startswith("REF:"):
            hop["candidates"] = len(edge["metadata"]["candidates"])
        return hop
