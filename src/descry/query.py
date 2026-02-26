import json
import logging
import sys
import argparse
import os
import math
import time
from collections import defaultdict
from functools import lru_cache
import re

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
MAX_INLINE_THRESHOLD = 300  # Maximum allowed inline threshold
MAX_CALLERS_SHOWN = 15  # Maximum callers to show in context

# Timeout (milliseconds)
TIMEOUT_MS = 4000  # Wall-clock timeout for recursive operations


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation: 4 chars ≈ 1 token)."""
    return len(text) // 4


# --- File Content LRU Cache ---
# Reduces I/O from ~150 to ~30 reads for deep context operations

@lru_cache(maxsize=128)
def _read_file_cached(file_path: str) -> tuple[str, ...]:
    """Read file content with LRU caching.

    Returns tuple of lines (tuple for hashability in cache).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return tuple(f.readlines())
    except Exception:
        return ()


def _clear_file_cache():
    """Clear the file content cache (useful after indexing)."""
    _read_file_cached.cache_clear()


def _get_syntax_lang(file_path: str) -> str:
    """Determine syntax highlighting language from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    lang_map = {
        ".rs": "rust",
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".svelte": "svelte",
        ".proto": "protobuf",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }
    return lang_map.get(ext, "")


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
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Insert underscore before uppercase letters that precede lowercase (for acronyms)
    # e.g., HTTPServer -> HTTP_Server -> http_server
    s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
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
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)", first_line)
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
    def __init__(self, graph_file):
        with open(graph_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.nodes = {n["id"]: n for n in self.data["nodes"]}
        self.edges = self.data["edges"]

        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)

        for edge in self.edges:
            self.outgoing[edge["source"]].append(edge)
            self.incoming[edge["target"]].append(edge)

        # Build IDF cache for TF-IDF search
        self._idf_cache = self._build_idf_cache()

        # Lazy filter indices - built on first use
        self._index_by_crate = None
        self._index_by_lang = None
        self._index_by_type = None
        self._index_is_test = None  # Test files and test functions
        self._indices_built = False

        # Memoization cache for recursive expansion
        self._expansion_cache = {}  # {(node_id, depth): (results, token_cost)}

    def get_source_segment(self, file_path, start_line, end_line):
        """Get source code segment using cached file content."""
        try:
            lines = _read_file_cached(file_path)
            if not lines:
                return "<Error: file not found or empty>"
            # line numbers are 1-based
            start = max(0, start_line - 1)
            end = min(len(lines), end_line)
            return "".join(lines[start:end])
        except Exception as e:
            return f"<Error reading file: {e}>"

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

        lang_map = {
            ".rs": "rust",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".svelte": "svelte",
            ".py": "python",
            ".proto": "protobuf",
        }

        # Test detection patterns
        test_path_patterns = (
            '/tests/', '/test/', '/_test/', '/spec/',
            '/testing/', '/fixtures/', '/mocks/', '/__tests__/',
        )
        test_file_suffixes = (
            '_test.rs', '.test.ts', '.spec.ts', '_test.py',
            '.test.js', '.spec.js', '.test.tsx', '.spec.tsx',
        )
        test_function_pattern = re.compile(r'^(test_|it_|describe_|spec_)')

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
                for ext, lang in lang_map.items():
                    if path.endswith(ext):
                        self._index_by_lang[lang].add(node_id)
                        break

                # Test detection: check path patterns and file suffixes
                is_test_file = (
                    any(p in path for p in test_path_patterns) or
                    any(path.endswith(s) for s in test_file_suffixes)
                )
                if is_test_file:
                    self._index_is_test.add(node_id)

            # Test detection: check function/method names
            if node_type in ("Function", "Method") and test_function_pattern.match(node_name):
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
        head_lines_only: int = None,
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
            all_lines = _read_file_cached(file_path)
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
                    "".join(head) +
                    f"\n    // ... ({remaining} more lines, use full=true to see all) ...\n"
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
                "".join(head) +
                f"\n    // ... ({omitted} lines omitted, use full=true to see complete source) ...\n\n" +
                "".join(tail)
            )

        except Exception as e:
            return f"<Error reading file: {e}>"

    def get_context_prompt(
        self,
        node_id,
        depth: int = 1,
        max_tokens: int = None,
        inline_threshold: int = None,
        full: bool = False,
        head_lines: int = None,
        expand_callees: bool = False,
        callee_budget: int = 2000,
        brief: bool = False,
        max_output_tokens: int = None,
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
        depth = min(max(depth, 1), MAX_DEPTH)  # Clamp to 1-3

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
        if not os.path.exists(file_path_rel):
            return f"ERROR: Source file '{file_path_rel}' not found on disk."

        # Determine syntax highlighting language
        source_lang = _get_syntax_lang(file_path_rel)

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
                doc_lines = [line.strip() for line in docstring.split("\n") if line.strip()]
                if doc_lines:
                    doc_preview = doc_lines[0][:120]

            # Count callers and callees
            callers_count = len(self.get_callers(meta.get("name", "")))
            callees_count = len(self.get_callees(node_id))

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

        source_code = self.get_smart_source(file_path_rel, start, end, token_count, full=full, head_lines_only=head_lines)

        # 2b. For Classes/Structs, include impl block methods
        impl_methods = []
        if node["type"] == "Class":
            # Find all methods that belong to this struct (children in the graph)
            for child_id, child_node in self.nodes.items():
                if child_node["type"] == "Method" and child_id.startswith(node_id + "::"):
                    child_meta = child_node["metadata"]
                    child_sig = child_meta.get("signature", child_meta.get("name", ""))
                    child_doc = child_meta.get("docstring", "").split("\n")[0][:80]
                    child_tokens = child_meta.get("token_count", 0)
                    impl_methods.append({
                        "name": child_meta.get("name", ""),
                        "signature": child_sig,
                        "docstring": child_doc,
                        "tokens": child_tokens,
                        "lineno": child_meta.get("lineno", 0),
                        "id": child_id,
                    })
            # Sort by line number
            impl_methods.sort(key=lambda m: m["lineno"])

        # 3. Find Dependencies (Callees) with Smart Inlining - deduplicated
        # Use recursive expansion if depth > 1
        if depth > 1:
            expanded = {node_id}  # Prevent cycles
            callee_info, _ = self._expand_callees_recursive(
                node_id,
                depth=depth,
                budget=max_tokens,
                expanded=expanded,
                current_depth=1,
                indent=0,
                inline_threshold=inline_threshold,
            )
        else:
            # Original single-level expansion
            callees = self.get_callees(node_id)
            callee_info = []

            for callee_id in callees:
                # Check if we have the node (resolved)
                callee_node = self.nodes.get(callee_id)

                if callee_node:
                    # Resolve file path for inlining
                    c_file = callee_node["id"].split("::")[0].replace("FILE:", "")
                    c_meta = callee_node["metadata"]
                    c_tokens = c_meta.get("token_count", 999)
                    c_lang = _get_syntax_lang(c_file)

                    # Header info
                    sig = c_meta.get("signature", f"{c_meta.get('name')}(...)")

                    # INLINING LOGIC: If small (< threshold tokens), show code.
                    if c_tokens < inline_threshold and os.path.exists(c_file):
                        c_start = c_meta.get("lineno", 1)
                        c_end = c_meta.get("end_lineno", c_start)
                        c_code = self.get_source_segment(c_file, c_start, c_end)
                        # Indent code content for proper markdown list nesting
                        c_code_indented = "\n".join("  " + line for line in c_code.split("\n"))
                        callee_info.append(
                            f"- **{c_meta.get('name')}** (Inlined, {c_tokens} toks):\n  ```{c_lang}\n{c_code_indented}\n  ```"
                        )
                    else:
                        # Summary mode
                        doc = c_meta.get("docstring", "").split("\n")[0]
                        callee_info.append(f"- `{sig}`\n  > {doc}")
                else:
                    # Unresolved REF - clean up for display
                    raw_name = callee_id.replace("REF:", "")
                    clean_name = _clean_ref_name(raw_name)
                    callee_info.append(f"- `{clean_name}` (External/Unresolved)")

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
                ex_lang = _get_syntax_lang(file_path)
                output.append(f"\n**From `{file_path}`:**")
                for ex in examples:
                    output.append(f"Line {ex['lineno']}:")
                    output.append(f"```{ex_lang}\n{ex['code']}\n```")
            output.append("")

        # Add explicit callers section
        if callers_summary:
            total_callers = len(self.get_callers(meta.get("name", "")))
            output.append(f"#### Callers ({len(callers_summary)} shown):")
            for caller in callers_summary:
                loc = f"{caller['file']}:{caller['lineno']}" if caller['lineno'] else caller['file']
                output.append(f"- `{caller['name']}` in {loc} ({caller['tokens']} toks)")
            if total_callers > len(callers_summary):
                output.append(f"  *({total_callers - len(callers_summary)} more callers)*")
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
                    output.append(f"\n**{exp['name']}** ({exp['file']}:{exp['lineno']}, {exp['tokens']} toks):")
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
                cfg_type = f" [{cfg['config_type']}]" if cfg['config_type'] else ""
                output.append(f"- L{cfg['lineno']}: `{cfg['signature']}`{cfg_type}")
                output.append(f"  Node: `{cfg['id']}`")
            output.append("")

        # Apply max_output_tokens progressive truncation if specified
        result = "\n".join(output)
        if max_output_tokens is not None and _estimate_tokens(result) > max_output_tokens:
            result = self._truncate_output_progressively(
                result, max_output_tokens, source_code, source_lang, file_path_rel, start
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
        for nid, node in self.nodes.items():
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
            for nid, node in self.nodes.items():
                if node["type"] in ("Function", "Method"):
                    name = node["metadata"]["name"]
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
        For example, when viewing setAuthTokenGetter, show the request interceptor that uses it.

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
                    configs.append({
                        "name": meta.get("name", "?"),
                        "signature": meta.get("signature", "?"),
                        "lineno": meta.get("lineno", 0),
                        "id": target_id,
                        "config_type": meta.get("config_type", ""),
                    })

        # Sort by line number
        configs.sort(key=lambda c: c["lineno"])
        return configs[:5]  # Limit to 5 config nodes

    def _truncate_output_progressively(
        self,
        result: str,
        max_tokens: int,
        source_code: str,
        source_lang: str,
        file_path: str,
        start_line: int,
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
            source_code: Original source code for fallback
            source_lang: Language for syntax highlighting
            file_path: File path for context
            start_line: Starting line number
        """
        import re

        current_tokens = _estimate_tokens(result)

        # Priority 1: Remove Expanded Callees section
        if current_tokens > max_tokens:
            result = re.sub(
                r'#### Expanded Callees \(Full Source\):.*?(?=####|\Z)',
                '',
                result,
                flags=re.DOTALL
            )
            result = re.sub(r'\n{3,}', '\n\n', result)
            current_tokens = _estimate_tokens(result)

        # Priority 2: Reduce Usage Examples to 2
        if current_tokens > max_tokens:
            # Find and truncate usage examples section
            match = re.search(r'(#### Usage Examples:.*?)(?=####|\Z)', result, re.DOTALL)
            if match:
                examples_section = match.group(1)
                # Keep header and first 2 code blocks
                code_blocks = list(re.finditer(r'```.*?```', examples_section, re.DOTALL))
                if len(code_blocks) > 2:
                    truncated_examples = examples_section[:code_blocks[1].end()]
                    truncated_examples += f"\n\n*({len(code_blocks) - 2} more examples omitted)*\n\n"
                    result = result[:match.start()] + truncated_examples + result[match.end():]
                    current_tokens = _estimate_tokens(result)

        # Priority 3: Remove Related Tests section
        if current_tokens > max_tokens:
            result = re.sub(
                r'#### Related Tests:.*?(?=####|\Z)',
                '',
                result,
                flags=re.DOTALL
            )
            result = re.sub(r'\n{3,}', '\n\n', result)
            current_tokens = _estimate_tokens(result)

        # Priority 4: Remove Related Configuration section
        if current_tokens > max_tokens:
            result = re.sub(
                r'#### Related Configuration.*?(?=####|\Z)',
                '',
                result,
                flags=re.DOTALL
            )
            result = re.sub(r'\n{3,}', '\n\n', result)
            current_tokens = _estimate_tokens(result)

        # Priority 5: Reduce Callers to 5
        if current_tokens > max_tokens:
            match = re.search(r'(#### Callers \(\d+ shown\):)(.*?)(?=####|\Z)', result, re.DOTALL)
            if match:
                callers_content = match.group(2)
                lines = callers_content.strip().split('\n')
                caller_lines = [line for line in lines if line.startswith('- `')]
                if len(caller_lines) > 5:
                    # Keep first 5 callers
                    new_content = '\n'.join(caller_lines[:5])
                    new_content += f"\n  *({len(caller_lines) - 5} more callers omitted)*\n"
                    result = result[:match.start()] + match.group(1) + '\n' + new_content + result[match.end():]
                    current_tokens = _estimate_tokens(result)

        # Priority 6: Reduce Dependencies to 5
        if current_tokens > max_tokens:
            match = re.search(r'(#### Dependencies \(Callees\):)(.*?)(?=####|\Z)', result, re.DOTALL)
            if match:
                deps_content = match.group(2)
                lines = deps_content.strip().split('\n')
                dep_lines = [line for line in lines if line.startswith('- ')]
                if len(dep_lines) > 5:
                    new_content = '\n'.join(dep_lines[:5])
                    new_content += f"\n*({len(dep_lines) - 5} more dependencies omitted)*\n"
                    result = result[:match.start()] + match.group(1) + '\n' + new_content + result[match.end():]
                    current_tokens = _estimate_tokens(result)

        # Priority 7: Truncate source code (last resort)
        if current_tokens > max_tokens:
            # Apply aggressive head_lines truncation
            match = re.search(r'(#### Source Code:\n```' + source_lang + r'\n)(.*?)(```)', result, re.DOTALL)
            if match:
                source = match.group(2)
                lines = source.split('\n')
                if len(lines) > 30:
                    truncated = '\n'.join(lines[:30])
                    truncated += f"\n// ... ({len(lines) - 30} lines omitted, max_output_tokens limit) ...\n"
                    result = result[:match.start()] + match.group(1) + truncated + match.group(3) + result[match.end():]

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
            if not os.path.exists(file_path):
                continue

            # Get source code
            start_line = meta.get("lineno", 1)
            end_line = meta.get("end_lineno", start_line + 50)
            source = self.get_source_segment(file_path, start_line, end_line)

            if source:
                expanded.append({
                    "name": meta.get("name", "?"),
                    "file": file_path,
                    "lineno": start_line,
                    "tokens": tokens,
                    "lang": _get_syntax_lang(file_path),
                    "source": source.rstrip(),
                })
                budget_remaining -= tokens

            # Stop if budget exhausted
            if budget_remaining <= 0:
                break

        return expanded

    def _find_similar_nodes(self, query: str, limit: int = 5) -> list[dict]:
        """Find nodes similar to the given query when exact match fails.

        Handles partial matches for:
        - Symbol name only (e.g., "authenticate" finds "FILE:auth.rs::authenticate")
        - Class::method format (e.g., "JwtAuth::from_request_parts")
        - Partial paths (e.g., "auth.rs::validate_token")
        - Naming convention variants (camelCase vs snake_case)

        Returns list of candidate nodes with match quality scores.
        """
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
            node_class = id_parts[-2] if len(id_parts) >= 2 and node["type"] == "Method" else None

            score = 0

            # Exact symbol name match (highest priority)
            if node_name == query_symbol:
                score += 100
            elif node_name_normalized == query_normalized:
                score += 80  # Naming convention variant

            # Partial symbol match
            elif query_normalized in node_name_normalized or node_name_normalized in query_normalized:
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
                candidates.append({
                    "id": node_id,
                    "name": node_name,
                    "type": node["type"],
                    "score": score,
                    "file": id_parts[0] if id_parts else "",
                    "lineno": meta.get("lineno", 0),
                })

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

        if len(similar) == 1 or (best["score"] >= 80 and best["score"] > second_score + 20):
            note = f"(Matched '{query}' → `{best['id']}`)"
            return best["id"], note

        # Multiple candidates - return suggestions
        suggestions = []
        for s in similar[:5]:
            suggestions.append(f"  - `{s['id']}` ({s['type']}, {s['file']}:{s['lineno']})")

        msg = f"Node '{query}' not found. Did you mean:\n" + "\n".join(suggestions)
        return None, msg

    def _get_usage_examples(self, target_node_id, limit=2):
        """
        Finds places where target_node_id is called and extracts code snippets.
        """
        target_node = self.nodes.get(target_node_id)
        if not target_node:
            return []

        target_name = target_node["metadata"]["name"]
        examples = []

        # We need to find edges that target REF:target_name or REF:parent.target_name
        # The graph stores targets as "REF:..."
        # We will scan edges.

        candidates = []

        # Optimization: Pre-filter edges by relation
        # But we need specific line numbers from the edges now.

        for edge in self.edges:
            if edge["relation"] == "CALLS":
                t = edge["target"].replace("REF:", "")
                # Flexible match: exact or attribute suffix
                if t == target_name or t.endswith(f".{target_name}"):
                    candidates.append(edge)

        # Sort candidates to prefer examples/tests if possible, or just by source path
        # Heuristic: put 'examples' or 'test' paths first
        def score_candidate(edge):
            src = edge["source"]
            if "example" in src:
                return 0
            if "test" in src:
                return 1
            return 2

        candidates.sort(key=score_candidate)

        for edge in candidates[:limit]:
            source_id = edge["source"]
            # Source ID is like FILE:path::Func
            file_path = source_id.split("::")[0].replace("FILE:", "")
            lineno = edge.get("metadata", {}).get("lineno")

            if lineno and os.path.exists(file_path):
                # Get context around the call (e.g. -1 to +1 lines)
                code = self.get_source_segment(file_path, lineno - 1, lineno + 2)
                examples.append(
                    {"file": file_path, "lineno": lineno, "code": code.strip()}
                )

        return examples

    def _get_callers_summary(self, target_node_id, limit=None):
        """Get summary of callers: name, file, lineno, tokens.

        Args:
            target_node_id: The node ID to find callers for
            limit: Maximum callers to return (defaults to MAX_CALLERS_SHOWN)

        Returns:
            List of dicts with caller info, sorted by token count (simpler first)
        """
        if limit is None:
            limit = MAX_CALLERS_SHOWN

        target_node = self.nodes.get(target_node_id)
        if not target_node:
            return []

        target_name = target_node.get("metadata", {}).get("name")
        if not target_name:
            return []

        # Use the optimized get_callers method
        caller_ids = self.get_callers(target_name)

        callers = []
        for caller_id in caller_ids:
            caller_node = self.nodes.get(caller_id)
            if caller_node:
                meta = caller_node.get("metadata", {})
                file_path = caller_id.split("::")[0].replace("FILE:", "")
                callers.append({
                    "id": caller_id,
                    "name": meta.get("name", "?"),
                    "file": file_path,
                    "lineno": meta.get("lineno", 0),
                    "tokens": meta.get("token_count", 0),
                    "type": caller_node.get("type", "?"),
                })

        # Sort by tokens (simpler callers first), then by name for consistency
        callers.sort(key=lambda c: (c["tokens"], c["name"]))
        return callers[:limit]

    def _expand_callees_recursive(
        self,
        node_id: str,
        depth: int,
        budget: int,
        expanded: set,
        current_depth: int = 1,
        indent: int = 0,
        start_time: float = None,
        inline_threshold: int = None,
    ) -> tuple[list[str], int]:
        """Recursively expand callees up to depth limit with token budget.

        Args:
            node_id: The node to expand callees for
            depth: Maximum recursion depth (1-3)
            budget: Remaining token budget
            expanded: Set of already expanded node IDs (cycle prevention)
            current_depth: Current recursion level
            indent: Current indentation level
            start_time: Start time for timeout checking
            inline_threshold: Token threshold for inlining code

        Returns:
            Tuple of (formatted lines list, remaining budget)
        """
        if start_time is None:
            start_time = time.time()
        if inline_threshold is None:
            inline_threshold = CALLEE_INLINE_THRESHOLD

        # Check timeout
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > TIMEOUT_MS:
            return [f"{'  ' * indent}*(timeout reached)*"], 0

        # Check depth limit
        if current_depth > depth or budget <= 0:
            return [], budget

        # Check memoization cache
        cache_key = (node_id, depth - current_depth + 1)  # Remaining depth
        if cache_key in self._expansion_cache:
            cached_results, cached_cost = self._expansion_cache[cache_key]
            if cached_cost <= budget:
                return list(cached_results), budget - cached_cost

        results = []
        tokens_used = 0
        callees = self.get_callees(node_id)[:MAX_CHILDREN_PER_LEVEL]

        for callee_id in callees:
            if callee_id in expanded:
                continue
            expanded.add(callee_id)

            # Check timeout periodically
            if (time.time() - start_time) * 1000 > TIMEOUT_MS:
                results.append(f"{'  ' * indent}*(timeout)*")
                break

            callee_node = self.nodes.get(callee_id)
            indent_str = "  " * indent

            if callee_node:
                # Resolved node
                c_file = callee_node["id"].split("::")[0].replace("FILE:", "")
                c_meta = callee_node.get("metadata", {})
                c_tokens = c_meta.get("token_count", 999)
                c_name = c_meta.get("name", "?")
                c_sig = c_meta.get("signature", f"{c_name}(...)")
                c_doc = (c_meta.get("docstring", "") or "").split("\n")[0][:80]
                c_lang = _get_syntax_lang(c_file)

                # Inline small functions if budget allows
                if c_tokens <= inline_threshold and c_tokens <= budget and os.path.exists(c_file):
                    c_start = c_meta.get("lineno", 1)
                    c_end = c_meta.get("end_lineno", c_start + 10)
                    code = self.get_source_segment(c_file, c_start, c_end)
                    # Indent code content for proper markdown list nesting
                    code_indent = indent_str + "  "
                    code_indented = "\n".join(code_indent + line for line in code.rstrip().split("\n"))
                    results.append(f"{indent_str}- **{c_name}** ({c_tokens} toks):")
                    results.append(f"{indent_str}  ```{c_lang}")
                    results.append(code_indented)
                    results.append(f"{indent_str}  ```")
                    tokens_used += c_tokens
                    budget -= c_tokens

                    # Recurse if depth allows
                    if current_depth < depth and budget > 0:
                        nested, budget = self._expand_callees_recursive(
                            callee_id, depth, budget, expanded,
                            current_depth + 1, indent + 1, start_time, inline_threshold
                        )
                        results.extend(nested)
                else:
                    # Summary mode for larger functions
                    results.append(f"{indent_str}- `{c_sig}` ({c_tokens} toks)")
                    if c_doc:
                        results.append(f"{indent_str}  > {c_doc}")
            else:
                # Unresolved REF
                raw_name = callee_id.replace("REF:", "")
                clean_name = _clean_ref_name(raw_name)
                results.append(f"{indent_str}- `{clean_name}` (External)")

        # Cache results
        self._expansion_cache[cache_key] = (results.copy(), tokens_used)

        return results, budget

    def trace_flow(
        self,
        start_name: str,
        direction: str = "forward",
        depth: int = 3,
        target: str = None,
        inline_threshold: int = 100,
        timeout_ms: int = None,
    ) -> str:
        """Trace call flow showing path with inline code.

        Args:
            start_name: Starting symbol name or node ID
            direction: 'forward' traces callees, 'backward' traces callers
            depth: Maximum depth (1-5, default: 3)
            target: Optional target to stop when reached
            inline_threshold: Inline code below this token count (default: 100)
            timeout_ms: Timeout in milliseconds (default: TIMEOUT_MS)

        Returns:
            Formatted flow trace as markdown string
        """
        if timeout_ms is None:
            timeout_ms = TIMEOUT_MS
        depth = min(depth, 5)  # Hard limit

        # Resolve start node
        start_nodes = self.find_nodes_by_name(start_name)
        if not start_nodes:
            # Try fuzzy match
            start_nodes = self.find_nodes_by_name(start_name, fuzzy=True)

        func_nodes = [n for n in start_nodes if n["type"] in ("Function", "Method")]
        if not func_nodes:
            if start_nodes:
                func_nodes = start_nodes[:1]  # Use first match even if not function
            else:
                return f"No function '{start_name}' found."

        start_node = func_nodes[0]
        start_id = start_node["id"]
        start_meta = start_node.get("metadata", {})

        output = [f"### Call Flow: `{start_meta.get('name', start_name)}` ({direction})"]
        output.append("")

        # Show disambiguation info if multiple matches
        if len(func_nodes) > 1:
            start_file = start_id.split("::")[0].replace("FILE:", "")
            output.append(f"**Note:** Found {len(func_nodes)} matches for `{start_name}`. Using:")
            output.append(f"  `{start_id}`")
            output.append(f"  ({start_file})")
            output.append("")
            output.append("**Alternatives** (use full node ID to select):")
            for alt in func_nodes[1:5]:  # Show up to 4 alternatives
                output.append(f"  - `{alt['id']}`")
            if len(func_nodes) > 5:
                output.append(f"  - ... and {len(func_nodes) - 5} more")
            output.append("")

        visited = set()
        path_to_target = []
        start_time = time.time()
        nodes_visited = 0

        def trace(node_id: str, current_depth: int, path: list) -> bool:
            nonlocal nodes_visited

            # Check limits
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                output.append("  " * (current_depth - 1) + "*(timeout)*")
                return False
            if current_depth > depth or node_id in visited:
                return False
            if nodes_visited >= MAX_NODES_PER_OPERATION:
                output.append("  " * (current_depth - 1) + "*(node limit)*")
                return False

            visited.add(node_id)
            nodes_visited += 1

            node = self.nodes.get(node_id)
            if not node:
                return False

            meta = node.get("metadata", {})
            name = meta.get("name", "?")
            tokens = meta.get("token_count", 0)
            node_type = node.get("type", "?")[:3]

            # Check target reached
            if target and name == target:
                path_to_target.extend(path + [node_id])
                return True

            # Get next level
            if direction == "forward":
                next_ids = self.get_callees(node_id)[:MAX_CHILDREN_PER_LEVEL]
            else:
                next_ids = self.get_callers(name)[:MAX_CHILDREN_PER_LEVEL]

            # Format node
            indent = "  " * (current_depth - 1)
            arrow = "├─" if current_depth > 1 else ""
            output.append(f"{indent}{arrow}[{node_type}] **{name}** ({tokens} toks)")

            # Inline small functions
            if tokens <= inline_threshold and tokens > 0:
                file_path = node_id.split("::")[0].replace("FILE:", "")
                if os.path.exists(file_path):
                    start_line = meta.get("lineno", 1)
                    end_line = meta.get("end_lineno", start_line + 10)
                    code = self.get_source_segment(file_path, start_line, end_line)
                    lang = _get_syntax_lang(file_path)
                    output.append(f"{indent}  ```{lang}")
                    output.append(code.rstrip())
                    output.append(f"{indent}  ```")

            # Recurse (limit branching)
            for next_id in next_ids:
                if trace(next_id, current_depth + 1, path + [node_id]):
                    return True
            return False

        trace(start_id, 1, [])

        if target and path_to_target:
            output.append("")
            output.append(f"#### Path to `{target}`:")
            path_names = [p.split("::")[-1] for p in path_to_target]
            output.append(" -> ".join(path_names))

        return "\n".join(output)

    def trace_flow_structured(
        self,
        start_name: str,
        direction: str = "forward",
        depth: int = 3,
        target: str = None,
        inline_threshold: int = 100,
        timeout_ms: int = None,
    ) -> dict:
        """Trace call flow, returning a structured tree for UI rendering.

        Returns dict with keys: root, alternatives (optional), note (optional).
        Each tree node has: id, name, type, tokens, file, line, code, lang, children.
        """
        if timeout_ms is None:
            timeout_ms = TIMEOUT_MS
        depth = min(depth, 5)

        # Resolve start node
        start_nodes = self.find_nodes_by_name(start_name)
        if not start_nodes:
            start_nodes = self.find_nodes_by_name(start_name, fuzzy=True)

        func_nodes = [n for n in start_nodes if n["type"] in ("Function", "Method")]
        if not func_nodes:
            if start_nodes:
                func_nodes = start_nodes[:1]
            else:
                return {"error": f"No function '{start_name}' found."}

        start_node = func_nodes[0]
        start_id = start_node["id"]

        # Build alternatives list for disambiguation
        alternatives = []
        note = None
        if len(func_nodes) > 1:
            start_file = start_id.split("::")[0].replace("FILE:", "")
            note = (
                f"Found {len(func_nodes)} matches for \"{start_name}\". "
                f"Using: {start_id} ({start_file})"
            )
            for alt in func_nodes[1:6]:
                alt_meta = alt.get("metadata", {})
                alt_file = alt["id"].split("::")[0].replace("FILE:", "")
                alternatives.append({
                    "id": alt["id"],
                    "name": alt_meta.get("name", "?"),
                    "file": alt_file,
                })

        visited = set()
        start_time = time.time()
        nodes_visited = 0

        def build_tree(node_id: str, current_depth: int) -> dict | None:
            nonlocal nodes_visited

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                return None
            if current_depth > depth or node_id in visited:
                return None
            if nodes_visited >= MAX_NODES_PER_OPERATION:
                return None

            visited.add(node_id)
            nodes_visited += 1

            node = self.nodes.get(node_id)
            if not node:
                return None

            meta = node.get("metadata", {})
            name = meta.get("name", "?")
            tokens = meta.get("token_count", 0)
            node_type = node.get("type", "?")[:3]
            file_path = node_id.split("::")[0].replace("FILE:", "")
            line = meta.get("lineno")
            lang = _get_syntax_lang(file_path)

            # Inline small functions
            code = None
            if tokens <= inline_threshold and tokens > 0:
                if os.path.exists(file_path):
                    start_line = meta.get("lineno", 1)
                    end_line = meta.get("end_lineno", start_line + 10)
                    code = self.get_source_segment(file_path, start_line, end_line)

            # Recurse into children
            if direction == "forward":
                next_ids = self.get_callees(node_id)[:MAX_CHILDREN_PER_LEVEL]
            else:
                next_ids = self.get_callers(name)[:MAX_CHILDREN_PER_LEVEL]

            children = []
            for next_id in next_ids:
                child = build_tree(next_id, current_depth + 1)
                if child:
                    children.append(child)

            return {
                "id": node_id,
                "name": name,
                "type": node_type,
                "tokens": tokens,
                "file": file_path,
                "line": line,
                "code": code,
                "lang": lang,
                "children": children,
            }

        root = build_tree(start_id, 1)

        result = {"root": root}
        if note:
            result["note"] = note
        if alternatives:
            result["alternatives"] = alternatives

        return result

    def flatten_class(self, class_node_id):
        """
        Traverses INHERITS edges to build a full list of methods for a class.
        """
        class_node = self.nodes.get(class_node_id)
        if not class_node:
            return f"ERROR: Node '{class_node_id}' not found in graph."
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
                if class_matches:
                    current_node = class_matches[0]
                else:
                    # External class or not found
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
        output.append(f"### Flattened View: `{class_node['metadata']['name']}`")
        output.append("#### Inheritance Chain:")
        output.append(
            " -> ".join(
                [c.get("metadata", {}).get("name", c.get("name")) for c in chain]
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
                        name = child["metadata"]["name"]
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
            tokens = set(re.findall(r"[a-z_][a-z0-9_]*", text.lower()))
            for token in tokens:
                doc_freq[token] += 1

        # Compute IDF with smoothing to avoid log(0)
        idf = {}
        for term, df in doc_freq.items():
            # Add 1 to df for smoothing (avoids division issues)
            idf[term] = math.log((total_docs + 1) / (df + 1)) + 1
        return idf

    def search_docs(self, query_terms, lang=None, crate=None, symbol_type=None, exclude_tests=False):
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
        terms = [t.lower() for t in query_terms]
        num_terms = len(terms)

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
            if lang and lang != "all" and lang in self._index_by_lang:
                candidate_ids &= self._index_by_lang[lang]
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
            name_tokens = re.findall(r"[a-z_][a-z0-9_]*", name)
            doc_tokens = re.findall(r"[a-z_][a-z0-9_]*", doc)
            full_tokens = re.findall(r"[a-z_][a-z0-9_]*", full_text)

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
                    phrase = "_".join(terms)  # e.g., "deployment_dispatch"
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
        - Qualified name: "ThoraxServer::start" matches node IDs ending with that pattern
        - File path suffix: "server.rs" matches files ending with that path
        - Partial qualified: "auth::check_permission" matches IDs containing that pattern
        - Fuzzy/prefix matching (when fuzzy=True): "validate_token" matches "validate_token_format"

        Args:
            name: Name or pattern to search for
            fuzzy: If True, also matches names that start with or contain the search term
        """
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

            # Exact name match
            if node_name == name:
                exact_matches.append(node)
                continue

            # Naming convention variant match (e.g., getClient matches get_client)
            if node_name and (node_name_normalized in name_variants or
                             node_name.lower() in [v.lower() for v in name_variants]):
                variant_matches.append(node)
                continue

            # File path suffix match
            if str(meta.get("path", "")).endswith(name):
                exact_matches.append(node)
                continue

            # Qualified name match (e.g., "ThoraxServer::start")
            if "::" in name:
                # Check if node ID ends with the qualified name
                if node_id.endswith(f"::{name}") or node_id.endswith(name):
                    exact_matches.append(node)
                    continue
                # Check if the qualified pattern appears in the node ID
                if name in node_id:
                    exact_matches.append(node)
                    continue
                # Also try with normalized variants
                for variant in name_variants:
                    if variant in node_id.lower():
                        variant_matches.append(node)
                        break

            # Fuzzy matching: prefix or contains (for simple names)
            if fuzzy and "::" not in name and len(name) >= 3:
                # Prefix match on node name (e.g., "validate_token" matches "validate_token_format")
                if node_name.startswith(name):
                    fuzzy_matches.append(node)
                    continue
                # Contains match in node ID (for cases like searching "dispatch" finding "dispatch_to_handler")
                if f"::{name}" in node_id.lower() or node_id.lower().endswith(name.lower()):
                    fuzzy_matches.append(node)
                    continue

        # Return exact matches first, then variant matches, then fuzzy if no others
        if exact_matches:
            return exact_matches
        if variant_matches:
            return variant_matches
        return fuzzy_matches

    def find_trait_impls(self, method_name: str, trait_name: str = None) -> list:
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
            # Returns JwtAuth, OptionalJwtAuth, SseAuth, Pagination, etc.

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
            if impl_trait:
                # If trait_name filter is specified, check it matches
                if trait_name is None or impl_trait == trait_name:
                    results.append(node)

        return results

    def get_callers(self, func_name, fuzzy=False):
        """Find all callers of a function or users of a struct/class.

        OPTIMIZED: Uses incoming index for O(in-degree) instead of O(E) scan.

        Supports:
        - Simple name: "start" matches calls to any function named start
        - Qualified name: "ThoraxServer::start" matches that specific method
        - Full node ID: "FILE:path::Class::method" for exact matching
        - Struct names: includes INSTANTIATES edges for struct usage tracking
        - Fuzzy matching (when fuzzy=True): "validate_token" matches "validate_token_format"

        Args:
            func_name: Function/method name to find callers for
            fuzzy: If True, also matches names that start with the search term
        """
        # Include INSTANTIATES for struct/class usage tracking
        relevant_relations = {
            "CALLS",
            "CALLS_RESOLVED",
            "INSTANTIATES",
            "INSTANTIATES_RESOLVED",
        }

        # Step 1: Find all target node IDs that match func_name
        target_ids = self._resolve_target_ids(func_name, fuzzy)

        # Step 2: Use incoming index to find callers - O(in-degree) per target
        callers = set()
        for target_id in target_ids:
            for edge in self.incoming.get(target_id, []):
                if edge["relation"] in relevant_relations:
                    callers.add(edge["source"])

        return list(callers)

    def _resolve_target_ids(self, func_name: str, fuzzy: bool = False) -> set:
        """Resolve a function name to all possible target IDs in the graph.

        Returns node IDs and REF: IDs that match the given name.
        """
        target_ids = set()
        base_name = func_name.split("::")[-1].split(".")[-1]

        # Direct matches - these are the most common
        target_ids.add(func_name)
        target_ids.add(f"REF:{func_name}")
        target_ids.add(f"REF:{base_name}")

        # Find resolved nodes that match the name
        matches = self.find_nodes_by_name(func_name, fuzzy=fuzzy)
        for match in matches:
            target_ids.add(match["id"])

        # For qualified names, also add the qualified REF
        if "::" in func_name:
            target_ids.add(f"REF:{func_name}")

        # Build additional targets from existing nodes/edges that end with our name
        # This catches cases like REF:Foo::Bar when searching for "Bar"
        for node_id in self.nodes.keys():
            if node_id.endswith(f"::{func_name}") or node_id.endswith(f"::{base_name}"):
                target_ids.add(node_id)

        # Also check for REF targets in incoming edges (for unresolved calls)
        # This is still needed because REF: targets may not be nodes themselves
        for target_id in list(self.incoming.keys()):
            if target_id.startswith("REF:"):
                ref_name = target_id.replace("REF:", "")
                # Exact match
                if ref_name == func_name or ref_name == base_name:
                    target_ids.add(target_id)
                # Suffix match
                elif ref_name.endswith(f"::{func_name}") or ref_name.endswith(f".{func_name}"):
                    target_ids.add(target_id)
                elif ref_name.endswith(f"::{base_name}") or ref_name.endswith(f".{base_name}"):
                    target_ids.add(target_id)
                # Fuzzy prefix match
                elif fuzzy and len(func_name) >= 3:
                    clean_ref = ref_name.split("::")[-1].split(".")[-1]
                    if clean_ref.startswith(func_name):
                        target_ids.add(target_id)

        return target_ids

    def get_callees(self, func_node_id):
        """Get unique callees for a function, deduplicated.

        For unresolved references (REF:...), deduplicates by cleaned/normalized name
        to avoid showing iter_mut and plan.steps.iter_mut as separate entries,
        and to handle duplicates like Uuid::parse_str appearing multiple times.
        """
        seen_ids = set()  # Full target IDs
        seen_clean_names = set()  # Cleaned names for unresolved refs
        callees = []

        for edge in self.outgoing[func_node_id]:
            # Updated to support both resolved and unresolved edges
            if edge["relation"] in ("CALLS", "CALLS_RESOLVED"):
                target = edge["target"]

                # For resolved references, dedupe by exact ID
                if not target.startswith("REF:"):
                    if target not in seen_ids:
                        seen_ids.add(target)
                        callees.append(target)
                else:
                    # For unresolved refs, use cleaned name for deduplication
                    # This handles both method chains (a.b.method) and qualified names (Type::method)
                    ref_name = target.replace("REF:", "")
                    clean_name = _clean_ref_name(ref_name)

                    if clean_name not in seen_clean_names:
                        seen_clean_names.add(clean_name)
                        callees.append(target)

        return callees

    def find_call_path(
        self,
        start_name: str,
        end_name: str,
        max_depth: int = 10,
        direction: str = "forward"
    ) -> list:
        """Find shortest call path between two symbols using BFS.

        Much more focused than trace_flow which shows entire call trees.
        Use for "how does X reach Y" questions.

        Args:
            start_name: Starting symbol name (e.g., 'handle_request')
            end_name: Target symbol name (e.g., 'validate_token')
            max_depth: Maximum path length (default: 10)
            direction: 'forward' (start calls end) or 'backward' (end calls start)

        Returns:
            List of hops, each containing:
            - caller_id: Full node ID of the caller
            - caller_name: Short name
            - callee_id: Full node ID of the callee
            - callee_name: Short name
            - call_line: Line number where call occurs
            - call_snippet: Code snippet around the call (3 lines)
            - file_path: File containing the call

        Example:
            find_call_path("create_deployment", "dispatch_to_handler")
            # Returns the direct path with code snippets at each hop
        """
        from collections import deque

        # Resolve start and end to node IDs
        start_nodes = self.find_nodes_by_name(start_name)
        end_nodes = self.find_nodes_by_name(end_name)

        if not start_nodes:
            return []
        if not end_nodes:
            return []

        start_id = start_nodes[0]["id"]
        end_ids = {n["id"] for n in end_nodes}

        # BFS with path tracking
        queue = deque([(start_id, [])])  # (node_id, path_so_far)
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            # Get edges based on direction
            if direction == "forward":
                edges = [e for e in self.outgoing.get(current_id, [])
                        if e["relation"] in ("CALLS", "CALLS_RESOLVED")]
            else:
                edges = [e for e in self.incoming.get(current_id, [])
                        if e["relation"] in ("CALLS", "CALLS_RESOLVED")]

            for edge in edges:
                next_id = edge["target"] if direction == "forward" else edge["source"]

                # Try to resolve REF: nodes to actual nodes
                if next_id.startswith("REF:"):
                    ref_name = next_id.replace("REF:", "")
                    # Try to find matching node
                    resolved = self.find_nodes_by_name(ref_name)
                    if resolved:
                        next_id = resolved[0]["id"]
                    else:
                        continue  # Skip unresolved references

                if next_id in visited:
                    continue
                visited.add(next_id)

                # Build hop info
                hop = self._build_hop_info(edge, current_id, next_id, direction)
                new_path = path + [hop]

                # Check if we reached the target
                if next_id in end_ids:
                    return new_path

                if len(new_path) < max_depth:
                    queue.append((next_id, new_path))

        return []  # No path found

    def _build_hop_info(self, edge: dict, caller_id: str, callee_id: str, direction: str) -> dict:
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
        if call_line and os.path.exists(file_path):
            snippet = self.get_source_segment(file_path, call_line - 1, call_line + 2)

        return {
            "caller_id": caller_id,
            "caller_name": caller_meta.get("name", caller_id.split("::")[-1]),
            "callee_id": callee_id,
            "callee_name": callee_meta.get("name", callee_id.split("::")[-1]),
            "call_line": call_line,
            "call_snippet": snippet.strip() if snippet else "",
            "file_path": file_path,
        }


def _find_default_graph() -> str:
    """Find the default graph file by searching for project markers."""
    from pathlib import Path

    # Check CODEGRAPH_CACHE_DIR first
    cache_dir_env = os.environ.get("CODEGRAPH_CACHE_DIR")
    if cache_dir_env:
        graph_path = Path(cache_dir_env) / "codebase_graph.json"
        if graph_path.exists():
            return str(graph_path)

    # Search for project root markers
    candidates = [Path.cwd()] + list(Path.cwd().parents)
    for path in candidates:
        if path == Path.home():
            break
        # Check for project markers
        if (path / ".mcp.json").exists() or (path / "Cargo.toml").exists():
            graph_path = path / ".codegraph_cache" / "codebase_graph.json"
            return str(graph_path)

    # Fall back to current directory
    return str(Path.cwd() / ".codegraph_cache" / "codebase_graph.json")


def _handle_status(graph_file: str) -> None:
    """Handle the status command - show graph and feature status."""
    from pathlib import Path
    import time

    graph_path = Path(graph_file)
    cache_dir = graph_path.parent

    print("Codegraph Status")
    print("=" * 50)

    # Graph status
    if graph_path.exists():
        mtime = graph_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600

        with open(graph_path) as f:
            data = json.load(f)
        nodes = len(data.get("nodes", []))
        edges = len(data.get("edges", []))

        stale = " (STALE)" if age_hours > 24 else ""
        print(f"Graph: {nodes:,} nodes, {edges:,} edges")
        print(f"Updated: {age_hours:.1f} hours ago{stale}")
        print(f"Size: {graph_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"Graph: NOT FOUND at {graph_path}")
        print("Run 'query_graph.py index' to generate.")
        return

    # SCIP status
    print()
    scip_dir = cache_dir / "scip"
    if os.environ.get("CODEGRAPH_NO_SCIP", "").lower() in ("1", "true"):
        print("SCIP: Disabled (CODEGRAPH_NO_SCIP=1)")
    elif scip_dir.exists():
        scip_files = list(scip_dir.glob("*.scip"))
        if scip_files:
            print(f"SCIP: Enabled ({len(scip_files)} project(s) indexed)")
            for sf in scip_files[:5]:
                print(f"  - {sf.name}")
            if len(scip_files) > 5:
                print(f"  ... and {len(scip_files) - 5} more")
        else:
            print("SCIP: Directory exists but no .scip files")
    else:
        print("SCIP: Not available (no scip/ directory)")

    # Embeddings status
    print()
    if os.environ.get("CODEGRAPH_NO_EMBEDDINGS", "").lower() in ("1", "true"):
        print("Embeddings: Disabled (CODEGRAPH_NO_EMBEDDINGS=1)")
    else:
        emb_files = list(cache_dir.glob("embeddings_*.npz"))
        if emb_files:
            emb_file = emb_files[0]
            emb_age = (time.time() - emb_file.stat().st_mtime) / 3600
            print(f"Embeddings: Ready ({emb_file.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  Updated: {emb_age:.1f} hours ago")
        else:
            print("Embeddings: Not generated")
            print("  Will build on first semantic search")

    print()
    print(f"Cache dir: {cache_dir}")


def _handle_index(graph_file: str, root_path: str) -> None:
    """Handle the index command - regenerate the graph."""
    import subprocess
    from pathlib import Path

    script_dir = Path(__file__).parent
    generate_script = script_dir / "generate_codegraph.py"

    if not generate_script.exists():
        print(f"Error: generate_codegraph.py not found at {generate_script}")
        sys.exit(1)

    print(f"Regenerating codebase graph from {root_path}...")
    print("This may take a few minutes for large codebases.")
    print()

    try:
        result = subprocess.run(
            ["uv", "run", str(generate_script), root_path],
            cwd=root_path if root_path != "." else None,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode == 0:
            print("\nIndex complete. Run 'status' to see details.")
        else:
            print(f"\nIndex failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("\nError: Index timed out after 10 minutes.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'uv' not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def _handle_semantic(graph_file: str, query: str, limit: int) -> None:
    """Handle the semantic command - search using embeddings."""
    from pathlib import Path

    # Try to import embeddings
    if os.environ.get("CODEGRAPH_NO_EMBEDDINGS", "").lower() in ("1", "true"):
        print("Semantic search disabled (CODEGRAPH_NO_EMBEDDINGS=1)")
        print("Falling back to keyword search...")
        # Fall back to keyword search
        q = GraphQuerier(graph_file)
        results = q.search_docs(query.split())
        for node in results[:limit]:
            meta = node["metadata"]
            print(f"\n[{node['type']}] {meta.get('name')}")
            print(f"  ID: {node['id']}")
        return

    try:
        from descry.embeddings import embeddings_available, SemanticSearcher

        if not embeddings_available():
            print("Embeddings not available (sentence-transformers not installed)")
            print("Install with: pip install sentence-transformers numpy")
            sys.exit(1)
    except ImportError:
        print("Embeddings module not found.")
        print("Install with: pip install sentence-transformers numpy")
        sys.exit(1)

    graph_path = Path(graph_file)
    if not graph_path.exists():
        print(f"Graph not found at {graph_path}")
        print("Run 'query_graph.py index' first.")
        sys.exit(1)

    print(f"Searching for: {query}")
    print("(First search may take a moment to load embeddings...)")
    print()

    try:
        searcher = SemanticSearcher(str(graph_path))
        results = searcher.search(query, limit=limit)

        if not results:
            print("No semantic matches found.")
            return

        print(f"Found {len(results)} semantic matches:\n")
        for node, score in results:
            meta = node.get("metadata", {})
            print(f"[{score:.3f}] [{node['type'][:3]}] {meta.get('name', 'unknown')}")
            print(f"        {node['id']}")
            if meta.get("docstring"):
                doc_line = meta["docstring"].split("\n")[0][:60]
                print(f"        {doc_line}...")
            print()
    except Exception as e:
        print(f"Semantic search error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Query the codebase knowledge graph",
        epilog="""
Examples:
  # Check status and feature availability
  %(prog)s status

  # Find callers of a function (with SCIP resolution + fuzzy matching)
  %(prog)s callers run_migrations

  # Find what a function calls
  %(prog)s callees "AppState::new"

  # Keyword search (TF-IDF)
  %(prog)s search dispatch queue

  # Semantic search (requires embeddings)
  %(prog)s semantic "authentication flow"

  # Regenerate the graph
  %(prog)s index

Environment variables:
  CODEGRAPH_CACHE_DIR    Override cache location (default: .codegraph_cache)
  CODEGRAPH_LOG_LEVEL    Logging verbosity (DEBUG/INFO/WARNING/ERROR)
  CODEGRAPH_NO_SCIP      Set to 1 to disable SCIP resolution
  CODEGRAPH_NO_EMBEDDINGS  Set to 1 to disable semantic search

Run from project root or set CODEGRAPH_CACHE_DIR.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-g", "--graph",
        dest="graph_file",
        default=None,
        help="Path to codebase_graph.json (auto-detected if not provided)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Status commands ---
    subparsers.add_parser("status", help="Check graph status and feature availability")

    p_index = subparsers.add_parser(
        "index", help="Regenerate the codebase graph"
    )
    p_index.add_argument(
        "--path", default=".", help="Root path to index (default: current directory)"
    )

    # --- Query commands ---
    p_callers = subparsers.add_parser(
        "callers", help="Find who calls a function (uses SCIP + fuzzy matching)"
    )
    p_callers.add_argument("name", help="Name of the function called")

    p_callees = subparsers.add_parser(
        "callees", help="Find what a function calls"
    )
    p_callees.add_argument("name", help="Name of the caller function")

    # --- Search commands ---
    p_search = subparsers.add_parser(
        "search", help="Search names and docstrings (TF-IDF keyword search)"
    )
    p_search.add_argument("terms", nargs="+", help="Keywords to search")

    p_semantic = subparsers.add_parser(
        "semantic", help="Semantic search using embeddings (natural language)"
    )
    p_semantic.add_argument("query", help="Natural language query")
    p_semantic.add_argument(
        "--limit", type=int, default=10, help="Max results (default: 10)"
    )

    # --- Info commands ---
    p_info = subparsers.add_parser("info", help="Get detailed info about a node")
    p_info.add_argument("node_id", help="Exact Node ID")

    p_struct = subparsers.add_parser("structure", help="Show structure of a file")
    p_struct.add_argument("file", help="Filename (e.g., server.rs)")

    p_context = subparsers.add_parser(
        "context", help="Get full context (source + dependencies) for a node"
    )
    p_context.add_argument("node_id", help="Exact Node ID")

    p_flatten = subparsers.add_parser("flatten", help="Flatten class hierarchy")
    p_flatten.add_argument("node_id", help="Class Node ID")

    args = parser.parse_args()

    # Determine graph file path
    graph_file = args.graph_file or _find_default_graph()

    # Handle commands that don't need the graph loaded
    if args.command == "status":
        _handle_status(graph_file)
        return

    if args.command == "index":
        _handle_index(graph_file, args.path)
        return

    # Load graph for other commands
    try:
        q = GraphQuerier(graph_file)
    except FileNotFoundError:
        print(f"Error: Could not find {graph_file}")
        print("Run 'query_graph.py index' to generate the graph.")
        sys.exit(1)

    if args.command == "semantic":
        _handle_semantic(graph_file, args.query, args.limit)
        return

    if args.command == "callers":
        results = q.get_callers(args.name)
        fuzzy_note = ""
        # Try fuzzy matching if no exact matches
        if not results:
            results = q.get_callers(args.name, fuzzy=True)
            if results:
                fuzzy_note = " (fuzzy match)"
        if not results:
            print(f"No callers found for '{args.name}'")
        else:
            print(f"Callers of '{args.name}'{fuzzy_note}:")
            for r in sorted(results):
                # Get line number from node metadata
                node_info = q.get_node_info(r)
                if node_info:
                    lineno = node_info.get("metadata", {}).get("lineno")
                    if lineno:
                        file_path = r.split("::")[0].replace("FILE:", "")
                        print(f"  - {r} ({file_path}:{lineno})")
                    else:
                        print(f"  - {r}")
                else:
                    print(f"  - {r}")

    elif args.command == "structure":
        # Find file node
        matches = q.find_nodes_by_name(args.file)
        file_matches = [m for m in matches if m["type"] == "File"]

        if not file_matches:
            print(f"File '{args.file}' not found in graph.")
        else:
            node_id = file_matches[0]["id"]
            print(f"Structure of {node_id}:")

            # Imports
            imports = []
            for edge in q.outgoing[node_id]:
                if edge["relation"] == "IMPORTS":
                    imports.append(edge["target"].replace("MODULE:", ""))

            if imports:
                print("  Imports:")
                for i in sorted(imports):
                    print(f"    {i}")

            # Definitions (Classes/Functions/Constants)
            defs = []
            for edge in q.outgoing[node_id]:
                if edge["relation"] == "DEFINES":
                    defs.append(q.nodes[edge["target"]])

            classes = [d["metadata"]["name"] for d in defs if d["type"] == "Class"]
            functions = [d["metadata"]["name"] for d in defs if d["type"] == "Function"]
            constants = [d["metadata"]["name"] for d in defs if d["type"] == "Constant"]

            if constants:
                print("  Constants:")
                for c in sorted(constants):
                    print(f"    {c}")
            if classes:
                print("  Classes:")
                for c in sorted(classes):
                    print(f"    {c}")
            if functions:
                print("  Functions:")
                for f in sorted(functions):
                    print(f"    {f}")

    elif args.command == "callees":
        matches = q.find_nodes_by_name(args.name)
        func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
        if not func_matches:
            print(f"No function found for '{args.name}'")
        else:
            node = func_matches[0]
            print(f"'{node['id']}' calls:")
            callees = q.get_callees(node["id"])
            for c in sorted(callees):
                # Clean up unresolved refs for better readability
                if c.startswith("REF:"):
                    clean_name = _clean_ref_name(c.replace("REF:", ""))
                    print(f"  - {clean_name} (External)")
                else:
                    print(f"  - {c}")

    elif args.command == "search":
        results = q.search_docs(args.terms)
        print(f"Found {len(results)} matches for '{' '.join(args.terms)}':")
        for node in results[:10]:  # Top 10
            meta = node["metadata"]
            print(f"\n[{node['type']}] {meta.get('name')}")
            print(f"  ID: {node['id']}")
            if meta.get("signature"):
                print(f"  Sig: {meta['signature']}")
            if meta.get("docstring"):
                print(f"  Doc: {meta['docstring'].splitlines()[0]}...")
            if node["type"] == "Constant":
                print(f"  Val: {meta.get('value')}")

    elif args.command == "info":
        node = q.get_node_info(args.node_id)
        if node:
            print(json.dumps(node, indent=2))
        else:
            print("Node not found.")

    elif args.command == "context":
        print(q.get_context_prompt(args.node_id))

    elif args.command == "flatten":
        print(q.flatten_class(args.node_id))


if __name__ == "__main__":
    main()
