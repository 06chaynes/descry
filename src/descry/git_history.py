#!/usr/bin/env python3
"""
Git History Analysis Module

Provides symbol-level git history analysis by combining git operations
with descry's structural knowledge. Enables churn hotspot detection,
symbol evolution tracking, and change impact analysis.

Usage:
    from git_history import GitHistoryAnalyzer, GitError
    analyzer = GitHistoryAnalyzer("/path/to/repo")
    churn = analyzer.get_churn(time_range="last 30 days", limit=20)
"""

import logging
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Raised for git-specific errors (not installed, not a repo, etc.)."""
    pass


DEFAULT_CHURN_EXCLUSIONS = [
    ".descry_cache/",
    ".beads/",
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
]

CODE_EXTENSIONS = {
    ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".svelte",
    ".go", ".java", ".css", ".scss", ".html",
}


class GitHistoryAnalyzer:
    """Analyzes git history at the symbol level using descry metadata."""

    def __init__(self, project_root: str, graph_querier=None, churn_exclusions=None, code_extensions=None, git_timeout=30):
        """Initialize the analyzer.

        Args:
            project_root: Path to the git repository root.
            graph_querier: Optional GraphQuerier instance for symbol resolution.
            churn_exclusions: List of path patterns to exclude from churn analysis.
                Defaults to DEFAULT_CHURN_EXCLUSIONS.
            code_extensions: Set of file extensions considered as code.
                Defaults to CODE_EXTENSIONS.
            git_timeout: Default timeout in seconds for git commands.
        """
        self.project_root = Path(project_root).resolve()
        self.querier = graph_querier
        self._verified = False
        self.churn_exclusions = churn_exclusions if churn_exclusions is not None else DEFAULT_CHURN_EXCLUSIONS
        self.code_extensions = code_extensions if code_extensions is not None else CODE_EXTENSIONS
        self.default_timeout = git_timeout

    def _run_git(self, args: list[str], timeout: int | None = None) -> str:
        """Run a git command and return stdout.

        Args:
            args: Git command arguments (without 'git' prefix).
            timeout: Timeout in seconds. Defaults to self.default_timeout.

        Returns:
            Command stdout as string.

        Raises:
            GitError: If git command fails or times out.
        """
        if timeout is None:
            timeout = self.default_timeout
        cmd = ["git"] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=timeout,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise GitError(f"git {' '.join(args[:3])}... failed: {stderr}")
            return result.stdout
        except FileNotFoundError:
            raise GitError("git not found. Install git to use history tools.")
        except subprocess.TimeoutExpired:
            raise GitError(
                f"Git timed out after {timeout}s. Try a narrower time/path range."
            )

    def _verify_git(self) -> None:
        """Verify we're in a git repo. Cached after first check."""
        if self._verified:
            return
        try:
            output = self._run_git(["rev-parse", "--is-inside-work-tree"])
            if output.strip() != "true":
                raise GitError("Not inside a git repository.")
        except GitError:
            raise GitError("Not inside a git repository.")
        self._verified = True

    def _is_shallow(self) -> bool:
        """Check if the repo is a shallow clone."""
        return (self.project_root / ".git" / "shallow").exists()

    def _parse_time_range(self, time_range: Optional[str]) -> list[str]:
        """Convert human-readable time range to git flags.

        Supported formats:
            - "last 30 days" / "last 2 weeks" -> ["--since=30 days ago"]
            - "since v1.0" / "since abc1234" -> ["v1.0..HEAD"]
            - None -> [] (no filter)

        Args:
            time_range: Human-readable time range string.

        Returns:
            List of git CLI flags.
        """
        if not time_range:
            return []

        time_range = time_range.strip()

        # "last N days/weeks/months"
        match = re.match(
            r"last\s+(\d+)\s+(days?|weeks?|months?)", time_range, re.IGNORECASE
        )
        if match:
            count = match.group(1)
            unit = match.group(2)
            return [f"--since={count} {unit} ago"]

        # "since <ref>"
        match = re.match(r"since\s+(.+)", time_range, re.IGNORECASE)
        if match:
            ref = match.group(1).strip()
            return [f"{ref}..HEAD"]

        # Treat as a direct git --since value
        return [f"--since={time_range}"]

    def _build_file_line_map(self, file_path: str) -> dict[int, str]:
        """Build a mapping from line numbers to symbol node IDs for a file.

        Uses the graph querier to find all functions/methods in the file
        and maps each line to its containing symbol.

        Args:
            file_path: Relative file path (e.g., "backend/src/config.rs").

        Returns:
            Dict mapping line_number -> node_id.
        """
        if not self.querier:
            return {}

        file_id = f"FILE:{file_path}"
        nodes = self.querier.data.get("nodes", [])

        # Collect function/method spans in this file
        spans = []
        for node in nodes:
            if node["id"].startswith(file_id + "::"):
                if node["type"] in ("Function", "Method", "Class"):
                    start = node["metadata"].get("lineno", 0)
                    end = node["metadata"].get("end_lineno", start)
                    if start > 0:
                        span_size = end - start
                        spans.append((span_size, start, end, node["id"]))

        # Sort by span size descending so outer spans are written first and inner spans overwrite them
        spans.sort(key=lambda x: -x[0])

        line_map = {}
        for _, start, end, node_id in spans:
            for ln in range(start, end + 1):
                line_map[ln] = node_id

        return line_map

    def _parse_diff_hunks(self, diff_output: str) -> list[dict]:
        """Parse unified diff output into structured hunk data.

        Args:
            diff_output: Raw git diff output.

        Returns:
            List of dicts with keys: file, old_start, old_count,
            new_start, new_count, header, lines.
        """
        hunks = []
        current_file = None
        current_hunk = None

        for line in diff_output.split("\n"):
            # Track file changes
            if line.startswith("diff --git"):
                current_file = None
                current_hunk = None
            elif line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("--- /dev/null"):
                # New file
                pass
            elif line.startswith("@@ "):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line
                )
                if match and current_file:
                    current_hunk = {
                        "file": current_file,
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2) or 1),
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4) or 1),
                        "header": match.group(5).strip(),
                        "lines": [],
                    }
                    hunks.append(current_hunk)
            elif current_hunk is not None:
                if line.startswith("+") or line.startswith("-") or line.startswith(" "):
                    current_hunk["lines"].append(line)

        return hunks

    def _get_node_display(self, node_id: str) -> tuple[str, str, str]:
        """Get display info for a node ID.

        Returns:
            Tuple of (type_badge, display_name, location).
        """
        if not self.querier:
            # Extract basic info from node ID
            parts = node_id.replace("FILE:", "").split("::")
            file_path = parts[0]
            name = parts[-1] if len(parts) > 1 else file_path
            return "[Fun]", name, file_path

        node = self.querier.get_node_info(node_id)
        if not node:
            parts = node_id.replace("FILE:", "").split("::")
            return "[???]", parts[-1] if len(parts) > 1 else node_id, ""

        meta = node.get("metadata", {})
        node_type = node.get("type", "?")[:3]
        name = meta.get("name", "unknown")
        parent = meta.get("parent_name", "")
        display_name = f"{parent}.{name}" if parent else name

        # Build location
        file_path = node_id.split("::")[0].replace("FILE:", "") if node_id.startswith("FILE:") else ""
        lineno = meta.get("lineno")
        location = f"{file_path}:{lineno}" if lineno else file_path

        return f"[{node_type}]", display_name, location

    # --- Main analysis methods ---

    def get_churn(
        self,
        time_range: Optional[str] = None,
        path_filter: Optional[str] = None,
        limit: int = 20,
        mode: str = "symbols",
        exclude_generated: bool = True,
    ) -> str:
        """Analyze code churn to find hotspots.

        Args:
            time_range: Time range filter (e.g., "last 30 days").
            path_filter: Path filter (e.g., "backend/", "*.rs").
            limit: Maximum results to return.
            mode: "symbols" (default), "files", or "co-change".

        Returns:
            Formatted analysis string.
        """
        self._verify_git()

        # Build git log command for file-level commit counts
        git_args = [
            "log", "--format=%H", "--name-only", "--no-merges",
        ]
        git_args.extend(self._parse_time_range(time_range))
        if path_filter:
            git_args.extend(["--", path_filter])
        elif exclude_generated:
            exclusions = [f":!{p}" for p in self.churn_exclusions]
            git_args.extend(["--", "."] + exclusions)

        output = self._run_git(git_args, timeout=60)

        if not output.strip():
            return "No commits found for the given time range."

        # Parse file-level commit counts
        file_commits: dict[str, set[str]] = defaultdict(set)
        current_commit = None
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                current_commit = line
            elif current_commit:
                file_commits[line].add(current_commit)

        if not file_commits:
            return "No file changes found for the given time range."

        # FILE MODE: Just return file-level results
        if mode == "files":
            sorted_files = sorted(
                file_commits.items(), key=lambda x: len(x[1]), reverse=True
            )[:limit]

            lines = []
            range_note = f" ({time_range})" if time_range else ""
            lines.append(f"### Top {len(sorted_files)} Most Changed Files{range_note}\n")

            for rank, (file_path, commits) in enumerate(sorted_files, 1):
                lines.append(f"{rank:>3}. {file_path} ({len(commits)} commits)")

            return "\n".join(lines)

        # SYMBOL MODE: Attribute changes to symbols
        if mode in ("symbols", "co-change"):
            # Sort files by commit count and take top 50 for symbol attribution
            sorted_files = sorted(
                file_commits.items(), key=lambda x: len(x[1]), reverse=True
            )[:50]

            symbol_commits: dict[str, set[str]] = defaultdict(set)
            symbol_lines: dict[str, tuple[int, int]] = {}  # node_id -> (added, removed)

            for file_path, commits in sorted_files:
                # Skip commits touching too many files (bulk refactors)
                file_line_map = self._build_file_line_map(file_path)
                if not file_line_map:
                    # No graph data for this file, count at file level
                    symbol_commits[f"FILE:{file_path}"].update(commits)
                    continue

                # Get per-commit diffs for this file
                for commit_hash in commits:
                    try:
                        diff_output = self._run_git(
                            ["diff", f"{commit_hash}~1..{commit_hash}", "--", file_path],
                            timeout=10,
                        )
                    except GitError:
                        # First commit or other issue - skip
                        continue

                    hunks = self._parse_diff_hunks(diff_output)
                    touched_symbols = set()

                    for hunk in hunks:
                        # Map changed lines to symbols
                        line = hunk["new_start"]
                        for diff_line in hunk["lines"]:
                            if diff_line.startswith("+"):
                                symbol = file_line_map.get(line)
                                if symbol:
                                    touched_symbols.add(symbol)
                                    old_add, old_rem = symbol_lines.get(symbol, (0, 0))
                                    symbol_lines[symbol] = (old_add + 1, old_rem)
                                line += 1
                            elif diff_line.startswith("-"):
                                # NOTE: Removed lines use new-side line numbers for symbol lookup,
                                # which is approximate when insertions shift line positions.
                                # Commit-count attribution (the primary ranking metric) is unaffected.
                                symbol = file_line_map.get(line)
                                if symbol:
                                    touched_symbols.add(symbol)
                                    old_add, old_rem = symbol_lines.get(symbol, (0, 0))
                                    symbol_lines[symbol] = (old_add, old_rem + 1)
                                # Removed lines don't advance new-side line counter
                            else:
                                line += 1

                    for sym in touched_symbols:
                        symbol_commits[sym].add(commit_hash)

            if not symbol_commits:
                return "No symbol-level changes detected. Graph may not cover these files."

            # Separate graph-resolved symbols (FILE:path::Symbol) from
            # file-level fallbacks (FILE:path with no ::).
            graph_resolved = {}
            file_fallbacks = {}
            for k, v in symbol_commits.items():
                is_file_fallback = k.startswith("FILE:") and "::" not in k
                if is_file_fallback:
                    ext = Path(k.replace("FILE:", "")).suffix.lower()
                    if ext not in self.code_extensions:
                        continue  # Drop non-code FILE entries
                    file_fallbacks[k] = v
                else:
                    graph_resolved[k] = v
            # Prefer graph-resolved symbols; fall back to file-level only if none
            symbol_commits = graph_resolved if graph_resolved else file_fallbacks

            if mode == "co-change":
                return self._format_co_change(symbol_commits, limit, file_commits)

            # Sort by commit count
            sorted_symbols = sorted(
                symbol_commits.items(), key=lambda x: len(x[1]), reverse=True
            )[:limit]

            lines = []
            range_note = f" ({time_range})" if time_range else ""
            path_note = f" in {path_filter}" if path_filter else ""
            lines.append(
                f"### Top {len(sorted_symbols)} Most Changed Symbols{range_note}{path_note}\n"
            )

            for rank, (node_id, commits) in enumerate(sorted_symbols, 1):
                type_badge, name, location = self._get_node_display(node_id)
                added, removed = symbol_lines.get(node_id, (0, 0))
                line_stats = f", +{added}/-{removed}" if added or removed else ""
                lines.append(
                    f"{rank:>3}. {type_badge} {name} | {location} "
                    f"({len(commits)} commits{line_stats})"
                )

            return "\n".join(lines)

        return f"Unknown mode '{mode}'. Use 'symbols', 'files', or 'co-change'."

    def get_churn_structured(
        self,
        time_range: Optional[str] = None,
        path_filter: Optional[str] = None,
        limit: int = 20,
        mode: str = "symbols",
        exclude_generated: bool = True,
    ) -> dict:
        """Analyze code churn, returning structured data for UI rendering.

        Returns dict with keys: mode, time_range, path_filter, and one of:
        - symbols: list of {id, type, name, file, line, commits, added, removed}
        - files: list of {file, commits}
        - pairs: list of {a_id, a_name, a_file, b_id, b_name, b_file, shared_commits}
        """
        self._verify_git()

        git_args = ["log", "--format=%H", "--name-only", "--no-merges"]
        git_args.extend(self._parse_time_range(time_range))
        if path_filter:
            git_args.extend(["--", path_filter])
        elif exclude_generated:
            exclusions = [f":!{p}" for p in self.churn_exclusions]
            git_args.extend(["--", "."] + exclusions)

        output = self._run_git(git_args, timeout=60)

        if not output.strip():
            return {"error": "No commits found for the given time range."}

        file_commits: dict[str, set[str]] = defaultdict(set)
        current_commit = None
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                current_commit = line
            elif current_commit:
                file_commits[line].add(current_commit)

        if not file_commits:
            return {"error": "No file changes found for the given time range."}

        base = {"mode": mode, "time_range": time_range or "", "path_filter": path_filter or ""}

        # FILE MODE
        if mode == "files":
            sorted_files = sorted(
                file_commits.items(), key=lambda x: len(x[1]), reverse=True
            )[:limit]
            base["files"] = [
                {"file": fp, "commits": len(commits)}
                for fp, commits in sorted_files
            ]
            return base

        # SYMBOL / CO-CHANGE: attribute changes to symbols
        sorted_files = sorted(
            file_commits.items(), key=lambda x: len(x[1]), reverse=True
        )[:50]

        symbol_commits: dict[str, set[str]] = defaultdict(set)
        symbol_lines: dict[str, tuple[int, int]] = {}

        for file_path, commits in sorted_files:
            file_line_map = self._build_file_line_map(file_path)
            if not file_line_map:
                symbol_commits[f"FILE:{file_path}"].update(commits)
                continue

            for commit_hash in commits:
                try:
                    diff_output = self._run_git(
                        ["diff", f"{commit_hash}~1..{commit_hash}", "--", file_path],
                        timeout=10,
                    )
                except GitError:
                    continue

                hunks = self._parse_diff_hunks(diff_output)
                touched_symbols = set()

                for hunk in hunks:
                    line = hunk["new_start"]
                    for diff_line in hunk["lines"]:
                        if diff_line.startswith("+"):
                            symbol = file_line_map.get(line)
                            if symbol:
                                touched_symbols.add(symbol)
                                old_add, old_rem = symbol_lines.get(symbol, (0, 0))
                                symbol_lines[symbol] = (old_add + 1, old_rem)
                            line += 1
                        elif diff_line.startswith("-"):
                            symbol = file_line_map.get(line)
                            if symbol:
                                touched_symbols.add(symbol)
                                old_add, old_rem = symbol_lines.get(symbol, (0, 0))
                                symbol_lines[symbol] = (old_add, old_rem + 1)
                        else:
                            line += 1

                for sym in touched_symbols:
                    symbol_commits[sym].add(commit_hash)

        if not symbol_commits:
            return {"error": "No symbol-level changes detected. Graph may not cover these files."}

        # Filter to graph-resolved symbols
        graph_resolved = {}
        file_fallbacks = {}
        for k, v in symbol_commits.items():
            is_file_fallback = k.startswith("FILE:") and "::" not in k
            if is_file_fallback:
                ext = Path(k.replace("FILE:", "")).suffix.lower()
                if ext not in self.code_extensions:
                    continue
                file_fallbacks[k] = v
            else:
                graph_resolved[k] = v
        symbol_commits = graph_resolved if graph_resolved else file_fallbacks

        # CO-CHANGE MODE
        if mode == "co-change":
            return self._structured_co_change(symbol_commits, limit, file_commits, base)

        # SYMBOLS MODE
        sorted_symbols = sorted(
            symbol_commits.items(), key=lambda x: len(x[1]), reverse=True
        )[:limit]

        symbols = []
        for node_id, commits in sorted_symbols:
            type_badge, name, location = self._get_node_display(node_id)
            file_path = ""
            line = None
            if ":" in location:
                file_path, line_str = location.rsplit(":", 1)
                try:
                    line = int(line_str)
                except ValueError:
                    file_path = location
            else:
                file_path = location
            added, removed = symbol_lines.get(node_id, (0, 0))
            symbols.append({
                "id": node_id,
                "type": type_badge.strip("[]"),
                "name": name,
                "file": file_path,
                "line": line,
                "commits": len(commits),
                "added": added,
                "removed": removed,
            })

        base["symbols"] = symbols
        return base

    def _structured_co_change(
        self,
        symbol_commits: dict[str, set[str]],
        limit: int,
        file_commits: dict[str, set[str]],
        base: dict,
    ) -> dict:
        """Build structured co-change data."""
        graph_symbols = {k: v for k, v in symbol_commits.items() if "::" in k}

        pairs = []
        if graph_symbols:
            commit_symbols: dict[str, set[str]] = defaultdict(set)
            for sym, commits in graph_symbols.items():
                for c in commits:
                    commit_symbols[c].add(sym)

            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for symbols in commit_symbols.values():
                sym_list = sorted(symbols)
                for i in range(len(sym_list)):
                    for j in range(i + 1, len(sym_list)):
                        pair_counts[(sym_list[i], sym_list[j])] += 1

            frequent_pairs = [
                (pair, count) for pair, count in pair_counts.items() if count >= 2
            ]
            frequent_pairs.sort(key=lambda x: x[1], reverse=True)

            for (sym_a, sym_b), count in frequent_pairs[:limit]:
                _, name_a, loc_a = self._get_node_display(sym_a)
                _, name_b, loc_b = self._get_node_display(sym_b)
                pairs.append({
                    "a_id": sym_a, "a_name": name_a, "a_file": loc_a,
                    "b_id": sym_b, "b_name": name_b, "b_file": loc_b,
                    "shared_commits": count,
                })

        if not pairs and file_commits:
            # Fall back to file-level co-change
            code_files = {
                f: commits for f, commits in file_commits.items()
                if Path(f).suffix.lower() in self.code_extensions
            }
            if code_files:
                commit_files: dict[str, set[str]] = defaultdict(set)
                for fp, commits in code_files.items():
                    for c in commits:
                        commit_files[c].add(fp)

                pair_counts: dict[tuple[str, str], int] = defaultdict(int)
                for files in commit_files.values():
                    file_list = sorted(files)
                    for i in range(len(file_list)):
                        for j in range(i + 1, len(file_list)):
                            pair_counts[(file_list[i], file_list[j])] += 1

                frequent_pairs = [
                    (pair, count) for pair, count in pair_counts.items() if count >= 2
                ]
                frequent_pairs.sort(key=lambda x: x[1], reverse=True)

                for (file_a, file_b), count in frequent_pairs[:limit]:
                    pairs.append({
                        "a_id": "", "a_name": file_a, "a_file": file_a,
                        "b_id": "", "b_name": file_b, "b_file": file_b,
                        "shared_commits": count,
                    })
                base["file_level"] = True

        base["pairs"] = pairs
        return base

    def _format_co_change(
        self,
        symbol_commits: dict[str, set[str]],
        limit: int,
        file_commits: Optional[dict[str, set[str]]] = None,
    ) -> str:
        """Format co-change analysis showing symbol pairs that change together.

        Args:
            symbol_commits: Mapping of symbol -> set of commit hashes.
            limit: Max pairs to show.
            file_commits: Optional file-level commit data for fallback.

        Returns:
            Formatted co-change analysis.
        """
        # Exclude file-level fallback entries (FILE:path with no ::) for meaningful co-change
        graph_symbols = {k: v for k, v in symbol_commits.items() if "::" in k}

        frequent_pairs = []
        if graph_symbols:
            # Build commit -> symbols map
            commit_symbols: dict[str, set[str]] = defaultdict(set)
            for sym, commits in graph_symbols.items():
                for c in commits:
                    commit_symbols[c].add(sym)

            # Count co-occurrences
            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for symbols in commit_symbols.values():
                sym_list = sorted(symbols)
                for i in range(len(sym_list)):
                    for j in range(i + 1, len(sym_list)):
                        pair_counts[(sym_list[i], sym_list[j])] += 1

            # Filter to pairs with >1 co-occurrence
            frequent_pairs = [
                (pair, count) for pair, count in pair_counts.items() if count >= 2
            ]
            frequent_pairs.sort(key=lambda x: x[1], reverse=True)
            frequent_pairs = frequent_pairs[:limit]

        if frequent_pairs:
            lines = [f"### Top {len(frequent_pairs)} Co-Change Pairs\n"]
            lines.append("Symbols that frequently change in the same commit:\n")

            for rank, ((sym_a, sym_b), count) in enumerate(frequent_pairs, 1):
                _, name_a, loc_a = self._get_node_display(sym_a)
                _, name_b, loc_b = self._get_node_display(sym_b)
                loc_suffix_a = f" ({loc_a})" if loc_a else ""
                loc_suffix_b = f" ({loc_b})" if loc_b else ""
                lines.append(
                    f"{rank:>3}. {name_a}{loc_suffix_a} <-> "
                    f"{name_b}{loc_suffix_b} ({count} shared commits)"
                )

            return "\n".join(lines)

        # Fall back to file-level co-change when symbol data is sparse
        if file_commits:
            return self._format_file_co_change(file_commits, limit)

        return "No co-change patterns found (need pairs with 2+ shared commits)."

    def _format_file_co_change(
        self, file_commits: dict[str, set[str]], limit: int
    ) -> str:
        """Format co-change analysis at the file level (fallback).

        Args:
            file_commits: Mapping of file path -> set of commit hashes.
            limit: Max pairs to show.

        Returns:
            Formatted file-level co-change analysis.
        """
        # Filter to code files only
        code_files = {
            f: commits for f, commits in file_commits.items()
            if Path(f).suffix.lower() in self.code_extensions
        }
        if not code_files:
            return "No co-change patterns found in code files."

        # Build commit -> files map
        commit_files: dict[str, set[str]] = defaultdict(set)
        for fp, commits in code_files.items():
            for c in commits:
                commit_files[c].add(fp)

        # Count co-occurrences
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for files in commit_files.values():
            file_list = sorted(files)
            for i in range(len(file_list)):
                for j in range(i + 1, len(file_list)):
                    pair_counts[(file_list[i], file_list[j])] += 1

        frequent_pairs = [
            (pair, count) for pair, count in pair_counts.items() if count >= 2
        ]
        frequent_pairs.sort(key=lambda x: x[1], reverse=True)
        frequent_pairs = frequent_pairs[:limit]

        if not frequent_pairs:
            return "No co-change patterns found (need pairs with 2+ shared commits)."

        lines = [f"### Top {len(frequent_pairs)} Co-Change Pairs (file-level)\n"]
        lines.append("*Symbol-level data was too sparse; showing file-level co-changes:*\n")

        for rank, ((file_a, file_b), count) in enumerate(frequent_pairs, 1):
            lines.append(f"{rank:>3}. {file_a} <-> {file_b} ({count} shared commits)")

        return "\n".join(lines)

    def get_evolution(
        self,
        name: str,
        time_range: Optional[str] = None,
        limit: int = 10,
        show_diff: bool = False,
        crate: Optional[str] = None,
    ) -> str:
        """Track the evolution of a specific symbol over time.

        Args:
            name: Symbol name (fuzzy-matched via graph querier).
            time_range: Time range filter.
            limit: Maximum commits to show.
            show_diff: Include actual diff hunks.

        Returns:
            Formatted evolution timeline.
        """
        self._verify_git()

        # Resolve symbol
        file_path = None
        symbol_name = name
        resolved_node_id = None
        match_count = 0

        if self.querier:
            matches = self.querier.find_nodes_by_name(name)
            # Prefer functions/methods
            func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
            if func_matches:
                matches = func_matches

            if not matches:
                matches = self.querier.find_nodes_by_name(name, fuzzy=True)
                func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
                if func_matches:
                    matches = func_matches

            if matches:
                all_matches = list(matches)
                match_count = len(all_matches)

                # Filter by crate if specified
                if crate:
                    crate_filtered = [m for m in matches if crate.lower() in m["id"].lower()]
                    if crate_filtered:
                        matches = crate_filtered

                # Rank by in_degree (callers), non-test preferred
                if len(matches) > 1:
                    def rank_match(m):
                        in_degree = m.get("metadata", {}).get("in_degree", 0)
                        is_test = "test" in m["id"].lower()
                        return (-in_degree, is_test)
                    matches.sort(key=rank_match)

                best = matches[0]
                resolved_node_id = best["id"]
                symbol_name = best["metadata"].get("name", name)
                # Extract file path from node ID
                if best["id"].startswith("FILE:"):
                    file_path = best["id"].split("::")[0].replace("FILE:", "")

        if not file_path:
            return (
                f"Could not resolve symbol '{name}'. "
                "Try descry_search to find the exact name."
            )

        # Try git log -L :name:file (git's native function tracking)
        # This works for top-level functions but fails for methods/nested fns
        git_args = ["log", f"-L:{symbol_name}:{file_path}",
                     "--format=%H %ad %an%n%s", "--date=short", "--no-merges"]
        git_args.extend(self._parse_time_range(time_range))

        try:
            output = self._run_git(git_args, timeout=30)
        except GitError:
            output = None

        # Fallback: try line-range based -L using descry metadata
        if not output and resolved_node_id and self.querier:
            node_info = self.querier.get_node_info(resolved_node_id)
            if node_info:
                meta = node_info.get("metadata", {})
                start_line = meta.get("lineno")
                end_line = meta.get("end_lineno")
                if start_line and end_line:
                    git_args = [
                        "log", f"-L{start_line},{end_line}:{file_path}",
                        "--format=%H %ad %an%n%s", "--date=short", "--no-merges",
                    ]
                    git_args.extend(self._parse_time_range(time_range))
                    try:
                        output = self._run_git(git_args, timeout=30)
                    except GitError:
                        output = None

        # Build disambiguation note if multiple matches
        disambig_note = ""
        if match_count > 1:
            _, _, location = self._get_node_display(resolved_node_id) if resolved_node_id else ("", "", file_path)
            disambig_note = (
                f"\n*{match_count} symbols named `{name}` — "
                f"showing: {location}. Use `crate` to disambiguate.*"
            )

        if output and output.strip():
            result = self._format_evolution_from_log_l(
                output, symbol_name, file_path, resolved_node_id, limit, show_diff
            )
            return result + disambig_note

        # Fallback: simple file log filtered by symbol
        git_args = ["log", "--format=%H %ad %an%n%s", "--date=short",
                     "--no-merges"]
        git_args.extend(self._parse_time_range(time_range))
        git_args.extend(["--", file_path])

        try:
            output = self._run_git(git_args, timeout=30)
        except GitError as e:
            return f"Error getting history: {e}"

        if not output.strip():
            return f"No commits found for '{symbol_name}' in {file_path}."

        result = self._format_evolution_fallback(
            output, symbol_name, file_path, resolved_node_id, limit, show_diff
        )
        return result + disambig_note

    def _format_evolution_from_log_l(
        self,
        output: str,
        symbol_name: str,
        file_path: str,
        node_id: Optional[str],
        limit: int,
        show_diff: bool,
    ) -> str:
        """Format evolution from git log -L output.

        The output format is: commit_hash date author\nsubject\ndiff...
        """
        commits = []
        current_commit = None
        diff_lines = []
        authors = set()

        for line in output.split("\n"):
            # New commit line: hash date author
            match = re.match(r"^([0-9a-f]{40})\s+(\S+)\s+(.+)$", line)
            if match:
                if current_commit:
                    current_commit["diff"] = "\n".join(diff_lines)
                    commits.append(current_commit)
                    diff_lines = []

                current_commit = {
                    "hash": match.group(1)[:8],
                    "date": match.group(2),
                    "author": match.group(3).strip(),
                    "subject": "",
                    "diff": "",
                    "added": 0,
                    "removed": 0,
                }
                authors.add(match.group(3).strip())
            elif current_commit and not current_commit["subject"]:
                current_commit["subject"] = line.strip()
            elif current_commit:
                diff_lines.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    current_commit["added"] += 1
                elif line.startswith("-") and not line.startswith("---"):
                    current_commit["removed"] += 1

        if current_commit:
            current_commit["diff"] = "\n".join(diff_lines)
            commits.append(current_commit)

        commits = commits[:limit]

        if not commits:
            return f"No evolution data found for '{symbol_name}'."

        # Format output
        location = file_path
        if node_id:
            _, _, loc = self._get_node_display(node_id)
            if loc:
                location = loc

        # Calculate date span
        dates = [c["date"] for c in commits]
        date_span = f"{dates[-1]} to {dates[0]}" if len(dates) > 1 else dates[0]

        lines = [f"### Evolution of `{symbol_name}` ({location})\n"]
        lines.append(
            f"{len(commits)} commit(s) | {len(authors)} author(s) | {date_span}\n"
        )

        shallow = " (shallow clone - history may be incomplete)" if self._is_shallow() else ""
        if shallow:
            lines.append(f"*Warning: {shallow}*\n")

        for i, commit in enumerate(commits, 1):
            lines.append(f"{i}. **{commit['hash']}** ({commit['date']}) {commit['author']}")
            lines.append(f"   {commit['subject']}")
            if commit["added"] or commit["removed"]:
                lines.append(f"   +{commit['added']}/-{commit['removed']} lines")

            if show_diff and commit["diff"].strip():
                # Show condensed diff - indent content for markdown list nesting
                diff_text = commit["diff"].strip()
                diff_lines_raw = diff_text.split("\n")[:30]
                # Indent each line by 3 spaces so marked.js keeps it inside the list item
                indented = "\n".join("   " + dl for dl in diff_lines_raw)
                if len(diff_text.split("\n")) > 30:
                    indented += "\n   ... (truncated)"
                lines.append(f"   ```diff\n{indented}\n   ```")
            lines.append("")

        return "\n".join(lines)

    def _format_evolution_fallback(
        self,
        output: str,
        symbol_name: str,
        file_path: str,
        node_id: Optional[str],
        limit: int,
        show_diff: bool,
    ) -> str:
        """Format evolution using simple file-level log (fallback mode)."""
        commits = []
        authors = set()

        for line in output.strip().split("\n"):
            match = re.match(r"^([0-9a-f]{40})\s+(\S+)\s+(.+)$", line)
            if match:
                commits.append({
                    "hash": match.group(1)[:8],
                    "date": match.group(2),
                    "author": match.group(3).strip(),
                    "subject": "",
                })
                authors.add(match.group(3).strip())
            elif commits and not commits[-1]["subject"]:
                commits[-1]["subject"] = line.strip()

        commits = commits[:limit]

        if not commits:
            return f"No commits found for '{symbol_name}' in {file_path}."

        location = file_path
        if node_id:
            _, _, loc = self._get_node_display(node_id)
            if loc:
                location = loc

        lines = [f"### Evolution of `{symbol_name}` ({location})\n"]
        note = "*Note: Using file-level history (git log -L failed for this symbol)*"
        if show_diff:
            note += "\n*Diffs were requested but are not available in file-level fallback mode.*"
        lines.append(note + "\n")
        lines.append(
            f"{len(commits)} commit(s) | {len(authors)} author(s)\n"
        )

        for i, commit in enumerate(commits, 1):
            lines.append(f"{i}. **{commit['hash']}** ({commit['date']}) {commit['author']}")
            lines.append(f"   {commit['subject']}")
            lines.append("")

        return "\n".join(lines)

    def get_changes(
        self,
        commit_range: Optional[str] = None,
        time_range: Optional[str] = None,
        path_filter: Optional[str] = None,
        show_callers: bool = True,
        limit: int = 50,
    ) -> str:
        """Analyze change impact for a commit range.

        Args:
            commit_range: Git commit range (e.g., "HEAD~5..HEAD").
            time_range: Time range (e.g., "last 7 days"). Used if commit_range not given.
            path_filter: Path filter (e.g., "backend/").
            show_callers: Include callers of changed symbols.
            limit: Maximum symbols to show.

        Returns:
            Formatted impact analysis.
        """
        self._verify_git()

        # Determine effective commit range
        effective_range = commit_range
        if not effective_range:
            if time_range:
                # Get commits in time range and build a range
                time_flags = self._parse_time_range(time_range)
                git_args = ["log", "--format=%H", "--no-merges"] + time_flags
                output = self._run_git(git_args, timeout=30)
                commit_hashes = [h.strip() for h in output.strip().split("\n") if h.strip()]
                if not commit_hashes:
                    return "No commits found for the given time range."
                # oldest..newest
                effective_range = f"{commit_hashes[-1]}~1..{commit_hashes[0]}"
            else:
                effective_range = "HEAD~1..HEAD"

        # Get changed files with stats
        diff_stat_args = ["diff", "--numstat", effective_range]
        if path_filter:
            diff_stat_args.extend(["--", path_filter])

        try:
            numstat_output = self._run_git(diff_stat_args, timeout=30)
        except GitError as e:
            return f"Invalid commit range '{effective_range}': {e}"

        if not numstat_output.strip():
            return f"No changes found in {effective_range}."

        # Parse numstat output
        changed_files: list[dict] = []
        for line in numstat_output.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 3:
                added = int(parts[0]) if parts[0] != "-" else 0
                removed = int(parts[1]) if parts[1] != "-" else 0
                changed_files.append({
                    "file": parts[2],
                    "added": added,
                    "removed": removed,
                })

        # Count commits in range
        try:
            commit_count_output = self._run_git(
                ["rev-list", "--count", effective_range], timeout=10
            )
            commit_count = int(commit_count_output.strip())
        except (GitError, ValueError):
            commit_count = 0

        # Get full diff for symbol attribution
        diff_args = ["diff", effective_range]
        if path_filter:
            diff_args.extend(["--", path_filter])

        try:
            diff_output = self._run_git(diff_args, timeout=60)
        except GitError:
            diff_output = ""

        hunks = self._parse_diff_hunks(diff_output) if diff_output else []

        # Attribute changes to symbols
        modified_symbols: dict[str, dict] = {}  # node_id -> {added, removed}
        added_symbols: list[str] = []  # new node_ids
        removed_symbols: list[str] = []  # removed node_ids

        # Group hunks by file
        file_hunks: dict[str, list[dict]] = defaultdict(list)
        for hunk in hunks:
            file_hunks[hunk["file"]].append(hunk)

        for file_path, fhunks in file_hunks.items():
            line_map = self._build_file_line_map(file_path)
            if not line_map:
                continue

            for hunk in fhunks:
                line = hunk["new_start"]
                for diff_line in hunk["lines"]:
                    if diff_line.startswith("+"):
                        symbol = line_map.get(line)
                        if symbol:
                            if symbol not in modified_symbols:
                                modified_symbols[symbol] = {"added": 0, "removed": 0}
                            modified_symbols[symbol]["added"] += 1
                        line += 1
                    elif diff_line.startswith("-"):
                        # NOTE: For removed lines, symbol attribution uses new-side line numbers
                        # (approximate — old-side would require building line map from pre-change state).
                        # The +N/-M stats may be slightly inaccurate but commit-level attribution is correct.
                        symbol = line_map.get(line)
                        if symbol:
                            if symbol not in modified_symbols:
                                modified_symbols[symbol] = {"added": 0, "removed": 0}
                            modified_symbols[symbol]["removed"] += 1
                    else:
                        line += 1

        # Format output
        lines = []
        range_display = commit_range or time_range or "HEAD~1..HEAD"
        count_note = f" ({commit_count} commit{'s' if commit_count != 1 else ''})" if commit_count else ""
        lines.append(f"### Change Impact: {range_display}{count_note}\n")
        lines.append(
            f"Scope: {len(changed_files)} file(s) | "
            f"{len(modified_symbols)} symbol(s) affected\n"
        )

        if not modified_symbols:
            # Show file-level summary if no symbol attribution possible
            lines.append("*No symbol-level attribution available (graph may not cover these files).*\n")
            lines.append("#### Changed Files")
            for f in changed_files[:limit]:
                lines.append(f"  - {f['file']} (+{f['added']}/-{f['removed']})")
            return "\n".join(lines)

        # Sort modified symbols by total change magnitude
        sorted_modified = sorted(
            modified_symbols.items(),
            key=lambda x: x[1]["added"] + x[1]["removed"],
            reverse=True,
        )[:limit]

        lines.append(f"#### Modified ({len(sorted_modified)})")
        for rank, (node_id, stats) in enumerate(sorted_modified, 1):
            type_badge, name, location = self._get_node_display(node_id)
            lines.append(
                f"{rank:>3}. {type_badge} {name} | {location} "
                f"(+{stats['added']}/-{stats['removed']})"
            )

            # Show callers if requested
            if show_callers and self.querier:
                sym_name = name.split(".")[-1]  # Get base name
                callers = self.querier.get_callers(sym_name)
                if callers:
                    caller_names = []
                    for caller_id in callers[:5]:
                        c_node = self.querier.get_node_info(caller_id)
                        if c_node:
                            caller_names.append(c_node["metadata"].get("name", "?"))
                        else:
                            # Extract name from node ID
                            parts = caller_id.split("::")
                            caller_names.append(parts[-1] if parts else "?")

                    more = f", +{len(callers) - 5} more" if len(callers) > 5 else ""
                    lines.append(f"     Callers: {', '.join(caller_names)}{more}")

        return "\n".join(lines)

    def get_changes_structured(
        self,
        commit_range: Optional[str] = None,
        time_range: Optional[str] = None,
        path_filter: Optional[str] = None,
        show_callers: bool = True,
        limit: int = 50,
    ) -> dict:
        """Analyze change impact, returning structured data for UI rendering.

        Returns dict with keys: range, commits, files_count, symbols_count,
        symbols (list), changed_files (list when no symbol attribution).
        """
        self._verify_git()

        effective_range = commit_range
        if not effective_range:
            if time_range:
                time_flags = self._parse_time_range(time_range)
                git_args = ["log", "--format=%H", "--no-merges"] + time_flags
                output = self._run_git(git_args, timeout=30)
                commit_hashes = [h.strip() for h in output.strip().split("\n") if h.strip()]
                if not commit_hashes:
                    return {"error": "No commits found for the given time range."}
                effective_range = f"{commit_hashes[-1]}~1..{commit_hashes[0]}"
            else:
                effective_range = "HEAD~1..HEAD"

        diff_stat_args = ["diff", "--numstat", effective_range]
        if path_filter:
            diff_stat_args.extend(["--", path_filter])

        try:
            numstat_output = self._run_git(diff_stat_args, timeout=30)
        except GitError as e:
            return {"error": f"Invalid commit range '{effective_range}': {e}"}

        if not numstat_output.strip():
            return {"error": f"No changes found in {effective_range}."}

        changed_files: list[dict] = []
        for line in numstat_output.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 3:
                added = int(parts[0]) if parts[0] != "-" else 0
                removed = int(parts[1]) if parts[1] != "-" else 0
                changed_files.append({"file": parts[2], "added": added, "removed": removed})

        try:
            commit_count_output = self._run_git(
                ["rev-list", "--count", effective_range], timeout=10
            )
            commit_count = int(commit_count_output.strip())
        except (GitError, ValueError):
            commit_count = 0

        diff_args = ["diff", effective_range]
        if path_filter:
            diff_args.extend(["--", path_filter])

        try:
            diff_output = self._run_git(diff_args, timeout=60)
        except GitError:
            diff_output = ""

        hunks = self._parse_diff_hunks(diff_output) if diff_output else []

        modified_symbols: dict[str, dict] = {}
        file_hunks: dict[str, list[dict]] = defaultdict(list)
        for hunk in hunks:
            file_hunks[hunk["file"]].append(hunk)

        for file_path, fhunks in file_hunks.items():
            line_map = self._build_file_line_map(file_path)
            if not line_map:
                continue
            for hunk in fhunks:
                line = hunk["new_start"]
                for diff_line in hunk["lines"]:
                    if diff_line.startswith("+"):
                        symbol = line_map.get(line)
                        if symbol:
                            if symbol not in modified_symbols:
                                modified_symbols[symbol] = {"added": 0, "removed": 0}
                            modified_symbols[symbol]["added"] += 1
                        line += 1
                    elif diff_line.startswith("-"):
                        symbol = line_map.get(line)
                        if symbol:
                            if symbol not in modified_symbols:
                                modified_symbols[symbol] = {"added": 0, "removed": 0}
                            modified_symbols[symbol]["removed"] += 1
                    else:
                        line += 1

        range_display = commit_range or time_range or "HEAD~1..HEAD"

        result = {
            "range": range_display,
            "commits": commit_count,
            "files_count": len(changed_files),
            "symbols_count": len(modified_symbols),
        }

        if not modified_symbols:
            result["changed_files"] = changed_files[:limit]
            return result

        sorted_modified = sorted(
            modified_symbols.items(),
            key=lambda x: x[1]["added"] + x[1]["removed"],
            reverse=True,
        )[:limit]

        symbols = []
        for node_id, stats in sorted_modified:
            type_badge, name, location = self._get_node_display(node_id)
            file_path = ""
            line = None
            if ":" in location:
                file_path, line_str = location.rsplit(":", 1)
                try:
                    line = int(line_str)
                except ValueError:
                    file_path = location
            else:
                file_path = location

            entry = {
                "id": node_id,
                "type": type_badge.strip("[]"),
                "name": name,
                "file": file_path,
                "line": line,
                "added": stats["added"],
                "removed": stats["removed"],
            }

            if show_callers and self.querier:
                sym_name = name.split(".")[-1]
                caller_ids = self.querier.get_callers(sym_name)
                if caller_ids:
                    callers = []
                    for caller_id in caller_ids[:5]:
                        c_node = self.querier.get_node_info(caller_id)
                        c_name = (
                            c_node["metadata"].get("name", "?")
                            if c_node
                            else caller_id.split("::")[-1] if "::" in caller_id else "?"
                        )
                        callers.append({"id": caller_id, "name": c_name})
                    entry["callers"] = callers
                    entry["callers_total"] = len(caller_ids)

            symbols.append(entry)

        result["symbols"] = symbols
        return result
