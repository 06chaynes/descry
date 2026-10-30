"""
Git History Analysis Module

Provides symbol-level git history analysis by combining git operations
with descry's structural knowledge. Enables churn hotspot detection,
symbol evolution tracking, and change impact analysis.
"""

import difflib
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from descry._answers import fail
from descry._files import open_regular
from descry._parsing import source_lines
from descry.handlers import (
    _DEFAULT_CHURN_EXCLUSIONS,
    _DEFAULT_CODE_EXTENSIONS,
    ChurnMode,
)
from descry.query import file_of, source_path

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Raised for git-specific errors (not installed, not a repo, etc.).
    `detail`, what git wrote, is for the text only."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message, self.detail = message, detail


class GitFailed(GitError):
    """Raised when git ran and exited non-zero."""


class PartialCloneError(GitError):
    """Raised when git needs objects a partial clone has not downloaded."""


# Accepts: HEAD, HEAD^, HEAD^^, HEAD~N, HEAD^{tree}, @, @{-1},
#          branch@{upstream}, master@{yesterday}, branch names, sha prefixes,
#          and the ~N / ^ / ^{type} suffixes on any of those names.
# Rejects: leading '-', '--' anywhere, empty peel, shell metacharacters.
_REF_REGEX = re.compile(
    r"(?:HEAD|@|[A-Za-z0-9][A-Za-z0-9._\-/]*)"
    r"(?:[~^]\d*)*(?:\^\{[a-z]+\})?(?:@\{(?:-\d+|[^}\-]+)\})?"
    r"|@\{-\d+\}"
)


def _validate_git_ref(ref: str) -> str:
    """Validate a single git ref. Raises GitError on invalid input."""
    if not ref or "--" in ref or not _REF_REGEX.fullmatch(ref):
        raise GitError(f"Invalid git ref: {ref!r}")
    return ref


def _validate_git_range(range_str: str) -> str:
    """Validate a git range (ref, ref..ref, or ref...ref).

    Raises GitError on invalid input.
    """
    if not range_str or "--" in range_str:
        raise GitError(f"Invalid git range: {range_str!r}")
    # Try triple-dot first, then double-dot, then single ref
    for sep in ("...", ".."):
        if sep in range_str:
            parts = range_str.split(sep)
            if len(parts) != 2:
                raise GitError(f"Invalid git range: {range_str!r}")
            _validate_git_ref(parts[0])
            _validate_git_ref(parts[1])
            return range_str
    _validate_git_ref(range_str)
    return range_str


def _validate_git_file(path: str) -> str:
    """Validate a file path for git (-L, pathspec).

    Rejects: absolute paths, '..' traversal, leading '-', pathspec magic
    prefix ':', shell metacharacters.
    """
    if not path:
        raise GitError("Empty git file path")
    if path.startswith(("-", ":", "/")):
        raise GitError(f"Invalid git file: {path!r}")
    if ".." in path.split("/"):
        raise GitError(f"Invalid git file: {path!r}")
    if any(c in path for c in "\x00;|&`$\n\r\t"):
        raise GitError(f"Invalid git file: {path!r}")
    return path


def _validate_pathspec(spec: str) -> str:
    """Validate a pathspec entry (no magic prefixes)."""
    if not spec:
        raise GitError("Empty pathspec")
    if spec.startswith((":", "-")):
        raise GitError(f"Invalid pathspec (magic prefix): {spec!r}")
    if ".." in spec.split("/"):
        raise GitError(f"Invalid pathspec: {spec!r}")
    if any(c in spec for c in "\x00\n\r"):
        raise GitError(f"Invalid pathspec: {spec!r}")
    return spec


# Placed before the subcommand on every invocation. All but the last stop
# the analysed repository's own config and hooks from making git run a
# command; `-c` overrides the repository's config, including files it
# includes.
_GIT_CONFIG_PINS = (
    # A path's history is its own; the churn walk reads renames itself.
    "-c",
    "log.follow=false",
    "-c",
    f"core.hooksPath={os.devnull}",
    "-c",
    "core.fsmonitor=",
    # Would run gpg.program for every signed commit a log walks.
    "-c",
    "log.showSignature=false",
    # Print non-ASCII paths as they are instead of as quoted octal escapes.
    "-c",
    "core.quotePath=false",
    # Keep blank context lines as " " so hunk line counts stay exact.
    "-c",
    "diff.suppressBlankEmpty=false",
)
# Ignore diff.external and the diff drivers and textconv filters that
# attributes select, and print the plain a/ b/ uncoloured form the diff
# parser reads whatever the repository's config asks for. --submodule=short
# keeps a submodule diff inline, and --ignore-submodules=dirty keeps the
# working-tree dirty check of a single-ref diff from starting a child git
# in the submodule that would read the submodule's own config and run its
# filter drivers; a submodule's moved commit still diffs.
_GIT_DIFF_OPTS = (
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "--submodule=short",
    "--ignore-submodules=dirty",
)
# Merged over the environment. A partial clone would otherwise fetch missing
# objects on demand through whatever transport command its remote config
# names. The C locale keeps the resulting error text matchable.
_GIT_ENV = {"GIT_NO_LAZY_FETCH": "1", "LC_ALL": "C"}

# A commit header line starts with a NUL and separates its fields with NULs.
# No path, subject, author name or diffed line can start with one, and no
# length is assumed for the commit id.
_NAME_ONLY_FORMAT = "--format=%x00%H%x00%P"
_HEADER_FORMAT = "--format=%x00%H%x00%ad%x00%an%x00%s"

# Symbol-mode churn runs one `git diff` per commit per file. Only this many
# of a file's most recent commits are diffed, and no further diff is started
# once the pass has run for this many seconds.
CHURN_MAX_COMMITS_PER_FILE = 100
CHURN_DIFF_BUDGET_SECONDS = 30
_UNKNOWN_CHURN_MODE = "Unknown mode '{}'. Use 'symbols', 'files', or 'co-change'."


def _parse_name_only_log(output: str) -> dict[str, list[str]]:
    """Each file's commits, newest first, of `git log --name-only` output
    written with _NAME_ONLY_FORMAT."""
    file_commits: dict[str, list[str]] = defaultdict(list)
    current_commit = None
    for line in output.split("\n"):
        if line.startswith("\x00"):
            current_commit = line[1:].strip().partition("\x00")[0]
        elif line.strip() and current_commit:
            file_commits[line.strip()].append(current_commit)
    return file_commits


def _parse_log_header(line: str) -> dict | None:
    """Parse a commit header line written with _HEADER_FORMAT, else None."""
    if not line.startswith("\x00"):
        return None
    commit_hash, date, author, subject = line[1:].split("\x00", 3)
    return {
        "hash": commit_hash[:8],
        "date": date,
        "author": author.strip(),
        "subject": subject.strip(),
    }


def _line_before(hunks: list[dict], line: int, first: bool) -> int:
    """Where line ``line`` of the new side of the diff ``hunks`` stands on
    its old side; one the diff wrote, at the first line of what it replaced
    for ``first``, else at the last."""
    shift = 0
    for hunk in hunks:
        new, new_count = hunk["new_start"], hunk["new_count"]
        old, old_count = hunk["old_start"], hunk["old_count"]
        if line < new + (not new_count):
            break
        if line < new + new_count:
            # With no old lines `old` is the line the new ones follow.
            return old + (not old_count) if first else old + old_count - 1
        shift += old_count - new_count
    return line + shift


def _spans_before(spans: list, hunks: list[dict]) -> list:
    """``spans`` of lines on the new side of the diff ``hunks`` as they
    stand on its old side; one wholly written by the diff is empty."""
    return [
        (_line_before(hunks, start, True), _line_before(hunks, end, False), held)
        for start, end, held in spans
    ]


def _line_diff(old: list[str], new: list[str]) -> list[dict]:
    """The hunks of a diff of the lines ``old`` to ``new``, as
    `_parse_diff_hunks` gives a diff's without context."""
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return [
        {
            "old_start": i1 + (i2 > i1),
            "old_count": i2 - i1,
            "new_start": j1 + (j2 > j1),
            "new_count": j2 - j1,
        }
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _line_map(spans: list) -> dict[int, str]:
    """Each line of ``spans`` to the innermost of them that holds it."""
    line_map = {}
    for start, end, node_id in sorted(spans, key=lambda s: s[0] - s[1]):
        for ln in range(start, end + 1):
            line_map[ln] = node_id
    return line_map


def _touched(hunks: list[dict], spans: list, before: list) -> dict[str, dict]:
    """``{symbol: {added, removed, diff}}`` of the `-U0` diff ``hunks``: its
    added lines counted by the new side's ``spans``, its removed ones by the
    old side's ``before``, ``diff`` the lines each symbol holds."""
    new_map, old_map = _line_map(spans), _line_map(before)
    touched: dict[str, dict] = {}
    for hunk in hunks:
        for side, mark, lines, start in (
            ("removed", "-", old_map, hunk["old_start"]),
            ("added", "+", new_map, hunk["new_start"]),
        ):
            written = [text for text in hunk["lines"] if text[:1] == mark]
            for line, text in enumerate(written, start):
                if symbol := lines.get(line):
                    held = touched.setdefault(
                        symbol, {"added": 0, "removed": 0, "diff": []}
                    )
                    held[side] += 1
                    held["diff"].append(text)
    return touched


def _churn_partial_note(skipped: dict[str, int]) -> str:
    """Describe what the churn limits left undiffed, or "" if nothing."""
    parts = []
    if skipped["capped_files"]:
        parts.append(
            f"only the {CHURN_MAX_COMMITS_PER_FILE} most recent commits were "
            f"read for {skipped['capped_files']} file(s) with a longer history"
        )
    if skipped["unread_files"]:
        parts.append(
            f"the {CHURN_DIFF_BUDGET_SECONDS}s time budget ran out with "
            f"{skipped['unread_commits']} commit(s) across "
            f"{skipped['unread_files']} file(s) unread"
        )
    if not parts:
        return ""
    return (
        "Partial results: "
        + "; ".join(parts)
        + ". Symbol commit and line counts are lower bounds - narrow the "
        "time range or path filter for complete numbers."
    )


def _symbol_line(rank: int, sym: dict) -> str:
    """A changed symbol's ranked line, without its counts."""
    location = f"{sym['file']}:{sym['line']}" if sym["line"] else sym["file"]
    return f"{rank:>3}. [{(sym['type'] or '???')[:3]}] {sym['name']} | {location} "


def _pairs(item_commits: dict[str, set[str]]) -> list[tuple[tuple[str, str], int]]:
    """The pairs of ``item_commits``' items changed together in at least two
    commits, with that count, most first, then by name."""
    commit_items: dict[str, set[str]] = defaultdict(set)
    for item, commits in item_commits.items():
        for c in commits:
            commit_items[c].add(item)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for items in commit_items.values():
        listed = sorted(items)
        for i in range(len(listed)):
            for j in range(i + 1, len(listed)):
                pair_counts[(listed[i], listed[j])] += 1
    frequent = [(pair, count) for pair, count in pair_counts.items() if count >= 2]
    return sorted(frequent, key=lambda x: (-x[1], x[0]))


def _narrowed(time_range, path_filter):
    """What a history query was narrowed by, as its message says it."""
    named = [
        what
        for what, given in (("time range", time_range), ("path filter", path_filter))
        if given
    ]
    return f" for the given {' and '.join(named)}" if named else ""


class GitHistoryAnalyzer:
    """Analyzes git history at the symbol level using descry metadata."""

    def __init__(
        self,
        project_root: str,
        graph_querier=None,
        churn_exclusions=_DEFAULT_CHURN_EXCLUSIONS,
        code_extensions=_DEFAULT_CODE_EXTENSIONS,
        git_timeout=30,
    ):
        """Initialize the analyzer.

        Args:
            project_root: Path to the git repository root.
            graph_querier: Optional GraphQuerier instance for symbol resolution.
            churn_exclusions: List of path patterns to exclude from churn analysis.
            code_extensions: Set of file extensions considered as code.
            git_timeout: Default timeout in seconds for git commands.
        """
        self.project_root = Path(project_root).resolve()
        self.querier = graph_querier
        self._verified = False
        self.churn_exclusions = churn_exclusions
        self.code_extensions = code_extensions
        self.default_timeout = git_timeout

    def _run_git(
        self,
        args: list[str],
        timeout: int | None = None,
        config: tuple[str, ...] = (),
    ) -> str:
        """Run a git command and return stdout.

        Args:
            args: Git command arguments (without 'git' prefix).
            timeout: Timeout in seconds. Defaults to self.default_timeout.
            config: Extra `-c key=value` options for this call.

        Returns:
            Command stdout as string.

        Raises:
            GitError: If git command fails or times out.
        """
        if timeout is None:
            timeout = self.default_timeout
        cmd = ["git", *_GIT_CONFIG_PINS, *config, args[0]]
        if args[0] in ("diff", "log"):
            cmd.extend(_GIT_DIFF_OPTS)
        cmd.extend(args[1:])
        try:
            # Captured as bytes: text mode would translate a bare CR inside
            # a diffed line into a line break of its own.
            result = subprocess.run(
                cmd,
                capture_output=True,
                cwd=str(self.project_root),
                timeout=timeout,
                env={**os.environ, **_GIT_ENV},
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                # An object a partial clone lacks: `cat-file` calls it a bad
                # file, other commands one they are unable to read.
                unread = "unable to read" in stderr or stderr.endswith(": bad file")
                if "lazy fetching disabled" in stderr or (
                    unread and self._is_partial_clone()
                ):
                    raise PartialCloneError(
                        "This repository is a partial clone and git objects this "
                        "analysis needs have not been downloaded. Descry does not "
                        "fetch them on demand: fetch the missing objects with git, "
                        "then retry."
                    )
                raise GitFailed(f"git {args[0]} failed.", detail=stderr)
            return result.stdout.decode("utf-8", errors="replace")
        except FileNotFoundError:
            raise GitError("git not found. Install git to use history tools.") from None
        except subprocess.TimeoutExpired:
            raise GitError(
                f"Git timed out after {timeout}s. Try a narrower time/path range."
            ) from None

    def _verify_git(self) -> None:
        """Verify we're in a git repo. Cached after first check."""
        if self._verified:
            return
        try:
            inside = self._run_git(["rev-parse", "--is-inside-work-tree"])
        except GitFailed as err:
            raise GitError("Not inside a git repository.") from err
        if inside.strip() != "true":
            raise GitError("Not inside a git repository.")
        self._verified = True

    def _is_shallow(self) -> bool:
        """Check if the repo is a shallow clone."""
        output = self._run_git(["rev-parse", "--is-shallow-repository"])
        return output.strip() == "true"

    def _is_partial_clone(self) -> bool:
        """Check if the repo has a promisor remote (was cloned with a filter)."""
        pattern = r"^(extensions\.partialclone|remote\..*\.promisor)$"
        try:
            self._run_git(["config", "--get-regexp", pattern])
        except GitError:
            return False
        return True

    def _worktree_filter_overrides(self, commit_range: str) -> tuple[str, ...]:
        """Return `-c` options that blank every configured filter driver.

        Only a single-ref range needs them: git diffs it against the working
        tree, running each modified file through its clean/process filter.
        """
        if ".." in commit_range:
            return ()
        keys = self._run_git(["config", "-z", "--name-only", "--list"]).split("\x00")
        names = {
            k.removeprefix("filter.").rpartition(".")[0]
            for k in keys
            if k.startswith("filter.")
        }
        overrides: list[str] = []
        for name in sorted(names - {""}):
            # `-c` splits key from value at the first '='.
            if "=" in name:
                raise GitError(
                    f"Refusing to diff the working tree: filter driver {name!r} "
                    "cannot be disabled. Use an explicit commit range (a..b)."
                )
            for setting in ("clean=", "process=", "required=false"):
                overrides += ["-c", f"filter.{name}.{setting}"]
        return tuple(overrides)

    def _empty_tree(self) -> str:
        """Return the id of the empty tree, the diff base of a root commit."""
        return self._run_git(["hash-object", "-t", "tree", os.devnull]).strip()

    def _file_walk(self, file_path: str):
        """Yield ``(commit, parents, path, parent path, start)`` for each
        commit that changed ``file_path``, children before parents: git's
        simplified history of the path, parents rewritten to those that
        changed it, and where a commit made the path by renaming another,
        that one's history from its parent on; ``start`` is the revision a
        history is read from, on its first commit, else None."""
        pending, seen = [("HEAD", file_path)], set()
        while pending:
            start, path = pending.pop(0)
            log = self._run_git(
                [
                    "log",
                    "--topo-order",
                    "--full-history",
                    "--parents",
                    "--format=%x00%H %P",
                    "--name-status",
                    "--no-renames",
                    start,
                    "--",
                    f":(literal){path}",
                ],
                timeout=30,
            )
            for index, record in enumerate(log.split("\x00")[1:]):
                head, _, body = record.partition("\n")
                commit, *parents = head.split()
                if (commit, path) in seen or not (body.strip() or parents):
                    # A commit read already, or one listed without the path
                    continue
                seen.add((commit, path))
                before = path
                if body.strip().startswith("A"):
                    # The path's history starts here; the commit's own parent
                    # may have had it under another name.
                    real = self._run_git(
                        ["rev-list", "--parents", "-n", "1", commit]
                    ).split()[1:2]
                    renames = real and self._run_git(
                        [
                            "diff",
                            "--name-status",
                            "-M",
                            "--diff-filter=R",
                            f"{real[0]}..{commit}",
                        ],
                        timeout=30,
                    )
                    for line in (renames or "").splitlines():
                        _, old, new = (line.split("\t") + ["", ""])[:3]
                        if new == path:
                            before, parents = old, real
                            pending.append((real[0], old))
                yield commit, parents, path, before, None if index else start

    def _churn_file(
        self,
        file_path,
        total,
        spans,
        counted,
        deadline,
        skipped,
        changed,
        most,
    ):
        """Credit each commit of ``counted`` (any where None) that changed
        ``file_path``, ``total`` of them in its log, to the symbols of
        ``spans`` its lines were in then, in ``changed[symbol][commit]``
        as `_touched` gives them: added lines by the commit's spans, removed
        ones by its parent's; a merge to those it changed against every
        parent. No more than ``most`` commits are read (None: all)."""
        # Revision -> {symbol: its span there}
        at = {"HEAD": {s[2]: s for s in self._spans_before(spans, "HEAD", file_path)}}
        read = 0
        for commit, parents, path, before, start in self._file_walk(file_path):
            # A history read from a revision starts with what it holds there.
            held = list((at.pop(commit, None) or at.pop(start, None) or {}).values())
            if not held:
                continue
            if time.monotonic() >= deadline:
                skipped["unread_files"] += 1
                skipped["unread_commits"] += max(total - read, 0)
                return
            wanted = counted is None or commit in counted
            if wanted and read == most:
                skipped["capped_files"] += 1
                return
            diffs = [
                self._parse_diff_hunks(
                    self._commit_file_diff(parent, commit, before, path)
                )
                for parent in parents or [self._empty_tree()]
            ]
            before_spans = [_spans_before(held, hunks) for hunks in diffs]
            touched = [
                _touched(hunks, held, old) for hunks, old in zip(diffs, before_spans)
            ]
            if wanted:
                read += 1
                for symbol in set(touched[0]).intersection(*touched[1:]):
                    changed[symbol][commit] = touched[0][symbol]
            # A symbol a merge leaves as one parent had it goes on in the
            # first such parent alone, as git follows a range.
            for index, (parent, old) in enumerate(zip(parents, before_spans)):
                for span, (_, _, symbol) in zip(old, held):
                    clean = [i for i, t in enumerate(touched) if symbol not in t]
                    if not clean or clean[0] == index:
                        at.setdefault(parent, {}).setdefault(symbol, span)

    def _commit_file_diff(
        self, parent: str, commit: str, before: str, path: str
    ) -> str:
        """The diff ``commit`` made to ``path``, ``before`` in ``parent``."""
        return self._run_git(
            [
                "diff",
                f"{parent}..{commit}",
                "-U0",
                "-M",
                "--",
                *(f":(literal){p}" for p in {before, path}),
            ],
            timeout=10,
        )

    def _ranges_covering(self, commit_hashes: list[str]) -> tuple[str, str]:
        """Return the diff range and rev-list range spanning `commit_hashes`.

        `commit_hashes` is newest first. A root commit has no parent to start
        from, so its diff starts at the empty tree.
        """
        oldest, newest = commit_hashes[-1], commit_hashes[0]
        try:
            self._run_git(["rev-parse", "--verify", "--quiet", f"{oldest}~1"])
        except GitError:
            return f"{self._empty_tree()}..{newest}", newest
        return f"{oldest}~1..{newest}", f"{oldest}~1..{newest}"

    def _counted(self, time_flags: list[str]) -> set[str] | None:
        """The commits ``time_flags`` select; None, every commit, for none."""
        if not time_flags:
            return None
        ranged = any(".." in flag for flag in time_flags)
        listed = ["rev-list", *time_flags, *([] if ranged else ["HEAD"])]
        return set(self._run_git(listed).split())

    def _parse_time_range(self, time_range: str | None) -> list[str]:
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
            _validate_git_ref(ref)
            return [f"{ref}..HEAD"]

        # Treat as a direct git --since value. Limit to characters git would
        # plausibly accept as a date expression so we don't pass crafted
        # user input straight through as a single argv token.
        if not re.match(r"^[A-Za-z0-9 :,.\-/+]{1,64}$", time_range):
            raise GitError(f"Invalid time_range: {time_range!r}")
        # git reads a date it cannot make out as now: one that is not "now"
        # and reads as it is no date at all.
        read, now = (
            int(self._run_git(["rev-parse", f"--since={at}"]).partition("=")[2])
            for at in (time_range, "now")
        )
        if time_range.lower() != "now" and abs(read - now) <= 1:
            raise GitError(
                f"Invalid time_range: {time_range!r} is no date git reads; "
                "use 'last N days', 'since <ref>' or a date"
            )
        return [f"--since={time_range}"]

    def _symbol_spans(self, file_path: str) -> list[tuple[int, int, str]]:
        """``(first line, last line, node ID)`` of each function, method and
        class of the graph in a file, as the file stands today."""
        if not self.querier:
            return []
        file_id = f"FILE:{file_path}"
        return [
            (start, node["metadata"].get("end_lineno", start), node["id"])
            for node in self.querier.data.get("nodes", [])
            if node["id"].startswith(file_id + "::")
            and node["type"] in ("Function", "Method", "Class")
            and (start := node["metadata"].get("lineno", 0)) > 0
        ]

    def _spans_before(self, spans: list, revision: str, file_path: str) -> list:
        """``spans`` of a file today as they stood in ``revision``, carried
        back through the line diff of the revision's copy against the file
        on disk, which no git filter reads."""
        old = self._run_git(["cat-file", "blob", f"{revision}:{file_path}"])
        disk = source_path(self.project_root, file_path)
        # Read as it is stored, a `\r` before each `\n` kept, as git does
        try:
            with open_regular(
                disk or os.devnull, "r", encoding="utf-8", errors="replace", newline=""
            ) as f:
                new = f.read()
        except OSError:
            new = ""
        # Lines counted at `\n` alone, as git and the parsers count them, and
        # compared without the `\r` a checkout may write before one
        old, new = (
            [line.removesuffix("\r") for line in source_lines(text)]
            for text in (old, new)
        )
        return _spans_before(spans, _line_diff(old, new))

    def _parse_diff_hunks(self, diff_output: str) -> list[dict]:
        """Parse unified diff output into structured hunk data.

        Args:
            diff_output: Raw git diff output.

        Returns:
            List of dicts with keys: file, old_start, old_count,
            new_start, new_count, lines.
        """
        hunks = []
        current_file = None
        current_hunk = None
        # Lines the current hunk's header says its body still holds. Inside
        # a body every line is content, even one that reads like a header.
        old_left = new_left = 0

        for line in diff_output.split("\n"):
            if (old_left > 0 or new_left > 0) and line.startswith(("+", "-", " ")):
                current_hunk["lines"].append(line)
                old_left -= not line.startswith("+")
                new_left -= not line.startswith("-")
            elif line.startswith("diff --git"):
                current_file = None
            elif line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("@@ "):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match and current_file:
                    current_hunk = {
                        "file": current_file,
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2) or 1),
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4) or 1),
                        "lines": [],
                    }
                    hunks.append(current_hunk)
                    old_left = current_hunk["old_count"]
                    new_left = current_hunk["new_count"]

        return hunks

    def _get_node_display(self, node_id: str) -> tuple[str, str]:
        """The name and location of a node ID."""
        if not self.querier:
            parts = node_id.replace("FILE:", "").split("::")
            file_path = parts[0]
            name = parts[-1] if len(parts) > 1 else file_path
            return name, file_path

        node = self.querier.nodes.get(node_id)
        if not node:
            parts = node_id.replace("FILE:", "").split("::")
            return parts[-1] if len(parts) > 1 else node_id, ""

        meta = node.get("metadata", {})
        name = meta.get("name", "unknown")

        # Build location
        file_path = file_of(node_id) if node_id.startswith("FILE:") else ""
        lineno = meta.get("lineno")
        location = f"{file_path}:{lineno}" if lineno else file_path

        return name, location

    def _symbol_entry(self, node_id: str) -> dict:
        """A changed symbol's `id`, `type` (its node's, None for a file or a
        node the graph lacks), `name`, `file` and `line`."""
        node = self.querier.nodes.get(node_id) if self.querier else None
        return {
            "id": node_id,
            "type": node.get("type") if node else None,
            "name": self._get_node_display(node_id)[0],
            "file": file_of(node_id),
            "line": (node or {}).get("metadata", {}).get("lineno"),
        }

    @staticmethod
    def render_churn(data: dict) -> str:
        """The markdown of a `get_churn_structured` result."""
        note = data.get("truncation_note")
        partial = f"\n\n*{note}*" if note else ""
        if "message" in data:
            return data["message"] + partial
        range_note = f" ({data['time_range']})" if data["time_range"] else ""
        if "files" in data:
            files = data["files"]
            lines = [f"### Top {len(files)} Most Changed Files{range_note}\n"]
            lines += [
                f"{rank:>3}. {f['file']} ({f['commits']} commits)"
                for rank, f in enumerate(files, 1)
            ]
            return "\n".join(lines)
        if "symbols" in data:
            symbols = data["symbols"]
            path_note = f" in {data['path_filter']}" if data["path_filter"] else ""
            lines = [
                f"### Top {len(symbols)} Most Changed Symbols{range_note}{path_note}\n"
            ]
            for rank, sym in enumerate(symbols, 1):
                stats = (
                    f", +{sym['added']}/-{sym['removed']}"
                    if sym["added"] or sym["removed"]
                    else ""
                )
                lines.append(
                    _symbol_line(rank, sym) + f"({sym['commits']} commits{stats})"
                )
            return "\n".join(lines) + partial
        pairs = data["pairs"]
        if not pairs:
            return (
                "No co-change patterns found (need pairs with 2+ shared commits)."
                if data.get("file_level")
                else "No co-change patterns found in code files."
            ) + partial
        if data.get("file_level"):
            lines = [
                f"### Top {len(pairs)} Co-Change Pairs (file-level)\n",
                "*Symbol-level data was too sparse; showing file-level co-changes:*\n",
            ]
            lines += [
                f"{rank:>3}. {p['a_name']} <-> {p['b_name']} ({p['shared_commits']} shared commits)"
                for rank, p in enumerate(pairs, 1)
            ]
        else:
            lines = [
                f"### Top {len(pairs)} Co-Change Pairs\n",
                "Symbols that frequently change in the same commit:\n",
            ]
            for rank, p in enumerate(pairs, 1):
                a = f" ({p['a_file']})" if p["a_file"] else ""
                b = f" ({p['b_file']})" if p["b_file"] else ""
                lines.append(
                    f"{rank:>3}. {p['a_name']}{a} <-> {p['b_name']}{b} ({p['shared_commits']} shared commits)"
                )
        return "\n".join(lines) + partial

    def get_churn_structured(
        self,
        time_range: str | None = None,
        path_filter: str | None = None,
        limit: int = 20,
        mode: str = "symbols",
        exclude_generated: bool = True,
    ) -> dict:
        """Analyze code churn, returning structured data for UI rendering.

        Returns dict with keys: mode, time_range, path_filter, and one of:
        - symbols: list of {id, type, name, file, line, commits, added, removed},
          `type` the node's type, None for a file or a node the graph lacks
        - files: list of {file, commits}
        - pairs: list of {a_id, a_name, a_file, b_id, b_name, b_file, shared_commits}
        Symbol and co-change results carry truncation_note when the churn
        limits left commits undiffed. Instead of the list, `message` says
        why there is nothing to list; an unknown mode is `{"error",
        "reason"}`.
        """
        if mode not in {m.value for m in ChurnMode}:
            return fail("invalid", _UNKNOWN_CHURN_MODE.format(mode))
        self._verify_git()
        try:
            if path_filter:
                _validate_pathspec(path_filter)
            time_flags = self._parse_time_range(time_range)
        except GitError as e:
            return fail("invalid", str(e))

        # Every commit that changed a file, through each parent of a merge,
        # whatever the pathspec
        git_args = [
            "log",
            _NAME_ONLY_FORMAT,
            "--name-only",
            "--no-merges",
            "--full-history",
        ]
        git_args.extend(time_flags)
        if path_filter:
            git_args.extend(["--", path_filter])
        elif exclude_generated:
            exclusions = []
            for p in self.churn_exclusions:
                _validate_pathspec(p)
                exclusions.append(f":!{p}")
            git_args.extend(["--", "."] + exclusions)

        output = self._run_git(git_args, timeout=60)
        base = {
            "mode": mode,
            "time_range": time_range or "",
            "path_filter": path_filter or "",
        }

        if not output.strip():
            return {
                **base,
                "message": f"No commits found{_narrowed(time_range, path_filter)}.",
            }

        file_commits = _parse_name_only_log(output)

        if not file_commits:
            return {
                **base,
                "message": "No file changes found"
                f"{_narrowed(time_range, path_filter)}.",
            }

        # FILE MODE
        if mode == "files":
            sorted_files = sorted(
                file_commits.items(), key=lambda x: len(x[1]), reverse=True
            )[:limit]
            base["files"] = [
                {"file": fp, "commits": len(commits)} for fp, commits in sorted_files
            ]
            return base

        # SYMBOL / CO-CHANGE: attribute changes to symbols
        sorted_files = sorted(
            file_commits.items(), key=lambda x: len(x[1]), reverse=True
        )[:50]

        symbol_commits: dict[str, set[str]] = defaultdict(set)
        changed: dict[str, dict[str, dict]] = defaultdict(dict)
        skipped: dict[str, int] = defaultdict(int)
        deadline = time.monotonic() + CHURN_DIFF_BUDGET_SECONDS

        # A commit's lines are read against the symbols as they stood in it,
        # as `git log -L` follows a range: today's carried back to HEAD, then
        # through the file's history, each commit's spans carried to each of
        # its parents through the diff between them, across renames.
        counted = self._counted(time_flags)
        for file_path, commits in sorted_files:
            spans = self._symbol_spans(file_path)
            if not spans:
                symbol_commits[f"FILE:{file_path}"].update(commits)
                continue
            try:
                self._churn_file(
                    file_path,
                    len(commits),
                    spans,
                    counted,
                    deadline,
                    skipped,
                    changed,
                    CHURN_MAX_COMMITS_PER_FILE,
                )
            except PartialCloneError:
                raise
            except GitError:
                skipped["unread_files"] += 1
        for symbol, commits in changed.items():
            symbol_commits[symbol].update(commits)

        partial_note = _churn_partial_note(skipped)

        if partial_note:
            base["truncation_note"] = partial_note
        if not symbol_commits:
            return {
                **base,
                "message": "No symbol-level changes detected. Graph may not cover these files.",
            }

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
        if not symbol_commits:
            return {
                **base,
                "message": "No changes to code files found"
                f"{_narrowed(time_range, path_filter)}.",
            }

        # CO-CHANGE MODE
        if mode == "co-change":
            return self._structured_co_change(symbol_commits, limit, file_commits, base)

        # SYMBOLS MODE
        sorted_symbols = sorted(
            symbol_commits.items(), key=lambda x: (-len(x[1]), x[0])
        )[:limit]

        symbols = []
        for node_id, commits in sorted_symbols:
            by_commit = changed.get(node_id, {}).values()
            symbols.append(
                {
                    **self._symbol_entry(node_id),
                    "commits": len(commits),
                    "added": sum(t["added"] for t in by_commit),
                    "removed": sum(t["removed"] for t in by_commit),
                }
            )

        base["symbols"] = symbols
        return base

    def _structured_co_change(
        self,
        symbol_commits: dict[str, set[str]],
        limit: int,
        file_commits: dict[str, list[str]],
        base: dict,
    ) -> dict:
        """Build structured co-change data."""
        graph_symbols = {k: v for k, v in symbol_commits.items() if "::" in k}

        pairs = []
        for (sym_a, sym_b), count in _pairs(graph_symbols)[:limit]:
            name_a, loc_a = self._get_node_display(sym_a)
            name_b, loc_b = self._get_node_display(sym_b)
            pairs.append(
                {
                    "a_id": sym_a,
                    "a_name": name_a,
                    "a_file": loc_a,
                    "b_id": sym_b,
                    "b_name": name_b,
                    "b_file": loc_b,
                    "shared_commits": count,
                }
            )

        if not pairs and file_commits:
            # Fall back to file-level co-change
            code_files = {
                f: commits
                for f, commits in file_commits.items()
                if Path(f).suffix.lower() in self.code_extensions
            }
            if code_files:
                for (file_a, file_b), count in _pairs(code_files)[:limit]:
                    pairs.append(
                        {
                            "a_id": None,
                            "a_name": file_a,
                            "a_file": file_a,
                            "b_id": None,
                            "b_name": file_b,
                            "b_file": file_b,
                            "shared_commits": count,
                        }
                    )
                base["file_level"] = True

        base["pairs"] = pairs
        return base

    def get_evolution_structured(
        self,
        name: str,
        time_range: str | None = None,
        limit: int = 10,
        show_diff: bool = False,
    ) -> dict:
        """The commits that changed a symbol, newest first.

        Args:
            name: The symbol's node ID.
            time_range: Time range filter.
            limit: Maximum commits to show.
            show_diff: Include actual diff hunks.

        Returns:
            `{"error", "reason"}` when the node is not a file's or git
            fails; else `symbol`, `node_id`, `location`,
            `mode` ("lines" from the symbol's own lines, those no definition
            inside it holds, followed as churn follows them; "file" from the
            file's log for a symbol with no line span), `shallow`,
            `show_diff`, `truncation_note` where the time budget cut the
            walk short, and `commits`, each `{hash, date, author, subject}`
            and, in "lines" mode, `added`, `removed` and `diff`.
        """
        self._verify_git()
        try:
            time_flags = self._parse_time_range(time_range)
        except GitError as e:
            return fail("invalid", str(e))

        if not name.startswith("FILE:"):
            return fail(
                "not_found",
                f"Could not resolve symbol '{name}'. "
                "Try descry_search to find the exact name.",
            )
        file_path = file_of(name)
        _validate_git_file(file_path)
        symbol_name, location = self._get_node_display(name)
        found = {
            "symbol": symbol_name,
            "node_id": name,
            "location": location or file_path,
            "show_diff": show_diff,
        }

        # The symbol's own lines, those no definition inside it holds,
        # followed through history as churn follows them; a symbol with no
        # line span has its file's commits.
        spans = self._symbol_spans(file_path)
        own = next((s for s in spans if s[2] == name), None)
        try:
            if own:
                held = [s for s in spans if own[0] <= s[0] and s[1] <= own[1]]
                changed, skipped = defaultdict(dict), defaultdict(int)
                deadline = time.monotonic() + CHURN_DIFF_BUDGET_SECONDS
                counted = self._counted(time_flags)
                self._churn_file(
                    file_path, 0, held, counted, deadline, skipped, changed, None
                )
                by_commit = list(changed[name].items())[:limit]
                log = (
                    self._run_git(
                        ["log", "--no-walk=unsorted", _HEADER_FORMAT, "--date=short"]
                        + [commit for commit, _ in by_commit]
                    )
                    if by_commit
                    else ""
                )
                headers = [
                    h for line in log.split("\n") if (h := _parse_log_header(line))
                ]
                commits = [
                    {**header, **lines, "diff": "\n".join(lines["diff"])}
                    for header, (_, lines) in zip(headers, by_commit)
                ]
                if skipped:
                    found["truncation_note"] = (
                        f"Partial results: the {CHURN_DIFF_BUDGET_SECONDS}s time "
                        "budget ran out before the symbol's first commit was read."
                    )
            else:
                log = self._run_git(
                    ["log", _HEADER_FORMAT, "--date=short", "--no-merges"]
                    + time_flags
                    + ["--", file_path],
                    timeout=30,
                )
                headers = (_parse_log_header(line) for line in log.split("\n"))
                commits = [h for h in headers if h][:limit]
        except PartialCloneError:
            raise
        except GitError as e:
            return fail("failed", f"Error getting history: {e}", _detail=e.detail)
        if not commits:
            return {
                "message": f"No commits found for '{symbol_name}' in {file_path}"
                f"{_narrowed(time_range, None)}."
            }
        return {
            **found,
            "mode": "lines" if own else "file",
            "shallow": bool(own) and self._is_shallow(),
            "commits": commits,
        }

    @staticmethod
    def render_evolution(data: dict) -> str:
        """The markdown of a `get_evolution_structured` result."""
        if "message" in data:
            return data["message"]
        commits = data["commits"]
        authors = len({c["author"] for c in commits})
        lines = [f"### Evolution of `{data['symbol']}` ({data['location']})\n"]
        if data["mode"] == "file":
            note = "*Note: Using file-level history (the graph has no lines for this symbol)*"
            if data["show_diff"]:
                note += "\n*Diffs were requested but are not available in file-level fallback mode.*"
            lines.append(note + "\n")
            lines.append(f"{len(commits)} commit(s) | {authors} author(s)\n")
        else:
            dates = [c["date"] for c in commits]
            span = f"{dates[-1]} to {dates[0]}" if len(dates) > 1 else dates[0]
            lines.append(f"{len(commits)} commit(s) | {authors} author(s) | {span}\n")
            if data["shallow"]:
                lines.append(
                    "*Warning:  (shallow clone - history may be incomplete)*\n"
                )
            if data.get("truncation_note"):
                lines.append(f"*{data['truncation_note']}*\n")

        for i, commit in enumerate(commits, 1):
            lines.append(
                f"{i}. **{commit['hash']}** ({commit['date']}) {commit['author']}"
            )
            lines.append(f"   {commit['subject']}")
            if commit.get("added") or commit.get("removed"):
                lines.append(f"   +{commit['added']}/-{commit['removed']} lines")
            if data["show_diff"] and commit.get("diff", "").strip():
                # Indented so marked.js keeps the diff inside the list item
                diff_lines = commit["diff"].strip().split("\n")
                indented = "\n".join("   " + dl for dl in diff_lines[:30])
                if len(diff_lines) > 30:
                    indented += "\n   ... (truncated)"
                lines.append(f"   ```diff\n{indented}\n   ```")
            lines.append("")

        return "\n".join(lines)

    def _attribute_hunks_to_symbols(self, hunks, spans_of) -> dict[str, dict]:
        """Attribute the added and removed lines of the `-U0` diff ``hunks``
        to the symbols holding them: added ones by the spans ``spans_of``
        gives a file on the diff's new side, removed ones by those spans
        carried back to its old side (see `_touched`)."""
        file_hunks: dict[str, list[dict]] = defaultdict(list)
        for hunk in hunks:
            file_hunks[hunk["file"]].append(hunk)
        modified_symbols: dict[str, dict] = {}
        for file_path, fhunks in file_hunks.items():
            spans = spans_of(file_path)
            touched = _touched(fhunks, spans, _spans_before(spans, fhunks))
            modified_symbols.update(touched)
        return modified_symbols

    def _collect_changes(
        self,
        commit_range: str | None,
        time_range: str | None,
        path_filter: str | None,
    ) -> tuple[list[dict], int, dict[str, dict]] | dict:
        """Changed files, commit count and per-symbol line counts for a range.

        Instead `{"error", "reason"}` when the arguments are wrong or the
        git run failed, or `{"message"}` saying why there is nothing to
        report.
        """
        self._verify_git()

        # Determine effective commit range. User-supplied commit_range is
        # validated; internally-built ranges use trusted sha output from
        # `git log --format=%H`.
        if commit_range:
            try:
                _validate_git_range(commit_range)
            except GitError as e:
                return fail(
                    "invalid",
                    f"Invalid commit range '{commit_range}': {e}",
                )
        if path_filter:
            try:
                _validate_pathspec(path_filter)
            except GitError as e:
                return fail(
                    "invalid",
                    f"Invalid path filter '{path_filter}': {e}",
                )

        effective_range = commit_range
        count_range = None
        if commit_range:
            if "..." in commit_range:
                # The diff spans the merge-base to the second ref; the
                # count is of those commits, not the symmetric difference.
                count_range = commit_range.replace("...", "..")
            elif ".." in commit_range:
                count_range = commit_range
                # Two refs that diverged: the commits the range counts run
                # from the merge base, and the diff of them does too; the
                # trees of both refs hold changes only the first made. Refs
                # of no common history, or a tree, have no merge base: the
                # diff is of the two trees.
                first, _, second = commit_range.partition("..")
                try:
                    self._run_git(["merge-base", first, second], timeout=10)
                except GitError:
                    pass
                else:
                    effective_range = commit_range.replace("..", "...")
            else:
                # A single ref diffs the working tree: nothing in it is
                # committed.
                count_range = ""
        elif not effective_range:
            if time_range:
                # Get commits in time range and build a range
                try:
                    time_flags = self._parse_time_range(time_range)
                except GitError as e:
                    return fail("invalid", str(e))
                git_args = ["log", "--format=%H", "--no-merges"] + time_flags
                output = self._run_git(git_args, timeout=30)
                commit_hashes = [
                    h.strip() for h in output.strip().split("\n") if h.strip()
                ]
                if not commit_hashes:
                    return {"message": "No commits found for the given time range."}
                # oldest..newest
                effective_range, count_range = self._ranges_covering(commit_hashes)
            else:
                effective_range, count_range = self._ranges_covering(["HEAD"])

        # Get changed files with stats
        diff_stat_args = ["diff", "--numstat", effective_range]
        if path_filter:
            diff_stat_args.extend(["--", path_filter])

        try:
            filter_overrides = self._worktree_filter_overrides(effective_range)
            numstat_output = self._run_git(
                diff_stat_args, timeout=30, config=filter_overrides
            )
        except PartialCloneError:
            raise
        except GitError as e:
            # The range passed validation above; the work itself failed
            # (an absent history in a shallow clone, a timeout).
            return fail(
                "failed",
                f"Getting changes for '{commit_range or effective_range}' failed: {e}",
                _detail=e.detail,
            )

        if not numstat_output.strip():
            return {
                "message": f"No changes found in {commit_range or effective_range}"
                f"{_narrowed(None if commit_range else time_range, path_filter)}."
            }

        # Parse numstat output
        changed_files: list[dict] = []
        for line in numstat_output.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 3:
                added = int(parts[0]) if parts[0] != "-" else 0
                removed = int(parts[1]) if parts[1] != "-" else 0
                changed_files.append(
                    {
                        "file": parts[2],
                        "added": added,
                        "removed": removed,
                    }
                )

        # Count commits in range; a single ref diffs the working tree,
        # where nothing is committed.
        if count_range == "":
            commit_count = 0
        else:
            try:
                commit_count_output = self._run_git(
                    ["rev-list", "--count", count_range or effective_range], timeout=10
                )
                commit_count = int(commit_count_output.strip())
            except (GitError, ValueError):
                commit_count = 0

        # The full diff, without context, for symbol attribution
        diff_args = ["diff", "-U0", effective_range]
        if path_filter:
            diff_args.extend(["--", path_filter])

        try:
            diff_output = self._run_git(diff_args, timeout=60, config=filter_overrides)
        except GitError:
            diff_output = ""

        hunks = self._parse_diff_hunks(diff_output) if diff_output else []

        # The range's new side is its last ref, a single ref's the working tree.
        end = effective_range.rpartition("..")[2] if ".." in effective_range else None
        end = end if end is None or end else "HEAD"
        try:
            symbols = self._attribute_hunks_to_symbols(
                hunks,
                lambda f: (
                    self._spans_before(self._symbol_spans(f), end, f)
                    if end
                    else self._symbol_spans(f)
                ),
            )
        except PartialCloneError:
            raise
        except GitError:
            symbols = {}
        return changed_files, commit_count, symbols

    @staticmethod
    def render_changes(data: dict) -> str:
        """The markdown of a `get_changes_structured` result."""
        from descry.query import GraphQuerier

        if "message" in data:
            return data["message"]
        commits = data["commits"]
        count_note = (
            f" ({commits} commit{'s' if commits != 1 else ''})" if commits else ""
        )
        lines = [
            f"### Change Impact: {data['range']}{count_note}\n",
            f"Scope: {data['files_count']} file(s) | {data['symbols_count']} symbol(s) affected\n",
        ]
        if "symbols" not in data:
            lines.append(
                "*No symbol-level attribution available (graph may not cover these files).*\n"
            )
            lines.append("#### Changed Files")
            lines += [
                f"  - {f['file']} (+{f['added']}/-{f['removed']})"
                for f in data["changed_files"]
            ]
            return "\n".join(lines)

        lines.append(f"#### Modified ({len(data['symbols'])})")
        for rank, sym in enumerate(data["symbols"], 1):
            lines.append(
                _symbol_line(rank, sym) + f"(+{sym['added']}/-{sym['removed']})"
            )
            if callers := sym.get("callers"):
                names = ", ".join(
                    c["name"] + GraphQuerier.one_of(c.get("candidates"))
                    for c in callers
                )
                hidden = sym["callers_possible"] - sum(
                    bool(c.get("candidates")) for c in callers
                )
                hidden_certain = sym["callers_total"] - len(callers) - hidden
                more = (
                    f"; {GraphQuerier.more_callers(hidden_certain, hidden)}"
                    if sym["callers_total"] > len(callers)
                    else ""
                )
                lines.append(f"     Callers: {names}{more}")
        return "\n".join(lines)

    def _symbol_callers(self, node_id):
        """The first five callers of the symbol ``node_id``, certain ones
        first, as `GraphQuerier.reference` gives each; the number of its
        callers; and how many of those only may be."""
        caller_ids, ambiguous = self.querier.ordered_callers(node_id)
        callers = [
            self.querier.reference(caller_id, ambiguous.get(caller_id))
            for caller_id in caller_ids[:5]
        ]
        return callers, len(caller_ids), len(ambiguous)

    def get_changes_structured(
        self,
        commit_range: str | None = None,
        time_range: str | None = None,
        path_filter: str | None = None,
        show_callers: bool = True,
        limit: int = 50,
    ) -> dict:
        """Analyze change impact, returning structured data for UI rendering.

        Returns dict with keys: range, commits, files_count, symbols_count,
        symbols (list), changed_files (list when no symbol attribution); or
        `{"error", "reason"}` for wrong arguments, or `message` saying why
        there is nothing to report. A symbol's
        `type` is its node's, None for a file or a node the graph lacks.
        """
        changes = self._collect_changes(commit_range, time_range, path_filter)
        if isinstance(changes, dict):
            return changes
        changed_files, commit_count, modified_symbols = changes

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
            entry = {
                **self._symbol_entry(node_id),
                "added": stats["added"],
                "removed": stats["removed"],
            }

            if show_callers and self.querier:
                callers, total, possible = self._symbol_callers(node_id)
                if callers:
                    entry["callers"] = callers
                    entry["callers_total"] = total
                    entry["callers_possible"] = possible

            symbols.append(entry)

        result["symbols"] = symbols
        return result
