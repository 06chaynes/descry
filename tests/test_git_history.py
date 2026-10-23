"""Tests for descry.git_history — Git history analysis and churn detection."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from descry import git_history
from descry.git_history import (
    GitError,
    GitHistoryAnalyzer,
    PartialCloneError,
    _validate_git_file,
    _validate_git_range,
    _validate_git_ref,
    _validate_git_symbol,
    _validate_pathspec,
)


def _git(repo, *args, check=True):
    """Run git in a fixture repo and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=check,
    )


def _init_repo(path, *init_args):
    """Create a repo that commits without consulting the user's git setup."""
    path.mkdir()
    _git(path, "init", "-q", *init_args)
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "Tester")
    _git(path, "config", "commit.gpgsign", "false")
    return path


def _commit(repo, message, **files):
    """Write `files` (name -> text or bytes), stage them and commit."""
    for name, content in files.items():
        if isinstance(content, bytes):
            (repo / name).write_bytes(content)
        else:
            (repo / name).write_text(content, encoding="utf-8")
        _git(repo, "add", "--", name)
    _git(repo, "commit", "-q", "-m", message)


class FakeQuerier:
    """The slice of GraphQuerier that GitHistoryAnalyzer uses."""

    def __init__(self, nodes):
        self.data = {"schema_version": 2, "nodes": nodes, "edges": []}

    def get_node_info(self, node_id):
        return next((n for n in self.data["nodes"] if n["id"] == node_id), None)

    def find_nodes_by_name(self, name, fuzzy=False):
        return [n for n in self.data["nodes"] if n["metadata"]["name"] == name]

    def ordered_callers(self, name):
        return [], {}


def _function_node(file_path, name, lineno, end_lineno):
    return {
        "id": f"FILE:{file_path}::{name}",
        "type": "Function",
        "metadata": {"name": name, "lineno": lineno, "end_lineno": end_lineno},
    }


@pytest.fixture
def isolated_git(monkeypatch):
    """Keep the user's global and system git config out of fixture repos."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


class TestGitHistoryParseTimeRange:
    """Tests for GitHistoryAnalyzer._parse_time_range."""

    def test_last_n_days(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range("last 30 days") == ["--since=30 days ago"]

    def test_last_n_weeks(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range("last 2 weeks") == ["--since=2 weeks ago"]

    def test_last_n_months(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range("last 3 months") == ["--since=3 months ago"]

    def test_since_ref(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range("since v1.0") == ["v1.0..HEAD"]

    def test_since_commit_hash(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range("since abc1234") == ["abc1234..HEAD"]

    def test_none_returns_empty(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range(None) == []


class TestGitHistoryParseDiffHunks:
    """Tests for GitHistoryAnalyzer._parse_diff_hunks."""

    def test_basic_diff(self):
        analyzer = GitHistoryAnalyzer(".")

        diff = """\
diff --git a/src/config.rs b/src/config.rs
--- a/src/config.rs
+++ b/src/config.rs
@@ -10,3 +10,4 @@ fn validate() {
 unchanged line
-removed line
+added line
+another added line
"""
        hunks = analyzer._parse_diff_hunks(diff)
        assert len(hunks) == 1
        assert hunks[0]["file"] == "src/config.rs"
        assert hunks[0]["old_start"] == 10
        assert hunks[0]["new_start"] == 10
        assert len(hunks[0]["lines"]) == 4

    def test_multiple_hunks(self):
        analyzer = GitHistoryAnalyzer(".")

        diff = """\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -5,2 +5,3 @@ fn foo() {
 line
+new line
@@ -20,3 +21,2 @@ fn bar() {
 context
-removed
"""
        hunks = analyzer._parse_diff_hunks(diff)
        assert len(hunks) == 2
        assert hunks[0]["new_start"] == 5
        assert hunks[1]["new_start"] == 21

    def test_new_file(self):
        analyzer = GitHistoryAnalyzer(".")

        diff = """\
diff --git a/src/new.rs b/src/new.rs
--- /dev/null
+++ b/src/new.rs
@@ -0,0 +1,3 @@
+fn hello() {
+    println!("hi");
+}
"""
        hunks = analyzer._parse_diff_hunks(diff)
        assert len(hunks) == 1
        assert hunks[0]["file"] == "src/new.rs"
        assert hunks[0]["new_start"] == 1

    def test_empty_context_line_does_not_swallow_the_next_file_header(self):
        analyzer = GitHistoryAnalyzer(".")

        # What git writes with diff.suppressBlankEmpty set.
        diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
 one

-two
+TWO
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-x
+y
"""
        hunks = analyzer._parse_diff_hunks(diff)
        assert [h["file"] for h in hunks] == ["a.py", "b.py"]
        assert hunks[1]["lines"] == ["-x", "+y"]


class TestGitHistoryBuildFileLineMap:
    """Tests for GitHistoryAnalyzer._build_file_line_map with mock graph data."""

    def test_maps_lines_to_symbols(self):
        # Create a mock querier with graph data
        graph_data = {
            "schema_version": 2,
            "nodes": [
                {
                    "id": "FILE:src/auth.rs::validate_token",
                    "type": "Function",
                    "metadata": {
                        "name": "validate_token",
                        "lineno": 10,
                        "end_lineno": 20,
                    },
                },
                {
                    "id": "FILE:src/auth.rs::check_perms",
                    "type": "Function",
                    "metadata": {"name": "check_perms", "lineno": 25, "end_lineno": 35},
                },
            ],
            "edges": [],
        }

        class MockQuerier:
            def __init__(self, data):
                self.data = data

        analyzer = GitHistoryAnalyzer(".")
        analyzer.querier = MockQuerier(graph_data)

        line_map = analyzer._build_file_line_map("src/auth.rs")

        # Lines 10-20 should map to validate_token
        assert line_map[10] == "FILE:src/auth.rs::validate_token"
        assert line_map[15] == "FILE:src/auth.rs::validate_token"
        assert line_map[20] == "FILE:src/auth.rs::validate_token"

        # Lines 25-35 should map to check_perms
        assert line_map[25] == "FILE:src/auth.rs::check_perms"
        assert line_map[30] == "FILE:src/auth.rs::check_perms"

        # Lines between functions should not be mapped
        assert 22 not in line_map

    def test_no_querier_returns_empty(self):
        analyzer = GitHistoryAnalyzer(".")
        analyzer.querier = None

        line_map = analyzer._build_file_line_map("src/auth.rs")
        assert line_map == {}


class TestGitHistoryEndToEnd:
    """End-to-end tests that use the actual git repository."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer rooted at the descry project."""
        import subprocess

        # Find project root (where .git exists)
        test_dir = Path(__file__).resolve().parent
        project_root = test_dir.parent
        if not (project_root / ".git").exists():
            pytest.skip("Not inside a git repository")
        # Check if the repo has any commits
        try:
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            pytest.skip("Git repository has no commits yet")
        return GitHistoryAnalyzer(str(project_root))

    def test_churn_files_mode(self, analyzer):
        """Should return file-level churn data."""
        result = analyzer.get_churn(time_range="last 30 days", mode="files", limit=5)
        assert "Most Changed Files" in result or "No commits" in result

    def test_churn_symbols_mode(self, analyzer):
        """Should return symbol-level churn data (may fall back to file-level)."""
        result = analyzer.get_churn(time_range="last 30 days", mode="symbols", limit=5)
        # Either has symbol results or "No commits" - both are valid
        assert "Changed" in result or "No commits" in result or "No symbol" in result

    def test_evolution_basic(self, analyzer):
        """Should return evolution for a known file."""
        # Use the git_history module itself as the target
        result = analyzer.get_evolution("git_history.py", time_range="last 30 days")
        # Either finds it or says it can't resolve
        assert "Evolution" in result or "Could not resolve" in result

    def test_changes_head(self, analyzer):
        """Should return change impact for HEAD~1..HEAD."""
        import subprocess

        test_dir = Path(__file__).resolve().parent
        project_root = test_dir.parent
        # Need at least 2 commits for HEAD~1..HEAD
        ret = subprocess.run(
            ["git", "rev-parse", "HEAD~1"],
            cwd=str(project_root),
            capture_output=True,
            check=False,
        )
        if ret.returncode != 0:
            pytest.skip("Need at least 2 commits for HEAD~1..HEAD")
        result = analyzer.get_changes(commit_range="HEAD~1..HEAD")
        assert "Change Impact" in result or "No changes" in result

    def test_verify_git(self, analyzer):
        """Should successfully verify git repo."""
        # Should not raise
        analyzer._verify_git()
        assert analyzer._verified

    def test_is_shallow(self, analyzer):
        """Should detect shallow clone status."""
        # Just verify it returns a boolean without error
        result = analyzer._is_shallow()
        assert isinstance(result, bool)


class TestGitHistoryNonUtf8Output:
    """Regression: git output containing non-UTF-8 bytes must not crash.

    Observed in the wild on c-postgres and rust-coreutils where `git diff`
    produced diff content containing Latin-1 bytes (0x92, 0xfd) — the
    previous `text=True` decode crashed with UnicodeDecodeError. The fix
    is to decode with errors='replace'.
    """

    def test_run_git_tolerates_non_utf8_bytes(self, tmp_path):
        """_run_git must not raise UnicodeDecodeError on binary-ish output."""
        import subprocess as sp

        # Build a minimal git repo and plant a commit whose diff contains a
        # non-UTF-8 byte sequence (0x92 — Windows-1252 right single quote).
        sp.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
        sp.run(["git", "config", "user.email", "t@t"], cwd=str(tmp_path), check=True)
        sp.run(["git", "config", "user.name", "t"], cwd=str(tmp_path), check=True)
        # Disable gpg signing inside the fixture repo so the test does not
        # depend on the user's signing setup (e.g. 1Password SSH agent).
        sp.run(
            ["git", "config", "commit.gpgsign", "false"],
            cwd=str(tmp_path),
            check=True,
        )
        sp.run(
            ["git", "config", "tag.gpgsign", "false"],
            cwd=str(tmp_path),
            check=True,
        )
        # File with non-UTF-8 content (Latin-1 curly quote, 0x92)
        target = tmp_path / "legacy.txt"
        target.write_bytes(b"hello \x92 world\n")
        sp.run(["git", "add", "legacy.txt"], cwd=str(tmp_path), check=True)
        sp.run(
            ["git", "commit", "-q", "-m", "initial"],
            cwd=str(tmp_path),
            check=True,
        )
        # Modify it so `git diff HEAD~` has real hunk content
        target.write_bytes(b"hello \x92 world\nsecond \xfd line\n")
        sp.run(["git", "add", "legacy.txt"], cwd=str(tmp_path), check=True)
        sp.run(
            ["git", "commit", "-q", "-m", "second"],
            cwd=str(tmp_path),
            check=True,
        )

        analyzer = GitHistoryAnalyzer(str(tmp_path))
        # This must not raise — pre-fix it raised UnicodeDecodeError on 0x92.
        out = analyzer._run_git(["diff", "HEAD~1..HEAD"])
        # Output contains the replacement char where bytes were invalid.
        assert isinstance(out, str)
        # Bytes 0x92 / 0xfd are not valid utf-8 continuation of a start
        # byte; they get replaced. The output must still contain recognizable
        # diff markers.
        assert "diff --git" in out
        # Do not crash on non-ascii either
        analyzer.get_changes(commit_range="HEAD~1..HEAD")


class TestGitArgValidators:
    """Negative cases for the validators guarding git's argv."""

    @pytest.mark.parametrize(
        "ref",
        [
            "HEAD",
            "HEAD~3",
            "HEAD^",
            "HEAD^^",
            "HEAD^{tree}",
            "HEAD@{1}",
            "main",
            "feature/login",
            "v1.0",
            "abc1234",
            "main@{upstream}",
            "main@{yesterday}",
            "@",
            "@{-1}",
            "main~1",
            "abc1234^",
            "v1.0^{commit}",
        ],
    )
    def test_ref_accepts(self, ref):
        assert _validate_git_ref(ref) == ref

    @pytest.mark.parametrize(
        "ref",
        [
            "",
            "-x",
            "--all",
            "--upload-pack=touch /tmp/x",
            "main--force",
            "main --all",
            "main;id",
            "main|id",
            "$(id)",
            "`id`",
            "main\n",
            "main\n--all",
            "main~x",
            "main^{}",
            "@{-x}",
            ".hidden",
            "/abs",
            ":path",
            "HEAD^{}",
            "main@{-x}",
            "ma*n",
        ],
    )
    def test_ref_rejects(self, ref):
        with pytest.raises(GitError):
            _validate_git_ref(ref)

    @pytest.mark.parametrize(
        "rng", ["HEAD", "HEAD~5..HEAD", "main~3..main", "main...feature/x"]
    )
    def test_range_accepts(self, rng):
        assert _validate_git_range(rng) == rng

    @pytest.mark.parametrize(
        "rng",
        [
            "",
            "..",
            "...",
            "main..",
            "..main",
            "a..b..c",
            "a....b",
            "a..-b",
            "--all",
            "a..b --all",
            "a..b\n",
            "a..$(id)",
        ],
    )
    def test_range_rejects(self, rng):
        with pytest.raises(GitError):
            _validate_git_range(rng)

    @pytest.mark.parametrize("name", ["foo", "_private", "Foo9"])
    def test_symbol_accepts(self, name):
        assert _validate_git_symbol(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "9lives",
            "foo bar",
            "foo:bar",
            "foo.bar",
            "foo-bar",
            "-L1,2",
            "foo\n",
            "foo/",
            "f(o)",
            "^foo",
        ],
    )
    def test_symbol_rejects(self, name):
        with pytest.raises(GitError):
            _validate_git_symbol(name)

    @pytest.mark.parametrize("path", ["src/a.py", "a b.py", "a..b.py", "dir.d/x"])
    def test_file_accepts(self, path):
        assert _validate_git_file(path) == path

    @pytest.mark.parametrize(
        "path",
        [
            "",
            "-x",
            "--output=x",
            ":(top)a.py",
            ":!a.py",
            "/etc/passwd",
            "..",
            "../a.py",
            "src/../../a.py",
            "a;id",
            "a|id",
            "a&id",
            "a`id`",
            "a$HOME",
            "a\nb",
            "a\rb",
            "a\tb",
            "a\x00b",
        ],
    )
    def test_file_rejects(self, path):
        with pytest.raises(GitError):
            _validate_git_file(path)

    @pytest.mark.parametrize("spec", ["backend/", "*.rs", "src/**/a.py", "a..b"])
    def test_pathspec_accepts(self, spec):
        assert _validate_pathspec(spec) == spec

    @pytest.mark.parametrize(
        "spec",
        [
            "",
            ":(exclude)a",
            ":!a",
            ":/",
            "-x",
            "--all",
            "..",
            "../a",
            "a/../b",
            "backend/\n",
            "a\x00b",
        ],
    )
    def test_pathspec_rejects(self, spec):
        with pytest.raises(GitError):
            _validate_pathspec(spec)

    @pytest.mark.parametrize(
        "time_range",
        [
            "since --all",
            "since -x",
            "since a;id",
            "yesterday; id",
            "$(id)",
            "2024-01-01\nx",
            "x" * 65,
        ],
    )
    def test_time_range_rejects(self, time_range):
        with pytest.raises(GitError):
            GitHistoryAnalyzer(".")._parse_time_range(time_range)

    def test_direct_since_value_is_one_argv_token(self):
        analyzer = GitHistoryAnalyzer(".")
        assert analyzer._parse_time_range("2024-01-01") == ["--since=2024-01-01"]
        # A flag-shaped value stays the value of --since, never its own flag.
        assert analyzer._parse_time_range("--all") == ["--since=--all"]

    @pytest.mark.parametrize("bad", ["--all", "a..b..c", "HEAD;id", "-x..HEAD"])
    def test_changes_rejects_bad_range_before_running_git(self, bad, monkeypatch):
        analyzer = GitHistoryAnalyzer(".")
        analyzer._verified = True
        monkeypatch.setattr(
            analyzer, "_run_git", lambda *a, **k: pytest.fail("git was run")
        )
        assert "Invalid commit range" in analyzer.get_changes(commit_range=bad)
        assert (
            "Invalid commit range"
            in (analyzer.get_changes_structured(commit_range=bad)["error"])
        )

    @pytest.mark.parametrize("bad", [":(top)x", "-x", "../x"])
    def test_churn_and_changes_reject_bad_path_filter(self, bad, monkeypatch):
        analyzer = GitHistoryAnalyzer(".")
        analyzer._verified = True
        monkeypatch.setattr(
            analyzer, "_run_git", lambda *a, **k: pytest.fail("git was run")
        )
        with pytest.raises(GitError):
            analyzer.get_churn(path_filter=bad)
        with pytest.raises(GitError):
            analyzer.get_churn_structured(path_filter=bad)
        assert "Invalid path filter" in analyzer.get_changes(
            commit_range="HEAD~1..HEAD", path_filter=bad
        )


def _a_py(alpha, beta=1):
    return f"def alpha():\n    return {alpha}\n\n\ndef beta():\n    return {beta}\n"


A_PY_V1 = _a_py(1)
A_PY_NODES = [
    _function_node("a.py", "alpha", 1, 2),
    _function_node("a.py", "beta", 5, 6),
]


@pytest.mark.skipif(sys.platform == "win32", reason="plants POSIX shell scripts")
@pytest.mark.usefixtures("isolated_git")
class TestHostileRepoConfig:
    """A repository's own config and hooks must not get commands run.

    Each test plants a command somewhere git would run it during a read-only
    operation, confirms with plain git that the plant is live, then drives
    every analysis entry point and checks the command never ran.
    """

    @pytest.fixture
    def marker(self, tmp_path):
        return tmp_path / "ran"

    @pytest.fixture
    def payload(self, tmp_path, marker):
        """A script that records that it was executed."""
        script = tmp_path / "payload.sh"
        script.write_text(f'#!/bin/sh\necho ran >> "{marker}"\n', encoding="utf-8")
        script.chmod(0o755)
        return script

    @pytest.fixture
    def repo(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"a.py": A_PY_V1})
        _commit(repo, "two", **{"a.py": _a_py(2)})
        _commit(repo, "three", **{"a.py": _a_py(3)})
        return repo

    def _assert_live(self, repo, marker, *git_args):
        """Plain git must trip the plant, or the test proves nothing."""
        _git(repo, *git_args, check=False)
        assert marker.exists(), f"plant not triggered by: git {' '.join(git_args)}"
        marker.unlink()

    def _set_attributes(self, repo, line):
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "attributes").write_text(line)

    def _exercise(self, repo):
        """Run every code path that shells out to git."""
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))
        assert "alpha" in analyzer.get_churn(mode="symbols")
        assert analyzer.get_churn_structured(mode="co-change")["mode"] == "co-change"
        assert "a.py" in analyzer.get_churn(mode="files")
        assert "3 commit(s)" in analyzer.get_evolution("alpha", show_diff=True)
        assert "alpha" in analyzer.get_changes(commit_range="HEAD~1..HEAD")
        assert analyzer.get_changes_structured(commit_range="HEAD~2..HEAD")["symbols"]
        # A single ref diffs against the working tree.
        analyzer.get_changes(commit_range="HEAD")
        analyzer.get_changes_structured(commit_range="HEAD")
        analyzer.get_changes(time_range="last 30 days")
        analyzer._is_shallow()

    def test_diff_external(self, repo, payload, marker):
        _git(repo, "config", "diff.external", str(payload))
        self._assert_live(repo, marker, "diff", "HEAD~1..HEAD")

        self._exercise(repo)
        assert not marker.exists()

    def test_diff_external_from_included_config(self, repo, payload, marker):
        (repo / ".git" / "extra.cfg").write_text(
            f"[diff]\n\texternal = {payload}\n", encoding="utf-8"
        )
        _git(repo, "config", "include.path", "extra.cfg")
        self._assert_live(repo, marker, "diff", "HEAD~1..HEAD")

        self._exercise(repo)
        assert not marker.exists()

    @pytest.mark.parametrize("key", ["command", "textconv"])
    def test_attribute_selected_diff_driver(self, repo, payload, marker, key):
        self._set_attributes(repo, "*.py diff=planted\n")
        _git(repo, "config", f"diff.planted.{key}", str(payload))
        self._assert_live(repo, marker, "diff", "HEAD~1..HEAD")
        if key == "textconv":
            self._assert_live(repo, marker, "log", "-L:alpha:a.py")

        self._exercise(repo)
        assert not marker.exists()

    def test_show_signature_runs_no_gpg_program(self, repo, payload, marker, tmp_path):
        signer = tmp_path / "signer.sh"
        signer.write_text(
            "#!/bin/sh\ncat >/dev/null\n"
            'echo "[GNUPG:] SIG_CREATED D 1 8 00 0 FAKE" >&2\n'
            "printf -- '-----BEGIN PGP SIGNATURE-----\\n\\nZmFrZQ==\\n"
            "-----END PGP SIGNATURE-----\\n'\n",
            encoding="utf-8",
        )
        signer.chmod(0o755)
        (repo / "a.py").write_text(_a_py(4), encoding="utf-8")
        _git(
            repo,
            "-c",
            "gpg.format=openpgp",
            "-c",
            f"gpg.program={signer}",
            "-c",
            "user.signingkey=x",
            "commit",
            "-q",
            "-S",
            "-am",
            "signed",
        )
        _git(repo, "config", "log.showSignature", "true")
        _git(repo, "config", "gpg.program", str(payload))
        self._assert_live(repo, marker, "log", "--format=%H")

        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))
        assert "a.py" in analyzer.get_churn(mode="files")
        assert "alpha" in analyzer.get_churn(mode="symbols")
        assert "4 commit(s)" in analyzer.get_evolution("alpha")
        analyzer.get_changes(time_range="last 30 days")
        assert not marker.exists()

    def test_post_index_change_hook(self, repo, payload, marker):
        hook = repo / ".git" / "hooks" / "post-index-change"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text(payload.read_text(encoding="utf-8"), encoding="utf-8")
        hook.chmod(0o755)

        def dirty_stat():
            # Same content, new mtime: the next worktree diff refreshes
            # the index and would write it back.
            stamp = (repo / "a.py").stat().st_mtime + 60
            os.utime(repo / "a.py", (stamp, stamp))

        dirty_stat()
        self._assert_live(repo, marker, "diff", "HEAD")

        dirty_stat()
        self._exercise(repo)
        assert not marker.exists()

    def test_fsmonitor_hook(self, repo, payload, marker):
        _git(repo, "config", "core.fsmonitor", str(payload))
        self._assert_live(repo, marker, "diff", "HEAD")

        self._exercise(repo)
        assert not marker.exists()

    @pytest.mark.parametrize("key", ["clean", "process"])
    def test_worktree_filter_driver(self, repo, payload, marker, key):
        self._set_attributes(repo, "*.py filter=planted\n")
        _git(repo, "config", f"filter.planted.{key}", str(payload))
        _git(repo, "config", "filter.planted.required", "true")
        (repo / "a.py").write_text(_a_py(3, beta=9), encoding="utf-8")
        self._assert_live(repo, marker, "diff", "HEAD")

        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))
        # The uncommitted edit is reported from the unfiltered file.
        assert "beta" in analyzer.get_changes(commit_range="HEAD")
        symbols = analyzer.get_changes_structured(commit_range="HEAD")["symbols"]
        assert [s["name"] for s in symbols] == ["beta"]
        assert not marker.exists()

    def test_filter_driver_that_cannot_be_disabled_is_refused(
        self, repo, payload, marker
    ):
        self._set_attributes(repo, "*.py filter=a=b\n")
        with open(repo / ".git" / "config", "a", encoding="utf-8") as config:
            config.write(f'[filter "a=b"]\n\tclean = {payload}\n')
        (repo / "a.py").write_text(_a_py(3, beta=9), encoding="utf-8")
        self._assert_live(repo, marker, "diff", "HEAD")

        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))
        assert "Refusing to diff the working tree" in analyzer.get_changes(
            commit_range="HEAD"
        )
        assert (
            "Refusing to diff the working tree"
            in (analyzer.get_changes_structured(commit_range="HEAD")["error"])
        )
        # A commit-to-commit range never reads the working tree.
        assert "alpha" in analyzer.get_changes(commit_range="HEAD~1..HEAD")
        assert not marker.exists()

    def test_partial_clone_runs_no_transport_and_says_what_is_missing(
        self, repo, payload, marker, tmp_path
    ):
        version = _git(repo, "version").stdout.split()[2].split(".")
        if (int(version[0]), int(version[1])) < (2, 44):
            pytest.skip("GIT_NO_LAZY_FETCH needs git 2.44+")
        bare = tmp_path / "origin.git"
        _git(tmp_path, "clone", "-q", "--bare", str(repo), str(bare))
        _git(bare, "config", "uploadpack.allowFilter", "true")
        _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
        partial = tmp_path / "partial"
        _git(
            tmp_path,
            "clone",
            "-q",
            "--no-checkout",
            "--filter=blob:none",
            f"file://{bare}",
            str(partial),
        )
        # Missing blobs are fetched on demand through the command this names.
        _git(partial, "config", "remote.origin.uploadpack", str(payload))
        self._assert_live(partial, marker, "diff", "HEAD~1..HEAD")

        analyzer = GitHistoryAnalyzer(
            str(partial), graph_querier=FakeQuerier(A_PY_NODES)
        )
        calls = [
            lambda: analyzer.get_churn(mode="symbols"),
            lambda: analyzer.get_churn_structured(mode="symbols"),
            lambda: analyzer.get_evolution("alpha"),
            lambda: analyzer.get_changes(commit_range="HEAD~1..HEAD"),
            lambda: analyzer.get_changes_structured(commit_range="HEAD~1..HEAD"),
        ]
        for call in calls:
            with pytest.raises(PartialCloneError, match="partial clone"):
                call()
        assert not marker.exists()
        # History that needs no blobs is still served.
        assert "a.py (3 commits)" in analyzer.get_churn(mode="files")


@pytest.mark.usefixtures("isolated_git")
class TestLogFraming:
    """Repository content must not be able to pose as a commit header."""

    FORGED_HASH = "deadbeef" * 5
    FORGED_HEADER = f"{FORGED_HASH} 2031-01-01 Mallory"

    def _evolution_repo(self, tmp_path, subject):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "add alpha", **{"a.py": A_PY_V1})
        _commit(repo, subject, **{"a.py": _a_py(2)})
        return repo

    def test_commit_subject_cannot_forge_a_commit(self, tmp_path):
        repo = self._evolution_repo(tmp_path, self.FORGED_HEADER)
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))

        result = analyzer.get_evolution("alpha")

        assert "2 commit(s) | 1 author(s)" in result
        assert f"**{self.FORGED_HASH[:8]}**" not in result
        assert ") Mallory" not in result
        # The crafted text is shown for what it is: a subject line.
        assert f"   {self.FORGED_HEADER}" in result

    def test_commit_subject_cannot_forge_a_commit_in_file_level_fallback(
        self, tmp_path
    ):
        repo = self._evolution_repo(tmp_path, self.FORGED_HEADER)
        # No such function in the file and no line span: `git log -L` cannot
        # be used, so the plain file log is parsed instead.
        node = {
            "id": "FILE:a.py::gamma",
            "type": "Function",
            "metadata": {"name": "gamma"},
        }
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier([node]))

        result = analyzer.get_evolution("gamma")

        assert "file-level history" in result
        assert "2 commit(s) | 1 author(s)" in result
        assert f"**{self.FORGED_HASH[:8]}**" not in result
        assert f"   {self.FORGED_HEADER}" in result

    def test_diffed_content_cannot_forge_a_commit(self, tmp_path):
        # Bare CRs followed by ready-made headers, space- and NUL-separated,
        # in a file forced to diff as text despite the NULs.
        forged = (
            f"\r{self.FORGED_HEADER}"
            f"\r\x00{self.FORGED_HASH}\x002031-01-01\x00Mallory\x00pwned"
        )
        repo = _init_repo(tmp_path / "repo")
        _commit(
            repo,
            "add alpha",
            **{".gitattributes": "*.py diff\n", "a.py": A_PY_V1},
        )
        body = f"def alpha():\n    x = 1{forged}\n    return 2\n".encode()
        _commit(repo, "edit alpha", **{"a.py": body})
        node = _function_node("a.py", "alpha", 1, 3)
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier([node]))

        result = analyzer.get_evolution("alpha", show_diff=True)

        assert "2 commit(s) | 1 author(s)" in result
        assert f"**{self.FORGED_HASH[:8]}**" not in result

    def test_hex_named_file_is_a_file_not_a_commit(self, tmp_path):
        hex_name = self.FORGED_HASH
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{hex_name: "x\n", "zeta.py": "y = 1\n"})
        _commit(repo, "two", **{"zeta.py": "y = 2\n"})
        analyzer = GitHistoryAnalyzer(str(repo))

        files = analyzer.get_churn_structured(mode="files")["files"]
        assert {f["file"]: f["commits"] for f in files} == {hex_name: 1, "zeta.py": 2}
        assert f"{hex_name} (1 commits)" in analyzer.get_churn(mode="files")

    def test_added_line_cannot_pose_as_a_file_header(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"a.py": "one\ntwo\n", "victim.py": "x\n"})
        _commit(repo, "two", **{"a.py": "one\n++ b/victim.py\ntwo\nthree\n"})
        analyzer = GitHistoryAnalyzer(str(repo))

        hunks = analyzer._parse_diff_hunks(analyzer._run_git(["diff", "HEAD~1..HEAD"]))

        assert [h["file"] for h in hunks] == ["a.py"]
        assert hunks[0]["lines"] == [" one", "+++ b/victim.py", " two", "+three"]

    def test_cr_in_git_output_is_not_a_line_break(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"crlf.txt": b"a\r\nb\r\n"})
        _commit(repo, "two", **{"crlf.txt": b"a\r\nb\rc\r\n"})
        analyzer = GitHistoryAnalyzer(str(repo))

        diff = analyzer._run_git(["diff", "HEAD~1..HEAD"])

        assert "+b\rc\r\n" in diff
        hunks = analyzer._parse_diff_hunks(diff)
        assert [line[0] for line in hunks[0]["lines"]] == [" ", "-", "+"]


@pytest.mark.usefixtures("isolated_git")
class TestChangesByTimeRange:
    """A time range is turned into a commit range that git can diff."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"a.py": A_PY_V1})
        _commit(repo, "two", **{"a.py": _a_py(2)})
        _commit(repo, "three", **{"b.py": "x = 1\n"})
        return GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))

    def test_range_reaching_the_root_commit(self, analyzer):
        result = analyzer.get_changes(time_range="last 30 days")

        assert "### Change Impact: last 30 days (3 commits)" in result
        assert "Scope: 2 file(s) | 2 symbol(s) affected" in result
        structured = analyzer.get_changes_structured(time_range="last 30 days")
        assert structured["commits"] == 3
        assert structured["files_count"] == 2
        assert {s["name"] for s in structured["symbols"]} == {"alpha", "beta"}

    def test_range_starting_after_the_root_commit(self, analyzer):
        ranges = analyzer._ranges_covering(
            _git(analyzer.project_root, "log", "--format=%H", "-2").stdout.split()
        )

        diff_range, count_range = ranges
        assert diff_range == count_range
        assert analyzer._run_git(["rev-list", "--count", count_range]).strip() == "2"


@pytest.mark.usefixtures("isolated_git")
class TestRootCommit:
    """A commit without a parent is diffed against the empty tree."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"a.py": A_PY_V1})
        return GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))

    def test_churn_attributes_the_root_commit(self, analyzer):
        result = analyzer.get_churn(mode="symbols")

        assert "alpha | a.py:1 (1 commits, +2/-0)" in result
        assert "beta | a.py:5 (1 commits, +2/-0)" in result
        symbols = analyzer.get_churn_structured(mode="symbols")["symbols"]
        assert {(s["name"], s["commits"], s["added"]) for s in symbols} == {
            ("alpha", 1, 2),
            ("beta", 1, 2),
        }

    def test_default_changes_range_in_a_one_commit_repository(self, analyzer):
        result = analyzer.get_changes()

        assert "(1 commit)" in result
        assert "Scope: 1 file(s) | 2 symbol(s) affected" in result
        structured = analyzer.get_changes_structured()
        assert structured["commits"] == 1
        assert {s["name"] for s in structured["symbols"]} == {"alpha", "beta"}

    def test_default_changes_range_is_the_last_commit(self, analyzer):
        _commit(analyzer.project_root, "two", **{"a.py": _a_py(2)})

        structured = analyzer.get_changes_structured()

        assert structured["commits"] == 1
        assert [s["name"] for s in structured["symbols"]] == ["alpha"]


@pytest.mark.usefixtures("isolated_git")
class TestOutputShapingConfig:
    @pytest.mark.parametrize(
        "setting",
        [
            "diff.mnemonicPrefix=true",
            "diff.noprefix=true",
            "diff.srcPrefix=old/",
            "color.ui=always",
            "diff.suppressBlankEmpty=true",
        ],
    )
    def test_diffs_are_attributed_whatever_the_repository_asks_git_to_print(
        self, tmp_path, setting
    ):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"a.py": _a_py(1)})
        _commit(repo, "two", **{"a.py": _a_py(2)})
        (repo / "a.py").write_text(_a_py(3), encoding="utf-8")
        _git(repo, "config", *setting.split("="))
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))

        committed = analyzer.get_changes_structured(commit_range="HEAD~1..HEAD")
        worktree = analyzer.get_changes_structured(commit_range="HEAD")

        assert [s["name"] for s in committed["symbols"]] == ["alpha"]
        assert [s["name"] for s in worktree["symbols"]] == ["alpha"]


@pytest.mark.usefixtures("isolated_git")
class TestNonAsciiPaths:
    """Paths outside ASCII are reported and attributed under their own name."""

    def test_churn_and_changes(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        _commit(repo, "one", **{"café.py": A_PY_V1})
        _commit(repo, "two", **{"café.py": _a_py(2)})
        nodes = [_function_node("café.py", "alpha", 1, 2)]
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(nodes))

        assert "café.py (2 commits)" in analyzer.get_churn(mode="files")
        assert "alpha | café.py:1 (2 commits" in analyzer.get_churn(mode="symbols")
        changes = analyzer.get_changes_structured(commit_range="HEAD~1..HEAD")
        assert [s["name"] for s in changes["symbols"]] == ["alpha"]


@pytest.mark.usefixtures("isolated_git")
class TestSha256Repository:
    """Commit ids are 64 hex digits in a SHA-256 repository."""

    @pytest.fixture
    def repo(self, tmp_path):
        try:
            repo = _init_repo(tmp_path / "repo", "--object-format=sha256")
        except subprocess.CalledProcessError:
            pytest.skip("this git cannot create SHA-256 repositories")
        _commit(repo, "one", **{"a.py": A_PY_V1})
        _commit(repo, "two", **{"a.py": _a_py(2)})
        _commit(repo, "three", **{"a.py": _a_py(3)})
        assert len(_git(repo, "rev-parse", "HEAD").stdout.strip()) == 64
        return repo

    def test_churn(self, repo):
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))

        assert "a.py (3 commits)" in analyzer.get_churn(mode="files")
        files = analyzer.get_churn_structured(mode="files")["files"]
        assert files == [{"file": "a.py", "commits": 3}]
        symbols = analyzer.get_churn_structured(mode="symbols")["symbols"]
        assert [(s["name"], s["commits"]) for s in symbols] == [
            ("alpha", 3),
            ("beta", 1),
        ]

    def test_evolution(self, repo):
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()

        result = analyzer.get_evolution("alpha")

        assert "3 commit(s) | 1 author(s)" in result
        assert f"1. **{head[:8]}**" in result


@pytest.mark.usefixtures("isolated_git")
class TestShallowDetection:
    """Shallow clones are detected wherever the git directory lives."""

    @pytest.fixture
    def origin(self, tmp_path):
        repo = _init_repo(tmp_path / "origin")
        _commit(repo, "one", **{"a.py": A_PY_V1})
        _commit(repo, "two", **{"a.py": _a_py(2)})
        return repo

    @pytest.fixture
    def shallow(self, origin, tmp_path):
        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{origin}", str(clone))
        return clone

    def test_full_repository_is_not_shallow(self, origin):
        assert GitHistoryAnalyzer(str(origin))._is_shallow() is False

    def test_linked_worktree_of_shallow_clone(self, shallow, tmp_path):
        worktree = tmp_path / "linked"
        _git(shallow, "worktree", "add", "-q", "--detach", str(worktree))
        assert (worktree / ".git").is_file()

        assert GitHistoryAnalyzer(str(worktree))._is_shallow() is True

    def test_separate_git_dir_shallow_clone(self, origin, tmp_path):
        clone = tmp_path / "split"
        _git(
            tmp_path,
            "clone",
            "-q",
            "--depth",
            "1",
            "--separate-git-dir",
            str(tmp_path / "split.git"),
            f"file://{origin}",
            str(clone),
        )
        assert (clone / ".git").is_file()

        assert GitHistoryAnalyzer(str(clone))._is_shallow() is True


@pytest.mark.usefixtures("isolated_git")
class TestChurnLimits:
    """Symbol-mode churn bounds its per-commit diffs and says when it did."""

    COMMITS = 6

    @pytest.fixture
    def analyzer(self, tmp_path):
        repo = _init_repo(tmp_path / "repo")
        for value in range(1, self.COMMITS + 1):
            _commit(repo, f"alpha = {value}", **{"a.py": _a_py(value)})
        return GitHistoryAnalyzer(str(repo), graph_querier=FakeQuerier(A_PY_NODES))

    def test_unknown_mode_is_an_error_in_both_forms(self, analyzer):
        assert "Unknown mode 'hot'" in analyzer.get_churn(mode="hot")
        assert (
            "Unknown mode 'hot'" in analyzer.get_churn_structured(mode="hot")["error"]
        )

    @pytest.fixture
    def diffed(self, analyzer, monkeypatch):
        """Record the revision argument of every `git diff` the analyzer runs."""
        calls = []
        run_git = analyzer._run_git

        def recording(args, *a, **kw):
            if args[0] == "diff":
                calls.append(args[1])
            return run_git(args, *a, **kw)

        monkeypatch.setattr(analyzer, "_run_git", recording)
        return calls

    def test_history_within_limits_is_not_marked_partial(self, analyzer, diffed):
        result = analyzer.get_churn(mode="symbols")

        assert len(diffed) == self.COMMITS
        assert f"alpha | a.py:1 ({self.COMMITS} commits" in result
        assert "Partial results" not in result
        structured = analyzer.get_churn_structured(mode="symbols")
        assert "truncation_note" not in structured

    def test_commit_cap_reads_only_the_most_recent_commits(
        self, analyzer, diffed, monkeypatch
    ):
        monkeypatch.setattr(git_history, "CHURN_MAX_COMMITS_PER_FILE", 2)
        newest = _git(analyzer.project_root, "log", "--format=%H", "-2").stdout.split()

        result = analyzer.get_churn(mode="symbols")

        assert diffed == [f"{sha}~1..{sha}" for sha in newest]
        assert "alpha | a.py:1 (2 commits" in result
        assert "only the 2 most recent commits were read for 1 file(s)" in result
        structured = analyzer.get_churn_structured(mode="symbols")
        assert structured["symbols"][0]["commits"] == 2
        assert "2 most recent commits" in structured["truncation_note"]

    def test_commit_cap_is_reported_in_co_change_mode(self, analyzer, monkeypatch):
        monkeypatch.setattr(git_history, "CHURN_MAX_COMMITS_PER_FILE", 2)

        assert "Partial results" in analyzer.get_churn(mode="co-change")
        structured = analyzer.get_churn_structured(mode="co-change")
        assert "Partial results" in structured["truncation_note"]

    def test_spent_time_budget_stops_diffing(self, analyzer, diffed, monkeypatch):
        monkeypatch.setattr(git_history, "CHURN_DIFF_BUDGET_SECONDS", 0)

        result = analyzer.get_churn(mode="symbols")

        assert diffed == []
        assert f"{self.COMMITS} commit(s) across 1 file(s) unread" in result
        structured = analyzer.get_churn_structured(mode="symbols")
        assert "time budget ran out" in structured["error"]

    def test_time_budget_running_out_midway_keeps_what_was_read(
        self, analyzer, diffed, monkeypatch
    ):
        # Each look at the clock advances it by one second.
        ticks = iter(range(1000))
        monkeypatch.setattr(git_history.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(git_history, "CHURN_DIFF_BUDGET_SECONDS", 3)

        result = analyzer.get_churn_structured(mode="symbols")

        assert len(diffed) == 2
        assert result["symbols"][0]["commits"] == 2
        assert (
            f"{self.COMMITS - 2} commit(s) across 1 file(s) unread"
            in (result["truncation_note"])
        )

    def test_files_mode_is_never_partial(self, analyzer, diffed, monkeypatch):
        monkeypatch.setattr(git_history, "CHURN_MAX_COMMITS_PER_FILE", 1)
        monkeypatch.setattr(git_history, "CHURN_DIFF_BUDGET_SECONDS", 0)

        result = analyzer.get_churn(mode="files")

        assert f"a.py ({self.COMMITS} commits)" in result
        assert "Partial results" not in result
        assert diffed == []


@pytest.mark.usefixtures("isolated_git")
class TestChangeImpactCallers:
    def test_a_changed_symbol_lists_its_own_callers(self, tmp_path):
        """Each changed symbol lists the callers of that symbol, not those
        of every symbol of its name."""
        from descry.generate import CodeGraphBuilder
        from descry.query import GraphQuerier

        def parse(value):
            return f"    def parse(self):\n        return {value}\n"

        repo = _init_repo(tmp_path / "repo")
        _commit(
            repo,
            "one",
            **{
                "a.py": "class A:\n" + parse(1),
                "b.py": "class B:\n" + parse(1),
                "use.py": "from a import A\nfrom b import B\n\n\n"
                "def only_a():\n    A().parse()\n\n\n"
                "def only_b():\n    B().parse()\n",
            },
        )
        _commit(
            repo,
            "two",
            **{"a.py": "class A:\n" + parse(2), "b.py": "class B:\n" + parse(2)},
        )
        builder = CodeGraphBuilder(repo)
        builder.process_directory()
        graph = tmp_path / "graph.json"
        builder.export(str(graph))
        analyzer = GitHistoryAnalyzer(str(repo), graph_querier=GraphQuerier(str(graph)))

        result = analyzer.get_changes_structured(commit_range="HEAD~1..HEAD")
        callers = {
            s["id"]: [c["name"] for c in s.get("callers", [])]
            for s in result["symbols"]
        }
        assert callers["FILE:a.py::A::parse"] == ["only_a"]
        assert callers["FILE:b.py::B::parse"] == ["only_b"]
        text = analyzer.get_changes(commit_range="HEAD~1..HEAD")
        assert text.count("Callers: only_a") == text.count("Callers: only_b") == 1


def test_tied_co_change_pairs_are_listed_by_name(tmp_path):
    """The same history lists the same pairs, whatever order sets iterate in."""
    analyzer = GitHistoryAnalyzer(str(tmp_path))
    symbol_commits = {
        "FILE:a.py::d": {"x1", "x2"},
        "FILE:a.py::c": {"x1", "x2"},
        "FILE:a.py::b": {"y1", "y2"},
        "FILE:a.py::a": {"y1", "y2"},
    }
    pairs = analyzer._structured_co_change(symbol_commits, 1, {}, {})["pairs"]
    assert [(p["a_name"], p["b_name"]) for p in pairs] == [("a", "b")]
    assert "1. a (a.py) <-> b (a.py)" in analyzer._format_co_change(symbol_commits, 1)
