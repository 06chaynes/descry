"""Tests for descry.git_history — Git history analysis and churn detection."""

from pathlib import Path

import pytest

from descry.git_history import GitHistoryAnalyzer


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


class TestGitHistoryBuildFileLineMap:
    """Tests for GitHistoryAnalyzer._build_file_line_map with mock graph data."""

    def test_maps_lines_to_symbols(self):
        # Create a mock querier with graph data
        graph_data = {
            "schema_version": 1,
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

    def test_run_git_tolerates_non_utf8_bytes(self, tmp_path, monkeypatch):
        """_run_git must not raise UnicodeDecodeError on binary-ish output."""
        import subprocess as sp
        import subprocess as _sp

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
        _ = analyzer.get_changes(commit_range="HEAD~1..HEAD")
        # silence unused-import lint
        assert _sp is sp
