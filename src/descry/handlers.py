"""DescryService — Core business logic for codebase knowledge graph tools.

No MCP dependency. All tool logic lives here; MCP/CLI/Web are thin wrappers.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from descry import __version__
from descry._answers import fail
from descry._env import env_flag
from descry._files import open_regular
from descry._graph import GraphSchemaError, load_graph_with_schema
from descry.query import (
    _DEFAULT_TEST_FILE_SUFFIXES,
    _DEFAULT_TEST_PATH_PATTERNS,
    HTTP_METHODS,
    LANG_ALIASES,
    GraphQuerier,
    _get_syntax_lang,
    crate_of,
    file_of,
)

logger = logging.getLogger(__name__)

# --- Configuration ---


_DEFAULT_PROJECT_MARKERS = [
    # Generic / explicit
    ".git",
    ".descry.toml",
    # Rust
    "Cargo.toml",
    # JavaScript / TypeScript / Node
    "package.json",
    # Python
    "pyproject.toml",
    "setup.py",
    # Go
    "go.mod",
    # Ruby
    "Gemfile",
    # PHP
    "composer.json",
    # Java / Kotlin / Scala (Gradle + Maven + sbt)
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "pom.xml",
    "build.sbt",
    # .NET
    "global.json",
    # Dart / Flutter
    "pubspec.yaml",
    # C / C++ (CMake + compdb)
    "CMakeLists.txt",
    "compile_commands.json",
]
_DEFAULT_API_PREFIXES = ["/api/v1", "/api/v2", "/api"]
_DEFAULT_EXCLUDED_DIRS = {
    # Generic
    ".git",
    ".hg",
    ".svn",
    "build",
    "dist",
    "docs",
    # Rust
    "target",
    # JS/TS/Node
    "node_modules",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".output",
    # Python
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    # Go
    "vendor",
    # .NET (its `bin`, and an Eclipse project's, is skipped beside the
    # project file)
    "obj",
    # Dart / Flutter
    ".dart_tool",
    # Java / Kotlin / Scala
    ".gradle",
    ".idea",
}

_DEFAULT_CODE_EXTENSIONS = {
    ".rs",
    ".py",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".svelte",
    ".proto",
    ".go",
    ".java",
    ".kt",
    ".scala",
    ".rb",
    ".rake",
    ".gemspec",
    ".php",
    ".cs",
    ".vb",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".dart",
    ".css",
    ".scss",
    ".html",
}
_DEFAULT_CHURN_EXCLUSIONS = [
    ".descry_cache/",
    ".beads/",
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
]
_DEFAULT_SYNTAX_LANG_MAP = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".svelte": "svelte",
    ".proto": "protobuf",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".vb": "vbnet",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cu": "cuda",
    ".h": "c",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".dart": "dart",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
}


_TOOLCHAIN_REGEX = re.compile(r"^[A-Za-z0-9._\-]+$")
_MODEL_HF_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]*$")
_MODEL_PATH_REGEX = re.compile(r"^[A-Za-z0-9._\-/~]+$")
_SHELL_METACHARS = set("\x00;|&`$\n\r\t<>")
_LANG_NAME_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+#.\-]*$")


def _toml_number(section, key, value, default, integer=False, positive=False):
    """A finite number of at least 0 and at most a million from TOML, or
    `default` with a warning if it is anything else; ``integer`` a whole
    one, ``positive`` one above 0."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value < 1e6
        or integer
        and isinstance(value, float)
        or positive
        and value <= 0
    ):
        want = " ".join(
            filter(
                None,
                (
                    "a finite",
                    "whole" if integer else "",
                    "number",
                    "above 0" if positive else "of at least 0",
                    "and below 1000000",
                ),
            )
        )
        logger.warning(
            f"Invalid [{section}] {key} in .descry.toml: {value!r} is not "
            f"{want}; keeping {default!r}"
        )
        return default
    return value


# TOML (section, key) -> the DescryConfig attribute it sets, for numbers.
_TOML_NUMBERS = {
    ("project", "max_stale_hours"): "max_stale_hours",
    ("code_files", "max_file_mb"): "max_file_mb",
    ("code_files", "max_parse_seconds"): "max_parse_seconds",
    ("git", "timeout"): "git_timeout",
    ("timeouts", "scip_minutes"): "scip_timeout_minutes",
    ("timeouts", "index_minutes"): "index_timeout_minutes",
    ("timeouts", "embedding_seconds"): "embedding_timeout",
    ("timeouts", "query_ms"): "query_timeout_ms",
    ("query", "max_depth"): "max_depth",
    ("query", "max_nodes"): "max_nodes",
    ("query", "max_children_per_level"): "max_children_per_level",
    ("query", "max_callers_shown"): "max_callers_shown",
}

# The attrs _TOML_NUMBERS sets that a fractional value breaks: they bound
# loops and ranges. Timeouts may be fractional.
_TOML_INT_KEYS = frozenset(
    {"max_depth", "max_nodes", "max_children_per_level", "max_callers_shown"}
)


def _validate_toolchain(value: str) -> str:
    """Validate scip_rust_toolchain TOML value. Raises ValueError on bad input."""
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("-")
        or not _TOOLCHAIN_REGEX.match(value)
    ):
        raise ValueError(f"Invalid scip.rust.toolchain: {value!r}")
    return value


def _from_root(value: str, project_root: Path) -> Path:
    """The place ``value``, read as a path from ``project_root`` in this
    platform's spelling, names."""
    return (project_root.resolve() / os.path.expanduser(value)).resolve()


def _outside_root(value: str, project_root: Path) -> bool:
    """Whether ``value``, read as a path from ``project_root``, names a
    place outside it."""
    root, resolved = project_root.resolve(), _from_root(value, project_root)
    return not (resolved == root or root in resolved.parents)


def _validate_scip_extra_arg(arg: str, project_root: Path) -> str:
    """Validate a scip_extra_args entry.

    Accepts bare positional args; flags must start with '--' (blocks short-flag
    smuggling). Rejects shell metacharacters, and a path outside the project,
    as the argument or as the value of `--flag=value`.
    """
    if not isinstance(arg, str) or not arg:
        raise ValueError(f"Invalid or empty scip extra_args entry: {arg!r}")
    if arg.startswith("-") and not arg.startswith("--"):
        raise ValueError(f"scip_extra_args short flag not allowed: {arg!r}")
    if any(c in _SHELL_METACHARS for c in arg):
        raise ValueError(f"scip_extra_args contains shell metacharacters: {arg!r}")
    if _outside_root(arg.partition("=")[2] or arg, project_root):
        raise ValueError(f"scip_extra_args path outside project root: {arg!r}")
    return arg


def _validate_embedding_model(value: str, project_root: Path) -> str:
    """Validate [embeddings] model TOML value.

    - Must be a string, carry no shell metacharacters, nor resolve, as a
      path from project_root, outside it.
    - Unless it looks like a local path (starts with '/', '~' or './' or
      has a '..' part), must match HuggingFace repo-id regex.
    - A local path is returned resolved from project_root, where it is
      loaded from whatever directory descry runs in.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid or empty embeddings.model: {value!r}")

    from descry.embeddings import MODEL_REGISTRY

    if value in MODEL_REGISTRY:
        return value
    if any(c in _SHELL_METACHARS for c in value):
        raise ValueError(f"embeddings.model contains shell metacharacters: {value!r}")
    if "/" not in value and not value.startswith("."):
        # Usually a mistyped alias, but HuggingFace hosts a few single-segment
        # repo ids, so warn and carry on.
        logger.warning(
            "embeddings.model %r is not a known alias (%s) and has no org prefix; "
            "treating it as a HuggingFace repo id",
            value,
            ", ".join(sorted(MODEL_REGISTRY)),
        )
    if _outside_root(value, project_root):
        raise ValueError(f"embeddings.model local path {value!r} outside project root")
    if value.startswith(("/", "~", "./")) or ".." in value.split("/"):
        # A path is written with path characters; prose is no model.
        if not _MODEL_PATH_REGEX.match(value):
            raise ValueError(f"Invalid embeddings.model path: {value!r}")
        return str(_from_root(value, project_root))
    if not _MODEL_HF_REGEX.match(value):
        raise ValueError(f"Invalid embeddings.model repo id: {value!r}")
    return value


@dataclass
class DescryConfig:
    """Configuration for a Descry project."""

    project_root: Path = field(default_factory=Path.cwd)
    cache_dir: Path | None = None
    max_stale_hours: float = 24
    enable_scip: bool = True
    enable_embeddings: bool = True
    openapi_path: Path | None = None
    backend_handler_patterns: list[str] = field(default_factory=list)
    api_prefixes: list[str] = field(default_factory=lambda: list(_DEFAULT_API_PREFIXES))
    excluded_dirs: set[str] = field(default_factory=lambda: set(_DEFAULT_EXCLUDED_DIRS))

    # Embeddings
    # None: the default model, or its ungated fallback when the default is gated.
    embedding_model: str | None = None

    # Test detection
    test_path_patterns: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_TEST_PATH_PATTERNS
    )
    test_file_suffixes: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_TEST_FILE_SUFFIXES
    )

    # Code files
    code_extensions: set[str] = field(
        default_factory=lambda: set(_DEFAULT_CODE_EXTENSIONS)
    )
    max_file_mb: float = (
        10  # larger files are not indexed, read or served; 0 = unlimited
    )
    # A file whose parsing takes longer is left as far as it got; 0 = unlimited
    max_parse_seconds: float = 120
    # In a git repository, files git ignores are not indexed
    respect_gitignore: bool = True

    # Git
    churn_exclusions: list[str] = field(
        default_factory=lambda: list(_DEFAULT_CHURN_EXCLUSIONS)
    )
    git_timeout: int = 30

    # Timeouts
    scip_timeout_minutes: int = 30  # per indexer; 0 = unlimited
    index_timeout_minutes: int = 30  # Overall index timeout (0 = unlimited)
    embedding_timeout: float | None = 60  # None: wait for the load
    query_timeout_ms: int = 4000

    # Query limits
    max_depth: int = 3
    max_nodes: int = 100
    max_children_per_level: int = 10
    max_callers_shown: int = 15

    # SCIP
    scip_extra_args: list[str] = field(
        default_factory=lambda: ["--exclude-vendored-libraries"]
    )
    scip_skip_crates: list[str] = field(default_factory=list)
    scip_rust_toolchain: str | None = (
        None  # e.g. "1.92.0" to use `rustup run 1.92.0 rust-analyzer`
    )

    # Syntax highlighting
    syntax_lang_map: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_SYNTAX_LANG_MAP)
    )

    def __post_init__(self):
        self.project_root = Path(self.project_root)
        # The default cache directory is inside the project; one given here
        # or by DESCRY_CACHE_DIR is the user's.
        self.cache_dir_given = self.cache_dir is not None
        if self.cache_dir is None:
            self.cache_dir = self.project_root / ".descry_cache"
        else:
            self.cache_dir = Path(self.cache_dir)

    def cache_dir_error(self) -> str | None:
        """Why descry may not write to the cache directory, or None if it may.

        A repository can ship the default directory, or its `scip` or
        `parse` subdirectory, as a symlink to a directory outside it, where
        descry would replace files and remove stale ones. The default must
        stay inside the project; DESCRY_CACHE_DIR names any other.
        """
        if self.cache_dir_given:
            return None
        root = self.project_root.resolve()
        for path in (self.cache_dir, self.cache_dir / "scip", self.cache_dir / "parse"):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                return (
                    f"The cache directory {path} resolves to {resolved}, outside "
                    "the project. Set DESCRY_CACHE_DIR to keep the cache there."
                )
        return None

    @property
    def graph_path(self) -> Path:
        return self.cache_dir / "codebase_graph.json"

    @property
    def max_file_bytes(self) -> int:
        """`max_file_mb` in bytes; 0 when files of any size are accepted."""
        return int(self.max_file_mb * 1024 * 1024)

    @property
    def resolved_project_root(self) -> Path:
        """Project root with all symlinks resolved.

        Use for containment checks (api_source, api_index, descry_index).
        """
        return self.project_root.resolve()

    @classmethod
    def auto_detect(cls, cwd: Path | None = None) -> "DescryConfig":
        """Auto-detect project root from cwd and build config."""
        start = Path(cwd) if cwd else Path.cwd()
        markers = _DEFAULT_PROJECT_MARKERS

        for path in [start] + list(start.parents):
            for marker in markers:
                if (path / marker).exists():
                    return cls(project_root=path)
            if path == Path.home():
                break

        return cls(project_root=start)

    @staticmethod
    def _load_toml(project_root: Path) -> dict:
        """Load .descry.toml from project root if it exists.

        Rejects files > 1 MiB (TOML-parse DoS guard).

        Returns:
            Parsed TOML data as dict, or empty dict if not found/invalid.
        """
        import tomllib

        toml_path = Path(project_root) / ".descry.toml"
        try:
            with open_regular(toml_path) as f:
                if os.fstat(f.fileno()).st_size > 1 * 1024 * 1024:
                    logger.warning(
                        f".descry.toml exceeds 1 MiB cap; ignoring: {toml_path}"
                    )
                    return {}
                return tomllib.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, tomllib.TOMLDecodeError) as e:
            logger.warning(f"Failed to parse .descry.toml: {e}")
            return {}

    def _apply_toml(self, data: dict) -> None:
        """Apply parsed TOML data to this config instance.

        Maps TOML sections/keys to config fields.
        """
        if not data:
            return

        def table(name):
            """The TOML table of ``name``; a value of another type is ignored
            with a warning rather than crashing on its keys."""
            held = data.get(name)
            if held is None:
                return {}
            if not isinstance(held, dict):
                logger.warning(f"[{name}] in .descry.toml is not a table; ignoring")
                return {}
            return held

        def listing(section, key, value):
            """The list of strings ``value`` of a key; a value of another
            type, and an entry that is not a string, are ignored with a
            warning. A string is one, not its characters."""
            if value is None:
                return None
            if not isinstance(value, list):
                logger.warning(
                    f"[{section}] {key} in .descry.toml is not a list; ignoring"
                )
                return None
            kept = [entry for entry in value if isinstance(entry, str)]
            if len(kept) != len(value):
                logger.warning(
                    f"[{section}] {key} in .descry.toml has entries that are "
                    "not strings; ignoring them"
                )
                if not kept:
                    return None
            return kept

        # [project]
        project = table("project")
        excluded = listing("project", "excluded_dirs", project.get("excluded_dirs"))
        if excluded is not None:
            self.excluded_dirs = set(excluded)
        if "respect_gitignore" in project:
            if isinstance(project["respect_gitignore"], bool):
                self.respect_gitignore = project["respect_gitignore"]
            else:
                logger.warning(
                    "[project] respect_gitignore in .descry.toml is not a "
                    f"boolean; keeping {self.respect_gitignore!r}"
                )

        # [features]
        features = table("features")
        for key, attr in (
            ("enable_scip", "enable_scip"),
            ("enable_embeddings", "enable_embeddings"),
        ):
            if key in features:
                if isinstance(features[key], bool):
                    setattr(self, attr, features[key])
                else:
                    logger.warning(
                        f"[features] {key} in .descry.toml is not a boolean; "
                        f"keeping {getattr(self, attr)!r}"
                    )

        # [embeddings]
        embeddings = table("embeddings")
        if "model" in embeddings:
            try:
                self.embedding_model = _validate_embedding_model(
                    embeddings["model"], self.project_root
                )
            except ValueError as e:
                logger.warning(
                    f"Invalid [embeddings] model in .descry.toml: {e}; "
                    "using the default model"
                )

        # [test_detection]
        test_detection = table("test_detection")
        for key, attr in (
            ("path_patterns", "test_path_patterns"),
            ("file_suffixes", "test_file_suffixes"),
        ):
            patterns = listing("test_detection", key, test_detection.get(key))
            if patterns is not None:
                setattr(self, attr, tuple(patterns))

        # [code_files]
        code_files = table("code_files")
        extensions = listing("code_files", "extensions", code_files.get("extensions"))
        if extensions is not None:
            self.code_extensions = set(extensions)

        # [git]
        git = table("git")
        exclusions = listing("git", "churn_exclusions", git.get("churn_exclusions"))
        if exclusions is not None:
            self.churn_exclusions = exclusions

        for (section, key), attr in _TOML_NUMBERS.items():
            held = data.get(section)
            if isinstance(held, dict) and key in held:
                setattr(
                    self,
                    attr,
                    _toml_number(
                        section,
                        key,
                        held[key],
                        getattr(self, attr),
                        integer=attr in _TOML_INT_KEYS,
                        positive=(section, key) == ("git", "timeout"),
                    ),
                )

        # [scip]
        scip = table("scip")
        args = listing("scip", "extra_args", scip.get("extra_args"))
        if args is not None:
            try:
                self.scip_extra_args = [
                    _validate_scip_extra_arg(a, self.project_root) for a in args
                ]
            except ValueError as e:
                logger.warning(
                    f"Invalid [scip] extra_args in .descry.toml: {e}; "
                    f"keeping default {self.scip_extra_args!r}"
                )
        skip = listing("scip", "skip_crates", scip.get("skip_crates"))
        if skip is not None:
            self.scip_skip_crates = skip

        # [cross_lang]
        cross_lang = table("cross_lang")
        patterns = listing(
            "cross_lang",
            "backend_handler_patterns",
            cross_lang.get("backend_handler_patterns"),
        )
        if patterns is not None:
            self.backend_handler_patterns = patterns
        prefixes = listing("cross_lang", "api_prefixes", cross_lang.get("api_prefixes"))
        if prefixes is not None:
            # An empty prefix would match every path and strip nothing.
            self.api_prefixes = [p for p in prefixes if p]
        if "openapi_path" in cross_lang:
            raw = cross_lang["openapi_path"]
            if not isinstance(raw, str) or not raw:
                logger.warning(
                    "[cross_lang] openapi_path in .descry.toml is not a path; ignoring"
                )
            else:
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = self.project_root / candidate
                # Containment check: keep the spec inside project_root so a
                # crafted .descry.toml can't point descry at arbitrary files.
                try:
                    resolved = candidate.resolve(strict=False)
                    root = self.project_root.resolve()
                    if root in resolved.parents:
                        self.openapi_path = resolved
                    elif resolved == root:
                        logger.warning(
                            f"[cross_lang] openapi_path {raw!r} is the project "
                            f"root, not a spec file; ignoring"
                        )
                    else:
                        logger.warning(
                            f"[cross_lang] openapi_path {raw!r} resolves outside "
                            f"project root; ignoring"
                        )
                except (OSError, RuntimeError) as e:
                    logger.warning(f"Invalid [cross_lang] openapi_path: {e}")

        # [scip.rust]
        scip_rust = scip.get("rust", {})
        if not isinstance(scip_rust, dict):
            logger.warning("[scip.rust] in .descry.toml is not a table; ignoring")
        elif "toolchain" in scip_rust:
            try:
                self.scip_rust_toolchain = _validate_toolchain(scip_rust["toolchain"])
            except ValueError as e:
                logger.warning(
                    f"Invalid [scip.rust] toolchain in .descry.toml: {e}; ignoring"
                )

        # [syntax.lang_map] — merges with defaults (additive)
        syntax = table("syntax")
        lang_map = syntax.get("lang_map", {})
        if not isinstance(lang_map, dict):
            logger.warning("[syntax] lang_map in .descry.toml is not a table; ignoring")
            lang_map = {}
        for ext, lang in lang_map.items():
            # The value becomes a code-fence tag and part of a regex.
            if isinstance(lang, str) and _LANG_NAME_REGEX.match(lang):
                self.syntax_lang_map[ext] = lang
            else:
                logger.warning(
                    f"Invalid [syntax.lang_map] {ext!r} in .descry.toml: "
                    f"{lang!r} is not a language name; ignoring"
                )

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "DescryConfig":
        """Build config: auto_detect, unless given the root -> apply TOML ->
        apply env vars."""
        log_level = os.environ.get("DESCRY_LOG_LEVEL", "WARNING")
        level = logging.getLevelNamesMapping().get(log_level.upper())
        if level is None:
            logger.warning(f"Invalid DESCRY_LOG_LEVEL {log_level!r}; using WARNING")
            level = logging.WARNING
        logging.getLogger("descry").setLevel(level)

        config = cls(project_root=project_root) if project_root else cls.auto_detect()

        toml_data = cls._load_toml(config.project_root)
        config._apply_toml(toml_data)

        config._apply_env()
        return config

    def _apply_env(self) -> None:
        """Apply DESCRY_CACHE_DIR, DESCRY_NO_SCIP and DESCRY_NO_EMBEDDINGS."""
        cache_dir_str = os.environ.get("DESCRY_CACHE_DIR")
        if cache_dir_str:
            self.cache_dir = Path(cache_dir_str)
            self.cache_dir_given = True
        if env_flag("DESCRY_NO_SCIP"):
            self.enable_scip = False
        if env_flag("DESCRY_NO_EMBEDDINGS"):
            self.enable_embeddings = False


# --- Format Helpers (module-level pure functions) ---


def format_search_result(
    node: dict, show_score: bool = False, score: float = 0.0
) -> str:
    """Format a single search result with enriched information."""
    meta = node.get("metadata", {})
    node_type = node.get("type", "?")[:3]
    name = meta.get("name", "unknown")

    node_id = node.get("id", "")
    file_path = ""
    lineno = meta.get("lineno")
    if node_id.startswith("FILE:"):
        file_path = file_of(node_id)

    if file_path and lineno is not None:
        location = f"{file_path}:{lineno}"
    elif file_path:
        location = file_path
    else:
        location = node_id

    token_count = meta.get("token_count", 0)
    in_degree = meta.get("in_degree", 0)

    score_str = f"[{score:.2f}] " if show_score else ""
    metrics_str = (
        f"({token_count} toks, {in_degree} caller{'' if in_degree == 1 else 's'})"
        if token_count or in_degree
        else ""
    )

    lines = [f"{score_str}[{node_type}] {name} | {location} {metrics_str}".rstrip()]

    sig = meta.get("signature", "")
    if sig:
        lines.append(f"      {sig}")

    docstring = meta.get("docstring", "")
    if docstring:
        doc_lines = [line.strip() for line in docstring.split("\n") if line.strip()]
        if doc_lines:
            doc_preview = doc_lines[0]
            if len(doc_preview) > 120:
                doc_preview = doc_preview[:117] + "..."
            lines.append(f"      {doc_preview}")

    return "\n".join(lines)


def format_compact_result(node: dict, rank: int = 0) -> str:
    """Format a single search result in compact single-line format."""
    meta = node.get("metadata", {})
    node_type = node.get("type", "?")[:3]
    name = meta.get("name", "unknown")

    node_id = node.get("id", "")
    file_path = ""
    lineno = meta.get("lineno")
    if node_id.startswith("FILE:"):
        file_path = file_of(node_id)
        parts = file_path.split("/")
        if len(parts) > 2:
            file_path = "/".join(parts[-2:])

    location = f"{file_path}:{lineno}" if lineno else file_path

    token_count = meta.get("token_count", 0)
    in_degree = meta.get("in_degree", 0)

    return f"{rank}. [{node_type}] {name} | {location} ({token_count}t, {in_degree}c)"


def is_natural_language_query(terms: list[str]) -> bool:
    """Detect if query terms represent natural language vs code identifiers."""
    text = " ".join(terms).lower()

    nl_indicators = [
        "how to",
        "what is",
        "where is",
        "where are",
        "find the",
        "show me",
        "get the",
        "look for",
        "search for",
        "related to",
        "that handles",
        "that does",
        "responsible for",
        "used for",
        "deals with",
    ]
    if any(p in text for p in nl_indicators):
        return True

    if terms and terms[0].lower() in ("how", "what", "where", "why", "which", "find"):
        return True

    code_patterns = [
        r"[a-z]+_[a-z]+",
        r"[a-z]+[A-Z][a-z]+",
        r"[A-Z][a-z]+[A-Z]",
        r"::",
    ]
    for pattern in code_patterns:
        if re.search(pattern, text):
            return False

    return len(terms) >= 3


def reciprocal_rank_fusion(
    tfidf_results: list, semantic_results: list, k: int = 60
) -> list:
    """Combine rankings using Reciprocal Rank Fusion."""
    rrf_scores = defaultdict(float)
    node_lookup = {}

    for rank, node in enumerate(tfidf_results):
        node_id = node["id"]
        rrf_scores[node_id] += 1.0 / (k + rank + 1)
        node_lookup[node_id] = node

    for rank, (node, _) in enumerate(semantic_results):
        node_id = node["id"]
        rrf_scores[node_id] += 1.0 / (k + rank + 1)
        node_lookup[node_id] = node

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(node_lookup[nid], rrf_scores[nid]) for nid in sorted_ids]


# --- Lazy imports ---


def _try_import_cross_lang():
    try:
        from descry.cross_lang import CrossLangTracer

        return CrossLangTracer
    except ImportError:
        return None


def _try_import_embeddings(enabled: bool):
    if not enabled:
        return False, None, None
    try:
        from descry.embeddings import (
            EMBEDDINGS_AVAILABLE,
            SemanticSearcher,
            get_embeddings_status,
        )

        return EMBEDDINGS_AVAILABLE, SemanticSearcher, get_embeddings_status
    except ImportError:
        return False, None, None


def _try_import_scip(enabled: bool):
    if not enabled:
        return None
    try:
        from descry.scip.support import get_scip_status

        return get_scip_status
    except ImportError:
        return None


def _try_import_git_history():
    try:
        from descry.git_history import GitError, GitHistoryAnalyzer

        return True, GitHistoryAnalyzer, GitError
    except ImportError:
        return False, None, Exception


# --- DescryService ---


class ServiceError(str):
    """A service result that reports a failure.

    Still a `str`, so every interface returns it unchanged; the CLI checks
    the type to choose its exit status. `reason` tells apart the failures an
    interface answers differently. `detail`, output a tool wrote, follows
    `message` in the text and stays out of structured answers.
    """

    def __new__(cls, message: str, reason: str = "failed", detail: str = ""):
        error = super().__new__(cls, f"{message}\n{detail}" if detail else message)
        error.message, error.reason, error.detail = message, reason, detail
        return error


def failure(error) -> dict:
    """The answer of a failed tool from its `ServiceError` or `GitError`;
    `refused` gives its text back."""
    return fail(getattr(error, "reason", "failed"), error.message, _detail=error.detail)


def refused(answer: dict) -> ServiceError:
    """The text of a failed tool's answer."""
    return ServiceError(
        answer["error"], answer.get("reason", "failed"), answer.get("_detail", "")
    )


_GRAPH_NOT_FOUND = "Graph not found. Run descry ensure first."
_INDEX_BUSY = "An index run is already in progress"
# What turns embeddings off: the environment or `.descry.toml`
_EMBEDDINGS_DISABLED = (
    "Embeddings are disabled by DESCRY_NO_EMBEDDINGS or "
    "[features] enable_embeddings = false."
)


def public(answer: dict) -> dict:
    """A service answer as an interface gives it: without `ok`, and without
    the keys starting with "_", which are for its text only."""
    return {k: v for k, v in answer.items() if k != "ok" and k[0] != "_"}


def limit_error(limit: int, name: str = "limit") -> str | None:
    """Why the limit or depth ``name`` cannot be ``limit``, or None: a
    negative one slices from the end, and none at all answers as if there
    were nothing to show."""
    if limit < 1:
        return f"{name} must be at least 1, got {limit}."
    return None


class Language(str, Enum):
    """The languages a `search` filter can name; the MCP schema and the
    service validation both read these values."""

    rust = "rust"
    typescript = "typescript"
    python = "python"
    svelte = "svelte"
    javascript = "javascript"
    go = "go"
    java = "java"
    kotlin = "kotlin"
    scala = "scala"
    ruby = "ruby"
    php = "php"
    csharp = "csharp"
    vbnet = "vbnet"
    c = "c"
    cpp = "cpp"
    dart = "dart"
    protobuf = "protobuf"
    all = "all"


class Direction(str, Enum):
    forward = "forward"
    backward = "backward"


class CrossLangMode(str, Enum):
    endpoint = "endpoint"
    list = "list"
    stats = "stats"


class ChurnMode(str, Enum):
    symbols = "symbols"
    files = "files"
    co_change = "co-change"


class SymbolType(str, Enum):
    function = "function"
    class_ = "class"
    method = "method"
    constant = "constant"
    file = "file"
    all = "all"


_LANGUAGE_VALUES = {member.value for member in Language} | set(LANG_ALIASES)
_TYPE_VALUES = {member.value for member in SymbolType}
# What a search's `lang` takes in every interface: a language or its alias
LanguageName = Enum(
    "LanguageName", {value: value for value in sorted(_LANGUAGE_VALUES)}, type=str
)


def _decode_output(raw: bytes) -> str:
    """Decode a child's captured output as `subprocess.run(text=True)` does."""
    text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a process started by `_process_tree` and everything it started."""
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        # No process groups to signal here; taskkill walks the tree instead.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        # The child may have exited without the event loop having seen it yet.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)


@contextlib.asynccontextmanager
async def _process_tree(argv: list[str], **kwargs):
    """Run `argv` in a session of its own, so that what it starts dies with it.

    Leaving the block kills whatever is still running, and returns only once
    the child has exited. The session keeps the terminal's Ctrl-C and hangup,
    and a signal sent to this process's group, from reaching the tree, so
    while the block runs each of those kills it first and is then handled as
    it would have been. The child's stdin is /dev/null, so neither it nor an
    indexer it starts reads the terminal or the stream this process was
    started with.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv, start_new_session=True, stdin=asyncio.subprocess.DEVNULL, **kwargs
    )

    def stop(signum, frame):
        _kill_process_tree(proc)
        previous = handlers[signum]
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signum, previous)
            signal.raise_signal(signum)

    handlers = {}
    # Windows starts no session, and only the main thread may set handlers.
    # A signal this process ignores, as under `nohup`, stays ignored.
    if (
        sys.platform != "win32"
        and threading.current_thread() is threading.main_thread()
    ):
        handlers = {
            signum: signal.signal(signum, stop)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
            if signal.getsignal(signum) != signal.SIG_IGN
        }
    try:
        yield proc
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
        _kill_process_tree(proc)
        await proc.wait()


class DescryService:
    """Core business logic. No MCP dependency."""

    def __init__(self, config: DescryConfig | None = None):
        self.config = config or DescryConfig.from_env()

        # Lazy-loaded modules
        self._CrossLangTracer = _try_import_cross_lang()

        self._get_scip_status_fn = _try_import_scip(self.config.enable_scip)

        emb_ok, self._SemanticSearcher, self._get_embeddings_status_fn = (
            _try_import_embeddings(self.config.enable_embeddings)
        )
        self._semantic_available = emb_ok

        git_ok, self._GitHistoryAnalyzer, self._GitError = _try_import_git_history()
        self._git_available = git_ok

        self._graph_cache_lock = asyncio.Lock()
        self._querier_cache_lock = asyncio.Lock()
        self._semantic_cache_lock = asyncio.Lock()
        self._git_cache_lock = asyncio.Lock()
        self._dedup_cache_lock = asyncio.Lock()

        self.reset_caches()
        self._max_dedup_entries = 100

        # Index runs write the same cache directory, so one runs at a time
        # and another is refused, not queued. Never waited on, so a thread
        # lock serves whichever event loop the service is called from.
        self._index_lock = threading.Lock()

    # --- Internal helpers ---

    def _graph_mtime(self) -> float | None:
        """The graph file's mtime, or None if it cannot be stat'ed."""
        try:
            return self.config.graph_path.stat().st_mtime
        except OSError:
            return None

    def _get_graph_status(self) -> tuple[bool, str, float | None]:
        mtime = self._graph_mtime()
        if mtime is None:
            return False, "Graph not found", None
        age_hours = (time.time() - mtime) / 3600
        return True, f"{age_hours:.1f}h ago", age_hours

    async def _update_cache(self):
        async with self._graph_cache_lock:
            gp = self.config.graph_path
            mtime = self._graph_mtime()
            if mtime is not None and mtime != self._graph_cache["mtime"]:
                try:
                    data = load_graph_with_schema(gp)
                    self._graph_cache = {
                        "mtime": mtime,
                        "nodes": len(data.get("nodes", [])),
                        "edges": len(data.get("edges", [])),
                        "languages": self._graph_languages(data),
                        "error": None,
                    }
                # ValueError: invalid JSON or invalid UTF-8.
                except (OSError, ValueError, GraphSchemaError) as e:
                    logger.warning(f"Failed to update graph cache: {e}")
                    # Zero counts, not those of the graph this one replaced.
                    self._graph_cache = {
                        "mtime": mtime,
                        "nodes": 0,
                        "edges": 0,
                        "languages": [],
                        "error": str(e),
                    }

    def _graph_languages(self, data) -> list[str]:
        """The languages of the graph's nodes that the `lang` search filter
        names, as it finds them: by the extension of the file a node is in.
        A `[syntax.lang_map]` name of no such language only tags fences."""
        paths = {
            file_of(node["id"])
            for node in data.get("nodes", [])
            if node.get("id", "").startswith("FILE:")
        }
        langs = {_get_syntax_lang(p, self.config.syntax_lang_map) for p in paths}
        return sorted(langs & {lang.value for lang in Language})

    async def _get_querier(self):
        """The cached GraphQuerier; None if the graph is missing or unloadable."""
        async with self._querier_cache_lock:
            gp = self.config.graph_path
            mtime = self._graph_mtime()
            if mtime is None:
                self._querier_cache = {"mtime": 0, "instance": None, "error": None}
                return None
            if mtime != self._querier_cache["mtime"]:
                try:
                    instance = GraphQuerier(str(gp), config=self.config)
                    error = None
                except (OSError, ValueError, GraphSchemaError) as e:
                    logger.warning(f"Cannot load graph {gp}: {e}")
                    instance, error = None, str(e)
                self._querier_cache = {
                    "mtime": mtime,
                    "instance": instance,
                    "error": error,
                }
            return self._querier_cache["instance"]

    def _no_graph(self) -> dict:
        """The failure of a tool that needs a graph it cannot load."""
        return fail("no_graph", self._no_graph_error())

    def _no_graph_error(self) -> str:
        """Why `_get_querier` returned None."""
        error = self._querier_cache["error"]
        if error:
            return f"Graph unusable ({error}). Run descry index to rebuild it."
        return _GRAPH_NOT_FOUND

    async def _get_git_analyzer(self):
        async with self._git_cache_lock:
            current_mtime = self._graph_mtime()
            if (
                self._git_cache["graph_mtime"] != current_mtime
                or self._git_cache["analyzer"] is None
            ):
                q = await self._get_querier()
                self._git_cache["analyzer"] = self._GitHistoryAnalyzer(
                    str(self.config.project_root),
                    graph_querier=q,
                    churn_exclusions=self.config.churn_exclusions,
                    code_extensions=self.config.code_extensions,
                    git_timeout=self.config.git_timeout,
                )
                self._git_cache["graph_mtime"] = current_mtime
            return self._git_cache["analyzer"]

    async def _check_dedup(self, content_hash: str, graph_mtime: float) -> str | None:
        async with self._dedup_cache_lock:
            if content_hash in self._dedup_cache:
                cached_mtime, node_id = self._dedup_cache[content_hash]
                if cached_mtime == graph_mtime:
                    return f"[Source shown above - see {node_id}]"
            return None

    async def _record_dedup(self, content_hash: str, graph_mtime: float, node_id: str):
        async with self._dedup_cache_lock:
            if len(self._dedup_cache) >= self._max_dedup_entries:
                keys_to_remove = list(self._dedup_cache.keys())[:20]
                for key in keys_to_remove:
                    del self._dedup_cache[key]
            self._dedup_cache[content_hash] = (graph_mtime, node_id)

    def reset_caches(self):
        """Reset all cached instances. Called on construction and after a
        reindex."""
        self._graph_cache = {
            "mtime": 0,
            "nodes": 0,
            "edges": 0,
            "languages": [],
            "error": None,
        }
        self._querier_cache = {"mtime": 0, "instance": None, "error": None}
        self._semantic_cache = {
            "mtime": 0,
            "instance": None,
            "load": None,
            "error": None,
        }
        self._git_cache = {"analyzer": None, "graph_mtime": None}
        self._dedup_cache: dict[str, tuple[float, str]] = {}
        # The module-level file-content cache in query.py is keyed on mtime,
        # so it self-invalidates after an edit; but a reindex can shift file
        # contents even without an mtime bump (copies, git checkouts). Clear
        # it explicitly so long-running MCP/web processes never serve stale
        # source lines after `descry index`.
        from descry.query import _clear_file_cache

        _clear_file_cache()

    async def _format_response(
        self, content: str, include_header: bool = True, max_lines: int = 500
    ) -> str:
        lines = content.split("\n")
        if len(lines) > max_lines:
            content = (
                "\n".join(lines[:max_lines])
                + f"\n\n[Safety truncated: {len(lines) - max_lines} more lines]"
            )

        if not include_header:
            return content

        exists, age_str, age_hours = self._get_graph_status()
        if not exists:
            return f"[Graph: NOT FOUND]\n\n{content}"

        await self._update_cache()
        stale = (
            " STALE" if age_hours and age_hours > self.config.max_stale_hours else ""
        )
        header = f"[{self._graph_cache['nodes']:,}n/{self._graph_cache['edges']:,}e | {age_str}{stale}]"
        return f"{header}\n\n{content}"

    def _new_searcher(self):
        """A SemanticSearcher over this service's graph, cache and model."""
        if error := self.config.cache_dir_error():
            raise RuntimeError(error)
        return self._SemanticSearcher(
            str(self.config.graph_path),
            cache_dir=str(self.config.cache_dir),
            model_name=self.config.embedding_model,
            source_root=str(self.config.project_root),
        )

    def _start_semantic_load(self, cache, build):
        """Run ``build`` in a thread as the load of ``cache``, which holds
        what it builds, or the error it raises, as of the graph it began
        on. Called holding the lock; returns the task."""
        mtime = self._graph_mtime() or 0

        async def load():
            try:
                # No timeout: the thread cannot be stopped, so a load given
                # up on would go on while another started.
                searcher = await asyncio.to_thread(build)
                async with self._semantic_cache_lock:
                    cache.update(mtime=mtime, instance=searcher, error=None)
                return searcher
            except Exception as e:  # noqa: BLE001 — boundary over sentence-transformers/torch model load; error is cached and logged
                async with self._semantic_cache_lock:
                    # Its message, which may name what the project
                    # configured, goes to the log, not into an answer.
                    cache["error"] = type(e).__name__
                logger.warning(f"Embeddings load failed: {e}")
                return None
            finally:
                async with self._semantic_cache_lock:
                    cache["load"] = None

        cache["load"] = asyncio.create_task(load())
        return cache["load"]

    async def _load_embeddings_background(self):
        """Load the searcher and its model, so the first search pays for
        neither."""

        def build():
            searcher = self._new_searcher()
            # On a warm cache the constructor never touches the model.
            searcher.ensure_model()
            return searcher

        async with self._semantic_cache_lock:
            cache = self._semantic_cache
            if cache["instance"] is not None:
                return
            load = cache["load"] or self._start_semantic_load(cache, build)
        if await asyncio.shield(load) is not None:
            logger.info("Pre-warm: embeddings ready")

    async def _load_indexed_embeddings(self, model_was_loaded):
        """Load the embeddings of a graph just indexed, and the model where
        it was loaded before; returns the line on them for the index's
        summary, empty without embeddings."""
        if not (self._semantic_available and self.config.graph_path.exists()):
            return ""
        logger.info("Loading embeddings for semantic search...")

        def build():
            # The child process has just embedded this graph and cached the
            # vectors under a key derived from the graph's content, so this
            # loads them. It encodes only when the child could not, since no
            # cache then matches.
            searcher = self._new_searcher()
            # Loading from cache never touches the model. A server that had
            # it loaded keeps it loaded, so the first search after a reindex
            # does not pay for the load; a one-shot `descry index` skips it.
            if model_was_loaded:
                searcher.ensure_model()
            return searcher

        async with self._semantic_cache_lock:
            cache = self._semantic_cache
            load = cache["load"] or self._start_semantic_load(cache, build)
        searcher = await asyncio.shield(load)
        if searcher is None:
            return f"Embeddings: Failed ({cache['error']})"
        return f"Embeddings: {len(searcher.nodes):,} nodes indexed"

    async def _get_semantic_searcher(self):
        """Single-flight accessor for the semantic searcher.

        Returns the cached SemanticSearcher, loading it again if
        codebase_graph.json mtime changed, and waits up to
        `embedding_timeout` for a load under way. Returns None if
        embeddings are unavailable (module missing or disabled), if the
        graph file does not exist, or while a load is not done; then
        _semantic_cache["error"] is the last failed load's until a load
        succeeds.
        """
        if not self._semantic_available:
            return None
        mtime = self._graph_mtime()
        if mtime is None:
            return None

        async with self._semantic_cache_lock:
            cache = self._semantic_cache
            if cache["instance"] is not None and cache["mtime"] == mtime:
                return cache["instance"]
            load = cache["load"] or self._start_semantic_load(cache, self._new_searcher)
        try:
            return await asyncio.wait_for(
                asyncio.shield(load), timeout=self.config.embedding_timeout
            )
        except TimeoutError:
            logger.warning(
                f"Embeddings still loading after {self.config.embedding_timeout}s"
            )
            return None

    # --- Public API (handle_* → service methods) ---

    async def embedding_models_answer(self) -> dict:
        """The embedding models descry ships pinned configurations for, which
        one is selected, and the model built with where the default is
        gated."""
        from descry.embeddings import (
            DEFAULT_MODEL_ALIAS,
            FALLBACK_MODEL_ALIAS,
            list_models,
        )

        configured = self.config.embedding_model
        selected = configured or DEFAULT_MODEL_ALIAS
        models = list_models()
        for m in models:
            m["selected"] = selected in (m["alias"], m["repo_id"])
        return {
            "ok": True,
            "configured": configured,
            "fallback_if_gated": None if configured else FALLBACK_MODEL_ALIAS,
            "models": models,
        }

    async def health(self) -> str:
        """Quick diagnostic check, as JSON text."""
        return await self.health_text(await self.health_answer())

    async def health_text(self, answer: dict) -> str:
        """The text of a `health_answer` or an `embedding_models_answer`: the
        answer as JSON."""
        return json.dumps(public(answer), indent=2)

    async def health_answer(self) -> dict:
        """Quick diagnostic check: version, paths, warm state, the depth a
        flow goes to at most, features and the graph's state."""
        from descry.query import flow_ceiling

        health = {
            "status": "ok",
            "version": __version__,
            "project_root": str(self.config.project_root),
            "cache_dir": str(self.config.cache_dir),
            "warm": False,
            "max_depth": flow_ceiling(self.config.max_depth),
            "features": {
                "scip": False,
                "embeddings": False,
                "embeddings_loading": False,
                "embeddings_error": None,
                "git_history": self._git_available,
                "cross_lang": self.cross_lang_error() is None,
            },
            "graph": public(await self.status_answer()),
        }

        async with self._querier_cache_lock:
            health["warm"] = self._querier_cache["instance"] is not None

        if self._get_scip_status_fn:
            scip_status = self._get_scip_status_fn()
            health["features"]["scip"] = scip_status.get("available", False)
            if scip_status.get("disabled_by_env"):
                health["features"]["scip_note"] = "Disabled by DESCRY_NO_SCIP"

        async with self._semantic_cache_lock:
            health["features"]["embeddings"] = (
                self._semantic_cache["instance"] is not None
            )
            health["features"]["embeddings_loading"] = (
                self._semantic_cache["load"] is not None
            )
            health["features"]["embeddings_error"] = self._semantic_cache.get("error")

        if not self._semantic_available:
            if not self.config.enable_embeddings:
                health["features"]["embeddings_note"] = _EMBEDDINGS_DISABLED
            else:
                health["features"]["embeddings_note"] = (
                    "sentence-transformers not installed"
                )

        graph = health["graph"]
        if not graph["exists"]:
            health["status"] = "no_graph"
        elif graph["error"]:
            health["status"] = "invalid_graph"
        elif graph["stale"]:
            health["status"] = "stale"

        return {"ok": True, **health}

    async def status_answer(self) -> dict:
        """Graph existence, age and size, and the state of SCIP and of the
        embeddings, as a structured answer.

        A graph that cannot be loaded counts no nodes or edges, and `error`
        says why. `scip` is `{state, indexers, cached}` with `state` one of
        "enabled", "disabled", "unavailable" or "not_loaded"; `embeddings`
        is `{state, nodes, model}` with `state` one of "ready", "stale",
        "missing", "disabled" or "unavailable".
        """
        exists, age_str, age_hours = self._get_graph_status()
        if not exists:
            graph = {
                "exists": False,
                "age_str": "N/A",
                "age_hours": None,
                "nodes": 0,
                "edges": 0,
                "languages": [],
                "stale": False,
                "error": None,
            }
        else:
            await self._update_cache()
            graph = {
                "exists": True,
                "age_str": age_str,
                "age_hours": round(age_hours, 2),
                "nodes": self._graph_cache["nodes"],
                "edges": self._graph_cache["edges"],
                "languages": self._graph_cache["languages"],
                "stale": age_hours > self.config.max_stale_hours,
                "error": self._graph_cache["error"],
            }
        return {
            "ok": True,
            **graph,
            "scip": self._scip_state(),
            "embeddings": self._embeddings_state(),
        }

    def _scip_state(self) -> dict:
        scip = {"state": "not_loaded", "indexers": [], "cached": None}
        if not self.config.enable_scip:
            scip["state"] = "disabled"
            return scip
        if not self._get_scip_status_fn:
            return scip
        status = self._get_scip_status_fn()
        if status.get("available"):
            scip_cache_dir = self.config.cache_dir / "scip"
            return {
                "state": "enabled",
                "indexers": [
                    name
                    for name, info in status.get("indexers", {}).items()
                    if info.get("available")
                ],
                "cached": len(list(scip_cache_dir.glob("*.scip")))
                if scip_cache_dir.exists()
                else None,
            }
        scip["state"] = "disabled" if status.get("disabled_by_env") else "unavailable"
        return scip

    def _embeddings_state(self) -> dict:
        if not self._semantic_available:
            state = "disabled" if not self.config.enable_embeddings else "unavailable"
            return {"state": state, "nodes": 0, "model": None}
        status = self._get_embeddings_status_fn(
            str(self.config.graph_path),
            model_name=self.config.embedding_model,
            cache_dir=str(self.config.cache_dir),
            source_root=str(self.config.project_root),
        )
        state = (
            "ready"
            if status.get("cached")
            else "stale"
            if status.get("stale")
            else "missing"
        )
        return {
            "state": state,
            "nodes": status.get("node_count", 0),
            "model": status.get("model"),
        }

    async def status(self) -> str:
        """Check graph existence and freshness."""
        return await self.status_text(await self.status_answer())

    async def status_text(self, answer: dict) -> str:
        """The text of a `status_answer`."""
        if not answer["exists"]:
            return "Graph: NOT FOUND. Run descry index first."
        if answer["error"]:
            return f"Graph: UNUSABLE ({answer['error']})"
        stale = " (STALE - run descry index)" if answer["stale"] else ""
        lines = [
            f"Graph: {answer['nodes']:,} nodes, {answer['edges']:,} edges, updated {answer['age_str']}{stale}"
        ]
        scip = answer["scip"]
        if scip["state"] == "enabled":
            cached = (
                f", {scip['cached']} project(s) cached"
                if scip["cached"] is not None
                else ""
            )
            lines.append(
                f"SCIP: Enabled ({', '.join(scip['indexers']) or 'none'}){cached}"
            )
        else:
            lines.append(
                {
                    "disabled": "SCIP: Disabled (DESCRY_NO_SCIP=1)",
                    "unavailable": "SCIP: Unavailable (no indexers found)",
                    "not_loaded": "SCIP: Not loaded",
                }[scip["state"]]
            )
        emb = answer["embeddings"]
        lines.append(
            {
                "ready": f"Embeddings: Ready ({emb['nodes']:,} nodes, {emb['model']})",
                "stale": "Embeddings: STALE (graph updated, will rebuild on first search)",
                "missing": "Embeddings: Not generated (will build on first search)",
                "disabled": "Embeddings: Disabled (DESCRY_NO_EMBEDDINGS=1 or "
                "[features] enable_embeddings = false)",
                "unavailable": "Embeddings: Unavailable",
            }[emb["state"]]
        )
        return "\n".join(lines)

    async def ensure_answer(self, max_age_hours: float | None = None) -> dict:
        """Ensure the graph exists and is fresh, as a structured answer.

        `max_age_hours` defaults to `[project] max_stale_hours`. `action` is
        "ready" when the graph was left alone, "refreshed" when it was stale
        and "generated" when there was none that could be used; `summary`
        says which in words; `graph` is the graph's status after it. The
        index report, None when "ready", is `_report`, for the text.
        On failure `reason` is "invalid" when `max_age_hours` is not an age,
        "busy" when the graph needs an index run and one is in progress, and
        "failed" when the index run failed.
        """
        if max_age_hours is None:
            max_age_hours = self.config.max_stale_hours
        # Chained, so that NaN is refused too.
        elif not 0 <= max_age_hours < math.inf:
            return fail(
                "invalid",
                "max_age_hours must be a finite, non-negative number, "
                f"got {max_age_hours}.",
            )
        exists, age_str, age_hours = self._get_graph_status()

        if not exists:
            action, summary = "generated", "Generated new graph."
        elif age_hours and age_hours > max_age_hours:
            action = "refreshed"
            summary = f"Refreshed stale graph ({age_hours:.0f}h old)."
        else:
            await self._update_cache()
            if not self._graph_cache["error"]:
                return {
                    "ok": True,
                    "action": "ready",
                    "summary": f"Graph ready: {self._graph_cache['nodes']:,}n/{self._graph_cache['edges']:,}e, {age_str}",
                    "graph": public(await self.status_answer()),
                    "_report": None,
                }
            action, summary = "generated", "Rebuilt unusable graph."

        result = await self.index(".")
        if isinstance(result, ServiceError):
            return failure(result)
        return {
            "ok": True,
            "action": action,
            "summary": summary,
            "graph": public(await self.status_answer()),
            "_report": result,
        }

    async def ensure(self, max_age_hours: float | None = None) -> str:
        """Ensure graph exists and is fresh."""
        return await self.ensure_text(await self.ensure_answer(max_age_hours))

    async def ensure_text(self, answer: dict) -> str:
        """The text of an `ensure_answer`."""
        if not answer["ok"]:
            return refused(answer)
        if answer["_report"] is None:
            return answer["summary"]
        return f"{answer['summary']}\n{answer['_report']}"

    def _indexer_child(self, index_path: str) -> tuple[list[str], dict[str, str]]:
        """argv and env for the `descry.generate` child process."""
        # -P keeps the cwd, which is the repository being indexed, off sys.path,
        # so a `descry` package inside it cannot shadow the installed one.
        argv = [
            sys.executable,
            "-P",
            "-m",
            "descry.generate",
            f"--project-root={self.config.project_root}",
            index_path,
        ]
        # The child writes to the cache dir this service reads from.
        env = {
            **os.environ,
            "DESCRY_CACHE_DIR": str(self.config.cache_dir.resolve()),
        }
        return argv, env

    def _index_timeout_message(self) -> str:
        return (
            f"Index timed out after {self.config.index_timeout_minutes} minutes. "
            "Set [timeouts] index_minutes in .descry.toml to increase."
        )

    async def index(self, path: str = ".") -> str:
        """Regenerate the codebase graph.

        `path` must resolve inside project_root. Prevents prompt-injected
        MCP clients from forcing indexing of arbitrary directories.

        A `path` below the root only narrows which files are walked. The graph
        replaces this service's graph, is built under the root's config, and
        records root-relative paths.

        While another run is in progress this one is refused: the result is
        a `ServiceError` whose `reason` is "busy".
        """
        if path == ".":
            index_path = str(self.config.project_root)
        else:
            try:
                # The path is read from the project root, not the process
                # cwd, so `./` is the root and `sub` the root's subdirectory.
                resolved = Path(path)
                if not resolved.is_absolute():
                    resolved = self.config.project_root / resolved
                resolved = resolved.resolve(strict=False)
                if not resolved.is_relative_to(self.config.resolved_project_root):
                    return ServiceError(
                        f"Index path {path!r} outside project root; "
                        f"must be relative to {self.config.project_root}.",
                        "invalid",
                    )
                # The existing graph is replaced, so a mistyped path must not
                # swap it for an empty one.
                if not resolved.is_dir():
                    return ServiceError(
                        f"Index path {path!r} is not a directory.", "invalid"
                    )
                index_path = str(resolved)
            except (OSError, ValueError) as e:
                return ServiceError(f"Invalid index path {path!r}: {e}", "invalid")
        if error := self.config.cache_dir_error():
            return ServiceError(error)

        if not self._index_lock.acquire(blocking=False):
            return ServiceError(_INDEX_BUSY, reason="busy")
        # Read before reset_caches() below discards it.
        model_was_loaded = self._semantic_cache["instance"] is not None

        try:
            timeout = (
                self.config.index_timeout_minutes * 60
                if self.config.index_timeout_minutes
                else None
            )
            argv, env = self._indexer_child(index_path)
            async with _process_tree(
                argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.config.project_root),
                env=env,
            ) as proc:
                stdout, stderr = map(
                    _decode_output,
                    await asyncio.wait_for(proc.communicate(), timeout),
                )

            if proc.returncode == 0:
                output = stdout.strip() or stderr.strip()
                self.reset_caches()

                scip = self._scip_state()
                scip_status = (
                    f"\nSCIP: {scip['cached']} project(s) indexed"
                    if scip["state"] == "enabled" and scip["cached"]
                    else ""
                )

                embeddings = await self._load_indexed_embeddings(model_was_loaded)
                embeddings_status = f"\n{embeddings}" if embeddings else ""
                return f"Index complete.{scip_status}{embeddings_status}\n{output}"
            else:
                return ServiceError(
                    f"Index failed: the indexer exited with code {proc.returncode}.",
                    detail=stderr,
                )

        except TimeoutError:
            return ServiceError(self._index_timeout_message())
        except Exception as e:  # noqa: BLE001 — tool boundary: every failure must come back as an error payload
            return ServiceError(f"Index error: {e}")
        finally:
            self._index_lock.release()

    @classmethod
    def _one_symbol(cls, q, symbol: str, crate: str | None = None):
        """``(node_id, None)`` for the definition ``symbol`` names, or
        ``(None, refusal)`` when it names none or several; with ``crate``
        only those of that crate, as `search --crate` reads it, count. A name
        no definition has is refused, with the close matches in the text."""
        if not symbol.strip():
            return None, fail(
                "invalid",
                "A symbol is required.",
            )
        node_id, choices, fuzzy = q.resolve_symbol(symbol)
        if fuzzy:
            return None, cls._refusal(symbol, [], [node_id] if node_id else choices)
        if crate:
            choices = [c for c in choices or [node_id] if c and crate_of(c) == crate]
            node_id = choices.pop() if len(choices) == 1 else None
        return (node_id, None) if node_id else (None, cls._refusal(symbol, choices))

    @staticmethod
    def _refusal(symbol: str, choices: list, close: list = ()) -> dict:
        """The answer of a tool given a ``symbol`` that names no definition,
        or several (``choices``): which, so one can be given by node ID.
        ``close``, definitions near ``symbol``, are named in the text of a
        refusal for none."""

        def listed(ids):
            more = f"\n  ... and {len(ids) - 20} more" if len(ids) > 20 else ""
            return "\n".join(f"  - {i}" for i in ids[:20]) + more

        if not choices:
            error = f"Symbol '{symbol}' not found. Try descry search."
            if close:
                error += f"\nClose matches:\n{listed(close)}"
            return fail("not_found", error)
        error = (
            f"'{symbol}' names {len(choices)} definitions; "
            f"give one by node ID:\n{listed(choices)}"
        )
        return fail("ambiguous", error, choices=choices)

    @staticmethod
    def _node_ref(node: dict) -> dict:
        """A definition as a structured answer gives it: ID, type, name,
        file, line, size in tokens and how many callers are certain. Never its source, signature or docstring."""
        meta = node.get("metadata", {})
        node_id = node["id"]
        return {
            "id": node_id,
            "type": node.get("type"),
            "name": meta.get("name"),
            "file": file_of(node_id) if node_id.startswith("FILE:") else None,
            "line": meta.get("lineno"),
            "tokens": meta.get("token_count", 0),
            "callers": meta.get("in_degree", 0),
        }

    async def _nodes(self, refs: list) -> list:
        """The graph nodes of `_node_ref` entries, for rendering them."""
        q = await self._get_querier()
        return [
            (q.nodes.get(ref["id"]) if q else None) or {"id": ref["id"]} for ref in refs
        ]

    @staticmethod
    def _counts(q, ids: list, marks: dict) -> dict:
        return dict(zip(("certain", "possible", "unresolved"), q.kinds(ids, marks)))

    async def callers_answer(self, name: str, limit: int = 20) -> dict:
        """The callers of a symbol as a structured answer: who calls it, by
        kind, how many of each, and how many of each a `limit` leaves out."""
        return await self._listing(self._callers_from, "callers", name, limit)

    async def callees_answer(self, name: str, limit: int = 20) -> dict:
        """The callees of a symbol as a structured answer, like
        `callers_answer`."""
        return await self._listing(self._callees_from, "callees", name, limit)

    async def _listing(self, fetch, key: str, name: str, limit: int) -> dict:
        if error := limit_error(limit):
            return fail("invalid", error)
        if not name.strip():
            return fail(
                "invalid",
                "A symbol is required.",
            )
        # One querier for lookup and the answer: a second fetch returns None
        # if the graph file has been removed in between.
        q = await self._get_querier()
        if not q:
            return self._no_graph()
        data = fetch(q, name, limit)
        if not data["ok"]:
            return data
        marks = data["ambiguous"]
        return {
            "ok": True,
            "symbol": data["symbol"],
            "total": data["total"],
            **self._counts(q, data[key] + data["left_out"], marks),
            "more": self._counts(q, data["left_out"], marks),
            key: [q.reference(i, marks.get(i)) for i in data[key]],
        }

    @staticmethod
    def _callers_from(q, name: str, limit: int) -> dict:
        node_id, choices, fuzzy = q.resolve_symbol(name)
        if fuzzy:
            # The name is no definition's; a close match is named, not
            # answered for, as the one-symbol tools do.
            return DescryService._refusal(name, [], [node_id] if node_id else choices)
        # A name no definition has may still be called; a node ID
        # the graph lacks names nothing
        if not node_id and (choices or name.startswith("FILE:")):
            return DescryService._refusal(name, choices)
        # Asked by name, the calls written with it come too, as unresolved
        # calls of the name; asked by node ID, those resolved or narrowed to it.
        asked = node_id if name == node_id else name
        all_callers, ambiguous = q.ordered_callers(asked)
        if not node_id and not all_callers and not name.startswith("FILE:"):
            # No definition has the name, and no call writes it.
            return DescryService._refusal(name, [], choices)
        return {
            "ok": True,
            "symbol": node_id or name,
            "total": len(all_callers),
            "callers": all_callers[:limit],
            "left_out": all_callers[limit:],
            "ambiguous": ambiguous,
        }

    @staticmethod
    def _callees_from(q, name: str, limit: int) -> dict:
        node_id, refusal = DescryService._one_symbol(q, name)
        if refusal:
            return refusal
        callees = sorted(q.get_callees(node_id))
        return {
            "ok": True,
            "symbol": node_id,
            "node": q.nodes[node_id],
            "total": len(callees),
            "callees": callees[:limit],
            "left_out": callees[limit:],
            "ambiguous": q.get_ambiguous_callees(node_id),
        }

    async def callers(self, name: str, limit: int = 20) -> str:
        """Find all callers of a symbol."""
        return await self.listing_text(
            await self.callers_answer(name, limit), "callers", limit
        )

    async def callees(self, name: str, limit: int = 20) -> str:
        """Find what a symbol calls."""
        return await self.listing_text(
            await self.callees_answer(name, limit), "callees", limit
        )

    async def listing_text(self, answer: dict, key: str, limit: int) -> str:
        """The markdown of a `callers_answer` or `callees_answer`."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        q = GraphQuerier
        symbol = answer["symbol"]
        extra = q.possible(answer["possible"], answer["unresolved"])
        if not answer[key]:
            result = (
                f"No callers of '{symbol}'. Try descry search to verify symbol exists."
                if key == "callers"
                else f"'{symbol}' calls no tracked functions."
            )
        else:
            lines = [
                f"{answer['certain']} caller(s) of '{symbol}'{extra}:"
                if key == "callers"
                else f"'{symbol}' calls {answer['certain']} function(s){extra}:"
            ]
            for ref in answer[key]:
                candidates = {"resolved": None, "unresolved": 0}.get(
                    ref["kind"], ref.get("candidates")
                )
                note = (q.one_of if key == "callers" else q.callee_note)(candidates)
                where = f" ({ref['file']}:{ref['line']})" if ref["line"] else ""
                lines.append(f"  {ref['id']}{where}{note}")
            more = answer["more"]
            if any(more.values()):
                of = "callers" if key == "callers" else "callees"
                left = q.more_callers(
                    more["certain"], more["possible"], of, more["unresolved"]
                )
                lines.append(f"  ... ({left}, limit {limit})")
            result = "\n".join(lines)
        return await self._format_response(result)

    async def context_answer(
        self,
        node_id: str,
        brief: bool = False,
        full: bool = False,
        expand_callees: bool = False,
        deduplicate: bool = False,
        depth: int = 1,
        max_tokens: int = 2000,
        callee_budget: int = 2000,
        head_lines: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict:
        """A symbol's context as a structured answer: the symbol and, brief,
        its counts; else its callers, its own callees and what each list
        leaves out by kind, where it is called (`examples`), its members,
        related tests and configuration, and what `max_output_tokens` left
        out (`omitted`, `truncated`). `repeated` when `deduplicate` finds it
        shown before. Source, signatures, docstrings and the callee
        expansion are only in the text (`_parts`)."""
        for value, name in (
            (depth, "depth"),
            (max_tokens, "max_tokens"),
            (callee_budget, "callee_budget"),
            (head_lines, "head_lines"),
            (max_output_tokens, "max_output_tokens"),
        ):
            if value is not None and (error := limit_error(value, name)):
                return fail("invalid", error)
        q = await self._get_querier()
        if not q:
            return self._no_graph()
        asked = node_id
        node_id, refusal = self._one_symbol(q, node_id)
        if refusal:
            return refusal

        # One mtime for the check and the record, so a record never pairs
        # one graph's content with another's mtime.
        graph_mtime = self._graph_mtime() if deduplicate else None
        request = {
            "asked": depth,
            "full": full,
            "head_lines": head_lines,
            "expand_callees": expand_callees,
        }
        symbol = self._node_ref(q.nodes[node_id])
        if graph_mtime is not None:
            # usedforsecurity=False: this hash is only a dedup key for the
            # in-memory context cache, never a trust anchor.
            content_hash = hashlib.md5(
                node_id.encode(), usedforsecurity=False
            ).hexdigest()
            dedup_ref = await self._check_dedup(content_hash, graph_mtime)
            if dedup_ref:
                parts = q.context_data(node_id, brief=True)
                return {
                    "ok": True,
                    "symbol": symbol,
                    "repeated": True,
                    "_parts": parts,
                    "_dedup": dedup_ref,
                    "_request": request,
                }

        parts = q.context_data(
            node_id,
            depth=depth,
            max_tokens=max_tokens,
            full=full,
            head_lines=head_lines,
            expand_callees=expand_callees,
            callee_budget=callee_budget,
            brief=brief,
            max_output_tokens=max_output_tokens,
        )
        if "error" in parts:
            return fail("not_found", parts["error"])
        if asked != node_id:
            parts["note"] = f"(Matched '{asked}' → `{node_id}`)"
        if graph_mtime is not None:
            await self._record_dedup(content_hash, graph_mtime, node_id)

        answer = {"ok": True, "symbol": symbol, "brief": brief}
        if brief:
            for key in ("callers", "callees"):
                answer[key] = dict(
                    zip(("certain", "possible", "unresolved"), parts[key])
                )
        else:
            ambiguous = parts["ambiguous"]
            shown = [c["id"] for c in parts["callers_summary"]]
            left = [c for c in parts["callers"] if c not in set(shown)]
            listed, left_out, marks = parts["callee_row"]
            answer.update(
                depth=q.flow_depth(depth),
                callers=[q.reference(c, ambiguous.get(c)) for c in shown],
                callers_more=self._counts(q, left, ambiguous),
                callees=[q.reference(c, marks.get(c)) for _, c in listed],
                callees_more=self._counts(q, left_out, marks),
                examples=[
                    {
                        "file": e["file"],
                        "line": e["lineno"],
                        "candidates": e["candidates"],
                    }
                    for e in parts["usage_examples"]
                ],
                members=[
                    {
                        "id": m["id"],
                        "name": m["name"],
                        "line": m["lineno"],
                        "tokens": m["tokens"],
                    }
                    for m in parts["impl_methods"]
                ],
                tests=parts["related_tests"],
                configs=[
                    {"id": c["id"], "line": c["lineno"], "type": c["config_type"]}
                    for c in parts["related_configs"]
                ],
                omitted=parts["omitted"],
                truncated=parts["truncated"],
            )
        return {**answer, "_parts": parts, "_request": request}

    async def context(
        self,
        node_id: str,
        brief: bool = False,
        full: bool = False,
        expand_callees: bool = False,
        deduplicate: bool = False,
        depth: int = 1,
        max_tokens: int = 2000,
        callee_budget: int = 2000,
        head_lines: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Get full context for a symbol."""
        return await self.context_text(
            await self.context_answer(
                node_id,
                brief,
                full,
                expand_callees,
                deduplicate,
                depth,
                max_tokens,
                callee_budget,
                head_lines,
                max_output_tokens,
            )
        )

    async def context_text(self, answer: dict) -> str:
        """The markdown of a `context_answer`."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        q = await self._get_querier()
        if not q:
            return ServiceError(f"ERROR: {self._no_graph_error()}")
        result = q.context_markdown(answer["_parts"])
        if answer.get("repeated"):
            return await self._format_response(
                f"{result}\n\n{answer['_dedup']}", include_header=False, max_lines=50
            )
        if answer["brief"]:
            return await self._format_response(
                result, include_header=False, max_lines=50
            )
        request = answer["_request"]
        traced = answer["depth"]
        depth_note = f" [depth={traced}]" if traced > 1 else ""
        if traced < request["asked"]:
            depth_note += f" [depth {request['asked']} is past [query] max_depth]"
        full_note = " [full=true]" if request["full"] else ""
        head_note = (
            f" [head_lines={request['head_lines']}]" if request["head_lines"] else ""
        )
        expand_note = " [expand_callees=true]" if request["expand_callees"] else ""
        ml = 2000 if (request["full"] or request["expand_callees"]) else 500
        return await self._format_response(
            result + depth_note + full_note + head_note + expand_note,
            include_header=False,
            max_lines=ml,
        )

    async def flow_answer(
        self,
        start: str,
        direction: str = "forward",
        depth: int = 3,
        target: str | None = None,
        inline_threshold: int = 100,
    ) -> dict:
        """Call flow from a starting symbol as a structured answer: the tree
        of `GraphQuerier.flow_data` without the code it inlines, which only
        `_flow`, for the text and the page, carries. `depth` is the depth
        traced, at most the project's ceiling."""
        if error := limit_error(depth, "depth"):
            return fail("invalid", error)
        q = await self._get_querier()
        if not q:
            return self._no_graph()

        start, refusal = self._one_symbol(q, start)
        if refusal:
            return refusal
        if target:
            target, refusal = self._one_symbol(q, target)
            if refusal:
                return refusal
        flow = q.flow_data(start, direction, depth, target, inline_threshold)

        def bare(node):
            return {
                **{
                    k: v
                    for k, v in node.items()
                    if k not in ("code", "lang", "children")
                },
                "kind": q.kind(node["candidates"]),
                "children": [bare(child) for child in node["children"]],
            }

        shown = {k: v for k, v in flow.items() if k not in ("root", "name")}
        return {"ok": True, **shown, "root": bare(flow["root"]), "_flow": flow}

    async def flow(
        self,
        start: str,
        direction: str = "forward",
        depth: int = 3,
        target: str | None = None,
        inline_threshold: int = 100,
    ) -> str:
        """Trace call flow from a starting symbol."""
        return await self.flow_text(
            await self.flow_answer(start, direction, depth, target, inline_threshold)
        )

    async def flow_text(self, answer: dict) -> str:
        """The markdown of a `flow_answer`, with the code it inlines."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        q = await self._get_querier()
        if not q:
            return ServiceError(f"ERROR: {self._no_graph_error()}")
        return await self._format_response(
            q.flow_markdown(answer["_flow"]), include_header=True, max_lines=300
        )

    async def search_answer(
        self,
        terms: list[str],
        limit: int = 10,
        lang: str | None = None,
        crate: str | None = None,
        symbol_type: str | None = None,
        exclude_tests: bool = False,
    ) -> dict:
        """Hybrid keyword + semantic search as a structured answer.

        The ranking rules live here so every interface ranks identically.
        `results` are `_node_ref` entries; `method` is "keyword" or "hybrid".
        """
        if error := limit_error(limit):
            return fail("invalid", error)
        if not any(term.strip() for term in terms):
            return fail("invalid", "Search terms are required.")
        if lang is not None and lang not in _LANGUAGE_VALUES:
            return fail(
                "invalid",
                f"Unknown language {lang!r}.",
            )
        if symbol_type is not None and symbol_type not in _TYPE_VALUES:
            return fail(
                "invalid",
                f"Unknown type {symbol_type!r}.",
            )
        q = await self._get_querier()
        if not q:
            return self._no_graph()

        # "all" is the web UI's wildcard; normalise it once, here.
        lang = None if lang == "all" else lang
        symbol_type = None if symbol_type == "all" else symbol_type

        tfidf_results = q.search_docs(
            terms,
            lang=lang,
            crate=crate,
            symbol_type=symbol_type,
            exclude_tests=exclude_tests,
        )[: limit * 2]

        semantic_results = []
        method = "keyword"

        if (
            self._semantic_available
            and self.config.graph_path.exists()
            and (is_natural_language_query(terms) or len(tfidf_results) < 3)
        ):
            try:
                # Single-flight async helper (runs in asyncio.to_thread
                # and respects the cache lock).
                searcher = await self._get_semantic_searcher()
                if searcher is not None:
                    # Under the filters the keyword side has
                    semantic_results = await asyncio.to_thread(
                        searcher.search,
                        " ".join(terms),
                        limit=limit * 2,
                        min_score=0.25,
                        keep=q.filtered_ids(lang, crate, symbol_type, exclude_tests),
                    )
            except Exception as e:  # noqa: BLE001 — boundary over the embedding backend; search degrades to keyword-only
                logger.warning(f"Semantic search failed, using keyword only: {e}")

        if semantic_results:
            combined = reciprocal_rank_fusion(tfidf_results, semantic_results)
            results = [node for node, _ in combined[:limit]]
            method = "hybrid"
        else:
            results = tfidf_results[:limit]

        return {
            "ok": True,
            "method": method,
            "query": " ".join(terms),
            "total": len(results),
            "results": [self._node_ref(node) for node in results],
        }

    async def search(
        self,
        terms: list[str],
        compact: bool = True,
        limit: int = 10,
        lang: str | None = None,
        crate: str | None = None,
        symbol_type: str | None = None,
        exclude_tests: bool = False,
    ) -> str:
        """Search symbol names and docstrings."""
        answer = await self.search_answer(
            terms,
            limit=limit,
            lang=lang,
            crate=crate,
            symbol_type=symbol_type,
            exclude_tests=exclude_tests,
        )
        return await self.search_text(
            answer, terms, compact, lang, crate, symbol_type, exclude_tests
        )

    async def search_text(
        self,
        answer: dict,
        terms: list[str],
        compact: bool = True,
        lang: str | None = None,
        crate: str | None = None,
        symbol_type: str | None = None,
        exclude_tests: bool = False,
    ) -> str:
        """The markdown of a `search_answer`, compact or with each result's
        signature and docstring."""
        if not answer["ok"]:
            return ServiceError(
                await self._format_response(
                    f"ERROR: {answer['error']}", include_header=True
                )
            )

        filters = []
        if lang and lang != "all":
            filters.append(f"lang={lang}")
        if crate:
            filters.append(f"crate={crate}")
        if symbol_type and symbol_type != "all":
            filters.append(f"type={symbol_type}")
        if exclude_tests:
            filters.append("exclude_tests")
        if compact:
            filters.append("compact")
        filter_note = f" [{', '.join(filters)}]" if filters else ""

        results = await self._nodes(answer["results"])
        search_method = answer["method"]

        if not results:
            result = f"No matches for '{' '.join(terms)}'{filter_note}."
        elif compact:
            lines = [
                f"{len(results)} match(es) for '{' '.join(terms)}' [{search_method}]{filter_note}:"
            ]
            for i, node in enumerate(results, 1):
                lines.append(format_compact_result(node, rank=i))
            lines.append("")
            lines.append("Use `descry context` with node ID for details.")
            result = "\n".join(lines)
        else:
            lines = [
                f"{len(results)} match(es) for '{' '.join(terms)}' [{search_method}]{filter_note}:\n"
            ]
            for node in results:
                lines.append(format_search_result(node))
                lines.append("")
            result = "\n".join(lines).rstrip()

        return await self._format_response(result)

    _STRUCTURE_KINDS = ("Constant", "Class", "Function", "Configuration")

    async def structure_answer(self, filename: str) -> dict:
        """A file's imports and definitions as a structured answer.

        `found` is False when no File node matches; a name several files
        share is refused with their node IDs. `imports` is the sorted
        module names; `definitions` maps "constant", "class", "function" and
        "configuration" to the file's definitions of that kind, each
        `{id, name, line}` in line order.
        """
        if not filename.strip():
            return fail("invalid", "A file name is required.")
        q = await self._get_querier()
        if not q:
            return self._no_graph()

        file_matches = [
            m for m in q.find_nodes_by_name(filename) if m["type"] == "File"
        ]
        if not file_matches:
            return {
                "ok": True,
                "found": False,
                "file": None,
                "imports": [],
                "definitions": {},
            }

        if len(file_matches) > 1:
            return self._refusal(filename, [m["id"] for m in file_matches])
        node_id = file_matches[0]["id"]
        definitions = defaultdict(list)
        imports = set()
        for edge in q.outgoing[node_id]:
            if edge["relation"] == "DEFINES":
                target = q.nodes.get(edge["target"])
                if target and target["type"] in self._STRUCTURE_KINDS:
                    meta = target["metadata"]
                    definitions[target["type"].lower()].append(
                        {
                            "id": target["id"],
                            "name": meta.get("name", ""),
                            "line": meta.get("lineno"),
                        }
                    )
            elif edge["relation"] == "IMPORTS":
                imports.add(edge["target"].removeprefix("MODULE:"))

        return {
            "ok": True,
            "found": True,
            "file": node_id,
            "imports": sorted(imports),
            "definitions": {
                kind.lower(): sorted(
                    definitions[kind.lower()], key=lambda d: d["line"] or 0
                )
                for kind in self._STRUCTURE_KINDS
                if kind.lower() in definitions
            },
        }

    async def structure(self, filename: str) -> str:
        """Show file structure."""
        return await self.structure_text(
            await self.structure_answer(filename), filename
        )

    async def structure_text(self, answer: dict, filename: str) -> str:
        """The markdown of a `structure_answer`."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        if not answer["found"]:
            return f"File '{filename}' not found. Check spelling."

        lines = [f"{answer['file']}:"]
        defs = answer["definitions"]
        sorted_imports = answer["imports"]

        if sorted_imports:
            lines.append(f"  Imports: {', '.join(sorted_imports[:10])}")
            if len(sorted_imports) > 10:
                lines.append(f"           (+{len(sorted_imports) - 10} more)")

        for type_name, kind in [
            ("Const", "constant"),
            ("Class", "class"),
            ("Fn", "function"),
        ]:
            items = sorted(d["name"] for d in defs.get(kind, []))
            if items:
                lines.append(f"  {type_name}: {', '.join(items[:15])}")
                if len(items) > 15:
                    lines.append(f"       (+{len(items) - 15} more)")

        if configs := defs.get("configuration"):
            q = await self._get_querier()
            lines.append("  Config:")
            for cfg in configs:
                meta = ((q.nodes.get(cfg["id"]) if q else None) or {}).get(
                    "metadata", {}
                )
                lines.append(
                    f"    L{cfg['line'] or '?'}: {meta.get('signature', cfg['name'] or '?')}"
                )

        return await self._format_response("\n".join(lines))

    async def flatten_answer(self, symbol: str) -> dict:
        """A class's effective API, inherited methods included, as a
        structured answer: `class`, `name`, `chain` and `methods` as
        `GraphQuerier.flatten_data` gives them."""
        q = await self._get_querier()
        if not q:
            return self._no_graph()
        node_id, refusal = self._one_symbol(q, symbol)
        if refusal:
            return refusal
        data = q.flatten_data(node_id)
        return {"ok": "error" not in data, **data}

    async def flatten(self, class_node_id: str) -> str:
        """Show effective API of a class including inherited methods."""
        return await self.flatten_text(await self.flatten_answer(class_node_id))

    async def flatten_text(self, answer: dict) -> str:
        """The markdown of a `flatten_answer`."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        q = await self._get_querier()
        if not q:
            return ServiceError(f"ERROR: {self._no_graph_error()}")
        return await self._format_response(
            q.flatten_markdown(answer), include_header=False, max_lines=150
        )

    async def semantic(self, query: str, limit: int = 10) -> str:
        """Pure semantic search using embeddings only."""
        return await self.semantic_text(await self.semantic_answer(query, limit), query)

    async def semantic_text(self, answer: dict, query: str) -> str:
        """The markdown of a `semantic_answer`."""
        if not answer["ok"]:
            if answer["reason"] == "no_graph":
                return f"ERROR: {answer['error']}"
            return refused(answer)
        results = answer["results"]

        if not results:
            result = f"No semantic matches for '{query}'."
        else:
            lines = [f"{len(results)} semantic match(es) for '{query}':\n"]
            nodes = await self._nodes(results)
            for node, ref in zip(nodes, results, strict=True):
                lines.append(
                    format_search_result(node, show_score=True, score=ref["score"])
                )
                lines.append("")
            result = "\n".join(lines).rstrip()

        return await self._format_response(result)

    async def quick_answer(
        self, name: str, full: bool = False, brief: bool = False
    ) -> dict:
        """The symbol a node ID or a name names, as a `_node_ref`, with its
        context's parts in `_parts` for the text; refused, as `_refusal`
        says, for a name of no definition or of several, and `not_found`
        when its source is gone."""
        q = await self._get_querier()
        if not q:
            return self._no_graph()
        node_id, refusal = self._one_symbol(q, name)
        if refusal:
            return refusal
        parts = q.context_data(node_id, full=full, brief=brief)
        if "error" in parts:
            return fail("not_found", parts["error"])
        return {"ok": True, "symbol": self._node_ref(q.nodes[node_id]), "_parts": parts}

    async def quick(self, name: str, full: bool = False, brief: bool = False) -> str:
        """Find symbol and show full context in one step."""
        return await self.quick_text(
            await self.quick_answer(name, full, brief), name, full, brief
        )

    async def quick_text(
        self, answer: dict, name: str, full: bool = False, brief: bool = False
    ) -> str:
        """The markdown of a `quick_answer`: the symbol and its context."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        q = await self._get_querier()
        if not q:
            return ServiceError(f"ERROR: {self._no_graph_error()}")
        context = q.context_markdown(answer["_parts"])
        if brief:
            return await self._format_response(
                context, include_header=False, max_lines=50
            )

        header_lines = [
            f"### Quick Context for `{name}`",
            "",
            format_search_result(q.nodes[answer["symbol"]["id"]]),
            "",
        ]

        result = "\n".join(header_lines) + "\n" + context
        if full:
            result += "\n [full=true]"

        ml = 2000 if full else 500
        return await self._format_response(result, include_header=False, max_lines=ml)

    async def semantic_answer(self, query: str, limit: int = 10) -> dict:
        """Pure semantic search as a structured answer: `results` are
        `_node_ref` entries, each with its `score`.

        `error` says which step failed, so a model that would not load is not
        reported as a missing graph; `reason` classifies it as "invalid",
        "unavailable", "no_graph", "loading" or "failed".
        """
        if refused := limit_error(limit):
            reason, error = "invalid", refused
        elif not query.strip():
            reason, error = "invalid", "A query is required."
        elif not self._semantic_available:
            reason, error = (
                "unavailable",
                _EMBEDDINGS_DISABLED
                if not self.config.enable_embeddings
                else (
                    "Semantic search not available. Install dependencies:\n"
                    "  pip install sentence-transformers numpy"
                ),
            )
        elif not self.config.graph_path.exists():
            reason, error = "no_graph", _GRAPH_NOT_FOUND
        elif (searcher := await self._get_semantic_searcher()) is None:
            # A failed load has set the error, the kind of what it raised, and
            # a load that succeeds clears it.
            load_error = self._semantic_cache["error"]
            if load_error is None:
                reason, error = "loading", "Embeddings are loading. Try again shortly."
            else:
                # The model is not there to use: gated, missing or broken
                reason = "unavailable"
                error = (
                    f"Failed to initialize semantic search ({load_error}); "
                    "the log says why."
                )
        else:
            try:
                results = await asyncio.to_thread(searcher.search, query, limit=limit)
            except Exception as e:  # noqa: BLE001 — boundary over the embedding backend; reported to the caller
                # A model loaded at the first search may fail as a load does:
                # its message, which may quote the model's files, is logged.
                logger.warning("Semantic search failed: %s", e)
                reason = "failed"
                error = (
                    f"Semantic search failed ({type(e).__name__}); the log says why."
                )
            else:
                return {
                    "ok": True,
                    "query": query,
                    "total": len(results),
                    "results": [
                        {**self._node_ref(node), "score": round(score, 4)}
                        for node, score in results
                    ],
                }
        return fail(reason, error)

    async def examples_data(self) -> dict:
        """Symbols of this graph worth trying each tool on, as data.

        `examples` maps a tool to names that give it something to show: the
        most-called functions, a class, a function with a wide call tree.
        It is None for a graph without nodes.
        """
        q = await self._get_querier()
        if not q:
            return self._no_graph()

        nodes = q.data.get("nodes", [])
        if not nodes:
            return {"ok": True, "examples": None}

        edges = q.data.get("edges", [])

        functions = []
        classes = []
        methods = []
        for n in nodes:
            meta = n.get("metadata", {})
            ntype = n.get("type", "")
            name = meta.get("name", "")
            # Skip test nodes and private/dunder names
            if not name or name.startswith("_") or q.is_test(n["id"]):
                continue
            entry = {
                "name": name,
                "id": n["id"],
                "type": ntype,
                "in_degree": meta.get("in_degree", 0),
                "token_count": meta.get("token_count", 0),
                "file": file_of(n["id"]) if "::" in n["id"] else "",
            }
            if ntype == "Function":
                functions.append(entry)
            elif ntype in ("Class", "Struct"):
                classes.append(entry)
            elif ntype == "Method":
                methods.append(entry)

        functions.sort(key=lambda x: x["in_degree"], reverse=True)
        methods.sort(key=lambda x: x["in_degree"], reverse=True)

        out_degree = {}
        for e in edges:
            src = e.get("source", "")
            out_degree[src] = out_degree.get(src, 0) + 1

        # Pick "interesting" symbols for each tool
        # Search: a popular function, a class, and a method
        search_examples = []
        if functions:
            search_examples.append(functions[0]["name"])
        if classes:
            search_examples.append(classes[0]["name"])
        if methods and len(search_examples) < 3:
            search_examples.append(methods[0]["name"])

        # Quick/callers/callees: functions with high in-degree (many callers)
        popular = [f for f in functions if f["in_degree"] >= 2][:3]
        popular_names = [f["name"] for f in popular] or [
            f["name"] for f in functions[:3]
        ]

        # Flow: find a function with good fanout (interesting forward trace)
        flow_candidates = []
        for f in functions[:20]:
            od = out_degree.get(f["id"], 0)
            if od >= 2:
                flow_candidates.append({"name": f["name"], "out_degree": od})
        flow_candidates.sort(key=lambda x: x["out_degree"], reverse=True)
        flow_fwd = (
            flow_candidates[0]["name"]
            if flow_candidates
            else (popular_names[0] if popular_names else "")
        )
        # Backward: pick a popular callee that differs from the forward symbol
        flow_bwd = ""
        for pn in popular_names:
            if pn != flow_fwd:
                flow_bwd = pn
                break
        if not flow_bwd:
            flow_bwd = popular_names[0] if popular_names else ""

        # Path: pick two symbols likely connected
        path_start = flow_fwd
        path_end = popular_names[0] if popular_names else ""
        if path_start == path_end and len(popular_names) > 1:
            path_end = popular_names[1]

        # Structure: pick files that have multiple symbols
        file_counts = {}
        for n in nodes:
            nid = n.get("id", "")
            if "::" in nid:
                f = file_of(nid)
                file_counts[f] = file_counts.get(f, 0) + 1
        structure_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        structure_examples = [f for f, _ in structure_files[:3]]

        flatten_example = classes[0]["id"] if classes else ""

        # Impls: pick method names that might have multiple implementations
        seen_names = {}
        for m in methods:
            n = m["name"]
            seen_names[n] = seen_names.get(n, 0) + 1
        impl_candidates = sorted(
            [(name, count) for name, count in seen_names.items() if count >= 2],
            key=lambda x: x[1],
            reverse=True,
        )
        impl_examples = [name for name, _ in impl_candidates[:3]]

        return {
            "ok": True,
            "examples": {
                "search": search_examples,
                "quick": popular_names[:3],
                "callers": popular_names[:3],
                "callees": popular_names[:2],
                "flow_forward": flow_fwd,
                "flow_backward": flow_bwd,
                "path_start": path_start,
                "path_end": path_end,
                "structure": structure_examples,
                "flatten": flatten_example,
                "impls": impl_examples,
                "evolution": popular_names[:2],
            },
        }

    async def index_answer(self, path: str = ".") -> dict:
        """`index` as a structured answer: the graph's status after the run.
        The indexer's report is `_report`, for the text."""
        report = await self.index(path)
        if isinstance(report, ServiceError):
            return failure(report)
        return {"ok": True, **public(await self.status_answer()), "_report": report}

    async def index_text(self, answer: dict) -> str:
        """The text of an `index_answer`: the indexer's report."""
        if not answer["ok"]:
            return refused(answer)
        return answer["_report"]

    async def index_stream(self):
        """Regenerate the graph, yielding the indexer's output as it arrives.

        Yields `{"type": "start", "message": str}`, `{"type": "output",
        "line": str}` per line, then one `{"type": "done", "success": bool,
        "message": str}`. Like `index()` it honors `[timeouts] index_minutes`
        and reloads the caches, and is refused, with only that `done`, while
        another run is in progress. The indexer is killed if the run times
        out or the generator is closed early.
        """
        # Read before reset_caches() below discards it.
        model_was_loaded = self._semantic_cache["instance"] is not None
        minutes = self.config.index_timeout_minutes
        loop = asyncio.get_running_loop()
        deadline = loop.time() + minutes * 60 if minutes else None

        def remaining() -> float | None:
            return None if deadline is None else max(deadline - loop.time(), 0)

        if error := self.config.cache_dir_error():
            yield {"type": "done", "success": False, "message": error}
            return
        argv, env = self._indexer_child(str(self.config.project_root))
        if not self._index_lock.acquire(blocking=False):
            yield {"type": "done", "success": False, "message": _INDEX_BUSY}
            return
        try:
            yield {"type": "start", "message": "Starting reindex..."}
            async with _process_tree(
                argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.config.project_root),
                env={**env, "PYTHONUNBUFFERED": "1"},
                limit=10 * 1024 * 1024,  # the indexer can emit very long lines
            ) as proc:
                while line := await asyncio.wait_for(
                    proc.stdout.readline(), remaining()
                ):
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        yield {"type": "output", "line": text}
                returncode = await asyncio.wait_for(proc.wait(), remaining())
        except TimeoutError:
            yield {
                "type": "done",
                "success": False,
                "message": self._index_timeout_message(),
            }
        # OSError: the indexer could not be started or read from.
        # ValueError: it printed a line longer than `limit`.
        except (OSError, ValueError) as e:
            yield {"type": "done", "success": False, "message": f"Index error: {e}"}
        else:
            if returncode != 0:
                yield {
                    "type": "done",
                    "success": False,
                    "message": f"Index failed: the indexer exited with code {returncode}.",
                }
                return

            self.reset_caches()
            if embeddings := await self._load_indexed_embeddings(model_was_loaded):
                yield {"type": "output", "line": embeddings}
            yield {"type": "done", "success": True, "message": "Index complete."}
        finally:
            self._index_lock.release()

    async def impls_answer(self, method: str, trait_name: str | None = None) -> dict:
        """The implementations of a trait method as a structured answer: each
        by ID, with the type it is implemented for, the trait, file and line."""
        if not method.strip():
            return fail("invalid", "A method name is required.")
        q = await self._get_querier()
        if not q:
            return self._no_graph()
        found = []
        for node in q.find_trait_impls(method, trait_name):
            node_id = node["id"]
            parts = node_id.split("::")
            found.append(
                {
                    "id": node_id,
                    "type": parts[-2] if len(parts) >= 2 else None,
                    "trait": node.get("metadata", {}).get("trait_impl"),
                    "file": parts[0].removeprefix("FILE:"),
                    "line": node.get("metadata", {}).get("lineno"),
                }
            )
        return {
            "ok": True,
            "method": method,
            "trait": trait_name,
            "total": len(found),
            "implementations": found,
        }

    async def impls(self, method: str, trait_name: str | None = None) -> str:
        """Find all implementations of a trait method."""
        return await self.impls_text(await self.impls_answer(method, trait_name))

    async def impls_text(self, answer: dict) -> str:
        """The markdown of an `impls_answer`, with each one's signature."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        method, trait_name = answer["method"], answer["trait"]
        found = answer["implementations"]
        if not found:
            trait_filter = f" for trait '{trait_name}'" if trait_name else ""
            return (
                f"No trait implementations found for method '{method}'{trait_filter}. "
                "Trait implementations are recorded for Rust code only."
            )

        by_trait = defaultdict(list)
        for impl in found:
            by_trait[impl["trait"] or "unknown"].append(impl)

        q = await self._get_querier()
        lines = [f"### Implementations of `{method}`"]
        if trait_name:
            lines.append(f"*Filtered to trait: {trait_name}*")
        lines.append("")
        lines.append(f"Found {len(found)} implementation(s):\n")

        for trait, impls in sorted(by_trait.items()):
            lines.append(f"**{trait}** ({len(impls)} impl):")
            for impl in impls:
                meta = ((q.nodes.get(impl["id"]) if q else None) or {}).get(
                    "metadata", {}
                )
                lines.append(f"  - **{impl['type'] or '?'}**::{method}")
                lines.append(f"    {impl['file']}:{impl['line'] or '?'}")
                lines.append(f"    `{meta.get('signature', f'fn {method}(...)')}`")
            lines.append("")

        return await self._format_response("\n".join(lines))

    async def path_answer(
        self,
        start: str,
        end: str,
        max_depth: int = 10,
        direction: str = "forward",
    ) -> dict:
        """The shortest call path between two symbols as a structured answer:
        how the search ended (`status`: "found", "not_found" or "timeout"),
        the two symbols by ID, and each hop's caller and
        callee with the file and line of the call."""
        if error := limit_error(max_depth, "max_depth"):
            return fail("invalid", error)
        q = await self._get_querier()
        if not q:
            return self._no_graph()

        ends = [self._one_symbol(q, symbol) for symbol in (start, end)]
        if refusal := next((r for _, r in ends if r), None):
            return refusal
        found = q.find_call_path_result(
            ends[0][0], ends[1][0], max_depth=max_depth, direction=direction
        )
        return {
            "ok": True,
            "status": found["status"],
            "start": ends[0][0],
            "end": ends[1][0],
            "direction": direction,
            "max_depth": found["max_depth"],
            "hops": [
                {
                    "caller": q.reference(hop["caller_id"], None),
                    "callee": q.reference(hop["callee_id"], hop.get("candidates")),
                    "file": hop["file_path"],
                    "line": hop["call_line"],
                }
                for hop in found["path"]
            ],
        }

    async def path(
        self,
        start: str,
        end: str,
        max_depth: int = 10,
        direction: str = "forward",
    ) -> str:
        """Find shortest call path between two symbols."""
        return await self.path_text(
            await self.path_answer(start, end, max_depth, direction), start, end
        )

    async def path_text(self, answer: dict, start: str, end: str) -> str:
        """The markdown of a `path_answer`, with the code at each call."""
        if not answer["ok"]:
            return ServiceError(f"ERROR: {answer['error']}")
        status = answer["status"]
        if status == "timeout":
            return (
                f"Path search from '{start}' to '{end}' stopped at the "
                f"{self.config.query_timeout_ms} ms time limit ([timeouts] "
                f"query_ms) before finishing; a path may still exist."
            )
        if status == "not_found":
            relation = "calls" if answer["direction"] == "forward" else "is called by"
            return (
                f"No path found from '{start}' to '{end}' within {answer['max_depth']} hops.\n\n"
                f"Direction: {answer['direction']} ('{start}' {relation} '{end}')\n\n"
                f"Try:\n"
                f"- Verify both symbols exist with descry search\n"
                f"- Increase max_depth if the path is longer\n"
                f"- Try the opposite direction"
            )
        q = await self._get_querier()
        hops = answer["hops"]
        lines = [
            f"### Call Path: `{start}` -> `{end}` ({len(hops)} hop{'s' if len(hops) != 1 else ''})\n"
        ]
        for i, hop in enumerate(hops, 1):
            file_path, call_line = hop["file"], hop["line"]
            lines.append(
                f"**{i}. {hop['caller']['name']}** -> **{hop['callee']['name']}**"
                f"{GraphQuerier.one_of(hop['callee'].get('candidates'))}"
            )
            lines.append(
                f"   {file_path}:{call_line}" if call_line else f"   {file_path}"
            )
            if snippet := q.call_snippet(file_path, call_line) if q else "":
                lang = _get_syntax_lang(file_path, self.config.syntax_lang_map)
                lines.append(f"```{lang}")
                lines.extend(f"   {line}" for line in snippet.split("\n"))
                lines.append("```")
            lines.append("")

        return await self._format_response("\n".join(lines), include_header=True)

    def _openapi_spec(self) -> Path | None:
        """The OpenAPI spec to trace against, or None if the project has none.

        That is `[cross_lang] openapi_path`. A project that leaves it unset
        may instead keep its spec at `public/api/latest.json` or
        `public/api/openapi.json` under the root.
        """
        if self.config.openapi_path is not None:
            return self.config.openapi_path
        from descry.query import source_path

        for name in ("latest.json", "openapi.json"):
            if path := source_path(self.config.project_root, f"public/api/{name}"):
                return Path(path)
        return None

    def cross_lang_error(self) -> str | None:
        """Why cross-language tracing cannot run here, or None if it can."""
        if not self._CrossLangTracer:
            return "Cross-language tracing not available. Missing cross_lang module."
        openapi_path = self._openapi_spec()
        if openapi_path is None:
            return "Cross-language tracing not configured. Set openapi_path in config."
        if not openapi_path.is_file():
            return f"OpenAPI spec not found at {openapi_path}."
        return None

    async def cross_lang_answer(
        self,
        mode: str = "endpoint",
        method: str | None = None,
        path: str | None = None,
        tag: str | None = None,
    ) -> dict:
        """Cross-language tracing as a structured answer: for mode "stats"
        `total_endpoints` and `linked_to_graph`; for "list" `tag`, `total`
        and `endpoints`; for "endpoint" `endpoint`, None when none matches.
        An endpoint is `{method, path, operation_id, tags, node_id}`. The
        tracer's own result, summaries included, is `_result`, for the text.
        On failure `reason` is "unavailable" when tracing cannot run here
        and "invalid" when the arguments are wrong.
        """
        if method is not None and method.upper() not in HTTP_METHODS:
            return fail(
                "invalid",
                f"Unknown HTTP method {method!r}; use one of "
                f"{', '.join(sorted(HTTP_METHODS))}.",
            )
        error = self.cross_lang_error()
        if error:
            return fail("unavailable", error)

        graph_path = (
            str(self.config.graph_path) if self.config.graph_path.exists() else None
        )
        querier = graph_path and await self._get_querier()
        if graph_path and querier is None:
            return self._no_graph()
        tracer = self._CrossLangTracer(
            str(self._openapi_spec()),
            graph_path,
            backend_handler_patterns=self.config.backend_handler_patterns or None,
            api_prefixes=self.config.api_prefixes or None,
            is_test=querier.is_test if querier else None,
            project_root=self.config.project_root,
            max_file_bytes=self.config.max_file_bytes,
        )

        if mode == "stats":
            result = tracer.get_stats()
        elif mode == "list":
            result = tracer.list_endpoints(tag=tag)
        elif mode == "endpoint":
            if not method or not path:
                return fail(
                    "invalid",
                    (
                        "Endpoint mode requires 'method' and 'path' arguments.\n"
                        "Example: method='GET', path='/api/v1/deployments'"
                    ),
                )
            result = tracer.get_handler_info(method.upper(), path)
        else:
            return fail(
                "invalid",
                f"Unknown mode '{mode}'. Use 'endpoint', 'list', or 'stats'.",
            )

        def endpoint(info):
            return {
                "method": info["method"],
                "path": info["path"],
                "operation_id": info["operationId"],
                "tags": info.get("tags", []),
                "node_id": info.get("node_id"),
            }

        if mode == "stats":
            shown = {k: result[k] for k in ("total_endpoints", "linked_to_graph")}
        elif mode == "list":
            shown = {
                "tag": tag,
                "total": len(result),
                "endpoints": [endpoint(e) for e in result],
            }
        else:
            shown = {"endpoint": endpoint(result) if result else None}
        return {"ok": True, "mode": mode, **shown, "_result": result}

    async def cross_lang(
        self,
        mode: str = "endpoint",
        method: str | None = None,
        path: str | None = None,
        tag: str | None = None,
    ) -> str:
        """Trace API calls from frontend to backend handlers via OpenAPI spec."""
        return await self.cross_lang_text(
            await self.cross_lang_answer(mode, method, path, tag), method, path, tag
        )

    async def cross_lang_text(
        self, answer: dict, method: str | None, path: str | None, tag: str | None
    ) -> str:
        """The markdown of a `cross_lang_answer`."""
        if not answer["ok"]:
            return refused(answer)
        mode = answer["mode"]

        if mode == "stats":
            stats = answer["_result"]
            lines = [
                "### Cross-Language Tracing Stats",
                "",
                f"**OpenAPI Spec**: {stats['openapi_path']}",
                f"**Total Endpoints**: {stats['total_endpoints']}",
                f"**Linked to Graph**: {stats['linked_to_graph']}",
                "",
                "Use mode='list' to see endpoints by tag.",
                "Use mode='endpoint' with method/path to lookup a specific handler.",
            ]
            return "\n".join(lines)

        elif mode == "list":
            endpoints = answer["_result"]
            if not endpoints:
                tag_note = f" for tag '{tag}'" if tag else ""
                return f"No endpoints found{tag_note}."

            tag_note = f" (tag: {tag})" if tag else ""
            lines = [f"### API Endpoints{tag_note}", ""]

            current_path = None
            for ep in endpoints:
                if ep["path"] != current_path:
                    if current_path is not None:
                        lines.append("")
                    current_path = ep["path"]
                    lines.append(f"**{ep['path']}**")

                handler = ep["operationId"]
                node_id = ep.get("node_id")
                summary = ep.get("summary", "")

                if node_id:
                    file_info = file_of(node_id)
                    lines.append(f"  {ep['method']:6s} -> `{handler}` ({file_info})")
                else:
                    lines.append(f"  {ep['method']:6s} -> `{handler}` (not in graph)")

                if summary:
                    lines.append(
                        f"           {summary[:60]}{'...' if len(summary) > 60 else ''}"
                    )

            lines.append("")
            lines.append(f"*{len(endpoints)} endpoint(s) total*")
            return "\n".join(lines)

        else:
            info = answer["_result"]
            if not info:
                return (
                    f"No handler found for {method.upper()} {path}.\n\n"
                    f"Try listing endpoints with: mode='list', tag='<resource>'"
                )

            lines = [
                f"### Handler for {method.upper()} {info['path']}",
                "",
                f"**Operation ID**: `{info['operationId']}`",
                f"**Tags**: {', '.join(info.get('tags', [])) or 'none'}",
            ]

            if info.get("summary"):
                lines.append(f"**Summary**: {info['summary']}")

            node_id = info.get("node_id")
            if node_id:
                lines.append("")
                lines.append(f"**Graph Node**: `{node_id}`")

                q = await self._get_querier()
                if q:
                    node_info = q.nodes.get(node_id)
                    if node_info:
                        meta = node_info.get("metadata", {})
                        if meta.get("lineno"):
                            file_path = file_of(node_id)
                            lines.append(f"**Location**: {file_path}:{meta['lineno']}")
                        if meta.get("signature"):
                            lines.append(f"**Signature**: `{meta['signature']}`")

                        lines.append("")
                        lines.append(
                            "Use `descry context` with this node_id for full details."
                        )
            else:
                lines.append("")
                lines.append(
                    "*Handler not found in graph - may need to run descry index*"
                )

            return "\n".join(lines)

    async def _git_history(self, tool: str, method: str, **kwargs) -> dict:
        """Run a GitHistoryAnalyzer method off the event loop, as data.

        `result` is what the method returned. On failure `reason` is
        "invalid" for a `limit` below 1, "unavailable" when the git history
        module is missing and "failed" when the analysis raised.
        """
        if error := limit_error(kwargs["limit"]):
            return fail("invalid", error)
        if not self._git_available:
            return fail(
                "unavailable",
                "Git history tools not available. Missing git_history module.",
            )

        try:
            analyzer = await self._get_git_analyzer()
            result = await asyncio.to_thread(getattr(analyzer, method), **kwargs)
        except self._GitError as e:
            return failure(e)
        # Tool boundary over git output and graph lookups: whatever the
        # analysis raises is logged with its traceback and reported.
        except Exception as e:
            logger.exception("Error in %s", tool)
            return fail(
                "failed",
                f"Error analyzing {tool}: {e}",
            )
        return {"ok": True, "result": result}

    @staticmethod
    def _git_answer(data: dict) -> dict:
        """A git-history result as an answer: a failure where the analysis
        refused its arguments or failed (`error`, `reason`)."""
        if not data["ok"]:
            return data
        result = data["result"]
        if "error" in result:
            return result
        return {"ok": True, **result}

    async def churn_answer(
        self,
        time_range: str | None = None,
        path_filter: str | None = None,
        limit: int = 20,
        mode: str = "symbols",
        exclude_generated: bool = True,
    ) -> dict:
        """Code churn hotspots as a structured answer: the result of
        `GitHistoryAnalyzer.get_churn_structured`, whose `message`, when set,
        says why there is nothing to list."""
        data = await self._git_history(
            "churn",
            "get_churn_structured",
            time_range=time_range,
            path_filter=path_filter,
            limit=limit,
            mode=mode,
            exclude_generated=exclude_generated,
        )
        return self._git_answer(data)

    async def churn(
        self,
        time_range: str | None = None,
        path_filter: str | None = None,
        limit: int = 20,
        mode: str = "symbols",
        exclude_generated: bool = True,
    ) -> str:
        """Find code churn hotspots."""
        return await self.churn_text(
            await self.churn_answer(
                time_range, path_filter, limit, mode, exclude_generated
            )
        )

    async def churn_text(self, answer: dict) -> str:
        """The markdown of a `churn_answer`."""
        if not answer["ok"]:
            return refused(answer)
        return await self._format_response(
            self._GitHistoryAnalyzer.render_churn(answer)
        )

    async def evolution_answer(
        self,
        name: str,
        time_range: str | None = None,
        limit: int = 10,
        show_diff: bool = False,
        crate: str | None = None,
    ) -> dict:
        """How a symbol has changed over time as a structured answer: the
        symbol, how many definitions its name fits, and each commit's hash,
        date and line counts. A name of no definition, or of several, is
        refused, as `_refusal` says. Authors, subjects and
        diffs are only in `_log`, for the text."""
        if not limit_error(limit):
            q = await self._get_querier()
            if q is None:
                return self._no_graph()
            node_id, refusal = self._one_symbol(q, name, crate)
            if refusal:
                return refusal
            name = node_id
        data = await self._git_history(
            "evolution",
            "get_evolution_structured",
            name=name,
            time_range=time_range,
            limit=limit,
            show_diff=show_diff,
        )
        if not (found := self._git_answer(data))["ok"]:
            return found
        log = data["result"]
        answer = {
            k: log[k]
            for k in ("message", "node_id", "mode", "truncation_note")
            if k in log
        }
        if "commits" in log:
            answer["authors"] = len({c["author"] for c in log["commits"]})
            answer["commits"] = [
                {k: c[k] for k in ("hash", "date", "added", "removed") if k in c}
                for c in log["commits"]
            ]
        return {"ok": True, **answer, "_log": log}

    async def evolution(
        self,
        name: str,
        time_range: str | None = None,
        limit: int = 10,
        show_diff: bool = False,
        crate: str | None = None,
    ) -> str:
        """Track how a symbol has changed over time."""
        return await self.evolution_text(
            await self.evolution_answer(name, time_range, limit, show_diff, crate)
        )

    async def evolution_text(self, answer: dict) -> str:
        """The markdown of an `evolution_answer`."""
        if not answer["ok"]:
            return refused(answer)
        return await self._format_response(
            self._GitHistoryAnalyzer.render_evolution(answer["_log"])
        )

    async def changes_answer(
        self,
        commit_range: str | None = None,
        time_range: str | None = None,
        path_filter: str | None = None,
        show_callers: bool = True,
        limit: int = 50,
    ) -> dict:
        """Change impact for a commit range as a structured answer: the
        result of `GitHistoryAnalyzer.get_changes_structured`, whose
        `message`, when set, says why there is nothing to report."""
        data = await self._git_history(
            "changes",
            "get_changes_structured",
            commit_range=commit_range,
            time_range=time_range,
            path_filter=path_filter,
            show_callers=show_callers,
            limit=limit,
        )
        return self._git_answer(data)

    async def changes(
        self,
        commit_range: str | None = None,
        time_range: str | None = None,
        path_filter: str | None = None,
        show_callers: bool = True,
        limit: int = 50,
    ) -> str:
        """Analyze change impact for a commit range."""
        return await self.changes_text(
            await self.changes_answer(
                commit_range, time_range, path_filter, show_callers, limit
            )
        )

    async def changes_text(self, answer: dict) -> str:
        """The markdown of a `changes_answer`."""
        if not answer["ok"]:
            return refused(answer)
        return await self._format_response(
            self._GitHistoryAnalyzer.render_changes(answer)
        )
