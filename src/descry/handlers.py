"""DescryService — Core business logic for codebase knowledge graph tools.

No MCP dependency. All tool logic lives here; MCP/CLI/Web are thin wrappers.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from descry._env import safe_env

logger = logging.getLogger(__name__)

# Server version for tracking
SERVER_VERSION = "0.1.0"


# --- Configuration ---


_DEFAULT_PROJECT_MARKERS = [
    ".git",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    ".descry.toml",
]
_DEFAULT_API_PREFIXES = ["/api/v1", "/api/v2", "/api"]
_DEFAULT_EXCLUDED_DIRS = {
    "target",
    "node_modules",
    "dist",
    "docs",
    ".git",
    "__pycache__",
    "build",
    "vendor",
}

_DEFAULT_TEST_PATH_PATTERNS = (
    "/tests/",
    "/test/",
    "/_test/",
    "/spec/",
    "/testing/",
    "/fixtures/",
    "/mocks/",
    "/__tests__/",
)
_DEFAULT_TEST_FILE_SUFFIXES = (
    "_test.rs",
    ".test.ts",
    ".spec.ts",
    "_test.py",
    ".test.js",
    ".spec.js",
    ".test.tsx",
    ".spec.tsx",
)
_DEFAULT_CODE_EXTENSIONS = {
    ".rs",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".svelte",
    ".go",
    ".java",
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
    ".js": "javascript",
    ".jsx": "javascript",
    ".svelte": "svelte",
    ".proto": "protobuf",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".rb": "ruby",
    ".go": "go",
    ".java": "java",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
}


def _env(key: str, default: str = "") -> str:
    """Read env var with default."""
    return os.environ.get(key, default)


_TOOLCHAIN_REGEX = re.compile(r"^[A-Za-z0-9._\-]+$")
_MODEL_HF_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]*$")
_SHELL_METACHARS = set("\x00;|&`$\n\r\t<>")


def _validate_toolchain(value: str) -> str:
    """Validate scip_rust_toolchain TOML value. Raises ValueError on bad input."""
    if not value or value.startswith("-") or not _TOOLCHAIN_REGEX.match(value):
        raise ValueError(f"Invalid scip.rust.toolchain: {value!r}")
    return value


def _validate_scip_extra_arg(arg: str) -> str:
    """Validate a scip_extra_args entry.

    Accepts bare positional args; flags must start with '--' (blocks short-flag
    smuggling). Rejects shell metacharacters.
    """
    if not arg:
        raise ValueError("Empty scip extra_args entry")
    if arg.startswith("-") and not arg.startswith("--"):
        raise ValueError(f"scip_extra_args short flag not allowed: {arg!r}")
    if any(c in _SHELL_METACHARS for c in arg):
        raise ValueError(f"scip_extra_args contains shell metacharacters: {arg!r}")
    return arg


def _validate_embedding_model(value: str, project_root: Path) -> str:
    """Validate [embeddings] model TOML value.

    - If looks like a local path (starts with '/' or contains '..'), must
      resolve inside project_root.
    - Otherwise must match HuggingFace repo-id regex.
    """
    if not value:
        raise ValueError("Empty embeddings.model")
    if value.startswith("/") or ".." in value.split("/"):
        resolved = Path(value).resolve()
        root = project_root.resolve()
        if not (resolved == root or root in resolved.parents):
            raise ValueError(
                f"embeddings.model local path {value!r} outside project root"
            )
        return value
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
    project_markers: list[str] = field(
        default_factory=lambda: list(_DEFAULT_PROJECT_MARKERS)
    )
    backend_handler_patterns: list[str] = field(default_factory=list)
    frontend_api_patterns: list[str] = field(default_factory=list)
    api_prefixes: list[str] = field(default_factory=lambda: list(_DEFAULT_API_PREFIXES))
    excluded_dirs: set[str] = field(default_factory=lambda: set(_DEFAULT_EXCLUDED_DIRS))

    # Embeddings
    embedding_model: str = "jinaai/jina-code-embeddings-0.5b"

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

    # Git
    churn_exclusions: list[str] = field(
        default_factory=lambda: list(_DEFAULT_CHURN_EXCLUSIONS)
    )
    git_timeout: int = 30

    # Timeouts
    scip_timeout_minutes: int = 0  # 0 = unlimited
    index_timeout_minutes: int = 30  # Overall index timeout (0 = unlimited)
    embedding_timeout: int = 60
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
        if self.cache_dir is None:
            self.cache_dir = self.project_root / ".descry_cache"
        else:
            self.cache_dir = Path(self.cache_dir)

    @property
    def graph_path(self) -> Path:
        return self.cache_dir / "codebase_graph.json"

    @property
    def resolved_project_root(self) -> Path:
        """Project root with all symlinks resolved.

        Use for containment checks (api_source, api_index, descry_index).
        Kept as a property instead of mutating project_root so existing test
        equality assertions against tmp_path continue to work.
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

        Rejects files > 1 MiB (TOML-parse DoS guard per A.4).

        Returns:
            Parsed TOML data as dict, or empty dict if not found/invalid.
        """
        import tomllib

        toml_path = Path(project_root) / ".descry.toml"
        if not toml_path.exists():
            return {}
        try:
            if toml_path.stat().st_size > 1 * 1024 * 1024:
                logger.warning(f".descry.toml exceeds 1 MiB cap; ignoring: {toml_path}")
                return {}
            with open(toml_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse .descry.toml: {e}")
            return {}

    def _apply_toml(self, data: dict) -> None:
        """Apply parsed TOML data to this config instance.

        Maps TOML sections/keys to config fields.
        """
        if not data:
            return

        # [project]
        project = data.get("project", {})
        if "excluded_dirs" in project:
            self.excluded_dirs = set(project["excluded_dirs"])
        if "max_stale_hours" in project:
            self.max_stale_hours = project["max_stale_hours"]

        # [features]
        features = data.get("features", {})
        if "enable_scip" in features:
            self.enable_scip = features["enable_scip"]
        if "enable_embeddings" in features:
            self.enable_embeddings = features["enable_embeddings"]

        # [embeddings]
        embeddings = data.get("embeddings", {})
        if "model" in embeddings:
            try:
                self.embedding_model = _validate_embedding_model(
                    embeddings["model"], self.project_root
                )
            except ValueError as e:
                logger.warning(
                    f"Invalid [embeddings] model in .descry.toml: {e}; "
                    f"falling back to default {self.embedding_model!r}"
                )

        # [test_detection]
        test_detection = data.get("test_detection", {})
        if "path_patterns" in test_detection:
            self.test_path_patterns = tuple(test_detection["path_patterns"])
        if "file_suffixes" in test_detection:
            self.test_file_suffixes = tuple(test_detection["file_suffixes"])

        # [code_files]
        code_files = data.get("code_files", {})
        if "extensions" in code_files:
            self.code_extensions = set(code_files["extensions"])

        # [git]
        git = data.get("git", {})
        if "churn_exclusions" in git:
            self.churn_exclusions = git["churn_exclusions"]
        if "timeout" in git:
            self.git_timeout = git["timeout"]

        # [timeouts]
        timeouts = data.get("timeouts", {})
        if "scip_minutes" in timeouts:
            self.scip_timeout_minutes = timeouts["scip_minutes"]
        if "index_minutes" in timeouts:
            self.index_timeout_minutes = timeouts["index_minutes"]
        if "embedding_seconds" in timeouts:
            self.embedding_timeout = timeouts["embedding_seconds"]
        if "query_ms" in timeouts:
            self.query_timeout_ms = timeouts["query_ms"]

        # [query]
        query = data.get("query", {})
        if "max_depth" in query:
            self.max_depth = query["max_depth"]
        if "max_nodes" in query:
            self.max_nodes = query["max_nodes"]
        if "max_children_per_level" in query:
            self.max_children_per_level = query["max_children_per_level"]
        if "max_callers_shown" in query:
            self.max_callers_shown = query["max_callers_shown"]

        # [scip]
        scip = data.get("scip", {})
        if "extra_args" in scip:
            try:
                self.scip_extra_args = [
                    _validate_scip_extra_arg(a) for a in scip["extra_args"]
                ]
            except ValueError as e:
                logger.warning(
                    f"Invalid [scip] extra_args in .descry.toml: {e}; "
                    f"keeping default {self.scip_extra_args!r}"
                )
        if "skip_crates" in scip:
            self.scip_skip_crates = scip["skip_crates"]

        # [scip.rust]
        scip_rust = scip.get("rust", {})
        if "toolchain" in scip_rust:
            try:
                self.scip_rust_toolchain = _validate_toolchain(scip_rust["toolchain"])
            except ValueError as e:
                logger.warning(
                    f"Invalid [scip.rust] toolchain in .descry.toml: {e}; ignoring"
                )

        # [syntax.lang_map] — merges with defaults (additive)
        syntax = data.get("syntax", {})
        lang_map = syntax.get("lang_map", {})
        if lang_map:
            self.syntax_lang_map.update(lang_map)

    @classmethod
    def from_env(cls) -> "DescryConfig":
        """Build config: auto_detect -> apply TOML -> apply env vars."""
        log_level = _env("DESCRY_LOG_LEVEL", "WARNING")
        logging.getLogger("descry").setLevel(
            getattr(logging, log_level.upper(), logging.WARNING)
        )

        # Step 1: auto-detect project root
        config = cls.auto_detect()

        # Step 2: apply TOML (overrides defaults)
        toml_data = cls._load_toml(config.project_root)
        config._apply_toml(toml_data)

        # Step 3: apply env vars (override TOML)
        cache_dir_str = _env("DESCRY_CACHE_DIR")
        no_scip = _env("DESCRY_NO_SCIP").lower() in ("1", "true", "yes")
        no_embeddings = _env("DESCRY_NO_EMBEDDINGS").lower() in ("1", "true", "yes")

        if cache_dir_str:
            config.cache_dir = Path(cache_dir_str)
        if no_scip:
            config.enable_scip = False
        if no_embeddings:
            config.enable_embeddings = False
        return config


# --- Format Helpers (module-level pure functions) ---


def format_search_result(
    node: dict, rank: int = 0, show_score: bool = False, score: float = 0.0
) -> str:
    """Format a single search result with enriched information."""
    meta = node.get("metadata", {})
    node_type = node.get("type", "?")[:3]
    name = meta.get("name", "unknown")

    node_id = node.get("id", "")
    file_path = ""
    lineno = meta.get("lineno")
    if node_id.startswith("FILE:"):
        file_path = node_id.split("::")[0].replace("FILE:", "")

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
        f"({token_count} toks, {in_degree} callers)" if token_count or in_degree else ""
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

    parent_name = meta.get("parent_name", "")
    display_name = f"{parent_name}.{name}" if parent_name else name

    node_id = node.get("id", "")
    file_path = ""
    lineno = meta.get("lineno")
    if node_id.startswith("FILE:"):
        file_path = node_id.split("::")[0].replace("FILE:", "")
        parts = file_path.split("/")
        if len(parts) > 2:
            file_path = "/".join(parts[-2:])

    location = f"{file_path}:{lineno}" if lineno else file_path

    token_count = meta.get("token_count", 0)
    in_degree = meta.get("in_degree", 0)

    return f"{rank}. [{node_type}] {display_name} | {location} ({token_count}t, {in_degree}c)"


def is_natural_language_query(terms: list[str]) -> bool:
    """Detect if query terms represent natural language vs code identifiers."""
    import re

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


def _try_import_query():
    try:
        from descry.query import GraphQuerier, _get_syntax_lang

        return GraphQuerier, _get_syntax_lang
    except ImportError:
        return None, None


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
            embeddings_available,
            SemanticSearcher,
            get_embeddings_status,
        )

        return embeddings_available(), SemanticSearcher, get_embeddings_status
    except ImportError:
        return False, None, None


def _try_import_scip(enabled: bool):
    if not enabled:
        return False, None, None
    try:
        from descry.scip.support import scip_available, get_scip_status

        return scip_available(), scip_available, get_scip_status
    except ImportError:
        return False, None, None


def _try_import_git_history():
    try:
        from descry.git_history import GitHistoryAnalyzer, GitError

        return True, GitHistoryAnalyzer, GitError
    except ImportError:
        return False, None, Exception


# --- DescryService ---


class DescryService:
    """Core business logic. No MCP dependency."""

    def __init__(self, config: DescryConfig | None = None):
        self.config = config or DescryConfig.from_env()

        # Lazy-loaded modules
        self._GraphQuerier, self._get_syntax_lang = _try_import_query()
        self._CrossLangTracer = _try_import_cross_lang()

        scip_ok, self._scip_available_fn, self._get_scip_status_fn = _try_import_scip(
            self.config.enable_scip
        )
        self._scip_loaded = scip_ok

        emb_ok, self._SemanticSearcher, self._get_embeddings_status_fn = (
            _try_import_embeddings(self.config.enable_embeddings)
        )
        self._semantic_available = emb_ok

        git_ok, self._GitHistoryAnalyzer, self._GitError = _try_import_git_history()
        self._git_available = git_ok

        # Instance-level caches (replaces module globals)
        self._graph_cache_lock = asyncio.Lock()
        self._querier_cache_lock = asyncio.Lock()
        self._semantic_cache_lock = asyncio.Lock()
        self._git_cache_lock = asyncio.Lock()
        self._dedup_cache_lock = asyncio.Lock()

        self._graph_cache = {"mtime": 0, "nodes": 0, "edges": 0}
        self._querier_cache = {"mtime": 0, "instance": None}
        self._semantic_cache = {
            "mtime": 0,
            "instance": None,
            "loading": False,
            "error": None,
        }
        self._git_cache = {"analyzer": None, "graph_mtime": None}
        self._dedup_cache: dict[str, tuple[float, str]] = {}
        self._max_dedup_entries = 100

    # --- Internal helpers ---

    def _get_graph_status(self) -> tuple[bool, str, float | None]:
        gp = self.config.graph_path
        if not gp.exists():
            return False, "Graph not found", None
        mtime = gp.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        return True, f"{age_hours:.1f}h ago", age_hours

    async def _update_cache(self):
        async with self._graph_cache_lock:
            gp = self.config.graph_path
            if gp.exists():
                mtime = gp.stat().st_mtime
                if mtime != self._graph_cache["mtime"]:
                    try:
                        from descry._graph import load_graph_with_schema

                        data = load_graph_with_schema(gp)
                        self._graph_cache = {
                            "mtime": mtime,
                            "nodes": len(data.get("nodes", [])),
                            "edges": len(data.get("edges", [])),
                        }
                    except Exception as e:
                        logger.warning(f"Failed to update graph cache: {e}")

    async def _get_querier(self):
        if self._GraphQuerier is None:
            return None
        async with self._querier_cache_lock:
            gp = self.config.graph_path
            if not gp.exists():
                self._querier_cache = {"mtime": 0, "instance": None}
                return None
            mtime = gp.stat().st_mtime
            if mtime != self._querier_cache["mtime"]:
                self._querier_cache = {
                    "mtime": mtime,
                    "instance": self._GraphQuerier(str(gp), config=self.config),
                }
            return self._querier_cache["instance"]

    async def _get_git_analyzer(self):
        async with self._git_cache_lock:
            gp = self.config.graph_path
            current_mtime = gp.stat().st_mtime if gp.exists() else None
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

    def _clear_dedup_cache(self):
        self._dedup_cache = {}

    def reset_caches(self):
        """Reset all cached instances. Call after reindex."""
        self._graph_cache = {"mtime": 0, "nodes": 0, "edges": 0}
        self._querier_cache = {"mtime": 0, "instance": None}
        self._semantic_cache = {
            "mtime": 0,
            "instance": None,
            "loading": False,
            "error": None,
        }
        self._git_cache = {"analyzer": None, "graph_mtime": None}
        self._dedup_cache = {}

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

    async def _load_embeddings_background(self):
        async with self._semantic_cache_lock:
            if (
                self._semantic_cache["instance"] is not None
                or self._semantic_cache["loading"]
            ):
                return
            self._semantic_cache["loading"] = True
            self._semantic_cache["error"] = None

        try:

            def load_sync():
                return self._SemanticSearcher(
                    str(self.config.graph_path), model_name=self.config.embedding_model
                )

            searcher = await asyncio.wait_for(
                asyncio.to_thread(load_sync), timeout=60.0
            )

            async with self._semantic_cache_lock:
                gp = self.config.graph_path
                mtime = gp.stat().st_mtime if gp.exists() else 0
                self._semantic_cache["mtime"] = mtime
                self._semantic_cache["instance"] = searcher
                logger.info("Pre-warm: embeddings ready")
        except asyncio.TimeoutError:
            async with self._semantic_cache_lock:
                self._semantic_cache["error"] = "Timeout loading embeddings (60s)"
            logger.warning("Pre-warm: embeddings load timed out")
        except Exception as e:
            async with self._semantic_cache_lock:
                self._semantic_cache["error"] = str(e)
            logger.warning(f"Pre-warm: embeddings failed: {e}")
        finally:
            async with self._semantic_cache_lock:
                self._semantic_cache["loading"] = False

    async def _get_semantic_searcher(self):
        """Single-flight accessor for the semantic searcher (E.1).

        Returns the cached SemanticSearcher, rebuilding if
        codebase_graph.json mtime changed. Returns None if embeddings are
        unavailable (module missing or disabled) or if the graph file does
        not exist.

        Uses _semantic_cache_lock; construction runs in asyncio.to_thread so
        the event loop is not blocked by model load. Callers that want to
        surface an "embeddings loading" UX string should check
        _semantic_cache["loading"] / ["error"] before calling this helper.
        """
        if not self._semantic_available or self._SemanticSearcher is None:
            return None
        gp = self.config.graph_path
        if not gp.exists():
            return None
        mtime = gp.stat().st_mtime

        async with self._semantic_cache_lock:
            if (
                self._semantic_cache["instance"] is not None
                and self._semantic_cache["mtime"] == mtime
            ):
                return self._semantic_cache["instance"]
            if self._semantic_cache["loading"]:
                return None
            self._semantic_cache["loading"] = True
            self._semantic_cache["error"] = None

        try:

            def load_sync():
                return self._SemanticSearcher(
                    str(gp), model_name=self.config.embedding_model
                )

            searcher = await asyncio.wait_for(
                asyncio.to_thread(load_sync), timeout=self.config.embedding_timeout
            )
            async with self._semantic_cache_lock:
                self._semantic_cache["mtime"] = mtime
                self._semantic_cache["instance"] = searcher
                self._semantic_cache["error"] = None
            return searcher
        except asyncio.TimeoutError:
            async with self._semantic_cache_lock:
                self._semantic_cache["error"] = (
                    f"Timeout loading embeddings ({self.config.embedding_timeout}s)"
                )
            logger.warning("Embeddings load timed out")
            return None
        except Exception as e:
            async with self._semantic_cache_lock:
                self._semantic_cache["error"] = str(e)
            logger.warning(f"Embeddings load failed: {e}")
            return None
        finally:
            async with self._semantic_cache_lock:
                self._semantic_cache["loading"] = False

    # --- Public API (handle_* → service methods) ---

    async def health(self) -> str:
        """Quick diagnostic check."""
        health = {
            "status": "ok",
            "version": SERVER_VERSION,
            "project_root": str(self.config.project_root),
            "cache_dir": str(self.config.cache_dir),
            "warm": False,
            "features": {
                "scip": False,
                "embeddings": False,
                "embeddings_loading": False,
                "embeddings_error": None,
                "git_history": self._git_available,
            },
            "graph": {
                "exists": False,
                "nodes": 0,
                "edges": 0,
                "age_hours": None,
            },
        }

        async with self._querier_cache_lock:
            health["warm"] = self._querier_cache["instance"] is not None

        exists, age_str, age_hours = self._get_graph_status()
        health["graph"]["exists"] = exists
        health["graph"]["age_hours"] = age_hours
        if exists:
            await self._update_cache()
            health["graph"]["nodes"] = self._graph_cache["nodes"]
            health["graph"]["edges"] = self._graph_cache["edges"]

        if self._scip_loaded and self._get_scip_status_fn:
            scip_status = self._get_scip_status_fn()
            health["features"]["scip"] = scip_status.get("available", False)
            if scip_status.get("disabled_by_env"):
                health["features"]["scip_note"] = "Disabled by DESCRY_NO_SCIP"

        async with self._semantic_cache_lock:
            health["features"]["embeddings"] = (
                self._semantic_cache["instance"] is not None
            )
            health["features"]["embeddings_loading"] = self._semantic_cache["loading"]
            health["features"]["embeddings_error"] = self._semantic_cache.get("error")

        if not self._semantic_available:
            if not self.config.enable_embeddings:
                health["features"]["embeddings_note"] = (
                    "Disabled by DESCRY_NO_EMBEDDINGS"
                )
            else:
                health["features"]["embeddings_note"] = (
                    "sentence-transformers not installed"
                )

        if not exists:
            health["status"] = "no_graph"
        elif age_hours and age_hours > self.config.max_stale_hours:
            health["status"] = "stale"

        return json.dumps(health, indent=2)

    async def status(self) -> str:
        """Check graph existence and freshness."""
        exists, age_str, age_hours = self._get_graph_status()
        if not exists:
            return "Graph: NOT FOUND. Run descry index first."

        await self._update_cache()
        stale = (
            " (STALE - run descry index)"
            if age_hours and age_hours > self.config.max_stale_hours
            else ""
        )
        result = f"Graph: {self._graph_cache['nodes']:,} nodes, {self._graph_cache['edges']:,} edges, updated {age_str}{stale}"

        if self._scip_loaded and self._get_scip_status_fn:
            scip_status = self._get_scip_status_fn()
            if scip_status.get("available"):
                indexers = scip_status.get("indexers", {})
                enabled = [
                    name for name, info in indexers.items() if info.get("available")
                ]
                scip_info = f"\nSCIP: Enabled ({', '.join(enabled) or 'none'})"
                scip_cache_dir = self.config.cache_dir / "scip"
                if scip_cache_dir.exists():
                    scip_files = list(scip_cache_dir.glob("*.scip"))
                    scip_info += f", {len(scip_files)} project(s) cached"
            elif scip_status.get("disabled_by_env"):
                scip_info = "\nSCIP: Disabled (DESCRY_NO_SCIP=1)"
            else:
                scip_info = "\nSCIP: Unavailable (no indexers found)"
        else:
            scip_info = "\nSCIP: Not loaded"
        result += scip_info

        if self._semantic_available and self._get_embeddings_status_fn:
            emb_status = self._get_embeddings_status_fn(str(self.config.graph_path))
            if emb_status.get("cached"):
                emb_info = (
                    f"\nEmbeddings: Ready ({emb_status.get('node_count', 0):,} nodes)"
                )
            elif emb_status.get("stale"):
                emb_info = (
                    "\nEmbeddings: STALE (graph updated, will rebuild on first search)"
                )
            else:
                emb_info = "\nEmbeddings: Not generated (will build on first search)"
        else:
            emb_info = "\nEmbeddings: Unavailable"
        result += emb_info

        return result

    async def ensure(self, max_age_hours: float = 24) -> str:
        """Ensure graph exists and is fresh."""
        exists, age_str, age_hours = self._get_graph_status()

        if not exists:
            result = await self.index(".")
            return f"Generated new graph.\n{result}"

        if age_hours and age_hours > max_age_hours:
            result = await self.index(".")
            return f"Refreshed stale graph ({age_hours:.0f}h old).\n{result}"

        await self._update_cache()
        return f"Graph ready: {self._graph_cache['nodes']:,}n/{self._graph_cache['edges']:,}e, {age_str}"

    async def index(self, path: str = ".") -> str:
        """Regenerate the codebase graph.

        A.9: `path` must resolve inside project_root. Prevents prompt-injected
        MCP clients from forcing indexing of arbitrary directories.
        """
        if path == ".":
            index_path = str(self.config.project_root)
        else:
            try:
                resolved = Path(path).resolve(strict=False)
                if not resolved.is_relative_to(self.config.resolved_project_root):
                    return (
                        f"Index path {path!r} outside project root; "
                        f"must be relative to {self.config.project_root}."
                    )
                index_path = str(resolved)
            except (OSError, ValueError) as e:
                return f"Invalid index path {path!r}: {e}"

        try:
            timeout = (
                self.config.index_timeout_minutes * 60
                if self.config.index_timeout_minutes
                else None
            )
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "descry.generate", index_path],
                capture_output=True,
                text=True,
                cwd=str(self.config.project_root),
                timeout=timeout,
                env=safe_env(),
            )

            if result.returncode == 0:
                output = result.stdout.strip() or result.stderr.strip()
                self._clear_dedup_cache()

                scip_status = ""
                if (
                    self._scip_loaded
                    and self._scip_available_fn
                    and self._scip_available_fn()
                ):
                    scip_cache_dir = self.config.cache_dir / "scip"
                    if scip_cache_dir.exists():
                        scip_files = list(scip_cache_dir.glob("*.scip"))
                        if scip_files:
                            scip_status = (
                                f"\nSCIP: {len(scip_files)} project(s) indexed"
                            )

                embeddings_status = ""
                if (
                    self._semantic_available
                    and self._SemanticSearcher
                    and self.config.graph_path.exists()
                ):
                    try:
                        # Force-rebuild: reset cache then construct in a thread
                        # so model load doesn't block the event loop (C.1).
                        async with self._semantic_cache_lock:
                            self._semantic_cache = {
                                "mtime": 0,
                                "instance": None,
                                "loading": False,
                                "error": None,
                            }
                        logger.info("Generating embeddings for semantic search...")

                        def _build_searcher():
                            return self._SemanticSearcher(
                                str(self.config.graph_path),
                                force_rebuild=True,
                                model_name=self.config.embedding_model,
                            )

                        searcher = await asyncio.to_thread(_build_searcher)
                        # Seed the cache with the freshly-built instance so
                        # the next search() call reuses it.
                        async with self._semantic_cache_lock:
                            gp = self.config.graph_path
                            mtime = gp.stat().st_mtime if gp.exists() else 0
                            self._semantic_cache["mtime"] = mtime
                            self._semantic_cache["instance"] = searcher
                        embeddings_status = (
                            f"\nEmbeddings: {len(searcher.nodes):,} nodes indexed"
                        )
                    except Exception as e:
                        embeddings_status = f"\nEmbeddings: Failed ({e})"
                        logger.warning(f"Embeddings generation failed: {e}")

                return f"Index complete.{scip_status}{embeddings_status}\n{output}"
            else:
                return f"Index failed:\n{result.stderr}"

        except subprocess.TimeoutExpired:
            mins = self.config.index_timeout_minutes
            return f"Index timed out after {mins} minutes. Set [timeouts] index_minutes in .descry.toml to increase."
        except Exception as e:
            return f"Index error: {e}"

    async def callers(self, name: str, limit: int = 20) -> str:
        """Find all callers of a symbol."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        all_callers = q.get_callers(name)
        fuzzy_note = ""

        if not all_callers:
            all_callers = q.get_callers(name, fuzzy=True)
            if all_callers:
                fuzzy_note = " (fuzzy match)"

        if not all_callers:
            result = (
                f"No callers of '{name}'. Try descry search to verify symbol exists."
            )
        else:
            total_count = len(all_callers)
            callers = sorted(all_callers)[:limit]
            lines = [f"{len(callers)} caller(s) of '{name}'{fuzzy_note}:"]
            for caller in callers:
                node_info = q.get_node_info(caller)
                if node_info:
                    lineno = node_info.get("metadata", {}).get("lineno")
                    if lineno:
                        file_path = caller.split("::")[0].replace("FILE:", "")
                        lines.append(f"  {caller} ({file_path}:{lineno})")
                    else:
                        lines.append(f"  {caller}")
                else:
                    lines.append(f"  {caller}")
            if len(callers) < total_count:
                lines.append(
                    f"  ... ({total_count - len(callers)} more, limit {limit})"
                )
            result = "\n".join(lines)

        return await self._format_response(result)

    async def callees(self, name: str, limit: int = 20) -> str:
        """Find what a symbol calls."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        matches = q.find_nodes_by_name(name)
        func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
        fuzzy_note = ""

        if not func_matches:
            matches = q.find_nodes_by_name(name, fuzzy=True)
            func_matches = [m for m in matches if m["type"] in ("Function", "Method")]
            if func_matches:
                fuzzy_note = " (fuzzy match)"

        if not func_matches:
            return f"No function '{name}' found. Try descry search."

        node = func_matches[0]
        callees = q.get_callees(node["id"])
        callees = sorted(callees)[:limit]

        display_name = node.get("metadata", {}).get("name", name)
        if not callees:
            result = f"'{display_name}'{fuzzy_note} calls no tracked functions."
        else:
            lines = [f"'{display_name}'{fuzzy_note} calls {len(callees)} function(s):"]
            for callee in callees:
                callee_info = q.get_node_info(callee)
                if callee_info:
                    lineno = callee_info.get("metadata", {}).get("lineno")
                    if lineno:
                        file_path = callee.split("::")[0].replace("FILE:", "")
                        lines.append(f"  {callee} ({file_path}:{lineno})")
                    else:
                        lines.append(f"  {callee}")
                else:
                    lines.append(f"  {callee}")
            result = "\n".join(lines)

        return await self._format_response(result)

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
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        gp = self.config.graph_path
        if deduplicate and gp.exists():
            graph_mtime = gp.stat().st_mtime
            content_hash = hashlib.md5(node_id.encode()).hexdigest()
            dedup_ref = await self._check_dedup(content_hash, graph_mtime)
            if dedup_ref:
                brief_result = q.get_context_prompt(node_id, brief=True)
                return await self._format_response(
                    f"{brief_result}\n\n{dedup_ref}", include_header=False, max_lines=50
                )

        result = q.get_context_prompt(
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

        if deduplicate and gp.exists():
            graph_mtime = gp.stat().st_mtime
            content_hash = hashlib.md5(node_id.encode()).hexdigest()
            await self._record_dedup(content_hash, graph_mtime, node_id)

        if brief:
            return await self._format_response(
                result, include_header=False, max_lines=50
            )

        depth_note = f" [depth={depth}]" if depth > 1 else ""
        full_note = " [full=true]" if full else ""
        head_note = f" [head_lines={head_lines}]" if head_lines else ""
        expand_note = " [expand_callees=true]" if expand_callees else ""
        ml = 2000 if (full or expand_callees) else 500
        return await self._format_response(
            result + depth_note + full_note + head_note + expand_note,
            include_header=False,
            max_lines=ml,
        )

    async def flow(
        self,
        start: str,
        direction: str = "forward",
        depth: int = 3,
        target: str | None = None,
        inline_threshold: int = 100,
    ) -> str:
        """Trace call flow from a starting symbol."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        result = q.trace_flow(
            start_name=start,
            direction=direction,
            depth=depth,
            target=target,
            inline_threshold=inline_threshold,
        )
        return await self._format_response(result, include_header=True, max_lines=300)

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
        q = await self._get_querier()
        if not q:
            return await self._format_response(
                "ERROR: Graph not found. Run descry ensure first.",
                include_header=True,
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

        tfidf_results = q.search_docs(
            terms,
            lang=lang,
            crate=crate,
            symbol_type=symbol_type,
            exclude_tests=exclude_tests,
        )[: limit * 2]

        semantic_results = []
        search_method = "keyword"

        if (
            self._semantic_available
            and self._SemanticSearcher
            and self.config.graph_path.exists()
        ):
            use_semantic = is_natural_language_query(terms) or len(tfidf_results) < 3
            if use_semantic:
                try:
                    # E.1/C.1: use the single-flight async helper (runs in
                    # asyncio.to_thread and respects the cache lock).
                    searcher = await self._get_semantic_searcher()
                    if searcher is not None:
                        query = " ".join(terms)
                        # searcher.search is CPU-bound; wrap in to_thread.
                        semantic_results = await asyncio.to_thread(
                            searcher.search, query, limit=limit * 2, min_score=0.25
                        )
                        search_method = "hybrid"
                except Exception as e:
                    logger.warning(f"Semantic search failed, using keyword only: {e}")

        if semantic_results and tfidf_results:
            combined = reciprocal_rank_fusion(tfidf_results, semantic_results)
            results = [node for node, _ in combined[:limit]]
            search_method = "hybrid"
        elif tfidf_results:
            results = tfidf_results[:limit]
        else:
            results = []

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
            for i, node in enumerate(results, 1):
                lines.append(format_search_result(node, rank=i))
                lines.append("")
            result = "\n".join(lines).rstrip()

        return await self._format_response(result)

    async def structure(self, filename: str) -> str:
        """Show file structure."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        matches = q.find_nodes_by_name(filename)
        file_matches = [m for m in matches if m["type"] == "File"]

        if not file_matches:
            return f"File '{filename}' not found. Check spelling."

        node_id = file_matches[0]["id"]
        lines = [f"{node_id}:"]

        defs = []
        imports = set()
        for edge in q.outgoing[node_id]:
            if edge["relation"] == "DEFINES":
                target = q.nodes.get(edge["target"])
                if target:
                    defs.append(target)
            elif edge["relation"] == "IMPORTS":
                target = edge["target"]
                if target.startswith("MODULE:"):
                    imports.add(target.replace("MODULE:", ""))
                else:
                    imports.add(target)

        if imports:
            sorted_imports = sorted(imports)
            lines.append(f"  Imports: {', '.join(sorted_imports[:10])}")
            if len(sorted_imports) > 10:
                lines.append(f"           (+{len(sorted_imports) - 10} more)")

        for type_name, type_filter in [
            ("Const", "Constant"),
            ("Class", "Class"),
            ("Fn", "Function"),
        ]:
            items = sorted(
                [d["metadata"]["name"] for d in defs if d["type"] == type_filter]
            )
            if items:
                lines.append(f"  {type_name}: {', '.join(items[:15])}")
                if len(items) > 15:
                    lines.append(f"       (+{len(items) - 15} more)")

        configs = [d for d in defs if d["type"] == "Configuration"]
        if configs:
            lines.append("  Config:")
            for cfg in sorted(configs, key=lambda x: x["metadata"].get("lineno", 0)):
                meta = cfg["metadata"]
                sig = meta.get("signature", meta.get("name", "?"))
                lineno = meta.get("lineno", "?")
                lines.append(f"    L{lineno}: {sig}")

        return await self._format_response("\n".join(lines))

    async def flatten(self, class_node_id: str) -> str:
        """Show effective API of a class including inherited methods."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        result = q.flatten_class(class_node_id)
        return await self._format_response(result, include_header=False, max_lines=150)

    async def semantic(self, query: str, limit: int = 10) -> str:
        """Pure semantic search using embeddings only."""
        if not self._semantic_available or not self._SemanticSearcher:
            return (
                "Semantic search not available. Install dependencies:\n"
                "  pip install sentence-transformers numpy\n\n"
                "Falling back to TF-IDF keyword search..."
            )

        gp = self.config.graph_path
        if not gp.exists():
            return "ERROR: Graph not found. Run descry ensure first."

        # Show a friendly message if a background pre-warm is in flight.
        async with self._semantic_cache_lock:
            if self._semantic_cache["loading"]:
                return "Embeddings loading in background (~2-3s). Try again shortly."

        # E.1/C.1: single-flight helper wraps the model load in asyncio.to_thread.
        searcher = await self._get_semantic_searcher()
        if searcher is None:
            err = self._semantic_cache.get("error")
            if err:
                return f"Failed to initialize semantic search: {err}"
            return "Semantic search unavailable."

        try:
            results = await asyncio.to_thread(searcher.search, query, limit=limit)
        except Exception as e:
            return f"Semantic search error: {e}"

        if not results:
            result = f"No semantic matches for '{query}'."
        else:
            lines = [f"{len(results)} semantic match(es) for '{query}':\n"]
            for i, (node, score) in enumerate(results, 1):
                lines.append(
                    format_search_result(node, rank=i, show_score=True, score=score)
                )
                lines.append("")
            result = "\n".join(lines).rstrip()

        return await self._format_response(result)

    async def quick(self, name: str, full: bool = False, brief: bool = False) -> str:
        """Find symbol and show full context in one step."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        matches = q.find_nodes_by_name(name)

        def type_priority(node):
            t = node.get("type", "")
            if t in ("Function", "Method"):
                return 0
            if t == "Class":
                return 1
            return 2

        matches.sort(key=type_priority)

        if not matches:
            matches = q.find_nodes_by_name(name, fuzzy=True)
            matches.sort(key=type_priority)

        if not matches:
            return f"No symbol found for '{name}'. Try descry search to explore."

        best_match = matches[0]
        node_id = best_match["id"]

        if brief:
            context = q.get_context_prompt(node_id, brief=True)
            other_matches = (
                f"\n*({len(matches) - 1} other matches)*" if len(matches) > 1 else ""
            )
            return await self._format_response(
                context + other_matches, include_header=False, max_lines=50
            )

        header_lines = [
            f"### Quick Context for `{name}`",
            "",
            format_search_result(best_match, rank=1),
            "",
        ]

        if len(matches) > 1:
            header_lines.append(f"*({len(matches) - 1} other matches available)*")
            header_lines.append("")

        context = q.get_context_prompt(node_id, full=full)
        result = "\n".join(header_lines) + "\n" + context
        if full:
            result += "\n [full=true]"

        ml = 2000 if full else 500
        return await self._format_response(result, include_header=False, max_lines=ml)

    async def impls(self, method: str, trait_name: str | None = None) -> str:
        """Find all implementations of a trait method."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        results = q.find_trait_impls(method, trait_name)

        if not results:
            trait_filter = f" for trait '{trait_name}'" if trait_name else ""
            return (
                f"No trait implementations found for method '{method}'{trait_filter}.\n\n"
                f"Note: trait_impl metadata is only available for recently indexed code. "
                f"Run descry index to regenerate the graph with this metadata."
            )

        by_trait = defaultdict(list)
        for node in results:
            meta = node.get("metadata", {})
            impl_trait = meta.get("trait_impl", "unknown")
            by_trait[impl_trait].append(node)

        lines = [f"### Implementations of `{method}`"]
        if trait_name:
            lines.append(f"*Filtered to trait: {trait_name}*")
        lines.append("")
        lines.append(f"Found {len(results)} implementation(s):\n")

        for trait, nodes in sorted(by_trait.items()):
            lines.append(f"**{trait}** ({len(nodes)} impl):")
            for node in nodes:
                meta = node.get("metadata", {})
                nid = node.get("id", "")
                parts = nid.split("::")
                struct_name = parts[-2] if len(parts) >= 2 else "?"
                file_path = (
                    nid.split("::")[0].replace("FILE:", "")
                    if nid.startswith("FILE:")
                    else ""
                )
                lineno = meta.get("lineno", "?")
                sig = meta.get("signature", f"fn {method}(...)")
                lines.append(f"  - **{struct_name}**::{method}")
                lines.append(f"    {file_path}:{lineno}")
                lines.append(f"    `{sig}`")
            lines.append("")

        return await self._format_response("\n".join(lines))

    async def path(
        self,
        start: str,
        end: str,
        max_depth: int = 10,
        direction: str = "forward",
    ) -> str:
        """Find shortest call path between two symbols."""
        q = await self._get_querier()
        if not q:
            return "ERROR: Graph not found. Run descry ensure first."

        result_path = q.find_call_path(
            start, end, max_depth=max_depth, direction=direction
        )

        if not result_path:
            return (
                f"No path found from '{start}' to '{end}' within {max_depth} hops.\n\n"
                f"Direction: {direction} ('{start}' {'calls' if direction == 'forward' else 'is called by'} '{end}')\n\n"
                f"Try:\n"
                f"- Verify both symbols exist with descry search\n"
                f"- Increase max_depth if the path is longer\n"
                f"- Try the opposite direction"
            )

        lines = [
            f"### Call Path: `{start}` -> `{end}` ({len(result_path)} hop{'s' if len(result_path) != 1 else ''})\n"
        ]

        for i, hop in enumerate(result_path, 1):
            caller_name = hop.get("caller_name", "?")
            callee_name = hop.get("callee_name", "?")
            file_path = hop.get("file_path", "")
            call_line = hop.get("call_line")
            snippet = hop.get("call_snippet", "")

            lines.append(f"**{i}. {caller_name}** -> **{callee_name}**")
            if file_path and call_line:
                lines.append(f"   {file_path}:{call_line}")
            elif file_path:
                lines.append(f"   {file_path}")

            if snippet:
                lang = (
                    self._get_syntax_lang(file_path)
                    if file_path and self._get_syntax_lang
                    else ""
                )
                lines.append(f"```{lang}")
                for line in snippet.split("\n"):
                    lines.append(f"   {line}")
                lines.append("```")
            lines.append("")

        return await self._format_response("\n".join(lines), include_header=True)

    async def cross_lang(
        self,
        mode: str = "endpoint",
        method: str | None = None,
        path: str | None = None,
        tag: str | None = None,
    ) -> str:
        """Trace API calls from frontend to backend handlers via OpenAPI spec."""
        if not self._CrossLangTracer:
            return "Cross-language tracing not available. Missing cross_lang module."

        openapi_path = self.config.openapi_path
        if openapi_path is None:
            return "Cross-language tracing not configured. Set openapi_path in config."
        if not openapi_path.exists():
            return f"OpenAPI spec not found at {openapi_path}."

        graph_path = (
            str(self.config.graph_path) if self.config.graph_path.exists() else None
        )
        tracer = self._CrossLangTracer(str(openapi_path), graph_path)

        if mode == "stats":
            stats = tracer.get_stats()
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
            endpoints = tracer.list_endpoints(tag=tag)
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
                    file_info = node_id.split("::")[0].replace("FILE:", "")
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

        elif mode == "endpoint":
            if not method or not path:
                return (
                    "Endpoint mode requires 'method' and 'path' arguments.\n"
                    "Example: method='GET', path='/api/v1/deployments'"
                )

            info = tracer.get_handler_info(method.upper(), path)
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
                    node_info = q.get_node_info(node_id)
                    if node_info:
                        meta = node_info.get("metadata", {})
                        if meta.get("lineno"):
                            file_path = node_id.split("::")[0].replace("FILE:", "")
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

        else:
            return f"Unknown mode '{mode}'. Use 'endpoint', 'list', or 'stats'."

    async def churn(
        self,
        time_range: str | None = None,
        path_filter: str | None = None,
        limit: int = 20,
        mode: str = "symbols",
        exclude_generated: bool = True,
    ) -> str:
        """Find code churn hotspots."""
        if not self._git_available:
            return "Git history tools not available. Missing git_history module."

        try:
            analyzer = await self._get_git_analyzer()
            result = await asyncio.to_thread(
                analyzer.get_churn,
                time_range=time_range,
                path_filter=path_filter,
                limit=limit,
                mode=mode,
                exclude_generated=exclude_generated,
            )
            return await self._format_response(result)
        except self._GitError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error in churn")
            return f"Error analyzing churn: {e}"

    async def evolution(
        self,
        name: str,
        time_range: str | None = None,
        limit: int = 10,
        show_diff: bool = False,
        crate: str | None = None,
    ) -> str:
        """Track how a symbol has changed over time."""
        if not self._git_available:
            return "Git history tools not available. Missing git_history module."

        try:
            analyzer = await self._get_git_analyzer()
            result = await asyncio.to_thread(
                analyzer.get_evolution,
                name=name,
                time_range=time_range,
                limit=limit,
                show_diff=show_diff,
                crate=crate,
            )
            return await self._format_response(result)
        except self._GitError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error in evolution")
            return f"Error analyzing evolution: {e}"

    async def changes(
        self,
        commit_range: str | None = None,
        time_range: str | None = None,
        path_filter: str | None = None,
        show_callers: bool = True,
        limit: int = 50,
    ) -> str:
        """Analyze change impact for a commit range."""
        if not self._git_available:
            return "Git history tools not available. Missing git_history module."

        try:
            analyzer = await self._get_git_analyzer()
            result = await asyncio.to_thread(
                analyzer.get_changes,
                commit_range=commit_range,
                time_range=time_range,
                path_filter=path_filter,
                show_callers=show_callers,
                limit=limit,
            )
            return await self._format_response(result)
        except self._GitError as e:
            return str(e)
        except Exception as e:
            logger.exception("Error in changes")
            return f"Error analyzing changes: {e}"
