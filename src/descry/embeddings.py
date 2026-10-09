"""Optional semantic search with embeddings for descry.

Uses sentence-transformers for embedding generation and numpy for similarity.
Falls back to TF-IDF search if dependencies are not available.

Usage:
    # Check if available
    if embeddings_available():
        searcher = SemanticSearcher(graph_path)
        results = searcher.search("authentication logic", limit=10)
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from descry.query import _read_file_cached, source_path

try:
    import fcntl as _fcntl  # Unix only
except ImportError:
    _fcntl = None


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout: float = 600.0):
    """Cross-process advisory lock.

    Uses fcntl.flock on Unix and msvcrt.locking on Windows. Both are held on
    an open handle, so the OS releases them when the holder exits.

    The lock file is never unlinked. Removing it would break mutual
    exclusion: a waiter blocked on the old inode and a newcomer that
    recreates the path hold locks on two different inodes and both enter.
    An empty lock file is cheap; correctness is not.
    """
    if _fcntl is not None:
        mode, poll_seconds = "w", 0.05

        def lock(f):
            _fcntl.flock(f, _fcntl.LOCK_EX | _fcntl.LOCK_NB)

        def unlock(f):
            _fcntl.flock(f, _fcntl.LOCK_UN)

    else:
        # Windows fallback: lock the first byte of the (always empty) lock file.
        import msvcrt

        mode, poll_seconds = "a", 0.1

        def lock(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

        def unlock(f):
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

    with open(lock_path, mode, encoding="utf-8") as f:
        # Poll non-blocking so the timeout is honoured; a blocking lock would
        # wait forever regardless of the argument.
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock(f)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for embeddings lock at {lock_path}"
                    ) from None
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            unlock(f)


logger = logging.getLogger(__name__)

# Identity of the embedding recipe. Anything that changes the vectors without
# changing the graph or the model name must be reflected here, or a stale
# cache is silently served against freshly-encoded queries. Bump RECIPE_VERSION
# whenever the text assembly or composition below changes.
RECIPE_VERSION = 4
DOCSTRING_CHAR_LIMIT = 500
CODE_CHAR_LIMIT = 1500
# Whitespace plus the quote and comment markers that surround a docstring line
# in source: `"""`, `'''`, `///`, `//!`, `//`, `#`, `*`, `/**`, `*/`.
_DOC_MARKER_CHARS = " \t\r\n\"'/*#!"
# A string prefix on a docstring's opening quotes, as in `r"""`.
_DOC_STRING_PREFIX_RE = re.compile(r"""^\s*[rRbBuU]{1,2}(?=["'])""")

# Instruction-tuned models score asymmetric retrieval better when the query and
# the document are encoded through their respective task prompts. Names differ
# per model family and are applied only when the loaded model registers them.
QUERY_PROMPT_CANDIDATES = ("nl2code_query", "query")
DOCUMENT_PROMPT_CANDIDATES = ("nl2code_document", "document")


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """A model descry has vetted: pinned and prompt-aware.

    A model named in `.descry.toml` that is not registered is loaded with no
    pinned revision. No model, registered or not, is loaded with
    `trust_remote_code`: descry never executes code from a model repository.
    """

    alias: str
    repo_id: str
    revision: str | None
    dim: int
    license: str
    context_tokens: int
    query_prompt: str | None
    document_prompt: str | None
    summary: str
    # `config_kwargs` for the SentenceTransformer constructor.
    config_kwargs: dict | None = None
    # Token cap set after loading, for a model that ships without one.
    max_seq_length: int | None = None
    # The descry extra that installs what this model needs beyond `embeddings`.
    extra: str | None = None


MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "jina-code": EmbeddingModelSpec(
        alias="jina-code",
        repo_id="jinaai/jina-code-embeddings-0.5b",
        revision="4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
        dim=896,
        license="CC-BY-NC-4.0",
        context_tokens=32768,
        query_prompt="nl2code_query",
        document_prompt="nl2code_document",
        summary="Code-specific. Non-commercial licence.",
    ),
    "qwen3": EmbeddingModelSpec(
        alias="qwen3",
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        dim=1024,
        license="Apache-2.0",
        context_tokens=32768,
        # Measured worse through its own generic query/document prompts than
        # with none at all, so this entry names none.
        query_prompt=None,
        document_prompt=None,
        summary="General purpose. Apache-2.0, the most permissive licence.",
    ),
    "embeddinggemma": EmbeddingModelSpec(
        alias="embeddinggemma",
        repo_id="google/embeddinggemma-300m",
        revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
        dim=768,
        license="Gemma",
        context_tokens=2048,
        # 'InstructionRetrieval' is this model's code-retrieval prompt,
        # 'task: code retrieval | query: '. The name is an MTEB task label, not
        # a description. It measured worse through its generic retrieval prompts.
        query_prompt="InstructionRetrieval",
        document_prompt="document",
        summary="Default. Gated: needs a HuggingFace login. 2K context.",
    ),
    "embeddinggemma-2": EmbeddingModelSpec(
        alias="embeddinggemma-2",
        repo_id="google/embeddinggemma-2",
        revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
        dim=768,
        license="Apache-2.0",
        context_tokens=8192,
        query_prompt="CodeRetrieval",
        document_prompt="Document",
        summary="Ungated, Apache-2.0. Needs the embeddinggemma-2 extra.",
        # Multimodal; without the vision and audio towers it is the 270M text
        # model and produces identical text vectors.
        config_kwargs={"vision_config": None, "audio_config": None},
        max_seq_length=8192,
        extra="embeddinggemma-2",
    ),
    "f2llm-v2-0.6b": EmbeddingModelSpec(
        alias="f2llm-v2-0.6b",
        repo_id="codefuse-ai/F2LLM-v2-0.6B",
        revision="8786315a8711c242ee03ec67c74dd9ad0a61e2cf",
        dim=1024,
        license="Apache-2.0",
        context_tokens=40960,
        # Measured better with no prompt than through its own query prompt.
        query_prompt=None,
        document_prompt=None,
        summary="Ungated, Apache-2.0. Used when the default is gated.",
    ),
}

DEFAULT_MODEL_ALIAS = "embeddinggemma"
# Used instead of the default, when no model is configured and the default is
# gated for this account.
FALLBACK_MODEL_ALIAS = "f2llm-v2-0.6b"


def resolve_model_spec(name: str) -> EmbeddingModelSpec | None:
    """Look a model up by alias or by full repo id. None if unregistered."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    for spec in MODEL_REGISTRY.values():
        if spec.repo_id == name:
            return spec
    return None


def list_models() -> list[dict]:
    """Registry contents, for `descry embedding-models` and diagnostics."""
    return [
        {
            "alias": spec.alias,
            "repo_id": spec.repo_id,
            "dim": spec.dim,
            "license": spec.license,
            "context_tokens": spec.context_tokens,
            "extra": spec.extra,
            "default": spec.alias == DEFAULT_MODEL_ALIAS,
            "summary": spec.summary,
        }
        for spec in MODEL_REGISTRY.values()
    ]


def _doc_line_content(line: str) -> str:
    """`line` without the indentation, quote and comment markers around it."""
    return _DOC_STRING_PREFIX_RE.sub("", line).strip(_DOC_MARKER_CHARS)


def code_excerpt(node: dict, source_root: str | Path | None) -> str:
    """The node's source lines without its docstring, up to CODE_CHAR_LIMIT.

    Empty without a `source_root`, for file nodes, for nodes lacking either
    line number and for files that cannot be read. A graph may be hand-edited,
    so only a regular file that resolves inside `source_root` is ever opened.

    Lines repeating a docstring line are dropped whatever markers surround
    them, so node_text() carries the docstring at most once and the excerpt
    does not depend on whether the docstring is included.
    """
    meta = node.get("metadata") or {}
    file_part, sep, _ = node["id"].partition("::")
    start, end = meta.get("lineno"), meta.get("end_lineno")
    if not (source_root and sep and file_part.startswith("FILE:") and start and end):
        return ""
    path = source_path(source_root, file_part.removeprefix("FILE:"))
    if not path:
        return ""
    doc_lines = {
        _doc_line_content(line) for line in (meta.get("docstring") or "").splitlines()
    } - {""}
    kept = [
        line
        for line in _read_file_cached(path)[start - 1 : end]
        if _doc_line_content(line) not in doc_lines
    ]
    return "".join(kept).strip()[:CODE_CHAR_LIMIT]


def node_text(node: dict, *, include_docstring: bool = True, code: str = "") -> str:
    """The document string for one graph node: name, signature, docstring, code.

    One text per node: averaging separate vectors for the parts pulls every
    node toward the centroid of its own parts.

    `code` is the node's code_excerpt(), or empty to leave source out.

    tests/eval calls this with `include_docstring=False` for its leave-one-out
    control, so it must not keep a copy: the substituted row has to match the
    index row it replaces.
    """
    meta = node.get("metadata") or {}
    name = meta.get("name") or node["id"].split("::")[-1]
    parts = [name, meta.get("signature") or ""]
    if include_docstring:
        parts.append((meta.get("docstring") or "")[:DOCSTRING_CHAR_LIMIT])
    parts.append(code)
    text = " ".join(p for p in parts if p).strip()
    return text or node["id"]


class EmbeddingCacheMismatch(RuntimeError):
    """Cached vectors are incompatible with the currently loaded model."""


class EmbeddingModelGated(RuntimeError):
    """The model's HuggingFace repo needs a licence acceptance and a login."""


class EmbeddingModelDependencyMissing(RuntimeError):
    """The model needs packages that its descry extra installs."""


def _is_gated_repo_error(exc: BaseException | None) -> bool:
    """True if `exc`, or anything it was raised from, is HF's gated-repo refusal.

    Matched by name: huggingface_hub is only a transitive dependency, and
    transformers re-raises the hub's error wrapped in a plain OSError.
    """
    while exc is not None:
        if type(exc).__name__ == "GatedRepoError":
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _resolve_cache_dir(graph_path: Path, cache_dir: str | None = None) -> Path:
    """Where embedding artifacts live for a given graph.

    Shared by SemanticSearcher and get_embeddings_status, so status looks for
    a cache in the directory the searcher writes it to.
    """
    if cache_dir:
        return Path(cache_dir).resolve()
    if graph_path.parent.name == ".descry_cache":
        return graph_path.parent
    return graph_path.parent / ".descry_cache"


# Try to import embedding dependencies
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None
    SentenceTransformer = None


def embeddings_available() -> bool:
    """Check if embedding dependencies are available."""
    return EMBEDDINGS_AVAILABLE


def _load_sentence_transformer(model_name: str, revision: str | None = None):
    """Construct a SentenceTransformer, pinned where descry has a pin.

    Registered models supply their own pinned revision. Anything else - an
    arbitrary repo id or a local path from `.descry.toml` [embeddings] model -
    is loaded unpinned.

    `trust_remote_code` is always False. There is deliberately no parameter,
    registry field or config key that turns it on: descry never executes code
    shipped inside a model repository.

    Args:
        model_name: Registry alias, HuggingFace repo id, or local path.
        revision: Overrides the registry's pinned revision.
    """
    spec = resolve_model_spec(model_name)
    target = spec.repo_id if spec else model_name
    if revision is None and spec:
        revision = spec.revision
    kwargs: dict = {"trust_remote_code": False}
    if revision:
        kwargs["revision"] = revision
    if spec and spec.config_kwargs:
        kwargs["config_kwargs"] = dict(spec.config_kwargs)
    try:
        model = SentenceTransformer(target, **kwargs)
        if spec and spec.max_seq_length:
            model.max_seq_length = spec.max_seq_length
        return model
    except ImportError as e:
        if not (spec and spec.extra):
            raise
        # Not re-raised as ImportError: callers treat that as "embeddings are
        # not installed" and skip silently.
        raise EmbeddingModelDependencyMissing(
            f"{spec.alias} needs packages that are not installed ({e}). "
            f"Install them with: pip install 'descry-codegraph[{spec.extra}]'"
        ) from e
    except OSError as e:
        if not _is_gated_repo_error(e):
            raise
        raise EmbeddingModelGated(
            f"{target} is a gated HuggingFace model. Accept its licence at "
            f"https://huggingface.co/{target}, then authenticate with "
            "`hf auth login` (or set HF_TOKEN). To use an ungated model "
            f'instead, set [embeddings] model = "{FALLBACK_MODEL_ALIAS}" in '
            ".descry.toml."
        ) from e


class SemanticSearcher:
    """Semantic search using sentence embeddings.

    Generates embeddings for node names and docstrings, then uses
    cosine similarity for semantic search.
    """

    # The default model, by repo id. Its dimensionality, licence and prompts
    # live on its MODEL_REGISTRY entry, not here.
    MODEL_NAME = MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].repo_id
    # Pinned HF revision (git sha) for the default model. The revision is part
    # of the cache key, so bumping it invalidates existing caches; the model
    # name alone would not, since the name is unchanged by a revision bump.
    DEFAULT_MODEL_REVISION = MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].revision

    def __init__(
        self,
        graph_path: str,
        cache_dir: str | None = None,
        force_rebuild: bool = False,
        model_name: str | None = None,
        source_root: str | None = None,
    ):
        """Initialize the semantic searcher.

        Args:
            graph_path: Path to codebase_graph.json
            cache_dir: Optional directory for caching embeddings
            force_rebuild: Force regeneration of embeddings even if cache exists
            model_name: Registry alias, HuggingFace repo id or local path.
                None selects the default model, or FALLBACK_MODEL_ALIAS when
                the default is gated for this account.
            source_root: Directory the graph's node ids are relative to. When
                given, each node's text includes its code_excerpt().
        """
        self._fallback_allowed = model_name is None
        self.model_name = model_name or self.MODEL_NAME
        self.source_root = source_root
        self._model_lock = threading.Lock()

        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embeddings require sentence-transformers and numpy. "
                "Install with: pip install descry-codegraph[embeddings]"
            )

        # Resolve to absolute path to avoid nested directory issues when CWD is inside cache
        self.graph_path = Path(graph_path).resolve()
        self.cache_dir = _resolve_cache_dir(self.graph_path, cache_dir)

        # Load graph (B.6: schema-checked)
        from descry._graph import load_graph_with_schema

        self.data = load_graph_with_schema(self.graph_path)
        self.nodes = self.data["nodes"]

        # Load or create embeddings
        self.model = None
        self.embeddings = None
        self.node_texts = None
        self._load_or_create_embeddings(force_rebuild=force_rebuild)

    def _recipe_fingerprint(self) -> str:
        """Identity of everything that shapes the vectors except the graph.

        Model name alone is not enough: a pinned-revision bump, a changed
        truncation limit, or a changed text assembly all produce different
        vectors under an unchanged name, and serving those from cache against
        freshly-encoded queries compares two different embedding spaces.
        """
        parts = [
            self._resolved_repo_id(),
            self._resolved_revision() or "",
            str(RECIPE_VERSION),
            str(DOCSTRING_CHAR_LIMIT),
            # The code excerpt's limit, or 0 for an index built without code.
            str(CODE_CHAR_LIMIT if self.source_root else 0),
            # An unregistered model's prompts are probed for once it is
            # loaded, after this key is needed, so the names probed for
            # stand in for the one that will be picked.
            str(spec.query_prompt if (spec := self.spec) else QUERY_PROMPT_CANDIDATES),
            str(spec.document_prompt if spec else DOCUMENT_PROMPT_CANDIDATES),
        ]
        # Appended only when set, so models without a cap keep their key.
        if spec and spec.max_seq_length:
            parts.append(f"max_seq_length={spec.max_seq_length}")
        if spec and spec.config_kwargs:
            parts.append(f"config_kwargs={sorted(spec.config_kwargs.items())!r}")
        # A local checkpoint can change under a stable path, so fold in the
        # directory's file sizes and mtimes rather than just its name.
        local = Path(self.model_name)
        if local.is_dir():
            for f in sorted(local.rglob("*")):
                if f.is_file():
                    st = f.stat()
                    parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]

    @property
    def spec(self) -> EmbeddingModelSpec | None:
        """Registry entry for this searcher's model, if it has one."""
        return resolve_model_spec(self.model_name)

    def _resolved_revision(self) -> str | None:
        spec = self.spec
        if spec is None:
            return None
        # Honour a monkeypatched class constant for the default model so the
        # cache key still moves when the pinned revision is bumped.
        if spec.alias == DEFAULT_MODEL_ALIAS:
            return self.DEFAULT_MODEL_REVISION
        return spec.revision

    def _resolved_repo_id(self) -> str:
        spec = self.spec
        return spec.repo_id if spec else self.model_name

    def _prompt_name(self, kind: str) -> str | None:
        """Prompt to encode `kind` ("query" or "document") through.

        A spec is authoritative even when it names no prompt — some models
        retrieve worse through their own generic prompts, so "none" must be
        expressible. An unregistered model is probed for the first
        candidate name it registers.
        """
        spec = self.spec
        if spec is not None:
            explicit = spec.query_prompt if kind == "query" else spec.document_prompt
            if not explicit:
                return None
            candidates: tuple[str, ...] = (explicit,)
        else:
            candidates = (
                QUERY_PROMPT_CANDIDATES
                if kind == "query"
                else DOCUMENT_PROMPT_CANDIDATES
            )
        return next((c for c in candidates if c in self.model.prompts), None)

    def _cache_key(self) -> str:
        """Compute content-addressed cache key.

        Composition: int(mtime) + sha256[:16] of graph bytes + sha256[:8] of the
        recipe fingerprint (model name, revision, recipe version, truncation).
        """
        graph_mtime = int(self.graph_path.stat().st_mtime)
        graph_hash = hashlib.sha256(self.graph_path.read_bytes()).hexdigest()[:16]
        return f"{graph_mtime}_{graph_hash}_{self._recipe_fingerprint()}"

    def _cache_paths(self) -> tuple[Path, Path]:
        """Return (npz_path, json_sidecar_path) for the current cache key."""
        key = self._cache_key()
        return (
            self.cache_dir / f"embeddings_{key}.npz",
            self.cache_dir / f"embeddings_{key}.json",
        )

    def _cleanup_old_embeddings(self, keep: set[Path]):
        """Remove embedding cache files not in the keep set."""
        try:
            # No writer's in-flight temporary is removed: these globs never
            # match _atomic_save's dot-prefixed names, and `.tmp.` excludes
            # the undotted names descry releases before 0.3 write here.
            old_files = [
                f
                for pattern in ("embeddings_*.npz", "embeddings_*.json")
                for f in self.cache_dir.glob(pattern)
                if f not in keep and ".tmp." not in f.name
            ]
            for old_file in old_files:
                logger.info(f"Removing stale embedding cache: {old_file.name}")
                old_file.unlink()
            if old_files:
                logger.info(f"Cleaned up {len(old_files)} old embedding file(s)")
        except OSError as e:
            logger.warning(f"Failed to cleanup old embeddings: {e}")

    def _atomic_save(self, npz_path: Path, json_path: Path) -> None:
        """Write .npz + JSON sidecar atomically (tmp then rename).

        Note: np.savez appends '.npz' to paths that don't end in '.npz'.
        We write to a temporary ending in '.tmp.npz' (which numpy preserves
        because the suffix is already .npz) and then rename to the final path.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Dot-prefixed and pid-tagged: invisible to the cleanup glob, and two
        # concurrent writers cannot collide on the same temporary.
        npz_tmp = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp.npz")
        json_tmp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp.json")
        cleanup = [npz_tmp, json_tmp]
        try:
            np.savez(npz_tmp, embeddings=self.embeddings)
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "node_count": len(self.node_texts),
                        "recipe_version": RECIPE_VERSION,
                        "model": self.model_name,
                        "revision": self._resolved_revision(),
                        "dim": int(self.embeddings.shape[1]),
                        "document_prompt": self._document_prompt,
                    },
                    f,
                )
            os.replace(npz_tmp, npz_path)
            os.replace(json_tmp, json_path)
        finally:
            for p in cleanup:
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def _load_or_create_embeddings(self, force_rebuild: bool = False):
        """Load embeddings from cache or create new ones.

        Serialized via a cache-dir lockfile so concurrent processes (e.g.
        descry-mcp + descry-web both starting on a cold cache) don't each
        load the model and write to the same files.
        """
        # Fast path: cache is hot, no lock needed.
        if not force_rebuild and self._load_cached():
            return

        # Cold path: acquire a cross-process lock, recheck, then generate.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / "embeddings.lock"
        with _file_lock(lock_path):
            if not force_rebuild and self._load_cached():
                return

            try:
                self._generate_embeddings()
            except EmbeddingModelGated:
                if not self._fallback_allowed:
                    raise
                logger.warning(
                    f"{self.model_name} is gated for this account; using "
                    f"{FALLBACK_MODEL_ALIAS} instead. To use the default model, "
                    f"accept its licence at https://huggingface.co/{self.model_name}, "
                    "authenticate with `hf auth login` (or set HF_TOKEN) and run "
                    "`descry index`. To keep this model without the warning, set "
                    f'[embeddings] model = "{FALLBACK_MODEL_ALIAS}" in .descry.toml.'
                )
                self.model_name = FALLBACK_MODEL_ALIAS
                self._generate_embeddings()
            npz_path, json_path = self._cache_paths()
            self._atomic_save(npz_path, json_path)
            logger.info(f"Cached {len(self.node_texts)} embeddings to {npz_path}")
            self._cleanup_old_embeddings(keep={npz_path, json_path})

    def _load_cached(self) -> bool:
        """Load this graph's cached vectors, whichever model built them.

        With no model configured the index may have been built by the default
        or by the fallback; `model_name` is left naming the one that was.
        """
        candidates = [self.model_name]
        if self._fallback_allowed:
            candidates.append(FALLBACK_MODEL_ALIAS)
        for self.model_name in candidates:
            npz_path, json_path = self._cache_paths()
            if (
                npz_path.exists()
                and json_path.exists()
                and self._try_load_cache(npz_path, json_path)
            ):
                return True
        self.model_name = candidates[0]
        return False

    def _try_load_cache(self, npz_path: Path, json_path: Path) -> bool:
        """Attempt to load embeddings from cache. Returns True on success."""
        try:
            data = np.load(npz_path, allow_pickle=False)
            embeddings = data["embeddings"]
            with open(json_path, encoding="utf-8") as f:
                sidecar = json.load(f)
            node_count = sidecar["node_count"]
            if sidecar.get("recipe_version") != RECIPE_VERSION:
                logger.warning(
                    "Embedding cache built by recipe v%s, current is v%s; regenerating",
                    sidecar.get("recipe_version"),
                    RECIPE_VERSION,
                )
                return False
            # Consistency check: an interrupted _atomic_save (or two racing
            # writers) could leave a mismatched pair on disk. Refuse to load
            # it — the cache will regenerate and overwrite with a consistent
            # pair. Cheaper than silently serving wrong embeddings.
            if node_count != embeddings.shape[0]:
                logger.warning(
                    "Embedding cache mismatch (%d nodes vs %d embeddings); regenerating",
                    node_count,
                    embeddings.shape[0],
                )
                return False
            self.embeddings = embeddings
            self._document_prompt = sidecar.get("document_prompt")
            logger.info(f"Loaded {node_count} embeddings from cache")
            # No pruning here: this runs on the lock-free fast path, where it
            # could delete files another process is writing. Pruning happens
            # under the lock in _load_or_create_embeddings.
            return True
        except Exception as e:  # noqa: BLE001 — corrupt cache regenerates; numpy/zipfile signal corruption with open-ended types
            logger.warning(f"Cache load failed: {e}, regenerating...")
            return False

    def ensure_model(self):
        """Load the model once, even under concurrent first searches.

        DescryService shares one searcher across worker threads; the lock
        keeps them from each constructing their own copy of a model several
        hundred megabytes in size.
        """
        if self.model is not None:
            return self.model
        with self._model_lock:
            if self.model is None:
                self.model = _load_sentence_transformer(self.model_name)
        return self.model

    def _generate_embeddings(self):
        """Encode one composite text per node in a single pass."""
        if not self.nodes:
            # An empty batch encodes to an array with no second axis to norm.
            self.node_texts = []
            self.embeddings = np.zeros((0, 0), dtype="float32")
            self._document_prompt = None
            return

        logger.info("Loading embedding model...")
        self.ensure_model()

        self.node_texts = [
            node_text(n, code=code_excerpt(n, self.source_root)) for n in self.nodes
        ]
        logger.info(f"Generating embeddings for {len(self.node_texts)} nodes...")

        prompt_name = self._prompt_name("document")
        # Recorded in the sidecar: the cache key cannot hold the prompt an
        # unregistered model resolves to, which is only known once it loads.
        self._document_prompt = prompt_name
        if prompt_name:
            logger.info(f"  Using document prompt {prompt_name!r}")
        kwargs = {"show_progress_bar": False, "convert_to_numpy": True}
        if prompt_name:
            kwargs["prompt_name"] = prompt_name

        embeddings = self.model.encode(self.node_texts, **kwargs)
        embeddings = np.asarray(embeddings, dtype="float32")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.where(norms == 0, 1, norms)
        logger.info("Embeddings generated successfully")

    def search(self, query: str, limit: int = 10, min_score: float = 0.3) -> list:
        """Search for nodes semantically similar to the query.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_score: Minimum final score, i.e. cosine similarity after
                the in-degree and type boosts are added (may exceed 1).

        Returns:
            List of (node, score) tuples sorted by relevance

        Scoring includes:
        - Cosine similarity (primary)
        - In-degree boost (symbols with more callers ranked higher)
        - Type preference (Functions/Classes over Constants)
        """
        import math

        if limit <= 0 or not self.nodes:
            return []

        self.ensure_model()

        document_prompt = self._prompt_name("document")
        if document_prompt != self._document_prompt:
            raise EmbeddingCacheMismatch(
                f"Cached embeddings were encoded through document prompt "
                f"{self._document_prompt!r} but {self.model_name} now resolves "
                f"{document_prompt!r}; re-index to rebuild them."
            )

        # Encode query through the model's query prompt when it has one, to
        # match the document prompt used when the index was built.
        kwargs = {"convert_to_numpy": True}
        prompt_name = self._prompt_name("query")
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        query_embedding = self.model.encode([query], **kwargs)[0]

        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise EmbeddingCacheMismatch(
                f"Cached embeddings are {self.embeddings.shape[1]}-dimensional but "
                f"{self.model_name} produced {query_embedding.shape[0]}; "
                "delete the embeddings cache and re-index."
            )

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Take a candidate pool by base score, then apply min_score to the
        # final score, so a node whose boosts (worth up to +0.17) lift it over
        # the threshold is kept.
        pool = min(len(similarities), limit * 3)
        top_indices = np.argpartition(-similarities, pool - 1)[:pool]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Re-rank with in-degree boost and type preference
        results = []
        for idx in top_indices:
            base_score = float(similarities[idx])
            node = self.nodes[idx]
            meta = node.get("metadata", {})

            # In-degree boost: log scaling to prevent domination by very popular symbols
            in_degree = meta.get("in_degree", 0)
            in_degree_boost = 0.0
            if in_degree > 0:
                # Boost ranges from 0 to ~0.15 for nodes with 1-100+ callers
                in_degree_boost = min(0.15, math.log(1 + in_degree) * 0.03)

            # Type preference: small tie-breaker
            node_type = node.get("type", "")
            type_boost = {
                "Function": 0.02,
                "Method": 0.02,
                "Class": 0.015,
                "Constant": 0.005,
                "File": 0.0,
            }.get(node_type, 0.01)

            final_score = base_score + in_degree_boost + type_boost
            # NaN compares False to everything; written so it is dropped.
            if not (final_score >= min_score):
                continue
            results.append((node, final_score))

        # Sort by final score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


def get_embeddings_status(
    graph_path: str = ".descry_cache/codebase_graph.json",
    model_name: str | None = None,
    cache_dir: str | None = None,
    source_root: str | None = None,
) -> dict:
    """Report embedding cache state for diagnostics.

    Recomputes the cache key the searcher would use, so a cache left over from
    an older graph or a different model is reported as stale rather than as
    Ready. Reads only the JSON sidecar - never np.load of a user-controlled
    .npz.

    Pass the `model_name`, `cache_dir` and `source_root` the searcher is
    constructed with, or the key will not match.
    """
    graph_path = Path(graph_path).resolve()

    status = {
        "available": EMBEDDINGS_AVAILABLE,
        "cached": False,
        "node_count": 0,
        "stale": False,
        "cache_path": None,
    }

    if not EMBEDDINGS_AVAILABLE or not graph_path.exists():
        return status

    resolved_cache_dir = _resolve_cache_dir(graph_path, cache_dir)
    if not resolved_cache_dir.is_dir():
        return status

    # With no model configured, the fallback model may have built the index.
    candidates = [model_name or SemanticSearcher.MODEL_NAME]
    if model_name is None:
        candidates.append(FALLBACK_MODEL_ALIAS)
    probe = SemanticSearcher.__new__(SemanticSearcher)
    probe.graph_path = graph_path
    probe.cache_dir = resolved_cache_dir
    probe.source_root = source_root
    for probe.model_name in candidates:
        try:
            expected_npz, expected_json = probe._cache_paths()
        except OSError:
            return status

        if expected_npz.exists() and expected_json.exists():
            status["cached"] = True
            status["model"] = probe._resolved_repo_id()
            status["cache_path"] = str(expected_npz)
            try:
                with open(expected_json, encoding="utf-8") as f:
                    status["node_count"] = json.load(f).get("node_count", 0)
            except (OSError, json.JSONDecodeError, KeyError):
                pass
            return status

    # Something is cached, but not for this graph/model/recipe combination.
    others = list(resolved_cache_dir.glob("embeddings_*.npz"))
    if others:
        status["stale"] = True
        status["cache_path"] = str(max(others, key=lambda p: p.stat().st_mtime))
    return status
