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
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl as _fcntl  # Unix only
except ImportError:
    _fcntl = None


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout: float = 600.0):
    """Cross-process advisory lock.

    Uses fcntl.flock on Unix; falls back to atomic-create sentinel on
    platforms without fcntl (Windows).

    The lock file is never unlinked. Removing it would break mutual
    exclusion: a waiter blocked on the old inode and a newcomer that
    recreates the path hold locks on two different inodes and both enter.
    An empty lock file is cheap; correctness is not.
    """
    if _fcntl is not None:
        with open(lock_path, "w", encoding="utf-8") as f:
            # Poll non-blocking so the documented timeout is actually honoured;
            # a plain LOCK_EX would block forever regardless of the argument.
            deadline = time.monotonic() + timeout
            while True:
                try:
                    _fcntl.flock(f, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for embeddings lock at {lock_path}"
                        ) from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                _fcntl.flock(f, _fcntl.LOCK_UN)
        return

    # Windows fallback: exclusive-create a sentinel file.
    start = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Timed out waiting for embeddings lock at {lock_path}"
                ) from None
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


logger = logging.getLogger(__name__)

# Identity of the embedding recipe. Anything that changes the vectors without
# changing the graph or the model name must be reflected here, or a stale
# cache is silently served against freshly-encoded queries. Bump RECIPE_VERSION
# whenever the text assembly or composition below changes.
RECIPE_VERSION = 2
DOCSTRING_CHAR_LIMIT = 500

# Instruction-tuned models score asymmetric retrieval markedly better when the
# query and the document are encoded through their respective task prompts.
# Measured on this repo's own graph (405 docstring->symbol queries, leave-one-out):
# R@1 0.360 -> 0.405, nDCG@10 0.550 -> 0.593. Names differ per model family and
# are applied only when the loaded model actually registers them.
QUERY_PROMPT_CANDIDATES = ("nl2code_query", "query")
DOCUMENT_PROMPT_CANDIDATES = ("nl2code_document", "document")


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """A model descry has vetted: pinned, prompt-aware, and load-safe.

    `trust_remote_code` is only ever True for entries in this registry. A model
    named in `.descry.toml` that is not registered is loaded with remote code
    disabled and no pinned revision, because descry cannot vouch for it.
    """

    alias: str
    repo_id: str
    revision: str | None
    trust_remote_code: bool
    dim: int
    license: str
    context_tokens: int
    query_prompt: str | None
    document_prompt: str | None
    summary: str


MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "jina-code": EmbeddingModelSpec(
        alias="jina-code",
        repo_id="jinaai/jina-code-embeddings-0.5b",
        revision="4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
        trust_remote_code=True,
        dim=896,
        license="CC-BY-NC-4.0",
        context_tokens=32768,
        query_prompt="nl2code_query",
        document_prompt="nl2code_document",
        summary="Code-specific. Non-commercial licence; needs remote code.",
    ),
    "qwen3": EmbeddingModelSpec(
        alias="qwen3",
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        trust_remote_code=False,
        dim=1024,
        license="Apache-2.0",
        context_tokens=32768,
        # Measured worse through its own generic query/document prompts
        # (R@1 0.360) than with none at all, so this entry names none.
        query_prompt=None,
        document_prompt=None,
        summary="General purpose. Apache-2.0, the most permissive licence.",
    ),
    "embeddinggemma": EmbeddingModelSpec(
        alias="embeddinggemma",
        repo_id="google/embeddinggemma-300m",
        revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
        trust_remote_code=False,
        dim=768,
        license="Gemma",
        context_tokens=2048,
        # 'InstructionRetrieval' is this model's code-retrieval prompt,
        # 'task: code retrieval | query: '. The name is an MTEB task label, not
        # a description. Through its generic retrieval prompts instead it drops
        # to R@1 0.385, below jina-code.
        query_prompt="InstructionRetrieval",
        document_prompt="document",
        summary="Default. Best measured retrieval; smallest index. 2K context.",
    ),
}

DEFAULT_MODEL_ALIAS = "embeddinggemma"
# Named in the error a gated default produces, so the fix is one line of toml.
UNGATED_FALLBACK_ALIAS = "qwen3"


def resolve_model_spec(name: str | None) -> EmbeddingModelSpec | None:
    """Look a model up by alias or by full repo id. None if unregistered."""
    if not name:
        return None
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
            "trust_remote_code": spec.trust_remote_code,
            "default": spec.alias == DEFAULT_MODEL_ALIAS,
            "summary": spec.summary,
        }
        for spec in MODEL_REGISTRY.values()
    ]


def node_text(node: dict, *, include_docstring: bool = True) -> str:
    """The document string for one graph node: name, signature, docstring.

    One text per node, not three averaged — averaging pulls every node toward
    the centroid of its own parts (R@1 0.185 vs 0.405 here).

    tests/eval calls this with `include_docstring=False` for its leave-one-out
    control, so it must not keep a copy: the substituted row has to match the
    index row it replaces.
    """
    meta = node.get("metadata") or {}
    name = meta.get("name") or node["id"].split("::")[-1]
    parts = [name, meta.get("signature") or ""]
    if include_docstring:
        parts.append((meta.get("docstring") or "")[:DOCSTRING_CHAR_LIMIT])
    text = " ".join(p for p in parts if p).strip()
    return text or node["id"]


class EmbeddingCacheMismatch(RuntimeError):
    """Cached vectors are incompatible with the currently loaded model."""


class EmbeddingModelGated(RuntimeError):
    """The model's HuggingFace repo needs a licence acceptance and a login."""


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

    Shared by SemanticSearcher and get_embeddings_status; they disagreed
    before, so status reported "not cached" for a perfectly good cache
    whenever the graph sat outside a .descry_cache directory.
    """
    if cache_dir:
        return Path(cache_dir).resolve()
    if graph_path.parent.name == ".descry_cache":
        return graph_path.parent
    return graph_path.parent / ".descry_cache"


def _pick_prompt(model, candidates: tuple[str, ...]) -> str | None:
    """First candidate the model actually registers, else None."""
    available = getattr(model, "prompts", None) or {}
    for name in candidates:
        if name in available:
            return name
    return None


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


def _load_sentence_transformer(
    model_name: str,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
):
    """Construct a SentenceTransformer with explicit trust/revision semantics.

    Registered models supply their own pinned revision and trust setting.
    Anything else - an arbitrary repo id or a local path from
    `.descry.toml` [embeddings] model - is loaded unpinned and with remote code
    disabled, since descry cannot vouch for code it has not reviewed.

    Args:
        model_name: Registry alias, HuggingFace repo id, or local path.
        revision: Overrides the registry's pinned revision.
        trust_remote_code: Overrides the registry's trust setting.
    """
    spec = resolve_model_spec(model_name)
    target = spec.repo_id if spec else model_name
    if trust_remote_code is None:
        trust_remote_code = bool(spec and spec.trust_remote_code)
    if revision is None and spec:
        revision = spec.revision
    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    try:
        return SentenceTransformer(target, **kwargs)
    except OSError as e:
        if not _is_gated_repo_error(e):
            raise
        raise EmbeddingModelGated(
            f"{target} is a gated HuggingFace model. Accept its licence at "
            f"https://huggingface.co/{target}, then authenticate with "
            "`hf auth login` (or set HF_TOKEN). To use an ungated model "
            f'instead, set [embeddings] model = "{UNGATED_FALLBACK_ALIAS}" in '
            ".descry.toml."
        ) from e


class SemanticSearcher:
    """Semantic search using sentence embeddings.

    Generates embeddings for node names and docstrings, then uses
    cosine similarity for semantic search.
    """

    # The default model, by repo id. Its dimensionality, licence, prompts and
    # trust setting live on its MODEL_REGISTRY entry, not here.
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
    ):
        """Initialize the semantic searcher.

        Args:
            graph_path: Path to codebase_graph.json
            cache_dir: Optional directory for caching embeddings
            force_rebuild: Force regeneration of embeddings even if cache exists
        """
        self.model_name = model_name or self.MODEL_NAME
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
            str(spec.query_prompt if (spec := self.spec) else None),
            str(spec.document_prompt if spec else None),
        ]
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
        expressible. Unregistered models fall back to probing.
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
        return _pick_prompt(self.model, candidates)

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
            # Temporaries are dot-prefixed and skipped here: a glob that
            # matches them lets a reader delete a file a writer is mid-rename.
            old_files = [
                f
                for pattern in ("embeddings_*.npz", "embeddings_*.json")
                for f in self.cache_dir.glob(pattern)
                if f not in keep
                and not f.name.startswith(".")
                and ".tmp." not in f.name
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
        We write to '<final>.tmp.npz' (which numpy preserves because the
        suffix is already .npz) and then rename to the final path.
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
                        "texts": self.node_texts,
                        "recipe_version": RECIPE_VERSION,
                        "model": self.model_name,
                        "revision": self._resolved_revision(),
                        "dim": int(self.embeddings.shape[1]),
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
        npz_path, json_path = self._cache_paths()

        # Fast path: cache is hot, no lock needed.
        if (
            not force_rebuild
            and npz_path.exists()
            and json_path.exists()
            and self._try_load_cache(npz_path, json_path)
        ):
            return

        # Cold path: acquire a cross-process lock, recheck, then generate.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / "embeddings.lock"
        with _file_lock(lock_path):
            if (
                not force_rebuild
                and npz_path.exists()
                and json_path.exists()
                and self._try_load_cache(npz_path, json_path)
            ):
                return

            self._generate_embeddings()
            self._atomic_save(npz_path, json_path)
            logger.info(f"Cached {len(self.node_texts)} embeddings to {npz_path}")
            self._cleanup_old_embeddings(keep={npz_path, json_path})

    def _try_load_cache(self, npz_path: Path, json_path: Path) -> bool:
        """Attempt to load embeddings from cache. Returns True on success."""
        try:
            data = np.load(npz_path, allow_pickle=False)
            embeddings = data["embeddings"]
            with open(json_path, encoding="utf-8") as f:
                sidecar = json.load(f)
            texts = sidecar["texts"]
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
            if len(texts) != embeddings.shape[0]:
                logger.warning(
                    "Embedding cache mismatch (%d texts vs %d embeddings); regenerating",
                    len(texts),
                    embeddings.shape[0],
                )
                return False
            self.embeddings = embeddings
            self.node_texts = texts
            logger.info(f"Loaded {len(self.node_texts)} embeddings from cache")
            # Deliberately no pruning here: this runs on the lock-free fast
            # path, where deleting files another process is writing is exactly
            # the race that made a concurrent index fail. Pruning happens under
            # the lock in _load_or_create_embeddings.
            return True
        except Exception as e:  # noqa: BLE001 — corrupt cache regenerates; numpy/zipfile signal corruption with open-ended types
            logger.warning(f"Cache load failed: {e}, regenerating...")
            return False

    def ensure_model(self):
        """Load the model once, even under concurrent first searches.

        DescryService shares one searcher across worker threads, so the bare
        check-then-set this replaces let several threads each construct their
        own copy of a model several hundred megabytes in size.
        """
        if self.model is not None:
            return self.model
        with self._model_lock:
            if self.model is None:
                self.model = _load_sentence_transformer(self.model_name)
        return self.model

    def _node_text(self, node: dict) -> str:
        return node_text(node)

    def _generate_embeddings(self):
        """Encode one composite text per node in a single pass."""
        logger.info("Loading embedding model...")
        self.ensure_model()

        self.node_texts = [self._node_text(n) for n in self.nodes]
        logger.info(f"Generating embeddings for {len(self.node_texts)} nodes...")

        prompt_name = self._prompt_name("document")
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

        self.ensure_model()

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
        # FINAL score. Cutting on the raw cosine first discarded nodes whose
        # boosts (worth up to +0.17) would have put them above the threshold
        # and above results that were returned.
        pool = min(len(similarities), max(limit * 3, limit))
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
            if final_score < min_score:
                continue
            results.append((node, final_score))

        # Sort by final score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


# Convenience function for one-off searches (with thread safety)
_cached_searcher = None
_searcher_lock = threading.Lock()


def _semantic_search(
    query: str,
    graph_path: str = ".descry_cache/codebase_graph.json",
    limit: int = 10,
) -> list:
    """Convenience function for semantic search (thread-safe).

    Args:
        query: Natural language search query
        graph_path: Path to graph file
        limit: Maximum results

    Returns:
        List of (node, score) tuples
    """
    global _cached_searcher

    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings not available, falling back to keyword search")
        return []

    with _searcher_lock:
        if _cached_searcher is None or str(_cached_searcher.graph_path) != graph_path:
            _cached_searcher = SemanticSearcher(graph_path)
        searcher = _cached_searcher

    return searcher.search(query, limit=limit)


def get_embeddings_status(
    graph_path: str = ".descry_cache/codebase_graph.json",
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> dict:
    """Report embedding cache state for diagnostics.

    Recomputes the cache key the searcher would use, so a cache left over from
    an older graph or a different model is reported as stale rather than as
    Ready. Reads only the JSON sidecar - never np.load of a user-controlled
    .npz.
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

    try:
        probe = SemanticSearcher.__new__(SemanticSearcher)
        probe.graph_path = graph_path
        probe.model_name = model_name or SemanticSearcher.MODEL_NAME
        expected_key = probe._cache_key()
    except OSError:
        return status
    status["expected_key"] = expected_key

    expected_npz = resolved_cache_dir / f"embeddings_{expected_key}.npz"
    expected_json = expected_npz.with_suffix(".json")
    if expected_npz.exists() and expected_json.exists():
        status["cached"] = True
        status["cache_path"] = str(expected_npz)
        try:
            with open(expected_json, encoding="utf-8") as f:
                status["node_count"] = len(json.load(f).get("texts", []))
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        return status

    # Something is cached, but not for this graph/model/recipe combination.
    others = [
        f
        for f in resolved_cache_dir.glob("embeddings_*.npz")
        if not f.name.startswith(".")
    ]
    if others:
        status["stale"] = True
        status["cache_path"] = str(max(others, key=lambda p: p.stat().st_mtime))
    return status


if __name__ == "__main__":
    # Test the embeddings module
    import sys

    if not EMBEDDINGS_AVAILABLE:
        print("Embeddings not available. Install with:")
        print("  pip install descry-codegraph[embeddings]")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = _semantic_search(query)

    print(f"\nSemantic search results for: '{query}'")
    print("-" * 50)
    for node, score in results:
        meta = node.get("metadata", {})
        print(f"[{score:.3f}] [{node['type'][:3]}] {meta.get('name', node['id'])}")
        if meta.get("docstring"):
            doc = meta["docstring"].split("\n")[0][:60]
            print(f"         {doc}...")
