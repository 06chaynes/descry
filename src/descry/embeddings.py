#!/usr/bin/env python3
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
from pathlib import Path
from typing import Optional


try:
    import fcntl as _fcntl  # Unix only
except ImportError:
    _fcntl = None


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout: float = 600.0):
    """Cross-process advisory lock.

    Uses fcntl.flock on Unix; falls back to atomic-create sentinel on
    platforms without fcntl (Windows).
    """
    if _fcntl is not None:
        with open(lock_path, "w") as f:
            _fcntl.flock(f, _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(f, _fcntl.LOCK_UN)
        try:
            lock_path.unlink()
        except OSError:
            pass
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
                )
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


logger = logging.getLogger(__name__)

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

    The default model (Jina code embeddings) requires remote-code loading; its
    revision is pinned for supply-chain integrity. User-supplied models (from
    `.descry.toml` [embeddings] model) default to trust_remote_code=False.

    Args:
        model_name: Model repo id or local path.
        revision: Pinned git sha for HF downloads; defaults to the pinned
            DEFAULT_MODEL_REVISION when model_name matches MODEL_NAME.
        trust_remote_code: Whether to allow model-provided Python code.
            Defaults to True only for the bundled default model; False for
            all user-supplied models.
    """
    if trust_remote_code is None:
        trust_remote_code = model_name == SemanticSearcher.MODEL_NAME
    if revision is None and model_name == SemanticSearcher.MODEL_NAME:
        revision = SemanticSearcher.DEFAULT_MODEL_REVISION
    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    return SentenceTransformer(model_name, **kwargs)


class SemanticSearcher:
    """Semantic search using sentence embeddings.

    Generates embeddings for node names and docstrings, then uses
    cosine similarity for semantic search.
    """

    # Code-optimized embedding model (896-dim, 494M params)
    # Significantly better code search quality than general-purpose models.
    # This model requires `trust_remote_code=True`; revision is pinned for
    # supply-chain integrity (A.3 Option B).
    MODEL_NAME = "jinaai/jina-code-embeddings-0.5b"
    # Pinned HF revision (git sha) for the default model. Update intentionally
    # when upgrading; cache auto-invalidates via model-name hash in cache key.
    DEFAULT_MODEL_REVISION = "4db235132dafbe56a8b9c5f59b59795ecf58a4a7"

    def __init__(
        self,
        graph_path: str,
        cache_dir: Optional[str] = None,
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

        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embeddings require sentence-transformers and numpy. "
                "Install with: pip install descry-codegraph[embeddings]"
            )

        # Resolve to absolute path to avoid nested directory issues when CWD is inside cache
        self.graph_path = Path(graph_path).resolve()
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        elif self.graph_path.parent.name == ".descry_cache":
            # Graph is already in cache dir, use it directly
            self.cache_dir = self.graph_path.parent
        else:
            self.cache_dir = self.graph_path.parent / ".descry_cache"

        # Load graph (B.6: schema-checked)
        from descry._graph import load_graph_with_schema

        self.data = load_graph_with_schema(self.graph_path)
        self.nodes = self.data["nodes"]

        # Load or create embeddings
        self.model = None
        self.embeddings = None
        self.node_texts = None
        self._load_or_create_embeddings(force_rebuild=force_rebuild)

    def _cache_key(self) -> str:
        """Compute content-addressed cache key.

        Composition: int(mtime) + sha256[:16] of graph bytes + sha256[:8] of
        model name. Model-hash component ensures A.3 model swap auto-invalidates
        old caches.
        """
        graph_mtime = int(self.graph_path.stat().st_mtime)
        graph_hash = hashlib.sha256(self.graph_path.read_bytes()).hexdigest()[:16]
        model_hash = hashlib.sha256(self.model_name.encode("utf-8")).hexdigest()[:8]
        return f"{graph_mtime}_{graph_hash}_{model_hash}"

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
            old_files = [
                f for f in self.cache_dir.glob("embeddings_*.npz") if f not in keep
            ] + [f for f in self.cache_dir.glob("embeddings_*.json") if f not in keep]
            for old_file in old_files:
                logger.info(f"Removing stale embedding cache: {old_file.name}")
                old_file.unlink()
            if old_files:
                logger.info(f"Cleaned up {len(old_files)} old embedding file(s)")
        except Exception as e:
            logger.warning(f"Failed to cleanup old embeddings: {e}")

    def _atomic_save(self, npz_path: Path, json_path: Path) -> None:
        """Write .npz + JSON sidecar atomically (tmp then rename).

        Note: np.savez appends '.npz' to paths that don't end in '.npz'.
        We write to '<final>.tmp.npz' (which numpy preserves because the
        suffix is already .npz) and then rename to the final path.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npz_tmp = npz_path.with_suffix(".tmp.npz")
        json_tmp = json_path.with_suffix(".tmp.json")
        cleanup = [npz_tmp, json_tmp]
        try:
            np.savez(npz_tmp, embeddings=self.embeddings)
            with open(json_tmp, "w") as f:
                json.dump({"texts": self.node_texts}, f)
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
            with open(json_path) as f:
                sidecar = json.load(f)
            texts = sidecar["texts"]
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
            self._cleanup_old_embeddings(keep={npz_path, json_path})
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}, regenerating...")
            return False

    def _generate_embeddings(self):
        """Generate weighted composite embeddings for all nodes.

        Uses separate embeddings for name, signature, and docstring with
        weighted combination. This improves semantic search by:
        - Prioritizing docstring content for meaning-based queries
        - Still matching on symbol names and signatures
        - Including caller context for disambiguation

        Weights: name=0.2, signature=0.3, docstring=0.5
        """
        logger.info("Loading embedding model...")
        self.model = _load_sentence_transformer(self.model_name)

        # Collect texts for each component
        names = []
        signatures = []
        docstrings = []
        self.node_texts = []  # Combined text for cache identification

        for node in self.nodes:
            meta = node.get("metadata", {})

            name = meta.get("name", "") or node["id"].split("::")[-1]
            sig = meta.get("signature", "")
            doc = meta.get("docstring", "")[:500]

            names.append(name)
            signatures.append(sig if sig else name)  # Fall back to name if no signature
            docstrings.append(doc if doc else name)  # Fall back to name if no docstring

            # Combined text for cache identification
            combined = f"{name} {sig} {doc}".strip()
            self.node_texts.append(combined if combined else node["id"])

        logger.info(
            f"Generating weighted embeddings for {len(self.node_texts)} nodes..."
        )

        # Generate embeddings for each component
        # batch_size keeps memory usage reasonable
        logger.info("  Encoding names...")
        name_embeddings = self.model.encode(
            names, show_progress_bar=False, convert_to_numpy=True
        )

        logger.info("  Encoding signatures...")
        sig_embeddings = self.model.encode(
            signatures, show_progress_bar=False, convert_to_numpy=True
        )

        logger.info("  Encoding docstrings...")
        doc_embeddings = self.model.encode(
            docstrings, show_progress_bar=False, convert_to_numpy=True
        )

        # Weighted combination: name=0.2, signature=0.3, docstring=0.5
        # Docstring gets highest weight for semantic/meaning-based queries
        logger.info("  Computing weighted composites...")
        self.embeddings = (
            0.2 * name_embeddings + 0.3 * sig_embeddings + 0.5 * doc_embeddings
        )

        # Normalize the combined embeddings for consistent cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        self.embeddings = self.embeddings / norms

        logger.info("Weighted embeddings generated successfully")

    def search(self, query: str, limit: int = 10, min_score: float = 0.3) -> list:
        """Search for nodes semantically similar to the query.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_score: Minimum cosine similarity score (0-1)

        Returns:
            List of (node, score) tuples sorted by relevance

        Scoring includes:
        - Cosine similarity (primary)
        - In-degree boost (symbols with more callers ranked higher)
        - Type preference (Functions/Classes over Constants)
        """
        import math

        if self.model is None:
            self.model = _load_sentence_transformer(self.model_name)

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get candidates above threshold
        top_indices = np.argsort(similarities)[::-1]
        candidates = []
        for idx in top_indices:
            base_score = float(similarities[idx])
            if base_score < min_score:
                break
            # Collect more candidates than needed for re-ranking
            if len(candidates) >= limit * 3:
                break
            candidates.append((idx, base_score))

        # Re-rank with in-degree boost and type preference
        results = []
        for idx, base_score in candidates:
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
) -> dict:
    """Get embeddings status for diagnostics.

    Reports whether a cache exists for the current graph and counts node
    texts via the JSON sidecar (no np.load of user-controlled .npz).
    """
    graph_path = Path(graph_path).resolve()
    cache_dir = graph_path.parent

    status = {
        "available": EMBEDDINGS_AVAILABLE,
        "cached": False,
        "node_count": 0,
        "stale": False,
        "cache_path": None,
    }

    if not EMBEDDINGS_AVAILABLE or not graph_path.exists():
        return status

    # Any embedding file (npz + matching json sidecar) counts; report the
    # most recent pair. No np.load — just read the JSON sidecar for count.
    npz_files = sorted(
        cache_dir.glob("embeddings_*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for npz_file in npz_files:
        sidecar = npz_file.with_suffix(".json")
        if sidecar.exists():
            status["cached"] = True
            status["cache_path"] = str(npz_file)
            try:
                with open(sidecar) as f:
                    texts = json.load(f).get("texts", [])
                status["node_count"] = len(texts)
            except Exception:
                pass
            return status

    # Fallback: orphan npz (no sidecar) is stale
    if npz_files:
        status["stale"] = True
        status["cache_path"] = str(npz_files[0])

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
