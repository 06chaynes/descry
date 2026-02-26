#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sentence-transformers>=2.2.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Optional semantic search with embeddings for codegraph.

Uses sentence-transformers for embedding generation and numpy for similarity.
Falls back to TF-IDF search if dependencies are not available.

Usage:
    # Check if available
    if embeddings_available():
        searcher = SemanticSearcher(graph_path)
        results = searcher.search("authentication logic", limit=10)
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

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


class SemanticSearcher:
    """Semantic search using sentence embeddings.

    Generates embeddings for node names and docstrings, then uses
    cosine similarity for semantic search.
    """

    # Code-optimized embedding model (896-dim, 494M params)
    # Significantly better code search quality than general-purpose models
    MODEL_NAME = "jinaai/jina-code-embeddings-0.5b"

    def __init__(self, graph_path: str, cache_dir: Optional[str] = None, force_rebuild: bool = False):
        """Initialize the semantic searcher.

        Args:
            graph_path: Path to codebase_graph.json
            cache_dir: Optional directory for caching embeddings
            force_rebuild: Force regeneration of embeddings even if cache exists
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embeddings require sentence-transformers and numpy. "
                "Install with: just codegraph-install-embeddings"
            )

        # Resolve to absolute path to avoid nested directory issues when CWD is inside cache
        self.graph_path = Path(graph_path).resolve()
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        elif self.graph_path.parent.name == ".codegraph_cache":
            # Graph is already in cache dir, use it directly
            self.cache_dir = self.graph_path.parent
        else:
            self.cache_dir = self.graph_path.parent / ".codegraph_cache"

        # Load graph
        with open(self.graph_path) as f:
            self.data = json.load(f)
        self.nodes = self.data["nodes"]

        # Load or create embeddings
        self.model = None
        self.embeddings = None
        self.node_texts = None
        self._load_or_create_embeddings(force_rebuild=force_rebuild)

    def _get_cache_path(self) -> Path:
        """Get path for cached embeddings."""
        graph_mtime = int(self.graph_path.stat().st_mtime)
        return self.cache_dir / f"embeddings_{graph_mtime}.npz"

    def _cleanup_old_embeddings(self, current_cache_path: Path):
        """Remove old embedding cache files, keeping only the current one.

        Args:
            current_cache_path: The current/valid cache file to keep
        """
        try:
            old_files = [
                f for f in self.cache_dir.glob("embeddings_*.npz")
                if f != current_cache_path
            ]
            for old_file in old_files:
                logger.info(f"Removing stale embedding cache: {old_file.name}")
                old_file.unlink()
            if old_files:
                logger.info(f"Cleaned up {len(old_files)} old embedding file(s)")
        except Exception as e:
            logger.warning(f"Failed to cleanup old embeddings: {e}")

    def _load_or_create_embeddings(self, force_rebuild: bool = False):
        """Load embeddings from cache or create new ones.

        Args:
            force_rebuild: Skip cache and regenerate embeddings
        """
        cache_path = self._get_cache_path()

        if not force_rebuild and cache_path.exists():
            # Load from cache
            try:
                data = np.load(cache_path, allow_pickle=True)
                self.embeddings = data["embeddings"]
                self.node_texts = data["texts"].tolist()
                logger.info(f"Loaded {len(self.node_texts)} embeddings from cache")
                # Cleanup old files on successful load
                self._cleanup_old_embeddings(cache_path)
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, regenerating...")

        # Generate new embeddings
        self._generate_embeddings()

        # Save to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            embeddings=self.embeddings,
            texts=np.array(self.node_texts, dtype=object),
        )
        logger.info(f"Cached {len(self.node_texts)} embeddings to {cache_path}")

        # Cleanup old embedding files after creating new one
        self._cleanup_old_embeddings(cache_path)

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
        self.model = SentenceTransformer(self.MODEL_NAME)

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

            # Add caller context to help disambiguation
            # e.g., "Token" used in auth context vs lexer context
            in_degree = meta.get("in_degree", 0)
            if in_degree > 0 and doc:
                # Note: We can't get actual caller names without loading the graph edges
                # So we'll use the node type and path for context
                node_id = node.get("id", "")
                if "auth" in node_id.lower():
                    doc += " Used in authentication context."
                elif "token" in node_id.lower() and "lex" in node_id.lower():
                    doc += " Used in lexer/tokenizer context."
                elif "deploy" in node_id.lower():
                    doc += " Used in deployment context."

            names.append(name)
            signatures.append(sig if sig else name)  # Fall back to name if no signature
            docstrings.append(doc if doc else name)  # Fall back to name if no docstring

            # Combined text for cache identification
            combined = f"{name} {sig} {doc}".strip()
            self.node_texts.append(combined if combined else node["id"])

        logger.info(f"Generating weighted embeddings for {len(self.node_texts)} nodes...")

        # Generate embeddings for each component
        # batch_size keeps memory usage reasonable
        logger.info("  Encoding names...")
        name_embeddings = self.model.encode(names, show_progress_bar=False, convert_to_numpy=True)

        logger.info("  Encoding signatures...")
        sig_embeddings = self.model.encode(signatures, show_progress_bar=False, convert_to_numpy=True)

        logger.info("  Encoding docstrings...")
        doc_embeddings = self.model.encode(docstrings, show_progress_bar=False, convert_to_numpy=True)

        # Weighted combination: name=0.2, signature=0.3, docstring=0.5
        # Docstring gets highest weight for semantic/meaning-based queries
        logger.info("  Computing weighted composites...")
        self.embeddings = (
            0.2 * name_embeddings +
            0.3 * sig_embeddings +
            0.5 * doc_embeddings
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
            self.model = SentenceTransformer(self.MODEL_NAME)

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

    def find_similar(self, node_id: str, limit: int = 5) -> list:
        """Find nodes semantically similar to a given node.

        Args:
            node_id: The node ID to find similar nodes for
            limit: Maximum number of results

        Returns:
            List of (node, score) tuples sorted by similarity
        """
        # Find the node index
        node_idx = None
        for i, node in enumerate(self.nodes):
            if node["id"] == node_id:
                node_idx = i
                break

        if node_idx is None:
            return []

        # Use that node's embedding as query
        node_embedding = self.embeddings[node_idx]

        # Compute similarities
        similarities = np.dot(self.embeddings, node_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(node_embedding)
        )

        # Get top results (excluding the query node itself)
        top_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in top_indices:
            if idx == node_idx:
                continue
            score = float(similarities[idx])
            results.append((self.nodes[idx], score))
            if len(results) >= limit:
                break

        return results


# Convenience function for one-off searches (with thread safety)
_cached_searcher = None
_searcher_lock = threading.Lock()


def semantic_search(
    query: str,
    graph_path: str = ".codegraph_cache/codebase_graph.json",
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


def get_embeddings_status(graph_path: str = ".codegraph_cache/codebase_graph.json") -> dict:
    """Get embeddings status for diagnostics.

    Args:
        graph_path: Path to the graph file

    Returns:
        Dictionary with status information
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

    if not EMBEDDINGS_AVAILABLE:
        return status

    if not graph_path.exists():
        return status

    # Find embedding cache file
    graph_mtime = int(graph_path.stat().st_mtime)
    cache_path = cache_dir / f"embeddings_{graph_mtime}.npz"

    if cache_path.exists():
        status["cached"] = True
        status["cache_path"] = str(cache_path)
        try:
            data = np.load(cache_path, allow_pickle=True)
            status["node_count"] = len(data["texts"])
        except Exception:
            pass
    else:
        # Check for any embedding files (might be stale)
        embedding_files = list(cache_dir.glob("embeddings_*.npz"))
        if embedding_files:
            status["stale"] = True
            latest = max(embedding_files, key=lambda p: p.stat().st_mtime)
            status["cache_path"] = str(latest)

    return status


if __name__ == "__main__":
    # Test the embeddings module
    import sys

    if not EMBEDDINGS_AVAILABLE:
        print("Embeddings not available. Install with:")
        print("  just codegraph-install-embeddings")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = semantic_search(query)

    print(f"\nSemantic search results for: '{query}'")
    print("-" * 50)
    for node, score in results:
        meta = node.get("metadata", {})
        print(f"[{score:.3f}] [{node['type'][:3]}] {meta.get('name', node['id'])}")
        if meta.get("docstring"):
            doc = meta["docstring"].split("\n")[0][:60]
            print(f"         {doc}...")
