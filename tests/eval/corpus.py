"""Graph -> query set and document texts for the embedding comparison.

Independent of SemanticSearcher: the harness must not inherit production's
min_score cut or its cache, or the comparison would be measuring those instead
of the model. Document text is production's own; see doc_text.
"""

import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from descry._graph import load_graph_with_schema
from descry.embeddings import DOCSTRING_CHAR_LIMIT, code_excerpt, node_text

WORD_RE = re.compile(r"[A-Za-z]+")
_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)(?:::|$)")


def language_of(node_id: str) -> str:
    """File extension of the node's defining file, for per-language breakdowns."""
    m = _EXT_RE.search(node_id.split("::")[0])
    return m.group(1) if m else "?"


def load_nodes(graph_path: str | Path) -> list[dict]:
    return load_graph_with_schema(graph_path)["nodes"]


def doc_text(
    node: dict, *, with_docstring: bool = True, source_root: str | Path | None = None
) -> str:
    """Index-side text, delegated to production.

    Shared rather than reimplemented so the leave-one-out substitute cannot
    drift from the index it replaces a row in.

    `source_root` is the directory node ids are relative to; with it the text
    carries the node's code excerpt, as the production index does.
    """
    return node_text(
        node,
        include_docstring=with_docstring,
        code=code_excerpt(node, source_root),
    )


def build_queries(
    nodes: list[dict], *, min_words: int = 6, seed: int = 0
) -> list[dict]:
    """Docstring-as-query, gold = the node that owns it.

    Filters, in order: a docstring must exist; its first line must not be
    shared with another node (otherwise two documents are correct and the
    metric punishes a right answer); and it must carry enough words to be
    answerable at all.
    """
    docstrings = [(n["metadata"].get("docstring") or "").strip() for n in nodes]
    first_lines = [d.splitlines()[0].strip() if d else "" for d in docstrings]
    first_line_counts = Counter(first_lines)

    queries = []
    for idx, (n, d, fl) in enumerate(zip(nodes, docstrings, first_lines, strict=True)):
        if not d or first_line_counts[fl] > 1:
            continue
        if len(WORD_RE.findall(fl)) < min_words:
            continue
        queries.append(
            {
                "gold_index": idx,
                "query": d[:DOCSTRING_CHAR_LIMIT],
                "gold_id": n["id"],
                "type": n["type"],
                "lang": language_of(n["id"]),
            }
        )
    random.Random(seed).shuffle(queries)
    return queries
