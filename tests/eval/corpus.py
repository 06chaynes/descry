"""Graph -> query set and document texts for the embedding comparison.

Kept deliberately independent of descry.embeddings: the harness must not
inherit production's composite, its min_score cut, or its cache, or the
comparison would be measuring those instead of the model.
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from descry.embeddings import DOCSTRING_CHAR_LIMIT, node_text

DOC_CHAR_LIMIT = DOCSTRING_CHAR_LIMIT
WORD_RE = re.compile(r"[A-Za-z]+")
_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)(?:::|$)")


def language_of(node_id: str) -> str:
    """File extension of the node's defining file, for per-language breakdowns."""
    m = _EXT_RE.search(node_id.split("::")[0])
    return m.group(1) if m else "?"


def load_nodes(graph_path: str | Path) -> list[dict]:
    with open(graph_path, encoding="utf-8") as f:
        return json.load(f)["nodes"]


def doc_text(node: dict, *, with_docstring: bool = True) -> str:
    """Index-side text, delegated to production.

    Shared rather than reimplemented so the leave-one-out substitute cannot
    drift from the index it replaces a row in. Alternative assemblies being
    *evaluated* (see composite_texts) still live here - the harness has to be
    able to score a candidate against the shipped one.
    """
    return node_text(node, include_docstring=with_docstring)


def composite_texts(node: dict, *, with_docstring: bool = True) -> tuple[str, str, str]:
    """The three texts production encodes separately, incl. its name fallbacks."""
    meta = node.get("metadata") or {}
    name = meta.get("name") or node["id"].split("::")[-1]
    sig = meta.get("signature") or ""
    doc = (meta.get("docstring") or "")[:DOC_CHAR_LIMIT] if with_docstring else ""
    return name, (sig or name), (doc or name)


def build_queries(
    nodes: list[dict], *, min_words: int = 6, seed: int = 0
) -> list[dict]:
    """Docstring-as-query, gold = the node that owns it.

    Filters, in order: a docstring must exist; its first line must not be
    shared with another node (otherwise two documents are correct and the
    metric punishes a right answer); and it must carry enough words to be
    answerable at all.
    """
    first_line_counts: dict[str, int] = {}
    for n in nodes:
        d = ((n.get("metadata") or {}).get("docstring") or "").strip()
        if d:
            fl = d.splitlines()[0].strip()
            first_line_counts[fl] = first_line_counts.get(fl, 0) + 1

    queries = []
    for idx, n in enumerate(nodes):
        d = ((n.get("metadata") or {}).get("docstring") or "").strip()
        if not d:
            continue
        fl = d.splitlines()[0].strip()
        if first_line_counts.get(fl, 0) > 1:
            continue
        if len(WORD_RE.findall(fl)) < min_words:
            continue
        queries.append(
            {
                "gold_index": idx,
                "query": d[:DOC_CHAR_LIMIT],
                "gold_id": n["id"],
                "type": n.get("type", ""),
                "lang": language_of(n["id"]),
            }
        )
    random.Random(seed).shuffle(queries)
    return queries
