"""Run the embedding model comparison on this repo's own graph.

    .venv/bin/python tests/eval/compare.py [graph_path]

Writes tests/eval/results/results.json and prints a table.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.corpus import (
    build_queries,
    doc_text,
    load_nodes,
)
from eval.metrics import (
    paired_bootstrap,
    permutation_test,
    ranks_to_metrics,
)

warnings.filterwarnings("ignore")

JINA = "jinaai/jina-code-embeddings-0.5b"
JINA_REV = "4db235132dafbe56a8b9c5f59b59795ecf58a4a7"
QWEN = "Qwen/Qwen3-Embedding-0.6B"


def norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n == 0, 1, n)


def encode(model, texts, prompt_name=None):
    kw = {"show_progress_bar": False, "convert_to_numpy": True}
    if prompt_name:
        kw["prompt_name"] = prompt_name
    t0 = time.time()
    v = model.encode(texts, **kw)
    return norm(v.astype("float32")), time.time() - t0


def rank_of_gold(doc_emb, loo_rows, queries, q_emb):
    """1-based rank of the gold node under leave-one-out substitution.

    `loo_rows` maps a node index to that node's docstring-free embedding. Only
    gold nodes need one - every other row keeps its full text - which is what
    makes this affordable on a large graph.
    """
    ranks = []
    for i, q in enumerate(queries):
        g = q["gold_index"]
        sims = doc_emb @ q_emb[i]
        if loo_rows is not None:
            sims[g] = loo_rows[g] @ q_emb[i]
        order = np.argsort(-sims)
        ranks.append(int(np.where(order == g)[0][0]) + 1)
    return ranks


def main(graph_path=".descry_cache/codebase_graph.json", only=None):
    from eval.registry import CONFIGS
    from sentence_transformers import SentenceTransformer

    nodes = load_nodes(graph_path)
    queries = build_queries(nodes)
    qtexts = [q["query"] for q in queries]
    full = [doc_text(n, with_docstring=True) for n in nodes]
    gold_idx = sorted({q["gold_index"] for q in queries})
    nodoc = [doc_text(nodes[i], with_docstring=False) for i in gold_idx]
    print(f"nodes={len(nodes)} queries={len(queries)}", flush=True)

    results, ranks_by_cfg, timings = {}, {}, {}
    loaded, model = None, None

    wanted = set(only.split(",")) if only else None
    for cfg in CONFIGS:
        if wanted and cfg.key not in wanted:
            continue
        if loaded != (cfg.repo_id, cfg.load_kwargs):
            print(f"\nloading {cfg.repo_id}", flush=True)
            del model
            model = SentenceTransformer(
                cfg.repo_id,
                revision=cfg.revision,
                trust_remote_code=cfg.trust_remote_code,
                **cfg.load_kwargs,
            )
            loaded = (cfg.repo_id, cfg.load_kwargs)
        dim = model.get_sentence_embedding_dimension()

        d_full, t1 = encode(model, full, cfg.document_prompt)
        d_nodoc, t2 = encode(model, nodoc, cfg.document_prompt)
        loo_rows = dict(zip(gold_idx, d_nodoc, strict=True))
        q_emb, _ = encode(model, qtexts, cfg.query_prompt)

        ranks = rank_of_gold(d_full, loo_rows, queries, q_emb)
        ranks_by_cfg[cfg.key] = ranks
        results[cfg.key] = ranks_to_metrics(ranks)

        # Per-language breakdown: the whole reason for running a second corpus
        # is to see whether a ranking holds outside the language it was tuned on.
        by_lang: dict[str, list[int]] = {}
        for q, r in zip(queries, ranks, strict=True):
            by_lang.setdefault(q.get("lang", "?"), []).append(r)
        results[cfg.key]["by_lang"] = {
            lang: dict(ranks_to_metrics(rs), n=len(rs))
            for lang, rs in sorted(by_lang.items())
            if len(rs) >= 30
        }
        timings[cfg.key] = {
            "encode_s": round(t1 + t2, 1),
            "dim": dim,
            "index_mb": round(len(nodes) * dim * 4 / 1e6, 2),
        }
        print(f"  {cfg.key:<18} {results[cfg.key]}", flush=True)

        # Same config without the leakage control, to show what the number
        # would be if the gold document still contained the query verbatim.
        if cfg.query_prompt is None:
            leaky = rank_of_gold(d_full, None, queries, q_emb)
            results[f"{cfg.key}_LEAKY"] = ranks_to_metrics(leaky)

    out = {"n_queries": len(queries), "metrics": results, "timings": timings}
    baseline = "jina_prompted"
    out["comparisons"] = {}
    for key, ranks in ranks_by_cfg.items():
        if key == baseline:
            continue
        e = {}
        for metric in ("R@1", "R@10", "MRR", "nDCG@10"):
            obs, lo, hi = paired_bootstrap(ranks_by_cfg[baseline], ranks, metric)
            p = permutation_test(ranks_by_cfg[baseline], ranks, metric)
            e[metric] = {
                "delta": round(obs, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "p": round(p, 4),
            }
        out["comparisons"][f"{key} - {baseline}"] = e

    out["corpus"] = str(Path(graph_path).resolve())
    out["nodes"] = len(nodes)
    label = Path(graph_path).resolve().parent.parent.name or "corpus"
    dest = Path(__file__).parent / "results"
    dest.mkdir(exist_ok=True)
    target = dest / f"results-{label}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {target}")
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
