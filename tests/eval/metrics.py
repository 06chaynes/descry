"""Retrieval metrics for the embedding model comparison.

Single-relevant-document setting: each query has exactly one gold node, so
nDCG@k reduces to 1/log2(rank+1) and MRR to 1/rank.
"""

import math
import random


def ranks_to_metrics(ranks: list[int], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """ranks: 1-based rank of the gold document, or 0 when it is unranked."""
    n = len(ranks)
    out = {}
    for k in ks:
        out[f"R@{k}"] = sum(1 for r in ranks if 0 < r <= k) / n
    out["MRR"] = sum(1.0 / r for r in ranks if r > 0) / n
    out["nDCG@10"] = sum(1.0 / math.log2(r + 1) for r in ranks if 0 < r <= 10) / n
    return out


def per_query_score(rank: int, metric: str) -> float:
    if metric.startswith("R@"):
        k = int(metric[2:])
        return 1.0 if 0 < rank <= k else 0.0
    if metric == "MRR":
        return 1.0 / rank if rank > 0 else 0.0
    if metric == "nDCG@10":
        return 1.0 / math.log2(rank + 1) if 0 < rank <= 10 else 0.0
    raise ValueError(metric)


def paired_bootstrap(ranks_a, ranks_b, metric, n=5000, seed=0):
    """Paired bootstrap CI on (b - a). Positive favours b."""
    rng = random.Random(seed)
    a = [per_query_score(r, metric) for r in ranks_a]
    b = [per_query_score(r, metric) for r in ranks_b]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    n_q = len(diffs)
    obs = sum(diffs) / n_q
    boot = []
    for _ in range(n):
        s = sum(diffs[rng.randrange(n_q)] for _ in range(n_q))
        boot.append(s / n_q)
    boot.sort()
    return obs, boot[int(0.025 * n)], boot[int(0.975 * n)]


def permutation_test(ranks_a, ranks_b, metric, n=5000, seed=0):
    """Two-sided paired permutation test (sign-flip) on the mean difference."""
    rng = random.Random(seed)
    a = [per_query_score(r, metric) for r in ranks_a]
    b = [per_query_score(r, metric) for r in ranks_b]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    obs = abs(sum(diffs) / len(diffs))
    hits = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / len(diffs)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)
