"""Score `search` end to end on prompt-like questions with known answers.

    .venv/bin/python tests/eval/search_eval.py OUT.json name=/path/to/repo ...

Each question in `search_questions.json` names its repository, the question as
an agent would send it, and the node IDs that answer it. The question's words
are the search terms, as a caller passing a whole prompt sends them. Each
repository must already be indexed with embeddings. Writes the rank of the
first answer per question (0 where none is in the results) to OUT.json, and
prints R@1, R@5, R@10 and MRR, all and per repository. Two runs compare with
`compare` below: `search_eval.py --compare A.json B.json`.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import paired_bootstrap, permutation_test, ranks_to_metrics

from descry.handlers import DescryConfig, DescryService

QUESTIONS = Path(__file__).with_name("search_questions.json")
LIMIT = 12


async def ranks(roots: dict) -> list:
    questions = json.loads(QUESTIONS.read_text())
    services = {
        name: DescryService(DescryConfig(project_root=root, enable_scip=False))
        for name, root in roots.items()
    }
    out = []
    for q in questions:
        svc = services[q["repo"]]
        # Wait for the model before the first question, as an index does.
        await svc._get_semantic_searcher()
        terms = re.findall(r"[\w'-]+", q["question"])
        answer = await svc.search_answer(terms, limit=LIMIT)
        found = [r["id"] for r in answer.get("results", [])]
        rank = next((i + 1 for i, n in enumerate(found) if n in q["answers"]), 0)
        out.append({**q, "rank": rank, "method": answer.get("method")})
    return out


def report(rows: list) -> None:
    for repo in sorted({r["repo"] for r in rows}) + [None]:
        picked = [r["rank"] for r in rows if repo in (None, r["repo"])]
        m = ranks_to_metrics(picked)
        print(
            f"{repo or 'all':10} n={len(picked):3} "
            + " ".join(f"{k}={v:.3f}" for k, v in m.items())
        )


def compare(a_path: str, b_path: str) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    assert [r["question"] for r in a] == [r["question"] for r in b]
    ra, rb = [r["rank"] for r in a], [r["rank"] for r in b]
    for metric in ("R@1", "R@5", "MRR"):
        obs, lo, hi = paired_bootstrap(ra, rb, metric)
        p = permutation_test(ra, rb, metric)
        print(f"{metric}: b - a = {obs:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit()
    roots = dict(arg.split("=", 1) for arg in sys.argv[2:])
    rows = asyncio.run(ranks({k: Path(v) for k, v in roots.items()}))
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
    report(rows)
