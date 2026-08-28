# Embedding model comparison

Answers one question with numbers instead of vendor claims: for **this**
repository's graph, does a different embedding model retrieve better?

```bash
# every config, this repo's own graph
.venv/bin/python tests/eval/compare.py

# selected configs against another project's graph
.venv/bin/python tests/eval/compare.py /path/to/.descry_cache/codebase_graph.json \
    jina_prompted,qwen3_plain,gemma_code
```

Writes `tests/eval/results/results-<project>.json`, so runs against different
graphs coexist. Read-only with respect to the target project: it never writes
into that project's cache directory. CPU-only; roughly 3 minutes per config per
10k nodes.

**Run it on more than one repository.** A ranking measured on a single codebase
is a prior, not a fact - descry indexes Rust, TypeScript, Java, Go and C++, and
this repo is pure Python. Results carry a `by_lang` breakdown (languages with at
least 30 queries) precisely so a model that wins overall but loses on the
language you care about is visible rather than averaged away.

## Protocol

**Ground truth.** A symbol's docstring is a description of that symbol written
by someone looking at it, so the docstring is the query and the owning node is
the single relevant document. No annotator required.

From 1140 nodes: drop nodes with no docstring, drop docstrings whose first line
is shared with another node (two documents would be correct and the metric
would punish a right answer), drop first lines under six words. Leaves **405
queries** — 244 Method, 109 Function, 52 Class. Shuffled with `Random(0)`.

**Leakage control is the whole ballgame.** The query is the gold node's
docstring, and that docstring is also inside the gold node's indexed text.
Uncontrolled, every model scores near 1.0 on its own document — measured R@1
0.933 rather than 0.360. For each query the harness swaps the gold row for that
node's docstring-free embedding, so the query is never inside its own document.

**Prompts.** Both candidate models are instruction-tuned. Evaluating them
without their task prompts measures the wrong thing, so each config encodes
documents and queries through the model's own prompt names.

## Configs

`registry.py` lists model/prompt combinations. Prompt choice is not incidental:
embeddinggemma scores R@1 0.479 through its code-retrieval prompt and 0.385
through its generic retrieval prompts, and Qwen3 scores 0.430 with no prompt but
0.360 through its own. Any new entry should be measured both ways before being
copied into `descry.embeddings.MODEL_REGISTRY`.

## Independent of `SemanticSearcher`, but not of `node_text`

The harness never constructs a `SemanticSearcher`: doing so would inherit its
cache, its `min_score` cut and its cache eviction, and the comparison would
measure those instead of the model.

Document text is a different matter. `corpus.doc_text` delegates to
`descry.embeddings.node_text`, because the leave-one-out substitute has to be
*the same assembly minus the docstring* as the row it replaces. A private copy
would let the two drift, and the leakage control would then quietly measure
something the index never contained.

Assemblies being *evaluated* rather than shipped — `composite_texts`, which
reproduces the pre-2026 three-vector average — stay here, since scoring a
candidate against the shipped one is the point.
