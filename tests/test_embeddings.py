"""Tests for descry.embeddings.

Runs without downloading a model: every test injects a deterministic stub over
`_load_sentence_transformer`, the single seam through which the searcher builds
a SentenceTransformer.
"""

import hashlib
import json
import os
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from descry import embeddings as E
from descry._graph import GraphSchemaError
from descry.embeddings import (
    EmbeddingCacheMismatch,
    SemanticSearcher,
    _file_lock,
    get_embeddings_status,
)

JINA_PROMPTS = {
    "nl2code_query": "Find the most relevant code snippet given the following query:\n",
    "nl2code_document": "Candidate code snippet:\n",
}


class FakeSentenceTransformer:
    """Deterministic stand-in. Vectors depend on text AND prompt, so a test can
    prove a prompt was actually applied rather than silently dropped."""

    def __init__(self, dim=8, prompts=None, recorder=None, load_delay=0.0):
        if load_delay:
            time.sleep(load_delay)
        self.dim = dim
        self.prompts = dict(prompts if prompts is not None else JINA_PROMPTS)
        self.recorder = recorder
        if recorder is not None:
            recorder["loads"] += 1

    def encode(self, texts, **kw):
        prompt = kw.get("prompt_name")
        if self.recorder is not None:
            self.recorder["encodes"].append((list(texts), prompt))
        out = np.empty((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            seed = hashlib.sha256(f"{prompt}|{t}".encode()).digest()
            v = np.frombuffer(seed * ((self.dim // 32) + 1), dtype="uint8")[: self.dim]
            vec = v.astype("float32") - 127.5
            out[i] = vec / (np.linalg.norm(vec) or 1.0)
        return out

    def get_sentence_embedding_dimension(self):
        return self.dim


@pytest.fixture
def fake_model(monkeypatch):
    """Patch the model factory; yields a recorder of loads and encode calls."""
    rec = {"loads": 0, "encodes": [], "prompts": None, "delay": 0.0}

    def factory(model_name, revision=None):
        return FakeSentenceTransformer(
            prompts=rec["prompts"],
            recorder=rec,
            load_delay=rec["delay"],
        )

    monkeypatch.setattr(E, "_load_sentence_transformer", factory)
    monkeypatch.setattr(E, "EMBEDDINGS_AVAILABLE", True)
    return rec


@pytest.fixture
def graph_writer(tmp_path):
    """Factory writing a graph into a .descry_cache dir; call again to rewrite."""

    def write(nodes=None, dirname=".descry_cache"):
        cache = tmp_path / dirname
        cache.mkdir(exist_ok=True)
        if nodes is None:
            nodes = [
                {
                    "id": "f.py::alpha",
                    "type": "Function",
                    "metadata": {
                        "name": "alpha",
                        "signature": "def alpha(a)",
                        "docstring": "Add two numbers together.",
                        "in_degree": 3,
                    },
                },
                {
                    "id": "f.py::Beta",
                    "type": "Class",
                    "metadata": {"name": "Beta", "docstring": "A container class."},
                },
                {
                    "id": "f.py::gamma",
                    "type": "Method",
                    "metadata": {"name": "gamma", "signature": "def gamma(self)"},
                },
                {"id": "f.py", "type": "File", "metadata": {}},
            ]
        p = cache / "codebase_graph.json"
        p.write_text(json.dumps({"schema_version": 2, "nodes": nodes, "edges": []}))
        return p

    return write


PY_SOURCE = '''def alpha(a, b):
    """Add two numbers together.

    Both must be ints.
    """
    return a + b
'''
PY_DOCSTRING = "Add two numbers together.\n\nBoth must be ints."
PY_EXCERPT = 'def alpha(a, b):\n\n    """\n    return a + b'

RS_SOURCE = """/// Strip credentials from the environment.
/// Returns the clean map.
fn scrub(env: &Env) -> Env {
    env.filter()
}
"""
RS_DOCSTRING = "Strip credentials from the environment.\nReturns the clean map."
RS_EXCERPT = "fn scrub(env: &Env) -> Env {\n    env.filter()\n}"


def _node(node_id, lineno=None, end_lineno=None, docstring=""):
    meta = {"name": node_id.split("::")[-1], "docstring": docstring}
    if lineno is not None:
        meta["lineno"] = lineno
    if end_lineno is not None:
        meta["end_lineno"] = end_lineno
    return {"id": node_id, "type": "Function", "metadata": meta}


@pytest.fixture
def source_nodes(tmp_path):
    """Write a Python and a Rust file under tmp_path; returns their nodes."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(PY_SOURCE)
    (tmp_path / "src" / "lib.rs").write_text(RS_SOURCE)
    return [
        _node("FILE:src/mod.py::alpha", 1, 6, PY_DOCSTRING),
        _node("FILE:src/lib.rs::scrub", 3, 5, RS_DOCSTRING),
        {"id": "FILE:src/mod.py", "type": "File", "metadata": {}},
    ]


def _peak_lock_holders(lock_path):
    """Most of four contending threads ever inside `_file_lock` at once."""
    inside, peak, guard = [0], [0], threading.Lock()

    def worker():
        with _file_lock(lock_path, timeout=10.0):
            with guard:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.05)
            with guard:
                inside[0] -= 1

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20.0)
    return peak[0]


class TestFileLock:
    def test_lock_file_survives_release(self, tmp_path):
        p = tmp_path / "x.lock"
        with _file_lock(p):
            pass
        assert p.exists(), "unlinking the lock destroys mutual exclusion"

    def test_lock_inode_is_stable_across_acquisitions(self, tmp_path):
        p = tmp_path / "x.lock"
        seen = []
        for _ in range(2):
            with _file_lock(p):
                seen.append(os.stat(p).st_ino)
        assert seen[0] == seen[1]

    def test_timeout_is_honoured(self, tmp_path):
        p = tmp_path / "x.lock"
        started, released = threading.Event(), threading.Event()

        def holder():
            with _file_lock(p):
                started.set()
                released.wait(5.0)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        assert started.wait(5.0)
        with pytest.raises(TimeoutError), _file_lock(p, timeout=0.2):
            pass
        released.set()
        t.join(5.0)

    def test_mutual_exclusion_under_contention(self, tmp_path):
        assert _peak_lock_holders(tmp_path / "x.lock") == 1


class TestCacheKeyIdentity:
    def test_key_changes_with_graph_content(self, fake_model, graph_writer):
        g = graph_writer()
        k1 = SemanticSearcher(str(g))._cache_key()
        graph_writer([{"id": "z::q", "type": "Function", "metadata": {"name": "q"}}])
        assert SemanticSearcher(str(g))._cache_key() != k1

    def test_key_changes_with_model_name(self, fake_model, graph_writer):
        g = graph_writer()
        a = SemanticSearcher(str(g), model_name="m/a")._cache_key()
        b = SemanticSearcher(str(g), model_name="m/b")._cache_key()
        assert a != b

    def test_key_changes_with_pinned_revision(
        self, fake_model, graph_writer, monkeypatch
    ):
        g = graph_writer()
        before = SemanticSearcher(str(g))._cache_key()
        monkeypatch.setattr(SemanticSearcher, "DEFAULT_MODEL_REVISION", "0" * 40)
        assert SemanticSearcher(str(g))._cache_key() != before

    def test_key_changes_with_recipe_version(
        self, fake_model, graph_writer, monkeypatch
    ):
        g = graph_writer()
        before = SemanticSearcher(str(g))._cache_key()
        monkeypatch.setattr(E, "RECIPE_VERSION", E.RECIPE_VERSION + 1)
        assert SemanticSearcher(str(g))._cache_key() != before

    def test_key_changes_with_docstring_limit(
        self, fake_model, graph_writer, monkeypatch
    ):
        g = graph_writer()
        before = SemanticSearcher(str(g))._cache_key()
        monkeypatch.setattr(E, "DOCSTRING_CHAR_LIMIT", 123)
        assert SemanticSearcher(str(g))._cache_key() != before

    def test_key_changes_with_code_inclusion(self, fake_model, graph_writer, tmp_path):
        g = graph_writer()
        with_code = SemanticSearcher(str(g), source_root=str(tmp_path))._cache_key()
        assert SemanticSearcher(str(g))._cache_key() != with_code

    def test_key_changes_with_code_limit_only_when_code_is_included(
        self, fake_model, graph_writer, tmp_path, monkeypatch
    ):
        g = graph_writer()
        root = str(tmp_path)
        with_code = SemanticSearcher(str(g), source_root=root)._cache_key()
        without = SemanticSearcher(str(g))._cache_key()
        monkeypatch.setattr(E, "CODE_CHAR_LIMIT", 123)
        assert SemanticSearcher(str(g), source_root=root)._cache_key() != with_code
        assert SemanticSearcher(str(g))._cache_key() == without

    def test_local_model_dir_contents_are_part_of_key(
        self, fake_model, graph_writer, tmp_path
    ):
        g = graph_writer()
        mdir = tmp_path / "localmodel"
        mdir.mkdir()
        (mdir / "w.bin").write_bytes(b"one")
        before = SemanticSearcher(str(g), model_name=str(mdir))._cache_key()
        (mdir / "w.bin").write_bytes(b"two-different-length")
        assert SemanticSearcher(str(g), model_name=str(mdir))._cache_key() != before

    def test_sidecar_records_identity(self, fake_model, graph_writer):
        g = graph_writer()
        s = SemanticSearcher(str(g))
        sidecar = json.loads(s._cache_paths()[1].read_text())
        assert sidecar["recipe_version"] == E.RECIPE_VERSION
        assert sidecar["dim"] == s.embeddings.shape[1]
        assert sidecar["model"] == s.model_name


class TestPrompts:
    def test_document_prompt_applied_at_index_time(self, fake_model, graph_writer):
        # Named explicitly: the prompt under test is jina-code's, so this must
        # not silently follow whichever model happens to be the default.
        SemanticSearcher(str(graph_writer()), model_name="jina-code")
        assert fake_model["encodes"], "nothing was encoded"
        assert all(p == "nl2code_document" for _, p in fake_model["encodes"])

    def test_query_prompt_applied_at_search_time(self, fake_model, graph_writer):
        s = SemanticSearcher(str(graph_writer()), model_name="jina-code")
        fake_model["encodes"].clear()
        s.search("add numbers", limit=2, min_score=-1.0)
        assert fake_model["encodes"][-1][1] == "nl2code_query"

    def test_model_without_prompts_is_tolerated(self, fake_model, graph_writer):
        fake_model["prompts"] = {}
        s = SemanticSearcher(str(graph_writer()))
        assert all(p is None for _, p in fake_model["encodes"])
        assert s.search("anything", limit=2, min_score=-1.0) is not None

    def test_single_encode_pass_per_index(self, fake_model, graph_writer):
        SemanticSearcher(str(graph_writer()))
        assert len(fake_model["encodes"]) == 1, "index should be one batched pass"


class TestCacheRoundTrip:
    def test_build_then_warm_load_skips_encoding(self, fake_model, graph_writer):
        g = graph_writer()
        first = SemanticSearcher(str(g))
        loads_after_build = fake_model["loads"]
        fake_model["encodes"].clear()
        second = SemanticSearcher(str(g))
        assert fake_model["encodes"] == []
        assert fake_model["loads"] == loads_after_build, (
            "warm load must not build a model"
        )
        assert np.array_equal(first.embeddings, second.embeddings)

    def test_embeddings_are_unit_norm(self, fake_model, graph_writer):
        s = SemanticSearcher(str(graph_writer()))
        assert np.allclose(np.linalg.norm(s.embeddings, axis=1), 1.0, atol=1e-5)

    def test_no_temp_files_left_behind(self, fake_model, graph_writer):
        g = graph_writer()
        SemanticSearcher(str(g))
        leftovers = [p.name for p in g.parent.iterdir() if ".tmp" in p.name]
        assert leftovers == []

    def test_force_rebuild_reencodes(self, fake_model, graph_writer):
        g = graph_writer()
        SemanticSearcher(str(g))
        fake_model["encodes"].clear()
        SemanticSearcher(str(g), force_rebuild=True)
        assert fake_model["encodes"], "force_rebuild must re-encode"


class TestCorruptCacheRecovery:
    def _build(self, graph_writer):
        g = graph_writer()
        s = SemanticSearcher(str(g))
        return g, s._cache_paths()

    def test_truncated_npz_regenerates(self, fake_model, graph_writer):
        g, (npz, _) = self._build(graph_writer)
        npz.write_bytes(b"not a zip file")
        fake_model["encodes"].clear()
        assert SemanticSearcher(str(g)).embeddings is not None
        assert fake_model["encodes"]

    def test_bad_sidecar_regenerates(self, fake_model, graph_writer):
        g, (_, js) = self._build(graph_writer)
        js.write_text("{not json")
        fake_model["encodes"].clear()
        SemanticSearcher(str(g))
        assert fake_model["encodes"]

    def test_node_count_mismatch_regenerates(self, fake_model, graph_writer):
        g, (_, js) = self._build(graph_writer)
        d = json.loads(js.read_text())
        d["node_count"] = 1
        js.write_text(json.dumps(d))
        fake_model["encodes"].clear()
        SemanticSearcher(str(g))
        assert fake_model["encodes"]

    def test_stale_recipe_version_regenerates(self, fake_model, graph_writer):
        g, (_, js) = self._build(graph_writer)
        d = json.loads(js.read_text())
        d["recipe_version"] = E.RECIPE_VERSION - 1
        js.write_text(json.dumps(d))
        fake_model["encodes"].clear()
        SemanticSearcher(str(g))
        assert fake_model["encodes"], "a cache from an older recipe must not be served"


class TestCleanup:
    def test_cleanup_ignores_in_flight_temp_files(self, fake_model, graph_writer):
        g = graph_writer()
        s = SemanticSearcher(str(g))
        npz, js = s._cache_paths()
        # The undotted temporary name of descry 0.2, which the cleanup glob
        # matches.
        temp = s.cache_dir / "embeddings_otherkey.tmp.npz"
        temp.write_bytes(b"in flight")
        s._cleanup_old_embeddings(keep={npz, js})
        assert temp.exists(), "cleanup deleted another writer's temporary"

    def test_cleanup_removes_superseded_pair(self, fake_model, graph_writer):
        g = graph_writer()
        s = SemanticSearcher(str(g))
        stale = s.cache_dir / "embeddings_old_key.npz"
        stale.write_bytes(b"x")
        s._cleanup_old_embeddings(keep=set(s._cache_paths()))
        assert not stale.exists()


class TestSearchScoring:
    def test_threshold_applied_after_boosts(self, fake_model, graph_writer):
        """A node under the raw-cosine bar whose boosts lift it over must survive.

        alpha carries in_degree=3 (boost min(0.15, log(4)*0.03) = 0.0416) plus
        the Function type boost (0.02), so its final score is base + 0.0616.
        With a base of 0.50 and a threshold of 0.52 it belongs in the results;
        cutting on the raw cosine first drops it.
        """
        s = SemanticSearcher(str(graph_writer()))
        dim = s.embeddings.shape[1]
        s.embeddings = np.zeros((len(s.nodes), dim), dtype="float32")
        s.embeddings[0, 0], s.embeddings[0, 1] = 0.50, np.sqrt(1 - 0.50**2)
        s.embeddings[3, 0], s.embeddings[3, 1] = 0.51, np.sqrt(1 - 0.51**2)
        for i in (1, 2):
            s.embeddings[i, 2] = 1.0
        q = np.zeros(dim, dtype="float32")
        q[0] = 1.0
        s.model = FakeSentenceTransformer(dim=dim, prompts={})
        s.model.encode = lambda texts, **kw: q.reshape(1, -1)

        hits = s.search("q", limit=5, min_score=0.52)
        ids = [n["id"] for n, _ in hits]
        assert "f.py::alpha" in ids, "boosted node was cut by the raw-cosine threshold"
        assert all(score >= 0.52 for _, score in hits)

    def test_dimension_mismatch_raises_typed_error(self, fake_model, graph_writer):
        s = SemanticSearcher(str(graph_writer()))
        s.model = FakeSentenceTransformer(dim=s.embeddings.shape[1] + 4, prompts={})
        with pytest.raises(EmbeddingCacheMismatch):
            s.search("anything", limit=3)

    def test_results_are_sorted_and_limited(self, fake_model, graph_writer):
        s = SemanticSearcher(str(graph_writer()))
        hits = s.search("alpha", limit=2, min_score=-1.0)
        assert len(hits) <= 2
        assert [sc for _, sc in hits] == sorted((sc for _, sc in hits), reverse=True)


class TestConcurrency:
    def test_model_loaded_once_under_concurrent_search(self, fake_model, graph_writer):
        s = SemanticSearcher(str(graph_writer()))
        s.model = None
        fake_model["loads"] = 0
        fake_model["delay"] = 0.2  # widen the race window deterministically

        ts = [threading.Thread(target=s.ensure_model) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(20.0)
        assert fake_model["loads"] == 1, f"{fake_model['loads']} concurrent model loads"


class TestNodeText:
    """The assembly shared with tests/eval. Drift here silently invalidates
    the harness's leave-one-out control, so pin the shape."""

    def test_orders_name_signature_docstring(self):
        n = {
            "id": "f.py::alpha",
            "metadata": {
                "name": "alpha",
                "signature": "def alpha(a)",
                "docstring": "Adds.",
            },
        }
        assert E.node_text(n) == "alpha def alpha(a) Adds."

    def test_excluding_docstring_leaves_the_rest_identical(self):
        n = {
            "id": "f.py::alpha",
            "metadata": {
                "name": "alpha",
                "signature": "def alpha(a)",
                "docstring": "Adds.",
            },
        }
        full = E.node_text(n)
        nodoc = E.node_text(n, include_docstring=False)
        assert full.startswith(nodoc)
        assert nodoc == "alpha def alpha(a)"

    def test_null_metadata_does_not_crash(self):
        assert E.node_text({"id": "f.py::alpha", "metadata": None}) == "alpha"

    def test_missing_name_falls_back_to_id_tail(self):
        assert E.node_text({"id": "f.py::alpha", "metadata": {}}) == "alpha"

    def test_docstring_is_truncated(self):
        n = {"id": "a::b", "metadata": {"name": "b", "docstring": "x" * 900}}
        assert len(E.node_text(n)) == 2 + E.DOCSTRING_CHAR_LIMIT

    def test_empty_node_falls_back_to_id(self):
        assert E.node_text({"id": "weird", "metadata": {}}) == "weird"

    def test_code_is_appended_after_the_docstring(self):
        n = {
            "id": "f.py::alpha",
            "metadata": {
                "name": "alpha",
                "signature": "def alpha(a)",
                "docstring": "Adds.",
            },
        }
        assert E.node_text(n, code="return a") == "alpha def alpha(a) Adds. return a"
        assert (
            E.node_text(n, include_docstring=False, code="return a")
            == "alpha def alpha(a) return a"
        )

    def test_empty_code_leaves_the_text_unchanged(self):
        n = {"id": "f.py::alpha", "metadata": {"name": "alpha", "docstring": "Adds."}}
        assert E.node_text(n, code="") == E.node_text(n) == "alpha Adds."

    def test_harness_delegates_to_this_function(self, source_nodes, tmp_path):
        """tests/eval must not carry its own copy, with or without code."""
        from tests.eval.corpus import doc_text

        n = {
            "id": "a::b",
            "metadata": {"name": "b", "signature": "s", "docstring": "d"},
        }
        assert doc_text(n) == E.node_text(n)
        assert doc_text(n, with_docstring=False) == E.node_text(
            n, include_docstring=False
        )

        alpha = source_nodes[0]
        full = doc_text(alpha, source_root=tmp_path)
        nodoc = doc_text(alpha, with_docstring=False, source_root=tmp_path)
        assert full == E.node_text(alpha, code=PY_EXCERPT)
        assert nodoc == E.node_text(alpha, include_docstring=False, code=PY_EXCERPT)
        assert nodoc == f"alpha {PY_EXCERPT}"


class TestEvalHarness:
    def test_a_graph_of_another_schema_is_rejected(self, tmp_path):
        from tests.eval.corpus import load_nodes

        g = tmp_path / "codebase_graph.json"
        g.write_text(json.dumps({"schema_version": 0, "nodes": [], "edges": []}))
        with pytest.raises(GraphSchemaError):
            load_nodes(g)

    def test_a_run_without_the_baseline_config_writes_no_comparisons(
        self, graph_writer, tmp_path, monkeypatch
    ):
        import sys
        import types

        from tests.eval import compare

        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = lambda repo_id, **kw: FakeSentenceTransformer()
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
        monkeypatch.setattr(compare, "__file__", str(tmp_path / "compare.py"))
        node = _node("f.py::alpha", docstring="Add two numbers together and return it.")

        out = compare.main(str(graph_writer([node])), "qwen3_plain")

        assert out["metrics"]["qwen3_plain"]["R@1"] == 1.0
        assert out["comparisons"] == {}
        assert list((tmp_path / "results").iterdir())


class TestCodeExcerpt:
    def test_python_function_without_its_docstring_lines(self, source_nodes, tmp_path):
        assert E.code_excerpt(source_nodes[0], tmp_path) == PY_EXCERPT

    def test_function_with_doc_comments_above_it(self, source_nodes, tmp_path):
        assert E.code_excerpt(source_nodes[1], tmp_path) == RS_EXCERPT

    @pytest.mark.parametrize(
        ("source", "docstring"),
        [
            (RS_SOURCE, RS_DOCSTRING),
            ("//! Strip it.\n// Twice.\nfn scrub() {}\n", "Strip it.\nTwice."),
            ("/**\n * Strip it.\n * Twice. */\nfn scrub() {}\n", "Strip it.\nTwice."),
            ("# Strip it.\n#   Twice.\ndef scrub; end\n", "Strip it.\nTwice."),
            ("def scrub():\n    '''Strip it.'''\n", "Strip it."),
            ('def scrub():\n    r"""Strip \\d+."""\n', "Strip \\d+."),
        ],
    )
    def test_docstring_lines_are_removed_whatever_markers_surround_them(
        self, tmp_path, source, docstring
    ):
        (tmp_path / "f.txt").write_text(source)
        node = _node("FILE:f.txt::scrub", 1, len(source.splitlines()), docstring)
        excerpt = E.code_excerpt(node, tmp_path)
        assert "scrub" in excerpt
        for line in docstring.splitlines():
            assert line not in excerpt

    def test_a_code_line_that_contains_a_docstring_line_is_kept(self, tmp_path):
        (tmp_path / "f.py").write_text('def f():\n    """Go."""\n    print("Go.")\n')
        node = _node("FILE:f.py::f", 1, 3, "Go.")
        assert E.code_excerpt(node, tmp_path) == 'def f():\n    print("Go.")'

    def test_truncated_to_the_limit(self, tmp_path):
        source = "def big():\n" + "    total += 1\n" * 400
        (tmp_path / "big.py").write_text(source)
        excerpt = E.code_excerpt(_node("FILE:big.py::big", 1, 401), tmp_path)
        assert len(excerpt) == E.CODE_CHAR_LIMIT
        assert source.startswith(excerpt)

    @pytest.mark.parametrize(
        "node",
        [
            {"id": "FILE:src/mod.py", "type": "File", "metadata": {}},
            {"id": "FILE:src/mod.py", "metadata": {"lineno": 1, "end_lineno": 6}},
            _node("FILE:src/mod.py::alpha", lineno=1),
            _node("FILE:src/mod.py::alpha", end_lineno=6),
            _node("src/mod.py::alpha", 1, 6),
            _node("FILE:src/missing.py::alpha", 1, 6),
            _node("FILE:src::alpha", 1, 6),
            _node("FILE:src/\0.py::alpha", 1, 6),
            _node("FILE:" + "x" * 300 + "::alpha", 1, 6),
        ],
    )
    def test_no_excerpt_without_a_readable_file_and_both_line_numbers(
        self, source_nodes, tmp_path, node
    ):
        assert E.code_excerpt(node, tmp_path) == ""

    def test_no_excerpt_without_a_source_root(self, source_nodes):
        assert E.code_excerpt(source_nodes[0], None) == ""

    @pytest.mark.parametrize(
        "node_id",
        [
            "FILE:../secret.py::x",
            "FILE:src/../../secret.py::x",
            "FILE:{outside}/secret.py::x",
            "FILE:link.py::x",
            "FILE:linkdir/secret.py::x",
        ],
    )
    def test_a_path_leaving_the_root_is_not_read(self, tmp_path, monkeypatch, node_id):
        root = tmp_path / "proj"
        (root / "src").mkdir(parents=True)
        (root / "inside.py").write_text("token = 1\n")
        (tmp_path / "secret.py").write_text("token = 2\n")
        os.symlink(tmp_path / "secret.py", root / "link.py")
        os.symlink(tmp_path, root / "linkdir")
        reads = []

        def read(path):
            reads.append(path)
            return ("token = 1\n",)

        monkeypatch.setattr(E, "_read_file_cached", read)
        node = _node(node_id.format(outside=tmp_path), 1, 1)
        assert E.code_excerpt(node, root) == ""
        assert reads == []
        assert E.code_excerpt(_node("FILE:inside.py::x", 1, 1), root) == "token = 1"
        assert reads == [str(root / "inside.py")]


class TestSourceRoot:
    """Code enters the index only through `source_root`."""

    def test_texts_include_code_when_given(
        self, fake_model, graph_writer, source_nodes, tmp_path
    ):
        g = graph_writer(source_nodes)
        s = SemanticSearcher(str(g), source_root=str(tmp_path))
        ((texts, _prompt),) = fake_model["encodes"]
        assert texts == s.node_texts
        assert texts == [
            f"alpha {PY_DOCSTRING} {PY_EXCERPT}",
            f"scrub {RS_DOCSTRING} {RS_EXCERPT}",
            "FILE:src/mod.py",
        ]

    def test_texts_have_no_code_otherwise(self, fake_model, graph_writer, source_nodes):
        g = graph_writer(source_nodes)
        SemanticSearcher(str(g))
        ((texts, _prompt),) = fake_model["encodes"]
        assert texts == [
            f"alpha {PY_DOCSTRING}",
            f"scrub {RS_DOCSTRING}",
            "FILE:src/mod.py",
        ]

    def test_an_index_without_code_is_not_loaded_when_code_is_expected(
        self, fake_model, graph_writer, source_nodes, tmp_path
    ):
        g = graph_writer(source_nodes)
        SemanticSearcher(str(g))
        s = SemanticSearcher(str(g), source_root=str(tmp_path))
        assert len(fake_model["encodes"]) == 2
        assert PY_EXCERPT in s.node_texts[0]

    def test_an_index_with_code_is_not_loaded_when_none_is_expected(
        self, fake_model, graph_writer, source_nodes, tmp_path
    ):
        g = graph_writer(source_nodes)
        SemanticSearcher(str(g), source_root=str(tmp_path))
        s = SemanticSearcher(str(g))
        assert len(fake_model["encodes"]) == 2
        assert PY_EXCERPT not in s.node_texts[0]

    def test_warm_load_with_the_same_source_root_skips_encoding(
        self, fake_model, graph_writer, source_nodes, tmp_path
    ):
        g = graph_writer(source_nodes)
        SemanticSearcher(str(g), source_root=str(tmp_path))
        SemanticSearcher(str(g), source_root=str(tmp_path))
        assert len(fake_model["encodes"]) == 1

    @pytest.mark.parametrize("built_with_code", [True, False])
    def test_status_agrees_with_the_searcher(
        self, fake_model, graph_writer, source_nodes, tmp_path, built_with_code
    ):
        g = graph_writer(source_nodes)
        root = {True: str(tmp_path), False: None}
        s = SemanticSearcher(str(g), source_root=root[built_with_code])

        same = get_embeddings_status(str(g), source_root=root[built_with_code])
        assert same["cached"] is True and same["stale"] is False
        assert same["cache_path"] == str(s._cache_paths()[0])
        assert same["node_count"] == 3

        other = get_embeddings_status(str(g), source_root=root[not built_with_code])
        assert other["cached"] is False and other["stale"] is True


class TestModelRegistry:
    def test_alias_resolves_to_repo_id(self):
        spec = E.resolve_model_spec("embeddinggemma")
        assert spec is not None
        assert spec.repo_id == "google/embeddinggemma-300m"

    def test_repo_id_resolves_to_same_spec(self):
        by_alias = E.resolve_model_spec("qwen3")
        by_repo = E.resolve_model_spec("Qwen/Qwen3-Embedding-0.6B")
        assert by_alias is by_repo

    def test_unregistered_model_resolves_to_none(self):
        assert E.resolve_model_spec("someone/not-in-registry") is None

    def test_every_entry_is_pinned(self):
        for alias, spec in E.MODEL_REGISTRY.items():
            assert spec.revision, f"{alias} has no pinned revision"
            assert spec.alias == alias

    def test_default_and_its_fallback(self):
        """Chosen on measured retrieval; see tests/eval."""
        assert E.DEFAULT_MODEL_ALIAS == "embeddinggemma"
        assert SemanticSearcher.MODEL_NAME == "google/embeddinggemma-300m"
        assert E.FALLBACK_MODEL_ALIAS == "f2llm-v2-0.6b"
        assert E.FALLBACK_MODEL_ALIAS in E.MODEL_REGISTRY

    def test_default_alias_is_registered(self):
        assert E.DEFAULT_MODEL_ALIAS in E.MODEL_REGISTRY
        assert (
            SemanticSearcher.MODEL_NAME
            == E.MODEL_REGISTRY[E.DEFAULT_MODEL_ALIAS].repo_id
        )

    @pytest.fixture
    def constructions(self, monkeypatch):
        """Stand a recorder in for SentenceTransformer; lists the arguments of
        each construction."""
        calls = []

        class Recorder:
            max_seq_length = 2048

            def __init__(self, name, **kw):
                calls.append({"name": name, **kw})

        monkeypatch.setattr(E, "SentenceTransformer", Recorder)
        return calls

    def test_remote_code_is_never_granted_to_unregistered_models(self, constructions):
        E._load_sentence_transformer("evil/model")
        (seen,) = constructions
        assert seen["trust_remote_code"] is False
        assert "revision" not in seen, "an unvetted model must not be given a pin"

    def test_no_model_is_ever_loaded_with_remote_code(self, constructions):
        """Nothing can turn it on: no registry field, no loader parameter."""
        import dataclasses
        import inspect

        for alias in E.MODEL_REGISTRY:
            E._load_sentence_transformer(alias)
        seen = [c["trust_remote_code"] for c in constructions]
        assert seen == [False] * len(E.MODEL_REGISTRY)
        fields = {f.name for f in dataclasses.fields(E.EmbeddingModelSpec)}
        assert "trust_remote_code" not in fields
        params = inspect.signature(E._load_sentence_transformer).parameters
        assert "trust_remote_code" not in params

    def test_registered_model_supplies_its_pin(self, constructions):
        E._load_sentence_transformer("jina-code")
        (seen,) = constructions
        spec = E.MODEL_REGISTRY["jina-code"]
        assert seen["name"] == spec.repo_id
        assert seen["trust_remote_code"] is False
        assert seen["revision"] == spec.revision

    def test_embeddinggemma2_loads_text_only_with_a_token_cap(self, constructions):
        model = E._load_sentence_transformer("embeddinggemma-2")
        (seen,) = constructions
        spec = E.MODEL_REGISTRY["embeddinggemma-2"]
        assert seen["name"] == "google/embeddinggemma-2"
        assert seen["trust_remote_code"] is False
        assert seen["revision"] == spec.revision
        assert seen["config_kwargs"] == {"vision_config": None, "audio_config": None}
        assert model.max_seq_length == 8192

    def test_models_without_loading_options_get_none(self, constructions):
        model = E._load_sentence_transformer("embeddinggemma")
        (seen,) = constructions
        assert "config_kwargs" not in seen
        assert model.max_seq_length == 2048

    def test_missing_extra_names_the_install_command(self, monkeypatch):
        def refuse(name, **kw):
            raise ImportError("requires the PIL library")

        monkeypatch.setattr(E, "SentenceTransformer", refuse)
        with pytest.raises(E.EmbeddingModelDependencyMissing) as err:
            E._load_sentence_transformer("embeddinggemma-2")
        assert "descry-codegraph[embeddinggemma-2]" in str(err.value)
        # A model with no extra of its own keeps the plain ImportError.
        with pytest.raises(ImportError) as plain:
            E._load_sentence_transformer("embeddinggemma")
        assert not isinstance(plain.value, E.EmbeddingModelDependencyMissing)

    def test_every_extra_a_model_names_is_declared(self):
        import pathlib
        import tomllib

        declared = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
        for spec in E.MODEL_REGISTRY.values():
            if spec.extra:
                assert spec.extra in declared

    def test_gated_repo_refusal_names_the_fix(self, monkeypatch):
        """HF's bare 401 becomes an error that says how to get past it."""

        class GatedRepoError(OSError):
            pass

        def refuse(name, **kw):
            # transformers re-raises the hub's refusal wrapped in an OSError.
            try:
                raise GatedRepoError("401 Client Error")
            except GatedRepoError as e:
                raise OSError("cannot access gated repo") from e

        monkeypatch.setattr(E, "SentenceTransformer", refuse)
        with pytest.raises(E.EmbeddingModelGated) as err:
            E._load_sentence_transformer("embeddinggemma")
        msg = str(err.value)
        assert "https://huggingface.co/google/embeddinggemma-300m" in msg
        assert f'model = "{E.FALLBACK_MODEL_ALIAS}"' in msg

    def test_other_load_failures_are_not_relabelled(self, monkeypatch):
        def refuse(name, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(E, "SentenceTransformer", refuse)
        with pytest.raises(OSError, match="disk full") as err:
            E._load_sentence_transformer("embeddinggemma")
        assert not isinstance(err.value, E.EmbeddingModelGated)

    def test_spec_prompts_are_authoritative_over_probing(
        self, fake_model, graph_writer
    ):
        """qwen3 names no prompt, and that must beat probing the model."""
        s = SemanticSearcher(str(graph_writer()), model_name="qwen3")
        s.ensure_model()
        # The stub registers jina's prompt names; the spec must still win.
        assert s._prompt_name("query") is None
        assert s._prompt_name("document") is None

    def test_unregistered_model_probes_for_prompts(self, fake_model, graph_writer):
        s = SemanticSearcher(str(graph_writer()), model_name="unknown/model")
        s.ensure_model()
        assert s._prompt_name("document") == "nl2code_document"

    def test_cache_key_differs_between_registry_entries(self, fake_model, graph_writer):
        g = graph_writer()
        keys = {
            alias: SemanticSearcher(str(g), model_name=alias)._cache_key()
            for alias in E.MODEL_REGISTRY
        }
        assert len(set(keys.values())) == len(keys), keys

    def test_alias_and_repo_id_share_a_cache(self, fake_model, graph_writer):
        g = graph_writer()
        a = SemanticSearcher(str(g), model_name="qwen3")._cache_key()
        b = SemanticSearcher(
            str(g), model_name="Qwen/Qwen3-Embedding-0.6B"
        )._cache_key()
        assert a == b, "selecting the same model two ways must not re-encode"


class TestGatedDefaultFallback:
    @pytest.fixture
    def gate(self, monkeypatch):
        """Model factory that refuses the default model while `closed` is set."""
        state = {"closed": True, "loads": []}

        def factory(model_name, revision=None):
            state["loads"].append(model_name)
            spec = E.resolve_model_spec(model_name)
            if state["closed"] and spec.alias == E.DEFAULT_MODEL_ALIAS:
                raise E.EmbeddingModelGated(f"{model_name} is gated")
            return FakeSentenceTransformer()

        monkeypatch.setattr(E, "_load_sentence_transformer", factory)
        monkeypatch.setattr(E, "EMBEDDINGS_AVAILABLE", True)
        return state

    def test_unconfigured_model_falls_back_with_a_warning(
        self, gate, graph_writer, caplog
    ):
        with caplog.at_level("WARNING", logger=E.logger.name):
            s = SemanticSearcher(str(graph_writer()))

        assert s.model_name == E.FALLBACK_MODEL_ALIAS
        assert gate["loads"] == [SemanticSearcher.MODEL_NAME, E.FALLBACK_MODEL_ALIAS]
        assert "hf auth login" in caplog.text
        assert f'model = "{E.FALLBACK_MODEL_ALIAS}"' in caplog.text
        assert len(s.search("add numbers", min_score=-1.0)) > 0

    def test_a_model_named_explicitly_does_not_fall_back(self, gate, graph_writer):
        with pytest.raises(E.EmbeddingModelGated):
            SemanticSearcher(str(graph_writer()), model_name="embeddinggemma")

    def test_searches_reuse_the_fallback_index_without_retrying_the_default(
        self, gate, graph_writer
    ):
        g = graph_writer()
        SemanticSearcher(str(g))
        gate["loads"].clear()

        s = SemanticSearcher(str(g))

        assert s.model_name == E.FALLBACK_MODEL_ALIAS
        assert gate["loads"] == []
        s.search("add numbers")
        assert gate["loads"] == [E.FALLBACK_MODEL_ALIAS]

    def test_a_rebuild_retries_the_default(self, gate, graph_writer):
        g = graph_writer()
        SemanticSearcher(str(g))
        gate["closed"] = False

        s = SemanticSearcher(str(g), force_rebuild=True)

        assert s.model_name == SemanticSearcher.MODEL_NAME
        assert SemanticSearcher(str(g)).model_name == SemanticSearcher.MODEL_NAME

    def test_status_names_the_model_that_built_the_index(self, gate, graph_writer):
        g = graph_writer()
        SemanticSearcher(str(g))

        st = get_embeddings_status(str(g))

        assert st["cached"] is True
        assert st["model"] == E.MODEL_REGISTRY[E.FALLBACK_MODEL_ALIAS].repo_id
        assert (
            get_embeddings_status(str(g), model_name="embeddinggemma")["cached"]
            is False
        )


class TestStatus:
    def test_reports_cached_for_current_graph(self, fake_model, graph_writer):
        g = graph_writer()
        SemanticSearcher(str(g))
        st = get_embeddings_status(str(g))
        assert st["cached"] is True and st["stale"] is False
        assert st["node_count"] == 4

    def test_reports_stale_after_graph_change(self, fake_model, graph_writer):
        g = graph_writer()
        SemanticSearcher(str(g))
        graph_writer([{"id": "n::x", "type": "Function", "metadata": {"name": "x"}}])
        st = get_embeddings_status(str(g))
        assert st["cached"] is False and st["stale"] is True

    def test_finds_cache_when_graph_sits_outside_descry_cache(
        self, fake_model, graph_writer
    ):
        g = graph_writer(dirname="elsewhere")
        SemanticSearcher(str(g))
        assert get_embeddings_status(str(g))["cached"] is True


class TestCacheDirPropagation:
    """A custom DESCRY_CACHE_DIR must hold the embeddings too.

    Without it the graph lands in the configured directory while embeddings go
    to `<configured>/.descry_cache`, so a warm index reads as missing and the
    60s pre-warm rebuilds it from scratch.
    """

    def test_service_passes_cache_dir_to_the_searcher(self):
        import inspect

        from descry.handlers import DescryService

        for name in ("_load_embeddings_background", "_get_semantic_searcher", "index"):
            src = inspect.getsource(getattr(DescryService, name))
            if "_SemanticSearcher(" in src:
                assert "cache_dir=" in src, (
                    f"{name} constructs a searcher without cache_dir"
                )

    def test_generate_passes_cache_dir(self):
        import pathlib

        src = pathlib.Path("src/descry/generate.py").read_text(encoding="utf-8")
        i = src.index("SemanticSearcher(")
        assert "cache_dir=" in src[i : i + 300]
        assert "source_root=" in src[i : i + 300]

    def test_embeddings_land_in_the_configured_dir(self, fake_model, graph_writer):
        """Graph outside a .descry_cache-named dir must not split the cache."""
        graph = graph_writer(dirname="custom-cache")
        cache = graph.parent
        SemanticSearcher(str(graph), cache_dir=str(cache))
        assert list(cache.glob("embeddings_*.npz")), "no index in the configured dir"
        assert not (cache / ".descry_cache").exists(), "cache split into a subdirectory"


class TestFileLockWithoutFcntl:
    """The msvcrt branch, with `msvcrt.locking` stood in for by `fcntl.flock`:
    both lock an open descriptor, refuse a held lock with OSError, and are
    dropped by the OS when the descriptor closes."""

    @pytest.fixture(autouse=True)
    def msvcrt_branch(self, monkeypatch):
        import sys
        import types

        fcntl = pytest.importorskip("fcntl")
        fake = types.ModuleType("msvcrt")
        fake.LK_NBLCK, fake.LK_UNLCK = 2, 0

        def locking(fd, mode, _nbytes):
            op = fcntl.LOCK_UN if mode == fake.LK_UNLCK else fcntl.LOCK_EX
            fcntl.flock(fd, op | fcntl.LOCK_NB)

        fake.locking = locking
        monkeypatch.setitem(sys.modules, "msvcrt", fake)
        monkeypatch.setattr(E, "_fcntl", None)

    def test_lock_file_left_by_a_dead_holder_does_not_block(self, tmp_path):
        p = tmp_path / "embeddings.lock"
        p.write_text("")
        start = time.monotonic()
        with _file_lock(p, timeout=5.0):
            pass
        assert time.monotonic() - start < 1.0

    def test_lock_file_is_never_unlinked(self, tmp_path):
        p = tmp_path / "embeddings.lock"
        inodes = []
        for _ in range(2):
            with _file_lock(p):
                inodes.append(os.stat(p).st_ino)
        assert p.exists()
        assert inodes[0] == inodes[1]

    def test_mutual_exclusion_under_contention(self, tmp_path):
        assert _peak_lock_holders(tmp_path / "embeddings.lock") == 1


class TestDegenerateInputs:
    def test_empty_graph_builds_and_searches_without_a_model(
        self, fake_model, graph_writer
    ):
        g = graph_writer(nodes=[])
        s = SemanticSearcher(str(g))
        assert s.embeddings.shape[0] == 0
        assert s.search("anything") == []
        assert fake_model["loads"] == 0
        assert SemanticSearcher(str(g)).search("anything") == []

    @pytest.mark.parametrize("limit", [0, -1])
    def test_non_positive_limit_returns_nothing(self, fake_model, graph_writer, limit):
        s = SemanticSearcher(str(graph_writer()))
        assert s.search("alpha", limit=limit, min_score=-1.0) == []

    @pytest.mark.parametrize("degenerate", [0.0, np.nan])
    def test_nan_similarity_never_passes_min_score(
        self, fake_model, graph_writer, degenerate
    ):
        s = SemanticSearcher(str(graph_writer()))
        s.embeddings = s.embeddings.copy()
        s.embeddings[1] = degenerate
        with np.errstate(invalid="ignore"):
            hits = s.search("alpha", limit=10, min_score=-1.0)
        assert "f.py::Beta" not in [n["id"] for n, _ in hits]
        assert not any(np.isnan(score) for _, score in hits)


class TestRecipeFingerprintInputs:
    def test_key_changes_with_config_kwargs(
        self, fake_model, graph_writer, monkeypatch
    ):
        import dataclasses

        g = graph_writer()
        before = SemanticSearcher(str(g), model_name="embeddinggemma-2")._cache_key()
        spec = E.MODEL_REGISTRY["embeddinggemma-2"]
        changed = dataclasses.replace(
            spec, config_kwargs={**spec.config_kwargs, "audio_config": {}}
        )
        monkeypatch.setitem(E.MODEL_REGISTRY, "embeddinggemma-2", changed)
        after = SemanticSearcher(str(g), model_name="embeddinggemma-2")._cache_key()
        assert before != after

    def test_key_changes_with_the_prompts_probed_for_unregistered_models(
        self, fake_model, graph_writer, monkeypatch
    ):
        g = graph_writer()
        before = SemanticSearcher(str(g), model_name="unknown/model")._cache_key()
        monkeypatch.setattr(E, "DOCUMENT_PROMPT_CANDIDATES", ("passage",))
        after = SemanticSearcher(str(g), model_name="unknown/model")._cache_key()
        assert before != after

    def test_warm_cache_is_refused_when_the_model_resolves_another_document_prompt(
        self, fake_model, graph_writer
    ):
        g = graph_writer()
        SemanticSearcher(str(g), model_name="unknown/model")
        fake_model["prompts"] = {}
        warm = SemanticSearcher(str(g), model_name="unknown/model")
        with pytest.raises(EmbeddingCacheMismatch, match="re-index"):
            warm.search("alpha")

    def test_warm_cache_is_searched_when_the_document_prompt_is_unchanged(
        self, fake_model, graph_writer
    ):
        g = graph_writer()
        SemanticSearcher(str(g), model_name="unknown/model")
        warm = SemanticSearcher(str(g), model_name="unknown/model")
        assert warm.search("alpha", min_score=-1.0)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
@pytest.mark.usefixtures("no_wait")
def test_fifo_sidecars_are_left(fake_model, graph_writer):
    """A FIFO where the cache's files are is not read."""
    g = graph_writer()
    s = SemanticSearcher(str(g))
    for path in s._cache_paths():
        path.unlink()
        os.mkfifo(path)

    assert s._load_cached() is False
    assert get_embeddings_status(str(g))["cached"] is False


def test_orphaned_embedding_temporaries_are_removed(fake_model, graph_writer):
    """A save killed mid-write leaves temporaries for a key the next index no
    longer uses; the cleanup removes them once nothing writes to them."""
    from descry import _files

    g = graph_writer()
    s = SemanticSearcher(str(g))
    orphan = s.cache_dir / ".embeddings_oldkey.npz.0123456789abcdef.tmp"
    orphan.write_bytes(b"x")
    stale = time.time() - 2 * _files._ORPHANED_AFTER
    os.utime(orphan, (stale, stale))

    s._cleanup_old_embeddings(keep=set(s._cache_paths()))

    assert not orphan.exists()
