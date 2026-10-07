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
    rec = {"loads": 0, "encodes": [], "dim": 8, "prompts": None, "delay": 0.0}

    def factory(model_name, revision=None, trust_remote_code=None):
        return FakeSentenceTransformer(
            dim=rec["dim"],
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
        p.write_text(json.dumps({"schema_version": 1, "nodes": nodes, "edges": []}))
        return p

    return write


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
        p = tmp_path / "x.lock"
        inside, peak, guard = [0], [0], threading.Lock()

        def worker():
            with _file_lock(p, timeout=10.0):
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
        assert peak[0] == 1, f"{peak[0]} threads held the lock at once"


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
        leftovers = [p.name for p in g.parent.iterdir() if ".tmp." in p.name]
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

    def test_text_count_mismatch_regenerates(self, fake_model, graph_writer):
        g, (_, js) = self._build(graph_writer)
        d = json.loads(js.read_text())
        d["texts"] = d["texts"][:1]
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
        # Named the way the previous implementation named its temporaries,
        # so this fails against a cleanup whose glob still matches them.
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

        errors = []

        def go():
            try:
                s.ensure_model()
            except (RuntimeError, OSError, ValueError) as exc:  # pragma: no cover
                errors.append(exc)

        ts = [threading.Thread(target=go) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(20.0)
        assert not errors
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

    def test_harness_delegates_to_this_function(self):
        """tests/eval must not carry its own copy."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from eval.corpus import doc_text

        n = {
            "id": "a::b",
            "metadata": {"name": "b", "signature": "s", "docstring": "d"},
        }
        assert doc_text(n) == E.node_text(n)
        assert doc_text(n, with_docstring=False) == E.node_text(
            n, include_docstring=False
        )


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

    def test_default_is_embeddinggemma(self):
        """Chosen on measured retrieval; see tests/eval."""
        assert E.DEFAULT_MODEL_ALIAS == "embeddinggemma"
        assert SemanticSearcher.MODEL_NAME == "google/embeddinggemma-300m"

    def test_default_alias_is_registered(self):
        assert E.DEFAULT_MODEL_ALIAS in E.MODEL_REGISTRY
        assert (
            SemanticSearcher.MODEL_NAME
            == E.MODEL_REGISTRY[E.DEFAULT_MODEL_ALIAS].repo_id
        )

    def test_remote_code_is_never_granted_to_unregistered_models(self, monkeypatch):
        """The registry is the only thing that can turn remote code on."""
        seen = {}

        class Recorder:
            def __init__(self, name, **kw):
                seen["name"] = name
                seen.update(kw)

        monkeypatch.setattr(E, "SentenceTransformer", Recorder)
        E._load_sentence_transformer("evil/model")
        assert seen["trust_remote_code"] is False
        assert "revision" not in seen, "an unvetted model must not be given a pin"

    def test_registered_model_supplies_pin_and_trust(self, monkeypatch):
        seen = {}

        class Recorder:
            def __init__(self, name, **kw):
                seen["name"] = name
                seen.update(kw)

        monkeypatch.setattr(E, "SentenceTransformer", Recorder)
        E._load_sentence_transformer("jina-code")
        spec = E.MODEL_REGISTRY["jina-code"]
        assert seen["name"] == spec.repo_id
        assert seen["trust_remote_code"] is True
        assert seen["revision"] == spec.revision

    def test_embeddinggemma2_loads_text_only_with_a_token_cap(self, monkeypatch):
        seen = {}

        class Recorder:
            max_seq_length = 10**30

            def __init__(self, name, **kw):
                seen["name"] = name
                seen.update(kw)

        monkeypatch.setattr(E, "SentenceTransformer", Recorder)
        model = E._load_sentence_transformer("embeddinggemma-2")
        spec = E.MODEL_REGISTRY["embeddinggemma-2"]
        assert seen["name"] == "google/embeddinggemma-2"
        assert seen["trust_remote_code"] is False
        assert seen["revision"] == spec.revision
        assert seen["config_kwargs"] == {"vision_config": None, "audio_config": None}
        assert model.max_seq_length == 8192

    def test_models_without_loading_options_get_none(self, monkeypatch):
        seen = {}

        class Recorder:
            max_seq_length = 2048

            def __init__(self, name, **kw):
                seen.update(kw)

        monkeypatch.setattr(E, "SentenceTransformer", Recorder)
        model = E._load_sentence_transformer("embeddinggemma")
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
        assert f'model = "{E.UNGATED_FALLBACK_ALIAS}"' in msg
        # The model the error steers people to must not itself need remote code.
        assert E.MODEL_REGISTRY[E.UNGATED_FALLBACK_ALIAS].trust_remote_code is False

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
        self, fake_model, tmp_path
    ):
        g = tmp_path / "codebase_graph.json"
        g.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "nodes": [
                        {"id": "a::b", "type": "Function", "metadata": {"name": "b"}}
                    ],
                    "edges": [],
                }
            )
        )
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

    def test_embeddings_land_in_the_configured_dir(self, fake_model, tmp_path):
        """Graph outside a .descry_cache-named dir must not split the cache."""
        import json

        cache = tmp_path / "custom-cache"
        cache.mkdir()
        graph = cache / "codebase_graph.json"
        graph.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "nodes": [
                        {"id": "a.py::f", "type": "Function", "metadata": {"name": "f"}}
                    ],
                    "edges": [],
                }
            )
        )
        SemanticSearcher(str(graph), cache_dir=str(cache))
        assert list(cache.glob("embeddings_*.npz")), "no index in the configured dir"
        assert not (cache / ".descry_cache").exists(), "cache split into a subdirectory"
