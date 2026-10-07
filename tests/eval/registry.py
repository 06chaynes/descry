"""Candidate models for the comparison harness.

Revisions are the snapshot shas actually exercised, so a rerun scores the same
weights. Keep in step with descry.embeddings.MODEL_REGISTRY.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalModel:
    key: str
    repo_id: str
    revision: str
    trust_remote_code: bool
    query_prompt: str | None
    document_prompt: str | None
    # Extra SentenceTransformer constructor arguments, e.g. `config_kwargs`.
    load_kwargs: dict = field(default_factory=dict)


_GEMMA2_TEXT_ONLY = {"config_kwargs": {"vision_config": None, "audio_config": None}}

CONFIGS = [
    EvalModel(
        "jina_plain",
        "jinaai/jina-code-embeddings-0.5b",
        "4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
        True,
        None,
        None,
    ),
    EvalModel(
        "jina_prompted",
        "jinaai/jina-code-embeddings-0.5b",
        "4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
        True,
        "nl2code_query",
        "nl2code_document",
    ),
    EvalModel(
        "qwen3_plain",
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        False,
        None,
        None,
    ),
    EvalModel(
        "qwen3_prompted",
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        False,
        "query",
        "document",
    ),
    EvalModel(
        "gemma_plain",
        "google/embeddinggemma-300m",
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
        False,
        None,
        None,
    ),
    EvalModel(
        "gemma_retrieval",
        "google/embeddinggemma-300m",
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
        False,
        "query",
        "document",
    ),
    # 'InstructionRetrieval' is embeddinggemma's code-retrieval prompt:
    # 'task: code retrieval | query: '. The name is the MTEB task label.
    EvalModel(
        "gemma_code",
        "google/embeddinggemma-300m",
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
        False,
        "InstructionRetrieval",
        "document",
    ),
    # EmbeddingGemma 2 is multimodal (740M); dropping the vision and audio
    # towers leaves the 270M text model and produces identical text vectors.
    EvalModel(
        "gemma2_plain",
        "google/embeddinggemma-2",
        "914f7f89142e33e77833254d9c9b90c3cef7303b",
        False,
        None,
        None,
        _GEMMA2_TEXT_ONLY,
    ),
    EvalModel(
        "gemma2_search",
        "google/embeddinggemma-2",
        "914f7f89142e33e77833254d9c9b90c3cef7303b",
        False,
        "SearchQuery",
        "Document",
        _GEMMA2_TEXT_ONLY,
    ),
    EvalModel(
        "gemma2_code",
        "google/embeddinggemma-2",
        "914f7f89142e33e77833254d9c9b90c3cef7303b",
        False,
        "CodeRetrieval",
        "Document",
        _GEMMA2_TEXT_ONLY,
    ),
]
