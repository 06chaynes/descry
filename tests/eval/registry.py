"""Candidate models for the comparison harness.

Revisions are the snapshot shas actually exercised, so a rerun scores the same
weights. Keep in step with descry.embeddings.MODEL_REGISTRY.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalModel:
    key: str
    repo_id: str
    revision: str
    trust_remote_code: bool
    query_prompt: str | None
    document_prompt: str | None


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
]
