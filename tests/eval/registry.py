"""Candidate models for the comparison harness.

Each config scores a descry.embeddings.MODEL_REGISTRY entry, at the revision
pinned there, through one choice of prompts. A candidate that is not registered
is written as a literal EvalModel.
"""

from dataclasses import dataclass, field

from descry.embeddings import MODEL_REGISTRY


@dataclass(frozen=True)
class EvalModel:
    key: str
    repo_id: str
    revision: str
    query_prompt: str | None
    document_prompt: str | None
    # Extra SentenceTransformer constructor arguments, e.g. `config_kwargs`.
    load_kwargs: dict = field(default_factory=dict)


def _config(key, alias, query_prompt, document_prompt):
    spec = MODEL_REGISTRY[alias]
    load_kwargs = {"config_kwargs": spec.config_kwargs} if spec.config_kwargs else {}
    return EvalModel(
        key, spec.repo_id, spec.revision, query_prompt, document_prompt, load_kwargs
    )


CONFIGS = [
    _config("jina_plain", "jina-code", None, None),
    _config("jina_prompted", "jina-code", "nl2code_query", "nl2code_document"),
    _config("qwen3_plain", "qwen3", None, None),
    _config("qwen3_prompted", "qwen3", "query", "document"),
    _config("gemma_plain", "embeddinggemma", None, None),
    _config("gemma_retrieval", "embeddinggemma", "query", "document"),
    # 'InstructionRetrieval' is embeddinggemma's code-retrieval prompt:
    # 'task: code retrieval | query: '. The name is the MTEB task label.
    _config("gemma_code", "embeddinggemma", "InstructionRetrieval", "document"),
    _config("gemma2_plain", "embeddinggemma-2", None, None),
    _config("gemma2_search", "embeddinggemma-2", "SearchQuery", "Document"),
    _config("gemma2_code", "embeddinggemma-2", "CodeRetrieval", "Document"),
    _config("f2llm_plain", "f2llm-v2-0.6b", None, None),
    _config("f2llm_prompted", "f2llm-v2-0.6b", "query", None),
]
