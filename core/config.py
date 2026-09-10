from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_db"
    similarity_metric: str = "cosine"

    retrieval_k: int = 3
    candidate_k: int = 10          # over-fetch before rerank
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    enable_reranker: bool = True

    llm_model: str = "openai/gpt-4o-mini"
    llm_base_url: str = "https://models.inference.ai.azure.com"
    llm_temperature: float = 0.0

CONFIG = Config()