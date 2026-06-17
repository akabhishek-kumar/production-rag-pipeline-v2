from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    groq_api_key: str = "gsk_..."
    groq_model: str = "llama-3.1-8b-instant"

    # Storage
    persist_dir: str = "./chroma_db"
    docs_dir: str = "./docs"
    bm25_index_path: str = "./bm25_index.pkl"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval — v2
    bm25_k: int = 10          # BM25 candidate pool
    vector_k: int = 10        # vector search candidate pool
    reranker_k: int = 5       # kept after cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Quality gates
    grade_threshold: int = 6
    max_retries: int = 2
    recursion_limit: int = 25


settings = Settings()
