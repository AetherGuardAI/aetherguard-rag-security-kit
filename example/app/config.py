from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'AetherGuard Local Chroma RAG'

    # AetherGuard SDK backend API. The wheel is only a client; it needs this backend.
    aetherguard_api_url: str = 'https://example.com'
    aetherguard_api_key: str = 'Your_Api_key'
    aetherguard_timeout: float = 30.0

    region: str = 'us-east-1'
    source_type: str = 'local-file'
    email: str = 'abc@gmail.com'

    chroma_host: str = 'localhost'
    chroma_port: int = 8000
    chroma_collection: str = 'secure_rag_chunks'

    ollama_base_url: str = 'http://localhost:11434'
    ollama_chat_model: str = 'llama3.1'
    ollama_embed_model: str = 'nomic-embed-text'

    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    max_context_tokens: int = 4096
    trust_threshold: str = 'trusted'
    max_injection_score: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
