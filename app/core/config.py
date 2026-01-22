# cymotive-cyber-copilot/app/core/config.py

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration loaded from env vars + optional .env
    Compatible with Pydantic v2 (BaseSettings lives in pydantic-settings).
    """

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


    # LLM configuration

    LLM_PROVIDER: str = Field(default="mock", description="mock|openai")
    OPENAI_API_KEY: str | None = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    RETRIEVAL_BACKEND: str = Field(default="tfidf", description="tfidf|faiss")

    # Embeddings model for FAISS retrieval (OpenAI)
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # Optional persistence (so you don't re-embed on every restart)
    FAISS_INDEX_PATH: str = Field(default="storage/faiss.index")
    FAISS_META_PATH: str = Field(default="storage/faiss_meta.json")

    # Retrieval configuration

    INCIDENTS_PATH: str = Field(default="app/data/incidents_seed.json")
    TOP_K_DEFAULT: int = Field(default=2, ge=1, le=20)


    # Debug / observability

    DEBUG_LOGS: bool = Field(default=False)
    DEBUG_LLM: bool = Field(default=False)


    # Cost configuration (placeholders OK)

    COST_PER_1K_INPUT_TOKENS_USD: float = Field(default=0.0005, ge=0.0)
    COST_PER_1K_OUTPUT_TOKENS_USD: float = Field(default=0.0015, ge=0.0)


settings = Settings()
