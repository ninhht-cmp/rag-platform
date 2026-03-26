"""
app/core/config.py
──────────────────
Central configuration using pydantic-settings.
All env vars validated at startup — fail fast, fail loud.
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL      = "local"
    STAGING    = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"
    LOCAL     = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────
    APP_NAME:    str         = "RAG Platform"
    APP_VERSION: str         = "1.0.0"
    ENVIRONMENT: Environment = Environment.LOCAL
    DEBUG:       bool        = False
    SECRET_KEY:  str         = Field(min_length=32)
    ALLOWED_HOSTS: list[str] = ["*"]
    API_PREFIX:  str         = "/api/v1"

    # ── LLM ──────────────────────────────────────────
    LLM_PROVIDER:         LLMProvider = LLMProvider.ANTHROPIC
    ANTHROPIC_API_KEY:    str         = ""
    OPENAI_API_KEY:       str         = ""
    LLM_MODEL_PRIMARY:    str         = "claude-sonnet-4-6"
    LLM_MODEL_SECONDARY:  str         = "claude-haiku-4-5-20251001"
    LLM_MAX_TOKENS:       int         = 4096
    LLM_TEMPERATURE:      float       = 0.1
    LLM_DAILY_BUDGET_USD: float       = 50.0

    # ── Embedding ────────────────────────────────────
    EMBEDDING_MODEL:      str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION:  int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE:     str = "cpu"

    # ── Qdrant ───────────────────────────────────────
    QDRANT_URL:     str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_TIMEOUT: int = 30

    # ── Redis ────────────────────────────────────────
    REDIS_URL:                RedisDsn = "redis://localhost:6379/0"  # type: ignore[assignment]
    REDIS_SESSION_TTL:        int      = 1800
    REDIS_SEMANTIC_CACHE_TTL: int      = 86400

    # ── PostgreSQL ───────────────────────────────────
    DATABASE_URL:          PostgresDsn = (
        "postgresql+asyncpg://rag:rag@localhost:5432/rag_platform"  # type: ignore[assignment]
    )
    DATABASE_POOL_SIZE:    int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── RAG ──────────────────────────────────────────
    RAG_CHUNK_SIZE:       int   = 512
    RAG_CHUNK_OVERLAP:    int   = 64
    RAG_TOP_K:            int   = 5
    RAG_SCORE_THRESHOLD:  float = 0.7
    RAG_RERANKER_ENABLED: bool  = True
    RAG_HYBRID_SEARCH:    bool  = True
    RAG_BM25_WEIGHT:      float = 0.3
    RAG_SEMANTIC_WEIGHT:  float = 0.7

    # ── Auth ─────────────────────────────────────────
    JWT_ALGORITHM:                    str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES:  int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS:    int = 7

    # ── Observability ────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str  = ""
    LANGFUSE_SECRET_KEY: str  = ""
    LANGFUSE_HOST:       str  = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED:    bool = False
    LOG_LEVEL:           str  = "INFO"
    LOG_FORMAT:          str  = "json"

    # ── Evaluation ───────────────────────────────────
    RAGAS_EVAL_ENABLED:       bool  = True
    RAGAS_MIN_FAITHFULNESS:   float = 0.82
    RAGAS_MIN_RELEVANCY:      float = 0.78
    RAGAS_MIN_CONTEXT_RECALL: float = 0.75

    # ── Feature flags ────────────────────────────────
    FEATURE_STREAMING:      bool = True
    FEATURE_AGENT_TOOLS:    bool = True
    FEATURE_SEMANTIC_CACHE: bool = True
    FEATURE_CITATION:       bool = True

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @model_validator(mode="after")
    def validate_api_keys(self) -> Settings:
        if self.ENVIRONMENT == Environment.PRODUCTION:
            if self.LLM_PROVIDER == LLMProvider.ANTHROPIC and not self.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY required in production")
            if self.LLM_PROVIDER == LLMProvider.OPENAI and not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY required in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @property
    def qdrant_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"url": self.QDRANT_URL, "timeout": self.QDRANT_TIMEOUT}
        if self.QDRANT_API_KEY:
            cfg["api_key"] = self.QDRANT_API_KEY
        return cfg

    def model_post_init(self, __context: Any) -> None:
        """Validate SECRET_KEY strength in production."""
        if self.ENVIRONMENT == Environment.PRODUCTION and len(self.SECRET_KEY) < 64:
            import warnings
            warnings.warn(
                "SECRET_KEY should be at least 64 characters in production",
                stacklevel=2,
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
