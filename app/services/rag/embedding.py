"""
app/services/rag/embedding.py
──────────────────────────────
Embedding service with:
- Lazy model loading (avoid cold-start penalty on import)
- Batch processing with configurable size
- Retry on transient failures
- Fallback: sentence-transformers (local) → OpenAI API

Backend priority:
1. sentence-transformers + torch — free, private, runs locally
2. OpenAI text-embedding-3-small — fallback when torch unavailable (e.g. Intel Mac)
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Async embedding interface with automatic backend selection.
    Model loaded lazily on first use — not at import time.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._backend: str = "unknown"
        self._lock = asyncio.Lock()

    async def _load_model(self) -> Any:
        """Load model once, thread-safe. Tries local first, falls back to OpenAI."""
        if self._model is not None:
            return self._model

        async with self._lock:
            if self._model is not None:  # double-check after acquiring lock
                return self._model

            loop = asyncio.get_running_loop()

            # Try sentence-transformers first (local, free, private)
            st_model = await loop.run_in_executor(None, self._load_sentence_transformer)

            if st_model is not None:
                self._model = st_model
                self._backend = "sentence_transformers"
                logger.info(
                    "embedding.model.loaded",
                    backend="local",
                    model=settings.EMBEDDING_MODEL,
                )
            else:
                # Fallback to OpenAI embedding API
                logger.warning(
                    "embedding.backend.fallback",
                    reason="sentence_transformers/torch not available",
                    fallback="openai",
                )
                from openai import AsyncOpenAI

                self._model = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                self._backend = "openai"
                logger.info("embedding.model.loaded", backend="openai")

        return self._model

    def _load_sentence_transformer(self) -> Any:
        """
        Try to load sentence-transformers model.
        Returns None (instead of raising) if torch/sentence-transformers unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            return SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
        except Exception:
            return None  # caller will use OpenAI fallback

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts. Returns list of float vectors.
        Routes to correct backend automatically.
        """
        if not texts:
            return []

        model = await self._load_model()

        # ── OpenAI API path ───────────────────────────────────────
        if self._backend == "openai":
            cleaned = [t.replace("\n", " ") for t in texts]
            response = await model.embeddings.create(
                model="text-embedding-3-small",
                input=cleaned,
            )
            embeddings = [item.embedding for item in response.data]
            logger.debug(
                "embedding.computed",
                backend="openai",
                count=len(texts),
                dimension=len(embeddings[0]) if embeddings else 0,
            )
            return embeddings

        # ── Local sentence-transformers path ──────────────────────
        loop = asyncio.get_running_loop()
        batch_size = settings.EMBEDDING_BATCH_SIZE
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings: np.ndarray = await loop.run_in_executor(
                None,
                lambda b=batch: model.encode(
                    b,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
            )
            all_embeddings.extend(embeddings.tolist())

        logger.debug(
            "embedding.computed",
            backend="local",
            count=len(texts),
            dimension=len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]

    @staticmethod
    def text_hash(text: str) -> str:
        """Stable hash for semantic cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
