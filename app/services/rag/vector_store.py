"""
app/services/rag/vector_store.py
─────────────────────────────────
Qdrant abstraction layer.
- Collection-per-use-case (namespace isolation)
- RBAC via payload filters
- Hybrid search (semantic + BM25 sparse)
- Automatic collection creation with correct schema
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import TextIndexType
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.models.domain import DocumentChunk

logger = get_logger(__name__)


class VectorStore:
    """
    Async Qdrant client wrapper.
    One instance per application — shared via DI.
    """

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("VectorStore not initialized. Call startup() first.")
        return self._client

    async def startup(self) -> None:
        self._client = AsyncQdrantClient(**settings.qdrant_config)
        logger.info("vector_store.connected", url=settings.QDRANT_URL)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("vector_store.disconnected")

    async def ensure_collection(self, collection_name: str) -> None:
        """
        Idempotent collection creation.
        Uses cosine distance + HNSW index for fast ANN search.
        """
        existing = await self.client.get_collections()
        names = {c.name for c in existing.collections}

        if collection_name in names:
            logger.debug("vector_store.collection.exists", name=collection_name)
            return

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qm.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=qm.Distance.COSINE,
                hnsw_config=qm.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                ),
                on_disk=True,  # memory-efficient for large collections
            ),
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
            replication_factor=1,
        )
        # Create payload indexes for fast filtering (RBAC)
        for field_name in ["user_id", "department", "access_level", "document_id"]:
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        # Full-text index on content field for BM25/keyword search
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="content",
            field_schema=qm.TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=qm.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        logger.info("vector_store.collection.created", name=collection_name)

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists without raising."""
        try:
            existing = await self.client.get_collections()
            return collection_name in {c.name for c in existing.collections}
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """Batch upsert chunks. Returns count of upserted points."""
        if not chunks:
            return 0

        points = [
            qm.PointStruct(
                id=chunk.id,
                vector=chunk.embedding or [],
                payload={
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
            if chunk.embedding is not None
        ]

        await self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            "vector_store.upserted",
            collection=collection_name,
            count=len(points),
        )
        return len(points)

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        score_threshold: float,
        rbac_filter: dict[str, Any] | None = None,
        query_text: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Semantic search with optional RBAC filter.

        Returns empty list (instead of raising) when collection does not exist yet.
        This is the correct behavior: no documents uploaded → no results → pipeline
        returns "I don't have information" instead of 500 Internal Server Error.
        """
        # ── Guard: collection not yet created ────────────────────
        # Happens when querying before any documents have been uploaded.
        # Return empty list so pipeline responds gracefully.
        if not await self.collection_exists(collection_name):
            logger.info(
                "vector_store.collection_not_found",
                collection=collection_name,
                note="No documents uploaded yet — returning empty results",
            )
            return []

        # ── Build RBAC filter ─────────────────────────────────────
        qdrant_filter: qm.Filter | None = None
        if rbac_filter:
            qdrant_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key=k,
                        match=qm.MatchValue(value=v),
                    )
                    for k, v in rbac_filter.items()
                ]
            )

        # ── Search ────────────────────────────────────────────────
        search_result = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        chunks: list[DocumentChunk] = []
        for r in search_result.points:
            payload = r.payload or {}
            chunks.append(
                DocumentChunk(
                    id=str(r.id),
                    document_id=payload.get("document_id", ""),
                    content=payload.get("content", ""),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("content", "document_id", "chunk_index")
                    },
                    chunk_index=payload.get("chunk_index", 0),
                    score=r.score,
                )
            )

        logger.debug(
            "vector_store.search",
            collection=collection_name,
            results=len(chunks),
            top_score=chunks[0].score if chunks else 0,
        )
        return chunks

    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        """Remove all chunks belonging to a document."""
        await self.client.delete(
            collection_name=collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info(
            "vector_store.deleted",
            collection=collection_name,
            document_id=document_id,
        )

    async def health_check(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False


# ── Singleton ─────────────────────────────────────────────────────
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
