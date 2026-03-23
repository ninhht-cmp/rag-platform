"""
app/services/rag/pipeline.py
─────────────────────────────
Core RAG pipeline.
Flow: Query → Intent route → Retrieve → Rerank → Generate → Cite

Design decisions:
- Stateless pipeline: all context passed explicitly (testable)
- Semantic cache: skip retrieval+generation for repeated queries
- Confidence scoring: detect low-confidence → escalation trigger
- Citation builder: every answer has traceable sources
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger
from app.core.plugin_registry import UseCasePlugin, registry
from app.models.domain import (
    Citation,
    DocumentChunk,
    QueryRequest,
    QueryResponse,
    QueryStatus,
    User,
)
from app.services.rag.embedding import get_embedding_service
from app.services.rag.llm_service import get_llm_service
from app.services.rag.vector_store import get_vector_store

logger = get_logger(__name__)

# ── Prompt injection sanitizer ────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions?",
    r"forget\s+(everything|all|your\s+instructions?)",
    r"you\s+are\s+now\s+(?!a\s+helpful)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak|dan\s+mode|developer\s+mode",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
]


def _sanitize_query(query: str) -> str:
    """
    Detect and neutralize obvious prompt injection attempts.
    Raises ValueError if malicious content detected.
    """
    import re
    lower = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            logger.warning("security.prompt_injection_detected", query_preview=query[:80])
            raise ValueError("Query contains potentially harmful content and cannot be processed.")
    return query[:2000]


# ── Reranker (cross-encoder) ──────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-encoder reranker for precise scoring.
    Falls back to original vector scores if model unavailable.
    """

    def __init__(self) -> None:
        self._model: object | None = None

    async def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int,
    ) -> list[DocumentChunk]:
        if not chunks:
            return chunks

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            import asyncio
            if self._model is None:
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    None, lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                )

            pairs = [(query, c.content) for c in chunks]
            loop = asyncio.get_running_loop()
            scores: list[float] = await loop.run_in_executor(
                None, lambda p=pairs: self._model.predict(p).tolist()  # type: ignore[union-attr]
            )

            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)

            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:top_k]

        except Exception as exc:
            logger.warning("reranker.fallback", error=str(exc))
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


_reranker = CrossEncoderReranker()


# ── Semantic cache ────────────────────────────────────────────────

class SemanticCache:
    """
    Cache query answers in Redis.
    Key: sha256(plugin_id + rbac_context + query)
    TTL: 24h (configurable)
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:  # type: ignore[type-arg]
        self._r = redis_client

    def _key(self, query: str, plugin_id: str, rbac_context: str = "") -> str:
        from app.services.rag.embedding import EmbeddingService
        # SECURITY: include RBAC context in key to prevent cross-user cache leakage
        h = EmbeddingService.text_hash(f"{plugin_id}:{rbac_context}:{query.lower().strip()}")
        return f"sem_cache:{h}"

    async def get(self, query: str, plugin_id: str, rbac_context: str = "") -> QueryResponse | None:
        import orjson
        key = self._key(query, plugin_id, rbac_context)
        data = await self._r.get(key)
        if data:
            logger.debug("semantic_cache.hit", key=key)
            return QueryResponse.model_validate(orjson.loads(data))
        return None

    async def set(self, query: str, plugin_id: str, response: QueryResponse, rbac_context: str = "") -> None:
        import orjson
        key = self._key(query, plugin_id, rbac_context)
        await self._r.set(
            key,
            orjson.dumps(response.model_dump(mode="json")),
            ex=settings.REDIS_SEMANTIC_CACHE_TTL,
        )
        logger.debug("semantic_cache.set", key=key)


# ── Main RAG Pipeline ─────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full RAG flow.
    Inject dependencies for testability.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._embedding = get_embedding_service()
        self._vector_store = get_vector_store()
        self._llm = get_llm_service()
        self._cache = SemanticCache(redis_client) if redis_client else None

    def _resolve_plugin(
        self,
        request: QueryRequest,
        user: User,
    ) -> UseCasePlugin:
        """Route request to correct plugin."""
        if request.use_case_id:
            plugin = registry.get(request.use_case_id)
            if plugin is None:
                raise ValueError(f"Unknown use_case_id: {request.use_case_id}")
            return plugin

        user_roles = [str(r) for r in user.roles]
        plugin = registry.route_by_intent(request.query, user_roles)
        if plugin:
            return plugin

        default = registry.get("knowledge_base")
        if default:
            return default
        raise ValueError("No active plugins registered")

    def _build_rbac_filter(
        self,
        plugin: UseCasePlugin,
        user: User,
    ) -> dict[str, str]:
        """Build Qdrant payload filter from RBAC rules."""
        filters = dict(plugin.rbac.metadata_filters)
        for k, v in list(filters.items()):
            if v == "${user.department}":
                filters[k] = user.department
            elif v == "${user.id}":
                filters[k] = str(user.id)
        return filters

    def _build_citations(self, chunks: list[DocumentChunk]) -> list[Citation]:
        seen: set[str] = set()
        citations: list[Citation] = []
        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id in seen:
                continue
            seen.add(doc_id)
            citations.append(
                Citation(
                    document_id=doc_id,
                    filename=chunk.metadata.get("filename", "Unknown"),
                    chunk_id=chunk.id,
                    content_preview=chunk.content[:200],
                    score=round(chunk.score, 4),
                    page_number=chunk.metadata.get("page_number"),
                    url=chunk.metadata.get("url"),
                )
            )
        return citations

    def _compute_confidence(self, chunks: list[DocumentChunk]) -> float:
        """Confidence from top-k retrieval scores (0-1)."""
        if not chunks:
            return 0.0
        top_scores = [c.score for c in chunks[:3]]
        return round(sum(top_scores) / len(top_scores), 3)

    def _should_escalate(
        self,
        plugin: UseCasePlugin,
        query: str,
        confidence: float,
    ) -> tuple[bool, str]:
        """Check escalation triggers."""
        if confidence < 0.50:
            return True, f"Low confidence score: {confidence:.2f}"

        if plugin.escalation_pattern:
            import re
            if re.search(plugin.escalation_pattern, query, re.IGNORECASE):
                return True, "Query matches escalation pattern"

        return False, ""

    async def query(
        self,
        request: QueryRequest,
        user: User,
    ) -> QueryResponse:
        """
        Full RAG pipeline:
        1. Resolve plugin
        2. Build RBAC filter (must be before cache check to prevent cross-user leak)
        3. Check semantic cache
        4. Sanitize + Embed query
        5. Retrieve chunks (with RBAC filter)
        6. Rerank
        7. Generate answer
        8. Build citations
        9. Evaluate confidence + escalation check
        10. Cache result
        """
        start_ms = int(time.monotonic() * 1000)

        # 1. Resolve plugin
        plugin = self._resolve_plugin(request, user)
        logger.info(
            "rag.pipeline.start",
            plugin=plugin.id,
            user=str(user.id),
            query_preview=request.query[:80],
        )

        # 2. Build RBAC filter — must happen before cache check
        rbac_filter = self._build_rbac_filter(plugin, user)
        rbac_ctx = ":".join(sorted(rbac_filter.values())) if rbac_filter else ""

        # 3. Semantic cache
        if settings.FEATURE_SEMANTIC_CACHE and self._cache:
            cached = await self._cache.get(request.query, plugin.id, rbac_ctx)
            if cached:
                logger.info("rag.pipeline.cache_hit", plugin=plugin.id)
                return cached

        # 4. Sanitize + Embed query
        try:
            safe_query = _sanitize_query(request.query)
        except ValueError as exc:
            return QueryResponse(
                query=request.query,
                answer=str(exc),
                use_case_id=plugin.id,
                confidence=0.0,
                status=QueryStatus.FAILED,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
            )
        query_vector = await self._embedding.embed_query(safe_query)

        # 5. Retrieve
        cfg = plugin.retrieval
        chunks = await self._vector_store.search(
            collection_name=plugin.collection_name,
            query_vector=query_vector,
            top_k=cfg.top_k * 2,    # over-fetch for reranking
            score_threshold=cfg.score_threshold,
            rbac_filter=rbac_filter if rbac_filter else None,
        )

        if not chunks:
            logger.warning("rag.pipeline.no_chunks", plugin=plugin.id)
            return QueryResponse(
                query=request.query,
                answer="I don't have information about this topic in the available documents.",
                use_case_id=plugin.id,
                confidence=0.0,
                status=QueryStatus.COMPLETED,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
            )

        # 6. Rerank
        if cfg.reranker_enabled:
            chunks = await _reranker.rerank(request.query, chunks, cfg.top_k)
        else:
            chunks = chunks[: cfg.top_k]

        # 7. Generate
        system_prompt = self._llm.render_prompt(
            template_path=plugin.system_prompt_path,
            context={"chunks": chunks, "query": request.query, "plugin": plugin},
        )
        answer, token_usage = await self._llm.generate(
            system_prompt=system_prompt,
            user_message=request.query,
        )

        # 8. Citations
        citations = self._build_citations(chunks) if plugin.citation_required else []

        # 9. Confidence + escalation
        confidence = self._compute_confidence(chunks)
        escalated, escalation_reason = self._should_escalate(
            plugin, request.query, confidence
        )

        latency_ms = int(time.monotonic() * 1000) - start_ms
        response = QueryResponse(
            query=request.query,
            answer=answer,
            use_case_id=plugin.id,
            citations=citations,
            confidence=confidence,
            status=QueryStatus.ESCALATED if escalated else QueryStatus.COMPLETED,
            escalated=escalated,
            escalation_reason=escalation_reason if escalated else None,
            session_id=request.session_id,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )

        # 10. Cache (only non-escalated responses)
        if settings.FEATURE_SEMANTIC_CACHE and self._cache and not escalated:
            await self._cache.set(request.query, plugin.id, response, rbac_ctx)

        # Estimate and log cost
        try:
            from app.utils.helpers import estimate_cost_usd
            cost = estimate_cost_usd(
                settings.LLM_MODEL_PRIMARY,
                token_usage.get("input_tokens", 0),
                token_usage.get("output_tokens", 0),
            )
        except Exception:
            cost = 0.0

        logger.info(
            "rag.pipeline.complete",
            plugin=plugin.id,
            latency_ms=latency_ms,
            confidence=confidence,
            escalated=escalated,
            chunks_used=len(chunks),
            cost_usd=round(cost, 6),
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0),
        )
        return response

    async def stream(
        self,
        request: QueryRequest,
        user: User,
    ) -> AsyncIterator[str]:
        """Streaming variant — yields tokens as they arrive."""
        plugin = self._resolve_plugin(request, user)
        query_vector = await self._embedding.embed_query(request.query)
        rbac_filter = self._build_rbac_filter(plugin, user)

        cfg = plugin.retrieval
        chunks = await self._vector_store.search(
            collection_name=plugin.collection_name,
            query_vector=query_vector,
            top_k=cfg.top_k * 2,
            score_threshold=cfg.score_threshold,
            rbac_filter=rbac_filter if rbac_filter else None,
        )

        # Apply same reranking as non-streaming for consistent quality
        if chunks and cfg.reranker_enabled:
            chunks = await _reranker.rerank(request.query, chunks, cfg.top_k)
        else:
            chunks = chunks[:cfg.top_k]

        system_prompt = self._llm.render_prompt(
            template_path=plugin.system_prompt_path,
            context={"chunks": chunks, "query": request.query, "plugin": plugin},
        )

        async for token in self._llm.stream(system_prompt, request.query):
            yield token