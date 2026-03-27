"""
app/services/rag/pipeline.py
─────────────────────────────
Core RAG pipeline.
Flow: Query → Intent route → Retrieve → Rerank → Generate → Cite
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

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
    import re

    lower = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            logger.warning("security.prompt_injection_detected", query_preview=query[:80])
            raise ValueError("Query contains potentially harmful content and cannot be processed.")
    return query[:2000]


# ── Reranker ──────────────────────────────────────────────────────


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model: Any | None = None

    async def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int,
    ) -> list[DocumentChunk]:
        if not chunks:
            return chunks
        try:
            import asyncio

            from sentence_transformers import CrossEncoder

            if self._model is None:
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    None, lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                )
            pairs = [(query, c.content) for c in chunks]
            loop = asyncio.get_running_loop()
            model_ref = self._model

            def _predict(p: list[tuple[str, str]]) -> list[float]:
                return list(model_ref.predict(p).tolist())  # type: ignore[union-attr]

            scores: list[float] = await loop.run_in_executor(None, lambda: _predict(pairs))
            for chunk, score in zip(chunks, scores, strict=True):
                chunk.score = float(score)
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:top_k]
        except Exception as exc:
            logger.warning("reranker.fallback", error=str(exc))
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


_reranker = CrossEncoderReranker()


# ── Semantic cache ────────────────────────────────────────────────


class SemanticCache:
    def __init__(self, redis_client: aioredis.Redis) -> None:  # type: ignore[type-arg]
        self._r = redis_client

    def _key(self, query: str, plugin_id: str, rbac_context: str = "") -> str:
        from app.services.rag.embedding import EmbeddingService

        h = EmbeddingService.text_hash(f"{plugin_id}:{rbac_context}:{query.lower().strip()}")
        return f"sem_cache:{h}"

    async def get(self, query: str, plugin_id: str, rbac_context: str = "") -> QueryResponse | None:
        import orjson

        key = self._key(query, plugin_id, rbac_context)
        try:
            data = await self._r.get(key)
        except Exception as exc:
            logger.warning("semantic_cache.get_error", error=str(exc))
            return None
        if data:
            logger.debug("semantic_cache.hit", key=key)
            return QueryResponse.model_validate(orjson.loads(data))
        return None

    async def set(
        self,
        query: str,
        plugin_id: str,
        response: QueryResponse,
        rbac_context: str = "",
    ) -> None:
        import orjson

        key = self._key(query, plugin_id, rbac_context)
        try:
            await self._r.set(
                key,
                orjson.dumps(response.model_dump(mode="json")),
                ex=settings.REDIS_SEMANTIC_CACHE_TTL,
            )
            logger.debug("semantic_cache.set", key=key)
        except Exception as exc:
            logger.warning("semantic_cache.set_error", error=str(exc))


# ── Main RAG Pipeline ─────────────────────────────────────────────

# Type alias for audit log callback: (response, user_id) -> None
QueryLogCallback = Callable[[QueryResponse, str], Awaitable[None]]


class RAGPipeline:
    """
    Orchestrates the full RAG flow.
    Inject dependencies for testability.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis | None = None,  # type: ignore[type-arg]
        query_log_callback: QueryLogCallback | None = None,
    ) -> None:
        self._embedding = get_embedding_service()
        self._vector_store = get_vector_store()
        self._llm = get_llm_service()
        self._cache = SemanticCache(redis_client) if redis_client else None
        self._query_log_callback = query_log_callback

    def set_query_log_callback(self, callback: QueryLogCallback) -> None:
        """Wire in audit logging after DB session factory is available."""
        self._query_log_callback = callback

    def _resolve_plugin(self, request: QueryRequest, user: User) -> UseCasePlugin:
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

    def _build_rbac_filter(self, plugin: UseCasePlugin, user: User) -> dict[str, str]:
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
        if not chunks:
            return 0.0
        top_scores = [c.score for c in chunks[:3]]
        return round(sum(top_scores) / len(top_scores), 3)

    def _should_escalate(
        self,
        plugin: UseCasePlugin,
        query: str,
        confidence: float,
        has_chunks: bool = True,
    ) -> tuple[bool, str]:
        # Check escalation pattern FIRST (more important)
        if plugin.escalation_pattern:
            import re

            if re.search(plugin.escalation_pattern, query, re.IGNORECASE):
                return True, "Query matches escalation pattern"

        # Only check confidence when we actually have chunks to score
        if has_chunks and confidence < 0.50:
            return True, f"Low confidence score: {confidence:.2f}"

        return False, ""

    async def _persist_audit_log(self, response: QueryResponse, user: User) -> None:
        """Fire-and-forget audit log — never lets failures surface to the caller."""
        if self._query_log_callback is None:
            return
        try:
            await self._query_log_callback(response, str(user.id))
        except Exception as exc:
            logger.error("audit_log.failed", error=str(exc))

    async def query(
        self,
        request: QueryRequest,
        user: User,
    ) -> QueryResponse:
        start_ms = int(time.monotonic() * 1000)

        plugin = self._resolve_plugin(request, user)
        logger.info(
            "rag.pipeline.start",
            plugin=plugin.id,
            user=str(user.id),
            query_preview=request.query[:80],
        )

        rbac_filter = self._build_rbac_filter(plugin, user)
        rbac_ctx = ":".join(sorted(rbac_filter.values())) if rbac_filter else ""

        if settings.FEATURE_SEMANTIC_CACHE and self._cache:
            cached = await self._cache.get(request.query, plugin.id, rbac_ctx)
            if cached:
                logger.info("rag.pipeline.cache_hit", plugin=plugin.id)
                return cached

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

        cfg = plugin.retrieval
        chunks = await self._vector_store.search(
            collection_name=plugin.collection_name,
            query_vector=query_vector,
            top_k=cfg.top_k * 2,
            score_threshold=cfg.score_threshold,
            rbac_filter=rbac_filter if rbac_filter else None,
        )

        if not chunks:
            logger.warning("rag.pipeline.no_chunks", plugin=plugin.id)

            # Check escalation pattern even without chunks
            escalated, escalation_reason = self._should_escalate(
                plugin, request.query, 0.0, has_chunks=False
            )

            response = QueryResponse(
                query=request.query,
                answer="I don't have information about this topic in the available documents.",
                use_case_id=plugin.id,
                confidence=0.0,
                status=QueryStatus.ESCALATED if escalated else QueryStatus.COMPLETED,
                escalated=escalated,
                escalation_reason=escalation_reason if escalated else None,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
            )
            await self._persist_audit_log(response, user)
            return response

        if cfg.reranker_enabled:
            chunks = await _reranker.rerank(request.query, chunks, cfg.top_k)
        else:
            chunks = chunks[: cfg.top_k]

        system_prompt = self._llm.render_prompt(
            template_path=plugin.system_prompt_path,
            context={"chunks": chunks, "query": request.query, "plugin": plugin},
        )
        answer, token_usage = await self._llm.generate(
            system_prompt=system_prompt,
            user_message=request.query,
            use_case_id=plugin.id,
        )

        citations = self._build_citations(chunks) if plugin.citation_required else []
        confidence = self._compute_confidence(chunks)
        escalated, escalation_reason = self._should_escalate(plugin, request.query, confidence)

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

        if settings.FEATURE_SEMANTIC_CACHE and self._cache and not escalated:
            await self._cache.set(request.query, plugin.id, response, rbac_ctx)

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

        await self._persist_audit_log(response, user)
        return response

    async def stream(
        self,
        request: QueryRequest,
        user: User,
    ) -> AsyncIterator[str]:
        """Streaming variant — yields tokens as they arrive."""
        plugin = self._resolve_plugin(request, user)

        try:
            safe_query = _sanitize_query(request.query)
        except ValueError as exc:
            # Capture error message before exc goes out of scope
            error_msg = str(exc)

            async def _error_gen() -> AsyncIterator[str]:
                yield error_msg

            async for token in _error_gen():
                yield token
            return

        query_vector = await self._embedding.embed_query(safe_query)
        rbac_filter = self._build_rbac_filter(plugin, user)

        cfg = plugin.retrieval
        chunks = await self._vector_store.search(
            collection_name=plugin.collection_name,
            query_vector=query_vector,
            top_k=cfg.top_k * 2,
            score_threshold=cfg.score_threshold,
            rbac_filter=rbac_filter if rbac_filter else None,
        )

        if chunks and cfg.reranker_enabled:
            chunks = await _reranker.rerank(request.query, chunks, cfg.top_k)
        else:
            chunks = chunks[: cfg.top_k]

        system_prompt = self._llm.render_prompt(
            template_path=plugin.system_prompt_path,
            context={"chunks": chunks, "query": request.query, "plugin": plugin},
        )

        async for token in self._llm.stream(system_prompt, request.query, use_case_id=plugin.id):
            yield token
