"""
app/api/v1/endpoints/query.py
──────────────────────────────
Query endpoints:
- POST /query         — standard query
- POST /query/stream  — SSE streaming

The pipeline singleton is initialized lazily so set_query_log_callback()
wired in main.py lifespan is applied before the first request.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.middleware.auth import get_current_user
from app.core.logging import get_logger
from app.models.domain import QueryRequest, QueryResponse, User
from app.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return shared pipeline instance. Redis cache wired after startup."""
    global _pipeline
    if _pipeline is None:
        try:
            from app.main import get_redis
            _pipeline = RAGPipeline(redis_client=get_redis())
        except Exception:
            _pipeline = RAGPipeline()
    return _pipeline


@router.post("", response_model=QueryResponse, summary="Query the RAG platform")
async def query(
    request: QueryRequest,
    user: User = Depends(get_current_user),  # noqa: B008
    pipeline: RAGPipeline = Depends(get_pipeline),  # noqa: B008
) -> QueryResponse:
    from app.core.plugin_registry import registry
    from app.services.agent.agent_service import AgentService

    if request.use_case_id:
        plugin = registry.get(request.use_case_id)
    else:
        user_roles = [r.value for r in user.roles]
        plugin = (
            registry.route_by_intent(request.query, user_roles)
            or registry.get("knowledge_base")
        )

    if plugin and plugin.agent_tools:
        agent = AgentService()
        return await agent.run(request, user, plugin)

    return await pipeline.query(request, user)


@router.post("/stream", summary="Stream query response (SSE)")
async def query_stream(
    request: QueryRequest,
    user: User = Depends(get_current_user),  # noqa: B008
    pipeline: RAGPipeline = Depends(get_pipeline),  # noqa: B008
) -> StreamingResponse:
    async def _generate() -> AsyncIterator[str]:
        async for token in pipeline.stream(request, user):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
