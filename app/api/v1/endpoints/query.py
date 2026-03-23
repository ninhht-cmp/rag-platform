"""
app/api/v1/endpoints/query.py
──────────────────────────────
Query endpoints:
- POST /query         — standard query
- POST /query/stream  — SSE streaming
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.middleware.auth import get_current_user
from app.core.logging import get_logger
from app.models.domain import QueryRequest, QueryResponse, User
from app.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

# Module-level singleton — reused across requests so redis_client is shared
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return shared pipeline instance with redis client for semantic cache."""
    global _pipeline
    if _pipeline is None:
        try:
            from app.main import get_redis
            _pipeline = RAGPipeline(redis_client=get_redis())
        except Exception:
            # Redis not available (tests, cold start) — cache disabled
            _pipeline = RAGPipeline()
    return _pipeline


@router.post("", response_model=QueryResponse, summary="Query the RAG platform")
async def query(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Submit a query to the RAG platform.
    The platform auto-routes to the correct use case unless `use_case_id` is specified.

    If the matched plugin has agent_tools configured, the AgentService runs
    a ReAct loop (RAG + tool execution). Otherwise, plain RAG pipeline.

    Returns:
    - `answer`: generated response
    - `citations`: source documents (if plugin has citation_required=True)
    - `confidence`: retrieval confidence score (0-1)
    - `escalated`: true if query was escalated to human agent
    """
    from app.core.plugin_registry import registry
    from app.services.agent.agent_service import AgentService

    # Resolve plugin to check if it needs agentic routing
    if request.use_case_id:
        plugin = registry.get(request.use_case_id)
    else:
        user_roles = [r.value for r in user.roles]
        plugin = registry.route_by_intent(request.query, user_roles) or registry.get("knowledge_base")

    # Use AgentService when plugin has tools configured
    if plugin and plugin.agent_tools:
        agent = AgentService()
        return await agent.run(request, user, plugin)

    # Default: plain RAG pipeline
    return await pipeline.query(request, user)


@router.post("/stream", summary="Stream query response (SSE)")
async def query_stream(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """
    Stream the answer token by token via Server-Sent Events.
    Use this for real-time UI updates.
    """
    async def _generate():  # type: ignore[return]
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
