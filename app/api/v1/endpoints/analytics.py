"""
app/api/v1/endpoints/analytics.py
───────────────────────────────────
Admin analytics endpoints.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.middleware.auth import require_roles
from app.core.logging import get_logger
from app.core.plugin_registry import registry
from app.models.domain import EvalMetrics, Role
from app.services.evaluation.eval_service import EvalSample, get_eval_service

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

_admin_required = require_roles(Role.ADMIN)


@router.get("/stats", summary="Query statistics per use case")
async def get_stats(
    use_case_id: str | None = None,
    days: int = 7,
    _: Annotated[object, Depends(_admin_required)] = None,  # noqa: B008
) -> dict[str, object]:
    """
    Returns query volume, avg confidence, escalation rate, and token usage per use case.
    Queries real DB; falls back to empty stats if DB is unavailable.
    """
    try:
        from app.repositories.document_repository import QueryLogRepository, get_session_factory

        async with get_session_factory()() as session:
            repo = QueryLogRepository(session)
            return await repo.get_stats(use_case_id=use_case_id, days=days)
    except Exception as exc:
        logger.warning("analytics.stats.db_unavailable", error=str(exc))
        # Graceful degradation: return empty stats with a note
        active_plugins = registry.get_active()
        stats = []
        for p in active_plugins:
            if use_case_id and p.id != use_case_id:
                continue
            stats.append(
                {
                    "use_case_id": p.id,
                    "name": p.name,
                    "total_queries": 0,
                    "avg_confidence": 0.0,
                    "avg_latency_ms": 0,
                    "escalated_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "note": "DB unavailable — showing empty stats",
                }
            )
        return {"period_days": days, "use_cases": stats}


@router.get(
    "/eval/{use_case_id}",
    response_model=EvalMetrics | None,
    summary="Latest evaluation results for a use case",
)
async def get_eval_results(
    use_case_id: str,
    _: Annotated[object, Depends(_admin_required)] = None,  # noqa: B008
) -> EvalMetrics | None:
    if registry.get(use_case_id) is None:
        raise HTTPException(status_code=404, detail=f"Use case '{use_case_id}' not found")

    try:
        from app.repositories.document_repository import EvalRepository, get_session_factory

        async with get_session_factory()() as session:
            repo = EvalRepository(session)
            return await repo.latest(use_case_id)
    except Exception as exc:
        logger.warning("analytics.eval.db_unavailable", error=str(exc))
        return None


@router.post(
    "/eval/{use_case_id}",
    response_model=EvalMetrics,
    summary="Trigger evaluation run",
)
async def trigger_eval(
    use_case_id: str,
    _: Annotated[object, Depends(_admin_required)] = None,  # noqa: B008
) -> EvalMetrics:
    if registry.get(use_case_id) is None:
        raise HTTPException(status_code=404, detail=f"Use case '{use_case_id}' not found")

    samples = [
        EvalSample(
            question="What is the main purpose of this system?",
            ground_truth="This is a RAG platform for enterprise AI use cases.",
            contexts=[
                "The RAG Platform enables companies to build AI applications on their own data.",
            ],
            answer="The system is an enterprise RAG platform for building AI applications.",
        ),
        EvalSample(
            question="How many use cases are supported?",
            ground_truth="Four use cases: knowledge base, support, document Q&A, and sales.",
            contexts=[
                "The platform supports four use cases: "
                "Internal KB, Customer Support, Document Q&A, Sales Automation.",
            ],
            answer="The platform supports four use cases.",
        ),
    ]

    svc = get_eval_service()
    result = await svc.evaluate_plugin(use_case_id, samples)

    # Persist result
    try:
        from app.repositories.document_repository import EvalRepository, get_session_factory

        async with get_session_factory()() as session:
            repo = EvalRepository(session)
            await repo.save_result(result)
    except Exception as exc:
        logger.warning("analytics.eval.persist_failed", error=str(exc))

    logger.info(
        "eval.triggered",
        use_case=use_case_id,
        passed=result.passed,
        faithfulness=result.faithfulness,
    )
    return result
