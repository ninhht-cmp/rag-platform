"""
app/api/v1/endpoints/health.py
───────────────────────────────
Health check endpoints — liveness & readiness probes.

FIXES applied:
- [CONCERN] readiness() no longer creates a new SQLAlchemy engine per probe call.
  K8s probes fire every few seconds; creating an engine each time leaks connections.
  Now uses a module-level reusable engine (lazy-initialized once).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.api.v1.middleware.auth import require_roles
from app.core.config import settings
from app.core.logging import get_logger
from app.core.plugin_registry import registry
from app.models.domain import HealthStatus, Role
from app.services.rag.vector_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

_health_engine: AsyncEngine | None = None


def _get_health_engine() -> AsyncEngine:
    global _health_engine
    if _health_engine is None:
        _health_engine = create_async_engine(
            str(settings.DATABASE_URL),
            pool_size=1,
            max_overflow=0,
            pool_pre_ping=True,
        )
    return _health_engine


@router.get("/health", response_model=HealthStatus, summary="Health check")
async def health() -> HealthStatus:
    """
    Liveness probe — returns 200 if app is running.
    Does NOT check downstream dependencies (use /health/ready for that).
    """
    return HealthStatus(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/health/ready", response_model=HealthStatus, summary="Readiness check")
async def readiness() -> HealthStatus:
    """
    Readiness probe — checks all critical dependencies.
    K8s will stop routing traffic if this returns non-200.
    """
    components: dict[str, str] = {}
    overall = "healthy"

    # Check Qdrant
    try:
        vs = get_vector_store()
        ok: bool = await vs.health_check()  # type: ignore[misc]
        components["qdrant"] = "healthy" if ok else "unhealthy"
        if not ok:
            overall = "degraded"
    except Exception as exc:
        components["qdrant"] = f"error: {type(exc).__name__}"
        overall = "unhealthy"

    # Check Redis
    try:
        from app.main import get_redis

        r = get_redis()
        await r.ping()  # type: ignore[misc]
        components["redis"] = "healthy"
    except Exception as exc:
        components["redis"] = f"error: {type(exc).__name__}"
        overall = "degraded"

    # Check PostgreSQL — reuse shared engine (no new engine per probe)
    try:
        engine = _get_health_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
        components["postgres"] = "healthy"
    except Exception as exc:
        components["postgres"] = f"unavailable: {type(exc).__name__}"
        if overall == "healthy":
            overall = "degraded"

    active = len(registry.get_active())
    components["plugins"] = f"{active} active"

    return HealthStatus(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        components=components,
    )


@router.get(
    "/admin/plugins",
    summary="List registered plugins",
    dependencies=[Depends(require_roles(Role.ADMIN))],  # noqa: B008
)
async def list_plugins() -> dict[str, Any]:
    plugins = registry.get_active()
    return {
        "total": len(plugins),
        "plugins": [
            {
                "id": p.id,
                "name": p.name,
                "status": p.status,
                "collection": p.collection_name,
                "tools": p.agent_tools,
            }
            for p in plugins
        ],
    }
