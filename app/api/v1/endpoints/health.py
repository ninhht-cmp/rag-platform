"""
app/api/v1/endpoints/health.py
───────────────────────────────
Health check and admin endpoints.
Used by load balancer, K8s liveness/readiness probes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.middleware.auth import require_roles
from app.core.config import settings
from app.core.plugin_registry import registry
from app.models.domain import HealthStatus, Role
from app.services.rag.vector_store import get_vector_store

router = APIRouter(tags=["Health"])


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
        ok = await vs.health_check()
        components["qdrant"] = "healthy" if ok else "unhealthy"
        if not ok:
            overall = "degraded"
    except Exception as exc:
        components["qdrant"] = f"error: {exc}"
        overall = "unhealthy"

    # Check Redis
    try:
        from app.main import get_redis
        r = get_redis()
        await r.ping()
        components["redis"] = "healthy"
    except Exception as exc:
        components["redis"] = f"error: {type(exc).__name__}"
        overall = "degraded"

    # Check PostgreSQL (lightweight check — just attempt connection)
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from app.core.config import settings
        engine = create_async_engine(str(settings.DATABASE_URL), pool_pre_ping=True)
        async with engine.connect():
            components["postgres"] = "healthy"
        await engine.dispose()
    except Exception as exc:
        components["postgres"] = f"unavailable: {type(exc).__name__}"
        # Postgres down is degraded, not unhealthy (app still works without audit log)
        if overall == "healthy":
            overall = "degraded"

    # Check plugin registry
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
    dependencies=[Depends(require_roles(Role.ADMIN))],
)
async def list_plugins() -> dict:
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
