"""
app/main.py
────────────
FastAPI application factory.
Lifespan: connect infra on startup, graceful shutdown.
Middleware: CORS, rate limiter, request ID, timing, error handling.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import analytics, auth, health, ingestion, query
from app.api.v1.middleware.rate_limiter import RateLimiterMiddleware
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.plugins import register_all_plugins
from app.services.rag.vector_store import get_vector_store

setup_logging()
logger = get_logger(__name__)


# ── Redis pool ────────────────────────────────────────────────────
_redis: aioredis.Redis | None = None  # type: ignore[type-arg]


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


# ── Budget checker (DB-backed) ─────────────────────────────────────
async def _budget_checker(use_case_id: str) -> None:
    """
    FIX: Enforce daily LLM budget. Raises HTTP 429 when limit exceeded.
    Called by LLMService.generate() before every LLM call.
    """
    from fastapi import HTTPException

    from app.repositories.document_repository import TokenUsageRepository, get_session_factory

    try:
        async with get_session_factory()() as session:
            repo = TokenUsageRepository(session)
            today_cost = await repo.daily_cost(use_case_id)
            if today_cost >= settings.LLM_DAILY_BUDGET_USD:
                logger.warning(
                    "budget.exceeded",
                    use_case=use_case_id,
                    today_cost=today_cost,
                    limit=settings.LLM_DAILY_BUDGET_USD,
                )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Daily LLM budget exceeded for use case '{use_case_id}'. "
                        f"Spent: ${today_cost:.2f} / ${settings.LLM_DAILY_BUDGET_USD:.2f}. "
                        "Budget resets at midnight UTC."
                    ),
                )
    except HTTPException:
        raise
    except Exception as exc:
        # DB down → fail open (don't block all requests)
        logger.error("budget.check.db_error", error=str(exc))


# ── Query audit log callback ───────────────────────────────────────
async def _query_log_callback(response: object, user_id: str) -> None:
    """
    FIX: Persist every query to query_logs table.
    Called by RAGPipeline after each successful query.
    """
    from app.repositories.document_repository import QueryLogRepository, get_session_factory

    try:
        async with get_session_factory()() as session:
            repo = QueryLogRepository(session)
            await repo.log(response, user_id)  # type: ignore[arg-type]
    except Exception as exc:
        logger.error("query_log.persist.failed", error=str(exc))


# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _redis
    logger.info("app.startup", env=settings.ENVIRONMENT, version=settings.APP_VERSION)

    # Validate CORS in production (same fail-fast philosophy as API key validation)
    if settings.is_production and "*" in settings.ALLOWED_HOSTS:
        raise RuntimeError(
            'ALLOWED_HOSTS=["*"] is not permitted in production. '
            "Set ALLOWED_HOSTS to your actual domain(s)."
        )

    # 1. Register plugins
    register_all_plugins()
    logger.info(
        "app.plugins.loaded",
        count=len(__import__("app.core.plugin_registry", fromlist=["registry"]).registry),
    )

    # 2. Connect Qdrant
    vs = get_vector_store()
    await vs.startup()

    # 3. Connect Redis
    _redis = aioredis.from_url(
        str(settings.REDIS_URL),
        encoding="utf-8",
        decode_responses=False,
        max_connections=20,
    )
    await _redis.ping()  # type: ignore[misc]
    logger.info("app.redis.connected")

    # 4. Wire budget guard into LLM service
    from app.services.rag.llm_service import get_llm_service

    get_llm_service().set_budget_checker(_budget_checker)
    logger.info("app.budget_guard.wired", limit_usd=settings.LLM_DAILY_BUDGET_USD)

    # 5. Wire query audit log into RAG pipeline
    from app.api.v1.endpoints.query import get_pipeline

    get_pipeline().set_query_log_callback(_query_log_callback)
    logger.info("app.query_audit.wired")

    yield

    # Graceful shutdown
    logger.info("app.shutdown")
    await vs.shutdown()
    if _redis:
        await _redis.aclose()


# ── App factory ───────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Enterprise AI RAG Platform — 4 use cases on one plugin platform",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        default_response_class=JSONResponse,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RateLimiterMiddleware, redis_client=None)

    @app.middleware("http")
    async def request_middleware(request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=elapsed_ms,
            request_id=request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    prefix = settings.API_PREFIX
    app.include_router(health.router)
    app.include_router(auth.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)
    app.include_router(ingestion.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)

    # ── Slack integration (opt-in via SLACK_ENABLED=true) ─────────
    if settings.SLACK_ENABLED:
        from app.api.v1.endpoints import slack as slack_endpoint  # noqa: PLC0415
        app.include_router(slack_endpoint.router, prefix=prefix)
        logger.info(
            "app.slack.enabled",
            bot_token_set=bool(settings.SLACK_BOT_TOKEN),
            signing_secret_set=bool(settings.SLACK_SIGNING_SECRET),
        )

    return app


app = create_app()
