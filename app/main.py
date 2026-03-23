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
from collections.abc import AsyncIterator
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


# ── Redis pool (shared across requests) ───────────────────────────
_redis: aioredis.Redis | None = None  # type: ignore[type-arg]


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _redis
    logger.info("app.startup", env=settings.ENVIRONMENT, version=settings.APP_VERSION)

    # 1. Register all use-case plugins
    register_all_plugins()
    logger.info("app.plugins.loaded", count=len(
        __import__("app.core.plugin_registry", fromlist=["registry"]).registry
    ))

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
    await _redis.ping()
    logger.info("app.redis.connected")

    yield  # ← application runs here

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

    # ── CORS ──────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiter ──────────────────────────────────────────────
    # redis_client=None → middleware fetches redis lazily via get_redis()
    # per request, after startup. Fails open if Redis not yet ready.
    app.add_middleware(RateLimiterMiddleware, redis_client=None)

    # ── Request ID + Timing middleware ────────────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
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

    # ── Global exception handler ──────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routers ───────────────────────────────────────────────────
    prefix = settings.API_PREFIX
    app.include_router(health.router)
    app.include_router(auth.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)
    app.include_router(ingestion.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)

    return app


app = create_app()