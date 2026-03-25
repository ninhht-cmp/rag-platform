"""
app/api/v1/middleware/rate_limiter.py
──────────────────────────────────────
Sliding-window rate limiter backed by Redis.

Limits:
- Anonymous:      20 req/min  (health checks etc.)
- Authenticated:  60 req/min  per user
- Admin:         300 req/min  per user
"""
from __future__ import annotations

import time
from functools import lru_cache

import redis.asyncio as aioredis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)

_LIMITS: dict[str, tuple[int, int]] = {
    "admin":  (300, 60),
    "user":   (60, 60),
    "anon":   (20, 60),
}


@lru_cache(maxsize=128)
def _decode_token_cached(token: str) -> tuple[str, str] | None:
    """
    FIX: Cache JWT decode so rate limiter doesn't re-verify the full HMAC
    on every request for the same token. LRU evicts old tokens automatically.
    Returns (user_id, tier) or None.
    """
    try:
        from jose import jwt
        from app.core.config import settings
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id = payload.get("sub", "unknown")
        roles = payload.get("roles", [])
        tier = "admin" if "admin" in roles else "user"
        return user_id, tier
    except Exception:
        return None


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, redis_client: aioredis.Redis | None = None) -> None:  # type: ignore[type-arg]
        super().__init__(app)  # type: ignore[arg-type]
        self._r = redis_client

    def _get_redis(self) -> aioredis.Redis:  # type: ignore[type-arg]
        if self._r is not None:
            return self._r
        try:
            from app.main import get_redis
            return get_redis()
        except Exception:
            raise RuntimeError("Redis not available")

    async def dispatch(self, request: Request, call_next: object) -> Response:
        if request.url.path in ("/health", "/health/ready"):
            return await call_next(request)  # type: ignore[misc]

        identity, tier = self._get_identity(request)
        max_reqs, window_secs = _LIMITS[tier]

        try:
            _ = self._get_redis()
            allowed, remaining = await self._check(identity, tier, max_reqs, window_secs)
        except RuntimeError:
            return await call_next(request)  # type: ignore[misc]
        except Exception as exc:
            logger.warning("rate_limiter.redis_error", error=str(exc))
            return await call_next(request)  # type: ignore[misc]

        if not allowed:
            logger.warning(
                "rate_limiter.exceeded",
                identity=identity[:20],
                tier=tier,
                path=request.url.path,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after_seconds": window_secs,
                },
            )
            response.headers["Retry-After"] = str(window_secs)
            return response

        response = await call_next(request)  # type: ignore[misc]
        response.headers["X-RateLimit-Limit"] = str(max_reqs)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(window_secs)
        return response

    def _get_identity(self, request: Request) -> tuple[str, str]:
        """
        Extract user identity and determine rate limit tier.
        """
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            result = _decode_token_cached(auth[7:])
            if result is not None:
                user_id, tier = result
                return f"user:{user_id}", tier

        # Fall back to direct TCP peer IP — do NOT trust X-Forwarded-For for rate limiting
        peer_ip = request.client.host if request.client else "unknown"
        return f"ip:{peer_ip}", "anon"

    async def _check(
        self,
        identity: str,
        tier: str,
        max_reqs: int,
        window_secs: int,
    ) -> tuple[bool, int]:
        key = f"rl:{tier}:{identity}"
        now = time.time()
        window_start = now - window_secs

        pipe = self._get_redis().pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window_secs + 1)
        results = await pipe.execute()

        count = int(results[1])
        allowed = count < max_reqs
        remaining = max(0, max_reqs - count - 1)
        return allowed, remaining
