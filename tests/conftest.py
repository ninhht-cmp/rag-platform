"""
tests/conftest.py
──────────────────
Shared pytest fixtures.
Patches heavy dependencies at session level so tests stay fast.
"""

from __future__ import annotations

import os
import unittest.mock as mock

import pytest

# ── Force test environment before any app import ──────────────────
os.environ.setdefault("ENVIRONMENT", "local")
os.environ.setdefault("SECRET_KEY", "test_secret_key_at_least_32_chars_long_!")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://rag:rag@localhost:5432/rag_platform",
)


@pytest.fixture(scope="session", autouse=True)
def register_plugins_once() -> None:
    """Register all plugins once per test session. Idempotent."""
    from app.plugins import register_all_plugins

    register_all_plugins()


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch: pytest.MonkeyPatch) -> mock.AsyncMock:
    """Prevent real Redis connections in all tests."""
    with mock.patch("redis.asyncio.from_url") as m:
        redis_mock = mock.AsyncMock()
        redis_mock.ping = mock.AsyncMock(return_value=True)
        redis_mock.get = mock.AsyncMock(return_value=None)
        redis_mock.set = mock.AsyncMock(return_value=True)
        m.return_value = redis_mock
        yield redis_mock


@pytest.fixture(autouse=True)
def mock_qdrant_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent real Qdrant connections during app startup."""
    from app.services.rag import vector_store as vs_module

    with (
        mock.patch.object(vs_module.VectorStore, "startup", mock.AsyncMock()),
        mock.patch.object(vs_module.VectorStore, "shutdown", mock.AsyncMock()),
        mock.patch.object(
            vs_module.VectorStore,
            "health_check",
            mock.AsyncMock(return_value=True),
        ),
    ):
        yield
