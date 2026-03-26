"""
tests/integration/test_api.py
──────────────────────────────
Integration tests for API endpoints.
Uses FastAPI TestClient — no real LLM/Qdrant calls.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from app.core.config import settings
from app.models.domain import QueryResponse, QueryStatus


def _make_token(roles: list[str] = None, user_id: str = "test_user") -> str:
    """Generate a valid test JWT."""
    payload = {
        "sub": user_id,
        "roles": roles or ["user"],
        "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
def client() -> TestClient:
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_token()}"}


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_token(['admin'])}"}


# ── Health endpoints ──────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_readiness_returns_200_or_503(self, client: TestClient) -> None:
        resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)


# ── Auth middleware ───────────────────────────────────────────────

class TestAuth:
    def test_query_without_token_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query",
            json={"query": "test query"},
        )
        assert resp.status_code == 401

    def test_query_with_invalid_token_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query",
            json={"query": "test query"},
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401

    def test_admin_endpoint_requires_admin_role(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        resp = client.get("/admin/plugins", headers=auth_headers)
        assert resp.status_code == 403

    def test_admin_endpoint_accessible_with_admin_role(
        self, client: TestClient, admin_headers: dict
    ) -> None:
        resp = client.get("/admin/plugins", headers=admin_headers)
        assert resp.status_code in (200, 500)  # 500 acceptable if DB not running


# ── Query endpoint ────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_query_returns_200_with_valid_token(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_response = QueryResponse(
            query="test",
            answer="Test answer from knowledge base.",
            use_case_id="knowledge_base",
            confidence=0.85,
            status=QueryStatus.COMPLETED,
        )

        with patch(
            "app.api.v1.endpoints.query.RAGPipeline.query",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            resp = client.post(
                "/api/v1/query",
                json={"query": "What is the vacation policy?"},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer from knowledge base."
        assert data["confidence"] == 0.85
        assert data["escalated"] is False

    def test_query_empty_string_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        resp = client.post(
            "/api/v1/query",
            json={"query": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_query_too_long_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        resp = client.post(
            "/api/v1/query",
            json={"query": "x" * 2001},  # exceeds max_length=2000
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_query_with_explicit_use_case_id(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_response = QueryResponse(
            query="test",
            answer="Support answer.",
            use_case_id="customer_support",
            confidence=0.80,
            status=QueryStatus.COMPLETED,
        )

        # Patch both RAGPipeline and AgentService since customer_support has agent_tools
        with patch(
            "app.api.v1.endpoints.query.RAGPipeline.query",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch(
            "app.services.agent.agent_service.AgentService",
        ) as mock_agent_cls:
            mock_agent_cls.return_value.run = AsyncMock(return_value=mock_response)
            resp = client.post(
                "/api/v1/query",
                json={
                    "query": "How do I reset my password?",
                    "use_case_id": "customer_support",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        assert resp.json()["use_case_id"] == "customer_support"


# ── Ingestion endpoint ────────────────────────────────────────────

class TestIngestionEndpoint:
    def test_upload_without_auth_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/ingest/upload",
            data={"use_case_id": "knowledge_base"},
            files={"file": ("test.txt", b"test content", "text/plain")},
        )
        assert resp.status_code == 401

    def test_upload_unsupported_type_returns_415(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        resp = client.post(
            "/api/v1/ingest/upload",
            data={"use_case_id": "knowledge_base"},
            files={"file": ("test.exe", b"binary content", "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 415

    def test_upload_valid_text_file(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        from app.models.domain import DocumentStatus, IngestionResult

        mock_result = IngestionResult(
            document_id="doc_abc123",
            chunks_created=5,
            status=DocumentStatus.INDEXED,
            processing_time_ms=250,
        )

        with patch(
            "app.api.v1.endpoints.ingestion.get_ingestion_service"
        ) as mock_svc:
            mock_svc.return_value.ingest_document = AsyncMock(return_value=mock_result)

            resp = client.post(
                "/api/v1/ingest/upload",
                data={"use_case_id": "knowledge_base"},
                files={"file": (
                    "policy.txt",
                    b"Company vacation policy: 15 days per year.",
                    "text/plain",
                )},
                headers=auth_headers,
            )

        assert resp.status_code == 202
        data = resp.json()
        assert data["chunks_created"] == 5
        assert data["status"] == "indexed"
