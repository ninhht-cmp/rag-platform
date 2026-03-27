"""
tests/e2e/test_workflows.py
────────────────────────────
End-to-end workflow tests.
These test complete user journeys from API call to response.
All LLM/Qdrant calls mocked — focus on data flow correctness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from app.core.config import settings
from app.models.domain import (
    DocumentStatus,
    IngestionResult,
    QueryResponse,
    QueryStatus,
)


def make_token(email: str, roles: list[str]) -> str:
    payload = {
        "sub": email,
        "roles": roles,
        "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


# ══════════════════════════════════════════════════════════════════
# Workflow 1: Login → Upload Doc → Query → Get Answer with Citation
# ══════════════════════════════════════════════════════════════════


class TestKnowledgeBaseWorkflow:
    def test_full_kb_workflow(self, client: TestClient) -> None:
        """
        GIVEN: Authenticated user with 'user' role
        WHEN:  Uploads a policy doc, then queries it
        THEN:  Gets answer with citations, no escalation
        """
        # Step 1: Login
        login_resp = client.post(
            "/api/v1/auth/token",
            data={"username": "user@company.com", "password": "user123"},
        )
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Step 2: Upload document
        mock_ingest = IngestionResult(
            document_id="doc_kb_001",
            chunks_created=8,
            status=DocumentStatus.INDEXED,
            processing_time_ms=320,
        )
        with patch("app.api.v1.endpoints.ingestion.get_ingestion_service") as mock_svc:
            mock_svc.return_value.ingest_document = AsyncMock(return_value=mock_ingest)
            upload_resp = client.post(
                "/api/v1/ingest/upload",
                data={"use_case_id": "knowledge_base"},
                files={
                    "file": (
                        "hr_policy.txt",
                        b"Vacation policy: 15 days per year.",
                        "text/plain",
                    )
                },
                headers=headers,
            )
        assert upload_resp.status_code == 202
        assert upload_resp.json()["chunks_created"] == 8

        # Step 3: Query
        mock_answer = QueryResponse(
            query="How many vacation days do I get?",
            answer=(
                "You receive 15 days of annual vacation leave per year. [Source 1 — hr_policy.txt]"
            ),
            use_case_id="knowledge_base",
            confidence=0.91,
            status=QueryStatus.COMPLETED,
            escalated=False,
        )
        with patch(
            "app.api.v1.endpoints.query.RAGPipeline.query",
            new_callable=AsyncMock,
            return_value=mock_answer,
        ):
            query_resp = client.post(
                "/api/v1/query",
                json={"query": "How many vacation days do I get?"},
                headers=headers,
            )

        assert query_resp.status_code == 200
        data = query_resp.json()
        assert "15 days" in data["answer"]
        assert data["escalated"] is False
        assert data["confidence"] > 0.8

    def test_escalation_on_sensitive_query(self, client: TestClient) -> None:
        """
        GIVEN: Query matches escalation pattern (termination)
        WHEN:  User queries about termination
        THEN:  Response is escalated
        """
        token = make_token("user@company.com", ["user"])
        headers = {"Authorization": f"Bearer {token}"}

        mock_escalated = QueryResponse(
            query="I want to file a lawsuit against the company",
            answer="This query has been escalated. Please consult HR or Legal.",
            use_case_id="knowledge_base",
            confidence=0.60,
            status=QueryStatus.ESCALATED,
            escalated=True,
            escalation_reason="Query matches escalation pattern",
        )
        with patch(
            "app.api.v1.endpoints.query.RAGPipeline.query",
            new_callable=AsyncMock,
            return_value=mock_escalated,
        ):
            resp = client.post(
                "/api/v1/query",
                json={"query": "I want to file a lawsuit against the company"},
                headers=headers,
            )

        assert resp.status_code == 200
        assert resp.json()["escalated"] is True
        assert resp.json()["status"] == "escalated"


# ══════════════════════════════════════════════════════════════════
# Workflow 2: Customer Support with Session Memory
# ══════════════════════════════════════════════════════════════════


class TestCustomerSupportWorkflow:
    def test_support_query_with_session(self, client: TestClient) -> None:
        """
        GIVEN: Customer (no auth required for support)
        WHEN:  Asks about password reset with session ID
        THEN:  Gets helpful response with session maintained
        """
        token = make_token("customer@external.com", ["user"])
        headers = {"Authorization": f"Bearer {token}"}

        session_id = "sess_test_001"
        mock_resp = QueryResponse(
            query="How do I reset my password?",
            answer=(
                "To reset your password: 1. Click 'Forgot Password' on the login page."
                " 2. Enter your email. 3. Check your inbox for the reset link."
            ),
            use_case_id="customer_support",
            confidence=0.88,
            status=QueryStatus.COMPLETED,
            escalated=False,
            session_id=session_id,
        )

        with (
            patch(
                "app.api.v1.endpoints.query.RAGPipeline.query",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch(
                "app.services.agent.agent_service.AgentService",
            ) as mock_agent_cls,
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=mock_resp)
            resp = client.post(
                "/api/v1/query",
                json={
                    "query": "How do I reset my password?",
                    "use_case_id": "customer_support",
                    "session_id": session_id,
                },
                headers=headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert "password" in data["answer"].lower()


# ══════════════════════════════════════════════════════════════════
# Workflow 3: RBAC — Sales restricted to sales_rep only
# ══════════════════════════════════════════════════════════════════


class TestRBACWorkflow:
    def test_sales_query_blocked_for_regular_user(self, client: TestClient) -> None:
        """
        GIVEN: User with 'user' role queries sales use case
        WHEN:  Query is explicitly routed to sales_automation
        THEN:  Platform either blocks or falls back (never serves restricted data)
        """
        token = make_token("user@company.com", ["user"])
        headers = {"Authorization": f"Bearer {token}"}

        # Plugin registry enforces roles — sales_automation needs sales_rep
        # The pipeline will route to a different plugin or return no results
        mock_fallback = QueryResponse(
            query="what is our pricing strategy?",
            answer="I don't have information about this in the available documents.",
            use_case_id="knowledge_base",  # fell back to KB, not sales
            confidence=0.0,
            status=QueryStatus.COMPLETED,
        )
        with patch(
            "app.api.v1.endpoints.query.RAGPipeline.query",
            new_callable=AsyncMock,
            return_value=mock_fallback,
        ):
            resp = client.post(
                "/api/v1/query",
                json={"query": "what is our pricing strategy?"},
                headers=headers,
            )

        # Must return 200 (not 403) — RBAC at retrieval level, not API level
        assert resp.status_code == 200
        # But should NOT return sales data
        assert resp.json()["use_case_id"] != "sales_automation"

    def test_sales_query_succeeds_for_sales_rep(self, client: TestClient) -> None:
        """
        GIVEN: User with 'sales_rep' role
        WHEN:  Queries pricing
        THEN:  Gets proper sales response
        """
        token = make_token("sales@company.com", ["sales_rep"])
        headers = {"Authorization": f"Bearer {token}"}

        mock_sales = QueryResponse(
            query="What is our enterprise pricing?",
            answer="Enterprise plan starts at $999/month. Contact sales for custom pricing.",
            use_case_id="sales_automation",
            confidence=0.85,
            status=QueryStatus.COMPLETED,
        )
        with (
            patch(
                "app.api.v1.endpoints.query.RAGPipeline.query",
                new_callable=AsyncMock,
                return_value=mock_sales,
            ),
            patch(
                "app.services.agent.agent_service.AgentService",
            ) as mock_agent_cls,
        ):
            mock_agent_cls.return_value.run = AsyncMock(return_value=mock_sales)
            resp = client.post(
                "/api/v1/query",
                json={
                    "query": "What is our enterprise pricing?",
                    "use_case_id": "sales_automation",
                },
                headers=headers,
            )

        assert resp.status_code == 200
        assert resp.json()["use_case_id"] == "sales_automation"


# ══════════════════════════════════════════════════════════════════
# Workflow 4: Admin — Plugin management and evaluation
# ══════════════════════════════════════════════════════════════════


class TestAdminWorkflow:
    def test_admin_can_list_plugins(self, client: TestClient) -> None:
        token = make_token("admin@company.com", ["admin"])
        resp = client.get(
            "/admin/plugins",
            headers={"Authorization": f"Bearer {token}"},
        )
        # May be 200 or 500 depending on DB, but not 401/403
        assert resp.status_code not in (401, 403)

    def test_non_admin_cannot_access_admin(self, client: TestClient) -> None:
        token = make_token("user@company.com", ["user"])
        resp = client.get(
            "/admin/plugins",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_admin_can_trigger_eval(self, client: TestClient) -> None:
        from app.models.domain import EvalMetrics

        token = make_token("admin@company.com", ["admin"])

        mock_metrics = EvalMetrics(
            use_case_id="knowledge_base",
            faithfulness=0.88,
            answer_relevancy=0.84,
            context_recall=0.79,
            passed=True,
            sample_size=2,
        )
        with patch("app.api.v1.endpoints.analytics.get_eval_service") as mock_svc:
            mock_svc.return_value.evaluate_plugin = AsyncMock(return_value=mock_metrics)
            resp = client.post(
                "/api/v1/admin/eval/knowledge_base",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
        assert data["faithfulness"] == 0.88
