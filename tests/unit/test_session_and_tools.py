"""
tests/unit/test_session_and_tools.py
──────────────────────────────────────
Unit tests for:
- SessionService (Redis-backed conversation memory)
- Agent tools (stubs validation)
- Helpers (text utils, cost estimation)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.services.agent.session_service import MAX_MESSAGES, SessionService
from app.utils.helpers import (
    clean_text,
    estimate_cost_usd,
    estimate_tokens,
    is_valid_email,
    paginate,
    sanitize_filename,
    stable_hash,
    truncate,
)

# ── Session Service tests ─────────────────────────────────────────


class TestSessionService:
    @pytest.fixture
    def redis_mock(self) -> AsyncMock:
        r = AsyncMock()
        r.get = AsyncMock(return_value=None)
        r.set = AsyncMock(return_value=True)
        r.delete = AsyncMock(return_value=1)
        r.exists = AsyncMock(return_value=0)
        return r

    @pytest.fixture
    def session(self, redis_mock: AsyncMock) -> SessionService:
        return SessionService(redis_mock)

    @pytest.mark.asyncio
    async def test_get_history_empty_session(self, session: SessionService) -> None:
        history = await session.get_history("sess_001")
        assert history == []

    @pytest.mark.asyncio
    async def test_append_stores_message(
        self, session: SessionService, redis_mock: AsyncMock
    ) -> None:
        await session.append("sess_001", "user", "Hello there")
        redis_mock.set.assert_called_once()
        # Verify the data stored contains our message
        call_args = redis_mock.set.call_args
        stored_data = json.loads(call_args[0][1])
        assert len(stored_data) == 1
        assert stored_data[0]["role"] == "user"
        assert stored_data[0]["content"] == "Hello there"

    @pytest.mark.asyncio
    async def test_append_respects_sliding_window(
        self, session: SessionService, redis_mock: AsyncMock
    ) -> None:
        # Build existing history at max
        existing = [
            {"role": "user", "content": f"msg {i}", "ts": "2026-01-01T00:00:00"}
            for i in range(MAX_MESSAGES)
        ]
        redis_mock.get = AsyncMock(return_value=json.dumps(existing).encode())

        await session.append("sess_001", "user", "new message")

        stored_data = json.loads(redis_mock.set.call_args[0][1])
        assert len(stored_data) == MAX_MESSAGES  # not MAX_MESSAGES + 1
        assert stored_data[-1]["content"] == "new message"
        assert stored_data[0]["content"] == "msg 1"  # oldest dropped

    @pytest.mark.asyncio
    async def test_clear_deletes_session(
        self, session: SessionService, redis_mock: AsyncMock
    ) -> None:
        await session.clear("sess_001")
        redis_mock.delete.assert_called_once_with("session:sess_001")

    @pytest.mark.asyncio
    async def test_format_for_prompt_empty(self, session: SessionService) -> None:
        result = await session.format_for_prompt("sess_001")
        assert result == ""

    @pytest.mark.asyncio
    async def test_format_for_prompt_with_history(
        self, session: SessionService, redis_mock: AsyncMock
    ) -> None:
        history = [
            {"role": "user", "content": "What is the policy?", "ts": "2026-01-01T00:00:00"},
            {
                "role": "assistant",
                "content": "The policy allows 15 days.",
                "ts": "2026-01-01T00:00:01",
            },
        ]
        redis_mock.get = AsyncMock(return_value=json.dumps(history).encode())
        result = await session.format_for_prompt("sess_001")
        assert "Previous conversation" in result
        assert "What is the policy?" in result
        assert "15 days" in result

    @pytest.mark.asyncio
    async def test_corrupt_session_returns_empty(
        self, session: SessionService, redis_mock: AsyncMock
    ) -> None:
        redis_mock.get = AsyncMock(return_value=b"not valid json{{{")
        history = await session.get_history("sess_corrupt")
        assert history == []

    @pytest.mark.asyncio
    async def test_session_key_format(self, session: SessionService, redis_mock: AsyncMock) -> None:
        await session.append("my_session_123", "user", "test")
        call_args = redis_mock.set.call_args
        key = call_args[0][0]
        assert key == "session:my_session_123"


# ── Agent Tools tests ─────────────────────────────────────────────


class TestAgentTools:
    @pytest.mark.asyncio
    async def test_create_ticket_returns_ticket_id(self) -> None:
        from app.services.agent.tools import create_support_ticket

        result = await create_support_ticket.ainvoke(
            {
                "subject": "Login issue",
                "description": "Cannot log in",
                "priority": "high",
            }
        )
        assert "TKT-" in result
        assert "high" in result.lower()

    @pytest.mark.asyncio
    async def test_lookup_order_returns_status(self) -> None:
        from app.services.agent.tools import lookup_order_status

        result = await lookup_order_status.ainvoke({"order_id": "ORD-12345"})
        assert "ORD-12345" in result
        assert "Status" in result

    @pytest.mark.asyncio
    async def test_draft_email_contains_review_warning(self) -> None:
        from app.services.agent.tools import draft_outreach_email

        result = await draft_outreach_email.ainvoke(
            {
                "prospect_name": "Jane Smith",
                "company_name": "Acme Corp",
                "pain_point": "slow onboarding",
                "product_value_prop": "automated onboarding",
            }
        )
        assert "REQUIRES HUMAN REVIEW" in result
        assert "Jane Smith" in result
        assert "Acme Corp" in result

    @pytest.mark.asyncio
    async def test_crm_activity_log_succeeds(self) -> None:
        from app.services.agent.tools import create_crm_activity

        result = await create_crm_activity.ainvoke(
            {
                "company_name": "TestCo",
                "activity_type": "call",
                "notes": "Discussed pricing",
            }
        )
        assert "ACT-" in result
        assert "call" in result.lower()

    def test_tool_registry_has_all_tools(self) -> None:
        from app.services.agent.tools import TOOL_REGISTRY

        expected = {
            "create_ticket",
            "lookup_order",
            "check_account_status",
            "web_search",
            "crm_lookup",
            "draft_email",
            "create_crm_activity",
        }
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_get_tools_for_plugin_returns_subset(self) -> None:
        from app.services.agent.tools import get_tools_for_plugin

        tools = get_tools_for_plugin(["create_ticket", "lookup_order"])
        assert len(tools) == 2

    def test_get_tools_unknown_id_skipped(self) -> None:
        from app.services.agent.tools import get_tools_for_plugin

        tools = get_tools_for_plugin(["create_ticket", "nonexistent_tool"])
        assert len(tools) == 1  # only the valid one


# ── Helpers tests ─────────────────────────────────────────────────


class TestHelpers:
    def test_truncate_short_text_unchanged(self) -> None:
        assert truncate("hello", 100) == "hello"

    def test_truncate_long_text(self) -> None:
        result = truncate("a" * 300, 200)
        assert len(result) == 200
        assert result.endswith("...")

    def test_clean_text_collapses_whitespace(self) -> None:
        text = "hello   world\n\n\n\nfoo"
        result = clean_text(text)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_clean_text_removes_null_bytes(self) -> None:
        text = "hello\x00world"
        result = clean_text(text)
        assert "\x00" not in result
        assert "helloworld" in result

    def test_estimate_tokens_reasonable(self) -> None:
        # ~100 words → ~100 tokens (rough estimate)
        text = "word " * 100
        tokens = estimate_tokens(text)
        assert 80 < tokens < 200

    def test_stable_hash_deterministic(self) -> None:
        h1 = stable_hash("hello world")
        h2 = stable_hash("hello world")
        h3 = stable_hash("different text")
        assert h1 == h2
        assert h1 != h3

    def test_stable_hash_length(self) -> None:
        assert len(stable_hash("test", length=8)) == 8
        assert len(stable_hash("test", length=32)) == 32

    def test_is_valid_email(self) -> None:
        assert is_valid_email("user@example.com") is True
        assert is_valid_email("user+tag@sub.domain.co") is True
        assert is_valid_email("not-an-email") is False
        assert is_valid_email("@nodomain.com") is False
        assert is_valid_email("") is False

    def test_sanitize_filename_removes_dangerous_chars(self) -> None:
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result and "/" not in result and "passwd" in result

    def test_sanitize_filename_no_directory_traversal(self) -> None:
        result = sanitize_filename("../../secret.txt")
        assert "../" not in result
        assert ".." not in result

    def test_estimate_cost_usd(self) -> None:
        # 1M input + 500K output tokens on Sonnet
        cost = estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 500_000)
        expected = 3.0 + (15.0 * 0.5)  # = 10.5 USD
        assert abs(cost - expected) < 0.01

    def test_estimate_cost_unknown_model_uses_default(self) -> None:
        cost = estimate_cost_usd("unknown-model", 100_000, 10_000)
        assert cost > 0

    def test_paginate_first_page(self) -> None:
        items = list(range(50))
        result = paginate(items, page=1, page_size=10)
        assert result["items"] == list(range(10))
        assert result["total"] == 50
        assert result["total_pages"] == 5
        assert result["has_next"] is True
        assert result["has_prev"] is False

    def test_paginate_last_page(self) -> None:
        items = list(range(25))
        result = paginate(items, page=3, page_size=10)
        assert result["items"] == [20, 21, 22, 23, 24]
        assert result["has_next"] is False
        assert result["has_prev"] is True

    def test_paginate_empty_list(self) -> None:
        result = paginate([], page=1, page_size=20)
        assert result["items"] == []
        assert result["total"] == 0
        assert result["total_pages"] == 1
