"""
app/services/agent/session_service.py
──────────────────────────────────────
Conversation session management via Redis.

Stores per-session message history so agents maintain context
across multiple turns. TTL auto-expires idle sessions.

Design:
- Key: session:{session_id}
- Value: JSON list of {role, content, timestamp} dicts
- TTL: 30 minutes (reset on each interaction)
- Max messages: 20 (sliding window — drop oldest)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_MESSAGES = 20


class SessionService:
    def __init__(self, redis_client: aioredis.Redis) -> None:  # type: ignore[type-arg]
        self._r = redis_client

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return conversation history for session. Empty list if not found."""
        raw = await self._r.get(self._key(session_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("session.corrupt", session_id=session_id)
            return []

    async def append(
        self,
        session_id: str,
        role: str,        # "user" | "assistant"
        content: str,
    ) -> None:
        """Append a message and reset TTL. Enforces sliding window."""
        history = await self.get_history(session_id)
        history.append({
            "role": role,
            "content": content,
            "ts": datetime.utcnow().isoformat(),
        })

        # Sliding window — drop oldest if over limit
        if len(history) > MAX_MESSAGES:
            history = history[-MAX_MESSAGES:]

        await self._r.set(
            self._key(session_id),
            json.dumps(history),
            ex=settings.REDIS_SESSION_TTL,
        )

    async def clear(self, session_id: str) -> None:
        """Delete session."""
        await self._r.delete(self._key(session_id))
        logger.info("session.cleared", session_id=session_id)

    async def format_for_prompt(self, session_id: str) -> str:
        """Format history as plain text for injection into system prompt."""
        history = await self.get_history(session_id)
        if not history:
            return ""
        lines: list[str] = ["Previous conversation:"]
        for msg in history[-6:]:  # last 6 messages for context window efficiency
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:500]}")  # truncate long messages
        return "\n".join(lines)

    async def session_exists(self, session_id: str) -> bool:
        return bool(await self._r.exists(self._key(session_id)))
