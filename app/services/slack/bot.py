"""
app/services/slack/bot.py
──────────────────────────
Slack Bot Service.

Architecture:
- 1 shared channel (#rag-bot) cho cả team
- Mỗi user's query/response là EPHEMERAL — chỉ người gửi thấy
- Support cả slash command (/ask) lẫn @mention message
- Session isolation: session_id = "slack:{slack_user_id}"
  → mỗi user có conversation memory riêng trong Redis
- Không ai đọc được query/answer của người khác

Privacy model:
    chat.postEphemeral → Slack API enforce, không phải convention.
    Dù bug trong code, Slack sẽ không bao giờ show message của A cho B.

Slack 3-second rule:
    Slack retry nếu không nhận 200 trong 3s.
    → ACK ngay lập tức, xử lý async qua asyncio.create_task.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import time
import uuid
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.models.domain import QueryRequest, QueryResponse, Role, User
from app.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


class SlackBotService:
    """
    Handles tất cả Slack interactions.

    Design decisions:
    - Tất cả responses dùng chat.postEphemeral → privacy at API level
    - session_id = "slack:{user_id}" → isolated per-user conversation history
    - Slash command trả về ACK ngay, process async qua response_url
    - @mention process async, send ephemeral sau khi pipeline chạy xong
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._signing_secret = settings.SLACK_SIGNING_SECRET
        self._bot_token = settings.SLACK_BOT_TOKEN
        self._http = httpx.AsyncClient(timeout=30.0)

    # ── Signature Verification ─────────────────────────────────────

    def verify_slack_signature(self, body: bytes, timestamp: str, signature: str) -> bool:
        """
        Verify request đến từ Slack (HMAC-SHA256).
        Ref: https://api.slack.com/authentication/verifying-requests-from-slack

        CRITICAL: Luôn verify — không có cái này bất kỳ ai cũng POST fake events được.
        Replay window: 5 phút (Slack standard).
        """
        try:
            req_time = int(timestamp)
        except (ValueError, TypeError):
            logger.warning("slack.signature.invalid_timestamp", timestamp=timestamp)
            return False

        # Reject requests cũ hơn 5 phút (replay attack)
        if abs(time.time() - req_time) > 60 * 5:
            logger.warning("slack.signature.stale", age_seconds=abs(time.time() - req_time))
            return False

        base = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = "v0=" + hmac.new(
            self._signing_secret.encode(),
            base.encode(),
            hashlib.sha256,
        ).hexdigest()

        # compare_digest để tránh timing attack
        return hmac.compare_digest(expected, signature)

    # ── Slash Command Handler ──────────────────────────────────────

    async def handle_slash_command(self, payload: dict[str, str]) -> dict[str, Any]:
        """
        Handle /ask <query> slash command.

        Flow:
            User: /ask Làm thế nào để reset password?
            → Slack POST → /api/v1/slack/commands
            → ACK ngay với "thinking..." ephemeral (< 3s)
            → asyncio.create_task → pipeline.query()
            → chat.postEphemeral qua response_url → chỉ user thấy

        Returns dict ngay lập tức cho FastAPI response (3s rule).
        """
        user_id: str = payload.get("user_id", "")
        user_name: str = payload.get("user_name", user_id)
        channel_id: str = payload.get("channel_id", "")
        text: str = payload.get("text", "").strip()
        response_url: str = payload.get("response_url", "")

        if not text:
            return {
                "response_type": "ephemeral",
                "text": (
                    "⚠️ *Cách dùng:* `/ask <câu hỏi của bạn>`\n"
                    "Ví dụ: `/ask Hướng dẫn deploy lên production là gì?`"
                ),
            }

        logger.info(
            "slack.slash_command",
            user=user_id,
            channel=channel_id,
            query_len=len(text),
        )

        user = self._build_user(user_id, user_name)

        # Fire-and-forget — không block response
        asyncio.create_task(
            self._process_and_respond(
                query=text,
                user=user,
                channel_id=channel_id,
                response_url=response_url,
                use_response_url=True,
            )
        )

        # Trả về "thinking..." ngay cho user thấy
        preview = text[:120] + ("..." if len(text) > 120 else "")
        return {
            "response_type": "ephemeral",
            "text": f"🤔 *Đang xử lý câu hỏi của bạn...*\n> {preview}",
        }

    # ── Event Handler (@mention / DM) ─────────────────────────────

    async def handle_event(self, event: dict[str, Any], channel_id: str) -> None:
        """
        Handle app_mention (@bot message) và DM.

        Strip bot mention prefix trước khi gửi vào pipeline.
        Respond ephemeral → chỉ người hỏi thấy.
        """
        user_id: str = event.get("user", "")
        text: str = event.get("text", "").strip()

        if not user_id or not text:
            return

        # Strip "<@UBOTID> " prefix
        clean_text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()

        if not clean_text:
            await self._post_ephemeral(
                channel_id=channel_id,
                user_id=user_id,
                text=(
                    "👋 Xin chào! Hỏi tôi bất cứ điều gì:\n"
                    "• Slash command: `/ask <câu hỏi>`\n"
                    "• Mention: `@RAG Bot <câu hỏi>`"
                ),
            )
            return

        logger.info(
            "slack.mention",
            user=user_id,
            channel=channel_id,
            query_len=len(clean_text),
        )

        user = self._build_user(user_id, event.get("username", user_id))

        # Báo "đang xử lý" ngay (ephemeral — chỉ user thấy)
        preview = clean_text[:120] + ("..." if len(clean_text) > 120 else "")
        await self._post_ephemeral(
            channel_id=channel_id,
            user_id=user_id,
            text=f"🤔 *Đang xử lý...*\n> {preview}",
        )

        # Xử lý và respond
        await self._process_and_respond(
            query=clean_text,
            user=user,
            channel_id=channel_id,
            response_url=None,
            use_response_url=False,
        )

    # ── Core: Pipeline + Respond ───────────────────────────────────

    async def _process_and_respond(
        self,
        query: str,
        user: User,
        channel_id: str,
        response_url: str | None,
        use_response_url: bool,
    ) -> None:
        """
        Chạy RAG pipeline và gửi ephemeral response.

        Session isolation:
            session_id = "slack:{slack_user_id}"
            → User A có "slack:UA12345", User B có "slack:UB67890"
            → Redis keys hoàn toàn tách biệt, TTL 30 phút mỗi session
            → Conversation history của A không bao giờ lẫn với B
        """
        # Slack user_id được lưu trong user.name (xem _build_user)
        slack_user_id = user.name
        session_id = f"slack:{slack_user_id}"

        try:
            request = QueryRequest(
                query=query,
                session_id=session_id,
                # use_case_id=None → auto-routed bởi intent trong pipeline
            )

            response: QueryResponse = await self._pipeline.query(request, user)
            answer_text = self._format_response(response)

            if use_response_url and response_url:
                # Slash command path: dùng response_url để replace "thinking..." message
                await self._respond_via_url(response_url, answer_text)
            else:
                # Mention/DM path: post ephemeral mới
                await self._post_ephemeral(
                    channel_id=channel_id,
                    user_id=slack_user_id,
                    text=answer_text,
                )

        except ValueError as exc:
            # Prompt injection hoặc validation error từ pipeline
            logger.warning("slack.query.rejected", user=slack_user_id, error=str(exc))
            err_text = f"⚠️ Câu hỏi không hợp lệ: {exc}"
            if use_response_url and response_url:
                await self._respond_via_url(response_url, err_text)
            else:
                await self._post_ephemeral(channel_id, slack_user_id, err_text)

        except Exception as exc:
            logger.error(
                "slack.query.failed",
                user=slack_user_id,
                session=session_id,
                error=str(exc),
                exc_info=True,
            )
            err_text = "❌ Có lỗi xảy ra. Vui lòng thử lại sau."
            if use_response_url and response_url:
                await self._respond_via_url(response_url, err_text)
            else:
                await self._post_ephemeral(channel_id, slack_user_id, err_text)

    # ── Slack API Calls ────────────────────────────────────────────

    async def _post_ephemeral(self, channel_id: str, user_id: str, text: str) -> None:
        """
        Post ephemeral message — CHỈ user_id thấy, không ai khác.

        Đây là privacy guarantee cốt lõi:
        Slack enforce ở API level — không phải chỉ code convention.
        Kể cả admin workspace cũng không thể đọc ephemeral messages.
        """
        resp = await self._http.post(
            "https://slack.com/api/chat.postEphemeral",
            headers={
                "Authorization": f"Bearer {self._bot_token}",
                "Content-Type": "application/json",
            },
            json={
                "channel": channel_id,
                "user": user_id,
                "text": text,
                "mrkdwn": True,
            },
        )
        data = resp.json()
        if not data.get("ok"):
            logger.error(
                "slack.ephemeral.post_failed",
                user=user_id,
                channel=channel_id,
                slack_error=data.get("error"),
            )

    async def _respond_via_url(self, response_url: str, text: str) -> None:
        """
        Respond qua Slack response_url (cho slash commands sau 3s delay).
        replace_original=True → thay "thinking..." bằng câu trả lời thật.
        response_type=ephemeral → chỉ người dùng thấy.
        """
        try:
            await self._http.post(
                response_url,
                json={
                    "response_type": "ephemeral",
                    "text": text,
                    "replace_original": True,
                    "mrkdwn": True,
                },
            )
        except Exception as exc:
            logger.error("slack.response_url.failed", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────

    def _build_user(self, slack_user_id: str, username: str) -> User:
        """
        Map Slack identity → RAG platform User.

        Quan trọng: user.name = slack_user_id
        → Được dùng để build session_id = "slack:{user.name}"
        → Stable UUID từ slack_user_id cho user.id
        """
        stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"slack:{slack_user_id}")
        return User(
            id=stable_uuid,
            email=f"{slack_user_id}@slack.workspace",
            name=slack_user_id,          # ← Slack user_id, dùng cho session key
            roles=[Role.USER],
            department="Slack",
        )

    def _format_response(self, response: QueryResponse) -> str:
        """
        Format QueryResponse thành Slack markdown.

        Layout:
        - Answer text
        - Citations (tối đa 3)
        - Confidence indicator
        - Escalation warning nếu có
        """
        lines: list[str] = [response.answer]

        # Citations
        if response.citations:
            lines.append("\n📚 *Nguồn tham khảo:*")
            for i, citation in enumerate(response.citations[:3], 1):
                preview = citation.content_preview[:100].replace("\n", " ")
                lines.append(f"  {i}. `{citation.filename}` — {preview}...")

        # Confidence bar
        if response.confidence > 0:
            if response.confidence >= 0.8:
                bar = "🟢"
            elif response.confidence >= 0.5:
                bar = "🟡"
            else:
                bar = "🔴"
            lines.append(f"\n{bar} Độ tin cậy: {response.confidence:.0%}")

        # Latency (debug info)
        if response.latency_ms > 0:
            lines.append(f"⏱ _{response.latency_ms}ms_")

        # Escalation
        if response.escalated:
            lines.append(f"\n⚠️ *Escalated:* {response.escalation_reason}")

        return "\n".join(lines)

    async def close(self) -> None:
        """Cleanup HTTP client khi app shutdown."""
        await self._http.aclose()


# ── Singleton ──────────────────────────────────────────────────────

_bot: SlackBotService | None = None


def get_slack_bot() -> SlackBotService:
    """
    Lazy singleton — initialized sau khi pipeline đã ready.
    Safe to call sau lifespan startup.
    """
    global _bot
    if _bot is None:
        from app.api.v1.endpoints.query import get_pipeline
        _bot = SlackBotService(pipeline=get_pipeline())
    return _bot
