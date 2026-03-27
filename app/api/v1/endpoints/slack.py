"""
app/api/v1/endpoints/slack.py
──────────────────────────────
Slack event ingestion endpoints.

Routes:
    POST /api/v1/slack/events    — Slack Events API (app_mention, message.im)
    POST /api/v1/slack/commands  — Slash commands (/ask)

Security:
    Mọi request đều verify X-Slack-Signature (HMAC-SHA256) trước khi xử lý.
    Request không có signature hợp lệ → 403, không xử lý gì thêm.

Slack's 3-second rule:
    Slack retry nếu không nhận HTTP 200 trong 3s.
    → Events: ACK ngay (200), xử lý async qua asyncio.create_task
    → Commands: trả về JSON ephemeral "thinking..." ngay, xử lý async
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.services.slack.bot import get_slack_bot

logger = get_logger(__name__)
router = APIRouter(prefix="/slack", tags=["Slack"])


def _verify_or_raise(request: Request, body: bytes) -> None:
    """
    Verify Slack HMAC signature.
    Raise 403 ngay nếu invalid — không để request đi tiếp.
    """
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not timestamp or not signature:
        logger.warning(
            "slack.request.missing_headers",
            has_timestamp=bool(timestamp),
            has_signature=bool(signature),
            ip=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing Slack signature headers",
        )

    bot = get_slack_bot()
    if not bot.verify_slack_signature(body, timestamp, signature):
        logger.warning(
            "slack.signature.invalid",
            ip=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Slack signature",
        )


@router.post("/events", summary="Slack Events API webhook")
async def slack_events(request: Request) -> JSONResponse:
    """
    Nhận Slack events: app_mention, message.im.

    Special case: URL verification challenge (one-time khi setup app).
    Slack gửi challenge JSON, ta phải echo lại đúng field "challenge".

    Các events khác:
    - ACK ngay (200 + {"ok": true})
    - Xử lý async qua asyncio.create_task
    """
    body = await request.body()
    _verify_or_raise(request, body)

    payload: dict[str, Any] = await request.json()

    # ── URL Verification (one-time setup) ──────────────────────────
    if payload.get("type") == "url_verification":
        logger.info("slack.url_verification.ok")
        return JSONResponse({"challenge": payload["challenge"]})

    # ── Event dispatch ─────────────────────────────────────────────
    event_data: dict[str, Any] = payload.get("event", {})
    event_type: str = event_data.get("type", "")

    logger.info(
        "slack.event.received",
        type=event_type,
        team=payload.get("team_id", ""),
    )

    # Bỏ qua messages từ chính bot (tránh vòng lặp vô tận)
    if event_data.get("bot_id") or event_data.get("subtype") == "bot_message":
        return JSONResponse({"ok": True})

    # Bỏ qua message_changed, message_deleted events
    if event_data.get("subtype") in ("message_changed", "message_deleted"):
        return JSONResponse({"ok": True})

    channel_id: str = event_data.get("channel", "")

    if event_type in ("app_mention", "message"):
        bot = get_slack_bot()
        import asyncio
        asyncio.create_task(bot.handle_event(event_data, channel_id))

    # Luôn trả 200 ngay (Slack 3s rule)
    return JSONResponse({"ok": True})


@router.post("/commands", summary="Slack slash commands (/ask)")
async def slack_commands(request: Request) -> JSONResponse:
    """
    Handle slash commands: /ask <question>

    Slack gửi slash commands dưới dạng application/x-www-form-urlencoded.
    Ta parse form, verify signature, xử lý async.

    Response: JSON ephemeral → chỉ người invoke thấy "thinking...".
    Câu trả lời thật sẽ đến sau qua response_url.
    """
    body = await request.body()
    _verify_or_raise(request, body)

    # Parse form data (Slack slash command format)
    form = await request.form()
    payload: dict[str, str] = {k: str(v) for k, v in form.items()}

    logger.info(
        "slack.command",
        command=payload.get("command", ""),
        user=payload.get("user_id", "")[:20],
        channel=payload.get("channel_id", ""),
    )

    bot = get_slack_bot()
    response = await bot.handle_slash_command(payload)

    return JSONResponse(response)
