# Slack Bot Setup Guide

## Kiến trúc

```
Team channel: #rag-bot  (1 channel duy nhất cho cả team)

User A  →  /ask câu hỏi A  →  ✅ Chỉ A thấy câu trả lời
User B  →  /ask câu hỏi B  →  ✅ Chỉ B thấy câu trả lời
User C  →  @RAG Bot ...    →  ✅ Chỉ C thấy câu trả lời
```

**Privacy đảm bảo bởi Slack API** (`chat.postEphemeral`), không phải chỉ code convention.

Session isolation trong Redis:
```
"slack:UA12345"  ← conversation history của User A
"slack:UB67890"  ← conversation history của User B
TTL: 30 phút (reset mỗi lần tương tác)
```

---

## Bước 1 — Tạo Slack App

1. Truy cập https://api.slack.com/apps
2. Nhấn **Create New App** → **From scratch**
3. Đặt tên app (ví dụ: `RAG Bot`) và chọn workspace của team
4. Nhấn **Create App**

---

## Bước 2 — Cấu hình Bot Permissions

**Settings → OAuth & Permissions → Scopes → Bot Token Scopes**

Thêm các scope sau:

| Scope | Mục đích |
|-------|----------|
| `chat:write` | Gửi ephemeral message |
| `chat:write.public` | Gửi vào channel chưa join |
| `commands` | Nhận slash command `/ask` |
| `app_mentions:read` | Nhận khi bị @mention |
| `channels:history` | Đọc events trong channel |
| `im:history` | Nhận Direct Messages (tuỳ chọn) |

---

## Bước 3 — Enable Event Subscriptions

**Settings → Event Subscriptions → Enable Events: ON**

**Request URL:**
```
https://your-domain.com/api/v1/slack/events
```

> 💡 Server phải đang chạy và accessible. Slack gửi challenge để verify URL.

**Subscribe to bot events** (nhấn "Add Bot User Event"):
- `app_mention` — khi user @RAG Bot trong channel
- `message.im` — nếu muốn support DM (tuỳ chọn)

Nhấn **Save Changes**.

---

## Bước 4 — Tạo Slash Command `/ask`

**Settings → Slash Commands → Create New Command**

| Field | Giá trị |
|-------|---------|
| Command | `/ask` |
| Request URL | `https://your-domain.com/api/v1/slack/commands` |
| Short Description | `Hỏi RAG Bot bất cứ điều gì` |
| Usage Hint | `[câu hỏi của bạn]` |

Nhấn **Save**.

---

## Bước 5 — Install App & Lấy Credentials

**Settings → Install App → Install to Workspace**

Sau khi install:

1. Copy **Bot User OAuth Token** (`xoxb-...`)
   → `SLACK_BOT_TOKEN` trong `.env`

2. Vào **Settings → Basic Information → Signing Secret** → **Show** → Copy
   → `SLACK_SIGNING_SECRET` trong `.env`

---

## Bước 6 — Cập nhật `.env`

```bash
SLACK_ENABLED=true
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
SLACK_SIGNING_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Restart server:
```bash
make dev
# hoặc
docker compose restart app
```

---

## Bước 7 — Tạo Channel và Thêm Members

```
# Trong Slack workspace:
/create #rag-bot

# Invite bot vào channel:
/invite @RAG Bot

# Invite team members:
/invite @alice @bob @charlie
```

---

## Cách Sử Dụng

```
# Slash command (khuyến nghị)
/ask Hướng dẫn deploy lên production là gì?

# Mention bot
@RAG Bot Làm thế nào để reset password?
```

Chỉ người hỏi thấy câu trả lời. Không ai trong channel đọc được query/answer của người khác.

---

## Local Development (không có public URL)

Dùng [ngrok](https://ngrok.com) để tunnel localhost:

```bash
# Terminal 1: chạy server
make dev

# Terminal 2: tunnel
ngrok http 8000
# → https://abc123.ngrok-free.app

# Cập nhật trong Slack App:
# Event Subscriptions URL:  https://abc123.ngrok-free.app/api/v1/slack/events
# Slash Command URL:        https://abc123.ngrok-free.app/api/v1/slack/commands
```

> ⚠️ URL ngrok thay đổi mỗi lần restart (free tier). Dùng ngrok paid hoặc subdomain cố định.

---

## Files đã thêm vào project

```
app/
├── core/
│   └── config.py              ← Thêm SLACK_ENABLED, SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET
├── main.py                    ← Register slack router khi SLACK_ENABLED=true
├── api/v1/endpoints/
│   └── slack.py               ← POST /slack/events, POST /slack/commands
└── services/slack/
    ├── __init__.py
    └── bot.py                 ← SlackBotService (verify, handle, respond)

.env.example                   ← Thêm Slack config block
SLACK_SETUP.md                 ← File này
```

---

## Troubleshooting

**"dispatch_failed" từ Slack:**
→ Server không respond trong 3s. Kiểm tra logs và network latency.

**"invalid_auth":**
→ `SLACK_BOT_TOKEN` sai hoặc hết hạn. Reinstall app trong workspace.

**"channel_not_found":**
→ Bot chưa được invite vào channel. Dùng `/invite @RAG Bot`.

**Signature verification failed:**
→ `SLACK_SIGNING_SECRET` sai. Copy lại từ **Basic Information**.

**Bot không respond sau /ask:**
→ Kiểm tra `SLACK_ENABLED=true` trong `.env` và restart server.
→ Xem logs: `docker compose logs app -f`

---

## Roadmap: Plans / Jira Integration

Khi sẵn sàng tích hợp task management, extend trong `bot.py`:

```python
# Thêm commands mới trong handle_slash_command()
if command == "/task":
    return await self.handle_task_command(payload)

# Implement:
async def handle_task_command(self, payload):
    action = payload.get("text", "").split()[0]  # create, list, done, sprint
    ...
```

Architecture đã sẵn sàng mở rộng — không cần refactor core.
