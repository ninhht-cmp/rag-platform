# ── Stage 1: dependency builder ──────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install all deps to /install prefix
RUN pip install --no-cache-dir --prefix=/install \
    click \
    "uvicorn[standard]" \
    fastapi \
    pydantic \
    pydantic-settings \
    langchain \
    langchain-openai \
    langchain-anthropic \
    langchain-community \
    langchain-text-splitters \
    langgraph \
    langfuse \
    qdrant-client \
    sentence-transformers \
    pypdf \
    python-docx \
    openpyxl \
    python-magic \
    asyncpg \
    "sqlalchemy[asyncio]" \
    alembic \
    "redis[hiredis]" \
    httpx \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    structlog \
    prometheus-client \
    tenacity \
    orjson \
    python-dotenv \
    jinja2 \
    python-multipart \
    ragas

# ── Stage 2: production runtime ───────────────────────────────────
FROM python:3.12-slim AS production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

COPY --from=builder /install /usr/local

COPY app/ ./app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-config", "/dev/null"]

# ── Stage 3: dev (hot reload) ─────────────────────────────────────
FROM production AS dev
USER root
RUN pip install --no-cache-dir watchfiles pytest pytest-asyncio httpx ruff mypy
USER appuser
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/app"]
