# =================================================================
# RAG Platform — Multi-stage Dockerfile
#
# Stages:
#   builder     — compile deps into /install (never ships)
#   production  — minimal runtime image
#   dev         — production + dev/test tooling + hot reload
#
# Build:
#   docker build --target production -t rag-platform:latest .
#   docker build --target dev       -t rag-platform:dev .
#
# Runtime env vars:
#   WEB_CONCURRENCY  — number of uvicorn workers (default: 2)
# =================================================================

# ── Stage 1: dependency builder ──────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile Python packages (gcc for asyncpg,
# libmagic-dev for python-magic, etc.). These stay in the builder
# stage only — never copied to production.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libmagic-dev \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip once, in its own layer — invalidated only on base image change.
RUN pip install --no-cache-dir --upgrade pip==26.0.1 hatchling==1.29.0

# Copy only the dependency manifest first.
# This layer is cached as long as pyproject.toml doesn't change,
# so rebuilds after editing app/ source skip the slow dep-install step.
COPY pyproject.toml ./

# Install ALL runtime dependencies (pinned via pyproject.toml) into
# an isolated prefix that we'll copy cleanly into the production stage.
# --no-deps is NOT used — we want the full dependency resolution to run
# so transitive pins are honoured.
#
# bcrypt is pinned to 4.0.1 here to match pyproject.toml: passlib's
# __version__ check breaks on bcrypt>=4.2 (passlib is unmaintained).
RUN pip install --no-cache-dir --prefix=/install \
        "fastapi==0.135.2" \
        "uvicorn[standard]==0.42.0" \
        "python-multipart==0.0.22" \
        "pydantic==2.12.5" \
        "pydantic-settings==2.13.1" \
        "langchain==1.2.13" \
        "langchain-openai==1.1.12" \
        "langchain-anthropic==1.4.0" \
        "langchain-text-splitters==1.1.1" \
        "langgraph==1.1.3" \
        "langfuse==4.0.1" \
        "qdrant-client==1.17.1" \
        "sentence-transformers==5.3.0" \
        "openai==2.30.0" \
        "jinja2==3.1.6" \
        "numpy==2.4.3" \
        "pypdf==6.9.2" \
        "python-docx==1.2.0" \
        "python-magic==0.4.27" \
        "asyncpg==0.31.0" \
        "sqlalchemy[asyncio]==2.0.48" \
        "alembic==1.18.4" \
        "redis[hiredis]==7.4.0" \
        "httpx==0.28.1" \
        "python-jose[cryptography]==3.5.0" \
        "passlib[bcrypt]==1.7.4" \
        "bcrypt==4.0.1" \
        "structlog==25.5.0" \
        "prometheus-client==0.24.1" \
        "opentelemetry-sdk==1.40.0" \
        "opentelemetry-instrumentation-fastapi==0.61b0" \
        "ragas==0.4.3" \
        "datasets==4.8.4" \
        "tenacity==9.1.4" \
        "orjson==3.11.7" \
        "python-dotenv==1.2.2" \
        "typing-extensions==4.15.0"


# ── Stage 2: production runtime ───────────────────────────────────
FROM python:3.12-slim AS production

WORKDIR /app

# Runtime-only system packages:
#   libmagic1  — required by python-magic (content-type validation)
#   curl       — used by HEALTHCHECK
# Create a non-root user in the same RUN to minimise layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser

# Pull compiled packages from builder — no compiler toolchain needed here.
COPY --from=builder /install /usr/local

# Copy application source last so source edits don't bust the dep-install cache.
COPY app/ ./app/

# Prevent Python from writing .pyc files to disk and enable stdout/stderr
# line buffering — both are best-practice for containerised apps.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Owning user for the process
    HOME=/tmp \
    # Default worker count — override at runtime with -e WEB_CONCURRENCY=N
    WEB_CONCURRENCY=2

USER appuser

EXPOSE 8000

# Probe the /health endpoint (no auth required).
# start-period gives the app time to connect to Qdrant/Redis/Postgres on cold start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Structured logging is handled by structlog (app/core/logging.py).
# --no-access-log suppresses uvicorn's own unstructured access log; the
# app-level request middleware in app/main.py logs every request in JSON.
CMD ["sh", "-c", \
     "uvicorn app.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers ${WEB_CONCURRENCY} \
      --no-access-log \
      --proxy-headers \
      --forwarded-allow-ips '*'"]


# ── Stage 3: dev (hot reload) ─────────────────────────────────────
FROM production AS dev

USER root

# Install the full dev dependency set (mirrors [project.optional-dependencies].dev
# in pyproject.toml) plus watchfiles for --reload support.
RUN pip install --no-cache-dir \
        "pytest==9.0.2" \
        "pytest-asyncio==1.3.0" \
        "pytest-cov==7.1.0" \
        "httpx==0.28.1" \
        "factory-boy==3.3.3" \
        "ruff==0.15.7" \
        "mypy==1.19.1" \
        "pre-commit==4.5.1" \
        "watchfiles==1.1.1"

# Mount the project root at /app in docker-compose for live reload.
# Source is bind-mounted, so we don't need it baked in here.
USER appuser

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/app", \
     "--no-access-log"]
