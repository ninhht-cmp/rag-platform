# RAG Enterprise Platform

Production-ready Retrieval-Augmented Generation platform with a plugin architecture, multi-tenant RBAC, hybrid search, and a LangGraph agent layer. Designed to be deployed in a single afternoon and extended with new use cases in minutes.

## Features

- **Plugin system** — each use case is a self-contained config block; routing, RBAC, retrieval tuning, and eval thresholds are all declared in one place
- **Hybrid search** — BM25 + semantic search with configurable weights per plugin, cross-encoder reranking
- **LangGraph agents** — multi-step tool use with human-in-the-loop gates for irreversible actions
- **Semantic cache** — Redis-backed response cache keyed on query + RBAC context
- **Token budget enforcement** — per-use-case monthly token limits, daily budget guard
- **Evaluation** — RAGAS faithfulness / relevancy / context recall on a schedule, with per-plugin pass/fail thresholds
- **Observability** — structured JSON logs (structlog), Langfuse LLM tracing, Prometheus metrics, OpenTelemetry
- **Auth** — JWT with refresh tokens, role-based access control at retrieval level (not just API level)

## Use cases (included)

| ID | Name | Roles | Agent tools |
|---|---|---|---|
| `knowledge_base` | Internal Knowledge Base | all | — |
| `customer_support` | Customer Support Bot | public | ticket, order lookup, account status |
| `document_qa` | Document Q&A | admin, analyst, user | — |
| `sales_automation` | Sales Automation | admin, sales_rep | CRM, web search, email draft |

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115 + Python 3.12 |
| LLM | Claude Sonnet (primary) · Claude Haiku (secondary) · GPT-4o (fallback) |
| Embedding | `BAAI/bge-m3` (1024-dim, multilingual) |
| Vector DB | Qdrant 1.13 (hybrid search, gRPC) |
| Agent | LangGraph 0.2 |
| Cache | Redis 7.4 (semantic cache + session memory) |
| Database | PostgreSQL 16 + SQLAlchemy 2 async |
| Evaluation | RAGAS 0.1 |
| Observability | Langfuse · Prometheus · OpenTelemetry · structlog |
| Auth | python-jose JWT · passlib bcrypt |
| Container | Docker multi-stage build (builder → production → dev) |

---

## Quick start

### Option A — Local dev (recommended)

Run infra in Docker, API on your machine with hot-reload.

```bash
# 1. Create venv and install all dependencies
make setup

# 2. Fill in required env vars
#    Minimum: SECRET_KEY (≥32 chars) + ANTHROPIC_API_KEY or OPENAI_API_KEY
nano .env

# 3. Start Qdrant, Redis, Postgres
make infra

# 4. Start API with hot-reload
make dev
# → http://localhost:8000/docs
```

### Option B — Full Docker stack

Everything in containers. No Python install needed.

```bash
cp .env.example .env
nano .env          # set SECRET_KEY + API key
make up
# → http://localhost:8000/docs
# → http://localhost:6333/dashboard  (Qdrant)
```

### With Langfuse observability

```bash
make up-obs
# → http://localhost:3000  (Langfuse)
```

---

## Environment variables

Copy `.env.example` to `.env` and set the following.

### Required

```env
SECRET_KEY=<random string, at least 32 characters>
ANTHROPIC_API_KEY=sk-ant-...    # or OPENAI_API_KEY if using OpenAI
```

### LLM

```env
LLM_PROVIDER=anthropic              # anthropic | openai
LLM_MODEL_PRIMARY=claude-sonnet-4-6
LLM_MODEL_SECONDARY=claude-haiku-4-5-20251001
LLM_TEMPERATURE=0.1
LLM_DAILY_BUDGET_USD=50.0
```

### Infrastructure (defaults work with make infra)

```env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag_platform
```

### Optional

```env
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=

FEATURE_SEMANTIC_CACHE=true
FEATURE_AGENT_TOOLS=true
FEATURE_STREAMING=true

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu             # set to "cuda" for GPU acceleration
```

---

## Project structure

```
rag-platform/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/           # query, ingest, auth, analytics, health
│   │   └── middleware/          # JWT auth, rate limiter
│   ├── core/
│   │   ├── config.py            # pydantic-settings, all env vars
│   │   ├── plugin_registry.py   # UseCasePlugin dataclass + registry
│   │   └── logging.py           # structlog setup
│   ├── models/
│   │   └── domain.py            # Pydantic domain models
│   ├── plugins/
│   │   └── __init__.py          # register_all_plugins() — add use cases here
│   ├── prompts/                 # Jinja2 system prompt templates
│   ├── repositories/            # SQLAlchemy async repositories
│   └── services/
│       ├── agent/               # LangGraph agent + tools + session memory
│       ├── evaluation/          # RAGAS evaluation service
│       ├── ingestion/           # Document → chunk → embed → upsert pipeline
│       └── rag/                 # Embedding, vector store, LLM, RAG pipeline
├── tests/
│   ├── unit/                    # Pure unit tests (no infra)
│   ├── integration/             # API endpoint tests (TestClient)
│   └── e2e/                     # Full workflow tests
├── infrastructure/
│   ├── postgres/init.sql
│   └── k8s/                     # Kubernetes manifests
├── docker/Dockerfile            # Multi-stage: builder → production → dev
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Development workflow

```bash
make lint        # ruff check + auto-fix
make format      # ruff format
make typecheck   # mypy strict

make test-unit         # fast, no infra needed
make test-integration  # requires: make infra
make test-e2e          # requires: make infra + make dev
make test-cov          # all tests + HTML coverage report
```

---

## Adding a use case

1. Open `app/plugins/__init__.py`
2. Add a `UseCasePlugin` block inside `register_all_plugins()`
3. Create a Jinja2 system prompt at `app/prompts/<id>_system.j2`
4. (Optional) Add a contract spec at `contracts/<id>.yaml`

The platform auto-wires routing, RBAC, evaluation, and token budgeting. No other files need to change.

```python
registry.register(UseCasePlugin(
    id="hr_chatbot",
    name="HR Chatbot",
    description="Employee HR queries",
    collection_name="uc_hr_chatbot",
    status=PluginStatus.PRODUCTION,
    intent_patterns=[r"leave|salary|benefits|payroll"],
    system_prompt_path="hr_chatbot_system.j2",
    retrieval=RetrievalConfig(top_k=5, hybrid_search=True),
    rbac=RBACRule(allowed_roles=["admin", "user"]),
    eval_thresholds=EvalThresholds(faithfulness=0.85),
    monthly_token_budget=3_000_000,
    citation_required=True,
))
```

---

## API reference

Interactive docs are available at `/docs` (Swagger) and `/redoc` when the API is running.

### Key endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/token` | Obtain JWT access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh an expired access token |
| `POST` | `/api/v1/query` | Submit a RAG query |
| `GET` | `/api/v1/query/stream` | Streaming SSE response |
| `POST` | `/api/v1/ingest/upload` | Upload and index a document |
| `DELETE` | `/api/v1/ingest/{document_id}` | Delete a document and its vectors |
| `POST` | `/api/v1/admin/eval/{use_case_id}` | Trigger RAGAS evaluation |
| `GET` | `/admin/plugins` | List registered plugins (admin only) |
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Readiness check (all deps) |

### Query a document

```bash
# Get a token
make token

# Upload a document
make upload FILE=docs/hr_policy.pdf

# Query it
make query Q="How many vacation days do I get?"
```

Or with curl directly:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin@company.com&password=admin123" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "use_case_id": "customer_support"}'
```

---

## Roles

| Role | Plugins accessible |
|---|---|
| `admin` | All |
| `user` | knowledge_base, document_qa |
| `analyst` | knowledge_base, document_qa |
| `sales_rep` | knowledge_base, sales_automation |
| `support_agent` | knowledge_base, customer_support |
| `*` (public) | customer_support |

RBAC is enforced at the vector retrieval level — a user with the wrong role receives no results, not a 403 error.

---

## Deployment

### Docker (single server)

```bash
make build
# Edit .env for production values
make up
```

Remove the dev volume mount in `docker-compose.yml` before deploying to production:

```yaml
# api service — remove this line:
- ./app:/app/app  # DEV ONLY
```

### Kubernetes

Manifests are in `infrastructure/k8s/`. Set secrets via your cluster's secret manager and apply:

```bash
kubectl apply -f infrastructure/k8s/
```

---

## License

MIT
