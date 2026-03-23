# RAG Enterprise Platform — Ultimate Pro 2026

Production-ready AI platform với 4 use cases, plugin architecture, full observability.

## Stack
- **Runtime**: Python 3.12 + FastAPI + LangGraph
- **Vector DB**: Qdrant (self-hosted)
- **Cache**: Redis 7
- **LLM Gateway**: LiteLLM
- **Observability**: Langfuse + Prometheus
- **Auth**: JWT + RBAC
- **Deploy**: Docker Compose → K8s ready

## Structure
```
rag-platform/
├── apps/
│   ├── api/          # FastAPI main application
│   ├── worker/       # Background job processor
│   └── agent/        # OpenClaw-compatible agent runner
├── core/
│   ├── config/       # Settings, environment management
│   ├── database/     # Qdrant, Redis clients
│   ├── security/     # JWT, RBAC, guardrails
│   ├── middleware/    # Logging, tracing, rate limiting
│   └── utils/        # Shared utilities
├── plugins/
│   ├── knowledge_base/    # Use case 1
│   ├── customer_support/  # Use case 2
│   ├── document_qa/       # Use case 3
│   └── sales_automation/  # Use case 4
├── services/
│   ├── llm/          # LiteLLM gateway wrapper
│   ├── rag/          # Retrieval pipeline
│   ├── embedding/    # Embedding service
│   └── cache/        # Semantic cache
├── schemas/          # Pydantic models
├── tests/            # Unit, integration, e2e
├── scripts/          # DB init, data seeding
└── deployments/      # Docker, K8s manifests
```

## Quick Start
```bash
cp .env.example .env
# Edit .env with your API keys
docker compose up -d
```

## API Docs
http://localhost:8000/docs
