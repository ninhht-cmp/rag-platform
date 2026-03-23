.PHONY: help setup infra infra-down dev \
        up up-obs down down-v restart logs logs-all \
        install env lint format typecheck \
        test test-unit test-integration test-cov \
        db-init clean build token smoke upload query

# ── Colors ────────────────────────────────────────────────────────
CYAN  := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

PYTHON := python3.12
VENV   := .venv
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
UV     := $(VENV)/bin/uvicorn

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

# ── Local dev setup (recommended) ────────────────────────────────
setup: ## Full local setup: create venv + install deps + copy .env
	@echo "$(CYAN)Step 1: Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(CYAN)Step 2: Upgrading pip...$(RESET)"
	$(PIP) install --upgrade pip
	@echo "$(CYAN)Step 3: Installing dependencies...$(RESET)"
	$(PIP) install \
		fastapi "uvicorn[standard]" pydantic pydantic-settings \
		langchain langchain-openai langchain-anthropic \
		langchain-community langchain-text-splitters langgraph \
		"qdrant-client==1.12.2" \
		pypdf python-docx openpyxl \
		asyncpg "sqlalchemy[asyncio]" \
		"redis[hiredis]" \
		"python-jose[cryptography]" "passlib[bcrypt]" "bcrypt==4.0.1" \
		structlog tenacity orjson \
		jinja2 python-multipart httpx numpy \
		pytest pytest-asyncio
	@echo "$(CYAN)Step 4: Setting up .env...$(RESET)"
	@[ -f .env ] && echo ".env already exists — skipping" || (cp .env.example .env && echo "Created .env — fill in SECRET_KEY and OPENAI_API_KEY")
	@echo ""
	@echo "$(GREEN)Setup complete!$(RESET)"
	@echo "$(CYAN)Next steps:$(RESET)"
	@echo "  1. Edit .env — fill in SECRET_KEY and OPENAI_API_KEY"
	@echo "  2. make infra   — start Qdrant + Redis + Postgres"
	@echo "  3. make dev     — start API with hot-reload"

infra: ## Start only infrastructure (Qdrant + Redis + Postgres) — no API container
	docker compose up -d qdrant redis postgres
	@echo "$(CYAN)→ Qdrant:    http://localhost:6333/dashboard$(RESET)"
	@echo "$(CYAN)→ Redis:     localhost:6379$(RESET)"
	@echo "$(CYAN)→ Postgres:  localhost:5432$(RESET)"
	@echo ""
	@echo "Run $(CYAN)make dev$(RESET) to start API locally with hot-reload"

infra-down: ## Stop only infrastructure containers
	docker compose stop qdrant redis postgres

infra-status: ## Show infrastructure container status
	docker compose ps qdrant redis postgres

# ── Full Docker stack (alternative to local dev) ──────────────────
up: ## Start full stack including API in Docker
	docker compose up -d
	@echo "$(CYAN)→ API:      http://localhost:8000/docs$(RESET)"
	@echo "$(CYAN)→ Qdrant:   http://localhost:6333/dashboard$(RESET)"

up-obs: ## Start full stack with observability (Langfuse)
	docker compose --profile observability up -d

down: ## Stop all containers
	docker compose down

down-v: ## Stop all containers and DELETE volumes (full reset)
	docker compose down -v

restart: down up ## Restart full Docker stack

logs: ## Tail API container logs
	docker compose logs -f api

logs-all: ## Tail all service logs
	docker compose logs -f

# ── Development ───────────────────────────────────────────────────
dev: ## Start API locally with hot-reload (needs: make infra first)
	@[ -f $(VENV)/bin/uvicorn ] || (echo "Run 'make setup' first" && exit 1)
	@[ -f .env ] || (echo "Run 'make env' first" && exit 1)
	$(UV) app.main:app --reload --reload-dir app --port 8000 --log-level info

env: ## Copy .env.example to .env (run once)
	@[ -f .env ] && echo ".env already exists" || (cp .env.example .env && echo "Created .env — fill in SECRET_KEY and OPENAI_API_KEY")

install: ## Install/update dependencies into existing venv
	@[ -f $(VENV)/bin/pip ] || (echo "Run 'make setup' first" && exit 1)
	$(PIP) install --upgrade \
		fastapi "uvicorn[standard]" pydantic pydantic-settings \
		langchain langchain-openai langchain-anthropic \
		langchain-community langchain-text-splitters langgraph \
		"qdrant-client==1.12.2" \
		pypdf python-docx openpyxl \
		asyncpg "sqlalchemy[asyncio]" \
		"redis[hiredis]" \
		"python-jose[cryptography]" "passlib[bcrypt]" "bcrypt==4.0.1" \
		structlog tenacity orjson \
		jinja2 python-multipart httpx numpy \
		pytest pytest-asyncio

# ── Quality ───────────────────────────────────────────────────────
lint: ## Run ruff linter
	$(VENV)/bin/ruff check app tests --fix

format: ## Format code with ruff
	$(VENV)/bin/ruff format app tests

typecheck: ## Run mypy type checker
	$(VENV)/bin/mypy app --ignore-missing-imports

# ── Tests ─────────────────────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only (fast, no infra needed)
	$(PYTEST) tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=app --cov-report=term-missing --cov-report=html

# ── Manual API testing ────────────────────────────────────────────
token: ## Get a fresh JWT token (admin)
	@curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin@company.com&password=admin123" \
		| python3 -m json.tool

smoke: ## Run smoke test against local API
	$(VENV)/bin/python scripts/smoke_test.py

upload: ## Upload document — usage: make upload FILE=path/to/file.pdf
	@[ -z "$(FILE)" ] && echo "Usage: make upload FILE=path/to/file.pdf" && exit 1 || true
	@TOKEN=$$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin@company.com&password=admin123" \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	curl -s -X POST http://localhost:8000/api/v1/ingest/upload \
		-H "Authorization: Bearer $$TOKEN" \
		-F "use_case_id=knowledge_base" \
		-F "file=@$(FILE)" | python3 -m json.tool

query: ## Query RAG — usage: make query Q="your question"
	@[ -z "$(Q)" ] && echo "Usage: make query Q=\"your question\"" && exit 1 || true
	@TOKEN=$$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin@company.com&password=admin123" \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	curl -s -X POST http://localhost:8000/api/v1/query \
		-H "Authorization: Bearer $$TOKEN" \
		-H "Content-Type: application/json" \
		-d "{\"query\": \"$(Q)\"}" | python3 -m json.tool

# ── Database ──────────────────────────────────────────────────────
db-init: ## Initialize database schema manually
	docker compose exec postgres psql -U rag -d rag_platform \
		-f /docker-entrypoint-initdb.d/init.sql

# ── Maintenance ───────────────────────────────────────────────────
clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

clean-venv: ## Remove virtual environment (rebuild with: make setup)
	rm -rf $(VENV)

build: ## Build Docker image for production
	docker build -f docker/Dockerfile -t rag-platform:latest .
