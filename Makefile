.PHONY: help \
        setup install env \
        infra infra-down infra-status \
        up up-obs down down-v restart \
        dev logs logs-all \
        lint format typecheck \
        test test-unit test-integration test-e2e test-cov \
        db-init build clean clean-venv \
        token smoke upload query

# ── Config ────────────────────────────────────────────────────────
PYTHON  := python3.12
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
UVICORN := $(VENV)/bin/uvicorn

CYAN  := \033[0;36m
GREEN := \033[0;32m
YELLOW:= \033[0;33m
RESET := \033[0m

# ── Help ──────────────────────────────────────────────────────────
help: ## Show available commands
	@echo ""
	@echo "  RAG Enterprise Platform — available commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────
setup: ## One-time setup: create venv, install deps, copy .env
	@echo "$(CYAN)Creating virtual environment (Python 3.12)...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	$(PIP) install ".[dev]"
	@echo "$(CYAN)Configuring environment...$(RESET)"
	@[ -f .env ] \
		&& echo "  $(YELLOW).env already exists — skipping$(RESET)" \
		|| (cp .env.example .env && echo "  Created .env — fill in required values")
	@echo ""
	@echo "$(GREEN)Setup complete.$(RESET)"
	@echo ""
	@echo "  Next steps:"
	@echo "    1. Edit $(CYAN).env$(RESET) — set SECRET_KEY and ANTHROPIC_API_KEY (or OPENAI_API_KEY)"
	@echo "    2. $(CYAN)make infra$(RESET)  — start Qdrant, Redis, Postgres"
	@echo "    3. $(CYAN)make dev$(RESET)    — start API with hot-reload"
	@echo ""

install: ## Reinstall/upgrade all dependencies into the existing venv
	@[ -f $(PIP) ] || (echo "$(YELLOW)Run 'make setup' first$(RESET)" && exit 1)
	$(PIP) install --upgrade ".[dev]"

env: ## Create .env from template (safe — skips if .env exists)
	@[ -f .env ] \
		&& echo "$(YELLOW).env already exists$(RESET)" \
		|| (cp .env.example .env && echo "$(GREEN)Created .env$(RESET) — fill in required values")

# ── Infrastructure ────────────────────────────────────────────────
infra: ## Start infrastructure only: Qdrant, Redis, Postgres (no API)
	docker compose up -d qdrant redis postgres
	@echo ""
	@echo "  $(CYAN)Qdrant$(RESET)    http://localhost:6333/dashboard"
	@echo "  $(CYAN)Redis$(RESET)     localhost:6379"
	@echo "  $(CYAN)Postgres$(RESET)  localhost:5432"
	@echo ""
	@echo "  Run $(CYAN)make dev$(RESET) to start the API locally with hot-reload."
	@echo ""

infra-down: ## Stop infrastructure containers (data preserved)
	docker compose stop qdrant redis postgres

infra-status: ## Show infrastructure container health
	docker compose ps qdrant redis postgres

# ── Full Docker stack ─────────────────────────────────────────────
up: ## Start full stack (infra + API) via Docker
	docker compose up -d
	@echo ""
	@echo "  $(CYAN)API$(RESET)        http://localhost:8000"
	@echo "  $(CYAN)Docs$(RESET)       http://localhost:8000/docs"
	@echo "  $(CYAN)Qdrant$(RESET)     http://localhost:6333/dashboard"
	@echo ""

up-obs: ## Start full stack + Langfuse observability
	docker compose --profile observability up -d
	@echo ""
	@echo "  $(CYAN)API$(RESET)        http://localhost:8000"
	@echo "  $(CYAN)Langfuse$(RESET)   http://localhost:3000"
	@echo ""

down: ## Stop all containers (data preserved)
	docker compose down

down-v: ## Stop all containers and delete volumes (full reset)
	@echo "$(YELLOW)This will delete all Qdrant vectors, Postgres data, and Redis cache.$(RESET)"
	@printf "Continue? [y/N] " && read ans && [ "$${ans}" = "y" ]
	docker compose down -v

restart: down up ## Restart the full Docker stack

logs: ## Tail API container logs
	docker compose logs -f api

logs-all: ## Tail all container logs
	docker compose logs -f

# ── Local development ─────────────────────────────────────────────
dev: ## Start API locally with hot-reload (requires: make infra)
	@[ -f $(UVICORN) ] || (echo "$(YELLOW)Run 'make setup' first$(RESET)" && exit 1)
	@[ -f .env ]       || (echo "$(YELLOW)Run 'make env' first$(RESET)"   && exit 1)
	$(UVICORN) app.main:app \
		--reload \
		--reload-dir app \
		--port 8000 \
		--log-level info

# ── Code quality ──────────────────────────────────────────────────
lint: ## Run ruff linter (auto-fix safe issues)
	$(RUFF) check app tests --fix

format: ## Format code with ruff
	$(RUFF) format app tests

typecheck: ## Run mypy static type checker
	$(MYPY) app --ignore-missing-imports

# ── Tests ─────────────────────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests (no infra required, fast)
	$(PYTEST) tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v --tb=short

test-e2e: ## Run end-to-end workflow tests
	$(PYTEST) tests/e2e/ -v --tb=short

test-cov: ## Run all tests with HTML coverage report
	$(PYTEST) tests/ \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=html:htmlcov
	@echo ""
	@echo "  Coverage report: $(CYAN)htmlcov/index.html$(RESET)"
	@echo ""

# ── Manual API interaction ────────────────────────────────────────
token: ## Fetch a JWT token for admin@company.com
	@curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin@company.com&password=admin123" \
		| python3 -m json.tool

smoke: ## Run smoke test suite against the local API
	$(VENV)/bin/python scripts/smoke_test.py

upload: ## Upload a document  —  usage: make upload FILE=path/to/file.pdf
	@[ -z "$(FILE)" ] && echo "$(YELLOW)Usage: make upload FILE=path/to/file.pdf$(RESET)" && exit 1 || true
	@TOKEN=$$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin@company.com&password=admin123" \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	curl -s -X POST http://localhost:8000/api/v1/ingest/upload \
		-H "Authorization: Bearer $$TOKEN" \
		-F "use_case_id=knowledge_base" \
		-F "file=@$(FILE)" \
		| python3 -m json.tool

query: ## Query the RAG API  —  usage: make query Q="your question"
	@[ -z "$(Q)" ] && echo "$(YELLOW)Usage: make query Q=\"your question\"$(RESET)" && exit 1 || true
	@TOKEN=$$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
		-d "username=admin@company.com&password=admin123" \
		| python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	curl -s -X POST http://localhost:8000/api/v1/query \
		-H "Authorization: Bearer $$TOKEN" \
		-H "Content-Type: application/json" \
		-d "{\"query\": \"$(Q)\"}" \
		| python3 -m json.tool

# ── Database ──────────────────────────────────────────────────────
db-init: ## Apply init.sql schema to the Postgres container
	docker compose exec postgres psql -U rag -d rag_platform \
		-f /docker-entrypoint-initdb.d/init.sql

# ── Build & maintenance ───────────────────────────────────────────
build: ## Build production Docker image
	docker build \
		-f docker/Dockerfile \
		--target production \
		-t rag-platform:latest \
		.

clean: ## Remove Python cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist .coverage

clean-venv: ## Delete the virtual environment (recreate with: make setup)
	rm -rf $(VENV)
