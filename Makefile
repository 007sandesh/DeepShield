.PHONY: install dev test lint format run serve docker docker-up clean help

# ─── Variables ──────────────────────────────────────────
PYTHON := python
PIP := pip
PYTEST := pytest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Installation ───────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install with dev dependencies
	$(PIP) install -e ".[dev]"

gpu: ## Install with GPU support
	$(PIP) install -e ".[gpu]"

# ─── Development ────────────────────────────────────────
format: ## Format code with black + isort
	black src/ tests/
	isort src/ tests/

lint: ## Lint with ruff + mypy
	ruff check src/ tests/
	mypy src/

test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only
	$(PYTEST) tests/ -v --tb=short -m "not slow and not integration"

test-integration: ## Run integration tests
	$(PYTEST) tests/ -v --tb=short -m "integration"

test-fast: ## Run tests (exclude slow)
	$(PYTEST) tests/ -v --tb=short -m "not slow"

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ─── Running ────────────────────────────────────────────
run: ## Run CLI detection on a file
	$(PYTHON) -m src.cli.main detect

serve: ## Start API server
	$(PYTHON) -m src.api.main

serve-dev: ## Start API server in dev mode
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

info: ## Show system info
	$(PYTHON) -m src.cli.main info

# ─── Docker ─────────────────────────────────────────────
docker: ## Build Docker image
	docker build -t deepshield:latest -f docker/Dockerfile .

docker-up: ## Start all services with docker-compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View service logs
	docker-compose logs -f api

# ─── Cleanup ────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Remove everything including venv
	rm -rf venv/ .venv/ env/
