.DEFAULT_GOAL := help
.PHONY: install install-dev lint format typecheck test test-unit test-integration \
        test-cov load-test security docker-build docker-up docker-down docker-logs \
        docker-dev run train export-onnx migrate migrate-new clean pre-commit help

PYTHON := python
PIP := pip

# ── Installation ─────────────────────────────────────────────────────────────

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

# ── Code Quality ─────────────────────────────────────────────────────────────

lint: ## Run linter (ruff)
	ruff check src/ tests/

format: ## Format code (ruff)
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	mypy src/

# ── Testing ──────────────────────────────────────────────────────────────────

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ \
		--cov=src/threat_id \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		-v

load-test: ## Run load tests with locust
	locust -f tests/load/locustfile.py --headless \
		-u 50 -r 10 --run-time 60s \
		--host http://localhost:8000

# ── Security ─────────────────────────────────────────────────────────────────

security: ## Run security checks (pip-audit + bandit)
	pip-audit
	bandit -r src/ -ll

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image (CPU)
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start all services
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

docker-logs: ## Tail service logs
	docker compose -f docker/docker-compose.yml logs -f

docker-dev: ## Start services with live reload (mount source)
	docker compose -f docker/docker-compose.yml up -d db redis prometheus grafana
	$(PYTHON) -m threat_id

# ── Application ──────────────────────────────────────────────────────────────

run: ## Run the application locally
	$(PYTHON) -m threat_id

train: ## Train YOLOv5 model
	$(PYTHON) scripts/train_yolov5.py --data data/dataset.yaml --epochs 100

export-onnx: ## Export model to ONNX format
	$(PYTHON) scripts/export_onnx.py --weights models/best.pt --output models/best.onnx

# ── Database ─────────────────────────────────────────────────────────────────

migrate: ## Run database migrations
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add users table")
	alembic revision --autogenerate -m "$(MSG)"

# ── Maintenance ──────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info coverage.xml .coverage

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# ── Help ─────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo "Threat Identification System - Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
