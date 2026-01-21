# NL2API Makefile
# Common commands for development workflow

.PHONY: help install dev db-up db-down db-reset db-logs load-tests test test-unit lint format clean

# Default target
help:
	@echo "NL2API Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install development dependencies"
	@echo ""
	@echo "Database:"
	@echo "  make db-up        Start PostgreSQL container"
	@echo "  make db-down      Stop PostgreSQL container"
	@echo "  make db-reset     Reset database (destroy and recreate)"
	@echo "  make db-logs      Show PostgreSQL logs"
	@echo "  make db-shell     Open psql shell"
	@echo ""
	@echo "Data:"
	@echo "  make load-tests   Load test cases from fixtures"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-cov     Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts"

# ============================================================================
# Setup
# ============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev,postgres]"

# ============================================================================
# Database
# ============================================================================

db-up:
	docker compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@docker compose exec -T postgres pg_isready -U nl2api -d nl2api || (echo "PostgreSQL not ready" && exit 1)
	@echo "PostgreSQL is ready!"

db-down:
	docker compose down

db-reset:
	docker compose down -v
	docker compose up -d
	@echo "Waiting for PostgreSQL to initialize..."
	@sleep 5
	@echo "Database reset complete!"

db-logs:
	docker compose logs -f postgres

db-shell:
	docker compose exec postgres psql -U nl2api -d nl2api

# ============================================================================
# Data Loading
# ============================================================================

load-tests:
	@if [ -d "tests/fixtures/lseg/generated" ]; then \
		python -m scripts.load_test_cases tests/fixtures/lseg/generated/; \
	elif [ -d "tests/fixtures" ]; then \
		python -m scripts.load_test_cases tests/fixtures/; \
	else \
		echo "No test fixtures found"; \
	fi

load-tests-dry:
	@if [ -d "tests/fixtures/lseg/generated" ]; then \
		python -m scripts.load_test_cases --dry-run tests/fixtures/lseg/generated/; \
	elif [ -d "tests/fixtures" ]; then \
		python -m scripts.load_test_cases --dry-run tests/fixtures/; \
	else \
		echo "No test fixtures found"; \
	fi

# ============================================================================
# Testing
# ============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# ============================================================================
# Code Quality
# ============================================================================

lint:
	ruff check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/

# ============================================================================
# Cleanup
# ============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
