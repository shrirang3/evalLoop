# EvalLoop developer tasks. `make` on its own lists them.
#
# Everything runs through `uv run`, so no virtualenv activation is needed and
# CI and local development execute identical commands.

COMPOSE := docker compose -f docker/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help install up down logs psql reset-db lint format typecheck test itest cov examples check ci clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync dependencies, including the dev extra
	uv sync --extra dev

up: ## Start the metastore and the stand-in source database
	$(COMPOSE) up -d
	@$(COMPOSE) ps

down: ## Stop the stack, keeping data volumes
	$(COMPOSE) down

logs: ## Tail the stack logs
	$(COMPOSE) logs -f

psql: ## Open a psql shell on the metastore
	$(COMPOSE) exec metastore psql -U evalloop -d evalloop

reset-db: ## Drop and rebuild the metastore schema (clears all rows)
	uv run alembic downgrade base
	uv run alembic upgrade head

lint: ## Ruff check
	uv run ruff check evalloop/ tests/

format: ## Ruff format, in place
	uv run ruff format evalloop/ tests/

typecheck: ## mypy --strict
	uv run mypy evalloop/

test: ## Unit tests only (no Docker required)
	uv run pytest -q -m "not integration and not gpu"

itest: up ## Integration tests (needs the compose stack)
	uv run pytest -q -m integration

cov: ## Unit tests with a coverage report
	uv run pytest -q -m "not integration and not gpu" \
		--cov=evalloop --cov-report=term-missing

examples: ## Validate the shipped example configs
	uv run evalloop validate examples/support-bot/*.yaml

check: lint typecheck test examples ## Lint, types, unit tests, examples

ci: check itest ## Everything CI runs, including integration tests
	uv run pytest -q --cov=evalloop --cov-report=term-missing --cov-fail-under=95

clean: ## Remove caches and coverage output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
