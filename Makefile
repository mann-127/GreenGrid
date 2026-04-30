.PHONY: install test format lint fix cov clean help

help:
	@echo "GreenGrid Development Commands"
	@echo ""
	@echo "  make install    Install package with dev dependencies (uv)"
	@echo "  make format     Auto-format code with ruff"
	@echo "  make lint       Run ruff linter"
	@echo "  make fix        Run ruff linter with auto-fix"
	@echo "  make test       Run pytest test suite"
	@echo "  make cov        Run tests with coverage report"
	@echo "  make clean      Remove build artefacts, caches, and coverage files"
	@echo ""

install:
	uv sync --all-extras --all-groups

format:
	uv run ruff format greengrid tests

lint:
	uv run ruff check greengrid tests

fix:
	uv run ruff check --fix greengrid tests

test:
	uv run pytest -v

cov:
	uv run pytest --cov=greengrid --cov-report=term-missing --cov-report=html

clean:
	rm -rf dist/ build/ *.egg-info/ greengrid.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
