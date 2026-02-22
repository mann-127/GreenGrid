.PHONY: install lint format test clean help

help:
	@echo "GreenGrid Development Commands"
	@echo ""
	@echo "  make install    Install package with dev dependencies (uv)"
	@echo "  make lint       Run ruff linter (uv)"
	@echo "  make format     Auto-format code with ruff (uv)"
	@echo "  make test       Run pytest test suite (uv)"
	@echo ""

install:
	uv sync --extra dev

lint:
	uv run ruff check greengrid tests

format:
	uv run ruff format greengrid tests

test:
	uv run pytest -v
