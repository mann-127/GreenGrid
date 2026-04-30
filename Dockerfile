# ────────────────────────────────────────────────
# GreenGrid — Multi-stage Docker build
# ────────────────────────────────────────────────
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade all base-image packages first to pick up OS-level security patches,
# then install only the minimal runtime deps (no compiler tools in prod).
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (separate layer) so code changes don't bust cache.
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install .

# Copy application code after dependencies are installed.
COPY greengrid/ greengrid/

# Run as non-root to limit blast radius of any container escape.
RUN useradd -m appuser
USER appuser

# ── Default: run the CLI ─────────────────────────
ENTRYPOINT ["greengrid"]
CMD ["--help"]

# ── Streamlit dashboard ─────────────────────────
FROM base AS dashboard
USER appuser
ENTRYPOINT []
EXPOSE 8501
CMD ["streamlit", "run", "greengrid/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

# ── Airflow ──────────────────────────────────────
# Airflow does not support Python 3.13 yet, so use 3.12 for this stage.
FROM python:3.12-slim AS airflow

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRFLOW_HOME=/app/airflow \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies before copying code.
COPY pyproject.toml .
# Install core package + airflow extra (Airflow requires Python <3.13)
RUN pip install --upgrade pip && pip install ".[airflow]"

COPY greengrid/ greengrid/
COPY airflow/ airflow/

RUN useradd -m appuser
USER appuser

EXPOSE 8080
CMD ["airflow", "standalone"]
