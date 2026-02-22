# ────────────────────────────────────────────────
# GreenGrid — Multi-stage Docker build
# ────────────────────────────────────────────────
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# ── Application code & Dependencies ─────────────
COPY . .
# Install the package and all dependencies directly from pyproject.toml
RUN pip install --upgrade pip && pip install .

# ── Default: run the CLI ─────────────────────────
ENTRYPOINT ["greengrid"]
CMD ["--help"]

# ── Streamlit dashboard ─────────────────────────
FROM base AS dashboard
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

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# ── Application code & Dependencies ─────────────
COPY . .
# The pyproject.toml environmental marker ensures airflow is only installed in this 3.12 stage
RUN pip install --upgrade pip && pip install .

EXPOSE 8080
CMD ["airflow", "standalone"]
