# GreenGrid: Renewable Energy Dispatch Optimizer

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-green.svg)](...)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/mann-127/BladeRunner/actions/workflows/ci.yml/badge.svg)](https://github.com/mann-127/BladeRunner/actions)
[![Permissions: Standard](https://img.shields.io/badge/permissions-standard-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![System Architecture](assets/system-architecture.png)

## Overview
GreenGrid is an end-to-end Machine Learning pipeline designed to harmonize human energy demands with the physical constraints of renewable resources. By generating probabilistic weather forecasts and optimizing Battery Energy Storage System (BESS) dispatch, GreenGrid prevents energy waste while actively protecting the physical lifespan of the hardware.

### The Problem
Renewable energy sources like wind and solar are highly intermittent. When energy production exceeds grid demand, the excess energy is often "curtailed" (wasted). While batteries can store this excess, naive dispatch algorithms often perform "micro-cycles"—charging and discharging for negligible profits, which severely degrades the battery's physical health over time. 

### Key Results
* **>15% Curtailment Reduction:** Effectively stores and shifts surplus renewable energy to peak demand hours.
* **Degradation-Aware AI:** The custom dispatch algorithm calculates a physical penalty cost ($/MWh) for battery wear-and-tear (cycle aging and calendar aging), refusing to cycle the battery if spot prices do not justify the physical damage.
* **End-to-End Automation:** Fully orchestrated via Apache Airflow, integrating real-world Open-Meteo API data seamlessly into PyTorch forecasting models.

## What it does
- **Real-world weather API integration** & synthetic physics-based data generation
- Probabilistic energy forecasts (Quantiles via LSTM & Temporal Fusion Transformers)
- **Degradation-aware** battery dispatch strategies (Cycle & Calendar aging)
- Interactive Streamlit visualization dashboard tracking financial and hardware metrics
- Automated Airflow orchestration and Docker support

---

## Quick start

### Prerequisites
- Python 3.13+
- uv (recommended) or pip 23+

### Install
```bash
git clone https://github.com/mann-127/GreenGrid.git
cd GreenGrid
uv sync --extra dev
```

### Alternative (pip/venv):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Run
The entire pipeline can be executed via the unified CLI:
```bash
greengrid generate-data                  # Generate synthetic weather & power data
greengrid generate-data --real-weather   # OR fetch real historical weather via API
greengrid train --model lstm             # Train the forecasting model (lstm or tft)
greengrid simulate                       # Run the degradation-aware dispatch simulation
greengrid simulate --model tft           # Use TFT model for simulation
greengrid simulate --skip-training       # Baseline-only mode (no model required)
greengrid dashboard                      # Launch the interactive UI
```

### Docker
```bash
docker compose up -d
# Dashboard: http://localhost:8501
# Airflow: http://localhost:8080
```

*Note: The Airflow service uses Python 3.12 because Airflow does not yet support 3.13.*

## Configuration
All central settings live in `greengrid/config.yaml`. You can tune physical battery constraints directly in this file, including `calendar_aging_per_hour` and the financial `cycle_penalty_cost_mwh`.

Override the random seed with `GREENGRID_SEED=42`. Other config values must be edited directly in `config.yaml`.

## Outputs
Generated data, checkpoints, logs, and results are created at runtime and are gitignored. Run `greengrid generate-data` to populate `data/raw/` before training.

## Project layout
```text
├── greengrid/            # Core Python package
│   ├── config.yaml       # Central configuration parameters
│   ├── settings.py       # Config loader with env-var overrides
│   ├── utils.py          # Shared helpers (seeding, device, paths)
│   ├── cli.py            # Unified command-line interface
│   ├── data/             # API fetching & synthetic generation
│   ├── models/           # LSTM, TFT, and baseline models
│   ├── optimizer/        # Degradation-aware dispatch & battery physics
│   ├── evaluation/       # Simulation engine & metrics
│   └── dashboard/        # Streamlit interactive UI
│
├── airflow/dags/         # Automated orchestration pipelines
├── assets/               # Architecture flowcharts & visuals
├── tests/                # Pytest suite (Logic & Physics constraints)
│
├── Dockerfile            # Multi-stage container build
├── docker-compose.yml    # Services configuration (App, UI, Airflow)
├── pyproject.toml        # Modern dependency & build config
├── uv.lock               # Exact dependency lockfile
└── Makefile              # Developer command shortcuts
```

## Development
```bash
make install       # Install with dev deps
make lint          # Ruff linting
make fix           # Ruff linting with auto-fix
make format        # Auto-format code
make test          # Run pytest suite
make clean         # Remove caches
```

### Or manually:
```bash
uv run ruff check greengrid tests
uv run ruff format greengrid tests
uv run pytest -v
```

## Testing
Tests cover battery physics constraints, dispatch logic, metrics, data generation, and preprocessing. Run the full suite with coverage:
```bash
make cov
# or: uv run pytest --cov=greengrid --cov-report=term-missing
```

## License
MIT License. See `LICENSE`.
