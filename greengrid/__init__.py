"""
GreenGrid — AI-Driven Smart Grid Controller
============================================
Predictive maintenance & load-balancing for renewable energy.

Core Modules
============
  greengrid.cli:           Command-line interface (entry point)
  greengrid.config:        Central configuration management
  greengrid.settings:      Config loader with env-var overrides
  
Data Pipeline
=============
  greengrid.data.generator:      Synthetic wind/solar data generation
  greengrid.data.preprocessing:  Feature engineering, scaling, windowing
  
Machine Learning Models
=======================
  greengrid.models.baseline:     Moving-average baseline forecaster
  greengrid.models.lstm_model:   Probabilistic Bi-LSTM with quantile regression
  greengrid.models.tft_model:    Temporal Fusion Transformer
  greengrid.models.trainer:      PyTorch Lightning training harness
  
Grid Optimization
=================
  greengrid.optimizer.battery:   BESS simulator with physical constraints
  greengrid.optimizer.dispatch:  Naive and forecast-aware dispatch strategies
  
Evaluation & Logging
====================
  greengrid.evaluation.metrics:     Point & probabilistic forecast metrics
  greengrid.evaluation.simulation:  End-to-end simulation engine
  greengrid.dashboard.app:          Streamlit interactive dashboard

Architecture
============
  Data Sources -> Data Pipeline -> Models -> Dispatch -> Dashboard
  
  Coordinated by:
    • Apache Airflow DAG (airflow/dags/greengrid_dag.py)
    • Docker Compose (docker-compose.yml)

Configuration
==============
  Primary config: greengrid/config.yaml
  Override with env vars: GREENGRID_SEED=42, etc.

Debugging & Logging
====================
  Uses loguru for structured logging:
    from loguru import logger
    logger.debug(\"[module_name] Debug message\")
    logger.info(\"[module_name] Info message\")
    logger.warning(\"[module_name] Warning message\")
    logger.error(\"[module_name] Error message\")
  
  All functions log entry/exit and intermediate state for traceability.
  Set environment variable to control verbosity:
    export LOGURU_LEVEL=DEBUG  # for maximum detail
    export LOGURU_LEVEL=WARNING  # for minimal output

Quick Start
===========
  greengrid --help                    # Show all commands
  greengrid generate-data             # Generate synthetic data
  greengrid train --model lstm        # Train LSTM forecaster
  greengrid simulate                  # Run end-to-end simulation
  greengrid dashboard                 # Launch interactive dashboard

Version
=======
"""

__version__ = "1.0.0"
