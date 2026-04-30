"""
Configuration Management
========================
Loads and exposes the YAML configuration file with environment variable overrides.

Usage:
    from greengrid.settings import CFG
    wind_farm_mw = CFG["data_generation"]["wind"][\"capacity_mw\"]

Environment Variables:
    GREENGRID_SEED: Override random seed (int).
    Any GREENGRID_* variable prefixed value overrides config key.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

_CFG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and parse YAML configuration file with env-var overrides.

    Args:
        path: Optional override path to config file. Defaults to greengrid/config.yaml.

    Returns:
        Configuration dictionary merged with environment variable overrides.

    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If config file has invalid YAML.
    """
    cfg_path = Path(path) if path else _CFG_PATH
    logger.debug(f"[settings] Loading config from {cfg_path}")

    if not cfg_path.exists():
        msg = f"Config file not found: {cfg_path}"
        logger.error(f"[settings] {msg}")
        raise FileNotFoundError(msg)

    try:
        with open(cfg_path) as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh)
        logger.info(f"[settings] Loaded config: {cfg_path.name} with {len(cfg)} top-level sections")
    except PermissionError as e:
        logger.error(f"[settings] Permission denied reading config: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"[settings] Failed to parse YAML: {e}")
        raise

    # Allow env-var overrides for CI / cloud / secrets
    # Example: GREENGRID_SEED=123 overrides config["project"]["seed"]
    if seed := os.getenv("GREENGRID_SEED"):
        try:
            cfg["project"]["seed"] = int(seed)
            logger.debug(f"[settings] Env override applied: GREENGRID_SEED={seed}")
        except (ValueError, KeyError) as e:
            logger.warning(f"[settings] Failed to apply GREENGRID_SEED override: {e}")

    logger.debug(f"[settings] Config ready (seed={cfg.get('project', {}).get('seed', 'N/A')})")
    return cfg


# Module-level singleton: helps avoid re-parsing config on every import
# Loads once at module initialization time
try:
    CFG: dict[str, Any] = load_config()
    logger.info("[settings] Configuration initialized successfully")
except Exception as e:
    logger.critical(f"[settings] Failed to initialize configuration: {e}")
    raise
