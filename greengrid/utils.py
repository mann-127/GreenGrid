"""
Shared helper utilities used across the GreenGrid codebase.
"""

import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42):
    """Deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    """Return best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hour_to_sincos(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclical encoding for hour-of-day (0-23)."""
    rad = 2 * np.pi * hour / 24.0
    return np.sin(rad), np.cos(rad)


def month_to_sincos(month: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclical encoding for month-of-year (1-12)."""
    rad = 2 * np.pi * (month - 1) / 12.0
    return np.sin(rad), np.cos(rad)
