"""
Baseline: Moving-Average Forecaster
=====================================
A simple yet surprisingly competitive baseline that uses multiple
window sizes and picks the best one on the validation set.  It also
produces naive probabilistic bands via historical residual quantiles.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from greengrid.settings import CFG


@dataclass
class BaselinePrediction:
    """Container for baseline forecasts."""

    point: np.ndarray  # (N, horizon, n_targets)
    quantiles: dict[float, np.ndarray]  # q: (N, horizon, n_targets)
    best_window: int


def moving_average_forecast(
    history: np.ndarray,
    horizon: int,
    window: int,
) -> np.ndarray:
    """For each sample, take the last window steps and tile the mean forward.

    Shape: (N, seq_len, F) -> (N, horizon, F)
    """
    # history: (N, seq_len, n_targets)
    tail = history[:, -window:, :]  # (N, window, T)
    avg = tail.mean(axis=1, keepdims=True)  # (N, 1, T)
    return np.tile(avg, (1, horizon, 1))  # (N, horizon, T)


def fit_baseline(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    cfg: dict | None = None,
) -> BaselinePrediction:
    """
    Evaluate multiple window sizes on *validation* data and return the
    best-performing moving-average forecast on ``val_X``.

    Parameters
    ----------
    train_X : (N_train, seq_len, n_features)  - we only use target cols
    train_y : (N_train, horizon, n_targets)
    val_X   : (N_val, seq_len, n_features)
    val_y   : (N_val, horizon, n_targets)

    Returns
    -------
    BaselinePrediction - point forecast + probabilistic quantiles.
    """
    if cfg is None:
        cfg = CFG
    mcfg = cfg["models"]["baseline"]
    window_sizes: list[int] = mcfg["window_sizes"]
    horizon = val_y.shape[1]

    # We need only target columns from X - they sit at the end after
    # preprocessing, but the caller should pass the target slice.
    # For safety, we use train_y's residual distribution.

    if not window_sizes:
        raise ValueError("models.baseline.window_sizes must not be empty")

    best_rmse = float("inf")
    best_window = window_sizes[0]
    best_pred: np.ndarray = moving_average_forecast(val_X, horizon, min(best_window, val_X.shape[1]))

    for w in window_sizes:
        pred = moving_average_forecast(val_X, horizon, min(w, val_X.shape[1]))
        rmse = float(np.sqrt(((pred - val_y) ** 2).mean()))
        logger.info(f"  MA window={w:>3d}  |  RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = w
            best_pred = pred

    logger.success(f"Best baseline window = {best_window}  (RMSE {best_rmse:.4f})")

    # ── Probabilistic bands from training residuals ───────────────────
    train_pred = moving_average_forecast(train_X, horizon, min(best_window, train_X.shape[1]))
    residuals = train_y - train_pred  # (N_train, horizon, T)

    # Use a shared quantile list if defined; fall back to sensible defaults.
    # Previously this accidentally read from the LSTM config block.
    quantile_levels = cfg["models"].get("quantiles") or mcfg.get("quantiles") or [0.05, 0.25, 0.50, 0.75, 0.95]
    quantile_forecasts: dict[float, np.ndarray] = {}
    for q in quantile_levels:
        q_offset = np.quantile(residuals, q, axis=0)  # (horizon, T)
        quantile_forecasts[q] = best_pred + q_offset[np.newaxis, :, :]

    return BaselinePrediction(
        point=best_pred,
        quantiles=quantile_forecasts,
        best_window=best_window,
    )
