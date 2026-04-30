"""
Evaluation Metrics
===================
Forecasting & grid-optimisation metrics used to compare models.
"""

import numpy as np
from loguru import logger

# ── Point-forecast metrics ───────────────────────────────────────────


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + epsilon))) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


# ── Probabilistic metrics ───────────────────────────────────────────


def pinball_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
    quantile: float,
) -> float:
    """
    Quantile / pinball loss for a single quantile level.

    L_q = mean( max[q·(y−ŷ), (q−1)·(y−ŷ)] )
    """
    diff = actual - predicted
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def mean_pinball_loss(
    actual: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> float:
    """Average pinball loss across all quantile levels."""
    losses = [pinball_loss(actual, pred, q) for q, pred in quantile_preds.items()]
    return float(np.mean(losses))


def coverage_probability(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage: fraction of actuals within [lower, upper]."""
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction-interval width — narrower is better (at same coverage)."""
    return float(np.mean(upper - lower))


def winkler_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Winkler interval score — rewards narrow intervals, penalises misses.
    Lower is better.
    """
    width = upper - lower
    below = lower - actual
    above = actual - upper
    penalty = (2 / alpha) * (np.maximum(below, 0) + np.maximum(above, 0))
    return float(np.mean(width + penalty))


# ── Grid / curtailment metrics ───────────────────────────────────────


def curtailment_reduction(
    baseline_curtailment_mwh: float,
    model_curtailment_mwh: float,
) -> float:
    """
    Percentage reduction in curtailment vs baseline.
    Positive means improvement.
    """
    if baseline_curtailment_mwh <= 0:
        return 0.0
    return float((baseline_curtailment_mwh - model_curtailment_mwh) / baseline_curtailment_mwh * 100)


def revenue_improvement(
    baseline_revenue: float,
    model_revenue: float,
) -> float:
    """Percentage revenue improvement vs baseline."""
    if baseline_revenue <= 0:
        return 0.0
    return float((model_revenue - baseline_revenue) / baseline_revenue * 100)


# ── Comprehensive report ─────────────────────────────────────────────


def compute_all_metrics(
    actual: np.ndarray,
    point_pred: np.ndarray,
    quantile_preds: dict[float, np.ndarray] | None = None,
    lower_90: np.ndarray | None = None,
    upper_90: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all relevant metrics and return as a flat dictionary.
    """
    metrics = {
        "mae": mae(actual, point_pred),
        "rmse": rmse(actual, point_pred),
        "mape": mape(actual, point_pred),
        "r_squared": r_squared(actual, point_pred),
    }

    if quantile_preds:
        metrics["mean_pinball_loss"] = mean_pinball_loss(actual, quantile_preds)

    if lower_90 is not None and upper_90 is not None:
        metrics["coverage_90"] = coverage_probability(actual, lower_90, upper_90)
        metrics["interval_width_90"] = interval_width(lower_90, upper_90)
        metrics["winkler_90"] = winkler_score(actual, lower_90, upper_90, alpha=0.10)

    logger.info("Metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics
