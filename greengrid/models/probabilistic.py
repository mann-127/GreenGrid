"""
Probabilistic Forecasting Utilities
=====================================
Post-processing & analysis of quantile forecasts:
  • Prediction-interval extraction
  • Coverage probability calculation
  • Calibration diagnostics
  • Confidence-band formatting for reports
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class PredictionInterval:
    """One named prediction interval (e.g., 90 %)."""

    level: float  # e.g. 0.90
    lower: np.ndarray  # (N, horizon, n_targets)
    upper: np.ndarray
    point: np.ndarray  # median


@dataclass
class ProbabilisticForecast:
    """Full probabilistic forecast with multiple intervals."""

    point: np.ndarray  # median (N, H, T)
    quantiles: dict[float, np.ndarray]  # q: (N, H, T)
    intervals: list[PredictionInterval]


def extract_intervals(
    quantile_preds: dict[float, np.ndarray],
    levels: list[float] | None = None,
) -> ProbabilisticForecast:
    """
    Convert raw quantile predictions into named prediction intervals.

    Parameters
    ----------
    quantile_preds : {0.05: array, 0.25: array, 0.50: array, ...}
    levels : Desired coverage levels, e.g. [0.50, 0.90].
             Automatically matched to closest available quantiles.

    Returns
    -------
    ProbabilisticForecast with intervals.
    """
    available_q = sorted(quantile_preds.keys())
    point = quantile_preds.get(0.50, quantile_preds[available_q[len(available_q) // 2]])

    if levels is None:
        levels = [0.50, 0.90]

    intervals = []
    for level in levels:
        alpha = (1 - level) / 2
        # Find closest available quantiles
        lower_q = min(available_q, key=lambda q: abs(q - alpha))
        upper_q = min(available_q, key=lambda q: abs(q - (1 - alpha)))
        intervals.append(
            PredictionInterval(
                level=level,
                lower=quantile_preds[lower_q],
                upper=quantile_preds[upper_q],
                point=point,
            )
        )
        logger.debug(f"  PI {level:.0%}: q_lower={lower_q}, q_upper={upper_q}")

    return ProbabilisticForecast(
        point=point,
        quantiles=quantile_preds,
        intervals=intervals,
    )


def empirical_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Fraction of actuals that fall within [lower, upper].

    Parameters
    ----------
    actual : (N, H, T)  or flattened
    lower  : same shape
    upper  : same shape

    Returns
    -------
    float in [0, 1]
    """
    inside = (actual >= lower) & (actual <= upper)
    return float(inside.mean())


def calibration_report(
    forecast: ProbabilisticForecast,
    actual: np.ndarray,
) -> dict[float, dict[str, float]]:
    """
    For each prediction interval, report empirical coverage vs nominal.

    Returns
    -------
    {level: {"nominal": 0.90, "empirical": 0.87, "gap": -0.03}, ...}
    """
    report = {}
    for pi in forecast.intervals:
        emp = empirical_coverage(actual, pi.lower, pi.upper)
        gap = emp - pi.level
        report[pi.level] = {"nominal": pi.level, "empirical": emp, "gap": gap}
        logger.info(f"  PI {pi.level:.0%}:  empirical={emp:.3f}  nominal={pi.level:.3f}  gap={gap:+.3f}")
    return report


def format_forecast_summary(
    forecast: ProbabilisticForecast,
    target_names: list[str],
    hour_offset: int = 0,
) -> str:
    """
    Pretty-print the next-24h forecast as a table.

    Example output::

        Hour | Wind (MW)          | Solar (MW)
        ─────┼────────────────────┼──────────────────
         +1  | 82.3 [71.5, 93.1]  | 0.0 [0.0, 0.0]
         +2  | 85.1 [73.0, 96.2]  | 0.0 [0.0, 0.0]
         ...
    """
    lines = []
    n_targets = forecast.point.shape[-1]
    pi_90 = next((pi for pi in forecast.intervals if pi.level >= 0.85), forecast.intervals[-1])

    header = "Hour |"
    for name in target_names:
        header += f" {name:>20s} |"
    lines.append(header)
    lines.append("─" * len(header))

    horizon = forecast.point.shape[1]
    for h in range(horizon):
        row = f" +{h + 1 + hour_offset:>2d}  |"
        for t in range(n_targets):
            p = forecast.point[0, h, t]
            lo = pi_90.lower[0, h, t]
            hi = pi_90.upper[0, h, t]
            row += f"  {p:6.1f} [{lo:5.1f}, {hi:5.1f}] |"
        lines.append(row)

    return "\n".join(lines)
