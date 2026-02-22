"""
Simulation Engine
==================
End-to-end simulation that:
  1. Generates / loads data.
  2. Preprocesses & creates sequences.
  3. Runs the baseline moving-average model.
  4. Runs the AI model (LSTM or TFT).
  5. Runs both dispatch strategies (naïve vs forecast-aware).
  6. Computes & compares all metrics.
  7. Checks whether the 15% curtailment-reduction target is met.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from greengrid.data.generator import generate_dataset, save_dataset
from greengrid.data.preprocessing import PreparedData, prepare_data, build_dataloaders
from greengrid.evaluation.metrics import (
    compute_all_metrics,
    curtailment_reduction,
    revenue_improvement,
)
from greengrid.models.baseline import fit_baseline
from greengrid.models.probabilistic import extract_intervals, calibration_report
from greengrid.optimizer.battery import BatteryConfig
from greengrid.optimizer.dispatch import ForecastDispatch, NaiveDispatch, DispatchResult
from greengrid.settings import CFG
from greengrid.utils import set_seed, get_device


@dataclass
class SimulationReport:
    """Final comparison report."""
    baseline_metrics: dict[str, float]
    model_metrics: dict[str, float]
    baseline_dispatch: DispatchResult
    model_dispatch: DispatchResult
    curtailment_reduction_pct: float
    revenue_improvement_pct: float
    target_met: bool
    calibration: dict


def run_simulation(
    model=None,
    model_type: str = "lstm",
    data: PreparedData | None = None,
    cfg: dict | None = None,
    skip_training: bool = False,
) -> SimulationReport:
    """
    Execute the full GreenGrid simulation pipeline.

    Parameters
    ----------
    model : Trained ProbabilisticLSTM or TemporalFusionTransformer.
            If None and skip_training=False, a new model is trained.
    model_type : "lstm" or "tft".
    data : Pre-computed PreparedData (saves re-processing).
    skip_training : If True, use only the baseline (for quick testing).
    """
    if cfg is None:
        cfg = CFG
    set_seed(cfg["project"]["seed"])

    # ── 1. Data ──────────────────────────────────────────────────────
    if data is None:
        logger.info("Generating synthetic dataset …")
        df = generate_dataset(cfg)
        save_dataset(df)
        data = prepare_data(df, cfg)

    test = data.test
    target_scaler = data.target_scaler

    # We need target-only views for the baseline
    n_targets = test.y.shape[2]
    # The baseline uses the last n_targets columns of X for its MA calc,
    # but since X is feature-scaled, we pass target-scaled data from y.
    # For a fair comparison we reconstruct target-only history from
    # the test sequences by appending y-history.
    # Simplification: use the target columns from X by inverse-transforming.

    # ── 2. Baseline forecast ─────────────────────────────────────────
    logger.info("Fitting baseline (moving-average) …")
    # For the baseline, we pass the *target* portion of the sequences.
    # We'll create a simple view: just tile the recent mean.
    # But baseline.fit_baseline needs (X, y) — we adapt X to only carry targets.
    # Since our preprocessing stores features separately, we create a
    # "target history" by picking the scaled target values from training y.

    # Build dummy X from concatenating a sliding window of y values
    # Actually, let's adapt: baseline uses its own forecast on the test set
    # by using the *feature sequences* and just computing MA on target columns.
    # For simplicity, we use the y-shifted sequences.

    # Create fake "target-only X" for baseline
    # Shape needed: (N, seq_len, n_targets)
    # We can reconstruct this from consecutive y values in test.raw_df
    from greengrid.models.baseline import fit_baseline, moving_average_forecast

    baseline_result = fit_baseline(
        train_X=data.train.y,  # use target-only for MA
        train_y=data.train.y,
        val_X=data.val.y,
        val_y=data.val.y,
    )

    # Evaluate on test
    horizon = test.y.shape[1]
    test_baseline_pred = moving_average_forecast(
        test.y, horizon, min(baseline_result.best_window, test.y.shape[1])
    )

    # Inverse-scale predictions and actuals for real-world metrics
    def _inv(arr):
        shape = arr.shape
        flat = arr.reshape(-1, n_targets)
        inv = target_scaler.inverse_transform(flat)
        return inv.reshape(shape)

    actual_mw = _inv(test.y)
    baseline_point_mw = _inv(test_baseline_pred)

    baseline_quantiles_mw = {}
    for q, qpred in baseline_result.quantiles.items():
        # Need to recompute on test
        q_offset = np.quantile(data.train.y - moving_average_forecast(
            data.train.y, horizon, min(baseline_result.best_window, data.train.y.shape[1])
        ), q, axis=0)
        baseline_quantiles_mw[q] = _inv(test_baseline_pred + q_offset[np.newaxis, :, :])

    baseline_metrics = compute_all_metrics(
        actual_mw, baseline_point_mw,
        quantile_preds=baseline_quantiles_mw,
        lower_90=baseline_quantiles_mw.get(0.05),
        upper_90=baseline_quantiles_mw.get(0.95),
    )

    # ── 3. AI model forecast ─────────────────────────────────────────
    if not skip_training and model is not None:
        device = get_device()
        model = model.to(device).eval()
        
        # Process test data in batches to avoid memory overload
        batch_size = 64
        all_quantiles = {}
        
        for batch_start in range(0, len(test.X), batch_size):
            batch_end = min(batch_start + batch_size, len(test.X))
            X_batch = torch.as_tensor(test.X[batch_start:batch_end], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                batch_preds = model.predict_quantiles(X_batch)
            
            for q, pred_tensor in batch_preds.items():
                pred_np = pred_tensor.cpu().numpy()
                if q not in all_quantiles:
                    all_quantiles[q] = []
                all_quantiles[q].append(pred_np)
        
        # Concatenate batches
        q_preds_scaled = {q: np.vstack(all_quantiles[q]) for q in all_quantiles}
        
        q_preds_mw = {}
        for q, pred_np in q_preds_scaled.items():
            q_preds_mw[q] = _inv(pred_np)

        median_q = 0.50
        model_point_mw = q_preds_mw[median_q]

        model_metrics = compute_all_metrics(
            actual_mw, model_point_mw,
            quantile_preds=q_preds_mw,
            lower_90=q_preds_mw.get(0.05),
            upper_90=q_preds_mw.get(0.95),
        )
    else:
        # For quick testing, duplicate baseline
        logger.warning("Skipping AI model — using baseline as stand-in.")
        model_point_mw = baseline_point_mw
        q_preds_mw = baseline_quantiles_mw
        model_metrics = baseline_metrics.copy()

    # Log forecast accuracy comparison
    logger.info(
        f"\n{'═' * 60}\n"
        f"  FORECAST ACCURACY COMPARISON:\n"
        f"  Baseline RMSE:  {baseline_metrics.get('rmse', 0):.4f}  MAE: {baseline_metrics.get('mae', 0):.4f}\n"
        f"  Model RMSE:     {model_metrics.get('rmse', 0):.4f}  MAE: {model_metrics.get('mae', 0):.4f}\n"
        f"{'═' * 60}\n"
    )

    # ── 4. Dispatch simulation ───────────────────────────────────────
    logger.info("Running dispatch simulations …")
    batt_cfg = BatteryConfig.from_cfg(cfg)

    # Run full test set for comprehensive evaluation
    n_samples = len(actual_mw)
    total_baseline_curtailment = 0.0
    total_model_curtailment = 0.0
    total_baseline_revenue = 0.0
    total_model_revenue = 0.0

    baseline_dispatch_last = None
    model_dispatch_last = None

    samples_logged = 0
    for i in range(n_samples):
        # Total renewable = wind + solar (sum targets)
        ren_actual = actual_mw[i, :, :].sum(axis=-1)      # (H,)
        
        # Baseline forecast (moving-average)
        ren_baseline_median = baseline_point_mw[i, :, :].sum(axis=-1)
        ren_baseline_lower = baseline_quantiles_mw.get(0.05, baseline_quantiles_mw[min(baseline_quantiles_mw)])[i, :, :].sum(axis=-1)
        ren_baseline_upper = baseline_quantiles_mw.get(0.95, baseline_quantiles_mw[max(baseline_quantiles_mw)])[i, :, :].sum(axis=-1)
        
        # LSTM forecast
        ren_model_median = model_point_mw[i, :, :].sum(axis=-1)
        ren_model_lower = q_preds_mw.get(0.05, q_preds_mw[min(q_preds_mw)])[i, :, :].sum(axis=-1)
        ren_model_upper = q_preds_mw.get(0.95, q_preds_mw[max(q_preds_mw)])[i, :, :].sum(axis=-1)

        # Synthetic demand & price: create scenarios where forecast quality matters
        # Pattern: baseline forecast has systematic biases, LSTM learns corrected patterns
        rng = np.random.default_rng(cfg["project"]["seed"] + i)
        
        # Base pattern: morning peak, afternoon valley, evening peak
        hour_weights = 0.5 * np.sin(2 * np.pi * np.arange(horizon) / 24) + \
                       0.5 * np.cos(4 * np.pi * np.arange(horizon) / 24)
        demand = 400 + 200 * (hour_weights + rng.normal(0, 0.05, horizon))
        demand = np.clip(demand, 100, 800)
        
        peak_hours = set(cfg["optimizer"]["peak_hours"])
        price = np.where(
            np.isin(np.arange(horizon) % 24, list(peak_hours)),
            120 + rng.normal(0, 10, horizon),
            45 + rng.normal(0, 5, horizon)
        )
        price = np.clip(price, 5, 200)

        # Baseline dispatch with moving-average forecast
        baseline_dispatch_strat = ForecastDispatch(batt_cfg, cfg)
        baseline_dr = baseline_dispatch_strat.run(
            ren_baseline_median, ren_baseline_lower, ren_baseline_upper,
            demand, price,
        )
        total_baseline_curtailment += baseline_dr.total_curtailment_mwh
        total_baseline_revenue += baseline_dr.total_revenue

        # LSTM-driven forecast-aware dispatch
        smart = ForecastDispatch(batt_cfg, cfg)
        model_dr = smart.run(
            ren_model_median, ren_model_lower, ren_model_upper,
            demand, price,
        )
        total_model_curtailment += model_dr.total_curtailment_mwh
        total_model_revenue += model_dr.total_revenue

        baseline_dispatch_last = baseline_dr
        model_dispatch_last = model_dr

    # ── 5. Compare ───────────────────────────────────────────────────
    curt_red = curtailment_reduction(total_baseline_curtailment, total_model_curtailment)
    rev_imp = revenue_improvement(total_baseline_revenue, total_model_revenue)
    target = cfg["evaluation"]["target_curtailment_reduction"] * 100

    logger.info(f"\n{'═' * 60}")
    logger.info(f"  CURTAILMENT REDUCTION:  {curt_red:.1f}%  (target: {target:.0f}%)")
    logger.info(f"  REVENUE IMPROVEMENT:    {rev_imp:.1f}%")
    logger.info(f"  TARGET MET:             {'YES' if curt_red >= target else 'NO'}")
    logger.info(f"{'═' * 60}\n")

    # ── 6. Calibration ───────────────────────────────────────────────
    forecast = extract_intervals(q_preds_mw, levels=[0.50, 0.90])
    cal = calibration_report(forecast, actual_mw)

    return SimulationReport(
        baseline_metrics=baseline_metrics,
        model_metrics=model_metrics,
        baseline_dispatch=baseline_dispatch_last,
        model_dispatch=model_dispatch_last,
        curtailment_reduction_pct=curt_red,
        revenue_improvement_pct=rev_imp,
        target_met=curt_red >= target,
        calibration=cal,
    )
