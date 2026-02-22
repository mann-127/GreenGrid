"""
Data Preprocessing Pipeline
============================
Transforms raw hourly data into windowed tensors ready for training:

1. Feature selection & NaN handling
2. Train / val / test chronological split
3. StandardScaler / MinMaxScaler fitting (train-only)
4. Sliding-window sequence creation
5. PyTorch ``Dataset`` / ``DataLoader`` wrappers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from greengrid.settings import CFG


# ─── Helpers ──────────────────────────────────────────────────────────

@dataclass
class SplitData:
    """Holds arrays and metadata for one split (train / val / test)."""
    X: np.ndarray          # (N, seq_len, n_features)
    y: np.ndarray          # (N, horizon, n_targets)
    timestamps: np.ndarray # (N,)  — timestamp of the *last* step in X
    raw_df: pd.DataFrame   # un-scaled slice (for plotting)


@dataclass
class PreparedData:
    """Full pipeline output."""
    train: SplitData
    val: SplitData
    test: SplitData
    feature_scaler: StandardScaler | MinMaxScaler
    target_scaler: StandardScaler | MinMaxScaler
    feature_columns: list[str]
    target_columns: list[str]


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset wrapping numpy arrays from ``SplitData``."""

    def __init__(self, split: SplitData):
        self.X = torch.as_tensor(split.X, dtype=torch.float32)
        self.y = torch.as_tensor(split.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─── Core pipeline ───────────────────────────────────────────────────

def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Parquet or CSV dataset from disk.
    
    Tries parquet first (faster), falls back to CSV.
    
    Args:
        path: Optional override path to data directory. Defaults to config.
        
    Returns:
        DataFrame with columns: timestamp, wind_power_mw, solar_power_mw, features.
        
    Raises:
        FileNotFoundError: If neither parquet nor CSV exists.
    """
    data_dir = Path(path) if path else Path(CFG["paths"]["raw_data"])
    logger.debug(f"[preprocessing] Loading raw data from {data_dir}")

    pq = data_dir / "greengrid_raw.parquet"
    if pq.exists():
        logger.debug(f"[preprocessing] Reading parquet: {pq}")
        df = pd.read_parquet(pq)
    else:
        csv = data_dir / "greengrid_raw.csv"
        logger.debug(f"[preprocessing] Reading CSV: {csv}")
        df = pd.read_csv(csv, parse_dates=["timestamp"])

    logger.info(f"[preprocessing] Loaded raw data: shape={df.shape}, date_range={df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def _chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict chronological (no shuffle) train/val/test split.
    
    Critical for time-series: we must never use future data for training.
    Test set is purely out-of-sample temporal data.
    
    Args:
        df: Input dataframe, assumed sorted by timestamp.
        train_ratio: Fraction for training (0-1).
        val_ratio: Fraction for validation (0-1). Test = 1 - train - val.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = i_train + int(n * val_ratio)

    logger.debug(
        f"[preprocessing] Chronological split: train=[0:{i_train}], "
        f"val=[{i_train}:{i_val}], test=[{i_val}:{n}]"
    )
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def _fit_scaler(
    kind: Literal["standard", "minmax"], train: np.ndarray
) -> StandardScaler | MinMaxScaler:
    """Fit a scaler (StandardScaler or MinMaxScaler) on training data only.
    
    IMPORTANT: Scaler is fit on training data only to prevent data leakage.
    Same fitted scaler is then applied to val and test.
    
    Args:
        kind: Type of scaler ('standard' or 'minmax').
        train: Training feature array of shape (n_train, n_features).
        
    Returns:
        Fitted scaler object.
        
    Raises:
        ValueError: If kind is not 'standard' or 'minmax'.
    """
    if kind == "standard":
        scaler = StandardScaler()
        logger.debug("[preprocessing] Using StandardScaler (z-score normalization)")
    elif kind == "minmax":
        scaler = MinMaxScaler()
        logger.debug("[preprocessing] Using MinMaxScaler (0-1 range normalization)")
    else:
        raise ValueError(f"Unknown scaler kind: {kind}")

    scaler.fit(train)
    logger.debug(f"[preprocessing] Scaler fitted on {train.shape[0]} training samples")
    return scaler


def _create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window with (seq_len history) -> (horizon future)."""
    X, y, ts = [], [], []
    for i in range(len(features) - seq_len - horizon + 1):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len : i + seq_len + horizon])
        ts.append(timestamps[i + seq_len - 1])
    return np.array(X), np.array(y), np.array(ts)


# ─── Public API ──────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> PreparedData:
    """
    Full preprocessing pipeline:
    raw DataFrame -> scaled sliding-window arrays.
    """
    if cfg is None:
        cfg = CFG
    pp = cfg["preprocessing"]

    if df is None:
        df = load_raw()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna()

    feature_cols = pp["feature_columns"]
    target_cols = pp["target_columns"]
    seq_len = pp["sequence_length"]
    horizon = pp["forecast_horizon"]

    # ── Split ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = _chronological_split(
        df, pp["train_ratio"], pp["val_ratio"]
    )
    logger.info(
        f"Split sizes  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    # ── Scale (fit on train only) ─────────────────────────────────────
    feat_scaler = _fit_scaler(pp["scaler"], train_df[feature_cols].values)
    tgt_scaler = _fit_scaler(pp["scaler"], train_df[target_cols].values)

    def _scale(split_df: pd.DataFrame):
        f = feat_scaler.transform(split_df[feature_cols].values)
        t = tgt_scaler.transform(split_df[target_cols].values)
        return f, t

    train_f, train_t = _scale(train_df)
    val_f, val_t = _scale(val_df)
    test_f, test_t = _scale(test_df)

    # ── Sequences ─────────────────────────────────────────────────────
    def _make_split(f, t, raw_df) -> SplitData:
        ts = raw_df["timestamp"].values
        X, y, timestamps = _create_sequences(f, t, ts, seq_len, horizon)
        return SplitData(X=X, y=y, timestamps=timestamps, raw_df=raw_df)

    return PreparedData(
        train=_make_split(train_f, train_t, train_df),
        val=_make_split(val_f, val_t, val_df),
        test=_make_split(test_f, test_t, test_df),
        feature_scaler=feat_scaler,
        target_scaler=tgt_scaler,
        feature_columns=feature_cols,
        target_columns=target_cols,
    )


def build_dataloaders(
    data: PreparedData,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience wrapper -> (train_loader, val_loader, test_loader)."""
    bs = batch_size or CFG["models"]["lstm"]["batch_size"]
    train_ds = TimeSeriesDataset(data.train)
    val_ds = TimeSeriesDataset(data.val)
    test_ds = TimeSeriesDataset(data.test)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    logger.info(
        f"DataLoaders ready  —  batch_size={bs}  |  "
        f"train_batches={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}"
    )
    return train_loader, val_loader, test_loader
