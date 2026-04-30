"""
Training Harness
=================
Unified training entry-point for both LSTM and TFT models using
PyTorch Lightning.
"""

from pathlib import Path

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from greengrid.data.preprocessing import PreparedData, build_dataloaders
from greengrid.models.lstm_model import ProbabilisticLSTM
from greengrid.models.tft_model import TemporalFusionTransformer
from greengrid.settings import CFG
from greengrid.utils import ensure_dir


def _build_callbacks(model_name: str, patience: int, ckpt_dir: Path) -> list:
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=True,
            min_delta=0.0001,
        ),
        checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
    ]


def train_lstm(data: PreparedData, cfg: dict | None = None) -> ProbabilisticLSTM:
    """Train the Probabilistic LSTM and return the best checkpoint."""
    if cfg is None:
        cfg = CFG
    mc = cfg["models"]["lstm"]
    pp = cfg["preprocessing"]

    n_features = len(pp["feature_columns"])
    n_targets = len(pp["target_columns"])
    horizon = pp["forecast_horizon"]

    model = ProbabilisticLSTM(
        n_features=n_features,
        n_targets=n_targets,
        horizon=horizon,
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
        bidirectional=mc["bidirectional"],
        learning_rate=mc["learning_rate"],
        quantiles=mc["quantiles"],
    )

    ckpt_dir = ensure_dir(cfg["paths"]["models"])
    train_loader, val_loader, _ = build_dataloaders(data, mc["batch_size"])

    trainer = pl.Trainer(
        max_epochs=mc["max_epochs"],
        callbacks=_build_callbacks("lstm", mc["patience"], ckpt_dir),
        gradient_clip_val=mc.get("gradient_clip_val", 1.0),
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    callbacks = _build_callbacks("lstm", mc["patience"], ckpt_dir)
    trainer = pl.Trainer(
        max_epochs=mc["max_epochs"],
        callbacks=callbacks,
        gradient_clip_val=mc.get("gradient_clip_val", 1.0),
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)

    checkpoint_callback = next(
        (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)),
        None,
    )
    if checkpoint_callback is None:
        raise RuntimeError(
            "LSTM training produced no checkpoint. "
            "Check that validation data is non-empty and at least one epoch completed."
        )
    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError(
            "LSTM training produced no checkpoint. "
            "Check that validation data is non-empty and at least one epoch completed."
        )
    logger.success(f"LSTM best checkpoint: {best_path}")
    return ProbabilisticLSTM.load_from_checkpoint(best_path)


def train_tft(data: PreparedData, cfg: dict | None = None) -> TemporalFusionTransformer:
    """Train the Temporal Fusion Transformer."""
    if cfg is None:
        cfg = CFG
    mc = cfg["models"]["tft"]
    pp = cfg["preprocessing"]

    n_features = len(pp["feature_columns"])
    n_targets = len(pp["target_columns"])
    horizon = pp["forecast_horizon"]

    model = TemporalFusionTransformer(
        n_features=n_features,
        n_targets=n_targets,
        horizon=horizon,
        hidden_size=mc["hidden_size"],
        n_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
        learning_rate=mc["learning_rate"],
        quantiles=mc["quantiles"],
    )

    ckpt_dir = ensure_dir(cfg["paths"]["models"])
    train_loader, val_loader, _ = build_dataloaders(data, mc["batch_size"])

    trainer = pl.Trainer(
        max_epochs=mc["max_epochs"],
        callbacks=_build_callbacks("tft", mc["patience"], ckpt_dir),
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    callbacks = _build_callbacks("tft", mc["patience"], ckpt_dir)
    trainer = pl.Trainer(
        max_epochs=mc["max_epochs"],
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)

    checkpoint_callback = next(
        (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)),
        None,
    )
    if checkpoint_callback is None:
        raise RuntimeError(
            "TFT training produced no checkpoint. "
            "Check that validation data is non-empty and at least one epoch completed."
        )

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError(
            "TFT training produced no checkpoint. "
            "Check that validation data is non-empty and at least one epoch completed."
        )
    logger.success(f"TFT best checkpoint: {best_path}")
    return TemporalFusionTransformer.load_from_checkpoint(best_path)


def load_model(path: str | Path, model_type: str = "lstm") -> pl.LightningModule:
    """Load a saved checkpoint."""
    if model_type == "lstm":
        return ProbabilisticLSTM.load_from_checkpoint(str(path))
    elif model_type == "tft":
        return TemporalFusionTransformer.load_from_checkpoint(str(path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
