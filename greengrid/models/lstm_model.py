"""
Probabilistic LSTM Forecaster
==============================
A stacked Bi-LSTM that outputs quantile forecasts instead of a
single point estimate.  Trained with the pinball loss (quantile loss)
so the model directly learns the conditional distribution:

    P(y <= y_q | X)  ~  q

Architecture
------------
Input (batch, seq_len, n_features)
  - LayerNorm
  - Bidirectional LSTM x N layers
  - Dropout
  - Linear projection to (batch, horizon x n_targets x n_quantiles)
  - Reshape to (batch, horizon, n_targets, n_quantiles)
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from greengrid.settings import CFG

# ── Quantile (pinball) loss ──────────────────────────────────────────


def quantile_loss(
    predictions: torch.Tensor,  # (B, H, T, Q)
    targets: torch.Tensor,  # (B, H, T)
    quantiles: list[float],
) -> torch.Tensor:
    """
    Pinball / quantile loss averaged over all quantile levels.

    L_q(y, ŷ) = max[q·(y − ŷ), (q−1)·(y − ŷ)]
    """
    targets = targets.unsqueeze(-1)  # (B, H, T, 1)
    q_tensor = torch.tensor(quantiles, device=predictions.device).reshape(1, 1, 1, -1)
    errors = targets - predictions  # (B, H, T, Q)
    loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
    return loss.mean()


# ── Model ────────────────────────────────────────────────────────────


class ProbabilisticLSTM(pl.LightningModule):
    """
    Quantile-regression LSTM for multi-horizon renewable-energy forecasting.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int = 24,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 1e-3,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.quantiles = quantiles or CFG["models"]["lstm"]["quantiles"]
        self.n_targets = n_targets
        self.horizon = horizon
        self.learning_rate = learning_rate
        n_quantiles = len(self.quantiles)

        direction_factor = 2 if bidirectional else 1

        self.layer_norm = nn.LayerNorm(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Project last hidden state to quantile outputs
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * n_targets * n_quantiles),
        )

        self.n_quantiles = n_quantiles

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        (batch, horizon, n_targets, n_quantiles)
        """
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)  # (B, S, H*dir)
        last = self.dropout(lstm_out[:, -1, :])  # (B, H*dir)
        out = self.fc(last)  # (B, horizon*T*Q)
        B = x.size(0)
        return out.view(B, self.horizon, self.n_targets, self.n_quantiles).contiguous()

    # ── Training step ────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = quantile_loss(pred, y, self.quantiles)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = quantile_loss(pred, y, self.quantiles)
        self.log("val_loss", loss, prog_bar=True)

        # Also log point-forecast RMSE (median quantile)
        median_idx = self.quantiles.index(0.50)
        point = pred[:, :, :, median_idx]
        rmse = torch.sqrt(((point - y) ** 2).mean())
        self.log("val_rmse", rmse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = quantile_loss(pred, y, self.quantiles)
        self.log("test_loss", loss)

    # ── Optimizer ────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── Convenience ──────────────────────────────────────────────────
    @torch.no_grad()
    def predict_quantiles(self, x: torch.Tensor) -> dict[float, torch.Tensor]:
        """Return a dict  {quantile_level: (B, H, T)}."""
        self.eval()
        pred = self(x)  # (B, H, T, Q)
        return {q: pred[:, :, :, i] for i, q in enumerate(self.quantiles)}
