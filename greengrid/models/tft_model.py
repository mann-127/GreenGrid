"""
Temporal Fusion Transformer (TFT) — Simplified Implementation
=============================================================
A streamlined TFT inspired by the paper:

    *Temporal Fusion Transformers for Interpretable Multi-horizon
     Time Series Forecasting*  (Lim et al., 2021)

Key components implemented:
  1. **Variable Selection Networks (VSN)**  — learn which features matter
  2. **Gated Residual Network (GRN)**        — non-linear feature mixing
  3. **Interpretable Multi-Head Attention**   — temporal attention
  4. **Quantile output layer**               — probabilistic forecasting

This is a *self-contained* implementation (no external TFT library
dependency) so the architecture is fully visible and hackable.
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from greengrid.models.lstm_model import quantile_loss
from greengrid.settings import CFG


# ═══════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════

class GatedLinearUnit(nn.Module):
    """GLU activation: σ(Wx + b) ⊙ (Vx + c)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * self.fc(x)


class GatedResidualNetwork(nn.Module):
    """
    GRN with optional context vector.
    out = LayerNorm(x + GLU(ELU(W1·x + W2·c + b)))
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1, context_dim: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.context_proj = nn.Linear(context_dim, d_hidden, bias=False) if context_dim else None
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.glu = GatedLinearUnit(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """
    Learns per-variable importance weights via a softmax gate, then
    applies a GRN to each selected variable.
    """

    def __init__(self, n_vars: int, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.flattened_grn = GatedResidualNetwork(n_vars * d_model, d_hidden, dropout)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_hidden, dropout) for _ in range(n_vars)
        ])
        self.gate = nn.Linear(n_vars * d_model, n_vars)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, n_vars, d_model)
        Returns: (B, T, d_model),  (B, T, n_vars)  — weights
        """
        B, T, V, D = x.shape
        flat = x.reshape(B, T, V * D)                       # (B, T, V*D)
        weights = self.softmax(self.gate(flat))              # (B, T, V)

        var_outputs = []
        for i in range(self.n_vars):
            var_outputs.append(self.var_grns[i](x[:, :, i, :]))  # (B, T, D)
        var_outputs = torch.stack(var_outputs, dim=2)        # (B, T, V, D)

        combined = (weights.unsqueeze(-1) * var_outputs).sum(dim=2)  # (B, T, D)
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention that shares value weights across heads for interpretability."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)  # shared value
        self.out_proj = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.shape

        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, dk)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v)  # (B, T, dk)  — shared

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))       # (B, H, T, T)

        # Apply attention to shared values
        V_exp = V.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, H, T, dk)
        context = torch.matmul(attn, V_exp).mean(dim=1)          # (B, T, dk)
        out = self.out_proj(context)
        return out, attn.mean(dim=1)  # average attention across heads


# ═══════════════════════════════════════════════════════════════════════
#  Full TFT Model
# ═══════════════════════════════════════════════════════════════════════

class TemporalFusionTransformer(pl.LightningModule):
    """
    Simplified TFT for renewable-energy probabilistic forecasting.

    Input : (B, seq_len, n_features)
    Output: (B, horizon, n_targets, n_quantiles)
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int = 24,
        hidden_size: int = 160,
        n_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 5e-4,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.quantiles = quantiles or CFG["models"]["tft"]["quantiles"]
        self.n_targets = n_targets
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.n_quantiles = len(self.quantiles)

        d = hidden_size

        # ── Input embedding (project each feature into d_model) ──────
        self.input_proj = nn.Linear(n_features, n_features * d)
        self.n_input_vars = n_features

        # ── Variable Selection ───────────────────────────────────────
        self.vsn = VariableSelectionNetwork(n_features, d, d, dropout)

        # ── Local processing (LSTM encoder) ──────────────────────────
        self.lstm_encoder = nn.LSTM(d, d, num_layers=2, dropout=dropout, batch_first=True)

        # ── Temporal self-attention ──────────────────────────────────
        self.attention = InterpretableMultiHeadAttention(d, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn_grn = GatedResidualNetwork(d, d, dropout)

        # ── Position-wise feed-forward ───────────────────────────────
        self.ff_grn = GatedResidualNetwork(d, d, dropout)
        self.ff_norm = nn.LayerNorm(d)

        # ── Quantile output ──────────────────────────────────────────
        self.output_proj = nn.Linear(d, horizon * n_targets * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        d = self.hparams.hidden_size

        # Embed each feature to d_model
        embedded = self.input_proj(x).view(B, T, F, d)  # (B, T, F, d)

        # Variable selection
        selected, var_weights = self.vsn(embedded)  # (B, T, d)
        self._var_weights = var_weights  # stash for interpretability

        # LSTM encoder
        lstm_out, _ = self.lstm_encoder(selected)  # (B, T, d)

        # Self-attention + residual
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        self._attn_weights = attn_weights
        attn_out = self.attn_norm(lstm_out + attn_out)
        attn_out = self.attn_grn(attn_out)

        # Feed-forward + residual
        ff_out = self.ff_grn(attn_out)
        ff_out = self.ff_norm(attn_out + ff_out)

        # Use last time-step for multi-horizon prediction
        last = ff_out[:, -1, :]  # (B, d)
        out = self.output_proj(last)
        return out.view(B, self.horizon, self.n_targets, self.n_quantiles)

    # ── Lightning steps ──────────────────────────────────────────────
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
        median_idx = self.quantiles.index(0.50) if 0.50 in self.quantiles else len(self.quantiles) // 2
        point = pred[:, :, :, median_idx]
        rmse = torch.sqrt(((point - y) ** 2).mean())
        self.log("val_rmse", rmse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = quantile_loss(pred, y, self.quantiles)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched}}

    @torch.no_grad()
    def predict_quantiles(self, x: torch.Tensor) -> dict[float, torch.Tensor]:
        self.eval()
        pred = self(x)
        return {q: pred[:, :, :, i] for i, q in enumerate(self.quantiles)}

    def get_variable_importances(self) -> torch.Tensor:
        """Return mean variable-selection weights from last forward pass."""
        return self._var_weights.mean(dim=(0, 1))  # (n_vars,)

    def get_attention_weights(self) -> torch.Tensor:
        """Return mean temporal attention weights from last forward pass."""
        return self._attn_weights.mean(dim=0)  # (T, T)
