"""
Tests for forecasting models.
"""

import pytest
import torch

from greengrid.models.lstm_model import ProbabilisticLSTM, quantile_loss
from greengrid.models.tft_model import TemporalFusionTransformer


class TestQuantileLoss:
    def test_zero_loss_for_perfect(self):
        pred = torch.ones(4, 24, 2, 5)
        target = torch.ones(4, 24, 2)
        loss = quantile_loss(pred, target, [0.05, 0.25, 0.5, 0.75, 0.95])
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_loss(self):
        pred = torch.randn(4, 24, 2, 5)
        target = torch.randn(4, 24, 2)
        loss = quantile_loss(pred, target, [0.05, 0.25, 0.5, 0.75, 0.95])
        assert loss.item() > 0


class TestLSTM:
    def test_output_shape(self):
        model = ProbabilisticLSTM(
            n_features=11,
            n_targets=2,
            horizon=24,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            quantiles=[0.1, 0.5, 0.9],
        )
        x = torch.randn(4, 168, 11)
        out = model(x)
        assert out.shape == (4, 24, 2, 3)

    def test_predict_quantiles(self):
        model = ProbabilisticLSTM(
            n_features=11,
            n_targets=2,
            horizon=24,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            quantiles=[0.1, 0.5, 0.9],
        )
        x = torch.randn(2, 168, 11)
        q_dict = model.predict_quantiles(x)
        assert len(q_dict) == 3
        assert q_dict[0.5].shape == (2, 24, 2)


class TestTFT:
    def test_output_shape(self):
        model = TemporalFusionTransformer(
            n_features=11,
            n_targets=2,
            horizon=24,
            hidden_size=32,
            n_heads=2,
            dropout=0.0,
            quantiles=[0.1, 0.5, 0.9],
        )
        x = torch.randn(4, 168, 11)
        out = model(x)
        assert out.shape == (4, 24, 2, 3)

    def test_variable_importances(self):
        model = TemporalFusionTransformer(
            n_features=11,
            n_targets=2,
            horizon=24,
            hidden_size=32,
            n_heads=2,
            dropout=0.0,
            quantiles=[0.1, 0.5, 0.9],
        )
        x = torch.randn(2, 168, 11)
        _ = model(x)
        vi = model.get_variable_importances()
        assert vi.shape == (11,)
        assert torch.allclose(vi.sum(), torch.tensor(1.0), atol=0.01)
