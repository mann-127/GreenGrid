"""
Tests for the preprocessing pipeline.
"""

import numpy as np
import pytest

from greengrid.data.generator import generate_dataset
from greengrid.data.preprocessing import prepare_data, TimeSeriesDataset, build_dataloaders
from greengrid.settings import CFG


@pytest.fixture
def small_data():
    cfg = CFG.copy()
    cfg["data_generation"] = {
        **cfg["data_generation"],
        "time_range": {"start": "2023-01-01", "end": "2023-06-30", "freq": "1h"},
    }
    df = generate_dataset(cfg)
    return prepare_data(df, cfg)


class TestPreprocessing:
    def test_splits_non_empty(self, small_data):
        assert len(small_data.train.X) > 0
        assert len(small_data.val.X) > 0
        assert len(small_data.test.X) > 0

    def test_sequence_shape(self, small_data):
        seq_len = CFG["preprocessing"]["sequence_length"]
        horizon = CFG["preprocessing"]["forecast_horizon"]
        n_feat = len(CFG["preprocessing"]["feature_columns"])
        n_tgt = len(CFG["preprocessing"]["target_columns"])

        assert small_data.train.X.shape[1] == seq_len
        assert small_data.train.X.shape[2] == n_feat
        assert small_data.train.y.shape[1] == horizon
        assert small_data.train.y.shape[2] == n_tgt

    def test_no_nans(self, small_data):
        assert not np.isnan(small_data.train.X).any()
        assert not np.isnan(small_data.train.y).any()

    def test_scaler_fitted(self, small_data):
        # StandardScaler should have non-zero std
        assert all(s > 0 for s in small_data.feature_scaler.scale_)

    def test_dataset_len(self, small_data):
        ds = TimeSeriesDataset(small_data.train)
        assert len(ds) == len(small_data.train.X)

    def test_dataloaders(self, small_data):
        tl, vl, tel = build_dataloaders(small_data, batch_size=16)
        batch = next(iter(tl))
        assert len(batch) == 2
        assert batch[0].shape[0] <= 16
