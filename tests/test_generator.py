"""
Tests for the synthetic data generator.
"""

import numpy as np
import pandas as pd
import pytest

from greengrid.data.generator import generate_dataset
from greengrid.settings import CFG


@pytest.fixture
def sample_cfg():
    """Reduced config for fast tests."""
    cfg = CFG.copy()
    cfg["data_generation"] = {
        **cfg["data_generation"],
        "time_range": {
            "start": "2023-01-01",
            "end": "2023-01-31",
            "freq": "1h",
        },
    }
    return cfg


class TestDataGenerator:
    def test_generates_dataframe(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        required = [
            "timestamp", "wind_speed_ms", "wind_power_mw",
            "solar_power_mw", "ghi_wm2", "temperature_c",
            "electricity_price_mwh", "hour_sin", "hour_cos",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_wind_power_non_negative(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        assert (df["wind_power_mw"] >= 0).all()

    def test_solar_power_non_negative(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        assert (df["solar_power_mw"] >= 0).all()

    def test_wind_power_within_capacity(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        max_mw = (
            sample_cfg["data_generation"]["wind"]["num_turbines"]
            * sample_cfg["data_generation"]["wind"]["rated_capacity_mw"]
        )
        assert df["wind_power_mw"].max() <= max_mw + 1e-3

    def test_solar_zero_at_night(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        night = df[df["timestamp"].dt.hour.isin([0, 1, 2, 3])]
        # Most nighttime solar should be zero or near-zero
        assert night["solar_power_mw"].mean() < 5.0

    def test_cyclical_features_range(self, sample_cfg):
        df = generate_dataset(sample_cfg)
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_reproducibility(self, sample_cfg):
        df1 = generate_dataset(sample_cfg)
        df2 = generate_dataset(sample_cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_real_weather_integration(self, sample_cfg):
        """Test that the generator correctly calls the external weather API."""
        from unittest.mock import patch
        
        with patch("greengrid.data.generator.fetch_real_weather") as mock_fetch:
            # Create dummy API data
            mock_df = pd.DataFrame({
                "timestamp": pd.date_range("2023-01-01", periods=24, freq="h"),
                "temperature_c": np.zeros(24),
                "humidity_pct": np.zeros(24),
                "pressure_hpa": np.zeros(24),
                "wind_speed_ms": np.ones(24) * 10,
                "wind_direction_deg": np.zeros(24),
                "ghi_wm2": np.zeros(24),
                "cloud_cover_pct": np.zeros(24),
            })
            mock_fetch.return_value = mock_df
            
            df = generate_dataset(sample_cfg, use_real_weather=True)
            mock_fetch.assert_called_once()
            assert len(df) == 24
