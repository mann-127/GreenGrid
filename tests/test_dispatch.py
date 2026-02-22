"""
Tests for battery and dispatch logic.
"""

import numpy as np
import pytest

from greengrid.optimizer.battery import Battery, BatteryConfig
from greengrid.optimizer.dispatch import NaiveDispatch, ForecastDispatch


class TestBattery:
    def test_initial_soc(self):
        b = Battery(initial_soc=0.5)
        assert b.soc == pytest.approx(0.5)

    def test_charge_increases_soc(self):
        b = Battery(initial_soc=0.3)
        state = b.step(50.0)  # charge at max rate
        assert b.soc > 0.3

    def test_discharge_decreases_soc(self):
        b = Battery(initial_soc=0.7)
        state = b.step(-50.0)
        assert b.soc < 0.7

    def test_soc_clipped_to_bounds(self):
        cfg = BatteryConfig(min_soc=0.1, max_soc=0.95)
        b = Battery(cfg, initial_soc=0.94)
        # Try to overcharge
        for _ in range(10):
            b.step(50.0)
        assert b.soc <= 0.95 + 1e-6

    def test_soc_min_bound(self):
        cfg = BatteryConfig(min_soc=0.1, max_soc=0.95)
        b = Battery(cfg, initial_soc=0.12)
        for _ in range(10):
            b.step(-50.0)
        assert b.soc >= 0.1 - 1e-6

    def test_history_tracked(self):
        b = Battery(initial_soc=0.5)
        b.step(10)
        b.step(-10)
        assert len(b.history) == 2

    def test_reset(self):
        b = Battery(initial_soc=0.5)
        b.step(10)
        b.reset(0.6)
        assert b.soc == pytest.approx(0.6)
        assert len(b.history) == 0


class TestDispatch:
    def test_naive_dispatch_runs(self):
        ren = np.full(24, 100.0)
        dem = np.full(24, 80.0)
        price = np.full(24, 50.0)
        d = NaiveDispatch()
        result = d.run(ren, dem, price)
        assert result.total_revenue > 0
        assert len(result.decisions) == 24

    def test_forecast_dispatch_less_curtailment(self):
        """Forecast dispatch should waste less energy than naïve."""
        rng = np.random.default_rng(42)
        ren = 100 + 50 * np.sin(2 * np.pi * np.arange(24) / 24) + rng.normal(0, 5, 24)
        ren = np.clip(ren, 0, 200)
        dem = np.full(24, 60.0)
        price = np.array([45 if h not in {7,8,17,18} else 120 for h in range(24)], dtype=float)

        naive = NaiveDispatch()
        naive_res = naive.run(ren, dem, price)

        smart = ForecastDispatch()
        smart_res = smart.run(ren, ren * 0.9, ren * 1.1, dem, price)

        # Smart should have ≤ curtailment
        assert smart_res.total_curtailment_mwh <= naive_res.total_curtailment_mwh + 1e-3
    
    def test_forecast_dispatch_degradation_awareness(self):
        """Ensure battery won't discharge if spot price is lower than the degradation penalty."""
        ren = np.zeros(24)       # 0 renewable generation (forcing a grid deficit)
        dem = np.full(24, 50.0)  # Constant demand
        price = np.full(24, 5.0) # Price is very low ($5/MWh)
        
        batt_cfg = BatteryConfig(min_soc=0.1, max_soc=0.9, capacity_mwh=100)
        # Custom config: Penalty is $15/MWh, no peak hours to force overrides
        custom_cfg = {"optimizer": {"cycle_penalty_cost_mwh": 15.0, "peak_hours": []}}
        
        smart = ForecastDispatch(batt_cfg, cfg=custom_cfg)
        res = smart.run(ren, ren, ren, dem, price)
        
        # Because the price ($5) is lower than the degradation penalty ($15), 
        # the AI should refuse to discharge the battery to save its lifespan.
        assert res.total_discharged_mwh == 0.0
