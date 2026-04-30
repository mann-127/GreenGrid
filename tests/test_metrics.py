"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest

from greengrid.evaluation.metrics import (
    coverage_probability,
    curtailment_reduction,
    mae,
    mape,
    pinball_loss,
    r_squared,
    revenue_improvement,
    rmse,
)


class TestPointMetrics:
    def test_mae_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_rmse_known(self):
        a = np.array([0.0, 0.0])
        p = np.array([1.0, 1.0])
        assert rmse(a, p) == pytest.approx(1.0)

    def test_r_squared_perfect(self):
        a = np.arange(100, dtype=float)
        assert r_squared(a, a) == pytest.approx(1.0)

    def test_mape_reasonable(self):
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 180.0])
        assert 0 < mape(a, p) < 100


class TestProbabilisticMetrics:
    def test_pinball_symmetric(self):
        a = np.zeros(100)
        p = np.ones(100)
        assert pinball_loss(a, p, 0.5) == pytest.approx(0.5)

    def test_coverage_all_inside(self):
        a = np.array([5.0, 6.0, 7.0])
        lo = np.array([4.0, 5.0, 6.0])
        hi = np.array([6.0, 7.0, 8.0])
        assert coverage_probability(a, lo, hi) == pytest.approx(1.0)

    def test_coverage_none_inside(self):
        a = np.array([10.0, 20.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([5.0, 5.0])
        assert coverage_probability(a, lo, hi) == pytest.approx(0.0)


class TestGridMetrics:
    def test_curtailment_reduction(self):
        assert curtailment_reduction(100, 80) == pytest.approx(20.0)

    def test_curtailment_reduction_zero_baseline(self):
        assert curtailment_reduction(0, 10) == pytest.approx(0.0)

    def test_revenue_improvement(self):
        assert revenue_improvement(1000, 1200) == pytest.approx(20.0)
