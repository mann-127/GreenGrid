"""
Grid Dispatch optimizer
========================
Decides hour-by-hour what to do with renewable energy production:

  1. **Meet demand first** — sell to grid at market price.
  2. **Store surplus** — charge battery when production > demand + sell.
  3. **Discharge strategically** — sell stored energy during peak prices.
  4. **Minimise curtailment** — avoid wasting any renewable energy.

Two strategies:
  • ``NaiveDispatch``   — simple rule-based baseline.
  • ``ForecastDispatch`` — uses the 24-h probabilistic forecast to plan
    ahead via a greedy look-ahead heuristic.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from greengrid.optimizer.battery import Battery, BatteryConfig
from greengrid.settings import CFG

# Minimum discharge score threshold; hours below this are skipped even if
# flagged as discharge slots, because the effective price is too low to justify
# battery wear.  Expressed in the same dimensionless score units as
# discharge_score (effective_price / price_max * 100 - surplus_lower).
_MIN_DISCHARGE_SCORE = -50.0

# Fraction of horizon hours pre-allocated for charging / discharging in the
# greedy look-ahead.  1/3 charge, 1/4 discharge chosen empirically to leave
# the remaining hours as neutral "sell surplus" hours.
_CHARGE_HOUR_FRACTION = 3
_DISCHARGE_HOUR_FRACTION = 4

# Weight applied to forecast confidence when scoring charge hours.
_CONFIDENCE_CHARGE_WEIGHT = 10.0
_CONFIDENCE_DISCHARGE_WEIGHT = 5.0


@dataclass
class DispatchDecision:
    """Single-hour dispatch outcome."""

    hour: int
    renewable_mw: float
    demand_mw: float
    price_mwh: float
    sold_mw: float
    charged_mw: float
    discharged_mw: float
    curtailed_mw: float
    battery_soc: float
    revenue: float


@dataclass
class DispatchResult:
    """Full simulation result over the dispatch horizon."""

    decisions: list[DispatchDecision]
    total_revenue: float
    total_curtailment_mwh: float
    total_sold_mwh: float
    total_charged_mwh: float
    total_discharged_mwh: float
    avg_soc: float


# ─────────────────────────────────────────────────────────────────────
#  Strategy 1: Naïve rule-based dispatch (baseline)
# ─────────────────────────────────────────────────────────────────────


class NaiveDispatch:
    """
    Simple greedy rule-based dispatch strategy (baseline).

    Decision logic per hour:
      1. Renewable generation > demand:
         - Sell surplus to grid
         - Charge battery with any excess
         - Curtail if battery is full
      2. Renewable < demand (deficit):
         - Sell all renewables to grid
         - Discharge battery to meet remaining demand
         - Cannot discharge below SoC min

    This strategy has no look-ahead and ignores electricity prices,
    making it a conservative baseline for comparison.
    """

    def __init__(self, battery_cfg: BatteryConfig | None = None):
        self.battery = Battery(battery_cfg)
        logger.debug("[dispatch] Initialized NaiveDispatch strategy")

    def run(
        self,
        renewable_mw: np.ndarray,  # (H,) renewable generation
        demand_mw: np.ndarray,  # (H,) electricity demand
        price_mwh: np.ndarray,  # (H,) electricity price $/MWh
    ) -> DispatchResult:
        """Execute naive dispatch strategy over horizon.

        Args:
            renewable_mw: Renewable generation (MW) for each hour.
            demand_mw: Grid demand (MW) for each hour.
            price_mwh: Electricity price ($/MWh) for each hour (unused).

        Returns:
            DispatchResult with decisions and metrics.
        """
        logger.info(
            f"[dispatch] Running NaiveDispatch: horizon={len(renewable_mw)}h, "
            f"total_gen={renewable_mw.sum():.1f}MWh, "
            f"total_demand={demand_mw.sum():.1f}MWh"
        )

        self.battery.reset()
        decisions = []

        for h in range(len(renewable_mw)):
            ren = float(renewable_mw[h])
            dem = float(demand_mw[h])
            price = float(price_mwh[h])

            surplus = ren - dem
            curtailed = 0.0
            sold = 0.0
            charged = 0.0
            discharged = 0.0

            if surplus > 0:
                # Try to sell what the grid needs, store the rest
                sold = min(surplus, dem)
                remainder = surplus - sold

                # Charge battery with remainder
                state = self.battery.step(remainder)
                charged = state.charge_mw
                curtailed = max(0, remainder - charged)
            else:
                # Deficit — sell all renewables, discharge battery for the gap
                sold = ren
                deficit = abs(surplus)
                state = self.battery.step(-deficit)
                discharged = state.discharge_mw

            revenue = sold * price + discharged * price

            decisions.append(
                DispatchDecision(
                    hour=h,
                    renewable_mw=ren,
                    demand_mw=dem,
                    price_mwh=price,
                    sold_mw=sold,
                    charged_mw=charged,
                    discharged_mw=discharged,
                    curtailed_mw=curtailed,
                    battery_soc=self.battery.soc,
                    revenue=revenue,
                )
            )

        return _aggregate(decisions)


# ─────────────────────────────────────────────────────────────────────
#  Strategy 2: Forecast-aware dispatch (AI-driven)
# ─────────────────────────────────────────────────────────────────────


class ForecastDispatch:
    """
    Uses the probabilistic 24-h forecast to optimise storage dispatch:
      1. Score each hour by expected price × surplus probability.
      2. Prioritise charging during cheap / high-surplus hours.
      3. Prioritise discharging during expensive / low-surplus hours.
      4. Use the lower confidence bound (P05) for conservative
         surplus estimation reduces curtailment.
    """

    def __init__(self, battery_cfg: BatteryConfig | None = None, cfg: dict | None = None):
        self.battery = Battery(battery_cfg)
        self.cfg = cfg or CFG
        self.peak_hours = set(self.cfg["optimizer"]["peak_hours"])
        # The financial cost to physically degrade the battery by 1 MWh of cycling
        optimizer_cfg = self.cfg.get("optimizer", {})
        self.cycle_penalty_cost_mwh = optimizer_cfg.get("cycle_penalty_cost_mwh", 15.0)

    def run(
        self,
        renewable_forecast_median: np.ndarray,  # (H,) — P50
        renewable_forecast_lower: np.ndarray,  # (H,) — P05
        renewable_forecast_upper: np.ndarray,  # (H,) — P95
        demand_mw: np.ndarray,
        price_mwh: np.ndarray,
    ) -> DispatchResult:
        self.battery.reset()
        H = len(renewable_forecast_median)
        decisions = []

        # ── Pre-compute priority scores ──────────────────────────────
        # Forecast confidence: (P95 - P05) / median  ==> wide band = low confidence
        forecast_band = renewable_forecast_upper - renewable_forecast_lower
        forecast_confidence = 1.0 / (1.0 + forecast_band / (renewable_forecast_median + 1))

        # "Charge score" = (high surplus probability + low price) + confidence bonus
        surplus_upper = renewable_forecast_upper - demand_mw
        base_charge = surplus_upper - price_mwh / price_mwh.max() * surplus_upper.max()
        charge_score = base_charge + _CONFIDENCE_CHARGE_WEIGHT * forecast_confidence

        # "Discharge score" = (high price + low surplus probability) + confidence bonus
        surplus_lower = renewable_forecast_lower - demand_mw

        # Effective price factors in the physical degradation cost of cycling
        effective_price = price_mwh - self.cycle_penalty_cost_mwh

        # If effective price is negative, penalise the discharge score
        base_discharge = np.where(
            effective_price > 0,
            effective_price / price_mwh.max() * 100 - surplus_lower,
            -100,
        )
        discharge_score = base_discharge + _CONFIDENCE_DISCHARGE_WEIGHT * forecast_confidence

        # Rank hours
        charge_priority = np.argsort(-charge_score)  # best charge hours first
        discharge_priority = np.argsort(-discharge_score)

        # Pre-allocate actions
        actions = np.zeros(H)
        # Mark top charge hours
        n_charge = max(1, H // _CHARGE_HOUR_FRACTION)
        for idx in charge_priority[:n_charge]:
            actions[idx] = 1  # want to charge

        n_discharge = max(1, H // _DISCHARGE_HOUR_FRACTION)
        for idx in discharge_priority[:n_discharge]:
            # Only tag for discharge if score clears the minimum threshold and
            # doesn't override a planned charge hour.
            if actions[idx] == 0 and discharge_score[idx] > _MIN_DISCHARGE_SCORE:
                actions[idx] = -1  # want to discharge

        # ── Execute dispatch ─────────────────────────────────────────
        for h in range(H):
            # Use conservative (P05) estimate to avoid over-promising
            ren = float(renewable_forecast_lower[h])
            dem = float(demand_mw[h])
            price = float(price_mwh[h])

            surplus = ren - dem
            curtailed = 0.0
            sold = 0.0
            charged = 0.0
            discharged = 0.0

            if surplus > 0:
                sold = min(surplus, dem)
                remainder = surplus - sold

                if actions[h] >= 0:
                    # Charge battery
                    state = self.battery.step(remainder)
                    charged = state.charge_mw
                    curtailed = max(0, remainder - charged)
                else:
                    # We planned to discharge but have surplus —
                    # charge a bit, sell the rest
                    state = self.battery.step(remainder * 0.5)
                    charged = state.charge_mw
                    extra_sold = remainder - charged
                    sold += extra_sold
                    curtailed = 0.0
            else:
                sold = ren
                deficit = abs(surplus)

                # Final check - do not discharge if market price is lower than
                # degradation penalty unless it is a peak hour where grid stability
                # demands it
                is_profitable_to_discharge = price >= self.cycle_penalty_cost_mwh

                if (actions[h] <= 0 and is_profitable_to_discharge) or h in self.peak_hours:
                    # Discharge battery
                    state = self.battery.step(-deficit)
                    discharged = state.discharge_mw
                else:
                    # Non-peak, planned to charge — skip discharge
                    state = self.battery.step(0)

            revenue = sold * price + discharged * price

            decisions.append(
                DispatchDecision(
                    hour=h,
                    renewable_mw=ren,
                    demand_mw=dem,
                    price_mwh=price,
                    sold_mw=sold,
                    charged_mw=charged,
                    discharged_mw=discharged,
                    curtailed_mw=curtailed,
                    battery_soc=self.battery.soc,
                    revenue=revenue,
                )
            )

        return _aggregate(decisions)


def _aggregate(decisions: list[DispatchDecision], dt_hours: float = 1.0) -> DispatchResult:
    """Summarise a list of hourly decisions.

    All MW values are multiplied by dt_hours to produce MWh totals.
    """
    total_rev = sum(d.revenue for d in decisions)
    total_curt = sum(d.curtailed_mw for d in decisions) * dt_hours
    total_sold = sum(d.sold_mw for d in decisions) * dt_hours
    total_charged = sum(d.charged_mw for d in decisions) * dt_hours
    total_discharged = sum(d.discharged_mw for d in decisions) * dt_hours
    avg_soc = np.mean([d.battery_soc for d in decisions])

    return DispatchResult(
        decisions=decisions,
        total_revenue=total_rev,
        total_curtailment_mwh=total_curt,
        total_sold_mwh=total_sold,
        total_charged_mwh=total_charged,
        total_discharged_mwh=total_discharged,
        avg_soc=float(avg_soc),
    )
