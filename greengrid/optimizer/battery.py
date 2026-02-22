"""
Battery Energy Storage System (BESS) Controller
=================================================
Simulates the physical state of a lithium-ion battery pack and enforces
operational constraints (SoC limits, charge/discharge rates, efficiency).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from greengrid.settings import CFG


@dataclass
class BatteryState:
    """Snapshot of battery at a single time-step."""
    soc: float              # state-of-charge [0, 1]
    energy_mwh: float       # absolute stored energy
    charge_mw: float        # positive = charging
    discharge_mw: float     # positive = discharging
    cycle_count: float      # cumulative fractional cycles


@dataclass
class BatteryConfig:
    capacity_mwh: float = 200.0
    max_charge_mw: float = 50.0
    max_discharge_mw: float = 50.0
    efficiency: float = 0.88          # round-trip
    min_soc: float = 0.10
    max_soc: float = 0.95
    degradation_per_cycle: float = 0.0001
    calendar_aging_per_hour: float = 0.000002  # NEW: Time-based degradation

    @classmethod
    def from_cfg(cls, cfg: dict | None = None) -> BatteryConfig:
        b = (cfg or CFG)["data_generation"]["battery"]
        return cls(
            capacity_mwh=b["capacity_mwh"],
            max_charge_mw=b["max_charge_rate_mw"],
            max_discharge_mw=b["max_discharge_rate_mw"],
            efficiency=b["round_trip_efficiency"],
            min_soc=b["min_soc"],
            max_soc=b["max_soc"],
            degradation_per_cycle=(cfg or CFG)["optimizer"]["storage_degradation_per_cycle"],
            # Fetch calendar aging if it exists in config, otherwise use default
            calendar_aging_per_hour=(cfg or CFG).get("optimizer", {}).get("calendar_aging_per_hour", 0.000002),
        )


class Battery:
    """Stateful battery simulator."""

    def __init__(self, config: BatteryConfig | None = None, initial_soc: float | None = None):
        self.cfg = config or BatteryConfig.from_cfg()
        self.soc = initial_soc if initial_soc is not None else 0.5
        self.cycle_count = 0.0
        self.total_hours_operated = 0.0  # NEW: Track total time passed
        self.history: list[BatteryState] = []

    @property
    def energy_mwh(self) -> float:
        return self.soc * self.cfg.capacity_mwh

    @property
    def effective_capacity(self) -> float:
        """Capacity degrades with both cycling and calendar aging."""
        cycle_degradation = self.cfg.degradation_per_cycle * self.cycle_count
        calendar_degradation = self.cfg.calendar_aging_per_hour * self.total_hours_operated
        
        # Ensure health doesn't drop below 0 (0% capacity)
        total_health = max(0.0, 1.0 - cycle_degradation - calendar_degradation) 
        return self.cfg.capacity_mwh * total_health

    def step(self, action_mw: float, dt_hours: float = 1.0) -> BatteryState:
        """Apply a charge/discharge action for ``dt_hours`` and return new state.

        Enforces all physical constraints:
          • Rate limits: max charge/discharge rates (MW)
          • SoC bounds: min/max state-of-charge [min_soc, max_soc]
          • Round-trip efficiency: losses proportional to sqrt(efficiency)
          • Degradation: tracks cycle count for capacity fade and time for calendar aging

        Args:
            action_mw: Power request in MW.
                Positive = charge (from renewables/grid).
                Negative = discharge (to grid/demand).
            dt_hours: Time-step duration in hours (default 1.0).

        Returns:
            BatteryState after action with all values clipped to constraints.
        """
        # NEW: Increment the total operational time for calendar aging
        self.total_hours_operated += dt_hours 
        
        cap = self.effective_capacity
        soc_before = self.soc

        # Clip to rate limits first
        action_mw_clipped = np.clip(action_mw, -self.cfg.max_discharge_mw, self.cfg.max_charge_mw)
        if abs(action_mw_clipped - action_mw) > 1e-6:
            logger.debug(
                f"[battery] Rate limit clipping: requested={action_mw:.2f}MW, "
                f"clipped={action_mw_clipped:.2f}MW"
            )

        if action_mw_clipped >= 0:
            # Charging: account for round-trip efficiency loss
            energy_in = action_mw_clipped * dt_hours * np.sqrt(self.cfg.efficiency)
            max_energy_in = (self.cfg.max_soc - self.soc) * cap
            energy_in = min(energy_in, max(max_energy_in, 0))
            actual_charge = energy_in / (np.sqrt(self.cfg.efficiency) * dt_hours) if dt_hours > 0 else 0
            self.soc += energy_in / cap
            charge_mw, discharge_mw = actual_charge, 0.0
        else:
            # Discharging
            energy_out = abs(action_mw_clipped) * dt_hours * np.sqrt(self.cfg.efficiency)
            max_energy_out = (self.soc - self.cfg.min_soc) * cap
            energy_out = min(energy_out, max(max_energy_out, 0))
            actual_discharge = energy_out / (np.sqrt(self.cfg.efficiency) * dt_hours) if dt_hours > 0 else 0
            self.soc -= energy_out / cap
            charge_mw, discharge_mw = 0.0, actual_discharge

        # Track degradation via cycle counting (each full cycle = 2× capacity throughput)
        energy_throughput = max(charge_mw, discharge_mw) * dt_hours
        self.cycle_count += energy_throughput / (2 * cap)

        # Enforce SoC bounds
        self.soc = np.clip(self.soc, self.cfg.min_soc, self.cfg.max_soc)
        soc_clipped = self.soc != (charge_mw - discharge_mw) * dt_hours / cap + soc_before
        if soc_clipped:
            logger.debug(f"[battery] SoC clipping: {soc_before:.2%} | {self.soc:.2%}")

        state = BatteryState(
            soc=float(self.soc),
            energy_mwh=float(self.soc * cap),
            charge_mw=float(charge_mw),
            discharge_mw=float(discharge_mw),
            cycle_count=float(self.cycle_count),
        )
        self.history.append(state)

        logger.debug(
            f"[battery] Step complete: SoC {soc_before:.1%}|{self.soc:.1%}, "
            f"charge={charge_mw:.1f}MW, discharge={discharge_mw:.1f}MW, "
            f"cycles={self.cycle_count:.2f}, hours_operated={self.total_hours_operated:.1f}"
        )
        return state

    def reset(self, soc: float = 0.5) -> None:
        """Reset battery to initial state for a new simulation.
        
        Args:
            soc: Initial state-of-charge [0, 1]. Default 50%.
        
        Clears history, cycle count, and hours operated (for a fresh run).
        """
        self.soc = np.clip(soc, self.cfg.min_soc, self.cfg.max_soc)
        self.cycle_count = 0.0
        self.total_hours_operated = 0.0  # NEW: Reset time
        self.history.clear()
        logger.debug(f"[battery] Reset: SoC={self.soc:.1%}, history cleared, cycles zeroed")
