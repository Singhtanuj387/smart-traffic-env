"""
AdaptiveDemand — Circadian spawn schedule (6 AM peak, 9 PM quiet).
"""

from __future__ import annotations

import math
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    try:
        from ..core.grid import TrafficGrid
    except ImportError:
        from core.grid import TrafficGrid


class AdaptiveDemand(BaseScenario):
    name = "adaptive_demand"

    def __init__(self):
        self._steps_per_hour = 150  # 3600 steps = 24 hours compressed

    def reset(self, grid: "TrafficGrid") -> None:
        pass

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        # Map simulation step to hour of day (0-24)
        hour = (t / self._steps_per_hour) % 24.0

        # Circadian curve: peaks at 8AM and 5PM, trough at 3AM
        morning_peak = math.exp(-((hour - 8.0) ** 2) / 8.0)
        evening_peak = math.exp(-((hour - 17.0) ** 2) / 8.0)
        night_trough = math.exp(-((hour - 3.0) ** 2) / 12.0)

        demand = 0.3 + 1.5 * (morning_peak + evening_peak) - 0.5 * night_trough
        demand = max(0.1, min(demand, 3.0))

        for i in range(grid.n_nodes):
            update.spawn_rate_multipliers[i] = [demand] * 12

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_recovery": 0.8}

    def get_flags(self, agent_id: int) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]
