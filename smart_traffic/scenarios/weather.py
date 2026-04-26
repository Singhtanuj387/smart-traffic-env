"""
AdverseWeather — speed_multiplier 0.6; spawn_rate 0.7; min_green bumped to 15.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    try:
        from ..core.grid import TrafficGrid
    except ImportError:
        from core.grid import TrafficGrid


class AdverseWeather(BaseScenario):
    name = "weather"

    def __init__(self):
        self._severity: float = 0.0  # 0-1 scale
        self._onset_step = 0
        self._duration = 1200

    def reset(self, grid: "TrafficGrid") -> None:
        self._severity = random.uniform(0.3, 1.0)
        self._onset_step = random.randint(0, 600)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        if t < self._onset_step or t > self._onset_step + self._duration:
            return update

        # Ramp up severity over first 100 steps
        elapsed = t - self._onset_step
        ramp = min(elapsed / 100.0, 1.0)
        current_severity = self._severity * ramp

        speed = max(0.3, 1.0 - 0.4 * current_severity)  # min 0.3
        spawn = max(0.4, 1.0 - 0.3 * current_severity)

        update.speed_multipliers = [speed] * grid.n_nodes
        update.spawn_rate_multipliers = [
            [spawn] * 12 for _ in range(grid.n_nodes)
        ]
        update.special_events = [f"weather_severity_{current_severity:.2f}"]

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_stability": 0.8, "w_throughput": 1.2}

    def get_flags(self, agent_id: int) -> List[float]:
        return [0.0, 0.0, 0.0, self._severity]
