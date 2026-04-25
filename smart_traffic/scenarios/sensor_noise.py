"""
SensorNoise — 15% of queue readings are stale/noisy; partial observability test.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class SensorNoise(BaseScenario):
    name = "sensor_noise"

    def __init__(self):
        self._noise_prob = 0.15
        self._stale_cache: Dict[int, List[float]] = {}

    def reset(self, grid: "TrafficGrid") -> None:
        self._stale_cache.clear()

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)
        update.special_events = ["sensor_noise_active"]
        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_stability": 0.5}

    def get_flags(self, agent_id: int) -> List[float]:
        # Flag indicates sensor reliability at this intersection
        noisy = random.random() < self._noise_prob
        return [0.0, 0.0, 0.0, 1.0 if noisy else 0.0]
