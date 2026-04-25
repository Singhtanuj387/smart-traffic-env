"""
PedestrianSurge — 30s mandatory hold at random intersections.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class PedestrianSurge(BaseScenario):
    name = "pedestrian_surge"

    def __init__(self):
        self._affected: List[int] = []
        self._hold_start: Dict[int, int] = {}
        self._hold_duration = 30

    def reset(self, grid: "TrafficGrid") -> None:
        # Pick 5-10 random intersections for pedestrian crossings
        n = random.randint(5, 10)
        self._affected = random.sample(range(grid.n_nodes), n)
        self._hold_start = {}
        for a in self._affected:
            self._hold_start[a] = random.randint(200, 3000)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        for node in self._affected:
            start = self._hold_start[node]
            if start <= t < start + self._hold_duration:
                # Force ALL_RED at these intersections (via reduced speed)
                update.speed_multipliers[node] = 0.0  # zero = no movement
                update.special_events.append(f"ped_hold_{node}")

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_fairness": 0.9, "w_stability": 0.6}

    def get_flags(self, agent_id: int) -> List[float]:
        return [0.0, 0.0, 0.0, 1.0 if agent_id in self._affected else 0.0]
