"""
EventSpike — Burst 15,000 cars from stadium node over 20 steps at t=900.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class EventSpike(BaseScenario):
    name = "event_spike"

    def __init__(self):
        self._epicenter = 40  # center of grid
        self._spike_start = 900
        self._spike_duration = 20
        self._cars_per_step = 750  # 15000 / 20

    def reset(self, grid: "TrafficGrid") -> None:
        g = grid.grid_size
        # Pick a random interior node as the stadium
        self._epicenter = random.randint(g + 1, grid.n_nodes - g - 2)
        self._spike_start = random.randint(600, 1200)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        if self._spike_start <= t < self._spike_start + self._spike_duration:
            # Massive spawn at epicenter and neighbors
            neighbors = grid.get_neighbors(self._epicenter)
            affected = [self._epicenter] + neighbors

            for node in affected:
                multiplier = 20.0 if node == self._epicenter else 10.0
                update.spawn_rate_multipliers[node] = [multiplier] * 12

            update.special_events = ["event_spike_active"]

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_cascade": 1.8, "w_coord": 1.4}

    def get_flags(self, agent_id: int) -> List[float]:
        is_epi = 1.0 if agent_id == self._epicenter else 0.0
        return [0.0, 0.0, is_epi, 0.0]
