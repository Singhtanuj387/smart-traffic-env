"""
RushHourWave — 4× spawn from one grid edge, wave propagates inward over t<600.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class RushHourWave(BaseScenario):
    name = "rush_hour"

    def __init__(self):
        self._wave_edge: str = "N"  # which edge the wave comes from
        self._grid_size = 9

    def reset(self, grid: "TrafficGrid") -> None:
        self._grid_size = grid.grid_size
        self._wave_edge = random.choice(["N", "S", "E", "W"])

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        if t >= 600:
            return update

        # Determine which row/col the wave has reached
        progress = min(t / 600.0, 1.0)
        wave_depth = int(progress * self._grid_size)

        for idx in range(grid.n_nodes):
            r, c = divmod(idx, self._grid_size)
            in_wave = False

            if self._wave_edge == "N" and r <= wave_depth:
                in_wave = True
            elif self._wave_edge == "S" and r >= self._grid_size - 1 - wave_depth:
                in_wave = True
            elif self._wave_edge == "W" and c <= wave_depth:
                in_wave = True
            elif self._wave_edge == "E" and c >= self._grid_size - 1 - wave_depth:
                in_wave = True

            if in_wave:
                update.spawn_rate_multipliers[idx] = [4.0] * 12

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_coord": 1.5}

    def get_flags(self, agent_id: int) -> List[float]:
        return [1.0, 0.0, 0.0, 0.0]
