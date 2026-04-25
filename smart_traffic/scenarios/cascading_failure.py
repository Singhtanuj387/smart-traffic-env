"""
CascadingFailure — 4× overload epicenter t<50; spillover multiplicatively.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class CascadingFailure(BaseScenario):
    name = "cascading_failure"

    def __init__(self):
        self._epicenter = 40
        self._cascade_radius = 0

    def reset(self, grid: "TrafficGrid") -> None:
        g = grid.grid_size
        self._epicenter = random.randint(g + 1, grid.n_nodes - g - 2)
        self._cascade_radius = 0

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        if t < 50:
            # Overload epicenter
            update.spawn_rate_multipliers[self._epicenter] = [4.0] * 12
            update.special_events = ["cascade_overload"]
        else:
            # Check if epicenter is congested — spread to neighbors
            epi_q = grid.get_queue(self._epicenter)
            if sum(epi_q) / (12 * 30) > 0.7:
                self._cascade_radius = min(self._cascade_radius + 1, 4)

            # Apply multiplicative spillover within radius
            g = grid.grid_size
            epi_r, epi_c = divmod(self._epicenter, g)

            for idx in range(grid.n_nodes):
                r, c = divmod(idx, g)
                dist = abs(r - epi_r) + abs(c - epi_c)

                if dist <= self._cascade_radius and dist > 0:
                    multiplier = max(1.0, 3.0 - dist * 0.5)
                    update.spawn_rate_multipliers[idx] = [multiplier] * 12

            if self._cascade_radius > 0:
                update.special_events = [f"cascade_radius_{self._cascade_radius}"]

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_cascade": 2.5, "w_recovery": 1.5}

    def get_flags(self, agent_id: int) -> List[float]:
        is_epi = 1.0 if agent_id == self._epicenter else 0.0
        return [is_epi, 0.0, 0.0, float(self._cascade_radius) / 4.0]
