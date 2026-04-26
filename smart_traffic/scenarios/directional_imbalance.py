"""
DirectionalImbalance — 90/10 E/W split; flip at step 1800; ±5% noise.
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


class DirectionalImbalance(BaseScenario):
    name = "directional_imbalance"

    def __init__(self):
        self._flip_step = 1800
        self._noise = 0.05

    def reset(self, grid: "TrafficGrid") -> None:
        self._flip_step = 1800

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        # Before flip: heavy E→W, light W→E
        # After flip: reverse
        flipped = t >= self._flip_step

        for idx in range(grid.n_nodes):
            mults = [1.0] * 12
            noise = random.uniform(-self._noise, self._noise)

            if not flipped:
                # E lanes (6-8) heavy, W lanes (9-11) light
                for l in range(6, 9):
                    mults[l] = 0.9 + noise  # 90%
                for l in range(9, 12):
                    mults[l] = 0.1 + noise  # 10%
            else:
                # Reversed
                for l in range(6, 9):
                    mults[l] = 0.1 + noise
                for l in range(9, 12):
                    mults[l] = 0.9 + noise

            # Normalize: N/S lanes remain at baseline
            update.spawn_rate_multipliers[idx] = [max(0.0, m) for m in mults]

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_fairness": 0.8}

    def get_flags(self, agent_id: int) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]
