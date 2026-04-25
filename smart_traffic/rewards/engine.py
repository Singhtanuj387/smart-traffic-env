"""
Reward Engine — combines 9 components with dynamic per-scenario weight overrides.

Fully decoupled from openenv: takes grid state, returns per-agent reward vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .components import (
    R_throughput,
    R_queue,
    R_wait,
    R_phase_stability,
    R_coordination,
    R_emergency,
    R_fairness,
    R_recovery,
    R_cascade_prevention,
)
from .shaping import shaped_reward

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_throughput": 2.0,
    "w_queue": 1.5,
    "w_wait": 1.0,
    "w_stability": 0.4,
    "w_coord": 0.8,
    "w_emergency": 3.0,
    "w_fairness": 0.3,
    "w_recovery": 0.5,
    "w_cascade": 1.2,
}


class RewardEngine:
    """Composable reward engine with 9 components and dynamic weight overrides."""

    def __init__(self):
        self._prev_queues_history: List[List[List[float]]] = []

    def reset(self) -> None:
        self._prev_queues_history.clear()

    def compute_all(
        self,
        grid: "TrafficGrid",
        prev_queues: List[List[float]],
        actions: list,
        scenario_update,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """Compute per-agent rewards for all 81 agents."""
        w = {**DEFAULT_WEIGHTS, **(weight_overrides or {})}
        rewards: List[float] = []

        for i in range(81):
            r = 0.0
            r += w["w_throughput"] * R_throughput(grid, i)
            r += w["w_queue"] * R_queue(grid.get_queue(i), prev_queues[i])
            r += w["w_wait"] * R_wait(grid.get_avg_wait(i))
            r += w["w_stability"] * R_phase_stability(
                actions[i].phase if hasattr(actions[i], "phase") else "ALL_RED_HOLD",
                grid,
                i,
            )
            r += w["w_coord"] * R_coordination(grid, i)
            r += w["w_emergency"] * R_emergency(scenario_update, grid, i)
            r += w["w_fairness"] * R_fairness(grid.get_queue(i))
            r += w["w_recovery"] * R_recovery(grid, i, self._prev_queues_history)
            r += w["w_cascade"] * R_cascade_prevention(grid, i)

            # Apply potential-based shaping
            r = shaped_reward(
                r,
                prev_state=prev_queues[i],
                next_state=grid.get_queue(i),
            )
            rewards.append(r)

        self._update_history(grid)
        return rewards

    def _update_history(self, grid: "TrafficGrid") -> None:
        self._prev_queues_history.append(grid.get_all_queues())
        if len(self._prev_queues_history) > 10:
            self._prev_queues_history.pop(0)
