"""
NetworkPartition — Disconnect subgraph (bridge collapse); dead-end handling.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class NetworkPartition(BaseScenario):
    name = "network_partition"

    def __init__(self):
        self._partition_edges: List[Tuple[int, int]] = []
        self._partition_step = 0
        self._active = False

    def reset(self, grid: "TrafficGrid") -> None:
        grid.pathfinder.clear_blocks()
        self._partition_edges.clear()
        self._active = False
        self._partition_step = random.randint(100, 500)

        # Create a partition by cutting a vertical or horizontal line
        g = grid.grid_size
        if random.random() < 0.5:
            # Vertical cut at random column
            cut_col = random.randint(2, g - 3)
            for r in range(g):
                left = r * g + cut_col
                right = r * g + cut_col + 1
                # Leave 1-2 bridge connections
                if r not in (0, g - 1):
                    self._partition_edges.append((left, right))
        else:
            # Horizontal cut
            cut_row = random.randint(2, g - 3)
            for c in range(g):
                top = cut_row * g + c
                bottom = (cut_row + 1) * g + c
                if c not in (0, g - 1):
                    self._partition_edges.append((top, bottom))

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        if t == self._partition_step:
            self._active = True
            for u, v in self._partition_edges:
                grid.pathfinder.block_edge(u, v)
            update.special_events = ["network_partition_active"]

        if self._active:
            update.blocked_roads = [[u, v] for u, v in self._partition_edges]

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_coord": 0.4, "w_cascade": 2.0}

    def get_flags(self, agent_id: int) -> List[float]:
        on_partition = any(agent_id in (u, v) for u, v in self._partition_edges)
        return [0.0, 0.0, 1.0 if on_partition else 0.0, 1.0 if self._active else 0.0]
