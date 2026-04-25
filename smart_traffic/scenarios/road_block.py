"""
RoadBlock — Remove 2–5 edges mid-episode; trigger A* reroute.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class RoadBlock(BaseScenario):
    name = "road_block"

    def __init__(self):
        self._blocked: List[Tuple[int, int]] = []
        self._block_start = 0
        self._block_duration = 600

    def reset(self, grid: "TrafficGrid") -> None:
        self._blocked.clear()
        grid.pathfinder.clear_blocks()
        # Schedule block between step 200 and 800
        self._block_start = random.randint(200, 800)
        n_blocks = random.randint(2, 5)
        # Pick random adjacent pairs to block
        candidates = []
        for idx in range(grid.n_nodes):
            for nb in grid.get_neighbors(idx):
                edge = (min(idx, nb), max(idx, nb))
                if edge not in candidates:
                    candidates.append(edge)
        if candidates:
            self._blocked = random.sample(candidates, min(n_blocks, len(candidates)))

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        if t == self._block_start:
            # Activate road blocks
            for u, v in self._blocked:
                grid.pathfinder.block_edge(u, v)
            update.blocked_roads = [[u, v] for u, v in self._blocked]
            update.special_events = ["road_block_activated"]

        elif t == self._block_start + self._block_duration:
            # Remove blocks
            for u, v in self._blocked:
                grid.pathfinder.unblock_edge(u, v)
            self._blocked.clear()
            update.special_events = ["road_block_cleared"]

        elif self._block_start <= t < self._block_start + self._block_duration:
            update.blocked_roads = [[u, v] for u, v in self._blocked]

        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_cascade": 2.0}

    def get_flags(self, agent_id: int) -> List[float]:
        on_block = any(agent_id in (u, v) for u, v in self._blocked)
        return [0.0, 0.0, 1.0 if on_block else 0.0, 0.0]
