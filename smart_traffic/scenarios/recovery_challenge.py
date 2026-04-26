"""
RecoveryChallenge — Start at 90% queue capacity; measure steps to restore normal flow.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate
try:
    from ..core.vehicle import Vehicle, VehicleType
except ImportError:
    from core.vehicle import Vehicle, VehicleType

if TYPE_CHECKING:
    try:
        from ..core.grid import TrafficGrid
    except ImportError:
        from core.grid import TrafficGrid


class RecoveryChallenge(BaseScenario):
    name = "recovery_challenge"

    def __init__(self):
        self._initial_fill = 0.9  # 90% capacity

    def reset(self, grid: "TrafficGrid") -> None:
        """Pre-fill all queues to 90% capacity."""
        try:
            from ..core.intersection import MAX_QUEUE_PER_LANE
        except ImportError:
            from core.intersection import MAX_QUEUE_PER_LANE

        target = int(MAX_QUEUE_PER_LANE * self._initial_fill)
        vid = 100000  # offset to avoid ID collisions

        for ix in grid.intersections:
            for lane_idx in range(12):
                for _ in range(target):
                    # Simple vehicle with no route (stays put until cleared)
                    dest = random.choice(list(grid._edge_nodes))
                    route = grid.pathfinder.find_path(ix.index, dest)
                    if route is None:
                        route = [ix.index]

                    veh = Vehicle(
                        vehicle_id=vid,
                        vehicle_type=VehicleType.CAR,
                        route=route,
                        lane=lane_idx,
                    )
                    vid += 1
                    ix.enqueue(veh, lane_idx)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)
        # Low spawn rate — challenge is to clear existing backlog
        for i in range(grid.n_nodes):
            update.spawn_rate_multipliers[i] = [0.3] * 12
        update.special_events = ["recovery_challenge"]
        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_recovery": 2.0, "w_cascade": 1.8}

    def get_flags(self, agent_id: int) -> List[float]:
        return [1.0, 0.0, 0.0, 0.0]
