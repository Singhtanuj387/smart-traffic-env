"""
VehicleMix — 25% trucks (3× clear time), 5% bikes (separate lane model).
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class VehicleMix(BaseScenario):
    name = "vehicle_mix"

    def reset(self, grid: "TrafficGrid") -> None:
        pass  # Vehicle type distribution is applied via spawn_vehicles

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)
        # Slightly reduce speed to model truck-heavy traffic
        update.speed_multipliers = [0.85] * grid.n_nodes
        update.special_events = ["vehicle_mix_active"]
        return update

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_stability": 0.7, "w_throughput": 1.3}

    def get_flags(self, agent_id: int) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]
