"""
MultiIncident — Combine 2–3 scenarios simultaneously; priority arbitration.
"""

from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, ScenarioUpdate
from .rush_hour import RushHourWave
from .emergency import EmergencyPriority
from .road_block import RoadBlock
from .weather import AdverseWeather
from .event_spike import EventSpike
from .cascading_failure import CascadingFailure

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class MultiIncident(BaseScenario):
    name = "multi_incident"

    def __init__(self):
        self._sub_scenarios: List[BaseScenario] = []

    def reset(self, grid: "TrafficGrid") -> None:
        # Pick 2-3 random scenarios to combine
        pool = [
            RushHourWave(),
            EmergencyPriority(),
            RoadBlock(),
            AdverseWeather(),
            EventSpike(),
            CascadingFailure(),
        ]
        n = random.randint(2, 3)
        self._sub_scenarios = random.sample(pool, n)
        for sc in self._sub_scenarios:
            sc.reset(grid)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        combined = ScenarioUpdate.default(grid.n_nodes)

        for sc in self._sub_scenarios:
            sub = sc.step(t, grid)
            # Merge: multiply spawn rates, min speed, union blocks/EVs
            for i in range(grid.n_nodes):
                for j in range(12):
                    combined.spawn_rate_multipliers[i][j] *= (
                        sub.spawn_rate_multipliers[i][j]
                        if i < len(sub.spawn_rate_multipliers)
                        and j < len(sub.spawn_rate_multipliers[i])
                        else 1.0
                    )
                if i < len(sub.speed_multipliers):
                    combined.speed_multipliers[i] = min(
                        combined.speed_multipliers[i], sub.speed_multipliers[i]
                    )
            combined.blocked_roads.extend(sub.blocked_roads)
            combined.emergency_vehicles.extend(sub.emergency_vehicles)
            combined.special_events.extend(sub.special_events)

        return combined

    def get_reward_overrides(self) -> Dict[str, float]:
        merged: Dict[str, float] = {"w_emergency": 4.0, "w_cascade": 2.0}
        for sc in self._sub_scenarios:
            for k, v in sc.get_reward_overrides().items():
                merged[k] = max(merged.get(k, 0.0), v)
        return merged

    def get_flags(self, agent_id: int) -> List[float]:
        flags = [0.0, 0.0, 0.0, 0.0]
        for sc in self._sub_scenarios:
            sf = sc.get_flags(agent_id)
            flags = [max(f, s) for f, s in zip(flags, sf)]
        return flags
