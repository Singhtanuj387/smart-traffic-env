"""
ScenarioManager — load, schedule, and mix scenarios per episode.

Returns ScenarioUpdate objects to the environment and provides
reward weight overrides and per-agent observation flags.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, TYPE_CHECKING

from ..models import ScenarioFlags
from .base import BaseScenario, ScenarioUpdate

# Import all 14 scenarios
from .rush_hour import RushHourWave
from .emergency import EmergencyPriority
from .road_block import RoadBlock
from .weather import AdverseWeather
from .event_spike import EventSpike
from .directional_imbalance import DirectionalImbalance
from .cascading_failure import CascadingFailure
from .network_partition import NetworkPartition
from .multi_incident import MultiIncident
from .pedestrian_surge import PedestrianSurge
from .vehicle_mix import VehicleMix
from .adaptive_demand import AdaptiveDemand
from .sensor_noise import SensorNoise
from .recovery_challenge import RecoveryChallenge

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


SCENARIO_REGISTRY: Dict[str, type] = {
    "rush_hour": RushHourWave,
    "emergency": EmergencyPriority,
    "road_block": RoadBlock,
    "weather": AdverseWeather,
    "event_spike": EventSpike,
    "directional_imbalance": DirectionalImbalance,
    "cascading_failure": CascadingFailure,
    "network_partition": NetworkPartition,
    "multi_incident": MultiIncident,
    "pedestrian_surge": PedestrianSurge,
    "vehicle_mix": VehicleMix,
    "adaptive_demand": AdaptiveDemand,
    "sensor_noise": SensorNoise,
    "recovery_challenge": RecoveryChallenge,
}


class ScenarioManager:
    """
    Manages scenario lifecycle: reset, step, reward overrides, flags.
    Supports single scenario selection or random per episode.
    """

    def __init__(self):
        self._active_scenarios: List[BaseScenario] = []

    def reset(
        self,
        grid: "TrafficGrid",
        scenario: Optional[str] = None,
    ) -> None:
        """
        Initialize scenario(s) for a new episode.
        - scenario=None → random single scenario
        - scenario="rush_hour" → specific scenario
        - scenario="random" → random from all 14
        - scenario="none" → no scenario (baseline)
        """
        self._active_scenarios.clear()

        if scenario == "none" or scenario == "baseline":
            return
        elif scenario is not None and scenario in SCENARIO_REGISTRY:
            sc = SCENARIO_REGISTRY[scenario]()
            sc.reset(grid)
            self._active_scenarios.append(sc)
        else:
            # Random scenario
            key = random.choice(list(SCENARIO_REGISTRY.keys()))
            sc = SCENARIO_REGISTRY[key]()
            sc.reset(grid)
            self._active_scenarios.append(sc)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        """Merge all active scenario updates."""
        if not self._active_scenarios:
            return ScenarioUpdate.default(grid.n_nodes)

        combined = ScenarioUpdate.default(grid.n_nodes)

        for sc in self._active_scenarios:
            sub = sc.step(t, grid)

            # Merge spawn rates (multiply)
            for i in range(min(grid.n_nodes, len(sub.spawn_rate_multipliers))):
                for j in range(min(12, len(sub.spawn_rate_multipliers[i]))):
                    combined.spawn_rate_multipliers[i][j] *= sub.spawn_rate_multipliers[i][j]

            # Merge speeds (min)
            for i in range(min(grid.n_nodes, len(sub.speed_multipliers))):
                combined.speed_multipliers[i] = min(
                    combined.speed_multipliers[i], sub.speed_multipliers[i]
                )

            combined.blocked_roads.extend(sub.blocked_roads)
            combined.emergency_vehicles.extend(sub.emergency_vehicles)
            combined.special_events.extend(sub.special_events)

        return combined

    @property
    def active_names(self) -> List[str]:
        return [sc.name for sc in self._active_scenarios]

    def get_reward_overrides(self) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for sc in self._active_scenarios:
            for k, v in sc.get_reward_overrides().items():
                merged[k] = max(merged.get(k, 0.0), v)
        return merged

    def get_flags(self, agent_id: int) -> List[float]:
        if not self._active_scenarios:
            return [0.0, 0.0, 0.0, 0.0]
        flags = [0.0, 0.0, 0.0, 0.0]
        for sc in self._active_scenarios:
            sf = sc.get_flags(agent_id)
            flags = [max(f, s) for f, s in zip(flags, sf)]
        return flags

    def get_scenario_flags(self) -> ScenarioFlags:
        emergency = any(
            hasattr(sc, "_emergency_vehicles") and sc._emergency_vehicles
            for sc in self._active_scenarios
        )
        blocked = []
        weather_sev = 0.0
        for sc in self._active_scenarios:
            if hasattr(sc, "_blocked"):
                blocked.extend([[u, v] for u, v in sc._blocked])
            if hasattr(sc, "_severity"):
                weather_sev = max(weather_sev, sc._severity)

        return ScenarioFlags(
            active_scenarios=self.active_names,
            emergency_active=emergency,
            blocked_roads=blocked,
            weather_severity=weather_sev,
        )
