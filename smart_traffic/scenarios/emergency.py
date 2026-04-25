"""
EmergencyPriority — ambulance node path; phase-clear check each step.
"""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, TYPE_CHECKING

from .base import BaseScenario, EmergencyVehicle, ScenarioUpdate

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class EmergencyPriority(BaseScenario):
    name = "emergency"

    def __init__(self):
        self._emergency_vehicles: List[EmergencyVehicle] = []
        self._spawn_interval = 300  # new EV every N steps

    def reset(self, grid: "TrafficGrid") -> None:
        self._emergency_vehicles.clear()
        self._spawn_emergency(grid)

    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        update = ScenarioUpdate.default(grid.n_nodes)

        # Spawn new emergency vehicle periodically
        if t > 0 and t % self._spawn_interval == 0:
            self._spawn_emergency(grid)

        # Advance existing emergency vehicles
        active_evs = []
        for ev in self._emergency_vehicles:
            ev.steps_active += 1
            if ev.steps_active < ev.max_steps and ev.current_node < len(ev.path) - 1:
                # Check if current intersection has cleared
                node = ev.path[ev.current_node]
                if grid.is_phase_aligned(node, ev.required_phase):
                    ev.current_node += 1
                active_evs.append(ev)

        self._emergency_vehicles = active_evs
        update.emergency_vehicles = list(self._emergency_vehicles)
        update.special_events = ["emergency_active"] if active_evs else []

        return update

    def _spawn_emergency(self, grid: "TrafficGrid") -> None:
        g = grid.grid_size
        # Random start/end on grid edges
        start = random.randint(0, g - 1)  # top row
        end = (g - 1) * g + random.randint(0, g - 1)  # bottom row
        path = grid.pathfinder.find_path(start, end)
        if path and len(path) >= 2:
            # Determine required phase based on primary direction
            required = "NS_STRAIGHT_GREEN"
            self._emergency_vehicles.append(
                EmergencyVehicle(
                    vehicle_id=str(uuid.uuid4())[:8],
                    path=path,
                    current_node=0,
                    required_phase=required,
                    max_steps=200,
                )
            )

    def get_reward_overrides(self) -> Dict[str, float]:
        return {"w_emergency": 5.0, "w_throughput": 0.8}

    def get_flags(self, agent_id: int) -> List[float]:
        for ev in self._emergency_vehicles:
            if agent_id in ev.path:
                progress = ev.current_node / max(len(ev.path), 1)
                return [0.0, 1.0, progress, 0.0]
        return [0.0, 0.0, 0.0, 0.0]
