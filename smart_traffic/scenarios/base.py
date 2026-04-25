"""
Base scenario protocol and shared data types.

All 14 scenarios implement BaseScenario.  ScenarioUpdate is the message
passed from scenario → environment each tick, modifying spawn rates,
speed multipliers, blocked roads, and emergency vehicles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..core.grid import TrafficGrid


class EmergencyVehicle(BaseModel):
    """An emergency vehicle requiring priority passage."""

    vehicle_id: str = ""
    path: List[int] = Field(default_factory=list)
    current_node: int = 0
    required_phase: str = "NS_STRAIGHT_GREEN"
    max_steps: int = 200
    steps_active: int = 0


class ScenarioUpdate(BaseModel):
    """Returned by scenario.step() — modifies environment dynamics."""

    spawn_rate_multipliers: List[List[float]] = Field(default_factory=list)  # 81×12
    speed_multipliers: List[float] = Field(default_factory=list)  # 81
    blocked_roads: List[List[int]] = Field(default_factory=list)
    emergency_vehicles: List[EmergencyVehicle] = Field(default_factory=list)
    special_events: List[str] = Field(default_factory=list)

    @classmethod
    def default(cls, n_agents: int = 81, n_lanes: int = 12) -> "ScenarioUpdate":
        """Create a neutral update (no modifications)."""
        return cls(
            spawn_rate_multipliers=[[1.0] * n_lanes for _ in range(n_agents)],
            speed_multipliers=[1.0] * n_agents,
        )


class BaseScenario(ABC):
    """Abstract base for all traffic scenarios."""

    name: str = "base"

    @abstractmethod
    def reset(self, grid: "TrafficGrid") -> None:
        """Initialize scenario state for a new episode."""
        ...

    @abstractmethod
    def step(self, t: int, grid: "TrafficGrid") -> ScenarioUpdate:
        """
        Return dynamic modifiers for this timestep.
        Called once per environment step *before* vehicle spawning.
        """
        ...

    def get_reward_overrides(self) -> Dict[str, float]:
        """Per-scenario reward weight overrides (merged into RewardEngine)."""
        return {}

    def get_flags(self, agent_id: int) -> List[float]:
        """4-element special flags for observation per agent."""
        return [0.0, 0.0, 0.0, 0.0]

    def is_active(self, t: int) -> bool:
        """Whether this scenario is currently active at timestep t."""
        return True
