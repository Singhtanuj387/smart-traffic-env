# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Traffic Control Environment.

9×9 MARL traffic signal control with 81 agents.
All types use Pydantic for automatic validation, serialization, and HTTP schema generation.
"""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# ACTION  (inherits from openenv Action — includes metadata dict)
# ─────────────────────────────────────────────────────────────

PhaseChoice = Literal[
    "NS_STRAIGHT_GREEN",
    "EW_STRAIGHT_GREEN",
    "PROTECTED_NS_LEFT",
    "PROTECTED_EW_LEFT",
    "ALL_RED_HOLD",
]

PHASE_LIST: list[str] = [
    "NS_STRAIGHT_GREEN",
    "EW_STRAIGHT_GREEN",
    "PROTECTED_NS_LEFT",
    "PROTECTED_EW_LEFT",
    "ALL_RED_HOLD",
]

NUM_PHASES = 5


class TrafficAction(Action):
    """Action sent by one agent for its intersection."""

    agent_id: int = Field(..., ge=0, lt=81, description="Agent index 0-80")
    phase: PhaseChoice = Field(..., description="Signal phase to apply")


class MultiAgentAction(Action):
    """Batch action for all 81 agents in one step call."""

    actions: list[TrafficAction] = Field(
        ..., min_length=81, max_length=81, description="Exactly 81 agent actions"
    )


# ─────────────────────────────────────────────────────────────
# OBSERVATION  (inherits from openenv Observation — includes done, reward, metadata)
# ─────────────────────────────────────────────────────────────


class GlobalMetrics(BaseModel):
    """Network-wide performance metrics."""

    avg_wait_time: float = 0.0
    total_throughput: int = 0
    network_efficiency: float = 0.0
    congestion_count: int = 0


class ScenarioFlags(BaseModel):
    """Active scenario information."""

    active_scenarios: list[str] = Field(default_factory=list)
    emergency_active: bool = False
    blocked_roads: list[list[int]] = Field(default_factory=list)
    weather_severity: float = 0.0


class AgentObservation(BaseModel):
    """67-dimensional observation for a single agent."""

    agent_id: int
    queue_lengths: list[float] = Field(..., min_length=12, max_length=12)
    wait_times: list[float] = Field(..., min_length=12, max_length=12)
    current_phase: list[float] = Field(..., min_length=5, max_length=5)
    phase_elapsed: float = Field(..., ge=0.0, le=1.0)
    yellow_active: float = Field(..., ge=0.0, le=1.0)
    neighbor_queues: list[float] = Field(..., min_length=8, max_length=8)
    neighbor_phases: list[float] = Field(..., min_length=20, max_length=20)
    congestion_index: list[float] = Field(..., min_length=4, max_length=4)
    special_flags: list[float] = Field(..., min_length=4, max_length=4)
    agent_reward: float = 0.0


class TrafficObservation(Observation):
    """
    Top-level observation containing all 81 agent observations.
    Inherits done, reward, metadata from openenv Observation.
    """

    step: int = 0
    active_agent: int = 0
    agents: list[AgentObservation] = Field(default_factory=list)
    global_metrics: GlobalMetrics = Field(default_factory=GlobalMetrics)
    scenario_flags: ScenarioFlags = Field(default_factory=ScenarioFlags)


# ─────────────────────────────────────────────────────────────
# STATE  (extends openenv State — has episode_id, step_count, extra="allow")
# ─────────────────────────────────────────────────────────────


class TrafficState(State):
    """
    Full serializable environment state.
    Extends openenv State (episode_id + step_count) with traffic-specific fields.
    """

    queues: list[list[float]] = Field(default_factory=list)  # 81 x 12
    wait_times: list[list[float]] = Field(default_factory=list)  # 81 x 12
    phases: list[str] = Field(default_factory=list)  # 81 current phases
    phase_timers: list[int] = Field(default_factory=list)  # 81 elapsed steps
    yellow_flags: list[bool] = Field(default_factory=list)  # 81 yellow states
    vehicles_passed: int = 0
    active_scenarios: list[str] = Field(default_factory=list)
    emergency_paths: list[list[int]] = Field(default_factory=list)
