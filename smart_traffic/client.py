# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Traffic Environment Client.

HTTP/WebSocket client for interacting with the running TrafficEnvironment server.
Training scripts use this client; never import the server class directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    AgentObservation,
    GlobalMetrics,
    MultiAgentAction,
    ScenarioFlags,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    PHASE_LIST,
)


class SmartTrafficEnv(
    EnvClient[MultiAgentAction, TrafficObservation, TrafficState]
):
    """
    Client for the Smart Traffic Environment.

    Connects to the running HTTP/WebSocket server and provides a clean
    Python API for training and evaluation.

    Example:
        >>> with SmartTrafficEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     # Create actions for all 81 agents
        ...     actions = [0] * 81  # phase indices
        ...     result = client.step_flat(actions)
    """

    def _step_payload(self, action: MultiAgentAction) -> Dict:
        """Convert MultiAgentAction to JSON payload for step message."""
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[TrafficObservation]:
        """Parse server response into StepResult[TrafficObservation]."""
        obs_data = payload.get("observation", {})

        # Parse agent observations
        agents = []
        for ag_data in obs_data.get("agents", []):
            agents.append(AgentObservation(**ag_data))

        # Parse global metrics
        gm_data = obs_data.get("global_metrics", {})
        global_metrics = GlobalMetrics(**gm_data) if gm_data else GlobalMetrics()

        # Parse scenario flags
        sf_data = obs_data.get("scenario_flags", {})
        scenario_flags = ScenarioFlags(**sf_data) if sf_data else ScenarioFlags()

        observation = TrafficObservation(
            step=obs_data.get("step", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            agents=agents,
            global_metrics=global_metrics,
            scenario_flags=scenario_flags,
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TrafficState:
        """Parse server response into TrafficState."""
        return TrafficState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            queues=payload.get("queues", []),
            wait_times=payload.get("wait_times", []),
            phases=payload.get("phases", []),
            phase_timers=payload.get("phase_timers", []),
            yellow_flags=payload.get("yellow_flags", []),
            vehicles_passed=payload.get("vehicles_passed", 0),
            active_scenarios=payload.get("active_scenarios", []),
        )

    # ── Convenience methods ──────────────────────────────────

    def step_flat(self, actions: List[int]) -> StepResult[TrafficObservation]:
        """
        Convenience wrapper: accepts flat list of 81 phase indices (0-4).
        Converts to MultiAgentAction and calls step().
        """
        batch = MultiAgentAction(
            actions=[
                TrafficAction(agent_id=i, phase=PHASE_LIST[a])
                for i, a in enumerate(actions)
            ]
        )
        return self.step(batch)

    @staticmethod
    def get_numpy_obs(obs: TrafficObservation) -> np.ndarray:
        """Return (81, 47) numpy array of agent observations."""
        rows = []
        for ag in obs.agents:
            row = (
                ag.queue_lengths      # 12
                + ag.wait_times       # 12
                + ag.current_phase    # 5
                + [ag.phase_elapsed]  # 1
                + [ag.yellow_active]  # 1
                + ag.neighbor_queues  # 8
                + ag.congestion_index # 4
                + ag.special_flags    # 4 → total 47
            )
            rows.append(row)
        return np.array(rows, dtype=np.float32)  # (81, 47)
