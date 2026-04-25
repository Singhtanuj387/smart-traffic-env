# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Traffic Environment — 9×9 MARL traffic signal control.

81 agents, each controlling one intersection's signal phases.
Implements openenv-core Environment[TrafficAction, TrafficObservation, TrafficState].
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        AgentObservation,
        TrafficAction,
        TrafficObservation,
        TrafficState,
    )
    from ..core.grid import TrafficGrid
    from ..rewards.engine import RewardEngine
    from ..scenarios.manager import ScenarioManager
except ImportError:
    from models import (
        AgentObservation,
        TrafficAction,
        TrafficObservation,
        TrafficState,
    )
    from core.grid import TrafficGrid
    from rewards.engine import RewardEngine
    from scenarios.manager import ScenarioManager


class SmartTrafficEnvironment(
    Environment[TrafficAction, TrafficObservation, TrafficState]
):
    """
    9×9 MARL traffic signal control environment.
    81 agents, each controlling one intersection.

    openenv step() returns TrafficObservation (which contains done, reward, metadata).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ── Constants ───────────────────────────────────────────
    GRID_SIZE = 9
    N_AGENTS = 81
    MAX_STEPS = 3600  # 1 simulated hour
    YELLOW_DURATION = 3
    SPAWN_RATE_BASE = 0.15

    def __init__(self):
        super().__init__()
        self._grid = TrafficGrid(self.GRID_SIZE)
        self._rewards = RewardEngine()
        self._scenarios = ScenarioManager()
        self._state = TrafficState()

    # ── OpenEnv API ─────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: Optional[str] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """Reset environment and return initial observation."""
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)

        eid = episode_id or str(uuid.uuid4())

        self._grid.reset()
        self._rewards.reset()
        self._scenarios.reset(self._grid, scenario)

        self._state = TrafficState(
            episode_id=eid,
            step_count=0,
            queues=self._grid.get_all_queues(),
            wait_times=[[0.0] * 12 for _ in range(self.N_AGENTS)],
            phases=self._grid.get_all_phases(),
            phase_timers=[0] * self.N_AGENTS,
            yellow_flags=[False] * self.N_AGENTS,
            vehicles_passed=0,
            active_scenarios=self._scenarios.active_names,
        )
        self._sub_step = 0
        self._current_actions = []

        return self._build_observation(rewards=[0.0] * self.N_AGENTS)

    def step(
        self,
        action: TrafficAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """
        Process single sequential agent step.
        Ticks physics only after all 81 agents have acted.
        """
        # Validate sequential ordering
        if action.agent_id != self._sub_step:
            raise ValueError(f"Expected agent {self._sub_step} to act, got {action.agent_id}")

        self._current_actions.append(action)

        # 1. Immediate Phase application for sequential observation updates
        agent_idx = action.agent_id
        prev_phase = self._state.phases[agent_idx]
        new_phase = action.phase

        if new_phase != prev_phase:
            self._state.yellow_flags[agent_idx] = True
            self._state.phase_timers[agent_idx] = 0
            self._grid.intersections[agent_idx].yellow_active = True

        self._grid.apply_phase(
            agent_idx,
            action.phase,
            self._state.yellow_flags[agent_idx],
        )

        # Increment turn
        self._sub_step += 1

        # If not all agents acted, return intermediate observation
        if self._sub_step < self.N_AGENTS:
            # Re-read phases into state so obs builder catches it
            self._state.phases = self._grid.get_all_phases()
            return self._build_observation(rewards=[0.0] * self.N_AGENTS)

        # --- PHYSICS TICK: All 81 agents acted ---
        t = self._state.step_count

        # 1. Apply scenario dynamics
        scenario_update = self._scenarios.step(t, self._grid)

        # 2. Spawn vehicles according to scenario rates
        self._grid.spawn_vehicles(
            spawn_rate_multipliers=scenario_update.spawn_rate_multipliers
            if scenario_update.spawn_rate_multipliers
            else None,
            base_rate=self.SPAWN_RATE_BASE,
        )

        prev_queues = [list(q) for q in self._state.queues]

        # 4. Advance yellow counters
        for i in range(self.N_AGENTS):
            if self._state.yellow_flags[i]:
                self._state.phase_timers[i] += 1
                if self._state.phase_timers[i] >= self.YELLOW_DURATION:
                    self._state.yellow_flags[i] = False
                    self._state.phase_timers[i] = 0
                    self._grid.intersections[i].yellow_active = False

        # 5. Move vehicles through green phases
        speed_mults = (
            scenario_update.speed_multipliers
            if scenario_update.speed_multipliers
            else None
        )
        self._grid.move_vehicles(speed_mults)

        # 6. Compute per-agent rewards
        rewards = self._rewards.compute_all(
            grid=self._grid,
            prev_queues=prev_queues,
            actions=self._current_actions,
            scenario_update=scenario_update,
            weight_overrides=self._scenarios.get_reward_overrides(),
        )

        # 7. Update state
        self._state.step_count += 1
        self._state.queues = self._grid.get_all_queues()
        self._state.wait_times = [
            self._grid.intersections[i].get_wait_times()
            for i in range(self.N_AGENTS)
        ]
        self._state.phases = self._grid.get_all_phases()
        self._state.vehicles_passed += self._grid.get_cleared_this_step()
        self._state.active_scenarios = self._scenarios.active_names

        done = self._state.step_count >= self.MAX_STEPS
        mean_reward = float(np.mean(rewards))

        # Reset turn loop BEFORE building observation so active_agent=0
        self._sub_step = 0
        self._current_actions = []

        obs = self._build_observation(rewards)
        obs.done = done
        obs.reward = mean_reward

        return obs

    @property
    def state(self) -> TrafficState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="smart-traffic-env",
            description="9×9 MARL traffic signal control — 81 agents, 14 scenarios",
            version="0.1.0",
        )

    def close(self) -> None:
        self._grid.close()

    # ── Private helpers ─────────────────────────────────────

    def _build_observation(self, rewards: list[float]) -> TrafficObservation:
        agents = []
        for i in range(self.N_AGENTS):
            agents.append(
                AgentObservation(
                    agent_id=i,
                    queue_lengths=self._grid.get_queue_obs(i),
                    wait_times=self._grid.get_wait_obs(i),
                    current_phase=self._grid.get_phase_onehot(i),
                    phase_elapsed=min(
                        self._state.phase_timers[i] / 30.0, 1.0
                    ),
                    yellow_active=float(self._state.yellow_flags[i]),
                    neighbor_queues=self._grid.get_neighbor_queues(i),
                    neighbor_phases=self._grid.get_neighbor_phases(i),
                    congestion_index=self._grid.get_congestion_index(i),
                    special_flags=self._scenarios.get_flags(i),
                    agent_reward=rewards[i],
                )
            )

        return TrafficObservation(
            step=self._state.step_count,
            active_agent=getattr(self, "_sub_step", 0),
            done=self._state.step_count >= self.MAX_STEPS,
            reward=float(np.mean(rewards)),
            agents=agents,
            global_metrics=self._grid.get_global_metrics(),
            scenario_flags=self._scenarios.get_scenario_flags(),
        )
