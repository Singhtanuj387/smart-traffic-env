"""
Thin Gymnasium wrapper around TrafficEnvClient.

The actual simulation runs in the Docker container (or locally).
This wrapper gives RL libraries (TorchRL, StableBaselines3) a familiar interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import asyncio

import gymnasium as gym
import numpy as np

from ..client import SmartTrafficEnv
from ..models import TrafficAction, PHASE_LIST


class TrafficGymAdapter(gym.Env):
    """
    Gymnasium wrapper around SmartTrafficEnv HTTP client.

    Usage:
        env = TrafficGymAdapter(server_url='http://localhost:8000')
        obs, info = env.reset(seed=42)
        actions = np.random.randint(0, 5, size=81)
        obs, reward, done, truncated, info = env.step(actions)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        scenario: Optional[str] = None,
    ):
        super().__init__()
        self.client = SmartTrafficEnv(base_url=server_url)
        self.scenario = scenario
        self._loop = asyncio.new_event_loop()

        # 81 agents × 67 observation dims
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(81, 67), dtype=np.float32
        )

        # 1 agent × 5 discrete actions
        self.action_space = gym.spaces.Discrete(5)

        self._last_obs = None
        self._current_agent = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        scenario = self.scenario
        if options and "scenario" in options:
            scenario = options["scenario"]

        result = self._loop.run_until_complete(
            self.client.reset(seed=seed, scenario=scenario)
        )
        obs = result.observation if hasattr(result, "observation") else result
        np_obs = SmartTrafficEnv.get_numpy_obs(obs)
        self._last_obs = obs
        self._current_agent = getattr(obs, "active_agent", 0)

        info = {
            "step": obs.step,
            "global_metrics": obs.global_metrics.model_dump(),
            "scenario_flags": obs.scenario_flags.model_dump(),
        }
        return np_obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Take single action for the currently active agent tracking locally
        traffic_action = TrafficAction(agent_id=self._current_agent, phase=PHASE_LIST[int(action)])

        result = self._loop.run_until_complete(
            self.client.step(traffic_action)
        )
        obs = result.observation
        np_obs = SmartTrafficEnv.get_numpy_obs(obs)
        self._last_obs = obs
        self._current_agent = getattr(obs, "active_agent", 0)

        # In sequential MDP, rewards are 0 until the final agent steps
        reward = result.reward if result.reward is not None else 0.0
        done = result.done
        truncated = False

        info = {
            "step": obs.step,
            "active_agent": self._current_agent,
            "global_metrics": obs.global_metrics.model_dump(),
            "per_agent_rewards": [ag.agent_reward for ag in obs.agents],
        }

        return np_obs, reward, done, truncated, info

    def close(self):
        if hasattr(self, "_loop") and self._loop.is_running():
            asyncio.create_task(self.client.close())
        elif hasattr(self, "_loop"):
            self._loop.run_until_complete(self.client.close())
            # We don't close the loop because standard RL workloads might reuse the environment instance
