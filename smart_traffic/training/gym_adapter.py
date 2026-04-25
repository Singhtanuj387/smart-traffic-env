"""
Thin Gymnasium wrapper around TrafficEnvClient.

The actual simulation runs in the Docker container (or locally).
This wrapper gives RL libraries (TorchRL, StableBaselines3) a familiar interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..client import SmartTrafficEnv
from ..models import MultiAgentAction, TrafficAction, PHASE_LIST


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

        # 81 agents × 47 observation dims
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(81, 47), dtype=np.float32
        )

        # 81 agents × 5 discrete actions
        self.action_space = gym.spaces.MultiDiscrete([5] * 81)

        self._last_obs = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        scenario = self.scenario
        if options and "scenario" in options:
            scenario = options["scenario"]

        result = self.client.reset(seed=seed, scenario=scenario)
        obs = result.observation if hasattr(result, "observation") else result
        np_obs = SmartTrafficEnv.get_numpy_obs(obs)
        self._last_obs = obs

        info = {
            "step": obs.step,
            "global_metrics": obs.global_metrics.model_dump(),
            "scenario_flags": obs.scenario_flags.model_dump(),
        }
        return np_obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_list = actions.tolist()
        batch = MultiAgentAction(
            actions=[
                TrafficAction(agent_id=i, phase=PHASE_LIST[a])
                for i, a in enumerate(action_list)
            ]
        )

        result = self.client.step(batch)
        obs = result.observation
        np_obs = SmartTrafficEnv.get_numpy_obs(obs)
        self._last_obs = obs

        reward = result.reward if result.reward is not None else 0.0
        done = result.done
        truncated = False

        info = {
            "step": obs.step,
            "global_metrics": obs.global_metrics.model_dump(),
            "per_agent_rewards": [ag.agent_reward for ag in obs.agents],
        }

        return np_obs, reward, done, truncated, info

    def close(self):
        self.client.close()
