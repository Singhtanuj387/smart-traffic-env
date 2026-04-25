"""
MAPPO (Multi-Agent Proximal Policy Optimization) for Smart Traffic.

Simplified MAPPO implementation using PyTorch for 81 cooperative agents.
Each agent shares policy parameters but receives individual observations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ActorCritic(nn.Module):
    """Shared actor-critic network for MAPPO agents."""

    def __init__(
        self,
        obs_dim: int = 47,
        n_actions: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Critic head (value function) — takes concatenated obs for centralized critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """Stores trajectories for PPO update."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.dones: List[bool] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: bool,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.observations)


class MAPPO:
    """
    Multi-Agent PPO trainer.

    All 81 agents share one ActorCritic network.
    Training uses centralized value function with decentralized policies.
    """

    def __init__(
        self,
        obs_dim: int = 67,
        n_actions: int = 5,
        n_agents: int = 81,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MAPPO training")

        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.policy = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for all agents.

        Args:
            obs: (81, 47) observation array

        Returns:
            actions: (81,) action indices
            log_probs: (81,) log probabilities
            values: (81,) value estimates
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            actions, log_probs, values = self.policy.act(obs_t)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def act_single(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action for a single agent (used in sequential environments).
        
        Args:
            obs: (67,) observation array
            
        Returns:
            action: integer action index
            log_prob: float log probability
            value: float value estimate
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions, log_probs, values = self.policy.act(obs_t)
            
        return (
            int(actions[0].item()),
            float(log_probs[0].item()),
            float(values[0].item()),
        )

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected trajectories."""
        if len(self.buffer) == 0:
            return {}

        # Compute GAE advantages
        advantages, returns = self._compute_gae()

        # Flatten across time and agents
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        acts = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(
            self.device
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Reshape: (T, 81, ...) → (T*81, ...)
        T = obs.shape[0]
        obs = obs.view(T * self.n_agents, -1)
        acts = acts.view(T * self.n_agents)
        old_log_probs = old_log_probs.view(T * self.n_agents)
        advantages = advantages.view(T * self.n_agents)
        returns = returns.view(T * self.n_agents)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(obs.shape[0])
            for start in range(0, obs.shape[0], self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                new_log_probs, values, entropy = self.policy.evaluate(
                    obs[idx], acts[idx]
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * advantages[idx]
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(values, returns[idx])

                # Total loss
                loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()

        return {
            "loss": total_loss / max(n_updates, 1),
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        T = len(self.buffer)
        advantages = np.zeros((T, self.n_agents))
        returns = np.zeros((T, self.n_agents))

        last_gae = np.zeros(self.n_agents)
        last_value = np.zeros(self.n_agents)

        for t in reversed(range(T)):
            if t == T - 1 or self.buffer.dones[t]:
                next_value = np.zeros(self.n_agents)
            else:
                next_value = self.buffer.values[t + 1]

            delta = (
                self.buffer.rewards[t]
                + self.gamma * next_value * (1 - float(self.buffer.dones[t]))
                - self.buffer.values[t]
            )
            last_gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - float(self.buffer.dones[t]))
                * last_gae
            )
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.buffer.values[t]

        return advantages, returns

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
