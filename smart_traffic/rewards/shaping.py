"""
Potential-based reward shaping.

φ(s, s') = γ · Φ(s') − Φ(s)

This form guarantees shaping does NOT change the optimal policy
(Ng, Harada & Russell 1999).
"""

from __future__ import annotations

from typing import List


def compute_potential(queue_state: List[float], gamma: float = 0.99) -> float:
    """
    Potential function Φ(s) based on total queue occupancy.
    Returns a negative value proportional to congestion.
    """
    total = sum(queue_state)
    return -0.5 * total / (12.0 * 30.0)


def shaped_reward(
    raw: float,
    prev_state: List[float],
    next_state: List[float],
    gamma: float = 0.99,
) -> float:
    """Apply potential-based shaping to a raw reward value."""
    return raw + gamma * compute_potential(next_state, gamma) - compute_potential(
        prev_state, gamma
    )
