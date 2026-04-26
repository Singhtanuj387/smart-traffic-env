"""
Nine composable reward component functions.

Each function takes grid state (and optionally previous state / actions)
and returns a scalar reward for a single agent.  Functions are pure —
they have no side effects and depend only on their arguments.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    try:
        from ..core.grid import TrafficGrid
    except ImportError:
        from core.grid import TrafficGrid


def R_throughput(grid: "TrafficGrid", agent_id: int) -> float:
    """Normalised vehicles cleared this step (0–1 range typical)."""
    return 2.0 * grid.cars_cleared_this_step(agent_id) / 10.0


def R_queue(curr_q: List[float], prev_q: List[float]) -> float:
    """Reward queue reduction; penalise growth."""
    delta = sum(prev_q) - sum(curr_q)
    return 1.5 * delta / 30.0


def R_wait(avg_wait: float, threshold: float = 60.0) -> float:
    """Linear below threshold, exponential above."""
    if avg_wait <= threshold:
        return -0.3 * (avg_wait / max(threshold, 1e-6))
    return -0.3 - 0.5 * (1.0 - math.exp(-(avg_wait - threshold) / 30.0))


def R_phase_stability(
    action_phase: str,
    grid: "TrafficGrid",
    agent_id: int,
    min_green: int = 10,
) -> float:
    """Penalise switching before minimum green duration."""
    elapsed = grid.get_phase_elapsed(agent_id)
    current = grid.get_current_phase(agent_id)
    switched = action_phase != current
    if switched and elapsed < min_green:
        return -0.4 * (1.0 - elapsed / max(min_green, 1))
    return 0.0


def R_coordination(grid: "TrafficGrid", agent_id: int) -> float:
    """Penalise releasing into congested downstream intersections."""
    neighbors = grid.get_neighbors(agent_id)
    if not neighbors:
        return 0.0
    neighbor_means = [
        float(np.mean(grid.get_queue(n))) for n in neighbors
    ]
    downstream_pressure = float(np.mean(neighbor_means)) / 30.0
    outflow = grid.cars_cleared_this_step(agent_id) / 10.0
    return -0.6 * downstream_pressure * outflow


def R_emergency(
    scenario_update,
    grid: "TrafficGrid",
    agent_id: int,
) -> float:
    """High reward for clearing emergency path; penalty for blocking."""
    evs = getattr(scenario_update, "emergency_vehicles", [])
    if not evs:
        return 0.0
    for ev in evs:
        if agent_id in ev.path:
            cleared = grid.is_phase_aligned(agent_id, ev.required_phase)
            return 3.0 if cleared else -1.5
    return 0.0


def R_fairness(queue: List[float]) -> float:
    """Penalise high Gini coefficient across lanes."""
    if not queue or max(queue) == 0:
        return 0.0
    sorted_q = sorted(queue)
    n = len(sorted_q)
    total = sum(sorted_q)
    if total == 0:
        return 0.0
    gini = sum(
        abs(sorted_q[i] - sorted_q[j]) for i in range(n) for j in range(n)
    )
    gini /= 2.0 * n * total
    return -0.3 * gini


def R_recovery(
    grid: "TrafficGrid",
    agent_id: int,
    history: List[List[List[float]]],
    k: int = 10,
) -> float:
    """Reward improvement from k steps ago."""
    if len(history) < k:
        return 0.0
    old_q = sum(history[-k][agent_id])
    curr_q = sum(grid.get_queue(agent_id))
    return 0.4 * max(old_q - curr_q, 0.0) / 30.0


def R_cascade_prevention(grid: "TrafficGrid", agent_id: int) -> float:
    """Penalise each neighbor whose queue overflows due to release."""
    neighbors = grid.get_neighbors(agent_id)
    overflow_count = sum(
        1
        for n in neighbors
        if float(np.mean(grid.get_queue(n))) > 0.85 * 30  # >85% of max
    )
    return -0.8 * overflow_count
