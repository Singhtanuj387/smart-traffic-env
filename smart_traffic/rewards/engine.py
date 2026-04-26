"""
Reward Engine — OpenEnv Rubric-based composable reward system.

Uses OpenEnv's Rubric primitives (WeightedSum, Gate) to build a
rich, informative, anti-gaming reward signal for 81-agent MARL.

Architecture:
    Anti-Gaming Gates  →  WeightedSum(9 composable rubrics)  →  Potential Shaping

Design principles:
  1. Rich signal: 9 continuous rubrics, not binary 0/1.
  2. Clever measurement: Gini fairness, exponential wait curves, spatial coordination.
  3. Composable: Each rubric can be inspected via rubric.last_score after evaluation.
  4. Hard to game: Gate rubrics zero the reward for ALL_RED exploit and rapid flickering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .rubrics import (
    build_traffic_rubric,
    build_anti_gaming_gates,
)
from .shaping import shaped_reward

if TYPE_CHECKING:
    try:
        from ..core.grid import TrafficGrid
    except ImportError:
        from core.grid import TrafficGrid


class RewardEngine:
    """Composable rubric-based reward engine with anti-gaming gates.
    
    The engine evaluates 9 composable rubrics via OpenEnv's WeightedSum,
    then applies two Gate rubrics that detect and zero-out degenerate
    strategies (permanent ALL_RED, rapid phase flickering).
    
    Finally, potential-based reward shaping (Ng et al. 1999) is applied
    to accelerate learning without changing the optimal policy.
    """

    def __init__(self):
        self._prev_queues_history: List[List[List[float]]] = []
        
        # Build the composable rubric tree
        self._rubric = build_traffic_rubric()
        
        # Build anti-gaming gate detectors (stateful, track per-agent history)
        self._all_red_gate, self._flicker_gate = build_anti_gaming_gates()

    def reset(self) -> None:
        """Reset all stateful rubrics and history."""
        self._prev_queues_history.clear()
        self._all_red_gate.reset()
        self._flicker_gate.reset()

    def compute_all(
        self,
        grid: "TrafficGrid",
        prev_queues: List[List[float]],
        actions: list,
        scenario_update,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """Compute per-agent rewards for all 81 agents using composable rubrics.
        
        Pipeline per agent:
          1. Build observation dict for rubric evaluation
          2. Check anti-gaming gates (ALL_RED exploit, flicker exploit)
          3. If gates pass → evaluate WeightedSum of 9 rubrics
          4. Apply potential-based reward shaping
        """
        rewards: List[float] = []

        # Store original weights
        original_weights = list(self._rubric._weights)
        
        # Apply scenario weight overrides if provided
        if weight_overrides:
            rubric_names = [
                "w_throughput", "w_queue", "w_wait", "w_coord",
                "w_fairness", "w_stability", "w_emergency",
                "w_cascade", "w_recovery"
            ]
            for idx, name in enumerate(rubric_names):
                if name in weight_overrides:
                    # Note: Original engine had arbitrary weights summing > 1
                    # WeightedSum expects weights scaling, so we just apply the multiplier
                    self._rubric._weights[idx] = weight_overrides[name]
                    
        try:
            for i in range(81):
                # Build the observation context dict for rubric evaluation
                action_phase = (
                    actions[i].phase 
                    if hasattr(actions[i], "phase") 
                    else "ALL_RED_HOLD"
                )
                
                obs_context = {
                    "grid": grid,
                    "agent_id": i,
                    "curr_queues": grid.get_queue(i),
                    "prev_queues": prev_queues[i],
                    "avg_wait": grid.get_avg_wait(i),
                    "action_phase": action_phase,
                    "scenario_update": scenario_update,
                    "queue_history": self._prev_queues_history,
                }
                
                # ── Anti-Gaming Gates ──────────────────────────────
                # If either gate returns 0.0, the agent gets ZERO reward.
                # This makes degenerate strategies strictly dominated.
                all_red_score = self._all_red_gate(None, obs_context)
                flicker_score = self._flicker_gate(None, obs_context)
                
                if all_red_score == 0.0 or flicker_score == 0.0:
                    # Agent is gaming! Zero out everything.
                    rewards.append(0.0)
                    continue
                
                # ── Core Rubric Evaluation ─────────────────────────
                # WeightedSum of 9 composable rubrics
                raw_reward = self._rubric(None, obs_context)
                
                # Scale to match the original reward magnitude
                # (rubric outputs are in [-1, 1], scale to training range)
                scaled_reward = raw_reward * 5.0
                
                # ── Potential-Based Shaping ────────────────────────
                # Guarantees shaping does NOT change optimal policy
                # (Ng, Harada & Russell 1999)
                shaped = shaped_reward(
                    scaled_reward,
                    prev_state=prev_queues[i],
                    next_state=grid.get_queue(i),
                )
                rewards.append(shaped)
        finally:
            # Restore original weights for safety
            self._rubric._weights = original_weights

        self._update_history(grid)
        return rewards

    def get_rubric_breakdown(self, grid: "TrafficGrid", agent_id: int, 
                              prev_queues: List[float], action_phase: str,
                              scenario_update=None) -> Dict[str, float]:
        """Get per-rubric score breakdown for debugging/visualization.
        
        Returns a dict like:
            {"throughput": 0.4, "queue_reduction": -0.1, "fairness": -0.05, ...}
        
        Useful for plotting ablation studies and understanding which
        rubric component is driving the reward signal.
        """
        obs_context = {
            "grid": grid,
            "agent_id": agent_id,
            "curr_queues": grid.get_queue(agent_id),
            "prev_queues": prev_queues,
            "avg_wait": grid.get_avg_wait(agent_id),
            "action_phase": action_phase,
            "scenario_update": scenario_update,
            "queue_history": self._prev_queues_history,
        }
        
        breakdown = {}
        rubric_names = [
            "throughput", "queue_reduction", "wait_time",
            "coordination", "fairness", "phase_stability",
            "emergency", "cascade_prevention", "recovery",
        ]
        
        for idx, name in enumerate(rubric_names):
            child = self._rubric._rubric_list[idx]
            score = child(None, obs_context)
            breakdown[name] = float(score)
        
        # Add gate status
        breakdown["gate_all_red"] = float(self._all_red_gate(None, obs_context))
        breakdown["gate_flicker"] = float(self._flicker_gate(None, obs_context))
        
        return breakdown

    def _update_history(self, grid: "TrafficGrid") -> None:
        self._prev_queues_history.append(grid.get_all_queues())
        if len(self._prev_queues_history) > 10:
            self._prev_queues_history.pop(0)
