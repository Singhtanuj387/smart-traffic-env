"""
OpenEnv Composable Rubric System for Smart Traffic MARL.

Each rubric is an independent, composable scoring module that evaluates
a single aspect of traffic control quality. They compose via WeightedSum,
Gate, and Sequential to form a rich, informative, anti-gaming reward signal.

Design principles:
  1. Rich signal: Every rubric returns a continuous float, not binary 0/1.
  2. Hard to game: Gate rubrics zero-out the entire reward if the agent
     exploits degenerate strategies (e.g., permanent ALL_RED or rapid flicker).
  3. Composable: Each rubric can be inspected, ablated, or reweighted
     independently without touching the others.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from openenv.core.rubrics import Rubric, WeightedSum, Gate

if TYPE_CHECKING:
    try:
        from ..core.grid import TrafficGrid
    except ImportError:
        from core.grid import TrafficGrid


# ═══════════════════════════════════════════════════════════════════════
# TIER 1: Core Traffic Rubrics (the "teaching" signal)
# ═══════════════════════════════════════════════════════════════════════

class ThroughputRubric(Rubric):
    """Reward vehicles actually clearing the intersection.
    
    This is the primary positive signal. Without throughput, nothing else
    matters. Normalized to [0, 1] range based on intersection capacity.
    """
    
    def forward(self, action: Any, observation: Any) -> float:
        grid: TrafficGrid = observation["grid"]
        agent_id: int = observation["agent_id"]
        cleared = grid.cars_cleared_this_step(agent_id)
        # Max ~10 cars per green phase per step
        return min(cleared / 10.0, 1.0)


class QueueReductionRubric(Rubric):
    """Reward queue shrinkage; penalize queue growth.
    
    Captures something hard to measure: are queues actually draining?
    A clever agent must balance which lanes to drain vs. let build up.
    Range: [-1, 1] where positive = queues shrinking.
    """
    
    def forward(self, action: Any, observation: Any) -> float:
        curr_q: List[float] = observation["curr_queues"]
        prev_q: List[float] = observation["prev_queues"]
        delta = sum(prev_q) - sum(curr_q)
        # Normalize: max delta is ~30 cars across 12 lanes
        return max(min(delta / 30.0, 1.0), -1.0)


class WaitTimePenaltyRubric(Rubric):
    """Exponentially penalize long wait times.
    
    Linear below 60s (acceptable), exponential above (unacceptable).
    This captures the non-linear human frustration curve — 30s feels OK,
    120s feels catastrophic. The exponential curve teaches urgency.
    Range: [-1, 0] where 0 = no waiting.
    """
    
    def __init__(self, threshold: float = 60.0):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, action: Any, observation: Any) -> float:
        avg_wait: float = observation["avg_wait"]
        if avg_wait <= self.threshold:
            return -0.3 * (avg_wait / max(self.threshold, 1e-6))
        # Exponential penalty kicks in above threshold
        return -0.3 - 0.7 * (1.0 - math.exp(-(avg_wait - self.threshold) / 30.0))


class CoordinationRubric(Rubric):
    """Penalize releasing vehicles into already-congested downstream intersections.
    
    This is the "cleverness" rubric — it captures spatial awareness.
    An agent shouldn't just clear its own queue if it's flooding its neighbor.
    This is extremely hard to game because clearing vehicles IS the goal,
    but doing it at the wrong time makes things worse globally.
    Range: [-1, 0].
    """
    
    def forward(self, action: Any, observation: Any) -> float:
        grid: TrafficGrid = observation["grid"]
        agent_id: int = observation["agent_id"]
        neighbors = grid.get_neighbors(agent_id)
        if not neighbors:
            return 0.0
        neighbor_pressure = float(np.mean([
            float(np.mean(grid.get_queue(n))) for n in neighbors
        ])) / 30.0
        outflow = grid.cars_cleared_this_step(agent_id) / 10.0
        # Penalty scales with both downstream pressure AND how much you released
        return -neighbor_pressure * outflow


class FairnessRubric(Rubric):
    """Penalize lane starvation using Gini coefficient.
    
    A lane that never gets green while others flow freely is unfair.
    This prevents the common gaming exploit where an agent only ever
    serves the highest-volume direction and starves cross-traffic forever.
    Range: [-1, 0] where 0 = perfectly fair.
    """
    
    def forward(self, action: Any, observation: Any) -> float:
        queue: List[float] = observation["curr_queues"]
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
        return -gini


# ═══════════════════════════════════════════════════════════════════════
# TIER 2: Stability & Safety Rubrics (prevent degenerate strategies)
# ═══════════════════════════════════════════════════════════════════════

class PhaseStabilityRubric(Rubric):
    """Penalize switching phases before minimum green duration.
    
    Anti-gaming measure: prevents the "rapid flicker" exploit where
    an agent switches every single step to artificially manipulate
    queue measurements without actually clearing traffic.
    Range: [-1, 0] where 0 = stable switching.
    """
    
    def __init__(self, min_green: int = 10):
        super().__init__()
        self.min_green = min_green
    
    def forward(self, action: Any, observation: Any) -> float:
        grid: TrafficGrid = observation["grid"]
        agent_id: int = observation["agent_id"]
        action_phase: str = observation["action_phase"]
        
        elapsed = grid.get_phase_elapsed(agent_id)
        current = grid.get_current_phase(agent_id)
        switched = action_phase != current
        if switched and elapsed < self.min_green:
            return -(1.0 - elapsed / max(self.min_green, 1))
        return 0.0


class EmergencyRubric(Rubric):
    """High reward for clearing emergency vehicle paths.
    
    When an ambulance is routing through, the agent MUST give it
    priority. This rubric provides a massive positive spike (+1.0)
    for compliance and a harsh penalty (-1.0) for blocking.
    Range: [-1, 1].
    """
    
    def forward(self, action: Any, observation: Any) -> float:
        grid: TrafficGrid = observation["grid"]
        agent_id: int = observation["agent_id"]
        scenario_update = observation.get("scenario_update")
        if scenario_update is None:
            return 0.0
        evs = getattr(scenario_update, "emergency_vehicles", [])
        if not evs:
            return 0.0
        for ev in evs:
            if agent_id in ev.path:
                cleared = grid.is_phase_aligned(agent_id, ev.required_phase)
                return 1.0 if cleared else -1.0
        return 0.0


class CascadePreventionRubric(Rubric):
    """Penalize causing downstream queue overflow.
    
    Anti-gaming: an agent that aggressively clears its own queue but
    overflows its neighbors gets punished. This forces the agent to
    think about the SYSTEM, not just its local score.
    Range: [-1, 0] per overflowing neighbor.
    """
    
    def forward(self, action: Any, observation: Any) -> float:
        grid: TrafficGrid = observation["grid"]
        agent_id: int = observation["agent_id"]
        neighbors = grid.get_neighbors(agent_id)
        overflow_count = sum(
            1 for n in neighbors
            if float(np.mean(grid.get_queue(n))) > 0.85 * 30
        )
        max_neighbors = max(len(neighbors), 1)
        return -overflow_count / max_neighbors


class RecoveryRubric(Rubric):
    """Reward long-term queue improvement (k-step lookback).
    
    Captures temporal credit assignment: did your actions k steps ago
    lead to better conditions now? This teaches strategic planning,
    not just greedy per-step optimization.
    Range: [0, 1].
    """
    
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
    
    def forward(self, action: Any, observation: Any) -> float:
        history: List = observation.get("queue_history", [])
        agent_id: int = observation["agent_id"]
        grid: TrafficGrid = observation["grid"]
        
        if len(history) < self.k:
            return 0.0
        old_q = sum(history[-self.k][agent_id])
        curr_q = sum(grid.get_queue(agent_id))
        improvement = max(old_q - curr_q, 0.0) / 30.0
        return min(improvement, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# TIER 3: Anti-Gaming Gate Rubrics (zero out degenerate strategies)
# ═══════════════════════════════════════════════════════════════════════

class AllRedExploitDetector(Rubric):
    """Gate: Detect and zero-out the "permanent ALL_RED" exploit.
    
    A degenerate agent can learn that ALL_RED prevents queue GROWTH
    on the serviced lanes (since no cars enter), getting positive
    queue-reduction scores while traffic dies globally.
    
    This gate returns 0.0 if the agent has been holding ALL_RED for
    more than `max_hold` consecutive steps, killing the entire reward.
    """
    
    def __init__(self, max_hold: int = 5):
        super().__init__()
        self.max_hold = max_hold
        self._consecutive_red: Dict[int, int] = {}
    
    def reset(self) -> None:
        self._consecutive_red.clear()
    
    def forward(self, action: Any, observation: Any) -> float:
        agent_id: int = observation["agent_id"]
        action_phase: str = observation["action_phase"]
        
        if action_phase == "ALL_RED_HOLD":
            self._consecutive_red[agent_id] = self._consecutive_red.get(agent_id, 0) + 1
        else:
            self._consecutive_red[agent_id] = 0
        
        if self._consecutive_red.get(agent_id, 0) > self.max_hold:
            return 0.0  # Kill the reward — you're gaming!
        return 1.0  # Pass-through


class FlickerExploitDetector(Rubric):
    """Gate: Detect and zero-out the "rapid phase flicker" exploit.
    
    A degenerate agent can learn to switch phases every single step
    to manipulate the queue measurement window, getting credit for
    "clearing" without actually serving any vehicles through.
    
    Returns 0.0 if the agent has switched phases more than `max_switches`
    times in the last `window` steps.
    """
    
    def __init__(self, window: int = 10, max_switches: int = 6):
        super().__init__()
        self.window = window
        self.max_switches = max_switches
        self._switch_history: Dict[int, List[str]] = {}
    
    def reset(self) -> None:
        self._switch_history.clear()
    
    def forward(self, action: Any, observation: Any) -> float:
        agent_id: int = observation["agent_id"]
        action_phase: str = observation["action_phase"]
        
        if agent_id not in self._switch_history:
            self._switch_history[agent_id] = []
        
        history = self._switch_history[agent_id]
        history.append(action_phase)
        if len(history) > self.window:
            history.pop(0)
        
        if len(history) < 2:
            return 1.0
        
        switches = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
        if switches > self.max_switches:
            return 0.0  # Kill the reward — you're flickering!
        return 1.0


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE: Build the full rubric tree
# ═══════════════════════════════════════════════════════════════════════

def build_traffic_rubric() -> Rubric:
    """Build the complete composable rubric tree for Smart Traffic.
    
    Architecture:
    
        Gate(AllRedExploitDetector)
          └── Gate(FlickerExploitDetector)
                └── WeightedSum(
                      ThroughputRubric        (0.25)  ← Primary positive signal
                      QueueReductionRubric    (0.20)  ← Queue management
                      WaitTimePenaltyRubric   (0.15)  ← Human frustration proxy
                      CoordinationRubric      (0.10)  ← Spatial awareness
                      FairnessRubric          (0.05)  ← Lane equity
                      PhaseStabilityRubric    (0.05)  ← Switching penalty
                      EmergencyRubric         (0.10)  ← Safety compliance
                      CascadePreventionRubric (0.05)  ← System-level thinking
                      RecoveryRubric          (0.05)  ← Temporal credit
                    )
    
    The Gate rubrics act as hard anti-gaming constraints.
    If the agent exploits ALL_RED or rapid flickering, the Gate
    returns 0.0, zeroing out the entire WeightedSum reward.
    This makes gaming strategies strictly dominated.
    """
    
    # Core teaching signal — weighted composition of 9 rubrics
    core_signal = WeightedSum(
        rubrics=[
            ThroughputRubric(),
            QueueReductionRubric(),
            WaitTimePenaltyRubric(threshold=60.0),
            CoordinationRubric(),
            FairnessRubric(),
            PhaseStabilityRubric(min_green=10),
            EmergencyRubric(),
            CascadePreventionRubric(),
            RecoveryRubric(k=10),
        ],
        weights=[0.25, 0.20, 0.15, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05],
    )
    
    # Layer 1: Kill reward if agent flickers phases too rapidly
    flicker_gated = Gate(
        rubric=core_signal,
        threshold=0.0,  # Gate passes if FlickerDetector returns > 0
    )
    
    # Layer 2: Kill reward if agent holds ALL_RED too long
    # Note: We use the anti-gaming detectors separately in the engine
    # because Gate() wraps a single child rubric
    
    return core_signal


def build_anti_gaming_gates() -> tuple:
    """Return the anti-gaming gate detectors as separate inspectable rubrics.
    
    These are evaluated before the core signal. If either returns 0.0,
    the entire reward for that agent is zeroed out.
    """
    return (
        AllRedExploitDetector(max_hold=5),
        FlickerExploitDetector(window=10, max_switches=6),
    )
