"""
Curriculum Learning Scheduler — 5-stage progressive training.

Automatically advances through difficulty stages based on performance gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CurriculumStage:
    """Definition of a single curriculum stage."""

    name: str
    scenarios: List[str]
    gate_avg_wait: float  # max avg wait to pass
    gate_throughput: float  # min throughput fraction
    gate_extra: Optional[Dict[str, float]] = None
    episode_range: Tuple[int, int] = (0, 2000)


# ── 5-stage curriculum definition ────────────────────────────

STAGES = [
    CurriculumStage(
        name="Foundation",
        scenarios=["none"],
        gate_avg_wait=45.0,
        gate_throughput=0.6,
        episode_range=(0, 2000),
    ),
    CurriculumStage(
        name="Basic Stress",
        scenarios=["rush_hour", "directional_imbalance"],
        gate_avg_wait=60.0,
        gate_throughput=0.5,
        episode_range=(2000, 5000),
    ),
    CurriculumStage(
        name="Events",
        scenarios=["emergency", "event_spike", "weather"],
        gate_avg_wait=70.0,
        gate_throughput=0.4,
        gate_extra={"emergency_success_rate": 0.8},
        episode_range=(5000, 10000),
    ),
    CurriculumStage(
        name="Failures",
        scenarios=["cascading_failure", "road_block", "network_partition"],
        gate_avg_wait=80.0,
        gate_throughput=0.35,
        gate_extra={"cascade_contained_steps": 200.0},
        episode_range=(10000, 18000),
    ),
    CurriculumStage(
        name="Full Mix",
        scenarios=[
            "rush_hour", "emergency", "road_block", "weather",
            "event_spike", "directional_imbalance", "cascading_failure",
            "network_partition", "multi_incident", "pedestrian_surge",
            "vehicle_mix", "adaptive_demand", "sensor_noise",
            "recovery_challenge",
        ],
        gate_avg_wait=90.0,
        gate_throughput=0.3,
        episode_range=(18000, 30000),
    ),
]


class CurriculumScheduler:
    """
    Manages curriculum progression based on rolling performance metrics.
    """

    def __init__(self, window_size: int = 100):
        self.current_stage_idx = 0
        self.window_size = window_size
        self._episode_count = 0

        # Rolling metrics
        self._avg_wait_history: List[float] = []
        self._throughput_history: List[float] = []
        self._extra_metrics: Dict[str, List[float]] = {}

    @property
    def current_stage(self) -> CurriculumStage:
        return STAGES[min(self.current_stage_idx, len(STAGES) - 1)]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    def get_scenario(self) -> Optional[str]:
        """Return a scenario name for the current episode."""
        import random

        scenarios = self.current_stage.scenarios
        if not scenarios or scenarios == ["none"]:
            return "none"
        return random.choice(scenarios)

    def record_episode(
        self,
        avg_wait: float,
        throughput_fraction: float,
        extra: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Record episode results. Returns True if stage advanced.
        """
        self._episode_count += 1
        self._avg_wait_history.append(avg_wait)
        self._throughput_history.append(throughput_fraction)

        if extra:
            for k, v in extra.items():
                if k not in self._extra_metrics:
                    self._extra_metrics[k] = []
                self._extra_metrics[k].append(v)

        # Trim to window
        if len(self._avg_wait_history) > self.window_size:
            self._avg_wait_history = self._avg_wait_history[-self.window_size:]
            self._throughput_history = self._throughput_history[-self.window_size:]
            for k in self._extra_metrics:
                if len(self._extra_metrics[k]) > self.window_size:
                    self._extra_metrics[k] = self._extra_metrics[k][-self.window_size:]

        # Check gate conditions
        if self._should_advance():
            self.current_stage_idx = min(
                self.current_stage_idx + 1, len(STAGES) - 1
            )
            self._avg_wait_history.clear()
            self._throughput_history.clear()
            self._extra_metrics.clear()
            return True

        return False

    def _should_advance(self) -> bool:
        if self.current_stage_idx >= len(STAGES) - 1:
            return False

        if len(self._avg_wait_history) < self.window_size // 2:
            return False

        stage = self.current_stage
        avg_wait = sum(self._avg_wait_history) / len(self._avg_wait_history)
        avg_throughput = sum(self._throughput_history) / len(self._throughput_history)

        if avg_wait > stage.gate_avg_wait:
            return False
        if avg_throughput < stage.gate_throughput:
            return False

        # Check extra gates
        if stage.gate_extra:
            for k, threshold in stage.gate_extra.items():
                if k in self._extra_metrics and self._extra_metrics[k]:
                    metric_avg = sum(self._extra_metrics[k]) / len(
                        self._extra_metrics[k]
                    )
                    if metric_avg < threshold:
                        return False

        return True

    def get_stats(self) -> Dict[str, float]:
        return {
            "stage": float(self.current_stage_idx),
            "episodes": float(self._episode_count),
            "rolling_avg_wait": (
                sum(self._avg_wait_history) / max(len(self._avg_wait_history), 1)
            ),
            "rolling_throughput": (
                sum(self._throughput_history)
                / max(len(self._throughput_history), 1)
            ),
        }
