"""Reward engine package — OpenEnv Rubric-based composable reward system."""

from .engine import RewardEngine
from .rubrics import (
    build_traffic_rubric,
    build_anti_gaming_gates,
    ThroughputRubric,
    QueueReductionRubric,
    WaitTimePenaltyRubric,
    CoordinationRubric,
    FairnessRubric,
    PhaseStabilityRubric,
    EmergencyRubric,
    CascadePreventionRubric,
    RecoveryRubric,
    AllRedExploitDetector,
    FlickerExploitDetector,
)

__all__ = [
    "RewardEngine",
    "build_traffic_rubric",
    "build_anti_gaming_gates",
    "ThroughputRubric",
    "QueueReductionRubric",
    "WaitTimePenaltyRubric",
    "CoordinationRubric",
    "FairnessRubric",
    "PhaseStabilityRubric",
    "EmergencyRubric",
    "CascadePreventionRubric",
    "RecoveryRubric",
    "AllRedExploitDetector",
    "FlickerExploitDetector",
]
