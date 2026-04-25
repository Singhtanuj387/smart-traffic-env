"""
Vehicle movement model for the Smart Traffic grid.

Each vehicle has a pre-computed route, a current position on that route,
and per-step wait tracking.  Vehicle types (car, truck, bike) are modelled
via *clear_time* — the number of green-phase ticks needed to clear a
single intersection.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List


class VehicleType(enum.Enum):
    CAR = "car"
    TRUCK = "truck"
    BIKE = "bike"


# Clear-time multiplier relative to a standard car (1 tick).
CLEAR_TIME: dict[VehicleType, int] = {
    VehicleType.CAR: 1,
    VehicleType.TRUCK: 3,
    VehicleType.BIKE: 1,
}

# How much queue capacity a vehicle occupies (trucks are bigger).
QUEUE_OCCUPANCY: dict[VehicleType, float] = {
    VehicleType.CAR: 1.0,
    VehicleType.TRUCK: 2.5,
    VehicleType.BIKE: 0.4,
}


@dataclass
class Vehicle:
    """A single vehicle traversing the 9×9 grid."""

    vehicle_id: int
    vehicle_type: VehicleType = VehicleType.CAR
    route: List[int] = field(default_factory=list)  # sequence of intersection indices
    route_idx: int = 0  # index into route — 0 means at origin
    wait_time: float = 0.0  # cumulative wait at current intersection
    lane: int = 0  # approach lane index (0-11)
    clear_ticks_remaining: int = 0  # ticks still needed to pass through
    is_emergency: bool = False

    # ── Derived helpers ────────────────────────────────────────

    @property
    def current_node(self) -> int:
        """Intersection the vehicle is currently at (or heading toward)."""
        if self.route_idx < len(self.route):
            return self.route[self.route_idx]
        return self.route[-1] if self.route else -1

    @property
    def next_node(self) -> int | None:
        """Next intersection on the route, or None if at destination."""
        nxt = self.route_idx + 1
        if nxt < len(self.route):
            return self.route[nxt]
        return None

    @property
    def has_arrived(self) -> bool:
        return self.route_idx >= len(self.route) - 1 and self.clear_ticks_remaining <= 0

    @property
    def occupancy(self) -> float:
        return QUEUE_OCCUPANCY[self.vehicle_type]

    def advance(self) -> None:
        """Move to the next node on the route."""
        if self.route_idx < len(self.route) - 1:
            self.route_idx += 1
            self.wait_time = 0.0
            self.clear_ticks_remaining = CLEAR_TIME[self.vehicle_type]

    def tick_wait(self, dt: float = 1.0) -> None:
        """Increment wait time while stuck at a red or yellow phase."""
        self.wait_time += dt

    def tick_clear(self) -> bool:
        """Decrement clear counter; return True when fully cleared."""
        if self.clear_ticks_remaining > 0:
            self.clear_ticks_remaining -= 1
        return self.clear_ticks_remaining <= 0
