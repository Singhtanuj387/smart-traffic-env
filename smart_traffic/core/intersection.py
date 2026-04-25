"""
Single intersection physics for the Smart Traffic grid.

Each intersection has 12 approach lanes (3 per cardinal direction —
through, left-turn, right-turn) and a signal phase controller.
Vehicles queue in lanes and pass through when their lane's phase is green.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional

from .vehicle import Vehicle, VehicleType

# ── Phase → lane mapping ─────────────────────────────────────
# Lane indices: N(0,1,2) S(3,4,5) E(6,7,8) W(9,10,11)
#   x+0 = through, x+1 = left, x+2 = right
PHASE_GREEN_LANES: Dict[str, List[int]] = {
    "NS_STRAIGHT_GREEN": [0, 2, 3, 5],      # N/S through + right
    "EW_STRAIGHT_GREEN": [6, 8, 9, 11],      # E/W through + right
    "PROTECTED_NS_LEFT": [1, 4],              # N/S left turn
    "PROTECTED_EW_LEFT": [7, 10],             # E/W left turn
    "ALL_RED_HOLD":      [],                  # everything red
}

# Direction from approach lane to downstream neighbor
# N lanes (0-2): vehicle came from North → goes South (row+1)
# S lanes (3-5): vehicle came from South → goes North (row-1)
# E lanes (6-8): vehicle came from East  → goes West  (col-1)
# W lanes (9-11): from West → goes East (col+1)
LANE_DIRECTION: Dict[int, str] = {}
for _l in range(0, 3):
    LANE_DIRECTION[_l] = "S"
for _l in range(3, 6):
    LANE_DIRECTION[_l] = "N"
for _l in range(6, 9):
    LANE_DIRECTION[_l] = "W"
for _l in range(9, 12):
    LANE_DIRECTION[_l] = "E"

MAX_QUEUE_PER_LANE = 30
NUM_LANES = 12


class Intersection:
    """Manages a single intersection's queues, phases, and vehicle flow."""

    def __init__(self, index: int, row: int, col: int, grid_size: int = 9):
        self.index = index
        self.row = row
        self.col = col
        self.grid_size = grid_size

        # Per-lane vehicle queues
        self.lanes: List[Deque[Vehicle]] = [deque() for _ in range(NUM_LANES)]
        self.current_phase: str = "ALL_RED_HOLD"
        self.phase_elapsed: int = 0
        self.yellow_active: bool = False
        self.yellow_timer: int = 0

        # Per-step bookkeeping
        self._cleared_this_step: int = 0
        self._cleared_vehicles: List[Vehicle] = []

        # Neighbor indices (set by TrafficGrid after construction)
        self.neighbors: Dict[str, Optional[int]] = {
            "N": None, "S": None, "E": None, "W": None
        }

    # ── Reset ─────────────────────────────────────────────────

    def reset(self) -> None:
        for lane in self.lanes:
            lane.clear()
        self.current_phase = "ALL_RED_HOLD"
        self.phase_elapsed = 0
        self.yellow_active = False
        self.yellow_timer = 0
        self._cleared_this_step = 0
        self._cleared_vehicles.clear()

    # ── Queue management ──────────────────────────────────────

    def enqueue(self, vehicle: Vehicle, lane: int) -> bool:
        """Add vehicle to a lane queue. Returns False if queue is full."""
        if len(self.lanes[lane]) >= MAX_QUEUE_PER_LANE:
            return False
        vehicle.lane = lane
        self.lanes[lane].append(vehicle)
        return True

    def get_queue_lengths(self) -> List[float]:
        """Return normalized queue lengths (0-1) for all 12 lanes."""
        return [len(q) / MAX_QUEUE_PER_LANE for q in self.lanes]

    def get_raw_queue_lengths(self) -> List[float]:
        """Return raw vehicle counts for all 12 lanes."""
        return [float(len(q)) for q in self.lanes]

    def get_wait_times(self) -> List[float]:
        """Return avg wait time per lane, normalized to [0,1] (cap at 120s)."""
        result = []
        for q in self.lanes:
            if q:
                avg = sum(v.wait_time for v in q) / len(q)
                result.append(min(avg / 120.0, 1.0))
            else:
                result.append(0.0)
        return result

    def get_raw_avg_wait(self) -> float:
        """Mean wait time across all vehicles at this intersection."""
        total_wait = 0.0
        total_veh = 0
        for q in self.lanes:
            for v in q:
                total_wait += v.wait_time
                total_veh += 1
        return total_wait / max(total_veh, 1)

    def total_vehicles(self) -> int:
        return sum(len(q) for q in self.lanes)

    # ── Phase control ─────────────────────────────────────────

    def apply_phase(self, new_phase: str, force_yellow: bool = False) -> None:
        """
        Set the signal phase. If the phase changes and yellow is not
        already active, the caller should set yellow_active = True externally
        (TrafficEnvironment handles yellow insertion).
        """
        if new_phase != self.current_phase and not self.yellow_active:
            self.current_phase = new_phase
            self.phase_elapsed = 0
        elif new_phase == self.current_phase:
            self.phase_elapsed += 1

    def get_phase_onehot(self) -> List[float]:
        """5-element one-hot of current phase."""
        phases = [
            "NS_STRAIGHT_GREEN", "EW_STRAIGHT_GREEN",
            "PROTECTED_NS_LEFT", "PROTECTED_EW_LEFT", "ALL_RED_HOLD",
        ]
        return [1.0 if self.current_phase == p else 0.0 for p in phases]

    def is_phase_aligned(self, required_phase: str) -> bool:
        """Check if the current phase matches the required one."""
        return self.current_phase == required_phase and not self.yellow_active

    # ── Vehicle movement ──────────────────────────────────────

    def move_vehicles(self, speed_mult: float = 1.0) -> List[Vehicle]:
        """
        Process one tick: clear vehicles from green lanes, increment
        wait on red lanes.  Returns list of vehicles that cleared this
        intersection (ready to enter next node).
        """
        self._cleared_this_step = 0
        self._cleared_vehicles.clear()

        green_lanes = PHASE_GREEN_LANES.get(self.current_phase, [])

        for lane_idx in range(NUM_LANES):
            if self.yellow_active:
                # All lanes red during yellow
                for v in self.lanes[lane_idx]:
                    v.tick_wait()
                continue

            if lane_idx in green_lanes:
                # Process front-of-queue vehicle(s)
                cleared_count = 0
                max_clear = max(1, int(speed_mult * 2))  # vehicles per tick
                while self.lanes[lane_idx] and cleared_count < max_clear:
                    veh = self.lanes[lane_idx][0]
                    if veh.clear_ticks_remaining > 0:
                        if veh.tick_clear():
                            self.lanes[lane_idx].popleft()
                            self._cleared_vehicles.append(veh)
                            self._cleared_this_step += 1
                            cleared_count += 1
                        else:
                            break
                    else:
                        # Start clearing
                        veh.clear_ticks_remaining = max(
                            1, int(veh.occupancy / max(speed_mult, 0.1))
                        )
                        if veh.tick_clear():
                            self.lanes[lane_idx].popleft()
                            self._cleared_vehicles.append(veh)
                            self._cleared_this_step += 1
                            cleared_count += 1
                        else:
                            break
                # Remaining vehicles in green lane still wait
                for v in self.lanes[lane_idx]:
                    v.tick_wait(0.5)  # slower wait in green because queue is moving
            else:
                # Red lane — all wait
                for v in self.lanes[lane_idx]:
                    v.tick_wait()

        return self._cleared_vehicles

    @property
    def cleared_this_step(self) -> int:
        return self._cleared_this_step

    def get_congestion_index(self) -> List[float]:
        """
        4-element congestion per direction: N, S, E, W.
        Normalized ratio of queue fullness.
        """
        result = []
        for start in [0, 3, 6, 9]:
            total = sum(len(self.lanes[start + j]) for j in range(3))
            result.append(total / (3 * MAX_QUEUE_PER_LANE))
        return result
