"""
9×9 Traffic Grid — the central simulation substrate.

Manages 81 intersections, vehicle spawning, routing, movement, and
provides observation accessors used by TrafficEnvironment.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .intersection import Intersection, LANE_DIRECTION, NUM_LANES, MAX_QUEUE_PER_LANE
from .pathfinder import Pathfinder
from .vehicle import Vehicle, VehicleType

# ── Models used for observations (imported lazily to avoid circular) ──
# We only need GlobalMetrics at runtime; import at module top is fine
# because models.py has no imports from core/.


class TrafficGrid:
    """
    9×9 grid of intersections connected by bidirectional roads.
    Responsible for vehicle lifecycle: spawn → route → queue → clear → exit.
    """

    def __init__(self, grid_size: int = 9):
        self.grid_size = grid_size
        self.n_nodes = grid_size * grid_size
        self.pathfinder = Pathfinder(grid_size)

        # Build intersections
        self.intersections: List[Intersection] = []
        for idx in range(self.n_nodes):
            r, c = divmod(idx, grid_size)
            self.intersections.append(Intersection(idx, r, c, grid_size))

        # Wire neighbor pointers
        self._wire_neighbors()

        # Vehicle tracking
        self._next_vehicle_id = 0
        self._active_vehicles: Dict[int, Vehicle] = {}
        self._cleared_this_step = 0
        self._total_cleared = 0

        # Edge nodes for spawning
        self._edge_nodes = self._compute_edge_nodes()

    # ── Reset ─────────────────────────────────────────────────

    def reset(self) -> None:
        for ix in self.intersections:
            ix.reset()
        self._active_vehicles.clear()
        self._next_vehicle_id = 0
        self._cleared_this_step = 0
        self._total_cleared = 0
        self.pathfinder.clear_blocks()

    def close(self) -> None:
        """Cleanup resources."""
        pass

    # ── Vehicle spawning ──────────────────────────────────────

    def spawn_vehicles(
        self,
        spawn_rate_multipliers: Optional[List[List[float]]] = None,
        base_rate: float = 0.15,
        vehicle_type_probs: Optional[Dict[VehicleType, float]] = None,
    ) -> int:
        """
        Spawn vehicles at edge intersections.
        spawn_rate_multipliers: 81×12 per-lane spawn probabilities (multiplied by base_rate).
        Returns number of vehicles spawned.
        """
        if vehicle_type_probs is None:
            vehicle_type_probs = {
                VehicleType.CAR: 0.90,
                VehicleType.TRUCK: 0.08,
                VehicleType.BIKE: 0.02,
            }

        spawned = 0
        for node_idx in range(self.n_nodes):
            ix = self.intersections[node_idx]
            for lane in range(NUM_LANES):
                if spawn_rate_multipliers is not None:
                    rate = base_rate * spawn_rate_multipliers[node_idx][lane]
                else:
                    # Only spawn at edge nodes by default
                    if node_idx in self._edge_nodes:
                        rate = base_rate
                    else:
                        rate = base_rate * 0.05  # very low interior rate

                if random.random() < rate:
                    # Pick destination (random non-same edge node)
                    dest = self._pick_destination(node_idx)
                    if dest is None:
                        continue
                    route = self.pathfinder.find_path(node_idx, dest)
                    if route is None or len(route) < 2:
                        continue

                    # Pick vehicle type
                    vtype = self._pick_vehicle_type(vehicle_type_probs)
                    veh = Vehicle(
                        vehicle_id=self._next_vehicle_id,
                        vehicle_type=vtype,
                        route=route,
                        lane=lane,
                    )
                    self._next_vehicle_id += 1

                    if ix.enqueue(veh, lane):
                        self._active_vehicles[veh.vehicle_id] = veh
                        spawned += 1

        return spawned

    # ── Phase application ────────────────────────────────────

    def apply_phase(self, agent_idx: int, phase: str, yellow_active: bool) -> None:
        ix = self.intersections[agent_idx]
        if not yellow_active:
            ix.apply_phase(phase)
        # Yellow state is managed externally by TrafficEnvironment

    # ── Vehicle movement ─────────────────────────────────────

    def move_vehicles(self, speed_multipliers: Optional[List[float]] = None) -> None:
        """
        Advance all intersections one tick.
        Cleared vehicles are forwarded to their next intersection.
        """
        self._cleared_this_step = 0

        # Process each intersection
        transfers: List[Tuple[Vehicle, int, int]] = []  # (vehicle, from_node, to_lane)
        for idx, ix in enumerate(self.intersections):
            speed = speed_multipliers[idx] if speed_multipliers else 1.0
            cleared = ix.move_vehicles(speed)
            self._cleared_this_step += ix.cleared_this_step

            for veh in cleared:
                veh.advance()
                if veh.has_arrived:
                    # Vehicle exited the network
                    self._active_vehicles.pop(veh.vehicle_id, None)
                    self._total_cleared += 1
                else:
                    next_node = veh.current_node
                    # Determine incoming lane at next intersection
                    to_lane = self._compute_incoming_lane(idx, next_node, veh)
                    transfers.append((veh, next_node, to_lane))

        # Execute transfers
        for veh, to_node, to_lane in transfers:
            ix = self.intersections[to_node]
            if not ix.enqueue(veh, to_lane):
                # Queue full — vehicle is stuck / dropped
                self._active_vehicles.pop(veh.vehicle_id, None)

    # ── Observation accessors ────────────────────────────────

    def get_all_queues(self) -> List[List[float]]:
        return [ix.get_queue_lengths() for ix in self.intersections]

    def get_all_phases(self) -> List[str]:
        return [ix.current_phase for ix in self.intersections]

    def get_queue_obs(self, agent_id: int) -> List[float]:
        return self.intersections[agent_id].get_queue_lengths()

    def get_wait_obs(self, agent_id: int) -> List[float]:
        return self.intersections[agent_id].get_wait_times()

    def get_phase_onehot(self, agent_id: int) -> List[float]:
        return self.intersections[agent_id].get_phase_onehot()

    def get_phase_elapsed(self, agent_id: int) -> int:
        return self.intersections[agent_id].phase_elapsed

    def get_current_phase(self, agent_id: int) -> str:
        return self.intersections[agent_id].current_phase

    def get_queue(self, agent_id: int) -> List[float]:
        return self.intersections[agent_id].get_raw_queue_lengths()

    def get_avg_wait(self, agent_id: int) -> float:
        return self.intersections[agent_id].get_raw_avg_wait()

    def get_neighbor_queues(self, agent_id: int) -> List[float]:
        """
        8-element list: mean queue of each of 4 neighbors (0 if no neighbor),
        plus max queue of each of 4 neighbors.
        """
        ix = self.intersections[agent_id]
        result = []
        for direction in ["N", "S", "E", "W"]:
            nb_idx = ix.neighbors.get(direction)
            if nb_idx is not None:
                qs = self.intersections[nb_idx].get_queue_lengths()
                result.append(sum(qs) / len(qs))
            else:
                result.append(0.0)
        for direction in ["N", "S", "E", "W"]:
            nb_idx = ix.neighbors.get(direction)
            if nb_idx is not None:
                qs = self.intersections[nb_idx].get_queue_lengths()
                result.append(max(qs))
            else:
                result.append(0.0)
        return result

    def get_congestion_index(self, agent_id: int) -> List[float]:
        return self.intersections[agent_id].get_congestion_index()

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Return list of neighbor intersection indices."""
        ix = self.intersections[agent_id]
        return [n for n in ix.neighbors.values() if n is not None]

    def get_cleared_this_step(self) -> int:
        return self._cleared_this_step

    def cars_cleared_this_step(self, agent_id: int) -> int:
        return self.intersections[agent_id].cleared_this_step

    def is_phase_aligned(self, agent_id: int, required_phase: str) -> bool:
        return self.intersections[agent_id].is_phase_aligned(required_phase)

    def get_global_metrics(self):
        """Return GlobalMetrics for the current state."""
        from ..models import GlobalMetrics

        total_wait = 0.0
        total_vehicles = 0
        congested = 0
        for ix in self.intersections:
            total_vehicles += ix.total_vehicles()
            total_wait += ix.get_raw_avg_wait() * ix.total_vehicles()
            if any(len(q) > MAX_QUEUE_PER_LANE * 0.8 for q in ix.lanes):
                congested += 1

        avg_wait = total_wait / max(total_vehicles, 1)
        capacity = self.n_nodes * NUM_LANES * MAX_QUEUE_PER_LANE
        efficiency = 1.0 - (total_vehicles / max(capacity, 1))

        return GlobalMetrics(
            avg_wait_time=avg_wait,
            total_throughput=self._total_cleared,
            network_efficiency=max(0.0, efficiency),
            congestion_count=congested,
        )

    # ── Private helpers ───────────────────────────────────────

    def _wire_neighbors(self) -> None:
        g = self.grid_size
        for ix in self.intersections:
            r, c = ix.row, ix.col
            if r > 0:
                ix.neighbors["N"] = (r - 1) * g + c
            if r < g - 1:
                ix.neighbors["S"] = (r + 1) * g + c
            if c > 0:
                ix.neighbors["W"] = r * g + (c - 1)
            if c < g - 1:
                ix.neighbors["E"] = r * g + (c + 1)

    def _compute_edge_nodes(self) -> set:
        g = self.grid_size
        edges = set()
        for i in range(self.n_nodes):
            r, c = divmod(i, g)
            if r == 0 or r == g - 1 or c == 0 or c == g - 1:
                edges.add(i)
        return edges

    def _pick_destination(self, origin: int) -> Optional[int]:
        candidates = list(self._edge_nodes - {origin})
        if not candidates:
            return None
        return random.choice(candidates)

    def _pick_vehicle_type(self, probs: Dict[VehicleType, float]) -> VehicleType:
        r = random.random()
        cumulative = 0.0
        for vtype, p in probs.items():
            cumulative += p
            if r < cumulative:
                return vtype
        return VehicleType.CAR

    def _compute_incoming_lane(
        self, from_node: int, to_node: int, veh: Vehicle
    ) -> int:
        """Determine which lane a vehicle enters at to_node based on direction."""
        from_ix = self.intersections[from_node]
        to_ix = self.intersections[to_node]

        # Direction of travel
        dr = to_ix.row - from_ix.row
        dc = to_ix.col - from_ix.col

        # Incoming direction determines lane group:
        # Vehicle coming from North (dr=1) → enters S approach lanes (3-5)
        # Vehicle coming from South (dr=-1) → enters N approach lanes (0-2)
        # Vehicle coming from West (dc=1) → enters E approach lanes (6-8)
        # Vehicle coming from East (dc=-1) → enters W approach lanes (9-11)
        if dr == 1:
            base = 0  # entering from North
        elif dr == -1:
            base = 3  # entering from South
        elif dc == 1:
            base = 9  # entering from West
        elif dc == -1:
            base = 6  # entering from East
        else:
            base = 0

        # Sub-lane: through(0), left(1), right(2) — random for now
        sub = random.randint(0, 2)
        return base + sub
