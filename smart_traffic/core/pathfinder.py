"""
A* pathfinding on the 9×9 traffic grid.

Computes shortest routes between intersections, supporting dynamic edge
removal (road blocks, network partitions) via a *blocked_edges* set.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple


class Pathfinder:
    """A* router over a grid graph."""

    def __init__(self, grid_size: int = 9):
        self.grid_size = grid_size
        self.n_nodes = grid_size * grid_size
        # Pre-build adjacency list for the full grid
        self._adj: Dict[int, List[int]] = self._build_adjacency()
        # Edges currently blocked (both directions stored)
        self.blocked_edges: Set[Tuple[int, int]] = set()

    # ── Public API ────────────────────────────────────────────

    def find_path(
        self, start: int, goal: int, blocked: Set[Tuple[int, int]] | None = None
    ) -> Optional[List[int]]:
        """
        Return shortest path from *start* to *goal* as list of node indices,
        or None if no path exists.
        """
        if start == goal:
            return [start]
        blocked = blocked if blocked is not None else self.blocked_edges
        open_set: list[Tuple[float, int]] = [(0.0, start)]
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbor in self._adj.get(current, []):
                edge = (current, neighbor)
                rev_edge = (neighbor, current)
                if edge in blocked or rev_edge in blocked:
                    continue
                tentative = g_score[current] + 1.0
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None  # no path

    def block_edge(self, u: int, v: int) -> None:
        self.blocked_edges.add((min(u, v), max(u, v)))

    def unblock_edge(self, u: int, v: int) -> None:
        self.blocked_edges.discard((min(u, v), max(u, v)))

    def clear_blocks(self) -> None:
        self.blocked_edges.clear()

    def get_neighbors(self, node: int) -> List[int]:
        """Return adjacent node indices, respecting blocked edges."""
        result = []
        for nb in self._adj.get(node, []):
            edge = (min(node, nb), max(node, nb))
            if edge not in self.blocked_edges:
                result.append(nb)
        return result

    # ── Private helpers ───────────────────────────────────────

    def _build_adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}
        g = self.grid_size
        for i in range(self.n_nodes):
            r, c = divmod(i, g)
            if r > 0:
                adj[i].append(i - g)  # north
            if r < g - 1:
                adj[i].append(i + g)  # south
            if c > 0:
                adj[i].append(i - 1)  # west
            if c < g - 1:
                adj[i].append(i + 1)  # east
        return adj

    def _heuristic(self, a: int, b: int) -> float:
        """Manhattan distance on grid."""
        g = self.grid_size
        ar, ac = divmod(a, g)
        br, bc = divmod(b, g)
        return float(abs(ar - br) + abs(ac - bc))

    def _reconstruct(self, came_from: Dict[int, int], current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
