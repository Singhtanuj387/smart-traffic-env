"""
Real-time grid renderer using pygame.

Displays the 9×9 traffic grid with color-coded congestion,
signal phases, and vehicle counts.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import TrafficObservation

try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# Color palette
BLACK = (20, 20, 30)
WHITE = (240, 240, 245)
DARK_GRAY = (60, 60, 70)
GREEN = (50, 205, 100)
YELLOW = (255, 220, 50)
RED = (220, 60, 60)
BLUE = (60, 120, 220)
ORANGE = (255, 160, 40)
PURPLE = (150, 80, 200)

PHASE_COLORS = {
    "NS_STRAIGHT_GREEN": GREEN,
    "EW_STRAIGHT_GREEN": BLUE,
    "PROTECTED_NS_LEFT": ORANGE,
    "PROTECTED_EW_LEFT": PURPLE,
    "ALL_RED_HOLD": RED,
}


class GridRenderer:
    """Pygame-based real-time traffic grid visualization."""

    def __init__(
        self,
        grid_size: int = 9,
        cell_size: int = 70,
        margin: int = 40,
    ):
        if not HAS_PYGAME:
            raise ImportError("pygame is required for GridRenderer")

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        self.width = grid_size * cell_size + 2 * margin
        self.height = grid_size * cell_size + 2 * margin + 60  # info bar

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Smart Traffic — 9×9 MARL Grid")
        self.font = pygame.font.SysFont("monospace", 12)
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)
        self.clock = pygame.time.Clock()

    def render(self, obs: "TrafficObservation") -> bool:
        """
        Render one frame. Returns False if window was closed.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BLACK)

        # Draw grid roads
        self._draw_roads()

        # Draw intersections
        for ag in obs.agents:
            self._draw_intersection(ag)

        # Draw info bar
        self._draw_info(obs)

        pygame.display.flip()
        self.clock.tick(30)
        return True

    def close(self):
        pygame.quit()

    def _draw_roads(self):
        """Draw road lines between intersections."""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = self.margin + c * self.cell_size + self.cell_size // 2
                y = self.margin + r * self.cell_size + self.cell_size // 2

                if c < self.grid_size - 1:
                    x2 = x + self.cell_size
                    pygame.draw.line(self.screen, DARK_GRAY, (x, y), (x2, y), 3)

                if r < self.grid_size - 1:
                    y2 = y + self.cell_size
                    pygame.draw.line(self.screen, DARK_GRAY, (x, y), (x, y2), 3)

    def _draw_intersection(self, ag):
        """Draw a single intersection with congestion coloring."""
        idx = ag.agent_id
        r, c = divmod(idx, self.grid_size)
        x = self.margin + c * self.cell_size + self.cell_size // 2
        y = self.margin + r * self.cell_size + self.cell_size // 2

        # Congestion color (green → yellow → red)
        avg_q = sum(ag.queue_lengths) / len(ag.queue_lengths)
        color = self._congestion_color(avg_q)

        # Phase indicator as outline
        phase_idx = ag.current_phase.index(1.0) if 1.0 in ag.current_phase else 4
        phases = [
            "NS_STRAIGHT_GREEN", "EW_STRAIGHT_GREEN",
            "PROTECTED_NS_LEFT", "PROTECTED_EW_LEFT", "ALL_RED_HOLD",
        ]
        phase_color = PHASE_COLORS.get(phases[phase_idx], RED)

        # Draw circle
        radius = int(self.cell_size * 0.35)
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, phase_color, (x, y), radius, 3)

        # Yellow indicator
        if ag.yellow_active > 0.5:
            pygame.draw.circle(self.screen, YELLOW, (x, y), radius + 4, 2)

        # Queue count text
        total_q = sum(ag.queue_lengths) * 30  # denormalize
        text = self.font.render(f"{int(total_q)}", True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)

    def _draw_info(self, obs: "TrafficObservation"):
        """Draw info bar at bottom."""
        y = self.height - 55
        gm = obs.global_metrics

        texts = [
            f"Step: {obs.step:>5d}",
            f"Wait: {gm.avg_wait_time:.1f}s",
            f"Through: {gm.total_throughput}",
            f"Eff: {gm.network_efficiency:.2%}",
            f"Congested: {gm.congestion_count}",
        ]

        scenarios = obs.scenario_flags.active_scenarios
        if scenarios:
            texts.append(f"Scenario: {','.join(scenarios)}")

        full_text = "  |  ".join(texts)
        surface = self.font_large.render(full_text, True, WHITE)
        self.screen.blit(surface, (10, y))

        # Reward bar
        reward = obs.reward if obs.reward is not None else 0.0
        bar_width = int(max(-2, min(reward, 5)) / 5 * 200) + 100
        bar_color = GREEN if reward > 0 else RED
        pygame.draw.rect(self.screen, bar_color, (10, y + 25, max(bar_width, 2), 12))
        rtext = self.font.render(f"R={reward:.3f}", True, WHITE)
        self.screen.blit(rtext, (220, y + 23))

    @staticmethod
    def _congestion_color(normalized_q: float) -> tuple:
        """Map 0-1 congestion to green→yellow→red."""
        if normalized_q < 0.3:
            return (50, 180, 80)
        elif normalized_q < 0.6:
            g = int(220 - normalized_q * 200)
            return (220, g, 40)
        else:
            r = min(255, int(180 + normalized_q * 75))
            return (r, 40, 40)
