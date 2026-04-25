"""
Training curve plotter using matplotlib.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Plotter:
    """Generate training curve plots from metrics CSV or in-memory data."""

    def __init__(self, save_dir: str = "./plots"):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for Plotter")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def plot_from_csv(
        self,
        csv_path: str,
        metrics: Optional[List[str]] = None,
        window: int = 50,
    ) -> str:
        """
        Read metrics CSV and generate training curves.
        Returns path to saved figure.
        """
        episodes = []
        data: Dict[str, List[float]] = {}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                for k, v in row.items():
                    if k in ("episode", "time"):
                        continue
                    if metrics and k not in metrics:
                        continue
                    if k not in data:
                        data[k] = []
                    try:
                        data[k].append(float(v))
                    except (ValueError, TypeError):
                        data[k].append(0.0)

        if not data:
            return ""

        n_plots = len(data)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), squeeze=False)

        for i, (name, values) in enumerate(data.items()):
            ax = axes[i, 0]
            ax.plot(episodes[: len(values)], values, alpha=0.3, color="steelblue")

            # Smoothed line
            smoothed = self._moving_avg(values, window)
            ax.plot(
                episodes[: len(smoothed)],
                smoothed,
                color="navy",
                linewidth=2,
                label=f"{name} (smoothed)",
            )

            ax.set_xlabel("Episode")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_comparison(
        self,
        data: Dict[str, List[float]],
        title: str = "Scenario Comparison",
        ylabel: str = "Average Reward",
    ) -> str:
        """Plot comparison bar chart across scenarios."""
        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(data.keys())
        means = [sum(v) / len(v) if v else 0 for v in data.values()]

        colors = plt.cm.viridis([i / len(names) for i in range(len(names))])
        bars = ax.bar(names, means, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(names, rotation=45, ha="right")

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        path = os.path.join(self.save_dir, "comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    @staticmethod
    def _moving_avg(values: List[float], window: int) -> List[float]:
        if len(values) < window:
            return values
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start : i + 1]) / (i - start + 1))
        return result
