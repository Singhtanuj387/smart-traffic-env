"""
Metrics logging for training — console, CSV, and optional TensorBoard.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Dict, Optional


class MetricsLogger:
    """
    Logs training metrics to console and CSV.
    Optionally integrates with TensorBoard.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        use_tensorboard: bool = False,
        print_interval: int = 10,
    ):
        self.log_dir = log_dir
        self.print_interval = print_interval
        self._step_count = 0
        self._episode_count = 0
        self._start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        # CSV file
        self._csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: TensorBoard not available, logging to CSV only")

        # Running averages
        self._running: Dict[str, list] = defaultdict(list)

    def log_step(self, metrics: Dict[str, float]) -> None:
        """Log per-step metrics."""
        self._step_count += 1
        for k, v in metrics.items():
            self._running[k].append(v)

        if self._tb_writer:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(f"step/{k}", v, self._step_count)

    def log_episode(self, metrics: Dict[str, float]) -> None:
        """Log per-episode metrics."""
        self._episode_count += 1

        # Write to CSV
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=["episode", "time"] + sorted(metrics.keys())
            )
            self._csv_writer.writeheader()

        row = {
            "episode": self._episode_count,
            "time": time.time() - self._start_time,
            **metrics,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(f"episode/{k}", v, self._episode_count)

        # Console output
        if self._episode_count % self.print_interval == 0:
            elapsed = time.time() - self._start_time
            eps_per_sec = self._episode_count / max(elapsed, 1)
            print(
                f"[Episode {self._episode_count:>6d}] "
                f"R={metrics.get('reward', 0):.3f}  "
                f"Wait={metrics.get('avg_wait', 0):.1f}s  "
                f"Through={metrics.get('throughput', 0):.0f}  "
                f"Eff={metrics.get('efficiency', 0):.3f}  "
                f"({eps_per_sec:.1f} eps/s)"
            )

    def get_running_avg(self, key: str, window: int = 100) -> float:
        vals = self._running.get(key, [])
        if not vals:
            return 0.0
        recent = vals[-window:]
        return sum(recent) / len(recent)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        if self._tb_writer:
            self._tb_writer.close()
