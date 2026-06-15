"""
dqn/training/callbacks.py:
Training callbacks: episode logging (CSV + JSON-lines) and early stopping.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


#  Training logger 

class TrainingLogger:
    """
    Writes per-episode metrics to:
      - <log_dir>/training_log.csv    (tabular, easy for pandas/Excel)
      - <log_dir>/training_log.jsonl  (one JSON object per line, full detail)
    """

    CSV_FIELDS = [
        "episode", "profile", "mean_reward", "ep_steps", "ep_time_s",
        "total_wait_s", "total_co2_mg", "epsilon_mean",
    ]

    def __init__(self, log_dir: str = "DeepQN/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path   = self.log_dir / f"training_{timestamp}.csv"
        self._jsonl_path = self.log_dir / f"training_{timestamp}.jsonl"

        self._csv_file  = open(self._csv_path,  "w", newline="")
        self._jsonl_file = open(self._jsonl_path, "w")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._writer.writeheader()

        logger.info("TrainingLogger writing to %s", self.log_dir)

    def log_episode(
        self,
        episode:     int,
        profile:     str,
        mean_reward: float,
        ep_steps:    int,
        ep_time_s:   float,
        agent_stats: Dict[str, Dict],
        info:        Dict[str, Any],
    ):
        epsilon_vals = [
            s["epsilon"] for s in agent_stats.values() if "epsilon" in s
        ]
        eps_mean = float(np.mean(epsilon_vals)) if epsilon_vals else 0.0

        row = {
            "episode":      episode,
            "profile":      profile,
            "mean_reward":  round(mean_reward, 4),
            "ep_steps":     ep_steps,
            "ep_time_s":    round(ep_time_s, 2),
            "total_wait_s": round(info.get("total_wait_s", 0.0), 1),
            "total_co2_mg": round(info.get("total_co2_mg", 0.0), 1),
            "epsilon_mean": round(eps_mean, 4),
        }
        self._writer.writerow(row)
        self._csv_file.flush()

        full_record = {
            **row,
            "agent_stats": agent_stats,
            "sim_time":    info.get("sim_time"),
        }
        self._jsonl_file.write(json.dumps(full_record) + "\n")
        self._jsonl_file.flush()

    def close(self):
        self._csv_file.close()
        self._jsonl_file.close()
        logger.info("TrainingLogger closed.")


# Episode callback 

class EpisodeCallback:
    """
    Hooks called by the training loop at the start/end of each episode.
    Subclass to add custom behaviour (e.g. Weights & Biases logging).
    """

    def on_episode_start(self, episode: int, profile: str): ...

    def on_episode_end(
        self,
        episode:     int,
        profile:     str,
        mean_reward: float,
        info:        dict,
    ): ...


#  Early stopping 

class EarlyStopping:
    """
    Stops training if the mean reward has not improved by ``min_delta``
    for ``patience`` consecutive episodes.

    Usage:
        stopper = EarlyStopping(patience=20, min_delta=0.5)
        ...
        if stopper.should_stop(mean_reward):
            break
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best:      float = float("-inf")
        self.wait:      int   = 0
        self.triggered: bool  = False

    def should_stop(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.triggered = True
                logger.info(
                    "EarlyStopping triggered after %d episodes without improvement.",
                    self.patience,
                )
                return True
        return False

    def reset(self):
        self.best = float("-inf")
        self.wait = 0
        self.triggered = False