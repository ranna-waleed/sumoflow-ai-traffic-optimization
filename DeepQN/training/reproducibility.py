"""Fail-closed reproducibility records for future preregistered training."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

TRAINING_RUN_SCHEMA_VERSION = "sumoflow-training-run-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int, deterministic_torch: bool = True) -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
    return {"python": seed, "numpy": seed, "torch": seed,
            "cuda": seed if torch.cuda.is_available() else None,
            "deterministic_torch": deterministic_torch}


@dataclass(frozen=True)
class ResearchRunContract:
    training_seed: int
    sumo_seed: int
    demand_realization_id: str
    demand_sha256: str
    config_sha256: str
    dataset_manifest_sha256: str | None
    forecast_mode: str
    git_commit: str
    dirty_worktree: bool

    def validate(self) -> None:
        if self.forecast_mode not in {"no_forecast", "verified_model"}:
            raise ValueError("research training requires an explicit forecast condition")
        if not self.demand_realization_id:
            raise ValueError("demand realization identity is required")
        for name, value in (("demand", self.demand_sha256), ("config", self.config_sha256)):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"{name} SHA-256 is invalid")

    def write_once(self, path: str | Path) -> str:
        self.validate(); destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite run contract: {destination}")
        payload = {"schema_version": TRAINING_RUN_SCHEMA_VERSION, **asdict(self),
                   "software": {"python": os.sys.version, "numpy": np.__version__,
                                "torch": torch.__version__}}
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
