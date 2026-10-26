"""Phase 3.0B research-only forecast, split, baseline and scoring contracts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from DeepQN.forecasting.contracts import ForecastContractError

TIME_CONTRACT_VERSION = "sumo-e1-post-step-right-closed-v1"
DATASET_CONTRACT_VERSION = "causal-junction-arrivals-60s-to-30s-v4-right-closed"
MODEL_CONTRACT_VERSION = "junction-bilstm-count-mean-v2"
SCORING_CONTRACT_VERSION = "junction-count-scoring-v1"
SCIENTIFIC_POOLS = (
    "training", "calibration", "validation", "forecast_test", "controller_evaluation"
)


def reporting_sample_time(entry_time_s: float, first_sample_s: float,
                          last_sample_s: float) -> float:
    """Map an E1 entry to its post-step sample label.

    A sample labelled ``s`` represents the completed one-second interval ``(s-1,s]``.
    Only the first recorded interval includes its left boundary, allowing an entry at the
    simulation begin time to be represented.  Exact interior boundaries belong to the
    interval that just completed, unlike ``floor(entry)+1``.
    """
    values = (entry_time_s, first_sample_s, last_sample_s)
    if any(not math.isfinite(value) for value in values):
        raise ForecastContractError("event/sample boundaries must be finite")
    if last_sample_s < first_sample_s:
        raise ForecastContractError("recorded sample interval is reversed")
    if entry_time_s == first_sample_s - 1.0:
        result = first_sample_s
    else:
        result = float(math.ceil(entry_time_s))
    if result < first_sample_s or result > last_sample_s:
        raise ForecastContractError(
            f"event at {entry_time_s} has no recorded sample bin"
        )
    return result


def poisson_count_loss(expected_counts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stable Poisson NLL for nonnegative expected 30-second counts."""
    if expected_counts.shape != targets.shape or expected_counts.ndim != 2 \
            or expected_counts.shape[-1] != 4:
        raise ForecastContractError("count loss requires matching [batch,4] tensors")
    if (not torch.isfinite(expected_counts).all() or not torch.isfinite(targets).all()
            or torch.any(expected_counts < 0) or torch.any(targets < 0)):
        raise ForecastContractError("count loss inputs must be finite and nonnegative")
    return F.poisson_nll_loss(expected_counts, targets, log_input=False,
                              full=False, eps=1e-8, reduction="mean")


def forecast_baselines(history: np.ndarray, *, historical_mean: np.ndarray | None = None,
                       moving_average_seconds: int = 30) -> dict[str, np.ndarray]:
    """Return causal four-direction 30-second count baselines from a [60,7] history."""
    values = np.asarray(history, dtype=np.float64)
    if values.shape != (60, 7) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ForecastContractError("baseline history must be finite nonnegative [60,7]")
    if not isinstance(moving_average_seconds, int) or not 1 <= moving_average_seconds <= 60:
        raise ForecastContractError("moving-average lookback must be an integer in [1,60]")
    arrivals = values[:, :4]
    result = {
        "zero": np.zeros(4, dtype=np.float64),
        "persistence": arrivals[-30:].sum(axis=0),
        "moving_average_rate": arrivals[-moving_average_seconds:].mean(axis=0) * 30.0,
    }
    if historical_mean is not None:
        mean = np.asarray(historical_mean, dtype=np.float64)
        if mean.shape != (4,) or not np.all(np.isfinite(mean)) or np.any(mean < 0):
            raise ForecastContractError("historical mean must be four finite nonnegative counts")
        result["historical_mean"] = mean.copy()
    return result


def fit_historical_mean(training_targets: np.ndarray, *, fit_pool: str = "training") -> np.ndarray:
    if fit_pool != "training":
        raise ForecastContractError("historical mean may be fit only from the training pool")
    targets = np.asarray(training_targets, dtype=np.float64)
    if (targets.ndim != 2 or targets.shape[1] != 4 or len(targets) == 0
            or not np.all(np.isfinite(targets)) or np.any(targets < 0)):
        raise ForecastContractError("historical mean requires finite nonnegative training [n,4]")
    return targets.mean(axis=0)


def score_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Score observations without percentage denominators; aggregate by realization upstream."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if (pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 4 or len(pred) == 0
            or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth))
            or np.any(pred < 0) or np.any(truth < 0)):
        raise ForecastContractError("scores require matching finite nonnegative [n,4] arrays")
    error = pred - truth
    eps = 1e-8
    poisson_deviance = np.empty_like(truth)
    positive = truth > 0
    poisson_deviance[~positive] = 2.0 * pred[~positive]
    poisson_deviance[positive] = 2.0 * (
        truth[positive] * np.log(truth[positive] / np.maximum(pred[positive], eps))
        - (truth[positive] - pred[positive])
    )
    return {
        "contract_version": SCORING_CONTRACT_VERSION,
        "n_windows": int(len(pred)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "directional_bias": error.mean(axis=0).tolist(),
        "mean_poisson_deviance": float(np.mean(poisson_deviance)),
        "percentage_error": None,
        "percentage_error_unavailable_reason": "undefined for zero-count denominators",
    }


def score_by_realization(prediction: np.ndarray, target: np.ndarray,
                         realization_ids: Iterable[str]) -> dict[str, dict[str, object]]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    ids = tuple(realization_ids)
    if pred.shape != truth.shape or pred.ndim != 2 or len(ids) != len(pred) or any(not i for i in ids):
        raise ForecastContractError("realization scoring inputs are inconsistent")
    return {
        identity: score_counts(pred[[index for index, value in enumerate(ids)
                                     if value == identity]],
                               truth[[index for index, value in enumerate(ids)
                                      if value == identity]])
        for identity in sorted(set(ids))
    }


@dataclass(frozen=True)
class PoolIdentity:
    realization_id: str
    pool: str
    source_sha256: str
    physical_fingerprint_sha256: str


def validate_scientific_pools(rows: Iterable[PoolIdentity]) -> tuple[PoolIdentity, ...]:
    """Fail if any physical realization, source, or identity crosses scientific pools."""
    values = tuple(rows)
    if not values:
        raise ForecastContractError("scientific split registry is empty")
    seen_ids: dict[str, PoolIdentity] = {}
    seen_sources: dict[str, PoolIdentity] = {}
    seen_physical: dict[str, PoolIdentity] = {}
    for row in values:
        if row.pool not in SCIENTIFIC_POOLS or not row.realization_id:
            raise ForecastContractError("split registry has an invalid pool or realization ID")
        for label, digest in (("source", row.source_sha256),
                              ("physical fingerprint", row.physical_fingerprint_sha256)):
            if not isinstance(digest, str) or len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest.lower()):
                raise ForecastContractError(f"invalid {label} SHA-256")
        for table, key, label in (
            (seen_ids, row.realization_id, "realization ID"),
            (seen_sources, row.source_sha256, "source content"),
            (seen_physical, row.physical_fingerprint_sha256, "physical demand"),
        ):
            prior = table.get(key)
            if prior is not None:
                raise ForecastContractError(
                    f"{label} reused by {prior.pool!r} and {row.pool!r}"
                )
            table[key] = row
    if {row.pool for row in values} != set(SCIENTIFIC_POOLS):
        raise ForecastContractError("split registry must contain every scientific pool")
    return values
