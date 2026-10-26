"""Strict Phase 2B junction-local forecast inference contracts."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from DeepQN.forecasting.contracts import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FEATURE_UNITS, load_approach_mapping,
)
from DeepQN.forecasting.detector_events import (
    DETECTOR_EVENT_SCHEMA_VERSION, E1EventCollector, build_detector_mapping,
    collect_approach_snapshot, detector_feature_row,
)
from DeepQN.forecasting.research_contract import (
    DATASET_CONTRACT_VERSION, MODEL_CONTRACT_VERSION, TIME_CONTRACT_VERSION,
)

FORECAST_INFERENCE_VERSION = "junction-bilstm-60s-to-30s-v2-nonnegative-counts"
RESEARCH_ARCHITECTURE = {
    "contract_version": MODEL_CONTRACT_VERSION,
    "input_size": 7,
    "history_steps": 60,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,
    "output_size": 4,
    "output_activation": "softplus",
}


class ForecastInferenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResearchBiLSTM(nn.Module):
    """BiLSTM producing four junction-local 30-second arrival counts."""
    def __init__(self, hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool = True):
        super().__init__()
        directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(7, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.output = nn.Linear(hidden_size * directions, 4)

    def forward(self, values):
        encoded, _ = self.lstm(values)
        return torch.nn.functional.softplus(self.output(encoded[:, -1, :]))


@dataclass(frozen=True)
class ForecastIdentity:
    model_sha256: str
    scaler_sha256: str
    dataset_manifest_sha256: str
    mapping_sha256: str
    detector_mapping_sha256: str
    network_sha256: str


class NoForecastAdapter:
    """Explicit ablation condition; zeros are intentional, never a fallback."""
    mode = "no_forecast"

    def __init__(self, tls_ids):
        self.tls_ids = tuple(tls_ids)

    def observe(self, tls_id: str, time_s: float, features) -> None:
        if tls_id not in self.tls_ids:
            raise ForecastInferenceError(f"unknown TLS {tls_id!r}")

    def predict_for(self, tls_id: str) -> np.ndarray:
        if tls_id not in self.tls_ids:
            raise ForecastInferenceError(f"unknown TLS {tls_id!r}")
        return np.zeros(4, dtype=np.float32)

    def reset(self):
        return None

    def tick(self):
        return None


class StrictForecastAdapter:
    """Fail-closed seven-feature, 60-second causal forecast adapter."""
    mode = "verified_model"

    def __init__(self, tls_ids, scaler: Mapping, identity: ForecastIdentity,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 output_scale, detector_mapping=None):
        self.tls_ids = tuple(tls_ids)
        if scaler.get("features") != list(FEATURE_NAMES) or scaler.get("units") != list(FEATURE_UNITS):
            raise ForecastInferenceError("scaler feature order or units are incompatible")
        minimum = np.asarray(scaler.get("minimum"), dtype=np.float64)
        maximum = np.asarray(scaler.get("maximum"), dtype=np.float64)
        if minimum.shape != (7,) or maximum.shape != (7,) or not np.all(np.isfinite([minimum, maximum])):
            raise ForecastInferenceError("scaler must contain seven finite minima/maxima")
        if np.any(maximum < minimum):
            raise ForecastInferenceError("scaler maxima cannot be below minima")
        self.minimum, self.maximum = minimum, maximum
        self.output_scale = np.asarray(output_scale, dtype=np.float64)
        if (self.output_scale.shape != (4,) or not np.all(np.isfinite(self.output_scale))
                or np.any(self.output_scale <= 0)):
            raise ForecastInferenceError("DQN forecast-state scale must contain four positive values")
        self.identity, self._predictor = identity, predictor
        self.detector_mapping = detector_mapping
        self._collector = E1EventCollector(detector_mapping) if detector_mapping else None
        self._history = defaultdict(lambda: deque(maxlen=60))
        self._last_time: dict[str, float] = {}

    def observe(self, tls_id: str, time_s: float, features) -> None:
        if tls_id not in self.tls_ids or not math.isfinite(time_s):
            raise ForecastInferenceError("unknown TLS or nonfinite observation time")
        row = np.asarray(features, dtype=np.float64)
        if row.shape != (7,) or not np.all(np.isfinite(row)) or np.any(row < 0):
            raise ForecastInferenceError("forecast observation must be seven finite nonnegative features")
        if not math.isclose(row[4], row[:4].sum(), rel_tol=0, abs_tol=1e-9):
            raise ForecastInferenceError("total feature does not equal directional arrivals")
        previous = self._last_time.get(tls_id)
        if previous is not None and time_s - previous != 1.0:
            raise ForecastInferenceError("forecast history requires consecutive one-second samples")
        self._history[tls_id].append(row)
        self._last_time[tls_id] = time_s

    def predict_for(self, tls_id: str) -> np.ndarray:
        history = self._history.get(tls_id)
        if history is None or len(history) != 60:
            raise ForecastInferenceError("verified forecast requires exactly 60 causal samples")
        values = np.stack(history)
        denominator = np.where(self.maximum > self.minimum,
                               self.maximum - self.minimum, 1.0)
        scaled = ((values - self.minimum) / denominator).astype(np.float32)
        prediction = np.asarray(self._predictor(scaled), dtype=np.float32)
        if prediction.shape != (4,) or not np.all(np.isfinite(prediction)) or np.any(prediction < 0):
            raise ForecastInferenceError("model prediction must be four finite nonnegative counts")
        return np.clip(prediction / self.output_scale, 0, 1).astype(np.float32)

    def history_ready(self) -> bool:
        """Return true only when every configured TLS has 60 observed samples."""
        return all(len(self._history.get(tls_id, ())) == 60 for tls_id in self.tls_ids)

    def validate_runtime_installation(self, connection, network_path: str | Path) -> dict:
        """Fail closed unless the live E1 installation matches the hashed mapping."""
        if self.detector_mapping is None:
            raise ForecastInferenceError("verified_model lacks a validated detector mapping")
        network = Path(network_path).resolve()
        if not network.is_file() or _sha256(network) != self.identity.network_sha256:
            raise ForecastInferenceError("running network does not match forecast network hash")
        installed = set(map(str, connection.inductionloop.getIDList()))
        required = {spec.detector_id for spec in self.detector_mapping.specs}
        missing = sorted(required - installed)
        if missing:
            raise ForecastInferenceError(f"required E1 detectors are not installed: {missing}")
        for spec in self.detector_mapping.specs:
            lane = str(connection.inductionloop.getLaneID(spec.detector_id))
            position = float(connection.inductionloop.getPosition(spec.detector_id))
            if lane != spec.lane_id or not math.isfinite(position) or not math.isclose(
                    position, spec.position_m, rel_tol=0, abs_tol=1e-6):
                raise ForecastInferenceError(
                    f"E1 detector {spec.detector_id!r} installation contradicts mapping"
                )
        return {
            "detector_mapping_sha256": self.identity.detector_mapping_sha256,
            "network_sha256": self.identity.network_sha256,
            "required_detector_ids": sorted(required),
            "verified": True,
        }

    def reset(self):
        self._history.clear(); self._last_time.clear()
        if self.detector_mapping is not None:
            self._collector = E1EventCollector(self.detector_mapping)

    def tick(self):
        if self._collector is None:
            raise ForecastInferenceError("verified_model lacks a validated detector mapping")
        import traci
        starting = tuple(map(str, traci.simulation.getStartingTeleportIDList()))
        ending = tuple(map(str, traci.simulation.getEndingTeleportIDList()))
        if starting or ending:
            raise ForecastInferenceError(
                "teleport intersects verified forecast collection; "
                f"starting={starting}, ending={ending}"
            )
        rows = {spec.detector_id: traci.inductionloop.getVehicleData(spec.detector_id)
                for spec in self.detector_mapping.specs}
        events = self._collector.collect(rows)
        time_s = float(traci.simulation.getTime())
        snapshot = collect_approach_snapshot(traci, self.detector_mapping, time_s)
        features = detector_feature_row(self.detector_mapping, events, snapshot)
        for tls_id in self.tls_ids:
            self.observe(tls_id, time_s, features[tls_id])

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, tls_ids):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "inference_contract_version": FORECAST_INFERENCE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "event_schema_version": DETECTOR_EVENT_SCHEMA_VERSION,
            "features": list(FEATURE_NAMES), "units": list(FEATURE_UNITS),
            "history_seconds": 60, "target_seconds": 30,
            "waiting_time_memory_s": 100,
            "target_shape": [4],
            "target_units": "vehicles per following 30-second interval",
            "output_activation": "softplus",
            "training_loss": "poisson_nll_expected_count",
            "time_contract_version": TIME_CONTRACT_VERSION,
        }
        if any(manifest.get(k) != v for k, v in required.items()):
            raise ForecastInferenceError("forecast manifest contract is incompatible")
        base = path.parent
        artifacts = {}
        for label in ("model", "scaler", "dataset_manifest", "mapping", "network"):
            item = manifest.get("artifacts", {}).get(label, {})
            artifact = (base / str(item.get("path", ""))).resolve()
            if not artifact.is_file() or _sha256(artifact) != item.get("sha256"):
                raise ForecastInferenceError(f"{label} artifact is missing or altered")
            artifacts[label] = artifact
        dataset_manifest = json.loads(artifacts["dataset_manifest"].read_text(encoding="utf-8"))
        dataset_required = {
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "time_contract_version": TIME_CONTRACT_VERSION,
            "history_seconds": 60,
            "target_seconds": 30,
            "target_shape": [4],
        }
        if any(dataset_manifest.get(key) != value for key, value in dataset_required.items()):
            raise ForecastInferenceError("dataset manifest contract is incompatible")
        approach_mapping = load_approach_mapping(artifacts["mapping"], artifacts["network"])
        detector_mapping = build_detector_mapping(approach_mapping, artifacts["network"])
        for label, actual in (("mapping_sha256", approach_mapping.sha256),
                              ("detector_mapping_sha256", detector_mapping.sha256),
                              ("network_sha256", approach_mapping.network_sha256)):
            if manifest.get(label) != actual or dataset_manifest.get(label) != actual:
                raise ForecastInferenceError(f"{label} is not bound to the verified dataset artifacts")
        scaler = json.loads(artifacts["scaler"].read_text(encoding="utf-8"))
        architecture = manifest.get("architecture", {})
        if architecture != RESEARCH_ARCHITECTURE:
            raise ForecastInferenceError("forecast architecture contract is incompatible")
        model = ResearchBiLSTM(
            int(architecture["hidden_size"]), int(architecture["num_layers"]),
            float(architecture["dropout"]), bool(architecture.get("bidirectional", True)))
        try:
            state = torch.load(artifacts["model"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ForecastInferenceError(f"forecast checkpoint is incompatible: {exc}") from exc
        model.eval()
        def predictor(values):
            with torch.no_grad():
                return model(torch.from_numpy(values).unsqueeze(0)).squeeze(0).numpy()
        identity = ForecastIdentity(
            manifest["artifacts"]["model"]["sha256"],
            manifest["artifacts"]["scaler"]["sha256"],
            manifest["artifacts"]["dataset_manifest"]["sha256"],
            manifest["mapping_sha256"], manifest["detector_mapping_sha256"],
            manifest["network_sha256"])
        return cls(tls_ids, scaler, identity, predictor,
                   manifest.get("dqn_state_forecast_scale"), detector_mapping)
