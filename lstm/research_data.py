"""Causal, leakage-safe dataset preparation for the Phase 2 forecast contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from DeepQN.forecasting.contracts import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_UNITS,
    ENTRY_EVENT_DEFINITION,
    WAITING_TIME_MEMORY_S,
    ApproachMapping,
    ForecastContractError,
    JunctionFeatureExtractor,
    TrafficSnapshot,
    VehicleObservation,
    load_approach_mapping,
)

HISTORY_SECONDS = 60
TARGET_SECONDS = 30
PARTITIONS = ("train", "validation", "test")
DATASET_CONTRACT_VERSION = "causal-junction-arrivals-60s-to-30s-v2-provisional"


@dataclass(frozen=True)
class SequenceSource:
    source_id: str
    partition: str
    snapshots: tuple[TrafficSnapshot, ...]
    demand_seed: int
    sumo_config_sha256: str
    source_sha256: str
    demand_realization_id: str
    demand_sha256: str
    waiting_time_memory_s: float = WAITING_TIME_MEMORY_S


@dataclass(frozen=True)
class WindowMetadata:
    source_id: str
    tls_id: str
    input_start_s: float
    input_end_s: float
    target_start_s: float
    target_end_s: float
    partition: str


@dataclass
class PartitionData:
    x: np.ndarray
    y: np.ndarray
    windows: list[WindowMetadata]


def _hash_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_snapshot_jsonl(path: str | Path) -> tuple[TrafficSnapshot, ...]:
    snapshots = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                vehicles = tuple(VehicleObservation(
                    vehicle_id=str(vehicle["id"]),
                    edge_id=str(vehicle["edge"]),
                    speed_m_s=float(vehicle["speed_m_s"]),
                    accumulated_waiting_s=float(vehicle["accumulated_waiting_s"]),
                ) for vehicle in row["vehicles"])
                snapshots.append(TrafficSnapshot(float(row["time_s"]), vehicles))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ForecastContractError(f"invalid snapshot {path}:{line_number}: {exc}") from exc
    if not snapshots:
        raise ForecastContractError(f"snapshot source is empty: {path}")
    return tuple(snapshots)


def _validate_sources(sources: Iterable[SequenceSource]) -> list[SequenceSource]:
    values = list(sources)
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    demand_ids: dict[str, tuple[str, str]] = {}
    demand_hashes: dict[str, str] = {}
    present = set()
    for source in values:
        if not source.source_id or source.source_id in ids:
            raise ForecastContractError(f"missing or duplicate source id {source.source_id!r}")
        ids.add(source.source_id)
        if source.partition not in PARTITIONS:
            raise ForecastContractError(f"unknown partition {source.partition!r}")
        present.add(source.partition)
        prior = hashes.get(source.source_sha256)
        if prior is not None:
            raise ForecastContractError(
                f"source content reused across partitions/sources: {prior!r}, {source.source_id!r}"
            )
        hashes[source.source_sha256] = source.source_id
        if not source.demand_realization_id or not source.demand_sha256:
            raise ForecastContractError(f"source {source.source_id!r} lacks frozen demand provenance")
        prior_id = demand_ids.get(source.demand_realization_id)
        if prior_id is not None and prior_id != (source.demand_sha256, source.partition):
            raise ForecastContractError(
                f"demand realization id has conflicting hash or partition: {source.demand_realization_id!r}"
            )
        prior_partition = demand_hashes.get(source.demand_sha256)
        if prior_partition is not None and prior_partition != source.partition:
            raise ForecastContractError(
                f"demand realization reused across partitions: {source.demand_realization_id!r}"
            )
        demand_ids[source.demand_realization_id] = (source.demand_sha256, source.partition)
        demand_hashes[source.demand_sha256] = source.partition
        if not math.isclose(source.waiting_time_memory_s, WAITING_TIME_MEMORY_S, abs_tol=1e-9):
            raise ForecastContractError(
                f"source {source.source_id!r} waiting-time memory must be {WAITING_TIME_MEMORY_S}s"
            )
        if not source.sumo_config_sha256:
            raise ForecastContractError(f"source {source.source_id!r} has no SUMO config hash")
    if present != set(PARTITIONS):
        raise ForecastContractError(f"all partitions are required; present={sorted(present)}")
    return values


def build_causal_windows(
    sources: Iterable[SequenceSource], mapping: ApproachMapping
) -> dict[str, PartitionData]:
    """Build windows within sources; no window may cross a source partition boundary."""
    sources = _validate_sources(sources)
    collected = {partition: {"x": [], "y": [], "windows": []} for partition in PARTITIONS}
    for source in sources:
        extractor = JunctionFeatureExtractor(mapping)
        per_tls = {tls_id: [] for tls_id in mapping.tls_directions}
        times = []
        for snapshot in source.snapshots:
            rows = extractor.extract(snapshot)
            times.append(snapshot.time_s)
            for tls_id, row in rows.items():
                per_tls[tls_id].append(row)
        if len(times) < HISTORY_SECONDS + TARGET_SECONDS:
            raise ForecastContractError(f"source {source.source_id!r} is too short")
        for tls_id, rows in per_tls.items():
            values = np.asarray(rows, dtype=np.float64)
            for input_end in range(HISTORY_SECONDS - 1, len(values) - TARGET_SECONDS):
                target_start = input_end + 1
                target_end = input_end + TARGET_SECONDS
                x = values[input_end - HISTORY_SECONDS + 1:input_end + 1]
                y = values[target_start:target_end + 1, :4].sum(axis=0)
                metadata = WindowMetadata(
                    source_id=source.source_id,
                    tls_id=tls_id,
                    input_start_s=times[input_end - HISTORY_SECONDS + 1],
                    input_end_s=times[input_end],
                    target_start_s=times[target_start],
                    target_end_s=times[target_end],
                    partition=source.partition,
                )
                if metadata.target_start_s <= metadata.input_end_s:
                    raise ForecastContractError("future target overlaps model input")
                collected[source.partition]["x"].append(x)
                collected[source.partition]["y"].append(y)
                collected[source.partition]["windows"].append(metadata)
    result = {}
    for partition, data in collected.items():
        if not data["x"]:
            raise ForecastContractError(f"partition {partition!r} contains no windows")
        result[partition] = PartitionData(
            x=np.asarray(data["x"], dtype=np.float64),
            y=np.asarray(data["y"], dtype=np.float64),
            windows=data["windows"],
        )
    return result


def fit_training_scaler(train_x: np.ndarray) -> dict:
    if train_x.ndim != 3 or train_x.shape[-1] != len(FEATURE_NAMES):
        raise ForecastContractError("training tensor has incompatible feature shape")
    minimum = train_x.reshape(-1, train_x.shape[-1]).min(axis=0)
    maximum = train_x.reshape(-1, train_x.shape[-1]).max(axis=0)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ForecastContractError("non-finite scaler values")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "units": list(FEATURE_UNITS),
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "fit_partition": "train",
    }


def transform_features(x: np.ndarray, scaler: dict) -> np.ndarray:
    if scaler.get("features") != list(FEATURE_NAMES) or scaler.get("units") != list(FEATURE_UNITS):
        raise ForecastContractError("scaler feature schema is incompatible")
    minimum = np.asarray(scaler["minimum"], dtype=np.float64)
    maximum = np.asarray(scaler["maximum"], dtype=np.float64)
    denominator = np.where(maximum > minimum, maximum - minimum, 1.0)
    transformed = (x - minimum) / denominator
    if not np.all(np.isfinite(transformed)):
        raise ForecastContractError("scaled features are non-finite")
    return transformed


def require_publication_training_ready(manifest: dict) -> None:
    """Fail closed until exact physical boundary-event labels are validated."""
    if manifest.get("publication_training_ready") is not True:
        raise ForecastContractError(
            "dataset is not approved for publication-grade training: "
            + str(manifest.get("publication_training_blocker", "unknown blocker"))
        )


def write_dataset(
    output_dir: str | Path,
    partitions: dict[str, PartitionData],
    sources: Iterable[SequenceSource],
    mapping: ApproachMapping,
) -> dict:
    destination = Path(output_dir)
    targets = [destination / f"{name}.npz" for name in PARTITIONS]
    targets += [destination / "scaler.json", destination / "manifest.json"]
    collisions = [str(path) for path in targets if path.exists()]
    if collisions:
        raise FileExistsError("refusing to overwrite dataset artifacts: " + ", ".join(collisions))
    destination.mkdir(parents=True, exist_ok=True)
    scaler = fit_training_scaler(partitions["train"].x)
    for name in PARTITIONS:
        data = partitions[name]
        np.savez_compressed(
            destination / f"{name}.npz",
            x=transform_features(data.x, scaler),
            y=data.y,
            source_id=np.asarray([row.source_id for row in data.windows]),
            tls_id=np.asarray([row.tls_id for row in data.windows]),
            input_start_s=np.asarray([row.input_start_s for row in data.windows]),
            input_end_s=np.asarray([row.input_end_s for row in data.windows]),
            target_start_s=np.asarray([row.target_start_s for row in data.windows]),
            target_end_s=np.asarray([row.target_end_s for row in data.windows]),
            partition=np.asarray([row.partition for row in data.windows]),
        )
    (destination / "scaler.json").write_text(
        json.dumps(scaler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_rows = [{
        "source_id": source.source_id,
        "partition": source.partition,
        "source_sha256": source.source_sha256,
        "demand_seed": source.demand_seed,
        "sumo_config_sha256": source.sumo_config_sha256,
        "demand_realization_id": source.demand_realization_id,
        "demand_sha256": source.demand_sha256,
        "waiting_time_memory_s": source.waiting_time_memory_s,
    } for source in sources]
    manifest = {
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "units": list(FEATURE_UNITS),
        "history_seconds": HISTORY_SECONDS,
        "target_seconds": TARGET_SECONDS,
        "target_definition": "directional approach entries in (t,t+30], vehicles/30s",
        "entry_event_definition": ENTRY_EVENT_DEFINITION,
        "publication_training_ready": False,
        "publication_training_blocker": "one-second snapshot differences are not validated as complete physical boundary crossings",
        "waiting_feature_definition": "mean TraCI vehicle.getAccumulatedWaitingTime over vehicles present on mapped approach edges at sample time",
        "waiting_time_memory_s": WAITING_TIME_MEMORY_S,
        "mapping_sha256": mapping.sha256,
        "network_sha256": mapping.network_sha256,
        "sources": source_rows,
        "partition_windows": {name: len(partitions[name].windows) for name in PARTITIONS},
    }
    artifact_hashes = {}
    for name in (*PARTITIONS, "scaler"):
        path = destination / (f"{name}.npz" if name in PARTITIONS else "scaler.json")
        artifact_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["artifact_sha256"] = artifact_hashes
    manifest["manifest_payload_sha256"] = _hash_json(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def generate_dataset_from_manifest(spec_path: str | Path) -> dict:
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    mapping_path = (base / spec["mapping_path"]).resolve()
    net_path = (base / spec["network_path"]).resolve()
    mapping = load_approach_mapping(mapping_path, net_path)
    sources = []
    for row in spec["sources"]:
        source_path = (base / row["path"]).resolve()
        config_path = (base / row["sumo_config_path"]).resolve()
        sources.append(SequenceSource(
            source_id=row["source_id"],
            partition=row["partition"],
            snapshots=load_snapshot_jsonl(source_path),
            demand_seed=int(row["demand_seed"]),
            sumo_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            demand_realization_id=str(row["demand_realization_id"]),
            demand_sha256=str(row["demand_sha256"]),
            waiting_time_memory_s=float(row.get("waiting_time_memory_s", WAITING_TIME_MEMORY_S)),
        ))
    partitions = build_causal_windows(sources, mapping)
    return write_dataset((base / spec["output_dir"]).resolve(), partitions, sources, mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="JSON generation manifest")
    args = parser.parse_args()
    print(json.dumps(generate_dataset_from_manifest(args.spec), indent=2))
