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
from DeepQN.forecasting.detector_events import (
    DETECTOR_EVENT_SCHEMA_VERSION, ApproachEntryEvent, DetectorMapping,
    build_detector_mapping, detector_feature_row, load_detector_events,
    validate_detector_events,
)

HISTORY_SECONDS = 60
TARGET_SECONDS = 30
PARTITIONS = ("train", "validation", "test")
DATASET_CONTRACT_VERSION = "causal-junction-arrivals-60s-to-30s-v3-detector-capable"

# Deliberately empty until an independently reviewed approval record is added
# by a separate, human-reviewed repository change. A caller-supplied boolean or
# approval JSON can never add itself to this trust root.
TRUSTED_MAPPING_APPROVALS: dict[str, dict[str, str]] = {}
MAPPING_APPROVAL_SCHEMA_VERSION = "traffic-engineering-mapping-approval-v1"


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
    detector_events: tuple[ApproachEntryEvent, ...] | None = None
    detector_mapping_sha256: str | None = None
    snapshot_path: str | None = None
    sumo_config_path: str | None = None
    demand_path: str | None = None
    event_source_path: str | None = None
    event_source_sha256: str | None = None
    event_interval_policy: str = "reject_outside_recorded_interval"


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


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_waiting_memory(config_path: str | Path) -> None:
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(config_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ForecastContractError(f"cannot inspect SUMO waiting-time contract: {exc}") from exc
    node = root.find(".//waiting-time-memory")
    if node is None or node.get("value") is None:
        raise ForecastContractError("SUMO config must explicitly set waiting-time-memory=100")
    try:
        value = float(node.get("value"))
    except ValueError as exc:
        raise ForecastContractError("invalid waiting-time-memory value") from exc
    if value != WAITING_TIME_MEMORY_S:
        raise ForecastContractError(f"waiting-time-memory must equal {WAITING_TIME_MEMORY_S}")


def _validate_sources(sources: Iterable[SequenceSource]) -> list[SequenceSource]:
    values = list(sources)
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    demand_ids: dict[str, tuple[str, str]] = {}
    demand_hashes: dict[str, str] = {}
    present = set()
    modes = set()
    for source in values:
        if not source.source_id or source.source_id in ids:
            raise ForecastContractError(f"missing or duplicate source id {source.source_id!r}")
        ids.add(source.source_id)
        if source.partition not in PARTITIONS:
            raise ForecastContractError(f"unknown partition {source.partition!r}")
        present.add(source.partition)
        modes.add("sumo_e1" if source.detector_events is not None else "snapshot_inference")
        times = [snapshot.time_s for snapshot in source.snapshots]
        if not times or any(not math.isfinite(value) for value in times):
            raise ForecastContractError(f"source {source.source_id!r} has nonfinite or missing timestamps")
        if any(after - before != 1.0 for before, after in zip(times, times[1:])):
            raise ForecastContractError(
                f"source {source.source_id!r} must contain monotonic consecutive one-second timestamps"
            )
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
        for label, digest in (
            ("snapshot", source.source_sha256),
            ("demand", source.demand_sha256),
            ("SUMO config", source.sumo_config_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise ForecastContractError(
                    f"source {source.source_id!r} has invalid {label} SHA-256"
                )
        if source.detector_events is not None:
            if source.event_interval_policy != "reject_outside_recorded_interval":
                raise ForecastContractError("unsupported detector-event crop policy")
            paths = {
                "snapshot": (source.snapshot_path, source.source_sha256),
                "SUMO config": (source.sumo_config_path, source.sumo_config_sha256),
                "demand": (source.demand_path, source.demand_sha256),
                "event": (source.event_source_path, source.event_source_sha256),
            }
            for label, (path, expected_hash) in paths.items():
                if not path or not Path(path).is_file():
                    raise ForecastContractError(
                        f"detector source {source.source_id!r} lacks actual {label} file"
                    )
                if _file_sha256(path) != expected_hash:
                    raise ForecastContractError(
                        f"detector source {source.source_id!r} {label} hash mismatch"
                    )
            authoritative_snapshots = load_snapshot_jsonl(source.snapshot_path)
            if authoritative_snapshots != source.snapshots:
                raise ForecastContractError(
                    f"detector source {source.source_id!r} in-memory snapshots differ from verified file"
                )
            _validate_waiting_memory(source.sumo_config_path)
    if present != set(PARTITIONS):
        raise ForecastContractError(f"all partitions are required; present={sorted(present)}")
    if len(modes) != 1:
        raise ForecastContractError("event-source mode must be consistent across all partitions")
    return values


def build_causal_windows(
    sources: Iterable[SequenceSource], mapping: ApproachMapping,
    detector_mapping: DetectorMapping | None = None,
) -> dict[str, PartitionData]:
    """Build windows within sources; no window may cross a source partition boundary."""
    sources = _validate_sources(sources)
    collected = {partition: {"x": [], "y": [], "windows": []} for partition in PARTITIONS}
    for source in sources:
        extractor = JunctionFeatureExtractor(mapping)
        if source.detector_events is not None:
            if detector_mapping is None or source.detector_mapping_sha256 != detector_mapping.sha256:
                raise ForecastContractError("detector-backed source lacks its exact detector mapping")
            file_events, file_hash = load_detector_events(
                source.event_source_path, detector_mapping, expected_source_id=source.source_id
            )
            if file_hash != source.event_source_sha256 or file_events != source.detector_events:
                raise ForecastContractError("in-memory detector events do not match their immutable source")
            events_by_sample: dict[float, list[ApproachEntryEvent]] = {}
            validated_events = validate_detector_events(source.detector_events, detector_mapping)
            sample_times = {snapshot.time_s for snapshot in source.snapshots}
            for event in validated_events:
                # SUMO reports E1 events after the step containing them. A row
                # labelled t therefore represents entry times in [t-1, t).
                sample_time = float(math.floor(event.entry_time_s) + 1)
                if sample_time not in sample_times:
                    raise ForecastContractError(
                        f"event {event.vehicle_id!r} at {event.entry_time_s} has no recorded sample bin {sample_time}"
                    )
                events_by_sample.setdefault(sample_time, []).append(event)
        per_tls = {tls_id: [] for tls_id in mapping.tls_directions}
        times = []
        for snapshot in source.snapshots:
            if source.detector_events is None:
                rows = extractor.extract(snapshot)
            else:
                rows = detector_feature_row(
                    detector_mapping, events_by_sample.get(snapshot.time_s, ()), snapshot
                )
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


def require_publication_training_ready(manifest: dict, manifest_path: str | Path | None = None) -> None:
    """Independently verify a dataset and a separately trusted manual approval.

    Manual process: an identified traffic engineer must review the exact
    approach/network/detector mapping, write an immutable approval record using
    ``traffic-engineering-mapping-approval-v1``, and provide its SHA-256. A
    separate human-reviewed code change must register that evidence hash and
    exact mapping hashes in ``TRUSTED_MAPPING_APPROVALS``. This function then
    re-hashes the manifest, dataset artifacts, source inputs, approval record,
    mappings, and network. No approval has been registered for this repository.
    """
    if manifest_path is None:
        raise ForecastContractError(
            "publication dataset is not approved: readiness requires the actual manifest file"
        )
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ForecastContractError("publication manifest file is missing")
    try:
        disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError(f"publication manifest cannot be read: {exc}") from exc
    if disk_manifest != manifest:
        raise ForecastContractError("supplied manifest differs from the manifest file")
    payload = dict(manifest)
    recorded_payload_hash = payload.pop("manifest_payload_sha256", None)
    if recorded_payload_hash != _hash_json(payload):
        raise ForecastContractError("manifest payload hash is invalid")
    expected_contract = {
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "units": list(FEATURE_UNITS),
        "history_seconds": HISTORY_SECONDS,
        "target_seconds": TARGET_SECONDS,
        "entry_event_definition": DETECTOR_EVENT_SCHEMA_VERSION,
        "waiting_time_memory_s": WAITING_TIME_MEMORY_S,
    }
    if any(manifest.get(key) != value for key, value in expected_contract.items()):
        raise ForecastContractError("manifest does not satisfy the publication data contract")
    base = manifest_path.parent
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = base / name
        if not path.is_file() or _file_sha256(path) != expected:
            raise ForecastContractError(f"dataset artifact is missing or altered: {name}")
    for source in manifest.get("sources", []):
        for path_key, hash_key in (
            ("snapshot_path", "source_sha256"),
            ("sumo_config_path", "sumo_config_sha256"),
            ("demand_path", "demand_sha256"),
            ("event_source_path", "event_source_sha256"),
        ):
            path_value = source.get(path_key)
            if not path_value or not Path(path_value).is_file() or _file_sha256(path_value) != source.get(hash_key):
                raise ForecastContractError(f"source artifact is missing or altered: {path_key}")
    mapping_path = manifest.get("mapping_path")
    network_path = manifest.get("network_path")
    if not mapping_path or not network_path:
        raise ForecastContractError("manifest lacks mapping/network source paths")
    verified_mapping = load_approach_mapping(mapping_path, network_path)
    verified_detectors = build_detector_mapping(verified_mapping, network_path)
    if verified_mapping.sha256 != manifest.get("mapping_sha256"):
        raise ForecastContractError("approach mapping is altered")
    if verified_mapping.network_sha256 != manifest.get("network_sha256"):
        raise ForecastContractError("network is altered")
    if verified_detectors.sha256 != manifest.get("detector_mapping_sha256"):
        raise ForecastContractError("detector mapping identity is altered")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ForecastContractError("publication manifest has no source records")
    if {row.get("partition") for row in sources} != set(PARTITIONS):
        raise ForecastContractError("publication sources must cover train/validation/test")
    for source in sources:
        if source.get("event_source") != "sumo_e1":
            raise ForecastContractError("publication sources must use verified SUMO E1 events")
        if source.get("detector_mapping_sha256") != verified_detectors.sha256:
            raise ForecastContractError("source detector mapping identity is inconsistent")
        if source.get("event_interval_policy") != "reject_outside_recorded_interval":
            raise ForecastContractError("source event interval policy is not publication-safe")
        if source.get("waiting_time_memory_s") != WAITING_TIME_MEMORY_S:
            raise ForecastContractError("source waiting-time contract is incompatible")
        _validate_waiting_memory(Path(source["sumo_config_path"]))
        load_snapshot_jsonl(source["snapshot_path"])
        load_detector_events(
            source["event_source_path"], verified_detectors,
            expected_source_id=source.get("source_id"),
        )
    approval = manifest.get("mapping_approval")
    if not isinstance(approval, dict):
        raise ForecastContractError("independent mapping approval record is missing")
    evidence_path = Path(str(approval.get("evidence_path", "")))
    if not evidence_path.is_file():
        raise ForecastContractError("independent approval evidence file is missing")
    evidence_hash = _file_sha256(evidence_path)
    if evidence_hash != approval.get("evidence_sha256"):
        raise ForecastContractError("independent approval evidence is altered")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError(f"independent approval evidence is invalid: {exc}") from exc
    required_evidence = {
        "schema_version": MAPPING_APPROVAL_SCHEMA_VERSION,
        "review_status": "approved",
        "evidence_id": approval.get("evidence_id"),
        "approach_mapping_sha256": verified_mapping.sha256,
        "detector_mapping_sha256": verified_detectors.sha256,
        "network_sha256": verified_mapping.network_sha256,
    }
    if any(evidence.get(key) != value for key, value in required_evidence.items()):
        raise ForecastContractError("approval evidence is not bound to the exact mappings/network")
    if not evidence.get("reviewer_identity"):
        raise ForecastContractError("approval evidence lacks an identified independent reviewer")
    trusted = TRUSTED_MAPPING_APPROVALS.get(evidence_hash)
    if trusted != {
        "approach_mapping_sha256": verified_mapping.sha256,
        "detector_mapping_sha256": verified_detectors.sha256,
        "network_sha256": verified_mapping.network_sha256,
    }:
        raise ForecastContractError("approval evidence is not registered in the reviewed trust root")
    if manifest.get("publication_training_ready") is not True:
        raise ForecastContractError("manifest is not marked publication-ready after verification")


def write_dataset(
    output_dir: str | Path,
    partitions: dict[str, PartitionData],
    sources: Iterable[SequenceSource],
    mapping: ApproachMapping,
    detector_mapping: DetectorMapping | None = None,
    mapping_approval: dict | None = None,
    mapping_path: str | Path | None = None,
    network_path: str | Path | None = None,
) -> dict:
    sources = list(sources)
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
        "event_source": "sumo_e1" if source.detector_events is not None else "snapshot_inference",
        "detector_mapping_sha256": source.detector_mapping_sha256,
        "snapshot_path": source.snapshot_path,
        "sumo_config_path": source.sumo_config_path,
        "demand_path": source.demand_path,
        "event_source_path": source.event_source_path,
        "event_source_sha256": source.event_source_sha256,
        "event_interval_policy": source.event_interval_policy,
    } for source in sources]
    detector_backed = all(source.detector_events is not None for source in sources)
    approval_path = Path(mapping_approval["evidence_path"]) if mapping_approval and mapping_approval.get("evidence_path") else None
    approval_hash = _file_sha256(approval_path) if approval_path and approval_path.is_file() else None
    trusted_approval = TRUSTED_MAPPING_APPROVALS.get(approval_hash or "")
    mapping_approved = bool(
        mapping_approval and detector_mapping is not None
        and mapping_approval.get("approved_mapping_sha256") == detector_mapping.sha256
        and mapping_approval.get("evidence_id")
        and approval_path and approval_path.is_file()
        and mapping_approval.get("evidence_sha256") == approval_hash
        and trusted_approval == {
            "approach_mapping_sha256": mapping.sha256,
            "detector_mapping_sha256": detector_mapping.sha256,
            "network_sha256": mapping.network_sha256,
        }
    )
    manifest = {
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "units": list(FEATURE_UNITS),
        "history_seconds": HISTORY_SECONDS,
        "target_seconds": TARGET_SECONDS,
        "target_definition": "directional E1 entries in [t,t+30), binned by reporting step, vehicles/30s",
        "entry_event_definition": (
            detector_mapping.schema_version if detector_backed and detector_mapping else ENTRY_EVENT_DEFINITION
        ),
        "detector_mapping_sha256": detector_mapping.sha256 if detector_mapping else None,
        "mapping_approval": mapping_approval,
        "publication_training_ready": bool(detector_backed and mapping_approved),
        "publication_training_blocker": (
            None if detector_backed and mapping_approved else
            "requires detector-backed sources and independently traffic-engineering-approved mapping"
        ),
        "waiting_feature_definition": "mean TraCI vehicle.getAccumulatedWaitingTime over vehicles present on mapped approach edges at sample time",
        "waiting_time_memory_s": WAITING_TIME_MEMORY_S,
        "mapping_sha256": mapping.sha256,
        "network_sha256": mapping.network_sha256,
        "mapping_path": str(Path(mapping_path).resolve()) if mapping_path else None,
        "network_path": str(Path(network_path).resolve()) if network_path else None,
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
    event_mode = spec.get("event_source_mode", "snapshot_inference")
    if event_mode not in {"snapshot_inference", "sumo_e1"}:
        raise ForecastContractError(f"unknown event source mode {event_mode!r}")
    detector_mapping = build_detector_mapping(mapping, net_path) if event_mode == "sumo_e1" else None
    for label, actual, expected in (
        ("approach mapping", mapping.sha256, spec.get("mapping_sha256")),
        ("network", mapping.network_sha256, spec.get("network_sha256")),
        ("detector mapping", detector_mapping.sha256 if detector_mapping else None,
         spec.get("detector_mapping_sha256")),
    ):
        if event_mode == "sumo_e1" and actual != expected:
            raise ForecastContractError(f"{label} hash is missing or changed")
    sources = []
    for row in spec["sources"]:
        source_path = (base / row["path"]).resolve()
        config_path = (base / row["sumo_config_path"]).resolve()
        demand_path = (base / row["demand_path"]).resolve()
        actual_snapshot_hash = _file_sha256(source_path)
        actual_config_hash = _file_sha256(config_path)
        actual_demand_hash = _file_sha256(demand_path)
        for label, actual, expected in (
            ("snapshot", actual_snapshot_hash, row.get("source_sha256")),
            ("SUMO config", actual_config_hash, row.get("sumo_config_sha256")),
            ("demand", actual_demand_hash, row.get("demand_sha256")),
        ):
            if actual != expected:
                raise ForecastContractError(
                    f"source {row['source_id']!r} {label} file hash is missing or changed"
                )
        events = None; event_path = None; event_hash = None
        if event_mode == "sumo_e1":
            event_path = (base / row["event_path"]).resolve()
            events, event_hash = load_detector_events(
                event_path, detector_mapping, expected_source_id=row["source_id"]
            )
            if event_hash != row.get("event_source_sha256"):
                raise ForecastContractError(
                    f"source {row['source_id']!r} detector-event file hash is missing or changed"
                )
        sources.append(SequenceSource(
            source_id=row["source_id"],
            partition=row["partition"],
            snapshots=load_snapshot_jsonl(source_path),
            demand_seed=int(row["demand_seed"]),
            sumo_config_sha256=actual_config_hash,
            source_sha256=actual_snapshot_hash,
            demand_realization_id=str(row["demand_realization_id"]),
            demand_sha256=actual_demand_hash,
            waiting_time_memory_s=float(row.get("waiting_time_memory_s", WAITING_TIME_MEMORY_S)),
            detector_events=events,
            detector_mapping_sha256=detector_mapping.sha256 if detector_mapping else None,
            snapshot_path=str(source_path),
            sumo_config_path=str(config_path),
            demand_path=str(demand_path),
            event_source_path=str(event_path) if event_path else None,
            event_source_sha256=event_hash,
            event_interval_policy=str(row.get(
                "event_interval_policy", "reject_outside_recorded_interval"
            )),
        ))
    partitions = build_causal_windows(sources, mapping, detector_mapping)
    return write_dataset(
        (base / spec["output_dir"]).resolve(), partitions, sources, mapping,
        detector_mapping, mapping_approval=spec.get("mapping_approval"),
        mapping_path=mapping_path, network_path=net_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="JSON generation manifest")
    args = parser.parse_args()
    print(json.dumps(generate_dataset_from_manifest(args.spec), indent=2))
