"""Versioned E1 detector contracts for exact approach-entry observations."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from DeepQN.forecasting.contracts import (
    DIRECTIONS, ApproachMapping, ForecastContractError, TrafficSnapshot, VehicleObservation,
)

DETECTOR_EVENT_SCHEMA_VERSION = "sumo-e1-approach-entry-v1"
DETECTOR_EVENT_FILE_VERSION = "sumo-e1-event-file-v1"
DETECTOR_POSITION_M = 0.1


@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    tls_id: str
    direction: str
    edge_id: str
    lane_id: str
    position_m: float
    lane_length_m: float


@dataclass(frozen=True)
class DetectorMapping:
    schema_version: str
    specs: tuple[DetectorSpec, ...]
    sha256: str
    network_sha256: str
    traffic_engineering_validated: bool = False


@dataclass(frozen=True)
class ApproachEntryEvent:
    detector_id: str
    tls_id: str
    direction: str
    lane_id: str
    vehicle_id: str
    entry_time_s: float
    leave_time_s: float


def _event_sort_key(event: ApproachEntryEvent) -> tuple:
    return (event.entry_time_s, event.detector_id, event.vehicle_id, event.leave_time_s)


def validate_detector_events(events: Iterable[ApproachEntryEvent],
                             mapping: DetectorMapping) -> tuple[ApproachEntryEvent, ...]:
    """Validate ownership, values, identity, uniqueness, and deterministic order."""
    values = tuple(events)
    specs = {spec.detector_id: spec for spec in mapping.specs}
    if list(values) != sorted(values, key=_event_sort_key):
        raise ForecastContractError("detector events are not in deterministic order")
    seen = set()
    tls_event_owner = {}
    for event in values:
        spec = specs.get(event.detector_id)
        if spec is None:
            raise ForecastContractError(f"event references unknown detector {event.detector_id!r}")
        if (event.tls_id, event.direction, event.lane_id) != (
            spec.tls_id, spec.direction, spec.lane_id
        ):
            raise ForecastContractError(f"event ownership contradicts detector {event.detector_id!r}")
        if not event.vehicle_id:
            raise ForecastContractError("detector event has empty vehicle identity")
        if not math.isfinite(event.entry_time_s) or event.entry_time_s < 0:
            raise ForecastContractError("detector event has invalid entry timestamp")
        if not math.isfinite(event.leave_time_s) or (
            event.leave_time_s != -1 and event.leave_time_s < event.entry_time_s
        ):
            raise ForecastContractError("detector event has invalid leave timestamp")
        identity = (event.detector_id, event.vehicle_id, event.entry_time_s)
        if identity in seen:
            raise ForecastContractError(f"duplicate detector event identity {identity!r}")
        owner_identity = (event.tls_id, event.vehicle_id, event.entry_time_s)
        prior_detector = tls_event_owner.get(owner_identity)
        if prior_detector is not None and prior_detector != event.detector_id:
            raise ForecastContractError(
                f"ambiguous multi-lane E1 entry for {event.vehicle_id!r} "
                f"at {event.entry_time_s}"
            )
        seen.add(identity)
        tls_event_owner[owner_identity] = event.detector_id
    return values


def write_detector_events(path: str | Path, events: Iterable[ApproachEntryEvent],
                          mapping: DetectorMapping, source_id: str) -> str:
    """Write deterministic, non-overwriting detector-event JSON and return SHA-256."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite detector events: {destination}")
    values = validate_detector_events(events, mapping)
    payload = {
        "schema_version": DETECTOR_EVENT_FILE_VERSION,
        "event_contract_version": DETECTOR_EVENT_SCHEMA_VERSION,
        "source_id": source_id,
        "detector_mapping_sha256": mapping.sha256,
        "network_sha256": mapping.network_sha256,
        "events": [asdict(event) for event in values],
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load_detector_events(path: str | Path, mapping: DetectorMapping,
                         expected_source_id: str | None = None) -> tuple[tuple[ApproachEntryEvent, ...], str]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ForecastContractError(f"invalid detector-event JSON: {exc}") from exc
    if payload.get("schema_version") != DETECTOR_EVENT_FILE_VERSION:
        raise ForecastContractError("unknown detector-event file schema")
    if payload.get("event_contract_version") != DETECTOR_EVENT_SCHEMA_VERSION:
        raise ForecastContractError("detector-event contract mismatch")
    if payload.get("detector_mapping_sha256") != mapping.sha256:
        raise ForecastContractError("detector-event mapping hash mismatch")
    if payload.get("network_sha256") != mapping.network_sha256:
        raise ForecastContractError("detector-event network hash mismatch")
    if expected_source_id is not None and payload.get("source_id") != expected_source_id:
        raise ForecastContractError("detector-event source identity mismatch")
    try:
        events = tuple(ApproachEntryEvent(**row) for row in payload["events"])
    except (KeyError, TypeError) as exc:
        raise ForecastContractError(f"malformed detector-event record: {exc}") from exc
    return validate_detector_events(events, mapping), hashlib.sha256(raw).hexdigest()


def build_detector_mapping(approaches: ApproachMapping, net_path: str | Path,
                           position_m: float = DETECTOR_POSITION_M) -> DetectorMapping:
    """Place one E1 on every lane at a common edge-start boundary."""
    if not math.isfinite(position_m) or position_m < 0:
        raise ForecastContractError("detector position must be finite and nonnegative")
    net_path = Path(net_path)
    root = ET.parse(net_path).getroot()
    edges = {edge.get("id"): edge for edge in root.findall("edge") if edge.get("id")}
    specs = []
    for tls_id, directions in approaches.tls_directions.items():
        for direction in DIRECTIONS:
            for edge_id in directions[direction]:
                for lane in edges[edge_id].findall("lane"):
                    lane_id = lane.get("id")
                    length = float(lane.get("length", "nan"))
                    if not lane_id or not math.isfinite(length) or length <= position_m:
                        raise ForecastContractError(
                            f"lane {lane_id!r} cannot host detector at {position_m}m"
                        )
                    specs.append(DetectorSpec(
                        detector_id=f"e1__{tls_id}__{direction}__{lane_id}",
                        tls_id=tls_id, direction=direction, edge_id=edge_id,
                        lane_id=lane_id, position_m=position_m, lane_length_m=length,
                    ))
    if not specs:
        raise ForecastContractError("detector mapping is empty")
    payload = {
        "schema_version": DETECTOR_EVENT_SCHEMA_VERSION,
        "position_semantics": "meters from mapped approach-edge start",
        "specs": [asdict(spec) for spec in specs],
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return DetectorMapping(
        schema_version=DETECTOR_EVENT_SCHEMA_VERSION,
        specs=tuple(specs), sha256=digest,
        network_sha256=hashlib.sha256(net_path.read_bytes()).hexdigest(),
    )


def write_detector_additional(mapping: DetectorMapping, path: str | Path,
                              output_file: str = "NUL") -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite detector definition: {destination}")
    root = ET.Element("additional")
    for spec in mapping.specs:
        ET.SubElement(root, "inductionLoop", {
            "id": spec.detector_id, "lane": spec.lane_id,
            "pos": format(spec.position_m, ".12g"), "period": "1",
            "file": output_file, "friendlyPos": "false",
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


class E1EventCollector:
    """Collect unique physical-boundary entry records from TraCI E1 data.

    Each event is owned by one TLS/direction and identified by detector,
    vehicle, and exact SUMO entry timestamp. A repeated response is ignored.
    Reports for one vehicle on multiple lane detectors of the same TLS at the
    same timestamp are ambiguous and rejected rather than silently merged.
    """

    def __init__(self, mapping: DetectorMapping):
        self.mapping = mapping
        self._by_id = {spec.detector_id: spec for spec in mapping.specs}
        self._seen: set[tuple[str, str, float]] = set()
        self._tls_event_owner: dict[tuple[str, str, float], str] = {}

    def collect(self, detector_rows: Mapping[str, Iterable[tuple]]) -> tuple[ApproachEntryEvent, ...]:
        events = []
        # Stage all identity changes so a malformed or ambiguous batch cannot
        # partially poison the collector and change the result of a retry.
        staged_seen = set(self._seen)
        staged_owners = dict(self._tls_event_owner)
        unknown = set(detector_rows) - set(self._by_id)
        if unknown:
            raise ForecastContractError(f"unknown E1 detectors: {sorted(unknown)}")
        for detector_id, rows in detector_rows.items():
            spec = self._by_id[detector_id]
            for row in rows:
                if len(row) != 5:
                    raise ForecastContractError(f"invalid E1 vehicle-data row for {detector_id}")
                vehicle_id, _length, entry_time, leave_time, _type_id = row
                try:
                    entry_time = float(entry_time); leave_time = float(leave_time)
                except (TypeError, ValueError) as exc:
                    raise ForecastContractError("E1 event has invalid timestamp") from exc
                if not vehicle_id or not math.isfinite(entry_time) or entry_time < 0:
                    raise ForecastContractError("E1 event has invalid identity or entry time")
                if not math.isfinite(leave_time) or (leave_time != -1 and leave_time < entry_time):
                    raise ForecastContractError("E1 event has invalid leave time")
                key = (detector_id, str(vehicle_id), entry_time)
                if key in staged_seen:
                    continue
                owner_key = (spec.tls_id, str(vehicle_id), entry_time)
                prior = staged_owners.get(owner_key)
                if prior is not None and prior != detector_id:
                    raise ForecastContractError(
                        f"ambiguous multi-lane E1 entry for {vehicle_id!r} at {entry_time}"
                    )
                staged_seen.add(key); staged_owners[owner_key] = detector_id
                events.append(ApproachEntryEvent(
                    detector_id, spec.tls_id, spec.direction, spec.lane_id,
                    str(vehicle_id), entry_time, leave_time,
                ))
        result = tuple(sorted(events, key=_event_sort_key))
        validate_detector_events(result, self.mapping)
        self._seen = staged_seen
        self._tls_event_owner = staged_owners
        return result


def detector_feature_row(mapping: DetectorMapping, events: Iterable[ApproachEntryEvent],
                         snapshot: TrafficSnapshot) -> dict[str, tuple[float, ...]]:
    """Combine exact E1 counts with causal approach occupancy state."""
    counts = {tls: {direction: 0 for direction in DIRECTIONS}
              for tls in {spec.tls_id for spec in mapping.specs}}
    for event in events:
        counts[event.tls_id][event.direction] += 1
    lanes_by_tls = {}
    for spec in mapping.specs:
        lanes_by_tls.setdefault(spec.tls_id, set()).add(spec.edge_id)
    rows = {}
    for tls_id in counts:
        present = [vehicle for vehicle in snapshot.vehicles
                   if vehicle.edge_id in lanes_by_tls[tls_id]]
        directional = tuple(float(counts[tls_id][d]) for d in DIRECTIONS)
        avg_speed = sum(v.speed_m_s for v in present) / len(present) if present else 0.0
        avg_wait = sum(v.accumulated_waiting_s for v in present) / len(present) if present else 0.0
        rows[tls_id] = (*directional, sum(directional), avg_speed, avg_wait)
    return rows


def collect_approach_snapshot(connection, mapping: DetectorMapping,
                              time_s: float) -> TrafficSnapshot:
    """Read causal speed/wait state using the same mapped approach edges.

    The SUMO process must be launched with ``--waiting-time-memory 100``.
    ``getAccumulatedWaitingTime`` is the rolling-memory value, not tripinfo's
    journey-total waiting time.
    """
    edges = {spec.edge_id for spec in mapping.specs}
    observations = []
    for vehicle_id in connection.vehicle.getIDList():
        edge_id = connection.vehicle.getRoadID(vehicle_id)
        if edge_id not in edges:
            continue
        speed = float(connection.vehicle.getSpeed(vehicle_id))
        waiting = float(connection.vehicle.getAccumulatedWaitingTime(vehicle_id))
        if not math.isfinite(speed) or speed < 0 or not math.isfinite(waiting) or waiting < 0:
            raise ForecastContractError(f"invalid live observation for {vehicle_id!r}")
        observations.append(VehicleObservation(str(vehicle_id), str(edge_id), speed, waiting))
    return TrafficSnapshot(float(time_s), tuple(observations))
