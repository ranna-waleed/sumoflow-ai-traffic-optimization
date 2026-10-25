"""Strict junction-local approach mapping and feature extraction contracts."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

DIRECTIONS = ("north", "south", "east", "west")
FEATURE_NAMES = (*DIRECTIONS, "total", "avg_speed", "avg_waiting")
FEATURE_UNITS = ("veh/s", "veh/s", "veh/s", "veh/s", "veh/s", "m/s", "s")
WAITING_TIME_MEMORY_S = 100.0
ENTRY_EVENT_DEFINITION = "inferred-consecutive-1s-edge-entry-provisional-v1"
FEATURE_SCHEMA_VERSION = "junction-approach-arrivals-v1"


class ForecastContractError(ValueError):
    pass


@dataclass(frozen=True)
class VehicleObservation:
    vehicle_id: str
    edge_id: str
    speed_m_s: float
    accumulated_waiting_s: float


@dataclass(frozen=True)
class TrafficSnapshot:
    time_s: float
    vehicles: tuple[VehicleObservation, ...]


@dataclass(frozen=True)
class ApproachMapping:
    network: str
    schema_version: str
    tls_directions: dict[str, dict[str, tuple[str, ...]]]
    sha256: str
    network_sha256: str


def _travel_direction(edge: ET.Element) -> str:
    lane = edge.find("lane")
    shape = [] if lane is None else (lane.get("shape") or "").split()
    if len(shape) < 2:
        raise ForecastContractError(f"edge {edge.get('id')!r} has no usable lane geometry")
    x1, y1 = (float(value) for value in shape[-2].split(",")[:2])
    x2, y2 = (float(value) for value in shape[-1].split(",")[:2])
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360
    if angle < 45 or angle >= 315:
        return "north"
    if angle < 135:
        return "east"
    if angle < 225:
        return "south"
    return "west"


def load_approach_mapping(path: str | Path, net_path: str | Path) -> ApproachMapping:
    mapping_path = Path(path)
    raw = mapping_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ForecastContractError(f"invalid mapping JSON: {exc}") from exc
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ForecastContractError("unknown forecast mapping schema version")
    tls_payload = payload.get("tls")
    if not isinstance(tls_payload, dict) or not tls_payload:
        raise ForecastContractError("mapping must contain at least one TLS")

    root = ET.parse(net_path).getroot()
    edges = {edge.get("id"): edge for edge in root.findall("edge") if edge.get("id")}
    tls_ids = {logic.get("id") for logic in root.findall("tlLogic")}
    used_edges: dict[str, tuple[str, str]] = {}
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for tls_id, tls_mapping in tls_payload.items():
        if tls_id not in tls_ids:
            raise ForecastContractError(f"mapping references unknown TLS {tls_id!r}")
        if set(tls_mapping) != set(DIRECTIONS):
            raise ForecastContractError(f"TLS {tls_id} must define exactly {DIRECTIONS}")
        normalized[tls_id] = {}
        for direction in DIRECTIONS:
            values = tls_mapping[direction]
            if not isinstance(values, list):
                raise ForecastContractError(f"TLS {tls_id} direction {direction} must be a list")
            if len(values) != len(set(values)):
                raise ForecastContractError(f"duplicate edge within TLS {tls_id}/{direction}")
            for edge_id in values:
                edge = edges.get(edge_id)
                if edge is None:
                    raise ForecastContractError(f"mapping references unknown edge {edge_id!r}")
                if edge.get("to") != tls_id:
                    raise ForecastContractError(f"edge {edge_id!r} does not enter TLS {tls_id!r}")
                if edge.find("lane") is None:
                    raise ForecastContractError(f"edge {edge_id!r} has no lanes")
                actual_direction = _travel_direction(edge)
                if actual_direction != direction:
                    raise ForecastContractError(
                        f"edge {edge_id!r} travels {actual_direction}, not declared {direction}"
                    )
                prior = used_edges.get(edge_id)
                if prior is not None:
                    raise ForecastContractError(
                        f"edge {edge_id!r} is double-counted by {prior} and {(tls_id, direction)}"
                    )
                used_edges[edge_id] = (tls_id, direction)
            normalized[tls_id][direction] = tuple(values)
    return ApproachMapping(
        network=str(payload.get("network", "")),
        schema_version=payload["schema_version"],
        tls_directions=normalized,
        sha256=hashlib.sha256(raw).hexdigest(),
        network_sha256=hashlib.sha256(Path(net_path).read_bytes()).hexdigest(),
    )


class JunctionFeatureExtractor:
    """Extract provisional inferred entries from consecutive 1-second snapshots.

    The first snapshot initializes occupancy and produces no entry events. A
    vehicle that enters and leaves between samples cannot be observed; exact
    physical crossing labels require a detector/event source.
    """

    def __init__(self, mapping: ApproachMapping):
        self.mapping = mapping
        self._previous_edge: dict[str, str] = {}
        self._previous_time: float | None = None
        self._edge_owner = {
            edge: (tls_id, direction)
            for tls_id, directions in mapping.tls_directions.items()
            for direction, edges in directions.items()
            for edge in edges
        }

    def extract(self, snapshot: TrafficSnapshot) -> dict[str, tuple[float, ...]]:
        if not math.isfinite(snapshot.time_s):
            raise ForecastContractError("snapshot time must be finite")
        if self._previous_time is not None and not math.isclose(
            snapshot.time_s - self._previous_time, 1.0, abs_tol=1e-9
        ):
            raise ForecastContractError("forecast contract requires consecutive one-second snapshots")
        vehicle_ids: set[str] = set()
        counts = {tls: {direction: 0 for direction in DIRECTIONS}
                  for tls in self.mapping.tls_directions}
        present = {tls: [] for tls in self.mapping.tls_directions}
        current_edges: dict[str, str] = {}
        counted_events: set[tuple[str, str]] = set()
        for vehicle in snapshot.vehicles:
            if vehicle.vehicle_id in vehicle_ids:
                raise ForecastContractError(f"duplicate vehicle observation {vehicle.vehicle_id!r}")
            vehicle_ids.add(vehicle.vehicle_id)
            if any(not math.isfinite(value) or value < 0 for value in (
                vehicle.speed_m_s, vehicle.accumulated_waiting_s
            )):
                raise ForecastContractError(f"invalid observation for {vehicle.vehicle_id!r}")
            current_edges[vehicle.vehicle_id] = vehicle.edge_id
            owner = self._edge_owner.get(vehicle.edge_id)
            if owner is None:
                continue
            tls_id, direction = owner
            present[tls_id].append(vehicle)
            if self._previous_time is not None and self._previous_edge.get(vehicle.vehicle_id) != vehicle.edge_id:
                event = (vehicle.vehicle_id, vehicle.edge_id)
                if event in counted_events:
                    raise ForecastContractError(f"duplicate crossing event {event}")
                counted_events.add(event)
                counts[tls_id][direction] += 1
        rows: dict[str, tuple[float, ...]] = {}
        for tls_id in self.mapping.tls_directions:
            directional = tuple(float(counts[tls_id][d]) for d in DIRECTIONS)
            vehicles = present[tls_id]
            avg_speed = sum(v.speed_m_s for v in vehicles) / len(vehicles) if vehicles else 0.0
            avg_wait = sum(v.accumulated_waiting_s for v in vehicles) / len(vehicles) if vehicles else 0.0
            rows[tls_id] = (*directional, sum(directional), avg_speed, avg_wait)
        self._previous_edge = current_edges
        self._previous_time = snapshot.time_s
        return rows

