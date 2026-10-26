"""Deterministic expansion of the narrow SUMO ``number``-flow subset.

This offline engineering contract converts only equally spaced, fixed-count
flows with complete named routes. It does not approve a demand model, create
independent stochastic realizations, or imply that requested vehicles insert.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from DeepQN.evaluation.demand_provenance import (
    DemandContractError, _canonical_xml, _config_inputs, _json_bytes,
    _network_inventory, _type_ids, sha256_file,
)

EXPANSION_SPEC_SCHEMA = "sumoflow-number-flow-expansion-spec-v1"
EXPANSION_MANIFEST_SCHEMA = "sumoflow-number-flow-expansion-manifest-v1"
EXPANSION_ALGORITHM = "sumo-number-equal-spacing-begin-inclusive-end-exclusive-v1"
SUPPORTED_SUMO_VERSION = "1.26.0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_FLOW_REQUIRED = {"id", "type", "route", "begin", "end", "number"}
_FLOW_OPTIONAL = {"departLane", "departSpeed"}
_FLOW_OPTIONAL_VALUES = {"departLane": "best", "departSpeed": "random"}
_FLOW_STOP_ATTRIBUTES = {"lane", "duration"}


def _identity(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise DemandContractError(f"{label} identity must contain only path and sha256")
    raw = value["path"]
    digest = value["sha256"]
    if not isinstance(raw, str) or not raw or not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", digest.lower()):
        raise DemandContractError(f"{label} identity is malformed")
    path = (Path(raw) if Path(raw).is_absolute() else base / raw).resolve()
    if not path.is_file() or sha256_file(path) != digest.lower():
        raise DemandContractError(f"{label} file is missing or changed")
    return path


def _decimal(value: Any, label: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DemandContractError(f"{label} must be a finite decimal") from exc
    if not number.is_finite():
        raise DemandContractError(f"{label} must be a finite decimal")
    return number


def _time_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _route_definitions(root: ET.Element, edges: set[str],
                       connections: set[tuple[str, str]]) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for route in root.findall("route"):
        identity = route.get("id")
        route_edges = (route.get("edges") or "").split()
        if (not identity or identity in result or not route_edges
                or set(route.attrib) != {"id", "edges"} or list(route)):
            raise DemandContractError("top-level route IDs and edge lists must be unique and nonempty")
        if any(edge not in edges for edge in route_edges) or any(
                pair not in connections for pair in zip(route_edges, route_edges[1:])):
            raise DemandContractError(f"route {identity!r} is unknown or disconnected")
        result[identity] = route
    return result


def _validate_route_edges(route: ET.Element, edges: set[str],
                          connections: set[tuple[str, str]], label: str) -> None:
    route_edges = (route.get("edges") or "").split()
    if not route_edges:
        raise DemandContractError(f"{label} route must have a nonempty edge list")
    if any(edge not in edges for edge in route_edges):
        raise DemandContractError(f"{label} route contains an unknown edge")
    if any(pair not in connections for pair in zip(route_edges, route_edges[1:])):
        raise DemandContractError(f"{label} route is disconnected")


def _inline_named_route(vehicle: ET.Element, routes: dict[str, ET.Element], edges: set[str],
                        connections: set[tuple[str, str]]) -> None:
    route_id = vehicle.attrib.pop("route", None)
    inline = vehicle.find("route")
    if route_id is not None:
        if inline is not None or route_id not in routes:
            raise DemandContractError("demand has contradictory or unknown route identity")
        route = copy.deepcopy(routes[route_id])
        route.attrib.pop("id", None)
        vehicle.insert(0, route)
    elif inline is None:
        raise DemandContractError("every explicit vehicle requires a complete route")
    _validate_route_edges(vehicle.find("route"), edges, connections,
                          f"vehicle {vehicle.get('id')!r}")


def _validate_flow_children(flow: ET.Element) -> None:
    """Accept only the bounded, shift-independent child subset proven by tests.

    In particular, SUMO shifts a flow stop's ``until`` for later vehicles.  Triggered,
    timetable, waypoint and transport semantics are likewise outside contract v1.
    """
    identity = flow.get("id")
    for child in flow:
        if child.tag != "stop" or set(child.attrib) != _FLOW_STOP_ATTRIBUTES:
            raise DemandContractError(
                f"flow {identity!r} has unsupported child or stop attributes"
            )
        if not child.get("lane"):
            raise DemandContractError(f"flow {identity!r} stop lane is missing")
        duration = _decimal(child.get("duration"), f"flow {identity} stop duration")
        if duration < 0:
            raise DemandContractError(f"flow {identity!r} stop duration must be nonnegative")


def _validate_flow_optional_attributes(flow: ET.Element) -> None:
    identity = flow.get("id")
    for key, supported in _FLOW_OPTIONAL_VALUES.items():
        value = flow.get(key)
        if value is not None and value != supported:
            raise DemandContractError(
                f"flow {identity!r} {key}={value!r} is outside contract v1"
            )


def _physical_fingerprint(vehicles: list[ET.Element]) -> str:
    rows = []
    for vehicle in vehicles:
        clone = copy.deepcopy(vehicle)
        clone.attrib.pop("id", None)
        rows.append({"depart": vehicle.get("depart"),
                     "xml": ET.tostring(clone, encoding="unicode")})
    rows.sort(key=lambda item: (Decimal(item["depart"]), item["xml"]))
    return hashlib.sha256(_json_bytes({"vehicles": rows})).hexdigest()


def expand_number_flows(spec_path: str | Path, expected_spec_sha256: str, *,
                        output_dir: str | Path, fail_after_files: int | None = None) -> dict:
    """Atomically produce explicit demand and a hash-bound engineering manifest."""
    spec_path = Path(spec_path).resolve()
    if (not spec_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_spec_sha256.lower())
            or sha256_file(spec_path) != expected_spec_sha256.lower()):
        raise DemandContractError("expansion specification is missing or changed")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemandContractError("expansion specification is invalid JSON") from exc
    required = {"schema", "expansion_id", "sumo_version", "algorithm", "source_routes",
                "configuration", "network", "additional_files", "begin_s", "end_s",
                "publication_approved"}
    if set(spec) != required or spec.get("schema") != EXPANSION_SPEC_SCHEMA:
        raise DemandContractError("expansion specification schema or fields are invalid")
    if spec.get("publication_approved") is not False:
        raise DemandContractError("an expansion specification cannot self-approve publication use")
    expansion_id = spec.get("expansion_id")
    if not isinstance(expansion_id, str) or _ID.fullmatch(expansion_id) is None:
        raise DemandContractError("expansion ID is invalid")
    if spec.get("sumo_version") != SUPPORTED_SUMO_VERSION:
        raise DemandContractError(f"contract v1 requires SUMO {SUPPORTED_SUMO_VERSION}")
    if spec.get("algorithm") != EXPANSION_ALGORITHM:
        raise DemandContractError("unsupported flow-expansion algorithm")
    base = spec_path.parent
    source = _identity(base, spec["source_routes"], "source route")
    config = _identity(base, spec["configuration"], "configuration")
    network = _identity(base, spec["network"], "network")
    if not isinstance(spec["additional_files"], list):
        raise DemandContractError("additional_files must be a list")
    additions = [_identity(base, value, "additional file") for value in spec["additional_files"]]
    begin, end, config_network, config_additions, config_routes = _config_inputs(config)
    if config_network != network or config_routes != [source] or config_additions != additions:
        raise DemandContractError("verified inputs differ from the effective SUMO configuration")
    if (_decimal(spec["begin_s"], "specification begin"),
            _decimal(spec["end_s"], "specification end")) != (
            Decimal(str(begin)), Decimal(str(end))):
        raise DemandContractError("expansion window differs from the SUMO configuration")

    try:
        source_root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise DemandContractError("source demand is malformed XML") from exc
    unsupported = [child.tag for child in source_root
                   if child.tag not in {"vType", "route", "vehicle", "flow"}]
    if unsupported:
        raise DemandContractError(f"unsupported demand element {unsupported[0]!r}")
    edges, connections = _network_inventory(network)
    routes = _route_definitions(source_root, edges, connections)
    types = _type_ids(additions) | {node.get("id") for node in source_root.findall("vType")
                                    if node.get("id")}
    explicit = [copy.deepcopy(node) for node in source_root.findall("vehicle")]
    source_ids = [node.get("id") for node in explicit + source_root.findall("flow")]
    if any(not identity for identity in source_ids) or len(source_ids) != len(set(source_ids)):
        raise DemandContractError("source vehicle and flow IDs must be present and unique")

    expanded: list[ET.Element] = []
    for vehicle in explicit:
        identity = vehicle.get("id")
        depart = _decimal(vehicle.get("depart"), f"vehicle {identity} departure")
        if not Decimal(str(begin)) <= depart < Decimal(str(end)):
            raise DemandContractError(f"vehicle {identity!r} departs outside [begin,end)")
        if vehicle.get("type") and vehicle.get("type") not in types:
            raise DemandContractError(f"vehicle {identity!r} has unknown type")
        _inline_named_route(vehicle, routes, edges, connections)
        expanded.append(vehicle)

    flow_counts: dict[str, int] = {}
    random_runtime_fields: set[str] = set()
    with localcontext() as context:
        context.prec = 40
        for flow in source_root.findall("flow"):
            identity = flow.get("id")
            keys = set(flow.attrib)
            if not _FLOW_REQUIRED <= keys or keys - (_FLOW_REQUIRED | _FLOW_OPTIONAL):
                raise DemandContractError(
                    f"flow {identity!r} uses unsupported or incomplete attributes"
                )
            _validate_flow_optional_attributes(flow)
            _validate_flow_children(flow)
            if flow.get("route") not in routes or flow.get("type") not in types:
                raise DemandContractError(f"flow {identity!r} has unknown route or type")
            start = _decimal(flow.get("begin"), f"flow {identity} begin")
            finish = _decimal(flow.get("end"), f"flow {identity} end")
            raw_number = flow.get("number")
            if not isinstance(raw_number, str) or re.fullmatch(r"[1-9][0-9]*", raw_number) is None:
                raise DemandContractError(f"flow {identity!r} number must be a positive integer")
            number = int(raw_number)
            if not Decimal(str(begin)) <= start < finish <= Decimal(str(end)):
                raise DemandContractError(f"flow {identity!r} interval is outside [begin,end]")
            spacing = (finish - start) / Decimal(number)
            for index in range(number):
                vehicle_attrs = {key: value for key, value in flow.attrib.items()
                                 if key not in {"id", "route", "begin", "end", "number"}}
                vehicle_attrs["id"] = f"{identity}.{index}"
                vehicle_attrs["depart"] = _time_text(start + Decimal(index) * spacing)
                vehicle = ET.Element("vehicle", dict(sorted(vehicle_attrs.items())))
                route = copy.deepcopy(routes[flow.get("route")])
                route.attrib.pop("id", None)
                vehicle.append(route)
                for child in flow:
                    vehicle.append(copy.deepcopy(child))
                expanded.append(vehicle)
            flow_counts[identity] = number
            for key in ("departLane", "departPos", "departSpeed", "departPosLat",
                        "arrivalLane", "arrivalPos", "arrivalSpeed", "arrivalPosLat",
                        "departEdge", "arrivalEdge"):
                if flow.get(key) == "random":
                    random_runtime_fields.add(key)

    ids = [vehicle.get("id") for vehicle in expanded]
    if len(ids) != len(set(ids)):
        raise DemandContractError("expanded vehicle identities collide")
    expanded.sort(key=lambda node: (Decimal(node.get("depart")), node.get("id")))
    output_root = ET.Element("routes", dict(sorted(source_root.attrib.items())))
    output_root.append(ET.Comment(
        "SUMOFLOW_NUMBER_FLOW_EXPANSION " + json.dumps({
            "schema": EXPANSION_MANIFEST_SCHEMA,
            "expansion_id": expansion_id,
            "source_sha256": sha256_file(source),
        }, sort_keys=True, separators=(",", ":"))
    ))
    for node in source_root.findall("vType"):
        output_root.append(copy.deepcopy(node))
    for node in source_root.findall("route"):
        output_root.append(copy.deepcopy(node))
    for vehicle in expanded:
        output_root.append(vehicle)
    route_bytes = _canonical_xml(output_root)
    manifest = {
        "schema": EXPANSION_MANIFEST_SCHEMA,
        "status": "expanded_offline_not_simulated",
        "publication_approved": False,
        "expansion_id": expansion_id,
        "algorithm": EXPANSION_ALGORITHM,
        "sumo_version_contract": SUPPORTED_SUMO_VERSION,
        "expansion_rng": "none; fixed-count number flows are equally spaced",
        "runtime_randomness": {
            "fields_preserved": sorted(random_runtime_fields),
            "requires_separate_recorded_sumo_seed": bool(random_runtime_fields),
        },
        "window": {"begin_s": begin, "end_s": end, "interval": "[begin,end)"},
        "inputs": {
            "specification": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            "source_routes": {"path": str(source), "sha256": sha256_file(source)},
            "configuration": {"path": str(config), "sha256": sha256_file(config)},
            "network": {"path": str(network), "sha256": sha256_file(network)},
            "additional_files": [{"path": str(path), "sha256": sha256_file(path)}
                                 for path in additions],
        },
        "counts": {
            "source_explicit_vehicles": len(explicit),
            "source_flows": len(flow_counts),
            "flow_requested_vehicles": sum(flow_counts.values()),
            "expanded_explicit_vehicles": len(expanded),
        },
        "flow_counts": dict(sorted(flow_counts.items())),
        "output": {
            "path": "routes.explicit.rou.xml",
            "sha256": hashlib.sha256(route_bytes).hexdigest(),
            "physical_fingerprint_sha256": _physical_fingerprint(expanded),
        },
        "interpretation": "scheduled explicit demand; insertion and arrival require SUMO outputs",
    }
    manifest_bytes = _json_bytes(manifest)
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite expansion directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".demand-expansion-", dir=target.parent))
    try:
        (staging / "routes.explicit.rou.xml").write_bytes(route_bytes)
        if fail_after_files == 1:
            raise OSError("injected expansion publication failure")
        (staging / "expansion_manifest.json").write_bytes(manifest_bytes)
        if fail_after_files == 2:
            raise OSError("injected expansion publication failure")
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_expansion(directory: str | Path, *, expected_manifest_sha256: str,
                     expected_spec_sha256: str) -> dict:
    """Verify against externally frozen identities and an independent regeneration.

    The caller must obtain both expected hashes from a separately frozen registry or
    review record.  Hashes establish identity only; they are not authentication or
    scientific approval.
    """
    root = Path(directory).resolve()
    manifest_path = root / "expansion_manifest.json"
    route_path = root / "routes.explicit.rou.xml"
    if (not re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_sha256).lower())
            or not re.fullmatch(r"[0-9a-f]{64}", str(expected_spec_sha256).lower())):
        raise DemandContractError("trusted expansion hashes are malformed")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256.lower():
        raise DemandContractError("expansion manifest differs from the trusted identity")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemandContractError("expansion manifest is missing or malformed") from exc
    expected_manifest_fields = {
        "schema", "status", "publication_approved", "expansion_id", "algorithm",
        "sumo_version_contract", "expansion_rng", "runtime_randomness", "window",
        "inputs", "counts", "flow_counts", "output", "interpretation",
    }
    if (set(manifest) != expected_manifest_fields
            or manifest.get("schema") != EXPANSION_MANIFEST_SCHEMA
            or manifest.get("publication_approved") is not False
            or manifest.get("status") != "expanded_offline_not_simulated"
            or manifest.get("algorithm") != EXPANSION_ALGORITHM
            or manifest.get("sumo_version_contract") != SUPPORTED_SUMO_VERSION):
        raise DemandContractError("expansion manifest status is invalid")
    if not route_path.is_file() or sha256_file(route_path) != manifest.get("output", {}).get("sha256"):
        raise DemandContractError("expanded route is missing or changed")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
            "specification", "source_routes", "configuration", "network",
            "additional_files"}:
        raise DemandContractError("expansion input identities are missing")
    specification = inputs.get("specification", {})
    if specification.get("sha256") != expected_spec_sha256.lower():
        raise DemandContractError("manifest specification differs from the trusted identity")
    for name in ("specification", "source_routes", "configuration", "network"):
        entry = inputs.get(name, {})
        path = Path(entry.get("path", ""))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise DemandContractError(f"expansion {name} is missing or changed")
    for entry in inputs.get("additional_files", []):
        path = Path(entry.get("path", ""))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise DemandContractError("expansion additional file is missing or changed")
    try:
        route_root = ET.parse(route_path).getroot()
    except ET.ParseError as exc:
        raise DemandContractError("expanded route is malformed") from exc
    if route_root.findall("flow") or route_root.findall("trip") or route_root.findall("person"):
        raise DemandContractError("expanded route is not explicit-only")
    vehicles = route_root.findall("vehicle")
    if len(vehicles) != manifest.get("counts", {}).get("expanded_explicit_vehicles"):
        raise DemandContractError("expanded route count conflicts with manifest")
    if _physical_fingerprint(vehicles) != manifest.get("output", {}).get(
            "physical_fingerprint_sha256"):
        raise DemandContractError("expanded route fingerprint conflicts with manifest")

    # The trusted specification binds every source identity.  Regeneration independently
    # reconstructs IDs, departures, routes, children, counts and the semantic fingerprint;
    # a coordinated output/manifest edit cannot satisfy this comparison.
    spec_path = Path(specification.get("path", ""))
    with tempfile.TemporaryDirectory(prefix="sumoflow-expansion-verification-") as temp:
        regenerated_dir = Path(temp) / "regenerated"
        regenerated = expand_number_flows(
            spec_path, expected_spec_sha256, output_dir=regenerated_dir
        )
        if (regenerated != manifest
                or (regenerated_dir / "routes.explicit.rou.xml").read_bytes()
                != route_path.read_bytes()):
            raise DemandContractError("expansion differs from independent regeneration")
    return manifest
