"""Deterministic, fail-closed demand realization and split-registry contracts.

This module is intentionally offline.  It does not claim that a source route file
or a randomization model represents observed traffic.  Publication use requires
an externally approved, hash-bound specification; the repository currently has
no such approval trust root.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


SPEC_SCHEMA = "sumoflow-demand-spec-v1"
SOURCE_MANIFEST_SCHEMA = "sumoflow-demand-source-v1"
REALIZATION_SCHEMA = "sumoflow-demand-realization-v1"
SPLIT_REGISTRY_SCHEMA = "sumoflow-experiment-split-registry-v1"
SEED_SCHEDULE_SCHEMA = "sumoflow-seed-schedule-proposal-v1"
PARTITIONS = {
    "training", "calibration", "validation", "forecast_test",
    "controller_evaluation",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ROUTE_PROVENANCE_PREFIX = "SUMOFLOW_DEMAND_PROVENANCE "
REGISTRY_STATUS = "frozen_assignments"
REGISTRY_PARTITION_POLICY = "supplied preregistration; no implicit random assignment"

# Deliberately empty.  A maintainer may only add an entry after an independent
# review record approves the exact spec, network and source identities.
TRUSTED_DEMAND_SPEC_APPROVALS: dict[str, dict[str, str]] = {}


class DemandContractError(ValueError):
    """A demand artifact or contract is invalid or cannot be verified."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value.lower()) is None:
        raise DemandContractError(f"{label} is not a valid SHA-256")
    return value.lower()


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DemandContractError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise DemandContractError(f"{label} must be finite")
    return number


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DemandContractError(f"{label} path is missing")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _verified_file(base: Path, entry: Any, label: str) -> Path:
    if not isinstance(entry, dict):
        raise DemandContractError(f"{label} identity is missing")
    path = _resolve(base, entry.get("path"), label)
    expected = _sha(entry.get("sha256"), f"{label} hash")
    if not path.is_file() or sha256_file(path) != expected:
        raise DemandContractError(f"{label} file is missing or its hash changed: {path}")
    return path


def _config_inputs(config_path: Path) -> tuple[float, float, Path, list[Path], list[Path]]:
    try:
        root = ET.parse(config_path).getroot()
        begin = _finite(root.find("./time/begin").get("value"), "configuration begin")
        end = _finite(root.find("./time/end").get("value"), "configuration end")
        input_node = root.find("input")
        network = _resolve(config_path.parent, input_node.find("net-file").get("value"), "network")
        route_node = input_node.find("route-files")
        routes = [_resolve(config_path.parent, item.strip(), "route file")
                  for item in route_node.get("value", "").split(",") if item.strip()]
        additions: list[Path] = []
        additional = input_node.find("additional-files")
        if additional is not None:
            additions = [_resolve(config_path.parent, item.strip(), "additional file")
                         for item in additional.get("value", "").split(",") if item.strip()]
    except (AttributeError, ET.ParseError) as exc:
        raise DemandContractError(f"invalid SUMO configuration: {config_path}") from exc
    if end <= begin:
        raise DemandContractError("configuration end must be greater than begin")
    if not routes:
        raise DemandContractError("configuration contains no route files")
    return begin, end, network, additions, routes


def _network_inventory(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise DemandContractError(f"invalid network XML: {path}") from exc
    edges = {edge.get("id") for edge in root.findall("edge")
             if edge.get("id") and edge.get("function") is None}
    connections = {(item.get("from"), item.get("to")) for item in root.findall("connection")}
    return edges, connections


def _type_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise DemandContractError(f"referenced additional file is missing: {path}")
        try:
            result.update(node.get("id") for node in ET.parse(path).getroot().iter("vType")
                          if node.get("id"))
        except ET.ParseError as exc:
            raise DemandContractError(f"invalid additional XML: {path}") from exc
    return result


def _load_explicit_vehicles(route_path: Path) -> tuple[ET.Element, list[ET.Element]]:
    try:
        root = ET.parse(route_path).getroot()
    except ET.ParseError as exc:
        raise DemandContractError(f"invalid route XML: {route_path}") from exc
    if any(root.findall(tag) != [] for tag in ("flow", "trip", "person", "container")):
        raise DemandContractError("only frozen explicit <vehicle> demand is supported")
    vehicles = root.findall("vehicle")
    if not vehicles:
        raise DemandContractError("source route contains no explicit vehicles")
    ids = [item.get("id") for item in vehicles]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise DemandContractError("source route has missing or duplicate vehicle IDs")
    return root, vehicles


def _validate_vehicle(vehicle: ET.Element, begin: float, end: float, edges: set[str],
                      connections: set[tuple[str, str]], types: set[str]) -> tuple[float, list[str]]:
    depart = _finite(vehicle.get("depart"), f"vehicle {vehicle.get('id')} depart")
    if not begin <= depart < end:
        raise DemandContractError(f"vehicle {vehicle.get('id')!r} departs outside [begin,end)")
    type_id = vehicle.get("type")
    if type_id and type_id not in types:
        raise DemandContractError(f"vehicle {vehicle.get('id')!r} references unknown type {type_id!r}")
    route = vehicle.find("route")
    if route is None or route.get("edges") is None:
        raise DemandContractError(f"vehicle {vehicle.get('id')!r} needs an inline explicit route")
    route_edges = route.get("edges").split()
    if not route_edges or any(edge not in edges for edge in route_edges):
        raise DemandContractError(f"vehicle {vehicle.get('id')!r} references an unknown network edge")
    invalid = [(left, right) for left, right in zip(route_edges, route_edges[1:])
               if (left, right) not in connections]
    if invalid:
        raise DemandContractError(
            f"vehicle {vehicle.get('id')!r} has disconnected route pair {invalid[0]!r}"
        )
    return depart, route_edges


def _canonical_xml(root: ET.Element) -> bytes:
    ET.indent(root, space="    ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True,
                          short_empty_elements=True)
    return payload + (b"" if payload.endswith(b"\n") else b"\n")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _route_provenance(root: ET.Element) -> dict:
    comments = [node.text for node in list(root) if node.tag is ET.Comment]
    matches = [text[len(ROUTE_PROVENANCE_PREFIX):] for text in comments
               if isinstance(text, str) and text.startswith(ROUTE_PROVENANCE_PREFIX)]
    if len(matches) != 1:
        raise DemandContractError("generated route must contain exactly one provenance comment")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise DemandContractError("generated route provenance comment is malformed") from exc
    if not isinstance(value, dict):
        raise DemandContractError("generated route provenance comment must be an object")
    return value


def _physical_fingerprint(root: ET.Element, realization_id: str) -> tuple[str, int]:
    vehicles = root.findall("vehicle")
    expected_prefix = f"{realization_id}__"
    ids = [vehicle.get("id") for vehicle in vehicles]
    if not vehicles or any(not identity or not identity.startswith(expected_prefix) for identity in ids):
        raise DemandContractError("generated vehicle identity conflicts with realization ID")
    if len(ids) != len(set(ids)):
        raise DemandContractError("generated route has duplicate vehicle IDs")
    physical_rows = []
    for vehicle in vehicles:
        depart = _finite(vehicle.get("depart"), f"vehicle {vehicle.get('id')} depart")
        physical = copy.deepcopy(vehicle)
        physical.attrib.pop("id", None)
        physical_rows.append({"depart_s": depart,
                              "vehicle_xml": ET.tostring(physical, encoding="unicode")})
    physical_rows.sort(key=lambda row: (row["depart_s"], row["vehicle_xml"]))
    return hashlib.sha256(_json_bytes({"vehicles": physical_rows})).hexdigest(), len(vehicles)


def _parse_route_with_comments(path: Path) -> ET.Element:
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        return ET.parse(path, parser=parser).getroot()
    except ET.ParseError as exc:
        raise DemandContractError(f"invalid generated route XML: {path}") from exc


def _verify_spec(spec_path: Path, expected_hash: str, *, purpose: str) -> tuple[dict, dict]:
    if sha256_file(spec_path) != _sha(expected_hash, "expected specification hash"):
        raise DemandContractError("demand specification hash mismatch")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemandContractError("invalid demand specification JSON") from exc
    if spec.get("schema") != SPEC_SCHEMA:
        raise DemandContractError("unsupported demand specification schema")
    if purpose not in {"engineering_fixture", "research_candidate"}:
        raise DemandContractError("purpose must be engineering_fixture or research_candidate")
    spec_hash = sha256_file(spec_path)
    if purpose == "research_candidate":
        approval = TRUSTED_DEMAND_SPEC_APPROVALS.get(spec_hash)
        if not approval:
            raise DemandContractError("demand specification lacks independent hash-bound approval")
        for key in ("evidence_path", "evidence_sha256", "network_sha256", "source_sha256"):
            if not approval.get(key):
                raise DemandContractError("trusted approval record is incomplete")
        if (_sha(approval["network_sha256"], "approved network hash")
                != _sha((spec.get("network") or {}).get("sha256"), "specification network hash")
                or _sha(approval["source_sha256"], "approved source hash")
                != _sha((spec.get("source_routes") or {}).get("sha256"),
                        "specification source hash")):
            raise DemandContractError("trusted approval is not bound to this network and source")
        evidence = _resolve(spec_path.parent, approval["evidence_path"], "approval evidence")
        if not evidence.is_file() or sha256_file(evidence) != _sha(
                approval["evidence_sha256"], "approval evidence hash"):
            raise DemandContractError("demand approval evidence is missing or changed")
    return spec, {"path": str(spec_path.resolve()), "sha256": spec_hash}


def generate_demand_realization(spec_path: str | Path, expected_spec_sha256: str, *,
                                seed: int, realization_id: str,
                                output_dir: str | Path,
                                purpose: str = "engineering_fixture") -> dict:
    """Generate one deterministic realization and two immutable manifests."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DemandContractError("seed must be a nonnegative integer")
    if not isinstance(realization_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", realization_id):
        raise DemandContractError("realization ID is missing or invalid")
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite realization directory: {target}")
    spec_path = Path(spec_path).resolve()
    spec, spec_identity = _verify_spec(spec_path, expected_spec_sha256, purpose=purpose)
    base = spec_path.parent
    source = _verified_file(base, spec.get("source_routes"), "source route")
    network = _verified_file(base, spec.get("network"), "network")
    config = _verified_file(base, spec.get("configuration"), "configuration")
    begin, end, config_network, config_additions, config_routes = _config_inputs(config)
    if config_network != network:
        raise DemandContractError("configuration network path differs from the verified network")
    if config_routes != [source]:
        raise DemandContractError(
            "contract v1 requires the verified source to be the only configured route file"
        )
    declared_begin = _finite(spec.get("begin_s"), "specification begin")
    declared_end = _finite(spec.get("end_s"), "specification end")
    if (declared_begin, declared_end) != (begin, end):
        raise DemandContractError("specification window conflicts with configuration")
    declared_additions = spec.get("additional_files", [])
    if not isinstance(declared_additions, list):
        raise DemandContractError("additional_files must be a list")
    additions = [_verified_file(base, item, "additional file") for item in declared_additions]
    if (set(additions) != set(config_additions)
            or len(additions) != len(config_additions)):
        raise DemandContractError("verified additional-file set differs from the configuration")
    edges, connections = _network_inventory(network)
    source_root, source_vehicles = _load_explicit_vehicles(source)
    types = _type_ids(additions) | {node.get("id") for node in source_root.findall("vType")
                                    if node.get("id")}
    model = spec.get("randomization")
    if not isinstance(model, dict) or model.get("evidence_status") not in {
            "provisional_simulation_assumption", "externally_approved"}:
        raise DemandContractError("randomization must declare its evidence status")
    retention = _finite(model.get("vehicle_retention_probability", 1.0), "retention probability")
    jitter = model.get("departure_jitter_s", {})
    if not 0 < retention <= 1:
        raise DemandContractError("vehicle retention probability must be in (0,1]")
    if not isinstance(jitter, dict) or jitter.get("distribution") != "discrete_uniform_seconds":
        raise DemandContractError("only explicit discrete-uniform departure jitter is supported")
    minimum = _finite(jitter.get("minimum"), "jitter minimum")
    maximum = _finite(jitter.get("maximum"), "jitter maximum")
    if not minimum.is_integer() or not maximum.is_integer() or minimum > maximum:
        raise DemandContractError("jitter bounds must be ordered integer seconds")
    if model.get("od_policy") != "preserve" or model.get("vehicle_type_policy") != "preserve":
        raise DemandContractError("OD routes and vehicle types must be preserved by contract v1")

    rng = random.Random(seed)
    generated: list[tuple[float, ET.Element, list[str], str]] = []
    for original in source_vehicles:
        depart, route_edges = _validate_vehicle(original, begin, end, edges, connections, types)
        if rng.random() > retention:
            continue
        offset = rng.randint(int(minimum), int(maximum))
        new_depart = depart + offset
        if not begin <= new_depart < end:
            raise DemandContractError(
                f"randomized departure for {original.get('id')!r} falls outside [begin,end)"
            )
        vehicle = copy.deepcopy(original)
        source_id = vehicle.get("id")
        vehicle.set("id", f"{realization_id}__{source_id}")
        vehicle.set("depart", f"{new_depart:.6f}".rstrip("0").rstrip("."))
        generated.append((new_depart, vehicle, route_edges, source_id))
    if not generated:
        raise DemandContractError("randomization produced an empty realization")
    generated.sort(key=lambda row: (row[0], row[3]))
    output_root = ET.Element("routes", dict(sorted(source_root.attrib.items())))
    route_provenance = {
        "schema": REALIZATION_SCHEMA,
        "realization_id": realization_id,
        "demand_seed": seed,
        "source_spec_sha256": spec_identity["sha256"],
        "purpose": purpose,
    }
    output_root.append(ET.Comment(
        ROUTE_PROVENANCE_PREFIX + json.dumps(route_provenance, sort_keys=True,
                                             separators=(",", ":"))
    ))
    for node in source_root.findall("vType"):
        output_root.append(copy.deepcopy(node))
    for _, vehicle, _, _ in generated:
        output_root.append(vehicle)
    route_bytes = _canonical_xml(output_root)
    route_hash = hashlib.sha256(route_bytes).hexdigest()
    fingerprint, vehicle_count = _physical_fingerprint(output_root, realization_id)
    input_files = {
        "specification": spec_identity,
        "source_routes": {"path": str(source), "sha256": sha256_file(source)},
        "network": {"path": str(network), "sha256": sha256_file(network)},
        "configuration": {"path": str(config), "sha256": sha256_file(config)},
        "additional_files": [{"path": str(path), "sha256": sha256_file(path)}
                             for path in sorted(additions)],
    }
    source_manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA, "profile": spec.get("profile"),
        "begin_s": begin, "end_s": end, "inputs": input_files,
        "origin_evidence": spec.get("origin_evidence", "unknown"),
        "randomization_evidence_status": model["evidence_status"],
        "publication_approved": purpose == "research_candidate",
    }
    source_manifest_bytes = _json_bytes(source_manifest)
    realization = {
        "schema": REALIZATION_SCHEMA, "realization_id": realization_id,
        "purpose": purpose, "status": "generated_offline",
        "profile": spec.get("profile"), "begin_s": begin, "end_s": end,
        "demand_seed": seed, "source_spec_sha256": spec_identity["sha256"],
        "source_manifest_path": "source_manifest.json",
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "network_sha256": sha256_file(network),
        "configuration_sha256": sha256_file(config),
        "generation_parameters": model,
        "route_path": "routes.rou.xml", "route_sha256": route_hash,
        "demand_fingerprint_sha256": fingerprint,
        "vehicle_count": vehicle_count, "publication_approved": purpose == "research_candidate",
        "simulation_status": "not_run",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / "routes.rou.xml").write_bytes(route_bytes)
        (staging / "source_manifest.json").write_bytes(source_manifest_bytes)
        (staging / "realization_manifest.json").write_bytes(_json_bytes(realization))
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return realization


def load_realization_manifest(path: str | Path) -> dict:
    manifest_path = Path(path).resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemandContractError(f"invalid realization manifest: {manifest_path}") from exc
    if data.get("schema") != REALIZATION_SCHEMA:
        raise DemandContractError("unsupported realization-manifest schema")
    route = _resolve(manifest_path.parent, data.get("route_path"), "realization route")
    expected = _sha(data.get("route_sha256"), "realization route hash")
    if not route.is_file() or sha256_file(route) != expected:
        raise DemandContractError("realization route is missing or changed")
    spec_hash = _sha(data.get("source_spec_sha256"), "source specification hash")
    source_manifest = _resolve(manifest_path.parent, data.get("source_manifest_path"),
                               "source manifest")
    source_manifest_hash = _sha(data.get("source_manifest_sha256"), "source manifest hash")
    if not source_manifest.is_file() or sha256_file(source_manifest) != source_manifest_hash:
        raise DemandContractError("source manifest is missing or changed")
    try:
        source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemandContractError("invalid source-manifest JSON") from exc
    if source_data.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise DemandContractError("unsupported source-manifest schema")
    inputs = source_data.get("inputs")
    if not isinstance(inputs, dict):
        raise DemandContractError("source manifest has no input identities")
    verified_inputs = {
        key: _verified_file(source_manifest.parent, inputs.get(key),
                            f"source-manifest {key}")
        for key in ("specification", "source_routes", "network", "configuration")
    }
    additional = inputs.get("additional_files")
    if not isinstance(additional, list):
        raise DemandContractError("source manifest additional-file identities are missing")
    for entry in additional:
        _verified_file(source_manifest.parent, entry, "source-manifest additional file")
    if (_sha(inputs["specification"].get("sha256"), "source specification hash")
            != data.get("source_spec_sha256")
            or _sha(inputs["network"].get("sha256"), "source network hash")
            != data.get("network_sha256")
            or _sha(inputs["configuration"].get("sha256"), "source configuration hash")
            != data.get("configuration_sha256")):
        raise DemandContractError("realization identity conflicts with its source manifest")
    _sha(data.get("network_sha256"), "network hash")
    _sha(data.get("configuration_sha256"), "configuration hash")
    if data.get("status") != "generated_offline" or data.get("simulation_status") != "not_run":
        raise DemandContractError("unexpected realization generation/simulation status")
    purpose = data.get("purpose")
    if purpose not in {"engineering_fixture", "research_candidate"}:
        raise DemandContractError("realization purpose is missing or invalid")
    expected_approval = purpose == "research_candidate"
    if data.get("publication_approved") is not expected_approval:
        raise DemandContractError("realization publication status conflicts with its purpose")
    if source_data.get("publication_approved") is not expected_approval:
        raise DemandContractError("source-manifest publication status conflicts with realization")

    route_root = _parse_route_with_comments(route)
    route_provenance = _route_provenance(route_root)
    expected_route_provenance = {
        "schema": REALIZATION_SCHEMA,
        "realization_id": data.get("realization_id"),
        "demand_seed": data.get("demand_seed"),
        "source_spec_sha256": spec_hash,
        "purpose": purpose,
    }
    if route_provenance != expected_route_provenance:
        raise DemandContractError("route-embedded provenance conflicts with realization manifest")
    fingerprint, vehicle_count = _physical_fingerprint(route_root, data.get("realization_id"))
    if fingerprint != _sha(data.get("demand_fingerprint_sha256"), "demand fingerprint"):
        raise DemandContractError("physical-demand fingerprint does not match verified route bytes")
    if isinstance(data.get("vehicle_count"), bool) or data.get("vehicle_count") != vehicle_count:
        raise DemandContractError("declared vehicle count does not match verified route bytes")

    # Deterministic offline replay authenticates every generated byte and every
    # generation field.  It also consults the approval trust root again for a
    # research candidate; mutable manifest flags cannot promote a fixture.
    with tempfile.TemporaryDirectory() as directory:
        replay_dir = Path(directory) / "replay"
        expected_manifest = generate_demand_realization(
            verified_inputs["specification"], spec_hash,
            seed=data.get("demand_seed"), realization_id=data.get("realization_id"),
            output_dir=replay_dir, purpose=purpose,
        )
        if (replay_dir / "routes.rou.xml").read_bytes() != route.read_bytes():
            raise DemandContractError("verified route bytes do not match deterministic replay")
        if (replay_dir / "source_manifest.json").read_bytes() != source_manifest.read_bytes():
            raise DemandContractError("source manifest does not match deterministic replay")
        if data != expected_manifest:
            raise DemandContractError("realization manifest does not match deterministic replay")
    return data


def write_split_registry(assignments: Iterable[dict], output_path: str | Path) -> dict:
    """Freeze supplied preregistered assignments; never choose partitions implicitly."""
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite split registry: {target}")
    rows: list[dict] = []
    identities: set[str] = set()
    route_hashes: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    seed_owners: dict[tuple[str, int], str] = {}
    profile_windows: dict[str, tuple[float, float, str, str]] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise DemandContractError("registry assignment must be an object")
        partition = assignment.get("partition")
        if partition not in PARTITIONS:
            raise DemandContractError(f"invalid experimental partition {partition!r}")
        manifest_path = Path(assignment.get("manifest_path", "")).resolve()
        manifest = load_realization_manifest(manifest_path)
        identity = manifest.get("realization_id")
        if identity in identities:
            raise DemandContractError(f"duplicate or conflicting realization ID {identity!r}")
        identities.add(identity)
        seed_key = (manifest["source_spec_sha256"], manifest["demand_seed"])
        if seed_key in seed_owners:
            raise DemandContractError(
                f"seed-to-ID collision between {seed_owners[seed_key]!r} and {identity!r}"
            )
        seed_owners[seed_key] = identity
        for value, owners, label in (
            (manifest["route_sha256"], route_hashes, "byte-identical demand"),
            (manifest["demand_fingerprint_sha256"], fingerprints, "physically identical demand"),
        ):
            if value in owners:
                raise DemandContractError(f"{label} reused by {owners[value]!r} and {identity!r}")
            owners[value] = identity
        profile = manifest.get("profile")
        window = (manifest["begin_s"], manifest["end_s"], manifest["network_sha256"],
                  manifest["configuration_sha256"])
        if profile in profile_windows and profile_windows[profile] != window:
            raise DemandContractError(f"conflicting profile window or frozen inputs for {profile!r}")
        profile_windows[profile] = window
        rows.append({
            "realization_id": identity, "partition": partition,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "demand_seed": manifest["demand_seed"], "profile": profile,
            "begin_s": manifest["begin_s"], "end_s": manifest["end_s"],
            "route_path": str((manifest_path.parent / manifest["route_path"]).resolve()),
            "route_sha256": manifest["route_sha256"],
            "demand_fingerprint_sha256": manifest["demand_fingerprint_sha256"],
            "source_spec_sha256": manifest["source_spec_sha256"],
            "network_sha256": manifest["network_sha256"],
            "configuration_sha256": manifest["configuration_sha256"],
            "generation_parameters": manifest["generation_parameters"],
        })
    if not rows:
        raise DemandContractError("split registry cannot be empty")
    rows.sort(key=lambda row: row["realization_id"])
    registry = {
        "schema": SPLIT_REGISTRY_SCHEMA,
        "status": REGISTRY_STATUS,
        "publication_approved": False,
        "partition_policy": REGISTRY_PARTITION_POLICY,
        "realizations": rows,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_json_bytes(registry))
    return registry


def verify_split_registry(path: str | Path, expected_sha256: str) -> dict:
    registry_path = Path(path).resolve()
    if not registry_path.is_file() or sha256_file(registry_path) != _sha(
            expected_sha256, "expected split-registry hash"):
        raise DemandContractError("split registry is missing or changed")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemandContractError("invalid split-registry JSON") from exc
    if registry.get("schema") != SPLIT_REGISTRY_SCHEMA:
        raise DemandContractError("unsupported split-registry schema")
    expected_keys = {
        "schema", "status", "publication_approved", "partition_policy", "realizations"
    }
    if set(registry) != expected_keys:
        raise DemandContractError("split registry has missing or unexpected top-level metadata")
    if registry.get("status") != REGISTRY_STATUS:
        raise DemandContractError("split registry status is not the frozen contract value")
    if registry.get("partition_policy") != REGISTRY_PARTITION_POLICY:
        raise DemandContractError("split registry partition policy is not the frozen contract value")
    if registry.get("publication_approved") is not False:
        raise DemandContractError(
            "split registry cannot self-declare independent scientific approval"
        )
    if not isinstance(registry.get("realizations"), list) or not registry["realizations"]:
        raise DemandContractError("split registry contains no realizations")
    # Re-run all cross-record checks into a temporary path without accepting the
    # registry's copied hashes as authority.
    assignments = [{"manifest_path": row.get("manifest_path"),
                    "partition": row.get("partition")}
                   for row in registry.get("realizations", [])]
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = write_split_registry(assignments, Path(directory) / "registry.json")
    if rebuilt["realizations"] != registry.get("realizations"):
        raise DemandContractError("split registry conflicts with verified realization artifacts")
    return registry


def _config_derivation_payload(path: Path) -> bytes:
    """Canonicalize a config while abstracting only approved run substitutions."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise DemandContractError(f"invalid effective SUMO configuration: {path}") from exc
    for output in (root.find("output"), root.find("o")):
        if output is not None:
            root.remove(output)
    input_section = root.find("input")
    routes = None if input_section is None else input_section.find("route-files")
    if routes is None:
        raise DemandContractError("SUMO configuration has no route-files")
    routes.set("value", "<VERIFIED_DEMAND_ROUTE>")
    random_number = root.find("random_number")
    if random_number is None:
        random_number = ET.SubElement(root, "random_number")
    seed = random_number.find("seed")
    if seed is None:
        seed = ET.SubElement(random_number, "seed")
    seed.set("value", "<VERIFIED_SUMO_SEED>")
    return ET.tostring(root, encoding="utf-8")


def _validate_effective_config(source: Path, effective: Path, *, route: Path,
                               network: Path, additions: list[Path], seed: int,
                               begin: float, end: float) -> None:
    effective_begin, effective_end, effective_network, effective_additions, effective_routes = (
        _config_inputs(effective)
    )
    if (effective_begin, effective_end) != (begin, end):
        raise DemandContractError("effective configuration changes the frozen simulation window")
    if effective_network != network or effective_routes != [route]:
        raise DemandContractError("effective configuration substitutes network or demand route")
    if set(effective_additions) != set(additions) or len(effective_additions) != len(additions):
        raise DemandContractError("effective configuration substitutes additional files")
    root = ET.parse(effective).getroot()
    seed_node = root.find("./random_number/seed")
    if seed_node is None or seed_node.get("value") != str(seed):
        raise DemandContractError("effective configuration has missing or incorrect SUMO seed")
    output = root.find("output")
    if output is None:
        output = root.find("o")
    if output is None:
        raise DemandContractError("effective configuration has no measurement outputs")
    values = {child.tag: child.get("value") for child in output}
    required_destinations = {"tripinfo-output", "emission-output", "summary-output"}
    if any(not values.get(tag) for tag in required_destinations):
        raise DemandContractError("effective configuration has incomplete measurement outputs")
    if values.get("tripinfo-output.write-unfinished") != "true":
        raise DemandContractError("effective configuration must write unfinished tripinfo")
    if values.get("tripinfo-output.write-undeparted") != "true":
        raise DemandContractError("effective configuration must write undeparted tripinfo")
    if values.get("emission-output.step-scaled") != "false":
        raise DemandContractError("effective configuration has incompatible emission scaling")
    allowed = required_destinations | {
        "tripinfo-output.write-unfinished", "tripinfo-output.write-undeparted",
        "emission-output.step-scaled",
    }
    if set(values) != allowed:
        raise DemandContractError("effective configuration contains unexpected output settings")
    if _config_derivation_payload(source) != _config_derivation_payload(effective):
        raise DemandContractError("effective configuration is not an approved derivation of source")


def bind_realization_to_run(manifest: Any, *, realization_manifest_path: str | Path,
                            split_registry_path: str | Path,
                            expected_split_registry_sha256: str,
                            effective_config_path: str | Path,
                            run_purpose: str) -> None:
    """Bind an evaluation manifest to exact demand bytes and frozen membership."""
    registry = verify_split_registry(split_registry_path, expected_split_registry_sha256)
    realization_path = Path(realization_manifest_path).resolve()
    realization = load_realization_manifest(realization_path)
    row = next((item for item in registry["realizations"]
                if item["realization_id"] == realization["realization_id"]), None)
    if row is None or Path(row["manifest_path"]) != realization_path:
        raise DemandContractError("realization is absent from the frozen split registry")
    usage_partitions = {
        "training": "training", "reward_calibration": "calibration",
        "model_validation": "validation", "forecast_test": "forecast_test",
        "controller_evaluation": "controller_evaluation",
    }
    if run_purpose != "engineering_diagnostic":
        expected_partition = usage_partitions.get(run_purpose)
        if expected_partition is None:
            raise DemandContractError("run purpose is missing or invalid")
        if row["partition"] != expected_partition:
            raise DemandContractError(
                f"partition {row['partition']!r} is ineligible for run purpose {run_purpose!r}"
            )
        if realization.get("purpose") != "research_candidate":
            raise DemandContractError("engineering demand cannot be bound to a scientific run purpose")
        raise DemandContractError("publication demand registry has no verified external authorization")

    realization_route = (realization_path.parent / realization["route_path"]).resolve()
    route_files = getattr(manifest, "route_files", {})
    if len(route_files) != 1:
        raise DemandContractError("run must use exactly one registered demand route")
    run_route_raw, run_route_hash = next(iter(route_files.items()))
    run_route = Path(run_route_raw).resolve()
    if (run_route != realization_route or not run_route.is_file()
            or sha256_file(run_route) != run_route_hash
            or run_route_hash != realization["route_sha256"]):
        raise DemandContractError("run route file or bytes differ from the registered realization")

    source_manifest_path = (realization_path.parent / realization["source_manifest_path"]).resolve()
    source_data = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    inputs = source_data["inputs"]
    source_network = Path(inputs["network"]["path"]).resolve()
    source_config = Path(inputs["configuration"]["path"]).resolve()
    source_additions = [Path(item["path"]).resolve() for item in inputs["additional_files"]]
    run_network = Path(getattr(manifest, "network_path", "")).resolve()
    if (run_network != source_network or not run_network.is_file()
            or sha256_file(run_network) != getattr(manifest, "network_sha256", None)
            or getattr(manifest, "network_sha256", None) != realization["network_sha256"]):
        raise DemandContractError("run network identity or bytes differ from the realization")
    run_config = Path(getattr(manifest, "simulation_config_path", "")).resolve()
    if (run_config != source_config or not run_config.is_file()
            or sha256_file(run_config) != getattr(manifest, "simulation_config_sha256", None)
            or getattr(manifest, "simulation_config_sha256", None)
            != realization["configuration_sha256"]):
        raise DemandContractError("run source configuration identity or bytes differ from realization")
    run_additional = getattr(manifest, "additional_files", {})
    expected_additional = {str(path): sha256_file(path) for path in source_additions}
    if run_additional != expected_additional or any(
            not Path(path).is_file() or sha256_file(path) != digest
            for path, digest in run_additional.items()):
        raise DemandContractError("run additional-file identities or bytes differ from realization")
    if getattr(manifest, "scenario", None) != realization["profile"]:
        raise DemandContractError("run profile differs from the registered realization")
    if (getattr(manifest, "simulation_begin_s", None), getattr(manifest, "simulation_end_s", None)) != (
            realization["begin_s"], realization["end_s"]):
        raise DemandContractError("run demand-normalization window differs from the realization")

    effective = Path(effective_config_path).resolve()
    if not effective.is_file():
        raise DemandContractError("effective SUMO configuration is missing")
    _validate_effective_config(
        source_config, effective, route=realization_route, network=source_network,
        additions=source_additions, seed=getattr(manifest, "seed", None),
        begin=realization["begin_s"], end=realization["end_s"],
    )
    from DeepQN.evaluation.provenance import hash_effective_sumocfg
    effective_semantic_hash = hash_effective_sumocfg(effective)
    recorded_effective_hash = getattr(manifest, "effective_simulation_config_sha256", None)
    if recorded_effective_hash not in {None, effective_semantic_hash}:
        raise DemandContractError("run effective-configuration hash conflicts with actual bytes")

    # Commit fields only after every check succeeds.
    manifest.demand_realization_id = realization["realization_id"]
    manifest.demand_manifest_sha256 = sha256_file(realization_path)
    manifest.demand_partition = row["partition"]
    manifest.split_registry_sha256 = expected_split_registry_sha256.lower()
    manifest.demand_usage = run_purpose
    manifest.effective_simulation_config_path = str(effective)
    manifest.effective_simulation_config_file_sha256 = sha256_file(effective)
    manifest.effective_simulation_config_sha256 = effective_semantic_hash


def proposed_seed_schedule() -> dict:
    """Return a collision-checked proposal; approval is required before use."""
    profiles = ("morning_rush", "midday", "evening_rush", "night")
    training = [{"condition": condition, "training_seed_id": seed,
                 "training_rng_seed": seed}
                for condition in ("no_forecast", "verified_forecast")
                for seed in range(11001, 11006)]
    evaluation = []
    for profile_index, profile in enumerate(profiles):
        for replicate in range(1, 11):
            evaluation.append({
                "profile": profile,
                "evaluation_replicate_id": f"{profile}-eval-{replicate:02d}",
                "demand_realization_id": f"{profile}-demand-{21000 + replicate}",
                "demand_rng_seed": 21000 + replicate,
                "sumo_rng_seed": 31000 + profile_index * 100 + replicate,
                "partition": "controller_evaluation",
            })
    identities = [row["evaluation_replicate_id"] for row in evaluation]
    if len(identities) != len(set(identities)):
        raise DemandContractError("proposed seed schedule has identifier collisions")
    return {
        "schema": SEED_SCHEDULE_SCHEMA,
        "status": "proposal_requires_protocol_approval",
        "training": training,
        "controller_evaluation": evaluation,
        "unallocated_partitions": ["training", "calibration", "validation", "forecast_test"],
        "note": "No numeric schedule is frozen until human protocol approval.",
    }


def audit_explicit_demand(*, route_path: str | Path, config_path: str | Path,
                          network_path: str | Path) -> dict:
    """Return byte-derived descriptive facts without assigning empirical origin."""
    route_path = Path(route_path).resolve()
    config_path = Path(config_path).resolve()
    network_path = Path(network_path).resolve()
    begin, end, configured_network, additions, configured_routes = _config_inputs(config_path)
    if configured_network != network_path:
        raise DemandContractError("configuration does not reference the audited network")
    if route_path not in configured_routes:
        raise DemandContractError("configuration does not reference the audited route file")
    try:
        root = ET.parse(route_path).getroot()
    except ET.ParseError as exc:
        raise DemandContractError(f"invalid route XML: {route_path}") from exc
    vehicles = root.findall("vehicle")
    flows = root.findall("flow")
    if root.findall("trip") or root.findall("person") or root.findall("container"):
        raise DemandContractError("unsupported trip/person/container demand in audited source")
    demand_ids = [node.get("id") for node in vehicles + flows]
    if any(not identity for identity in demand_ids) or len(demand_ids) != len(set(demand_ids)):
        raise DemandContractError("audited source has missing or duplicate demand IDs")
    edges, connections = _network_inventory(network_path)
    types = _type_ids(additions) | {node.get("id") for node in root.findall("vType")
                                    if node.get("id")}
    departures: list[float] = []
    type_counts: dict[str, int] = {}
    od_counts: dict[str, int] = {}
    attribute_counts: dict[str, int] = {}
    route_definitions: dict[str, list[str]] = {}
    for route in root.findall("route"):
        identity = route.get("id")
        route_edges = (route.get("edges") or "").split()
        if not identity or identity in route_definitions or not route_edges:
            raise DemandContractError("missing, duplicate, or empty top-level route definition")
        if any(edge not in edges for edge in route_edges) or any(
                (left, right) not in connections for left, right in zip(route_edges, route_edges[1:])):
            raise DemandContractError(f"invalid top-level route definition {identity!r}")
        route_definitions[identity] = route_edges
    for vehicle in vehicles:
        depart, route_edges = _validate_vehicle(
            vehicle, begin, end, edges, connections, types
        )
        departures.append(depart)
        type_id = vehicle.get("type", "<default>")
        type_counts[type_id] = type_counts.get(type_id, 0) + 1
        od = f"{route_edges[0]} -> {route_edges[-1]}"
        od_counts[od] = od_counts.get(od, 0) + 1
        for attribute in vehicle.attrib:
            attribute_counts[attribute] = attribute_counts.get(attribute, 0) + 1
    flow_vehicle_count = 0
    flow_intervals: dict[str, int] = {}
    for flow in flows:
        identity = flow.get("id")
        route_id = flow.get("route")
        type_id = flow.get("type")
        start = _finite(flow.get("begin"), f"flow {identity} begin")
        finish = _finite(flow.get("end"), f"flow {identity} end")
        try:
            number = int(flow.get("number"))
        except (TypeError, ValueError) as exc:
            raise DemandContractError(f"flow {identity!r} requires an integer number") from exc
        if not identity or route_id not in route_definitions or type_id not in types:
            raise DemandContractError(f"flow {identity!r} has invalid route or vehicle type")
        if not begin <= start < finish <= end or number < 0:
            raise DemandContractError(f"flow {identity!r} has invalid interval or count")
        flow_vehicle_count += number
        interval = f"[{start:g},{finish:g})"
        flow_intervals[interval] = flow_intervals.get(interval, 0) + number
        route_edges = route_definitions[route_id]
        od = f"{route_edges[0]} -> {route_edges[-1]}"
        od_counts[od] = od_counts.get(od, 0) + number
        type_counts[type_id] = type_counts.get(type_id, 0) + number
    departures.sort()

    def quantile(fraction: float) -> float:
        position = (len(departures) - 1) * fraction
        lower = int(math.floor(position)); upper = int(math.ceil(position))
        if lower == upper:
            return departures[lower]
        weight = position - lower
        return departures[lower] * (1 - weight) + departures[upper] * weight

    return {
        "route_path": str(route_path), "route_sha256": sha256_file(route_path),
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "network_path": str(network_path), "network_sha256": sha256_file(network_path),
        "begin_s": begin, "end_s": end, "explicit_vehicle_count": len(vehicles),
        "flow_definition_count": len(flows), "flow_requested_vehicle_count": flow_vehicle_count,
        "total_requested_vehicle_count": len(vehicles) + flow_vehicle_count,
        "top_level_route_definition_count": len(route_definitions),
        "trip_count": len(root.findall("trip")),
        "explicit_departure_s": ({"minimum": departures[0], "q25": quantile(0.25),
                                  "median": quantile(0.5), "q75": quantile(0.75),
                                  "maximum": departures[-1]} if departures else None),
        "flow_requested_vehicles_by_interval": dict(sorted(flow_intervals.items())),
        "vehicle_type_counts": dict(sorted(type_counts.items())),
        "unique_od_count": len(od_counts),
        "top_od_counts": [{"od": key, "count": value} for key, value in
                          sorted(od_counts.items(), key=lambda item: (-item[1], item[0]))[:10]],
        "vehicle_attribute_counts": dict(sorted(attribute_counts.items())),
        "route_validity": "all inline and top-level routes use known, connected frozen-network edges",
        "origin_classification": "unknown; source bytes contain no empirical provenance",
    }
