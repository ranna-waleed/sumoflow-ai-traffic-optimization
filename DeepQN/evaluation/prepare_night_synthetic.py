"""Prepare one immutable engineering candidate for bounded night diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from DeepQN.evaluation.demand_expansion import (
    EXPANSION_ALGORITHM, EXPANSION_SPEC_SCHEMA, expand_number_flows,
    verify_expansion, _physical_fingerprint,
)
from DeepQN.evaluation.demand_provenance import _canonical_xml, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sumoflow-night-synthetic-input-candidate-v6"
MIN_LANE_CHANGE_DISTANCE_M = 25.0

# This is a reviewable correction to one source record, not an automatic route
# repair rule.  The historical record calls the object a microbus but assigns a
# bicycle vClass to a route whose next edge prohibits bicycles.
VEHICLE_CLASS_CORRECTIONS = {
    "microbus_982": {
        "source_type": "type_bicycle",
        "candidate_type": "type_microbus",
        "source_vclass": "bicycle",
        "candidate_vclass": "bus",
        "incompatible_pair": ["28718647#3", "690516091#1"],
        "policy": "explicit_reviewed_vehicle_class_correction_preserve_id_depart_route_and_stops",
    },
}


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _network_variant(source: Path, target: Path) -> dict:
    # Export the compiled source to SUMO plain XML, edit topology/programs in
    # their supported source representations, and rebuild all internal lanes,
    # junction logic, via lanes, and right-of-way data with netconvert.
    with tempfile.TemporaryDirectory(prefix="sumoflow-v6-net-") as temp_name:
        temp = Path(temp_name)
        prefix = temp / "base"
        exported = subprocess.run(
            ["netconvert", "--sumo-net-file", str(source),
             "--plain-output-prefix", str(prefix)], capture_output=True,
            text=True, check=False)
        if exported.returncode != 0:
            raise RuntimeError(f"netconvert plain export failed: {exported.stderr}")
        con_path, tll_path = temp / "base.con.xml", temp / "base.tll.xml"
        con_root, root = ET.parse(con_path).getroot(), ET.parse(tll_path).getroot()
        changes = _edit_plain_network(con_root, root)
        _write_xml(con_root, con_path); _write_xml(root, tll_path)
        rebuilt = subprocess.run([
            "netconvert", "--node-files", str(temp / "base.nod.xml"),
            "--edge-files", str(temp / "base.edg.xml"),
            "--connection-files", str(con_path),
            "--tllogic-files", str(tll_path),
            "--type-files", str(temp / "base.typ.xml"),
            "--output-file", str(target), "--no-turnarounds", "true",
            "--offset.disable-normalization", "true", "--walkingareas", "false",
        ], capture_output=True, text=True, check=False)
        if rebuilt.returncode != 0 or not target.is_file():
            raise RuntimeError(f"netconvert rebuild failed: {rebuilt.stderr}")
        changes["supported_rebuild"] = {
            "tool": "netconvert", "source_representation": "plain XML",
            "internal_geometry_recomputed": True,
            "export_warnings": exported.stderr.splitlines(),
            "rebuild_warnings": rebuilt.stderr.splitlines(),
        }
        return changes


def _edit_plain_network(con_root: ET.Element, root: ET.Element) -> dict:
    changes = {}
    tls_a = next(node for node in root.findall("tlLogic")
                 if node.get("id") == "96621068" and node.get("programID") == "0")
    phases_a = tls_a.findall("phase")
    if len(phases_a) != 1 or phases_a[0].get("state") != "GGGG":
        raise ValueError("unexpected source program for 96621068")
    phases_a[0].set("state", "gggg")
    changes["96621068"] = {
        "source": "GGGG", "candidate": "gggg",
        "rationale": "all four same-direction merge links are permissive; no protected-conflict claim",
    }
    tls_b = next(node for node in root.findall("tlLogic")
                 if node.get("id") == "96621100" and node.get("programID") == "0")
    phases_b = tls_b.findall("phase")
    expected = ["rrrrrrrrrr", "yyyyyyyyyy", "GGGGGGGGGG"]
    if [phase.get("state") for phase in phases_b] != expected:
        raise ValueError("unexpected source program for 96621100")
    for phase in phases_b:
        tls_b.remove(phase)
    for duration, state, following in (
            (45, "GGGGGrrrrr", "1"),
            (3, "yyyyyrrrrr", "2"),
            (45, "rrrrrGGGGG", "3"),
            (3, "rrrrryyyyy", "0")):
        ET.SubElement(tls_b, "phase", duration=str(duration), state=state,
                      next=following)
    remapped_links = []
    for collection in (con_root, root):
        for connection in collection.findall("connection"):
            if not (connection.get("from") == "10873186#8"
                    and connection.get("to") == "10873186#9"):
                continue
            old = connection.get("toLane")
            new = ("5" if connection.get("fromLane") == "4"
                   else connection.get("fromLane"))
            connection.set("toLane", new)
            if connection.get("linkIndex") is not None:
                remapped_links.append({"link_index": int(connection.get("linkIndex")),
                                       "from_lane": int(connection.get("fromLane")),
                                       "source_to_lane": int(old),
                                       "candidate_to_lane": int(new)})
    changes["96621100"] = {
        "native_cycle": "0(approach-A-green)->1(A-yellow)->2(approach-B-green)->3(B-yellow)->0",
        "candidate_phases": ["GGGGGrrrrr", "yyyyyrrrrr",
                             "rrrrrGGGGG", "rrrrryyyyy"],
        "controlled_link_indices": {"approach_A": [0, 1, 2, 3, 4],
                                    "approach_B": [5, 6, 7, 8, 9]},
        "secondary_approach_lane_remapping": remapped_links,
        "rationale": "synthetic protected alternation prevents simultaneous merge and gives the secondary approach direct access to downstream lane 0",
    }
    changes["synthetic_effective_edge_lengths"] = {
        "changes": [],
        "assumption": "v5 lane geometry and lengths are preserved; direct lane-chain topology removes the prior need for compiled length edits",
    }
    turn_additions = []
    existing = {(node.get("fromLane"), node.get("toLane"))
                for node in con_root.findall("connection")
                if node.get("from") == "10873186#0"
                and node.get("to") == "28718874#0"}
    for from_lane, to_lane in (("1", "0"), ("2", "1"), ("3", "1"),
                               ("4", "2"), ("5", "2")):
        if (from_lane, to_lane) in existing:
            continue
        ET.SubElement(con_root, "connection", {
            "from": "10873186#0", "to": "28718874#0",
            "fromLane": from_lane, "toLane": to_lane,
        })
        turn_additions.append({"from_lane": int(from_lane),
                               "to_lane": int(to_lane)})
    changes["96621145_turn_access"] = {
        "source_only_from_lane": 0,
        "added_plain_connections": turn_additions,
        "rationale": "six fixed-time wrong-lane teleports occurred on lanes 2/3 for routes requiring 28718874#0; netconvert rebuilds junction internals and right-of-way",
    }
    return changes


def _sumocfg(path: Path, *, network: Path, routes: Path, begin: int,
             end: int, additions: list[Path]) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=str(network.resolve()))
    ET.SubElement(inputs, "route-files", value=str(routes.resolve()))
    ET.SubElement(inputs, "additional-files",
                  value=",".join(str(item.resolve()) for item in additions))
    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value=str(begin)); ET.SubElement(time, "end", value=str(end))
    ET.SubElement(time, "step-length", value="1")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.SubElement(processing, "time-to-teleport", value="300")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-warnings", value="false")
    _write_xml(root, path)


def _correct_stopping_places(source: Path, target: Path) -> list[dict]:
    root = ET.parse(source).getroot()
    corrections = {
        "bs_5": {"lane": "28718647#3_0", "startPos": "0.1", "endPos": "12.1",
                 "rationale": "relocated to the next common route edge before bs_3"},
        "bs_2": {"startPos": "1", "endPos": "13",
                 "rationale": "extended within the existing 13.34 m lane"},
        "bs_3": {"startPos": "13.1", "endPos": "25.1",
                 "rationale": "extended and placed after relocated bs_5 within the existing 25.10 m lane"},
        "bs_9": {"startPos": "1", "endPos": "13",
                 "rationale": "extended within the existing 96.17 m lane"},
    }
    records = []
    for stop_id, replacement in corrections.items():
        stop = next((node for node in root.findall("busStop")
                     if node.get("id") == stop_id), None)
        if stop is None:
            raise ValueError(f"missing reviewed stopping place {stop_id}")
        before = dict(stop.attrib)
        rationale = replacement["rationale"]
        stop.attrib.update({key: value for key, value in replacement.items()
                            if key != "rationale"})
        records.append({"stop_id": stop_id, "source": before,
                        "candidate": dict(stop.attrib),
                        "required_vehicle_length_m": 12.0,
                        "rationale": rationale})
    _write_xml(root, target)
    return records


def _correct_invalid_stops(source: Path, network: Path, additions: list[Path],
                           target: Path) -> list[dict]:
    root = ET.parse(source).getroot()
    net = ET.parse(network).getroot()
    lengths = {lane.get("id"): float(lane.get("length"))
               for lane in net.findall("edge/lane")}
    lane_edges = {lane.get("id"): edge.get("id") for edge in net.findall("edge")
                  for lane in edge.findall("lane")}
    bus_stops = {}
    for addition in additions:
        for stop in ET.parse(addition).getroot().findall("busStop"):
            bus_stops[stop.get("id")] = {
                "lane": stop.get("lane"),
                "start_pos": float(stop.get("startPos", 0)),
            }
    corrections = []
    for vehicle in root.findall("vehicle"):
        route = vehicle.find("route")
        route_sequence = (route.get("edges") or "").split() if route is not None else []
        route_edges = set(route_sequence)
        indexed_stops = []
        original_stops = list(vehicle.findall("stop"))
        for index, stop in enumerate(original_stops):
            bus_stop_id = stop.get("busStop")
            if bus_stop_id is not None:
                definition = bus_stops.get(bus_stop_id) or {}
                lane_id = definition.get("lane")
                if lane_id not in lane_edges:
                    raise ValueError(f"unknown bus stop {bus_stop_id}")
                edge_id = lane_edges[lane_id]
                if edge_id not in route_edges:
                    raise ValueError(f"vehicle {vehicle.get('id')} bus stop is outside its route")
                edge_index = route_sequence.index(edge_id)
                indexed_stops.append((edge_index, definition["start_pos"],
                                      index, stop, lane_id))
                continue
            lane_id = stop.get("lane")
            if lane_id is None:
                continue
            if lane_id not in lengths:
                raise ValueError(f"stop references unknown lane {lane_id}")
            edge_id = lane_edges[lane_id]
            if edge_id not in route_edges:
                raise ValueError(f"vehicle {vehicle.get('id')} lane stop is outside its route")
            edge_index = route_sequence.index(edge_id)
            indexed_stops.append((edge_index, float(stop.get("startPos", 0)),
                                  index, stop, lane_id))
            start = float(stop.get("startPos", 0))
            end = float(stop.get("endPos", lengths[lane_id]))
            if start < 0 or end > lengths[lane_id] or start >= end:
                if stop.get("friendlyPos") not in (None, "false"):
                    raise ValueError("source stop has unexpected friendlyPos semantics")
                stop.set("friendlyPos", "true")
                corrections.append({"vehicle_id": vehicle.get("id"),
                    "stop_index": index, "lane_id": lane_id,
                    "lane_length_m": lengths[lane_id],
                    "requested_start_pos_m": start, "requested_end_pos_m": end,
                    "policy": "sumo_friendlyPos_true_preserve_stop_and_clamp_position"})
        sorted_stops = sorted(indexed_stops,
                              key=lambda item: (item[0], item[1], item[2]))
        if [item[2] for item in sorted_stops] != list(range(len(original_stops))):
            for stop in original_stops:
                vehicle.remove(stop)
            for _, _, _, stop, _ in sorted_stops:
                vehicle.append(stop)
            corrections.append({
                "vehicle_id": vehicle.get("id"),
                "source_stop_order": [item[2] for item in indexed_stops],
                "candidate_stop_order": [item[2] for item in sorted_stops],
                "bus_stop_ids": [item[3].get("busStop") for item in sorted_stops],
                "lane_ids": [item[4] for item in sorted_stops],
                "policy": "reorder_stops_to_route_order_preserve_all_obligations",
            })
    target.write_bytes(_canonical_xml(root))
    return corrections


def _vehicle_types(additions: list[Path]) -> dict[str, str]:
    result = {}
    for addition in additions:
        for node in ET.parse(addition).getroot().findall("vType"):
            identity = node.get("id")
            if identity:
                result[identity] = node.get("vClass", "passenger")
    return result


def _lane_allows(lane: ET.Element, vclass: str) -> bool:
    allowed = set((lane.get("allow") or "").split())
    disallowed = set((lane.get("disallow") or "").split())
    return (not allowed or vclass in allowed) and vclass not in disallowed


def validate_explicit_route_integrity(route_path: str | Path,
                                      network_path: str | Path,
                                      additions: list[str | Path]) -> dict:
    """Validate every explicit route, vClass movement, and required stop.

    This source-level check proves that at least one permitted lane-to-lane
    connection exists for every consecutive edge pair.  It cannot prove that a
    congested vehicle can reach the needed lane at runtime; the full-horizon
    SUMO probe remains mandatory for that dynamic property.
    """
    route_path = Path(route_path).resolve()
    network_path = Path(network_path).resolve()
    additions = [Path(item).resolve() for item in additions]
    route_root = ET.parse(route_path).getroot()
    net_root = ET.parse(network_path).getroot()
    types = _vehicle_types(additions)
    lanes = {lane.get("id"): lane for edge in net_root.findall("edge")
             for lane in edge.findall("lane") if lane.get("id")}
    edge_lanes = {edge.get("id"): [lane.get("id") for lane in edge.findall("lane")]
                  for edge in net_root.findall("edge") if edge.get("id")}
    lane_edges = {lane.get("id"): edge.get("id")
                  for edge in net_root.findall("edge")
                  for lane in edge.findall("lane") if lane.get("id")}
    edges = {edge.get("id") for edge in net_root.findall("edge")
             if edge.get("id") and not edge.get("id").startswith(":")}
    connections: dict[tuple[str, str], list[ET.Element]] = {}
    for connection in net_root.findall("connection"):
        connections.setdefault((connection.get("from"), connection.get("to")), []).append(connection)
    bus_stops = {}
    for addition in additions:
        for stop in ET.parse(addition).getroot().findall("busStop"):
            bus_stops[stop.get("id")] = dict(stop.attrib)

    checked_stops = 0
    vehicles = route_root.findall("vehicle")
    for vehicle in vehicles:
        identity = vehicle.get("id")
        type_id = vehicle.get("type", "passenger")
        vclass = types.get(type_id, type_id)
        route = vehicle.find("route")
        route_edges = ((route.get("edges") or "").split()
                       if route is not None else [])
        if not route_edges or any(edge not in edges for edge in route_edges):
            raise ValueError(f"vehicle {identity!r} has an empty or unknown-edge route")
        for left, right in zip(route_edges, route_edges[1:]):
            candidates = connections.get((left, right), [])
            compatible = False
            for connection in candidates:
                from_lane = lanes.get(f"{left}_{connection.get('fromLane')}")
                to_lane = lanes.get(f"{right}_{connection.get('toLane')}")
                via_lane = lanes.get(connection.get("via")) if connection.get("via") else None
                if (from_lane is not None and to_lane is not None
                        and _lane_allows(from_lane, vclass)
                        and _lane_allows(to_lane, vclass)
                        and (via_lane is None or _lane_allows(via_lane, vclass))):
                    compatible = True
                    break
            if not compatible:
                raise ValueError(
                    f"vehicle {identity!r} vClass={vclass!r} has no permitted "
                    f"lane connection for {left!r}->{right!r}")

        last_stop_index = -1
        stop_constraints = []
        for stop in vehicle.findall("stop"):
            lane_id = stop.get("lane")
            if stop.get("busStop") is not None:
                bus_stop = bus_stops.get(stop.get("busStop"))
                lane_id = bus_stop.get("lane") if bus_stop else None
            lane = lanes.get(lane_id)
            if lane is None:
                raise ValueError(f"vehicle {identity!r} has an unresolved stop lane")
            edge_id = lane_edges[lane_id]
            indices = [index for index, value in enumerate(route_edges)
                       if value == edge_id and index >= last_stop_index]
            if not indices or not _lane_allows(lane, vclass):
                raise ValueError(
                    f"vehicle {identity!r} has an unreachable or prohibited stop on {lane_id!r}")
            last_stop_index = indices[0]
            lane_length = float(lane.get("length"))
            requested_end = float(
                bus_stop.get("endPos", lane_length)
                if stop.get("busStop") is not None
                else stop.get("endPos", lane_length))
            effective_end = min(requested_end, lane_length)
            stop_constraints.append((last_stop_index, lane_id, effective_end))
            if (last_stop_index + 1 < len(route_edges)
                    and effective_end >= lane_length - 1e-9):
                next_edge = route_edges[last_stop_index + 1]
                lane_index = lane.get("index")
                can_depart_stop_lane = any(
                    connection.get("fromLane") == lane_index
                    and _lane_allows(lanes[f"{next_edge}_{connection.get('toLane')}"] , vclass)
                    for connection in connections.get((edge_id, next_edge), [])
                )
                if not can_depart_stop_lane:
                    raise ValueError(
                        f"vehicle {identity!r} stop on {lane_id!r} reaches the "
                        f"edge end but that lane cannot serve next edge {next_edge!r}")
            checked_stops += 1

        stops_by_edge: dict[int, list[tuple[str, float]]] = {}
        for edge_index, lane_id, effective_end in stop_constraints:
            stops_by_edge.setdefault(edge_index, []).append(
                (lane_id, effective_end))
        reachable = {lane_id for lane_id in edge_lanes[route_edges[0]]
                     if _lane_allows(lanes[lane_id], vclass)}
        for edge_index, edge_id in enumerate(route_edges):
            permitted = {lane_id for lane_id in edge_lanes[edge_id]
                         if _lane_allows(lanes[lane_id], vclass)}
            edge_length = max(float(lanes[lane_id].get("length"))
                              for lane_id in permitted)
            if edge_length >= MIN_LANE_CHANGE_DISTANCE_M:
                reachable = permitted
            for stop_lane, effective_end in stops_by_edge.get(edge_index, []):
                if stop_lane not in reachable:
                    raise ValueError(
                        f"vehicle {identity!r} cannot reach required stop lane "
                        f"{stop_lane!r} under the synthetic lane-chain contract")
                reachable = {stop_lane}
                if edge_length - effective_end >= MIN_LANE_CHANGE_DISTANCE_M:
                    reachable = permitted
            if edge_index == len(route_edges) - 1:
                break
            next_edge = route_edges[edge_index + 1]
            next_reachable = set()
            for connection in connections.get((edge_id, next_edge), []):
                from_lane = f"{edge_id}_{connection.get('fromLane')}"
                to_lane = f"{next_edge}_{connection.get('toLane')}"
                if (from_lane in reachable and to_lane in lanes
                        and _lane_allows(lanes[to_lane], vclass)):
                    next_reachable.add(to_lane)
            if not next_reachable:
                raise ValueError(
                    f"vehicle {identity!r} has no continuous feasible lane chain "
                    f"for {edge_id!r}->{next_edge!r}")
            reachable = next_reachable
    return {
        "schema": "sumoflow-explicit-route-integrity-v1",
        "vehicle_count": len(vehicles),
        "required_stop_count": checked_stops,
        "checks": ["known_edges", "ordered_edge_connectivity",
                   "vehicle_class_lane_permissions", "ordered_required_stops",
                   "stop_lane_vehicle_class_permissions",
                   "end_of_edge_stop_outgoing_lane_reachability",
                   "continuous_lane_chain_with_25m_lane_change_contract"],
        "runtime_limitation": (
            "connectivity and permissions do not prove lane reachability under "
            "congestion; full-horizon SUMO execution is required"),
    }


def _apply_vehicle_class_corrections(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    records = []
    vehicles = {node.get("id"): node for node in root.findall("vehicle")}
    for identity, correction in VEHICLE_CLASS_CORRECTIONS.items():
        vehicle = vehicles.get(identity)
        if vehicle is None or vehicle.get("type") != correction["source_type"]:
            raise ValueError(f"unexpected source identity/type for {identity}")
        route = vehicle.find("route")
        edges = (route.get("edges") or "").split() if route is not None else []
        pair = correction["incompatible_pair"]
        if not any([left, right] == pair for left, right in zip(edges, edges[1:])):
            raise ValueError(f"expected incompatible pair is absent for {identity}")
        vehicle.set("type", correction["candidate_type"])
        records.append({"vehicle_id": identity, **correction,
                        "depart_s": float(vehicle.get("depart")),
                        "route_edges": edges,
                        "stop_count": len(vehicle.findall("stop"))})
    path.write_bytes(_canonical_xml(root))
    return records


def _apply_stop_lane_corrections(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    records = []
    for vehicle in root.findall("vehicle"):
        route = (vehicle.find("route").get("edges") or "").split()
        if "28718874#0" not in route:
            continue
        route_index = route.index("28718874#0")
        following = route[route_index + 1] if route_index + 1 < len(route) else None
        for stop in vehicle.findall("stop"):
            if stop.get("lane") != "28718874#0_0":
                continue
            identity = vehicle.get("id")
            if following not in {"10873193#0", "28718874#1"}:
                raise ValueError(f"unreviewed stop exit for {identity}: {following}")
            before = dict(stop.attrib)
            if following == "10873193#0":
                stop.set("lane", "28718874#0_2")
            requested_length = float(stop.get("endPos")) - float(stop.get("startPos"))
            if float(stop.get("endPos")) > 20.69:
                corrected_end = 20.69
                stop.set("endPos", f"{corrected_end:.2f}")
                stop.set("startPos", f"{corrected_end - requested_length:.2f}")
            stop.attrib.pop("friendlyPos", None)
            if dict(stop.attrib) != before:
                records.append({
                    "vehicle_id": identity, "following_edge": following,
                    "source": before, "candidate": dict(stop.attrib),
                    "policy": "route_aware_same_edge_stop_lane_and_extent_correction",
                    "assumption": "left-turn services use synthetic in-lane lane 2; the original stop extent is retained and only shifted when it exceeded the unchanged v5 edge boundary",
                })
    path.write_bytes(_canonical_xml(root))
    return records


def prepare(output_dir: str | Path) -> Path:
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite synthetic input candidate: {target}")
    target.mkdir(parents=True)
    try:
        source_network = ROOT / "simulation/maps/tahrirupdated.net.xml"
        source_routes = ROOT / "simulation/maps/routes_night.rou.xml"
        source_additions = [ROOT / "simulation/maps/tahrir.add.xml",
                            ROOT / "simulation/maps/vehtype.add.xml"]
        corrected_stops = target / "tahrir.synthetic-stops.add.xml"
        stopping_place_corrections = _correct_stopping_places(
            source_additions[0], corrected_stops)
        additions = [corrected_stops, source_additions[1]]
        network = target / "tahrir_synthetic_signal_v1.net.xml"
        changes = _network_variant(source_network, network)
        expansion_cfg = target / "expansion-source.sumocfg"
        _sumocfg(expansion_cfg, network=network, routes=source_routes,
                 begin=79200, end=86400, additions=additions)
        spec = {
            "schema": EXPANSION_SPEC_SCHEMA,
            "expansion_id": "night-explicit-sorted-diagnostic-v1",
            "sumo_version": "1.26.0", "algorithm": EXPANSION_ALGORITHM,
            "source_routes": _identity(source_routes),
            "configuration": _identity(expansion_cfg), "network": _identity(network),
            "additional_files": [_identity(item) for item in additions],
            "begin_s": 79200, "end_s": 86400, "publication_approved": False,
        }
        spec_path = target / "expansion-spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        expanded = target / "expanded"
        expansion = expand_number_flows(spec_path, sha256_file(spec_path), output_dir=expanded)
        expansion_manifest = expanded / "expansion_manifest.json"
        verify_expansion(expanded, expected_manifest_sha256=sha256_file(expansion_manifest),
                         expected_spec_sha256=sha256_file(spec_path))
        expanded_explicit = expanded / "routes.explicit.rou.xml"
        explicit = target / "routes.explicit.corrected.rou.xml"
        stop_corrections = _correct_invalid_stops(
            expanded_explicit, network, additions, explicit)
        class_corrections = _apply_vehicle_class_corrections(explicit)
        stop_lane_corrections = _apply_stop_lane_corrections(explicit)
        route_integrity = validate_explicit_route_integrity(
            explicit, network, additions)
        smoke = target / "night-900s.sumocfg"
        overnight = target / "night-full.sumocfg"
        _sumocfg(smoke, network=network, routes=explicit, begin=79200, end=80100,
                 additions=additions)
        _sumocfg(overnight, network=network, routes=explicit, begin=79200, end=86400,
                 additions=additions)
        manifest = {
            "schema": SCHEMA, "status": "engineering_candidate_not_publication_approved",
            "publication_approved": False,
            "source_network": _identity(source_network), "source_routes": _identity(source_routes),
            "source_additional_files": [_identity(item) for item in source_additions],
            "stopping_place_corrections": stopping_place_corrections,
            "network_variant": {**_identity(network), "signal_changes": changes},
            "expansion_spec": _identity(spec_path),
            "expansion_manifest": _identity(expansion_manifest),
            "expanded_routes_intermediate": _identity(expanded_explicit),
            "explicit_routes": {**_identity(explicit),
                "physical_fingerprint_sha256": _physical_fingerprint(
                    ET.parse(explicit).getroot().findall("vehicle")),
                "counts": expansion["counts"],
                "vehicle_class_corrections": class_corrections,
                "vehicle_class_correction_count": len(class_corrections),
                "stop_lane_corrections": stop_lane_corrections,
                "stop_lane_correction_count": len(stop_lane_corrections),
                "route_integrity": route_integrity,
                "invalid_stop_corrections": stop_corrections,
                "invalid_stop_correction_count": len(stop_corrections)},
            "sumo_configs": {"smoke": _identity(smoke), "full": _identity(overnight)},
            "limitations": [
                "synthetic signal priority assumptions require bounded verification",
                "not field calibrated or traffic-engineering approved",
                "no model checkpoint is authorized by this manifest",
            ],
        }
        manifest_path = target / "input-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        return manifest_path
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(prepare(args.output_dir))
