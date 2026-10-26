"""Bounded real-SUMO characterization of E1 detector edge cases.

This engineering harness creates isolated networks under a caller-selected
directory.  Its trajectory samples are an independent reference for the E1
``getVehicleData`` records under test.  It is not a Tahrir traffic experiment.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import traci


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_xml(path: Path, root: ET.Element) -> None:
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _netgenerate(path: Path, *, loop: bool, lanes: int = 1) -> None:
    command = ["netgenerate", "--grid", "--grid.x-number", "2" if loop else "4",
               "--grid.y-number", "2" if loop else "1", "--grid.x-length", "40",
               "--grid.y-length", "40", "--default.lanenumber", str(lanes),
               "--default.speed", "10", "--no-turnarounds", "true",
               "--output-file", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)


def _short_network(root: Path, length: float) -> Path:
    nodes = root / "nodes.nod.xml"
    edges = root / "edges.edg.xml"
    net = root / "network.net.xml"
    nr = ET.Element("nodes")
    positions = (0.0, 25.0, 25.0 + length, 45.0 + length)
    for index, x in enumerate(positions):
        ET.SubElement(nr, "node", id=f"n{index}", x=str(x), y="0")
    _write_xml(nodes, nr)
    er = ET.Element("edges")
    for index in range(3):
        ET.SubElement(er, "edge", id=f"e{index}", **{"from": f"n{index}",
                     "to": f"n{index + 1}", "numLanes": "1", "speed": "10"})
    _write_xml(edges, er)
    completed = subprocess.run(["netconvert", "--node-files", str(nodes),
                                "--edge-files", str(edges), "--output-file", str(net),
                                "--no-turnarounds", "true"],
                               capture_output=True, text=True, timeout=15)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return net


def _inputs(root: Path, net: Path, edges: list[str], detector_lanes: list[str],
            detector_position: float, *, step: float = .1) -> tuple[Path, Path, Path]:
    route = root / "demand.rou.xml"
    rr = ET.Element("routes")
    ET.SubElement(rr, "vType", id="probe", accel="10", decel="10", length="2",
                  minGap="0.1", maxSpeed="10", sigma="0", lcStrategic="0",
                  lcSpeedGain="0", lcCooperative="0")
    ET.SubElement(rr, "route", id="trajectory", edges=" ".join(edges))
    ET.SubElement(rr, "vehicle", id="probe", route="trajectory", type="probe",
                  depart="0", departLane="0", departSpeed="10")
    _write_xml(route, rr)
    additional = root / "detectors.add.xml"
    ar = ET.Element("additional")
    for index, lane in enumerate(detector_lanes):
        ET.SubElement(ar, "inductionLoop", id=f"e1_{index}", lane=lane,
                      pos=str(detector_position), period="1", file="e1.xml")
    _write_xml(additional, ar)
    config = root / "scenario.sumocfg"
    cr = ET.Element("configuration")
    inp = ET.SubElement(cr, "input")
    ET.SubElement(inp, "net-file", value=net.name)
    ET.SubElement(inp, "route-files", value=route.name)
    ET.SubElement(inp, "additional-files", value=additional.name)
    tm = ET.SubElement(cr, "time")
    ET.SubElement(tm, "begin", value="0")
    ET.SubElement(tm, "end", value="40")
    ET.SubElement(tm, "step-length", value=str(step))
    _write_xml(config, cr)
    return route, additional, config


def _run(root: Path, config: Path, *, lane_change=False, teleport=False,
         maximum_steps: int = 400) -> dict:
    label = f"e1-characterization-{root.name}"
    traci.start(["sumo", "-c", str(config), "--seed", "17", "--no-step-log",
                 "true", "--fcd-output", str(root / "trajectory.xml")],
                label=label, numRetries=0)
    connection = traci.getConnection(label)
    trajectory, raw_events, commands, teleports = [], {}, [], []
    changed = jumped = False
    try:
        for _ in range(maximum_steps):
            connection.simulationStep()
            now = float(connection.simulation.getTime())
            teleports.extend({"time_s": now, "kind": "start", "vehicle_id": vid}
                             for vid in connection.simulation.getStartingTeleportIDList())
            teleports.extend({"time_s": now, "kind": "end", "vehicle_id": vid}
                             for vid in connection.simulation.getEndingTeleportIDList())
            for detector_id in connection.inductionloop.getIDList():
                for row in connection.inductionloop.getVehicleData(detector_id):
                    key = (str(detector_id), str(row[0]), float(row[2]))
                    raw_events[key] = {"detector_id": str(detector_id),
                        "vehicle_id": str(row[0]), "entry_time_s": float(row[2]),
                        "leave_time_s": float(row[3])}
            if "probe" in connection.vehicle.getIDList():
                lane = connection.vehicle.getLaneID("probe")
                position = float(connection.vehicle.getLanePosition("probe"))
                x, y = connection.vehicle.getPosition("probe")
                trajectory.append({"time_s": now, "lane_id": lane,
                                   "lane_position_m": position, "x": float(x), "y": float(y)})
                if lane_change and not changed and lane == "B0C0_0" and position >= 14:
                    connection.vehicle.changeLane("probe", 1, 2.0)
                    commands.append({"time_s": now, "command": "changeLane", "target_lane": 1})
                    changed = True
                if teleport and not jumped and lane == "B0C0_0" and position >= 10:
                    before = position
                    connection.vehicle.moveTo("probe", "B0C0_0", 25.0)
                    commands.append({"time_s": now, "command": "moveTo",
                                     "from_position_m": before, "to_position_m": 25.0})
                    jumped = True
            if connection.simulation.getMinExpectedNumber() == 0:
                break
    finally:
        connection.close(True)
    return {"trajectory": trajectory,
            "events": [raw_events[key] for key in sorted(raw_events)],
            "commands": commands, "sumo_teleports": teleports}


def characterize(output_root: str | Path) -> dict:
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite characterization: {destination}")
    destination.mkdir(parents=True)
    scenarios = {}

    for name, lane_change, teleport in (("lane_change", True, False),
                                         ("controlled_move_to", False, True)):
        root = destination / name; root.mkdir()
        net = root / "network.net.xml"; _netgenerate(net, loop=False, lanes=2)
        route, additional, config = _inputs(
            root, net, ["A0B0", "B0C0", "C0D0"],
            ["B0C0_0", "B0C0_1"] if lane_change else ["B0C0_0"], 20)
        result = _run(root, config, lane_change=lane_change, teleport=teleport)
        scenarios[name] = {**result, "files": {p.name: _sha(p)
            for p in (net, route, additional, config, root / "trajectory.xml")}}

    root = destination / "cyclic_reentry"; root.mkdir()
    net = root / "network.net.xml"; _netgenerate(net, loop=True)
    route, additional, config = _inputs(root, net,
        ["A0B0", "B0B1", "B1A1", "A1A0", "A0B0", "B0B1"], ["A0B0_0"], 20)
    result = _run(root, config)
    scenarios["cyclic_reentry"] = {**result, "files": {p.name: _sha(p)
        for p in (net, route, additional, config, root / "trajectory.xml")}}

    for name, length, detector_position in (("short_edge", 5.0, .1),
                                             ("extreme_short_edge", .2, .1)):
        root = destination / name; root.mkdir()
        net = _short_network(root, length)
        route, additional, config = _inputs(
            root, net, ["e0", "e1", "e2"], ["e1_0"], detector_position, step=.1)
        result = _run(root, config)
        scenarios[name] = {**result, "files": {p.name: _sha(p)
            for p in (net, route, additional, config, root / "trajectory.xml")},
            "edge_length_m": length}

    payload = {"schema": "sumoflow-e1-edge-characterization-v1",
               "sumo_version": subprocess.run(["sumo", "--version"], capture_output=True,
                                                text=True, timeout=10).stdout.splitlines()[0],
               "step_length_s": .1, "seed": 17, "scenarios": scenarios}
    target = destination / "result.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory")
    arguments = parser.parse_args()
    characterize(arguments.output_directory)
