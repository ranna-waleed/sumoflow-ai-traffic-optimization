"""Bounded helper used by the real-SUMO E1 regression test."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import traci

from DeepQN.forecasting.contracts import load_approach_mapping
from DeepQN.forecasting.detector_events import (
    E1EventCollector, build_detector_mapping, collect_approach_snapshot,
    detector_feature_row,
    write_detector_events,
)


def run(config: Path, mapping_path: Path, net_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = build_detector_mapping(load_approach_mapping(mapping_path, net_path), net_path)
    collector = E1EventCollector(mapping)
    events = []
    feature_rows = []
    label = "detector_fixture"
    config_root = ET.parse(config).getroot()
    waiting_node = config_root.find(".//waiting-time-memory")
    if waiting_node is None or waiting_node.get("value") != "100":
        raise ValueError("fixture must explicitly configure waiting-time-memory=100")
    input_node = config_root.find("input")
    if input_node is None:
        raise ValueError("fixture config has no input section")
    for tag in ("net-file", "route-files"):
        node = input_node.find(tag)
        node.set("value", str((config.parent / node.get("value")).resolve()))
    additional_node = input_node.find("additional-files")
    additional_source = (config.parent / additional_node.get("value")).resolve()
    additional_root = ET.parse(additional_source).getroot()
    for detector in additional_root:
        filename = "detector-instant.xml" if detector.tag == "instantInductionLoop" else "detector-aggregate.xml"
        detector.set("file", str((output_dir / filename).resolve()))
    additional_copy = output_dir / "detectors.add.xml"
    ET.ElementTree(additional_root).write(additional_copy, encoding="utf-8", xml_declaration=True)
    additional_node.set("value", str(additional_copy.resolve()))
    config_copy = output_dir / "detector.sumocfg"
    ET.ElementTree(config_root).write(config_copy, encoding="utf-8", xml_declaration=True)
    traci.start([
        "sumo", "-c", str(config_copy.resolve()), "--no-step-log", "true",
        "--waiting-time-memory", "100",
    ], label=label, numRetries=10)
    connection = traci.getConnection(label)
    try:
        while connection.simulation.getTime() < 8:
            connection.simulationStep()
            rows = {spec.detector_id: connection.inductionloop.getVehicleData(spec.detector_id)
                    for spec in mapping.specs}
            new_events = collector.collect(rows)
            events.extend(new_events)
            snapshot = collect_approach_snapshot(
                connection, mapping, connection.simulation.getTime()
            )
            feature_rows.append({
                "time_s": snapshot.time_s,
                "rows": detector_feature_row(mapping, new_events, snapshot),
            })
    finally:
        connection.close()
    instant_path = output_dir / "detector-instant.xml"
    instant = []
    if instant_path.is_file():
        for event in ET.parse(instant_path).iter("instantOut"):
            instant.append({key: event.get(key) for key in ("id", "time", "state", "vehID")})
    event_path = output_dir / "detector-events.json"
    event_hash = write_detector_events(event_path, events, mapping, source_id="detector-fixture")
    return {
        "schema_version": mapping.schema_version,
        "mapping_sha256": mapping.sha256,
        "events": [event.__dict__ for event in events],
        "instant_events": instant,
        "feature_rows": feature_rows,
        "waiting_time_memory_s": 100.0,
        "event_source_path": str(event_path.resolve()),
        "event_source_sha256": event_hash,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("network", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = run(args.config, args.mapping, args.network, args.output_dir)
    if args.result.exists():
        raise FileExistsError(f"refusing to overwrite {args.result}")
    args.result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
