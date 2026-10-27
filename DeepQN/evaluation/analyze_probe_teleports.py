"""Extract unique SUMO teleport starts and bind them to frozen routes."""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

PATTERN = re.compile(
    r"Teleporting vehicle '([^']+)'; waited too long \(([^)]+)\), "
    r"lane='([^']+)', time=([0-9]+(?:\.[0-9]+)?)")


def analyze(probe_dir: str | Path) -> dict:
    root = Path(probe_dir).resolve()
    report = json.loads((root / "probe-report.json").read_text(encoding="utf-8"))
    cfg = ET.parse(root / "effective.sumocfg").getroot()
    route_path = Path(cfg.find("input/route-files").get("value"))
    routes = {v.get("id"): (v.find("route").get("edges") or "").split()
              for v in ET.parse(route_path).getroot().findall("vehicle")}
    events, seen = [], set()
    for match in PATTERN.finditer(report.get("stderr", "")):
        key = match.groups()
        if key in seen:
            continue
        seen.add(key)
        events.append({"vehicle_id": key[0], "reason": key[1],
                       "lane_id": key[2], "time_s": float(key[3]),
                       "route_edges": routes.get(key[0])})
    return {"schema": "sumoflow-teleport-analysis-v1",
            "source_probe": str(root), "unique_event_count": len(events),
            "events": events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(args.probe_dir)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
