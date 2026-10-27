"""Bounded, reproducible isolation of compiled-network v6 changes.

This engineering utility never modifies the source v5/v6 artifacts.  It builds
diagnostic variants in a caller-selected directory and executes only the
requested bounded interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tls(root: ET.Element, tls_id: str) -> ET.Element:
    return next(x for x in root.findall("tlLogic")
                if x.get("id") == tls_id and x.get("programID") == "0")


def _copy_component(base: ET.Element, candidate: ET.Element,
                    component: str) -> None:
    if component == "tls":
        for tls_id in ("96621068", "96621100"):
            old = _tls(base, tls_id)
            index = list(base).index(old)
            base.remove(old)
            base.insert(index, ET.fromstring(ET.tostring(_tls(candidate, tls_id))))
    elif component == "connections":
        wanted = [(x.get("from"), x.get("to"), x.get("fromLane"))
                  for x in candidate.findall("connection")
                  if x.get("from") == "10873186#8"
                  and x.get("to") == "10873186#9"]
        for src, dst, lane in wanted:
            source = next(x for x in base.findall("connection")
                          if (x.get("from"), x.get("to"), x.get("fromLane"))
                          == (src, dst, lane))
            target = next(x for x in candidate.findall("connection")
                          if (x.get("from"), x.get("to"), x.get("fromLane"))
                          == (src, dst, lane))
            source.attrib.clear(); source.attrib.update(target.attrib)
    elif component == "lengths":
        candidate_lanes = {x.get("id"): x for e in candidate.findall("edge")
                           for x in e.findall("lane")}
        for edge in base.findall("edge"):
            for lane in edge.findall("lane"):
                other = candidate_lanes.get(lane.get("id"))
                if other is not None and lane.get("length") != other.get("length"):
                    lane.set("length", other.get("length"))
    else:
        raise ValueError(component)


def _inventory(v5: ET.Element, v6: ET.Element) -> dict:
    lane5 = {x.get("id"): x.attrib for e in v5.findall("edge")
             for x in e.findall("lane")}
    lane6 = {x.get("id"): x.attrib for e in v6.findall("edge")
             for x in e.findall("lane")}
    lane_changes = {key: {"v5": lane5[key], "v6": lane6[key]}
                    for key in sorted(lane5)
                    if lane5[key] != lane6.get(key)}
    def conns(root: ET.Element) -> dict:
        return {(x.get("from"), x.get("to"), x.get("fromLane"),
                 x.get("linkIndex")): x.attrib
                for x in root.findall("connection")}
    c5, c6 = conns(v5), conns(v6)
    conn_changes = [{"key": key, "v5": c5.get(key), "v6": c6.get(key)}
                    for key in sorted(set(c5) | set(c6), key=str)
                    if c5.get(key) != c6.get(key)]
    tls = {}
    for tls_id in ("96621068", "96621100"):
        tls[tls_id] = {
            label: [{"duration": p.get("duration"), "state": p.get("state"),
                     "next": p.get("next")}
                    for p in _tls(root, tls_id).findall("phase")]
            for label, root in (("v5", v5), ("v6", v6))}
    return {"lane_attribute_changes": lane_changes,
            "connection_attribute_changes": conn_changes,
            "tls_program_changes": tls}


def _last_summary(path: Path) -> tuple[int, str | None]:
    if not path.is_file():
        return 0, None
    count, last = 0, None
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "step":
                count += 1; last = elem.get("time")
            elem.clear()
    except ET.ParseError:
        pass
    return count, last


def run(v5_path: Path, v6_path: Path, routes: Path, additions: list[Path],
        output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    v5, v6 = ET.parse(v5_path).getroot(), ET.parse(v6_path).getroot()
    variants = {"v5": (), "tls_only": ("tls",),
                "connections_only": ("connections",),
                "lengths_only": ("lengths",)}
    results = []
    for name, components in variants.items():
        folder = output / name; folder.mkdir()
        root = ET.fromstring(ET.tostring(v5))
        for component in components:
            _copy_component(root, v6, component)
        net = folder / "variant.net.xml"
        ET.ElementTree(root).write(net, encoding="utf-8", xml_declaration=True)
        summary, messages, errors = (folder / x for x in
                                     ("summary.xml", "messages.log", "errors.log"))
        command = ["sumo", "-n", str(net), "-r", str(routes),
                   "-a", ",".join(str(x) for x in additions),
                   "--begin", "79200", "--end", "79300", "--step-length", "1",
                   "--seed", "42001", "--no-step-log", "true",
                   "--ignore-route-errors", "false", "--summary-output", str(summary),
                   "--message-log", str(messages), "--error-log", str(errors)]
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        count, final = _last_summary(summary)
        results.append({"name": name, "components": list(components),
                        "network_sha256": _sha(net), "command": command,
                        "exit_code": exit_code, "timed_out": timed_out,
                        "wall_clock_s": time.monotonic()-started,
                        "summary_complete_steps": count, "final_summary_time": final,
                        "stdout": stdout, "stderr": stderr,
                        "message_log": messages.read_text(errors="replace") if messages.is_file() else None,
                        "error_log": errors.read_text(errors="replace") if errors.is_file() else None})
    report = {"schema": "sumoflow-v6-native-crash-ablation-v1",
              "sumo_version": subprocess.run(["sumo", "--version"], capture_output=True,
                                             text=True).stdout.splitlines()[0],
              "interval": {"begin": 79200, "end_exclusive": 79300,
                           "step_length": 1, "seed": 42001},
              "inputs": {"v5_network": {"path": str(v5_path), "sha256": _sha(v5_path)},
                         "v6_network": {"path": str(v6_path), "sha256": _sha(v6_path)},
                         "routes": {"path": str(routes), "sha256": _sha(routes)}},
              "change_inventory": _inventory(v5, v6), "results": results}
    path = output / "ablation-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--v5-network", required=True, type=Path)
    p.add_argument("--v6-network", required=True, type=Path)
    p.add_argument("--routes", required=True, type=Path)
    p.add_argument("--additional", required=True, type=Path, nargs="+")
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    print(run(a.v5_network.resolve(), a.v6_network.resolve(), a.routes.resolve(),
              [x.resolve() for x in a.additional], a.output_dir.resolve()))
