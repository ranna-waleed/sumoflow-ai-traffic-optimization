"""Bounded fixed-time probe for the hash-bound synthetic input candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path

from DeepQN.evaluation.demand_provenance import sha256_file
from DeepQN.evaluation.metrics import load_metrics_from_paths

SCHEMA = "sumoflow-fixed-time-synthetic-probe-v1"


def _parse_summary_with_partial_evidence(path: Path) -> tuple[list[ET.Element], str | None]:
    """Return every complete step preceding a truncated/native-crash tail."""
    if not path.is_file():
        return [], "summary output is missing"
    steps: list[ET.Element] = []
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "step":
                steps.append(ET.Element("step", dict(elem.attrib)))
            elem.clear()
    except (ET.ParseError, OSError) as exc:
        return steps, f"{type(exc).__name__}: {exc}"
    return steps, None


def _warning_categories(text: str) -> dict:
    lowered = text.lower()
    return {
        "ignored_departure_order": lowered.count("should be sorted"),
        "emergency_braking": lowered.count("emergency braking"),
        "teleport": lowered.count("teleporting vehicle"),
        "unsafe_signal": lowered.count("unsafe green") + lowered.count("missing yellow"),
    }


def _warning_lines(text: str) -> list[str]:
    return [line for line in text.splitlines()
            if ("warning" in line.lower() or "error" in line.lower()
                or "teleport" in line.lower())]


def _summary_horizon_coverage(config_root: ET.Element,
                              summary_steps: list[ET.Element]) -> dict:
    """Validate SUMO's half-open summary timestamps against effective config.

    SUMO writes one summary record for each simulated step timestamp in
    ``[begin, end)``. Thus a one-second run ending at 80100 has its final record
    at 80099, not 80100.
    """
    try:
        begin_node = config_root.find("time/begin")
        end_node = config_root.find("time/end")
        step_node = config_root.find("time/step-length")
        if any(node is None or node.get("value") is None
               for node in (begin_node, end_node, step_node)):
            raise ValueError("effective SUMO timing is incomplete")
        begin = Decimal(begin_node.get("value"))
        end = Decimal(end_node.get("value"))
        step_length = Decimal(step_node.get("value"))
    except (InvalidOperation, ValueError, TypeError) as exc:
        return {"valid": False, "reason": str(exc)}
    if (not begin.is_finite() or not end.is_finite()
            or not step_length.is_finite() or step_length <= 0 or end <= begin):
        return {"valid": False, "reason": "invalid effective SUMO timing"}

    expected_count = int(
        ((end - begin) / step_length).to_integral_value(
            rounding=ROUND_CEILING))
    actual_times = []
    try:
        for node in summary_steps:
            value = Decimal(node.get("time"))
            if not value.is_finite():
                raise ValueError("nonfinite summary timestamp")
            actual_times.append(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        return {"valid": False, "reason": str(exc),
                "expected_step_count": expected_count,
                "actual_step_count": len(summary_steps)}

    mismatch_index = next((index for index, actual in enumerate(actual_times)
                           if actual != begin + step_length * index), None)
    valid = len(actual_times) == expected_count and mismatch_index is None
    reason = None
    if len(actual_times) != expected_count:
        reason = (f"summary step count {len(actual_times)} does not match "
                  f"expected {expected_count}")
    elif mismatch_index is not None:
        reason = (f"summary timestamp mismatch at index {mismatch_index}: "
                  f"{actual_times[mismatch_index]} != "
                  f"{begin + step_length * mismatch_index}")
    return {
        "valid": valid,
        "reason": reason,
        "begin_s": str(begin),
        "end_s_exclusive": str(end),
        "step_length_s": str(step_length),
        "expected_step_count": expected_count,
        "actual_step_count": len(actual_times),
        "expected_final_timestamp_s": str(
            begin + step_length * (expected_count - 1)),
        "actual_final_timestamp_s": (str(actual_times[-1])
                                     if actual_times else None),
    }


def run(*, manifest_path: str | Path, output_dir: str | Path, seed: int,
        wall_clock_timeout_s: float, configuration: str = "smoke") -> Path:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") not in {
            "sumoflow-night-synthetic-input-candidate-v4",
            "sumoflow-night-synthetic-input-candidate-v5",
            "sumoflow-night-synthetic-input-candidate-v6"}
            or manifest.get("publication_approved") is not False):
        raise ValueError("invalid synthetic input candidate")
    if configuration not in {"smoke", "full"}:
        raise ValueError("configuration must be 'smoke' or 'full'")
    timeout = float(wall_clock_timeout_s)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("wall-clock timeout must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    base = manifest_path.parent
    identities = {
        "network": manifest["network_variant"],
        "routes": manifest["explicit_routes"],
        "sumocfg": manifest["sumo_configs"][configuration],
    }
    paths = {}
    for name, identity in identities.items():
        path = Path(identity["path"]).resolve()
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"{name} input is missing or changed")
        paths[name] = path
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(paths["sumocfg"]); root = tree.getroot()
    output = ET.SubElement(root, "output")
    outputs = {}
    for tag, filename in (("tripinfo-output", "tripinfo.xml"),
                          ("emission-output", "emission.xml"),
                          ("summary-output", "summary.xml"),
                          ("statistic-output", "statistic.xml"),
                          ("collision-output", "collisions.xml")):
        path = target / filename; outputs[filename] = path
        ET.SubElement(output, tag, value=str(path))
    ET.SubElement(output, "tripinfo-output.write-unfinished", value="true")
    ET.SubElement(output, "tripinfo-output.write-undeparted", value="true")
    random = ET.SubElement(root, "random_number")
    ET.SubElement(random, "seed", value=str(seed))
    effective = target / "effective.sumocfg"
    tree.write(effective, encoding="utf-8", xml_declaration=True)
    message_log = target / "sumo-messages.log"
    error_log = target / "sumo-errors.log"
    command = ["sumo", "-c", str(effective), "--no-step-log", "true",
               "--waiting-time-memory", "100", "--message-log", str(message_log),
               "--error-log", str(error_log)]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    wall = time.monotonic() - started
    text = "\n".join((result.stdout, result.stderr,
                       message_log.read_text(encoding="utf-8", errors="replace")
                       if message_log.is_file() else "",
                       error_log.read_text(encoding="utf-8", errors="replace")
                       if error_log.is_file() else ""))
    categories = _warning_categories(text)
    warning_lines = _warning_lines(text)
    summary, summary_parse_error = _parse_summary_with_partial_evidence(
        outputs["summary.xml"])
    horizon_coverage = _summary_horizon_coverage(root, summary)
    if summary_parse_error:
        horizon_coverage["valid"] = False
        horizon_coverage["parse_error"] = summary_parse_error
    final = summary[-1].attrib if summary else {}
    parse_errors = ({"summary.xml": summary_parse_error}
                    if summary_parse_error else {})
    try:
        metrics = load_metrics_from_paths(
            tripinfo_path=outputs["tripinfo.xml"],
            emission_path=outputs["emission.xml"],
            summary_path=outputs["summary.xml"],
            profile="night", mode="fixed_time", step_length_s=1.0,
            expected_begin_s=float(root.find("time/begin").get("value")),
            expected_end_s=float(root.find("time/end").get("value")),
            route_paths=[paths["routes"]], require_demand_contract=True,
        ).to_dict()
    except Exception as exc:
        parse_errors["metrics"] = f"{type(exc).__name__}: {exc}"
        metrics = {"valid": False, "errors": [parse_errors["metrics"]]}
    traffic_tokens = ("emergency braking", "teleport", "junction collision")
    integrity_warning_lines = [line for line in warning_lines
                               if not any(token in line.lower()
                                          for token in traffic_tokens)]
    execution_integrity_valid = (
        result.returncode == 0 and horizon_coverage["valid"]
        and metrics.get("valid") is True
        and categories["ignored_departure_order"] == 0
        and categories["unsafe_signal"] == 0
        and not integrity_warning_lines)
    traffic_behavior_valid = (
        not warning_lines and all(value == 0 for value in categories.values())
        and int(final.get("collisions", 0)) == 0
        and int(final.get("teleports", 0)) == 0)
    valid = (execution_integrity_valid if configuration == "full"
             else execution_integrity_valid and traffic_behavior_valid)
    report = {
        "schema": SCHEMA, "classification": "engineering_probe",
        "publication_approved": False, "valid": valid,
        "configuration": configuration,
        "execution_integrity_valid": execution_integrity_valid,
        "traffic_behavior_valid": traffic_behavior_valid,
        "command": command, "returncode": result.returncode, "wall_clock_s": wall,
        "warning_categories": categories, "warning_error_lines": warning_lines,
        "integrity_warning_error_lines": integrity_warning_lines,
        "horizon_coverage": horizon_coverage, "final_summary": final,
        "metrics": metrics, "partial_artifact_parse_errors": parse_errors,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                   for name, path in paths.items()},
        "input_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in outputs.items() if path.is_file()},
        "telemetry_logs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (("message_log", message_log),
                               ("error_log", error_log)) if path.is_file()},
        "effective_sumocfg": {"path": str(effective), "sha256": sha256_file(effective)},
        "stdout": result.stdout, "stderr": result.stderr,
    }
    report_path = target / "probe-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not valid:
        raise RuntimeError(f"fixed-time probe failed physical/input checks: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--wall-clock-timeout", required=True, type=float)
    parser.add_argument("--configuration", choices=("smoke", "full"),
                        default="smoke")
    args = parser.parse_args()
    print(run(manifest_path=args.manifest, output_dir=args.output_dir,
              seed=args.seed, wall_clock_timeout_s=args.wall_clock_timeout,
              configuration=args.configuration))
