"""One-run, nonpublication SUMO/DQN diagnostic with explicit telemetry.

This entry point does not call or weaken ``train_research``. It creates fresh,
untrained agents, never resumes a checkpoint, and labels every artifact as an
uncalibrated engineering diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.env.sumo_env import (
    BOUNDED_OCCUPANCY_OBSERVATION_CONTRACT_VERSION,
    DQN_OBSERVATION_CONTRACT_VERSION,
    SumoEnv,
)
from DeepQN.training.reproducibility import seed_all

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sumoflow-exploratory-training-diagnostic-v1"
_EMERGENCY_COUNTER = re.compile(
    r"^\s*Emergency\s+Braking\s*:\s*(\d+)\s*$", re.IGNORECASE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32; psapi = ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize) if ok else None


def _sumo_version(binary: str) -> str:
    result = subprocess.run([binary, "--version"], capture_output=True, text=True,
                            timeout=15, check=True)
    return (result.stdout or result.stderr).splitlines()[0].strip()


def _collect_sumo_telemetry(*, error_log: Path, message_log: Path,
                            diagnostic_log: Path) -> dict:
    """Reconcile required SUMO logs without double-counting repeated lines."""
    contents = {}
    failures = []
    for label, path in (("error_log", error_log), ("message_log", message_log),
                        ("diagnostic_log", diagnostic_log)):
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            contents[label] = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{label} missing or unreadable: {exc}")

    counter_values = []
    for line in contents.get("message_log", "").splitlines():
        match = _EMERGENCY_COUNTER.match(line)
        if match:
            counter_values.append(int(match.group(1)))
    if not counter_values:
        failures.append("message_log lacks required Emergency Braking counter")
        emergency_count = None
    elif len(set(counter_values)) != 1:
        failures.append("message_log has contradictory Emergency Braking counters")
        emergency_count = None
    else:
        emergency_count = counter_values[0]

    events = {}
    for source, content in contents.items():
        for raw in content.splitlines():
            line = " ".join(raw.strip().split())
            lowered = line.lower()
            if not line or _EMERGENCY_COUNTER.match(line):
                continue
            if not any(token in lowered for token in (
                    "warning", "error", "teleport", "emergency braking")):
                continue
            key = lowered
            event = events.setdefault(key, {"message": line, "sources": []})
            if source not in event["sources"]:
                event["sources"].append(source)

    adverse_events = list(events.values())
    detailed_emergency_count = sum(
        "emergency braking" in event["message"].lower()
        for event in adverse_events)
    if (emergency_count is not None and detailed_emergency_count
            and detailed_emergency_count != emergency_count):
        failures.append(
            "Emergency Braking detail count contradicts message-log counter: "
            f"details={detailed_emergency_count}, counter={emergency_count}")

    categories = {
        "ignored_departure_order": sum(
            "should be sorted" in event["message"].lower()
            for event in adverse_events),
        "emergency_braking": emergency_count,
        "emergency_braking_detailed_events": detailed_emergency_count,
        "emergency_stops": sum(
            "performs emergency stop" in event["message"].lower()
            for event in adverse_events),
        "teleport": sum("teleport" in event["message"].lower()
                        for event in adverse_events),
        "unsafe_signal": sum(
            "unsafe green" in event["message"].lower()
            or "missing yellow" in event["message"].lower()
            for event in adverse_events),
    }
    return {
        "valid": not failures,
        "failures": failures,
        "required_sources": {
            "error_log": str(error_log), "message_log": str(message_log),
            "diagnostic_log": str(diagnostic_log)},
        "adverse_events": adverse_events,
        "categories": categories,
    }


def _save_diagnostic_checkpoints(agents, checkpoint_dir: str | Path,
                                 *, behavior_valid: bool) -> list[dict]:
    if not behavior_valid:
        return []
    agents.save(checkpoint_dir, 1)
    return [
        {"path": str(item), "sha256": sha256_file(item)}
        for item in sorted(Path(checkpoint_dir).glob("*.pt"))
    ]


def _patch_sumo_config(source: Path, run_dir: Path, *, begin_s: float,
                       end_s: float) -> Path:
    tree = ET.parse(source); root = tree.getroot()
    input_node = root.find("input")
    if input_node is None:
        raise ValueError("SUMO configuration has no input section")
    time_node = root.find("time")
    if time_node is None:
        time_node = ET.SubElement(root, "time")
    for tag, value in (("begin", begin_s), ("end", end_s)):
        node = time_node.find(tag)
        if node is None:
            node = ET.SubElement(time_node, tag)
        node.set("value", f"{float(value):g}")
    for tag in ("net-file", "route-files", "additional-files"):
        node = input_node.find(tag)
        if node is None or not node.get("value"):
            raise ValueError(f"SUMO configuration lacks {tag}")
        resolved = []
        for item in node.get("value").split(","):
            path = (source.parent / item.strip()).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing SUMO {tag} input: {path}")
            resolved.append(path.as_posix())
        node.set("value", ",".join(resolved))
    gui = input_node.find("gui-settings-file")
    if gui is not None and gui.get("value"):
        gui_path = (source.parent / gui.get("value")).resolve()
        if not gui_path.is_file():
            raise FileNotFoundError(f"missing SUMO gui-settings-file input: {gui_path}")
        gui.set("value", gui_path.as_posix())
    output = root.find("output")
    if output is None:
        output = root.find("o")
    if output is None:
        output = ET.SubElement(root, "output")
    desired = {
        "tripinfo-output": "tripinfo.xml", "emission-output": "emission.xml",
        "summary-output": "summary.xml", "statistic-output": "statistic.xml",
        "collision-output": "collisions.xml",
    }
    for tag, filename in desired.items():
        node = output.find(tag)
        if node is None:
            node = ET.SubElement(output, tag)
        node.set("value", (run_dir / filename).resolve().as_posix())
    for tag in ("fcd-output",):
        node = output.find(tag)
        if node is not None:
            output.remove(node)
    processing = root.find("processing")
    if processing is None:
        processing = ET.SubElement(root, "processing")
    for tag, value in (("collision.action", "warn"),
                       ("tripinfo-output.write-unfinished", "true"),
                       ("tripinfo-output.write-undeparted", "true")):
        node = processing.find(tag)
        if node is None:
            node = ET.SubElement(processing, tag)
        node.set("value", value)
    report = root.find("report")
    if report is None:
        report = ET.SubElement(root, "report")
    node = report.find("no-warnings")
    if node is None:
        node = ET.SubElement(report, "no-warnings")
    node.set("value", "false")
    target = run_dir / "effective.sumocfg"
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target


def _validate(config: dict, diagnostic: dict) -> None:
    if diagnostic.get("schema") != SCHEMA or diagnostic.get("episodes") != 1:
        raise ValueError("diagnostic schema requires exactly one episode")
    if diagnostic.get("classification") != "nonpublication_uncalibrated_synthetic_diagnostic":
        raise ValueError("diagnostic classification is invalid")
    if diagnostic.get("profile") != "night":
        raise ValueError("this diagnostic is restricted to the night profile")
    timeout = diagnostic.get("wall_clock_timeout_s")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("wall-clock timeout must be finite and positive")
    simulation = config["simulation"]
    if (simulation.get("research_evaluation") is not True
            or simulation.get("comparative_research") is not False
            or simulation.get("physical_trace_mode") != "research_fail_closed"
            or simulation.get("lane_observation_mode") != "research_fail_closed"
            or simulation.get("waiting_time_memory_s") != 100
            or simulation.get("terminate_on_empty") is not False):
        raise ValueError("strict fixed-horizon diagnostic simulation contract is incomplete")
    if config["bilstm"].get("mode") != "no_forecast":
        raise ValueError("diagnostic must be forecast-free")
    observation_contract = config["state"].get("observation_contract")
    if observation_contract not in {
            DQN_OBSERVATION_CONTRACT_VERSION,
            BOUNDED_OCCUPANCY_OBSERVATION_CONTRACT_VERSION}:
        raise ValueError("diagnostic observation contract is unsupported")
    if (observation_contract == BOUNDED_OCCUPANCY_OBSERVATION_CONTRACT_VERSION
            and config["state"].get("lane_occupancy_mode")
            != "bounded_saturation_v1"):
        raise ValueError("v4 diagnostic requires explicit bounded occupancy")
    if config["state"].get("state_dim") != 37 or config["dqn"].get("state_dim") != 37:
        raise ValueError("diagnostic requires matching 37-dimensional state")
    if len(config.get("tls_junctions", {})) != 7:
        raise ValueError("diagnostic requires exactly seven configured TLS agents")


def run(path: str | Path) -> Path:
    diagnostic_path = Path(path).resolve()
    diagnostic = yaml.safe_load(diagnostic_path.read_text(encoding="utf-8"))
    base_path = (diagnostic_path.parent / diagnostic["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = _merge(base, diagnostic["overrides"])
    config["simulation"]["seed"] = diagnostic["sumo_seed"]
    _validate(config, diagnostic)

    output_root = (diagnostic_path.parent / diagnostic["output_root"]).resolve()
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    profile = diagnostic["profile"]
    source_sumocfg = (ROOT / config["profiles"][profile]["config"]).resolve()
    effective_sumocfg = _patch_sumo_config(
        source_sumocfg, run_dir,
        begin_s=config["profiles"][profile]["begin"],
        end_s=config["profiles"][profile]["end"],
    )
    config["profiles"][profile]["config"] = str(effective_sumocfg)
    config["training"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    config["training"]["log_dir"] = str(run_dir / "logs")
    config["simulation"]["message_log_path"] = str(run_dir / "sumo-messages.log")
    config["simulation"]["error_log_path"] = str(run_dir / "sumo-errors.log")
    effective_yaml = run_dir / "effective-diagnostic.yaml"
    effective_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    started = time.monotonic(); cpu_started = time.process_time()
    deadline = started + float(diagnostic["wall_clock_timeout_s"])
    memory_samples = [value for value in [_working_set_bytes()] if value is not None]
    seeds = seed_all(int(diagnostic["training_seed"]), deterministic_torch=True)
    agents = MultiAgentDQN(list(config["tls_junctions"]), config["dqn"],
                           observation_contract=config["state"]["observation_contract"])
    env = SumoEnv(config, profile=profile, port=int(diagnostic["traci_port"]))
    intervals = []; reward_totals = {key: 0.0 for key in (
        "interval_delay_vehicle_s", "interval_co2_mg", "interval_arrivals",
        "interval_pressure_vehicle_s", "reward_unclipped", "reward_clipped")}
    requested = {tid: {"0": 0, "1": 0} for tid in agents.agents}
    accepted_keep = {tid: 0 for tid in agents.agents}
    accepted_switch = {tid: 0 for tid in agents.agents}; reason_counts = {}
    optimizer_updates = 0; loss_values = []; q_values = []
    completed = False; failure = None; final_info = {}
    try:
        observations = env.reset()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic wall-clock limit exceeded")
            with torch.no_grad():
                for tid, agent in agents.agents.items():
                    tensor = torch.as_tensor(observations[tid], dtype=torch.float32,
                                             device=agent.device)
                    values = agent.online_net(tensor).detach().cpu().numpy().reshape(-1)
                    q_values.extend(float(value) for value in values)
            actions = agents.act(observations, eval_mode=False)
            next_obs, rewards, done, info = env.step(actions)
            for tid, action in actions.items():
                requested[tid][str(action)] += 1
                decision = info["action_decisions"][tid]
                reason = decision["reason"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if reason == "accepted_keep": accepted_keep[tid] += 1
                if reason == "accepted_switch":
                    accepted_switch[tid] += 1
            for key in reward_totals:
                reward_totals[key] += float(info.get(key, 0.0))
            agents.push(observations, actions, rewards, next_obs,
                        terminated=info.get("terminated", done),
                        truncated=info.get("truncated", False),
                        valid=info.get("transition_valid", True),
                        interval_duration_s=info.get("interval_duration_s", 0.0))
            losses = agents.update()
            for value in losses.values():
                if value is not None:
                    optimizer_updates += 1; loss_values.append(float(value))
            intervals.append({"decision_index": len(intervals), "sim_time": info["sim_time"],
                "interval_duration_s": info["interval_duration_s"], "rewards": rewards,
                "requested_actions": info["requested_actions"],
                "action_decisions": info["action_decisions"],
                "physical_action_trace": info["action_trace"],
                "action_trace_sha256": info["action_trace_sha256"],
                **{key: info.get(key) for key in reward_totals}})
            observations = next_obs; final_info = info
            value = _working_set_bytes()
            if value is not None: memory_samples.append(value)
            if done:
                completed = bool(info.get("truncated") and info.get("sim_time") >= env.sim_end)
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        diagnostic_log = str(getattr(env, "_sumo_diagnostic_path", ""))
        env.close()

    wall_s = time.monotonic() - started; cpu_s = time.process_time() - cpu_started
    stats = agents.episode_stats()
    outputs = {}
    for name in ("tripinfo.xml", "emission.xml", "summary.xml", "statistic.xml",
                 "collisions.xml", "effective.sumocfg", "effective-diagnostic.yaml",
                 "sumo-messages.log", "sumo-errors.log"):
        item = run_dir / name
        if item.is_file(): outputs[name] = {"path": str(item), "sha256": sha256_file(item)}
    message_log = run_dir / "sumo-messages.log"
    error_log = run_dir / "sumo-errors.log"
    telemetry = _collect_sumo_telemetry(
        error_log=error_log, message_log=message_log,
        diagnostic_log=Path(diagnostic_log))
    categories = telemetry["categories"]
    final_summary = {}
    summary_path = run_dir / "summary.xml"
    if summary_path.is_file():
        steps = ET.parse(summary_path).getroot().findall("step")
        if steps:
            final_summary = dict(steps[-1].attrib)
    required_behavior_valid = (
        telemetry["valid"]
        and not telemetry["adverse_events"]
        and categories["emergency_braking"] == 0
        and int(final_summary.get("collisions", 0)) == 0
        and int(final_summary.get("teleports", 0)) == 0
    )
    behavior_valid = bool(completed and required_behavior_valid)
    checkpoint_files = _save_diagnostic_checkpoints(
        agents, config["training"]["checkpoint_dir"],
        behavior_valid=behavior_valid)
    report = {
        "schema": SCHEMA, "classification": diagnostic["classification"],
        "publication_compatible": False,
        "diagnostic_behavior_valid": behavior_valid,
        "completed_full_horizon": completed,
        "failure": failure, "profile": profile, "sumo_version": _sumo_version(config["simulation"]["sumo_binary"]),
        "sim_begin_s": config["profiles"][profile]["begin"],
        "sim_end_s": config["profiles"][profile]["end"],
        "final_sim_time_s": final_info.get("sim_time"), "wall_clock_s": wall_s,
        "process_cpu_s": cpu_s, "process_cpu_to_wall_ratio": cpu_s / wall_s if wall_s else None,
        "python_process_peak_observed_working_set_bytes": max(memory_samples) if memory_samples else None,
        "resource_scope_limitation": "CPU/RAM cover the Python process; SUMO child-process peaks are not measured",
        "device": str(next(iter(agents.agents.values())).device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "seeds": seeds, "sumo_seed": diagnostic["sumo_seed"], "decision_count": len(intervals),
        "optimizer_updates": optimizer_updates, "loss": {
            "count": len(loss_values), "mean": float(np.mean(loss_values)) if loss_values else None,
            "min": min(loss_values) if loss_values else None, "max": max(loss_values) if loss_values else None},
        "q_values": {"count": len(q_values), "mean": float(np.mean(q_values)) if q_values else None,
            "min": min(q_values) if q_values else None, "max": max(q_values) if q_values else None},
        "agent_stats": stats, "reward_totals": reward_totals,
        "requested_action_counts": requested,
        "accepted_keep_counts": accepted_keep,
        "accepted_switch_counts": accepted_switch,
        "action_reason_counts": reason_counts, "final_info": final_info,
        "sumo_diagnostic_log": diagnostic_log,
        "sumo_message_log": str(message_log),
        "sumo_error_log": str(error_log),
        "sumo_telemetry_valid": telemetry["valid"],
        "sumo_telemetry_failures": telemetry["failures"],
        "sumo_warning_categories": categories,
        "sumo_warning_error_events": telemetry["adverse_events"],
        "final_summary": final_summary,
        "inputs": {"diagnostic_yaml": {"path": str(diagnostic_path), "sha256": sha256_file(diagnostic_path)},
            "base_yaml": {"path": str(base_path), "sha256": sha256_file(base_path)},
            "source_sumocfg": {"path": str(source_sumocfg), "sha256": sha256_file(source_sumocfg)}},
        "outputs": outputs,
        "diagnostic_checkpoints": checkpoint_files,
        "checkpoint_manifest_limitation": "legacy diagnostic weight dictionaries only; no research ensemble manifest or publication compatibility",
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (run_dir / "decision-telemetry.json").write_text(
        json.dumps(intervals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = run_dir / "diagnostic-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not completed or not required_behavior_valid:
        raise RuntimeError(
            f"diagnostic did not pass completion/behavior checks; evidence: "
            f"{report_path}; {failure}"
        )
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
