"""Persistent full-night DQN training for nonpublication engineering evidence.

This entry point neither calls nor weakens ``train_research``.  Its snapshots
are complete interruption-resume states but are always classified as
nonpublication synthetic exploratory artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.evaluation.demand_provenance import sha256_file
from DeepQN.evaluation.metrics import load_metrics_from_paths, parse_scheduled_demand
from DeepQN.evaluation.prepare_night_synthetic import validate_explicit_route_integrity
from DeepQN.env.sumo_env import SumoEnv
from DeepQN.training.diagnostic_train import (
    _collect_sumo_telemetry,
    _merge,
    _patch_sumo_config,
)
from DeepQN.training.reproducibility import seed_all
from DeepQN.training.research_resume import (
    ResumeContracts,
    TrainingCursor,
    publish_training_snapshot,
    restore_training_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sumoflow-fullnight-exploratory-training-v1"
CLASSIFICATION = "nonpublication_uncalibrated_synthetic_exploratory_training"
SNAPSHOT_CLASSIFICATION = "nonpublication_exploratory_training_state"
IMPLEMENTATION_FILES = (
    "DeepQN/training/exploratory_train.py",
    "DeepQN/training/diagnostic_train.py",
    "DeepQN/training/research_resume.py",
    "DeepQN/env/sumo_env.py",
    "DeepQN/env/reward.py",
    "DeepQN/agent/dqn_agent.py",
    "DeepQN/agent/replay_buffer.py",
    "DeepQN/agent/network.py",
    "DeepQN/evaluation/metrics.py",
    "DeepQN/evaluation/prepare_night_synthetic.py",
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _git_identity() -> dict:
    safe = f"safe.directory={ROOT.as_posix()}"
    commit = subprocess.run(
        ["git", "-c", safe, "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(
        ["git", "-c", safe, "status", "--porcelain"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _validate_input_manifest(path: Path) -> tuple[dict, dict[str, Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "sumoflow-night-synthetic-input-candidate-v6"
            or manifest.get("publication_approved") is not False):
        raise ValueError("exploratory training requires the unapproved v6 synthetic input manifest")
    integrity = manifest.get("explicit_routes", {}).get("route_integrity", {})
    if (integrity.get("schema") != "sumoflow-explicit-route-integrity-v1"
            or integrity.get("vehicle_count") != 2106):
        raise ValueError("v6 route-integrity evidence is missing or incompatible")
    identities = {
        "network": manifest["network_variant"],
        "routes": manifest["explicit_routes"],
        "sumocfg": manifest["sumo_configs"]["full"],
    }
    paths = {}
    for name, identity in identities.items():
        candidate = Path(identity["path"]).resolve()
        if not candidate.is_file() or sha256_file(candidate) != identity.get("sha256"):
            raise ValueError(f"{name} input is missing or changed")
        paths[name] = candidate
    expansion_spec_path = Path(manifest["expansion_spec"]["path"]).resolve()
    if (not expansion_spec_path.is_file()
            or sha256_file(expansion_spec_path) != manifest["expansion_spec"]["sha256"]):
        raise ValueError("v6 expansion specification is missing or changed")
    expansion_spec = json.loads(expansion_spec_path.read_text(encoding="utf-8"))
    additions = []
    for identity in expansion_spec.get("additional_files", []):
        addition = Path(identity["path"]).resolve()
        if not addition.is_file() or sha256_file(addition) != identity.get("sha256"):
            raise ValueError("v6 additional input is missing or changed")
        additions.append(addition)
    try:
        validate_explicit_route_integrity(paths["routes"], paths["network"], additions)
    except (OSError, ValueError) as exc:
        raise ValueError(f"v6 source-level route integrity failed: {exc}") from exc
    return manifest, paths


def _validate_config(config: dict, spec: dict) -> None:
    if spec.get("schema") != SCHEMA or spec.get("classification") != CLASSIFICATION:
        raise ValueError("invalid exploratory training identity")
    if spec.get("profile") != "night" or spec.get("episodes") != 3:
        raise ValueError("this entry point requires exactly three night episodes")
    port = spec.get("traci_port")
    if (not isinstance(port, int) or isinstance(port, bool)
            or not 1 <= port <= 65535):
        raise ValueError("traci_port must be an integer in [1, 65535]")
    if config["simulation"].get("port") != port:
        raise ValueError("effective simulation.port must equal the declared traci_port")
    timeout = spec.get("wall_clock_timeout_per_episode_s")
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("per-episode wall-clock timeout must be finite and positive")
    simulation = config["simulation"]
    if (simulation.get("step_length") != 1
            or simulation.get("decision_interval") != 10
            or simulation.get("terminate_on_empty") is not False
            or simulation.get("research_evaluation") is not True
            or simulation.get("physical_trace_mode") != "research_fail_closed"
            or simulation.get("lane_observation_mode") != "research_fail_closed"
            or simulation.get("waiting_time_memory_s") != 100):
        raise ValueError("full-night strict simulation contract is incomplete")
    if config["profiles"]["night"].get("begin") != 79200 \
            or config["profiles"]["night"].get("end") != 86400:
        raise ValueError("night training window must be [79200, 86400)")
    if config["bilstm"].get("mode") != "no_forecast":
        raise ValueError("exploratory training must be forecast-free")
    if config["dqn"].get("min_replay_size") != 1000:
        raise ValueError("normal min_replay_size=1000 is required")
    if len(config.get("tls_junctions", {})) != 7:
        raise ValueError("exactly seven TLS agents are required")
    if config["state"].get("observation_contract") \
            != "sumoflow-dqn-observation-v4-lane-occupancy-saturated":
        raise ValueError("v4 observation contract is required")


def _apply_runtime_spec(config: dict, spec: dict) -> dict:
    effective = copy.deepcopy(config)
    effective["simulation"]["port"] = spec.get("traci_port")
    return effective


def _implementation_identities() -> dict[str, dict]:
    identities = {}
    for relative in IMPLEMENTATION_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required implementation file is missing: {path}")
        identities[relative] = {"path": relative, "sha256": sha256_file(path)}
    return identities


def _verify_implementation_identities(expected: dict) -> None:
    actual = _implementation_identities()
    if expected != actual:
        changed = sorted(set(expected or {}) | set(actual))
        differences = [name for name in changed
                       if (expected or {}).get(name) != actual.get(name)]
        raise ValueError(f"resume implementation identity changed: {differences}")


def _environment_identity() -> dict:
    result = subprocess.run(["sumo", "--version"], capture_output=True,
                            text=True, timeout=15, check=True)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "sumo": (result.stdout or result.stderr).splitlines()[0].strip()}


def _contracts(run_id: str, run_manifest_hash: str, input_manifest_hash: str,
               routes_hash: str, network_hash: str, config: dict,
               environment_hash: str, git_commit: str) -> ResumeContracts:
    return ResumeContracts(
        run_id=run_id, run_manifest_sha256=run_manifest_hash,
        dataset_manifest_sha256=input_manifest_hash,
        demand_registry_sha256=routes_hash,
        split_registry_sha256=network_hash,
        observation_contract=config["state"]["observation_contract"],
        reward_contract=config["reward"].get("contract", "interval_v1"),
        forecast_contract="no_forecast",
        git_commit=git_commit, environment_sha256=environment_hash,
    )


def _finite_observations(observations: dict) -> bool:
    return bool(observations) and all(
        np.asarray(value).shape == (37,) and np.isfinite(value).all()
        for value in observations.values())


def _episode_integrity(telemetry: dict, metrics: dict, due_count: int,
                       final_summary: dict, completed: bool) -> tuple[bool, list[str]]:
    reasons = list(telemetry["failures"])
    def allowed_event(event: dict) -> bool:
        message = event["message"].lower()
        return ((message.startswith("warning:")
                 and any(token in message for token in (
                     "emergency braking", "emergency stop", "teleport",
                     "junction collision")))
                or message.startswith("teleports:"))
    unknown = [event["message"] for event in telemetry["adverse_events"]
               if not allowed_event(event)]
    if unknown:
        reasons.append(f"non-traffic SUMO warnings/errors: {unknown[:5]}")
    if telemetry["categories"]["ignored_departure_order"]:
        reasons.append("SUMO ignored out-of-order demand")
    if telemetry["categories"]["unsafe_signal"]:
        reasons.append("SUMO reported unsafe signal behavior")
    if not completed:
        reasons.append("episode did not reach the configured horizon")
    if not metrics.get("valid"):
        reasons.append(f"metric/demand reconciliation failed: {metrics.get('errors')}")
    try:
        if int(final_summary["loaded"]) != due_count:
            reasons.append(
                f"loaded demand {final_summary['loaded']} != due demand {due_count}")
    except (KeyError, TypeError, ValueError):
        reasons.append("final summary lacks a valid loaded-demand count")
    return not reasons, reasons


def _snapshot_classification(emergency: int, collisions: int,
                             teleports: int, emergency_stops: int = 0) -> str:
    return ("nonpublication_behavior_flagged"
            if emergency or emergency_stops or collisions or teleports
            else SNAPSHOT_CLASSIFICATION)


def _run_episode(*, agents: MultiAgentDQN, config: dict, episode: int,
                 sumo_seed: int, episode_dir: Path, wall_timeout_s: float,
                 route_path: Path, progress_path: Path) -> dict:
    episode_dir.mkdir(parents=True, exist_ok=False)
    effective = copy.deepcopy(config)
    profile = effective["profiles"]["night"]
    source = Path(profile["config"]).resolve()
    sumocfg = _patch_sumo_config(source, episode_dir,
        begin_s=profile["begin"], end_s=profile["end"])
    profile["config"] = str(sumocfg)
    effective["simulation"]["seed"] = sumo_seed
    effective["simulation"]["message_log_path"] = str(episode_dir / "sumo-messages.log")
    effective["simulation"]["error_log_path"] = str(episode_dir / "sumo-errors.log")
    (episode_dir / "effective-training.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")

    env = SumoEnv(effective, profile="night", port=int(effective["simulation"].get("port", 8813)))
    started = time.monotonic(); deadline = started + wall_timeout_s
    decisions = 0; losses = []; completed = False; failure = None
    reward_totals = {name: 0.0 for name in (
        "interval_delay_vehicle_s", "interval_co2_mg", "interval_arrivals",
        "interval_pressure_vehicle_s", "reward_unclipped", "reward_clipped")}
    saturation_count = 0; saturation_events = []
    updates_before = {tid: agent.total_updates for tid, agent in agents.agents.items()}
    try:
        observations = env.reset()
        if set(observations) != set(agents.agents) or not _finite_observations(observations):
            raise RuntimeError("initial observations are missing, malformed, or nonfinite")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("exploratory episode wall-clock limit exceeded")
            actions = agents.act(observations, eval_mode=False)
            next_obs, rewards, done, info = env.step(actions)
            if (not _finite_observations(next_obs)
                    or set(rewards) != set(agents.agents)
                    or any(not math.isfinite(float(value)) for value in rewards.values())):
                raise RuntimeError("nonfinite or incomplete environment transition")
            agents.push(observations, actions, rewards, next_obs,
                terminated=info.get("terminated", done),
                truncated=info.get("truncated", False),
                valid=info.get("transition_valid", False),
                interval_duration_s=info.get("interval_duration_s", 0.0))
            step_losses = agents.update()
            for value in step_losses.values():
                if value is not None:
                    if not math.isfinite(float(value)):
                        raise RuntimeError("nonfinite DQN loss")
                    losses.append(float(value))
            for name in reward_totals:
                value = float(info[name])
                if not math.isfinite(value):
                    raise RuntimeError(f"nonfinite reward component {name}")
                reward_totals[name] += value
            saturation_count = int(info.get("occupancy_saturation_count", 0))
            saturation_events = info.get("occupancy_saturation_events", [])
            observations = next_obs; decisions += 1
            if decisions == 1 or decisions % 10 == 0:
                _atomic_json(progress_path, {
                    "status": "episode_running", "episode": episode,
                    "decision_count": decisions,
                    "sim_time_s": float(info.get("sim_time")),
                    "wall_clock_s": time.monotonic() - started,
                    "replay_size_by_agent": {
                        tid: len(agent.replay)
                        for tid, agent in agents.agents.items()},
                    "optimizer_updates_cumulative_by_agent": {
                        tid: agent.total_updates
                        for tid, agent in agents.agents.items()},
                })
            if done:
                completed = bool(info.get("truncated") and info.get("sim_time") >= 86400)
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        diagnostic_path = Path(getattr(env, "_sumo_diagnostic_path", ""))
        env.close()

    telemetry = _collect_sumo_telemetry(
        error_log=episode_dir / "sumo-errors.log",
        message_log=episode_dir / "sumo-messages.log",
        diagnostic_log=diagnostic_path)
    metrics_obj = load_metrics_from_paths(
        tripinfo_path=episode_dir / "tripinfo.xml",
        emission_path=episode_dir / "emission.xml",
        summary_path=episode_dir / "summary.xml",
        profile="night", mode="exploratory_dqn", step_length_s=1.0,
        expected_begin_s=79200, expected_end_s=86400,
        route_paths=[route_path], require_demand_contract=True)
    metrics = metrics_obj.to_dict()
    final_summary = {}
    summary_path = episode_dir / "summary.xml"
    if summary_path.is_file():
        import xml.etree.ElementTree as ET
        steps = ET.parse(summary_path).getroot().findall("step")
        if steps:
            final_summary = dict(steps[-1].attrib)
    due_count = len(parse_scheduled_demand([route_path], 79200, 86400))
    integrity_valid, integrity_failures = _episode_integrity(
        telemetry, metrics, due_count, final_summary, completed)
    if failure:
        integrity_valid = False; integrity_failures.append(failure)

    agent_stats = agents.episode_stats()
    updates = {tid: agents.agents[tid].total_updates - updates_before[tid]
               for tid in agents.agents}
    emergency = telemetry["categories"]["emergency_braking"]
    emergency_stops = telemetry["categories"].get("emergency_stops", 0)
    collisions = int(metrics.get("collisions") or 0)
    teleports = int(metrics.get("teleports") or 0)
    record = {
        "schema": "sumoflow-exploratory-episode-record-v1",
        "classification": _snapshot_classification(
            int(emergency or 0), collisions, teleports,
            int(emergency_stops or 0)),
        "publication_compatible": False,
        "episode": episode, "sumo_seed": sumo_seed,
        "traci_port": int(effective["simulation"]["port"]),
        "sim_begin_s": 79200, "sim_end_s": 86400,
        "decision_count": decisions, "completed_horizon": completed,
        "execution_integrity_valid": integrity_valid,
        "integrity_failures": integrity_failures,
        "traffic_outcomes_require_review": bool(
            emergency or emergency_stops or collisions or teleports),
        "optimizer_updates_by_agent": updates,
        "optimizer_updates_aggregate": sum(updates.values()),
        "agent_stats": agent_stats,
        "loss": {"count": len(losses),
                 "mean": float(np.mean(losses)) if losses else None,
                 "min": min(losses) if losses else None,
                 "max": max(losses) if losses else None},
        "reward_totals": reward_totals,
        "occupancy_saturation_count": saturation_count,
        "occupancy_saturation_events": saturation_events,
        "sumo_telemetry": telemetry,
        "metrics": metrics, "final_summary": final_summary,
        "wall_clock_s": time.monotonic() - started,
        "input_hashes": {
            "route_file": sha256_file(route_path),
            "source_sumocfg": sha256_file(source),
            "effective_sumocfg": sha256_file(sumocfg),
            "network": sha256_file(Path(effective["simulation"]["net_file"])),
        },
        "outputs": {item.name: {"path": str(item.resolve()),
                    "sha256": sha256_file(item)}
                    for item in episode_dir.iterdir() if item.is_file()},
    }
    _atomic_json(episode_dir / "episode-report.json", record)
    return record


def _latest(run_dir: Path) -> dict:
    candidates = []
    path = run_dir / "LATEST.json"
    if path.is_file():
        pointer = json.loads(path.read_text(encoding="utf-8"))
        candidates.append(pointer)
    snapshot_root = run_dir / "snapshots"
    if snapshot_root.is_dir():
        for snapshot in snapshot_root.iterdir():
            manifest_path = snapshot / "manifest.json"
            marker_path = snapshot / "COMPLETE.json"
            if not manifest_path.is_file() or not marker_path.is_file():
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            manifest_hash = sha256_file(manifest_path)
            if marker.get("manifest_sha256") != manifest_hash:
                continue
            candidates.append({
                "schema": "sumoflow-exploratory-latest-v1",
                "run_id": manifest.get("contracts", {}).get("run_id"),
                "completed_episode": manifest.get("cursor", {}).get(
                    "completed_episode"),
                "snapshot_path": str(snapshot.resolve()),
                "manifest_sha256": manifest_hash,
                "checkpoint_classification": (
                    manifest.get("episode_record", {}).get(
                        "checkpoint_classification")),
                "publication_compatible": False,
            })
    valid = [candidate for candidate in candidates
             if isinstance(candidate.get("completed_episode"), int)
             and not isinstance(candidate.get("completed_episode"), bool)]
    if not valid:
        raise FileNotFoundError(f"no complete snapshot found under {run_dir}")
    return max(valid, key=lambda candidate: candidate["completed_episode"])


def _next_episode_attempt(run_dir: Path, episode: int) -> Path:
    parent = run_dir / "episodes"
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"episode-{episode:06d}-attempt-"
    attempts = [int(path.name.removeprefix(prefix)) for path in parent.glob(
        prefix + "*") if path.name.removeprefix(prefix).isdigit()]
    return parent / f"{prefix}{max(attempts, default=0) + 1:03d}"


def _authorize_resume_status(run_dir: Path, prior_status: str | None,
                             acknowledge_traffic_review: bool) -> None:
    if prior_status == "failed_integrity":
        raise RuntimeError(
            f"resume is blocked pending review of prior status {prior_status}")
    if prior_status != "stopped_for_collision_or_teleport_review":
        return
    if not acknowledge_traffic_review:
        raise RuntimeError(
            "resume requires --acknowledge-traffic-review; this does not "
            "certify safety or publication validity")
    _atomic_json(run_dir / "traffic-review-acknowledgment.json", {
        "schema": "sumoflow-exploratory-traffic-review-ack-v1",
        "classification": "nonpublication_behavior_flagged",
        "publication_compatible": False,
        "meaning": "operator reviewed preserved adverse traffic evidence and authorized exploratory replay continuation; no safety approval",
        "prior_status": prior_status,
    })


def run(config_path: str | Path, *, run_id: str | None = None,
        resume_run_dir: str | Path | None = None,
        acknowledge_traffic_review: bool = False,
        episode_runner: Callable = _run_episode) -> Path:
    spec_path = Path(config_path).resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    base_path = (spec_path.parent / spec["base_config"]).resolve()
    config = _merge(yaml.safe_load(base_path.read_text(encoding="utf-8")),
                    spec["overrides"])
    config = _apply_runtime_spec(config, spec)
    _validate_config(config, spec)
    manifest_path = (spec_path.parent / spec["input_manifest"]).resolve()
    input_manifest, inputs = _validate_input_manifest(manifest_path)
    if Path(config["profiles"]["night"]["config"]).resolve() != inputs["sumocfg"]:
        raise ValueError("configured full-night SUMO file differs from input manifest")
    if Path(config["simulation"]["net_file"]).resolve() != inputs["network"]:
        raise ValueError("configured network differs from input manifest")

    if resume_run_dir is None:
        if not run_id or not run_id.strip() or Path(run_id).name != run_id:
            raise ValueError("a safe explicit --run-id is required for a new run")
        output_root = (spec_path.parent / spec["output_root"]).resolve()
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        environment = _environment_identity()
        git = _git_identity()
        run_manifest = {
            "schema": SCHEMA, "classification": CLASSIFICATION,
            "publication_compatible": False, "run_id": run_id,
            "config": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            "base_config": {"path": str(base_path), "sha256": sha256_file(base_path)},
            "input_manifest": {"path": str(manifest_path),
                               "sha256": sha256_file(manifest_path)},
            "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                       for name, path in inputs.items()},
            "environment": environment, "git": git,
            "traci_port": int(spec["traci_port"]),
            "implementation_files": _implementation_identities(),
            "seeds": {"training": spec["training_seed"],
                      "sumo_start": spec["sumo_seed_start"]},
            "episodes": 3,
        }
        _atomic_json(run_dir / "run-manifest.json", run_manifest)
        seed_all(int(spec["training_seed"]), deterministic_torch=True)
        cursor = TrainingCursor(0, 1, 0, 0, 0,
                                int(spec["sumo_seed_start"]), True)
    else:
        run_dir = Path(resume_run_dir).resolve()
        run_manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
        if run_manifest.get("schema") != SCHEMA or run_manifest.get("publication_compatible") is not False:
            raise ValueError("resume run manifest is incompatible")
        run_id = run_manifest["run_id"]
        if (run_manifest["config"]["sha256"] != sha256_file(spec_path)
                or run_manifest["base_config"]["sha256"] != sha256_file(base_path)
                or run_manifest["input_manifest"]["sha256"] != sha256_file(manifest_path)):
            raise ValueError("resume configuration or input manifest changed")
        if run_manifest.get("traci_port") != spec.get("traci_port"):
            raise ValueError("resume TraCI port differs from the run manifest")
        _verify_implementation_identities(run_manifest.get("implementation_files"))
        if _environment_identity() != run_manifest["environment"]:
            raise ValueError("resume software/SUMO environment differs from the run manifest")
        if _git_identity()["commit"] != run_manifest["git"]["commit"]:
            raise ValueError("resume Git commit differs from the run manifest")
        progress_file = run_dir / "progress.json"
        if progress_file.is_file():
            prior_status = json.loads(progress_file.read_text(encoding="utf-8")).get("status")
            _authorize_resume_status(
                run_dir, prior_status, acknowledge_traffic_review)

    environment_hash = _canonical_hash(run_manifest["environment"])
    contracts = _contracts(
        run_id, sha256_file(run_dir / "run-manifest.json"),
        sha256_file(manifest_path), input_manifest["explicit_routes"]["sha256"],
        input_manifest["network_variant"]["sha256"], config,
        environment_hash, run_manifest["git"]["commit"])
    resume_pointer = None
    if resume_run_dir is not None:
        try:
            resume_pointer = _latest(run_dir)
        except FileNotFoundError:
            resume_pointer = None
    resume_has_snapshot = resume_pointer is not None
    if resume_run_dir is not None and not resume_has_snapshot:
        # Seed before constructing agents so a first-episode restart recreates
        # the same initial ensemble, not merely the same action RNG stream.
        seed_all(int(spec["training_seed"]), deterministic_torch=True)
    agents = MultiAgentDQN(list(config["tls_junctions"]), config["dqn"],
                           observation_contract=config["state"]["observation_contract"])
    if resume_run_dir is not None:
        if resume_has_snapshot:
            pointer = resume_pointer
            cursor = restore_training_snapshot(
                pointer["snapshot_path"], agents, contracts,
                pointer["manifest_sha256"])
            _atomic_json(run_dir / "LATEST.json", pointer)
        else:
            # An interruption before the first completed episode has no learned
            # state to restore. Re-seeding restarts that same episode exactly;
            # the prior partial attempt remains immutable evidence.
            cursor = TrainingCursor(0, 1, 0, 0, 0,
                                    int(spec["sumo_seed_start"]), True)

    progress_path = run_dir / "progress.json"
    while cursor.next_episode <= int(spec["episodes"]):
        current_manifest, current_inputs = _validate_input_manifest(manifest_path)
        if (current_manifest["explicit_routes"]["sha256"]
                != input_manifest["explicit_routes"]["sha256"]
                or current_manifest["network_variant"]["sha256"]
                != input_manifest["network_variant"]["sha256"]
                or current_inputs != inputs):
            raise RuntimeError("exploratory inputs changed between episodes")
        episode = cursor.next_episode
        episode_dir = _next_episode_attempt(run_dir, episode)
        record = episode_runner(
            agents=agents, config=config, episode=episode,
            sumo_seed=cursor.next_sumo_seed, episode_dir=episode_dir,
            wall_timeout_s=float(spec["wall_clock_timeout_per_episode_s"]),
            route_path=inputs["routes"], progress_path=progress_path)
        if not record["execution_integrity_valid"]:
            _atomic_json(progress_path, {"status": "failed_integrity",
                "episode": episode, "report": str(episode_dir / "episode-report.json"),
                "failures": record["integrity_failures"]})
            raise RuntimeError(f"episode {episode} failed execution integrity")

        next_cursor = TrainingCursor(
            completed_episode=episode, next_episode=episode + 1,
            profile_cursor=0, demand_cursor=0, sumo_run_index=episode,
            next_sumo_seed=int(spec["sumo_seed_start"]) + episode,
            episode_complete=True)
        published = publish_training_snapshot(
            run_dir / "snapshots", agents, next_cursor, contracts,
            episode_record={**record,
                "checkpoint_classification": record["classification"],
                "publication_compatible": False})
        snapshot_manifest = json.loads(
            (Path(published["path"]) / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_evidence = {
            "schema": "sumoflow-exploratory-checkpoint-evidence-v1",
            "classification": record["classification"],
            "publication_compatible": False,
            "snapshot_path": published["path"],
            "snapshot_manifest_sha256": published["manifest_sha256"],
            "constituent_hashes": {
                row["filename"]: row["sha256"]
                for row in snapshot_manifest["agents"]},
            "global_state_sha256": snapshot_manifest["global_state"]["sha256"],
        }
        _atomic_json(episode_dir / "checkpoint-evidence.json", checkpoint_evidence)
        pointer = {"schema": "sumoflow-exploratory-latest-v1",
                   "run_id": run_id, "completed_episode": episode,
                   "snapshot_path": published["path"],
                   "manifest_sha256": published["manifest_sha256"],
                   "checkpoint_classification": record["classification"],
                   "publication_compatible": False}
        _atomic_json(run_dir / "LATEST.json", pointer)
        _atomic_json(progress_path, {"status": "episode_complete",
            "completed_episode": episode, "target_episodes": spec["episodes"],
            "next_episode": episode + 1, "latest_snapshot": pointer,
            "traffic_outcomes_require_review": record["traffic_outcomes_require_review"]})
        cursor = next_cursor
        if (record["metrics"].get("collisions") or record["metrics"].get("teleports")):
            _atomic_json(progress_path, {"status": "stopped_for_collision_or_teleport_review",
                "completed_episode": episode, "latest_snapshot": pointer,
                "report": str(episode_dir / "episode-report.json")})
            raise RuntimeError("collision/teleport observed; manual implementation review required")

    _atomic_json(progress_path, {"status": "complete",
        "completed_episode": cursor.completed_episode,
        "target_episodes": spec["episodes"],
        "latest_snapshot": _latest(run_dir)})
    return progress_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id")
    group.add_argument("--resume-run-dir")
    parser.add_argument("--acknowledge-traffic-review", action="store_true")
    args = parser.parse_args()
    print(run(args.config, run_id=args.run_id,
              resume_run_dir=args.resume_run_dir,
              acknowledge_traffic_review=args.acknowledge_traffic_review))
