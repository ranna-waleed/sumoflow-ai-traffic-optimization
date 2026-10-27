"""Paired, evaluation-only comparison for the frozen synthetic v6 scenario.

This is deliberately nonpublication infrastructure.  A held-out SUMO random
seed with one frozen route file is not an independent held-out demand sample.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.evaluation.demand_provenance import sha256_file
from DeepQN.evaluation.evaluate import run_controlled_episode
from DeepQN.evaluation.metrics import EpisodeMetrics
from DeepQN.evaluation.synthetic_probe import run as run_fixed_probe
from DeepQN.training.diagnostic_train import _merge
from DeepQN.training.exploratory_train import (
    _apply_runtime_spec, _validate_config, _validate_input_manifest,
)
from DeepQN.training.reproducibility import seed_all
from DeepQN.training.research_resume import ResumeContracts, restore_training_snapshot

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sumoflow-heldout-dqn-evaluation-v1"
CLASSIFICATION = "nonpublication_synthetic_exploratory_evaluation"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _failure_record(output: Path, error: Exception) -> dict:
    manifests = sorted(output.glob("manifest_*.json")) if output.is_dir() else []
    manifest = None
    if manifests:
        try:
            manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
    return {"status": "postprocessing_failed", "error_type": type(error).__name__,
            "error": str(error), "output_dir": str(output.resolve()),
            "manifest_path": str(manifests[-1].resolve()) if manifests else None,
            "manifest": manifest}


def _load_spec(path: Path) -> tuple[dict, dict, dict, dict[str, Path]]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (spec.get("schema") != SCHEMA or spec.get("classification") != CLASSIFICATION
            or spec.get("publication_approved") is not False):
        raise ValueError("invalid nonpublication evaluation specification")
    seeds = spec.get("heldout_seeds")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in seeds)):
        raise ValueError("heldout_seeds must be unique nonnegative integers")
    timeout = spec.get("wall_clock_timeout_per_run_s")
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("wall-clock timeout must be finite and positive")
    port = spec.get("traci_port_start")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65533:
        raise ValueError("traci_port_start must leave room for paired controlled runs")
    training_path = (path.parent / spec["training_config"]).resolve()
    training_spec = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    base_path = (training_path.parent / training_spec["base_config"]).resolve()
    config = _merge(yaml.safe_load(base_path.read_text(encoding="utf-8")),
                    training_spec["overrides"])
    config = _apply_runtime_spec(config, training_spec)
    _validate_config(config, training_spec)
    manifest_path = (path.parent / spec["input_manifest"]).resolve()
    input_manifest, inputs = _validate_input_manifest(manifest_path)
    if Path(config["profiles"]["night"]["config"]).resolve() != inputs["sumocfg"]:
        raise ValueError("evaluation config does not reference the frozen full-night input")
    return spec, config, input_manifest, inputs


def _ensemble_fingerprint(agents: MultiAgentDQN) -> dict:
    rows = {}
    for tls_id, agent in sorted(agents.agents.items()):
        digest = hashlib.sha256()
        for label, state in (("online", agent.online_net.state_dict()),
                             ("target", agent.target_net.state_dict())):
            digest.update(label.encode())
            for name, tensor in sorted(state.items()):
                digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        rows[tls_id] = {
            "weights_sha256": digest.hexdigest(), "epsilon": agent.epsilon,
            "replay_size": len(agent.replay), "total_steps": agent.total_steps,
            "total_updates": agent.total_updates,
        }
    return rows


def _restore_episode3(spec: dict, config: dict) -> tuple[MultiAgentDQN, dict]:
    run_dir = (Path(spec["_spec_path"]).parent / spec["training_run_dir"]).resolve()
    pointer = json.loads((run_dir / "LATEST.json").read_text(encoding="utf-8"))
    expected = spec["snapshot_manifest_sha256"]
    if pointer.get("completed_episode") != 3 or pointer.get("manifest_sha256") != expected:
        raise ValueError("the pinned episode-3 snapshot pointer is incompatible")
    snapshot = Path(pointer["snapshot_path"]).resolve()
    manifest_path = snapshot / "manifest.json"
    if sha256_file(manifest_path) != expected:
        raise ValueError("episode-3 snapshot manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = ResumeContracts(**manifest["contracts"])
    if (contracts.dataset_manifest_sha256 != spec["_input_manifest_sha256"]
            or contracts.demand_registry_sha256 != spec["_route_sha256"]
            or contracts.split_registry_sha256 != spec["_network_sha256"]
            or contracts.observation_contract != config["state"]["observation_contract"]
            or contracts.reward_contract != "interval_v1"
            or contracts.forecast_contract != "no_forecast"):
        raise ValueError("snapshot input, observation, reward, or forecast contracts differ")
    agents = MultiAgentDQN(list(config["tls_junctions"]), config["dqn"],
                           observation_contract=config["state"]["observation_contract"])
    cursor = restore_training_snapshot(snapshot, agents, contracts, expected)
    return agents, {"path": str(snapshot), "manifest_sha256": expected,
                    "completed_episode": cursor.completed_episode,
                    "classification": pointer["checkpoint_classification"],
                    "publication_compatible": False,
                    "constituent_hashes": {r["tls_id"]: r["sha256"]
                                             for r in manifest["agents"]}}


def _controlled(*, label: str, agents: MultiAgentDQN, config: dict, seed: int,
                output: Path, port: int, checkpoint: dict | None,
                wall_clock_timeout_s: float) -> dict:
    before = _ensemble_fingerprint(agents)
    metrics, manifest = run_controlled_episode(
        cfg=config, profile="night", agents=agents, controller=label,
        output_dir=output, run_index=0, seed=seed, checkpoint_manifest=None,
        model_mode="no_forecast", config_path=config["_evaluation_spec_path"],
        port=port, require_forecast_protocol=False,
        wall_clock_timeout_s=wall_clock_timeout_s)
    after = _ensemble_fingerprint(agents)
    if before != after:
        raise RuntimeError("evaluation mutated weights, replay, epsilon, or training counters")
    manifest.checkpoint_manifest = checkpoint
    manifest.controller_diagnostics["evaluation_only_state_before"] = before
    manifest.controller_diagnostics["evaluation_only_state_after"] = after
    manifest.controller_diagnostics["evaluation_state_unchanged"] = True
    manifest.write(output / f"manifest_{manifest.run_id}.json")
    return {"execution_integrity_valid": manifest.completion_status == "complete",
            "metrics": metrics.to_dict(), "manifest": manifest.to_dict(),
            "manifest_path": str((output / f"manifest_{manifest.run_id}.json").resolve())}


def _metrics(value: dict) -> EpisodeMetrics:
    allowed = EpisodeMetrics.__dataclass_fields__
    return EpisodeMetrics(**{k: v for k, v in value.items() if k in allowed and k != "_trip_records"})


def _validate_fixed_identity(fixed: dict, *, seed: int,
                             input_manifest: dict) -> None:
    if fixed.get("returncode") != 0 or fixed.get("metrics", {}).get("valid") is not True:
        raise ValueError("fixed-time member is not a complete valid SUMO run")
    if (fixed.get("inputs", {}).get("network", {}).get("sha256")
            != input_manifest["network_variant"]["sha256"]
            or fixed.get("inputs", {}).get("routes", {}).get("sha256")
            != input_manifest["explicit_routes"]["sha256"]):
        raise ValueError("fixed-time member does not use the frozen network and demand")
    effective = Path(fixed["effective_sumocfg"]["path"])
    if not effective.is_file() or sha256_file(effective) != fixed["effective_sumocfg"]["sha256"]:
        raise ValueError("fixed-time effective configuration integrity failed")
    import xml.etree.ElementTree as ET
    random_seed = ET.parse(effective).getroot().find("random_number/seed")
    if random_seed is None or int(random_seed.get("value")) != seed:
        raise ValueError("fixed-time member uses a different SUMO seed")


def _validate_pair_identity(fixed: dict, controlled: list[dict], *, seed: int,
                            input_manifest: dict, begin_s: float, end_s: float) -> None:
    _validate_fixed_identity(fixed, seed=seed, input_manifest=input_manifest)
    for record in controlled:
        manifest = record["manifest"]; metrics = record["metrics"]
        if (manifest["seed"] != seed or manifest["simulation_begin_s"] != begin_s
                or manifest["simulation_end_s"] != end_s
                or manifest["step_length_s"] != 1.0
                or manifest.get("controller_diagnostics", {}).get(
                    "effective_sumo_seed") != seed
                or manifest["network_sha256"] != input_manifest["network_variant"]["sha256"]
                or sorted(manifest["route_files"].values())
                   != [input_manifest["explicit_routes"]["sha256"]]
                or metrics.get("simulation_begin_s") != begin_s
                or metrics.get("reached_horizon") is not True):
            raise ValueError("controlled member violates paired input/horizon identity")
    versions = {record["manifest"]["sumo_version"] for record in controlled}
    if len(versions) != 1 or not next(iter(versions), None):
        raise ValueError("controlled members lack a common SUMO version")
    result = subprocess.run(["sumo", "--version"], cwd=ROOT, capture_output=True,
                            text=True, check=True, timeout=15)
    fixed_version = (result.stdout or result.stderr).splitlines()[0].strip()
    if versions != {fixed_version}:
        raise ValueError("fixed-time and controlled members use different SUMO versions")


def run(config_path: str | Path, *, output_dir: str | Path,
        seeds: list[int] | None = None, smoke: bool = False,
        reuse_fixed_report: str | Path | None = None) -> Path:
    spec_path = Path(config_path).resolve()
    spec, config, input_manifest, inputs = _load_spec(spec_path)
    selected = spec["heldout_seeds"] if seeds is None else seeds
    if any(seed not in spec["heldout_seeds"] for seed in selected):
        raise ValueError("only preregistered held-out seeds may be selected")
    spec = copy.deepcopy(spec)
    spec["_spec_path"] = str(spec_path)
    spec["_input_manifest_sha256"] = sha256_file((spec_path.parent / spec["input_manifest"]).resolve())
    spec["_route_sha256"] = input_manifest["explicit_routes"]["sha256"]
    spec["_network_sha256"] = input_manifest["network_variant"]["sha256"]
    config["_evaluation_spec_path"] = str(spec_path)
    if smoke:
        smoke_cfg = Path(input_manifest["sumo_configs"]["smoke"]["path"]).resolve()
        root = __import__("xml.etree.ElementTree", fromlist=["ElementTree"]).parse(smoke_cfg).getroot()
        config["profiles"]["night"].update({"config": str(smoke_cfg),
            "begin": float(root.find("time/begin").get("value")),
            "end": float(root.find("time/end").get("value"))})
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summary = {"schema": SCHEMA, "classification": CLASSIFICATION,
        "publication_approved": False, "synthetic": True,
        "heldout_scope": "SUMO stochastic seed only; frozen demand is shared with training",
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "input_manifest": {"path": str((spec_path.parent / spec["input_manifest"]).resolve()),
                           "sha256": spec["_input_manifest_sha256"]}, "pairs": []}
    result = destination / "paired-comparison.json"
    _write_json_atomic(result, summary)
    if reuse_fixed_report is not None and len(selected) != 1:
        raise ValueError("one reused fixed-time report requires exactly one selected seed")
    for index, seed in enumerate(selected):
        pair_dir = destination / f"seed-{seed}"; pair_dir.mkdir()
        pair = {"seed": seed, "matched": False, "status": "in_progress",
                "fixed_time": None, "untrained_dqn": None,
                "trained_dqn": None, "comparisons": None}
        summary["pairs"].append(pair); _write_json_atomic(result, summary)
        if reuse_fixed_report is None:
            fixed_dir = pair_dir / "fixed_time"
            fixed_report_path = run_fixed_probe(
                manifest_path=(spec_path.parent / spec["input_manifest"]).resolve(),
                output_dir=fixed_dir, seed=seed,
                wall_clock_timeout_s=float(spec["wall_clock_timeout_per_run_s"]),
                configuration="smoke" if smoke else "full")
        else:
            fixed_report_path = Path(reuse_fixed_report).resolve()
            if not fixed_report_path.is_file():
                raise FileNotFoundError(f"reused fixed-time report is missing: {fixed_report_path}")
        fixed = json.loads(fixed_report_path.read_text(encoding="utf-8"))
        _validate_fixed_identity(fixed, seed=seed, input_manifest=input_manifest)
        pair["fixed_time"] = fixed
        pair["fixed_time_reuse"] = ({"reused": reuse_fixed_report is not None,
            "report_path": str(fixed_report_path),
            "report_sha256": sha256_file(fixed_report_path)})
        _write_json_atomic(result, summary)
        seed_all(seed + 1_000_000, deterministic_torch=True)
        untrained = MultiAgentDQN(list(config["tls_junctions"]), config["dqn"],
            observation_contract=config["state"]["observation_contract"])
        try:
            untrained_record = _controlled(label="untrained_dqn", agents=untrained,
                config=config, seed=seed, output=pair_dir / "untrained_dqn",
                port=int(spec["traci_port_start"]) + index * 2, checkpoint=None,
                wall_clock_timeout_s=float(spec["wall_clock_timeout_per_run_s"]))
            pair["untrained_dqn"] = untrained_record; _write_json_atomic(result, summary)
        except Exception as exc:
            pair["untrained_dqn"] = _failure_record(pair_dir / "untrained_dqn", exc)
            pair["status"] = "untrained_dqn_failed"; _write_json_atomic(result, summary)
            raise
        trained, checkpoint = _restore_episode3(spec, config)
        try:
            trained_record = _controlled(label="trained_dqn", agents=trained,
                config=config, seed=seed, output=pair_dir / "trained_dqn",
                port=int(spec["traci_port_start"]) + index * 2 + 1, checkpoint=checkpoint,
                wall_clock_timeout_s=float(spec["wall_clock_timeout_per_run_s"]))
            pair["trained_dqn"] = trained_record; _write_json_atomic(result, summary)
        except Exception as exc:
            pair["trained_dqn"] = _failure_record(pair_dir / "trained_dqn", exc)
            pair["status"] = "trained_dqn_failed"; _write_json_atomic(result, summary)
            raise
        _validate_pair_identity(fixed, [untrained_record, trained_record], seed=seed,
            input_manifest=input_manifest,
            begin_s=float(config["profiles"]["night"]["begin"]),
            end_s=float(config["profiles"]["night"]["end"]))
        valid = (fixed["execution_integrity_valid"]
                 and untrained_record["execution_integrity_valid"]
                 and trained_record["execution_integrity_valid"])
        pair["matched"] = valid
        if valid:
            baseline = _metrics(fixed["metrics"])
            pair["comparisons"] = {
                "untrained_dqn_vs_fixed_time": _metrics(untrained_record["metrics"]).improvement_vs(baseline),
                "trained_dqn_vs_fixed_time": _metrics(trained_record["metrics"]).improvement_vs(baseline),
                "trained_dqn_vs_untrained_dqn": _metrics(trained_record["metrics"]).improvement_vs(
                    _metrics(untrained_record["metrics"])),
            }
        pair["status"] = "complete" if valid else "invalid_unmatched"
        _write_json_atomic(result, summary)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-fixed-report")
    args = parser.parse_args()
    print(run(args.config, output_dir=args.output_dir, seeds=args.seeds,
              smoke=args.smoke, reuse_fixed_report=args.reuse_fixed_report))
