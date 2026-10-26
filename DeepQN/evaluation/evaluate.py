"""Provenance-aware SUMO evaluation using one XML metric pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, Optional

import numpy as np
import yaml

from DeepQN.agent.dqn_agent import CheckpointManifest, MultiAgentDQN
from DeepQN.evaluation.metrics import (
    EpisodeMetrics,
    load_metrics_from_outputs,
    load_metrics_from_paths,
)
from DeepQN.evaluation.provenance import (
    RunManifest,
    build_run_manifest,
    capture_output_hashes,
    hash_effective_sumocfg,
    validate_comparability,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _set_or_create(parent: ET.Element, tag: str, value: str) -> None:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.set("value", value)


def _patch_sumocfg(original_path: str | Path, outputs: Dict[str, Path], seed: int) -> Path:
    """Create a temporary config beside the original; never alter the source."""
    original = Path(original_path).resolve()
    tree = ET.parse(original)
    root = tree.getroot()
    output = root.find("output")
    if output is None:
        output = root.find("o")
    if output is None:
        output = ET.SubElement(root, "output")
    tags = {
        "tripinfo": "tripinfo-output",
        "emission": "emission-output",
        "summary": "summary-output",
    }
    retained_tags = set(tags.values())
    for child in list(output):
        if child.tag.endswith("-output") and child.tag not in retained_tags:
            output.remove(child)
    for key, tag in tags.items():
        _set_or_create(output, tag, str(outputs[key].resolve()))
    _set_or_create(output, "tripinfo-output.write-unfinished", "true")
    _set_or_create(output, "tripinfo-output.write-undeparted", "true")
    _set_or_create(output, "emission-output.step-scaled", "false")
    random_number = root.find("random_number")
    if random_number is None:
        random_number = ET.SubElement(root, "random_number")
    _set_or_create(random_number, "seed", str(seed))
    temporary = tempfile.NamedTemporaryFile(
        suffix=".sumocfg", mode="wb", dir=original.parent, delete=False
    )
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.close()
    return Path(temporary.name)


def _sumocfg_inputs(config_path: str | Path) -> tuple[Path, list[Path], list[Path]]:
    config = Path(config_path).resolve()
    root = ET.parse(config).getroot()
    input_section = root.find("input")
    if input_section is None:
        raise ValueError(f"SUMO config has no input section: {config}")
    net = input_section.find("net-file")
    routes = input_section.find("route-files")
    if net is None or not net.get("value"):
        raise ValueError(f"SUMO config has no net-file: {config}")
    if routes is None or not routes.get("value"):
        raise ValueError(f"SUMO config has no route-files: {config}")
    network_path = config.parent / net.get("value")
    route_paths = [config.parent / item.strip() for item in routes.get("value").split(",")]
    additional = input_section.find("additional-files")
    additional_paths = [] if additional is None or not additional.get("value") else [
        config.parent / item.strip() for item in additional.get("value").split(",") if item.strip()
    ]
    return network_path, route_paths, additional_paths


def _ensure_outputs_available(paths: Iterable[Path]) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError("refusing to overwrite existing evaluation artifacts: " + ", ".join(collisions))


def _paired_baseline_manifest_path(directory, profile, seed, run_index):
    """Return the exact non-overwriting filename emitted by baseline.py."""
    return Path(directory) / f"manifest_fixed_time_{profile}_seed{seed}_run{run_index}.json"


def _fixed_horizon_config(cfg: dict, profile: str, patched_path: Path,
                          *, require_forecast_protocol: bool = False) -> dict:
    """Return an isolated evaluation config that cannot stop on empty demand."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("simulation", {})["terminate_on_empty"] = False
    run_cfg["simulation"]["research_evaluation"] = True
    run_cfg["simulation"]["comparative_research"] = require_forecast_protocol
    run_cfg["simulation"]["require_matched_forecast_warmup"] = require_forecast_protocol
    run_cfg["profiles"][profile]["config"] = str(patched_path)
    return run_cfg


def run_controlled_episode(
    *,
    cfg: dict,
    profile: str,
    agents: Optional[MultiAgentDQN],
    controller: str,
    output_dir: Path,
    run_index: int,
    seed: int,
    checkpoint_manifest: Optional[CheckpointManifest],
    model_mode: str,
    config_path: str | Path,
    port: int,
    environment_mutator=None,
    action_provider=None,
) -> tuple[EpisodeMetrics, RunManifest]:
    """Run one explicit learned, random, or validated rule-based controller."""
    if controller not in {"dqn", "random", "max_pressure"}:
        raise ValueError("controller must be 'dqn', 'random', or 'max_pressure'")
    if controller == "max_pressure" and action_provider is None:
        raise ValueError("max_pressure requires an explicit validated action provider")
    if controller == "max_pressure" and (checkpoint_manifest is not None or model_mode != "not_applicable"):
        raise ValueError("max_pressure must not use a DQN checkpoint or forecast model")
    if controller in {"dqn", "random"} and agents is None:
        raise ValueError(f"{controller} requires an explicit agent ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{controller}_{profile}_seed{seed}_run{run_index}"
    outputs = {
        kind: (output_dir / f"{kind}_{run_id}.xml").resolve()
        for kind in ("tripinfo", "emission", "summary")
    }
    original_sumocfg = Path(cfg["profiles"][profile]["config"])
    network_path, route_paths, additional_paths = _sumocfg_inputs(original_sumocfg)
    manifest_path = output_dir / f"manifest_{run_id}.json"
    _ensure_outputs_available([*outputs.values(), manifest_path])
    manifest = build_run_manifest(
        project_root=PROJECT_ROOT,
        run_id=run_id,
        controller=controller,
        scenario=profile,
        seed=seed,
        simulation_begin_s=float(cfg["profiles"][profile]["begin"]),
        simulation_end_s=float(cfg["profiles"][profile]["end"]),
        step_length_s=float(cfg.get("simulation", {}).get("step_length", 1.0)),
        config_path=config_path,
        simulation_config_path=original_sumocfg,
        network_path=network_path,
        route_paths=route_paths,
        additional_paths=additional_paths,
        output_paths={key: str(value) for key, value in outputs.items()},
        model_mode=model_mode,
        checkpoint_manifest=(checkpoint_manifest.to_dict() if checkpoint_manifest else None),
        emission_step_scaled=False,
        sumo_binary=cfg.get("simulation", {}).get("sumo_binary", "sumo"),
    )
    patched_path: Optional[Path] = None
    env = None
    try:
        from DeepQN.env.sumo_env import SumoEnv

        patched_path = _patch_sumocfg(original_sumocfg, outputs, seed)
        manifest.effective_simulation_config_sha256 = hash_effective_sumocfg(patched_path)
        run_cfg = _fixed_horizon_config(
            cfg, profile, patched_path,
            require_forecast_protocol=(controller in {"dqn", "max_pressure"})
        )
        env = SumoEnv(run_cfg, profile=profile, port=port)
        if environment_mutator is not None:
            environment_mutator(env)
        obs = env.reset()
        warmup_diagnostics = env._warmup_diagnostics
        warmup_trace = env._warmup_trace
        detector_preflight = env._detector_preflight
        waiting_time_memory_s = env._waiting_time_memory_s
        observation_contract = env.observation_contract
        comparative_research = env.comparative_research
        done = False
        rng = np.random.default_rng(seed)
        decision_count = 0
        requested_switches = 0
        requested_action_trace = []
        physical_action_trace = []
        while not done:
            if controller == "random":
                actions = {tls_id: int(rng.integers(0, 2)) for tls_id in agents.agents}
            elif controller == "max_pressure":
                actions = action_provider.actions(env, obs)
            else:
                actions = agents.act(obs, eval_mode=True)
            decision_count += 1
            requested_switches += sum(int(action == 1) for action in actions.values())
            requested_action_trace.append({tls_id: int(action) for tls_id, action in sorted(actions.items())})
            obs, _, done, step_info = env.step(actions)
            physical_action_trace.append({
                "decisions": step_info.get("action_decisions", {}),
                "physical_steps": step_info.get("action_trace", {}),
                "interval_duration_s": step_info.get("interval_duration_s"),
            })
        env.close()
        env = None
        metrics = load_metrics_from_outputs(
            output_dir,
            profile,
            controller,
            run_id,
            step_length_s=manifest.step_length_s,
            expected_begin_s=manifest.simulation_begin_s,
            expected_end_s=manifest.simulation_end_s,
            emission_step_scaled=bool(manifest.emission_step_scaled),
            route_paths=route_paths,
            require_demand_contract=True,
        )
        trace_path = Path(output_dir) / f"action_trace_{run_id}.json"
        if trace_path.exists():
            raise FileExistsError(f"refusing to overwrite action trace: {trace_path}")
        trace_payload = {
            "initial_control_warmup": warmup_trace,
            "requested": requested_action_trace,
            "physical": physical_action_trace,
        }
        trace_raw = (json.dumps(trace_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        trace_path.write_bytes(trace_raw)
        manifest.controller_diagnostics = {
            "decision_count": decision_count,
            "requested_switches": requested_switches,
            "controlled_tls_ids": (list(action_provider.tls_ids) if action_provider is not None
                                   else sorted(agents.agents)),
            "requested_action_trace_sha256": hashlib.sha256(json.dumps(
                requested_action_trace, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "physical_action_trace_path": str(trace_path.resolve()),
            "physical_action_trace_sha256": hashlib.sha256(trace_raw).hexdigest(),
            "initial_control_warmup": warmup_diagnostics,
            "detector_runtime_preflight": detector_preflight,
            "waiting_time_memory_s": waiting_time_memory_s,
            "observation_contract": observation_contract,
            "comparative_research": comparative_research,
        }
        if action_provider is not None:
            manifest.controller_diagnostics["rule_based_controller"] = action_provider.diagnostics()
        capture_output_hashes(manifest)
        manifest.completion_status = "complete" if metrics.valid and metrics.reached_horizon else "incomplete"
        if not metrics.valid:
            manifest.errors.extend(metrics.errors)
        elif not metrics.reached_horizon:
            manifest.errors.append("SUMO terminated before the configured horizon")
    except Exception as exc:
        manifest.completion_status = "failed"
        manifest.errors.append(str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        if patched_path is not None:
            patched_path.unlink(missing_ok=True)
        manifest.write(manifest_path)
    return metrics, manifest


def _aggregate(runs: Iterable[EpisodeMetrics]) -> dict:
    records = list(runs)
    fields = (
        "tripinfo_records", "successful_arrivals", "removed_or_vaporized",
        "teleport_or_vaporization_tagged_arrivals", "unfinished_tripinfo_records",
        "scheduled_demand", "inserted_vehicles", "undeparted_vehicles",
        "demand_normalized_waiting_s", "censored_preinsertion_delay_per_due_vehicle_s",
        "mean_departure_delay_inserted_s", "insertion_rate", "completion_rate",
        "conditional_completion_rate", "arrival_throughput_veh_per_hour",
        "avg_waiting_time_s", "avg_travel_time_s", "avg_time_loss_s",
        "total_co2_mg", "network_co2_per_successful_arrival_mg", "mean_successful_trip_co2_mg", "throughput",
        "unfinished_vehicles", "teleports", "collisions", "avg_speed_ms",
    )
    result: dict = {"n_runs": len(records), "all_valid": all(run.valid for run in records)}
    for field_name in fields:
        values = [getattr(run, field_name) for run in records]
        if not values or any(value is None for value in values):
            result[field_name] = {"mean": None, "stddev": None}
        else:
            numeric = [float(value) for value in values]
            result[field_name] = {
                "mean": mean(numeric),
                "stddev": stdev(numeric) if len(numeric) > 1 else 0.0,
            }
    return result


def evaluate(
    *,
    config_path: str,
    checkpoint_dir: str,
    checkpoint_episode: Optional[int],
    controller: str,
    profiles: Optional[list[str]],
    runs: int,
    seed: int,
    output_dir: str,
    baseline_manifest_dir: Optional[str],
    model_mode: str,
    port: int,
) -> dict:
    with Path(config_path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    selected_profiles = profiles or list(cfg["profiles"])
    unknown = sorted(set(selected_profiles) - set(cfg["profiles"]))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    agents = MultiAgentDQN(list(cfg["tls_junctions"]), cfg["dqn"])
    checkpoint_manifest = None
    if controller == "dqn":
        if checkpoint_episode is None:
            raise ValueError("trained DQN evaluation requires --checkpoint-episode")
        checkpoint_manifest = agents.load_for_research_evaluation(
            checkpoint_dir, checkpoint_episode
        )
    elif checkpoint_episode is not None:
        raise ValueError("--checkpoint-episode is not valid for random evaluation")

    destination = Path(output_dir)
    report_path = destination / f"evaluation_{controller}_report.json"
    _ensure_outputs_available([report_path])
    report: dict = {
        "controller": controller,
        "checkpoint_manifest": checkpoint_manifest.to_dict() if checkpoint_manifest else None,
        "profiles": {},
    }
    for profile in selected_profiles:
        run_metrics: list[EpisodeMetrics] = []
        run_manifests: list[RunManifest] = []
        comparisons: list[dict] = []
        for run_index in range(runs):
            run_seed = seed + run_index
            metrics, manifest = run_controlled_episode(
                cfg=cfg,
                profile=profile,
                agents=agents,
                controller=controller,
                output_dir=destination,
                run_index=run_index,
                seed=run_seed,
                checkpoint_manifest=checkpoint_manifest,
                model_mode=model_mode,
                config_path=config_path,
                port=port,
            )
            run_metrics.append(metrics)
            run_manifests.append(manifest)
            if baseline_manifest_dir:
                baseline_path = _paired_baseline_manifest_path(
                    baseline_manifest_dir, profile, run_seed, run_index
                )
                baseline_manifest = RunManifest.read(baseline_path)
                validate_comparability(baseline_manifest, manifest)
                baseline_metrics = load_metrics_from_paths(
                    tripinfo_path=baseline_manifest.output_paths["tripinfo"],
                    emission_path=baseline_manifest.output_paths["emission"],
                    summary_path=baseline_manifest.output_paths["summary"],
                    profile=profile,
                    mode="fixed_time",
                    step_length_s=baseline_manifest.step_length_s,
                    expected_begin_s=baseline_manifest.simulation_begin_s,
                    expected_end_s=baseline_manifest.simulation_end_s,
                    emission_step_scaled=bool(baseline_manifest.emission_step_scaled),
                    route_paths=list(baseline_manifest.route_files),
                    require_demand_contract=True,
                )
                comparisons.append(metrics.improvement_vs(baseline_metrics))
        report["profiles"][profile] = {
            "runs": [item.to_dict() for item in run_metrics],
            "manifests": [item.to_dict() for item in run_manifests],
            "aggregate": _aggregate(run_metrics),
            "paired_improvements": comparisons,
        }
    destination.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="DeepQN/configs/dqn_config.yaml")
    parser.add_argument("--controller", choices=("dqn", "random"), required=True)
    parser.add_argument("--checkpoint-dir", default="DeepQN/checkpoints")
    parser.add_argument("--checkpoint-episode", type=int)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="DeepQN/results/phase1a")
    parser.add_argument("--baseline-manifest-dir")
    parser.add_argument("--model-mode", default="as_configured")
    parser.add_argument("--port", type=int, default=8813)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_episode=args.checkpoint_episode,
        controller=args.controller,
        profiles=args.profiles,
        runs=args.runs,
        seed=args.seed,
        output_dir=args.output_dir,
        baseline_manifest_dir=args.baseline_manifest_dir,
        model_mode=args.model_mode,
        port=args.port,
    )
