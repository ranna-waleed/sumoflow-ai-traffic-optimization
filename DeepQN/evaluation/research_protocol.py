"""Fail-closed Phase 3.0C calibration and experiment-analysis contracts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from DeepQN.evaluation.demand_expansion import _physical_fingerprint

CALIBRATION_SCHEMA_V3 = "sumoflow-reward-calibration-v3"
CALIBRATION_TRACE_SCHEMA = "sumoflow-interval-reward-trace-v1"
SCIENTIFIC_SPLIT_SCHEMA = "sumoflow-five-pool-split-registry-v2"
CALIBRATION_PROTOCOL_SCHEMA = "sumoflow-reward-calibration-protocol-v1"
EXPERIMENT_SCHEMA = "sumoflow-publication-experiment-candidate-v1"
READINESS_SCHEMA = "sumoflow-phase3c-readiness-v1"
POOLS = {"training", "calibration", "validation", "forecast_test", "controller_evaluation"}
ELIGIBLE_CALIBRATION_POOLS = {"training", "calibration"}
METRICS = {
    "delay_vehicle_s": "vehicle-seconds",
    "co2_mg": "mg",
    "arrivals_vehicles": "vehicles",
    "pressure_vehicle_s": "vehicle-seconds",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value.lower()) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _load_bound_json(path: str | Path, expected_sha256: str, label: str) -> tuple[Path, dict]:
    source = Path(path).resolve()
    if not source.is_file() or sha256_file(source) != _sha(expected_sha256, label):
        raise ValueError(f"{label} is missing or changed")
    try:
        return source, json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc


def calibrate_publication_reward_references(
    registry_path: str | Path, expected_registry_sha256: str, output_path: str | Path,
    *, protocol_path: str | Path, expected_protocol_sha256: str,
) -> dict:
    """Freeze per-profile scales from hash-bound training/calibration traces only."""
    protocol_file, protocol = _load_bound_json(
        protocol_path, expected_protocol_sha256, "calibration protocol"
    )
    expected_protocol = {
        "schema": CALIBRATION_PROTOCOL_SCHEMA, "estimator": "positive_quantile",
        "quantile": 0.95, "numpy_method": "linear",
        "pooling_rule": "separate_reference_per_profile",
        "zero_policy": "retain_in_trace_exclude_from_scale",
        "eligible_partitions": ["training", "calibration"],
        "required_profiles": ["morning", "midday", "evening", "night"],
        "trace_generation_protocol": "frozen_non_evaluation_controller_diagnostic_v1",
        "publication_approved": False,
    }
    if protocol != expected_protocol:
        raise ValueError("calibration protocol differs from the frozen candidate")
    quantile = protocol["quantile"]
    registry_file, registry = _load_bound_json(
        registry_path, expected_registry_sha256, "split registry"
    )
    if set(registry) != {"schema", "realizations"} or registry["schema"] != SCIENTIFIC_SPLIT_SCHEMA:
        raise ValueError("unsupported or noncanonical split registry")
    rows = registry["realizations"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("split registry has no realizations")
    ids = set(); physical = {}; demand_bytes = {}
    samples: dict[str, dict[str, list[float]]] = {}
    source_rows = []
    for row in rows:
        required = {
            "realization_id", "profile", "partition", "template_sha256",
            "explicit_demand_path", "explicit_demand_sha256",
            "physical_fingerprint_sha256", "demand_generator_seed", "sumo_seed",
            "reward_trace_path", "reward_trace_sha256",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("split-registry realization fields are noncanonical")
        identity = row["realization_id"]
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("duplicate or invalid realization identity")
        ids.add(identity)
        if row["partition"] not in POOLS:
            raise ValueError(f"unknown scientific partition for {identity}")
        for seed_name in ("demand_generator_seed", "sumo_seed"):
            if type(row[seed_name]) is not int or row[seed_name] < 0:
                raise ValueError(f"{seed_name} must be a nonnegative integer")
        template_hash = _sha(row["template_sha256"], "template hash")
        demand_hash = _sha(row["explicit_demand_sha256"], "explicit demand hash")
        fingerprint = _sha(row["physical_fingerprint_sha256"], "physical fingerprint")
        demand_path = (registry_file.parent / row["explicit_demand_path"]).resolve()
        if not demand_path.is_file() or sha256_file(demand_path) != demand_hash:
            raise ValueError(f"explicit demand artifact is missing or changed for {identity}")
        try:
            root = ET.parse(demand_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"explicit demand is malformed for {identity}") from exc
        if root.findall("flow") or root.findall("trip") or root.findall("person"):
            raise ValueError(f"explicit demand is not explicit-only for {identity}")
        derived_fingerprint = _physical_fingerprint(root.findall("vehicle"))
        if derived_fingerprint != fingerprint:
            raise ValueError(f"declared physical fingerprint is untrusted for {identity}")
        trace_value = row["reward_trace_path"]
        trace_hash_value = row["reward_trace_sha256"]
        if fingerprint in physical:
            raise ValueError(
                f"physical demand reused by {physical[fingerprint]!r} and {identity!r}"
            )
        if demand_hash in demand_bytes:
            raise ValueError(
                f"explicit demand bytes reused by {demand_bytes[demand_hash]!r} and {identity!r}"
            )
        physical[fingerprint] = identity; demand_bytes[demand_hash] = identity
        if row["partition"] not in ELIGIBLE_CALIBRATION_POOLS:
            if trace_value is not None or trace_hash_value is not None:
                raise ValueError(f"held-out reward trace is forbidden for {identity}")
            continue
        if not isinstance(trace_value, str) or not trace_value:
            raise ValueError(f"eligible reward trace is missing for {identity}")
        trace_path = (registry_file.parent / trace_value).resolve()
        trace_hash = _sha(trace_hash_value, "reward trace hash")
        if not trace_path.is_file() or sha256_file(trace_path) != trace_hash:
            raise ValueError(f"eligible reward trace is missing or changed for {identity}")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        expected_trace = {
            "schema", "realization_id", "profile", "partition", "template_sha256",
            "explicit_demand_sha256", "physical_fingerprint_sha256",
            "demand_generator_seed", "sumo_seed", "intervals",
        }
        if set(trace) != expected_trace or trace["schema"] != CALIBRATION_TRACE_SCHEMA:
            raise ValueError(f"reward trace schema is invalid for {identity}")
        for key in expected_trace - {"schema", "intervals"}:
            if trace[key] != row[key if key != "explicit_demand_sha256" else key]:
                raise ValueError(f"reward trace provenance conflicts for {identity}: {key}")
        intervals = trace["intervals"]
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"eligible reward trace is empty for {identity}")
        profile_values = samples.setdefault(row["profile"], {name: [] for name in METRICS})
        for interval in intervals:
            if not isinstance(interval, dict) or set(interval) != set(METRICS) | {"duration_s"}:
                raise ValueError(f"reward interval fields are invalid for {identity}")
            duration = float(interval["duration_s"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("reward interval duration must be finite and positive")
            for metric in METRICS:
                value = float(interval[metric])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid calibration value for {metric}")
                profile_values[metric].append(value)
        source_rows.append({
            "realization_id": identity, "profile": row["profile"],
            "partition": row["partition"], "template_sha256": template_hash,
            "explicit_demand_path": str(demand_path), "explicit_demand_sha256": demand_hash,
            "physical_fingerprint_sha256": fingerprint,
            "demand_generator_seed": row["demand_generator_seed"], "sumo_seed": row["sumo_seed"],
            "reward_trace_path": str(trace_path), "reward_trace_sha256": trace_hash,
        })
    if set(samples) != set(protocol["required_profiles"]):
        missing = sorted(set(protocol["required_profiles"]) - set(samples))
        raise ValueError(f"calibration profile coverage is incomplete: {missing}")
    if not samples:
        raise ValueError("registry has no eligible training/calibration reward traces")
    references = {}
    for profile, profile_values in sorted(samples.items()):
        references[profile] = {}
        for metric, values in profile_values.items():
            positive = [value for value in values if value > 0]
            if not positive:
                raise ValueError(f"no positive {metric} samples for profile {profile}")
            references[profile][metric] = {
                "value": float(np.quantile(positive, quantile, method="linear")),
                "unit": METRICS[metric], "positive_samples": len(positive),
            }
    manifest = {
        "schema": CALIBRATION_SCHEMA_V3,
        "publication_approved": False,
        "protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file),
                     **protocol},
        "split_registry": {"path": str(registry_file),
                           "sha256": sha256_file(registry_file),
                           "schema": SCIENTIFIC_SPLIT_SCHEMA},
        "references_by_profile": references,
        "sources": source_rows,
    }
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
        Path(temp_name).replace(target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return manifest


def validate_experiment_candidate(manifest: dict, *, require_execution_ready: bool = False) -> None:
    required = {
        "schema", "publication_approved", "phase3_authorized", "contracts", "network",
        "effective_configuration", "controlled_tls", "demand", "warmup",
        "simulation_step_s", "decision_interval_s", "policy",
        "training", "evaluation", "arms", "profiles", "failure_policy", "artifacts",
    }
    if set(manifest) != required or manifest.get("schema") != EXPERIMENT_SCHEMA:
        raise ValueError("experiment manifest schema is noncanonical")
    if manifest["publication_approved"] is not False or manifest["phase3_authorized"] is not False:
        raise ValueError("candidate cannot self-authorize")
    config = manifest["effective_configuration"]
    if set(config) != {"path", "sha256", "status"}:
        raise ValueError("effective configuration identity is noncanonical")
    _sha(config["sha256"], "effective configuration hash")
    if config["status"] not in {"provisional_not_effective_for_publication", "frozen_publication"}:
        raise ValueError("effective configuration status is unknown")
    if manifest["simulation_step_s"] != 1:
        raise ValueError("publication experiment requires one-second simulation steps")
    if len(manifest["controlled_tls"]) != 7 or len(set(manifest["controlled_tls"])) != 7:
        raise ValueError("experiment requires exactly seven distinct controlled TLS")
    if manifest["warmup"] != {"duration_s": 60, "protocol": "phase_controller_keep_v1",
                               "included_in_outcomes": True}:
        raise ValueError("matched warm-up contract is invalid")
    if manifest["training"].get("seeds_per_arm") != 5:
        raise ValueError("publication candidate requires five training seeds per learned arm")
    if manifest["evaluation"].get("paired_demands_per_profile") != 10:
        raise ValueError("publication candidate requires ten paired demands per profile")
    core_arms = {"fixed_time_secondary", "no_forecast_dqn", "verified_forecast_dqn"}
    if not core_arms.issubset(manifest["arms"]):
        raise ValueError("experiment arms are incomplete")
    if set(manifest["profiles"]) != {"morning", "midday", "evening", "night"}:
        raise ValueError("four profiles are required")
    if manifest["contracts"] != {
        "observation": "sumoflow-dqn-observation-v3-lane-occupancy-fraction",
        "forecast_inference": "junction-bilstm-60s-to-30s-v2-nonnegative-counts",
        "forecast_dataset": "causal-junction-arrivals-60s-to-30s-v4-right-closed",
        "reward": "interval-global-reward-v1",
        "metrics": "sumo-xml-demand-normalized-horizon-wait-v5",
    }:
        raise ValueError("experiment contract versions are incompatible")
    if require_execution_ready:
        if not manifest["publication_approved"] or not manifest["phase3_authorized"]:
            raise ValueError("experiment candidate is not externally authorized")
        for name, identity in manifest["artifacts"].items():
            if not isinstance(identity, dict) or identity.get("sha256") is None:
                raise ValueError(f"required experiment artifact is unresolved: {name}")
        if config["status"] != "frozen_publication":
            raise ValueError("effective publication configuration is not frozen")


def validate_readiness_report(report: dict) -> None:
    """Validate structure only; local evidence can never confer external approval."""
    required = {
        "schema", "publication_approved", "phase3_authorized", "trust_roots",
        "engineering_prerequisites", "scientific_blockers", "artifacts",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("readiness report schema is noncanonical")
    if report.get("schema") != READINESS_SCHEMA:
        raise ValueError("unsupported readiness report schema")
    if report["publication_approved"] is not False or report["phase3_authorized"] is not False:
        raise ValueError("readiness report cannot self-authorize")
    if report["trust_roots"] != []:
        raise ValueError("publication trust roots must remain empty until independently supplied")
    if not isinstance(report["scientific_blockers"], list) or not report["scientific_blockers"]:
        raise ValueError("candidate readiness report must retain explicit blockers")
    for name, identity in report["artifacts"].items():
        if not isinstance(name, str) or not isinstance(identity, dict):
            raise ValueError("readiness artifact identity is malformed")
        digest = identity.get("sha256")
        if digest is not None:
            _sha(digest, f"{name} hash")


def paired_primary_estimand(rows: Iterable[dict]) -> dict:
    """Known-answer paired estimator; inferential hierarchy is specified externally."""
    cells = {}
    for row in rows:
        required = {"training_seed", "demand_id", "sumo_seed", "profile", "arm", "waiting_s", "valid"}
        if set(row) != required or row["arm"] not in {"no_forecast_dqn", "verified_forecast_dqn"}:
            raise ValueError("analysis row is malformed")
        if row["valid"] is not True:
            raise ValueError("invalid or failed runs require the prespecified failure analysis")
        value = float(row["waiting_s"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("primary outcome must be finite and nonnegative")
        key = (row["training_seed"], row["demand_id"], row["sumo_seed"], row["profile"])
        cell = cells.setdefault(key, {})
        if row["arm"] in cell:
            raise ValueError("duplicate arm within paired analysis cell")
        cell[row["arm"]] = value
    if not cells or any(set(cell) != {"no_forecast_dqn", "verified_forecast_dqn"}
                        for cell in cells.values()):
        raise ValueError("every analysis cell requires both learned arms")
    differences = {key: cell["verified_forecast_dqn"] - cell["no_forecast_dqn"]
                   for key, cell in cells.items()}
    profiles = sorted({key[3] for key in differences})
    profile_means = {
        profile: float(np.mean([value for key, value in differences.items()
                                if key[3] == profile])) for profile in profiles
    }
    return {
        "estimand": "verified_forecast_minus_no_forecast_demand_normalized_waiting_s",
        "overall_mean_difference_s": float(np.mean(list(profile_means.values()))),
        "per_profile_mean_difference_s": profile_means,
        "paired_cells": len(differences),
        "independence_warning": "cells are not independent across shared training policies; use hierarchical inference",
    }


def validate_publication_pairing(rows: Iterable[dict], manifest: dict) -> None:
    """Require complete frozen coverage; this does not perform hierarchical inference."""
    validate_experiment_candidate(manifest)
    if manifest["training"].get("pairing_rule") != "matched_training_seed_v1":
        raise ValueError("publication training-seed pairing rule is not frozen")
    profiles = set(manifest["profiles"])
    arms = {"no_forecast_dqn", "verified_forecast_dqn"}
    cells = {}
    policy_identity = {}
    physical_by_demand = {}
    for row in rows:
        required = {"profile", "training_seed", "policy_id", "checkpoint_sha256",
                    "demand_id", "physical_fingerprint_sha256", "sumo_seed", "arm", "valid"}
        if not isinstance(row, dict) or set(row) != required or row["arm"] not in arms:
            raise ValueError("publication pairing row is malformed")
        if row["valid"] is not True:
            raise ValueError("publication pairing contains a failed or omitted run")
        if row["profile"] not in profiles or type(row["training_seed"]) is not int:
            raise ValueError("publication pairing profile or training seed is invalid")
        _sha(row["checkpoint_sha256"], "checkpoint hash")
        fingerprint = _sha(row["physical_fingerprint_sha256"], "physical fingerprint")
        demand_key = (row["profile"], row["demand_id"])
        previous = physical_by_demand.setdefault(demand_key, fingerprint)
        if previous != fingerprint:
            raise ValueError("paired demand identity is contradictory")
        policy_key = (row["arm"], row["training_seed"])
        identity = (row["policy_id"], row["checkpoint_sha256"])
        if policy_identity.setdefault(policy_key, identity) != identity:
            raise ValueError("policy/checkpoint provenance is inconsistent")
        key = (row["profile"], row["training_seed"], row["demand_id"], row["sumo_seed"])
        cell = cells.setdefault(key, {})
        if row["arm"] in cell:
            raise ValueError("duplicate publication arm cell")
        cell[row["arm"]] = fingerprint
    expected_seeds = set(range(5))
    for arm in arms:
        if {seed for candidate, seed in policy_identity if candidate == arm} != expected_seeds:
            raise ValueError("publication pairing lacks five policy identities per arm")
    for profile in profiles:
        demands = {key[2] for key in cells if key[0] == profile}
        if len(demands) != 10:
            raise ValueError("publication pairing lacks ten distinct demands per profile")
        fingerprints = {physical_by_demand[(profile, demand)] for demand in demands}
        if len(fingerprints) != 10:
            raise ValueError("publication pairing reuses a physical demand under another ID")
        for seed in expected_seeds:
            selected = [cell for key, cell in cells.items()
                        if key[0] == profile and key[1] == seed]
            if len(selected) != 10 or any(set(cell) != arms or len(set(cell.values())) != 1
                                          for cell in selected):
                raise ValueError("publication pairing is missing, unbalanced, or physically unmatched")
