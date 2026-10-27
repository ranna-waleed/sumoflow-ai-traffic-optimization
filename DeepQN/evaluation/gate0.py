"""Phase 3.0E offline Gate-0 contracts; never starts SUMO or constructs a model."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from DeepQN.evaluation.demand_provenance import verify_split_registry
from DeepQN.evaluation.research_protocol import validate_experiment_candidate

SEED_SCHEMA = "sumoflow-publication-seed-allocation-candidate-v2"
USAGE_SCHEMA = "sumoflow-demand-usage-subpools-v1"
GATE0_SCHEMA = "sumoflow-synthetic-gate0-preflight-v1"
PARTITIONS = {"training", "calibration", "validation", "forecast_test",
              "controller_evaluation"}
PROFILES = {"morning", "midday", "evening", "night"}
SHA256 = re.compile(r"[0-9a-f]{64}")


class Gate0Error(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value.lower()) is None:
        raise Gate0Error(f"{label} must be a SHA-256 digest")
    return value.lower()


def _bound_json(path: str | Path, expected_hash: str, label: str) -> tuple[Path, dict]:
    source = Path(path).resolve()
    if not source.is_file() or sha256_file(source) != _hash(expected_hash, label):
        raise Gate0Error(f"{label} is missing or changed")
    try:
        return source, json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Gate0Error(f"{label} is invalid JSON") from exc


def validate_seed_allocation(path: str | Path, expected_hash: str, *,
                             require_frozen: bool = False) -> dict:
    """Validate identities without converting a candidate into scientific approval."""
    _, data = _bound_json(path, expected_hash, "seed allocation")
    required = {"schema", "status", "publication_approved", "scientific_arms",
                "policy_replicates", "forecast_training", "controller_evaluation",
                "checkpoint_identity_policy", "training_seed_pairing_rule",
                "proposed_training_seed_pairing_rule"}
    if set(data) != required or data.get("schema") != SEED_SCHEMA:
        raise Gate0Error("seed allocation schema is noncanonical")
    if data["publication_approved"] is not False:
        raise Gate0Error("seed allocation cannot self-approve")
    if data["scientific_arms"] != ["no_forecast_dqn", "verified_forecast_dqn"]:
        raise Gate0Error("scientific arm identities are invalid")
    policy = data["policy_replicates"]
    if not isinstance(policy, list) or len(policy) != 5:
        raise Gate0Error("exactly five policy replicate mappings are required")
    if {row.get("policy_replicate_index") for row in policy} != set(range(5)):
        raise Gate0Error("policy replicate indices must be exactly 0..4")
    seeds = [row.get("dqn_rng_seed") for row in policy]
    if seeds != list(range(11001, 11006)) or len(set(seeds)) != 5:
        raise Gate0Error("DQN RNG seed mapping is invalid")
    evaluation = data["controller_evaluation"]
    if set(evaluation) != {"crossing_requirement", "demand_replicates", "profiles"}:
        raise Gate0Error("controller-evaluation seed fields are noncanonical")
    demands = evaluation["demand_replicates"]
    if ([row.get("replicate_index") for row in demands] != list(range(10))
            or [row.get("demand_generation_seed") for row in demands]
            != list(range(21001, 21011))):
        raise Gate0Error("demand replicate/seed mapping is invalid")
    if set(evaluation["profiles"]) != PROFILES:
        raise Gate0Error("four canonical profile seed mappings are required")
    all_sumo = []
    for profile, row in evaluation["profiles"].items():
        if set(row) != {"profile_index", "sumo_execution_seeds"}:
            raise Gate0Error(f"profile seed mapping is malformed for {profile}")
        seeds = row["sumo_execution_seeds"]
        if not isinstance(seeds, list) or len(seeds) != 10 or len(set(seeds)) != 10:
            raise Gate0Error(f"profile requires ten distinct SUMO seeds: {profile}")
        all_sumo.extend(seeds)
    if len(all_sumo) != len(set(all_sumo)):
        raise Gate0Error("SUMO seeds collide across profiles")
    forecast = data["forecast_training"]
    if set(forecast) != {"replicate_indices", "rng_seeds", "status"}:
        raise Gate0Error("forecast seed allocation is malformed")
    if require_frozen:
        if (data["status"] != "frozen_candidate" or
                data["training_seed_pairing_rule"] != "matched_training_seed_v1"):
            raise Gate0Error("training-seed pairing rule is not frozen")
        if (forecast["status"] != "frozen" or forecast["replicate_indices"] != list(range(5))
                or forecast["rng_seeds"] != list(range(12001, 12006))):
            raise Gate0Error("forecast-training seeds are not allocated")
    else:
        if (data["status"] != "candidate_not_frozen"
                or data["training_seed_pairing_rule"] is not None
                or data["proposed_training_seed_pairing_rule"] != "matched_training_seed_v1"
                or forecast != {"replicate_indices": list(range(5)),
                                "rng_seeds": list(range(12001, 12006)),
                                "status": "allocated_candidate_not_frozen"}):
            raise Gate0Error("unfrozen seed candidate misrepresents readiness")
    return data


def write_usage_subpools(split_registry_path: str | Path, expected_registry_hash: str,
                         assignments: list[dict], output_path: str | Path) -> dict:
    """Bind versioned usage labels to an already verified five-pool registry."""
    registry = verify_split_registry(split_registry_path, expected_registry_hash)
    rows = registry["realizations"]
    if {row["partition"] for row in rows} != PARTITIONS:
        raise Gate0Error("usage registry requires all five scientific partitions")
    by_id = {row["realization_id"]: row for row in rows}
    if not isinstance(assignments, list) or len(assignments) != len(rows):
        raise Gate0Error("every realization requires exactly one usage assignment")
    output = []; seen = set()
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {
                "realization_id", "usage_subpool"}:
            raise Gate0Error("usage assignment fields are noncanonical")
        identity = assignment["realization_id"]
        usage = assignment["usage_subpool"]
        if identity not in by_id or identity in seen:
            raise Gate0Error("usage assignment has unknown or duplicate realization")
        if not isinstance(usage, str) or re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", usage) is None:
            raise Gate0Error("usage subpool identifier is invalid")
        if not usage.startswith(by_id[identity]["partition"] + "_"):
            raise Gate0Error("usage subpool conflicts with scientific partition")
        seen.add(identity)
        output.append({"realization_id": identity,
                       "partition": by_id[identity]["partition"],
                       "usage_subpool": usage,
                       "manifest_sha256": by_id[identity]["manifest_sha256"],
                       "physical_fingerprint_sha256": by_id[identity]["demand_fingerprint_sha256"]})
    result = {"schema": USAGE_SCHEMA, "publication_approved": False,
              "split_registry": {"path": str(Path(split_registry_path).resolve()),
                                 "sha256": expected_registry_hash.lower()},
              "assignments": sorted(output, key=lambda row: row["realization_id"])}
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite usage registry: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
        Path(temp_name).replace(target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return result


def validate_gate0_preflight(path: str | Path, expected_hash: str) -> dict:
    """Fail before runtime construction; this validator grants no execution authority."""
    manifest_path, data = _bound_json(path, expected_hash, "Gate-0 manifest")
    required = {"schema", "study_scope", "status", "phase3_authorized",
                "publication_approved", "trust_roots", "artifacts", "contracts",
                "access_policy", "signal_restrictions", "failure_policy",
                "output_requirements"}
    if set(data) != required or data.get("schema") != GATE0_SCHEMA:
        raise Gate0Error("Gate-0 manifest schema is noncanonical")
    if (data["study_scope"] != "uncalibrated_tahrir_derived_synthetic_simulation"
            or data["status"] != "candidate_not_executable" or data["phase3_authorized"] is not False
            or data["publication_approved"] is not False or data["trust_roots"] != []):
        raise Gate0Error("Gate-0 scope or authority is invalid")
    artifacts = data["artifacts"]
    required_artifacts = {"experiment_candidate", "seed_allocation", "split_registry",
                          "usage_subpools", "network", "effective_configuration",
                          "detector_mapping", "signal_review_candidate",
                          "reward_calibration_protocol"}
    if set(artifacts) != required_artifacts:
        raise Gate0Error("Gate-0 artifact inventory is incomplete")
    resolved = {}
    for name, identity in artifacts.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise Gate0Error(f"Gate-0 artifact identity is malformed: {name}")
        value = identity["sha256"]
        if value is None:
            raise Gate0Error(f"Gate-0 artifact is unresolved: {name}")
        source = (manifest_path.parent / identity["path"]).resolve()
        if not source.is_file() or sha256_file(source) != _hash(value, f"{name} hash"):
            raise Gate0Error(f"Gate-0 artifact is missing or changed: {name}")
        resolved[name] = source
    validate_experiment_candidate(json.loads(resolved["experiment_candidate"].read_text()))
    validate_seed_allocation(resolved["seed_allocation"], artifacts["seed_allocation"]["sha256"],
                             require_frozen=True)
    verify_split_registry(resolved["split_registry"], artifacts["split_registry"]["sha256"])
    if data["access_policy"] != {
            "training_only": ["training", "calibration", "validation"],
            "forecast_test": "forecast_test_only_after_model_selection",
            "controller_evaluation": "prohibited_until_held_out_evaluation"}:
        raise Gate0Error("Gate-0 data-access policy is invalid")
    if not data["signal_restrictions"] or not data["failure_policy"] or not data["output_requirements"]:
        raise Gate0Error("Gate-0 safety/failure/output contracts are incomplete")
    return data
