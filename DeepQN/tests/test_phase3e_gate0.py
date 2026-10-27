from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import DeepQN.tests.test_phase2c2b_demand_provenance as demand_fixture
from DeepQN.evaluation.demand_provenance import sha256_file, write_split_registry
from DeepQN.evaluation.gate0 import (
    GATE0_SCHEMA, Gate0Error, validate_gate0_preflight,
    validate_seed_allocation, write_usage_subpools,
)

ROOT = Path(__file__).resolve().parents[2]
SEED_CANDIDATE = ROOT / "research/2026-10-27-phase3e-seed-allocation-candidate.json"


class Gate0ContractsTests(unittest.TestCase):
    def test_seed_identities_are_separate_and_candidate_cannot_be_frozen_implicitly(self):
        payload = validate_seed_allocation(SEED_CANDIDATE, sha256_file(SEED_CANDIDATE))
        self.assertEqual([row["policy_replicate_index"] for row in payload["policy_replicates"]],
                         list(range(5)))
        self.assertEqual([row["dqn_rng_seed"] for row in payload["policy_replicates"]],
                         list(range(11001, 11006)))
        self.assertEqual(payload["forecast_training"]["rng_seeds"],
                         list(range(12001, 12006)))
        self.assertIsNone(payload["training_seed_pairing_rule"])
        with self.assertRaisesRegex(Gate0Error, "pairing rule is not frozen"):
            validate_seed_allocation(SEED_CANDIDATE, sha256_file(SEED_CANDIDATE),
                                     require_frozen=True)

    def test_seed_tampering_collision_and_self_approval_fail(self):
        original = json.loads(SEED_CANDIDATE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            for mutation, message in (
                (lambda item: item.update(publication_approved=True), "self-approve"),
                (lambda item: item["policy_replicates"][1].update(dqn_rng_seed=11001),
                 "DQN RNG seed"),
                (lambda item: item["controller_evaluation"]["profiles"]["night"]
                 ["sumo_execution_seeds"].__setitem__(0, 31001), "collide"),
            ):
                payload = copy.deepcopy(original); mutation(payload)
                path = Path(directory) / f"candidate-{len(list(Path(directory).iterdir()))}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(Gate0Error, message):
                    validate_seed_allocation(path, sha256_file(path))

    def test_bounded_five_pool_generation_registry_and_usage_pipeline(self):
        fixture = demand_fixture.DemandProvenanceTests("test_byte_identical_regeneration_and_distinct_seed")
        fixture.setUp()
        try:
            partitions = ("training", "calibration", "validation", "forecast_test",
                          "controller_evaluation")
            assignments = []
            for index, partition in enumerate(partitions):
                identity = f"gate0-{partition}"
                fixture._generate(identity, 101 + index)
                assignments.append({"manifest_path":
                    fixture.root / identity / "realization_manifest.json",
                    "partition": partition})
            registry_path = fixture.root / "split-registry.json"
            write_split_registry(assignments, registry_path)
            usage_path = fixture.root / "usage.json"
            usage = write_usage_subpools(
                registry_path, sha256_file(registry_path),
                [{"realization_id": f"gate0-{partition}",
                  "usage_subpool": f"{partition}_candidate_v1"}
                 for partition in partitions], usage_path)
            self.assertEqual({row["partition"] for row in usage["assignments"]}, set(partitions))
            self.assertFalse(usage["publication_approved"])
            with self.assertRaises(FileExistsError):
                write_usage_subpools(registry_path, sha256_file(registry_path),
                    [{"realization_id": f"gate0-{partition}",
                      "usage_subpool": f"{partition}_candidate_v1"}
                     for partition in partitions], usage_path)
        finally:
            fixture.tearDown()

    def test_preflight_rejects_unresolved_artifact_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate0.json"
            artifacts = {name: {"path": "missing", "sha256": None} for name in (
                "experiment_candidate", "seed_allocation", "split_registry",
                "usage_subpools", "network", "effective_configuration",
                "detector_mapping", "signal_review_candidate",
                "reward_calibration_protocol")}
            payload = {"schema": GATE0_SCHEMA,
                "study_scope": "uncalibrated_tahrir_derived_synthetic_simulation",
                "status": "candidate_not_executable", "phase3_authorized": False,
                "publication_approved": False, "trust_roots": [], "artifacts": artifacts,
                "contracts": {}, "access_policy": {}, "signal_restrictions": ["unresolved"],
                "failure_policy": ["fail_closed"], "output_requirements": ["manifest"]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Gate0Error, "artifact is unresolved"):
                validate_gate0_preflight(path, sha256_file(path))


if __name__ == "__main__": unittest.main()
