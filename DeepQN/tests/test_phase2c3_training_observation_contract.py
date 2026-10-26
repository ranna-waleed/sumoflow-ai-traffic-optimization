from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from DeepQN.agent.dqn_agent import MultiAgentDQN, RESEARCH_OBSERVATION_CONTRACT
from DeepQN.training.research_resume import (
    ResearchResumeError,
    ResumeContracts,
    TrainingCursor,
    observation_config_sha256,
    publish_training_snapshot,
    validate_publication_observation_contract,
)
from DeepQN.training.train import train_research


def _config() -> dict:
    return {
        "simulation": {"step_length": 1, "waiting_time_memory_s": 100,
                       "lane_observation_mode": "research_fail_closed"},
        "profiles": {"p": {"begin": 100, "end": 200, "config": "p.sumocfg"}},
        "tls_junctions": {"tls": {"incoming_lanes": ["lane"],
                                      "green_phases": [0]}},
        "state": {"observation_contract": RESEARCH_OBSERVATION_CONTRACT,
                  "state_dim": 37, "max_lanes": 10, "features_per_lane": 3,
                  "junction_features": 2, "global_features": 5,
                  "max_queue": 50.0, "max_wait": 300.0,
                  "max_occupancy": 1.0, "max_phase_elapsed": 120.0},
        "bilstm": {"mode": "no_forecast", "history_warmup_seconds": 60,
                   "initial_control_protocol": "phase_controller_keep_v1"},
        "dqn": {"state_dim": 37},
    }


def _contracts(config: dict, **changes) -> ResumeContracts:
    values = dict(
        run_id="synthetic-publication-contract-test",
        run_manifest_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        demand_registry_sha256="3" * 64,
        split_registry_sha256="4" * 64,
        observation_contract=RESEARCH_OBSERVATION_CONTRACT,
        reward_contract="interval-reward-v1",
        forecast_contract="no-forecast-v1",
        git_commit="test-commit",
        environment_sha256="5" * 64,
        observation_config_sha256=observation_config_sha256(config),
    )
    values.update(changes)
    return ResumeContracts(**values)


class PublicationObservationContractTests(unittest.TestCase):
    def test_matching_v3_configuration_is_bound(self):
        config = _config()
        self.assertEqual(
            validate_publication_observation_contract(config, _contracts(config)),
            observation_config_sha256(config),
        )

    def test_both_intended_forecast_arms_share_the_v3_gate(self):
        for mode in ("no_forecast", "verified_model"):
            with self.subTest(mode=mode):
                config = _config()
                config["bilstm"]["mode"] = mode
                self.assertEqual(
                    validate_publication_observation_contract(config, _contracts(config)),
                    observation_config_sha256(config),
                )

    def test_v3_configuration_rejects_v2_resume_contract(self):
        config = _config()
        with self.assertRaisesRegex(ResearchResumeError, "contracts differ"):
            validate_publication_observation_contract(
                config, _contracts(config, observation_contract="sumoflow-dqn-observation-v2")
            )

    def test_v2_configuration_cannot_claim_v3_resume_contract(self):
        config = _config()
        config["state"]["observation_contract"] = "sumoflow-dqn-observation-v2"
        with self.assertRaisesRegex(ResearchResumeError, "requires"):
            validate_publication_observation_contract(config, _contracts(_config()))

    def test_modified_observation_configuration_is_rejected(self):
        original = _config()
        contracts = _contracts(original)
        agent_cfg = {"state_dim": 2, "action_dim": 2, "hidden_dims": [2],
                     "learning_rate": .001, "replay_buffer_size": 4,
                     "min_replay_size": 1, "batch_size": 1, "gamma": .9}
        with tempfile.TemporaryDirectory() as temp:
            publish_training_snapshot(
                temp, MultiAgentDQN(["tls"], agent_cfg),
                TrainingCursor(0, 1, 0, 0, 0, 1), contracts,
            )
            modified = copy.deepcopy(original)
            modified["state"]["max_queue"] = 51.0
            with self.assertRaisesRegex(ResearchResumeError, "hash mismatch"):
                validate_publication_observation_contract(modified, contracts)

    def test_missing_malformed_and_inconsistent_identifiers_fail(self):
        base = _config()
        cases = []
        for value in (None, "", " ", 3, "sumoflow-dqn-observation-v4"):
            config = copy.deepcopy(base)
            config["state"]["observation_contract"] = value
            cases.append((config, _contracts(base)))
        config = copy.deepcopy(base)
        del config["state"]["observation_contract"]
        cases.append((config, _contracts(base)))
        for config, contracts in cases:
            with self.subTest(value=config["state"].get("observation_contract")):
                with self.assertRaises((ResearchResumeError, ValueError)):
                    validate_publication_observation_contract(config, contracts)

    def test_missing_or_malformed_configuration_hash_fails(self):
        config = _config()
        for digest in ("", "0" * 63, "g" * 64):
            with self.subTest(digest=digest), self.assertRaisesRegex(
                    ResearchResumeError, "observation-config SHA-256"):
                validate_publication_observation_contract(
                    config, _contracts(config, observation_config_sha256=digest)
                )

    def test_missing_or_malformed_resume_identifier_fails(self):
        config = _config()
        for identifier in (None, "", " ", 7):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                validate_publication_observation_contract(
                    config, _contracts(config, observation_contract=identifier)
                )

    def test_generic_snapshot_contract_is_not_publication_validation(self):
        generic = _contracts(_config(), observation_contract="obs-v1",
                             observation_config_sha256="")
        generic.validate()
        with self.assertRaises(ResearchResumeError):
            validate_publication_observation_contract(_config(), generic)

    def test_fingerprint_is_deterministic_across_yaml_round_trip(self):
        config = _config()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(observation_config_sha256(config), observation_config_sha256(loaded))

    def test_dimension_inconsistency_fails_closed(self):
        config = _config()
        contracts = _contracts(config)
        config["dqn"]["state_dim"] = 36
        # Hash check is deliberately first and proves every effective change is bound.
        contracts = _contracts(config)
        with self.assertRaisesRegex(ResearchResumeError, "dimensions differ"):
            validate_publication_observation_contract(config, contracts)

    def test_hash_consistent_semantic_regressions_fail_early(self):
        cases = (
            ("simulation", "lane_observation_mode", "legacy_zero_fill",
             "research_fail_closed"),
            ("state", "features_per_lane", 4, "features_per_lane=3"),
            ("simulation", "step_length", 5, "one-second timestep"),
            ("bilstm", "history_warmup_seconds", 0, "60-second shared warm-up"),
        )
        for section, field, value, message in cases:
            with self.subTest(field=field):
                config = _config()
                config[section][field] = value
                # A matching digest establishes identity, not semantic validity.
                with self.assertRaisesRegex(ResearchResumeError, message):
                    validate_publication_observation_contract(config, _contracts(config))

    def test_wrong_feature_arithmetic_and_missing_layout_fields_fail(self):
        for field in ("max_lanes", "features_per_lane", "junction_features",
                      "global_features", "state_dim"):
            with self.subTest(missing=field):
                config = _config()
                del config["state"][field]
                with self.assertRaisesRegex(ResearchResumeError, field):
                    validate_publication_observation_contract(config, _contracts(config))
        config = _config()
        config["state"].update(max_lanes=9, features_per_lane=3,
                               junction_features=5, global_features=5)
        # The declared total remains 37, but 9*3+5+5 is not 37 and the
        # canonical v3 component inventory is independently enforced.
        with self.assertRaises(ResearchResumeError):
            validate_publication_observation_contract(config, _contracts(config))

    def test_nonfinite_nonpositive_and_missing_normalizers_fail(self):
        for field in ("max_queue", "max_wait", "max_occupancy", "max_phase_elapsed"):
            for value in (0, -1, float("nan"), float("inf"), None):
                with self.subTest(field=field, value=value):
                    config = _config()
                    if value is None:
                        del config["state"][field]
                    else:
                        config["state"][field] = value
                    with self.assertRaisesRegex(ResearchResumeError, field):
                        validate_publication_observation_contract(config, _contracts(config))

    def test_forecast_mode_protocol_waiting_memory_and_timestep_are_strict(self):
        cases = (
            ("bilstm", "mode", "legacy_adapter", "forecast mode"),
            ("bilstm", "initial_control_protocol", "other", "initial_control_protocol"),
            ("simulation", "waiting_time_memory_s", 99, "waiting memory"),
            ("simulation", "step_length", float("nan"), "one-second timestep"),
        )
        for section, field, value, message in cases:
            with self.subTest(field=field):
                config = _config(); config[section][field] = value
                with self.assertRaisesRegex(ResearchResumeError, message):
                    validate_publication_observation_contract(config, _contracts(config))

    def test_production_entrypoint_validates_and_passes_v3_to_ensemble(self):
        config = _config()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("DeepQN.training.train.require_research_training_authority"), \
                    patch("DeepQN.training.train.MultiAgentDQN") as ensemble, \
                    patch("DeepQN.training.train.run_research_episode_transactions",
                          return_value="validated"):
                result = train_research(
                    config_path=str(path), contracts=_contracts(config), schedule=object(),
                    initial_cursor=TrainingCursor(0, 1, 0, 0, 0, 1),
                    snapshot_root=temp, episode_count=0,
                    episode_runner=lambda *_args, **_kwargs: None,
                )
        self.assertEqual(result, "validated")
        ensemble.assert_called_once_with(
            ["tls"], config["dqn"], observation_contract=RESEARCH_OBSERVATION_CONTRACT
        )

    def test_semantic_failure_precedes_agent_creation_and_restoration(self):
        config = _config(); config["simulation"]["lane_observation_mode"] = "legacy_zero_fill"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("DeepQN.training.train.require_research_training_authority"), \
                    patch("DeepQN.training.train.MultiAgentDQN") as ensemble, \
                    patch("DeepQN.training.train.restore_training_snapshot") as restore:
                with self.assertRaisesRegex(ResearchResumeError, "research_fail_closed"):
                    train_research(
                        config_path=str(path), contracts=_contracts(config), schedule=object(),
                        initial_cursor=TrainingCursor(0, 1, 0, 0, 0, 1),
                        snapshot_root=temp, episode_count=0,
                        episode_runner=lambda *_args, **_kwargs: None,
                        resume_snapshot="must-not-be-read",
                        expected_resume_manifest_sha256="a" * 64,
                    )
            ensemble.assert_not_called()
            restore.assert_not_called()

    def test_research_authority_gate_remains_closed(self):
        from DeepQN.training.train import require_research_training_authority
        with self.assertRaisesRegex(RuntimeError, "trust roots are absent"):
            require_research_training_authority()

    def test_external_decisions_are_unapproved_and_bound_to_actual_sources(self):
        root = Path(__file__).resolve().parents[2]
        register_path = root / "research/2026-10-26-phase2c3-external-decision-register.json"
        register = json.loads(register_path.read_text(encoding="utf-8"))
        self.assertFalse(register["publication_approved"])
        self.assertFalse(register["phase3_authorized"])
        self.assertEqual(register["reviewers"], [])
        for decision in register["decisions"]:
            with self.subTest(decision=decision["id"]):
                self.assertEqual(decision["status"], "unresolved")
                self.assertIsNone(decision["decision"])
                self.assertIsNone(decision["reviewer"])
                self.assertIsNone(decision["approval_evidence_sha256"])
                self.assertTrue(decision["source_artifacts"])
                for artifact in decision["source_artifacts"]:
                    path = root / artifact["path"]
                    self.assertNotEqual(path.resolve(), register_path.resolve())
                    self.assertTrue(path.is_file(), artifact["path"])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                     artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
