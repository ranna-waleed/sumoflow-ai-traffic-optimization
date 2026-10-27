from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.env.sumo_env import SumoEnv
from DeepQN.training.exploratory_train import (
    CLASSIFICATION,
    _apply_runtime_spec,
    _authorize_resume_status,
    _episode_integrity,
    _implementation_identities,
    _latest,
    _merge,
    _next_episode_attempt,
    _run_episode,
    _snapshot_classification,
    _validate_config,
    _validate_input_manifest,
    _verify_implementation_identities,
)
from DeepQN.training.research_resume import (
    ResumeContracts,
    TrainingCursor,
    publish_training_snapshot,
    restore_training_snapshot,
)


class ExploratoryTrainingPersistenceTests(unittest.TestCase):
    def test_behavior_flagged_resume_requires_explicit_nonpublication_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = "stopped_for_collision_or_teleport_review"
            with self.assertRaisesRegex(RuntimeError,
                                        "acknowledge-traffic-review"):
                _authorize_resume_status(root, status, False)
            _authorize_resume_status(root, status, True)
            evidence = json.loads((root / "traffic-review-acknowledgment.json")
                                  .read_text(encoding="utf-8"))
            self.assertFalse(evidence["publication_compatible"])
            self.assertEqual(evidence["classification"],
                             "nonpublication_behavior_flagged")
            with self.assertRaisesRegex(RuntimeError, "failed_integrity"):
                _authorize_resume_status(root, "failed_integrity", True)

    def test_existing_episode_emergency_stops_are_behavior_not_integrity(self):
        root = Path(__file__).resolve().parents[2]
        report_path = root / (
            "DeepQN/diagnostics/phase3_fullnight_exploratory_v3/"
            "night-v6-stopturn-r1-3ep-seed41001/episodes/"
            "episode-000001-attempt-001/episode-report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        telemetry = report["sumo_telemetry"]
        # Reclassify the immutable evidence with the corrected taxonomy.
        telemetry["categories"]["emergency_stops"] = sum(
            "performs emergency stop" in event["message"].lower()
            for event in telemetry["adverse_events"])
        valid, failures = _episode_integrity(
            telemetry, report["metrics"], 2106,
            report["final_summary"], report["completed_horizon"])
        self.assertTrue(valid, failures)
        self.assertEqual(telemetry["categories"]["emergency_stops"], 2)
        self.assertEqual(
            _snapshot_classification(
                telemetry["categories"]["emergency_braking"], 0, 1,
                telemetry["categories"]["emergency_stops"]),
            "nonpublication_behavior_flagged")

    def test_actual_sumo_errors_remain_integrity_failures(self):
        telemetry = {"failures": [], "adverse_events": [
            {"message": "Error: invalid route for vehicle v", "sources": ["error_log"]}],
            "categories": {"ignored_departure_order": 0, "unsafe_signal": 0}}
        valid, failures = _episode_integrity(
            telemetry, {"valid": True}, 1, {"loaded": "1"}, True)
        self.assertFalse(valid)
        self.assertIn("non-traffic SUMO warnings/errors", failures[0])

    def test_interrupted_episode_uses_new_attempt_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _next_episode_attempt(root, 2)
            first.mkdir()
            (first / "partial.txt").write_text("preserve", encoding="utf-8")
            second = _next_episode_attempt(root, 2)
            self.assertEqual(second.name, "episode-000002-attempt-002")
            self.assertEqual((first / "partial.txt").read_text(), "preserve")

    def test_fullnight_configuration_uses_normal_learning_threshold(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "DeepQN/configs/diagnostics/night_three_episode_exploratory_v3.yaml"
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = yaml.safe_load((path.parent / spec["base_config"]).read_text(encoding="utf-8"))
        config = _apply_runtime_spec(_merge(base, spec["overrides"]), spec)
        _validate_config(config, spec)
        self.assertEqual(config["simulation"]["port"], 18813)
        self.assertEqual(config["dqn"]["min_replay_size"], 1000)
        self.assertEqual((config["profiles"]["night"]["begin"],
                          config["profiles"]["night"]["end"]),
                         (79200, 86400))

    def test_declared_traci_port_reaches_actual_environment_constructor(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "DeepQN/configs/diagnostics/night_three_episode_exploratory_v3.yaml"
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = yaml.safe_load((path.parent / spec["base_config"]).read_text(encoding="utf-8"))
        config = _apply_runtime_spec(_merge(base, spec["overrides"]), spec)
        _validate_config(config, spec)
        input_manifest = json.loads(
            (root / "DeepQN/diagnostics/inputs/night_synthetic_v6/input-manifest.json")
            .read_text(encoding="utf-8"))
        route_path = Path(input_manifest["explicit_routes"]["path"])
        agents = Mock()
        agents.agents = {"tls": Mock(total_updates=0, replay=[])}
        agents.episode_stats.return_value = {"tls": {}}
        telemetry = {
            "valid": False, "failures": ["deliberate test stop"],
            "adverse_events": [], "categories": {
                "ignored_departure_order": 0, "unsafe_signal": 0,
                "emergency_braking": 0, "collisions": 0, "teleports": 0,
            },
        }
        metrics = Mock()
        metrics.to_dict.return_value = {
            "valid": False, "errors": ["deliberate test stop"],
            "collisions": 0, "teleports": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            diagnostic = tmp_path / "diagnostic.json"
            diagnostic.write_text("{}", encoding="utf-8")
            env = Mock()
            env._sumo_diagnostic_path = diagnostic
            env.reset.side_effect = RuntimeError("stop after construction")
            with patch("DeepQN.training.exploratory_train.SumoEnv",
                       return_value=env) as env_class, \
                 patch("DeepQN.training.exploratory_train._collect_sumo_telemetry",
                       return_value=telemetry), \
                 patch("DeepQN.training.exploratory_train.load_metrics_from_paths",
                       return_value=metrics), \
                 patch("DeepQN.training.exploratory_train.parse_scheduled_demand",
                       return_value={"vehicle": 79200.0}):
                record = _run_episode(
                    agents=agents, config=config, episode=1, sumo_seed=31001,
                    episode_dir=tmp_path / "episode", wall_timeout_s=60,
                    route_path=route_path,
                    progress_path=tmp_path / "progress.json")
            self.assertEqual(env_class.call_args.kwargs["port"], 18813)
            self.assertEqual(record["traci_port"], 18813)
            effective = yaml.safe_load(
                (tmp_path / "episode/effective-training.yaml").read_text(
                    encoding="utf-8"))
            self.assertEqual(effective["simulation"]["port"], 18813)
            actual_env = SumoEnv(effective, profile="night",
                                 port=env_class.call_args.kwargs["port"])
            self.assertEqual(actual_env.port, 18813)

    def test_invalid_declared_ports_are_rejected(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "DeepQN/configs/diagnostics/night_three_episode_exploratory_v3.yaml"
        original = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = yaml.safe_load((path.parent / original["base_config"]).read_text(encoding="utf-8"))
        for invalid in (True, 0, 65536, 18813.0, "18813"):
            with self.subTest(port=invalid):
                spec = dict(original, traci_port=invalid)
                config = _apply_runtime_spec(_merge(base, spec["overrides"]), spec)
                with self.assertRaisesRegex(ValueError, "traci_port"):
                    _validate_config(config, spec)

    def test_resume_rejects_modified_implementation_identity(self):
        identities = _implementation_identities()
        self.assertIn("DeepQN/training/exploratory_train.py", identities)
        altered = json.loads(json.dumps(identities))
        altered["DeepQN/training/exploratory_train.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "implementation identity changed"):
            _verify_implementation_identities(altered)

    def test_integrity_failure_is_separate_from_adverse_traffic(self):
        telemetry = {"valid": False, "failures": ["missing error log"],
                     "adverse_events": [], "categories": {
                         "ignored_departure_order": 0, "unsafe_signal": 0}}
        valid, reasons = _episode_integrity(
            telemetry, {"valid": True}, 10, {"loaded": "10"}, True)
        self.assertFalse(valid)
        self.assertIn("missing error log", reasons)

        traffic_only = {"valid": True, "failures": [],
            "adverse_events": [{"message":
                "Warning: Vehicle v performs emergency braking", "sources": ["error_log"]}],
            "categories": {"ignored_departure_order": 0, "unsafe_signal": 0}}
        valid, reasons = _episode_integrity(
            traffic_only, {"valid": True}, 10, {"loaded": "10"}, True)
        self.assertTrue(valid)
        self.assertEqual(reasons, [])

        collision_traffic = {
            "valid": True, "failures": [], "adverse_events": [
                {"message": "Warning: Vehicle 'a'; junction collision with vehicle 'b'",
                 "sources": ["error_log"]},
                {"message": "Teleports: 2 (Collisions: 1, Wrong Lane: 1)",
                 "sources": ["message_log"]},
            ], "categories": {"ignored_departure_order": 0,
                               "unsafe_signal": 0},
        }
        valid, reasons = _episode_integrity(
            collision_traffic, {"valid": True}, 10, {"loaded": "10"}, True)
        self.assertTrue(valid)
        self.assertEqual(reasons, [])

        skipped_stop = {"valid": True, "failures": [], "adverse_events": [
            {"message": "Warning: Vehicle 'v' skips stop on lane 'lane_0'",
             "sources": ["error_log"]}],
            "categories": {"ignored_departure_order": 0, "unsafe_signal": 0}}
        valid, reasons = _episode_integrity(
            skipped_stop, {"valid": True}, 10, {"loaded": "10"}, True)
        self.assertFalse(valid)
        self.assertIn("skips stop", reasons[0])

    def test_v6_manifest_passes_source_route_integrity_gate(self):
        root = Path(__file__).resolve().parents[2]
        manifest = root / "DeepQN/diagnostics/inputs/night_synthetic_v6/input-manifest.json"
        value, paths = _validate_input_manifest(manifest)
        self.assertEqual(value["schema"],
                         "sumoflow-night-synthetic-input-candidate-v6")
        self.assertTrue(all(path.is_file() for path in paths.values()))

    @staticmethod
    def _cfg():
        return {
            "state_dim": 3, "action_dim": 2, "hidden_dims": [8],
            "dueling": True, "learning_rate": 0.001, "gamma": 0.95,
            "batch_size": 1, "replay_buffer_size": 20,
            "min_replay_size": 1, "epsilon_start": 1.0,
            "epsilon_end": 0.05, "epsilon_decay": 0.9,
            "tau": 0.005, "seed": 17,
        }

    @staticmethod
    def _contracts(run_id="exploratory-test"):
        digest = "1" * 64
        return ResumeContracts(
            run_id=run_id, run_manifest_sha256=digest,
            dataset_manifest_sha256="2" * 64,
            demand_registry_sha256="3" * 64,
            split_registry_sha256="4" * 64,
            observation_contract="engineering-v4", reward_contract="interval-v1",
            forecast_contract="no_forecast", git_commit="test-commit",
            environment_sha256="5" * 64)

    @staticmethod
    def _transition(agents, value):
        state = {"tls": np.array([value, 0, 1], dtype=np.float32)}
        next_state = {"tls": np.array([value + 1, 0, 1], dtype=np.float32)}
        agents.push(state, {"tls": 0}, {"tls": 1.0}, next_state,
                    terminated=False, truncated=False, valid=True,
                    interval_duration_s=10)
        agents.update()

    def test_three_episode_snapshots_preserve_one_ensemble_and_counters(self):
        agents = MultiAgentDQN(["tls"], self._cfg(), "engineering-v4")
        identity = id(agents.agents["tls"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for episode in range(1, 4):
                self._transition(agents, episode)
                cursor = TrainingCursor(episode, episode + 1, 0, 0,
                                        episode, 100 + episode, True)
                result = publish_training_snapshot(
                    root, agents, cursor, self._contracts(),
                    episode_record={"classification": CLASSIFICATION,
                                    "publication_compatible": False})
                self.assertTrue(Path(result["path"], "COMPLETE.json").is_file())
                self.assertEqual(id(agents.agents["tls"]), identity)
                self.assertEqual(len(agents.agents["tls"].replay), episode)
                self.assertEqual(agents.agents["tls"].total_updates, episode)

    def test_snapshot_resume_restores_full_state_and_rng_deterministically(self):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        agents = MultiAgentDQN(["tls"], self._cfg(), "engineering-v4")
        self._transition(agents, 1)
        with tempfile.TemporaryDirectory() as tmp:
            published = publish_training_snapshot(
                tmp, agents, TrainingCursor(1, 2, 0, 0, 1, 102, True),
                self._contracts("resume-test"),
                episode_record={"classification":
                    "nonpublication_behavior_flagged",
                    "publication_compatible": False})
            expected_rng = (random.random(), float(np.random.random()),
                            float(torch.rand(1)))
            restored = MultiAgentDQN(["tls"], self._cfg(), "engineering-v4")
            cursor = restore_training_snapshot(
                published["path"], restored, self._contracts("resume-test"),
                published["manifest_sha256"])
            actual_rng = (random.random(), float(np.random.random()),
                          float(torch.rand(1)))
            self.assertEqual(cursor.next_episode, 2)
            self.assertEqual(actual_rng, expected_rng)
            self.assertEqual(len(restored.agents["tls"].replay), 1)
            self.assertEqual(restored.agents["tls"].total_updates, 1)
            self.assertEqual(restored.agents["tls"].epsilon,
                             agents.agents["tls"].epsilon)
            self.assertEqual(
                restored.agents["tls"].optimiser.state_dict()["state"].keys(),
                agents.agents["tls"].optimiser.state_dict()["state"].keys())

    def test_atomic_snapshot_failure_publishes_no_complete_state(self):
        agents = MultiAgentDQN(["tls"], self._cfg(), "engineering-v4")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                publish_training_snapshot(
                    tmp, agents, TrainingCursor(1, 2, 0, 0, 1, 2, True),
                    self._contracts("failure-test"), fail_after_files=1,
                    episode_record={"classification": CLASSIFICATION})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_resume_recovers_complete_snapshot_after_stale_pointer(self):
        agents = MultiAgentDQN(["tls"], self._cfg(), "engineering-v4")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snapshots = run_dir / "snapshots"
            for episode in (1, 2):
                self._transition(agents, episode)
                result = publish_training_snapshot(
                    snapshots, agents,
                    TrainingCursor(episode, episode + 1, 0, 0, episode,
                                   100 + episode, True),
                    self._contracts("recover-test"),
                    episode_record={"checkpoint_classification":
                        "nonpublication_exploratory_training_state"})
                if episode == 1:
                    (run_dir / "LATEST.json").write_text(json.dumps({
                        "completed_episode": 1,
                        "snapshot_path": result["path"],
                        "manifest_sha256": result["manifest_sha256"],
                    }), encoding="utf-8")
            recovered = _latest(run_dir)
            self.assertEqual(recovered["completed_episode"], 2)

    def test_adverse_outcomes_are_explicitly_nonpublication_flagged(self):
        self.assertEqual(_snapshot_classification(1, 0, 0),
                         "nonpublication_behavior_flagged")
        self.assertEqual(_snapshot_classification(0, 1, 0),
                         "nonpublication_behavior_flagged")
        self.assertEqual(_snapshot_classification(0, 0, 1),
                         "nonpublication_behavior_flagged")
        self.assertEqual(_snapshot_classification(0, 0, 0),
                         "nonpublication_exploratory_training_state")


if __name__ == "__main__":
    unittest.main()
