import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.evaluation.heldout_dqn_evaluation import (
    ROOT, _controlled, _ensemble_fingerprint, _failure_record, _load_spec,
    _restore_episode3, _validate_fixed_identity, _write_json_atomic,
)
from DeepQN.evaluation.metrics import EpisodeMetrics
from DeepQN.evaluation.evaluate import (
    _collect_evaluation_telemetry, _is_traffic_behavior_event,
    _persist_action_trace, _set_controlled_evaluation_seed,
)
from DeepQN.training.reproducibility import seed_all


SPEC = ROOT / "DeepQN/configs/diagnostics/night_v6_heldout_evaluation_v1.yaml"


class _Manifest:
    def __init__(self):
        self.run_id = "untrained_dqn_night_seed42004_run0"
        self.completion_status = "complete"
        self.checkpoint_manifest = None
        self.controller_diagnostics = {}
    def write(self, path):
        Path(path).write_text("{}\n", encoding="utf-8")
    def to_dict(self):
        return {"run_id": self.run_id, "completion_status": self.completion_status,
                "checkpoint_manifest": self.checkpoint_manifest,
                "controller_diagnostics": self.controller_diagnostics}


class HeldoutEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.config, cls.input_manifest, cls.inputs = _load_spec(SPEC)
        cls.spec = copy.deepcopy(cls.spec)
        cls.spec["_spec_path"] = str(SPEC)
        from DeepQN.evaluation.demand_provenance import sha256_file
        cls.spec["_input_manifest_sha256"] = sha256_file(
            (SPEC.parent / cls.spec["input_manifest"]).resolve())
        cls.spec["_route_sha256"] = cls.input_manifest["explicit_routes"]["sha256"]
        cls.spec["_network_sha256"] = cls.input_manifest["network_variant"]["sha256"]
        cls.config["_evaluation_spec_path"] = str(SPEC)

    def test_pinned_episode3_snapshot_loads_all_seven_agents(self):
        agents, evidence = _restore_episode3(self.spec, self.config)
        self.assertEqual(len(agents.agents), 7)
        self.assertEqual(evidence["completed_episode"], 3)
        self.assertFalse(evidence["publication_compatible"])
        self.assertEqual(len(evidence["constituent_hashes"]), 7)
        self.assertTrue(all(agent.total_updates == 1161
                            for agent in agents.agents.values()))

    def test_wrong_external_snapshot_hash_is_rejected(self):
        altered = dict(self.spec)
        altered["snapshot_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "snapshot pointer"):
            _restore_episode3(altered, self.config)

    def test_fresh_ensemble_is_untrained_and_reproducible(self):
        seed_all(1042004, deterministic_torch=True)
        first = MultiAgentDQN(list(self.config["tls_junctions"]), self.config["dqn"],
            observation_contract=self.config["state"]["observation_contract"])
        seed_all(1042004, deterministic_torch=True)
        second = MultiAgentDQN(list(self.config["tls_junctions"]), self.config["dqn"],
            observation_contract=self.config["state"]["observation_contract"])
        self.assertEqual(_ensemble_fingerprint(first), _ensemble_fingerprint(second))
        self.assertTrue(all(row["replay_size"] == 0 and row["total_updates"] == 0
                            for row in _ensemble_fingerprint(first).values()))

    def test_evaluation_only_runner_rejects_agent_mutation(self):
        seed_all(11, deterministic_torch=True)
        agents = MultiAgentDQN(list(self.config["tls_junctions"]), self.config["dqn"],
            observation_contract=self.config["state"]["observation_contract"])
        def mutating(**kwargs):
            next(iter(kwargs["agents"].agents.values())).epsilon -= 0.1
            return EpisodeMetrics(profile="night", mode="untrained_dqn"), _Manifest()
        with tempfile.TemporaryDirectory() as tmp, patch(
                "DeepQN.evaluation.heldout_dqn_evaluation.run_controlled_episode",
                side_effect=mutating):
            with self.assertRaisesRegex(RuntimeError, "evaluation mutated"):
                _controlled(label="untrained_dqn", agents=agents,
                    config=self.config, seed=42004, output=Path(tmp), port=18831,
                    checkpoint=None, wall_clock_timeout_s=60)

    def test_evaluation_only_runner_records_unchanged_state(self):
        seed_all(12, deterministic_torch=True)
        agents = MultiAgentDQN(list(self.config["tls_junctions"]), self.config["dqn"],
            observation_contract=self.config["state"]["observation_contract"])
        with tempfile.TemporaryDirectory() as tmp, patch(
                "DeepQN.evaluation.heldout_dqn_evaluation.run_controlled_episode",
                return_value=(EpisodeMetrics(profile="night", mode="untrained_dqn"),
                              _Manifest())):
            result = _controlled(label="untrained_dqn", agents=agents,
                config=self.config, seed=42004, output=Path(tmp), port=18831,
                checkpoint=None, wall_clock_timeout_s=60)
        self.assertTrue(result["manifest"]["controller_diagnostics"]
                        ["evaluation_state_unchanged"])

    def test_structured_safety_telemetry_supports_clean_traci_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("errors.log", "messages.log", "diagnostic.log"):
                (root / name).write_text("Simulation ended by TraCI.\n", encoding="utf-8")
            (root / "statistic.xml").write_text(
                '<statistics><teleports total="0"/><safety collisions="0" '
                'emergencyStops="0" emergencyBraking="0"/></statistics>', encoding="utf-8")
            result = _collect_evaluation_telemetry(
                error_log=root / "errors.log", message_log=root / "messages.log",
                diagnostic_log=root / "diagnostic.log", statistic_path=root / "statistic.xml")
        self.assertTrue(result["valid"])
        self.assertFalse(any(result["categories"].values()))

    def test_structured_safety_telemetry_preserves_adverse_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "errors.log").write_text(
                "Warning: Vehicle 'v' performs emergency braking.\n", encoding="utf-8")
            (root / "messages.log").write_text("", encoding="utf-8")
            (root / "diagnostic.log").write_text("", encoding="utf-8")
            (root / "statistic.xml").write_text(
                '<statistics><teleports total="1"/><safety collisions="0" '
                'emergencyStops="0" emergencyBraking="1"/></statistics>', encoding="utf-8")
            result = _collect_evaluation_telemetry(
                error_log=root / "errors.log", message_log=root / "messages.log",
                diagnostic_log=root / "diagnostic.log", statistic_path=root / "statistic.xml")
        self.assertEqual(result["categories"]["emergency_braking"], 1)
        self.assertEqual(result["categories"]["teleports"], 1)

    def test_real_sumo_collision_warning_is_traffic_evidence(self):
        warning = ("Warning: Vehicle 'flow_584.1'; collision with vehicle "
                   "'flow_604.1', lane='10873186#0_3', gap=-0.24, "
                   "time=82632.00, stage=move.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "errors.log").write_text(warning + "\n", encoding="utf-8")
            (root / "messages.log").write_text(warning + "\n", encoding="utf-8")
            (root / "diagnostic.log").write_text("", encoding="utf-8")
            (root / "statistic.xml").write_text(
                '<statistics><teleports total="0"/><safety collisions="1" '
                'emergencyStops="0" emergencyBraking="0"/></statistics>', encoding="utf-8")
            result = _collect_evaluation_telemetry(
                error_log=root / "errors.log", message_log=root / "messages.log",
                diagnostic_log=root / "diagnostic.log", statistic_path=root / "statistic.xml")
        self.assertEqual(result["categories"]["collisions"], 1)
        self.assertEqual(len(result["adverse_events"]), 1)
        self.assertIn("collision with vehicle", result["adverse_events"][0]["message"])
        self.assertTrue(_is_traffic_behavior_event(warning))
        self.assertFalse(_is_traffic_behavior_event(
            "Error: Vehicle 'x' has no valid route."))

    def test_collision_detail_counter_contradiction_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "errors.log").write_text(
                "Warning: Vehicle 'a'; collision with vehicle 'b'.\n", encoding="utf-8")
            (root / "messages.log").write_text("", encoding="utf-8")
            (root / "diagnostic.log").write_text("", encoding="utf-8")
            (root / "statistic.xml").write_text(
                '<statistics><teleports total="0"/><safety collisions="0" '
                'emergencyStops="0" emergencyBraking="0"/></statistics>', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "collision details exceed"):
                _collect_evaluation_telemetry(
                    error_log=root / "errors.log", message_log=root / "messages.log",
                    diagnostic_log=root / "diagnostic.log", statistic_path=root / "statistic.xml")

    def test_exact_r2_summary_counter_is_not_a_second_detail(self):
        collision = ("Warning: Vehicle 'flow_584.1'; collision with vehicle "
                     "'flow_604.1', lane='10873186#0_3', gap=-0.24, "
                     "time=82632.00, stage=move.")
        braking = ("Warning: Vehicle 'flow_584.1' performs emergency braking on "
                   "lane '10873186#0_3' with decel=9.00, wished=4.50, "
                   "severity=1.00, time=82633.00.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "errors.log").write_text(
                collision + "\n" + braking + "\n", encoding="utf-8")
            (root / "messages.log").write_text(
                "Teleports: 0 (Collisions: 1)\n Emergency Braking: 1\n",
                encoding="utf-8")
            (root / "diagnostic.log").write_text("", encoding="utf-8")
            (root / "statistic.xml").write_text(
                '<statistics><teleports total="0"/><safety collisions="1" '
                'emergencyStops="0" emergencyBraking="1"/></statistics>', encoding="utf-8")
            result = _collect_evaluation_telemetry(
                error_log=root / "errors.log", message_log=root / "messages.log",
                diagnostic_log=root / "diagnostic.log", statistic_path=root / "statistic.xml")
        self.assertEqual(result["categories"]["emergency_braking"], 1)
        self.assertEqual(result["reconciliation"]["emergency_braking"], {
            "structured_count": 1, "deduplicated_detail_count": 1,
            "summary_counter_values": [1], "status": "matched"})
        self.assertEqual(result["source_evidence"]["error_log"]
                         ["emergency_braking_detail_count"], 1)
        self.assertEqual(result["source_evidence"]["message_log"]
                         ["emergency_braking_summary_counters"], [1])
        self.assertEqual(len(result["adverse_events"]), 2)

    def test_structured_event_without_detail_is_explicit_not_zeroed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("errors.log", "messages.log", "diagnostic.log"):
                (root / name).write_text("", encoding="utf-8")
            (root / "statistic.xml").write_text(
                '<statistics><teleports total="0"/><safety collisions="1" '
                'emergencyStops="0" emergencyBraking="2"/></statistics>', encoding="utf-8")
            result = _collect_evaluation_telemetry(
                error_log=root / "errors.log", message_log=root / "messages.log",
                diagnostic_log=root / "diagnostic.log", statistic_path=root / "statistic.xml")
        self.assertEqual(result["categories"]["emergency_braking"], 2)
        self.assertEqual(result["reconciliation"]["emergency_braking"]["status"],
                         "structured_count_exceeds_detailed_warnings")
        self.assertEqual(result["reconciliation"]["collisions"]["status"],
                         "structured_count_exceeds_detailed_warnings")

    def test_action_trace_and_partial_summary_are_atomic_and_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); trace = root / "action.json"
            digest, raw = _persist_action_trace(trace, {
                "requested": [{"tls": 1}], "physical": [{"phase": 2}]})
            import hashlib
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            self.assertTrue(trace.is_file()); self.assertFalse((root / "action.json.tmp").exists())
            manifest = root / "manifest_failed.json"
            manifest.write_text('{"completion_status":"failed"}\n', encoding="utf-8")
            failure = _failure_record(root, RuntimeError("postprocess failed"))
            self.assertEqual(failure["status"], "postprocessing_failed")
            self.assertEqual(failure["manifest"]["completion_status"], "failed")
            summary = root / "paired.json"
            _write_json_atomic(summary, {"pairs": [{"status": "untrained_dqn_failed"}]})
            self.assertIn("untrained_dqn_failed", summary.read_text(encoding="utf-8"))

    def test_controlled_evaluation_seed_overrides_inherited_training_seed(self):
        config = {"simulation": {"seed": 42}}
        _set_controlled_evaluation_seed(config, 42004)
        self.assertEqual(config["simulation"]["seed"], 42004)
        for invalid in (True, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _set_controlled_evaluation_seed(config, invalid)

    def test_missing_structured_safety_telemetry_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("errors.log", "messages.log", "diagnostic.log"):
                (root / name).write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "statistic output is missing"):
                _collect_evaluation_telemetry(
                    error_log=root / "errors.log", message_log=root / "messages.log",
                    diagnostic_log=root / "diagnostic.log", statistic_path=root / "missing.xml")


if __name__ == "__main__":
    unittest.main()
