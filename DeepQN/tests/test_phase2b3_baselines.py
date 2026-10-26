from __future__ import annotations

import json
import hashlib
import shutil
import socket
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import yaml

from DeepQN.agent.replay_buffer import PrioritisedReplayBuffer, UniformReplayBuffer
from DeepQN.evaluation.compute_pilot import measure_callable, measure_traci_fixture
from DeepQN.evaluation.controller_baselines import (
    MaxPressureActionProvider, MaxPressureController, load_max_pressure_definition,
    load_signal_program, verify_loaded_signal_programs_bounded,
)
from DeepQN.evaluation.reward_calibration import calibrate_reward_references
from DeepQN.evaluation.baseline import run_native_baseline_episode
from DeepQN.evaluation.evaluate import run_controlled_episode
from DeepQN.env.sumo_env import SumoEnv, _PhaseController


ROOT = Path(__file__).resolve().parent
TINY = ROOT / "fixtures" / "tiny_sumo"
FIXTURE = ROOT / "fixtures" / "controller_baselines"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ControllerDefinitionTests(unittest.TestCase):
    def test_fixed_time_schedule_is_explicit_and_frozen(self):
        program = load_signal_program(TINY / "tiny.add.xml", "center", "static")
        self.assertEqual([p["duration_s"] for p in program.phases], [10, 3, 10, 3])
        self.assertEqual([p["state"] for p in program.phases],
                         ["GGggrrrrGGggrrrr", "yyyyrrrryyyyrrrr",
                          "rrrrGGggrrrrGGgg", "rrrryyyyrrrryyyy"])

    def test_actuated_definition_has_variable_greens_and_parameters(self):
        program = load_signal_program(FIXTURE / "actuated.add.xml", "center", "actuated")
        self.assertEqual(program.program_type, "actuated")
        self.assertEqual(program.phases[0]["min_duration_s"], 5)
        self.assertEqual(program.phases[0]["max_duration_s"], 20)

    def test_static_tahrir_style_program_cannot_be_called_actuated(self):
        with self.assertRaisesRegex(ValueError, "not 'actuated'"):
            load_signal_program(TINY / "tiny.add.xml", "center", "actuated")

    def test_native_baseline_cannot_masquerade_as_matched_warmup(self):
        with self.assertRaisesRegex(ValueError, "not phase-synchronized"):
            run_native_baseline_episode(
                cfg={}, profile="tiny", output_dir=Path("unused"), run_index=0, seed=17,
                config_path=TINY / "tiny.sumocfg", controller_type="fixed_time",
                tls_program_path=TINY / "tiny.add.xml", tls_ids=["center"],
                comparison_scope="primary_matched_policy")

    def test_native_baseline_requires_explicit_wall_clock_limit(self):
        with self.assertRaisesRegex(ValueError, "explicit wall_clock_timeout_s"):
            run_native_baseline_episode(
                cfg={}, profile="tiny", output_dir=Path("unused"), run_index=0, seed=17,
                config_path=TINY / "tiny.sumocfg", controller_type="fixed_time",
                tls_program_path=TINY / "tiny.add.xml", tls_ids=["center"])

    def test_max_pressure_rejects_ai_artifact_labels(self):
        provider = type("Provider", (), {})()
        with self.assertRaisesRegex(ValueError, "must not use"):
            run_controlled_episode(
                cfg={}, profile="tiny", agents=None, controller="max_pressure",
                output_dir=Path("unused"), run_index=0, seed=1,
                checkpoint_manifest=object(), model_mode="verified_model",
                config_path=TINY / "tiny.sumocfg", port=0,
                action_provider=provider)

    def test_genuine_movement_pressure_uses_upstream_minus_downstream(self):
        definition = load_max_pressure_definition(
            FIXTURE / "max_pressure.json", TINY / "tiny.net.xml", "center")
        controller = MaxPressureController(definition)
        queues = {f"{direction}_{kind}_0": 0.0
                  for direction in "nesw" for kind in ("in", "out")}
        queues.update(n_in_0=12, s_in_0=8, e_in_0=2, w_in_0=1,
                      s_out_0=1, n_out_0=1, e_out_0=1, w_out_0=1)
        self.assertEqual(controller.select_phase(queues), 0)
        queues.update(n_in_0=0, s_in_0=0, e_in_0=15, w_in_0=15)
        self.assertEqual(controller.select_phase(queues), 2)

    def test_incomplete_or_wrong_movement_fails_closed(self):
        data = json.loads((FIXTURE / "max_pressure.json").read_text(encoding="utf-8"))
        data["movements"][0]["link_index"] = 99
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a controlled network connection"):
                load_max_pressure_definition(path, TINY / "tiny.net.xml", "center")

    def test_incomplete_movement_inventory_fails_closed(self):
        data = json.loads((FIXTURE / "max_pressure.json").read_text(encoding="utf-8"))
        data["movements"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every controlled link"):
                load_max_pressure_definition(path, TINY / "tiny.net.xml", "center")

    def test_action_provider_reads_physical_queues_before_complete_request(self):
        definition = load_max_pressure_definition(
            FIXTURE / "max_pressure.json", TINY / "tiny.net.xml", "center")
        provider = MaxPressureActionProvider({"center": MaxPressureController(definition)})
        env = object.__new__(SumoEnv)
        env._phase_controllers = {"center": _PhaseController("center", {
            "green_phases": [0, 2], "green_to_yellow": {0: 1, 2: 3},
            "yellow_to_next": {1: 2, 3: 0}, "min_green": 2,
        })}
        def queue(lane):
            return 12 if lane in {"e_in_0", "w_in_0"} else 0
        with patch("traci.lane.getLastStepHaltingNumber", side_effect=queue):
            self.assertEqual(provider.actions(env, {}), {"center": 1})
            self.assertEqual(provider.last_selected_targets, {"center": 2})

    def test_binary_adapter_rejects_three_green_and_red_hold_topologies(self):
        definition = load_max_pressure_definition(
            FIXTURE / "max_pressure.json", TINY / "tiny.net.xml", "center")
        provider = MaxPressureActionProvider({"center": MaxPressureController(definition)})
        for cfg in (
            {"green_phases": [0, 2, 4]},
            {"green_phases": [0, 2], "red_hold_phase": 4},
        ):
            env = object.__new__(SumoEnv)
            env._phase_controllers = {"center": _PhaseController("center", cfg)}
            with patch("traci.lane.getLastStepHaltingNumber", return_value=0):
                with self.assertRaisesRegex(ValueError, "unsupported by binary"):
                    provider.actions(env, {})

    def test_multi_phase_movement_service_must_be_complete(self):
        network_text = (TINY / "tiny.net.xml").read_text(encoding="utf-8")
        network_text = network_text.replace(
            'state="rrrrGGggrrrrGGgg"', 'state="GrrrGGggrrrrGGgg"', 1)
        definition = json.loads((FIXTURE / "max_pressure.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); network = root / "network.net.xml"; mapping = root / "map.json"
            network.write_text(network_text, encoding="utf-8")
            mapping.write_text(json.dumps(definition), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "service inventory mismatch"):
                load_max_pressure_definition(mapping, network, "center")
            definition["movements"][0]["phase_indices"] = [0, 2]
            mapping.write_text(json.dumps(definition), encoding="utf-8")
            loaded = load_max_pressure_definition(mapping, network, "center")
            pressure = MaxPressureController(loaded).pressures({
                f"{direction}_{kind}_0": 0 for direction in "nesw" for kind in ("in", "out")})
            self.assertIn(0, pressure)
            self.assertIn(2, pressure)


class CalibrationTests(unittest.TestCase):
    def _sample(self, path: Path, partition: str):
        path.write_text(json.dumps({"partition": partition, "demand_sha256": "a" * 64,
                                    "intervals": {"delay": [1, 2], "co2": [10, 20],
                                                  "arrivals": [1, 2], "pressure": [3, 4]}}),
                        encoding="utf-8")

    def test_calibration_is_frozen_and_unapproved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "sample.json"; output = root / "manifest.json"
            self._sample(source, "calibration")
            manifest = calibrate_reward_references([source], output)
            self.assertFalse(manifest["publication_approved"])
            self.assertEqual(manifest["references"]["co2"]["unit"], "mg")
            with self.assertRaises(FileExistsError):
                calibrate_reward_references([source], output)

    def test_held_out_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "sample.json"
            self._sample(source, "test")
            with self.assertRaisesRegex(ValueError, "held-out"):
                calibrate_reward_references([source], root / "manifest.json")

    def test_publication_candidate_is_bound_to_files_and_frozen_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); demand = root / "demand.rou.xml"; sample = root / "sample.json"
            demand.write_text("<routes/>", encoding="utf-8")
            demand_hash = hashlib.sha256(demand.read_bytes()).hexdigest()
            sample.write_text(json.dumps({"realization_id": "cal-1", "partition": "calibration",
                "demand_sha256": demand_hash, "intervals": {"delay": [1], "co2": [2],
                "arrivals": [1], "pressure": [3]}}), encoding="utf-8")
            sample_hash = hashlib.sha256(sample.read_bytes()).hexdigest()
            registry = root / "splits.json"
            registry.write_text(json.dumps({"schema": "sumoflow-calibration-split-registry-v1",
                "realizations": [{"realization_id": "cal-1", "partition": "calibration",
                    "demand_path": demand.name, "demand_sha256": demand_hash,
                    "sample_path": sample.name, "sample_sha256": sample_hash}]}), encoding="utf-8")
            registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
            manifest = calibrate_reward_references([sample], root / "out.json",
                mode="publication_candidate", dataset_label="unapproved_test",
                split_registry_path=registry,
                expected_split_registry_sha256=registry_hash)
            self.assertFalse(manifest["publication_approved"])
            self.assertEqual(manifest["split_registry"]["sha256"], registry_hash)
            demand.write_text("<routes><vehicle/></routes>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "demand artifact hash mismatch"):
                calibrate_reward_references([sample], root / "out2.json",
                    mode="publication_candidate", split_registry_path=registry,
                    expected_split_registry_sha256=registry_hash)
            demand.write_text("<routes/>", encoding="utf-8")
            altered = json.loads(sample.read_text(encoding="utf-8"))
            altered["intervals"]["co2"] = [999]
            sample.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration source hash mismatch"):
                calibrate_reward_references([sample], root / "out3.json",
                    mode="publication_candidate", split_registry_path=registry,
                    expected_split_registry_sha256=registry_hash)

    def test_publication_registry_rejects_invalid_hash_duplicates_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); demand = root / "demand"; sample = root / "sample"
            demand.write_text("demand", encoding="utf-8"); demand_hash = hashlib.sha256(demand.read_bytes()).hexdigest()
            sample.write_text(json.dumps({"realization_id": "a", "partition": "calibration",
                "demand_sha256": demand_hash, "intervals": {"delay":[1],"co2":[1],"arrivals":[1],"pressure":[1]}}), encoding="utf-8")
            sample_hash = hashlib.sha256(sample.read_bytes()).hexdigest()
            base = {"realization_id":"a","partition":"calibration","demand_path":"demand",
                    "demand_sha256":demand_hash,"sample_path":"sample","sample_sha256":sample_hash}
            for rows, pattern in (([base, dict(base)], "duplicate"),
                                  ([base, {**base, "realization_id":"held", "partition":"test"}], "byte-identical"),
                                  ([base, {**base, "realization_id":"second"}], "byte-identical")):
                registry = root / f"{pattern}.json"
                registry.write_text(json.dumps({"schema":"sumoflow-calibration-split-registry-v1",
                                                "realizations":rows}), encoding="utf-8")
                digest = hashlib.sha256(registry.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, pattern):
                    calibrate_reward_references([sample], root / f"{pattern}.out",
                        mode="publication_candidate", split_registry_path=registry,
                        expected_split_registry_sha256=digest)
            with self.assertRaisesRegex(ValueError, "SHA-256 syntax"):
                calibrate_reward_references([sample], root / "bad.out",
                    mode="publication_candidate", split_registry_path=root / "duplicate.json",
                    expected_split_registry_sha256="g" * 64)


class ReplayResumeTests(unittest.TestCase):
    @staticmethod
    def _fill(buffer):
        for index in range(8):
            state = np.array([index, index + 0.5], dtype=np.float32)
            buffer.push(state, index % 2, float(index), state + 1,
                        terminated=index == 7, interval_duration_s=10)

    def test_uniform_sampling_resumes_deterministically(self):
        original = UniformReplayBuffer(capacity=10, seed=7); self._fill(original)
        restored = UniformReplayBuffer(capacity=10, seed=999)
        restored.load_state_dict(original.state_dict())
        self.assertTrue(np.array_equal(original.sample(4).states, restored.sample(4).states))

    def test_prioritised_sampling_and_annealing_resume_deterministically(self):
        original = PrioritisedReplayBuffer(capacity=10, seed=7); self._fill(original)
        original.update_priorities(np.arange(8), np.arange(1, 9, dtype=float))
        restored = PrioritisedReplayBuffer(capacity=10, seed=999)
        restored.load_state_dict(original.state_dict())
        first, second = original.sample(4), restored.sample(4)
        self.assertTrue(np.array_equal(first.indices, second.indices))
        self.assertEqual(original.beta, restored.beta)


class ComputePilotTests(unittest.TestCase):
    def test_resource_record_is_explicitly_bounded_and_labelled(self):
        record = measure_callable(lambda: sum(range(20)), repetitions=3,
                                  wall_clock_timeout_s=1, label="synthetic_test")
        self.assertEqual(len(record["latency_ns"]), 3)
        self.assertIn("no full-scale extrapolation", record["scope"])
        self.assertGreater(record["peak_python_bytes"], 0)

    def test_timeout_is_reported(self):
        with self.assertRaises(TimeoutError):
            measure_callable(lambda: time.sleep(0.02), repetitions=3,
                             wall_clock_timeout_s=0.01, label="timeout_test")


@unittest.skipUnless(shutil.which("sumo"), "SUMO executable is unavailable")
class RealActuatedActivationTests(unittest.TestCase):
    def test_live_program_preflight_has_enforceable_wall_timeout(self):
        program = load_signal_program(TINY / "tiny.add.xml", "center", "static",
                                      TINY / "tiny.net.xml")
        with self.assertRaisesRegex(TimeoutError, "exceeded wall-clock timeout"):
            verify_loaded_signal_programs_bounded(
                TINY / "tiny.sumocfg", [program], "sumo", wall_clock_timeout_s=0.001)

    def test_sumo_uses_actuated_program(self):
        import traci
        import traci.constants as tc
        traci.start(["sumo", "-c", str(FIXTURE / "actuated.sumocfg"), "--no-step-log", "true"])
        try:
            logics = traci.trafficlight.getAllProgramLogics("center")
            selected = next(logic for logic in logics if logic.programID == "actuated_fixture")
            self.assertEqual(selected.type, tc.TRAFFICLIGHT_TYPE_ACTUATED)
            traci.trafficlight.setProgram("center", "actuated_fixture")
            self.assertEqual(traci.trafficlight.getProgram("center"), "actuated_fixture")
            self.assertEqual(traci.trafficlight.getNextSwitch("center"), 5.0)
            traci.simulationStep()
        finally:
            traci.close(True)

    def test_real_fixed_time_phase_schedule_fidelity(self):
        import traci
        traci.start(["sumo", "-c", str(TINY / "tiny.sumocfg"), "--no-step-log", "true"])
        try:
            phases = []
            for _ in range(26):
                traci.simulationStep()
                phases.append(traci.trafficlight.getPhase("center"))
            self.assertEqual(phases, [0] * 10 + [1] * 3 + [2] * 10 + [3] * 3)
        finally:
            traci.close(True)

    def test_real_sumo_compute_pilot_is_bounded_and_explicit(self):
        record = measure_traci_fixture(
            ["sumo", "-c", str(FIXTURE / "actuated.sumocfg"), "--no-step-log", "true"],
            maximum_steps=3, wall_clock_timeout_s=10)
        self.assertEqual(record["simulation_steps"], 3)
        self.assertEqual(len(record["traci_step_roundtrip_latency_ns"]), 3)
        self.assertIn("TraCI IPC", record["traci_method"])

    def test_actuated_native_runner_emits_xml_and_provenance_without_ai(self):
        cfg = {"simulation": {"sumo_binary": "sumo", "step_length": 1.0},
               "profiles": {"tiny": {"config": str(FIXTURE / "actuated.sumocfg"),
                                      "begin": 0, "end": 40}}}
        with tempfile.TemporaryDirectory() as directory:
            metrics, manifest = run_native_baseline_episode(
                cfg=cfg, profile="tiny", output_dir=Path(directory), run_index=0,
                seed=17, config_path=FIXTURE / "actuated.sumocfg",
                controller_type="sumo_actuated",
                tls_program_path=FIXTURE / "actuated.add.xml", tls_ids=["center"],
                wall_clock_timeout_s=15)
            self.assertTrue(metrics.valid)
            self.assertEqual(manifest.controller, "sumo_actuated")
            self.assertEqual(manifest.model_mode, "not_applicable")
            self.assertIsNone(manifest.checkpoint_manifest)
            self.assertEqual(manifest.completion_status, "complete")
            self.assertEqual(set(manifest.output_hashes), {"tripinfo", "emission", "summary"})
            program = manifest.controller_diagnostics["phase_programs"][0]
            observed = manifest.controller_diagnostics["live_program_preflight"][0]
            self.assertEqual(program["program_sha256"], observed["observed_program_sha256"])

    def test_native_runner_rejects_unloaded_and_differently_selected_programs(self):
        cfg = {"simulation": {"sumo_binary": "sumo", "step_length": 1.0},
               "profiles": {"tiny": {"config": str(FIXTURE / "actuated.sumocfg"),
                                      "begin": 0, "end": 40}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); outside = root / "outside.add.xml"
            outside.write_bytes((FIXTURE / "actuated.add.xml").read_bytes())
            for index, (kind, source, message) in enumerate((
                ("sumo_actuated", outside, "not loaded"),
                ("fixed_time", TINY / "tiny.net.xml", "selected TLS program"),
            )):
                output = root / f"run{index}"
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    run_native_baseline_episode(
                        cfg=cfg, profile="tiny", output_dir=output, run_index=index,
                        seed=17, config_path=FIXTURE / "actuated.sumocfg",
                        controller_type=kind, tls_program_path=source,
                        tls_ids=["center"], wall_clock_timeout_s=15)
                manifest_path = output / f"manifest_{kind}_tiny_seed17_run{index}.json"
                self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))
                                 ["completion_status"], "failed")

    def test_max_pressure_uses_real_sumo_env_and_target_is_only_a_request(self):
        with (TINY / "tiny_eval.yaml").open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        env = SumoEnv(cfg, profile="tiny", port=_free_port())
        definition = load_max_pressure_definition(
            FIXTURE / "max_pressure.json", TINY / "tiny.net.xml", "center")
        provider = MaxPressureActionProvider({"center": MaxPressureController(definition)})
        try:
            observations = env.reset()
            def east_west_pressure(lane):
                return 20 if lane in {"e_in_0", "w_in_0"} else 0
            with patch("traci.lane.getLastStepHaltingNumber", side_effect=east_west_pressure):
                first = provider.actions(env, observations)
            self.assertEqual(provider.last_selected_targets, {"center": 2})
            observations, _, _, first_info = env.step(first)
            self.assertFalse(first_info["action_decisions"]["center"]["accepted"])
            self.assertEqual(env.phase_controller_states()["center"].current_green_phase, 0)
            with patch("traci.lane.getLastStepHaltingNumber", side_effect=east_west_pressure):
                second = provider.actions(env, observations)
            _, _, _, second_info = env.step(second)
            self.assertTrue(second_info["action_decisions"]["center"]["accepted"])
            self.assertEqual(env.phase_controller_states()["center"].current_green_phase, 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
