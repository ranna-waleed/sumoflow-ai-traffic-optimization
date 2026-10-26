from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from DeepQN.agent.dqn_agent import CheckpointValidationError, DQNAgent, MultiAgentDQN
from DeepQN.agent.replay_buffer import Batch, PrioritisedReplayBuffer, UniformReplayBuffer
from DeepQN.env.reward import IntervalRewardFunction, RewardConfig
from DeepQN.env.sumo_env import SumoEnv, _PhaseController
from DeepQN.forecasting.contracts import FEATURE_NAMES, FEATURE_UNITS
from DeepQN.forecasting.inference import (
    ForecastIdentity, ForecastInferenceError, NoForecastAdapter,
    ResearchBiLSTM, StrictForecastAdapter,
)
from DeepQN.forecasting.contracts import load_approach_mapping
from DeepQN.forecasting.detector_events import build_detector_mapping
from DeepQN.training.reproducibility import ResearchRunContract, seed_all


def scaler():
    return {"features": list(FEATURE_NAMES), "units": list(FEATURE_UNITS),
            "minimum": [0] * 7, "maximum": [10] * 7}


class StrictForecastTests(unittest.TestCase):
    def _artifact_manifest(self, root: Path, model_source: Path | None = None):
        project = Path(__file__).resolve().parents[2]
        mapping_src = project / "DeepQN/tests/fixtures/detector_sumo/mapping.json"
        network_src = project / "DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml"
        mapping = root / "mapping.json"; network = root / "network.net.xml"
        shutil.copyfile(mapping_src, mapping); shutil.copyfile(network_src, network)
        approach = load_approach_mapping(mapping, network)
        detectors = build_detector_mapping(approach, network)
        scaler_path = root / "scaler.json"
        scaler_path.write_text(json.dumps(scaler()), encoding="utf-8")
        model_path = root / "model.pt"
        if model_source is None:
            torch.save(ResearchBiLSTM(4, 1, 0).state_dict(), model_path)
        else:
            shutil.copyfile(model_source, model_path)
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps({"mapping_sha256": approach.sha256,
            "detector_mapping_sha256": detectors.sha256,
            "network_sha256": approach.network_sha256}), encoding="utf-8")
        def item(path):
            return {"path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {
            "inference_contract_version": "junction-bilstm-60s-to-30s-v1",
            "feature_schema_version": "junction-approach-arrivals-v1",
            "event_schema_version": "sumo-e1-approach-entry-v1",
            "features": list(FEATURE_NAMES), "units": list(FEATURE_UNITS),
            "history_seconds": 60, "target_seconds": 30, "waiting_time_memory_s": 100,
            "mapping_sha256": approach.sha256,
            "detector_mapping_sha256": detectors.sha256,
            "network_sha256": approach.network_sha256,
            "dqn_state_forecast_scale": [10, 10, 10, 10],
            "architecture": {"input_size": 7, "output_size": 4, "hidden_size": 4,
                             "num_layers": 1, "dropout": 0, "bidirectional": True},
            "artifacts": {"model": item(model_path), "scaler": item(scaler_path),
                          "dataset_manifest": item(dataset), "mapping": item(mapping),
                          "network": item(network)},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, manifest

    def test_valid_causal_history_and_explicit_no_forecast(self):
        identity = ForecastIdentity(*(["a" * 64] * 6))
        adapter = StrictForecastAdapter(["tls"], scaler(), identity,
                                        lambda values: np.array([1, 2, 3, 4]), [10] * 4)
        for time_s in range(60):
            adapter.observe("tls", float(time_s), [1, 2, 3, 4, 10, 5, 2])
        np.testing.assert_allclose(adapter.predict_for("tls"), [.1, .2, .3, .4])
        ablation = NoForecastAdapter(["tls"])
        self.assertEqual(ablation.mode, "no_forecast")
        self.assertEqual(ablation.predict_for("tls").tolist(), [0, 0, 0, 0])

    def test_schema_scaler_history_mapping_and_prediction_fail_closed(self):
        identity = ForecastIdentity(*(["b" * 64] * 6))
        for changed in (
            {**scaler(), "features": list(reversed(FEATURE_NAMES))},
            {**scaler(), "units": ["wrong"] * 7},
        ):
            with self.assertRaises(ForecastInferenceError):
                StrictForecastAdapter(["tls"], changed, identity, lambda _: np.zeros(4), [1] * 4)
        adapter = StrictForecastAdapter(["tls"], scaler(), identity, lambda _: np.zeros(4), [1] * 4)
        with self.assertRaisesRegex(ForecastInferenceError, "exactly 60"):
            adapter.predict_for("tls")
        with self.assertRaisesRegex(ForecastInferenceError, "total"):
            adapter.observe("tls", 0, [1, 1, 1, 1, 3, 0, 0])
        with self.assertRaisesRegex(ForecastInferenceError, "unknown TLS"):
            adapter.observe("other", 0, [0] * 7)

    def test_missing_and_historical_six_input_checkpoint_are_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"inference_contract_version": "junction-bilstm-60s-to-30s-v1"}
            path = root / "manifest.json"; path.write_text(json.dumps(manifest))
            with self.assertRaises(ForecastInferenceError):
                StrictForecastAdapter.from_manifest(path, ["tls"])
        historical = Path(__file__).resolve().parents[2] / "lstm/models/lstm_best.pth"
        if historical.is_file():
            state = torch.load(historical, map_location="cpu", weights_only=True)
            self.assertEqual(state["lstm.weight_ih_l0"].shape[1], 6)
            with tempfile.TemporaryDirectory() as tmp:
                manifest_path, _ = self._artifact_manifest(Path(tmp), historical)
                with self.assertRaisesRegex(ForecastInferenceError, "checkpoint is incompatible"):
                    StrictForecastAdapter.from_manifest(manifest_path, ["center"])

    def test_artifact_mapping_hash_binding_and_valid_strict_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest_path, manifest = self._artifact_manifest(root)
            adapter = StrictForecastAdapter.from_manifest(manifest_path, ["center"])
            self.assertEqual(adapter.mode, "verified_model")
            manifest["mapping_sha256"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ForecastInferenceError, "mapping_sha256"):
                StrictForecastAdapter.from_manifest(manifest_path, ["center"])

    def test_runtime_detector_installation_is_exact_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest_path, _ = self._artifact_manifest(root)
            adapter = StrictForecastAdapter.from_manifest(manifest_path, ["center"])
            specs = adapter.detector_mapping.specs
            lanes = {spec.detector_id: spec.lane_id for spec in specs}
            positions = {spec.detector_id: spec.position_m for spec in specs}
            loops = SimpleNamespace(
                getIDList=lambda: list(lanes),
                getLaneID=lambda detector_id: lanes[detector_id],
                getPosition=lambda detector_id: positions[detector_id],
            )
            result = adapter.validate_runtime_installation(
                SimpleNamespace(inductionloop=loops), root / "network.net.xml")
            self.assertTrue(result["verified"])
            missing = SimpleNamespace(inductionloop=SimpleNamespace(
                getIDList=lambda: list(lanes)[1:], getLaneID=loops.getLaneID,
                getPosition=loops.getPosition))
            with self.assertRaisesRegex(ForecastInferenceError, "not installed"):
                adapter.validate_runtime_installation(missing, root / "network.net.xml")
            wrong_lane = SimpleNamespace(inductionloop=SimpleNamespace(
                getIDList=loops.getIDList, getLaneID=lambda _detector_id: "wrong",
                getPosition=loops.getPosition))
            with self.assertRaisesRegex(ForecastInferenceError, "contradicts"):
                adapter.validate_runtime_installation(wrong_lane, root / "network.net.xml")
            (root / "network.net.xml").write_text("altered", encoding="utf-8")
            with self.assertRaisesRegex(ForecastInferenceError, "network hash"):
                adapter.validate_runtime_installation(
                    SimpleNamespace(inductionloop=loops), root / "network.net.xml")


class StrictForecastStartupTests(unittest.TestCase):
    class _Adapter:
        mode = "verified_model"
        def __init__(self, ready_after=60):
            self.ticks = 0; self.ready_after = ready_after
        def reset(self): self.ticks = 0
        def tick(self): self.ticks += 1
        def history_ready(self): return self.ticks == self.ready_after
        def validate_runtime_installation(self, _connection, _network):
            return {"verified": True}
        def predict_for(self, _tls):
            if not self.history_ready():
                raise ForecastInferenceError("verified forecast requires exactly 60 causal samples")
            return np.zeros(4, dtype=np.float32)

    def _env(self, adapter, warmup=60):
        env = object.__new__(SumoEnv)
        env._connected = False; env._step_count = 0; env._episode_done = False
        env.research_evaluation = True
        env.physical_trace_mode = "research_fail_closed"
        env.lane_observation_mode = "research_fail_closed"
        env._forecast_mode = "verified_model"; env._bilstm = adapter
        env._warmup_seconds = float(warmup)
        env._initial_control_protocol = "phase_controller_keep_v1" if warmup else None
        env._phase_controllers = {"tls": _PhaseController("tls", {
            "green_phases": [0], "min_green": 0, "max_green": 1000})}
        env.tls_ids = ["tls"]; env.sim_begin = 0; env.sim_end = 120
        env.net_file = "network.net.xml"; env._waiting_time_memory_s = 100
        env._reward_fn = SimpleNamespace(reset=lambda: None)
        env._launch_sumo = lambda: None
        env._get_all_states = lambda: {"tls": adapter.predict_for("tls")}
        return env

    def _traci(self):
        clock = SimpleNamespace(time=0.0)
        return clock, SimpleNamespace(
            TraCIException=RuntimeError,
            simulation=SimpleNamespace(getTime=lambda: clock.time,
                getArrivedNumber=lambda: 0),
            simulationStep=lambda: setattr(clock, "time", clock.time + 1.0),
            trafficlight=SimpleNamespace(setPhase=lambda *_: None,
                getPhase=lambda _tls: 0, getRedYellowGreenState=lambda _tls: "Gr"),
            close=lambda **_kwargs: None,
        )

    def test_verified_reset_rejects_unprimed_history_and_closes(self):
        adapter = self._Adapter(); env = self._env(adapter, warmup=0)
        _clock, fake = self._traci()
        with patch("DeepQN.env.sumo_env.traci", fake), self.assertRaisesRegex(
                RuntimeError, "did not produce 60"):
            env.reset()
        self.assertFalse(env._connected)

    def test_verified_reset_primes_real_sixty_second_history_before_first_decision(self):
        adapter = self._Adapter(); env = self._env(adapter)
        clock, fake = self._traci()
        with patch("DeepQN.env.sumo_env.traci", fake):
            observation = env.reset()
        self.assertEqual(adapter.ticks, 60)
        self.assertEqual(clock.time, 60.0)
        self.assertEqual(env._warmup_diagnostics["first_forecast_decision_s"], 60.0)
        self.assertEqual(observation["tls"].tolist(), [0, 0, 0, 0])

    def test_forecast_and_no_forecast_use_equal_initial_control_timing(self):
        results = []
        for adapter in (self._Adapter(), NoForecastAdapter(["tls"])):
            env = self._env(adapter); env._bilstm = adapter
            clock, fake = self._traci()
            with patch("DeepQN.env.sumo_env.traci", fake):
                adapter.reset(); results.append(env._run_initial_control_warmup())
            self.assertEqual(clock.time, 60.0)
        self.assertEqual(results[0], results[1])

    def _config(self):
        return {
            "simulation": {
                "research_evaluation": True, "physical_trace_mode": "research_fail_closed",
                "lane_observation_mode": "research_fail_closed",
                "comparative_research": False,
                "waiting_time_memory_s": 100, "step_length": 1,
                "decision_interval": 10, "sumo_binary": "sumo",
                "net_file": "network.net.xml",
            },
            "profiles": {"tiny": {"config": "tiny.sumocfg", "begin": 0, "end": 120}},
            "tls_junctions": {},
            "state": {"state_dim": 37, "max_lanes": 10, "features_per_lane": 3,
                "observation_contract": "sumoflow-dqn-observation-v2-occupancy-percent"},
            "bilstm": {"mode": "no_forecast", "history_warmup_seconds": 0},
            "reward": {},
        }

    def test_research_mode_rejects_implicit_legacy_and_waiting_memory_mismatch(self):
        config = self._config(); config["simulation"].pop("comparative_research")
        with self.assertRaisesRegex(ValueError, "explicit boolean.*comparative_research"):
            SumoEnv(config, profile="tiny", port=0)
        config = self._config(); config["state"].pop("observation_contract")
        with self.assertRaisesRegex(ValueError, "observation_contract"):
            SumoEnv(config, profile="tiny", port=0)
        config = self._config(); config["bilstm"].pop("mode")
        with self.assertRaisesRegex(ValueError, "explicit bilstm.mode"):
            SumoEnv(config, profile="tiny", port=0)
        config = self._config(); config["bilstm"]["mode"] = "legacy_adapter"
        with self.assertRaisesRegex(ValueError, "cannot use legacy_adapter"):
            SumoEnv(config, profile="tiny", port=0)
        for value in (None, 99):
            config = self._config()
            if value is None: config["simulation"].pop("waiting_time_memory_s")
            else: config["simulation"]["waiting_time_memory_s"] = value
            with self.assertRaisesRegex(ValueError, "waiting_time_memory_s=100"):
                SumoEnv(config, profile="tiny", port=0)
        config = self._config(); config["simulation"].pop("physical_trace_mode")
        with self.assertRaisesRegex(ValueError, "explicit physical_trace_mode"):
            SumoEnv(config, profile="tiny", port=0)
        config = self._config(); config["simulation"]["physical_trace_mode"] = "legacy_estimate"
        with self.assertRaisesRegex(ValueError, "requires physical_trace_mode=research_fail_closed"):
            SumoEnv(config, profile="tiny", port=0)
        config = self._config(); config["simulation"]["lane_observation_mode"] = "legacy_zero_fill"
        with self.assertRaisesRegex(ValueError, "lane_observation_mode=research_fail_closed"):
            SumoEnv(config, profile="tiny", port=0)

    def test_research_dqn_conditions_require_equal_sixty_second_protocol(self):
        config = self._config()
        config["simulation"]["comparative_research"] = True
        for value in (None, False):
            candidate = json.loads(json.dumps(config))
            if value is None:
                candidate["simulation"].pop("require_matched_forecast_warmup", None)
            else:
                candidate["simulation"]["require_matched_forecast_warmup"] = value
            with self.assertRaisesRegex(ValueError, "requires require_matched"):
                SumoEnv(candidate, profile="tiny", port=0)
        config["simulation"]["require_matched_forecast_warmup"] = True
        with self.assertRaisesRegex(ValueError, "shared 60-second"):
            SumoEnv(config, profile="tiny", port=0)
        config["bilstm"].update({
            "history_warmup_seconds": 60,
            "initial_control_protocol": "phase_controller_keep_v1",
        })
        env = SumoEnv(config, profile="tiny", port=0)
        self.assertEqual(env._warmup_seconds, 60)

    def test_research_launch_explicitly_passes_waiting_memory(self):
        env = SumoEnv(self._config(), profile="tiny", port=0)
        with patch("DeepQN.env.sumo_env.traci.start") as start:
            env._launch_sumo()
        command = start.call_args.args[0]
        index = command.index("--waiting-time-memory")
        self.assertEqual(command[index + 1], "100")

    def test_repeated_reset_clears_red_yellow_and_forced_controller_state(self):
        adapter = self._Adapter(); env = self._env(adapter)
        clock, fake = self._traci()
        with patch("DeepQN.env.sumo_env.traci", fake):
            env.reset()
            ctrl = env._phase_controllers["tls"]
            for dirty_state in (
                {"in_red_hold": True, "forced_switch": False, "in_yellow": False},
                {"in_red_hold": False, "forced_switch": True, "in_yellow": True},
            ):
                ctrl.in_red_hold = dirty_state["in_red_hold"]
                ctrl.forced_switch = dirty_state["forced_switch"]
                ctrl.in_yellow = dirty_state["in_yellow"]
                ctrl.yellow_elapsed = 2; ctrl.phase_elapsed_steps = 17
                env._connected = False; clock.time = 0.0
                env.reset()
                self.assertFalse(ctrl.in_red_hold)
                self.assertFalse(ctrl.forced_switch)
                self.assertFalse(ctrl.in_yellow)
                self.assertEqual(ctrl.yellow_elapsed, 0)
                self.assertEqual(env._initial_tls_states["tls"]["phase"], 0)
                self.assertEqual(env._initial_tls_states["tls"]["state"], "Gr")

    def test_research_reset_fails_when_initial_physical_phase_cannot_be_confirmed(self):
        adapter = self._Adapter(); env = self._env(adapter)
        _clock, fake = self._traci()
        fake.trafficlight.getPhase = lambda _tls: 2
        with patch("DeepQN.env.sumo_env.traci", fake), self.assertRaisesRegex(
                RuntimeError, "failed to initialize and confirm physical TLS tls"):
            env.reset()
        self.assertFalse(env._connected)


class ResearchLaneObservationTests(unittest.TestCase):
    def _env(self, mode="research_fail_closed"):
        env = object.__new__(SumoEnv)
        env.lane_observation_mode = mode
        env._lanes = {"tls": ["lane-a"]}
        env._phase_controllers = {"tls": _PhaseController("tls", {
            "green_phases": [0], "min_green": 0, "max_green": 100})}
        env.max_lanes = 1; env.fperlane = 3; env.max_queue = 10
        env.max_wait = 20; env.max_elapsed = 100; env.state_dim = 10
        return env

    def test_each_lane_measurement_fails_closed_with_exact_identity_after_warmup(self):
        global_features = np.zeros(5, dtype=np.float32)
        for measurement, method in (
            ("halting_number", "getLastStepHaltingNumber"),
            ("waiting_time", "getWaitingTime"),
            ("occupancy", "getLastStepOccupancy"),
        ):
            with self.subTest(measurement=measurement):
                methods = {
                    "getLastStepHaltingNumber": lambda _lane: 1,
                    "getWaitingTime": lambda _lane: 2,
                    "getLastStepOccupancy": lambda _lane: .3,
                }
                methods[method] = lambda _lane: (_ for _ in ()).throw(RuntimeError("sensor down"))
                fake = SimpleNamespace(lane=SimpleNamespace(**methods))
                env = self._env(); env._warmup_diagnostics = {
                    "duration_s": 60, "first_forecast_decision_s": 60,
                }
                with patch("DeepQN.env.sumo_env.traci", fake), self.assertRaisesRegex(
                        RuntimeError,
                        f"TLS=tls, lane=lane-a, measurement={measurement}"):
                    env._junction_state("tls", global_features)

    def test_legacy_lane_zero_substitution_requires_labeled_mode(self):
        env = self._env("legacy_zero_fill")
        failing = lambda _lane: (_ for _ in ()).throw(RuntimeError("offline"))
        fake = SimpleNamespace(lane=SimpleNamespace(
            getLastStepHaltingNumber=failing, getWaitingTime=failing,
            getLastStepOccupancy=failing))
        with patch("DeepQN.env.sumo_env.traci", fake):
            state = env._junction_state("tls", np.zeros(5, dtype=np.float32))
        self.assertEqual(state[:3].tolist(), [0, 0, 0])

    def test_raw_occupancy_percent_is_normalized_to_fraction(self):
        for raw, expected in (
            (0, 0), (.25, .0025), (1, .01), (5, .05),
            (30, .30), (80, .80), (100, 1),
        ):
            with self.subTest(raw=raw):
                env = self._env()
                fake = SimpleNamespace(lane=SimpleNamespace(
                    getLastStepHaltingNumber=lambda _lane: 0,
                    getWaitingTime=lambda _lane: 0,
                    getLastStepOccupancy=lambda _lane: raw))
                with patch("DeepQN.env.sumo_env.traci", fake):
                    state = env._junction_state("tls", np.zeros(5, dtype=np.float32))
                self.assertAlmostEqual(float(state[2]), expected)

    def test_invalid_and_nonfinite_occupancy_are_rejected(self):
        for raw in (-.01, 100.01, float("nan"), float("inf")):
            with self.subTest(raw=raw):
                env = self._env()
                fake = SimpleNamespace(lane=SimpleNamespace(
                    getLastStepHaltingNumber=lambda _lane: 0,
                    getWaitingTime=lambda _lane: 0,
                    getLastStepOccupancy=lambda _lane: raw))
                with patch("DeepQN.env.sumo_env.traci", fake), self.assertRaisesRegex(
                        RuntimeError, "TLS=tls, lane=lane-a, measurement=occupancy"):
                    env._junction_state("tls", np.zeros(5, dtype=np.float32))

    def test_post_warmup_sensor_failure_prevents_a_valid_step_result(self):
        env = self._env(); env._connected = True; env.decision_int = 1
        env.terminate_on_empty = False; env.sim_end = 120; env._step_count = 60
        env.tls_ids = ["tls"]; env.profile = "tiny"; env.sim_begin = 0
        env.physical_trace_mode = "research_fail_closed"
        env._bilstm = NoForecastAdapter(["tls"])
        env._reward_fn = SimpleNamespace(compute=lambda: 0.0, get_metrics=lambda: {})
        clock = SimpleNamespace(time=60.0)
        fake = SimpleNamespace(
            simulation=SimpleNamespace(getTime=lambda: clock.time,
                getArrivedNumber=lambda: 0, getMinExpectedNumber=lambda: 1),
            simulationStep=lambda: setattr(clock, "time", clock.time + 1),
            trafficlight=SimpleNamespace(setPhase=lambda *_: None,
                getPhase=lambda _tls: 0, getRedYellowGreenState=lambda _tls: "Gr"),
            lane=SimpleNamespace(
                getLastStepHaltingNumber=lambda _lane: 0,
                getWaitingTime=lambda _lane: (_ for _ in ()).throw(RuntimeError("sensor down")),
                getLastStepOccupancy=lambda _lane: 0),
        )
        with patch("DeepQN.env.sumo_env.traci", fake), self.assertRaisesRegex(
                RuntimeError, "TLS=tls, lane=lane-a, measurement=waiting_time"):
            env.step({"tls": 0})


class ResearchActionDictionaryTests(unittest.TestCase):
    def _env(self):
        env = object.__new__(SumoEnv)
        env._connected = True; env.comparative_research = True
        env.tls_ids = ["a", "b"]; env.decision_int = 0
        env.terminate_on_empty = False; env.sim_end = 100; env._step_count = 0
        env.profile = "tiny"
        env._phase_controllers = {
            tid: _PhaseController(tid, {"green_phases": [0],
                "min_green": 0, "max_green": 100}) for tid in env.tls_ids
        }
        env._reward_fn = SimpleNamespace(compute=lambda: 0.0, get_metrics=lambda: {})
        env._get_all_states = lambda: {}
        return env

    def test_invalid_research_action_dictionaries_are_atomic(self):
        invalid_cases = (
            {"a": 0},
            {"a": 0, "b": 1, "extra": 0},
            {"a": 0, "unknown": 1},
            {"a": 0, "b": True},
            {"a": 0, "b": 2},
            {"a": 0, "b": -1},
            {"a": 0, "b": 1.0},
        )
        env = self._env()
        for actions in invalid_cases:
            with self.subTest(actions=actions):
                before = {
                    tid: (ctrl.current_green_phase, ctrl.phase_elapsed_steps,
                          ctrl.in_yellow, ctrl.forced_switch)
                    for tid, ctrl in env._phase_controllers.items()
                }
                with self.assertRaises(ValueError):
                    env.step(actions)
                after = {
                    tid: (ctrl.current_green_phase, ctrl.phase_elapsed_steps,
                          ctrl.in_yellow, ctrl.forced_switch)
                    for tid, ctrl in env._phase_controllers.items()
                }
                self.assertEqual(after, before)

    def test_complete_multi_tls_actions_are_accepted_and_remain_requests(self):
        env = self._env()
        fake = SimpleNamespace(simulation=SimpleNamespace(
            getTime=lambda: 0.0, getMinExpectedNumber=lambda: 1),
            trafficlight=SimpleNamespace(setPhase=lambda *_: None))
        with patch("DeepQN.env.sumo_env.traci", fake):
            _obs, _rewards, _done, info = env.step({"a": 0, "b": 1})
        self.assertEqual(info["requested_actions"], {"a": 0, "b": 1})
        self.assertEqual(info["action_decisions"]["a"]["reason"], "accepted_keep")
        self.assertEqual(info["action_decisions"]["b"]["reason"], "accepted_switch")


class IntervalRewardTests(unittest.TestCase):
    def _reward(self, step_length_rates=(100.0, 200.0), clip=(-100, 100)):
        config = RewardConfig(wait_time_weight=1, co2_weight=1, throughput_weight=1,
            pressure_weight=1, clip_min=clip[0], clip_max=clip[1],
            delay_reference_vehicle_s=10, co2_reference_mg=100,
            arrivals_reference_vehicles=1, pressure_reference_vehicle_s=10)
        reward = IntervalRewardFunction({"tls": ["a", "b"]}, ["edge"], config)
        rates = iter(step_length_rates)
        fake = SimpleNamespace(
            lane=SimpleNamespace(getLastStepHaltingNumber=lambda lane: 2 if lane == "a" else 0),
            edge=SimpleNamespace(getCO2Emission=lambda edge: next(rates)))
        return reward, fake

    def test_nonunit_timestep_arrivals_components_and_clipping(self):
        reward, fake = self._reward()
        with patch("DeepQN.env.reward.traci", fake):
            reward.observe_step(.5, 0); reward.observe_step(.5, 2)
        # delay=2 vehicle-s, CO2=150mg, arrivals=2, pressure=1 vehicle-s
        self.assertAlmostEqual(reward.compute(), .2)
        metrics = reward.get_metrics()
        self.assertEqual(metrics["interval_delay_vehicle_s"], 2)
        self.assertEqual(metrics["interval_co2_mg"], 150)
        self.assertEqual(metrics["interval_arrivals"], 2)
        clipped, fake = self._reward((1000,), clip=(-.5, .5))
        with patch("DeepQN.env.reward.traci", fake): clipped.observe_step(2, 0)
        self.assertEqual(clipped.compute(), -.5)

    def test_zero_traffic_and_invalid_timestep(self):
        config = RewardConfig(delay_reference_vehicle_s=1, co2_reference_mg=1,
                              arrivals_reference_vehicles=1, pressure_reference_vehicle_s=1)
        reward = IntervalRewardFunction({}, [], config)
        fake = SimpleNamespace(lane=SimpleNamespace(), edge=SimpleNamespace())
        with patch("DeepQN.env.reward.traci", fake): reward.observe_step(1, 0)
        self.assertEqual(reward.compute(), 0)
        with self.assertRaises(ValueError): reward.observe_step(0, 0)


class ActionReplayReproducibilityTests(unittest.TestCase):
    def test_deterministic_inference_action_reward_transition_path(self):
        def execute_once():
            clock = SimpleNamespace(time=0.0, phase=0)
            simulation = SimpleNamespace(
                getTime=lambda: clock.time, getArrivedNumber=lambda: 0,
                getMinExpectedNumber=lambda: 1)
            def step(): clock.time += 1.0
            trafficlight = SimpleNamespace(
                setPhase=lambda _tls, phase: setattr(clock, "phase", phase),
                getPhase=lambda _tls: clock.phase,
                getRedYellowGreenState=lambda _tls: "Gr" if clock.phase == 0 else "yr")
            fake = SimpleNamespace(simulation=simulation, simulationStep=step,
                trafficlight=trafficlight,
                lane=SimpleNamespace(getLastStepHaltingNumber=lambda _lane: 1),
                edge=SimpleNamespace(getCO2Emission=lambda _edge: 10.0))
            env = object.__new__(SumoEnv)
            env._connected = True; env.decision_int = 3; env.terminate_on_empty = False
            env.sim_end = 3; env._step_count = 0; env.tls_ids = ["tls"]; env.profile = "tiny"
            env.physical_trace_mode = "research_fail_closed"
            env._phase_controllers = {"tls": _PhaseController("tls", {
                "green_phases": [0, 2], "green_to_yellow": {"0": 1},
                "yellow_to_next": {"1": 2}, "yellow_duration": 1,
                "min_green": 0, "max_green": 10})}
            cfg = RewardConfig(delay_reference_vehicle_s=1, co2_reference_mg=1,
                arrivals_reference_vehicles=1, pressure_reference_vehicle_s=1,
                clip_min=-1000, clip_max=1000)
            env._reward_fn = IntervalRewardFunction({"tls": ["lane"]}, ["edge"], cfg)
            env._bilstm = NoForecastAdapter(["tls"])
            env._get_all_states = lambda: {"tls": np.zeros(37, dtype=np.float32)}
            with patch("DeepQN.env.sumo_env.traci", fake), patch("DeepQN.env.reward.traci", fake):
                next_obs, rewards, done, info = env.step({"tls": 1})
            replay = UniformReplayBuffer(2, seed=1)
            replay.push(np.zeros(37), 1, rewards["tls"], next_obs["tls"],
                        terminated=info["terminated"], truncated=info["truncated"],
                        interval_duration_s=info["interval_duration_s"])
            return rewards["tls"], done, info["action_trace_sha256"], replay.sample(1), info
        first = execute_once(); second = execute_once()
        self.assertEqual(first[:3], second[:3])
        self.assertTrue(first[1]); self.assertEqual(first[3].truncated.tolist(), [1.0])
        self.assertEqual(first[3].terminated.tolist(), [0.0])
        trace = first[4]["action_trace"]["tls"]
        self.assertEqual(trace[0]["executed_state"]["phase"], 1)
        self.assertEqual(trace[0]["after_controller_update"]["phase"], 2)
        self.assertEqual(trace[-1]["end_s"], 3.0)

    def test_physical_trace_failure_invalidates_research_step_without_hash(self):
        clock = SimpleNamespace(time=0.0)
        fake = SimpleNamespace(
            simulation=SimpleNamespace(getTime=lambda: clock.time,
                getArrivedNumber=lambda: 0, getMinExpectedNumber=lambda: 1),
            simulationStep=lambda: setattr(clock, "time", clock.time + 1),
            trafficlight=SimpleNamespace(
                getPhase=lambda _tid: (_ for _ in ()).throw(RuntimeError("query failed")),
                getRedYellowGreenState=lambda _tid: "Gr", setPhase=lambda *_: None))
        env = object.__new__(SumoEnv); env._connected = True; env.decision_int = 1
        env.terminate_on_empty = False; env.sim_end = 1; env._step_count = 0
        env.tls_ids = ["tls"]; env.profile = "tiny"; env.physical_trace_mode = "research_fail_closed"
        env._phase_controllers = {"tls": _PhaseController("tls", {
            "green_phases": [0], "min_green": 0, "max_green": 10})}
        env._reward_fn = SimpleNamespace(compute=lambda: 0, get_metrics=lambda: {})
        env._bilstm = NoForecastAdapter(["tls"]); env._get_all_states = lambda: {}
        with patch("DeepQN.env.sumo_env.traci", fake), self.assertRaisesRegex(
                RuntimeError, "physical TLS trace unavailable"):
            env.step({"tls": 0})

    def test_legacy_trace_fallback_is_explicitly_labeled(self):
        env = object.__new__(SumoEnv); env.physical_trace_mode = "legacy_estimate"
        ctrl = _PhaseController("tls", {"green_phases": [0]})
        fake = SimpleNamespace(trafficlight=SimpleNamespace(
            getPhase=lambda _tid: (_ for _ in ()).throw(RuntimeError("offline"))))
        with patch("DeepQN.env.sumo_env.traci", fake):
            sample = env._read_physical_tls_state("tls", ctrl)
        self.assertFalse(sample["available"])
        self.assertEqual(sample["source"], "legacy_controller_estimate")

    def test_requested_action_acceptance_deferral_and_physical_transition(self):
        ctrl = _PhaseController("tls", {"green_phases": [0, 2],
            "green_to_yellow": {"0": 1}, "yellow_to_next": {"1": 2},
            "yellow_duration": 2, "min_green": 3, "max_green": 5})
        fake = SimpleNamespace(trafficlight=SimpleNamespace(setPhase=lambda *_: None))
        with patch("DeepQN.env.sumo_env.traci", fake):
            deferred = ctrl.apply_action(1)
            self.assertFalse(deferred.accepted); self.assertEqual(deferred.reason, "deferred_min_green")
            for _ in range(3): ctrl.tick()
            accepted = ctrl.apply_action(1)
            self.assertTrue(accepted.accepted); self.assertTrue(ctrl.in_yellow)
            ctrl.tick(); ctrl.tick()
        self.assertEqual(ctrl.current_green_phase, 2)

    def test_red_hold_and_minimum_green_contracts(self):
        ctrl = _PhaseController("tls", {"green_phases": [0], "red_hold_phase": 2,
            "green_to_yellow": {"0": 1}, "yellow_to_next": {"1": 2},
            "red_to_green_phase": 0, "yellow_duration": 1,
            "min_green": 2, "max_green": 10})
        phases = []
        fake = SimpleNamespace(trafficlight=SimpleNamespace(
            setPhase=lambda _tls, phase: phases.append(phase)))
        with patch("DeepQN.env.sumo_env.traci", fake):
            self.assertEqual(ctrl.apply_action(1).reason, "deferred_min_green")
            ctrl.tick(); ctrl.tick()
            self.assertTrue(ctrl.apply_action(1).accepted)
            ctrl.tick()
            self.assertTrue(ctrl.in_red_hold); self.assertEqual(ctrl.current_green_phase, 2)
            ctrl.tick(); ctrl.tick()
            self.assertTrue(ctrl.apply_action(1).accepted)
        self.assertEqual(phases, [1, 2, 0])

    def test_replay_bootstraps_truncation_but_not_termination_and_is_deterministic(self):
        first = UniformReplayBuffer(10, seed=7); second = UniformReplayBuffer(10, seed=7)
        for value in range(4):
            kwargs = dict(terminated=value == 0, truncated=value == 1,
                          interval_duration_s=3 if value == 3 else 10)
            first.push(np.array([value]), 0, value, np.array([value + 1]), **kwargs)
            second.push(np.array([value]), 0, value, np.array([value + 1]), **kwargs)
        left, right = first.sample(3), second.sample(3)
        np.testing.assert_array_equal(left.states, right.states)
        self.assertTrue(np.any(left.truncated >= 0))
        with self.assertRaises(ValueError):
            first.push(np.zeros(1), 0, 0, np.zeros(1), valid=False)

        cfg = {"state_dim": 1, "action_dim": 2, "hidden_dims": [2], "gamma": .5,
               "min_replay_size": 99}
        agent = DQNAgent("tls", cfg, torch.device("cpu"))
        for parameter in (*agent.online_net.parameters(), *agent.target_net.parameters()):
            parameter.data.zero_()
        agent.target_net.value_stream[-1].bias.data.fill_(2)
        batch = Batch(np.zeros((2, 1)), np.zeros(2, dtype=np.int64), np.array([1, 1]),
                      np.zeros((2, 1)), np.array([1, 0], dtype=np.float32),
                      np.array([0, 1], dtype=np.float32), np.ones(2), np.ones(2), np.zeros(2))
        _, errors = agent._compute_loss(batch)
        self.assertEqual(errors.detach().tolist(), [1.0, 2.0])

    def test_seed_and_run_contract_are_explicit_and_nonoverwriting(self):
        self.assertEqual(seed_all(17)["torch"], 17)
        contract = ResearchRunContract(1, 2, "demand-2", "a" * 64, "b" * 64,
                                       None, "no_forecast", "c" * 40, True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            digest = contract.write_once(path)
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError): contract.write_once(path)

    def test_research_checkpoint_schema_is_complete_and_nonoverwriting(self):
        cfg = {"state_dim": 2, "action_dim": 2, "hidden_dims": [2],
               "min_replay_size": 99, "seed": 4}
        ensemble = MultiAgentDQN(["a", "b"], cfg)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = ensemble.save_research_ensemble(tmp, 3, "d" * 64)
            self.assertEqual({row.tls_id for row in manifest.checkpoints}, {"a", "b"})
            payload = torch.load(Path(tmp) / "a_ep0003.pt", map_location="cpu",
                                 weights_only=True)
            self.assertEqual(payload["checkpoint_schema"], "sumoflow-dqn-checkpoint-v2")
            for key in ("optimizer", "architecture_config_sha256", "run_manifest_sha256",
                        "python_rng_state", "numpy_rng_state", "torch_rng_state"):
                self.assertIn(key, payload)
            with self.assertRaises(FileExistsError):
                ensemble.save_research_ensemble(tmp, 3, "d" * 64)

    def test_research_checkpoint_manifest_load_and_hash_validation_are_atomic(self):
        cfg = {"state_dim": 2, "action_dim": 2, "hidden_dims": [2],
               "min_replay_size": 99, "seed": 4}
        source = MultiAgentDQN(["a", "b"], cfg)
        with tempfile.TemporaryDirectory() as tmp:
            source.save_research_ensemble(tmp, 3, "d" * 64)
            target = MultiAgentDQN(["a", "b"], cfg)
            loaded = target.load_for_research_evaluation(tmp, 3)
            self.assertEqual(loaded.schema_version, "sumoflow-dqn-ensemble-v2")
            before = {tid: {key: value.clone() for key, value in agent.online_net.state_dict().items()}
                      for tid, agent in target.agents.items()}
            checkpoint = Path(tmp) / "b_ep0003.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"altered")
            with self.assertRaisesRegex(CheckpointValidationError, "hash mismatch"):
                target.load_for_research_evaluation(tmp, 3)
            for tid, agent in target.agents.items():
                for key, value in agent.online_net.state_dict().items():
                    self.assertTrue(torch.equal(value, before[tid][key]))

    def test_research_loader_rejects_legacy_mixed_identity_and_run_manifest(self):
        cfg = {"state_dim": 2, "action_dim": 2, "hidden_dims": [2],
               "min_replay_size": 99, "seed": 4}
        for mutation in ("missing_schema", "wrong_tls", "wrong_run"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                ensemble = MultiAgentDQN(["a", "b"], cfg)
                ensemble.save_research_ensemble(tmp, 3, "d" * 64)
                checkpoint_path = Path(tmp) / "b_ep0003.pt"
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if mutation == "missing_schema": payload.pop("checkpoint_schema")
                elif mutation == "wrong_tls": payload["tls_id"] = "a"
                else: payload["run_manifest_sha256"] = "e" * 64
                torch.save(payload, checkpoint_path)
                manifest_path = Path(tmp) / "ensemble_ep0003.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for row in manifest["checkpoints"]:
                    if row["tls_id"] == "b":
                        row["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(CheckpointValidationError, "provenance"):
                    ensemble.load_for_research_evaluation(tmp, 3)

    def test_research_checkpoint_publish_failure_leaves_no_complete_ensemble(self):
        cfg = {"state_dim": 2, "action_dim": 2, "hidden_dims": [2],
               "min_replay_size": 99, "seed": 4}
        ensemble = MultiAgentDQN(["a", "b"], cfg)
        with tempfile.TemporaryDirectory() as tmp:
            original_replace = Path.replace; calls = {"count": 0}
            def fail_second(source, target):
                if source.suffix == ".pt":
                    calls["count"] += 1
                    if calls["count"] == 2:
                        raise OSError("injected publication failure")
                return original_replace(source, target)
            with patch.object(Path, "replace", autospec=True, side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    ensemble.save_research_ensemble(tmp, 3, "d" * 64)
            self.assertFalse((Path(tmp) / "ensemble_ep0003.json").exists())
            self.assertEqual(list(Path(tmp).glob("*_ep0003.pt")), [])
            with self.assertRaisesRegex(CheckpointValidationError, "manifest not found"):
                ensemble.load_for_research_evaluation(tmp, 3)

    def test_discount_mode_is_explicitly_decision_step_not_elapsed_time(self):
        cfg = {"state_dim": 1, "action_dim": 2, "hidden_dims": [2], "gamma": .5,
               "discount_mode": "decision_step_v1", "min_replay_size": 99}
        agent = DQNAgent("tls", cfg, torch.device("cpu"))
        for parameter in (*agent.online_net.parameters(), *agent.target_net.parameters()):
            parameter.data.zero_()
        agent.target_net.value_stream[-1].bias.data.fill_(2)
        batch = Batch(np.zeros((2, 1)), np.zeros(2, dtype=np.int64), np.ones(2),
            np.zeros((2, 1)), np.zeros(2), np.array([0, 1]),
            np.array([1, 10]), np.ones(2), np.zeros(2))
        _, errors = agent._compute_loss(batch)
        self.assertEqual(errors.detach().tolist(), [2.0, 2.0])
        with self.assertRaisesRegex(ValueError, "unsupported discount_mode"):
            DQNAgent("tls", {**cfg, "discount_mode": "elapsed_time"}, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
