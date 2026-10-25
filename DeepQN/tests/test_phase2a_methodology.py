from __future__ import annotations

import subprocess
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from DeepQN.env.reward import RewardConfig, RewardFunction
from DeepQN.env.sumo_env import SumoEnv
from DeepQN.evaluation.baseline import _run_sumo_command, _validate_wall_clock_timeout
from DeepQN.evaluation.evaluate import _fixed_horizon_config


class EvaluationPreflightTests(unittest.TestCase):
    def test_controlled_evaluation_forces_fixed_horizon_without_mutating_source(self):
        cfg = {
            "simulation": {"terminate_on_empty": True},
            "profiles": {"case": {"config": "original.sumocfg"}},
        }
        result = _fixed_horizon_config(cfg, "case", Path("temporary.sumocfg"))
        self.assertFalse(result["simulation"]["terminate_on_empty"])
        self.assertEqual(result["profiles"]["case"]["config"], "temporary.sumocfg")
        self.assertTrue(cfg["simulation"]["terminate_on_empty"])
        self.assertEqual(cfg["profiles"]["case"]["config"], "original.sumocfg")

    def test_sumo_env_training_default_remains_terminate_on_empty(self):
        cfg = {
            "simulation": {"net_file": "unused.net.xml"},
            "profiles": {"case": {"begin": 0, "end": 20}},
            "tls_junctions": {},
            "state": {},
        }
        env = SumoEnv(cfg, profile="case")
        self.assertTrue(env.terminate_on_empty)

    def test_nondivisible_horizon_stops_during_final_decision_interval(self):
        clock = SimpleNamespace(time=20.0, min_expected=1)

        def simulation_step():
            clock.time += 1.0

        simulation = SimpleNamespace(
            getTime=lambda: clock.time,
            getArrivedNumber=lambda: 0,
            getMinExpectedNumber=lambda: clock.min_expected,
        )
        fake_traci = SimpleNamespace(simulation=simulation, simulationStep=simulation_step)

        env = object.__new__(SumoEnv)
        env._connected = True
        env._phase_controllers = {}
        env.decision_int = 10
        env.terminate_on_empty = False
        env.sim_end = 23
        env._step_count = 20
        env.tls_ids = []
        env.profile = "case"
        env._bilstm = SimpleNamespace(tick=lambda: None)
        env._reward_fn = SimpleNamespace(compute=lambda: 0.0, get_metrics=lambda: {})
        env._get_all_states = lambda: {}

        with patch("DeepQN.env.sumo_env.traci", fake_traci):
            _, _, done, info = env.step({})

        self.assertTrue(done)
        self.assertEqual(info["sim_time"], 23.0)
        self.assertEqual(info["interval_steps"], 3)
        self.assertEqual(info["interval_duration_s"], 3.0)
        self.assertEqual(info["step_count"], 23)

    def test_fixed_horizon_does_not_stop_when_demand_empties(self):
        clock = SimpleNamespace(time=0.0)

        def simulation_step():
            clock.time += 1.0

        simulation = SimpleNamespace(
            getTime=lambda: clock.time,
            getArrivedNumber=lambda: 0,
            getMinExpectedNumber=lambda: 0,
        )
        fake_traci = SimpleNamespace(simulation=simulation, simulationStep=simulation_step)
        env = object.__new__(SumoEnv)
        env._connected = True
        env._phase_controllers = {}
        env.decision_int = 4
        env.terminate_on_empty = False
        env.sim_end = 4
        env._step_count = 0
        env.tls_ids = []
        env.profile = "case"
        env._bilstm = SimpleNamespace(tick=lambda: None)
        env._reward_fn = SimpleNamespace(compute=lambda: 0.0, get_metrics=lambda: {})
        env._get_all_states = lambda: {}

        with patch("DeepQN.env.sumo_env.traci", fake_traci):
            _, _, done, info = env.step({})

        self.assertTrue(done)
        self.assertEqual(info["interval_steps"], 4)
        self.assertEqual(info["sim_time"], 4.0)

    def test_wall_clock_timeout_is_independent_and_validated(self):
        self.assertIsNone(_validate_wall_clock_timeout(None))
        self.assertEqual(_validate_wall_clock_timeout(12), 12.0)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_wall_clock_timeout(invalid)

    @patch("DeepQN.evaluation.baseline.subprocess.run")
    def test_baseline_forwards_timeout_to_subprocess(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        _run_sumo_command(["sumo", "-c", "case.sumocfg"], 123.5)
        self.assertEqual(run.call_args.kwargs["timeout"], 123.5)

    @patch("DeepQN.evaluation.baseline.subprocess.run")
    def test_baseline_timeout_preserves_command_and_output(self, run):
        run.side_effect = subprocess.TimeoutExpired(
            cmd=["sumo"], timeout=2, stderr="startup diagnostics"
        )
        with self.assertRaisesRegex(RuntimeError, "startup diagnostics") as raised:
            _run_sumo_command(["sumo", "-c", "case.sumocfg"], 2.0)
        self.assertIn("case.sumocfg", str(raised.exception))


class RewardCharacterizationTests(unittest.TestCase):
    def test_current_reward_uses_endpoint_differences_not_interval_quantities(self):
        cfg = RewardConfig(
            wait_time_weight=1.0,
            co2_weight=0.001,
            throughput_weight=0.5,
            pressure_weight=0.0,
            clip_min=-1000.0,
            clip_max=1000.0,
        )
        reward = RewardFunction([], [], cfg)
        reward._prev_total_wait = 20.0
        reward._prev_total_co2 = 100.0
        reward._prev_arrived = 2
        reward._get_total_wait = Mock(return_value=5.0)
        reward._get_total_co2 = Mock(return_value=50.0)
        fake_traci = SimpleNamespace(
            simulation=SimpleNamespace(getArrivedNumber=lambda: 0)
        )

        with patch("DeepQN.env.reward.traci", fake_traci):
            actual = reward.compute()

        # Despite positive waiting and emissions during an interval, falling
        # endpoint snapshots contribute +15 and +0.05; resetting last-step
        # arrivals contributes -1. This characterizes, but does not fix, the
        # current reward semantics.
        self.assertAlmostEqual(actual, 14.05)

    def test_rate_integration_requires_actual_timestep_duration(self):
        rates = (100.0, 200.0)
        self.assertEqual(sum(0.5 * value for value in rates), 150.0)
        self.assertEqual(sum(1.0 * value for value in rates), 300.0)
        self.assertEqual(sum(2.0 * value for value in rates), 600.0)

    def test_historical_bilstm_checkpoint_shape_conflicts_with_declared_input(self):
        import torch

        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "lstm/models/config.json"
        checkpoint_path = project_root / "lstm/models/lstm_best.pth"
        if not config_path.is_file() or not checkpoint_path.is_file():
            self.skipTest("historical BiLSTM artifacts are unavailable")
        with config_path.open(encoding="utf-8") as stream:
            declared = json.load(stream)["input_size"]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_input = state["lstm.weight_ih_l0"].shape[1]
        self.assertEqual(declared, 7)
        self.assertEqual(checkpoint_input, 6)
        self.assertNotEqual(declared, checkpoint_input)


if __name__ == "__main__":
    unittest.main()
