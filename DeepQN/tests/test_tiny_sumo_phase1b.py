from __future__ import annotations

import socket
import json
import tempfile
import unittest
from pathlib import Path

from DeepQN.evaluation.tiny_sumo_validation import run_validation
from DeepQN.evaluation.tiny_sumo_validation import DEFAULT_CONFIG
from DeepQN.env.sumo_env import SumoEnv
import yaml


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TinySumoIntegrationTests(unittest.TestCase):
    def test_occupied_traci_port_is_rejected_before_startup(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0)); occupied.listen()
            port = occupied.getsockname()[1]
            env = SumoEnv(cfg, profile="tiny", port=port)
            with self.assertRaisesRegex(RuntimeError, "port .* unavailable"):
                env.reset()

    def test_fixed_time_and_random_pair_is_valid_and_comparable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_validation(Path(directory) / "outputs", seed=17, repeat=1, base_port=_free_port())
            record = result["records"][0]
            self.assertTrue(record["comparable"])
            self.assertEqual(record["random_manifest"]["controller"], "random")
            self.assertIsNone(record["random_manifest"]["checkpoint_manifest"])
            self.assertGreater(record["random_manifest"]["controller_diagnostics"]["decision_count"], 0)
            trace_path = Path(record["random_manifest"]["controller_diagnostics"]
                              ["physical_action_trace_path"])
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertTrue(trace["physical"])
            samples = [sample for decision in trace["physical"]
                       for rows in decision["physical_steps"].values() for sample in rows]
            self.assertTrue(samples)
            self.assertTrue(all(sample["start_boundary"]["source"] == "traci_physical_query"
                                and sample["end_boundary_before_controller"]["source"] == "traci_physical_query"
                                and sample["after_controller_update"]["source"] == "traci_physical_query"
                                for sample in samples))
            self.assertTrue(any(sample["executed_state"] is not None for sample in samples))
            for mode in ("fixed_time", "random"):
                self.assertTrue(record[mode]["valid"])
                self.assertTrue(record[mode]["reached_horizon"])
                self.assertEqual(record[mode]["successful_arrivals"], record[mode]["summary_arrived"])
            for manifest_name in ("fixed_manifest", "random_manifest"):
                self.assertEqual(set(record[manifest_name]["output_hashes"]), {"tripinfo", "emission", "summary"})


if __name__ == "__main__":
    unittest.main()
