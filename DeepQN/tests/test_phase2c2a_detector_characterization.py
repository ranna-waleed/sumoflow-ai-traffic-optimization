from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from DeepQN.forecasting.inference import ForecastInferenceError, StrictForecastAdapter


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    all(shutil.which(tool) for tool in ("sumo", "netgenerate", "netconvert")),
    "SUMO fixture tools are unavailable",
)
class RealE1EdgeCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "characterization"
        completed = subprocess.run(
            [sys.executable, "-m", "DeepQN.forecasting.characterize_e1_edge_cases",
             str(cls.output)], cwd=ROOT, capture_output=True, text=True, timeout=45,
        )
        if completed.returncode:
            raise AssertionError(completed.stderr or completed.stdout)
        cls.payload = json.loads((cls.output / "result.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_lane_change_is_one_destination_lane_entry_not_two_arrivals(self):
        case = self.payload["scenarios"]["lane_change"]
        self.assertEqual(case["commands"][0]["command"], "changeLane")
        self.assertEqual([(row["detector_id"], row["vehicle_id"])
                          for row in case["events"]], [("e1_1", "probe")])
        lanes = {row["lane_id"] for row in case["trajectory"]
                 if 14 <= row["lane_position_m"] <= 21 and row["lane_id"].startswith("B0C0")}
        self.assertEqual(lanes, {"B0C0_0", "B0C0_1"})

    def test_controlled_move_across_plane_is_not_a_physical_e1_entry(self):
        case = self.payload["scenarios"]["controlled_move_to"]
        command = case["commands"][0]
        self.assertLess(command["from_position_m"], 20)
        self.assertGreater(command["to_position_m"], 20)
        self.assertEqual(case["events"], [])
        # TraCI moveTo is an external discontinuity and is not reported by
        # SUMO's ordinary teleport lists. Collection must prohibit/log it.
        self.assertEqual(case["sumo_teleports"], [])

    def test_real_cyclic_route_preserves_two_distinct_crossings(self):
        case = self.payload["scenarios"]["cyclic_reentry"]
        events = case["events"]
        self.assertEqual(len(events), 2)
        self.assertEqual({row["vehicle_id"] for row in events}, {"probe"})
        self.assertLess(events[0]["entry_time_s"], events[1]["entry_time_s"])
        crossings = []
        prior = None
        for row in case["trajectory"]:
            if row["lane_id"] == "A0B0_0" and row["lane_position_m"] >= 20 and (
                    prior is None or prior["lane_id"] != "A0B0_0"
                    or prior["lane_position_m"] < 20):
                crossings.append(row["time_s"])
            prior = row
        self.assertEqual(len(crossings), 2)
        for event, crossing in zip(events, crossings):
            self.assertLessEqual(abs(event["entry_time_s"] - crossing), .11)

    def test_short_edges_are_crossed_between_one_second_snapshots(self):
        for name in ("short_edge", "extreme_short_edge"):
            with self.subTest(name=name):
                case = self.payload["scenarios"][name]
                self.assertEqual(len(case["events"]), 1)
                trajectory = case["trajectory"]
                integer_lanes = {row["lane_id"] for row in trajectory
                    if abs(row["time_s"] - round(row["time_s"])) < 1e-9}
                self.assertNotIn("e1_0", integer_lanes)
                fine_lanes = {row["lane_id"] for row in trajectory}
                self.assertIn("e1_0", fine_lanes)
                # The physical detector plane is global x=25.1 m. On the
                # 0.20 m edge the vehicle front can advance from the edge
                # start to e2 between adjacent 0.1 s samples, so lane ID alone
                # is not a sufficient trajectory reference.
                first_fine_sample = next(row["time_s"] for row in trajectory
                                         if row["x"] >= 25.1)
                self.assertLessEqual(abs(case["events"][0]["entry_time_s"]
                                         - first_fine_sample), .11)

    def test_every_effective_input_and_reference_is_hash_bound(self):
        for case in self.payload["scenarios"].values():
            self.assertEqual(set(case["files"]), {
                "network.net.xml", "demand.rou.xml", "detectors.add.xml",
                "scenario.sumocfg", "trajectory.xml",
            })
            self.assertTrue(all(len(value) == 64 for value in case["files"].values()))


class LiveTeleportSafetyTests(unittest.TestCase):
    def test_verified_adapter_rejects_standard_sumo_teleport_before_e1_collection(self):
        adapter = object.__new__(StrictForecastAdapter)
        adapter._collector = object()
        with patch("traci.simulation.getStartingTeleportIDList", return_value=("v",)), \
             patch("traci.simulation.getEndingTeleportIDList", return_value=()):
            with self.assertRaisesRegex(ForecastInferenceError, "teleport intersects"):
                adapter.tick()


if __name__ == "__main__":
    unittest.main()
