from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import traci


FIXTURES = Path(__file__).parent / "fixtures"


@unittest.skipUnless(shutil.which("sumo"), "SUMO executable is unavailable")
class Sumo126OccupancyContractTests(unittest.TestCase):
    def tearDown(self):
        try:
            traci.close(False)
        except Exception:
            pass

    def test_lane_occupancy_is_fraction_and_e1_occupancy_is_percentage(self):
        tiny = (FIXTURES / "tiny_sumo" / "tiny.sumocfg").resolve()
        traci.start(["sumo", "-c", str(tiny), "--no-step-log", "true",
                     "--duration-log.disable", "true"])
        traci.simulationStep()
        lane_id = "n_in_0"
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        self.assertEqual(vehicle_ids, ("ns0",))
        expected_fraction = sum(
            traci.vehicle.getLength(vehicle_id) for vehicle_id in vehicle_ids
        ) / traci.lane.getLength(lane_id)
        self.assertAlmostEqual(
            traci.lane.getLastStepOccupancy(lane_id), expected_fraction, places=12
        )
        self.assertLess(traci.lane.getLastStepOccupancy(lane_id), 1.0)
        traci.close()

        detector = (FIXTURES / "detector_sumo" / "detector.sumocfg").resolve()
        traci.start(["sumo", "-c", str(detector), "--no-step-log", "true",
                     "--duration-log.disable", "true"])
        traci.simulationStep()
        self.assertEqual(
            traci.inductionloop.getLastStepOccupancy(
                "e1__center__south__n_in_0"
            ),
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
