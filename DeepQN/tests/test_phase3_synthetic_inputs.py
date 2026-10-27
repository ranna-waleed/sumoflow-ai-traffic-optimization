from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from DeepQN.env.sumo_env import SumoEnv, _PhaseController
from DeepQN.evaluation.demand_provenance import sha256_file
from DeepQN.evaluation.prepare_night_synthetic import (
    validate_explicit_route_integrity,
)
from DeepQN.evaluation.synthetic_probe import (
    _parse_summary_with_partial_evidence,
    run as run_synthetic_probe,
    _summary_horizon_coverage,
    _warning_lines,
)
from DeepQN.training.diagnostic_train import (
    _collect_sumo_telemetry,
    _merge,
    _save_diagnostic_checkpoints,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS_V4 = ROOT / "DeepQN/diagnostics/inputs/night_synthetic_v4"
INPUTS_V5 = ROOT / "DeepQN/diagnostics/inputs/night_synthetic_v5"
INPUTS = ROOT / "DeepQN/diagnostics/inputs/night_synthetic_v6"


class SyntheticInputContractTests(unittest.TestCase):
    def _telemetry(self, error_text: str, message_text: str,
                   diagnostic_text: str = "") -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / name for name in (
                "errors.log", "messages.log", "diagnostic.log")}
            paths["errors.log"].write_text(error_text, encoding="utf-8")
            paths["messages.log"].write_text(message_text, encoding="utf-8")
            paths["diagnostic.log"].write_text(
                diagnostic_text, encoding="utf-8")
            return _collect_sumo_telemetry(
                error_log=paths["errors.log"],
                message_log=paths["messages.log"],
                diagnostic_log=paths["diagnostic.log"])

    def test_diagnostic_detects_error_log_warning_without_double_counting(self):
        warning = ("Warning: Vehicle 'v' performs emergency braking on lane "
                   "':tls_0_0' at time=10.00")
        result = self._telemetry(
            warning, "Emergency Braking: 1\n", warning + "\n")
        self.assertTrue(result["valid"])
        self.assertEqual(result["categories"]["emergency_braking"], 1)
        self.assertEqual(len(result["adverse_events"]), 1)
        self.assertEqual(set(result["adverse_events"][0]["sources"]),
                         {"error_log", "diagnostic_log"})

    def test_diagnostic_detects_message_summary_emergency_without_warning(self):
        result = self._telemetry("", "Emergency Braking: 1\n")
        self.assertTrue(result["valid"])
        self.assertEqual(result["categories"]["emergency_braking"], 1)
        self.assertFalse(result["adverse_events"])

    def test_diagnostic_accepts_complete_zero_warning_telemetry(self):
        result = self._telemetry("", "Emergency Braking: 0\n")
        self.assertTrue(result["valid"])
        self.assertEqual(result["categories"]["emergency_braking"], 0)
        self.assertFalse(result["adverse_events"])

    def test_diagnostic_rejects_missing_and_contradictory_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            message = root / "messages.log"
            diagnostic = root / "diagnostic.log"
            message.write_text("Emergency Braking: 0\nEmergency Braking: 1\n",
                               encoding="utf-8")
            diagnostic.write_text("", encoding="utf-8")
            result = _collect_sumo_telemetry(
                error_log=root / "missing-errors.log",
                message_log=message, diagnostic_log=diagnostic)
        self.assertFalse(result["valid"])
        self.assertTrue(any("missing or unreadable" in reason
                            for reason in result["failures"]))
        self.assertTrue(any("contradictory" in reason
                            for reason in result["failures"]))

    def test_behavior_invalid_run_does_not_save_checkpoints(self):
        agents = Mock()
        files = _save_diagnostic_checkpoints(
            agents, "unused", behavior_valid=False)
        self.assertEqual(files, [])
        agents.save.assert_not_called()

    @staticmethod
    def _timing_config(begin: str = "79200", end: str = "80100",
                       step: str = "1") -> ET.Element:
        return ET.fromstring(
            f"<configuration><time><begin value='{begin}'/>"
            f"<end value='{end}'/><step-length value='{step}'/>"
            f"</time></configuration>")

    def test_probe_accepts_complete_half_open_summary_timeline(self):
        steps = [ET.Element("step", time=str(value))
                 for value in range(79200, 80100)]
        coverage = _summary_horizon_coverage(self._timing_config(), steps)
        self.assertTrue(coverage["valid"])
        self.assertEqual(coverage["expected_step_count"], 900)
        self.assertEqual(coverage["actual_final_timestamp_s"], "80099")

    def test_probe_rejects_missing_final_or_interior_summary_step(self):
        complete = [ET.Element("step", time=str(value))
                    for value in range(79200, 80100)]
        missing_final = _summary_horizon_coverage(
            self._timing_config(), complete[:-1])
        self.assertFalse(missing_final["valid"])
        self.assertIn("step count 899", missing_final["reason"])

        missing_interior = _summary_horizon_coverage(
            self._timing_config(), complete[:400] + complete[401:]
            + [ET.Element("step", time="80100")])
        self.assertFalse(missing_interior["valid"])
        self.assertIn("timestamp mismatch at index 400",
                      missing_interior["reason"])

    def test_probe_preserves_partial_steps_from_native_crash_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.xml"
            path.write_text(
                '<summary><step time="79200"/><step time="79201"/>',
                encoding="utf-8")
            steps, error = _parse_summary_with_partial_evidence(path)
            self.assertEqual([node.get("time") for node in steps],
                             ["79200", "79201"])
            self.assertIn("ParseError", error)

    @patch("DeepQN.evaluation.synthetic_probe.subprocess.run")
    def test_probe_writes_structured_report_after_native_crash(self, run_mock):
        def crash(command, **_kwargs):
            cfg = ET.parse(command[command.index("-c") + 1]).getroot()
            outputs = cfg.find("output")
            for node in outputs:
                value = node.get("value")
                if not value or not node.tag.endswith("-output"):
                    continue
                path = Path(value)
                if node.tag == "summary-output":
                    path.write_text('<summary><step time="79200"/>',
                                    encoding="utf-8")
                else:
                    path.write_text("<root>", encoding="utf-8")
            Path(command[command.index("--message-log") + 1]).write_text(
                "", encoding="utf-8")
            Path(command[command.index("--error-log") + 1]).write_text(
                "", encoding="utf-8")
            return Mock(returncode=-1073741819, stdout="", stderr="")

        run_mock.side_effect = crash
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "probe"
            with self.assertRaises(RuntimeError):
                run_synthetic_probe(
                    manifest_path=INPUTS / "input-manifest.json",
                    output_dir=output, seed=42001,
                    wall_clock_timeout_s=30, configuration="smoke")
            report = json.loads(
                (output / "probe-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["returncode"], -1073741819)
            self.assertFalse(report["execution_integrity_valid"])
            self.assertEqual(report["final_summary"]["time"], "79200")
            self.assertIn("summary.xml",
                          report["partial_artifact_parse_errors"])

    def test_probe_fail_closed_warning_inventory_includes_uncategorized_errors(self):
        lines = _warning_lines(
            "Warning: an otherwise uncategorized simulator condition\n"
            "Error: route loading failed\nTeleporting vehicle 'v0'\n"
        )
        self.assertEqual(len(lines), 3)

    def test_manifest_hashes_counts_and_sorted_explicit_departures(self):
        manifest = json.loads((INPUTS / "input-manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["publication_approved"])
        for key in ("source_network", "source_routes", "network_variant",
                    "expansion_spec", "expansion_manifest", "explicit_routes"):
            identity = manifest[key]
            self.assertEqual(sha256_file(identity["path"]), identity["sha256"])
        counts = manifest["explicit_routes"]["counts"]
        self.assertEqual(counts, {"source_explicit_vehicles": 186,
            "source_flows": 840, "flow_requested_vehicles": 1920,
            "expanded_explicit_vehicles": 2106})
        root = ET.parse(manifest["explicit_routes"]["path"]).getroot()
        vehicles = root.findall("vehicle")
        self.assertEqual(len(vehicles), 2106)
        self.assertFalse(root.findall("flow"))
        departures = [float(vehicle.get("depart")) for vehicle in vehicles]
        self.assertEqual(departures, sorted(departures))
        self.assertEqual(len({vehicle.get("id") for vehicle in vehicles}), 2106)
        self.assertGreater(manifest["explicit_routes"]["invalid_stop_correction_count"], 0)
        corrected = manifest["explicit_routes"]["invalid_stop_corrections"]
        self.assertTrue(all(row["policy"] in {
            "sumo_friendlyPos_true_preserve_stop_and_clamp_position",
            "remove_topologically_impossible_or_out_of_order_stop_preserve_vehicle_route",
            "reorder_stops_to_route_order_preserve_all_obligations",
        } for row in corrected))
        expected_friendly = (
            sum(row["policy"].startswith("sumo_friendlyPos_true")
                for row in corrected)
            - sum(row["source"].get("friendlyPos") == "true"
                  for row in manifest["explicit_routes"]["stop_lane_corrections"]))
        self.assertEqual(sum(stop.get("friendlyPos") == "true"
                             for vehicle in vehicles for stop in vehicle.findall("stop")),
                         expected_friendly)

    def test_v4_route_check_reproduces_bicycle_permission_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            route_path = Path(tmp) / "v4-without-stops.xml"
            root = ET.parse(
                INPUTS_V4 / "routes.explicit.corrected.rou.xml").getroot()
            for vehicle in list(root.findall("vehicle")):
                if vehicle.get("id") != "microbus_982":
                    root.remove(vehicle)
                    continue
                for stop in list(vehicle.findall("stop")):
                    vehicle.remove(stop)
            ET.ElementTree(root).write(route_path, encoding="utf-8",
                                       xml_declaration=True)
            with self.assertRaisesRegex(
                    ValueError,
                    "microbus_982.*vClass='bicycle'.*28718647#3.*690516091#1"):
                validate_explicit_route_integrity(
                    route_path,
                    INPUTS_V4 / "tahrir_synthetic_signal_v1.net.xml",
                    [ROOT / "simulation/maps/tahrir.add.xml",
                     ROOT / "simulation/maps/vehtype.add.xml"])

    def test_v5_records_class_correction_but_fails_end_stop_reachability(self):
        manifest = json.loads(
            (INPUTS_V5 / "input-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"],
                         "sumoflow-night-synthetic-input-candidate-v5")
        explicit = manifest["explicit_routes"]
        self.assertEqual(explicit["vehicle_class_correction_count"], 1)
        correction = explicit["vehicle_class_corrections"][0]
        self.assertEqual(correction["vehicle_id"], "microbus_982")
        self.assertEqual(correction["source_type"], "type_bicycle")
        self.assertEqual(correction["candidate_type"], "type_microbus")
        self.assertEqual(correction["incompatible_pair"],
                         ["28718647#3", "690516091#1"])
        root = ET.parse(explicit["path"]).getroot()
        vehicle = next(node for node in root.findall("vehicle")
                       if node.get("id") == "microbus_982")
        self.assertEqual(vehicle.get("type"), "type_microbus")
        self.assertEqual(vehicle.get("depart"), "86261")
        self.assertEqual(vehicle.find("route").get("edges").split(),
                         correction["route_edges"])
        with tempfile.TemporaryDirectory() as tmp:
            route_path = Path(tmp) / "microbus-21.xml"
            root = ET.parse(explicit["path"]).getroot()
            for vehicle in list(root.findall("vehicle")):
                if vehicle.get("id") != "microbus_21":
                    root.remove(vehicle)
            ET.ElementTree(root).write(route_path, encoding="utf-8",
                                       xml_declaration=True)
            with self.assertRaisesRegex(
                    ValueError,
                    "microbus_21.*stop.*28718874#0_0.*cannot serve next edge.*10873193#0"):
                validate_explicit_route_integrity(
                    route_path, manifest["network_variant"]["path"],
                    [ROOT / "simulation/maps/tahrir.add.xml",
                     ROOT / "simulation/maps/vehtype.add.xml"])

    def test_v6_preserves_demand_and_all_stop_obligations(self):
        manifest = json.loads(
            (INPUTS / "input-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"],
                         "sumoflow-night-synthetic-input-candidate-v6")
        explicit = manifest["explicit_routes"]
        result = validate_explicit_route_integrity(
            explicit["path"], manifest["network_variant"]["path"],
            [ROOT / "simulation/maps/tahrir.add.xml",
             ROOT / "simulation/maps/vehtype.add.xml"])
        self.assertEqual(result, explicit["route_integrity"])
        self.assertEqual(result["vehicle_count"], 2106)
        self.assertEqual(result["required_stop_count"], 242)
        expanded = ET.parse(
            manifest["expanded_routes_intermediate"]["path"]).getroot()
        candidate = ET.parse(explicit["path"]).getroot()
        self.assertEqual(len(expanded.findall("vehicle")), 2106)
        self.assertEqual(sum(len(v.findall("stop")) for v in expanded.findall("vehicle")),
                         242)
        self.assertEqual(sum(len(v.findall("stop")) for v in candidate.findall("vehicle")),
                         242)
        self.assertEqual(explicit["stop_lane_correction_count"], 49)
        corrected_ids = {row["vehicle_id"]
                         for row in explicit["stop_lane_corrections"]}
        self.assertTrue({"microbus_21", "microbus_532", "microbus_634"}
                        <= corrected_ids)
        for identity in ("microbus_21", "microbus_634"):
            vehicle = next(v for v in candidate.findall("vehicle")
                           if v.get("id") == identity)
            self.assertTrue(any(stop.get("lane") == "28718874#0_2"
                                for stop in vehicle.findall("stop")))

    def test_route_integrity_rejects_prohibited_vehicle_class_movement(self):
        with tempfile.TemporaryDirectory() as tmp:
            route_path = Path(tmp) / "routes.xml"
            root = ET.parse(INPUTS / "routes.explicit.corrected.rou.xml").getroot()
            for item in root.findall("vehicle"):
                for stop in list(item.findall("stop")):
                    item.remove(stop)
            vehicle = next(node for node in root.findall("vehicle")
                           if node.get("id") == "microbus_982")
            vehicle.set("type", "type_bicycle")
            ET.ElementTree(root).write(route_path, encoding="utf-8",
                                       xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "no permitted lane connection"):
                validate_explicit_route_integrity(
                    route_path, INPUTS / "tahrir_synthetic_signal_v1.net.xml",
                    [ROOT / "simulation/maps/tahrir.add.xml",
                     ROOT / "simulation/maps/vehtype.add.xml"])

    def test_v5_microbus_532_requires_lane_change_across_short_connector_edges(self):
        network = ET.parse(
            INPUTS_V5 / "tahrir_synthetic_signal_v1.net.xml").getroot()
        connections = network.findall("connection")
        approach_targets = {
            int(node.get("toLane")) for node in connections
            if node.get("from") == "10873186#8"
            and node.get("to") == "10873186#9"
        }
        turn_sources = {
            int(node.get("fromLane")) for node in connections
            if node.get("from") == "10873186#0"
            and node.get("to") == "28718874#0"
        }
        self.assertEqual(approach_targets, {1, 2, 3, 4, 5})
        self.assertEqual(turn_sources, {0})
        lengths = {}
        for edge in network.findall("edge"):
            if edge.get("id") in {"10873186#9", "10873186#0"}:
                lengths[edge.get("id")] = float(edge.find("lane").get("length"))
        self.assertAlmostEqual(sum(lengths.values()), 11.64, places=2)

    def test_v6_secondary_approach_has_direct_lane_chain_without_length_edits(self):
        network = ET.parse(
            INPUTS / "tahrir_synthetic_signal_v1.net.xml").getroot()
        mappings = {
            (int(node.get("fromLane")), int(node.get("toLane")))
            for node in network.findall("connection")
            if node.get("from") == "10873186#8"
            and node.get("to") == "10873186#9"
        }
        self.assertEqual(mappings, {(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)})
        lengths = {}
        for edge in network.findall("edge"):
            if edge.get("id") in {"10873186#8", "10873186#9", "10873186#0"}:
                values = {float(lane.get("length")) for lane in edge.findall("lane")}
                self.assertEqual(len(values), 1)
                lengths[edge.get("id")] = values.pop()
        self.assertEqual(lengths, {"10873186#8": 0.2,
                                   "10873186#9": 5.41,
                                   "10873186#0": 6.23})

    def test_v6_stopping_places_fit_scheduled_vehicle_lengths(self):
        manifest = json.loads(
            (INPUTS / "input-manifest.json").read_text(encoding="utf-8"))
        additions = ET.parse(
            INPUTS / "tahrir.synthetic-stops.add.xml").getroot()
        stops = {node.get("id"): node for node in additions.findall("busStop")}
        self.assertEqual(stops["bs_5"].attrib,
                         {"id": "bs_5", "lane": "28718647#3_0",
                          "startPos": "0.1", "endPos": "12.1"})
        self.assertEqual(stops["bs_3"].get("startPos"), "13.1")
        self.assertEqual(stops["bs_3"].get("endPos"), "25.1")
        for stop_id in ("bs_2", "bs_3", "bs_5", "bs_9"):
            stop = stops[stop_id]
            self.assertGreaterEqual(float(stop.get("endPos")) -
                                    float(stop.get("startPos")), 12.0)
        self.assertEqual({row["stop_id"] for row in
                          manifest["stopping_place_corrections"]},
                         {"bs_2", "bs_3", "bs_5", "bs_9"})

    def test_v6_turn_is_reachable_from_every_corridor_lane(self):
        root = ET.parse(
            INPUTS / "tahrir_synthetic_signal_v1.net.xml").getroot()
        sources = {int(node.get("fromLane")) for node in root.findall("connection")
                   if node.get("from") == "10873186#0"
                   and node.get("to") == "28718874#0"}
        self.assertEqual(sources, set(range(6)))

    @patch("DeepQN.env.sumo_env.traci")
    def test_v6_phase_controller_matches_protected_program(self, traci):
        spec = yaml.safe_load((
            ROOT / "DeepQN/configs/diagnostics/night_three_episode_exploratory_v3.yaml"
        ).read_text(encoding="utf-8"))
        cfg = spec["overrides"]["tls_junctions"]["96621100"]
        ctrl = _PhaseController("96621100", cfg)
        ctrl.phase_elapsed_steps = cfg["min_green"]
        ctrl.apply_action(1)
        traci.trafficlight.setPhase.assert_called_with("96621100", 1)
        for _ in range(3):
            ctrl.tick()
        self.assertEqual(ctrl.current_green_phase, 2)
        traci.trafficlight.setPhase.assert_called_with("96621100", 2)
        ctrl.phase_elapsed_steps = cfg["min_green"]
        ctrl.apply_action(1)
        traci.trafficlight.setPhase.assert_called_with("96621100", 3)
        for _ in range(3):
            ctrl.tick()
        self.assertEqual(ctrl.current_green_phase, 0)
        traci.trafficlight.setPhase.assert_called_with("96621100", 0)


    def test_signal_variant_cycle_and_protected_links_are_nonconflicting(self):
        root = ET.parse(INPUTS / "tahrir_synthetic_signal_v1.net.xml").getroot()
        programs = {node.get("id"): node for node in root.findall("tlLogic")}
        p68 = programs["96621068"].findall("phase")
        self.assertEqual([phase.get("state") for phase in p68], ["gggg"])
        p100 = programs["96621100"].findall("phase")
        self.assertEqual([phase.get("next") for phase in p100],
                         ["1", "2", "3", "0"])
        self.assertEqual([phase.get("state") for phase in p100],
                         ["GGGGGrrrrr", "yyyyyrrrrr",
                          "rrrrrGGGGG", "rrrrryyyyy"])
        for tls_id in ("96621068", "96621100"):
            connections = {int(node.get("linkIndex")): (node.get("to"), node.get("toLane"))
                           for node in root.findall("connection") if node.get("tl") == tls_id}
            for phase in programs[tls_id].findall("phase"):
                protected = [connections[index] for index, state in enumerate(phase.get("state"))
                             if state == "G"]
                self.assertEqual(len(protected), len(set(protected)))

    def test_corrected_diagnostic_keeps_all_lanes_and_matches_sumo_horizon(self):
        base = yaml.safe_load((ROOT / "DeepQN/configs/dqn_config.yaml").read_text())
        for name, expected_end in (("night_900s_synthetic_v1.yaml", 80100),
                                   ("night_full_synthetic_v1.yaml", 86400)):
            diagnostic_path = ROOT / "DeepQN/configs/diagnostics" / name
            diagnostic = yaml.safe_load(diagnostic_path.read_text())
            config = _merge(base, diagnostic["overrides"])
            self.assertEqual(config["tls_junctions"], base["tls_junctions"])
            self.assertEqual(config["profiles"]["night"]["end"], expected_end)
            sumocfg = ROOT / config["profiles"]["night"]["config"]
            declared = float(ET.parse(sumocfg).getroot().find("time/end").get("value"))
            self.assertEqual(declared, expected_end)
            self.assertEqual(config["state"]["lane_occupancy_mode"],
                             "bounded_saturation_v1")

    @patch("DeepQN.env.sumo_env.traci")
    def test_v4_saturates_and_records_occupancy_without_dropping_lane(self, traci):
        traci.lane.getLastStepHaltingNumber.return_value = 0
        traci.lane.getWaitingTime.return_value = 0
        traci.lane.getLastStepOccupancy.return_value = 10.817458
        traci.simulation.getTime.return_value = 79210
        env = SumoEnv.__new__(SumoEnv)
        env.max_lanes = 10; env.fperlane = 3; env.max_queue = 50; env.max_wait = 300
        env.max_elapsed = 120; env.state_dim = 37
        env.lane_observation_mode = "research_fail_closed"
        env.lane_occupancy_mode = "bounded_saturation_v1"
        env._lanes = {"tls": ["short_lane"]}
        env._phase_controllers = {"tls": _PhaseController("tls", {
            "green_phases": [0], "min_green": 1, "max_green": 10})}
        env._occupancy_saturation_count = 0; env._occupancy_saturation_events = []
        state = env._junction_state("tls", np.zeros(5, dtype=np.float32))
        self.assertEqual(state.shape, (37,)); self.assertEqual(float(state[2]), 1.0)
        self.assertEqual(env._occupancy_saturation_count, 1)
        self.assertEqual(env._occupancy_saturation_events[0]["lane_id"], "short_lane")

    @patch("DeepQN.env.sumo_env.traci")
    def test_v4_normalizes_only_negligible_negative_occupancy(self, traci):
        env = SumoEnv.__new__(SumoEnv)
        env.max_lanes = 10; env.fperlane = 3; env.max_queue = 50; env.max_wait = 300
        env.max_elapsed = 120; env.state_dim = 37
        env.lane_observation_mode = "research_fail_closed"
        env.lane_occupancy_mode = "bounded_saturation_v1"
        env._lanes = {"tls": ["10873193#1_0"]}
        env._phase_controllers = {"tls": _PhaseController("tls", {
            "green_phases": [0], "min_green": 1, "max_green": 10})}
        env._occupancy_saturation_count = 0; env._occupancy_saturation_events = []
        env._occupancy_zero_correction_count = 0
        env._occupancy_zero_correction_events = []
        traci.lane.getLastStepHaltingNumber.return_value = 0
        traci.lane.getWaitingTime.return_value = 0
        traci.simulation.getTime.return_value = 82632

        for raw, expected in ((-6.687083418913758e-18, 0.0), (0.0, 0.0),
                              (0.30, 0.30), (10.817458, 1.0)):
            with self.subTest(raw=raw):
                traci.lane.getLastStepOccupancy.return_value = raw
                state = env._junction_state("tls", np.zeros(5, dtype=np.float32))
                self.assertAlmostEqual(float(state[2]), expected)
        self.assertEqual(env._occupancy_zero_correction_count, 1)
        self.assertEqual(env._occupancy_zero_correction_events[0]["raw_occupancy"],
                         -6.687083418913758e-18)

        for raw in (-1.0001e-12, -0.01, float("nan"), float("inf")):
            with self.subTest(invalid=raw):
                traci.lane.getLastStepOccupancy.return_value = raw
                with self.assertRaisesRegex(RuntimeError, "measurement=occupancy"):
                    env._junction_state("tls", np.zeros(5, dtype=np.float32))

    @patch("DeepQN.env.sumo_env.traci")
    def test_v3_still_rejects_greater_than_one_occupancy(self, traci):
        traci.lane.getLastStepHaltingNumber.return_value = 0
        traci.lane.getWaitingTime.return_value = 0
        traci.lane.getLastStepOccupancy.return_value = 1.01
        env = SumoEnv.__new__(SumoEnv)
        env.max_lanes = 10; env.fperlane = 3; env.max_queue = 50; env.max_wait = 300
        env.max_elapsed = 120; env.state_dim = 37
        env.lane_observation_mode = "research_fail_closed"
        env.lane_occupancy_mode = "strict_fraction_v1"
        env._lanes = {"tls": ["lane"]}
        env._phase_controllers = {"tls": _PhaseController("tls", {
            "green_phases": [0], "min_green": 1, "max_green": 10})}
        with self.assertRaisesRegex(RuntimeError, "outside \\[0, 1\\]"):
            env._junction_state("tls", np.zeros(5, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
