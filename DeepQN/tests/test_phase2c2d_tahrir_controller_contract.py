from __future__ import annotations

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "DeepQN/configs/controller_contracts/tahrir_controller_review_candidate_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TahrirControllerContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.network = ET.parse(ROOT / self.contract["sources"]["network"]["path"]).getroot()

    def test_sources_are_actual_hash_bound_files(self):
        for record in self.contract["sources"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(_sha256(path), record["sha256"])

    def test_all_nine_tls_and_every_controlled_link_are_accounted_for(self):
        controlled = self.contract["controlled_tls"]
        unselected = self.contract["unselected_tls"]
        self.assertEqual(len(controlled), 7)
        self.assertEqual(len(unselected), 2)
        network_tls = {node.get("id") for node in self.network.iter("tlLogic")}
        self.assertEqual(set(controlled) | set(unselected), network_tls)
        for tls_id, record in controlled.items():
            logic = next(node for node in self.network.iter("tlLogic") if node.get("id") == tls_id)
            physical = [(float(p.get("duration")), p.get("state")) for p in logic.findall("phase")]
            declared = [(float(p["duration_s"]), p["state"]) for p in record["program"]["phases"]]
            self.assertEqual(physical, declared)
            links = {int(node.get("linkIndex")) for node in self.network.iter("connection")
                     if node.get("tl") == tls_id}
            declared_links = {index for group in record["service_groups"] for index in group["link_indices"]}
            self.assertEqual(links, declared_links)
            for group in record["service_groups"]:
                phase_state = record["program"]["phases"][group["phase"]]["state"]
                self.assertTrue(all(phase_state[index] in "Gg" for index in group["link_indices"]))
                self.assertIsNone(group["saturation_weights"])

    def test_only_two_tls_have_competing_service_phases(self):
        eligible = {tls_id for tls_id, record in self.contract["controlled_tls"].items()
                    if record["max_pressure_eligibility"].startswith("topology_eligible")}
        self.assertEqual(eligible, {"315744796", "2031414903"})
        for tls_id, record in self.contract["controlled_tls"].items():
            green_phases = {group["phase"] for group in record["service_groups"]}
            if tls_id not in eligible:
                self.assertEqual(len(green_phases), 1)
                self.assertEqual(record["max_pressure_eligibility"],
                                 "ineligible_no_competing_service_phase")

    def test_all_native_cycles_and_exact_green_link_inventories(self):
        expected_states = {
            "2031414899": ("GGGG", "yyyy", "rrrr"),
            "2031414903": ("GGrr", "yyrr", "rrGG", "rryy"),
            "271064234": ("GGG", "yyy", "rrr"),
            "315743335": ("GGG", "yyy", "rrr"),
            "315744796": ("GGGGrrrrrr", "yyyyrrrrrr", "rrrrGGGGGG", "rrrryyyyyy"),
            "6288771431": ("GGGG", "yyyy", "rrrr"),
            "6288771435": ("GGGGGGGG",),
            "96621068": ("GGGG",),
            "96621100": ("rrrrrrrrrr", "yyyyyyyyyy", "GGGGGGGGGG"),
        }
        inventory = self.contract["native_cycle_inventory"]
        self.assertEqual(set(inventory), set(expected_states))
        for tls_id, expected in expected_states.items():
            logic = next(node for node in self.network.iter("tlLogic") if node.get("id") == tls_id)
            states = tuple(phase.get("state") for phase in logic.findall("phase"))
            self.assertEqual(states, expected)
            expected_transitions = [f"{index}->{(index + 1) % len(states)}"
                                    for index in range(len(states))]
            self.assertEqual(inventory[tls_id]["cyclic_transitions"], expected_transitions)
            actual_green = [[index for index, signal in enumerate(state) if signal in "Gg"]
                            for state in states]
            declared_green = [phase["green_links"] for phase in inventory[tls_id]["ordered_phases"]]
            self.assertEqual(declared_green, actual_green)

    def test_96621100_native_and_commanded_orders_are_distinct_and_flagged(self):
        native = self.contract["native_cycle_inventory"]["96621100"]
        self.assertEqual(native["cyclic_transitions"], ["0->1", "1->2", "2->0"])
        self.assertIn("anomaly", native["review_status"])
        cfg = yaml.safe_load((ROOT / "DeepQN/configs/dqn_config.yaml").read_text(encoding="utf-8"))
        tls = cfg["tls_junctions"]["96621100"]
        self.assertEqual(tls["green_to_yellow"], {"2": 1})
        self.assertEqual(tls["yellow_to_next"], {"1": 0})
        self.assertEqual(tls["red_to_green_phase"], 2)
        self.assertEqual(native["commanded_controller_difference"],
            "DQN phase controller commands 2->1->0 and 0->2, not the native 0->1->2->0 order")

    def test_candidate_cannot_self_approve_or_supply_invented_parameters(self):
        self.assertFalse(self.contract["publication_approved"])
        self.assertEqual(self.contract["approval_trust_roots"], [])
        review = self.contract["external_review"]
        self.assertTrue(all(review[key] is None for key in (
            "decision", "reviewer_identity", "organization", "date",
            "signature_or_external_record", "approved_mapping_sha256",
            "approved_saturation_definition_sha256", "approved_actuated_program_sha256")))
        detector = self.contract["detector_and_actuation_status"]
        self.assertFalse(detector["effective_tahrir_additional_file_contains_traffic_detectors"])
        self.assertIsNone(detector["approved_gap_parameters"])
        self.assertIsNone(detector["approved_detector_assignment"])


if __name__ == "__main__":
    unittest.main()
