from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from DeepQN.evaluation.demand_expansion import _physical_fingerprint
from DeepQN.evaluation.research_protocol import (
    CALIBRATION_PROTOCOL_SCHEMA, CALIBRATION_TRACE_SCHEMA, EXPERIMENT_SCHEMA,
    SCIENTIFIC_SPLIT_SCHEMA, calibrate_publication_reward_references,
    paired_primary_estimand, validate_experiment_candidate,
    validate_publication_pairing, validate_readiness_report,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RewardCalibrationV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps({
            "schema": CALIBRATION_PROTOCOL_SCHEMA, "estimator": "positive_quantile",
            "quantile": 0.95, "numpy_method": "linear",
            "pooling_rule": "separate_reference_per_profile",
            "zero_policy": "retain_in_trace_exclude_from_scale",
            "eligible_partitions": ["training", "calibration"],
            "required_profiles": ["morning", "midday", "evening", "night"],
            "trace_generation_protocol": "frozen_non_evaluation_controller_diagnostic_v1",
            "publication_approved": False,
        }, sort_keys=True), encoding="utf-8")

    def tearDown(self): self.temp.cleanup()

    def _calibrate(self, registry, output="calibration.json", protocol=None):
        selected = protocol or self.protocol
        return calibrate_publication_reward_references(
            registry, digest(registry), self.root / output,
            protocol_path=selected, expected_protocol_sha256=digest(selected))

    def _registry(self, mutate=None):
        layout = [(profile, pool) for profile in ("morning", "midday", "evening", "night")
                  for pool in ("training", "calibration")]
        layout += [("morning", pool) for pool in
                   ("validation", "forecast_test", "controller_evaluation")]
        rows = []; template_hash = hashlib.sha256(b"common template").hexdigest()
        for index, (profile, pool) in enumerate(layout):
            identity = f"r-{index}"; demand = self.root / f"demand-{index}.xml"
            demand.write_text(f"<routes><vehicle id='{identity}' depart='{index}' type='car'>"
                              f"<route edges='a b'/></vehicle></routes>", encoding="utf-8")
            fingerprint = _physical_fingerprint(ET.parse(demand).getroot().findall("vehicle"))
            eligible = pool in {"training", "calibration"}; trace = self.root / f"trace-{index}.json"
            payload = {"schema": CALIBRATION_TRACE_SCHEMA, "realization_id": identity,
                "profile": profile, "partition": pool, "template_sha256": template_hash,
                "explicit_demand_sha256": digest(demand),
                "physical_fingerprint_sha256": fingerprint,
                "demand_generator_seed": 100 + index, "sumo_seed": 200 + index,
                "intervals": [{"duration_s": 10, "delay_vehicle_s": index + 1,
                    "co2_mg": 10 * (index + 1), "arrivals_vehicles": index + 1,
                    "pressure_vehicle_s": 2 * (index + 1)}]}
            if eligible: trace.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            rows.append({"realization_id": identity, "profile": profile, "partition": pool,
                "template_sha256": template_hash, "explicit_demand_path": demand.name,
                "explicit_demand_sha256": digest(demand),
                "physical_fingerprint_sha256": fingerprint,
                "demand_generator_seed": 100 + index, "sumo_seed": 200 + index,
                "reward_trace_path": trace.name if eligible else None,
                "reward_trace_sha256": digest(trace) if eligible else None})
        if mutate: mutate(rows)
        registry = self.root / f"registry-{len(list(self.root.glob('registry-*')))}.json"
        registry.write_text(json.dumps({"schema": SCIENTIFIC_SPLIT_SCHEMA,
                                        "realizations": rows}, sort_keys=True), encoding="utf-8")
        return registry, rows

    def test_missing_heldout_traces_succeeds_and_cannot_affect_bytes(self):
        registry, rows = self._registry(); first = self._calibrate(registry, "first.json")
        self.assertEqual({row["partition"] for row in first["sources"]}, {"training", "calibration"})
        self.assertEqual(first, self._calibrate(registry, "second.json"))
        heldout = next(row for row in rows if row["partition"] == "controller_evaluation")
        heldout["reward_trace_path"] = "forbidden.json"; heldout["reward_trace_sha256"] = "a" * 64
        forged = self.root / "heldout.json"
        forged.write_text(json.dumps({"schema": SCIENTIFIC_SPLIT_SCHEMA,
                                      "realizations": rows}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "held-out reward trace is forbidden"):
            self._calibrate(forged, "forbidden-output.json")

    def test_missing_eligible_trace_and_modified_source_fail(self):
        registry, rows = self._registry(); (self.root / rows[0]["reward_trace_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "eligible reward trace is missing"):
            self._calibrate(registry)
        registry, rows = self._registry(); (self.root / rows[0]["explicit_demand_path"]).write_text(
            "<routes/>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "demand artifact is missing or changed"):
            self._calibrate(registry, "changed.json")
        registry, rows = self._registry(); trace = self.root / rows[0]["reward_trace_path"]
        trace.write_text(trace.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reward trace is missing or changed"):
            self._calibrate(registry, "changed-trace.json")

    def test_renamed_physical_duplicate_rejected_even_with_forged_fingerprints(self):
        def duplicate(rows):
            source = self.root / rows[0]["explicit_demand_path"]
            target = self.root / rows[-1]["explicit_demand_path"]
            root = ET.parse(source).getroot(); root.find("vehicle").set("id", "attacker-renamed")
            target.write_bytes(ET.tostring(root, encoding="utf-8"))
            rows[-1]["explicit_demand_sha256"] = digest(target)
            rows[-1]["physical_fingerprint_sha256"] = hashlib.sha256(b"forged").hexdigest()
        registry, _ = self._registry(duplicate)
        with self.assertRaisesRegex(ValueError, "declared physical fingerprint is untrusted"):
            self._calibrate(registry)

    def test_distinct_schedules_same_template_are_allowed(self):
        registry, rows = self._registry()
        self.assertEqual(len({row["template_sha256"] for row in rows}), 1)
        self.assertEqual(len({row["physical_fingerprint_sha256"] for row in rows}), len(rows))
        self._calibrate(registry)

    def test_profile_coverage_and_frozen_protocol_fail_closed(self):
        registry, _ = self._registry(lambda rows: rows.__setitem__(slice(None),
            [row for row in rows if row["profile"] != "night"]))
        with self.assertRaisesRegex(ValueError, "profile coverage is incomplete"):
            self._calibrate(registry)
        changed = json.loads(self.protocol.read_text()); changed["quantile"] = .5
        alternative = self.root / "alternative.json"
        alternative.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
        registry, _ = self._registry()
        with self.assertRaisesRegex(ValueError, "differs from the frozen candidate"):
            self._calibrate(registry, "alternative-output.json", alternative)

    def test_invalid_intervals_and_zero_scale_fail_closed(self):
        for field, value, message in (("duration_s", 0, "duration"),
                                      ("co2_mg", -1, "invalid calibration"),
                                      ("co2_mg", float("nan"), "invalid calibration")):
            def mutate(rows, field=field, value=value):
                trace = self.root / rows[0]["reward_trace_path"]
                payload = json.loads(trace.read_text()); payload["intervals"][0][field] = value
                trace.write_text(json.dumps(payload), encoding="utf-8")
                rows[0]["reward_trace_sha256"] = digest(trace)
            registry, _ = self._registry(mutate)
            with self.assertRaisesRegex(ValueError, message): self._calibrate(registry)
        def zero(rows):
            for row in rows:
                if row["partition"] in {"training", "calibration"}:
                    trace = self.root / row["reward_trace_path"]
                    payload = json.loads(trace.read_text()); payload["intervals"][0]["arrivals_vehicles"] = 0
                    trace.write_text(json.dumps(payload), encoding="utf-8")
                    row["reward_trace_sha256"] = digest(trace)
        registry, _ = self._registry(zero)
        with self.assertRaisesRegex(ValueError, "no positive arrivals_vehicles"):
            self._calibrate(registry)


class ExperimentAndAnalysisContractTests(unittest.TestCase):
    @staticmethod
    def manifest():
        return {"schema": EXPERIMENT_SCHEMA, "publication_approved": False,
            "phase3_authorized": False,
            "contracts": {"observation": "sumoflow-dqn-observation-v3-lane-occupancy-fraction",
                "forecast_inference": "junction-bilstm-60s-to-30s-v2-nonnegative-counts",
                "forecast_dataset": "causal-junction-arrivals-60s-to-30s-v4-right-closed",
                "reward": "interval-global-reward-v1",
                "metrics": "sumo-xml-demand-normalized-horizon-wait-v5"},
            "network": {"path": "network", "sha256": "a" * 64},
            "effective_configuration": {"path": "config", "sha256": "b" * 64,
                "status": "provisional_not_effective_for_publication"},
            "controlled_tls": [str(index) for index in range(7)],
            "demand": {"contract": "explicit-id", "partition": "controller_evaluation"},
            "warmup": {"duration_s": 60, "protocol": "phase_controller_keep_v1",
                       "included_in_outcomes": True}, "simulation_step_s": 1,
            "decision_interval_s": 10, "policy": {},
            "training": {"seeds_per_arm": 5, "budget": "unresolved",
                         "selection": "validation_only", "pairing_rule": None},
            "evaluation": {"paired_demands_per_profile": 10},
            "arms": {"fixed_time_secondary": {}, "no_forecast_dqn": {},
                     "verified_forecast_dqn": {}},
            "profiles": {name: {} for name in ("morning", "midday", "evening", "night")},
            "failure_policy": {}, "artifacts": {"reward_calibration": {"sha256": None}}}

    @staticmethod
    def _pair(profile, demand, difference):
        return [{"training_seed": 0, "demand_id": demand, "sumo_seed": 7,
                 "profile": profile, "arm": arm, "waiting_s": value, "valid": True}
                for arm, value in (("no_forecast_dqn", 20),
                                   ("verified_forecast_dqn", 20 + difference))]

    def test_equal_profile_estimator_is_not_cell_weighted(self):
        rows = self._pair("morning", "m0", -10) + self._pair("morning", "m1", -10)
        rows += self._pair("night", "n0", 10)
        result = paired_primary_estimand(rows)
        self.assertEqual(result["overall_mean_difference_s"], 0)
        self.assertEqual(result["per_profile_mean_difference_s"], {"morning": -10, "night": 10})

    def _publication_rows(self):
        rows = []
        for profile in ("morning", "midday", "evening", "night"):
            for seed in range(5):
                for demand in range(10):
                    fingerprint = hashlib.sha256(f"{profile}-{demand}".encode()).hexdigest()
                    for arm in ("no_forecast_dqn", "verified_forecast_dqn"):
                        rows.append({"profile": profile, "training_seed": seed,
                            "policy_id": f"{arm}-{seed}",
                            "checkpoint_sha256": hashlib.sha256(f"{arm}-{seed}".encode()).hexdigest(),
                            "demand_id": f"{profile}-{demand}",
                            "physical_fingerprint_sha256": fingerprint,
                            "sumo_seed": 1000 + demand, "arm": arm, "valid": True})
        return rows

    def test_publication_pairing_requires_frozen_rule_and_full_balanced_cells(self):
        manifest = self.manifest(); rows = self._publication_rows()
        with self.assertRaisesRegex(ValueError, "pairing rule is not frozen"):
            validate_publication_pairing(rows, manifest)
        manifest["training"]["pairing_rule"] = "matched_training_seed_v1"
        validate_publication_pairing(rows, manifest)
        for changed in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError): validate_publication_pairing(changed, manifest)
        unbalanced = copy.deepcopy(rows); unbalanced[0]["physical_fingerprint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "contradictory|physically unmatched"):
            validate_publication_pairing(unbalanced, manifest)

    def test_candidate_and_readiness_authority_gates_remain_closed(self):
        manifest = self.manifest(); validate_experiment_candidate(manifest)
        with self.assertRaisesRegex(ValueError, "not externally authorized"):
            validate_experiment_candidate(manifest, require_execution_ready=True)
        report = {"schema": "sumoflow-phase3c-readiness-v1", "publication_approved": False,
                  "phase3_authorized": False, "trust_roots": [],
                  "engineering_prerequisites": {}, "scientific_blockers": ["absent"],
                  "artifacts": {}}
        validate_readiness_report(report); report["trust_roots"] = ["local"]
        with self.assertRaises(ValueError): validate_readiness_report(report)


if __name__ == "__main__": unittest.main()
