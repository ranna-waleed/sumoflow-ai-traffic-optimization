from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from DeepQN.evaluation.metrics import (
    EpisodeMetrics,
    IncomparableRunsError,
    load_metrics_from_outputs,
    parse_tripinfo,
)
from DeepQN.evaluation.evaluate import _patch_sumocfg
from DeepQN.evaluation.evaluate import _ensure_outputs_available
from DeepQN.evaluation.provenance import RunManifest, sha256_file, validate_comparability


class MetricParsingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name: str, text: str) -> None:
        (self.root / name).write_text(text, encoding="utf-8")

    def test_streaming_pipeline_preserves_genuine_zero_values(self):
        self._write(
            "tripinfo_case.xml",
            '<tripinfos><tripinfo id="v0" arrival="0" waitingTime="0" duration="0" timeLoss="0"/></tripinfos>',
        )
        self._write(
            "emission_case.xml",
            '<emission-export><timestep time="0"><vehicle id="v0" CO2="0"/></timestep></emission-export>',
        )
        self._write(
            "summary_case.xml",
            '<summary><step time="0" arrived="1" ended="1" running="0" waiting="0" teleports="0" collisions="0" meanSpeed="0"/></summary>',
        )
        metrics = load_metrics_from_outputs(self.root, "case", "fixed", "case", expected_end_s=1)
        self.assertTrue(metrics.valid)
        self.assertTrue(metrics.completed)
        self.assertEqual(metrics.n_trips, 1)
        self.assertEqual(metrics.avg_waiting_time_s, 0.0)
        self.assertEqual(metrics.total_co2_mg, 0.0)
        self.assertEqual(metrics.co2_per_completed_trip_mg, 0.0)
        self.assertEqual(metrics.teleports, 0)

    def test_missing_file_is_invalid_and_not_zero(self):
        metrics = parse_tripinfo(self.root / "missing.xml", "case", "fixed")
        self.assertFalse(metrics.valid)
        self.assertIsNone(metrics.n_trips)
        self.assertIsNone(metrics.avg_waiting_time_s)

    def test_malformed_xml_is_invalid(self):
        self._write("broken.xml", "<tripinfos><tripinfo")
        metrics = parse_tripinfo(self.root / "broken.xml", "case", "fixed")
        self.assertFalse(metrics.valid)
        self.assertIn("malformed XML", metrics.errors[0])

    def test_missing_required_attribute_is_invalid(self):
        self._write(
            "partial.xml",
            '<tripinfos><tripinfo id="v0" arrival="1" duration="1" timeLoss="1"/></tripinfos>',
        )
        metrics = parse_tripinfo(self.root / "partial.xml", "case", "fixed")
        self.assertFalse(metrics.valid)
        self.assertIsNone(metrics.n_trips)

    def test_co2_denominator_is_completed_trips_not_vehicle_samples(self):
        self._write(
            "tripinfo_case.xml",
            '<tripinfos><tripinfo arrival="1" waitingTime="1" duration="2" timeLoss="1"/>'
            '<tripinfo arrival="1" waitingTime="3" duration="4" timeLoss="2"/></tripinfos>',
        )
        self._write(
            "emission_case.xml",
            '<emission-export><timestep time="0"><vehicle CO2="5"/><vehicle CO2="10"/></timestep>'
            '<timestep time="1"><vehicle CO2="15"/></timestep></emission-export>',
        )
        self._write(
            "summary_case.xml",
            '<summary><step time="0" arrived="0" ended="0" running="1" waiting="0" teleports="2" collisions="1" meanSpeed="4"/>'
            '<step time="1" arrived="2" ended="2" running="0" waiting="0" teleports="2" collisions="1" meanSpeed="6"/></summary>',
        )
        metrics = load_metrics_from_outputs(self.root, "case", "dqn", "case", expected_end_s=2)
        self.assertEqual(metrics.total_co2_mg, 30.0)
        self.assertEqual(metrics.co2_per_completed_trip_mg, 15.0)
        self.assertEqual(metrics.unfinished_vehicles, 0)
        self.assertEqual(metrics.teleports, 2)
        self.assertEqual(metrics.collisions, 1)
        self.assertEqual(metrics.avg_speed_ms, 5.0)

    def test_co2_rate_is_integrated_using_step_length(self):
        self._write(
            "tripinfo_case.xml",
            '<tripinfos><tripinfo arrival="0" waitingTime="0" duration="1" timeLoss="0"/></tripinfos>',
        )
        self._write(
            "emission_case.xml",
            '<emission-export><timestep time="0"><vehicle CO2="10"/></timestep></emission-export>',
        )
        self._write(
            "summary_case.xml",
            '<summary><step time="0" arrived="1" ended="1" running="0" waiting="0" meanSpeed="1"/></summary>',
        )
        metrics = load_metrics_from_outputs(
            self.root, "case", "fixed", "case", step_length_s=0.5, expected_end_s=0.5
        )
        self.assertEqual(metrics.total_co2_mg, 5.0)
        self.assertEqual(metrics.co2_per_completed_trip_mg, 5.0)

    def test_missing_metric_and_zero_baseline_reject_improvement(self):
        baseline = EpisodeMetrics(profile="case", mode="fixed", throughput=1)
        candidate = EpisodeMetrics(profile="case", mode="dqn", throughput=1)
        with self.assertRaises(IncomparableRunsError):
            candidate.improvement_vs(baseline)
        baseline.avg_waiting_time_s = baseline.avg_travel_time_s = 0.0
        baseline.avg_time_loss_s = baseline.total_co2_mg = 0.0
        candidate.avg_waiting_time_s = candidate.avg_travel_time_s = 0.0
        candidate.avg_time_loss_s = candidate.total_co2_mg = 0.0
        comparison = candidate.improvement_vs(baseline)
        self.assertIsNone(comparison["avg_wait_pct"])
        self.assertIn("baseline is zero", comparison["avg_wait_pct_unavailable_reason"])
        self.assertEqual(comparison["avg_wait_absolute_delta"], 0.0)
        self.assertEqual(comparison["throughput_delta"], 0)

    def test_temporary_sumo_config_redirects_only_phase1a_outputs(self):
        source = self.root / "case.sumocfg"
        original = (
            '<configuration><input/><o><tripinfo-output value="historical-trip.xml"/>'
            '<fcd-output value="historical-fcd.xml"/></o></configuration>'
        )
        source.write_text(original, encoding="utf-8")
        outputs = {
            name: self.root / f"new-{name}.xml"
            for name in ("tripinfo", "emission", "summary")
        }
        temporary = _patch_sumocfg(source, outputs, seed=17)
        try:
            patched = temporary.read_text(encoding="utf-8")
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            self.assertNotIn("historical-fcd.xml", patched)
            self.assertNotIn("historical-trip.xml", patched)
            self.assertIn("new-tripinfo.xml", patched)
            self.assertIn('seed value="17"', patched)
        finally:
            temporary.unlink(missing_ok=True)

    def test_horizon_success_does_not_require_network_to_clear(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="9" waitingTime="1" duration="2" timeLoss="1"/></tripinfos>')
        self._write("emission_x.xml", '<emission-export/>')
        self._write("summary_x.xml", '<summary><step time="0" arrived="0" ended="0" running="0" waiting="0"/>'
                    '<step time="9" arrived="1" ended="1" running="3" waiting="2"/></summary>')
        metrics = load_metrics_from_outputs(self.root, "x", "dqn", "x", expected_end_s=10)
        self.assertTrue(metrics.valid); self.assertTrue(metrics.reached_horizon)
        self.assertFalse(metrics.network_cleared); self.assertEqual(metrics.unfinished_vehicles, 5)

    def test_premature_termination_is_invalid(self):
        self._write("tripinfo_x.xml", '<tripinfos/>'); self._write("emission_x.xml", '<emission-export/>')
        self._write("summary_x.xml", '<summary><step time="7" arrived="0" ended="0" running="0" waiting="0"/></summary>')
        metrics = load_metrics_from_outputs(self.root, "x", "dqn", "x", expected_end_s=10)
        self.assertFalse(metrics.valid); self.assertFalse(metrics.reached_horizon)

    def test_zero_arrivals_have_zero_throughput_and_undefined_means(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="-1" waitingTime="5" duration="5" timeLoss="5"/></tripinfos>')
        metrics = parse_tripinfo(self.root / "tripinfo_x.xml", "x", "fixed")
        self.assertEqual(metrics.throughput, 0); self.assertIsNone(metrics.avg_travel_time_s)

    def test_vaporized_and_unfinished_records_are_not_arrivals(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="-1" vaporized="collision" waitingTime="1" duration="2" timeLoss="1"/>'
                    '<tripinfo arrival="-1" waitingTime="1" duration="2" timeLoss="1"/>'
                    '<tripinfo arrival="4"><emissions CO2_abs="8"/></tripinfo></tripinfos>')
        # The successful record must carry required statistics.
        text = (self.root / "tripinfo_x.xml").read_text().replace('arrival="4"', 'arrival="4" waitingTime="0" duration="4" timeLoss="1"')
        (self.root / "tripinfo_x.xml").write_text(text)
        metrics = parse_tripinfo(self.root / "tripinfo_x.xml", "x", "fixed")
        self.assertEqual((metrics.tripinfo_records, metrics.successful_arrivals, metrics.removed_or_vaporized, metrics.unfinished_tripinfo_records), (3, 1, 1, 1))
        self.assertEqual(metrics.mean_successful_trip_co2_mg, 8)

    def test_teleport_tag_does_not_override_reported_arrival(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="4" vaporized="teleport" waitingTime="2" duration="4" timeLoss="1"/></tripinfos>')
        self._write("emission_x.xml", '<emission-export/>')
        self._write("summary_x.xml", '<summary><step time="0" arrived="0" ended="0" running="1" waiting="0"/>'
                    '<step time="4" arrived="1" ended="1" running="0" waiting="0" teleports="1"/></summary>')
        metrics = load_metrics_from_outputs(self.root, "x", "fixed", "x", expected_end_s=5)
        self.assertTrue(metrics.valid)
        self.assertEqual(metrics.successful_arrivals, 1)
        self.assertEqual(metrics.teleport_or_vaporization_tagged_arrivals, 1)
        self.assertEqual(metrics.removed_or_vaporized, 0)

    def test_nonarrival_removal_reconciles_with_summary_ended(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="-1" vaporized="collision" waitingTime="2" duration="4" timeLoss="1"/></tripinfos>')
        self._write("emission_x.xml", '<emission-export/>')
        self._write("summary_x.xml", '<summary><step time="0" arrived="0" ended="0" running="1" waiting="0"/>'
                    '<step time="4" arrived="0" ended="1" running="0" waiting="0"/></summary>')
        metrics = load_metrics_from_outputs(self.root, "x", "fixed", "x", expected_end_s=5)
        self.assertTrue(metrics.valid); self.assertEqual(metrics.removed_or_vaporized, 1)

    def test_contradictory_summary_arrival_count_is_rejected(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="4" waitingTime="0" duration="4" timeLoss="0"/></tripinfos>')
        self._write("emission_x.xml", '<emission-export/>')
        self._write("summary_x.xml", '<summary><step time="0" arrived="0" ended="0" running="1" waiting="0"/>'
                    '<step time="4" arrived="0" ended="0" running="0" waiting="0"/></summary>')
        metrics = load_metrics_from_outputs(self.root, "x", "fixed", "x", expected_end_s=5)
        self.assertFalse(metrics.valid)
        self.assertIn("tripinfo arrivals 1 != summary arrived 0", metrics.errors[-1])

    def test_trip_co2_requires_complete_co2_abs_measurements(self):
        self._write("complete.xml", '<tripinfos><tripinfo arrival="1" waitingTime="0" duration="1" timeLoss="0">'
                    '<emissions CO2_abs="12.5"/></tripinfo></tripinfos>')
        complete = parse_tripinfo(self.root / "complete.xml", "x", "fixed")
        self.assertEqual(complete.mean_successful_trip_co2_mg, 12.5)
        self._write("partial.xml", '<tripinfos><tripinfo arrival="1" waitingTime="0" duration="1" timeLoss="0">'
                    '<emissions CO2_abs="12.5"/></tripinfo><tripinfo arrival="2" waitingTime="0" duration="1" timeLoss="0"/></tripinfos>')
        self.assertIsNone(parse_tripinfo(self.root / "partial.xml", "x", "fixed").mean_successful_trip_co2_mg)

    def test_step_scaled_emissions_are_not_integrated_twice(self):
        self._write("tripinfo_x.xml", '<tripinfos><tripinfo arrival="1" waitingTime="0" duration="1" timeLoss="0"/></tripinfos>')
        self._write("emission_x.xml", '<emission-export><vehicle CO2="10"/></emission-export>')
        self._write("summary_x.xml", '<summary><step time="1" arrived="1" ended="1" running="0" waiting="0"/></summary>')
        scaled = load_metrics_from_outputs(self.root, "x", "fixed", "x", step_length_s=.5, expected_end_s=1.5, emission_step_scaled=True)
        self.assertEqual(scaled.total_co2_mg, 10); self.assertEqual(scaled.network_co2_per_successful_arrival_mg, 10)

    def test_nonfinite_values_and_output_collisions_are_rejected(self):
        self._write("bad.xml", '<tripinfos><tripinfo arrival="1" waitingTime="nan" duration="1" timeLoss="0"/></tripinfos>')
        self.assertFalse(parse_tripinfo(self.root / "bad.xml", "x", "fixed").valid)
        occupied = self.root / "occupied.xml"; occupied.touch()
        with self.assertRaises(FileExistsError): _ensure_outputs_available([occupied])

    def test_summary_mean_speed_minus_one_is_unavailable_sentinel(self):
        self._write("tripinfo_x.xml", '<tripinfos/>'); self._write("emission_x.xml", '<emission-export/>')
        self._write("summary_x.xml", '<summary><step time="0" arrived="0" ended="0" running="0" waiting="0" meanSpeed="-1.00"/></summary>')
        metrics = load_metrics_from_outputs(self.root, "x", "fixed", "x", expected_end_s=1)
        self.assertTrue(metrics.valid); self.assertIsNone(metrics.avg_speed_ms)


def _manifest(**updates) -> RunManifest:
    fixture = str(Path(__file__).resolve())
    fixture_hash = sha256_file(fixture)
    values = dict(
        run_id="run",
        controller="fixed_time",
        scenario="case",
        seed=7,
        simulation_begin_s=0.0,
        simulation_end_s=10.0,
        step_length_s=1.0,
        config_path="controller.yaml",
        config_sha256="controller",
        simulation_config_path="case.sumocfg",
        simulation_config_sha256="sumocfg",
        network_path="case.net.xml",
        network_sha256="network",
        route_files={"case.rou.xml": "demand"},
        effective_simulation_config_sha256="effective",
        emission_step_scaled=False,
        git_commit="abc",
        git_dirty=False,
        output_paths={name: fixture for name in ("tripinfo", "emission", "summary")},
        output_hashes={name: fixture_hash for name in ("tripinfo", "emission", "summary")},
        sumo_version="Eclipse SUMO sumo 1.26.0",
        completion_status="complete",
    )
    values.update(updates)
    return RunManifest(**values)


class ComparabilityTests(unittest.TestCase):
    def test_matching_manifests_are_comparable(self):
        validate_comparability(_manifest(), _manifest(controller="dqn"))

    def test_sumo_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(IncomparableRunsError, "SUMO version"):
            validate_comparability(
                _manifest(sumo_version="Eclipse SUMO sumo 1.24.0"),
                _manifest(sumo_version="Eclipse SUMO sumo 1.26.0"),
            )

    def test_horizon_and_demand_mismatches_are_rejected(self):
        with self.assertRaises(IncomparableRunsError):
            validate_comparability(_manifest(), _manifest(simulation_end_s=11.0))
        with self.assertRaises(IncomparableRunsError):
            validate_comparability(_manifest(), _manifest(route_files={"case.rou.xml": "other"}))

    def test_incomplete_manifest_is_rejected(self):
        with self.assertRaises(IncomparableRunsError):
            validate_comparability(_manifest(), _manifest(completion_status="incomplete"))

    def test_modified_or_missing_additional_file_provenance_is_rejected(self):
        with self.assertRaisesRegex(IncomparableRunsError, "additional-file"):
            validate_comparability(_manifest(additional_files={"a.xml":"one"}), _manifest(additional_files={"a.xml":"two"}))
        with self.assertRaises(IncomparableRunsError):
            validate_comparability(_manifest(effective_simulation_config_sha256=None), _manifest())


if __name__ == "__main__":
    unittest.main()
