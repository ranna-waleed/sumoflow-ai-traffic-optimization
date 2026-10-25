from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from DeepQN.evaluation.baseline import run_fixed_time_episode
from DeepQN.evaluation.evaluate import _paired_baseline_manifest_path, _patch_sumocfg
from DeepQN.evaluation.metrics import EpisodeMetrics, IncomparableRunsError, load_metrics_from_paths
from DeepQN.forecasting.contracts import (
    FEATURE_NAMES,
    ForecastContractError,
    JunctionFeatureExtractor,
    TrafficSnapshot,
    VehicleObservation,
    load_approach_mapping,
)
from lstm.research_data import (
    SequenceSource,
    build_causal_windows,
    fit_training_scaler,
    transform_features,
    require_publication_training_ready,
    write_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
TINY_NET = ROOT / "DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml"
BACKLOG_CONFIG = ROOT / "DeepQN/tests/fixtures/congested_sumo/backlog_eval.yaml"


def _mapping_file(directory: Path, tls=None) -> Path:
    path = directory / "mapping.json"
    path.write_text(json.dumps({
        "schema_version": "junction-approach-arrivals-v1",
        "network": "tiny.net.xml",
        "tls": tls or {
            "center": {"north": ["s_in"], "south": ["n_in"],
                       "east": ["w_in"], "west": ["e_in"]},
        },
    }), encoding="utf-8")
    return path


def _source(source_id: str, partition: str, crossing_time: int) -> SequenceSource:
    snapshots = []
    for time_s in range(95):
        vehicles = ()
        if crossing_time <= time_s <= crossing_time + 1:
            vehicles = (VehicleObservation("v", "n_in", 5.0, 2.0),)
        snapshots.append(TrafficSnapshot(float(time_s), vehicles))
    return SequenceSource(
        source_id=source_id,
        partition=partition,
        snapshots=tuple(snapshots),
        demand_seed={"train": 1, "validation": 2, "test": 3}[partition],
        sumo_config_sha256=f"config-{partition}",
        source_sha256=hashlib.sha256(source_id.encode()).hexdigest(),
        demand_realization_id=f"demand-{source_id}",
        demand_sha256=hashlib.sha256(f"demand-{source_id}".encode()).hexdigest(),
    )


class DemandContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name: str, value: str) -> Path:
        path = self.root / name
        path.write_text(value, encoding="utf-8")
        return path

    def test_censored_waiting_reconciles_all_populations_and_teleports(self):
        route = self._write("demand.rou.xml", '<routes>' + ''.join(
            f'<vehicle id="{name}" depart="0"/>'
            for name in ("arrived", "removed", "running", "undeparted")
        ) + '<vehicle id="future" depart="20"/></routes>')
        trip = self._write("trip.xml", '<tripinfos>'
            '<tripinfo id="arrived" depart="0" arrival="8" waitingTime="2" duration="8" timeLoss="3" vaporized="teleport"/>'
            '<tripinfo id="removed" depart="0" arrival="-1" waitingTime="1" vaporized="collision"/>'
            '<tripinfo id="running" depart="0" arrival="-1" waitingTime="3" vaporized="end"/>'
            '<tripinfo id="undeparted" depart="-1" arrival="-1" waitingTime="0" vaporized="end"/>'
            '</tripinfos>')
        emission = self._write("emission.xml", '<emission-export/>')
        summary = self._write("summary.xml", '<summary>'
            '<step time="0" arrived="0" ended="0" inserted="1" running="1" waiting="3"/>'
            '<step time="9" arrived="1" ended="2" inserted="3" running="1" waiting="1" teleports="1"/>'
            '</summary>')
        metrics = load_metrics_from_paths(
            tripinfo_path=trip, emission_path=emission, summary_path=summary,
            profile="case", mode="fixed", expected_end_s=10,
            route_paths=[route], require_demand_contract=True,
        )
        self.assertTrue(metrics.valid, metrics.errors)
        self.assertEqual(metrics.scheduled_demand, 4)
        self.assertEqual(metrics.inserted_vehicles, 3)
        self.assertEqual(metrics.undeparted_vehicles, 1)
        self.assertEqual(metrics.successful_arrivals, 1)
        self.assertEqual(metrics.removed_or_vaporized, 1)
        self.assertEqual(metrics.running_vehicles, 1)
        self.assertAlmostEqual(metrics.demand_normalized_waiting_s, 4.0)
        self.assertAlmostEqual(metrics.insertion_rate, 0.75)
        self.assertAlmostEqual(metrics.completion_rate, 0.25)
        self.assertAlmostEqual(metrics.conditional_completion_rate, 1 / 3)

    def test_unknown_or_missing_demand_ids_are_rejected(self):
        route = self._write("demand.rou.xml", '<routes><vehicle id="expected" depart="0"/></routes>')
        trip = self._write("trip.xml", '<tripinfos><tripinfo id="unknown" depart="-1" arrival="-1" waitingTime="0" vaporized="end"/></tripinfos>')
        emission = self._write("emission.xml", '<emission-export/>')
        summary = self._write("summary.xml", '<summary><step time="9" arrived="0" ended="0" inserted="0" running="0" waiting="1"/></summary>')
        metrics = load_metrics_from_paths(
            tripinfo_path=trip, emission_path=emission, summary_path=summary,
            profile="case", mode="fixed", expected_end_s=10,
            route_paths=[route], require_demand_contract=True,
        )
        self.assertFalse(metrics.valid)
        self.assertTrue(any("ID mismatch" in error for error in metrics.errors))

    def test_fixed_and_controlled_patch_uses_complete_measurement_settings(self):
        outputs = {name: self.root / f"{name}.xml" for name in ("tripinfo", "emission", "summary")}
        patched = _patch_sumocfg(
            ROOT / "DeepQN/tests/fixtures/congested_sumo/backlog.sumocfg", outputs, 23
        )
        try:
            text = patched.read_text(encoding="utf-8")
            self.assertIn('tripinfo-output.write-unfinished value="true"', text)
            self.assertIn('tripinfo-output.write-undeparted value="true"', text)
            self.assertIn('emission-output.step-scaled value="false"', text)
        finally:
            patched.unlink(missing_ok=True)

    def test_real_sumo_backlog_fixture_has_zero_arrivals_and_valid_primary_metric(self):
        with BACKLOG_CONFIG.open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        output = self.root / "sumo-output"
        metrics, manifest = run_fixed_time_episode(
            cfg=cfg, profile="backlog", output_dir=output,
            run_index=0, seed=23, config_path=BACKLOG_CONFIG,
            wall_clock_timeout_s=20,
        )
        self.assertTrue(metrics.valid, metrics.errors)
        self.assertEqual(metrics.successful_arrivals, 0)
        self.assertEqual(metrics.scheduled_demand, 10)
        self.assertEqual(metrics.inserted_vehicles, 2)
        self.assertEqual(metrics.running_vehicles, 2)
        self.assertEqual(metrics.undeparted_vehicles, 8)
        self.assertAlmostEqual(metrics.demand_normalized_waiting_s, 6.9)
        self.assertIsNone(metrics.avg_waiting_time_s)
        self.assertEqual(manifest.completion_status, "complete")

    def test_zero_arrival_comparison_keeps_primary_and_marks_secondary_unavailable(self):
        baseline = EpisodeMetrics("case", "fixed", throughput=0, demand_normalized_waiting_s=8.0,
                                  total_co2_mg=0.0)
        candidate = EpisodeMetrics("case", "random", throughput=0, demand_normalized_waiting_s=6.0,
                                   total_co2_mg=5.0)
        result = candidate.improvement_vs(baseline)
        self.assertEqual(result["demand_normalized_waiting_absolute_delta"], -2.0)
        self.assertEqual(result["demand_normalized_waiting_pct"], 25.0)
        self.assertIsNone(result["avg_wait_absolute_delta"])
        self.assertIn("unavailable", result["avg_wait_pct_unavailable_reason"])
        self.assertEqual(result["total_co2_absolute_delta"], 5.0)
        self.assertIsNone(result["total_co2_pct"])

    def test_missing_primary_rejects_comparison(self):
        with self.assertRaises(IncomparableRunsError):
            EpisodeMetrics("case", "random", throughput=1).improvement_vs(
                EpisodeMetrics("case", "fixed", throughput=1)
            )

    def test_mixed_arrivals_keep_one_sided_secondary_unavailable(self):
        baseline = EpisodeMetrics("case", "fixed", throughput=0, demand_normalized_waiting_s=4.0,
                                  avg_waiting_time_s=None, total_co2_mg=10.0)
        candidate = EpisodeMetrics("case", "random", throughput=2, demand_normalized_waiting_s=3.0,
                                   avg_waiting_time_s=2.0, total_co2_mg=8.0)
        result = candidate.improvement_vs(baseline)
        self.assertEqual(result["throughput_delta"], 2)
        self.assertIsNone(result["avg_wait_absolute_delta"])
        self.assertEqual(result["total_co2_absolute_delta"], -2.0)

    def test_nonzero_begin_uses_half_open_interval_and_empty_is_undefined(self):
        route = self._write("window.rou.xml", '<routes><vehicle id="old" depart="4"/>'
            '<vehicle id="begin" depart="5"/><vehicle id="inside" depart="9"/>'
            '<vehicle id="end" depart="10"/></routes>')
        trip = self._write("trip.xml", '<tripinfos><tripinfo id="begin" depart="5" arrival="7" waitingTime="0" duration="2" timeLoss="0"/>'
            '<tripinfo id="inside" depart="9" arrival="-1" waitingTime="0" vaporized="end"/>'
            '<tripinfo id="end" depart="-1" arrival="-1" waitingTime="0" vaporized="end"/></tripinfos>')
        emission = self._write("emission.xml", '<emission-export/>')
        summary = self._write("summary.xml", '<summary><step time="5" arrived="0" ended="0" inserted="1" running="1" waiting="0"/>'
            '<step time="9" arrived="1" ended="1" inserted="2" running="1" waiting="0"/></summary>')
        metrics = load_metrics_from_paths(tripinfo_path=trip, emission_path=emission, summary_path=summary,
            profile="case", mode="fixed", expected_begin_s=5, expected_end_s=10,
            route_paths=[route], require_demand_contract=True)
        self.assertTrue(metrics.valid, metrics.errors)
        self.assertEqual((metrics.scheduled_demand, metrics.inserted_vehicles), (2, 2))
        empty_route = self._write("empty.rou.xml", '<routes><vehicle id="old" depart="4"/></routes>')
        empty_trip = self._write("empty-trip.xml", '<tripinfos/>')
        empty_summary = self._write("empty-summary.xml", '<summary><step time="5" arrived="0" ended="0" inserted="0" running="0" waiting="0"/>'
            '<step time="9" arrived="0" ended="0" inserted="0" running="0" waiting="0"/></summary>')
        empty = load_metrics_from_paths(tripinfo_path=empty_trip, emission_path=emission, summary_path=empty_summary,
            profile="empty", mode="fixed", expected_begin_s=5, expected_end_s=10,
            route_paths=[empty_route], require_demand_contract=True)
        self.assertTrue(empty.valid, empty.errors)
        self.assertEqual(empty.scheduled_demand, 0)
        self.assertIsNone(empty.demand_normalized_waiting_s)

    def test_large_horizon_fractional_near_end_is_eligible_and_exact_end_is_boundary(self):
        route = self._write("large-window.rou.xml", '<routes>'
            '<vehicle id="near" depart="71999.99995"/>'
            '<vehicle id="exact" depart="72000"/>'
            '<vehicle id="after" depart="72000.00005"/></routes>')
        trip = self._write("large-trip.xml", '<tripinfos>'
            '<tripinfo id="near" depart="71999.99995" arrival="-1" waitingTime="0" vaporized="end"/>'
            '<tripinfo id="exact" depart="-1" arrival="-1" waitingTime="0" vaporized="end"/>'
            '</tripinfos>')
        emission = self._write("large-emission.xml", '<emission-export/>')
        summary = self._write("large-summary.xml", '<summary>'
            '<step time="71900" arrived="0" ended="0" inserted="0" running="0" waiting="0"/>'
            '<step time="71999" arrived="0" ended="0" inserted="1" running="1" waiting="0"/>'
            '</summary>')
        metrics = load_metrics_from_paths(
            tripinfo_path=trip, emission_path=emission, summary_path=summary,
            profile="large", mode="fixed", expected_begin_s=71900, expected_end_s=72000,
            route_paths=[route], require_demand_contract=True,
        )
        self.assertTrue(metrics.valid, metrics.errors)
        self.assertEqual(metrics.scheduled_demand, 1)
        self.assertEqual(metrics.inserted_vehicles, 1)
        self.assertEqual(metrics.undeparted_vehicles, 0)
        self.assertAlmostEqual(metrics.demand_normalized_waiting_s, 0.0)

    def test_paired_baseline_manifest_name_matches_runner(self):
        self.assertEqual(_paired_baseline_manifest_path(self.root, "case", 7, 2).name,
                         "manifest_fixed_time_case_seed7_run2.json")

    def test_real_sumo_uses_half_open_begin_end_interval(self):
        config = ROOT / "DeepQN/tests/fixtures/demand_window_sumo/window.sumocfg"
        trip = self.root / "window-trip.xml"
        summary = self.root / "window-summary.xml"
        result = subprocess.run(["sumo", "-c", str(config), "--tripinfo-output", str(trip),
            "--summary-output", str(summary), "--tripinfo-output.write-unfinished", "true",
            "--tripinfo-output.write-undeparted", "true", "--no-step-log", "true"],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        records = {row.get("id"): row for row in ET.parse(trip).iter("tripinfo")}
        self.assertNotIn("before_begin", records)
        self.assertIn("at_begin", records)
        self.assertIn("before_end", records)
        self.assertEqual(records["at_end"].get("depart"), "-1")
        self.assertNotIn("after_end", records)
        last = list(ET.parse(summary).iter("step"))[-1]
        self.assertEqual((last.get("time"), last.get("inserted")), ("9.00", "2"))


class ForecastContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mapping = load_approach_mapping(_mapping_file(self.root), TINY_NET)

    def tearDown(self):
        self.temp.cleanup()

    def test_committed_tahrir_mapping_validates_against_network(self):
        mapping = load_approach_mapping(
            ROOT / "DeepQN/configs/forecast_mappings/tahrir_v1.json",
            ROOT / "simulation/maps/tahrirupdated.net.xml",
        )
        self.assertEqual(len(mapping.tls_directions), 7)
        self.assertEqual(len(mapping.sha256), 64)

    def test_feature_order_and_crossing_is_counted_once_while_vehicle_remains(self):
        self.assertEqual(FEATURE_NAMES, ("north", "south", "east", "west", "total", "avg_speed", "avg_waiting"))
        extractor = JunctionFeatureExtractor(self.mapping)
        first = extractor.extract(TrafficSnapshot(0, (
            VehicleObservation("v", "n_in", 6, 3),
        )))["center"]
        second = extractor.extract(TrafficSnapshot(1, (
            VehicleObservation("v", "n_in", 4, 5),
        )))["center"]
        self.assertEqual(first, (0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 3.0))
        self.assertEqual(second, (0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0))

    def test_reentry_after_absence_is_a_new_inferred_event(self):
        extractor = JunctionFeatureExtractor(self.mapping)
        self.assertEqual(extractor.extract(TrafficSnapshot(0, (VehicleObservation("v", "n_in", 1, 0),)))["center"][1], 0)
        extractor.extract(TrafficSnapshot(1, ()))
        self.assertEqual(extractor.extract(TrafficSnapshot(2, (VehicleObservation("v", "n_in", 1, 0),)))["center"][1], 1)

    def test_invalid_edge_and_duplicate_boundary_are_rejected(self):
        invalid = _mapping_file(self.root, {
            "center": {"north": ["missing"], "south": [], "east": [], "west": []}
        })
        with self.assertRaises(ForecastContractError):
            load_approach_mapping(invalid, TINY_NET)
        duplicate = _mapping_file(self.root, {
            "center": {"north": ["s_in"], "south": ["s_in"], "east": [], "west": []}
        })
        with self.assertRaises(ForecastContractError):
            load_approach_mapping(duplicate, TINY_NET)

    def test_future_arrivals_never_enter_history(self):
        sources_a = [_source("train-a", "train", 70), _source("val-a", "validation", 70), _source("test-a", "test", 70)]
        sources_b = [_source("train-b", "train", 90), _source("val-b", "validation", 90), _source("test-b", "test", 90)]
        a = build_causal_windows(sources_a, self.mapping)["train"]
        b = build_causal_windows(sources_b, self.mapping)["train"]
        # The first window observes t=0..59. Changing an event at t=70 to t=90
        # changes its future target but cannot change the input tensor.
        np.testing.assert_array_equal(a.x[0], b.x[0])
        self.assertFalse(np.array_equal(a.y[0], b.y[0]))
        self.assertGreater(a.windows[0].target_start_s, a.windows[0].input_end_s)

    def test_partition_content_reuse_is_rejected(self):
        train = _source("train", "train", 70)
        validation = _source("validation", "validation", 70)
        validation = SequenceSource(
            validation.source_id, validation.partition, validation.snapshots,
            validation.demand_seed, validation.sumo_config_sha256, train.source_sha256,
            validation.demand_realization_id, validation.demand_sha256,
        )
        with self.assertRaisesRegex(ForecastContractError, "reused"):
            build_causal_windows([train, validation, _source("test", "test", 70)], self.mapping)

    def test_demand_realization_cannot_cross_partitions_when_snapshots_differ(self):
        train = _source("train", "train", 70)
        validation = _source("validation", "validation", 75)
        validation = SequenceSource(validation.source_id, validation.partition, validation.snapshots,
            validation.demand_seed, validation.sumo_config_sha256, validation.source_sha256,
            train.demand_realization_id, train.demand_sha256)
        with self.assertRaisesRegex(ForecastContractError, "demand realization (id has conflicting|reused)"):
            build_causal_windows([train, validation, _source("test", "test", 80)], self.mapping)

    def test_scaler_is_fit_from_training_and_schema_checked(self):
        partitions = build_causal_windows([
            _source("train", "train", 70),
            _source("validation", "validation", 75),
            _source("test", "test", 80),
        ], self.mapping)
        scaler = fit_training_scaler(partitions["train"].x)
        transformed = transform_features(partitions["test"].x, scaler)
        self.assertTrue(np.all(np.isfinite(transformed)))
        incompatible = dict(scaler, features=list(reversed(scaler["features"])))
        with self.assertRaises(ForecastContractError):
            transform_features(partitions["test"].x, incompatible)

    def test_dataset_manifest_hashes_schema_mapping_sources_and_outputs(self):
        sources = [
            _source("train", "train", 70),
            _source("validation", "validation", 75),
            _source("test", "test", 80),
        ]
        partitions = build_causal_windows(sources, self.mapping)
        output = self.root / "dataset"
        manifest = write_dataset(output, partitions, sources, self.mapping)
        self.assertEqual(manifest["mapping_sha256"], self.mapping.sha256)
        self.assertEqual(manifest["network_sha256"], self.mapping.network_sha256)
        self.assertEqual({row["partition"] for row in manifest["sources"]}, {"train", "validation", "test"})
        self.assertEqual(set(manifest["artifact_sha256"]), {"train.npz", "validation.npz", "test.npz", "scaler.json"})
        self.assertFalse(manifest["publication_training_ready"])
        with self.assertRaisesRegex(ForecastContractError, "not approved"):
            require_publication_training_ready(manifest)
        with np.load(output / "train.npz") as exported:
            required = {"source_id", "tls_id", "input_start_s", "input_end_s",
                        "target_start_s", "target_end_s", "partition"}
            self.assertTrue(required.issubset(exported.files))
            self.assertTrue(np.all(exported["partition"] == "train"))
        with self.assertRaises(FileExistsError):
            write_dataset(output, partitions, sources, self.mapping)


if __name__ == "__main__":
    unittest.main()
