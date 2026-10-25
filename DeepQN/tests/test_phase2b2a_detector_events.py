from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from DeepQN.forecasting.contracts import (
    ForecastContractError, TrafficSnapshot, VehicleObservation, load_approach_mapping,
)
from DeepQN.forecasting.detector_events import (
    ApproachEntryEvent, E1EventCollector, build_detector_mapping,
    load_detector_events, validate_detector_events, write_detector_events,
)
from lstm.research_data import (
    SequenceSource, build_causal_windows, generate_dataset_from_manifest,
    load_snapshot_jsonl, require_publication_training_ready, write_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "DeepQN/tests/fixtures/detector_sumo"
NET = ROOT / "DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml"


class DetectorEventTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        self.mapping = build_detector_mapping(approaches, NET)

    def tearDown(self):
        self.temp.cleanup()

    def _file_backed_source(self, partition: str, index: int,
                            events: tuple[ApproachEntryEvent, ...]) -> SequenceSource:
        snapshot_path = self.root / f"{partition}.jsonl"
        rows = []
        for time_s in range(95):
            rows.append(json.dumps({"time_s": time_s, "vehicles": [{
                "id": f"present-{partition}", "edge": "s_in",
                "speed_m_s": 5.0, "accumulated_waiting_s": 2.0,
            }]}, sort_keys=True))
        snapshot_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        config_path = self.root / f"{partition}.sumocfg"
        config_path.write_text('<configuration><processing><waiting-time-memory value="100"/>'
                               '</processing></configuration>', encoding="utf-8")
        demand_path = self.root / f"{partition}.rou.xml"
        demand_path.write_text(f'<routes><vehicle id="d-{partition}" depart="0"/></routes>',
                               encoding="utf-8")
        event_path = self.root / f"{partition}.events.json"
        event_hash = write_detector_events(event_path, events, self.mapping, partition)
        return SequenceSource(
            partition, partition, load_snapshot_jsonl(snapshot_path), index,
            hashlib.sha256(config_path.read_bytes()).hexdigest(),
            hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            f"demand-{index}", hashlib.sha256(demand_path.read_bytes()).hexdigest(),
            detector_events=events, detector_mapping_sha256=self.mapping.sha256,
            snapshot_path=str(snapshot_path), sumo_config_path=str(config_path),
            demand_path=str(demand_path), event_source_path=str(event_path),
            event_source_sha256=event_hash,
        )

    def test_mapping_places_one_valid_detector_per_lane(self):
        self.assertEqual(len(self.mapping.specs), 4)
        self.assertTrue(all(spec.position_m == .1 and spec.lane_length_m > .1
                            for spec in self.mapping.specs))
        self.assertFalse(self.mapping.traffic_engineering_validated)

    def test_tahrir_mapping_generates_valid_lane_specific_detectors(self):
        approaches = load_approach_mapping(
            ROOT / "DeepQN/configs/forecast_mappings/tahrir_v1.json",
            ROOT / "simulation/maps/tahrirupdated.net.xml",
        )
        mapping = build_detector_mapping(approaches, ROOT / "simulation/maps/tahrirupdated.net.xml")
        self.assertGreaterEqual(len(mapping.specs), 10)
        self.assertEqual(len({spec.detector_id for spec in mapping.specs}), len(mapping.specs))
        self.assertFalse(mapping.traffic_engineering_validated)

    def test_collector_deduplicates_repeated_response_but_allows_later_reentry(self):
        collector = E1EventCollector(self.mapping)
        detector = self.mapping.specs[0]
        first = ("v", 5.0, 0.25, -1.0, "car")
        self.assertEqual(len(collector.collect({detector.detector_id: [first]})), 1)
        self.assertEqual(collector.collect({detector.detector_id: [first]}), ())
        later = ("v", 5.0, 4.25, 4.6, "car")
        self.assertEqual(len(collector.collect({detector.detector_id: [later]})), 1)

    def test_ambiguous_same_tls_multilane_duplicate_fails_closed(self):
        # Construct two lane reports with the same TLS/vehicle/time. This is
        # treated as a lane-change ambiguity, never two approach arrivals.
        specs = list(self.mapping.specs)
        same_direction = specs[0].__class__(
            specs[1].detector_id, specs[0].tls_id, specs[0].direction,
            specs[1].edge_id, specs[1].lane_id, specs[1].position_m, specs[1].lane_length_m,
        )
        altered = self.mapping.__class__(self.mapping.schema_version,
            (specs[0], same_direction), self.mapping.sha256, self.mapping.network_sha256)
        collector = E1EventCollector(altered)
        with self.assertRaisesRegex(ForecastContractError, "ambiguous multi-lane"):
            collector.collect({
                specs[0].detector_id: [("v", 5, .2, .3, "car")],
                same_direction.detector_id: [("v", 5, .2, .4, "car")],
            })
        # The failed batch is atomic: its first row was not remembered.
        accepted = collector.collect({specs[0].detector_id: [("v", 5, .2, .3, "car")]})
        self.assertEqual(len(accepted), 1)

        offline_events = tuple(sorted((
            ApproachEntryEvent(specs[0].detector_id, specs[0].tls_id,
                               specs[0].direction, specs[0].lane_id, "offline", .5, .6),
            ApproachEntryEvent(same_direction.detector_id, same_direction.tls_id,
                               same_direction.direction, same_direction.lane_id,
                               "offline", .5, .7),
        ), key=lambda event: (event.entry_time_s, event.detector_id,
                              event.vehicle_id, event.leave_time_s)))
        with self.assertRaisesRegex(ForecastContractError, "ambiguous multi-lane"):
            validate_detector_events(offline_events, altered)
        output = self.root / "ambiguous.events.json"
        with self.assertRaisesRegex(ForecastContractError, "ambiguous multi-lane"):
            write_detector_events(output, offline_events, altered, "offline")
        self.assertFalse(output.exists())

    def test_real_sumo_e1_matches_instant_enter_events(self):
        result_path = self.root / "result.json"
        completed = subprocess.run([
            sys.executable, "-m", "DeepQN.forecasting.validate_detector_fixture",
            str(FIXTURE / "detector.sumocfg"), str(FIXTURE / "mapping.json"),
            str(NET), str(self.root / "outputs"), str(result_path),
        ], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        events = payload["events"]
        enters = [row for row in payload["instant_events"] if row["state"] == "enter"]
        leaves = [row for row in payload["instant_events"] if row["state"] == "leave"]
        self.assertEqual(len(events), 4)
        self.assertEqual({row["vehicle_id"] for row in events},
                         {"north0", "south0", "east0", "west0"})
        self.assertNotIn("at_end", {row["vehicle_id"] for row in events})
        self.assertEqual(len(enters), 4)
        self.assertEqual(len(leaves), 4)
        self.assertTrue(all(row["entry_time_s"] == 0.0 for row in events))
        self.assertTrue(all(0 <= float(row["time"]) <= 1 for row in leaves))
        self.assertIn("1.00", {row["time"] for row in leaves})  # one vehicle remained on its loop
        first = payload["feature_rows"][0]
        self.assertEqual(first["time_s"], 1.0)
        self.assertEqual(sum(first["rows"]["center"][:4]), 4.0)
        self.assertEqual(payload["waiting_time_memory_s"], 100.0)
        self.assertEqual(len(payload["event_source_sha256"]), 64)

    def test_detector_events_drive_causal_targets_and_manifest_stays_fail_closed(self):
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        sources = []
        for index, partition in enumerate(("train", "validation", "test"), 1):
            spec = next(spec for spec in self.mapping.specs if spec.direction == "north")
            event = ApproachEntryEvent(spec.detector_id, spec.tls_id, spec.direction,
                spec.lane_id, f"cross-{partition}", 70.25, 70.4)
            sources.append(self._file_backed_source(partition, index, (event,)))
        partitions = build_causal_windows(sources, approaches, self.mapping)
        self.assertEqual(partitions["train"].x[0, :, :4].sum(), 0)
        self.assertEqual(partitions["train"].y[0].tolist(), [1.0, 0.0, 0.0, 0.0])
        output = self.root / "dataset"
        manifest = write_dataset(output, partitions, sources, approaches, self.mapping)
        self.assertEqual(manifest["entry_event_definition"], "sumo-e1-approach-entry-v1")
        self.assertEqual(manifest["detector_mapping_sha256"], self.mapping.sha256)
        self.assertFalse(manifest["publication_training_ready"])
        with self.assertRaises(ForecastContractError):
            require_publication_training_ready(manifest)
        with np.load(output / "train.npz") as archive:
            self.assertEqual(archive["y"][0].tolist(), [1.0, 0.0, 0.0, 0.0])
        second = self.root / "dataset-repeat"
        repeated = write_dataset(second, partitions, sources, approaches, self.mapping)
        self.assertEqual(manifest["artifact_sha256"], repeated["artifact_sha256"])
        self.assertEqual(manifest["manifest_payload_sha256"], repeated["manifest_payload_sha256"])

    def test_all_modes_reject_irregular_sampling(self):
        base = self._file_backed_source("train", 1, ())
        validation = self._file_backed_source("validation", 2, ())
        test = self._file_backed_source("test", 3, ())
        cases = {
            "missing": tuple(base.snapshots[:20] + base.snapshots[21:]),
            "duplicate": tuple(base.snapshots[:2] + (base.snapshots[1],) + base.snapshots[2:]),
            "reversed": tuple(base.snapshots[:2] + (replace(base.snapshots[2], time_s=0.0),) + base.snapshots[3:]),
            "fractional-gap": tuple(base.snapshots[:2] + (replace(base.snapshots[2], time_s=2.5),) + base.snapshots[3:]),
            "nonfinite": tuple((replace(base.snapshots[0], time_s=float("nan")),) + base.snapshots[1:]),
        }
        for label, snapshots in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ForecastContractError, "timestamp"):
                build_causal_windows([replace(base, snapshots=snapshots), validation, test],
                                     load_approach_mapping(FIXTURE / "mapping.json", NET), self.mapping)
        inferred = replace(base, detector_events=None, detector_mapping_sha256=None,
                           event_source_path=None, event_source_sha256=None)
        inferred_validation = replace(validation, detector_events=None, detector_mapping_sha256=None,
                                      event_source_path=None, event_source_sha256=None)
        inferred_test = replace(test, detector_events=None, detector_mapping_sha256=None,
                                event_source_path=None, event_source_sha256=None)
        with self.assertRaisesRegex(ForecastContractError, "timestamp"):
            build_causal_windows([
                replace(inferred, snapshots=cases["missing"]), inferred_validation, inferred_test
            ], load_approach_mapping(FIXTURE / "mapping.json", NET))

    def test_event_bins_boundaries_and_missing_bins(self):
        spec = self.mapping.specs[0]
        events = tuple(ApproachEntryEvent(spec.detector_id, spec.tls_id, spec.direction,
            spec.lane_id, f"v{i}", time_s, time_s + .01)
            for i, time_s in enumerate((0.0, .999999, 1.0, 1.000001)))
        train = self._file_backed_source("train", 1, events)
        validation = self._file_backed_source("validation", 2, ())
        test = self._file_backed_source("test", 3, ())
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        windows = build_causal_windows([train, validation, test], approaches, self.mapping)
        self.assertEqual(windows["train"].x[0, :3, :4].sum(), 4.0)
        missing = ApproachEntryEvent(spec.detector_id, spec.tls_id, spec.direction,
            spec.lane_id, "outside", 94.1, 94.2)
        bad = self._file_backed_source("outside", 4, (missing,))
        bad = replace(bad, partition="train")
        with self.assertRaisesRegex(ForecastContractError, "no recorded sample bin"):
            build_causal_windows([bad, validation, test], approaches, self.mapping)

    def test_malformed_unsorted_duplicate_and_wrong_owner_events_are_rejected(self):
        spec = self.mapping.specs[0]
        first = ApproachEntryEvent(spec.detector_id, spec.tls_id, spec.direction,
                                   spec.lane_id, "a", .2, .3)
        later = replace(first, vehicle_id="b", entry_time_s=.1, leave_time_s=.2)
        with self.assertRaisesRegex(ForecastContractError, "deterministic order"):
            validate_detector_events((first, later), self.mapping)
        with self.assertRaisesRegex(ForecastContractError, "duplicate"):
            validate_detector_events((first, first), self.mapping)
        with self.assertRaisesRegex(ForecastContractError, "ownership"):
            validate_detector_events((replace(first, direction="west"),), self.mapping)
        with self.assertRaisesRegex(ForecastContractError, "timestamp"):
            validate_detector_events((replace(first, entry_time_s=float("inf")),), self.mapping)

    def test_manifest_generator_loads_events_and_rejects_modified_file(self):
        spec_rows = []
        for index, partition in enumerate(("train", "validation", "test"), 1):
            detector = self.mapping.specs[0]
            event = ApproachEntryEvent(detector.detector_id, detector.tls_id, detector.direction,
                detector.lane_id, f"v-{partition}", 70.25, 70.4)
            source = self._file_backed_source(partition, index, (event,))
            spec_rows.append({
                "source_id": partition, "partition": partition,
                "path": Path(source.snapshot_path).name,
                "source_sha256": source.source_sha256,
                "sumo_config_path": Path(source.sumo_config_path).name,
                "sumo_config_sha256": source.sumo_config_sha256,
                "demand_path": Path(source.demand_path).name,
                "demand_sha256": source.demand_sha256,
                "demand_realization_id": source.demand_realization_id,
                "demand_seed": source.demand_seed,
                "event_path": Path(source.event_source_path).name,
                "event_source_sha256": source.event_source_sha256,
                "waiting_time_memory_s": 100,
            })
        approach_path = FIXTURE / "mapping.json"
        payload = {
            "mapping_path": str(approach_path.resolve()), "network_path": str(NET.resolve()),
            "mapping_sha256": load_approach_mapping(approach_path, NET).sha256,
            "network_sha256": self.mapping.network_sha256,
            "detector_mapping_sha256": self.mapping.sha256,
            "event_source_mode": "sumo_e1", "output_dir": "generated", "sources": spec_rows,
        }
        spec_path = self.root / "dataset-spec.json"
        spec_path.write_text(json.dumps(payload), encoding="utf-8")
        manifest = generate_dataset_from_manifest(spec_path)
        self.assertEqual(manifest["sources"][0]["event_source_sha256"],
                         spec_rows[0]["event_source_sha256"])
        event_path = self.root / spec_rows[0]["event_path"]
        event_path.write_text(event_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        payload["output_dir"] = "changed-output"
        spec_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ForecastContractError, "file hash.*changed"):
            generate_dataset_from_manifest(spec_path)

    def test_waiting_memory_and_mixed_event_modes_fail_closed(self):
        train = self._file_backed_source("train", 1, ())
        validation = self._file_backed_source("validation", 2, ())
        test = self._file_backed_source("test", 3, ())
        config = Path(train.sumo_config_path)
        config.write_text('<configuration><processing><waiting-time-memory value="99"/>'
                          '</processing></configuration>', encoding="utf-8")
        train = replace(train, sumo_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        with self.assertRaisesRegex(ForecastContractError, "waiting-time-memory"):
            build_causal_windows([train, validation, test], approaches, self.mapping)
        train = replace(validation, source_id="inferred-train", partition="train",
                        detector_events=None, detector_mapping_sha256=None,
                        event_source_path=None, event_source_sha256=None,
                        demand_realization_id="inferred-demand",
                        demand_sha256=hashlib.sha256(b"inferred-demand").hexdigest(),
                        source_sha256=hashlib.sha256(b"inferred-snapshot").hexdigest())
        with self.assertRaisesRegex(ForecastContractError, "event-source mode"):
            build_causal_windows([train, validation, test], approaches, self.mapping)

    def test_mutable_mapping_boolean_cannot_enable_publication_readiness(self):
        detector = self.mapping.specs[0]
        sources = []
        for index, partition in enumerate(("train", "validation", "test"), 1):
            event = ApproachEntryEvent(detector.detector_id, detector.tls_id, detector.direction,
                detector.lane_id, f"v-{partition}", 70.25, 70.4)
            sources.append(self._file_backed_source(partition, index, (event,)))
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        mutable_true = replace(self.mapping, traffic_engineering_validated=True)
        partitions = build_causal_windows(sources, approaches, mutable_true)
        manifest = write_dataset(self.root / "boolean-not-approval", partitions, sources,
                                 approaches, mutable_true)
        self.assertFalse(manifest["publication_training_ready"])

    def _publication_manifest(self, approval: dict | None = None):
        sources = [self._file_backed_source(partition, index, ())
                   for index, partition in enumerate(("train", "validation", "test"), 1)]
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        partitions = build_causal_windows(sources, approaches, self.mapping)
        output = self.root / f"publication-{len(list(self.root.glob('publication-*')))}"
        manifest = write_dataset(
            output, partitions, sources, approaches, self.mapping,
            mapping_approval=approval, mapping_path=FIXTURE / "mapping.json",
            network_path=NET,
        )
        return output, manifest

    @staticmethod
    def _force_ready(output: Path, manifest: dict) -> dict:
        altered = dict(manifest)
        altered["publication_training_ready"] = True
        altered.pop("manifest_payload_sha256", None)
        canonical = json.dumps(altered, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode()
        altered["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        (output / "manifest.json").write_text(
            json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return altered

    def test_publication_gate_rejects_boolean_fabrication_and_missing_evidence(self):
        with self.assertRaisesRegex(ForecastContractError, "actual manifest"):
            require_publication_training_ready({"publication_training_ready": True})

        missing = self.root / "missing-approval.json"
        approval = {
            "evidence_id": "invented", "evidence_path": str(missing),
            "evidence_sha256": "0" * 64,
            "approved_mapping_sha256": self.mapping.sha256,
        }
        output, manifest = self._publication_manifest(approval)
        manifest = self._force_ready(output, manifest)
        with self.assertRaisesRegex(ForecastContractError, "evidence file is missing"):
            require_publication_training_ready(manifest, output / "manifest.json")

    def test_publication_gate_rejects_untrusted_or_altered_approval(self):
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        evidence = self.root / "approval.json"
        evidence_payload = {
            "schema_version": "traffic-engineering-mapping-approval-v1",
            "review_status": "approved", "evidence_id": "self-asserted",
            "reviewer_identity": "fabricated-test-reviewer",
            "approach_mapping_sha256": approaches.sha256,
            "detector_mapping_sha256": self.mapping.sha256,
            "network_sha256": approaches.network_sha256,
        }
        evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")
        evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
        approval = {
            "evidence_id": "self-asserted", "evidence_path": str(evidence),
            "evidence_sha256": evidence_hash,
            "approved_mapping_sha256": self.mapping.sha256,
        }
        output, manifest = self._publication_manifest(approval)
        manifest = self._force_ready(output, manifest)
        with self.assertRaisesRegex(ForecastContractError, "reviewed trust root"):
            require_publication_training_ready(manifest, output / "manifest.json")
        evidence.write_text(json.dumps({**evidence_payload, "review_status": "withdrawn"}),
                            encoding="utf-8")
        with self.assertRaisesRegex(ForecastContractError, "evidence is altered"):
            require_publication_training_ready(manifest, output / "manifest.json")

    def test_publication_gate_rejects_incorrect_mapping_hash(self):
        output, manifest = self._publication_manifest()
        manifest["detector_mapping_sha256"] = "f" * 64
        manifest = self._force_ready(output, manifest)
        with self.assertRaisesRegex(ForecastContractError, "detector mapping identity"):
            require_publication_training_ready(manifest, output / "manifest.json")

    def test_detector_snapshots_must_match_verified_file_in_every_field(self):
        train = self._file_backed_source("train", 1, ())
        validation = self._file_backed_source("validation", 2, ())
        test = self._file_backed_source("test", 3, ())
        approaches = load_approach_mapping(FIXTURE / "mapping.json", NET)
        original = train.snapshots[0]
        vehicle = original.vehicles[0]
        mutations = {
            "speed": replace(vehicle, speed_m_s=42.0),
            "waiting": replace(vehicle, accumulated_waiting_s=99.0),
            "edge": replace(vehicle, edge_id="n_in"),
            "identity": replace(vehicle, vehicle_id="substituted"),
        }
        for label, changed_vehicle in mutations.items():
            changed = replace(original, vehicles=(changed_vehicle,))
            snapshots = (changed,) + train.snapshots[1:]
            with self.subTest(label=label), self.assertRaisesRegex(
                    ForecastContractError, "in-memory snapshots differ"):
                build_causal_windows([replace(train, snapshots=snapshots), validation, test],
                                     approaches, self.mapping)
        shifted = replace(original, time_s=.5)
        with self.assertRaisesRegex(ForecastContractError, "timestamp"):
            build_causal_windows([
                replace(train, snapshots=(shifted,) + train.snapshots[1:]), validation, test
            ], approaches, self.mapping)


if __name__ == "__main__":
    unittest.main()
