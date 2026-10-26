import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from DeepQN.evaluation.demand_provenance import (
    DemandContractError,
    SPEC_SCHEMA,
    bind_realization_to_run,
    generate_demand_realization,
    load_realization_manifest,
    proposed_seed_schedule,
    sha256_file,
    verify_split_registry,
    write_split_registry,
)
from DeepQN.evaluation.provenance import RunManifest
from DeepQN.evaluation.evaluate import _patch_sumocfg


class DemandProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.network = self.root / "network.net.xml"
        self.network.write_text(
            """<?xml version="1.0"?><net>
<edge id="a"><lane id="a_0" index="0" speed="10" length="100"/></edge>
<edge id="b"><lane id="b_0" index="0" speed="10" length="100"/></edge>
<edge id="c"><lane id="c_0" index="0" speed="10" length="100"/></edge>
<connection from="a" to="b" fromLane="0" toLane="0"/>
<connection from="b" to="c" fromLane="0" toLane="0"/>
</net>\n""", encoding="utf-8")
        self.types = self.root / "types.add.xml"
        self.types.write_text('<additional><vType id="car" vClass="passenger"/></additional>\n',
                              encoding="utf-8")
        self.source = self.root / "source.rou.xml"
        vehicles = "".join(
            f'<vehicle id="v{i}" type="car" depart="{10+i}"><route edges="a b c"/></vehicle>'
            for i in range(12)
        )
        self.source.write_text(f'<routes>{vehicles}</routes>\n', encoding="utf-8")
        self.config = self.root / "case.sumocfg"
        self.config.write_text(
            '<configuration><input><net-file value="network.net.xml"/>'
            '<route-files value="source.rou.xml"/><additional-files value="types.add.xml"/>'
            '</input><time><begin value="10"/><end value="40"/></time></configuration>\n',
            encoding="utf-8")
        self.spec_path = self.root / "spec.json"
        self.spec = {
            "schema": SPEC_SCHEMA, "profile": "synthetic",
            "begin_s": 10, "end_s": 40,
            "source_routes": self._identity(self.source),
            "network": self._identity(self.network),
            "configuration": self._identity(self.config),
            "additional_files": [self._identity(self.types)],
            "origin_evidence": "synthetic engineering fixture",
            "randomization": {
                "evidence_status": "provisional_simulation_assumption",
                "departure_jitter_s": {"distribution": "discrete_uniform_seconds",
                                       "minimum": 0, "maximum": 5},
                "vehicle_retention_probability": 0.8,
                "od_policy": "preserve", "vehicle_type_policy": "preserve",
            },
        }
        self._write_spec(self.spec)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _identity(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _write_spec(self, value):
        self.spec_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
        return sha256_file(self.spec_path)

    def _generate(self, identity="r1", seed=17, directory=None):
        target = directory or self.root / identity
        return generate_demand_realization(
            self.spec_path, sha256_file(self.spec_path), seed=seed,
            realization_id=identity, output_dir=target,
        )

    def test_byte_identical_regeneration_and_distinct_seed(self):
        first = self._generate("same", 17, self.root / "one")
        second = self._generate("same", 17, self.root / "two")
        self.assertEqual((self.root / "one/routes.rou.xml").read_bytes(),
                         (self.root / "two/routes.rou.xml").read_bytes())
        self.assertEqual(first["route_sha256"], second["route_sha256"])
        third = self._generate("different", 18, self.root / "three")
        self.assertNotEqual(first["demand_fingerprint_sha256"],
                            third["demand_fingerprint_sha256"])
        self.assertFalse(first["publication_approved"])

    def test_output_nonoverwrite_and_publication_gate(self):
        self._generate()
        with self.assertRaises(FileExistsError):
            self._generate()
        with self.assertRaisesRegex(DemandContractError, "independent hash-bound approval"):
            generate_demand_realization(
                self.spec_path, sha256_file(self.spec_path), seed=2,
                realization_id="candidate", output_dir=self.root / "candidate",
                purpose="research_candidate",
            )

    def test_tampered_source_config_and_spec_hashes(self):
        original = self.source.read_text(encoding="utf-8")
        self.source.write_text(original.replace('depart="10"', 'depart="11"'), encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "source route file"):
            self._generate()
        self.source.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "specification hash mismatch"):
            generate_demand_realization(self.spec_path, "0" * 64, seed=1,
                                        realization_id="x", output_dir=self.root / "x")
        original_config = self.config.read_text(encoding="utf-8")
        self.config.write_text(original_config.replace('end value="40"', 'end value="41"'),
                               encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "configuration file"):
            self._generate()

    def test_invalid_routes_types_windows_and_duplicate_ids(self):
        cases = [
            (lambda text: text.replace('edges="a b c"', 'edges="a missing"', 1), "unknown network edge"),
            (lambda text: text.replace('edges="a b c"', 'edges="a c"', 1), "disconnected route pair"),
            (lambda text: text.replace('type="car"', 'type="unknown"', 1), "unknown type"),
            (lambda text: text.replace('depart="10"', 'depart="40"', 1), "outside"),
            (lambda text: text.replace('id="v1"', 'id="v0"', 1), "duplicate vehicle IDs"),
        ]
        original = self.source.read_text(encoding="utf-8")
        for index, (mutation, message) in enumerate(cases):
            with self.subTest(case=index):
                self.source.write_text(mutation(original), encoding="utf-8")
                spec = copy.deepcopy(self.spec)
                spec["source_routes"] = self._identity(self.source)
                self._write_spec(spec)
                with self.assertRaisesRegex(DemandContractError, message):
                    self._generate(f"bad-{index}")
        self.source.write_text(original, encoding="utf-8")
        self._write_spec(self.spec)

    def test_registry_rejects_seed_id_copy_leakage_and_conflicts(self):
        first = self._generate("r1", 17)
        second = self._generate("r2", 18)
        registry_path = self.root / "registry.json"
        registry = write_split_registry([
            {"manifest_path": self.root / "r1/realization_manifest.json", "partition": "training"},
            {"manifest_path": self.root / "r2/realization_manifest.json", "partition": "controller_evaluation"},
        ], registry_path)
        self.assertEqual(len(registry["realizations"]), 2)
        self.assertEqual(verify_split_registry(registry_path, sha256_file(registry_path)), registry)
        with self.assertRaises(FileExistsError):
            write_split_registry([], registry_path)
        with self.assertRaisesRegex(DemandContractError, "duplicate or conflicting"):
            write_split_registry([
                {"manifest_path": self.root / "r1/realization_manifest.json", "partition": "training"},
                {"manifest_path": self.root / "r1/realization_manifest.json", "partition": "training"},
            ], self.root / "duplicate-id.json")
        with self.assertRaisesRegex(DemandContractError, "invalid experimental partition"):
            write_split_registry([
                {"manifest_path": self.root / "r1/realization_manifest.json", "partition": "test"},
            ], self.root / "invalid-partition.json")

        copied = self.root / "copied"
        copied.mkdir()
        (copied / "routes.rou.xml").write_bytes((self.root / "r1/routes.rou.xml").read_bytes())
        (copied / "source_manifest.json").write_bytes(
            (self.root / "r1/source_manifest.json").read_bytes())
        copied_manifest = dict(first, realization_id="copied", route_path="routes.rou.xml",
                               demand_seed=999)
        (copied / "realization_manifest.json").write_text(
            json.dumps(copied_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "route-embedded provenance"):
            write_split_registry([
                {"manifest_path": self.root / "r1/realization_manifest.json", "partition": "training"},
                {"manifest_path": copied / "realization_manifest.json", "partition": "forecast_test"},
            ], self.root / "copy-registry.json")

        collision = self.root / "collision"
        self._generate("collision", 17, collision)
        with self.assertRaisesRegex(DemandContractError, "seed-to-ID collision"):
            write_split_registry([
                {"manifest_path": self.root / "r1/realization_manifest.json", "partition": "training"},
                {"manifest_path": collision / "realization_manifest.json", "partition": "calibration"},
            ], self.root / "collision-registry.json")

    def test_registry_detects_manifest_tampering_and_conflicting_window(self):
        self._generate("r1", 17)
        manifest_path = self.root / "r1/realization_manifest.json"
        registry_path = self.root / "registry.json"
        write_split_registry([{"manifest_path": manifest_path, "partition": "training"}], registry_path)
        expected = sha256_file(registry_path)
        source_bytes = self.source.read_bytes()
        self.source.write_bytes(source_bytes + b"<!--tampered-->\n")
        with self.assertRaisesRegex(DemandContractError, "source-manifest source_routes"):
            verify_split_registry(registry_path, expected)
        self.source.write_bytes(source_bytes)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["demand_seed"] = 99
        manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "conflicts"):
            verify_split_registry(registry_path, expected)

        registry_bytes = registry_path.read_bytes()
        registry_path.write_bytes(registry_bytes + b" ")
        with self.assertRaisesRegex(DemandContractError, "split registry is missing or changed"):
            verify_split_registry(registry_path, expected)
        registry_path.write_bytes(registry_bytes)

        # A second valid artifact with the same profile but a contradictory
        # declared window is rejected even if its route bytes differ.
        self._generate("r2", 18)
        second_path = self.root / "r2/realization_manifest.json"
        second = json.loads(second_path.read_text(encoding="utf-8"))
        second["end_s"] = 39
        second_path.write_text(json.dumps(second, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "route-embedded provenance"):
            write_split_registry([
                {"manifest_path": manifest_path, "partition": "training"},
                {"manifest_path": second_path, "partition": "validation"},
            ], self.root / "windows.json")

    def test_run_binding_requires_registered_exact_route_and_window(self):
        self._generate("r1", 17)
        realization_path = self.root / "r1/realization_manifest.json"
        registry_path = self.root / "registry.json"
        write_split_registry([{"manifest_path": realization_path,
                               "partition": "controller_evaluation"}], registry_path)
        route = self.root / "r1/routes.rou.xml"
        effective = _patch_sumocfg(
            self.config,
            {name: self.root / f"{name}.xml" for name in ("tripinfo", "emission", "summary")},
            seed=71, route_paths=[route],
        )
        manifest = RunManifest(
            "run", "fixed", "synthetic", 71, 10, 40, 1,
            str(self.spec_path), sha256_file(self.spec_path),
            str(self.config), sha256_file(self.config),
            str(self.network), sha256_file(self.network),
            {str(route): sha256_file(route)},
            additional_files={str(self.types.resolve()): sha256_file(self.types)},
        )
        bind_realization_to_run(
            manifest, realization_manifest_path=realization_path,
            split_registry_path=registry_path,
            expected_split_registry_sha256=sha256_file(registry_path),
            effective_config_path=effective, run_purpose="engineering_diagnostic",
        )
        self.assertEqual(manifest.demand_realization_id, "r1")
        self.assertEqual(manifest.demand_partition, "controller_evaluation")
        wrong = copy.deepcopy(manifest)
        wrong.route_files = {str(route): "0" * 64}
        for name in ("demand_realization_id", "demand_manifest_sha256", "demand_partition",
                     "split_registry_sha256", "demand_usage"):
            setattr(wrong, name, None)
        with self.assertRaisesRegex(DemandContractError, "route file or bytes"):
            bind_realization_to_run(
                wrong, realization_manifest_path=realization_path,
                split_registry_path=registry_path,
                expected_split_registry_sha256=sha256_file(registry_path),
                effective_config_path=effective, run_purpose="engineering_diagnostic",
            )
        self.assertEqual((wrong.demand_realization_id, wrong.demand_manifest_sha256,
                          wrong.demand_partition, wrong.split_registry_sha256,
                          wrong.demand_usage), (None, None, None, None, None))
        Path(effective).unlink()

    def test_seed_schedule_is_dry_run_collision_free_and_not_approved(self):
        schedule = proposed_seed_schedule()
        self.assertEqual(schedule["status"], "proposal_requires_protocol_approval")
        self.assertEqual(len(schedule["training"]), 10)
        self.assertEqual(len(schedule["controller_evaluation"]), 40)
        self.assertEqual({row["training_rng_seed"] for row in schedule["training"]},
                         set(range(11001, 11006)))
        identifiers = [row["evaluation_replicate_id"]
                       for row in schedule["controller_evaluation"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_manifest_replay_rejects_forged_identity_seed_count_and_approval(self):
        deterministic = copy.deepcopy(self.spec)
        deterministic["randomization"]["vehicle_retention_probability"] = 1.0
        deterministic["randomization"]["departure_jitter_s"] = {
            "distribution": "discrete_uniform_seconds", "minimum": 0, "maximum": 0,
        }
        self._write_spec(deterministic)
        first = self._generate("physical-a", 1)
        second = self._generate("physical-b", 2)
        self.assertNotEqual(first["route_sha256"], second["route_sha256"])
        self.assertEqual(first["demand_fingerprint_sha256"],
                         second["demand_fingerprint_sha256"])
        with self.assertRaisesRegex(DemandContractError, "physically identical demand"):
            write_split_registry([
                {"manifest_path": self.root / "physical-a/realization_manifest.json",
                 "partition": "training"},
                {"manifest_path": self.root / "physical-b/realization_manifest.json",
                 "partition": "forecast_test"},
            ], self.root / "physical-registry.json")

        manifest_path = self.root / "physical-b/realization_manifest.json"
        original = manifest_path.read_bytes()
        mutations = (
            ({"demand_fingerprint_sha256": "f" * 64}, "fingerprint"),
            ({"demand_seed": 999}, "route-embedded provenance"),
            ({"vehicle_count": 999}, "vehicle count"),
            ({"realization_id": "forged"}, "route-embedded provenance"),
            ({"purpose": "research_candidate", "publication_approved": True},
             "publication status"),
            ({"generation_parameters": {"forged": True}}, "deterministic replay"),
            ({"profile": "forged"}, "deterministic replay"),
            ({"end_s": 39}, "deterministic replay"),
        )
        for values, message in mutations:
            with self.subTest(values=values):
                data = json.loads(original)
                data.update(values)
                manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
                with self.assertRaisesRegex(DemandContractError, message):
                    load_realization_manifest(manifest_path)
                manifest_path.write_bytes(original)

    def test_registry_top_level_metadata_cannot_self_approve_or_change_policy(self):
        self._generate("r1", 17)
        manifest = self.root / "r1/realization_manifest.json"
        for field, value, message in (
            ("publication_approved", True, "cannot self-declare"),
            ("status", "draft", "status"),
            ("partition_policy", "caller decides later", "partition policy"),
        ):
            with self.subTest(field=field):
                path = self.root / f"registry-{field}.json"
                registry = write_split_registry(
                    [{"manifest_path": manifest, "partition": "training"}], path)
                registry[field] = value
                path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
                with self.assertRaisesRegex(DemandContractError, message):
                    verify_split_registry(path, sha256_file(path))
        path = self.root / "registry-extra.json"
        registry = write_split_registry(
            [{"manifest_path": manifest, "partition": "training"}], path)
        registry["approval_note"] = "fabricated"
        path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "unexpected top-level"):
            verify_split_registry(path, sha256_file(path))

    def test_multiple_source_routes_are_rejected(self):
        second = self.root / "extra.rou.xml"
        second.write_text('<routes><vehicle id="extra" type="car" depart="12">'
                          '<route edges="a b c"/></vehicle></routes>\n', encoding="utf-8")
        original = self.config.read_text(encoding="utf-8")
        self.config.write_text(original.replace(
            'route-files value="source.rou.xml"',
            'route-files value="source.rou.xml,extra.rou.xml"'), encoding="utf-8")
        spec = copy.deepcopy(self.spec)
        spec["configuration"] = self._identity(self.config)
        self._write_spec(spec)
        with self.assertRaisesRegex(DemandContractError, "only configured route"):
            self._generate("multi-route")

    def test_binding_rejects_substitutions_partition_misuse_and_is_atomic(self):
        self._generate("r1", 17)
        realization_path = self.root / "r1/realization_manifest.json"
        registry_path = self.root / "registry.json"
        write_split_registry([{"manifest_path": realization_path,
                               "partition": "controller_evaluation"}], registry_path)
        route = self.root / "r1/routes.rou.xml"
        effective = _patch_sumocfg(
            self.config,
            {name: self.root / f"{name}.xml" for name in ("tripinfo", "emission", "summary")},
            seed=71, route_paths=[route],
        )
        base = RunManifest(
            "run", "fixed", "synthetic", 71, 10, 40, 1,
            str(self.spec_path), sha256_file(self.spec_path),
            str(self.config), sha256_file(self.config),
            str(self.network), sha256_file(self.network),
            {str(route): sha256_file(route)},
            additional_files={str(self.types.resolve()): sha256_file(self.types)},
        )

        cases = []
        wrong_network = copy.deepcopy(base); wrong_network.network_sha256 = "0" * 64
        cases.append((wrong_network, effective, "engineering_diagnostic", "network"))
        wrong_config = copy.deepcopy(base); wrong_config.simulation_config_sha256 = "0" * 64
        cases.append((wrong_config, effective, "engineering_diagnostic", "source configuration"))
        wrong_additional = copy.deepcopy(base); wrong_additional.additional_files = {
            str(self.types.resolve()): "0" * 64}
        cases.append((wrong_additional, effective, "engineering_diagnostic", "additional-file"))
        wrong_profile = copy.deepcopy(base); wrong_profile.scenario = "other"
        cases.append((wrong_profile, effective, "engineering_diagnostic", "profile"))
        cases.append((copy.deepcopy(base), effective, "training", "ineligible"))

        altered_effective = self.root / "altered.sumocfg"
        altered_effective.write_bytes(Path(effective).read_bytes().replace(
            b'<seed value="71"', b'<seed value="72"'))
        cases.append((copy.deepcopy(base), altered_effective, "engineering_diagnostic", "SUMO seed"))
        for manifest, config, purpose, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DemandContractError, message):
                    bind_realization_to_run(
                        manifest, realization_manifest_path=realization_path,
                        split_registry_path=registry_path,
                        expected_split_registry_sha256=sha256_file(registry_path),
                        effective_config_path=config, run_purpose=purpose,
                    )
                self.assertIsNone(manifest.demand_realization_id)
                self.assertIsNone(manifest.demand_usage)
        Path(effective).unlink()


if __name__ == "__main__":
    unittest.main()
