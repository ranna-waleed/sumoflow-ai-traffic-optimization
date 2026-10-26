from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from DeepQN.evaluation.demand_expansion import (
    EXPANSION_ALGORITHM, EXPANSION_SPEC_SCHEMA, DemandContractError,
    expand_number_flows, verify_expansion,
)
from DeepQN.evaluation.metrics import parse_scheduled_demand
from DeepQN.evaluation.demand_provenance import _config_inputs


class NumberFlowExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.network = self.root / "tiny.net.xml"
        self.network.write_text(
            '<net><edge id="a"><lane id="a_0" index="0" speed="10" length="20"/></edge>'
            '<edge id="b"><lane id="b_0" index="0" speed="10" length="20"/></edge>'
            '<connection from="a" to="b" fromLane="0" toLane="0"/></net>\n',
            encoding="utf-8",
        )
        self.types = self.root / "types.add.xml"
        self.types.write_text('<additional><vType id="car"/></additional>\n', encoding="utf-8")
        self.source = self.root / "source.rou.xml"
        self.source.write_text(
            '<routes><route id="r" edges="a b"/>'
            '<flow id="f" type="car" route="r" begin="0" end="10" number="4" '
            'departLane="best" departSpeed="random"><stop lane="b_0" duration="2"/></flow>'
            '<vehicle id="explicit" type="car" depart="12"><route edges="a b"/></vehicle>'
            '</routes>\n', encoding="utf-8",
        )
        self.config = self.root / "tiny.sumocfg"
        self.config.write_text(
            '<configuration><input><net-file value="tiny.net.xml"/>'
            '<route-files value="source.rou.xml"/>'
            '<additional-files value="types.add.xml"/></input>'
            '<time><begin value="0"/><end value="20"/><step-length value="1"/></time>'
            '</configuration>\n', encoding="utf-8",
        )
        self.spec_path = self.root / "expansion-spec.json"
        self.spec = {
            "schema": EXPANSION_SPEC_SCHEMA,
            "expansion_id": "tiny-canonical-v1",
            "sumo_version": "1.26.0",
            "algorithm": EXPANSION_ALGORITHM,
            "source_routes": self._identity(self.source),
            "configuration": self._identity(self.config),
            "network": self._identity(self.network),
            "additional_files": [self._identity(self.types)],
            "begin_s": 0,
            "end_s": 20,
            "publication_approved": False,
        }
        self._write_spec()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _identity(self, path: Path) -> dict:
        return {"path": path.name, "sha256": self._digest(path)}

    def _write_spec(self):
        self.spec_path.write_text(json.dumps(self.spec, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")

    def _expand(self, name="expanded", **kwargs):
        return expand_number_flows(
            self.spec_path, self._digest(self.spec_path), output_dir=self.root / name, **kwargs
        )

    def _verify(self, name="expanded", *, spec_sha256=None, manifest_sha256=None):
        directory = self.root / name
        return verify_expansion(
            directory,
            expected_spec_sha256=spec_sha256 or self._digest(self.spec_path),
            expected_manifest_sha256=(manifest_sha256 or self._digest(
                directory / "expansion_manifest.json"
            )),
        )

    def test_identical_inputs_produce_identical_bytes_and_explicit_denominator(self):
        first = self._expand("one")
        second = self._expand("two")
        one = self.root / "one/routes.explicit.rou.xml"
        two = self.root / "two/routes.explicit.rou.xml"
        self.assertEqual(one.read_bytes(), two.read_bytes())
        self.assertEqual(first["output"], second["output"])
        self.assertEqual(first["counts"]["expanded_explicit_vehicles"], 5)
        due = parse_scheduled_demand([one], 0, 20)
        self.assertEqual(set(due), {"f.0", "f.1", "f.2", "f.3", "explicit"})
        self.assertEqual(self._verify("one"), first)

    def test_equal_spacing_is_begin_inclusive_end_exclusive(self):
        self._expand()
        root = ET.parse(self.root / "expanded/routes.explicit.rou.xml").getroot()
        times = {node.get("id"): node.get("depart") for node in root.findall("vehicle")}
        self.assertEqual([times[f"f.{i}"] for i in range(4)], ["0", "2.5", "5", "7.5"])
        self.assertNotIn("10", times.values())

    def test_vehicle_type_route_departure_attributes_and_stop_are_preserved(self):
        manifest = self._expand()
        root = ET.parse(self.root / "expanded/routes.explicit.rou.xml").getroot()
        vehicle = next(node for node in root.findall("vehicle") if node.get("id") == "f.0")
        self.assertEqual(vehicle.get("type"), "car")
        self.assertEqual(vehicle.get("departLane"), "best")
        self.assertEqual(vehicle.get("departSpeed"), "random")
        self.assertEqual(vehicle.find("route").get("edges"), "a b")
        self.assertEqual(vehicle.find("stop").attrib, {"lane": "b_0", "duration": "2"})
        self.assertEqual(manifest["runtime_randomness"], {
            "fields_preserved": ["departSpeed"],
            "requires_separate_recorded_sumo_seed": True,
        })

    def test_stochastic_rate_and_incomplete_route_flows_are_rejected(self):
        original = self.source.read_text(encoding="utf-8")
        mutations = (
            (original.replace('number="4"', 'probability="0.5"'), "unsupported"),
            (original.replace('number="4"', 'period="2"'), "unsupported"),
            (original.replace('route="r"', 'from="a" to="b"'), "unsupported"),
        )
        for index, (text, message) in enumerate(mutations):
            with self.subTest(index=index):
                self.source.write_text(text, encoding="utf-8")
                self.spec["source_routes"] = self._identity(self.source)
                self._write_spec()
                with self.assertRaisesRegex(DemandContractError, message):
                    self._expand(f"bad-{index}")
        self.source.write_text(original, encoding="utf-8")

    def test_until_and_unverified_flow_child_or_attribute_semantics_are_rejected(self):
        original = self.source.read_text(encoding="utf-8")
        mutations = (
            (original.replace('duration="2"', 'until="10"'), "unsupported child"),
            (original.replace('duration="2"', 'duration="2" triggered="person"'),
             "unsupported child"),
            (original.replace('<stop lane="b_0" duration="2"/>', '<param key="x" value="y"/>'),
             "unsupported child"),
            (original.replace('departLane="best"', 'departLane="free"'),
             "outside contract v1"),
            (original.replace('departSpeed="random"', 'departSpeed="max"'),
             "outside contract v1"),
        )
        for index, (text, message) in enumerate(mutations):
            with self.subTest(index=index):
                self.source.write_text(text, encoding="utf-8")
                self.spec["source_routes"] = self._identity(self.source); self._write_spec()
                with self.assertRaisesRegex(DemandContractError, message):
                    self._expand(f"unsupported-semantic-{index}")
        self.source.write_text(original, encoding="utf-8")

    def test_inline_routes_must_use_known_connected_edges(self):
        original = self.source.read_text(encoding="utf-8")
        for suffix, edges, message in (
            ("unknown", "a missing", "unknown edge"),
            ("disconnected", "b a", "disconnected"),
        ):
            with self.subTest(suffix=suffix):
                text = original.replace(
                    '<vehicle id="explicit" type="car" depart="12"><route edges="a b"/>',
                    f'<vehicle id="explicit" type="car" depart="12"><route edges="{edges}"/>',
                )
                self.source.write_text(text, encoding="utf-8")
                self.spec["source_routes"] = self._identity(self.source); self._write_spec()
                with self.assertRaisesRegex(DemandContractError, message):
                    self._expand(f"inline-{suffix}")
        self.source.write_text(original, encoding="utf-8")

    def test_duplicate_generated_identity_and_unknown_elements_fail(self):
        original = self.source.read_text(encoding="utf-8")
        for text, message in (
            (original.replace('id="explicit"', 'id="f.0"'), "collide"),
            (original.replace('</routes>', '<trip id="t" depart="1" from="a" to="b"/></routes>'),
             "unsupported demand element"),
        ):
            with self.subTest(message=message):
                self.source.write_text(text, encoding="utf-8")
                self.spec["source_routes"] = self._identity(self.source); self._write_spec()
                with self.assertRaisesRegex(DemandContractError, message):
                    self._expand("bad-" + str(len(list(self.root.glob("bad-*")))))
        self.source.write_text(original, encoding="utf-8")

    def test_changed_source_config_and_output_bytes_are_detected(self):
        self._expand()
        trusted_manifest = self._digest(self.root / "expanded/expansion_manifest.json")
        trusted_spec = self._digest(self.spec_path)
        original = self.source.read_bytes()
        self.source.write_bytes(original + b"<!--changed-->\n")
        with self.assertRaisesRegex(DemandContractError, "missing or changed"):
            self._verify(manifest_sha256=trusted_manifest, spec_sha256=trusted_spec)
        self.source.write_bytes(original)
        route = self.root / "expanded/routes.explicit.rou.xml"
        route.write_bytes(route.read_bytes() + b" ")
        with self.assertRaisesRegex(DemandContractError, "expanded route"):
            self._verify(manifest_sha256=trusted_manifest, spec_sha256=trusted_spec)

        self.source.write_text(self.source.read_text(encoding="utf-8").replace(
            'number="4"', 'number="5"'), encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "source route"):
            self._expand("changed-source")
        self.source.write_bytes(original)
        original_config = self.config.read_bytes()
        self.config.write_bytes(original_config.replace(b'end value="20"', b'end value="21"'))
        with self.assertRaisesRegex(DemandContractError, "configuration"):
            self._expand("changed-config")

    def test_coordinated_route_and_manifest_forgery_fails_trusted_identity(self):
        self._expand()
        directory = self.root / "expanded"
        trusted_manifest = self._digest(directory / "expansion_manifest.json")
        trusted_spec = self._digest(self.spec_path)
        route_path = directory / "routes.explicit.rou.xml"
        route_root = ET.parse(route_path).getroot()
        next(node for node in route_root.findall("vehicle") if node.get("id") == "f.0").set(
            "depart", "0.25"
        )
        route_path.write_bytes(ET.tostring(route_root, encoding="utf-8", xml_declaration=True))
        manifest_path = directory / "expansion_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output"]["sha256"] = self._digest(route_path)
        # A forged fingerprint is deliberately self-consistent with the forged bytes.
        from DeepQN.evaluation.demand_expansion import _physical_fingerprint
        manifest["output"]["physical_fingerprint_sha256"] = _physical_fingerprint(
            route_root.findall("vehicle")
        )
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                                 encoding="utf-8")
        with self.assertRaisesRegex(DemandContractError, "trusted identity"):
            self._verify(manifest_sha256=trusted_manifest, spec_sha256=trusted_spec)

    def test_wrong_trusted_hashes_and_mutably_rehashed_source_fail(self):
        self._expand()
        manifest_hash = self._digest(self.root / "expanded/expansion_manifest.json")
        spec_hash = self._digest(self.spec_path)
        with self.assertRaisesRegex(DemandContractError, "trusted identity"):
            self._verify(manifest_sha256="0" * 64, spec_sha256=spec_hash)
        with self.assertRaisesRegex(DemandContractError, "trusted identity"):
            self._verify(manifest_sha256=manifest_hash, spec_sha256="0" * 64)

        # Updating mutable source/spec/manifest hashes does not satisfy the frozen references.
        self.source.write_bytes(self.source.read_bytes() + b"<!--coordinated source edit-->\n")
        self.spec["source_routes"] = self._identity(self.source); self._write_spec()
        with self.assertRaisesRegex(DemandContractError, "specification is missing or changed"):
            self._verify(manifest_sha256=manifest_hash, spec_sha256=spec_hash)

    @unittest.skipUnless(shutil.which("sumo"), "SUMO executable unavailable")
    def test_bounded_source_flow_and_expansion_runtime_parity(self):
        repository = Path(__file__).resolve().parents[2]
        self.network = repository / "DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml"

        def run_case(label: str, random_speed: bool):
            random_attr = ' departSpeed="random"' if random_speed else ""
            self.source.write_text(
                '<routes><vType id="car"/><route id="r" edges="n_in s_out"/>'
                f'<flow id="f" type="car" route="r" begin="0" end="8" number="4"'
                f'{random_attr}><stop lane="s_out_0" duration="2"/></flow></routes>\n',
                encoding="utf-8",
            )
            self.config.write_text(
                f'<configuration><input><net-file value="{self.network.as_posix()}"/>'
                '<route-files value="source.rou.xml"/></input>'
                '<time><begin value="0"/><end value="80"/><step-length value="1"/></time>'
                '</configuration>\n', encoding="utf-8",
            )
            self.spec.update(
                source_routes=self._identity(self.source),
                configuration=self._identity(self.config),
                network={"path": str(self.network), "sha256": self._digest(self.network)},
                additional_files=[], begin_s=0, end_s=80,
            )
            self._write_spec()
            manifest = self._expand(label)

            results = []
            for representation, route in (
                ("flow", self.source),
                ("explicit", self.root / label / "routes.explicit.rou.xml"),
            ):
                tripinfo = self.root / f"{label}-{representation}.tripinfo.xml"
                summary = self.root / f"{label}-{representation}.summary.xml"
                run_config = self.root / f"{label}-{representation}.sumocfg"
                run_config.write_text(
                    f'<configuration><input><net-file value="{self.network.as_posix()}"/>'
                    f'<route-files value="{route.as_posix()}"/></input>'
                    '<time><begin value="0"/><end value="80"/><step-length value="1"/></time>'
                    '<random_number><seed value="17"/></random_number></configuration>\n',
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    ["sumo", "-c", str(run_config), "--tripinfo-output", str(tripinfo),
                     "--summary-output", str(summary), "--no-step-log", "true"],
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                trips = ET.parse(tripinfo).getroot().findall("tripinfo")
                last = ET.parse(summary).getroot().findall("step")[-1]
                results.append((
                    [(trip.get("id"), trip.get("depart"), trip.get("arrival"),
                      trip.get("duration"), trip.get("waitingTime")) for trip in trips],
                    (last.get("loaded"), last.get("inserted"), last.get("arrived")),
                ))
            self.assertEqual([row[0] for row in results[0][0]],
                             [f"f.{index}" for index in range(4)])
            self.assertEqual(results[0][1], ("4", "4", "4"))
            self.assertEqual(results[1][1], ("4", "4", "4"))
            if not random_speed:
                self.assertEqual(results[0], results[1])
            return manifest, results

        deterministic_manifest, deterministic = run_case("parity-deterministic", False)
        self.assertFalse(deterministic_manifest["runtime_randomness"]
                         ["requires_separate_recorded_sumo_seed"])
        random_manifest, random_results = run_case("parity-random", True)
        self.assertTrue(random_manifest["runtime_randomness"]
                        ["requires_separate_recorded_sumo_seed"])
        # Population parity is established.  No trajectory equality assertion is made for
        # runtime-random attributes because representation changes may alter RNG consumption.
        self.assertEqual(random_results[0][1], random_results[1][1])

    def test_self_approval_wrong_version_and_overwrite_are_rejected(self):
        for field, value, message in (
            ("publication_approved", True, "cannot self-approve"),
            ("sumo_version", "1.27.0", "requires SUMO 1.26.0"),
            ("algorithm", "random", "unsupported"),
        ):
            with self.subTest(field=field):
                spec = copy.deepcopy(self.spec); spec[field] = value
                self.spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
                with self.assertRaisesRegex(DemandContractError, message):
                    self._expand("invalid-" + field)
        self._write_spec(); self._expand("once")
        with self.assertRaises(FileExistsError):
            self._expand("once")

    def test_failure_injection_leaves_no_partial_publication(self):
        target = self.root / "failed"
        with self.assertRaisesRegex(OSError, "injected"):
            self._expand("failed", fail_after_files=1)
        self.assertFalse(target.exists())

    def test_historical_profile_counts_remain_unchanged_offline(self):
        repository = Path(__file__).resolve().parents[2]
        maps = repository / "simulation/maps"
        profiles = {
            "morning": (maps / "config_morning_rush.sumocfg", 6000),
            "midday": (maps / "config_midday.sumocfg", 9072),
            "evening": (maps / "config_evening_rush.sumocfg", 20199),
            "night": (maps / "config_night.sumocfg", 2106),
        }
        for index, (profile, (config, expected_count)) in enumerate(profiles.items()):
            with self.subTest(profile=profile):
                begin, end, network, additions, routes = _config_inputs(config.resolve())
                self.assertEqual(len(routes), 1)
                spec_path = self.root / f"historical-{profile}.json"
                spec = {
                    "schema": EXPANSION_SPEC_SCHEMA,
                    "expansion_id": f"historical-{profile}-compatibility",
                    "sumo_version": "1.26.0",
                    "algorithm": EXPANSION_ALGORITHM,
                    "source_routes": {"path": str(routes[0]), "sha256": self._digest(routes[0])},
                    "configuration": {"path": str(config), "sha256": self._digest(config)},
                    "network": {"path": str(network), "sha256": self._digest(network)},
                    "additional_files": [
                        {"path": str(path), "sha256": self._digest(path)} for path in additions
                    ],
                    "begin_s": begin,
                    "end_s": end,
                    "publication_approved": False,
                }
                spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
                manifest = expand_number_flows(
                    spec_path, self._digest(spec_path),
                    output_dir=self.root / f"historical-output-{index}",
                )
                self.assertEqual(manifest["counts"]["expanded_explicit_vehicles"],
                                 expected_count)

    @unittest.skipUnless(shutil.which("sumo"), "SUMO executable unavailable")
    def test_bounded_sumo_accepts_expanded_explicit_demand(self):
        repository = Path(__file__).resolve().parents[2]
        self.network = repository / "DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml"
        self.source.write_text(
            '<routes><vType id="car"/><route id="r" edges="n_in s_out"/>'
            '<flow id="f" type="car" route="r" begin="0" end="8" number="4"/>'
            '</routes>\n', encoding="utf-8",
        )
        self.config.write_text(
            f'<configuration><input><net-file value="{self.network.as_posix()}"/>'
            '<route-files value="source.rou.xml"/></input>'
            '<time><begin value="0"/><end value="40"/><step-length value="1"/></time>'
            '</configuration>\n', encoding="utf-8",
        )
        self.spec.update(
            source_routes=self._identity(self.source),
            configuration=self._identity(self.config),
            network={"path": str(self.network), "sha256": self._digest(self.network)},
            additional_files=[], begin_s=0, end_s=40,
        )
        self._write_spec(); self._expand()
        summary = self.root / "summary.xml"
        run_config = self.root / "run.sumocfg"
        route = self.root / "expanded/routes.explicit.rou.xml"
        run_config.write_text(
            f'<configuration><input><net-file value="{self.network.as_posix()}"/>'
            f'<route-files value="{route.as_posix()}"/></input>'
            '<time><begin value="0"/><end value="40"/><step-length value="1"/></time>'
            '<random_number><seed value="17"/></random_number></configuration>\n',
            encoding="utf-8",
        )
        completed = subprocess.run(
            ["sumo", "-c", str(run_config), "--summary-output", str(summary),
             "--no-step-log", "true"], capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        steps = ET.parse(summary).getroot().findall("step")
        self.assertTrue(steps)
        self.assertEqual(int(steps[-1].get("loaded")), 4)
        self.assertEqual(int(steps[-1].get("arrived")), 4)


if __name__ == "__main__":
    unittest.main()
