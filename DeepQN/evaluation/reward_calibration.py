"""Leakage-resistant reward-reference calibration contracts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


CALIBRATION_SCHEMA = "sumoflow-reward-calibration-v2"
SPLIT_REGISTRY_SCHEMA = "sumoflow-calibration-split-registry-v1"
UNITS = {"delay": "vehicle-seconds", "co2": "mg", "arrivals": "vehicles",
         "pressure": "vehicle-seconds"}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label} is not valid SHA-256 syntax")
    return value.lower()


def calibrate_reward_references(sample_files: list[str | Path], output: str | Path,
                                *, quantile: float = 0.95,
                                dataset_label: str = "engineering_fixture",
                                mode: str = "engineering_fixture",
                                split_registry_path: str | Path | None = None,
                                expected_split_registry_sha256: str | None = None) -> dict:
    """Freeze positive reference scales from training/calibration partitions only."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    values = {name: [] for name in UNITS}
    sources = []
    demand_hashes = set()
    registry_identity = None
    registry_entries = None
    if mode not in {"engineering_fixture", "publication_candidate"}:
        raise ValueError("calibration mode must be engineering_fixture or publication_candidate")
    if mode == "publication_candidate":
        if split_registry_path is None or expected_split_registry_sha256 is None:
            raise ValueError("publication calibration requires a hash-bound split registry")
        registry_path = Path(split_registry_path).resolve()
        registry_hash = _hash(registry_path)
        if registry_hash != _require_sha256(expected_split_registry_sha256,
                                            "expected split-registry hash"):
            raise ValueError("split-registry hash mismatch")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema") != SPLIT_REGISTRY_SCHEMA:
            raise ValueError("unsupported split-registry schema")
        rows = registry.get("realizations")
        if not isinstance(rows, list) or not rows:
            raise ValueError("split registry contains no realizations")
        registry_entries = {}
        demand_partitions = {}
        for row in rows:
            identity = row.get("realization_id")
            if not isinstance(identity, str) or not identity or identity in registry_entries:
                raise ValueError("duplicate or invalid split-registry realization identity")
            partition = row.get("partition")
            if partition not in {"training", "calibration", "validation", "test", "evaluation"}:
                raise ValueError(f"invalid split partition for {identity}")
            demand_path = (registry_path.parent / row.get("demand_path", "")).resolve()
            sample_path = (registry_path.parent / row.get("sample_path", "")).resolve()
            demand_hash = _require_sha256(row.get("demand_sha256"), "registry demand hash")
            sample_hash = _require_sha256(row.get("sample_sha256"), "registry sample hash")
            if not demand_path.is_file() or _hash(demand_path) != demand_hash:
                raise ValueError(f"demand artifact hash mismatch for {identity}")
            if not sample_path.is_file() or _hash(sample_path) != sample_hash:
                raise ValueError(f"calibration source hash mismatch for {identity}")
            previous = demand_partitions.setdefault(demand_hash, partition)
            if previous != partition:
                raise ValueError("a demand artifact overlaps conflicting split partitions")
            registry_entries[identity] = {
                "partition": partition, "demand_path": str(demand_path),
                "demand_sha256": demand_hash, "sample_path": str(sample_path),
                "sample_sha256": sample_hash,
            }
        registry_identity = {"path": str(registry_path), "sha256": registry_hash,
                             "schema": SPLIT_REGISTRY_SCHEMA,
                             "review_status": "externally supplied; not inferred by calibration code"}
    elif split_registry_path is not None or expected_split_registry_sha256 is not None:
        raise ValueError("engineering fixture mode must not imply a publication split registry")

    seen_realizations = set()
    for raw_path in sample_files:
        path = Path(raw_path).resolve()
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        partition = data.get("partition")
        if partition not in {"training", "calibration"}:
            raise ValueError("held-out validation/test/evaluation samples cannot calibrate rewards")
        demand_hash = _require_sha256(data.get("demand_sha256"), "sample demand hash")
        realization_id = data.get("realization_id")
        if mode == "publication_candidate":
            if not isinstance(realization_id, str) or realization_id in seen_realizations:
                raise ValueError("duplicate or missing calibration realization identity")
            entry = registry_entries.get(realization_id)
            if entry is None:
                raise ValueError(f"calibration source is absent from split registry: {realization_id}")
            if (entry["partition"] != partition or entry["demand_sha256"] != demand_hash
                    or Path(entry["sample_path"]) != path or entry["sample_sha256"] != _hash(path)):
                raise ValueError(f"calibration source conflicts with split registry: {realization_id}")
            seen_realizations.add(realization_id)
        demand_hashes.add(demand_hash)
        for name in UNITS:
            for value in data.get("intervals", {}).get(name, []):
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"invalid {name} calibration value")
                values[name].append(number)
        sources.append({"path": str(path), "sha256": _hash(path), "partition": partition,
                        "realization_id": realization_id,
                        "demand_path": (None if mode == "engineering_fixture"
                                        else registry_entries[realization_id]["demand_path"]),
                        "demand_sha256": demand_hash})
    references = {}
    for name, samples in values.items():
        positive = [value for value in samples if value > 0]
        if not positive:
            raise ValueError(f"no positive eligible samples for {name}")
        references[name] = {"value": float(np.quantile(positive, quantile, method="linear")),
                            "unit": UNITS[name], "eligible_positive_samples": len(positive)}
    manifest = {"schema": CALIBRATION_SCHEMA,
                "method": f"positive_quantile_{quantile:g}_linear",
                "aggregation": "all nonnegative interval values; zero excluded from scale selection",
                "numpy_version": np.__version__,
                "dataset_label": dataset_label, "mode": mode,
                "publication_approved": False,
                "split_registry": registry_identity,
                "references": references, "sources": sources,
                "demand_sha256": sorted(demand_hashes)}
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite calibration manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

