"""Experiment manifests and strict comparability checks for SUMO runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from DeepQN.evaluation.metrics import IncomparableRunsError, METRIC_DEFINITION


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(args: list[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            args, cwd=cwd, check=True, capture_output=True, text=True, timeout=15
        )
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(project_root: str | Path) -> tuple[Optional[str], Optional[bool]]:
    root = Path(project_root).resolve()
    # Git for Windows matches safe.directory using forward-slash paths.
    prefix = ["git", "-c", f"safe.directory={root.as_posix()}"]
    commit = _command_output(prefix + ["rev-parse", "HEAD"], root)
    status = _command_output(prefix + ["status", "--porcelain"], root)
    return commit, None if status is None else bool(status)


def sumo_version(sumo_binary: str, cwd: str | Path) -> Optional[str]:
    output = _command_output([sumo_binary, "--version"], Path(cwd))
    return output.splitlines()[0] if output else None


@dataclass
class RunManifest:
    run_id: str
    controller: str
    scenario: str
    seed: int
    simulation_begin_s: float
    simulation_end_s: float
    step_length_s: float
    config_path: str
    config_sha256: str
    simulation_config_path: str
    simulation_config_sha256: str
    network_path: str
    network_sha256: str
    route_files: Dict[str, str]
    additional_files: Dict[str, str] = field(default_factory=dict)
    effective_simulation_config_sha256: Optional[str] = None
    emission_step_scaled: Optional[bool] = None
    metric_definition: str = METRIC_DEFINITION
    model_mode: str = "not_applicable"
    checkpoint_manifest: Optional[dict] = None
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    sumo_version: Optional[str] = None
    software_versions: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    output_hashes: Dict[str, str] = field(default_factory=dict)
    controller_diagnostics: Dict[str, Any] = field(default_factory=dict)
    demand_realization_id: Optional[str] = None
    demand_manifest_sha256: Optional[str] = None
    demand_partition: Optional[str] = None
    split_registry_sha256: Optional[str] = None
    demand_usage: Optional[str] = None
    effective_simulation_config_path: Optional[str] = None
    effective_simulation_config_file_sha256: Optional[str] = None
    completion_status: str = "planned"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        with Path(path).open(encoding="utf-8") as stream:
            return cls(**json.load(stream))


def build_run_manifest(
    *,
    project_root: str | Path,
    run_id: str,
    controller: str,
    scenario: str,
    seed: int,
    simulation_begin_s: float,
    simulation_end_s: float,
    step_length_s: float,
    config_path: str | Path,
    simulation_config_path: str | Path,
    network_path: str | Path,
    route_paths: Iterable[str | Path],
    additional_paths: Iterable[str | Path] = (),
    output_paths: Dict[str, str],
    model_mode: str,
    checkpoint_manifest: Optional[dict],
    emission_step_scaled: bool = False,
    sumo_binary: str = "sumo",
) -> RunManifest:
    root = Path(project_root).resolve()

    def resolve(path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else root / candidate

    config = resolve(config_path)
    simulation_config = resolve(simulation_config_path)
    network = resolve(network_path)
    routes = {str(resolve(path)): sha256_file(resolve(path)) for path in route_paths}
    additional = {str(resolve(path)): sha256_file(resolve(path)) for path in additional_paths}
    commit, dirty = git_state(root)
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    return RunManifest(
        run_id=run_id,
        controller=controller,
        scenario=scenario,
        seed=seed,
        simulation_begin_s=simulation_begin_s,
        simulation_end_s=simulation_end_s,
        step_length_s=step_length_s,
        config_path=str(config),
        config_sha256=sha256_file(config),
        simulation_config_path=str(simulation_config),
        simulation_config_sha256=sha256_file(simulation_config),
        effective_simulation_config_sha256=hash_effective_sumocfg(simulation_config),
        network_path=str(network),
        network_sha256=sha256_file(network),
        route_files=routes,
        additional_files=additional,
        model_mode=model_mode,
        checkpoint_manifest=checkpoint_manifest,
        emission_step_scaled=emission_step_scaled,
        git_commit=commit,
        git_dirty=dirty,
        sumo_version=sumo_version(sumo_binary, root),
        software_versions=versions,
        output_paths=dict(output_paths),
    )


def hash_effective_sumocfg(path: str | Path) -> str:
    """Hash behavior settings while ignoring only output destination paths."""
    root = ET.parse(path).getroot()
    for section in (root.find("output"), root.find("o")):
        if section is None:
            continue
        for child in section:
            if child.tag.endswith("-output") and child.tag not in {
                "tripinfo-output.write-unfinished", "tripinfo-output.write-undeparted",
                "emission-output.step-scaled"
            }:
                child.set("value", "<OUTPUT_DESTINATION>")
    payload = ET.tostring(root, encoding="utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_output_hashes(manifest: RunManifest) -> None:
    manifest.output_hashes = {}
    for name, raw_path in manifest.output_paths.items():
        path = Path(raw_path)
        if not path.is_file():
            raise IncomparableRunsError(f"missing required output {name}: {path}")
        manifest.output_hashes[name] = sha256_file(path)


def validate_output_integrity(manifest: RunManifest) -> None:
    required = {"tripinfo", "emission", "summary"}
    if set(manifest.output_paths) != required or set(manifest.output_hashes) != required:
        raise IncomparableRunsError("missing or unknown required output provenance")
    for name in required:
        path = Path(manifest.output_paths[name])
        if not path.is_file() or sha256_file(path) != manifest.output_hashes[name]:
            raise IncomparableRunsError(f"output integrity failure: {name}")


def validate_comparability(left: RunManifest, right: RunManifest) -> None:
    """Raise unless two runs differ only in controller/model/checkpoint choice."""
    checks: Dict[str, tuple[Any, Any]] = {
        "scenario": (left.scenario, right.scenario),
        "simulation begin": (left.simulation_begin_s, right.simulation_begin_s),
        "simulation end": (left.simulation_end_s, right.simulation_end_s),
        "step length": (left.step_length_s, right.step_length_s),
        "effective simulation config hash": (
            left.effective_simulation_config_sha256,
            right.effective_simulation_config_sha256,
        ),
        "network hash": (left.network_sha256, right.network_sha256),
        "route hashes": (sorted(left.route_files.values()), sorted(right.route_files.values())),
        "additional-file hashes": (sorted(left.additional_files.values()), sorted(right.additional_files.values())),
        "emission scaling mode": (left.emission_step_scaled, right.emission_step_scaled),
        "metric definition": (left.metric_definition, right.metric_definition),
        "SUMO version": (left.sumo_version, right.sumo_version),
        "evaluation seed": (left.seed, right.seed),
    }
    for manifest in (left, right):
        missing = []
        if not manifest.git_commit: missing.append("git_commit")
        if manifest.git_dirty is None: missing.append("git_dirty")
        if not manifest.sumo_version: missing.append("sumo_version")
        if not manifest.route_files: missing.append("route_files")
        if not manifest.network_sha256: missing.append("network_sha256")
        if not math.isfinite(manifest.step_length_s) or manifest.step_length_s <= 0:
            missing.append("step_length_s")
        if missing:
            raise IncomparableRunsError(
                f"manifest {manifest.run_id} has missing or invalid required provenance: "
                + ", ".join(missing)
            )
        validate_output_integrity(manifest)
    mismatches = [name for name, (a, b) in checks.items() if a is None or b is None or a != b]
    if mismatches:
        raise IncomparableRunsError(
            "runs are not experimentally comparable; mismatched or unknown: "
            + ", ".join(mismatches)
        )
    demand_fields = (
        "demand_realization_id", "demand_manifest_sha256", "demand_partition",
        "split_registry_sha256", "demand_usage",
    )
    if any(getattr(item, name) is not None for item in (left, right) for name in demand_fields):
        for item in (left, right):
            missing = [name for name in demand_fields if not getattr(item, name)]
            if missing:
                raise IncomparableRunsError(
                    f"manifest {item.run_id} has incomplete demand provenance: "
                    + ", ".join(missing)
                )
        mismatched = [name for name in demand_fields
                      if getattr(left, name) != getattr(right, name)]
        if mismatched:
            raise IncomparableRunsError(
                "runs are not experimentally comparable; mismatched demand provenance: "
                + ", ".join(mismatched)
            )
    if left.completion_status != "complete" or right.completion_status != "complete":
        raise IncomparableRunsError("both run manifests must have completion_status='complete'")
