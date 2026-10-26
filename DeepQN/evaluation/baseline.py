"""Fixed-time SUMO baseline using the common Phase 1 evaluation contract."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

from DeepQN.evaluation.evaluate import PROJECT_ROOT, _ensure_outputs_available, _patch_sumocfg, _sumocfg_inputs
from DeepQN.evaluation.metrics import EpisodeMetrics, load_metrics_from_outputs
from DeepQN.evaluation.provenance import (
    RunManifest, build_run_manifest, capture_output_hashes, hash_effective_sumocfg,
)
from DeepQN.evaluation.controller_baselines import (
    CONTROLLER_CONTRACT_VERSION, load_signal_program, verify_loaded_signal_programs,
)


def _validate_wall_clock_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("wall_clock_timeout_s must be finite and greater than zero")
    return timeout


def _run_sumo_command(command: list[str], wall_clock_timeout_s: float | None):
    """Run SUMO with an optional wall-clock guard and diagnostic failure."""
    try:
        return subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=wall_clock_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        detail = (stderr or stdout).strip()
        suffix = f"; output: {detail}" if detail else ""
        raise RuntimeError(
            f"SUMO baseline exceeded wall-clock timeout "
            f"{wall_clock_timeout_s:g}s; command: {command!r}{suffix}"
        ) from exc


def run_native_baseline_episode(*, cfg: dict, profile: str, output_dir: Path,
                                run_index: int, seed: int, config_path: str | Path,
                                controller_type: str,
                                tls_program_path: str | Path, tls_ids: list[str],
                                wall_clock_timeout_s: float | None = None,
                                comparison_scope: str = "secondary_full_operation",
                                ) -> tuple[EpisodeMetrics, RunManifest]:
    """Run a verified native static/actuated SUMO program without a DQN."""
    expected_type = {"fixed_time": "static", "sumo_actuated": "actuated"}.get(controller_type)
    if expected_type is None:
        raise ValueError("native controller_type must be fixed_time or sumo_actuated")
    if comparison_scope != "secondary_full_operation":
        raise ValueError(
            "native SUMO baselines are not phase-synchronized with the 60-second "
            "DQN handoff; use secondary_full_operation or an approved matched protocol"
        )
    wall_clock_timeout_s = _validate_wall_clock_timeout(wall_clock_timeout_s)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{controller_type}_{profile}_seed{seed}_run{run_index}"
    outputs = {kind: (output_dir / f"{kind}_{run_id}.xml").resolve()
               for kind in ("tripinfo", "emission", "summary")}
    manifest_path = output_dir / f"manifest_{run_id}.json"
    _ensure_outputs_available([*outputs.values(), manifest_path])
    source = Path(cfg["profiles"][profile]["config"])
    network, routes, additional = _sumocfg_inputs(source)
    manifest = build_run_manifest(
        project_root=PROJECT_ROOT, run_id=run_id, controller=controller_type,
        scenario=profile, seed=seed,
        simulation_begin_s=float(cfg["profiles"][profile]["begin"]),
        simulation_end_s=float(cfg["profiles"][profile]["end"]),
        step_length_s=float(cfg["simulation"].get("step_length", 1.0)),
        config_path=config_path, simulation_config_path=source,
        network_path=network, route_paths=routes, additional_paths=additional,
        output_paths={key: str(value) for key, value in outputs.items()},
        model_mode="not_applicable", checkpoint_manifest=None,
        emission_step_scaled=False,
        sumo_binary=cfg["simulation"].get("sumo_binary", "sumo"),
    )
    manifest.controller_diagnostics = {
        "controller_contract": CONTROLLER_CONTRACT_VERSION,
        "initial_control_protocol": "native_full_operation_from_begin",
        "comparison_scope": comparison_scope,
        "phase_programs": [],
    }
    patched = None
    try:
        declared_source = Path(tls_program_path).resolve()
        effective_inputs = {network.resolve(), *(path.resolve() for path in additional)}
        if declared_source not in effective_inputs:
            raise ValueError(
                f"declared TLS program source is not loaded by the SUMO configuration: "
                f"{declared_source}"
            )
        programs = [load_signal_program(declared_source, tls_id, expected_type, network)
                    for tls_id in tls_ids]
        manifest.controller_diagnostics["phase_programs"] = [
            {"tls_id": item.tls_id, "program_id": item.program_id,
             "program_type": item.program_type, "source_path": item.source_path,
             "source_sha256": item.source_sha256,
             "program_sha256": item.program_sha256, "phases": list(item.phases),
             "parameters": dict(item.parameters)}
            for item in programs
        ]
        patched = _patch_sumocfg(source, outputs, seed)
        manifest.effective_simulation_config_sha256 = hash_effective_sumocfg(patched)
        manifest.controller_diagnostics["live_program_preflight"] = (
            verify_loaded_signal_programs(
                patched, programs, cfg["simulation"].get("sumo_binary", "sumo")
            )
        )
        command = [cfg["simulation"].get("sumo_binary", "sumo"), "-c", str(patched), "--no-step-log"]
        result = _run_sumo_command(command, wall_clock_timeout_s)
        if result.returncode:
            raise RuntimeError(f"{controller_type} SUMO exited {result.returncode}: {(result.stderr or result.stdout).strip()}")
        metrics = load_metrics_from_outputs(
            output_dir, profile, controller_type, run_id,
            step_length_s=manifest.step_length_s,
            expected_begin_s=manifest.simulation_begin_s,
            expected_end_s=manifest.simulation_end_s,
            emission_step_scaled=False,
            route_paths=routes,
            require_demand_contract=True,
        )
        capture_output_hashes(manifest)
        manifest.completion_status = "complete" if metrics.valid and metrics.reached_horizon else "incomplete"
        manifest.errors.extend(metrics.errors)
        return metrics, manifest
    except Exception as exc:
        manifest.completion_status = "failed"
        manifest.errors.append(str(exc))
        raise
    finally:
        if patched is not None:
            patched.unlink(missing_ok=True)
        manifest.write(manifest_path)


def run_fixed_time_episode(*, cfg: dict, profile: str, output_dir: Path,
                           run_index: int, seed: int, config_path: str | Path,
                           wall_clock_timeout_s: float | None = None,
                           tls_program_path: str | Path | None = None,
                           tls_ids: list[str] | None = None,
                           ) -> tuple[EpisodeMetrics, RunManifest]:
    """Compatibility wrapper for a verified native fixed-time program."""
    source = Path(cfg["profiles"][profile]["config"])
    network, _, additional = _sumocfg_inputs(source)
    program_path = Path(tls_program_path) if tls_program_path else (additional[0] if additional else network)
    if tls_ids is None:
        import xml.etree.ElementTree as ET
        tls_ids = sorted({node.get("id") for node in ET.parse(program_path).getroot().iter("tlLogic")
                          if node.get("id")})
    if not tls_ids:
        raise ValueError(
            "no explicit TLS programs found; supply tls_program_path and tls_ids "
            "rather than running an unverified fixed-time baseline"
        )
    return run_native_baseline_episode(
        cfg=cfg, profile=profile, output_dir=output_dir, run_index=run_index,
        seed=seed, config_path=config_path, controller_type="fixed_time",
        tls_program_path=program_path, tls_ids=tls_ids,
        wall_clock_timeout_s=wall_clock_timeout_s,
    )
