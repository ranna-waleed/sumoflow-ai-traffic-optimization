"""Fixed-time SUMO baseline using the common Phase 1 evaluation contract."""
from __future__ import annotations

import subprocess
from pathlib import Path

from DeepQN.evaluation.evaluate import PROJECT_ROOT, _ensure_outputs_available, _patch_sumocfg, _sumocfg_inputs
from DeepQN.evaluation.metrics import EpisodeMetrics, load_metrics_from_outputs
from DeepQN.evaluation.provenance import (
    RunManifest, build_run_manifest, capture_output_hashes, hash_effective_sumocfg,
)


def run_fixed_time_episode(*, cfg: dict, profile: str, output_dir: Path,
                           run_index: int, seed: int, config_path: str | Path
                           ) -> tuple[EpisodeMetrics, RunManifest]:
    """Run SUMO without a TraCI controller, preserving its fixed TLS program."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"fixed_time_{profile}_seed{seed}_run{run_index}"
    outputs = {kind: (output_dir / f"{kind}_{run_id}.xml").resolve()
               for kind in ("tripinfo", "emission", "summary")}
    manifest_path = output_dir / f"manifest_{run_id}.json"
    _ensure_outputs_available([*outputs.values(), manifest_path])
    source = Path(cfg["profiles"][profile]["config"])
    network, routes, additional = _sumocfg_inputs(source)
    manifest = build_run_manifest(
        project_root=PROJECT_ROOT, run_id=run_id, controller="fixed_time",
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
    patched = None
    try:
        patched = _patch_sumocfg(source, outputs, seed)
        manifest.effective_simulation_config_sha256 = hash_effective_sumocfg(patched)
        command = [cfg["simulation"].get("sumo_binary", "sumo"), "-c", str(patched), "--no-step-log"]
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"fixed-time SUMO exited {result.returncode}: {(result.stderr or result.stdout).strip()}")
        metrics = load_metrics_from_outputs(
            output_dir, profile, "fixed_time", run_id,
            step_length_s=manifest.step_length_s,
            expected_begin_s=manifest.simulation_begin_s,
            expected_end_s=manifest.simulation_end_s,
            emission_step_scaled=False,
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
