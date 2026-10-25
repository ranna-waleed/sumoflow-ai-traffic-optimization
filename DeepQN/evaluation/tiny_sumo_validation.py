"""Execute the bounded Phase 1B fixed-time/random SUMO validation pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import numpy as np

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.evaluation.baseline import run_fixed_time_episode
from DeepQN.evaluation.evaluate import run_controlled_episode
from DeepQN.evaluation.provenance import validate_comparability

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "DeepQN/tests/fixtures/tiny_sumo/tiny_eval.yaml"


class _NoForecastFixtureAdapter:
    """Deterministic smoke-test adapter; production BiLSTM behavior is untouched."""
    def reset(self): pass
    def load(self): return False
    def tick(self): pass
    def predict(self): return np.zeros(4, dtype=np.float32)


def run_validation(output_root: str | Path, seed: int = 17, repeat: int = 2,
                   base_port: int = 18913) -> dict:
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"validation output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    with DEFAULT_CONFIG.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    agents = MultiAgentDQN(list(cfg["tls_junctions"]), cfg["dqn"])
    records = []
    for index in range(repeat):
        pair_dir = output_root / f"repeat_{index}"
        fixed, fixed_manifest = run_fixed_time_episode(
            cfg=cfg, profile="tiny", output_dir=pair_dir / "fixed_time",
            run_index=index, seed=seed, config_path=DEFAULT_CONFIG)
        random, random_manifest = run_controlled_episode(
            cfg=cfg, profile="tiny", agents=agents, controller="random",
            output_dir=pair_dir / "random", run_index=index, seed=seed,
            checkpoint_manifest=None, model_mode="not_applicable_random_policy",
            config_path=DEFAULT_CONFIG, port=base_port + index,
            environment_mutator=lambda env: setattr(env, "_bilstm", _NoForecastFixtureAdapter()))
        validate_comparability(fixed_manifest, random_manifest)
        records.append({
            "fixed_time": fixed.to_dict(), "random": random.to_dict(),
            "fixed_manifest": fixed_manifest.to_dict(),
            "random_manifest": random_manifest.to_dict(),
            "comparison": random.improvement_vs(fixed),
            "comparable": True,
        })
    def scientific_metrics(record: dict) -> dict:
        return {key: value for key, value in record.items()
                if key not in {"source_files"}}
    reproducible_metrics = len(records) < 2 or all(
        scientific_metrics(records[0][mode]) == scientific_metrics(item[mode])
        for item in records[1:] for mode in ("fixed_time", "random")
    )
    reproducible_controller_diagnostics = len(records) < 2 or all(
        records[0]["random_manifest"]["controller_diagnostics"]
        == item["random_manifest"]["controller_diagnostics"]
        for item in records[1:]
    )
    identical_raw_output_hashes = len(records) < 2 or all(
        records[0][f"{mode}_manifest"]["output_hashes"]
        == item[f"{mode}_manifest"]["output_hashes"]
        for item in records[1:] for mode in ("fixed", "random")
    )
    report = {"seed": seed, "repeat": repeat, "records": records,
              "reproducible_metrics": reproducible_metrics,
              "reproducible_controller_diagnostics": reproducible_controller_diagnostics,
              "identical_raw_output_hashes": identical_raw_output_hashes}
    report_path = output_root / "tiny_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=18913)
    args = parser.parse_args()
    result = run_validation(args.output_root, args.seed, args.repeat, args.base_port)
    print(json.dumps({"reproducible_metrics": result["reproducible_metrics"],
                      "reproducible_controller_diagnostics": result["reproducible_controller_diagnostics"],
                      "identical_raw_output_hashes": result["identical_raw_output_hashes"],
                      "runs": len(result["records"])}, sort_keys=True))
