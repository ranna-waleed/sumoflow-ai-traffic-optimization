"""
DeepQN/evaluation/evaluate.py
------------------------------
Before/after comparison — supports both Tahrir and Taksim configs.

BASELINE: read from XML files whose path comes from the config YAML
          (evaluation.baseline_output_dir) — not hardcoded.

DQN:      collect live from TraCI per-vehicle metrics.
          If no checkpoints exist, runs a RANDOM policy (Option A test).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.env.sumo_env import SumoEnv
from DeepQN.evaluation.metrics import EpisodeMetrics, load_metrics_from_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ── Helpers ───────────────────────────────────────────────────────

def _baseline_dir(cfg: dict) -> Path:
    """Read baseline output dir from config — works for any map."""
    ev = cfg.get("evaluation", {})
    d  = ev.get("baseline_output_dir", "simulation/maps/baseline_outputs")
    return Path(d)


def _suffix_map(cfg: dict) -> dict:
    """Map profile names to XML file suffixes."""
    # Custom map from config (e.g. taksim_morning → taksim)
    sm = cfg.get("baseline_suffix_map", {})
    # Tahrir defaults
    defaults = {
        "morning_rush": "morning",
        "midday":       "midday",
        "evening_rush": "evening",
        "night":        "night",
    }
    defaults.update(sm)
    return defaults


# ── Baseline ──────────────────────────────────────────────────────

def get_baseline_metrics(cfg: dict, profile: str) -> EpisodeMetrics:
    """Read baseline KPIs from SUMO XML files — no re-run needed."""
    base_dir = _baseline_dir(cfg)
    smap     = _suffix_map(cfg)
    suffix   = smap.get(profile, profile)

    logger.info("[BASELINE] Reading XML files for profile: %s (suffix: %s)", profile, suffix)
    m = load_metrics_from_outputs(base_dir, profile, "fixed_time", suffix_override=suffix)
    logger.info(
        "[BASELINE] %s — avg_wait=%.1fs  total_co2=%.0fmg  throughput=%d",
        profile, m.avg_waiting_time_s, m.total_co2_mg, m.throughput,
    )
    return m


# ── DQN / random runner ───────────────────────────────────────────

def run_dqn(
    cfg:        dict,
    profile:    str,
    agents:     MultiAgentDQN,
    port:       int  = 8813,
    random_policy: bool = False,
) -> EpisodeMetrics:
    """
    Run one episode. If random_policy=True, actions are random (no trained weights).
    Metrics collected per-vehicle to match baseline XML.
    """
    mode_label = "RANDOM" if random_policy else "DQN"
    logger.info("[%s] Running profile: %s", mode_label, profile)

    try:
        import traci
    except ImportError:
        raise ImportError("TraCI not found. Check SUMO_HOME and PYTHONPATH.")

    env  = SumoEnv(cfg, profile=profile, port=port)
    obs  = env.reset()
    done = False

    total_co2_mg    = 0.0
    total_arrived   = 0
    vehicle_waits: Dict[str, float] = {}
    steps = 0

    while not done:
        try:
            active_vids = traci.vehicle.getIDList()
        except Exception:
            active_vids = []

        for vid in active_vids:
            try:
                vehicle_waits[vid] = traci.vehicle.getWaitingTime(vid)
            except Exception:
                pass

        for vid in active_vids:
            try:
                total_co2_mg += traci.vehicle.getCO2Emission(vid)
            except Exception:
                pass

        try:
            total_arrived += traci.simulation.getArrivedNumber()
        except Exception:
            pass

        if random_policy:
            actions = {tid: np.random.randint(0, 2) for tid in agents.agents}
        else:
            actions = agents.act(obs, eval_mode=True)

        obs, rewards, done, info = env.step(actions)
        steps += 1

    env.close()

    wait_values = list(vehicle_waits.values())
    avg_wait = float(np.mean(wait_values)) if wait_values else 0.0

    m = EpisodeMetrics(
        profile            = profile,
        mode               = "dqn_adaptive",
        avg_waiting_time_s = avg_wait,
        total_co2_mg       = total_co2_mg,
        throughput         = total_arrived,
    )
    logger.info(
        "[%s] %s done — avg_wait=%.1fs  total_co2=%.0fmg  throughput=%d  steps=%d",
        mode_label, profile, avg_wait, total_co2_mg, total_arrived, steps,
    )
    return m


# ── Comparison ────────────────────────────────────────────────────

def compare(
    config_path:    str                 = "DeepQN/configs/dqn_config.yaml",
    checkpoint_dir: str                 = "DeepQN/checkpoints",
    profiles:       Optional[List[str]] = None,
    n_runs:         int                 = 3,
    output_dir:     str                 = "DeepQN/results",
    port:           int                 = 8813,
) -> Dict[str, dict]:

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    all_profiles = profiles or list(cfg["profiles"].keys())
    tls_ids      = list(cfg["tls_junctions"].keys())
    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load checkpoints — fall back to random policy if none exist
    agents = MultiAgentDQN(tls_ids, cfg["dqn"])
    try:
        agents.load_latest(checkpoint_dir)
        random_policy = False
        logger.info("Loaded trained checkpoints from %s", checkpoint_dir)
    except Exception as e:
        random_policy = True
        logger.warning("No checkpoints found (%s) — using RANDOM policy", e)
        logger.warning("Results show pipeline functionality, not trained performance.")

    results: Dict[str, dict] = {}

    for profile in all_profiles:
        logger.info("=" * 60)
        logger.info(
            "Evaluating profile: %s  (%d runs, %s)",
            profile, n_runs, "RANDOM" if random_policy else "DQN"
        )
        logger.info("=" * 60)

        base_m = get_baseline_metrics(cfg, profile)

        dqn_waits, dqn_co2s, dqn_thrus = [], [], []
        for run in range(n_runs):
            logger.info("  Run %d/%d", run + 1, n_runs)
            dm = run_dqn(cfg, profile, agents, port=port, random_policy=random_policy)
            dqn_waits.append(dm.avg_waiting_time_s)
            dqn_co2s.append(dm.total_co2_mg)
            dqn_thrus.append(dm.throughput)

        def _avg(lst): return float(np.mean(lst)) if lst else 0.0

        dqn_avg = EpisodeMetrics(
            profile            = profile,
            mode               = "dqn_adaptive" if not random_policy else "random_policy",
            avg_waiting_time_s = _avg(dqn_waits),
            total_co2_mg       = _avg(dqn_co2s),
            throughput         = int(_avg(dqn_thrus)),
        )

        improvements  = dqn_avg.improvement_vs(base_m)
        kpi_wait_pass = improvements["avg_wait_pct"] > 0
        kpi_co2_pass  = improvements["total_co2_pct"] > 0

        results[profile] = {
            "baseline":      base_m.to_dict(),
            "dqn":           dqn_avg.to_dict(),
            "improvements":  improvements,
            "kpi_wait_pass": kpi_wait_pass,
            "kpi_co2_pass":  kpi_co2_pass,
            "random_policy": random_policy,
        }

        _print_profile_table(profile, base_m, dqn_avg, improvements, random_policy)

    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Report saved: %s", report_path)

    _print_final_summary(results, random_policy)
    return results


# ── Print helpers ─────────────────────────────────────────────────

def _print_profile_table(
    profile:       str,
    base:          EpisodeMetrics,
    dqn:           EpisodeMetrics,
    imp:           dict,
    random_policy: bool = False,
):
    label = "RANDOM" if random_policy else "DQN"
    print(f"\n  Profile: {profile.upper()}  [{label}]")
    print(f"  {'Metric':<30} {'Baseline':>16} {label:>16} {'Change':>9}")
    print("  " + "─" * 72)

    rows = [
        ("Avg waiting time (s)",  base.avg_waiting_time_s, dqn.avg_waiting_time_s, imp.get("avg_wait_pct")),
        ("Avg travel time (s)",   base.avg_travel_time_s,  dqn.avg_travel_time_s,  imp.get("avg_travel_pct")),
        ("Total CO2 (mg)",        base.total_co2_mg,       dqn.total_co2_mg,       imp.get("total_co2_pct")),
        ("Throughput (vehicles)", float(base.throughput),  float(dqn.throughput),  None),
    ]
    for name, bv, dv, pct in rows:
        if pct is not None:
            pct_str = f"{pct:+.1f}%"
            arrow   = "✓" if pct > 0 else "✗"
        else:
            delta   = imp.get("throughput_delta", 0)
            pct_str = f"Δ{delta:+d}"
            arrow   = "✓" if delta >= 0 else "✗"
        print(f"  {name:<30} {bv:>16.1f} {dv:>16.1f} {pct_str:>9} {arrow}")


def _print_final_summary(results: dict, random_policy: bool = False):
    label = "RANDOM POLICY" if random_policy else "DQN"
    print("\n" + "=" * 60)
    print(f"  SUMMARY [{label}]")
    print("=" * 60)
    print(f"  {'Profile':<22} {'Wait ↓':>10} {'CO2 ↓':>10}")
    print("  " + "─" * 44)
    for profile, r in results.items():
        w = "✓ PASS" if r["kpi_wait_pass"] else "✗ FAIL"
        c = "✓ PASS" if r["kpi_co2_pass"]  else "✗ FAIL"
        print(f"  {profile:<22} {w:>10} {c:>10}")
    print("=" * 60)
    if random_policy:
        print("  ℹ Pipeline test complete. Train DQN for real results.")
    print()


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate SUMOFlow AI DQN vs fixed-time baseline"
    )
    p.add_argument("--config",     default="DeepQN/configs/dqn_config.yaml")
    p.add_argument("--checkpoint", default="DeepQN/checkpoints")
    p.add_argument("--profiles",   nargs="+", default=None)
    p.add_argument("--runs",  type=int, default=1)
    p.add_argument("--output", default="DeepQN/results")
    p.add_argument("--port",   type=int, default=8813)
    args = p.parse_args()

    compare(
        config_path    = args.config,
        checkpoint_dir = args.checkpoint,
        profiles       = args.profiles,
        n_runs         = args.runs,
        output_dir     = args.output,
        port           = args.port,
    )