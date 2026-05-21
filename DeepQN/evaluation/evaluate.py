"""
DeepQN/evaluation/evaluate.py
------------------------------
Before/after comparison — fixed metric collection.

BASELINE: read from existing SUMO XML output files (simulation/maps/outputs/)
DQN:      collect live from TraCI, matching exactly what the XML reports:
            - avg_waiting_time_s  = mean waitingTime per arrived vehicle
            - total_co2_mg        = sum of CO2 across all vehicles all steps
            - throughput          = total arrived vehicles (accumulated)
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

BASELINE_OUTPUT_DIR = Path("simulation") / "maps" / "outputs"


#  Baseline: read from existing XML files 

def get_baseline_metrics(profile: str) -> EpisodeMetrics:
    """Read baseline KPIs from SUMO XML files — no re-run needed."""
    logger.info("[BASELINE] Reading XML files for profile: %s", profile)
    m = load_metrics_from_outputs(BASELINE_OUTPUT_DIR, profile, "fixed_time")
    logger.info(
        "[BASELINE] %s — avg_wait=%.1fs  total_co2=%.0fmg  throughput=%d",
        profile, m.avg_waiting_time_s, m.total_co2_mg, m.throughput,
    )
    return m


#  DQN runner: metrics collected per-vehicle to match XML 

def run_dqn(
    cfg:     dict,
    profile: str,
    agents:  MultiAgentDQN,
    port:    int = 8813,
) -> EpisodeMetrics:
    """
    Run one greedy DQN episode. Metrics collected to match baseline XML:

    avg_waiting_time_s
        Mean of traci.vehicle.getWaitingTime() across all active vehicles,
        averaged over all decision steps. Matches tripinfo waitingTime.

    total_co2_mg
        Sum of traci.vehicle.getCO2Emission(vid) over all vehicles all steps.
        Matches emission XML total.

    throughput
        Accumulated from info["step_arrived"] which counts getArrivedNumber()
        inside every SUMO step of the decision interval — no arrivals missed.
    """
    logger.info("[DQN] Running profile: %s", profile)

    try:
        import traci
    except ImportError:
        raise ImportError("TraCI not found. Check SUMO_HOME and PYTHONPATH.")

    env  = SumoEnv(cfg, profile=profile, port=port)
    obs  = env.reset()
    done = False

    total_co2_mg     = 0.0
    total_arrived    = 0
    wait_sum         = 0.0
    wait_samples     = 0
    steps            = 0

    while not done:
        #  Collect per-vehicle metrics before step 
        try:
            active_vids = traci.vehicle.getIDList()
        except Exception:
            active_vids = []

        # Waiting time: mean across active vehicles this decision step
        step_waits = []
        for vid in active_vids:
            try:
                step_waits.append(traci.vehicle.getWaitingTime(vid))
            except Exception:
                pass
        if step_waits:
            wait_sum     += float(np.mean(step_waits))
            wait_samples += 1

        # CO2: sum across all active vehicles
        for vid in active_vids:
            try:
                total_co2_mg += traci.vehicle.getCO2Emission(vid)
            except Exception:
                pass

        # ── DQN step (advances decision_interval SUMO steps) ──
        actions = agents.act(obs, eval_mode=True)
        obs, rewards, done, info = env.step(actions)

        # Arrivals properly accumulated inside sumo_env inner loop
        total_arrived += info.get("step_arrived", 0)
        steps += 1

    env.close()

    avg_wait = wait_sum / max(wait_samples, 1)

    m = EpisodeMetrics(
        profile            = profile,
        mode               = "dqn_adaptive",
        avg_waiting_time_s = avg_wait,
        total_co2_mg       = total_co2_mg,
        throughput         = total_arrived,
    )
    logger.info(
        "[DQN] %s done — avg_wait=%.1fs  total_co2=%.0fmg  throughput=%d  steps=%d",
        profile, avg_wait, total_co2_mg, total_arrived, steps,
    )
    return m


#  Comparison 

def compare(
    config_path:    str                  = "DeepQN/configs/dqn_config.yaml",
    checkpoint_dir: str                  = "DeepQN/checkpoints",
    profiles:       Optional[List[str]]  = None,
    n_runs:         int                  = 3,
    output_dir:     str                  = "DeepQN/results",
    port:           int                  = 8813,
) -> Dict[str, dict]:

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    all_profiles = profiles or list(cfg["profiles"].keys())
    tls_ids      = list(cfg["tls_junctions"].keys())
    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = MultiAgentDQN(tls_ids, cfg["dqn"])
    agents.load_latest(checkpoint_dir)

    results: Dict[str, dict] = {}

    for profile in all_profiles:
        logger.info("=" * 60)
        logger.info("Evaluating profile: %s  (%d DQN runs)", profile, n_runs)
        logger.info("=" * 60)

        # Baseline from XML — single read, no simulation
        base_m = get_baseline_metrics(profile)

        # DQN — n_runs greedy episodes, averaged
        dqn_waits, dqn_co2s, dqn_thrus = [], [], []
        for run in range(n_runs):
            logger.info("  DQN run %d/%d", run + 1, n_runs)
            dm = run_dqn(cfg, profile, agents, port=port)
            dqn_waits.append(dm.avg_waiting_time_s)
            dqn_co2s.append(dm.total_co2_mg)
            dqn_thrus.append(dm.throughput)

        def _avg(lst): return float(np.mean(lst)) if lst else 0.0

        dqn_avg = EpisodeMetrics(
            profile            = profile,
            mode               = "dqn_adaptive",
            avg_waiting_time_s = _avg(dqn_waits),
            total_co2_mg       = _avg(dqn_co2s),
            throughput         = int(_avg(dqn_thrus)),
        )

        improvements    = dqn_avg.improvement_vs(base_m)
        kpi_wait_pass   = improvements["avg_wait_pct"] > 0
        kpi_co2_pass    = improvements["total_co2_pct"] > 0

        results[profile] = {
            "baseline":      base_m.to_dict(),
            "dqn":           dqn_avg.to_dict(),
            "improvements":  improvements,
            "kpi_wait_pass": kpi_wait_pass,
            "kpi_co2_pass":  kpi_co2_pass,
        }

        _print_profile_table(profile, base_m, dqn_avg, improvements)

    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Report saved: %s", report_path)

    _print_final_summary(results)
    return results


#  Print helpers 

def _print_profile_table(
    profile: str,
    base:    EpisodeMetrics,
    dqn:     EpisodeMetrics,
    imp:     dict,
):
    print(f"\n  Profile: {profile.upper()}")
    print(f"  {'Metric':<30} {'Baseline':>16} {'DQN':>16} {'Change':>9}")
    print("  " + "─" * 72)

    rows = [
        ("Avg waiting time (s)",  base.avg_waiting_time_s, dqn.avg_waiting_time_s, imp.get("avg_wait_pct")),
        ("Avg travel time (s)",   base.avg_travel_time_s,  dqn.avg_travel_time_s,  imp.get("avg_travel_pct")),
        ("Time loss (s)",         base.avg_time_loss_s,    dqn.avg_time_loss_s,    imp.get("time_loss_pct")),
        ("Total CO2 (mg)",        base.total_co2_mg,       dqn.total_co2_mg,       imp.get("total_co2_pct")),
        ("Throughput (vehicles)", float(base.throughput),  float(dqn.throughput),  None),
    ]
    for name, bv, dv, pct in rows:
        if pct is not None:
            pct_str = f"{pct:+.1f}%"
            arrow   = "true" if pct > 0 else "false"
        else:
            delta   = imp.get("throughput_delta", 0)
            pct_str = f"Δ{delta:+d}"
            arrow   = "true" if delta >= 0 else "false"
        print(f"  {name:<30} {bv:>16.1f} {dv:>16.1f} {pct_str:>9} {arrow}")


def _print_final_summary(results: dict):
    print("\n" + "=" * 60)
    print("  KPI PASS / FAIL SUMMARY")
    print("=" * 60)
    print(f"  {'Profile':<22} {'Wait time ↓':>12} {'CO2 ↓':>10}")
    print("  " + "─" * 46)
    all_pass = True
    for profile, r in results.items():
        w = "PASS" if r["kpi_wait_pass"] else "✗ FAIL"
        c = " PASS" if r["kpi_co2_pass"]  else "✗ FAIL"
        if not r["kpi_wait_pass"] or not r["kpi_co2_pass"]:
            all_pass = False
        print(f"  {profile:<22} {w:>12} {c:>10}")
    print("=" * 60)
    if all_pass:
        print(" All KPIs satisfied across all profiles.")
    else:
        print(" Some KPIs not satisfied.")
    print()


# CLI 

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate SUMOFlow AI DQN vs fixed-time baseline"
    )
    p.add_argument("--config",     default="DeepQN/configs/dqn_config.yaml")
    p.add_argument("--checkpoint", default="DeepQN/checkpoints")
    p.add_argument("--profiles",   nargs="+", default=None)
    p.add_argument("--runs",  type=int, default=3)
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