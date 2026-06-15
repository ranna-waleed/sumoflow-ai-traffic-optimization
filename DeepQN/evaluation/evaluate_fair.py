"""
DeepQN/evaluation/evaluate_fair.py
FAIR comparison:
  - Baseline : loaded directly from existing evaluation_report.json
               (the real fixed-time simulation results already on disk)
  - DQN      : run live simulation, save SUMO XML, parse via metrics.py

This guarantees the baseline numbers are the genuine fixed-time results
while the DQN uses identical XML-based measurement methodology.

Usage:
    python -m DeepQN.evaluation.evaluate_fair
    python -m DeepQN.evaluation.evaluate_fair --profiles morning_rush --runs 1
    python -m DeepQN.evaluation.evaluate_fair --baseline DeepQN/results/evaluation_report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

#Patch SUMO sleep time BEFORE importing SumoEnv
# Windows needs more time than Linux for SUMO to open the TraCI port.
import subprocess
import DeepQN.env.sumo_env as _sumo_env_module

_original_launch = _sumo_env_module.SumoEnv._launch_sumo

def _patched_launch(self):
    import traci
    profile_config = self.profile_cfg["config"]
    cmd = [
        self.sumo_binary,
        "-c", profile_config,
        "--start",
        "--quit-on-end",
        "--no-step-log",
        "--remote-port", str(self.port),
    ]
    logging.getLogger(__name__).info(
        "Launching SUMO: %s", " ".join(cmd)
    )
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6.0)   # Windows needs more time than default 1.5s
    traci.init(self.port)

_sumo_env_module.SumoEnv._launch_sumo = _patched_launch

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.env.sumo_env import SumoEnv
from DeepQN.evaluation.metrics import EpisodeMetrics, load_metrics_from_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# Load baseline from existing evaluation_report.json

def get_baseline_from_report(
    report_path: Path,
    profile:     str,
) -> EpisodeMetrics:
    """
    Load baseline metrics directly from the existing evaluation_report.json.
    These are the genuine fixed-time simulation results already on disk.
    No re-simulation needed — these numbers are the real baseline.
    """
    if not report_path.exists():
        raise FileNotFoundError(
            f"evaluation_report.json not found at: {report_path}\n"
            f"Make sure you have run the original evaluate.py first."
        )

    with open(report_path) as f:
        report = json.load(f)

    if profile not in report:
        raise KeyError(
            f"Profile '{profile}' not found in report.\n"
            f"Available profiles: {list(report.keys())}"
        )

    b = report[profile]["baseline"]

    m = EpisodeMetrics(
        profile            = profile,
        mode               = "fixed_time",
        n_trips            = b.get("n_trips", 0),
        avg_waiting_time_s = b.get("avg_waiting_time_s", 0.0),
        avg_travel_time_s  = b.get("avg_travel_time_s",  0.0),
        avg_time_loss_s    = b.get("avg_time_loss_s",    0.0),
        total_co2_mg       = b.get("total_co2_mg",       0.0),
        avg_co2_per_veh_mg = b.get("avg_co2_per_veh_mg", 0.0),
        throughput         = b.get("throughput",          0),
        avg_speed_ms       = b.get("avg_speed_ms",        0.0),
    )

    logger.info(
        "[BASELINE] %s loaded from JSON — "
        "avg_wait=%.1fs  co2=%.0fmg  throughput=%d",
        profile, m.avg_waiting_time_s, m.total_co2_mg, m.throughput,
    )
    return m


# Patch sumocfg to add XML output flags

def _set_or_create(parent: ET.Element, tag: str, value: str) -> None:
    """Set attribute 'value' on child tag, creating child if missing."""
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.set("value", value)


def _patch_sumocfg(
    original_path:   str,
    tripinfo_output: str,
    emission_output: str,
) -> str:
    """
    Read original .sumocfg, inject output flags, write temp file
    IN THE SAME DIRECTORY as original so relative map paths work.
    Handles both <output> and <o> tag names.
    Original file is NEVER modified.
    """
    original = Path(original_path)
    tree     = ET.parse(str(original))
    root     = tree.getroot()

    # Remove existing output children from ANY output-like section
    # The config uses <o> tag not <output> : handle both
    for section in list(root):
        if section.tag in ("output", "o"):
            for child in list(section):
                if child.tag in (
                    "tripinfo-output", "emission-output",
                    "fcd-output", "summary-output", "lanechange-output",
                ):
                    section.remove(child)

    # Find output section,try <output> first, then <o>, then create new
    output_elem = root.find("output")
    if output_elem is None:
        output_elem = root.find("o")
    if output_elem is None:
        output_elem = ET.SubElement(root, "output")

    # Use absolute paths so SUMO writes files to the right place
    _set_or_create(output_elem, "tripinfo-output",
                   str(Path(tripinfo_output).resolve()))
    _set_or_create(output_elem, "emission-output",
                   str(Path(emission_output).resolve()))

    # Write temp config NEXT TO original so map paths resolve
    tmp = tempfile.NamedTemporaryFile(
        suffix=".sumocfg",
        mode="wb",
        dir=str(original.parent),
        delete=False,
    )
    tree.write(tmp)
    tmp.close()
    logger.info("[PATCH] Temp config: %s", tmp.name)
    return tmp.name


#  DQN run: saves XML, parses via metrics.py

def run_dqn_fair(
    cfg:           dict,
    profile:       str,
    agents:        MultiAgentDQN,
    output_dir:    Path,
    run_index:     int  = 0,
    port:          int  = 8813,
    random_policy: bool = False,
) -> EpisodeMetrics:
    """
    Run one DQN episode with SUMO XML output enabled.
    Metrics parsed via same metrics.py pipeline as original baseline.
    """
    mode_label = "RANDOM" if random_policy else "DQN"

    # Absolute output paths
    suffix        = f"dqn_{profile}_run{run_index}"
    tripinfo_path = (output_dir / f"tripinfo_{suffix}.xml").resolve()
    emission_path = (output_dir / f"emission_{suffix}.xml").resolve()

    # Patch sumocfg next to original
    profile_cfg_path = cfg["profiles"][profile]["config"]
    patched_cfg_path = _patch_sumocfg(
        original_path   = profile_cfg_path,
        tripinfo_output = str(tripinfo_path),
        emission_output = str(emission_path),
    )

    # Patch config dict
    patched_cfg = copy.deepcopy(cfg)
    patched_cfg["profiles"][profile]["config"] = patched_cfg_path

    logger.info("[%s] profile=%s  run=%d", mode_label, profile, run_index)
    logger.info("[%s] tripinfo -> %s", mode_label, tripinfo_path)
    logger.info("[%s] emission -> %s", mode_label, emission_path)

    # Run simulation
    env   = SumoEnv(patched_cfg, profile=profile, port=port)
    obs   = env.reset()
    done  = False
    steps = 0

    while not done:
        if random_policy:
            actions = {tid: np.random.randint(0, 2)
                       for tid in agents.agents}
        else:
            actions = agents.act(obs, eval_mode=True)

        obs, rewards, done, info = env.step(actions)
        steps += 1

    env.close()
    logger.info("[%s] Simulation done in %d steps.", mode_label, steps)

    # Clean up temp sumocfg
    try:
        Path(patched_cfg_path).unlink()
    except Exception:
        pass

    # Parse XML: same metrics.py pipeline as original baseline
    m = load_metrics_from_outputs(
        output_dir      = output_dir,
        profile         = profile,
        mode            = "dqn_adaptive" if not random_policy else "random_policy",
        suffix_override = suffix,
    )
    logger.info(
        "[%s] %s — avg_wait=%.1fs  co2=%.0fmg  throughput=%d",
        mode_label, profile,
        m.avg_waiting_time_s, m.total_co2_mg, m.throughput,
    )
    return m


# Main comparison 

def compare_fair(
    config_path:    str                 = "DeepQN/configs/dqn_config.yaml",
    checkpoint_dir: str                 = "DeepQN/checkpoints",
    baseline_report:str                 = "DeepQN/results/evaluation_report.json",
    profiles:       Optional[List[str]] = None,
    n_runs:         int                 = 3,
    output_dir:     str                 = "DeepQN/results/fair",
    port:           int                 = 8813,
) -> Dict[str, dict]:

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    all_profiles = profiles or list(cfg["profiles"].keys())
    tls_ids      = list(cfg["tls_junctions"].keys())
    out_dir      = Path(output_dir)
    report_path  = Path(baseline_report)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load agents
    agents = MultiAgentDQN(tls_ids, cfg["dqn"])
    try:
        agents.load_latest(checkpoint_dir)
        random_policy = False
        logger.info("Loaded checkpoints from %s", checkpoint_dir)
    except Exception as e:
        random_policy = True
        logger.warning("No checkpoints (%s) — using RANDOM policy", e)

    results: Dict[str, dict] = {}

    for profile in all_profiles:
        logger.info("=" * 60)
        logger.info("FAIR Evaluation — profile: %s  (%d runs)", profile, n_runs)
        logger.info("=" * 60)

        # Baseline from existing JSON , no re-simulation needed
        base_m = get_baseline_from_report(report_path, profile)

        # DQN runs: save XML, parse via metrics.py
        dqn_waits, dqn_co2s, dqn_thrus = [], [], []
        for run in range(n_runs):
            logger.info("  Run %d / %d", run + 1, n_runs)
            dm = run_dqn_fair(
                cfg           = cfg,
                profile       = profile,
                agents        = agents,
                output_dir    = out_dir,
                run_index     = run,
                port          = port,
                random_policy = random_policy,
            )
            dqn_waits.append(dm.avg_waiting_time_s)
            dqn_co2s.append(dm.total_co2_mg)
            dqn_thrus.append(dm.throughput)

        def _avg(lst): return float(np.mean(lst)) if lst else 0.0

        dqn_avg = EpisodeMetrics(
            profile            = profile,
            mode               = "dqn_adaptive" if not random_policy
                                 else "random_policy",
            avg_waiting_time_s = _avg(dqn_waits),
            total_co2_mg       = _avg(dqn_co2s),
            throughput         = int(_avg(dqn_thrus)),
        )

        improvements  = dqn_avg.improvement_vs(base_m)
        kpi_wait_pass = improvements["avg_wait_pct"] > 0
        kpi_co2_pass  = improvements["total_co2_pct"] > 0

        results[profile] = {
            "baseline":         base_m.to_dict(),
            "dqn":              dqn_avg.to_dict(),
            "improvements":     improvements,
            "kpi_wait_pass":    kpi_wait_pass,
            "kpi_co2_pass":     kpi_co2_pass,
            "random_policy":    random_policy,
            "measurement_note": (
                "FAIR: baseline from original evaluation_report.json "
                "(genuine fixed-time results). DQN measured via "
                "SUMO XML + identical metrics.py pipeline."
            ),
        }

        _print_table(profile, base_m, dqn_avg, improvements, random_policy)

    report_out = out_dir / "evaluation_report_fair.json"
    with open(report_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Fair report saved: %s", report_out)

    _print_summary(results, random_policy)
    return results


# Print helpers

def _print_table(profile, base, dqn, imp, random_policy):
    label = "RANDOM" if random_policy else "DQN FAIR"
    print(f"\n  Profile: {profile.upper()}  [{label}]")
    print(f"  {'Metric':<30} {'Baseline':>16} {label:>14} {'Change':>9}")
    print("  " + "-" * 72)
    rows = [
        ("Avg waiting time (s)",
         base.avg_waiting_time_s, dqn.avg_waiting_time_s,
         imp.get("avg_wait_pct")),
        ("Avg travel time (s)",
         base.avg_travel_time_s,  dqn.avg_travel_time_s,
         imp.get("avg_travel_pct")),
        ("Total CO2 (mg)",
         base.total_co2_mg,       dqn.total_co2_mg,
         imp.get("total_co2_pct")),
        ("Throughput (vehicles)",
         float(base.throughput),  float(dqn.throughput),
         None),
    ]
    for name, bv, dv, pct in rows:
        if pct is not None:
            pct_str = f"{pct:+.1f}%"
            mark    = "PASS" if pct > 0 else "FAIL"
        else:
            delta   = imp.get("throughput_delta", 0)
            pct_str = f"D{delta:+d}"
            mark    = "PASS" if delta >= 0 else "FAIL"
        print(f"  {name:<30} {bv:>16.1f} {dv:>14.1f} {pct_str:>9} {mark}")


def _print_summary(results, random_policy):
    label = "RANDOM" if random_policy else "DQN FAIR"
    print("\n" + "=" * 65)
    print(f"  FAIR COMPARISON SUMMARY [{label}]")
    print("  Baseline: original evaluation_report.json (genuine fixed-time)")
    print("  DQN:      fresh XML simulation via identical metrics.py")
    print("=" * 65)
    print(f"  {'Profile':<22} {'Wait':>10} {'CO2':>10}")
    print("  " + "-" * 44)
    for profile, r in results.items():
        w = "PASS" if r["kpi_wait_pass"] else "FAIL"
        c = "PASS" if r["kpi_co2_pass"]  else "FAIL"
        print(f"  {profile:<22} {w:>10} {c:>10}")
    print("=" * 65)
    print()


# CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Fair DQN evaluation — baseline from existing JSON, "
            "DQN from fresh SUMO XML via identical metrics.py pipeline"
        )
    )
    p.add_argument("--config",     default="DeepQN/configs/dqn_config.yaml")
    p.add_argument("--checkpoint", default="DeepQN/checkpoints")
    p.add_argument("--baseline",
                   default="DeepQN/results/evaluation_report.json",
                   help="Path to existing evaluation_report.json")
    p.add_argument("--profiles",   nargs="+", default=None)
    p.add_argument("--runs",       type=int,  default=3)
    p.add_argument("--output",     default="DeepQN/results/fair")
    p.add_argument("--port",       type=int,  default=8813)
    args = p.parse_args()

    compare_fair(
        config_path     = args.config,
        checkpoint_dir  = args.checkpoint,
        baseline_report = args.baseline,
        profiles        = args.profiles,
        n_runs          = args.runs,
        output_dir      = args.output,
        port            = args.port,
    )