# dqn/run_baseline.py
# Run SUMO simulation with FIXED timing (no DQN) to get baseline metrics
# This is the BEFORE data for Before/After comparison
# Run: python dqn/run_baseline.py

import os, sys, json
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR    = os.path.join(BASE_DIR, "simulation", "maps")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PROFILES = {
    "morning_rush": os.path.join(MAPS_DIR, "config_morning_rush.sumocfg"),
    "evening_rush": os.path.join(MAPS_DIR, "config_evening_rush.sumocfg"),
    "midday":       os.path.join(MAPS_DIR, "config_midday.sumocfg"),
    "night":        os.path.join(MAPS_DIR, "config_night.sumocfg"),
}

MAX_STEPS = 3600


def run_fixed_timing(profile: str) -> dict:
    """Run one simulation with SUMO's original fixed timing."""
    print(f"  Running fixed timing: {profile}...")

    try:
        traci.close()
    except Exception:
        pass

    traci.start([
        "sumo", "-c", PROFILES[profile],
        "--no-warnings", "--no-step-log",
    ], port=8816)

    step       = 0
    waits      = []
    co2_steps  = []
    throughput = 0

    while step < MAX_STEPS:
        if traci.simulation.getMinExpectedNumber() == 0:
            break

        traci.simulationStep()
        step += 1

        vehicle_ids = traci.vehicle.getIDList()
        step_waits, step_co2 = [], []

        for v in vehicle_ids:
            try:
                step_waits.append(traci.vehicle.getWaitingTime(v))
                step_co2.append(traci.vehicle.getCO2Emission(v))
            except Exception:
                continue

        if step_waits:
            waits.append(sum(step_waits) / len(step_waits))
        if step_co2:
            co2_steps.append(sum(step_co2) / max(step, 1))  # per step like run_dqn

        throughput = max(throughput, len(vehicle_ids))

    departed = traci.simulation.getDepartedNumber()

    try:
        traci.close()
    except Exception:
        pass

    return {
        "profile":      profile,
        "avg_wait_s":   float(np.mean(waits))     if waits     else 0.0,
        "max_wait_s":   float(np.max(waits))       if waits     else 0.0,
        "avg_co2_mg":   float(np.mean(co2_steps))  if co2_steps else 0.0,
        "total_co2_mg": float(np.sum(co2_steps))   if co2_steps else 0.0,
        "peak_vehicles": throughput,
        "total_steps":  step,
    }


def main():
    print("  SUMOFlow AI — Fixed Timing Baseline")
    print("  Recording BEFORE metrics (no DQN)")

    results = {}
    for profile in PROFILES:
        results[profile] = run_fixed_timing(profile)
        r = results[profile]
        print(f"  {profile:<15}: wait={r['avg_wait_s']:.3f}s | "
              f"CO2={r['avg_co2_mg']/1000:.1f}k mg | "
              f"vehicles={r['peak_vehicles']}")

    # Overall averages
    all_waits = [r["avg_wait_s"]   for r in results.values()]
    all_co2   = [r["avg_co2_mg"]   for r in results.values()]

    summary = {
        "type":         "fixed_timing_baseline",
        "profiles":     results,
        "overall": {
            "avg_wait_s": float(np.mean(all_waits)),
            "avg_co2_mg": float(np.mean(all_co2)),
        }
    }

    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Overall avg wait:  {summary['overall']['avg_wait_s']:.3f}s")
    print(f"  Overall avg CO2:   {summary['overall']['avg_co2_mg']/1000:.1f}k mg")
    print(f"  Saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()