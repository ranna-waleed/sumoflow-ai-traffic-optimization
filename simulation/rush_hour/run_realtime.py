import traci
import time
import csv
import os
import sys
from datetime import datetime

# ─── Config Files per Profile ─────────────────────────
CONFIGS = {
    "morning_rush": "simulation/maps/config_morning_rush.sumocfg",
    "evening_rush": "simulation/maps/config_evening_rush.sumocfg",
    "midday":       "simulation/maps/config_midday.sumocfg",
    "night":        "simulation/maps/config_night.sumocfg",
    "realtime":     "simulation/maps/config_realtime.sumocfg",
}

# ─── Auto Detect Current Profile ─────────────────────
def get_current_profile():
    hour = datetime.now().hour
    if   8  <= hour < 10: return "morning_rush"
    elif 16 <= hour < 19: return "evening_rush"
    elif 12 <= hour < 14: return "midday"
    else:                 return "night"

def get_period_label(profile):
    labels = {
        "morning_rush": " MORNING RUSH HOUR (8AM-10AM)",
        "evening_rush": " EVENING RUSH HOUR (4PM-7PM)",
        "midday":       " MIDDAY TRAFFIC (12PM-2PM)",
        "night":        " NIGHT LIGHT TRAFFIC (10PM-6AM)",
        "realtime":     " FULL 24H REAL-TIME",
    }
    return labels.get(profile, profile)

# ─── Main Simulation ──────────────────────────────────
def run_simulation(profile_name=None):

    # Auto detect if not given
    if profile_name is None:
        profile_name = get_current_profile()
        print(f"Auto-detected profile: {profile_name}")

    if profile_name not in CONFIGS:
        print(f" Unknown profile: {profile_name}")
        print(f"   Options: {list(CONFIGS.keys())}")
        return

    config = CONFIGS[profile_name]
    now    = datetime.now()

    print(f"  SUMO REAL-TIME RUSH HOUR SIMULATION")
    print(f"  Profile:  {get_period_label(profile_name)}")
    print(f"  Time:     {now.strftime('%H:%M:%S')}")
    print(f"  Config:   {config}")

    # ─── Start SUMO ───────────────────────────────────
    sumo_cmd = [
        "sumo-gui",
        "-c",            config,
        "--step-length", "1",
        "--delay",       "0",  # 1 real sec = 1 sim sec
        "--start",
        "--quit-on-end",
        "--no-warnings"
    ]

    traci.start(sumo_cmd)
    print(" SUMO started!")
    print("   Speed: 1 simulation second = 1 real second\n")

    # ─── Collect Metrics ──────────────────────────────
    metrics    = []
    step       = 0
    start_time = time.time()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Print every 60 steps = every 60 real seconds
        if step % 60 == 0:
            vehicles   = traci.vehicle.getIDList()
            n_vehicles = len(vehicles)

            if n_vehicles > 0:
                avg_wait = sum(
                    traci.vehicle.getWaitingTime(v)
                    for v in vehicles
                ) / n_vehicles

                total_co2 = sum(
                    traci.vehicle.getCO2Emission(v)
                    for v in vehicles
                )

                metrics.append({
                    "step":      step,
                    "real_time": datetime.now().strftime("%H:%M:%S"),
                    "vehicles":  n_vehicles,
                    "avg_wait":  round(avg_wait, 2),
                    "co2":       round(total_co2, 2),
                    "profile":   profile_name
                })

                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Step {step:5d} | "
                    f"Vehicles: {n_vehicles:3d} | "
                    f"Avg Wait: {avg_wait:.1f}s | "
                    f"CO2: {total_co2:.0f}mg"
                )

    traci.close()

    # ─── Save Results ─────────────────────────────────
    os.makedirs("simulation/maps/outputs", exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M")
    csv_path  = (f"simulation/maps/outputs/"
                 f"metrics_{profile_name}_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "real_time", "vehicles",
            "avg_wait", "co2", "profile"
        ])
        writer.writeheader()
        writer.writerows(metrics)

    elapsed = time.time() - start_time
    print(f"  SIMULATION COMPLETE ")
    print(f"  Profile:  {profile_name}")
    print(f"  Steps:    {step}")
    print(f"  Elapsed:  {elapsed:.0f}s real time")
    print(f"  Results:  {csv_path}")

    return metrics

# ─── Entry Point ──────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_simulation(sys.argv[1])
    else:
        run_simulation()