import traci
import time
import csv
import os
import sys
import argparse
from datetime import datetime

# Config Files per Profile
CONFIGS = {
    "morning_rush": "simulation/maps/config_morning_rush.sumocfg",
    "evening_rush": "simulation/maps/config_evening_rush.sumocfg",
    "midday":       "simulation/maps/config_midday.sumocfg",
    "night":        "simulation/maps/config_night.sumocfg",
    "realtime":     "simulation/maps/config_realtime.sumocfg",
}

def get_current_profile():
    hour = datetime.now().hour
    if 7 <= hour < 11:
        return "morning_rush"
    elif 15 <= hour < 20:
        return "evening_rush"
    elif 12 <= hour < 15:
        return "midday"
    else:
        return "night"


def get_period_label(profile):
    labels = {
        "morning_rush": "MORNING RUSH HOUR (7:30AM-10:30AM)",
        "evening_rush": "EVENING RUSH HOUR (3PM-8PM)",
        "midday":       "MIDDAY TRAFFIC (12PM-3PM)",
        "night":        "NIGHT LIGHT TRAFFIC (10PM-12AM)",
        "realtime":     "FULL 24H REAL-TIME",
    }
    return labels.get(profile, profile)


def run_simulation(profile_name=None):

    if profile_name is None:
        profile_name = get_current_profile()
        print(f"Auto-detected profile: {profile_name}")

    if profile_name not in CONFIGS:
        print(f"Unknown profile: {profile_name}")
        print(f"  Options: {list(CONFIGS.keys())}")
        return

    config = CONFIGS[profile_name]
    now    = datetime.now()

    print("  SUMO REAL-TIME RUSH HOUR SIMULATION")
    print(f"  Profile:  {get_period_label(profile_name)}")
    print(f"  Time:     {now.strftime('%H:%M:%S')}")
    print(f"  Config:   {config}")

    sumo_cmd = [
        "sumo-gui",
        "-c",            config,
        "--step-length", "1",
        "--delay",       "0",
        "--start",
        "--quit-on-end",
        "--no-warnings",
    ]

    traci.start(sumo_cmd)
    print("SUMO started!")
    print("  Speed: 1 simulation second = 1 real second\n")

    metrics    = []
    step       = 0
    start_time = time.time()

    # End times matching the config files
    END_TIMES = {
        "morning_rush": 37800,   # 10:30 AM
        "evening_rush": 72000,   # 8:00 PM
        "midday":       54000,   # 3:00 PM
        "night":        86400,   # 12:00 AM
        "realtime":     86400,
    }

    end_time = END_TIMES.get(profile_name, 86400)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Collect metrics every 60 steps — BEFORE break check
        if step % 60 == 0:
            vehicles   = traci.vehicle.getIDList()
            n_vehicles = len(vehicles)

            if n_vehicles > 0:
                avg_wait  = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / n_vehicles
                total_co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)

                metrics.append({
                    "step":      step,
                    "real_time": datetime.now().strftime("%H:%M:%S"),
                    "vehicles":  n_vehicles,
                    "avg_wait":  round(avg_wait, 2),
                    "co2":       round(total_co2, 2),
                    "profile":   profile_name,
                })

                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Step {step:5d} | "
                    f"Vehicles: {n_vehicles:3d} | "
                    f"Avg Wait: {avg_wait:.1f}s | "
                    f"CO2: {total_co2:.0f}mg"
                )

        # Stop at config end time — AFTER collecting metrics
        if traci.simulation.getTime() >= end_time:
            break
    traci.close()   

    os.makedirs("simulation/maps/outputs", exist_ok=True)

    os.makedirs("simulation/maps/outputs", exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M")
    csv_path  = f"simulation/maps/outputs/metrics_{profile_name}_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "real_time", "vehicles",
            "avg_wait", "co2", "profile"
        ])
        writer.writeheader()
        writer.writerows(metrics)

    elapsed = time.time() - start_time
    print("  SIMULATION COMPLETE ")
    print(f"  Profile:  {profile_name}")
    print(f"  Steps:    {step}")
    print(f"  Elapsed:  {elapsed:.0f}s real time")
    print(f"  Results:  {csv_path}")

    return metrics


if __name__ == "__main__":
    # proper argument parsing — supports both:
    #   python run_realtime.py morning_rush
    #   python run_realtime.py --profile morning_rush
    parser = argparse.ArgumentParser(description="SUMOFlow Rush Hour Simulation")
    parser.add_argument(
        "profile",
        nargs="?",          # optional positional
        default=None,
        help="Profile to run",
    )
    parser.add_argument(
        "--profile",
        dest="profile_flag",
        default=None,
        help="Profile to run (flag form)",
    )
    args = parser.parse_args()

    # --profile flag takes priority, then positional, then auto-detect
    chosen = args.profile_flag or args.profile
    run_simulation(chosen)