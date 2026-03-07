import csv
import os
import glob

def compare_profiles():
    """
    Reads all 4 profile CSV files and creates
    a single comparison summary
    """

    output_dir = "simulation/maps/outputs"
    profiles   = ["morning_rush", "evening_rush", "midday", "night"]
    summary    = []

    for profile in profiles:
        # Find latest CSV for this profile
        pattern = f"{output_dir}/metrics_{profile}_*.csv"
        files   = glob.glob(pattern)

        if not files:
            print(f" No file found for {profile}")
            continue

        # Use most recent file
        latest = max(files)
        print(f"Reading: {latest}")

        vehicles_list = []
        wait_list     = []
        co2_list      = []

        with open(latest, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vehicles_list.append(int(row["vehicles"]))
                wait_list.append(float(row["avg_wait"]))
                co2_list.append(float(row["co2"]))

        if not vehicles_list:
            continue

        summary.append({
            "profile":       profile,
            "time_period":   {
                "morning_rush": "8AM - 10AM",
                "evening_rush": "4PM - 7PM",
                "midday":       "12PM - 2PM",
                "night":        "10PM - 12AM"
            }[profile],
            "avg_vehicles":  round(sum(vehicles_list) / len(vehicles_list), 1),
            "peak_vehicles": max(vehicles_list),
            "min_vehicles":  min(vehicles_list),
            "avg_wait_s":    round(sum(wait_list) / len(wait_list), 1),
            "max_wait_s":    round(max(wait_list), 1),
            "avg_co2_mg":    round(sum(co2_list) / len(co2_list), 0),
            "peak_co2_mg":   round(max(co2_list), 0),
            "total_samples": len(vehicles_list),
        })

    #Print Table 
    print("  TRAFFIC PROFILE COMPARISON — EL TAHRIR SQUARE")

    print(f"\n{'Profile':<15} {'Period':<15} {'Avg Veh':>8} "
          f"{'Peak Veh':>9} {'Avg Wait':>9} {'Max Wait':>9} "
          f"{'Avg CO2':>12}")
    print("-"*70)

    for s in summary:
        print(
            f"{s['profile']:<15} "
            f"{s['time_period']:<15} "
            f"{s['avg_vehicles']:>8} "
            f"{s['peak_vehicles']:>9} "
            f"{s['avg_wait_s']:>8}s "
            f"{s['max_wait_s']:>8}s "
            f"{s['avg_co2_mg']:>10.0f}mg"
        )

    print("-"*70)

    #Save Summary CSV 
    summary_path = f"{output_dir}/comparison_summary.csv"
    fields = [
        "profile", "time_period", "avg_vehicles", "peak_vehicles",
        "min_vehicles", "avg_wait_s", "max_wait_s",
        "avg_co2_mg", "peak_co2_mg", "total_samples"
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n Summary saved: {summary_path}")
    print("\nUse this file for:")
    print(" # Report tables")
    print(" # Frontend charts")
    print(" # GitHub documentation")

if __name__ == "__main__":
    compare_profiles()
