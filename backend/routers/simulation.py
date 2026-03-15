# backend/routers/simulation.py
from fastapi import APIRouter, HTTPException
import csv
import os
import glob

router = APIRouter()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "simulation", "maps", "outputs")

PROFILE_LABELS = {
    "morning_rush": "8AM - 10AM",
    "evening_rush": "4PM - 7PM",
    "midday":       "12PM - 2PM",
    "night":        "10PM - 12AM",
}

#  GET /api/simulation/profiles 
@router.get("/profiles")
def get_profiles():
    return {
        "profiles": [
            {"id": "morning_rush", "label": "Morning Rush", "period": "8AM - 10AM"},
            {"id": "evening_rush", "label": "Evening Rush", "period": "4PM - 7PM"},
            {"id": "midday",       "label": "Midday",       "period": "12PM - 2PM"},
            {"id": "night",        "label": "Night",        "period": "10PM - 12AM"},
        ]
    }

#  GET /api/simulation/summary 
@router.get("/summary")
def get_summary():
    summary_path = os.path.join(OUTPUTS_DIR, "comparison_summary.csv")

    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="comparison_summary.csv not found")

    profiles = []
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles.append({
                "profile":       row["profile"],
                "time_period":   row["time_period"],
                "avg_vehicles":  float(row["avg_vehicles"]),
                "peak_vehicles": int(row["peak_vehicles"]),
                "min_vehicles":  int(row["min_vehicles"]),
                "avg_wait_s":    float(row["avg_wait_s"]),
                "max_wait_s":    float(row["max_wait_s"]),
                "avg_co2_mg":    float(row["avg_co2_mg"]),
                "peak_co2_mg":   float(row["peak_co2_mg"]),
                "total_samples": int(row["total_samples"]),
            })

    return {
        "location": "El-Tahrir Square, Cairo, Egypt",
        "network": {
            "total_edges":   75,
            "active_edges":  48,
            "usage_rate":    "64%",
            "unique_routes": 164,
        },
        "profiles": profiles
    }

#  GET /api/simulation/metrics/{profile} 
@router.get("/metrics/{profile}")
def get_metrics(profile: str):
    if profile not in PROFILE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile. Choose from: {list(PROFILE_LABELS.keys())}"
        )

    pattern = os.path.join(OUTPUTS_DIR, f"metrics_{profile}_*.csv")
    files   = glob.glob(pattern)

    if not files:
        raise HTTPException(status_code=404, detail=f"No CSV found for: {profile}")

    latest = max(files)
    rows   = []

    with open(latest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step":     int(row["step"]),
                "vehicles": int(row["vehicles"]),
                "avg_wait": float(row["avg_wait"]),
                "co2":      float(row["co2"]),
            })

    if not rows:
        raise HTTPException(status_code=404, detail="CSV is empty")

    vehicles = [r["vehicles"] for r in rows]
    waits    = [r["avg_wait"] for r in rows]
    co2s     = [r["co2"]      for r in rows]

    return {
        "profile":       profile,
        "time_period":   PROFILE_LABELS[profile],
        "total_steps":   len(rows),
        "peak_vehicles": max(vehicles),
        "avg_vehicles":  round(sum(vehicles) / len(vehicles), 1),
        "avg_wait_s":    round(sum(waits) / len(waits), 1),
        "max_wait_s":    round(max(waits), 1),
        "peak_co2_mg":   round(max(co2s), 0),
        "avg_co2_mg":    round(sum(co2s) / len(co2s), 0),
        "timeseries":    rows,
    }