# lstm/evaluate.py
# Run this AFTER a simulation to see how accurate predictions were

import os, csv, json, pickle
import numpy as np
import torch
from predict import predict, _load, TrafficLSTM

LSTM_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(LSTM_DIR, "data", "sequences.csv")
MODELS_DIR = os.path.join(LSTM_DIR, "models")

SEQ_LEN  = 60
PRED_LEN = 30
FEATURES = ["north", "south", "east", "west", "total", "avg_speed", "avg_waiting"]  # FIX 1: match train features

# FIX 4: evaluate all profiles instead of only morning_rush
ALL_PROFILES = ["morning_rush", "evening_rush", "midday", "night"]

def evaluate():
    # Load all data grouped by profile
    by_profile = {p: [] for p in ALL_PROFILES}
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profile = row["profile"]
            if profile in by_profile:
                by_profile[profile].append(
                    {k: float(row[k]) if k != "profile" else row[k] for k in row}
                )

    # Load scaler
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print("\n=== LSTM Evaluation — All Profiles ===")
    print(f"{'Profile':<16} {'Direction':<10} {'MAE':>8}")
    print("-" * 38)

    all_maes = []

    for profile in ALL_PROFILES:
        rows = sorted(by_profile[profile], key=lambda x: x["timestep"])
        if len(rows) < SEQ_LEN + PRED_LEN:
            print(f"  [{profile}] Not enough data, skipping.")
            continue

        errors = {"north": [], "south": [], "east": [], "west": []}

        for i in range(len(rows) - SEQ_LEN - PRED_LEN):
            history = rows[i : i + SEQ_LEN]
            actual  = rows[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]

            pred = predict(history)

            for dir in ["north", "south", "east", "west"]:
                pred_vals   = pred[dir]
                actual_vals = [r[dir] for r in actual]
                mae = np.mean(np.abs(np.array(pred_vals) - np.array(actual_vals)))
                errors[dir].append(mae)

        profile_mae = np.mean([np.mean(v) for v in errors.values()])
        all_maes.append(profile_mae)

        for dir, errs in errors.items():
            print(f"  {profile:<14} {dir:<10} {np.mean(errs):>8.2f} vehicles")
        print(f"  {'':14} {'overall':<10} {profile_mae:>8.2f} vehicles avg\n")

    if all_maes:
        overall = np.mean(all_maes)
        print(f"Overall MAE across all profiles: {overall:.2f} vehicles")
        print(f"(e.g. 3.2 means predictions are off by ~3 vehicles on average)")

if __name__ == "__main__":
    evaluate()