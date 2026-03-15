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
FEATURES = ["north", "south", "east", "west", "total", "avg_speed"]

def evaluate():
    # Load data
    rows = []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["profile"] == "morning_rush":  # test on one profile
                rows.append({k: float(row[k]) if k != "profile" else row[k] for k in row})

    rows.sort(key=lambda x: x["timestep"])

    # Load scaler
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    errors = {"north": [], "south": [], "east": [], "west": []}

    # Slide through sequences
    for i in range(len(rows) - SEQ_LEN - PRED_LEN):
        history = rows[i : i + SEQ_LEN]
        actual  = rows[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]

        pred = predict(history)

        for j, dir in enumerate(["north", "south", "east", "west"]):
            pred_vals   = pred[dir]
            actual_vals = [r[dir] for r in actual]
            mae = np.mean(np.abs(np.array(pred_vals) - np.array(actual_vals)))
            errors[dir].append(mae)

    print("\n=== LSTM Evaluation on morning_rush ===")
    print(f"{'Direction':<10} {'MAE':>8} {'Meaning'}")
    print("-" * 45)
    for dir, errs in errors.items():
        mae = np.mean(errs)
        print(f"{dir:<10} {mae:>8.2f} vehicles avg error")

    overall = np.mean([np.mean(v) for v in errors.values()])
    print(f"\nOverall MAE: {overall:.2f} vehicles")
    print(f"(e.g. 3.2 means predictions are off by ~3 vehicles on average)")

if __name__ == "__main__":
    evaluate()