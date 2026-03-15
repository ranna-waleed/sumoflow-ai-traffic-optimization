# lstm/prepare_data.py (Enhanced)
# Parses FCD XML/CSV + summary XML → extracts direction counts + waiting times
# Output: lstm/data/sequences.csv

import os
import csv
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS   = os.path.join(BASE_DIR, "simulation", "maps", "outputs")
LSTM_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(LSTM_DATA, exist_ok=True)


#  Direction from angle 
def angle_to_direction(angle: float) -> str:
    angle = float(angle) % 360
    if angle < 45 or angle >= 315:  return "north"
    if 45  <= angle < 135:          return "east"
    if 135 <= angle < 225:          return "south"
    return "west"


#  Parse summary XML : waiting times per timestep 
def parse_summary_xml(filepath: str) -> dict:
    """Returns {timestep: avg_waiting_time}"""
    waiting = {}
    if not os.path.exists(filepath):
        return waiting
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for step in root.findall("step"):
            time = float(step.get("time", 0))
            wt   = float(step.get("meanWaitingTime", 0))
            waiting[time] = wt
    except Exception as e:
        print(f"[prepare] summary parse error: {e}")
    return waiting


#  Parse FCD XML 
def parse_fcd_xml(filepath: str, summary_path: str, profile: str) -> list:
    print(f"[prepare] Parsing {os.path.basename(filepath)}...")
    waiting_map = parse_summary_xml(summary_path)
    rows = []

    tree = ET.parse(filepath)
    root = tree.getroot()

    for ts in root.findall("timestep"):
        time     = float(ts.get("time", 0))
        vehicles = ts.findall("vehicle")
        counts   = {"north": 0, "south": 0, "east": 0, "west": 0}
        speeds   = []

        for v in vehicles:
            angle = v.get("angle")
            speed = v.get("speed")
            if angle is not None:
                counts[angle_to_direction(angle)] += 1
            if speed is not None:
                try: speeds.append(float(speed))
                except: pass

        rows.append({
            "timestep":    time,
            "profile":     profile,
            "north":       counts["north"],
            "south":       counts["south"],
            "east":        counts["east"],
            "west":        counts["west"],
            "total":       sum(counts.values()),
            "avg_speed":   round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
            "avg_waiting": round(waiting_map.get(time, 0.0), 2),
        })

    print(f"[prepare] done {len(rows)} timesteps from {profile}")
    return rows


#  Parse FCD CSV 
def parse_fcd_csv(filepath: str, summary_path: str, profile: str) -> list:
    print(f"[prepare] Parsing {os.path.basename(filepath)}...")
    waiting_map  = parse_summary_xml(summary_path)
    timestep_data = {}

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            time  = row.get("timestep_time", "").strip()
            angle = row.get("vehicle_angle", "").strip()
            speed = row.get("vehicle_speed", "").strip()
            if not time or not angle:
                continue
            try:
                time  = float(time)
                angle = float(angle)
            except ValueError:
                continue
            if time not in timestep_data:
                timestep_data[time] = {"north":0,"south":0,"east":0,"west":0,"speeds":[]}
            timestep_data[time][angle_to_direction(angle)] += 1
            if speed:
                try: timestep_data[time]["speeds"].append(float(speed))
                except: pass

    rows = []
    for time in sorted(timestep_data.keys()):
        d      = timestep_data[time]
        speeds = d["speeds"]
        rows.append({
            "timestep":    time,
            "profile":     profile,
            "north":       d["north"],
            "south":       d["south"],
            "east":        d["east"],
            "west":        d["west"],
            "total":       d["north"] + d["south"] + d["east"] + d["west"],
            "avg_speed":   round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
            "avg_waiting": round(waiting_map.get(time, 0.0), 2),
        })

    print(f"[prepare] done {len(rows)} timesteps from {profile}")
    return rows


def normalize_timesteps(rows):
    if not rows: return rows
    start = rows[0]["timestep"]
    for r in rows:
        r["timestep"] = r["timestep"] - start
    return rows


#  Main 
def prepare_all():
    all_rows = []

    xml_files = {
        "morning_rush": ("fcd_morning.xml",  "summary_morning.xml"),
        "evening_rush": ("fcd_evening.xml",  "summary_evening.xml"),
        "midday":       ("fcd_midday.xml",   "summary_midday.xml"),
        "night":        ("fcd_night.xml",    "summary_night.xml"),
    }

    for profile, (fcd_f, sum_f) in xml_files.items():
        fcd_path = os.path.join(OUTPUTS, fcd_f)
        sum_path = os.path.join(OUTPUTS, sum_f)
        if os.path.exists(fcd_path):
            rows = parse_fcd_xml(fcd_path, sum_path, profile)
            rows = normalize_timesteps(rows)
            all_rows.extend(rows)
        else:
            print(f"[prepare] ⚠ Not found: {fcd_path}")

    # Custom OD CSV
    csv_path = os.path.join(OUTPUTS, "fcd.csv")
    sum_path = os.path.join(OUTPUTS, "summary.xml")
    if os.path.exists(csv_path):
        rows = parse_fcd_csv(csv_path, sum_path, "custom")
        rows = normalize_timesteps(rows)
        all_rows.extend(rows)

    out_path   = os.path.join(LSTM_DATA, "sequences.csv")
    fieldnames = ["timestep","profile","north","south","east","west",
                  "total","avg_speed","avg_waiting"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[prepare] done Saved {len(all_rows)} rows → {out_path}")
    profiles = defaultdict(int)
    for r in all_rows:
        profiles[r["profile"]] += 1
    for p, c in profiles.items():
        print(f"  {p}: {c} timesteps")
    return out_path


if __name__ == "__main__":
    prepare_all()