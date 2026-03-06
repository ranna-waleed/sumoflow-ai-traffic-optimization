import xml.etree.ElementTree as ET
import os
import random

# ─── Hourly Traffic Demand (vehicles/hour) ────────────
# Based on typical Cairo traffic patterns
HOURLY_DEMAND = {
    0:  200,
    1:  150,
    2:  100,
    3:  100,
    4:  200,
    5:  500,
    6:  1500,
    7:  3000,
    8:  5000,   # MORNING RUSH PEAK → 150+ vehicles
    9:  4000,
    10: 2500,
    11: 2000,
    12: 2500,   # MIDDAY
    13: 2000,
    14: 2000,
    15: 2500,
    16: 3500,
    17: 5500,   # EVENING RUSH PEAK → 200+ vehicles
    18: 4500,
    19: 3000,
    20: 2000,
    21: 1500,
    22: 800,
    23: 400,
}


# ─── Profile Definitions ──────────────────────────────
PROFILES = {
    "morning_rush": {
        "start_hour": 8,
        "end_hour":   10,
        "output": "simulation/maps/routes_morning_rush.rou.xml"
    },
    "evening_rush": {
        "start_hour": 16,
        "end_hour":   19,
        "output": "simulation/maps/routes_evening_rush.rou.xml"
    },
    "midday": {
        "start_hour": 12,
        "end_hour":   14,
        "output": "simulation/maps/routes_midday.rou.xml"
    },
    "night": {
        "start_hour": 22,
        "end_hour":   24,
        "output": "simulation/maps/routes_night.rou.xml"
    },
    "realtime": {
        "start_hour": 0,
        "end_hour":   24,
        "output": "simulation/maps/routes_realtime.rou.xml"
    }
}

# ─── Vehicle Types ────────────────────────────────────
VEHICLE_TYPES = [
    {"id": "passenger",  "ratio": 0.45,
     "maxSpeed": "13.89", "length": "4.5",
     "color": "255,255,255"},   # white  → private cars
    {"id": "taxi",       "ratio": 0.13,
     "maxSpeed": "13.89", "length": "4.5",
     "color": "255,200,0"},     # yellow → taxis
    {"id": "microbus",   "ratio": 0.13,
     "maxSpeed": "11.11", "length": "6.0",
     "color": "0,0,255"},       # blue   → microbuses
    {"id": "bus",        "ratio": 0.08,
     "maxSpeed": "11.11", "length": "12.0",
     "color": "255,0,0"},       # red    → buses
    {"id": "truck",      "ratio": 0.07,
     "maxSpeed": "11.11", "length": "8.0",
     "color": "128,128,128"},   # gray   → trucks
    {"id": "motorcycle", "ratio": 0.03,
     "maxSpeed": "16.67", "length": "2.0",
     "color": "255,165,0"},     # orange → motorcycles
    {"id": "bicycle",    "ratio": 0.02,
     "maxSpeed": "5.56",  "length": "1.8",
     "color": "0,200,0"},       # green  → bicycles
]

def get_existing_routes():
    """Read real edge IDs from existing route file"""
    input_file = "simulation/maps/tahrir_fixed.rou.xml"
    print(f"Reading routes from: {input_file}")
    tree     = ET.parse(input_file)
    root     = tree.getroot()
    vehicles = root.findall("vehicle")

    routes = []
    for v in vehicles:
        route_elem = v.find("route")
        if route_elem is not None:
            edges = route_elem.get("edges")
            if edges and edges not in routes:
                routes.append(edges)

    print(f"Found {len(routes)} unique routes")
    return routes

def generate_profile(profile_name):
    """Generate realistic route file for one time profile"""

    profile    = PROFILES[profile_name]
    start_hour = profile["start_hour"]
    end_hour   = profile["end_hour"]
    output     = profile["output"]

    print(f"\n{'─'*40}")
    print(f"Generating: {profile_name}")
    print(f"Time: {start_hour:02d}:00 → {end_hour:02d}:00")

    existing_routes = get_existing_routes()
    num_routes      = min(20, len(existing_routes))

    new_root = ET.Element("routes")

    # ─── Vehicle Type Definitions ─────────────────────
    for vtype in VEHICLE_TYPES:
        ET.SubElement(new_root, "vType", {
            "id":       vtype["id"],
            "maxSpeed": vtype["maxSpeed"],
            "length":   vtype["length"],
            "accel":    "2.6",
            "decel":    "4.5",
            "sigma":    "0.5",   # driver imperfection
            "minGap":   "2.5",   # gap between vehicles
            "color":    vtype["color"]
        })

    # ─── Route Definitions ────────────────────────────
    for i, edges in enumerate(existing_routes[:num_routes]):
        ET.SubElement(new_root, "route", {
            "id":    f"route_{i}",
            "edges": edges
        })

    # ─── Generate Realistic Hourly Flows ──────────────
    flow_id = 0

    for hour in range(start_hour, end_hour):
        demand    = HOURLY_DEMAND[hour % 24]
        begin_sec = hour * 3600
        end_sec   = (hour + 1) * 3600

        # Split demand into 10-minute intervals
        # for more realistic gradual buildup
        interval  = 600   # 10 minutes in seconds
        intervals = 6     # 6 x 10min = 1 hour

        for interval_num in range(intervals):
            iv_begin = begin_sec + (interval_num * interval)
            iv_end   = iv_begin + interval

            # Gradually increase during rush hour
            if profile_name in ["morning_rush", "evening_rush"]:
                # Build up traffic gradually
                ramp = 0.7 + (0.3 * interval_num / intervals)
            else:
                ramp = 1.0

            interval_demand = int((demand / intervals) * ramp)

            print(
                f"  {hour:02d}:{interval_num*10:02d} → "
                f"{interval_demand} vehicles/10min"
            )

            for vtype in VEHICLE_TYPES:
                count = int(interval_demand * vtype["ratio"])
                if count == 0:
                    continue

                # More vehicles per route = more congestion
                per_route = max(5, count // 8)

                for r in range(min(10, num_routes)):
                    ET.SubElement(new_root, "flow", {
                        "id":          f"flow_{flow_id}",
                        "type":        vtype["id"],
                        "route":       f"route_{r}",
                        "begin":       str(iv_begin),
                        "end":         str(iv_end),
                        "number":      str(per_route),
                        "departLane":  "best",
                        "departSpeed": "random",  # realistic speeds
                    })
                    flow_id += 1

    # ─── Save ─────────────────────────────────────────
    tree_out = ET.ElementTree(new_root)
    ET.indent(tree_out, space="    ")
    tree_out.write(
        output,
        encoding="unicode",
        xml_declaration=True
    )

    print(f"\n  Total flows: {flow_id}")
    print(f"  Saved: {output} ")

def generate_all():
    """Generate route files for all profiles"""
    print("  GENERATING RUSH HOUR ROUTES")

    for profile in PROFILES.keys():
        generate_profile(profile)

    print("  ALL PROFILES DONE ")
    print("\nNext step:")
    print("python simulation/rush_hour/run_realtime.py")

if __name__ == "__main__":
    generate_all()