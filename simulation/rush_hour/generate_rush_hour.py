import xml.etree.ElementTree as ET
import random
import os

# Hourly Traffic Demand (vehicles/hour)
HOURLY_DEMAND = {
    0:  200, 1:  150, 2:  100, 3:  100, 4:  200,
    5:  600, 6:  1800, 7:  3500, 8:  5500, 9:  4500,
    10: 3000, 11: 2500, 12: 3000, 13: 2500, 14: 2200,
    15: 3000, 16: 4000, 17: 6000, 18: 5500, 19: 4000,
    20: 2500, 21: 1800, 22: 900,  23: 450,
}

# Profile Definitions
PROFILES = {
    "morning_rush": {
        "start_hour": 7,
        "end_hour":   10,
        "output": "simulation/maps/routes_morning_rush.rou.xml"
    },
    "evening_rush": {
        "start_hour": 15,
        "end_hour":   20,
        "output": "simulation/maps/routes_evening_rush.rou.xml"
    },
    "midday": {
        "start_hour": 12,
        "end_hour":   15,
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

# Vehicle Composition per Profile 
VEHICLE_COMPOSITION = {
    "morning_rush": {
        "passenger":   0.57, "microbus": 0.20, "taxi":       0.12,
        "bus":         0.04, "truck":    0.02, "motorcycle": 0.04,
        "bicycle":     0.01,
    },
    "evening_rush": {
        "passenger":   0.55, "microbus": 0.22, "taxi":       0.12,
        "bus":         0.04, "truck":    0.01, "motorcycle": 0.05,
        "bicycle":     0.01,
    },
    "midday": {
        "passenger":   0.58, "microbus": 0.18, "taxi":       0.13,
        "bus":         0.05, "truck":    0.03, "motorcycle": 0.03,
        "bicycle":     0.01,
    },
    "night": {
        "passenger":   0.65, "microbus": 0.12, "taxi":       0.18,
        "bus":         0.02, "truck":    0.01, "motorcycle": 0.02,
        "bicycle":     0.00,
    },
    "realtime": {
        "passenger":   0.57, "microbus": 0.20, "taxi":       0.12,
        "bus":         0.04, "truck":    0.02, "motorcycle": 0.04,
        "bicycle":     0.01,
    },
}

# Vehicle Type Specs
VEHICLE_SPECS = {
    "passenger":  {"maxSpeed": "13.89", "length": "4.5",  "color": "255,255,255"},
    "taxi":       {"maxSpeed": "13.89", "length": "4.5",  "color": "255,200,0"},
    "microbus":   {"maxSpeed": "11.11", "length": "6.0",  "color": "0,100,255"},
    "bus":        {"maxSpeed": "11.11", "length": "12.0", "color": "255,0,0"},
    "truck":      {"maxSpeed": "11.11", "length": "8.0",  "color": "128,128,128"},
    "motorcycle": {"maxSpeed": "16.67", "length": "2.0",  "color": "255,165,0"},
    "bicycle":    {"maxSpeed": "5.56",  "length": "1.8",  "color": "0,200,0"},
}

#  Bus Stop Definitions
# Maps bus stop ID to the edge it's on (extracted from lane ID)
BUS_STOPS = {
    "bs_1":  {"lane": "50211834#1_0",   "edge": "50211834#1",   "pos": 4.5},
    "bs_2":  {"lane": "192591565#1_0",  "edge": "192591565#1",  "pos": 5.5},
    "bs_3":  {"lane": "28718647#3_0",   "edge": "28718647#3",   "pos": 5.5},
    "bs_4":  {"lane": "28718647#3_4",   "edge": "28718647#3",   "pos": 1.2},
    "bs_5":  {"lane": "28718647#2_0",   "edge": "28718647#2",   "pos": 1.5},
    "bs_6":  {"lane": "28718647#2_4",   "edge": "28718647#2",   "pos": 1.2},
    "bs_7":  {"lane": "28718647#1_0",   "edge": "28718647#1",   "pos": 12.5},
    "bs_8":  {"lane": "28718647#1_4",   "edge": "28718647#1",   "pos": 12.5},
    "bs_9":  {"lane": "28718874#1_0",   "edge": "28718874#1",   "pos": 4.5},
    "bs_10": {"lane": "28718874#1_4",   "edge": "28718874#1",   "pos": 4.5},
    "bs_11": {"lane": "10873193#1_0",   "edge": "10873193#1",   "pos": 4.5},
}

#  Bus Stop Lane Type (for microbus stop filtering) 
# Edges where microbuses CAN stop (primary + secondary only)
MICROBUS_STOPPABLE_TYPES = ["highway.primary", "highway.secondary"]

# Known primary/secondary edges around Tahrir Square
# (trunk roads excluded: too fast and dangerous for random stops)
PRIMARY_EDGES = [
    "50211834#0", "50211834#1",
    "192591565#0", "192591565#1",
    "28718874#0",  "28718874#1",
    "10873193#0",  "10873193#1",
    "29181613#0",  "29181613#1",
]


def get_existing_routes():
    """Read real edge IDs from existing route file."""
    input_file = "simulation/maps/tahrir_fixed.rou.xml"
    print(f"Reading routes from: {input_file}")
    tree = ET.parse(input_file)
    root = tree.getroot()

    routes = []
    for v in root.findall("vehicle"):
        route_elem = v.find("route")
        if route_elem is not None:
            edges = route_elem.get("edges")
            if edges and edges not in routes:
                routes.append(edges)

    print(f"Found {len(routes)} unique routes")
    return routes


def route_passes_bus_stop(edges_str):
    """
    Check which bus stops a route passes through.
    Returns list of (stop_id, stop_data) tuples.
    """
    edges = edges_str.split()
    passed_stops = []
    seen_edges = set()

    for stop_id, stop_data in BUS_STOPS.items():
        edge = stop_data["edge"]
        if edge in edges and edge not in seen_edges:
            passed_stops.append((stop_id, stop_data))
            seen_edges.add(edge)

    return passed_stops


def find_route_with_bus_stops(routes, min_stops=1):
    """
    Find routes that pass through at least min_stops bus stops.
    Option B: If none found, extend search to find best route.
    """
    best_route = None
    best_stop_count = 0

    for route in routes:
        stops = route_passes_bus_stop(route)
        if len(stops) >= min_stops:
            return route, stops
        if len(stops) > best_stop_count:
            best_stop_count = len(stops)
            best_route = (route, stops)

    # Option B: Return best available even if below min_stops
    if best_route:
        return best_route
    return routes[0], []


def get_microbus_stop_edges(edges_str):
    """
    Get edges in route where microbus can randomly stop.
    Only primary/secondary roads — no trunk roads.
    """
    edges = edges_str.split()
    stoppable = [e for e in edges if e in PRIMARY_EDGES]
    return stoppable


def generate_bus_vehicles(new_root, routes, composition_ratio,
                          interval_demand, iv_begin, iv_end,
                          flow_id_counter):
    """
    Generate individual bus vehicles with stops at bus stops.
    Returns updated flow_id_counter.
    """
    if composition_ratio == 0.0:
        return flow_id_counter

    bus_count = int(interval_demand * composition_ratio)
    if bus_count == 0:
        return flow_id_counter

    # Find routes that pass through bus stops
    bus_routes_with_stops = []
    for route_edges in routes[:20]:
        stops = route_passes_bus_stop(route_edges)
        if stops:
            bus_routes_with_stops.append((route_edges, stops))

    # Option B: if no routes pass bus stops, use best available
    if not bus_routes_with_stops:
        best_route, best_stops = find_route_with_bus_stops(routes)
        bus_routes_with_stops = [(best_route, best_stops)]

    # Generate individual bus vehicles spread across time interval
    interval_duration = iv_end - iv_begin
    time_gap = max(1, interval_duration // max(bus_count, 1))

    for i in range(bus_count):
        depart_time = iv_begin + (i * time_gap) + random.randint(0, max(1, time_gap - 1))
        depart_time = min(depart_time, iv_end - 1)

        # Pick a route with bus stops
        route_edges, stops = random.choice(bus_routes_with_stops)

        # Build vehicle element
        vehicle = ET.SubElement(new_root, "vehicle", {
            "id":          f"bus_{flow_id_counter}",
            "type":        "bus",
            "depart":      str(depart_time),
            "departLane":  "best",
            "departSpeed": "random",
        })

        ET.SubElement(vehicle, "route", {"edges": route_edges})

        # Add stops at bus stop positions along route
        for stop_id, stop_data in stops:
            duration = random.randint(20, 45)  # 20-45 seconds per stop
            ET.SubElement(vehicle, "stop", {
                "busStop":  stop_id,
                "duration": str(duration),
            })

        flow_id_counter += 1

    return flow_id_counter


def generate_microbus_vehicles(new_root, routes, composition_ratio,
                               interval_demand, iv_begin, iv_end,
                               flow_id_counter):
    """
    Generate individual microbus vehicles with random stops.
    Stops on primary/secondary roads only — 2-4 stops per trip,
    10-60 seconds each.
    Returns updated flow_id_counter.
    """
    if composition_ratio == 0.0:
        return flow_id_counter

    microbus_count = int(interval_demand * composition_ratio)
    if microbus_count == 0:
        return flow_id_counter

    interval_duration = iv_end - iv_begin
    time_gap = max(1, interval_duration // max(microbus_count, 1))

    for i in range(microbus_count):
        depart_time = iv_begin + (i * time_gap) + random.randint(0, max(1, time_gap - 1))
        depart_time = min(depart_time, iv_end - 1)

        # Pick route from top 20
        route_edges = random.choice(routes[:20])
        stoppable_edges = get_microbus_stop_edges(route_edges)

        vehicle = ET.SubElement(new_root, "vehicle", {
            "id":          f"microbus_{flow_id_counter}",
            "type":        "microbus",
            "depart":      str(depart_time),
            "departLane":  "best",
            "departSpeed": "random",
        })

        ET.SubElement(vehicle, "route", {"edges": route_edges})

        # Add random stops on primary/secondary edges
        if stoppable_edges:
            num_stops  = random.randint(2, 4)
            # Don't stop on same edge twice
            chosen = random.sample(
                stoppable_edges,
                min(num_stops, len(stoppable_edges))
            )
            # Sort by position in route to stop in order
            edge_list = route_edges.split()
            chosen.sort(key=lambda e: edge_list.index(e)
                        if e in edge_list else 999)

            for edge in chosen:
                duration   = random.randint(10, 60)  # 10-60 seconds
                # Random position on the edge (5-15m from start)
                stop_pos   = random.uniform(5.0, 15.0)
                lane_id    = f"{edge}_0"

                ET.SubElement(vehicle, "stop", {
                    "lane":       lane_id,
                    "startPos":   str(round(stop_pos, 1)),
                    "endPos":     str(round(stop_pos + 10.0, 1)),
                    "duration":   str(duration),
                    "parking":    "false",  # stays on road — blocks traffic
                })

        flow_id_counter += 1

    return flow_id_counter


def generate_profile(profile_name):
    """Generate realistic route file for one time profile."""
    profile     = PROFILES[profile_name]
    start_hour  = profile["start_hour"]
    end_hour    = profile["end_hour"]
    output      = profile["output"]
    composition = VEHICLE_COMPOSITION[profile_name]

    print(f"\nGenerating: {profile_name}")
    print(f"Time: {start_hour:02d}:00 → {end_hour:02d}:00")

    existing_routes = get_existing_routes()
    num_routes      = min(20, len(existing_routes))
    new_root        = ET.Element("routes")

    # Vehicle Type Definitions 
    for vtype_id, specs in VEHICLE_SPECS.items():
        ET.SubElement(new_root, "vType", {
            "id":       vtype_id,
            "maxSpeed": specs["maxSpeed"],
            "length":   specs["length"],
            "accel":    "2.6",
            "decel":    "4.5",
            "sigma":    "0.5",
            "minGap":   "2.5",
            "color":    specs["color"],
        })

    # Route Definitions
    for i, edges in enumerate(existing_routes[:num_routes]):
        ET.SubElement(new_root, "route", {
            "id":    f"route_{i}",
            "edges": edges
        })

    # Generate Flows + Individual Vehicles 
    flow_id_counter = 0

    for hour in range(start_hour, end_hour):
        demand        = HOURLY_DEMAND[hour % 24]
        begin_sec     = hour * 3600
        interval      = 600    # 10 minutes
        num_intervals = 6

        for interval_num in range(num_intervals):
            iv_begin = begin_sec + (interval_num * interval)
            iv_end   = iv_begin + interval

            # Ramp up during rush hours
            if profile_name in ["morning_rush", "evening_rush"]:
                ramp = 0.7 + (0.3 * interval_num / num_intervals)
            else:
                ramp = 1.0

            interval_demand = int((demand / num_intervals) * ramp)

            print(f"  {hour:02d}:{interval_num*10:02d} → "
                  f"{interval_demand} vehicles/10min")

            for vtype_id, ratio in composition.items():
                if ratio == 0.0:
                    continue

                # ── Bus: individual vehicles with bus stop stops ──
                if vtype_id == "bus":
                    flow_id_counter = generate_bus_vehicles(
                        new_root, existing_routes, ratio,
                        interval_demand, iv_begin, iv_end,
                        flow_id_counter
                    )

                # Microbus: individual vehicles with random stops
                elif vtype_id == "microbus":
                    flow_id_counter = generate_microbus_vehicles(
                        new_root, existing_routes, ratio,
                        interval_demand, iv_begin, iv_end,
                        flow_id_counter
                    )

                # All other types: flow elements (unchanged)
                else:
                    count = int(interval_demand * ratio)
                    if count == 0:
                        continue

                    per_route = max(2, count // 20)

                    for r in range(min(20, num_routes)):
                        ET.SubElement(new_root, "flow", {
                            "id":          f"flow_{flow_id_counter}",
                            "type":        vtype_id,
                            "route":       f"route_{r}",
                            "begin":       str(iv_begin),
                            "end":         str(iv_end),
                            "number":      str(per_route),
                            "departLane":  "best",
                            "departSpeed": "random",
                        })
                        flow_id_counter += 1

    #  Save
    tree_out = ET.ElementTree(new_root)
    ET.indent(tree_out, space="    ")
    tree_out.write(output, encoding="unicode", xml_declaration=True)

    print(f"\n  Total elements: {flow_id_counter}")
    print(f"  Saved: {output}")


def generate_all():
    """Generate route files for all profiles."""
    print("  GENERATING RUSH HOUR ROUTES — EGYPT REALISTIC")
    print("  Buses stop at bus stops (20-45s)")
    print("  Microbuses stop randomly (10-60s, 2-4 times)")

    for profile in PROFILES.keys():
        generate_profile(profile)

    print("  ALL PROFILES DONE ")
    print("\nNext steps:")
    print("  1. python simulation/rush_hour/run_realtime.py --profile morning_rush")
    print("  2. python simulation/rush_hour/run_realtime.py --profile evening_rush")
    print("  3. python simulation/rush_hour/run_realtime.py --profile midday")
    print("  4. python simulation/rush_hour/run_realtime.py --profile night")
    print("  5. python lstm/prepare_data.py")
    print("  6. python lstm/train_lstm.py")
    print("  7. python dqn/run_baseline.py")
    print("  8. python dqn/train_dqn.py")
    print("  9. python dqn/run_dqn.py")
    print(" 10. python dqn/compare.py")


if __name__ == "__main__":
    generate_all()