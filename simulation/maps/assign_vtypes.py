import xml.etree.ElementTree as ET
import random

# Cairo El-Tahrir realistic traffic composition
VEHICLE_MIX = {
    "car":        0.50,   # 50% private cars
    "taxi":       0.13,   # 13% taxis
    "microbus":   0.13,   # 13% microbuses
    "bus":        0.06,   # 6%  buses
    "truck":      0.05,   # 5%  trucks
    "motorcycle": 0.10,   # 10% motorcycles (very common in Cairo)
    "bicycle":    0.03,   # 3%  bicycles
}

# TAZ sources on TRUNK roads → disallow bicycle (trunk disallows bicycle in net.xml)
TRUNK_TAZ_SOURCES = {"taz_south_in", "taz_east_in"}

# Buses only on these TAZ pairs (routes that pass near bus stops)
BUS_ALLOWED_TAZ_PAIRS = [
    ("taz_north_in", "taz_west_out"),
    ("taz_east_in",  "taz_west_out"),
    ("taz_south_in", "taz_north_out"),
    ("taz_west_in",  "taz_east_out"),
]

INPUT_FILE  = "od_tahrir.odtrips.xml"
OUTPUT_FILE = "od_tahrir_typed.odtrips.xml"

random.seed(42)

def normalize(mix):
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}

def pick_vtype(from_taz, to_taz):
    mix = dict(VEHICLE_MIX)

    # Remove bicycle & truck from trunk road TAZs
    if from_taz in TRUNK_TAZ_SOURCES:
        mix.pop("bicycle", None)

    # Remove bus from non-bus routes
    if (from_taz, to_taz) not in BUS_ALLOWED_TAZ_PAIRS:
        mix.pop("bus", None)

    mix = normalize(mix)

    r = random.random()
    cumulative = 0.0
    for vtype, prob in mix.items():
        cumulative += prob
        if r <= cumulative:
            return vtype
    return "car"  # fallback

tree = ET.parse(INPUT_FILE)
root = tree.getroot()

type_counts = {}

for trip in root.findall("trip"):
    from_taz = trip.get("fromTaz", "")
    to_taz   = trip.get("toTaz", "")
    vtype    = pick_vtype(from_taz, to_taz)
    trip.set("type", vtype)
    type_counts[vtype] = type_counts.get(vtype, 0) + 1

tree.write(OUTPUT_FILE, encoding="unicode", xml_declaration=True)

print("Vehicle types assigned successfully!")
print(f" Output: {OUTPUT_FILE}")
print("\n Traffic Composition:")
total = sum(type_counts.values())
for vtype, count in sorted(type_counts.items()):
    print(f" {vtype:<12}: {count:>4} vehicles  ({count/total*100:.1f}%)")