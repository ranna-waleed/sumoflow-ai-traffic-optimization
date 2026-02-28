import traci
import os
import time

# ─── Paths ───────────────────────────────────────────
SUMO_CFG    = "simulation/maps/config_file.sumocfg"
OUTPUT_DIR  = "detection/dataset/images/raw"
TOTAL_STEPS = 3600
CAPTURE_EVERY = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── NEON COLOR MAPPING (R, G, B, Alpha) ─────────────
NEON_COLORS = {
    "car": (255, 0, 255, 255),        # Magenta
    "bus": (0, 255, 255, 255),        # Cyan
    "taxi": (150, 0, 255, 255),       # Electric Purple
    "microbus": (255, 100, 200, 255), # Hot Pink
    "bicycle": (0, 150, 255, 255),    # Bright Blue
    "truck": (0, 255, 150, 255),      # Mint Green
    "motorcycle": (255, 150, 0, 255)  # Neon Orange
}

# ─── Start SUMO ──────────────────────────────────────
traci.start([
    "sumo-gui",
    "-c", SUMO_CFG,
    "--delay", "10",
    "--quit-on-end", "false",
    "--end", str(TOTAL_STEPS),
    "--no-step-log", "true",
    "--window-size", "1920,1080"
])

step = 0
frame_count = 0

print("Ready! Adjust your zoom now, then click Play in SUMO.")

while step < TOTAL_STEPS:
    traci.simulationStep()

    # ─── THE PAINT GUN ────────────────────────────────
    # Find all vehicles that just spawned in this exact step
    for veh_id in traci.simulation.getDepartedIDList():
        vtype = traci.vehicle.getTypeID(veh_id)
        
        # Apply the neon color if it matches our dictionary
        if vtype in NEON_COLORS:
            traci.vehicle.setColor(veh_id, NEON_COLORS[vtype])
            
    # ─── CAPTURE LOGIC ────────────────────────────────
    if step % CAPTURE_EVERY == 0:
        time.sleep(0.05) # Tiny pause for screen buffer
        
        filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.png")
        traci.gui.screenshot("View #0", os.path.abspath(filename))
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"   Captured {frame_count} frames so far...")

    step += 1

traci.close()
print(f"\nDone! {frame_count} frames saved in {OUTPUT_DIR}")