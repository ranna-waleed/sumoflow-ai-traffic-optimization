import traci
import os

# ─── Paths ───────────────────────────────────────────
SUMO_CFG    = "simulation/maps/config_file.sumocfg"
OUTPUT_DIR  = "detection/dataset/images/raw"
TOTAL_STEPS = 3600
CAPTURE_EVERY = 10

# ─── Setup ───────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Start SUMO ──────────────────────────────────────
traci.start([
    "sumo-gui",
    "-c", SUMO_CFG,
    "--delay", "0",
    "--quit-on-end", "false",   # CHANGED → don't quit when no vehicles
    "--end", "3600",             # ADDED   → force run until 3600
    "--no-step-log", "true",    # ADDED   → suppress step warnings
    "--window-size", "1920,1080"  # higher resolution screenshots
])

step        = 0
frame_count = 0

print("Starting frame extraction...")
print(f"   Total steps : {TOTAL_STEPS}")
print(f"   Capture every : {CAPTURE_EVERY} steps")
print(f"   Expected frames : ~{TOTAL_STEPS // CAPTURE_EVERY}\n")

while step < TOTAL_STEPS:
    traci.simulationStep()

    if step % CAPTURE_EVERY == 0:
        filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.png")
        traci.gui.screenshot("View #0", os.path.abspath(filename))
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"   Captured {frame_count} frames so far...")

    step += 1

traci.close()
print(f"\nDone! {frame_count} frames saved in {OUTPUT_DIR}")