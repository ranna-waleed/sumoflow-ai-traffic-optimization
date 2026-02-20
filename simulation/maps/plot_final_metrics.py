import pandas as pd
import matplotlib.pyplot as plt

print("Loading final datasets...")

# Load the CSV files
try:
    summary = pd.read_csv(r'outputs/summary.csv', sep=';')
    tripinfo = pd.read_csv(r'outputs/tripinfo.csv', sep=';')
    lanechange = pd.read_csv(r'outputs/lanechange.csv', sep=';')
except FileNotFoundError as e:
    print(f"Error finding file: {e}. Make sure they are in the outputs folder!")
    exit()

# ---------------------------------------------------------
# Plot 3: Network Congestion (Active Vehicles Over Time)
# ---------------------------------------------------------
print("Generating Plot 3: Network Congestion...")
time_col_sum = [c for c in summary.columns if 'time' in c.lower()][0]
running_col = [c for c in summary.columns if 'running' in c.lower()][0]

plt.figure(figsize=(10, 5))
plt.plot(summary[time_col_sum], summary[running_col], color='purple', linewidth=2)
plt.title("Traffic Density: Active Vehicles in El-Tahrir Square", fontsize=14, fontweight='bold')
plt.xlabel("Simulation Time (Seconds)", fontsize=12)
plt.ylabel("Number of Vehicles on Map", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.fill_between(summary[time_col_sum], summary[running_col], color='purple', alpha=0.1)
plt.tight_layout()
plt.savefig(r'outputs/Report_Plot_Congestion.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 4: Lane Changing & Weaving Behavior
# ---------------------------------------------------------
print("Generating Plot 4: Lane Changes...")
time_col_lc = [c for c in lanechange.columns if 'time' in c.lower()][0]

plt.figure(figsize=(10, 5))
plt.hist(lanechange[time_col_lc], bins=40, color='darkorange', edgecolor='black', alpha=0.8)
plt.title("Lane Change Frequency Over Time (Roundabout Weaving)", fontsize=14, fontweight='bold')
plt.xlabel("Simulation Time (Seconds)", fontsize=12)
plt.ylabel("Number of Lane Changes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'outputs/Report_Plot_LaneChanges.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 5: Time Loss vs. Total Travel Time (Delays)
# ---------------------------------------------------------
print("Generating Plot 5: Traffic Delays...")
duration_col = [c for c in tripinfo.columns if 'duration' in c.lower()][0]
loss_col = [c for c in tripinfo.columns if 'timeloss' in c.lower()][0]

plt.figure(figsize=(8, 6))
plt.scatter(tripinfo[duration_col], tripinfo[loss_col], alpha=0.5, color='teal', edgecolors='k')
plt.title("Vehicle Delay: Time Loss vs Total Travel Duration", fontsize=14, fontweight='bold')
plt.xlabel("Total Travel Duration (Seconds)", fontsize=12)
plt.ylabel("Time Loss due to Congestion/Lights (Seconds)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'outputs/Report_Plot_TimeLoss.png', dpi=300)
plt.close()

print("Success! Check your 'outputs' folder for the final 3 PNGs.")