import pandas as pd
import matplotlib.pyplot as plt
import os

print("Loading massive simulation data... ")

# SUMO's xml2csv uses ';' as a delimiter
fcd = pd.read_csv(r'outputs/fcd.csv', sep=';')
em = pd.read_csv(r'outputs/emission.csv', sep=';')

# Dynamically find columns to prevent errors
time_col = [c for c in em.columns if 'time' in c.lower()][0]
co2_col = [c for c in em.columns if 'co2' in c.lower()][0]
speed_col = [c for c in fcd.columns if 'speed' in c.lower()][0]
time_col_fcd = [c for c in fcd.columns if 'time' in c.lower()][0]

# ---------------------------------------------------------
# Plot 1: Total CO2 Emissions Over Time
# ---------------------------------------------------------
print("Generating Report Plot 1: CO2 Emissions...")
# Group by time and SUM the CO2 of all vehicles on the map
co2_trend = em.groupby(time_col)[co2_col].sum()

plt.figure(figsize=(10, 5))
plt.plot(co2_trend.index, co2_trend.values, color='#d62728', linewidth=2)
plt.title("Total CO2 Emissions Over Time (El-Tahrir Square)", fontsize=14, fontweight='bold')
plt.xlabel("Simulation Time (Seconds)", fontsize=12)
plt.ylabel("Total CO2 (mg/s)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.fill_between(co2_trend.index, co2_trend.values, color='#d62728', alpha=0.1)
plt.tight_layout()
plt.savefig(r'outputs/Report_Plot_CO2.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Average Network Speed Over Time
# ---------------------------------------------------------
print("Generating Report Plot 2: Average Speed...")
# Group by time, average the speeds, and convert m/s to km/h
speed_trend = fcd.groupby(time_col_fcd)[speed_col].mean() * 3.6

plt.figure(figsize=(10, 5))
plt.plot(speed_trend.index, speed_trend.values, color='#1f77b4', linewidth=2)
plt.title("Average Traffic Speed Over Time (El-Tahrir Square)", fontsize=14, fontweight='bold')
plt.xlabel("Simulation Time (Seconds)", fontsize=12)
plt.ylabel("Average Speed (km/h)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.fill_between(speed_trend.index, speed_trend.values, color='#1f77b4', alpha=0.1)
plt.tight_layout()
plt.savefig(r'outputs/Report_Plot_Average_Speed.png', dpi=300)
plt.close()

print("Success! Check your 'outputs' folder for the high-resolution PNGs.")