import React from "react";
import { Compass } from "lucide-react";

// FIXED: proper direction parsing from signal string
// SUMO signal strings like "GGGGrrrr" — G=green, y=yellow, r=red
function parsePhase(signalStr) {
  if (!signalStr) return "red";
  const s = signalStr.toLowerCase();
  if (s.includes("y")) return "yellow";
  if (s.includes("g")) return "green";
  return "red";
}

// FIXED: map TL IDs to directions based on known Tahrir Square TL positions
// Instead of blindly mapping first 4 IDs, we use the known IDs from the project
const TL_DIRECTION_MAP = {
  "315744796":  "North",   // Main TL — north approach
  "2031414903": "South",   // South approach
  "96621100":   "East",    // East approach
  "2031414899": "West",    // West approach
  // Fallback for any other TLs
  "6288771431": "North-East",
  "96621068":   "South-East",
  "271064234":  "South-West",
  "315743335":  "North-West",
  "6288771435": "Centre",
};

// Static fallback when simulation not running
const STATIC = [
  { id: "north", label: "North",     phase: "red", signal: "——" },
  { id: "south", label: "South",     phase: "red", signal: "——" },
  { id: "east",  label: "East",      phase: "red", signal: "——" },
  { id: "west",  label: "West",      phase: "red", signal: "——" },
];

// Show only the 4 main directions in the UI
const MAIN_DIRECTIONS = ["North", "South", "East", "West"];

function TrafficLightStatus({ trafficLights, typeCounts }) {
  // FIXED: removed console.log that was left in production

  let rows;
  if (trafficLights && Object.keys(trafficLights).length > 0) {
    // Build direction → signal map using known TL IDs
    const directionMap = {};
    for (const [tlId, signal] of Object.entries(trafficLights)) {
      const direction = TL_DIRECTION_MAP[tlId];
      if (direction && MAIN_DIRECTIONS.includes(direction)) {
        directionMap[direction] = signal;
      }
    }

    // Build rows for the 4 cardinal directions
    rows = MAIN_DIRECTIONS.map(label => {
      const signal = directionMap[label] || "rrrr";
      return {
        id:     label.toLowerCase(),
        label,
        phase:  parsePhase(signal),
        signal,
      };
    });
  } else {
    rows = STATIC;
  }

  const totalVehicles = typeCounts
    ? Object.values(typeCounts).reduce((a, b) => a + b, 0)
    : null;

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Compass className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-white">Traffic Light Status</h3>
        </div>
        {totalVehicles !== null && (
          <span className="text-[11px] font-mono text-[#64748b]">
            {totalVehicles} total
          </span>
        )}
      </div>

      <div className="space-y-3">
        {rows.map(row => {
          const isGreen  = row.phase === "green";
          const isYellow = row.phase === "yellow";
          const isRed    = row.phase === "red";
          return (
            <div
              key={row.label}
              className={`flex items-center justify-between p-3 rounded-xl transition-colors ${
                isGreen  ? "bg-emerald-500/10 border border-emerald-500/25" :
                isYellow ? "bg-amber-500/10   border border-amber-500/25"  :
                           "bg-white/5         border border-white/5"
              }`}
            >
              <div>
                <div className="text-xs font-medium text-white">{row.label}</div>
                <div className="text-[11px] font-mono text-[#94a3b8] truncate max-w-[80px]">
                  {trafficLights ? row.signal : "No data"}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`text-[11px] font-mono font-medium ${
                  isGreen ? "text-emerald-400" : isYellow ? "text-amber-400" : "text-red-400"
                }`}>
                  {isGreen ? "GREEN" : isYellow ? "YELLOW" : "RED"}
                </span>
                <div className="traffic-light">
                  <span className={`traffic-light-dot red    ${isRed    ? "active" : ""}`} />
                  <span className={`traffic-light-dot yellow ${isYellow ? "active" : ""}`} />
                  <span className={`traffic-light-dot green  ${isGreen  ? "active" : ""}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {!trafficLights && (
        <p className="mt-3 text-[11px] text-[#64748b] text-center">
          Start simulation to see live signal states
        </p>
      )}
    </div>
  );
}

export default TrafficLightStatus;