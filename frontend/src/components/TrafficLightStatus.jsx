import React from "react";
import { Compass } from "lucide-react";



// Determine overall phase from a SUMO signal string like "GGGGrrrrrr"
function parsePhase(signalStr) {
  if (!signalStr) return "red";
  const s = signalStr.toLowerCase();
  if (s.includes("y")) return "yellow";
  if (s.includes("g")) return "green";   // any green = green
  return "red";
}

// Static fallback when simulation is not running
const STATIC = [
  { id: "north", label: "North", phase: "red",   signal: "——" },
  { id: "south", label: "South", phase: "red",   signal: "——" },
  { id: "east",  label: "East",  phase: "red",   signal: "——" },
  { id: "west",  label: "West",  phase: "red",   signal: "——" },
];

function TrafficLightStatus({ trafficLights, typeCounts }) {
  console.log("trafficLights received:", trafficLights);  
  // Build display rows from real TraCI data
  let rows;

  if (trafficLights && Object.keys(trafficLights).length > 0) {
    const ids = Object.keys(trafficLights);
    const labels = ["North", "South", "East", "West"];

    // Map first 4 traffic light IDs to cardinal directions
    rows = labels.map((label, i) => {
      const id     = ids[i] || ids[0];
      const signal = trafficLights[id] || "rrrr";
      const phase  = parsePhase(signal);
      return { id, label, phase, signal };
    });
  } else {
    rows = STATIC;
  }

  // Vehicle counts per direction — approximate from type_counts total
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
        {rows.map((row) => {
          const isGreen  = row.phase === "green";
          const isYellow = row.phase === "yellow";
          const isRed    = row.phase === "red";

          return (
            <div key={row.label}
              className={`flex items-center justify-between p-3 rounded-xl transition-colors ${
                isGreen  ? "bg-emerald-500/10 border border-emerald-500/25" :
                isYellow ? "bg-amber-500/10 border border-amber-500/25" :
                           "bg-white/5 border border-white/5"
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