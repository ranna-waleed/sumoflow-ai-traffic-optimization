import React from "react";
import { Compass } from "lucide-react";

const directions = [
  { id: "north", label: "North", vehicles: 82, status: "green" },
  { id: "south", label: "South", vehicles: 64, status: "red" },
  { id: "east", label: "East", vehicles: 56, status: "red" },
  { id: "west", label: "West", vehicles: 45, status: "red" },
];

function TrafficLightStatus() {
  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Compass className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-white">Traffic Light Status</h3>
      </div>
      <div className="space-y-3">
        {directions.map((dir) => {
          const isGreen = dir.status === "green";
          return (
            <div
              key={dir.id}
              className={`flex items-center justify-between p-3 rounded-xl transition-colors ${
                isGreen
                  ? "bg-emerald-500/10 border border-emerald-500/25"
                  : "bg-white/5 border border-white/5"
              }`}
            >
              <div>
                <div className="text-xs font-medium text-white">{dir.label}</div>
                <div className="text-[11px] font-mono text-[#94a3b8]">
                  {dir.vehicles} vehicles
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span
                  className={`text-[11px] font-mono font-medium ${
                    isGreen ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {isGreen ? "GREEN" : "RED"}
                </span>
                <div className="traffic-light">
                  <span className={`traffic-light-dot red ${dir.status === "red" ? "active" : ""}`} />
                  <span
                    className={`traffic-light-dot yellow ${dir.status === "yellow" ? "active" : ""}`}
                  />
                  <span
                    className={`traffic-light-dot green ${dir.status === "green" ? "active" : ""}`}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default TrafficLightStatus;
