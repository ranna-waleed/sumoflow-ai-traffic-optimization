import React from "react";
import { Maximize2, ZoomIn, ZoomOut, Camera, Play, Square } from "lucide-react";

const detectionClasses = [
  { name: "Car", count: 142, color: "#94a3b8" },
  { name: "Taxi", count: 48, color: "#eab308" },
  { name: "Bus", count: 22, color: "#f97316" },
  { name: "Microbus", count: 24, color: "#38bdf8" },
  { name: "Truck", count: 8, color: "#a78bfa" },
  { name: "Motorcycle", count: 3, color: "#ef4444" },
];

const lstmPredictions = [
  { dir: "North", value: 30, color: "#38bdf8" },
  { dir: "South", value: 12, color: "#34d399" },
  { dir: "East", value: 18, color: "#f97316" },
  { dir: "West", value: 6, color: "#a78bfa" },
];

const maxVehicles = 247;

function LiveSimulation() {
  return (
    <div className="space-y-6 pt-4">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Live Simulation</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">
            Real-time SUMO stream with YOLO detection
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            Live
          </span>
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-mono font-medium">
            Epoch: 42/50
          </span>
          <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 text-white text-sm font-medium transition-colors">
            <Play className="w-4 h-4" strokeWidth={2.5} />
            Run Simulation
          </button>
          <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/40 text-sm font-medium transition-colors">
            <Square className="w-4 h-4" strokeWidth={2.5} />
            Stop
          </button>
          <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-[#94a3b8] border border-white/10 text-sm font-medium transition-colors">
            <Camera className="w-4 h-4" strokeWidth={2.5} />
            Capture Frame
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        {/* Simulation viewport */}
        <div className="card p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <h2 className="text-sm font-semibold text-white">Simulation Viewport</h2>
            <div className="flex gap-2">
              <button className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-[#94a3b8] border border-white/10 text-xs font-medium transition-colors">
                <ZoomIn className="w-3.5 h-3.5" />
                Zoom In
              </button>
              <button className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-[#94a3b8] border border-white/10 text-xs font-medium transition-colors">
                <ZoomOut className="w-3.5 h-3.5" />
                Zoom Out
              </button>
              <button className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-300 border border-indigo-500/30 text-xs font-medium transition-colors">
                <Maximize2 className="w-3.5 h-3.5" />
                Fullscreen
              </button>
            </div>
          </div>

          <div className="relative viewport-frame scanline h-80 lg:h-96 flex items-center justify-center rounded-xl overflow-hidden">
            <div className="absolute top-4 right-4 z-20">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-[11px] font-mono font-medium">
                YOLO Detection On
              </span>
            </div>
            <div className="relative z-10 text-center px-6">
              <p className="text-base font-semibold text-indigo-300 mb-1">SUMO-GUI Stream</p>
              <p className="text-sm text-[#94a3b8] mb-2">
                Real-time El-Tahrir Square with vehicle detection overlays
              </p>
              <p className="text-xs text-[#64748b] font-mono">
                Frame: 12,406 · Model: YOLOv8 · Threshold: 0.45
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-[#94a3b8]">
            <span>Step: <span className="text-indigo-400">1,240</span></span>
            <span>Time: <span className="text-indigo-400">620s</span></span>
            <span>Vehicles: <span className="text-indigo-400">247</span></span>
            <span>FPS: <span className="text-indigo-400">126</span></span>
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-4">
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-white">Detection Results</h3>
              <span className="text-[11px] font-mono text-[#64748b]">Max: {maxVehicles}</span>
            </div>
            <div className="space-y-3">
              {detectionClasses.map((cls) => {
                const pct = Math.min(100, (cls.count / maxVehicles) * 100);
                return (
                  <div key={cls.name}>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="text-[#e2e8f0]">{cls.name}</span>
                      <span className="font-mono text-[#94a3b8]">{cls.count}</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, backgroundColor: cls.color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card p-5">
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-white">LSTM Prediction</h3>
              <p className="text-xs text-[#94a3b8] mt-0.5">Next 30 seconds</p>
            </div>
            <div className="space-y-2">
              {lstmPredictions.map((row) => (
                <div
                  key={row.dir}
                  className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/5"
                >
                  <span className="text-xs font-medium text-[#e2e8f0]">{row.dir}</span>
                  <span
                    className="text-sm font-mono font-medium"
                    style={{ color: row.color }}
                  >
                    {row.value} vehicles
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LiveSimulation;
