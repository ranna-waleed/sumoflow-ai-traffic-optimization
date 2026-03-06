import React from "react";
import { Play, Square, Pause, Gauge } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import MetricCard from "../components/MetricCard";
import TrafficLightStatus from "../components/TrafficLightStatus";

const co2Series = [320, 410, 380, 450, 420, 390, 360, 340, 380, 350, 330, 310];
const waitSeries = [45, 52, 48, 60, 55, 50, 44, 40, 48, 42, 38, 35];
const chartData = Array.from({ length: 12 }).map((_, i) => ({
  t: i + 1,
  co2: co2Series[i],
  wait: waitSeries[i],
}));

const chartStyle = {
  grid: { stroke: "rgba(255,255,255,0.06)", strokeDasharray: "4 4" },
  tick: { fill: "#64748b", fontSize: 11 },
  tooltip: {
    backgroundColor: "#161c24",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "10px",
    fontSize: "12px",
  },
};

function Dashboard() {
  return (
    <div className="space-y-6 pt-4">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Dashboard</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">
            El-Tahrir Square — Cairo, EG
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            System Online
          </span>
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-mono font-medium">
            Step: 1,240
          </span>
          <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-medium">
            ⚡ AI Optimizer Active
          </span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Vehicles"
          value="247"
          unit="active"
          color="#818cf8"
          data={[220, 230, 210, 260, 247, 240, 245]}
        />
        <MetricCard
          label="Avg Wait Time"
          value="28.4"
          unit="seconds"
          color="#34d399"
          data={[40, 38, 35, 32, 30, 29, 28]}
        />
        <MetricCard
          label="CO₂ Emissions"
          value="312"
          unit="mg/step"
          color="#fbbf24"
          data={[380, 360, 340, 330, 320, 315, 312]}
        />
        <MetricCard
          label="Detection FPS"
          value="126"
          unit="fps"
          color="#a78bfa"
          data={[110, 115, 120, 118, 122, 124, 126]}
        />
      </div>

      {/* Main content */}
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        {/* Simulation frame */}
        <div className="card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Gauge className="w-5 h-5 text-indigo-400" />
              <h2 className="text-sm font-semibold text-white">Live Simulation Feed</h2>
            </div>
            <span className="text-xs font-mono text-[#64748b]">SUMO + TraCI</span>
          </div>

          <div className="flex gap-2">
            <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 text-white text-sm font-medium transition-colors">
              <Play className="w-4 h-4" strokeWidth={2.5} />
              Start
            </button>
            <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/40 text-sm font-medium transition-colors">
              <Square className="w-4 h-4" strokeWidth={2.5} />
              Stop
            </button>
            <button className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-[#94a3b8] border border-white/10 text-sm font-medium transition-colors">
              <Pause className="w-4 h-4" strokeWidth={2.5} />
              Pause
            </button>
          </div>

          <div className="relative viewport-frame scanline h-72 md:h-80 flex items-center justify-center rounded-xl overflow-hidden">
            <div className="relative z-10 text-center px-6">
              <p className="text-base font-semibold text-indigo-300 mb-1">
                SUMO Simulation Frame
              </p>
              <p className="text-sm text-[#94a3b8] mb-4">El-Tahrir Square — Cairo</p>
              <p className="text-xs text-[#64748b] max-w-sm mx-auto">
                Real-time vehicle flows, adaptive signal phases, and AI-driven optimization.
              </p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex-1 max-w-xs">
              <div className="flex justify-between text-xs text-[#94a3b8] mb-1">
                <span>Simulation Speed</span>
                <span className="font-mono text-white">1.0x</span>
              </div>
              <input
                type="range"
                min="0"
                max="4"
                defaultValue="1"
                className="w-full h-2 rounded-full appearance-none bg-white/10 accent-indigo-500"
              />
            </div>
            <span className="text-xs font-mono text-[#64748b]">
              Delay: <span className="text-indigo-400">0ms</span>
            </span>
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-4">
          <TrafficLightStatus />
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <h3 className="text-sm font-semibold text-white">DQN Optimizer</h3>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-[#94a3b8]">Action</span>
                <span className="px-2.5 py-1 rounded-lg bg-indigo-500/15 text-indigo-300 font-mono text-xs">
                  NORTH GREEN 60s
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#94a3b8]">Reward</span>
                <span className="font-mono text-red-400">-28.4</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#94a3b8]">Epsilon</span>
                <span className="font-mono text-indigo-400">0.12</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">CO₂ Emissions Over Time</h3>
            <span className="text-xs font-mono text-[#64748b]">mg/step</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                <CartesianGrid strokeDasharray="4 4" stroke={chartStyle.grid.stroke} />
                <XAxis dataKey="t" tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <YAxis tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <Tooltip
                  contentStyle={chartStyle.tooltip}
                  labelFormatter={(t) => `t = ${t}`}
                  formatter={(v) => [`${v} mg/step`, "CO₂"]}
                />
                <Line
                  type="monotone"
                  dataKey="co2"
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={{ r: 2, fill: "#fbbf24" }}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Average Waiting Time</h3>
            <span className="text-xs font-mono text-[#64748b]">seconds</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                <CartesianGrid strokeDasharray="4 4" stroke={chartStyle.grid.stroke} />
                <XAxis dataKey="t" tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <YAxis tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <Tooltip
                  contentStyle={chartStyle.tooltip}
                  labelFormatter={(t) => `t = ${t}`}
                  formatter={(v) => [`${v} s`, "Avg Wait"]}
                />
                <Line
                  type="monotone"
                  dataKey="wait"
                  stroke="#34d399"
                  strokeWidth={2}
                  dot={{ r: 2, fill: "#34d399" }}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
