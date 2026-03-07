import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { Leaf, Timer, Car } from "lucide-react";

const co2Before = [420, 410, 400, 420, 410, 400, 415, 405, 420, 410, 400, 420];
const co2After = [420, 400, 380, 360, 345, 335, 325, 320, 315, 312, 310, 308];

const waitBefore = [45, 46, 44, 47, 45, 46, 45, 44, 46, 45, 45, 44];
const waitAfter = [45, 42, 40, 37, 35, 33, 31, 30, 29, 28, 28, 27];

const ticks = Array.from({ length: 12 }).map((_, i) => i + 1);

function buildSeries(a, b) {
  return ticks.map((t, i) => ({ t, before: a[i], after: b[i] }));
}

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

function BeforeAfter() {
  return (
    <div className="space-y-6 pt-4">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Before vs After</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">
            Static lights vs AI adaptive lights
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="px-3 py-1.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-medium">
            Before: Static Lights
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
            After: AI Adaptive Lights
          </span>
        </div>
      </div>

      {/* Comparison cards */}
      <div className="grid md:grid-cols-3 gap-4 items-stretch">
        <div className="card border-red-500/30 p-5">
          <h3 className="text-sm font-semibold text-red-400 mb-4 flex items-center gap-2">
            <span>⚠</span> Before — Static Lights
          </h3>
          <div className="space-y-4">
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">CO₂ Emissions</div>
              <div className="font-mono text-red-400 font-semibold">1,200 mg/step</div>
            </div>
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">Avg Wait Time</div>
              <div className="font-mono text-amber-400 font-semibold">45 seconds</div>
            </div>
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">Throughput</div>
              <div className="font-mono text-[#94a3b8] font-semibold">180 vehicles/hr</div>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-center">
          <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-[#64748b] text-sm font-semibold">
            VS
          </div>
        </div>

        <div className="card border-emerald-500/30 p-5">
          <h3 className="text-sm font-semibold text-emerald-400 mb-4 flex items-center gap-2">
            <span>✓</span> After — AI Adaptive
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs text-[#94a3b8] mb-0.5">CO₂ Emissions</div>
                <div className="font-mono text-emerald-400 font-semibold">800 mg/step</div>
              </div>
              <span className="shrink-0 px-2 py-0.5 rounded-lg bg-emerald-500/20 text-emerald-300 text-[11px] font-mono font-medium">
                ↓33%
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs text-[#94a3b8] mb-0.5">Avg Wait Time</div>
                <div className="font-mono text-emerald-400 font-semibold">28 seconds</div>
              </div>
              <span className="shrink-0 px-2 py-0.5 rounded-lg bg-emerald-500/20 text-emerald-300 text-[11px] font-mono font-medium">
                ↓38%
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs text-[#94a3b8] mb-0.5">Throughput</div>
                <div className="font-mono text-emerald-400 font-semibold">247 vehicles/hr</div>
              </div>
              <span className="shrink-0 px-2 py-0.5 rounded-lg bg-emerald-500/20 text-emerald-300 text-[11px] font-mono font-medium">
                ↑37%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="card p-6 border-emerald-500/20 bg-gradient-to-r from-emerald-500/5 via-transparent to-indigo-500/5">
        <h3 className="text-sm font-semibold text-white text-center mb-6">
          System Improvement Summary
        </h3>
        <div className="grid sm:grid-cols-3 gap-4">
          <div className="card p-5 bg-white/5 border-white/10 text-center">
            <Leaf className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
            <div className="text-xs text-[#94a3b8] mb-1">CO₂ Reduction</div>
            <div className="font-mono text-2xl font-semibold text-emerald-400">33%</div>
          </div>
          <div className="card p-5 bg-white/5 border-white/10 text-center">
            <Timer className="w-8 h-8 text-indigo-400 mx-auto mb-2" />
            <div className="text-xs text-[#94a3b8] mb-1">Wait Time Reduction</div>
            <div className="font-mono text-2xl font-semibold text-emerald-400">38%</div>
          </div>
          <div className="card p-5 bg-white/5 border-white/10 text-center">
            <Car className="w-8 h-8 text-indigo-400 mx-auto mb-2" />
            <div className="text-xs text-[#94a3b8] mb-1">Throughput Increase</div>
            <div className="font-mono text-2xl font-semibold text-emerald-400">37%</div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">CO₂ Over Time</h3>
            <span className="text-xs font-mono text-[#64748b]">mg/step</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={buildSeries(co2Before, co2After)}
                margin={{ left: -20, right: 8, top: 5, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="4 4" stroke={chartStyle.grid.stroke} />
                <XAxis dataKey="t" tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <YAxis tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <Tooltip contentStyle={chartStyle.tooltip} labelFormatter={(t) => `t = ${t}`} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="before"
                  name="Before"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={{ r: 2, fill: "#ef4444" }}
                  activeDot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="after"
                  name="After"
                  stroke="#34d399"
                  strokeWidth={2}
                  dot={{ r: 2, fill: "#34d399" }}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Wait Time Over Time</h3>
            <span className="text-xs font-mono text-[#64748b]">seconds</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={buildSeries(waitBefore, waitAfter)}
                margin={{ left: -20, right: 8, top: 5, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="4 4" stroke={chartStyle.grid.stroke} />
                <XAxis dataKey="t" tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <YAxis tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                <Tooltip contentStyle={chartStyle.tooltip} labelFormatter={(t) => `t = ${t}`} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="before"
                  name="Before"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={{ r: 2, fill: "#ef4444" }}
                  activeDot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="after"
                  name="After"
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

export default BeforeAfter;
