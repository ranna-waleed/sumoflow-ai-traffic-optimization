import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { CheckCircle2, Loader2, Clock } from "lucide-react";

const models = [
  {
    name: "YOLO",
    color: "#818cf8",
    status: "Complete",
    statusIcon: CheckCircle2,
    statusColor: "text-emerald-400",
    map: "41.3%",
    precision: "68.3%",
    recall: "37.1%",
    fps: "126",
    time: "7.9ms",
    highlight: true,
  },
  {
    name: "RetinaNet",
    color: "#34d399",
    status: "Training",
    statusIcon: Loader2,
    statusColor: "text-amber-400",
    map: "—",
    precision: "—",
    recall: "—",
    fps: "—",
    time: "—",
    highlight: false,
  },
  {
    name: "Faster RCNN",
    color: "#f97316",
    status: "Pending",
    statusIcon: Clock,
    statusColor: "text-red-400",
    map: "—",
    precision: "—",
    recall: "—",
    fps: "—",
    time: "—",
    highlight: false,
  },
];

const perClass = [
  { cls: "Car", yolo: 46.3, retinanet: null, faster: null },
  { cls: "Bus", yolo: 94.4, retinanet: null, faster: null },
  { cls: "Truck", yolo: 65.4, retinanet: null, faster: null },
  { cls: "Taxi", yolo: 42.9, retinanet: null, faster: null },
  { cls: "Microbus", yolo: 38.1, retinanet: null, faster: null },
  { cls: "Motorcycle", yolo: 0.4, retinanet: null, faster: null },
  { cls: "Bicycle", yolo: 1.7, retinanet: null, faster: null },
];

const barData = [
  { model: "YOLO", value: 0.413 },
  { model: "RetinaNet", value: 0 },
  { model: "Faster RCNN", value: 0 },
];

function percentageTone(val) {
  if (val == null) return "";
  if (val > 50) return "text-emerald-400";
  if (val >= 10) return "text-indigo-400";
  return "text-red-400";
}

function ModelComparison() {
  return (
    <div className="space-y-6 pt-4">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Model Comparison</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">
            YOLO vs RetinaNet vs Faster RCNN
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-medium">
            3 Models Comparison
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
            YOLO ✓ Complete
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-medium">
            RetinaNet ⟳ Training
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-medium">
            Faster RCNN ● Pending
          </span>
        </div>
      </div>

      {/* Model cards */}
      <div className="grid md:grid-cols-3 gap-4">
        {models.map((m) => {
          const StatusIcon = m.statusIcon;
          return (
            <div
              key={m.name}
              className={`card p-5 ${
                m.highlight
                  ? "ring-1 ring-indigo-500/50 shadow-lg shadow-indigo-500/10"
                  : ""
              }`}
            >
              <div className="flex items-start justify-between gap-3 mb-4">
                <div>
                  <h2
                    className="text-lg font-semibold"
                    style={{ color: m.color }}
                  >
                    {m.name}
                  </h2>
                  <p className="text-xs text-[#94a3b8] mt-0.5">
                    Vehicle detection model
                  </p>
                </div>
                <span className={`inline-flex items-center gap-1 text-xs font-medium ${m.statusColor}`}>
                  <StatusIcon className="w-3.5 h-3.5" />
                  {m.status}
                </span>
              </div>
              <div className="space-y-2.5 text-sm">
                {[
                  { label: "mAP@0.5", value: m.map },
                  { label: "Precision", value: m.precision },
                  { label: "Recall", value: m.recall },
                  { label: "FPS", value: m.fps },
                  { label: "Inference Time", value: m.time },
                ].map((row) => (
                  <div key={row.label} className="flex items-center justify-between">
                    <span className="text-[#94a3b8]">{row.label}</span>
                    <span className="font-mono text-white">{row.value}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Per class table */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-white">Per Class mAP@0.5</h3>
          <span className="text-xs text-[#64748b] font-mono">Higher is better</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[#94a3b8] border-b border-white/10">
                <th className="pb-3 pr-4 font-medium">Class</th>
                <th className="pb-3 px-4 font-medium">YOLO</th>
                <th className="pb-3 px-4 font-medium">RetinaNet</th>
                <th className="pb-3 px-4 font-medium">Faster RCNN</th>
              </tr>
            </thead>
            <tbody>
              {perClass.map((row) => (
                <tr key={row.cls} className="border-b border-white/5">
                  <td className="py-2.5 pr-4 font-mono text-[#e2e8f0]">{row.cls}</td>
                  <td className={`py-2.5 px-4 font-mono ${percentageTone(row.yolo)}`}>
                    {row.yolo != null ? `${row.yolo.toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-2.5 px-4 font-mono text-[#64748b]">—</td>
                  <td className="py-2.5 px-4 font-mono text-[#64748b]">—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bar chart */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-white">Global mAP@0.5 by Model</h3>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="model"
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickLine={false}
                tickMargin={8}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickLine={false}
                tickMargin={8}
                domain={[0, 0.5]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#161c24",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "10px",
                  fontSize: "12px",
                }}
                formatter={(value, _, item) => [
                  `${(value * 100).toFixed(1)}%`,
                  item?.payload?.model || "mAP",
                ]}
              />
              <Bar
                dataKey="value"
                radius={[6, 6, 0, 0]}
                fill="#818cf8"
                fillOpacity={0.9}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default ModelComparison;
