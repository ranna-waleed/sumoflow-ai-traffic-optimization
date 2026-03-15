import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, CartesianGrid,
} from "recharts";
import { CheckCircle2, Loader2, Clock } from "lucide-react";

const API = "http://127.0.0.1:8000";

const MODEL_META = {
  "YOLOv8s":     { color: "#818cf8", highlight: true  },
  "Faster RCNN": { color: "#f97316", highlight: false },
  "RetinaNet":   { color: "#34d399", highlight: false },
};

function StatusBadge({ status }) {
  if (status === "complete")   return <span className="inline-flex items-center gap-1 text-xs text-emerald-400"><CheckCircle2 className="w-3.5 h-3.5" />Complete</span>;
  if (status === "retraining") return <span className="inline-flex items-center gap-1 text-xs text-amber-400"><Loader2 className="w-3.5 h-3.5 animate-spin" />Retraining</span>;
  return <span className="inline-flex items-center gap-1 text-xs text-red-400"><Clock className="w-3.5 h-3.5" />Pending</span>;
}

function tone(val) {
  if (val == null) return "text-[#64748b]";
  if (val > 0.5)   return "text-emerald-400";
  if (val > 0.1)   return "text-indigo-400";
  return "text-red-400";
}

function ModelComparison() {
  const [models, setModels]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    fetch(`${API}/api/models/comparison`)
      .then(r => r.json())
      .then(data => { setModels(data.models); setLoading(false); })
      .catch(() => { setError("Failed to connect to backend"); setLoading(false); });
  }, []);

  const barData = models.map(m => ({ model: m.name, mAP: m.mAP50 || 0 }));
  const classes = ["car", "bus", "truck", "taxi", "microbus", "motorcycle", "bicycle"];

  return (
    <div className="space-y-6 pt-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Model Comparison</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">YOLOv8s vs Faster RCNN vs RetinaNet</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-medium">3 Models · 7 Classes</span>
          <span className="px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-medium">180 Test Images</span>
        </div>
      </div>

      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-sm">
          ⚠ {error} — make sure the backend is running on port 8000
        </div>
      )}

      {loading ? (
        <div className="text-sm text-[#64748b] animate-pulse">Loading model data...</div>
      ) : (
        <div className="grid md:grid-cols-3 gap-4">
          {models.map(m => {
            const meta = MODEL_META[m.name] || {};
            return (
              <div key={m.name} className={`card p-5 ${meta.highlight ? "ring-1 ring-indigo-500/50 shadow-lg shadow-indigo-500/10" : ""}`}>
                <div className="flex items-start justify-between gap-3 mb-4">
                  <div>
                    <h2 className="text-lg font-semibold" style={{ color: meta.color }}>{m.name}</h2>
                    <p className="text-xs text-[#94a3b8] mt-0.5">{meta.highlight ? "★ Selected Model" : "Detection model"}</p>
                  </div>
                  <StatusBadge status={m.status} />
                </div>
                <div className="space-y-2.5 text-sm">
                  {[
                    { label: "mAP@0.5",     value: m.mAP50        ? `${(m.mAP50 * 100).toFixed(1)}%`        : "—" },
                    { label: "mAP@0.5:0.95",value: m.mAP50_95     ? `${(m.mAP50_95 * 100).toFixed(1)}%`     : "—" },
                    { label: "Precision",   value: m.precision    ? `${(m.precision * 100).toFixed(1)}%`    : "—" },
                    { label: "Recall",      value: m.recall       ? `${(m.recall * 100).toFixed(1)}%`       : "—" },
                    { label: "FPS",         value: m.fps          ? `${m.fps}`                               : "—" },
                    { label: "Inference",   value: m.inference_ms ? `${m.inference_ms}ms`                   : "—" },
                  ].map(row => (
                    <div key={row.label} className="flex items-center justify-between">
                      <span className="text-[#94a3b8]">{row.label}</span>
                      <span className="font-mono text-white">{row.value}</span>
                    </div>
                  ))}
                </div>
                {m.note && (
                  <p className="mt-3 text-[11px] text-amber-400 bg-amber-500/10 rounded-lg px-3 py-2">⚠ {m.note}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {!loading && models.length > 0 && (
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Per-Class AP@0.5</h3>
            <span className="text-xs text-[#64748b] font-mono">Higher is better</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[#94a3b8] border-b border-white/10">
                  <th className="pb-3 pr-4 font-medium">Class</th>
                  {models.map(m => (
                    <th key={m.name} className="pb-3 px-4 font-medium" style={{ color: MODEL_META[m.name]?.color }}>{m.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {classes.map(cls => (
                  <tr key={cls} className="border-b border-white/5">
                    <td className="py-2.5 pr-4 font-mono text-[#e2e8f0] capitalize">{cls}</td>
                    {models.map(m => {
                      const val = m.per_class_ap?.[cls];
                      return (
                        <td key={m.name} className={`py-2.5 px-4 font-mono ${tone(val)}`}>
                          {val != null ? `${(val * 100).toFixed(1)}%` : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && (
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Global mAP@0.5 Comparison</h3>
            <span className="text-xs text-[#64748b] font-mono">Higher is better</span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="model" tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} tickMargin={8} />
                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} tickMargin={8} domain={[0, 0.6]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#161c24", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "10px", fontSize: "12px" }}
                  formatter={(v) => [`${(v * 100).toFixed(1)}%`, "mAP@0.5"]}
                />
                <Bar dataKey="mAP" radius={[6, 6, 0, 0]} fill="#818cf8" fillOpacity={0.9} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelComparison;