import React, { useState, useEffect, useRef, useCallback } from "react";
import { Play, Square, Pause, Maximize2, Activity, Brain } from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";

const API = "http://127.0.0.1:8000";

const CLASS_COLORS = {
  car:        "#94a3b8",
  taxi:       "#eab308",
  bus:        "#f97316",
  microbus:   "#38bdf8",
  truck:      "#a78bfa",
  motorcycle: "#ef4444",
  bicycle:    "#34d399",
};

const DIR_COLORS = {
  North: "#38bdf8",
  South: "#34d399",
  East:  "#f97316",
  West:  "#a78bfa",
};

const chartStyle = {
  tooltip: {
    backgroundColor: "#161c24",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "10px",
    fontSize: "12px",
  },
};

// FIXED: profile selector — was always hardcoded to "custom"
const PROFILES = [
  { key: "morning_rush", label: "Morning Rush", icon: "🌅", time: "7:30–10:30 AM" },
  { key: "evening_rush", label: "Evening Rush", icon: "🌆", time: "3:00–8:00 PM"  },
  { key: "midday",       label: "Midday",       icon: "☀️", time: "12:00–3:00 PM" },
  { key: "night",        label: "Night",        icon: "🌙", time: "10:00 PM–12 AM"},
  { key: "custom",       label: "Custom OD",    icon: "🗺️", time: "config_file.sumocfg" },
];

function LiveSimulation() {
  // FIXED: profile selector state — was hardcoded "custom"
  const [selectedProfile, setSelectedProfile] = useState("morning_rush");

  const [simRunning,  setSimRunning]  = useState(false);
  const [simPaused,   setSimPaused]   = useState(false);
  const [simStep,     setSimStep]     = useState(0);
  const [liveData,    setLiveData]    = useState(null);
  const [liveChart,   setLiveChart]   = useState([]);
  const [frame,       setFrame]       = useState(null);
  const [simError,    setSimError]    = useState(null);
  const [simLoading,  setSimLoading]  = useState(false);
  const [fullscreen,  setFullscreen]  = useState(false);

  // LSTM state
  const [lstmPred,       setLstmPred]       = useState(null);
  const [lstmStatus,     setLstmStatus]     = useState("waiting");
  const [lstmHistoryLen, setLstmHistoryLen] = useState(0);

  // FIXED: fetch real LSTM model info from API — was hardcoded
  const [lstmModelInfo, setLstmModelInfo] = useState(null);

  const pollRef = useRef(null);
  const lstmRef = useRef(null);

  // Fetch LSTM model info on mount
  useEffect(() => {
    fetch(`${API}/api/lstm/status`)
      .then(r => r.json())
      .then(data => setLstmModelInfo(data))
      .catch(() => setLstmModelInfo(null));
  }, []);

  // FIXED: handleStop stable with useCallback
  const handleStop = useCallback(async () => {
    clearInterval(pollRef.current);
    clearInterval(lstmRef.current);
    setSimRunning(false);
    setSimPaused(false);
    setLiveData(null);
    setFrame(null);
    setLstmPred(null);
    setLstmStatus("waiting");
    try { await fetch(`${API}/api/sumo/stop`, { method: "POST" }); } catch {}
  }, []);

  // Poll simulation every 2s
  useEffect(() => {
    if (simRunning && !simPaused) {
      pollRef.current = setInterval(async () => {
        try {
          const res  = await fetch(`${API}/api/sumo/step`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ steps: 30 }),
          });
          const data = await res.json();
          if (data.latest) {
            const s = data.latest;
            setSimStep(s.step);
            setLiveData(s);
            setLiveChart(prev => [...prev, {
              t: s.step, vehicles: s.vehicles,
              wait: s.avg_wait_s,
              co2:  Math.round(s.total_co2_mg / 1000),
            }].slice(-40));
            if (s.simulation_done) handleStop();
          }
          if (data.image) setFrame(data.image);
        } catch {
          setSimError("Lost connection to simulation");
          handleStop();
        }
      }, 2000);
    }
    return () => clearInterval(pollRef.current);
  }, [simRunning, simPaused, handleStop]);

  // Poll LSTM every 4s
  useEffect(() => {
    if (simRunning && !simPaused) {
      lstmRef.current = setInterval(async () => {
        try {
          const res  = await fetch(`${API}/api/lstm/predict/live`);
          const data = await res.json();
          if (data.status === "collecting") {
            setLstmStatus("collecting");
            setLstmHistoryLen(data.history_len || 0);
          } else if (data.status === "ok" && data.next_30s) {
            setLstmStatus("ready");
            setLstmPred(data.next_30s);
            setLstmHistoryLen(data.history_len || 0);
          }
        } catch {
          setLstmStatus("error");
        }
      }, 4000);
    } else {
      clearInterval(lstmRef.current);
      if (!simRunning) {
        setLstmStatus("waiting");
        setLstmPred(null);
        setLstmHistoryLen(0);
      }
    }
    return () => clearInterval(lstmRef.current);
  }, [simRunning, simPaused]);

  const handleStart = async () => {
    setSimError(null);
    setSimLoading(true);
    setLiveChart([]);
    setFrame(null);
    setLstmPred(null);
    setLstmStatus("collecting");
    try {
      const res  = await fetch(`${API}/api/sumo/start`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        // FIXED: use selectedProfile — was hardcoded "custom"
        body:    JSON.stringify({ profile: selectedProfile, gui: true }),
      });
      const data = await res.json();
      if (data.status === "started" || data.status === "already_running") {
        setSimRunning(true);
        setSimPaused(false);
        setSimStep(0);
      } else {
        setSimError(data.detail || "Failed to start");
      }
    } catch {
      setSimError("Cannot reach backend — is it running?");
    }
    setSimLoading(false);
  };

  const handlePause = () => {
    setSimPaused(p => !p);
    if (!simPaused) clearInterval(lstmRef.current);
  };

  const typeCounts    = liveData?.type_counts || {};
  const totalVehicles = Object.values(typeCounts).reduce((a, b) => a + b, 0) || 1;
  const maxPred       = lstmPred ? Math.max(...Object.values(lstmPred), 1) : 1;

  // FIXED: LSTM model description from API — was hardcoded text
  const lstmModelLabel = lstmModelInfo?.model || "BiLSTM 2×128";
  const lstmPredictsLabel = lstmModelInfo?.predicts || "Next 30s per direction";

  return (
    <div className="space-y-6 pt-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Live Simulation</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">
            Real-time traffic · El-Tahrir Square, Cairo
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border ${
            simRunning
              ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
              : "bg-white/5 border-white/10 text-[#94a3b8]"}`}>
            <span className={`w-2 h-2 rounded-full ${simRunning ? "bg-emerald-400 animate-pulse" : "bg-[#64748b]"}`} />
            {simRunning ? (simPaused ? "Paused" : "Live") : "Stopped"}
          </span>
          {simRunning && (
            <span className="px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-mono">
              Step: {simStep.toLocaleString()}
            </span>
          )}
        </div>
      </div>

      {/* FIXED: Profile selector */}
      <div className="flex flex-wrap gap-2 items-center">
        {PROFILES.map(p => (
          <button key={p.key}
            onClick={() => { if (!simRunning) setSelectedProfile(p.key); }}
            disabled={simRunning}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
              selectedProfile === p.key
                ? "bg-indigo-500 border-indigo-400 text-white"
                : simRunning
                ? "bg-white/5 border-white/10 text-[#64748b] cursor-not-allowed"
                : "bg-white/5 border-white/10 text-[#94a3b8] hover:bg-white/10"}`}>
            <span>{p.icon}</span>
            <span>{p.label}</span>
          </button>
        ))}
        {simRunning && <span className="text-xs text-[#64748b]">← Stop to change profile</span>}
      </div>

      {simError && (
        <div className="px-4 py-3 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-sm">
          ⚠ {simError}
        </div>
      )}

      {/* Top row */}
      <div className="grid lg:grid-cols-[1fr_300px] gap-6">

        {/* Viewport */}
        <div className="card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              <h2 className="text-sm font-semibold text-white">Simulation Viewport</h2>
            </div>
            <div className="flex items-center gap-2">
              {frame && (
                <button onClick={() => setFullscreen(true)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-[#94a3b8] border border-white/10 text-xs">
                  <Maximize2 className="w-3.5 h-3.5" /> Fullscreen
                </button>
              )}
              <span className="text-xs font-mono text-[#64748b]">SUMO + TraCI</span>
            </div>
          </div>

          <div className="flex gap-2">
            <button onClick={handleStart} disabled={simRunning || simLoading}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                simRunning || simLoading
                  ? "bg-indigo-500/30 text-indigo-300 cursor-not-allowed"
                  : "bg-indigo-500 hover:bg-indigo-400 text-white"}`}>
              <Play className="w-4 h-4" strokeWidth={2.5} />
              {simLoading ? "Starting..." : "Start"}
            </button>
            <button onClick={handleStop} disabled={!simRunning}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                !simRunning
                  ? "bg-red-500/10 text-red-300/40 border-red-500/20 cursor-not-allowed"
                  : "bg-red-500/20 hover:bg-red-500/30 text-red-400 border-red-500/40"}`}>
              <Square className="w-4 h-4" strokeWidth={2.5} /> Stop
            </button>
            <button onClick={handlePause} disabled={!simRunning}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                !simRunning
                  ? "bg-white/5 text-[#94a3b8]/40 border-white/10 cursor-not-allowed"
                  : simPaused
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                  : "bg-white/5 hover:bg-white/10 text-[#94a3b8] border-white/10"}`}>
              <Pause className="w-4 h-4" strokeWidth={2.5} />
              {simPaused ? "Resume" : "Pause"}
            </button>
          </div>

          <div className="relative h-72 md:h-80 rounded-xl overflow-hidden bg-[#0a0f16]">
            {frame ? (
              <>
                <img src={frame} alt="SUMO" className="w-full h-full object-cover" />
                <div className="absolute top-3 left-3 flex gap-2 z-10">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-black/70 border border-emerald-500/40 text-emerald-300 text-[11px] font-mono">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    {simPaused ? "PAUSED" : "LIVE"}
                  </span>
                  <span className="px-2.5 py-1 rounded-lg bg-black/70 border border-indigo-500/30 text-indigo-300 text-[11px] font-mono">
                    Step {simStep.toLocaleString()}
                  </span>
                </div>
                {liveData && (
                  <div className="absolute bottom-3 left-3 right-3 z-10 flex justify-between">
                    <span className="px-2.5 py-1 rounded-lg bg-black/70 text-[11px] font-mono text-white">
                      🚗 {liveData.vehicles} vehicles
                    </span>
                    <span className="px-2.5 py-1 rounded-lg bg-black/70 text-[11px] font-mono text-white">
                      ⏱ {liveData.avg_wait_s}s avg wait
                    </span>
                    <span className="px-2.5 py-1 rounded-lg bg-black/70 text-[11px] font-mono text-white">
                      💨 {(liveData.total_co2_mg / 1000).toFixed(0)}k mg CO₂
                    </span>
                  </div>
                )}
              </>
            ) : (
              <div className="scanline h-full flex items-center justify-center">
                <div className="text-center px-6">
                  {simRunning ? (
                    <>
                      <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                      <p className="text-sm text-indigo-300">Loading SUMO-GUI...</p>
                    </>
                  ) : (
                    <>
                      <p className="text-base font-semibold text-indigo-300 mb-1">Live Traffic Simulation</p>
                      <p className="text-sm text-[#94a3b8] mb-2">El-Tahrir Square, Cairo</p>
                      <p className="text-xs text-[#64748b]">Select a profile and press <span className="text-indigo-400">Start</span></p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right panel */}
        <div className="space-y-4">

          {/* Vehicle Classes */}
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-white">Vehicle Classes</h3>
              {liveData && (
                <span className="text-[11px] font-mono text-[#64748b]">
                  Total: {liveData.vehicles}
                </span>
              )}
            </div>
            <div className="space-y-3">
              {Object.entries(CLASS_COLORS).map(([cls, color]) => {
                const count = typeCounts[cls] || 0;
                const pct   = totalVehicles > 0 ? Math.round((count / totalVehicles) * 100) : 0;
                return (
                  <div key={cls}>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-[#e2e8f0] capitalize">{cls}</span>
                      </div>
                      <span className="font-mono text-[#94a3b8]">
                        {simRunning ? `${count} (${pct}%)` : "—"}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-500"
                        style={{ width: simRunning ? `${pct}%` : "0%", backgroundColor: color }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* LSTM Prediction */}
          <div className="card p-5">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-semibold text-white">BiLSTM Prediction</h3>
              </div>
              <span className={`text-[11px] font-mono px-2 py-0.5 rounded ${
                lstmStatus === "ready"      ? "bg-emerald-500/15 text-emerald-400" :
                lstmStatus === "collecting" ? "bg-amber-500/15 text-amber-400"    :
                                              "bg-white/5 text-[#64748b]"
              }`}>
                {lstmStatus === "ready"      ? "● Live"                    :
                 lstmStatus === "collecting" ? `${lstmHistoryLen}/60`      :
                 lstmStatus === "error"      ? "Error"                     : "Waiting"}
              </span>
            </div>

            {/* FIXED: model info from API — was hardcoded */}
            <p className="text-xs text-[#94a3b8] mb-4">
              {lstmModelLabel} · {lstmPredictsLabel}
            </p>

            {lstmStatus === "waiting" && (
              <p className="text-xs text-[#64748b] text-center py-4">
                Start simulation to see predictions
              </p>
            )}

            {lstmStatus === "collecting" && (
              <div className="space-y-2">
                <p className="text-xs text-amber-400 text-center">
                  Collecting history... {lstmHistoryLen}/60 steps
                </p>
                <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div className="h-full rounded-full bg-amber-500 transition-all duration-500"
                    style={{ width: `${(lstmHistoryLen / 60) * 100}%` }} />
                </div>
              </div>
            )}

            {lstmStatus === "ready" && lstmPred && (
              <div className="space-y-3">
                {Object.entries(DIR_COLORS).map(([dir, color]) => {
                  const key   = dir.toLowerCase();
                  const count = lstmPred[key] || 0;
                  const pct   = Math.round((count / maxPred) * 100);
                  return (
                    <div key={dir}>
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <span className="text-[#e2e8f0] font-medium">{dir}</span>
                        <span className="font-mono" style={{ color }}>{count} vehicles</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-700"
                          style={{ width: `${pct}%`, backgroundColor: color }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {lstmStatus === "error" && (
              <p className="text-xs text-red-400 text-center py-2">
                ⚠ LSTM prediction failed
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Live Charts */}
      <div className="grid lg:grid-cols-3 gap-6">
        {[
          { key: "vehicles", label: "Vehicles",       unit: "count",    color: "#818cf8", id: "gV", fmt: v => [v, "Vehicles"]       },
          { key: "wait",     label: "Avg Wait Time",  unit: "seconds",  color: "#34d399", id: "gW", fmt: v => [`${v}s`, "Avg Wait"] },
          { key: "co2",      label: "CO₂ Emissions",  unit: "×1000 mg", color: "#fbbf24", id: "gC", fmt: v => [`${v}k mg`, "CO₂"]   },
        ].map(chart => (
          <div key={chart.key} className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-white">{chart.label}</h3>
              <span className="text-xs font-mono text-[#64748b]">
                {simRunning ? "● Live · " : ""}{chart.unit}
              </span>
            </div>
            <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={liveChart} margin={{ left: -20, right: 4, top: 4, bottom: 0 }}>
                  <defs>
                    <linearGradient id={chart.id} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={chart.color} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={chart.color} stopOpacity={0}   />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} />
                  <Tooltip contentStyle={chartStyle.tooltip} formatter={chart.fmt} />
                  <Area type="monotone" dataKey={chart.key} stroke={chart.color}
                    strokeWidth={2} fill={`url(#${chart.id})`} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>

      {/* Fullscreen */}
      {fullscreen && frame && (
        <div className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center"
          onClick={() => setFullscreen(false)}>
          <img src={frame} alt="fullscreen" className="max-w-full max-h-full rounded-xl" />
          <button className="absolute top-4 right-4 px-3 py-1.5 rounded-lg bg-white/10 text-white text-sm hover:bg-white/20"
            onClick={() => setFullscreen(false)}>✕ Close</button>
        </div>
      )}
    </div>
  );
}

export default LiveSimulation;