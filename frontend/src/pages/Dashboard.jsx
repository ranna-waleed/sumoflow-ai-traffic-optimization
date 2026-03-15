import React, { useState, useEffect, useRef } from "react";
import { Play, Square, Pause, Gauge, Maximize2 } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import MetricCard from "../components/MetricCard";
import TrafficLightStatus from "../components/TrafficLightStatus";

const API = "http://127.0.0.1:8000";

const chartStyle = {
  tooltip: {
    backgroundColor: "#161c24",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "10px",
    fontSize: "12px",
  },
};

const PROFILES = ["morning_rush", "evening_rush", "midday", "night"];
const PROFILE_LABELS = {
  morning_rush: "Morning Rush",
  evening_rush: "Evening Rush",
  midday:       "Midday",
  night:        "Night",
};

function Dashboard() {
  const [selectedProfile, setSelectedProfile] = useState("morning_rush");
  const [metrics, setMetrics]       = useState(null);
  const [timeseries, setTimeseries] = useState([]);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState(null);

  const [simRunning, setSimRunning] = useState(false);
  const [simPaused, setSimPaused]   = useState(false);
  const [simStep, setSimStep]       = useState(0);
  const [liveData, setLiveData]     = useState(null);
  const [liveChart, setLiveChart]   = useState([]);
  const [simError, setSimError]     = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const [frame, setFrame]           = useState(null);
  const [fullscreen, setFullscreen] = useState(false);

  const pollRef = useRef(null);

  // Load CSV metrics
  useEffect(() => {
    setLoading(true);
    fetch(`${API}/api/simulation/metrics/${selectedProfile}`)
      .then(r => r.json())
      .then(data => {
        setMetrics(data);
        const sampled = data.timeseries.filter((_, i) => i % 10 === 0);
        setTimeseries(sampled.map(d => ({
          t: d.step, co2: Math.round(d.co2 / 1000),
          wait: d.avg_wait, vehicles: d.vehicles,
        })));
        setLoading(false);
      })
      .catch(() => { setError("Failed to connect to backend"); setLoading(false); });
  }, [selectedProfile]);

  // Poll /step every 2s — gets state + screenshot in one call
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

          // Update state
          if (data.latest) {
            const s = data.latest;
            setSimStep(s.step);
            setLiveData(s);
            setLiveChart(prev => [...prev, {
              t: s.step, co2: Math.round(s.total_co2_mg / 1000),
              wait: s.avg_wait_s, vehicles: s.vehicles,
            }].slice(-30));
            if (s.simulation_done) handleStop();
          }

          // Update screenshot — comes from same response!
          if (data.image) {
            setFrame(data.image);
          }

        } catch {
          setSimError("Lost connection to simulation");
          handleStop();
        }
      }, 2000);
    }
    return () => clearInterval(pollRef.current);
  }, [simRunning, simPaused]);

  const handleStart = async () => {
    setSimError(null);
    setSimLoading(true);
    setLiveChart([]);
    setFrame(null);
    try {
      const res  = await fetch(`${API}/api/sumo/start`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
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

  const handleStop = async () => {
    clearInterval(pollRef.current);
    setSimRunning(false);
    setSimPaused(false);
    setLiveData(null);
    setFrame(null);
    try { await fetch(`${API}/api/sumo/stop`, { method: "POST" }); } catch {}
  };

  const handlePause = () => setSimPaused(p => !p);
  const chartData = simRunning && liveChart.length > 0 ? liveChart : timeseries;

  return (
    <div className="space-y-6 pt-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Dashboard</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">El-Tahrir Square — Cairo, EG</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border ${
            simRunning ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
                       : "bg-white/5 border-white/10 text-[#94a3b8]"}`}>
            <span className={`w-2 h-2 rounded-full ${simRunning ? "bg-emerald-400 animate-pulse" : "bg-[#64748b]"}`} />
            {simRunning ? (simPaused ? "Paused" : "Simulation Running") : "Stopped"}
          </span>
          {simRunning && (
            <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-mono">
              Step: {simStep.toLocaleString()}
            </span>
          )}
        </div>
      </div>

      {/* Profile selector */}
      <div className="flex flex-wrap gap-2 items-center">
        {PROFILES.map(p => (
          <button key={p} onClick={() => { if (!simRunning) setSelectedProfile(p); }}
            disabled={simRunning}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
              selectedProfile === p ? "bg-indigo-500 border-indigo-400 text-white"
              : simRunning ? "bg-white/5 border-white/10 text-[#64748b] cursor-not-allowed"
              : "bg-white/5 border-white/10 text-[#94a3b8] hover:bg-white/10"}`}>
            {PROFILE_LABELS[p]}
          </button>
        ))}
        {simRunning && <span className="text-xs text-[#64748b]">← Stop to change profile</span>}
      </div>

      {error    && <div className="px-4 py-3 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-sm">⚠ {error}</div>}
      {simError && <div className="px-4 py-3 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-sm">⚠ {simError}</div>}

      {/* Metric cards */}
      {!loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="Vehicles"
            value={simRunning && liveData ? String(liveData.vehicles) : String(metrics?.peak_vehicles ?? "—")}
            unit={simRunning ? "live" : "peak"} color="#818cf8"
            data={timeseries.slice(-7).map(d => d.vehicles)} />
          <MetricCard label="Avg Wait"
            value={simRunning && liveData ? String(liveData.avg_wait_s) : String(metrics?.avg_wait_s ?? "—")}
            unit="seconds" color="#34d399"
            data={timeseries.slice(-7).map(d => d.wait)} />
          <MetricCard label="CO₂ Emissions"
            value={simRunning && liveData ? (liveData.total_co2_mg/1000).toFixed(0) : metrics ? (metrics.peak_co2_mg/1000).toFixed(0) : "—"}
            unit="k mg/step" color="#fbbf24"
            data={timeseries.slice(-7).map(d => d.co2)} />
          <MetricCard label="Max Wait"
            value={simRunning && liveData ? String(liveData.max_wait_s) : String(metrics?.max_wait_s ?? "—")}
            unit="seconds" color="#a78bfa"
            data={timeseries.slice(-7).map(d => d.wait)} />
        </div>
      )}

      {/* Main content */}
      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        <div className="card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Gauge className="w-5 h-5 text-indigo-400" />
              <h2 className="text-sm font-semibold text-white">Live Simulation Feed</h2>
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

          {/* Buttons */}
          <div className="flex gap-2">
            <button onClick={handleStart} disabled={simRunning || simLoading}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                simRunning || simLoading ? "bg-indigo-500/30 text-indigo-300 cursor-not-allowed"
                : "bg-indigo-500 hover:bg-indigo-400 text-white"}`}>
              <Play className="w-4 h-4" strokeWidth={2.5} />
              {simLoading ? "Starting..." : "Start"}
            </button>
            <button onClick={handleStop} disabled={!simRunning}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                !simRunning ? "bg-red-500/10 text-red-300/40 border-red-500/20 cursor-not-allowed"
                : "bg-red-500/20 hover:bg-red-500/30 text-red-400 border-red-500/40"}`}>
              <Square className="w-4 h-4" strokeWidth={2.5} /> Stop
            </button>
            <button onClick={handlePause} disabled={!simRunning}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                !simRunning ? "bg-white/5 text-[#94a3b8]/40 border-white/10 cursor-not-allowed"
                : simPaused ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-white/5 hover:bg-white/10 text-[#94a3b8] border-white/10"}`}>
              <Pause className="w-4 h-4" strokeWidth={2.5} />
              {simPaused ? "Resume" : "Pause"}
            </button>
          </div>

          {/* Viewport */}
          <div className="relative h-72 md:h-80 rounded-xl overflow-hidden bg-[#0a0f16]">
            {frame ? (
              <>
                <img src={frame} alt="SUMO simulation" className="w-full h-full object-cover" />
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
                      <p className="text-xs text-[#64748b] mt-1">First frame incoming</p>
                    </>
                  ) : (
                    <>
                      <p className="text-base font-semibold text-indigo-300 mb-1">SUMO Simulation</p>
                      <p className="text-sm text-[#94a3b8] mb-3">El-Tahrir Square — Cairo</p>
                      <p className="text-xs text-[#64748b]">
                        Select a profile and press <span className="text-indigo-400">Start</span>
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Stats bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-[#94a3b8]">
            <span>Profile: <span className="text-indigo-400">{PROFILE_LABELS[selectedProfile]}</span></span>
            {simRunning && liveData ? (
              <>
                <span>Vehicles: <span className="text-indigo-400">{liveData.vehicles}</span></span>
                <span>CO₂: <span className="text-indigo-400">{(liveData.total_co2_mg/1000).toFixed(0)}k mg</span></span>
                <span>Step: <span className="text-indigo-400">{simStep.toLocaleString()}</span></span>
              </>
            ) : metrics && (
              <>
                <span>Avg Vehicles: <span className="text-indigo-400">{metrics.avg_vehicles}</span></span>
                <span>Steps: <span className="text-indigo-400">{metrics.total_steps}</span></span>
                <span>Period: <span className="text-indigo-400">{metrics.time_period}</span></span>
              </>
            )}
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-4">
          <TrafficLightStatus
          trafficLights={liveData?.traffic_lights}
          typeCounts={liveData?.type_counts}
          />
          <div className="card p-5">
            <h3 className="text-sm font-semibold text-white mb-4">DQN Optimizer</h3>
            <div className="space-y-3 text-sm">
              {[
                { label: "Status",      value: "In Development", cls: "bg-amber-500/15 text-amber-300 rounded px-2 py-0.5" },
                { label: "Model",       value: "Deep Q-Network",  cls: "text-indigo-400" },
                { label: "Environment", value: "SUMO + TraCI",    cls: "text-indigo-400" },
              ].map(row => (
                <div key={row.label} className="flex items-center justify-between">
                  <span className="text-[#94a3b8]">{row.label}</span>
                  <span className={`font-mono text-xs ${row.cls}`}>{row.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        {[
          { key: "co2",  label: "CO₂ Emissions",      unit: "×1000 mg/step", color: "#fbbf24", fmt: v => [`${v}k mg`, "CO₂"] },
          { key: "wait", label: "Average Waiting Time", unit: "seconds",      color: "#34d399", fmt: v => [`${v}s`, "Avg Wait"] },
        ].map(chart => (
          <div key={chart.key} className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-white">{chart.label}</h3>
              <span className="text-xs font-mono text-[#64748b]">{simRunning ? "● Live · " : ""}{chart.unit}</span>
            </div>
            <div className="h-56">
              {loading ? (
                <div className="h-full flex items-center justify-center text-sm text-[#64748b] animate-pulse">Loading...</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} tickMargin={8} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} tickMargin={8} />
                    <Tooltip contentStyle={chartStyle.tooltip} formatter={chart.fmt} />
                    <Line type="monotone" dataKey={chart.key} stroke={chart.color} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Fullscreen overlay */}
      {fullscreen && frame && (
        <div className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center" onClick={() => setFullscreen(false)}>
          <img src={frame} alt="SUMO fullscreen" className="max-w-full max-h-full rounded-xl" />
          <button className="absolute top-4 right-4 px-3 py-1.5 rounded-lg bg-white/10 text-white text-sm hover:bg-white/20"
            onClick={() => setFullscreen(false)}>✕ Close</button>
        </div>
      )}
    </div>
  );
}

export default Dashboard;