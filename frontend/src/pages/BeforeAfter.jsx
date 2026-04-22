import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, Tooltip, CartesianGrid,
  BarChart, Bar,
} from "recharts";
import { Brain, Play, Square, Zap } from "lucide-react";

const API = "http://127.0.0.1:8000";

const chartStyle = {
  tooltip: {
    backgroundColor: "#161c24",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "10px",
    fontSize: "12px",
  },
  tick: { fill: "#64748b", fontSize: 11 },
};

const PROFILES = [
  { key: "morning_rush", label: "Morning Rush", icon: "🌅", time: "8:00–11:00 AM" },
  { key: "evening_rush", label: "Evening Rush", icon: "🌆", time: "4:00–7:00 PM" },
  { key: "midday",       label: "Midday",       icon: "☀️", time: "12:00–3:00 PM" },
  { key: "night",        label: "Night",        icon: "🌙", time: "10:00 PM–1:00 AM" },
];

const PROFILE_LABELS = {
  morning_rush: "Morning Rush",
  evening_rush: "Evening Rush",
  midday:       "Midday",
  night:        "Night",
};

function BeforeAfter() {
  const [results,       setResults]       = useState(null);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState(null);
  const [simRunning,    setSimRunning]     = useState(false);
  const [simProfile,    setSimProfile]     = useState(null);
  const [simMetrics,    setSimMetrics]     = useState(null);
  const [simFrame,      setSimFrame]       = useState(null);
  const [startingProfile, setStartingProfile] = useState(null);
  const [qValues,       setQValues]        = useState(null);  
  const [fallbackMode,  setFallbackMode]   = useState(false); 

  const frameRef    = useRef(null);
  const pollRef     = useRef(null);

  // Load DQN results
  useEffect(() => {
    fetch(`${API}/api/dqn/results`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { setResults(data); setLoading(false); })
      .catch(() => { setError("DQN not trained yet"); setLoading(false); });
  }, []);

  // Poll simulation status + screenshot
  const pollSim = useCallback(async () => {
    try {
      const status = await fetch(`${API}/api/dqn/sim/status`).then(r => r.json());
      setSimRunning(status.running);
      if (status.running) {
        setSimMetrics(status.metrics);
        // extract Q-values and fallback mode
        if (status.metrics?.q_values) {
          setQValues(status.metrics.q_values);
        }
        setFallbackMode(status.metrics?.fallback_mode || false);
        const imgRes = await fetch(`${API}/api/dqn/sim/screenshot`);
        if (imgRes.ok) {
          const blob = await imgRes.blob();
          const url  = URL.createObjectURL(blob);
          setSimFrame(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
        }
      }
    } catch (e) {}
  }, []);

  useEffect(() => {
    if (simRunning) {
      pollRef.current = setInterval(pollSim, 500);
    } else {
      clearInterval(pollRef.current);
    }
    return () => clearInterval(pollRef.current);
  }, [simRunning, pollSim]);

  const handleStart = async (profileKey) => {
    setStartingProfile(profileKey);
    try {
      // Stop existing if running
      if (simRunning) {
        await fetch(`${API}/api/dqn/sim/stop`, { method: "POST" });
        await new Promise(r => setTimeout(r, 1500));
      }
      const res = await fetch(`${API}/api/dqn/sim/start/${profileKey}`, { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail);
      setSimProfile(profileKey);
      setSimRunning(true);
      setSimMetrics(null);
      setSimFrame(null);
    } catch (e) {
      alert(`Failed to start: ${e.message}`);
    } finally {
      setStartingProfile(null);
    }
  };

  const handleStop = async () => {
    await fetch(`${API}/api/dqn/sim/stop`, { method: "POST" });
    setSimRunning(false);
    setSimMetrics(null);
  };

  // Chart data
  const waitBefore      = results?.avg_wait_fixed      || 0;
  const waitAfter       = results?.avg_wait_dqn        || 0;
  const waitImprovement = results?.improvement_pct     || 0;
  const co2Improvement  = results?.co2_improvement_pct || 0;
  const fixedCo2        = results?.fixed_co2_mg        || 0;
  const dqnCo2          = results?.dqn_co2_mg          || 0;

  const episodeData = results?.episode_waits
    ? results.episode_waits
        .filter((_, i) => i % 2 === 0)
        .map((w, i) => ({ episode: (i * 2) + 1, wait: parseFloat(w.toFixed(2)) }))
    : [];

  const waitCompData = [
    { name: "Fixed Timing",  wait: parseFloat(waitBefore.toFixed(2)) },
    { name: "DQN Optimizer", wait: parseFloat(waitAfter.toFixed(2))  },
  ];

  const co2CompData = [
    { name: "Fixed Timing",  co2: parseFloat(fixedCo2.toFixed(1)) },
    { name: "DQN Optimizer", co2: parseFloat(dqnCo2.toFixed(1))   },
  ];

  // Profile results from comparison
  const profileData = results?.profiles || {};
  const activeProfile = PROFILES.find(p => p.key === simProfile);

  return (
    <div className="space-y-6 pt-4">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-white">Before vs After</h1>
          <p className="text-sm text-[#94a3b8] mt-0.5">
            Fixed timing vs DQN AI-adaptive signal control — El-Tahrir Square
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="px-3 py-1.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-medium">
            Before: Fixed Timing
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
            After: DQN Optimizer
          </span>
          {results?.episodes && (
            <span className="px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-xs font-mono">
              {results.episodes} episodes trained
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="px-4 py-3 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 text-sm">
          ⚠ {error} — run <code>python dqn/train_dqn.py</code> first.
        </div>
      )}

      {/* Comparison Cards */}
      <div className="grid md:grid-cols-3 gap-4 items-stretch">
        <div className="card border-red-500/30 p-5">
          <h3 className="text-sm font-semibold text-red-400 mb-4">⚠ Before — Fixed Timing</h3>
          <div className="space-y-3">
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">Avg Wait Time</div>
              <div className="font-mono text-red-400 font-semibold text-lg">{results ? `${waitBefore.toFixed(2)}s` : "—"}</div>
            </div>
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">CO₂ per Step</div>
              <div className="font-mono text-amber-400 font-semibold">{results ? `${fixedCo2.toFixed(1)} mg` : "—"}</div>
            </div>
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">Signal Strategy</div>
              <div className="font-mono text-[#94a3b8] text-sm">Fixed 39s cycles</div>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center gap-3">
          <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-[#64748b] text-sm font-semibold">VS</div>
          {results && waitImprovement > 0 && (
            <div className="text-center space-y-1">
              <div className="text-3xl font-bold text-emerald-400">↓{waitImprovement.toFixed(1)}%</div>
              <div className="text-xs text-[#94a3b8]">Wait time reduction</div>
              <div className="text-2xl font-bold text-emerald-400">↓{co2Improvement.toFixed(1)}%</div>
              <div className="text-xs text-[#94a3b8]">CO₂ reduction</div>
            </div>
          )}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-500/15 border border-indigo-500/30">
            <Brain className="w-4 h-4 text-indigo-400" />
            <span className="text-xs text-indigo-300 font-medium">DQN Agent</span>
          </div>
        </div>

        <div className="card border-emerald-500/30 p-5">
          <h3 className="text-sm font-semibold text-emerald-400 mb-4">✓ After — DQN Optimizer</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs text-[#94a3b8] mb-0.5">Avg Wait Time</div>
                <div className="font-mono text-emerald-400 font-semibold text-lg">{results ? `${waitAfter.toFixed(2)}s` : "—"}</div>
              </div>
              {results && <span className="px-2 py-0.5 rounded-lg bg-emerald-500/20 text-emerald-300 text-[11px] font-mono">↓{waitImprovement.toFixed(1)}%</span>}
            </div>
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs text-[#94a3b8] mb-0.5">CO₂ per Step</div>
                <div className="font-mono text-emerald-400 font-semibold">{results ? `${dqnCo2.toFixed(1)} mg` : "—"}</div>
              </div>
              {results && <span className="px-2 py-0.5 rounded-lg bg-emerald-500/20 text-emerald-300 text-[11px] font-mono">↓{co2Improvement.toFixed(1)}%</span>}
            </div>
            <div>
              <div className="text-xs text-[#94a3b8] mb-0.5">Episodes Trained</div>
              <div className="font-mono text-emerald-400 text-sm">{results?.episodes || "—"} episodes</div>
            </div>
          </div>
        </div>
      </div>

      {/* ── DQN Live Simulation ───────────────────────────────── */}
      <div className="card p-5 border-indigo-500/20">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-indigo-400" />
            <h2 className="text-sm font-semibold text-white">DQN Live Simulation</h2>
            <span className="text-xs text-[#64748b]">— watch DQN control traffic lights in real time</span>
          </div>
          {simRunning && (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 border border-red-500/30 text-red-300 text-xs font-medium hover:bg-red-500/30 transition-colors"
            >
              <Square className="w-3 h-3" /> Stop
            </button>
          )}
        </div>

        {/* Profile selector */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          {PROFILES.map(p => {
            const pd       = profileData[p.key];
            const isActive = simProfile === p.key && simRunning;
            const starting = startingProfile === p.key;
            return (
              <button
                key={p.key}
                onClick={() => handleStart(p.key)}
                disabled={starting || (simRunning && simProfile !== p.key && startingProfile !== null)}
                className={`flex flex-col items-start gap-1 px-3 py-3 rounded-xl border transition-all text-left
                  ${isActive
                    ? "bg-indigo-500/20 border-indigo-500/50 text-white"
                    : "bg-white/5 border-white/10 text-[#94a3b8] hover:bg-white/10 hover:border-white/20"
                  }`}
              >
                <div className="flex items-center gap-2 w-full">
                  <span className="text-base">{p.icon}</span>
                  <span className="text-xs font-semibold truncate">{p.label}</span>
                  {isActive && <span className="ml-auto w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />}
                  {starting && <span className="ml-auto text-[10px] text-indigo-300 animate-pulse">starting...</span>}
                </div>
                <span className="text-[10px] text-[#64748b]">{p.time}</span>
                {pd && (
                  <div className="flex gap-1 mt-1">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">↓{pd.wait_improvement.toFixed(0)}% wait</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-300">↓{pd.co2_improvement.toFixed(0)}% CO₂</span>
                  </div>
                )}
                {!isActive && !starting && (
                  <div className="flex items-center gap-1 mt-1 text-[10px] text-indigo-400">
                    <Play className="w-2.5 h-2.5" /> Click to run DQN
                  </div>
                )}
              </button>
            );
          })}
        </div>

        {/* Simulation viewport */}
        <div className="grid lg:grid-cols-3 gap-4">
          {/* Stream */}
          <div className="lg:col-span-2">
            <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-[#0a0f1a] border border-white/10 flex items-center justify-center">
              {simFrame && simRunning ? (
                <img src={simFrame} alt="DQN simulation" className="w-full h-full object-cover" />
              ) : (
                <div className="text-center space-y-2">
                  <Brain className="w-10 h-10 text-indigo-500/40 mx-auto" />
                  <p className="text-sm text-[#64748b]">
                    {simRunning ? "Loading simulation..." : "Select a profile and click to start DQN simulation"}
                  </p>
                </div>
              )}
              {simRunning && (
                <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2 py-1 rounded-lg bg-black/60 backdrop-blur">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs text-emerald-300 font-mono">DQN ACTIVE</span>
                </div>
              )}
              {simRunning && activeProfile && (
                <div className="absolute top-3 right-3 px-2 py-1 rounded-lg bg-black/60 backdrop-blur">
                  <span className="text-xs text-white font-mono">{activeProfile.icon} {activeProfile.label}</span>
                </div>
              )}
              {simMetrics?.current_action && (
                <div className="absolute bottom-3 left-3 px-2 py-1 rounded-lg bg-indigo-500/80 backdrop-blur">
                  <span className="text-xs text-white font-mono">⚡ {simMetrics.current_action}</span>
                </div>
              )}
            </div>
          </div>

          {/* Live metrics */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-[#94a3b8] mb-2">
              {simRunning ? "Live Metrics" : "Profile Results"}
            </div>

            {simRunning && simMetrics ? (
              // Live metrics while running
              <>
                {[
                  { label: "Vehicles",      value: simMetrics.vehicles,                    unit: "" },
                  { label: "Avg Wait",      value: `${simMetrics.avg_wait_s}`,             unit: "s" },
                  { label: "Avg Speed",     value: `${simMetrics.avg_speed}`,              unit: "m/s" },
                  { label: "Total CO₂",     value: `${(simMetrics.total_co2_mg/1000).toFixed(1)}`, unit: "k mg" },
                  { label: "Sim Step",      value: simMetrics.step,                        unit: "" },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                    <span className="text-xs text-[#94a3b8]">{m.label}</span>
                    <span className="font-mono text-white text-xs">{m.value}{m.unit}</span>
                  </div>
                ))}
                <div className="px-3 py-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
                  <div className="text-[10px] text-[#94a3b8] mb-0.5">Current Action</div>
                  <div className="font-mono text-indigo-300 text-[11px]">{simMetrics.current_action}</div>
                </div>
                {/* ── Q-Value Visualization — WHY DQN chose this action ── */}
                {qValues && (
                  <div className="px-3 py-3 rounded-lg bg-white/5 border border-white/5 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-[10px] font-semibold text-[#94a3b8]">
                        DQN Q-Values
                      </div>
                      <div className="text-[10px] text-[#64748b]">
                        higher = preferred
                      </div>
                    </div>
                    {Object.entries(qValues).map(([actionName, qVal]) => {
                      const isChosen = actionName === simMetrics.current_action;
                      // Normalize bars: shift all values so min=0
                      const allVals  = Object.values(qValues);
                      const minVal   = Math.min(...allVals);
                      const maxVal   = Math.max(...allVals);
                      const range    = maxVal - minVal || 1;
                      const pct      = Math.round(((qVal - minVal) / range) * 100);
                      return (
                        <div key={actionName}>
                          <div className="flex items-center justify-between text-[10px] mb-1">
                            <span className={isChosen ? "text-indigo-300 font-semibold" : "text-[#94a3b8]"}>
                              {isChosen ? "⚡ " : ""}{actionName}
                            </span>
                            <span className={`font-mono ${isChosen ? "text-indigo-300" : "text-[#64748b]"}`}>
                              {qVal.toFixed(4)}
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width:           `${pct}%`,
                                backgroundColor: isChosen ? "#818cf8" : "#334155",
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                    <div className="text-[10px] text-[#64748b] pt-1 border-t border-white/5">
                      DQN picks action with highest Q-value
                    </div>
                  </div>
                )}

                {/* Fallback mode warning */}
                {fallbackMode && (
                  <div className="px-3 py-2 rounded-lg bg-amber-500/15 border border-amber-500/30">
                    <div className="text-[10px] text-amber-400 font-semibold">
                      ⚠ Fallback Mode Active
                    </div>
                    <div className="text-[10px] text-[#94a3b8] mt-0.5">
                      DQN inference failed — using fixed 39s timing
                    </div>
                  </div>
                )}
              </>
            ) : simProfile && profileData[simProfile] ? (
              // Profile comparison results
              <>
                {[
                  { label: "Fixed Wait",      value: `${profileData[simProfile].fixed_wait_s.toFixed(2)}s`,  color: "text-red-400" },
                  { label: "DQN Wait",        value: `${profileData[simProfile].dqn_wait_s.toFixed(2)}s`,    color: "text-emerald-400" },
                  { label: "Wait Reduction",  value: `${profileData[simProfile].wait_improvement.toFixed(1)}%`, color: "text-emerald-400" },
                  { label: "Fixed CO₂",       value: `${profileData[simProfile].fixed_co2_mg.toFixed(0)} mg`, color: "text-red-400" },
                  { label: "DQN CO₂",         value: `${profileData[simProfile].dqn_co2_mg.toFixed(0)} mg`,   color: "text-emerald-400" },
                  { label: "CO₂ Reduction",   value: `${profileData[simProfile].co2_improvement.toFixed(1)}%`, color: "text-emerald-400" },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                    <span className="text-xs text-[#94a3b8]">{m.label}</span>
                    <span className={`font-mono text-xs ${m.color}`}>{m.value}</span>
                  </div>
                ))}
              </>
            ) : (
              // No profile selected
              <div className="text-xs text-[#64748b] text-center py-8">
                Select a profile to see results
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">

        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Avg Wait Time Comparison</h3>
            <span className="text-xs font-mono text-[#64748b]">seconds</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={waitCompData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="name" tick={chartStyle.tick} tickLine={false} />
                <YAxis tick={chartStyle.tick} tickLine={false} />
                <Tooltip contentStyle={chartStyle.tooltip} formatter={v => [`${v}s`, "Avg Wait"]} />
                <Bar dataKey="wait" radius={[6, 6, 0, 0]} fill="#818cf8"
                  label={{ position: "top", fill: "#94a3b8", fontSize: 11, formatter: v => `${v}s` }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">CO₂ Emissions Comparison</h3>
            <span className="text-xs font-mono text-[#64748b]">mg/step</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={co2CompData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="name" tick={chartStyle.tick} tickLine={false} />
                <YAxis tick={chartStyle.tick} tickLine={false} />
                <Tooltip contentStyle={chartStyle.tooltip} formatter={v => [`${v} mg`, "CO₂"]} />
                <Bar dataKey="co2" radius={[6, 6, 0, 0]} fill="#34d399"
                  label={{ position: "top", fill: "#94a3b8", fontSize: 11, formatter: v => `${v}` }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">DQN Learning Curve</h3>
            <span className="text-xs font-mono text-[#64748b]">avg wait per episode</span>
          </div>
          <div className="h-56">
            {!results || episodeData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm text-[#64748b]">
                {loading ? "Loading..." : "Train DQN to see learning curve"}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={episodeData} margin={{ left: -20, right: 8, top: 5, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="episode" tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                  <YAxis tick={chartStyle.tick} tickLine={false} tickMargin={8} />
                  <Tooltip contentStyle={chartStyle.tooltip} formatter={v => [`${v}s`, "Avg Wait"]} />
                  <Line type="monotone" dataKey="wait" stroke="#34d399" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {results && (
          <div className="card p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Training Details</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Episodes",         value: results.episodes },
                { label: "Profiles Trained", value: "4 profiles" },
                { label: "Fixed Avg Wait",   value: `${waitBefore.toFixed(2)}s` },
                { label: "DQN Avg Wait",     value: `${waitAfter.toFixed(2)}s` },
                { label: "Wait Improvement", value: `${waitImprovement.toFixed(1)}%` },
                { label: "CO₂ Improvement",  value: `${co2Improvement.toFixed(1)}%` },
              ].map(row => (
                <div key={row.label} className="flex flex-col gap-1 px-3 py-2.5 rounded-lg bg-white/5 border border-white/5">
                  <span className="text-[10px] text-[#94a3b8]">{row.label}</span>
                  <span className="font-mono text-white text-sm font-medium">{row.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default BeforeAfter;