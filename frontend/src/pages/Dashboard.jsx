import React, { useState, useEffect, useRef, useCallback } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Play, Square, Pause, Maximize2 } from "lucide-react";
import MetricCard from "../components/MetricCard";
import TrafficLightStatus from "../components/TrafficLightStatus";
import { useSimMode, VIDEO_MAP, API, STATIC_METRICS, STATIC_SIGNALS, STATIC_DIRECTIONS } from "../hooks/useSimMode";
import { useIsMobile, useIsTablet, gridCols } from "../hooks/useIsMobile";

const TT   = { backgroundColor:"#fff", border:"1px solid #e2e8f0", borderRadius:"4px", fontSize:"12px", color:"#1e293b" };
const TICK = { fill:"#94a3b8", fontSize:11 };
const card   = { background:"#fff", border:"1px solid #e2e8f0", borderRadius:"6px", boxShadow:"0 1px 3px rgba(0,0,0,.04)" };
const slabel = { fontSize:"11px", fontWeight:600, letterSpacing:"0.06em", textTransform:"uppercase", color:"#64748b" };
const btnS   = (v, dis) => ({
  display:"inline-flex", alignItems:"center", gap:"6px",
  padding:"6px 14px", fontSize:"13px", fontWeight:500,
  borderRadius:"4px", border:"1px solid",
  cursor: dis?"not-allowed":"pointer", opacity: dis?0.45:1, transition:"all 0.15s",
  ...(v==="primary" ? { background:"#1d4ed8", color:"#fff", borderColor:"#1d4ed8" }
    : v==="danger"  ? { background:"#fff", color:"#dc2626", borderColor:"#fca5a5" }
    :                 { background:"#fff", color:"#374151", borderColor:"#d1d5db" }),
});

const PROFILES = ["morning_rush","evening_rush","midday","night"];
const PLABELS  = { morning_rush:"Morning Rush", evening_rush:"Evening Rush", midday:"Midday", night:"Night" };

export default function Dashboard() {
  const mode = useSimMode();
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const [sel,      setSel]      = useState("morning_rush");
  const [met,      setMet]      = useState(null);
  const [ts,       setTs]       = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [err,      setErr]      = useState(null);

  // Live mode state
  const [run,      setRun]      = useState(false);
  const [paused,   setPaused]   = useState(false);
  const [step,     setStep]     = useState(0);
  const [live,     setLive]     = useState(null);
  const [chart,    setChart]    = useState([]);
  const [simErr,   setSimErr]   = useState(null);
  const [simState, setSimState] = useState("idle");
  const [frame,    setFrame]    = useState(null);
  const [full,     setFull]     = useState(false);

  const poll = useRef(null);

  useEffect(() => {
    setLoading(true);
    if (mode === "video") {
      const d = STATIC_METRICS[sel];
      setMet(d);
      setTs(d.timeseries.filter((_,i)=>i%10===0).map(d=>({
        t:d.step, co2:Math.round(d.co2/1000), wait:d.avg_wait, vehicles:d.vehicles,
      })));
      setLoading(false);
      return;
    }
    fetch(`${API}/api/simulation/metrics/${sel}`)
      .then(r => r.json())
      .then(d => {
        setMet(d);
        setTs(d.timeseries.filter((_,i) => i%10===0).map(d => ({
          t:d.step, co2:Math.round(d.co2/1000), wait:d.avg_wait, vehicles:d.vehicles,
        })));
        setLoading(false);
      })
      .catch(() => {
        const d = STATIC_METRICS[sel];
        setMet(d);
        setTs(d.timeseries.filter((_,i)=>i%10===0).map(d=>({
          t:d.step, co2:Math.round(d.co2/1000), wait:d.avg_wait, vehicles:d.vehicles,
        })));
        setLoading(false);
      });
  }, [sel, mode]);

  const handleStop = useCallback(async () => {
    clearInterval(poll.current);
    setRun(false); setPaused(false); setLive(null);
    setFrame(null); setSimState("idle");
    try { await fetch(`${API}/api/sumo/stop`, { method:"POST" }); } catch {}
  }, []);

  useEffect(() => {
    if (run && !paused) {
      poll.current = setInterval(async () => {
        try {
          const res  = await fetch(`${API}/api/sumo/step`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({ steps:30 }),
          });
          const data = await res.json();
          if (data.latest) {
            const s = data.latest;
            setStep(s.step); setLive(s);
            setChart(p => [...p, {
              t:s.step, co2:Math.round(s.total_co2_mg/1000),
              wait:s.avg_wait_s, vehicles:s.vehicles,
            }].slice(-40));
            if (s.simulation_done) handleStop();
          }
          if (data.image) setFrame(data.image);
        } catch { setSimErr("Lost connection"); handleStop(); }
      }, 2000);
    }
    return () => clearInterval(poll.current);
  }, [run, paused, handleStop]);

  const handleStart = async () => {
    setSimErr(null); setSimState("starting");
    setChart([]); setFrame(null); setLive(null);
    try {
      const res  = await fetch(`${API}/api/sumo/start`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ profile:sel, gui:true }),
      });
      const data = await res.json();
      if (data.status==="started"||data.status==="already_running") {
        setRun(true); setPaused(false); setStep(0); setSimState("running");
        try {
          const fr = await fetch(`${API}/api/sumo/step`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({ steps:1 }),
          }).then(r=>r.json());
          if (fr.image) setFrame(fr.image);
        } catch {}
      } else { setSimErr(data.detail||"Failed"); setSimState("idle"); }
    } catch { setSimErr("Cannot reach backend"); setSimState("idle"); }
  };

  const chartData = run && chart.length>0 ? chart : ts;

  // ── Video mode viewport ──────────────────────────────────────
  const VideoViewport = () => (
    <div style={{ margin:"12px", height:isMobile?"200px":"320px", background:"#0f172a", borderRadius:"4px", overflow:"hidden", position:"relative" }}>
      <video
        key={sel}
        src={VIDEO_MAP[sel]}
        autoPlay loop muted playsInline
        style={{ width:"100%", height:"100%", objectFit:"cover" }}
      />
      <div style={{ position:"absolute", top:"10px", left:"10px", display:"flex", gap:"6px" }}>
        <span style={{ padding:"3px 8px", borderRadius:"99px", fontSize:"11px", fontWeight:600, fontFamily:"monospace", background:"rgba(0,0,0,0.75)", color:"#4ade80" }}>● SIMULATION</span>
        <span style={{ padding:"3px 8px", borderRadius:"99px", fontSize:"11px", fontFamily:"monospace", background:"rgba(0,0,0,0.75)", color:"#93c5fd" }}>{PLABELS[sel]}</span>
      </div>
    </div>
  );

  // ── Live mode viewport ───────────────────────────────────────
  const LiveViewport = () => (
    <div style={{ margin:"12px", height:isMobile?"200px":"320px", background:"#0f172a", borderRadius:"4px", border:"1px solid #e2e8f0", overflow:"hidden", position:"relative" }}>
      {frame ? (
        <>
          <img src={frame} alt="SUMO" style={{ width:"100%", height:"100%", objectFit:"cover" }}/>
          <div style={{ position:"absolute", top:"10px", left:"10px", display:"flex", gap:"6px" }}>
            <span style={{ padding:"3px 8px", borderRadius:"99px", fontSize:"11px", fontWeight:600, fontFamily:"monospace", background:"rgba(0,0,0,0.75)", color:paused?"#fbbf24":"#4ade80" }}>
              {paused?"⏸ PAUSED":"● LIVE"}
            </span>
            <span style={{ padding:"3px 8px", borderRadius:"99px", fontSize:"11px", fontFamily:"monospace", background:"rgba(0,0,0,0.75)", color:"#93c5fd" }}>Step {step.toLocaleString()}</span>
          </div>
          {live && (
            <div style={{ position:"absolute", bottom:"10px", left:"10px", right:"10px", display:"flex", justifyContent:"space-between" }}>
              {[`🚗 ${live.vehicles}`,`⏱ ${live.avg_wait_s}s`,`💨 ${(live.total_co2_mg/1000).toFixed(0)}k mg CO₂`].map(t => (
                <span key={t} style={{ padding:"3px 8px", borderRadius:"99px", fontSize:"11px", fontFamily:"monospace", background:"rgba(0,0,0,0.75)", color:"#fff" }}>{t}</span>
              ))}
            </div>
          )}
          {frame && <button style={{ position:"absolute", top:"10px", right:"10px", padding:"4px 8px", background:"rgba(0,0,0,0.6)", color:"#fff", border:"none", borderRadius:"99px", cursor:"pointer" }} onClick={()=>setFull(true)}><Maximize2 size={12}/></button>}
        </>
      ) : (
        <div style={{ height:"100%", display:"flex", alignItems:"center", justifyContent:"center", flexDirection:"column", gap:"8px" }}>
          {simState==="starting"
            ? <><div style={{ width:"28px", height:"28px", border:"2px solid #3b82f6", borderTopColor:"transparent", borderRadius:"50%", animation:"spin 0.8s linear infinite" }}/><span style={{ color:"#60a5fa", fontSize:"13px" }}>Opening SUMO-GUI...</span></>
            : <><span style={{ fontSize:"28px" }}></span><span style={{ color:"#64748b", fontSize:"13px" }}>Select a profile and press <strong style={{ color:"#1d4ed8" }}>Start</strong></span></>
          }
        </div>
      )}
    </div>
  );

  if (mode === "checking") return (
    <div style={{ paddingTop:"48px", textAlign:"center", color:"#64748b", fontSize:"14px" }}>
      <div style={{ width:"24px", height:"24px", border:"2px solid #3b82f6", borderTopColor:"transparent", borderRadius:"50%", animation:"spin 0.8s linear infinite", margin:"0 auto 12px" }}/>
      Connecting...
    </div>
  );

  return (
    <div style={{ paddingTop:"24px" }}>
      {/* Header */}
      <div style={{ paddingBottom:"16px", borderBottom:"1px solid #e2e8f0", marginBottom:"24px", display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
        <div>
          <h1 style={{ margin:0, fontSize:"20px", fontWeight:700, color:"#0f172a" }}>Dashboard</h1>
          <p style={{ margin:"4px 0 0", fontSize:"13px", color:"#64748b" }}>El-Tahrir Square — Cairo, Egypt</p>
        </div>
        <div style={{ display:"flex", gap:"8px", alignItems:"center" }}>
          <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"12px", fontWeight:500,
            background: mode==="live" ? "#dbeafe" : "#f1f5f9",
            color: mode==="live" ? "#1d4ed8" : "#64748b",
            border: `1px solid ${mode==="live"?"#bfdbfe":"#e2e8f0"}` }}>
            {mode==="live" ? " Live Mode" : "▶ Video Mode"}
          </span>
          {mode==="live" && run && <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"12px", fontFamily:"monospace", background:"#dbeafe", border:"1px solid #bfdbfe", color:"#1d4ed8" }}>Step {step.toLocaleString()}</span>}
        </div>
      </div>

      {/* Profile selector */}
      <div style={{ display:"flex", gap:"6px", marginBottom:"20px", flexWrap:"wrap", alignItems:"center" }}>
        <span style={{ ...slabel, marginRight:"4px" }}>Profile:</span>
        {PROFILES.map(p => (
          <button key={p}
            onClick={() => { if(mode==="video"||(!run&&simState==="idle")) setSel(p); }}
            disabled={mode==="live"&&(run||simState==="starting")}
            style={{ padding:"5px 14px", fontSize:"13px", fontWeight:500, borderRadius:"4px", border:"1px solid",
              cursor: mode==="live"&&(run||simState==="starting") ? "not-allowed" : "pointer",
              background:sel===p?"#1d4ed8":"#fff", color:sel===p?"#fff":"#374151",
              borderColor:sel===p?"#1d4ed8":"#d1d5db" }}>
            {PLABELS[p]}
          </button>
        ))}
      </div>

      {simErr && <div style={{ padding:"8px 12px", background:"#fee2e2", border:"1px solid #fca5a5", borderRadius:"4px", color:"#dc2626", fontSize:"13px", marginBottom:"16px" }}>⚠ {simErr}</div>}

      {/* KPI cards */}
      {!loading && met && (
        <div style={{ display:"grid", gridTemplateColumns:gridCols(4,2,2,isMobile,isTablet), gap:"12px", marginBottom:"20px" }}>
          <MetricCard label="Vehicles"      value={run&&live?live.vehicles:met.peak_vehicles}                              unit={run?"live count":"peak count"} accent="#1d4ed8"/>
          <MetricCard label="Avg Wait Time" value={run&&live?live.avg_wait_s:met.avg_wait_s}                               unit="seconds"                      accent="#15803d"/>
          <MetricCard label="CO₂ Emissions" value={run&&live?(live.total_co2_mg/1000).toFixed(0):(met.peak_co2_mg/1000).toFixed(0)} unit="×1,000 mg/step"    accent="#b45309"/>
          <MetricCard label="Max Wait Time" value={run&&live?live.max_wait_s:met.max_wait_s}                               unit="seconds"                      accent="#7c3aed"/>
        </div>
      )}

      {/* Main grid */}
      <div style={{ display:"grid", gridTemplateColumns:isMobile||isTablet?"1fr":"1fr 300px", gap:"16px", marginBottom:"20px" }}>
        <div style={card}>
          <div style={{ padding:"12px 16px", borderBottom:"1px solid #f1f5f9", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <span style={slabel}>
              {mode==="video" ? "Simulation Recording" : "Live Simulation Feed"}
            </span>
            {mode==="live" && (
              <div style={{ display:"flex", gap:"6px" }}>
                <button style={btnS("primary", run||simState==="starting")} onClick={handleStart} disabled={run||simState==="starting"}>
                  <Play size={13} strokeWidth={2.5}/>{simState==="starting"?"Starting...":"Start"}
                </button>
                <button style={btnS("danger", !run)} onClick={handleStop} disabled={!run}>
                  <Square size={13} strokeWidth={2.5}/> Stop
                </button>
                <button style={btnS("neutral", !run)} onClick={()=>setPaused(p=>!p)} disabled={!run}>
                  <Pause size={13} strokeWidth={2.5}/>{paused?"Resume":"Pause"}
                </button>
              </div>
            )}
          </div>

          {mode==="video" ? <VideoViewport/> : <LiveViewport/>}

          <div style={{ padding:"8px 16px", borderTop:"1px solid #f1f5f9", display:"flex", gap:"16px", fontSize:"12px", color:"#64748b" }}>
            <span>Profile: <strong style={{ color:"#1d4ed8" }}>{PLABELS[sel]}</strong></span>
            {met && <><span>Avg vehicles: <strong>{met.avg_vehicles}</strong></span><span>Steps: <strong>{met.total_steps}</strong></span></>}
          </div>
        </div>

        {/* Right column */}
        <div style={{ display:"flex", flexDirection:"column", gap:"12px" }}>
          <TrafficLightStatus trafficLights={mode==="video" ? STATIC_SIGNALS : live?.traffic_lights} typeCounts={mode==="video" ? {"car":280,"taxi":76,"microbus":61,"bus":40,"truck":25,"motorcycle":20,"bicycle":5} : live?.type_counts}/>
          <div style={{ ...card, padding:"16px" }}>
            <div style={{ ...slabel, marginBottom:"12px" }}>Profile Summary</div>
            {met && [
              { label:"Time period",   value: met.time_period },
              { label:"Avg vehicles",  value: met.avg_vehicles },
              { label:"Peak vehicles", value: met.peak_vehicles },
              { label:"Avg wait",      value: `${met.avg_wait_s}s` },
              { label:"Max wait",      value: `${met.max_wait_s}s` },
            ].map(row => (
              <div key={row.label} style={{ display:"flex", justifyContent:"space-between", padding:"7px 0", borderBottom:"1px solid #f1f5f9" }}>
                <span style={{ fontSize:"13px", color:"#64748b" }}>{row.label}</span>
                <span style={{ fontSize:"13px", fontWeight:500, color:"#1e293b" }}>{row.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div style={{ display:"grid", gridTemplateColumns:isMobile?"1fr":"1fr 1fr", gap:"16px" }}>
        {[
          { key:"co2",  label:"CO₂ Emissions",       unit:"×1,000 mg/step", color:"#b45309" },
          { key:"wait", label:"Average Waiting Time", unit:"seconds",        color:"#15803d" },
        ].map(c => (
          <div key={c.key} style={{ ...card, padding:"16px" }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"14px" }}>
              <span style={{ fontSize:"14px", fontWeight:600, color:"#0f172a" }}>{c.label}</span>
              <span style={{ fontSize:"11px", color:"#94a3b8", fontFamily:"monospace" }}>{c.unit}</span>
            </div>
            <div style={{ height:"200px" }}>
              {loading
                ? <div style={{ height:"100%", display:"flex", alignItems:"center", justifyContent:"center", color:"#94a3b8" }}>Loading...</div>
                : <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ left:-20, right:8, top:4, bottom:0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="t" tick={TICK} tickLine={false}/>
                      <YAxis tick={TICK} tickLine={false}/>
                      <Tooltip contentStyle={TT} formatter={v=>[c.key==="co2"?`${v}k mg`:`${v}s`,c.label]}/>
                      <Line type="monotone" dataKey={c.key} stroke={c.color} strokeWidth={2} dot={false} activeDot={{r:3}}/>
                    </LineChart>
                  </ResponsiveContainer>
              }
            </div>
          </div>
        ))}
      </div>

      {full && frame && (
        <div style={{ position:"fixed", inset:0, zIndex:100, background:"rgba(0,0,0,0.95)", display:"flex", alignItems:"center", justifyContent:"center" }} onClick={()=>setFull(false)}>
          <img src={frame} alt="fs" style={{ maxWidth:"100%", maxHeight:"100%" }}/>
          <button style={{ position:"absolute", top:"16px", right:"16px", padding:"6px 12px", background:"rgba(255,255,255,0.15)", color:"#fff", border:"none", borderRadius:"4px", cursor:"pointer" }} onClick={()=>setFull(false)}>✕ Close</button>
        </div>
      )}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}