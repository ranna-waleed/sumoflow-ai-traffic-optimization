import React, { useState, useEffect, useRef, useCallback } from "react";
import { Play, Square, Pause, Maximize2 } from "lucide-react";
import { useSimMode, VIDEO_MAP, API, STATIC_DIRECTIONS, STATIC_BILSTM, STATIC_METRICS } from "../hooks/useSimMode";
import { useIsMobile, useIsTablet, gridCols } from "../hooks/useIsMobile";

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

const PROFILES = [
  { key:"morning_rush", label:"Morning Rush", time:"07:30–10:30" },
  { key:"evening_rush", label:"Evening Rush", time:"15:00–20:00" },
  { key:"midday",       label:"Midday",       time:"12:00–15:00" },
  { key:"night",        label:"Night",        time:"22:00–24:00" },
];
const DIRS = [
  { key:"north", label:"North ↑", color:"#3b82f6" },
  { key:"south", label:"South ↓", color:"#16a34a" },
  { key:"east",  label:"East →",  color:"#f97316" },
  { key:"west",  label:"West ←",  color:"#7c3aed" },
];
const CLASS_COLORS = { car:"#3b82f6", taxi:"#eab308", bus:"#f97316", microbus:"#06b6d4", truck:"#7c3aed", motorcycle:"#dc2626", bicycle:"#16a34a" };

function FlowBar({ dir, color, current, predicted, maxVal }) {
  const curPct  = Math.round((current  / Math.max(maxVal,1)) * 100);
  const predPct = Math.round((predicted / Math.max(maxVal,1)) * 100);
  return (
    <div style={{ marginBottom:"14px" }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"6px" }}>
        <span style={{ fontSize:"13px", fontWeight:600, color }}>{dir.label}</span>
        <div style={{ display:"flex", gap:"16px", fontSize:"12px" }}>
          <span>Now: <strong style={{ color, fontFamily:"monospace" }}>{current}</strong></span>
          <span style={{ color:"#94a3b8" }}>+30s: <strong style={{ color:"#64748b", fontFamily:"monospace" }}>{predicted}</strong></span>
        </div>
      </div>
      <div style={{ height:"8px", background:"#f1f5f9", borderRadius:"99px", overflow:"hidden", marginBottom:"3px" }}>
        <div style={{ height:"100%", width:`${curPct}%`, background:color, borderRadius:"99px", transition:"width 0.6s" }}/>
      </div>
      <div style={{ height:"4px", background:"#f1f5f9", borderRadius:"99px", overflow:"hidden" }}>
        <div style={{ height:"100%", width:`${predPct}%`, background:color, opacity:0.35, borderRadius:"99px", transition:"width 0.8s" }}/>
      </div>
    </div>
  );
}

function VehicleMix({ typeCounts }) {
  const entries = Object.entries(typeCounts||{}).filter(([,v])=>v>0);
  const total   = entries.reduce((a,[,v])=>a+v,0)||1;
  if (!entries.length) return <div style={{ fontSize:"12px", color:"#94a3b8", textAlign:"center", padding:"16px 0" }}>No vehicles detected</div>;
  return (
    <div>
      <div style={{ display:"flex", height:"20px", borderRadius:"4px", overflow:"hidden", marginBottom:"10px" }}>
        {entries.map(([cls,cnt])=>(
          <div key={cls} style={{ width:`${(cnt/total)*100}%`, background:CLASS_COLORS[cls]||"#94a3b8", transition:"width 0.5s" }} title={`${cls}: ${cnt}`}/>
        ))}
      </div>
      <div style={{ display:"flex", flexWrap:"wrap", gap:"8px" }}>
        {entries.map(([cls,cnt])=>(
          <div key={cls} style={{ display:"flex", alignItems:"center", gap:"5px" }}>
            <div style={{ width:"10px", height:"10px", borderRadius:"2px", background:CLASS_COLORS[cls]||"#94a3b8" }}/>
            <span style={{ fontSize:"12px", color:"#374151", textTransform:"capitalize" }}>{cls} <strong>{cnt}</strong> <span style={{ color:"#94a3b8" }}>({Math.round((cnt/total)*100)}%)</span></span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LiveSimulation() {
  const mode = useSimMode();
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const [sel,      setSel]      = useState("morning_rush");
  const [run,      setRun]      = useState(false);
  const [pau,      setPau]      = useState(false);
  const [step,     setStep]     = useState(0);
  const [live,     setLive]     = useState(null);
  const [err,      setErr]      = useState(null);
  const [simState, setSimState] = useState("idle");
  const [frame,    setFrame]    = useState(null);
  const [full,     setFull]     = useState(false);
  const [lp,       setLp]       = useState(null);
  const [lst,      setLst]      = useState("waiting");
  const [lhlen,    setLhlen]    = useState(0);
  const poll  = useRef(null);
  const lstmR = useRef(null);

  const handleStop = useCallback(async () => {
    clearInterval(poll.current); clearInterval(lstmR.current);
    setRun(false); setPau(false); setLive(null);
    setFrame(null); setSimState("idle");
    setLp(null); setLst("waiting"); setLhlen(0);
    try { await fetch(`${API}/api/sumo/stop`, { method:"POST" }); } catch {}
  }, []);

  useEffect(() => {
    if (run && !pau) {
      poll.current = setInterval(async () => {
        try {
          const d = await fetch(`${API}/api/sumo/step`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({ steps:30 }),
          }).then(r=>r.json());
          if (d.latest) { setStep(d.latest.step); setLive(d.latest); if(d.latest.simulation_done) handleStop(); }
          if (d.image) setFrame(d.image);
        } catch { setErr("Lost connection"); handleStop(); }
      }, 2000);
    }
    return () => clearInterval(poll.current);
  }, [run, pau, handleStop]);

  useEffect(() => {
    if (run && !pau) {
      lstmR.current = setInterval(async () => {
        try {
          const d = await fetch(`${API}/api/lstm/predict/live`).then(r=>r.json());
          if (d.status==="collecting") { setLst("collecting"); setLhlen(d.history_len||0); }
          else if (d.status==="ok"&&d.next_30s) { setLst("ready"); setLp(d.next_30s); setLhlen(d.history_len||0); }
        } catch { setLst("error"); }
      }, 4000);
    } else clearInterval(lstmR.current);
    return () => clearInterval(lstmR.current);
  }, [run, pau]);

  const handleStart = async () => {
    setErr(null); setSimState("starting"); setFrame(null); setLive(null);
    setLp(null); setLst("collecting");
    try {
      const d = await fetch(`${API}/api/sumo/start`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ profile:sel, gui:true }),
      }).then(r=>r.json());
      if (d.status==="started"||d.status==="already_running") {
        setRun(true); setPau(false); setStep(0); setSimState("running");
        try {
          const fr = await fetch(`${API}/api/sumo/step`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ steps:1 }) }).then(r=>r.json());
          if (fr.image) setFrame(fr.image);
        } catch {}
      } else { setErr(d.detail||"Failed"); setSimState("idle"); }
    } catch { setErr("Cannot reach backend"); setSimState("idle"); }
  };

  const dirCounts = live?.direction_counts||{};
  const dirMax    = Math.max(...DIRS.map(d=>dirCounts[d.key]||0),1);
  const predMax   = Math.max(...DIRS.map(d=>lp?.[d.key]||0),dirMax,1);
  const barMax    = Math.max(dirMax,predMax);

  if (mode==="checking") return <div style={{ paddingTop:"48px", textAlign:"center", color:"#64748b" }}>Connecting...</div>;

  return (
    <div style={{ paddingTop:"24px" }}>
      {/* Header */}
      <div style={{ paddingBottom:"16px", borderBottom:"1px solid #e2e8f0", marginBottom:"20px", display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
        <div>
          <h1 style={{ margin:0, fontSize:"20px", fontWeight:700, color:"#0f172a" }}>Live Simulation</h1>
          <p style={{ margin:"4px 0 0", fontSize:"13px", color:"#64748b" }}>Real-time traffic flow monitoring — El-Tahrir Square, Cairo</p>
        </div>
        <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"12px", fontWeight:500,
          background: mode==="live"?"#dbeafe":"#f1f5f9",
          color: mode==="live"?"#1d4ed8":"#64748b",
          border:`1px solid ${mode==="live"?"#bfdbfe":"#e2e8f0"}` }}>
          {mode==="live" ? " Live Mode" : "▶ Video Mode"}
        </span>
      </div>

      {/* Profile + controls */}
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"16px", flexWrap:"wrap", gap:"10px" }}>
        <div style={{ display:"flex", gap:"6px", alignItems:"center", flexWrap:"wrap" }}>
          <span style={{ ...slabel, marginRight:"4px" }}>Profile:</span>
          {PROFILES.map(p => (
            <button key={p.key}
              onClick={() => setSel(p.key)}
              disabled={mode==="live"&&(run||simState==="starting")}
              title={p.time}
              style={{ padding:"5px 12px", fontSize:"13px", fontWeight:500, borderRadius:"4px", border:"1px solid",
                cursor: mode==="live"&&(run||simState==="starting")?"not-allowed":"pointer",
                background:sel===p.key?"#1d4ed8":"#fff", color:sel===p.key?"#fff":"#374151",
                borderColor:sel===p.key?"#1d4ed8":"#d1d5db" }}>
              {p.label}
            </button>
          ))}
        </div>
        {mode==="live" && (
          <div style={{ display:"flex", gap:"6px" }}>
            <button style={btnS("primary", run||simState==="starting")} onClick={handleStart} disabled={run||simState==="starting"}>
              <Play size={13} strokeWidth={2.5}/>{simState==="starting"?"Opening...":"Start"}
            </button>
            <button style={btnS("danger", !run)} onClick={handleStop} disabled={!run}>
              <Square size={13} strokeWidth={2.5}/> Stop
            </button>
            <button style={btnS("neutral", !run)} onClick={()=>setPau(p=>!p)} disabled={!run}>
              <Pause size={13} strokeWidth={2.5}/>{pau?"Resume":"Pause"}
            </button>
          </div>
        )}
      </div>

      {err && <div style={{ padding:"8px 12px", background:"#fee2e2", border:"1px solid #fca5a5", borderRadius:"4px", color:"#dc2626", fontSize:"13px", marginBottom:"16px" }}>⚠ {err}</div>}

      {/* Main feed */}
      <div style={{ ...card, marginBottom:"16px" }}>
        <div style={{ position:"relative", height:isMobile?"240px":"420px", background:"#0f172a", borderRadius:"6px", overflow:"hidden" }}>

          {/* VIDEO MODE */}
          {mode==="video" && (
            <>
              <video key={sel} src={VIDEO_MAP[sel]} autoPlay loop muted playsInline
                style={{ width:"100%", height:"100%", objectFit:"cover" }}/>
              <div style={{ position:"absolute", top:"12px", left:"12px", display:"flex", gap:"6px" }}>
                <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"11px", fontWeight:600, fontFamily:"monospace", background:"rgba(0,0,0,0.7)", color:"#4ade80" }}>● SIMULATION</span>
                <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"11px", fontFamily:"monospace", background:"rgba(0,0,0,0.7)", color:"#93c5fd" }}>
                  {PROFILES.find(p=>p.key===sel)?.label}
                </span>
              </div>
            </>
          )}

          {/* LIVE MODE */}
          {mode==="live" && (
            frame ? (
              <>
                <img src={frame} alt="SUMO" style={{ width:"100%", height:"100%", objectFit:"cover" }}/>
                <div style={{ position:"absolute", top:"12px", left:"12px", display:"flex", gap:"6px" }}>
                  <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"11px", fontWeight:600, fontFamily:"monospace", background:"rgba(0,0,0,0.7)", color:pau?"#fbbf24":"#4ade80" }}>
                    {pau?"⏸ PAUSED":"● LIVE"}
                  </span>
                  <span style={{ padding:"4px 10px", borderRadius:"4px", fontSize:"11px", fontFamily:"monospace", background:"rgba(0,0,0,0.7)", color:"#93c5fd" }}>Step {step.toLocaleString()}</span>
                </div>
                {live && (
                  <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"linear-gradient(transparent,rgba(0,0,0,0.75))", padding:"24px 16px 12px" }}>
                    <div style={{ display:"flex", justifyContent:"space-around" }}>
                      {[
                        { label:"Vehicles", value:live.vehicles, unit:"" },
                        { label:"Avg Wait", value:live.avg_wait_s, unit:"s" },
                        { label:"Avg Speed", value:live.avg_speed, unit:"m/s" },
                        { label:"CO₂", value:`${(live.total_co2_mg/1000).toFixed(0)}k`, unit:"mg" },
                      ].map(m => (
                        <div key={m.label} style={{ textAlign:"center" }}>
                          <div style={{ fontSize:"20px", fontWeight:700, color:"#fff", fontFamily:"monospace", lineHeight:1 }}>{m.value}<span style={{ fontSize:"12px", color:"rgba(255,255,255,0.6)", marginLeft:"2px" }}>{m.unit}</span></div>
                          <div style={{ fontSize:"11px", color:"rgba(255,255,255,0.55)", marginTop:"3px" }}>{m.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <button style={{ position:"absolute", top:"12px", right:"12px", padding:"4px 8px", background:"rgba(0,0,0,0.6)", color:"#fff", border:"none", borderRadius:"99px", cursor:"pointer" }} onClick={()=>setFull(true)}><Maximize2 size={12}/></button>
              </>
            ) : (
              <div style={{ height:"100%", display:"flex", alignItems:"center", justifyContent:"center", flexDirection:"column", gap:"10px" }}>
                {simState==="starting"
                  ? <><div style={{ width:"32px", height:"32px", border:"2px solid #3b82f6", borderTopColor:"transparent", borderRadius:"50%", animation:"spin 0.8s linear infinite" }}/><span style={{ color:"#60a5fa", fontSize:"14px" }}>Opening SUMO-GUI...</span></>
                  : <><span style={{ fontSize:"36px" }}></span><span style={{ color:"#64748b", fontSize:"14px" }}>Select a profile and press <strong style={{ color:"#1d4ed8" }}>Start</strong></span></>
                }
              </div>
            )
          )}
        </div>
      </div>

      {/* 3 insight cards */}
      <div style={{ display:"grid", gridTemplateColumns:gridCols(3,2,1,isMobile,isTablet), gap:"14px" }}>

        {/* Direction flow */}
        <div style={{ ...card, padding:"16px" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"14px" }}>
            <span style={slabel}>Traffic Flow by Direction</span>
            {mode==="live" && (
              <div style={{ display:"flex", gap:"10px", fontSize:"11px" }}>
                <span style={{ display:"flex", alignItems:"center", gap:"4px" }}><div style={{ width:"20px", height:"6px", borderRadius:"99px", background:"#3b82f6" }}/> Now</span>
                <span style={{ display:"flex", alignItems:"center", gap:"4px" }}><div style={{ width:"20px", height:"3px", borderRadius:"99px", background:"#94a3b8" }}/> +30s</span>
              </div>
            )}
          </div>
          {mode==="video"
            ? (() => {
                const staticDir = STATIC_DIRECTIONS[sel]||{};
                const staticPred = STATIC_BILSTM[sel]||{};
                const maxV = Math.max(...Object.values(staticDir), ...Object.values(staticPred), 1);
                return DIRS.map(d => <FlowBar key={d.key} dir={d} color={d.color} current={staticDir[d.key]||0} predicted={staticPred[d.key]||0} maxVal={maxV}/>);
              })()
            : !run
              ? <div style={{ fontSize:"12px", color:"#94a3b8", textAlign:"center", padding:"20px 0" }}>Start simulation to see live flow</div>
              : DIRS.map(d => <FlowBar key={d.key} dir={d} color={d.color} current={dirCounts[d.key]||0} predicted={lp?.[d.key]||0} maxVal={barMax}/>)
          }
        </div>

        {/* BiLSTM */}
        <div style={{ ...card, padding:"16px" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"14px" }}>
            <span style={slabel}>BiLSTM Flow Forecast</span>
            {mode==="live" && (
              <span style={{ fontSize:"11px", fontWeight:600, padding:"3px 8px", borderRadius:"99px",
                background:lst==="ready"?"#dcfce7":lst==="collecting"?"#fef3c7":"#f1f5f9",
                color:lst==="ready"?"#15803d":lst==="collecting"?"#b45309":"#64748b" }}>
                {lst==="ready"?" Live":lst==="collecting"?`${lhlen}/60 steps`:lst==="error"?"Error":"Waiting"}
              </span>
            )}
          </div>
          {mode==="video"
            ? (() => {
                const sp = STATIC_BILSTM[sel]||{};
                return (
                  <div>
                    <p style={{ fontSize:"12px", color:"#64748b", marginBottom:"12px" }}>Predicted arrivals in next 30 seconds:</p>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px" }}>
                      {DIRS.map(d => (
                        <div key={d.key} style={{ padding:"10px 12px", background:"#f8fafc", border:"1px solid #e2e8f0", borderRadius:"4px", borderLeft:`3px solid ${d.color}` }}>
                          <div style={{ fontSize:"11px", color:"#64748b", marginBottom:"4px" }}>{d.label}</div>
                          <div style={{ fontSize:"20px", fontWeight:700, color:d.color, fontFamily:"monospace" }}>{sp[d.key]||0}</div>
                          <div style={{ fontSize:"11px", color:"#94a3b8" }}>vehicles</div>
                        </div>
                      ))}
                    </div>
                    <p style={{ fontSize:"11px", color:"#94a3b8", marginTop:"10px", borderTop:"1px solid #f1f5f9", paddingTop:"8px" }}>BiLSTM 2×128 · trained on Tahrir Square data</p>
                  </div>
                );
              })()
            : lst==="waiting"
              ? <div style={{ fontSize:"12px", color:"#94a3b8", textAlign:"center", padding:"24px 0" }}>Start simulation to see predictions</div>
              : lst==="collecting"
                ? <div><p style={{ fontSize:"13px", color:"#b45309", marginBottom:"10px" }}>Collecting history...</p><div style={{ height:"8px", background:"#f1f5f9", borderRadius:"99px", overflow:"hidden" }}><div style={{ height:"100%", background:"#f59e0b", width:`${(lhlen/60)*100}%`, transition:"width 0.5s" }}/></div><p style={{ fontSize:"11px", color:"#94a3b8", marginTop:"6px" }}>{lhlen}/60 steps</p></div>
                : lst==="ready"&&lp
                  ? <div>
                      <p style={{ fontSize:"12px", color:"#64748b", marginBottom:"12px" }}>Predicted arrivals in next 30 seconds:</p>
                      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px" }}>
                        {DIRS.map(d => (
                          <div key={d.key} style={{ padding:"10px 12px", background:"#f8fafc", border:"1px solid #e2e8f0", borderRadius:"4px", borderLeft:`3px solid ${d.color}` }}>
                            <div style={{ fontSize:"11px", color:"#64748b", marginBottom:"4px" }}>{d.label}</div>
                            <div style={{ fontSize:"20px", fontWeight:700, color:d.color, fontFamily:"monospace" }}>{lp[d.key]||0}</div>
                            <div style={{ fontSize:"11px", color:"#94a3b8" }}>vehicles</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  : null
          }
        </div>

        {/* Vehicle mix */}
        <div style={{ ...card, padding:"16px" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"14px" }}>
            <span style={slabel}>Vehicle Mix</span>
            {live && <span style={{ fontSize:"12px", color:"#64748b" }}>Total: <strong style={{ color:"#0f172a" }}>{live.vehicles}</strong></span>}
          </div>
          {mode==="video"
            ? (
              <div>
                <p style={{ fontSize:"12px", color:"#64748b", marginBottom:"10px" }}>Cairo traffic composition:</p>
                <div style={{ display:"flex", height:"16px", borderRadius:"4px", overflow:"hidden", marginBottom:"10px" }}>
                  {[["car","#3b82f6",55],["taxi","#eab308",15],["microbus","#06b6d4",12],["bus","#f97316",8],["truck","#7c3aed",5],["motorcycle","#dc2626",4],["bicycle","#16a34a",1]].map(([cls,color,pct])=>(
                    <div key={cls} style={{ width:`${pct}%`, background:color }} title={`${cls}: ${pct}%`}/>
                  ))}
                </div>
                <div style={{ display:"flex", flexWrap:"wrap", gap:"6px" }}>
                  {[["car","#3b82f6","55%"],["taxi","#eab308","15%"],["microbus","#06b6d4","12%"],["bus","#f97316","8%"],["truck","#7c3aed","5%"],["motorcycle","#dc2626","4%"],["bicycle","#16a34a","1%"]].map(([cls,color,pct])=>(
                    <div key={cls} style={{ display:"flex", alignItems:"center", gap:"4px" }}>
                      <div style={{ width:"8px", height:"8px", borderRadius:"2px", background:color }}/>
                      <span style={{ fontSize:"11px", color:"#374151" }}>{cls} <span style={{ color:"#94a3b8" }}>{pct}</span></span>
                    </div>
                  ))}
                </div>
              </div>
            )
            : !run
              ? <div style={{ fontSize:"12px", color:"#94a3b8", textAlign:"center", padding:"20px 0" }}>Start simulation to see vehicle types</div>
              : <VehicleMix typeCounts={live?.type_counts||{}}/>
          }
        </div>
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