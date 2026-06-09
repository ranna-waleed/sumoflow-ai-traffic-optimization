import React,{useState,useEffect,useRef,useCallback} from "react";
import {ResponsiveContainer,BarChart,Bar,XAxis,YAxis,Tooltip,CartesianGrid,LineChart,Line,Cell} from "recharts";
import {Play,Square} from "lucide-react";
import { useSimMode, DQN_VIDEO_MAP, API, STATIC_DQN_RESULTS } from "../hooks/useSimMode";
import { useIsMobile, useIsTablet, gridCols } from "../hooks/useIsMobile";

const TT={backgroundColor:"#fff",border:"1px solid #e2e8f0",borderRadius:"4px",fontSize:"12px",color:"#1e293b"};
const TICK={fill:"#94a3b8",fontSize:11};
const card={background:"#fff",border:"1px solid #e2e8f0",borderRadius:"6px",boxShadow:"0 1px 3px rgba(0,0,0,.04)"};
const slabel={fontSize:"11px",fontWeight:600,letterSpacing:"0.06em",textTransform:"uppercase",color:"#64748b"};
const btn=(v)=>({display:"inline-flex",alignItems:"center",gap:"6px",padding:"6px 14px",fontSize:"13px",fontWeight:500,borderRadius:"4px",border:"1px solid",cursor:"pointer",transition:"all 0.15s",
  ...(v==="primary"?{background:"#1d4ed8",color:"#fff",borderColor:"#1d4ed8"}
    :v==="danger"?{background:"#fff",color:"#dc2626",borderColor:"#fca5a5"}
    :{background:"#fff",color:"#374151",borderColor:"#d1d5db"})});

const PROFILES=[
  {key:"morning_rush",label:"Morning Rush",time:"07:30–10:30"},
  {key:"evening_rush",label:"Evening Rush",time:"15:00–20:00"},
  {key:"midday",      label:"Midday",      time:"12:00–15:00"},
  {key:"night",       label:"Night",       time:"22:00–24:00"},
];

export default function BeforeAfter(){
  const mode = useSimMode();
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const [results,setResults]         = useState(null);
  const [loading,setLoading]         = useState(true);
  const [error,setError]             = useState(null);
  const [simRunning,setSimRunning]   = useState(false);
  const [simProfile,setSimProfile]   = useState(null);
  const [simMetrics,setSimMetrics]   = useState(null);
  const [simFrame,setSimFrame]       = useState(null);
  const [startingProfile,setStartingProfile] = useState(null);
  const [qValues,setQValues]         = useState(null);
  const pollRef = useRef(null);

  useEffect(()=>{
    if (mode === "checking") return; // wait for mode detection
    if (mode === "video") {
      setResults(STATIC_DQN_RESULTS);
      setError(null);
      setLoading(false);
      return;
    }
    fetch(`${API}/api/dqn/results`)
      .then(r=>{ if(!r.ok) throw new Error(); return r.json(); })
      .then(d=>{ setResults(d); setLoading(false); })
      .catch(()=>{
        // Backend online but no results yet — use static
        setResults(STATIC_DQN_RESULTS);
        setError(null);
        setLoading(false);
      });
  },[mode]);

  const pollSim = useCallback(async()=>{
    try{
      const status = await fetch(`${API}/api/dqn/sim/status`).then(r=>r.json());
      setSimRunning(status.running);
      if(status.running){
        setSimMetrics(status.metrics);
        if(status.metrics?.q_values) setQValues(status.metrics.q_values);
        const imgRes = await fetch(`${API}/api/dqn/sim/screenshot`);
        if(imgRes.ok){
          const blob = await imgRes.blob();
          const url  = URL.createObjectURL(blob);
          setSimFrame(prev=>{ if(prev) URL.revokeObjectURL(prev); return url; });
        }
      }
    }catch{}
  },[]);

  useEffect(()=>{
    if(simRunning&&mode==="live"){ pollRef.current=setInterval(pollSim,500); }
    else{ clearInterval(pollRef.current); }
    return()=>clearInterval(pollRef.current);
  },[simRunning,pollSim,mode]);

  const handleStart = async(profileKey)=>{
    if(mode==="video"){ setSimProfile(profileKey); setSimRunning(true); return; }
    setStartingProfile(profileKey);
    try{
      if(simRunning){
        await fetch(`${API}/api/dqn/sim/stop`,{method:"POST"});
        await new Promise(r=>setTimeout(r,1500));
      }
      const res = await fetch(`${API}/api/dqn/sim/start/${profileKey}`,{method:"POST"});
      if(!res.ok) throw new Error((await res.json()).detail);
      setSimProfile(profileKey); setSimRunning(true);
      setSimMetrics(null); setSimFrame(null);
    }catch(e){ alert(`Failed to start: ${e.message}`); }
    finally{ setStartingProfile(null); }
  };

  const handleStop = async()=>{
    if(mode==="video"){ setSimRunning(false); setSimProfile(null); return; }
    await fetch(`${API}/api/dqn/sim/stop`,{method:"POST"});
    setSimRunning(false); setSimMetrics(null); setSimFrame(null);
  };

  const waitBefore      = results?.avg_wait_fixed      || 0;
  const waitAfter       = results?.avg_wait_dqn        || 0;
  const waitImprovement = results?.improvement_pct     || 0;
  const co2Improvement  = results?.co2_improvement_pct || 0;
  const fixedCo2        = results?.fixed_co2_mg        || 0;
  const dqnCo2          = results?.dqn_co2_mg          || 0;
  const profileData     = results?.profiles || {};

  const episodeData = (results?.episode_waits||[]).filter((_,i)=>i%3===0).map((w,i)=>({episode:(i*3)+1,wait:parseFloat(w.toFixed(1))}));
  const waitBarData = [{name:"Fixed-Time",value:parseFloat(waitBefore.toFixed(1))},{name:"DQN Adaptive",value:parseFloat(waitAfter.toFixed(1))}];
  const co2BarData  = [{name:"Fixed-Time",value:parseFloat((fixedCo2/1e9).toFixed(2))},{name:"DQN Adaptive",value:parseFloat((dqnCo2/1e9).toFixed(2))}];

  return(
    <div style={{paddingTop:"24px"}}>
      <div style={{paddingBottom:"16px",borderBottom:"1px solid #e2e8f0",marginBottom:"24px",display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
        <div>
          <h1 style={{margin:0,fontSize:"20px",fontWeight:700,color:"#0f172a"}}>Before vs After — DQN Performance</h1>
          <p style={{margin:"4px 0 0",fontSize:"13px",color:"#64748b"}}>Fixed-time baseline compared to DQN adaptive control — El-Tahrir Square</p>
        </div>
        <div style={{display:"flex",gap:"6px"}}>
          {results&&<><span style={{padding:"4px 10px",borderRadius:"99px",fontSize:"12px",fontWeight:500,background:"#dbeafe",color:"#1d4ed8"}}>{results.episodes} episodes trained</span><span style={{padding:"4px 10px",borderRadius:"99px",fontSize:"12px",fontWeight:500,background:"#dcfce7",color:"#15803d"}}>All KPIs met</span></>}
          <span style={{padding:"4px 10px",borderRadius:"99px",fontSize:"12px",fontWeight:500,background:mode==="live"?"#dbeafe":"#f1f5f9",color:mode==="live"?"#1d4ed8":"#64748b",border:`1px solid ${mode==="live"?"#bfdbfe":"#e2e8f0"}`}}>{mode==="live"?" Live":"▶ Video"}</span>
        </div>
      </div>

      {error&&<div style={{padding:"8px 14px",background:"#fef3c7",border:"1px solid #fde68a",borderRadius:"4px",color:"#92400e",fontSize:"13px",marginBottom:"16px"}}>⚠ {error}</div>}

      {/* KPI cards */}
      {results&&(
        <div style={{display:"grid",gridTemplateColumns:gridCols(4,2,1,isMobile,isTablet),gap:"12px",marginBottom:"20px"}}>
          {[
            {label:"Avg Wait: Baseline",value:`${waitBefore.toFixed(1)}s`,sub:"fixed-time signals",accent:"#dc2626"},
            {label:"Avg Wait: DQN",     value:`${waitAfter.toFixed(1)}s`, sub:"adaptive control", accent:"#15803d"},
            {label:"Wait Time Reduction",value:`↓${waitImprovement.toFixed(1)}%`,sub:"across all profiles",accent:"#15803d"},
            {label:"CO₂ Reduction",      value:`↓${co2Improvement.toFixed(1)}%`, sub:"total emissions",   accent:"#15803d"},
          ].map(c=>(
            <div key={c.label} style={{...card,padding:"16px 20px",borderTop:`3px solid ${c.accent}`}}>
              <div style={{fontSize:"11px",fontWeight:600,letterSpacing:"0.06em",textTransform:"uppercase",color:"#64748b",marginBottom:"8px"}}>{c.label}</div>
              <div style={{fontSize:"26px",fontWeight:700,color:c.accent,lineHeight:1,fontVariantNumeric:"tabular-nums"}}>{c.value}</div>
              <div style={{fontSize:"12px",color:"#94a3b8",marginTop:"6px"}}>{c.sub}</div>
            </div>
          ))}
        </div>
      )}

      {/* Per-profile table */}
      {results&&Object.keys(profileData).length>0&&(
        <div style={{...card,marginBottom:"16px"}}>
          <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            <span style={slabel}>Per-Profile KPI Results</span>
            <div style={{display:"flex",gap:"16px",fontSize:"12px"}}>
              <span style={{color:"#dc2626"}}> Baseline (fixed-time)</span>
              <span style={{color:"#15803d"}}> DQN (adaptive)</span>
            </div>
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse"}}>
              <thead>
                <tr style={{background:"#f8fafc"}}>
                  {["Profile","Baseline Wait","DQN Wait","Wait ↓","Baseline CO₂","DQN CO₂","CO₂ ↓","Throughput Δ","Status"].map(h=>(
                    <th key={h} style={{padding:"10px 14px",textAlign:"left",fontSize:"12px",fontWeight:600,color:"#64748b",borderBottom:"1px solid #e2e8f0",whiteSpace:"nowrap"}}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PROFILES.map(p=>{
                  const d=profileData[p.key]; if(!d) return null;
                  return(
                    <tr key={p.key} style={{borderBottom:"1px solid #f1f5f9"}}>
                      <td style={{padding:"10px 14px",fontSize:"14px",fontWeight:600,color:"#0f172a"}}>{p.label}</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",color:"#dc2626",fontWeight:500}}>{d.fixed_wait_s?.toFixed(1)}s</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",color:"#15803d",fontWeight:500}}>{d.dqn_wait_s?.toFixed(1)}s</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",fontWeight:700,color:"#15803d"}}>↓{d.wait_improvement?.toFixed(1)}%</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",color:"#dc2626"}}>{(d.fixed_co2_mg/1e9).toFixed(2)}B mg</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",color:"#15803d"}}>{(d.dqn_co2_mg/1e9).toFixed(2)}B mg</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",fontWeight:700,color:"#15803d"}}>↓{d.co2_improvement?.toFixed(1)}%</td>
                      <td style={{padding:"10px 14px",fontSize:"13px",fontFamily:"monospace",color:d.throughput_delta>=0?"#15803d":"#64748b"}}>{d.throughput_delta>=0?"+":""}{d.throughput_delta}</td>
                      <td style={{padding:"10px 14px"}}>
                        <span style={{padding:"3px 8px",borderRadius:"99px",fontSize:"11px",fontWeight:600,background:d.kpi_wait_pass&&d.kpi_co2_pass?"#dcfce7":"#fee2e2",color:d.kpi_wait_pass&&d.kpi_co2_pass?"#15803d":"#dc2626"}}>
                          {d.kpi_wait_pass&&d.kpi_co2_pass?" Pass":" Fail"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* DQN Simulation panel */}
      <div style={{...card,marginBottom:"16px"}}>
        <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={slabel}>DQN Adaptive Control: {mode==="video"?"Recorded Simulation":"Live Simulation"}</span>
          {simRunning&&<button style={btn("danger")} onClick={handleStop}><Square size={13} strokeWidth={2}/> Stop</button>}
        </div>

        {/* Profile buttons */}
        <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9",display:"flex",gap:"8px",flexWrap:"wrap"}}>
          {PROFILES.map(p=>{
            const pd=profileData[p.key];
            const isActive=simProfile===p.key&&simRunning;
            const starting=startingProfile===p.key;
            return(
              <button key={p.key} onClick={()=>handleStart(p.key)} disabled={starting}
                style={{padding:"10px 16px",borderRadius:"4px",border:"1px solid",cursor:starting?"not-allowed":"pointer",
                  background:isActive?"#1d4ed8":"#fff",color:isActive?"#fff":"#374151",
                  borderColor:isActive?"#1d4ed8":"#d1d5db",transition:"all 0.15s",textAlign:"left"}}>
                <div style={{display:"flex",alignItems:"center",gap:"8px",marginBottom:pd?"4px":"0"}}>
                  {isActive&&<span style={{width:"7px",height:"7px",borderRadius:"50%",background:"#4ade80",display:"inline-block"}}/>}
                  <span style={{fontSize:"13px",fontWeight:600}}>{p.label}</span>
                  {starting&&<span style={{fontSize:"11px",color:"#60a5fa"}}>starting...</span>}
                </div>
                <div style={{fontSize:"11px",color:isActive?"rgba(255,255,255,0.7)":"#94a3b8"}}>{p.time}</div>
                {pd&&(
                  <div style={{display:"flex",gap:"6px",marginTop:"6px"}}>
                    <span style={{padding:"2px 6px",borderRadius:"99px",fontSize:"11px",fontWeight:500,background:isActive?"rgba(255,255,255,0.2)":"#dcfce7",color:isActive?"#fff":"#15803d"}}>↓{pd.wait_improvement?.toFixed(0)}% wait</span>
                    <span style={{padding:"2px 6px",borderRadius:"99px",fontSize:"11px",fontWeight:500,background:isActive?"rgba(255,255,255,0.2)":"#dcfce7",color:isActive?"#fff":"#15803d"}}>↓{pd.co2_improvement?.toFixed(0)}% CO₂</span>
                  </div>
                )}
                {!isActive&&!starting&&<div style={{fontSize:"11px",color:"#1d4ed8",marginTop:"4px",display:"flex",alignItems:"center",gap:"4px"}}><Play size={10}/> {mode==="video"?"Watch DQN":"Run DQN"}</div>}
              </button>
            );
          })}
        </div>

        {/* Viewport + metrics */}
        <div style={{display:"grid",gridTemplateColumns:isMobile||isTablet?"1fr":"1fr 300px"}}>
          <div style={{padding:"12px",borderRight:"1px solid #f1f5f9"}}>
            <div style={{height:isMobile?"200px":"300px",background:"#0f172a",borderRadius:"4px",border:"1px solid #e2e8f0",overflow:"hidden",position:"relative"}}>

              {/* VIDEO MODE */}
              {mode==="video" && simProfile && simRunning ? (
                <>
                  <video key={simProfile} src={DQN_VIDEO_MAP[simProfile]} autoPlay loop muted playsInline
                    style={{width:"100%",height:"100%",objectFit:"cover"}}/>
                  <div style={{position:"absolute",top:"10px",left:"10px",display:"flex",gap:"6px"}}>
                    <span style={{padding:"3px 8px",borderRadius:"99px",fontSize:"11px",fontWeight:600,fontFamily:"monospace",background:"rgba(0,0,0,0.75)",color:"#4ade80"}}>● DQN ACTIVE</span>
                    <span style={{padding:"3px 8px",borderRadius:"99px",fontSize:"11px",fontFamily:"monospace",background:"rgba(29,78,216,0.85)",color:"#fff"}}>⚡ Adaptive Control</span>
                  </div>
                </>
              ) : mode==="live" && simFrame && simRunning ? (
                <>
                  <img src={simFrame} alt="DQN" style={{width:"100%",height:"100%",objectFit:"cover"}}/>
                  <div style={{position:"absolute",top:"10px",left:"10px"}}>
                    <span style={{padding:"3px 8px",borderRadius:"99px",fontSize:"11px",fontWeight:600,fontFamily:"monospace",background:"rgba(0,0,0,0.75)",color:"#4ade80"}}>● DQN ACTIVE</span>
                  </div>
                  {simMetrics?.current_action&&(
                    <div style={{position:"absolute",bottom:"10px",left:"10px"}}>
                      <span style={{padding:"3px 8px",borderRadius:"99px",fontSize:"11px",fontFamily:"monospace",background:"rgba(29,78,216,0.85)",color:"#fff"}}>⚡ {simMetrics.current_action}</span>
                    </div>
                  )}
                </>
              ) : (
                <div style={{height:"100%",display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column",gap:"8px"}}>
                  <span style={{fontSize:"24px"}}></span>
                  <span style={{color:"#94a3b8",fontSize:"13px"}}>{simRunning?"Loading...":"Select a profile to watch DQN"}</span>
                </div>
              )}
            </div>
          </div>

          {/* Metrics */}
          <div style={{padding:"16px"}}>
            <div style={{...slabel,marginBottom:"12px"}}>{simRunning&&mode==="live"?"Live Metrics":"Profile Results"}</div>
            {simRunning&&mode==="live"&&simMetrics?( 
              <>
                {[
                  {label:"Vehicles", value:simMetrics.vehicles},
                  {label:"Avg Wait", value:`${simMetrics.avg_wait_s}s`},
                  {label:"Avg Speed",value:`${simMetrics.avg_speed} m/s`},
                  {label:"Total CO₂",value:`${(simMetrics.total_co2_mg/1000).toFixed(0)}k mg`},
                  {label:"Sim Step", value:simMetrics.step},
                ].map(row=>(
                  <div key={row.label} style={{display:"flex",justifyContent:"space-between",padding:"7px 0",borderBottom:"1px solid #f1f5f9"}}>
                    <span style={{fontSize:"13px",color:"#64748b"}}>{row.label}</span>
                    <span style={{fontSize:"13px",fontWeight:600,color:"#0f172a",fontFamily:"monospace"}}>{row.value}</span>
                  </div>
                ))}
                {simMetrics.current_action&&<div style={{marginTop:"10px",padding:"8px 10px",background:"#dbeafe",borderRadius:"4px"}}><div style={{fontSize:"11px",color:"#64748b",marginBottom:"2px"}}>Current Action</div><div style={{fontSize:"13px",fontWeight:600,color:"#1d4ed8"}}>{simMetrics.current_action}</div></div>}
                {qValues&&Object.keys(qValues).length>0&&(
                  <div style={{marginTop:"14px"}}>
                    <div style={{...slabel,marginBottom:"8px"}}>Junction Decisions</div>
                    {Object.entries(qValues).map(([junc,val])=>{
                      const switched=val>0.5;
                      return(
                        <div key={junc} style={{marginBottom:"7px"}}>
                          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"3px"}}>
                            <span style={{fontSize:"12px",color:"#374151",fontWeight:500}}>{junc}</span>
                            <span style={{padding:"2px 7px",borderRadius:"99px",fontSize:"11px",fontWeight:600,background:switched?"#dbeafe":"#f1f5f9",color:switched?"#1d4ed8":"#64748b"}}>{switched?"SWITCH":"KEEP"}</span>
                          </div>
                          <div style={{height:"4px",background:"#f1f5f9",borderRadius:"99px",overflow:"hidden"}}>
                            <div style={{height:"100%",borderRadius:"99px",width:`${val*100}%`,background:switched?"#1d4ed8":"#94a3b8",transition:"width 0.4s"}}/>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </>
            ):simProfile&&profileData[simProfile]?(
              <>
                {[
                  {label:"Fixed Wait",    value:`${profileData[simProfile].fixed_wait_s?.toFixed(1)}s`,color:"#dc2626"},
                  {label:"DQN Wait",      value:`${profileData[simProfile].dqn_wait_s?.toFixed(1)}s`, color:"#15803d"},
                  {label:"Reduction",     value:`↓${profileData[simProfile].wait_improvement?.toFixed(1)}%`,color:"#15803d"},
                  {label:"Fixed CO₂",     value:`${(profileData[simProfile].fixed_co2_mg/1e9).toFixed(2)}B mg`,color:"#dc2626"},
                  {label:"DQN CO₂",       value:`${(profileData[simProfile].dqn_co2_mg/1e9).toFixed(2)}B mg`,color:"#15803d"},
                  {label:"CO₂ Reduction", value:`↓${profileData[simProfile].co2_improvement?.toFixed(1)}%`,color:"#15803d"},
                ].map(row=>(
                  <div key={row.label} style={{display:"flex",justifyContent:"space-between",padding:"7px 0",borderBottom:"1px solid #f1f5f9"}}>
                    <span style={{fontSize:"13px",color:"#64748b"}}>{row.label}</span>
                    <span style={{fontSize:"13px",fontWeight:600,color:row.color,fontFamily:"monospace"}}>{row.value}</span>
                  </div>
                ))}
              </>
            ):(
              <div style={{fontSize:"13px",color:"#94a3b8",paddingTop:"16px"}}>Select a profile above to see results</div>
            )}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div style={{display:"grid",gridTemplateColumns:gridCols(3,2,1,isMobile,isTablet),gap:"12px"}}>
        <div style={{...card,padding:"16px"}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:"14px"}}>
            <span style={{fontSize:"14px",fontWeight:600,color:"#0f172a"}}>Avg Wait Time</span>
            <span style={{fontSize:"11px",color:"#94a3b8",fontFamily:"monospace"}}>seconds</span>
          </div>
          <div style={{height:"180px"}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={waitBarData} margin={{left:-20,right:8,top:4,bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false}/>
                <XAxis dataKey="name" tick={TICK} tickLine={false}/>
                <YAxis tick={TICK} tickLine={false}/>
                <Tooltip contentStyle={TT} formatter={v=>[`${v}s`,"Avg Wait"]}/>
                <Bar dataKey="value" radius={[3,3,0,0]}><Cell fill="#dc2626"/><Cell fill="#15803d"/></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div style={{...card,padding:"16px"}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:"14px"}}>
            <span style={{fontSize:"14px",fontWeight:600,color:"#0f172a"}}>CO₂ Emissions</span>
            <span style={{fontSize:"11px",color:"#94a3b8",fontFamily:"monospace"}}>billion mg</span>
          </div>
          <div style={{height:"180px"}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={co2BarData} margin={{left:-20,right:8,top:4,bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false}/>
                <XAxis dataKey="name" tick={TICK} tickLine={false}/>
                <YAxis tick={TICK} tickLine={false}/>
                <Tooltip contentStyle={TT} formatter={v=>[`${v}B mg`,"CO₂"]}/>
                <Bar dataKey="value" radius={[3,3,0,0]}><Cell fill="#dc2626"/><Cell fill="#15803d"/></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div style={{...card,padding:"16px"}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:"14px"}}>
            <span style={{fontSize:"14px",fontWeight:600,color:"#0f172a"}}>DQN Learning Curve</span>
            <span style={{fontSize:"11px",color:"#94a3b8",fontFamily:"monospace"}}>approx wait / episode</span>
          </div>
          <div style={{height:"180px"}}>
            {episodeData.length===0
              ? <div style={{height:"100%",display:"flex",alignItems:"center",justifyContent:"center",color:"#94a3b8",fontSize:"13px"}}>{loading?"Loading...":"No training data"}</div>
              : <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={episodeData} margin={{left:-20,right:8,top:4,bottom:0}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                    <XAxis dataKey="episode" tick={TICK} tickLine={false}/>
                    <YAxis tick={TICK} tickLine={false}/>
                    <Tooltip contentStyle={TT} formatter={v=>[`${v}`,"Approx Wait"]}/>
                    <Line type="monotone" dataKey="wait" stroke="#1d4ed8" strokeWidth={2} dot={false} activeDot={{r:3}}/>
                  </LineChart>
                </ResponsiveContainer>
            }
          </div>
        </div>
      </div>
    </div>
  );
}