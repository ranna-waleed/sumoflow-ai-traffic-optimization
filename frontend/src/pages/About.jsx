import { useIsMobile, useIsTablet, gridCols } from "../hooks/useIsMobile";
import React from "react";

const card={background:"#fff",border:"1px solid #e2e8f0",borderRadius:"6px",boxShadow:"0 1px 3px rgba(0,0,0,.04)"};
const slabel={fontSize:"11px",fontWeight:600,letterSpacing:"0.06em",textTransform:"uppercase",color:"#64748b"};

const TEAM=[
  {name:"Rana",   role:"YOLOv8s · BiLSTM · Frontend",  initials:"RA"},
  {name:"Roaa",   role:"RetinaNet · Visualization",     initials:"RO"},
  {name:"Mariam", role:"Faster R-CNN · DQN Optimizer",  initials:"MA"},
];
const STACK=[
  ["Simulation",   "Eclipse SUMO 1.24 + TraCI API"],
  ["Detection",    "YOLOv8s · Faster R-CNN · RetinaNet"],
  ["Prediction",   "BiLSTM (2×128 hidden units)"],
  ["Optimizer",    "Dueling Double DQN — 7 agents"],
  ["Backend",      "FastAPI + Python 3.12"],
  ["Frontend",     "React 18 + Recharts + Tailwind CSS"],
  ["Training",     "CPU — 350 episodes, 4 profiles"],
  ["Evaluation",   "TraCI live + SUMO XML output"],
];
const PIPELINE=[
  {n:"01",label:"SUMO Simulation",   desc:"Microscopic traffic model of El-Tahrir Square. 4 traffic profiles. 7 vehicle classes in route files."},
  {n:"02",label:"Vehicle Detection", desc:"YOLOv8s selected as primary model (best mAP@0.5). Detects car, bus, truck, taxi, microbus, motorcycle, bicycle."},
  {n:"03",label:"Flow Prediction",   desc:"BiLSTM predicts traffic volume per direction (N/S/E/W) 30 seconds ahead. Feeds into DQN state vector."},
  {n:"04",label:"DQN Control",       desc:"7 Dueling Double DQN agents. 37-feature state. Binary action: keep or switch phase. 350 episodes trained."},
  {n:"05",label:"TraCI Execution",   desc:"DQN decisions applied to SUMO junctions via TraCI every 10 simulation seconds."},
  {n:"06",label:"Evaluation",        desc:"Before/after comparison across all 4 profiles. 92.6% avg wait reduction, 90.3% CO₂ reduction."},
];
const RESULTS=[
  {profile:"Morning Rush",wait:"↓92.6%",co2:"↓90.3%",base:"626.6s",dqn:"46.7s"},
  {profile:"Midday",      wait:"↓90.6%",co2:"↓90.1%",base:"629.1s",dqn:"59.2s"},
  {profile:"Evening Rush",wait:"↓92.7%",co2:"↓90.0%",base:"632.1s",dqn:"46.1s"},
  {profile:"Night",       wait:"↓57.9%",co2:"↓91.4%",base:"24.3s", dqn:"10.2s"},
];
const STATS=[
  ["1,800","Training images"],["7","Vehicle classes"],
  ["3","Detection models"],["60","Training epochs"],
  ["350","DQN episodes"],["7","DQN agents"],
  ["4","Traffic profiles"],["El-Tahrir Sq.","Simulation area"],
];

export default function About(){
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  return(
    <div style={{paddingTop:"24px"}}>
      <div style={{paddingBottom:"16px",borderBottom:"1px solid #e2e8f0",marginBottom:"24px",display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
        <div>
          <h1 style={{margin:0,fontSize:"20px",fontWeight:700,color:"#0f172a"}}>About — SUMOFlow AI</h1>
          <p style={{margin:"4px 0 0",fontSize:"13px",color:"#64748b"}}>El-Tahrir Square Traffic Optimization · Graduation Project</p>
        </div>
        <span style={{padding:"4px 10px",borderRadius:"4px",fontSize:"12px",fontWeight:500,background:"#dbeafe",color:"#1d4ed8"}}>v1.0.0</span>
      </div>

      {/* Overview */}
      <div style={{...card,padding:"24px",marginBottom:"16px",borderLeft:"4px solid #1d4ed8"}}>
        <h2 style={{margin:"0 0 12px",fontSize:"16px",fontWeight:700,color:"#0f172a"}}>Project Overview</h2>
        <p style={{margin:0,fontSize:"14px",color:"#374151",lineHeight:1.75,maxWidth:"860px"}}>
          SUMOFlow AI is an adaptive traffic signal control system combining vehicle detection,
          flow prediction, and deep reinforcement learning to optimize traffic light timing at
          El-Tahrir Square, Cairo. The DQN controller reduces vehicle waiting time by up to 92.7%
          and CO₂ emissions by over 90% compared to static baseline signals — verified through
          controlled SUMO simulation experiments across all 4 daily traffic profiles.
        </p>
      </div>

      {/* Results table */}
      <div style={{...card,marginBottom:"16px"}}>
        <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={slabel}>Evaluation Results</span>
          <span style={{padding:"3px 8px",borderRadius:"3px",fontSize:"12px",fontWeight:500,background:"#dcfce7",color:"#15803d"}}>✓ All KPIs Passed</span>
        </div>
        <table style={{width:"100%",borderCollapse:"collapse"}}>
          <thead>
            <tr style={{background:"#f8fafc"}}>
              {["Profile","Baseline Wait","DQN Wait","Wait Reduction","CO₂ Reduction","Status"].map(h=>(
                <th key={h} style={{padding:"10px 14px",textAlign:"left",fontSize:"12px",fontWeight:600,color:"#64748b",borderBottom:"1px solid #e2e8f0"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {RESULTS.map(r=>(
              <tr key={r.profile} style={{borderBottom:"1px solid #f1f5f9"}}>
                <td style={{padding:"10px 14px",fontSize:"14px",fontWeight:600,color:"#0f172a"}}>{r.profile}</td>
                <td style={{padding:"10px 14px",fontSize:"14px",fontFamily:"monospace",color:"#dc2626",fontWeight:500}}>{r.base}</td>
                <td style={{padding:"10px 14px",fontSize:"14px",fontFamily:"monospace",color:"#15803d",fontWeight:500}}>{r.dqn}</td>
                <td style={{padding:"10px 14px",fontSize:"14px",fontFamily:"monospace",fontWeight:700,color:"#15803d"}}>{r.wait}</td>
                <td style={{padding:"10px 14px",fontSize:"14px",fontFamily:"monospace",fontWeight:700,color:"#15803d"}}>{r.co2}</td>
                <td style={{padding:"10px 14px"}}><span style={{padding:"3px 8px",borderRadius:"3px",fontSize:"12px",fontWeight:600,background:"#dcfce7",color:"#15803d"}}>✓ Pass</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pipeline */}
      <div style={{...card,marginBottom:"16px"}}>
        <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9"}}>
          <span style={slabel}>System Pipeline</span>
        </div>
        <div style={{padding:"20px 24px"}}>
          {PIPELINE.map((s,i)=>(
            <div key={s.n} style={{display:"flex",gap:"20px",paddingBottom:i<PIPELINE.length-1?"20px":"0",marginBottom:i<PIPELINE.length-1?"0":"0"}}>
              <div style={{display:"flex",flexDirection:"column",alignItems:"center",flexShrink:0}}>
                <div style={{width:"32px",height:"32px",borderRadius:"50%",background:"#1d4ed8",color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"12px",fontWeight:700}}>
                  {s.n}
                </div>
                {i<PIPELINE.length-1&&<div style={{width:"2px",flex:1,background:"#e2e8f0",marginTop:"6px",marginBottom:"0",minHeight:"20px"}}/>}
              </div>
              <div style={{paddingBottom:i<PIPELINE.length-1?"20px":"0"}}>
                <div style={{fontSize:"15px",fontWeight:600,color:"#0f172a",marginBottom:"4px"}}>{s.label}</div>
                <div style={{fontSize:"13px",color:"#64748b",lineHeight:1.65}}>{s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Team + Stack */}
      <div style={{display:"grid",gridTemplateColumns:isMobile?"1fr":"1fr 1fr",gap:"16px",marginBottom:"16px"}}>
        <div style={{...card,padding:"16px"}}>
          <div style={{...slabel,marginBottom:"14px"}}>Development Team</div>
          {TEAM.map(m=>(
            <div key={m.name} style={{display:"flex",alignItems:"center",gap:"14px",padding:"12px",background:"#f8fafc",border:"1px solid #e2e8f0",borderRadius:"4px",marginBottom:"8px"}}>
              <div style={{width:"40px",height:"40px",borderRadius:"50%",background:"#1d4ed8",color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"13px",fontWeight:700,flexShrink:0}}>
                {m.initials}
              </div>
              <div>
                <div style={{fontSize:"15px",fontWeight:600,color:"#0f172a"}}>{m.name}</div>
                <div style={{fontSize:"12px",color:"#64748b",marginTop:"2px"}}>{m.role}</div>
              </div>
            </div>
          ))}
        </div>
        <div style={{...card,padding:"16px"}}>
          <div style={{...slabel,marginBottom:"14px"}}>Technology Stack</div>
          <table style={{width:"100%",borderCollapse:"collapse"}}>
            <tbody>
              {STACK.map(([layer,value])=>(
                <tr key={layer} style={{borderBottom:"1px solid #f1f5f9"}}>
                  <td style={{padding:"8px 0",fontSize:"13px",color:"#64748b",paddingRight:"16px",whiteSpace:"nowrap"}}>{layer}</td>
                  <td style={{padding:"8px 0",fontSize:"13px",color:"#1e293b",fontFamily:"monospace"}}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Stats grid */}
      <div style={{...card,padding:"16px"}}>
        <div style={{...slabel,marginBottom:"14px"}}>Project Statistics</div>
        <div style={{display:"grid",gridTemplateColumns:gridCols(4,2,2,isMobile,isTablet),gap:"10px"}}>
          {STATS.map(([value,label])=>(
            <div key={label} style={{padding:"14px 16px",background:"#f8fafc",border:"1px solid #e2e8f0",borderRadius:"4px"}}>
              <div style={{fontSize:"22px",fontWeight:700,color:"#1d4ed8",marginBottom:"4px",fontVariantNumeric:"tabular-nums"}}>{value}</div>
              <div style={{fontSize:"12px",color:"#64748b"}}>{label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}