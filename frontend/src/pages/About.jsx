import { useIsMobile, useIsTablet } from "../hooks/useIsMobile";
import React from "react";

const card   = { background:"#fff", border:"1px solid #e2e8f0", borderRadius:"6px", boxShadow:"0 1px 3px rgba(0,0,0,.04)" };
const slabel = { fontSize:"11px", fontWeight:600, letterSpacing:"0.06em", textTransform:"uppercase", color:"#64748b" };

const TEAM = [
  {
    name: "Rana Waleed", id: "202201737",
    rows: [
      ["Detection",  "YOLOv8s training & evaluation (mAP=0.478, 208 FPS)"],
      ["Frontend",   "Dashboard, Before/After, About pages"],
      ["SUMO",       "Map extraction (OSM -> netconvert) + SSM safety config"],
      ["DQN",        "DQN live simulation viewer + Before/After page"],
    ]
  },
  {
    name: "Roaa Raafat", id: "202202079",
    rows: [
      ["Detection",  "RetinaNet training & evaluation (mAP=0.413)"],
      ["Prediction", "BiLSTM traffic predictor training (MAE=3.15)"],
      ["Frontend",   "Live Simulation page"],
      ["SUMO",       "TAZ definition, SUMO config, output devices, visualization"],
    ]
  },
  {
    name: "Mariam Alhaj", id: "202200529",
    rows: [
      ["Detection",  "Faster R-CNN training & evaluation (mAP=0.470)"],
      ["Backend",    "FastAPI backend: 15+ REST endpoints"],
      ["Frontend",   "Model Comparison page"],
      ["SUMO",       "Network validation, OD matrix generation, route files"],
    ]
  },
  {
    name: "All Team Members", id: null,
    rows: [
      ["SUMO", "El-Tahrir Square simulation setup and testing"],
      ["DQN",  "Environment, agent, training, evaluation"],
    ]
  },
];

const STACK = [
  ["Simulation",  "Eclipse SUMO 1.24 + TraCI API"],
  ["Detection",   "YOLOv8s · Faster R-CNN · RetinaNet"],
  ["Prediction",  "BiLSTM (2×128 hidden units)"],
  ["Optimizer",   "Dueling Double DQN: 7 agents"],
  ["Backend",     "FastAPI + Python 3.12"],
  ["Frontend",    "React 18 + Recharts"],
  ["Training",    "350 episodes, 4 profiles"],
];

function MemberCard({ member }) {
  return (
    <div style={{ ...card, overflow:"hidden", flex:1 }}>
      <div style={{ padding:"10px 16px", background:"#0f2644", borderBottom:"1px solid #1e3a5f" }}>
        <div style={{ fontSize:"13px", fontWeight:700, color:"#fff" }}>{member.name}</div>
        {member.id && <div style={{ fontSize:"11px", color:"#94a3b8", marginTop:"2px" }}>{member.id}</div>}
      </div>
      <table style={{ width:"100%", borderCollapse:"collapse" }}>
        <thead>
          <tr style={{ background:"#f8fafc" }}>
            <th style={{ padding:"7px 14px", textAlign:"left", fontSize:"11px", fontWeight:600, color:"#1d4ed8", borderBottom:"1px solid #e2e8f0", width:"110px" }}>Component</th>
            <th style={{ padding:"7px 14px", textAlign:"left", fontSize:"11px", fontWeight:600, color:"#1d4ed8", borderBottom:"1px solid #e2e8f0" }}>Responsibility</th>
          </tr>
        </thead>
        <tbody>
          {member.rows.map(([comp, resp], i) => (
            <tr key={comp} style={{ borderBottom: i < member.rows.length-1 ? "1px solid #f1f5f9" : "none" }}>
              <td style={{ padding:"8px 14px", fontSize:"12px", fontWeight:600, color:"#374151", whiteSpace:"nowrap" }}>{comp}</td>
              <td style={{ padding:"8px 14px", fontSize:"12px", color:"#64748b" }}>{resp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function About() {
  const isMobile = useIsMobile();

  return (
    <div style={{ paddingTop:"24px" }}>

      {/* Header */}
      <div style={{ paddingBottom:"16px", borderBottom:"1px solid #e2e8f0", marginBottom:"24px", display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
        <div>
          <h1 style={{ margin:0, fontSize:"20px", fontWeight:700, color:"#0f172a" }}>About — SUMOFlow AI</h1>
          <p style={{ margin:"4px 0 0", fontSize:"13px", color:"#64748b" }}>El-Tahrir Square Traffic Optimization · Graduation Project · Zewail City</p>
        </div>
        <span style={{ padding:"4px 12px", borderRadius:"99px", fontSize:"12px", fontWeight:500, background:"#dbeafe", color:"#1d4ed8" }}>v1.0.0</span>
      </div>

      {/* Overview */}
      <div style={{ ...card, padding:"20px 24px", marginBottom:"20px", borderLeft:"4px solid #1d4ed8" }}>
        <p style={{ margin:0, fontSize:"14px", color:"#374151", lineHeight:1.75 }}>
          SUMOFlow AI replaces fixed-time traffic signals at El-Tahrir Square, Cairo with an adaptive
          system using computer vision, flow prediction, and deep reinforcement learning. Seven DQN
          agents control 7 junctions simultaneously, reducing average waiting time by up to
          <strong style={{ color:"#15803d" }}> 92.7%</strong> and CO₂ emissions by over
          <strong style={{ color:"#15803d" }}> 90%</strong> compared to the fixed-time baseline,
          verified across 4 daily traffic profiles.
        </p>
      </div>

      {/* Team Responsibilities */}
      <div style={{ ...card, marginBottom:"20px" }}>
        <div style={{ padding:"12px 16px", borderBottom:"1px solid #f1f5f9" }}>
          <span style={slabel}>Team Responsibilities</span>
        </div>
        <div style={{ padding:"16px", display:"flex", flexDirection:"column", gap:"12px" }}>

          {/* Row 1: Rana + Roaa */}
          <div style={{ display:"flex", gap:"12px", flexDirection: isMobile ? "column" : "row" }}>
            <MemberCard member={TEAM[0]} />
            <MemberCard member={TEAM[1]} />
          </div>

          {/* Row 2: Mariam + All Team */}
          <div style={{ display:"flex", gap:"12px", flexDirection: isMobile ? "column" : "row" }}>
            <MemberCard member={TEAM[2]} />
            <MemberCard member={TEAM[3]} />
          </div>

        </div>
      </div>

      {/* Technology Stack */}
      <div style={{ ...card, padding:"16px" }}>
        <div style={{ ...slabel, marginBottom:"12px" }}>Technology Stack</div>
        <div style={{ display:"grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap:"0" }}>
          {STACK.map(([layer, value], i) => (
            <div key={layer} style={{ display:"flex", gap:"12px", padding:"8px 0", borderBottom:"1px solid #f1f5f9" }}>
              <span style={{ fontSize:"13px", color:"#64748b", minWidth:"90px", flexShrink:0 }}>{layer}</span>
              <span style={{ fontSize:"13px", color:"#1e293b", fontFamily:"monospace" }}>{value}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}