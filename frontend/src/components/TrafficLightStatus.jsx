import React from "react";

const NAMES = {
  "315744796":"N-Trunk Entry","96621100":"Ring N-Entry",
  "2031414903":"Ring W-Entry","2031414899":"S-Gate 1",
  "6288771431":"S-Gate 2","271064234":"E-Exit 1","315743335":"E-Exit 2",
};

function phaseInfo(state=""){
  const s = state.toLowerCase();
  if(s.includes("g")) return {bg:"#dcfce7",color:"#15803d",label:"Green"};
  if(s.includes("y")) return {bg:"#fef3c7",color:"#b45309",label:"Yellow"};
  return {bg:"#fee2e2",color:"#dc2626",label:"Red"};
}

const card = {background:"#fff",border:"1px solid #e2e8f0",borderRadius:"6px",boxShadow:"0 1px 3px rgba(0,0,0,.04)"};
const slabel = {fontSize:"11px",fontWeight:600,letterSpacing:"0.06em",textTransform:"uppercase",color:"#64748b"};

export default function TrafficLightStatus({trafficLights,typeCounts}){
  const hasTL = trafficLights && Object.keys(trafficLights).length>0;
  return(
    <div style={card}>
      <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9"}}>
        <span style={slabel}>Traffic Signal Status</span>
      </div>
      {!hasTL
        ? <div style={{padding:"24px 16px",textAlign:"center",color:"#94a3b8",fontSize:"13px"}}>No live data — start simulation</div>
        : <table style={{width:"100%",borderCollapse:"collapse"}}>
            <thead>
              <tr style={{background:"#f8fafc"}}>
                {["Junction","Phase","State"].map(h=>(
                  <th key={h} style={{padding:"8px 12px",textAlign:h==="State"?"right":"left",fontSize:"11px",fontWeight:600,color:"#64748b",borderBottom:"1px solid #e2e8f0"}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(trafficLights).map(([id,state])=>{
                const {bg,color,label}=phaseInfo(state);
                return(
                  <tr key={id} style={{borderBottom:"1px solid #f1f5f9"}}>
                    <td style={{padding:"8px 12px",fontSize:"12px",fontWeight:600,color:"#1e293b"}}>{NAMES[id]||id}</td>
                    <td style={{padding:"8px 12px",fontSize:"11px",fontFamily:"monospace",color:"#64748b"}}>{state?.slice(0,8)||"—"}</td>
                    <td style={{padding:"8px 12px",textAlign:"right"}}>
                      <span style={{background:bg,color,padding:"2px 8px",borderRadius:"3px",fontSize:"11px",fontWeight:600}}>{label}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
      }
      {typeCounts && Object.keys(typeCounts).length>0 && (
        <div style={{borderTop:"1px solid #f1f5f9",padding:"12px 16px"}}>
          <div style={{...slabel,marginBottom:"8px"}}>Vehicle Mix</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:"6px"}}>
            {Object.entries(typeCounts).map(([t,c])=>(
              <span key={t} style={{background:"#f1f5f9",border:"1px solid #e2e8f0",padding:"3px 8px",borderRadius:"3px",fontSize:"12px",color:"#374151",fontWeight:500}}>
                {t}: <strong style={{color:"#1d4ed8"}}>{c}</strong>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}