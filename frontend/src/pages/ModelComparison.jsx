import React,{useState,useEffect} from "react";
import {ResponsiveContainer,BarChart,Bar,XAxis,YAxis,Tooltip,CartesianGrid,Cell} from "recharts";
import { useSimMode, API, STATIC_MODELS } from "../hooks/useSimMode";
import { useIsMobile, useIsTablet, gridCols } from "../hooks/useIsMobile";

const TT={backgroundColor:"#fff",border:"1px solid #e2e8f0",borderRadius:"4px",fontSize:"12px",color:"#1e293b"};
const TICK={fill:"#94a3b8",fontSize:11};
const card={background:"#fff",border:"1px solid #e2e8f0",borderRadius:"6px",boxShadow:"0 1px 3px rgba(0,0,0,.04)"};
const slabel={fontSize:"11px",fontWeight:600,letterSpacing:"0.06em",textTransform:"uppercase",color:"#64748b"};
const MCOLORS={"YOLOv8s":"#1d4ed8","Faster RCNN":"#f97316","RetinaNet":"#16a34a"};
const CLASSES=["car","bus","truck","taxi","microbus","motorcycle","bicycle"];

function pct(v){return v==null?"—":`${(v*100).toFixed(1)}%`;}
function vcol(v){if(v==null)return"#94a3b8";if(v>0.5)return"#15803d";if(v>0.2)return"#1d4ed8";return"#dc2626";}

export default function ModelComparison(){
  const mode = useSimMode();
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const [models,setModels]=useState([]);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState(null);

  useEffect(()=>{
    if(mode==="video"){
      setModels(STATIC_MODELS.models);
      setLoading(false);
      return;
    }
    fetch(`${API}/api/models/comparison`).then(r=>r.json())
      .then(d=>{setModels(d.models);setLoading(false);})
      .catch(()=>{
        setModels(STATIC_MODELS.models);
        setLoading(false);
      });
  },[mode]);

  const barData=models.map(m=>({model:m.name,mAP:m.mAP50||0}));

  return(
    <div style={{paddingTop:"24px"}}>
      <div style={{paddingBottom:"16px",borderBottom:"1px solid #e2e8f0",marginBottom:"24px",display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
        <div>
          <h1 style={{margin:0,fontSize:"20px",fontWeight:700,color:"#0f172a"}}>Detection Model Comparison</h1>
          <p style={{margin:"4px 0 0",fontSize:"13px",color:"#64748b"}}>YOLOv8s · Faster R-CNN · RetinaNet — 7 vehicle classes</p>
        </div>
        <div style={{display:"flex",gap:"6px"}}>
          {[["3 Models","#dbeafe","#1d4ed8"],["7 Classes","#f1f5f9","#374151"]].map(([t,bg,c])=>(
            <span key={t} style={{padding:"4px 10px",borderRadius:"99px",fontSize:"12px",fontWeight:500,background:bg,color:c}}>{t}</span>
          ))}
        </div>
      </div>

      {loading
        ? <div style={{padding:"48px",textAlign:"center",color:"#64748b",fontSize:"14px"}}>Loading model data...</div>
        : <>
          <div style={{display:"grid",gridTemplateColumns:gridCols(3,2,1,isMobile,isTablet),gap:"14px",marginBottom:"16px"}}>
            {models.map(m=>{
              const color=MCOLORS[m.name]||"#64748b";
              const isPrimary=m.name==="YOLOv8s";
              return(
                <div key={m.name} style={{...card,padding:"20px",borderLeft:`4px solid ${color}`,
                  ...(isPrimary?{boxShadow:"0 0 0 1px #bfdbfe, 0 1px 3px rgba(0,0,0,.04)"}:{})}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:"16px"}}>
                    <div>
                      <div style={{fontSize:"17px",fontWeight:700,color}}>{m.name}</div>
                      {isPrimary&&<span style={{display:"inline-block",marginTop:"4px",padding:"2px 8px",borderRadius:"99px",fontSize:"11px",fontWeight:600,background:"#dbeafe",color:"#1d4ed8"}}> Selected</span>}
                    </div>
                  </div>
                  {[
                    ["mAP@0.5",      pct(m.mAP50)],
                    ["mAP@0.5:0.95", pct(m.mAP50_95)],
                    ["Precision",    pct(m.precision)],
                    ["Recall",       pct(m.recall)],
                    ["FPS",          m.fps?`${m.fps} fps`:"—"],
                    ["Inference",    m.inference_ms?`${m.inference_ms} ms`:"—"],
                  ].map(([label,value])=>(
                    <div key={label} style={{display:"flex",justifyContent:"space-between",padding:"7px 0",borderBottom:"1px solid #f1f5f9"}}>
                      <span style={{fontSize:"13px",color:"#64748b"}}>{label}</span>
                      <span style={{fontSize:"13px",fontWeight:600,color:"#0f172a",fontFamily:"monospace"}}>{value}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          <div style={{...card,marginBottom:"16px"}}>
            <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9",display:"flex",justifyContent:"space-between"}}>
              <span style={slabel}>Per-Class AP@0.5</span>
              <span style={{fontSize:"12px",color:"#94a3b8"}}>Higher is better</span>
            </div>
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse"}}>
                <thead>
                  <tr style={{background:"#f8fafc"}}>
                    <th style={{padding:"10px 14px",textAlign:"left",fontSize:"12px",fontWeight:600,color:"#64748b",borderBottom:"1px solid #e2e8f0"}}>Vehicle Class</th>
                    {models.map(m=>(<th key={m.name} style={{padding:"10px 14px",textAlign:"left",fontSize:"12px",fontWeight:600,color:MCOLORS[m.name],borderBottom:"1px solid #e2e8f0"}}>{m.name}</th>))}
                    <th style={{padding:"10px 14px",textAlign:"left",fontSize:"12px",fontWeight:600,color:"#64748b",borderBottom:"1px solid #e2e8f0"}}>Best</th>
                  </tr>
                </thead>
                <tbody>
                  {CLASSES.map(cls=>{
                    const vals=models.map(m=>m.per_class_ap?.[cls]);
                    const best=Math.max(...vals.filter(v=>v!=null),0);
                    return(
                      <tr key={cls} style={{borderBottom:"1px solid #f1f5f9"}}>
                        <td style={{padding:"9px 14px",fontSize:"14px",fontWeight:600,color:"#1e293b",textTransform:"capitalize"}}>{cls}</td>
                        {vals.map((v,i)=>(
                          <td key={i} style={{padding:"9px 14px",fontSize:"13px",fontFamily:"monospace",fontWeight:v===best&&v!=null?700:400,color:vcol(v)}}>{pct(v)}</td>
                        ))}
                        <td style={{padding:"9px 14px",fontSize:"13px",fontFamily:"monospace",fontWeight:700,color:"#1d4ed8"}}>{best>0?pct(best):"—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{...card,padding:"16px"}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:"16px"}}>
              <span style={{fontSize:"14px",fontWeight:600,color:"#0f172a"}}>Overall mAP@0.5 Comparison</span>
              <span style={{fontSize:"12px",color:"#94a3b8"}}>Higher is better</span>
            </div>
            <div style={{height:"220px"}}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} margin={{left:-20,right:8,top:4,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false}/>
                  <XAxis dataKey="model" tick={TICK} tickLine={false}/>
                  <YAxis tick={TICK} tickLine={false} domain={[0,0.8]} tickFormatter={v=>`${(v*100).toFixed(0)}%`}/>
                  <Tooltip contentStyle={TT} formatter={v=>[`${(v*100).toFixed(1)}%`,"mAP@0.5"]}/>
                  <Bar dataKey="mAP" radius={[3,3,0,0]}>
                    {barData.map((d,i)=><Cell key={i} fill={MCOLORS[d.model]||"#94a3b8"}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      }
    </div>
  );
}