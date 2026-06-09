// frontend/src/hooks/useSimMode.js
import { useState, useEffect } from "react";

const API = "http://127.0.0.1:8000";

export const VIDEO_MAP = {
  morning_rush: "/videos/morning_rush.mp4",
  evening_rush: "/videos/evening_rush.mp4",
  midday:       "/videos/midday.mp4",
  night:        "/videos/night.mp4",
};

export const DQN_VIDEO_MAP = {
  morning_rush: "/videos/dqn_morning_rush.mp4",
  evening_rush: "/videos/dqn_evening_rush.mp4",
  midday:       "/videos/dqn_midday.mp4",
  night:        "/videos/dqn_night.mp4",
};

// ── Static fallback data (used in video/deployment mode) ──────────

export const STATIC_METRICS = {
  morning_rush: {
    profile:"morning_rush", time_period:"8AM - 10AM",
    total_steps:750, peak_vehicles:512, avg_vehicles:398,
    avg_wait_s:626.6, max_wait_s:1847.2,
    peak_co2_mg:2850000, avg_co2_mg:1920000,
    timeseries: Array.from({length:75},(_,i)=>({
      step:i*10, vehicles:Math.round(250+i*3.5+Math.sin(i*0.4)*30),
      avg_wait:Math.round(20+i*8+Math.sin(i*0.3)*15),
      co2:Math.round(800000+i*28000+Math.sin(i*0.5)*80000),
    }))
  },
  evening_rush: {
    profile:"evening_rush", time_period:"4PM - 7PM",
    total_steps:900, peak_vehicles:589, avg_vehicles:441,
    avg_wait_s:632.1, max_wait_s:1923.4,
    peak_co2_mg:3100000, avg_co2_mg:2100000,
    timeseries: Array.from({length:90},(_,i)=>({
      step:i*10, vehicles:Math.round(270+i*3.8+Math.sin(i*0.35)*35),
      avg_wait:Math.round(22+i*7+Math.sin(i*0.28)*18),
      co2:Math.round(900000+i*26000+Math.sin(i*0.45)*90000),
    }))
  },
  midday: {
    profile:"midday", time_period:"12PM - 2PM",
    total_steps:720, peak_vehicles:463, avg_vehicles:351,
    avg_wait_s:629.1, max_wait_s:1764.3,
    peak_co2_mg:2650000, avg_co2_mg:1780000,
    timeseries: Array.from({length:72},(_,i)=>({
      step:i*10, vehicles:Math.round(220+i*3.2+Math.sin(i*0.38)*28),
      avg_wait:Math.round(18+i*8.5+Math.sin(i*0.32)*14),
      co2:Math.round(750000+i*25000+Math.sin(i*0.42)*75000),
    }))
  },
  night: {
    profile:"night", time_period:"10PM - 12AM",
    total_steps:420, peak_vehicles:87, avg_vehicles:52,
    avg_wait_s:24.3, max_wait_s:198.7,
    peak_co2_mg:380000, avg_co2_mg:210000,
    timeseries: Array.from({length:42},(_,i)=>({
      step:i*10, vehicles:Math.round(30+i*1.2+Math.sin(i*0.5)*12),
      avg_wait:Math.round(5+i*0.4+Math.sin(i*0.4)*4),
      co2:Math.round(120000+i*6000+Math.sin(i*0.6)*20000),
    }))
  },
};

export const STATIC_DQN_RESULTS = {
  avg_wait_fixed:   478.0,
  avg_wait_dqn:     40.5,
  improvement_pct:  91.5,
  co2_improvement_pct: 90.5,
  fixed_co2_mg:     12500000000,
  dqn_co2_mg:       1187500000,
  episodes:         350,
  episode_waits:    Array.from({length:350},(_,i)=>Math.max(15, 600-i*1.7+Math.sin(i*0.3)*40)),
  profiles: {
    morning_rush: { fixed_wait_s:626.6, dqn_wait_s:46.7,  wait_improvement:92.6, fixed_co2_mg:4200000000, dqn_co2_mg:408000000,  co2_improvement:90.3, throughput_delta:-97,  kpi_wait_pass:true, kpi_co2_pass:true },
    midday:       { fixed_wait_s:629.1, dqn_wait_s:59.2,  wait_improvement:90.6, fixed_co2_mg:3800000000, dqn_co2_mg:380000000,  co2_improvement:90.1, throughput_delta:-61,  kpi_wait_pass:true, kpi_co2_pass:true },
    evening_rush: { fixed_wait_s:632.1, dqn_wait_s:46.1,  wait_improvement:92.7, fixed_co2_mg:4500000000, dqn_co2_mg:450000000,  co2_improvement:90.0, throughput_delta:138,  kpi_wait_pass:true, kpi_co2_pass:true },
    night:        { fixed_wait_s:24.3,  dqn_wait_s:10.2,  wait_improvement:57.9, fixed_co2_mg:380000000,  dqn_co2_mg:32680000,   co2_improvement:91.4, throughput_delta:0,    kpi_wait_pass:true, kpi_co2_pass:true },
  },
};

export const STATIC_MODELS = {
  models: [
    {
      name:"YOLOv8s", status:"complete",
      mAP50:0.712, mAP50_95:0.489, precision:0.748, recall:0.681, fps:42, inference_ms:24,
      per_class_ap:{ car:0.821, bus:0.756, truck:0.698, taxi:0.743, microbus:0.672, motorcycle:0.634, bicycle:0.589 },
    },
    {
      name:"Faster RCNN", status:"complete",
      mAP50:0.681, mAP50_95:0.461, precision:0.712, recall:0.649, fps:14, inference_ms:71,
      per_class_ap:{ car:0.798, bus:0.731, truck:0.672, taxi:0.718, microbus:0.641, motorcycle:0.598, bicycle:0.551 },
    },
    {
      name:"RetinaNet", status:"complete",
      mAP50:0.658, mAP50_95:0.442, precision:0.689, recall:0.623, fps:18, inference_ms:56,
      per_class_ap:{ car:0.776, bus:0.708, truck:0.651, taxi:0.694, microbus:0.618, motorcycle:0.571, bicycle:0.524 },
    },
  ],
};

// ── Hook ────────────────────────────────────────────────────────

export function useSimMode() {
  const [mode, setMode] = useState("checking");

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API}/health`, {
          signal: AbortSignal.timeout(3000),
        });
        setMode(res.ok ? "live" : "video");
      } catch {
        setMode("video");
      }
    };
    check();
  }, []);

  return mode;
}

export { API };

// Static traffic signal states for video mode
export const STATIC_SIGNALS = {
  "315744796": "GGGgrrrrGGGgrrrr",
  "96621100":  "GGrrGGrr",
  "2031414903":"GGGGrrrr",
  "2031414899":"GyrrGyrr",
  "6288771431":"GGrrGGrr",
  "271064234": "GGGGrrrr",
  "315743335": "GrrrGrrr",
};

// Static direction counts per profile
export const STATIC_DIRECTIONS = {
  morning_rush: { north:142, south:98,  east:167, west:134 },
  evening_rush: { north:178, south:121, east:143, west:156 },
  midday:       { north:112, south:87,  east:134, west:108 },
  night:        { north:24,  south:18,  east:29,  west:22  },
};

// Static BiLSTM predictions per profile
export const STATIC_BILSTM = {
  morning_rush: { north:158, south:112, east:184, west:149 },
  evening_rush: { north:195, south:138, east:161, west:172 },
  midday:       { north:126, south:98,  east:148, west:121 },
  night:        { north:28,  south:21,  east:33,  west:26  },
};