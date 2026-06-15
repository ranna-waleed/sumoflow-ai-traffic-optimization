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

// Static fallback data (used in video/deployment mode)

export const STATIC_METRICS = {
  morning_rush: {
    profile:"morning_rush", time_period:"8AM - 10AM",
    total_steps:750, peak_vehicles:3482, avg_vehicles:2400,
    avg_wait_s:626.6, max_wait_s:1848.5,
    avg_time_loss_s: 715.5,
    total_co2_mg: 4581366000.9,
    throughput: 3482,
    peak_co2_mg:6108488, avg_co2_mg:6108488,
    timeseries: Array.from({length:75},(_,i)=>({
      step:i*10, vehicles:Math.round(180+i*1.7+Math.sin(i*0.4)*20),
      avg_wait:Math.round(20+i*8+Math.sin(i*0.3)*15),
      co2:Math.round(4000000+i*80000+Math.sin(i*0.5)*400000),
    }))
  },
  evening_rush: {
    profile:"evening_rush", time_period:"4PM - 7PM",
    total_steps:900, peak_vehicles:5711, avg_vehicles:3800,
    avg_wait_s:632.1, max_wait_s:1864.7,
    avg_time_loss_s: 720.9,
    total_co2_mg: 7460325091.0,
    throughput: 5711,
    peak_co2_mg:8289250, avg_co2_mg:8289250,
    timeseries: Array.from({length:90},(_,i)=>({
      step:i*10, vehicles:Math.round(200+i*2.2+Math.sin(i*0.35)*25),
      avg_wait:Math.round(22+i*7+Math.sin(i*0.28)*18),
      co2:Math.round(5500000+i*85000+Math.sin(i*0.45)*500000),
    }))
  },
  midday: {
    profile:"midday", time_period:"12PM - 2PM",
    total_steps:720, peak_vehicles:2937, avg_vehicles:2000,
    avg_wait_s:629.1, max_wait_s:1855.8,
    avg_time_loss_s: 719.4,
    total_co2_mg: 4109020468.3,
    throughput: 2937,
    peak_co2_mg:5706973, avg_co2_mg:5706973,
    timeseries: Array.from({length:72},(_,i)=>({
      step:i*10, vehicles:Math.round(150+i*1.9+Math.sin(i*0.38)*18),
      avg_wait:Math.round(18+i*8.5+Math.sin(i*0.32)*14),
      co2:Math.round(3800000+i*75000+Math.sin(i*0.42)*400000),
    }))
  },
  night: {
    profile:"night", time_period:"10PM - 12AM",
    total_steps:420, peak_vehicles:882, avg_vehicles:600,
    avg_wait_s:24.3, max_wait_s:71.7,
    avg_time_loss_s: 36.2,
    total_co2_mg: 135595522.8,
    throughput: 882,
    peak_co2_mg:322846, avg_co2_mg:322846,
    timeseries: Array.from({length:42},(_,i)=>({
      step:i*10, vehicles:Math.round(25+i*0.8+Math.sin(i*0.5)*8),
      avg_wait:Math.round(5+i*0.4+Math.sin(i*0.4)*4),
      co2:Math.round(200000+i*3000+Math.sin(i*0.6)*50000),
    }))
  },
};

export const STATIC_DQN_RESULTS = {
  avg_wait_fixed:      478.0,
  avg_wait_dqn:        341.6,
  improvement_pct:     25.2,
  co2_improvement_pct: 17.9,
  fixed_co2_mg:        16286307082.97,
  dqn_co2_mg:          11485086252.8,
  episodes:            350,
  episode_waits:       Array.from({length:350},(_,i)=>Math.max(15, 600-i*1.7+Math.sin(i*0.3)*40)),
  profiles: {
    morning_rush: {
      fixed_wait_s:     626.6,
      dqn_wait_s:       84.5,
      wait_improvement: 86.5,
      fixed_co2_mg:     4581366000.9,
      dqn_co2_mg:       1897381341.5,
      co2_improvement:  58.6,
      throughput_delta: 1732,
      kpi_wait_pass:    true,
      kpi_co2_pass:     true,
    },
    midday: {
      fixed_wait_s:     629.1,
      dqn_wait_s:       738.2,
      wait_improvement: -17.3,
      fixed_co2_mg:     4109020468.3,
      dqn_co2_mg:       4693491364.7,
      co2_improvement:  -14.2,
      throughput_delta: -171,
      kpi_wait_pass:    false,
      kpi_co2_pass:     false,
    },
    evening_rush: {
      fixed_wait_s:     632.1,
      dqn_wait_s:       522.8,
      wait_improvement: 17.4,
      fixed_co2_mg:     7460325091.0,
      dqn_co2_mg:       4746263752.8,
      co2_improvement:  36.3,
      throughput_delta: -1937,
      kpi_wait_pass:    true,
      kpi_co2_pass:     true,
    },
    night: {
      fixed_wait_s:     24.3,
      dqn_wait_s:       20.8,
      wait_improvement: 14.2,
      fixed_co2_mg:     135595522.8,
      dqn_co2_mg:       147949793.8,
      co2_improvement:  -9.1,
      throughput_delta: -1,
      kpi_wait_pass:    true,
      kpi_co2_pass:     false,
    },
  },
};

export const STATIC_MODELS = {
  models: [
    {
      name:"YOLOv8s", status:"complete",
      mAP50:0.483, mAP50_95:0.344, precision:0.818, recall:0.411, fps:208, inference_ms:4.8,
      per_class_ap:{ car:0.527, bus:0.972, truck:0.714, taxi:0.517, microbus:0.506, motorcycle:0.059, bicycle:0.049 },
    },
    {
      name:"Faster RCNN", status:"complete",
      mAP50:0.470, mAP50_95:0.342, precision:0.946, recall:0.374, fps:9, inference_ms:109,
      per_class_ap:{ car:0.312, bus:0.797, truck:0.496, taxi:0.308, microbus:0.275, motorcycle:0.107, bicycle:0.097 },
    },
    {
      name:"RetinaNet", status:"complete",
      mAP50:0.417, mAP50_95:0.288, precision:0, recall:0.374, fps:13, inference_ms:75,
      per_class_ap:{ car:0.241, bus:0.725, truck:0.482, taxi:0.237, microbus:0.258, motorcycle:0.039, bicycle:0.037 },
    },
  ],
};

// Hook

export function useSimMode() {
  const [mode, setMode] = useState("checking");

  useEffect(() => {
    const check = async () => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5000);
      try {
        const res = await fetch(`${API}/health`, { signal: controller.signal });
        setMode(res.ok ? "live" : "video");
      } catch {
        setMode("video");
      } finally {
        clearTimeout(timer);
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