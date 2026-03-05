import React from "react";
import {
  Cpu,
  MapPin,
  Network,
  Brain,
  Database,
  BarChart3,
  Image,
  Tag,
  Bot,
  RefreshCw,
} from "lucide-react";

const team = [
  { name: "Rana", role: "YOLO + LSTM + Frontend", color: "#818cf8" },
  { name: "Roaa", role: "RetinaNet + Visualization", color: "#34d399" },
  { name: "Mariam", role: "Faster RCNN + Optimizer", color: "#f97316" },
];

const stack = [
  { label: "Simulation", value: "SUMO + TraCI" },
  { label: "Detection", value: "YOLO / RetinaNet / Faster RCNN" },
  { label: "Optimizer", value: "DQN (Deep Q-Network)" },
  { label: "Backend", value: "FastAPI (Python)" },
  { label: "Frontend", value: "React.js" },
  { label: "Training", value: "Google Colab (T4 GPU)" },
];

const pipeline = [
  { label: "SUMO Simulation", icon: MapPin, desc: "Microscopic traffic simulation of El-Tahrir Square." },
  { label: "Detection Model", icon: Cpu, desc: "Vehicle detection via YOLO / RetinaNet / Faster RCNN." },
  { label: "DQN Optimizer", icon: Network, desc: "Learns optimal signal timings to minimize delay and CO₂." },
  { label: "TraCI Controller", icon: Database, desc: "Applies actions to SUMO via TraCI API." },
  { label: "Results & Visualization", icon: BarChart3, desc: "KPIs, charts, and live views." },
];

const stats = [
  { icon: Image, label: "1,800 Training Images" },
  { icon: Tag, label: "7 Vehicle Classes" },
  { icon: Bot, label: "3 Detection Models" },
  { icon: RefreshCw, label: "60 Training Epochs" },
];

function About() {
  return (
    <div className="space-y-6 pt-4">
      {/* Page header */}
      <div>
        <h1 className="text-xl font-semibold text-white">About</h1>
        <p className="text-sm text-[#94a3b8] mt-0.5">
          El-Tahrir Square AI Traffic Optimization — Graduation Project
        </p>
      </div>

      {/* Project description */}
      <div className="card p-6 border-indigo-500/20">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-5 h-5 text-indigo-400" />
          <h2 className="text-base font-semibold text-white">SUMOFLOW — AI Traffic Optimization</h2>
        </div>
        <p className="text-sm text-[#94a3b8] leading-relaxed max-w-3xl">
          An AI-based adaptive traffic control system that uses vehicle detection and deep
          reinforcement learning to optimize traffic light timing in a SUMO simulation of
          El-Tahrir Square, Cairo — reducing CO₂ emissions and vehicle waiting time compared to
          static traffic lights.
        </p>
      </div>

      {/* Team & Tech stack */}
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Team Members</h3>
          <div className="space-y-3">
            {team.map((m) => (
              <div
                key={m.name}
                className="flex items-center justify-between px-4 py-3 rounded-xl bg-white/5 border border-white/5"
              >
                <span className="font-medium text-white">{m.name}</span>
                <span className="text-sm font-mono" style={{ color: m.color }}>
                  {m.role}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Technology Stack</h3>
          <div className="space-y-3">
            {stack.map((row) => (
              <div
                key={row.label}
                className="flex items-center justify-between py-2 border-b border-white/5 last:border-0"
              >
                <span className="text-sm text-[#94a3b8]">{row.label}</span>
                <span className="text-sm font-mono text-indigo-300">{row.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System architecture */}
      <div className="card p-5">
        <h3 className="text-sm font-semibold text-white mb-4">System Architecture</h3>
        <div className="hidden md:flex items-stretch gap-4 overflow-x-auto pb-2">
          {pipeline.map((step, idx) => {
            const Icon = step.icon;
            return (
              <React.Fragment key={step.label}>
                <div className="shrink-0 flex-1 min-w-[120px] card p-4 bg-white/5 border-white/5 text-center">
                  <div className="flex justify-center mb-2">
                    <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-500/15 border border-indigo-500/30">
                      <Icon className="w-5 h-5 text-indigo-400" />
                    </div>
                  </div>
                  <div className="text-xs font-medium text-white mb-1">{step.label}</div>
                  <div className="text-[11px] text-[#94a3b8] leading-tight">{step.desc}</div>
                </div>
                {idx < pipeline.length - 1 && (
                  <div className="shrink-0 flex items-center text-[#64748b]">→</div>
                )}
              </React.Fragment>
            );
          })}
        </div>
        <div className="md:hidden space-y-3">
          {pipeline.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.label}
                className="flex items-center gap-4 p-3 rounded-xl bg-white/5 border border-white/5"
              >
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-500/15 border border-indigo-500/30 shrink-0">
                  <Icon className="w-5 h-5 text-indigo-400" />
                </div>
                <div>
                  <div className="text-xs font-medium text-white">{step.label}</div>
                  <div className="text-[11px] text-[#94a3b8]">{step.desc}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Project stats */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map(({ icon: Icon, label }) => (
          <div
            key={label}
            className="card p-5 flex flex-col items-center justify-center text-center"
          >
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-indigo-500/15 border border-indigo-500/25 mb-3">
              <Icon className="w-6 h-6 text-indigo-400" />
            </div>
            <div className="text-xs font-medium text-[#e2e8f0]">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default About;
