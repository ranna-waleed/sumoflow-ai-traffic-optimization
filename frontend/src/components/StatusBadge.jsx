import React from "react";

const variants = {
  default: "bg-white/5 text-[#94a3b8] border-white/10",
  success: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  warning: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  danger: "bg-red-500/15 text-red-300 border-red-500/30",
  primary: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
};

function StatusBadge({ label, variant = "default" }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium font-mono border ${variants[variant]}`}
    >
      {label}
    </span>
  );
}

export default StatusBadge;
