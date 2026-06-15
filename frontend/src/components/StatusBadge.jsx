import React from "react";

const variants = {
  default: "bg-slate-100 text-slate-500 border-slate-200",
  success: "bg-green-50 text-green-700 border-green-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  danger:  "bg-red-50 text-red-700 border-red-200",
  primary: "bg-blue-50 text-blue-700 border-blue-200",
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
