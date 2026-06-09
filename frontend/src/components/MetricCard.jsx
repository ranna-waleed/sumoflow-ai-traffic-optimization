import React from "react";

export default function MetricCard({ label, value, unit, accent = "#1d4ed8" }) {
  return (
    <div style={{
      background: "#fff",
      border: "1px solid #e2e8f0",
      borderTop: `3px solid ${accent}`,
      borderRadius: "6px",
      padding: "16px 20px",
      boxShadow: "0 1px 3px rgba(0,0,0,.04)",
    }}>
      <div style={{
        fontSize: "11px", fontWeight: 600,
        letterSpacing: "0.06em", textTransform: "uppercase",
        color: "#64748b", marginBottom: "8px",
      }}>
        {label}
      </div>
      <div style={{
        fontSize: "28px", fontWeight: 700,
        color: "#0f172a", lineHeight: 1,
        fontVariantNumeric: "tabular-nums",
      }}>
        {value ?? "—"}
      </div>
      {unit && (
        <div style={{ fontSize: "12px", color: "#94a3b8", marginTop: "6px" }}>
          {unit}
        </div>
      )}
    </div>
  );
}