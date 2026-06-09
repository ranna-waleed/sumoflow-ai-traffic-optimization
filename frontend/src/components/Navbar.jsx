import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";

const API = "http://127.0.0.1:8000";

const NAV = [
  { to: "/",             end: true,  label: "Dashboard"       },
  { to: "/live",         end: false, label: "Live Simulation"  },
  { to: "/models",       end: false, label: "Model Comparison" },
  { to: "/before-after", end: false, label: "Before vs After"  },
  { to: "/about",        end: false, label: "About"            },
];

const S = {
  header: {
    position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
    background: "#0f2644", borderBottom: "1px solid #1e3a5f",
  },
  inner: {
    maxWidth: "1280px", margin: "0 auto", padding: "0 24px",
    height: "60px", display: "flex", alignItems: "center",
    justifyContent: "space-between", gap: "16px",
  },
  logo: {
    display: "flex", alignItems: "center", gap: "12px", flexShrink: 0,
  },
  logoIcon: {
    width: "34px", height: "34px", background: "#1d4ed8",
    borderRadius: "6px", display: "flex", alignItems: "center",
    justifyContent: "center", fontSize: "18px",
  },
  logoName: {
    fontSize: "15px", fontWeight: 700, color: "#fff",
    letterSpacing: "-0.01em", lineHeight: 1,
  },
  logoSub: {
    fontSize: "10px", color: "#94a3b8", letterSpacing: "0.08em",
    textTransform: "uppercase", lineHeight: 1, marginTop: "2px",
  },
  nav: {
    display: "flex", alignItems: "center", gap: "2px",
    flex: 1, justifyContent: "center",
  },
  navLink: (active) => ({
    display: "flex", alignItems: "center", gap: "6px",
    padding: "6px 12px", borderRadius: "4px",
    fontSize: "13px", fontWeight: 500,
    textDecoration: "none", transition: "all 0.15s",
    color: active ? "#fff" : "#94a3b8",
    background: active ? "#1d4ed8" : "transparent",
  }),
  right: { display: "flex", alignItems: "center", gap: "12px" },
  badge: (online) => ({
    display: "flex", alignItems: "center", gap: "6px",
    padding: "4px 10px", borderRadius: "3px",
    background: online ? "#d1fae5" : "#fee2e2",
    border: `1px solid ${online ? "#6ee7b7" : "#fca5a5"}`,
  }),
  dot: (online) => ({
    width: "7px", height: "7px", borderRadius: "50%",
    background: online ? "#16a34a" : "#dc2626",
    animation: online ? "pulse-dot 2s infinite" : "none",
  }),
  badgeText: (online) => ({
    fontSize: "12px", fontWeight: 500,
    color: online ? "#15803d" : "#b91c1c",
  }),
  version: { fontSize: "12px", color: "#475569", fontFamily: "monospace" },
};

export default function Navbar() {
  const [online, setOnline] = useState(false);
  const [width, setWidth]   = useState(window.innerWidth);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
        setOnline(res.ok);
      } catch { setOnline(false); }
    };
    check();
    const id = setInterval(check, 30000);
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => { clearInterval(id); window.removeEventListener("resize", onResize); };
  }, []);

  const isMobile = width < 768;

  return (
    <header style={S.header}>
      <div style={S.inner}>
        {/* Logo */}
        <div style={S.logo}>
          <div style={S.logoIcon}>🚦</div>
          <div>
            <div style={S.logoName}>
              SUMOFlow <span style={{ color: "#60a5fa" }}>AI</span>
            </div>
            <div style={S.logoSub}>Traffic Optimization — El-Tahrir Square</div>
          </div>
        </div>

        {/* Desktop nav only */}
        {!isMobile && (
          <nav style={S.nav}>
            {NAV.map(({ to, end, label }) => (
              <NavLink key={to} to={to} end={end}
                style={({ isActive }) => S.navLink(isActive)}>
                {label}
              </NavLink>
            ))}
          </nav>
        )}

        {/* Right */}
        <div style={S.right}>
          <div style={S.badge(online)}>
            <div style={S.dot(online)} />
            <span style={S.badgeText(online)}>
              {online ? "Online" : "Offline"}
            </span>
          </div>
          {!isMobile && <span style={S.version}>v1.0.0</span>}
        </div>
      </div>

      {/* Mobile nav — only when narrow */}
      {isMobile && (
        <nav style={{
          background: "#0d2040", borderTop: "1px solid #1e3a5f",
          padding: "6px 12px", display: "flex", gap: "4px", overflowX: "auto",
        }}>
          {NAV.map(({ to, end, label }) => (
            <NavLink key={to} to={to} end={end}
              style={({ isActive }) => ({
                flexShrink: 0, padding: "5px 10px", borderRadius: "3px",
                fontSize: "12px", fontWeight: 500, textDecoration: "none",
                color: isActive ? "#fff" : "#94a3b8",
                background: isActive ? "#1d4ed8" : "transparent",
              })}>
              {label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  );
}