import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import { LayoutDashboard, Activity, BarChart2, ArrowLeftRight, Info } from "lucide-react";

const API = "http://127.0.0.1:8000";

const NAV = [
  { to: "/",             end: true,  label: "Dashboard",   icon: LayoutDashboard },
  { to: "/live",         end: false, label: "Live Sim",     icon: Activity        },
  { to: "/models",       end: false, label: "Models",       icon: BarChart2       },
  { to: "/before-after", end: false, label: "Before/After", icon: ArrowLeftRight  },
  { to: "/about",        end: false, label: "About",        icon: Info            },
];

export default function Navbar() {
  const [online, setOnline] = useState(false);
  const [width,  setWidth]  = useState(window.innerWidth);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
        setOnline(r.ok);
      } catch { setOnline(false); }
    };
    check();
    const id = setInterval(check, 30000);
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => { clearInterval(id); window.removeEventListener("resize", onResize); };
  }, []);

  const isMobile = width < 640;

  return (
    <>
      {/* ── Top header ── */}
      <header style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
        background: "#fff",
        borderBottom: "1px solid #e2e8f0",
        boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
      }}>
        <div style={{
          maxWidth: "1280px", margin: "0 auto", padding: "0 24px",
          height: "52px", display: "flex", alignItems: "center",
          justifyContent: "space-between",
        }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{
              width: "32px", height: "32px", background: "#2563eb",
              borderRadius: "8px", display: "flex", alignItems: "center",
              justifyContent: "center", fontSize: "16px",
              boxShadow: "0 2px 8px rgba(37,99,235,0.25)",
            }}>🚦</div>
            <div>
              <div style={{ fontSize: "14px", fontWeight: 700, color: "#0f172a", lineHeight: 1 }}>
                SUMOFlow <span style={{ color: "#2563eb" }}>AI</span>
              </div>
              {!isMobile && (
                <div style={{ fontSize: "10px", color: "#296ed0", letterSpacing: "0.07em", textTransform: "uppercase", marginTop: "2px" }}>
                  Traffic Optimization · El-Tahrir Square
                </div>
              )}
            </div>
          </div>

          {/* Status */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{
              display: "flex", alignItems: "center", gap: "6px",
              padding: "4px 12px", borderRadius: "99px",
              background: online ? "#f0fdf4" : "#fef2f2",
              border: `1px solid ${online ? "#bbf7d0" : "#fecaca"}`,
            }}>
              <div style={{
                width: "7px", height: "7px", borderRadius: "50%",
                background: online ? "#16a34a" : "#dc2626",
                animation: online ? "pulse-dot 2s infinite" : "none",
              }}/>
              <span style={{ fontSize: "12px", fontWeight: 600, color: online ? "#15803d" : "#b91c1c" }}>
                {online ? "Online" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Bottom floating navbar ── */}
      <nav style={{
        position: "fixed",
        bottom: "16px",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 50,
        background: "#fff",
        borderRadius: "28px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06)",
        border: "1px solid #f1f5f9",
        padding: "0 12px",
        height: "64px",
        display: "flex",
        alignItems: "flex-end",
        paddingBottom: "10px",
        gap: isMobile ? "0px" : "4px",
        width: isMobile ? "calc(100vw - 32px)" : "auto",
        justifyContent: isMobile ? "space-around" : "center",
      }}>
        {NAV.map(({ to, end, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            style={{ textDecoration: "none" }}
          >
            {({ isActive }) => (
              <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "3px",
                padding: "0 10px",
                cursor: "pointer",
              }}>
                {/* Icon bubble */}
                <div style={{
                  width: isActive ? "50px" : "40px",
                  height: isActive ? "50px" : "40px",
                  borderRadius: isActive ? "16px" : "12px",
                  background: isActive ? "#2563eb" : "#f1f5f9",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: isActive ? "-28px" : "-18px",
                  transition: "all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)",
                  boxShadow: isActive ? "0 6px 18px rgba(37,99,235,0.35)" : "none",
                }}>
                  <Icon
                    size={isActive ? 22 : 18}
                    strokeWidth={isActive ? 2.5 : 1.8}
                    color={isActive ? "#fff" : "#94a3b8"}
                  />
                </div>
                {/* Label */}
                <span style={{
                  fontSize: "10px",
                  fontWeight: isActive ? 700 : 500,
                  color: isActive ? "#2563eb" : "#94a3b8",
                  whiteSpace: "nowrap",
                  letterSpacing: "0.01em",
                }}>
                  {label}
                </span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </>
  );
}