import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import { LayoutDashboard, Radio, GitCompare, Scale, Info } from "lucide-react";

const API = "http://127.0.0.1:8000";

const navItems = [
  { to: "/",            end: true,  label: "Dashboard",       icon: LayoutDashboard },
  { to: "/live",        end: false, label: "Live Simulation",  icon: Radio },
  { to: "/models",      end: false, label: "Model Comparison", icon: Scale },
  { to: "/before-after",end: false, label: "Before vs After",  icon: GitCompare },
  { to: "/about",       end: false, label: "About",            icon: Info },
];

function Navbar() {
  // ── FIXED: real health check — was always green static badge ──
  const [online, setOnline] = useState(false);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
        setOnline(res.ok);
      } catch {
        setOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 30000);  // re-check every 30s
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="fixed top-0 inset-x-0 z-40 border-b border-white/8 bg-[#0a0d12]/90 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between gap-4">

        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500/30 to-violet-500/20 border border-white/10">
            <span className="text-xl">🚦</span>
          </div>
          <div>
            <div className="font-semibold text-white tracking-tight">SUMOFLOW</div>
            <div className="text-[10px] uppercase tracking-widest text-[#94a3b8]">
              AI Traffic Control
            </div>
          </div>
        </div>

        {/* Nav links — desktop */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map(({ to, end, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-indigo-500/15 text-indigo-300"
                    : "text-[#94a3b8] hover:text-white hover:bg-white/5"
                }`
              }
            >
              <Icon className="w-4 h-4" strokeWidth={2} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Right */}
        <div className="flex items-center gap-3">
          {/* FIXED: real online/offline status from /health */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
            online
              ? "bg-emerald-500/15 border-emerald-500/30"
              : "bg-red-500/15 border-red-500/30"
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              online ? "bg-emerald-400 animate-pulse" : "bg-red-400"
            }`} />
            <span className={`text-xs font-medium ${
              online ? "text-emerald-300" : "text-red-300"
            }`}>
              {online ? "Online" : "Offline"}
            </span>
          </div>
          <span className="hidden sm:inline text-xs font-mono text-[#64748b]">v1.0.0</span>
        </div>
      </div>

      {/* Mobile nav */}
      <nav className="md:hidden border-t border-white/8 bg-[#0a0d12]/95 px-4 py-2 flex items-center gap-1 overflow-x-auto">
        {navItems.map(({ to, end, label }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `shrink-0 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                isActive ? "bg-indigo-500/20 text-indigo-300" : "text-[#94a3b8]"
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}

export default Navbar;