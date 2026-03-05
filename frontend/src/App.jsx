import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import LiveSimulation from "./pages/LiveSimulation";
import ModelComparison from "./pages/ModelComparison";
import BeforeAfter from "./pages/BeforeAfter";
import About from "./pages/About";

function App() {
  return (
    <div className="min-h-screen text-[#e5e7eb]">
      <Navbar />
      <main className="pt-[72px] pb-12 px-4 md:px-6">
        <div className="max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveSimulation />} />
            <Route path="/models" element={<ModelComparison />} />
            <Route path="/before-after" element={<BeforeAfter />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
