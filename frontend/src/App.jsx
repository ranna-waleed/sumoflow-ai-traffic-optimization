import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard       from "./pages/Dashboard";
import LiveSimulation  from "./pages/LiveSimulation";
import ModelComparison from "./pages/ModelComparison";
import BeforeAfter     from "./pages/BeforeAfter";
import About           from "./pages/About";

export default function App() {
  return (
    <>
      <Navbar />
      <div style={{
        paddingTop: "68px",
        paddingBottom: "100px",
        minHeight: "100vh",
        background: "#f1f5f9",
      }}>
        <div style={{
          maxWidth: "1280px",
          margin: "0 auto",
          padding: "0 16px",
        }}>
          <Routes>
            <Route path="/"             element={<Dashboard/>}/>
            <Route path="/live"         element={<LiveSimulation/>}/>
            <Route path="/models"       element={<ModelComparison/>}/>
            <Route path="/before-after" element={<BeforeAfter/>}/>
            <Route path="/about"        element={<About/>}/>
          </Routes>
        </div>
      </div>
    </>
  );
}