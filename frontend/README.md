# SUMOFLOW Frontend

SUMOFLOW is an **AI Traffic Control System** for **El-Tahrir Square, Cairo**, 
built as a graduation project by DSAI students.

This frontend is a dark, tech-themed dashboard that visualizes the SUMO-based 
simulation, vehicle detection models, and AI optimization results.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| React 18 | Frontend framework |
| React Router v6 | Navigation between pages |
| Recharts | Charts and data visualization |
| lucide-react | Icons |
| Tailwind CSS | Styling |

---

## Structure
- public/index.html – Root HTML shell  
- src/index.js – React entrypoint with BrowserRouter  
- src/App.jsx – App shell and route definitions  
- src/components/* – Reusable UI components (navbar, cards, traffic lights, etc.)  
- src/pages/* – 5 main pages:
  - Dashboard  
  - Live Simulation  
  - Model Comparison  
  - Before vs After  
  - About  
---

## Pages Overview

| Page | Description |
|------|-------------|
| **Dashboard** | Live vehicle counts, CO₂ emissions, wait time, traffic light status |
| **Live Simulation** | SUMO-GUI stream, detection results per class, LSTM predictions |
| **Model Comparison** | YOLO / RetinaNet / Faster RCNN metrics comparison table |
| **Before vs After** | Static lights vs AI adaptive system improvements |
| **About** | Team members, technology stack, system architecture |

---

## Running the App
```bash
# From the frontend folder
cd frontend
npm install
npm start
```

App runs at: **http://localhost:3000**

---

