# SUMOFlow AI 🚦
### AI-Driven Traffic Optimization for Next-Generation Smart Cities: A YOLO-SUMO Integration Approach

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![SUMO](https://img.shields.io/badge/SUMO-1.24-orange?style=for-the-badge)

**Team 46 | Zewail City University of Science and Technology**  
**Department of Data Science and Artificial Intelligence (DSAI)**  
**Supervisor: Dr. Mohamed Maher Ata**

[Rana Waleed (202201737)](https://github.com/ranna-waleed) · Roaa Raafat (202202079) · Mariam Alhaj (202200529)

</div>

---

## Overview

SUMOFlow AI is a complete smart city traffic optimization system for **El-Tahrir Square, Cairo**. It integrates real-time vehicle detection, traffic flow prediction, and deep reinforcement learning to reduce congestion and CO₂ emissions.

### Key Results
| Metric | Before (Fixed Timing) | After (DQN) | Improvement |
|---|---|---|---|
| Avg Wait Time | 10.09s | 6.10s | **↓39.49%** |
| CO₂ Emissions | 273.3 mg/step | 154.3 mg/step | **↓43.57%** |
| BiLSTM MAE | — | 3.15 vehicles | — |
| YOLOv8s mAP@0.5 | — | 0.478 | 208 FPS |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SUMOFlow AI Pipeline                    │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  SUMO Sim    │  Detection   │  BiLSTM      │  DQN           │
│  El-Tahrir   │  YOLOv8s     │  Predictor   │  Optimizer     │
│  5 profiles  │  FasterRCNN  │  MAE=3.15    │  ↓39.5% wait   │
│  TraCI API   │  RetinaNet   │  30s horizon │  ↓43.6% CO₂    │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              FastAPI Backend (15+ REST endpoints)            │
├─────────────────────────────────────────────────────────────┤
│         React Dashboard (5 pages, real-time charts)         │
└─────────────────────────────────────────────────────────────┘
```

---

##  Features

### Live SUMO Simulation
- El-Tahrir Square network with 5 traffic profiles (morning rush, evening rush, midday, night, custom OD)
- Real-time screenshot streaming to React dashboard via TraCI
- Live metrics: vehicle count, waiting time, CO₂ emissions, avg speed
- Traffic light status display (real TraCI data)

###  Vehicle Detection (3 Models)
| Model | mAP@0.5 | mAP@0.5:0.95 | FPS | Inference |
|---|---|---|---|---|
| **YOLOv8s**  | **0.478** | 0.341 | **208** | 4.8ms |
| Faster RCNN | 0.470 | 0.343 | 7.08 | 141.2ms |
| RetinaNet | 0.413 | 0.285 | 9.38 | 106.6ms |

7 vehicle classes: car, bus, truck, taxi, microbus, motorcycle, bicycle

###  BiLSTM Traffic Predictor
- 2-layer Bidirectional LSTM, hidden=128
- Trained on 10,167 sequences across 5 profiles
- **Overall MAE: 3.15 vehicles** (30-second prediction horizon)
- Live predictions for N/S/E/W directions every 4 seconds

###  DQN Signal Optimizer
- Deep Q-Network with experience replay and target network
- State: `[N, S, E, W vehicle counts, current phase, avg wait]`
- 4 actions using real SUMO phase definitions for TL 315744796
- Trained for 50 episodes across 4 rush hour profiles
- **Results: ↓39.49% wait time, ↓43.57% CO₂**

---

##  Project Structure

```
sumoflow-ai-traffic-optimization/
├── backend/                    # FastAPI backend
│   ├── main.py
│   ├── routers/
│   │   ├── simulation.py       # SUMO simulation endpoints
│   │   ├── models.py           # Detection model comparison
│   │   ├── detection.py        # YOLO inference
│   │   ├── sumo_control.py     # TraCI control
│   │   ├── lstm.py             # BiLSTM predictions
│   │   └── dqn.py              # DQN optimizer + live sim
│   └── services/
│       ├── sumo_runner.py      # TraCI manager
│       ├── yolo_detect.py      # YOLOv8 inference
│       └── dqn_runner.py       # DQN live simulation
├── frontend/                   # React dashboard (5 pages)
│   └── src/pages/
│       ├── Dashboard.jsx        # Live SUMO stream + metrics
│       ├── LiveSimulation.jsx   # Custom OD + LSTM predictions
│       ├── ModelComparison.jsx  # 3-model comparison table
│       ├── BeforeAfter.jsx      # DQN results + live DQN sim
│       └── About.jsx
├── dqn/                        # DQN optimizer
│   ├── environment.py          # SUMO-TraCI environment
│   ├── agent.py                # DQN agent + QNetwork
│   ├── train_dqn.py            # Training script
│   ├── run_baseline.py         # Fixed timing baseline
│   ├── run_dqn.py              # DQN evaluation
│   ├── compare.py              # Before/After comparison
│   └── visualize.py            # Watch DQN control lights
├── lstm/                       # BiLSTM predictor
│   ├── train_lstm.py           # Training script (BiLSTM)
│   ├── predict.py              # Inference module
│   └── evaluate.py             # MAE evaluation
├── detection/                  # Detection models
│   ├── yolo/                   # YOLOv8s weights + results
│   └── FasterRCNN/             # Faster RCNN weights + results
└── simulation/
    └── maps/                   # SUMO network + route files
```

---

##  Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- [SUMO 1.24](https://sumo.dlr.de/docs/Downloads.php) with `SUMO_HOME` set
- CUDA GPU (optional — CPU works)

### Backend Setup
```bash
# Clone repo
git clone https://github.com/ranna-waleed/sumoflow-ai-traffic-optimization.git
cd sumoflow-ai-traffic-optimization

# Create virtual environment
python -m venv sumoflow_env
sumoflow_env\Scripts\activate        # Windows
# source sumoflow_env/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

---

##  Running the System

### 1. Start Backend
```bash
cd backend
uvicorn main:app --port 8000 --workers 1
```

### 2. Start Frontend
```bash
cd frontend
npm start
```

Open `http://localhost:3000`

### 3. Train DQN (optional — model included)
```bash
# Train
python dqn/train_dqn.py

# Run baseline comparison
python dqn/run_baseline.py
python dqn/run_dqn.py
python dqn/compare.py
```

### 4. Watch DQN Control Traffic Lights
```bash
python dqn/visualize.py
```

---

##  API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/sumo/start` | Start SUMO simulation |
| GET | `/api/sumo/screenshot` | Get live frame |
| GET | `/api/sumo/state` | Get live metrics |
| GET | `/api/models/comparison` | 3-model comparison |
| GET | `/api/lstm/predict/live` | Live N/S/E/W predictions |
| GET | `/api/dqn/results` | DQN comparison results |
| POST | `/api/dqn/sim/start/{profile}` | Start DQN live simulation |
| GET | `/api/dqn/sim/screenshot` | DQN simulation frame |

Full API docs: `http://localhost:8000/docs`

---

##  DQN Results by Profile

| Profile | Fixed Wait | DQN Wait | Wait ↓ | CO₂ ↓ |
|---|---|---|---|---|
| Morning Rush  | 10.67s | 9.52s | 10.8% | 38.26% |
| Evening Rush  | 10.72s | 5.16s | **51.91%** | 44.22% |
| Midday  | 10.71s | 4.78s | **55.36%** | 44.48% |
| Night  | 8.25s | 4.96s | 39.87% | **50.11%** |
| **Overall** | **10.09s** | **6.10s** | **39.49%** | **43.57%** |

---

##  MLflow Experiment Tracking

```bash
# View all training runs
python -m mlflow ui --port 5000
# Open http://127.0.0.1:5000
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Simulation | SUMO 1.24, TraCI |
| Detection | YOLOv8s, Faster RCNN, RetinaNet |
| Prediction | BiLSTM (PyTorch) |
| Optimization | DQN (PyTorch) |
| Backend | FastAPI, Python |
| Frontend | React 18, Recharts, Tailwind CSS |
| Experiment Tracking | MLflow |
| Version Control | Git, GitHub |

---

## License

This project is developed as a graduation project at Zewail City University of Science and Technology. All rights reserved.

---
