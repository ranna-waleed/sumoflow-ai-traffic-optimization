# SUMOFlow AI 
### AI-Driven Traffic Optimization for Next-Generation Smart Cities
> A YOLO-SUMO Integration Approach — Team 46 | Zewail City of Science and Technology

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![SUMO](https://img.shields.io/badge/SUMO-1.x-green)](https://sumo.dlr.de)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-red)](https://ultralytics.com)
[![React](https://img.shields.io/badge/Frontend-React_18-61DAFB)](https://reactjs.org)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org)

---

##  Project Overview

SUMOFlow AI is a graduation project that integrates AI-based vehicle detection with the SUMO traffic simulator to optimize traffic light timing at El-Tahrir Square, Cairo. The system detects 7 vehicle classes in real time, simulates 4 rush hour profiles, and uses optimizer to reduce congestion and CO2 emissions.

---

##  System Architecture

```
1- INPUT LAYER
   ├── SUMO Network Files (.net.xml, .rou.xml) — El-Tahrir Square (OSM)
   ├── Simulation Frames (1,800 JPEG frames via TraCI)
   └── Rush Hour Profiles (Morning / Evening / Midday / Night)

2- DETECTION & SIMULATION LAYER
   ├── YOLOv8s       — mAP@0.5: 0.478 | 208 FPS  
   ├── Faster RCNN   — mAP@0.5: 0.430 | 10.93 FPS
   ├── RetinaNet     — mAP@0.5: 0.248 | 10.94 FPS
   └── SUMO Simulator — 4 rush hour profiles | peak 275 vehicles

3- DATA PIPELINE & TRACKING
   ├── Dataset Pipeline — 1,800 images | 7 classes | 70/20/10 split
   ├── MLflow Tracking  — hyperparams, metrics, artifacts
   └── Simulation Metrics — vehicle count, wait time, CO2 (CSV)

4- CONTROL LAYER (TraCI)
   ├── Metrics Calculator — wait times, queue stats, CO2, KPIs
   └── Optimization Engine —  Optimizer + LSTM Predictor (in progress)

5- OUTPUT LAYER
   ├── Performance Reports — CSV metrics per profile
   ├── Signal Timing Logs  — phase changes, durations
   └── React Dashboard     — 5 pages, live charts
```

---

## Model Comparison

| Model | mAP@0.5 | mAP@0.5:95 | Precision | Recall | FPS |
|-------|---------|------------|-----------|--------|-----|
| YOLOv8s  | 0.478  | 0.341 | 0.801 | 0.407 | 208 |
| Faster RCNN | 0.430 | 0.310 | 0.863 | 0.338 | 10.93 |
| RetinaNet | 0.248 | — | — | 0.274 | 10.94 |

---

##  Rush Hour Simulation Results

| Profile | Time | Peak Vehicles | Avg Wait | Peak CO2 |
|---------|------|--------------|----------|----------|
| Morning Rush | 8–10 AM | 275 | ~15s | 435,006 mg |
| Evening Rush | 4–7 PM | 194 | ~12s | 335,171 mg |
| Midday | 12–2 PM | 90 | ~12s | 181,981 mg |
| Night | 10 PM–12 AM | 61 | ~10s | 143,066 mg |

---

##  Project Structure

```
sumoflow-ai-traffic-optimization/
├── simulation/
│   ├── maps/
│   │   ├── tahrirupdated.net.xml       ← SUMO network (OSM)
│   │   ├── tahrir.rou.xml              ← base routes (250 vehicles)
│   │   ├── tahrir_fixed.rou.xml        ← extended routes (164 unique)
│   │   ├── routes_morning_rush.rou.xml
│   │   ├── routes_evening_rush.rou.xml
│   │   ├── routes_midday.rou.xml
│   │   ├── routes_night.rou.xml
│   │   ├── config_*.sumocfg            ← 4 rush hour configs
│   │   └── outputs/
│   │       ├── metrics_morning_rush_*.csv
│   │       ├── metrics_evening_rush_*.csv
│   │       ├── metrics_midday_*.csv
│   │       ├── metrics_night_*.csv
│   │       └── comparison_summary.csv
│   └── rush_hour/
│       ├── generate_rush_hour.py       ← generate route files
│       ├── run_realtime.py             ← run simulation profiles
│       ├── check_edges.py              ← edge analysis + fix short routes
│       └── compare_profiles.py        ← cross-profile comparison
│
├── detection/
│   ├── yolo/
│   │   ├── train.py                    ← YOLOv8s training (60 epochs)
│   │   ├── evaluate.py
│   │   └── results/                    ← metrics CSV, confusion matrix
│   ├── FasterRCNN/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── dataloader.py
│   │   └── outputs/
│   │       ├── eval_results.json
│   │       └── detection_images/
│   ├── RetinaNet/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── results/
│       └── model_comparison.csv        ← YOLO vs Faster RCNN vs RetinaNet
│
├── frontend/
│   ├── src/
│   │   ├── pages/                      ← Dashboard, LiveSim, ModelComparison, etc.
│   │   └── components/                 ← Navbar, MetricCard, Charts
│   ├── package.json
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Installation & Setup

**Step 1: Clone the repository**
```bash
git clone https://github.com/ranna-waleed/sumoflow-ai-traffic-optimization.git
cd sumoflow-ai-traffic-optimization
```

**Step 2: Create and activate virtual environment**
```bash
python -m venv sumoflow_env

# Windows
sumoflow_env\Scripts\activate

# Mac/Linux
source sumoflow_env/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Verify installation**
```bash
python -c "import traci; import cv2; import ultralytics; print('All packages installed!')"
```

---

##  Running the Simulation

**Run a specific rush hour profile:**
```bash
python simulation/rush_hour/run_realtime.py morning_rush
python simulation/rush_hour/run_realtime.py evening_rush
python simulation/rush_hour/run_realtime.py midday
python simulation/rush_hour/run_realtime.py night
```

**Regenerate all route files:**
```bash
python simulation/rush_hour/generate_rush_hour.py
```

**Analyze network edges:**
```bash
python simulation/rush_hour/check_edges.py
```

**Compare all 4 profiles:**
```bash
python simulation/rush_hour/compare_profiles.py
```

---

##  Training the Models

**YOLOv8s (Google Colab recommended):**
```bash
python detection/yolo/train.py
```

**Faster RCNN:**
```bash
python detection/FasterRCNN/train.py
```

**RetinaNet:**
```bash
python detection/RetinaNet/train.py
```

---

##  Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Large Files (Google Drive)

Model weights and large simulation outputs are stored on Google Drive:

 [Google Drive — Outputs & Weights](https://drive.google.com/drive/u/0/folders/1ULhxaaJfYKngSDmra5949AVY5koPnaeH)

Includes:
- `best.pt` — YOLOv8s weights
- `best_faster_rcnn.pth` — Faster RCNN weights
- `retinanet_best.pth` — RetinaNet weights
- `mlflow.db` — MLflow experiment tracking database
- Simulation XML outputs (fcd, emission, ssm)
- dataset_v2.zip
- SumoFlowAI_DVC_Storage
- Rush hour output files

---

## Team

| Member | ID | Role |
|--------|----|------|
| Rana Waleed | 202201737 | SUMO Simulation, YOLO Training, Frontend, Rush Hour |
| Roaa Raafat | 202202079 | SUMO Simulation,Dataset Labeling,Frontend, RetinaNet |
| Mariam Alhaj | 202200529 | SUMO Simulation,Faster RCNN, Frontend |

**Supervisor:** Dr. Mohamed Maher Ata
**University:** Zewail City of Science and Technology
**Program:** Data Science and Artificial Intelligence (DSAI)

---

##  Project Status

| Component | Status |
|-----------|--------|
| SUMO Simulation | Complete |
| Rush Hour Profiles | Complete |
| Dataset Pipeline |  Complete |
| YOLOv8s Training | Complete |
| Faster RCNN Training | Complete |
| RetinaNet Training | Complete |
| Frontend Dashboard |  Complete (static) |
| MLflow Tracking |  Complete |
| DQN Optimizer |  In Progress |
| LSTM Predictor | In Progress |
| FastAPI Backend | In Progress |
| Frontend-Backend Integration |  In Progress |
