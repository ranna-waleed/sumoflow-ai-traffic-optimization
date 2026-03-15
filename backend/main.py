# backend/main.py
# SUMOFlow AI — FastAPI Backend
# Run: uvicorn main:app --reload --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import simulation, models, detection, sumo_control, lstm


app = FastAPI(
    title="SUMOFlow AI API",
    description="AI-Driven Traffic Optimization — El-Tahrir Square, Cairo",
    version="1.0.0"
)

#  CORS — allow React frontend on port 3000 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Routers 
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(models.router,     prefix="/api/models",     tags=["Models"])
app.include_router(detection.router,  prefix="/api/detection",  tags=["Detection"])
app.include_router(sumo_control.router, prefix="/api/sumo", tags=["SUMO Control"])
app.include_router(lstm.router, prefix="/api/lstm", tags=["LSTM"])
#  Health Check 
@app.get("/")
def root():
    return {
        "project": "SUMOFlow AI",
        "status":  "running",
        "version": "1.0.0",
        "endpoints": [
            "/api/simulation/summary",
            "/api/simulation/metrics/{profile}",
            "/api/simulation/profiles",
            "/api/models/comparison",
            "/api/models/yolo",
            "/api/models/faster_rcnn",
            "/api/detection/status",
            "/api/detection/detect",
            "/api/detection/detect-batch",
            "/docs"
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}