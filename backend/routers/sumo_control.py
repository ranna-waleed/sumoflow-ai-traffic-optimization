# backend/routers/sumo_control.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import sumo_runner

router = APIRouter()

class StartRequest(BaseModel):
    profile: str = "morning_rush"
    gui:     bool = True

class StepRequest(BaseModel):
    steps: int = 30


@router.get("/status")
def get_status():
    return sumo_runner.get_status()


@router.post("/start")
def start(req: StartRequest):
    try:
        return sumo_runner.start(profile=req.profile, gui=req.gui)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start SUMO: {str(e)}")


@router.post("/stop")
def stop():
    return sumo_runner.stop()


@router.post("/step")
def run_steps(req: StepRequest):
    """
    Run N steps + capture screenshot in ONE TraCI call.
    Returns: { latest: state, image: base64 }
    """
    if not sumo_runner.is_running():
        raise HTTPException(status_code=400, detail="Simulation not running.")
    try:
        return sumo_runner.run_steps_with_screenshot(req.steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
def get_state():
    if not sumo_runner.is_running():
        raise HTTPException(status_code=400, detail="Simulation not running.")
    return sumo_runner.get_state()


@router.get("/screenshot")
def get_screenshot():
    if not sumo_runner.is_running():
        raise HTTPException(status_code=400, detail="Simulation not running. Use /step instead.")
    return {"status": "use POST /step — screenshot is included in step response"}