# backend/routers/dqn.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import sys, os, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()

DQN_DIR         = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dqn")
MODELS_DIR      = os.path.join(DQN_DIR, "models")
RESULTS_PATH    = os.path.join(DQN_DIR, "results", "training_results.json")
COMPARISON_PATH = os.path.join(DQN_DIR, "results", "comparison_results.json")

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        try:
            from dqn.agent import DQNAgent
            _agent = DQNAgent(state_size=6, action_size=4)
            _agent.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"DQN not ready: {str(e)}")
    return _agent


class ActionRequest(BaseModel):
    state: list


#  GET /api/dqn/status 
@router.get("/status")
def dqn_status():
    model_path = os.path.join(MODELS_DIR, "dqn_best.pth")
    trained    = os.path.exists(model_path)
    results    = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    return {
        "status":              "ready" if trained else "not_trained",
        "trained":             trained,
        "model":               "DQN (2 layers, hidden=128)",
        "state_size":          6,
        "action_size":         4,
        "actions":             ["N-S Green (39s)", "E-W Green (39s)", "N-S Green (20s)", "E-W Green (20s)"],
        "episodes_trained":    results.get("episodes", 0),
        "improvement_pct":     results.get("improvement_pct", 0),
        "co2_improvement_pct": results.get("co2_improvement_pct", 0),
    }


#  POST /api/dqn/action 
@router.post("/action")
def get_action(req: ActionRequest):
    if len(req.state) != 6:
        raise HTTPException(status_code=400, detail="State must have 6 values")
    try:
        agent   = _get_agent()
        action  = agent.act(req.state)
        actions = ["N-S Green (39s)", "E-W Green (39s)", "N-S Green (20s)", "E-W Green (20s)"]
        return {"action": action, "action_name": actions[action], "epsilon": agent.epsilon}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  GET /api/dqn/results 
@router.get("/results")
def get_results():
    training = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            training = json.load(f)

    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            comparison = json.load(f)
        return {
            "avg_wait_fixed":      comparison["overall"]["fixed_wait_s"],
            "avg_wait_dqn":        comparison["overall"]["dqn_wait_s"],
            "improvement_pct":     comparison["overall"]["wait_improvement"],
            "co2_improvement_pct": comparison["overall"]["co2_improvement"],
            "fixed_co2_mg":        comparison["overall"]["fixed_co2_mg"],
            "dqn_co2_mg":          comparison["overall"]["dqn_co2_mg"],
            "profiles":            comparison.get("profiles", {}),
            "episodes":            training.get("episodes", 50),
            "episode_waits":       training.get("episode_waits", []),
            "episode_rewards":     training.get("episode_rewards", []),
            "episode_co2":         training.get("episode_co2", []),
        }

    if training:
        return {
            "avg_wait_fixed":      training.get("avg_wait_first10", 0),
            "avg_wait_dqn":        training.get("avg_wait_last10", 0),
            "improvement_pct":     training.get("improvement_pct", 0),
            "co2_improvement_pct": training.get("co2_improvement_pct", 0),
            "fixed_co2_mg":        training.get("avg_co2_first10", 0),
            "dqn_co2_mg":          training.get("avg_co2_last10", 0),
            "profiles":            {},
            "episodes":            training.get("episodes", 50),
            "episode_waits":       training.get("episode_waits", []),
            "episode_rewards":     training.get("episode_rewards", []),
            "episode_co2":         training.get("episode_co2", []),
        }

    raise HTTPException(status_code=404, detail="Run python dqn/train_dqn.py first")


#  POST /api/dqn/sim/start/{profile} 
@router.post("/sim/start/{profile}")
def start_dqn_sim(profile: str):
    """Start DQN-controlled SUMO simulation for a given profile."""
    valid = ["morning_rush", "evening_rush", "midday", "night"]
    if profile not in valid:
        raise HTTPException(status_code=400, detail=f"Profile must be one of {valid}")
    try:
        from services.dqn_runner import start
        ok = start(profile)
        if not ok:
            raise HTTPException(status_code=503, detail="Failed to start DQN simulation. Is dqn_best.pth trained?")
        return {"status": "started", "profile": profile}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  POST /api/dqn/sim/stop 
@router.post("/sim/stop")
def stop_dqn_sim():
    try:
        from services.dqn_runner import stop
        stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  GET /api/dqn/sim/status 
@router.get("/sim/status")
def dqn_sim_status():
    try:
        from services.dqn_runner import get_status
        return get_status()
    except Exception as e:
        return {"running": False, "profile": None, "step": 0, "metrics": {}}


#  GET /api/dqn/sim/screenshot 
@router.get("/sim/screenshot")
def dqn_screenshot():
    try:
        from services.dqn_runner import get_screenshot
        data = get_screenshot()
        if data is None:
            raise HTTPException(status_code=404, detail="No simulation running")
        return Response(content=data, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))