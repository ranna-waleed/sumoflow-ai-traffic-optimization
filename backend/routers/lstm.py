# backend/routers/lstm.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()

try:
    from lstm.predict import predict, is_ready
    LSTM_AVAILABLE = True
except Exception as e:
    LSTM_AVAILABLE = False
    LSTM_ERROR     = str(e)


class PredictRequest(BaseModel):
    history: list


#  GET /api/lstm/status 
@router.get("/status")
def lstm_status():
    if not LSTM_AVAILABLE:
        return {"status": "unavailable", "ready": False,
                "detail": LSTM_ERROR if not LSTM_AVAILABLE else ""}
    return {
        "status":   "ready" if is_ready() else "not_trained",
        "ready":    is_ready(),
        "model":    "LSTM (2 layers, hidden=128)",
        "predicts": "North/South/East/West vehicle counts for next 30 timesteps",
    }


#  GET /api/lstm/predict/live 
@router.get("/predict/live")
def predict_live():
    """
    Predict using current simulation history from sumo_runner.
    No body needed — pulls history automatically.
    """
    if not LSTM_AVAILABLE or not is_ready():
        raise HTTPException(status_code=503, detail="LSTM not ready")

    try:
        from services.sumo_runner import get_lstm_history, is_running
        if not is_running():
            raise HTTPException(status_code=400, detail="Simulation not running")

        history = get_lstm_history()
        if len(history) < 10:
            return {
                "status":  "collecting",
                "message": f"Collecting history... {len(history)}/60 timesteps",
                "next_30s": {"north": 0, "south": 0, "east": 0, "west": 0}
            }

        result = predict(history)
        result["status"] = "ok"
        result["history_len"] = len(history)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  POST /api/lstm/predict 
@router.post("/predict")
def lstm_predict(req: PredictRequest):
    if not LSTM_AVAILABLE or not is_ready():
        raise HTTPException(status_code=503, detail="LSTM not ready")
    if len(req.history) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 timesteps")
    try:
        return predict(req.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))