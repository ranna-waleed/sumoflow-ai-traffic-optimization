# backend/routers/dqn.py  , returns data in format BeforeAfter.jsx expects
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import sys, os, json, glob, csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH    = os.path.join(BASE_DIR, "DeepQN", "results","fair", "evaluation_report_fair.json")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "DeepQN", "checkpoints")
LOGS_DIR        = os.path.join(BASE_DIR, "DeepQN", "logs")


def _load_episode_waits() -> list:
    """
    Read mean_reward per episode from the DeepQN training CSV logs.
    Returns a list of avg_wait values approximated from reward
    (reward ≈ -wait, so avg_wait ≈ abs(reward) / steps).
    """
    csv_files = sorted(glob.glob(os.path.join(LOGS_DIR, "training_*.csv")))
    if not csv_files:
        return []

    waits = []
    try:
        # Use the most recent CSV log
        with open(csv_files[-1], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # mean_reward is negative of wait-dominated signal
                    reward = float(row.get("mean_reward", 0))
                    steps  = int(row.get("ep_steps", 1080))
                    # Approximate avg wait per vehicle per step
                    # reward = -sum_wait so avg_wait ≈ abs(reward)/steps
                    approx_wait = abs(reward) / max(steps, 1)
                    waits.append(round(approx_wait, 2))
                except Exception:
                    pass
    except Exception:
        pass
    return waits


def _count_episodes() -> int:
    """Count total episodes trained from checkpoint files."""
    checkpoints = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pt"))
    if not checkpoints:
        return 0
    # Extract episode numbers from filenames like 315744796_ep0200.pt
    ep_nums = []
    for c in checkpoints:
        try:
            ep_str = os.path.basename(c).split("_ep")[1].replace(".pt", "")
            ep_nums.append(int(ep_str))
        except Exception:
            pass
    return max(ep_nums) if ep_nums else 0


#  GET /api/dqn/status
@router.get("/status")
def dqn_status():
    checkpoints = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pt"))
    trained     = len(checkpoints) > 0
    n_agents    = len(set(
        os.path.basename(c).split("_ep")[0] for c in checkpoints
    )) if checkpoints else 0

    return {
        "status":       "ready" if trained else "not_trained",
        "trained":      trained,
        "model":        "Dueling Double DQN — 7 agents",
        "state_size":   37,
        "action_size":  2,
        "actions":      ["keep current phase", "switch to next phase"],
        "n_agents":     n_agents,
        "checkpoints":  len(checkpoints),
    }


#  GET /api/dqn/results 
@router.get("/results")
def get_results():
    """
    Returns evaluation results in the exact format BeforeAfter.jsx expects:

    Top-level fields:
        avg_wait_fixed, avg_wait_dqn, improvement_pct,
        co2_improvement_pct, fixed_co2_mg, dqn_co2_mg,
        episodes, episode_waits

    Per-profile (results.profiles[profile]):
        fixed_wait_s, dqn_wait_s, wait_improvement,
        fixed_co2_mg, dqn_co2_mg, co2_improvement
    """
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Run: python -m DeepQN.evaluation.evaluate"
        )

    with open(RESULTS_PATH) as f:
        report = json.load(f)

    #  Per-profile data 
    profiles_out = {}
    all_wait_improvements = []
    all_co2_improvements  = []
    all_fixed_waits       = []
    all_dqn_waits         = []
    all_fixed_co2         = []
    all_dqn_co2           = []

    for profile, data in report.items():
        base = data["baseline"]
        dqn  = data["dqn"]
        imp  = data["improvements"]

        profiles_out[profile] = {
            # Field names BeforeAfter.jsx reads
            "fixed_wait_s":    round(base["avg_waiting_time_s"], 2),
            "dqn_wait_s":      round(dqn["avg_waiting_time_s"],  2),
            "wait_improvement": round(imp["avg_wait_pct"],        1),
            "fixed_co2_mg":    round(base["total_co2_mg"],        1),
            "dqn_co2_mg":      round(dqn["total_co2_mg"],         1),
            "co2_improvement": round(imp["total_co2_pct"],        1),
            "baseline_throughput": base["throughput"],
            "dqn_throughput":      dqn["throughput"],
            "throughput_delta":    imp["throughput_delta"],
            "kpi_wait_pass":       data["kpi_wait_pass"],
            "kpi_co2_pass":        data["kpi_co2_pass"],
        }

        all_wait_improvements.append(imp["avg_wait_pct"])
        all_co2_improvements.append(imp["total_co2_pct"])
        all_fixed_waits.append(base["avg_waiting_time_s"])
        all_dqn_waits.append(dqn["avg_waiting_time_s"])
        all_fixed_co2.append(base["total_co2_mg"])
        all_dqn_co2.append(dqn["total_co2_mg"])

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    # Top-level fields BeforeAfter.jsx reads directly 
    return {
        # Overall averages across all 4 profiles
        "avg_wait_fixed":      round(_avg(all_fixed_waits), 2),
        "avg_wait_dqn":        round(_avg(all_dqn_waits),   2),
        "improvement_pct":     round(_avg(all_wait_improvements), 1),
        "co2_improvement_pct": round(_avg(all_co2_improvements),  1),
        "fixed_co2_mg":        round(_avg(all_fixed_co2), 1),
        "dqn_co2_mg":          round(_avg(all_dqn_co2),   1),

        # Training info
        "episodes":      _count_episodes(),
        "episode_waits": _load_episode_waits(),

        # Per-profile breakdown
        "profiles": profiles_out,
    }


#  GET /api/dqn/cycle/status 
@router.get("/cycle/status")
def cycle_status():
    try:
        from services.dqn_runner import get_pipeline_state, get_status
        pipeline = get_pipeline_state()
        sim      = get_status()

        return {
            "cycle": {
                "sumo": {
                    "running":  sim["running"],
                    "profile":  sim["profile"],
                    "step":     sim["step"],
                    "vehicles": pipeline.get("sumo_vehicles", 0),
                },
                "yolo": {
                    "counts":         pipeline.get("yolo_counts", {}),
                    "total_detected": sum(
                        pipeline.get("yolo_counts", {}).values()
                    ),
                },
                "lstm": {
                    "prediction": pipeline.get("lstm_prediction",
                        {"north": 0, "south": 0, "east": 0, "west": 0}
                    ),
                },
                "dqn": {
                    "action":      pipeline.get("dqn_action", 0),
                    "action_name": pipeline.get("dqn_action_name", "—"),
                    "avg_wait_s":  pipeline.get("avg_wait_s", 0.0),
                    "agents":      7,
                    "model":       "Dueling Double DQN",
                },
            },
            "pipeline_connected": sim["running"],
        }
    except Exception as e:
        return {
            "cycle": {},
            "pipeline_connected": False,
            "error": str(e),
        }


#  POST /api/dqn/sim/start/{profile} 
@router.post("/sim/start/{profile}")
def start_dqn_sim(profile: str):
    valid = ["morning_rush", "evening_rush", "midday", "night"]
    if profile not in valid:
        raise HTTPException(
            status_code=400, detail=f"Profile must be one of {valid}"
        )
    try:
        from services.dqn_runner import start
        ok = start(profile)
        if not ok:
            raise HTTPException(
                status_code=503,
                detail="Failed to start. Are DeepQN checkpoints present?"
            )
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
    except Exception:
        return {"running": False, "profile": None, "step": 0, "metrics": {}}


#  GET /api/dqn/sim/screenshot 
@router.get("/sim/screenshot")
def dqn_screenshot():
    try:
        from services.dqn_runner import get_screenshot
        data = get_screenshot()
        if data is None:
            raise HTTPException(
                status_code=404, detail="No simulation running"
            )
        return Response(content=data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))