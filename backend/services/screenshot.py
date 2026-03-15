# backend/services/screenshot.py
# Uses TraCI's built-in screenshot — works even when SUMO is minimized

import base64
import os
import time
import traci

# Temp file to save SUMO screenshot into
SCREENSHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "temp_frame.jpg"
)

def capture_sumo_window() -> dict:
    """
    Asks SUMO-GUI to save its own screenshot via TraCI.
    Works even when the SUMO window is behind other windows or minimized.
    """
    try:
        # Tell SUMO to write its current frame to a file
        traci.gui.screenshot("default", SCREENSHOT_PATH, width=960, height=600)

        # Give SUMO a moment to write the file
        time.sleep(0.1)

        if not os.path.exists(SCREENSHOT_PATH):
            return {"status": "error", "detail": "SUMO did not write screenshot file", "image": None}

        # Read file and encode as base64
        with open(SCREENSHOT_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status":    "ok",
            "image":     f"data:image/jpeg;base64,{b64}",
            "width":     960,
            "height":    600,
            "timestamp": time.time(),
        }

    except traci.exceptions.TraCIException as e:
        return {"status": "error", "detail": f"TraCI error: {str(e)}", "image": None}
    except Exception as e:
        return {"status": "error", "detail": str(e), "image": None}


def check_deps() -> dict:
    # No extra deps needed — just TraCI which is already installed
    return {"traci": True, "ready": True}