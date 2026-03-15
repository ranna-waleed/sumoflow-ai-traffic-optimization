# backend/services/yolo_detect.py
# Loads YOLOv8s best.pt once at startup and runs inference on frames

import io
import os
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

#  Config 
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "detection", "yolo", "results", "best.pt"
)

CONFIDENCE_THRESHOLD = 0.45

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "taxi",
    4: "microbus",
    5: "motorcycle",
    6: "bicycle",
}

CLASS_COLORS = {
    "car":        "#94a3b8",
    "bus":        "#f97316",
    "truck":      "#a78bfa",
    "taxi":       "#eab308",
    "microbus":   "#38bdf8",
    "motorcycle": "#ef4444",
    "bicycle":    "#34d399",
}

#  Model singleton — loaded once at startup 
_model = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"YOLOv8s weights not found at: {MODEL_PATH}\n"
                "Download best.pt from Google Drive and place it at the path above."
            )
        print(f"[YOLO] Loading model from {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
        print("[YOLO] Model loaded successfully ✓")
    return _model


#  Core inference function 
def detect_image(image_bytes: bytes, conf: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Run YOLOv8s inference on raw image bytes.
    Returns bounding boxes, class names, confidence scores, and per-class counts.
    """
    model = get_model()

    # Convert bytes , PIL Image , numpy array
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    img_w, img_h = image.size

    t0 = time.time()
    results = model(img_array, conf=conf, verbose=False)
    inference_ms = round((time.time() - t0) * 1000, 1)

    detections = []
    class_counts = {name: 0 for name in CLASS_NAMES.values()}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id     = int(box.cls[0].item())
            cls_name   = CLASS_NAMES.get(cls_id, "unknown")
            confidence = round(float(box.conf[0].item()), 3)

            # Bounding box in pixels [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": confidence,
                "color":      CLASS_COLORS.get(cls_name, "#ffffff"),
                "bbox": {
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                    "width":  round(x2 - x1),
                    "height": round(y2 - y1),
                    # Normalized 0–1 for frontend canvas rendering
                    "x1_norm": round(x1 / img_w, 4),
                    "y1_norm": round(y1 / img_h, 4),
                    "x2_norm": round(x2 / img_w, 4),
                    "y2_norm": round(y2 / img_h, 4),
                },
            })

            if cls_name in class_counts:
                class_counts[cls_name] += 1

    return {
        "total_detections": len(detections),
        "inference_ms":     inference_ms,
        "image_size":       {"width": img_w, "height": img_h},
        "class_counts":     class_counts,
        "detections":       detections,
        "model":            "YOLOv8s",
        "confidence_threshold": conf,
    }


#  Batch detection (list of frames) 
def detect_batch(frames: list[bytes], conf: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    return [detect_image(f, conf) for f in frames]