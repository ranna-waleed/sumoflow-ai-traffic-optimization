# backend/routers/models.py
from fastapi import APIRouter, HTTPException
import json
import os

router = APIRouter()

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FASTER_RCNN_DIR = os.path.join(BASE_DIR, "detection", "FasterRCNN", "outputs")

#  GET /api/models/comparison 
@router.get("/comparison")
def get_comparison():
    rcnn_path = os.path.join(FASTER_RCNN_DIR, "eval_results.json")
    rcnn_data = {}
    if os.path.exists(rcnn_path):
        with open(rcnn_path) as f:
            rcnn_data = json.load(f)

    return {
        "models": [
            {
                "name":         "YOLOv8s",
                "status":       "complete",
                "selected":     True,
                "mAP50":        0.478,
                "mAP50_95":     0.341,
                "precision":    0.801,
                "recall":       0.407,
                "fps":          208,
                "inference_ms": 4.8,
                "per_class_ap": {
                    "car":        0.527,
                    "bus":        0.972,
                    "truck":      0.714,
                    "taxi":       0.517,
                    "microbus":   0.506,
                    "motorcycle": 0.059,
                    "bicycle":    0.049,
                }
            },
            {
                "name":         "Faster RCNN",
                "status":       "complete",
                "selected":     False,
                "mAP50":        rcnn_data.get("mAP@0.5", 0),
                "mAP50_95":     rcnn_data.get("mAP@0.5:0.95", 0),
                "precision":    rcnn_data.get("precision", 0),
                "recall":       rcnn_data.get("recall", 0),
                "fps":          rcnn_data.get("fps", 0),
                "inference_ms": rcnn_data.get("avg_inference_ms", 0),
                "per_class_ap": rcnn_data.get("per_class_ap", {}),
            },
            {
                "name":         "RetinaNet",
                "status":       "retraining",
                "selected":     False,
                "mAP50":        0.2485,
                "recall":       0.2746,
                "fps":          10.94,
                "inference_ms": 91.4,
                "note":         "NUM_CLASSES bug fixed — retrain in progress",
            },
        ]
    }

#  GET /api/models/yolo 
@router.get("/yolo")
def get_yolo():
    return {
        "name":         "YOLOv8s",
        "status":       "complete",
        "selected":     True,
        "epochs":       60,
        "gpu":          "Google Colab T4",
        "mAP50":        0.478,
        "mAP50_95":     0.341,
        "precision":    0.801,
        "recall":       0.407,
        "fps":          208,
        "inference_ms": 4.8,
        "per_class_ap": {
            "car":        0.527,
            "bus":        0.972,
            "truck":      0.714,
            "taxi":       0.517,
            "microbus":   0.506,
            "motorcycle": 0.059,
            "bicycle":    0.049,
        }
    }

#  GET /api/models/faster_rcnn
@router.get("/faster_rcnn")
def get_faster_rcnn():
    rcnn_path = os.path.join(FASTER_RCNN_DIR, "eval_results.json")

    if not os.path.exists(rcnn_path):
        raise HTTPException(status_code=404, detail="eval_results.json not found")

    with open(rcnn_path) as f:
        data = json.load(f)

    return {
        "name":         "Faster RCNN",
        "status":       "complete",
        "selected":     False,
        "epochs":       60,
        "backbone":     "ResNet-50",
        "mAP50":        data.get("mAP@0.5"),
        "mAP50_95":     data.get("mAP@0.5:0.95"),
        "precision":    data.get("precision"),
        "recall":       data.get("recall"),
        "fps":          data.get("fps"),
        "inference_ms": data.get("avg_inference_ms"),
        "per_class_ap": data.get("per_class_ap", {}),
    }