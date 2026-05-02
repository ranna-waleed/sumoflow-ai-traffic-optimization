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
    rcnn_path = os.path.join(FASTER_RCNN_DIR, "eval_results_run3.json")
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
                "mAP50":        0.483,
                "mAP50_95":     0.344,
                "precision":    0.750,
                "recall":       0.818,
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
                "mAP50":        rcnn_data.get("mAP@0.5",         0.4697),
                "mAP50_95":     rcnn_data.get("mAP@0.5:0.95",    0.3416),
                "precision":    rcnn_data.get("precision",        0.9459),
                "recall":       rcnn_data.get("recall",           0.3735),
                "fps":          rcnn_data.get("fps",              9.15),
                "inference_ms": rcnn_data.get("avg_inference_ms", 109.332),
                "per_class_ap": rcnn_data.get("per_class_ap", {
                    "car":        0.3117,
                    "bus":        0.7966,
                    "truck":      0.4961,
                    "motorcycle": 0.1065,
                    "taxi":       0.3082,
                    "microbus":   0.2748,
                    "bicycle":    0.0973,
                }),
            },
            {
                "name":         "RetinaNet",
                "status":       "complete",
                "selected":     False,
                "mAP50":        0.4172,
                "mAP50_95":     0.2884,
                "precision":    0,
                "recall":       0.3735,
                "fps":          13.40,
                "inference_ms": 74.6,
                "note":         "Precision metric unavailable due to torchmetrics 1.8.2 bug",
                "per_class_ap": {
                    "car":        0.2406,
                    "bus":        0.7249,
                    "truck":      0.4819,
                    "motorcycle": 0.0392,
                    "taxi":       0.2372,
                    "microbus":   0.2580,
                    "bicycle":    0.0370,
                },
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
        "mAP50":        0.483,
        "mAP50_95":     0.344,
        "precision":    0.750,
        "recall":       0.818,
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
    rcnn_path = os.path.join(FASTER_RCNN_DIR, "eval_results_run3.json")

    if not os.path.exists(rcnn_path):
        return {
            "name":         "Faster RCNN",
            "status":       "complete",
            "selected":     False,
            "epochs":       60,
            "backbone":     "ResNet-50",
            "mAP50":        0.4697,
            "mAP50_95":     0.3416,
            "precision":    0.9459,
            "recall":       0.3735,
            "fps":          9.15,
            "inference_ms": 109.332,
            "per_class_ap": {
                "car":        0.3117,
                "bus":        0.7966,
                "truck":      0.4961,
                "motorcycle": 0.1065,
                "taxi":       0.3082,
                "microbus":   0.2748,
                "bicycle":    0.0973,
            }
        }

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


#  GET /api/models/retinanet
@router.get("/retinanet")
def get_retinanet():
    return {
        "name":         "RetinaNet",
        "status":       "complete",
        "selected":     False,
        "epochs":       60,
        "backbone":     "ResNet-50 + FPN",
        "mAP50":        0.4172,
        "mAP50_95":     0.2884,
        "precision":    0,
        "recall":       0.3735,
        "fps":          13.40,
        "inference_ms": 74.6,
        "note":         "Precision metric unavailable due to torchmetrics 1.8.2 bug",
        "per_class_ap": {
            "car":        0.2406,
            "bus":        0.7249,
            "truck":      0.4819,
            "motorcycle": 0.0392,
            "taxi":       0.2372,
            "microbus":   0.2580,
            "bicycle":    0.0370,
        }
    }