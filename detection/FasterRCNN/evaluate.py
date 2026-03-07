"""
Part 3: Evaluation & Inference
- Loads best trained weights
- Calculates mAP@0.5, Precision, Recall, FPS, Inference time (ms)
- Saves sample detection images with bounding boxes
- Logs all results to MLflow
"""

import os
import sys
import time
import json
import torch
import cv2
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataloader import TahrirTrafficDataset, collate_fn

BASE_DIR     = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "best_faster_rcnn.pth")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
DETECTIONS_DIR = os.path.join(OUTPUT_DIR, "detection_images")
os.makedirs(DETECTIONS_DIR, exist_ok=True)

TEST_IMGS_DIR = "detection/dataset/images/test"
TEST_XML_DIR  = "detection/dataset/annotations/test"

NUM_CLASSES      = 8
SCORE_THRESHOLD  = 0.5
NUM_SAMPLE_IMGS  = 10   # how many images to save with drawn boxes

IDX_TO_CLASS = {
    1: 'car', 2: 'bus', 3: 'truck', 4: 'motorcycle',
    5: 'taxi', 6: 'microbus', 7: 'bicycle'
}
COLORS = {
    'car':        (255,  80,  80),
    'bus':        ( 80, 200,  80),
    'truck':      ( 80, 100, 255),
    'motorcycle': (255, 200,  40),
    'taxi':       (255, 140,   0),
    'microbus':   (180,  80, 255),
    'bicycle':    ( 40, 220, 220),
}


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def draw_detections(image_tensor, pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels, save_path: str):
    """Draw predicted (solid) and ground-truth (dashed) boxes on image and save."""
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw ground truth in white dashed style
    for box, lbl in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Draw predictions with class colour
    for box, lbl, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_name = IDX_TO_CLASS.get(int(lbl), "unknown")
        color = COLORS.get(class_name, (200, 200, 200))
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        label_txt = f"{class_name}: {score:.2f}"
        cv2.putText(img_bgr, label_txt, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, img_bgr)


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── load model ────────────────────────────────────────────────────────────
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights not found at: {WEIGHTS_PATH}\n"
            "Run train.py first."
        )
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"[OK] Loaded weights from {WEIGHTS_PATH}")

    # ── test dataset ──────────────────────────────────────────────────────────
    test_dataset = TahrirTrafficDataset(imgs_dir=TEST_IMGS_DIR, xml_dir=TEST_XML_DIR)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"Test images: {len(test_dataset)}")

    # ── metric accumulators ───────────────────────────────────────────────────
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)    
    all_inference_times_ms = []
    saved_count = 0

    with torch.no_grad():
        for img_idx, (images, targets) in enumerate(test_loader):
            images_gpu  = [img.to(device) for img in images]

            # ── timed inference ────────────────────────────────────────────────
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(images_gpu)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            all_inference_times_ms.append(elapsed_ms)

            # ── accumulate metrics ─────────────────────────────────────────────
            preds = [{
                "boxes":  o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu(),
            } for o in outputs]

            gts = [{
                "boxes":  t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            } for t in targets]

            metric.update(preds, gts)

            # ── save sample detection images ───────────────────────────────────
            if saved_count < NUM_SAMPLE_IMGS:
                out_path = os.path.join(DETECTIONS_DIR, f"detection_{img_idx:04d}.jpg")
                draw_detections(
                    image_tensor = images[0],
                    pred_boxes   = outputs[0]["boxes"].cpu().tolist(),
                    pred_labels  = outputs[0]["labels"].cpu().tolist(),
                    pred_scores  = outputs[0]["scores"].cpu().tolist(),
                    gt_boxes     = targets[0]["boxes"].cpu().tolist(),
                    gt_labels    = targets[0]["labels"].cpu().tolist(),
                    save_path    = out_path,
                )
                saved_count += 1

    # ── compute final metrics ─────────────────────────────────────────────────
    map_result = metric.compute()

    map50       = float(map_result["map_50"])
    map_val     = float(map_result["map"])
    recall_val = float(map_result.get("mar_100", torch.tensor(float("nan"))))

    # Compute precision manually: TP / (TP + FP) across all test images
    total_tp = 0
    total_fp = 0
    with torch.no_grad():
        for img_idx, (images, targets) in enumerate(test_loader):
            images_gpu = [img.to(device) for img in images]
            outputs = model(images_gpu)
            for out, tgt in zip(outputs, targets):
                pred_boxes  = out["boxes"][out["scores"] >= SCORE_THRESHOLD]
                pred_labels = out["labels"][out["scores"] >= SCORE_THRESHOLD]
                gt_boxes    = tgt["boxes"]
                gt_labels   = tgt["labels"]
                matched_gt  = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    matched = False
                    for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gi in matched_gt or pl != gl:
                            continue
                        # compute IoU
                        ix1 = max(pb[0], gb[0]); iy1 = max(pb[1], gb[1])
                        ix2 = min(pb[2], gb[2]); iy2 = min(pb[3], gb[3])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        union = ((pb[2]-pb[0])*(pb[3]-pb[1]) +
                                 (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
                        if union > 0 and (inter / union) >= 0.5:
                            matched_gt.add(gi)
                            matched = True
                            break
                    if matched:
                        total_tp += 1
                    else:
                        total_fp += 1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")

    # FPS and latency
    avg_ms  = float(np.mean(all_inference_times_ms))
    std_ms  = float(np.std(all_inference_times_ms))
    fps     = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print("\n" + "="*55)
    print("EVALUATION RESULTS")
    print("="*55)
    print(f"  mAP@0.5           : {map50:.4f}")
    print(f"  mAP@0.5:0.95      : {map_val:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall_val:.4f}")
    print(f"  Avg Inference Time: {avg_ms:.2f} ms  ± {std_ms:.2f} ms")
    print(f"  FPS               : {fps:.2f}")
    print(f"  Saved {saved_count} detection images → {DETECTIONS_DIR}")
    print("="*55)

    # ── per-class AP ──────────────────────────────────────────────────────────
    per_class_ap = {}
    if "map_per_class" in map_result:
        for cls_idx, ap in enumerate(map_result["map_per_class"].tolist()):
            cls_name = IDX_TO_CLASS.get(cls_idx + 1, f"class_{cls_idx+1}")
            per_class_ap[cls_name] = round(ap, 4)
    
    print("\nPer-class AP@0.5:")
    for cls, ap in per_class_ap.items():
        print(f"  {cls:<14}: {ap:.4f}")

    # ── save results to JSON ──────────────────────────────────────────────────
    results = {
        "mAP@0.5":           round(map50, 6),
        "mAP@0.5:0.95":      round(map_val, 6),
        "precision":         round(precision, 6),
        "recall":            round(recall_val, 6),
        "avg_inference_ms":  round(avg_ms, 3),
        "std_inference_ms":  round(std_ms, 3),
        "fps":               round(fps, 2),
        "per_class_ap":      per_class_ap,
        "num_test_images":   len(test_dataset),
    }
    results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── log to MLflow ──────────────────────────────────────────────────────────
    mlflow.set_experiment("FasterRCNN_TahrirTraffic")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "mAP_50":            map50,
            "mAP_50_95":         map_val,
            "precision":         precision,
            "recall":            recall_val,
            "avg_inference_ms":  avg_ms,
            "fps":               fps,
        })
        for cls, ap in per_class_ap.items():
            mlflow.log_metric(f"AP_{cls}", ap)

        mlflow.log_artifact(results_path, artifact_path="eval")
        mlflow.log_artifacts(DETECTIONS_DIR, artifact_path="detection_images")

    print("\nAll metrics logged to MLflow.")
    return results


if __name__ == "__main__":
    run_evaluation()