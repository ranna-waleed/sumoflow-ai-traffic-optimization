import torch
import time
import mlflow
import torchvision
import argparse
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader import get_test_loader   # FIX: use the held-out test split
import os

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES  = 8
BATCH_SIZE   = 1
WEIGHTS_PATH = "detection/RetinaNet/retinanet_best.pth"

IDX_TO_CLASS = {
    1: 'car', 2: 'bus', 3: 'truck', 4: 'motorcycle',
    5: 'taxi', 6: 'microbus', 7: 'bicycle'
}

# ─── 2. Model Initialization ───────────────────────────────────
def get_retinanet_model(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    return model


# ─── 3. Evaluation Loop ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate RetinaNet and log to MLflow")
    parser.add_argument("--run-id", type=str, required=True,
                        help="The MLflow Run ID to attach metrics to")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Evaluating on device: {device}")

    model = get_retinanet_model(NUM_CLASSES)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights not found at '{WEIGHTS_PATH}'.\n"
            f"Train the model first, or verify WEIGHTS_PATH."
        )

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    print(f"Loaded weights from {WEIGHTS_PATH}")
    model.to(device)
    model.eval()

    # FIX: evaluate on the true held-out test set
    test_loader = get_test_loader(BATCH_SIZE)

    metric = MeanAveragePrecision(
        box_format='xyxy',
        class_metrics=True,                          # enables per-class AP
        max_detection_thresholds=[1, 10, 300]
    )

    total_inference_time = 0.0
    num_images           = 0

    print("Starting evaluation on Test Set (held-out)...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [img.to(device) for img in images]

            start_time = time.time()
            outputs    = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            num_images           += len(images)

            # No confidence filtering here — metric sweeps all thresholds internally
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            metric.update(outputs, targets)

            if batch_idx % 10 == 0:
                print(f"Evaluated [{batch_idx}/{len(test_loader)}] images...")

    if num_images == 0:
        raise RuntimeError("No images evaluated. Check test loader paths.")

    avg_inference_time = total_inference_time / num_images
    fps                = 1.0 / avg_inference_time
    mAP_results        = metric.compute()

    map_50         = mAP_results['map_50'].item()
    map_50_95      = mAP_results['map'].item()
    mar_300        = mAP_results['mar_300'].item()   # Mean Average Recall @ 300 dets/img
    map_per_class  = mAP_results.get('map_per_class', None)

    print("\n══════════════════════════════════")
    print("  Evaluation Results (Test Set)  ")
    print("══════════════════════════════════")
    print(f"  FPS:                    {fps:.2f}")
    print(f"  Inference Time:         {avg_inference_time:.4f} sec/image")
    print(f"  mAP@0.5:                {map_50:.4f}")
    print(f"  mAP@0.5:0.95 (COCO):    {map_50_95:.4f}   ← strict metric")
    print(f"  Mean Avg Recall @300:   {mar_300:.4f}   ← COCO MAR, not plain recall")

    if map_per_class is not None:
        print("\n  Per-class AP@0.5:")
        for idx, ap in enumerate(mAP_results['map_per_class']):
            class_name = IDX_TO_CLASS.get(idx + 1, f"class_{idx+1}")
            print(f"    {class_name:<12}: {ap.item():.4f}")
    print("══════════════════════════════════\n")

    # ── Log to MLflow ─────────────────────────────────────────
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics({
            "test_FPS":               fps,
            "test_Inference_Time_sec": avg_inference_time,
            "test_mAP_0.5":           map_50,
            "test_mAP_0.5_0.95":      map_50_95,
            "test_MAR_at_300":        mar_300,
        })

        if map_per_class is not None:
            for idx, ap in enumerate(mAP_results['map_per_class']):
                class_name = IDX_TO_CLASS.get(idx + 1, f"class_{idx+1}")
                mlflow.log_metric(f"test_AP_{class_name}", ap.item())

        print("✅ Evaluation metrics logged to MLflow run:", args.run_id)


if __name__ == "__main__":
    main()