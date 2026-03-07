import torch
import time
import mlflow
import torchvision
import argparse # NEW: For command-line arguments
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader import get_val_loader
import os

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES = 8 
BATCH_SIZE = 1 
WEIGHTS_PATH = "detection/RetinaNet/retinanet_best.pth"

# ─── 2. Model Initialization ───────────────────────────────────
def get_retinanet_model(num_classes):
    # weights=None — we load our own trained weights immediately after,
    # so downloading DEFAULT pretrained weights is wasteful and misleading.
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
    # NEW: Setup argument parser
    parser = argparse.ArgumentParser(description="Evaluate RetinaNet and log to MLflow")
    parser.add_argument("--run-id", type=str, required=True, help="The MLflow Run ID to attach metrics to")
    args = parser.parse_args()
    
    mlflow_run_id = args.run_id

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Evaluating on device: {device}")

    # Load Model and Weights
    model = get_retinanet_model(NUM_CLASSES)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights not found at '{WEIGHTS_PATH}'.\n"
            f"Check that SAVE_PATH in train_retinanet.py matches WEIGHTS_PATH here.\n"
            f"Expected: detection/RetinaNet/retinanet_best.pth"
        )
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    print(f"Successfully loaded weights from {WEIGHTS_PATH}")
        
    model.to(device)
    model.eval() # CRITICAL: Set to evaluation mode!

    val_loader = get_val_loader(BATCH_SIZE)

    # Initialize TorchMetrics mAP calculator
# We keep exactly 3 values, but bump the highest one up to 300!
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True, max_detection_thresholds=[1, 10, 300])
    total_inference_time = 0
    num_images = 0

    print("Starting evaluation on Validation Set...")
    
    with torch.no_grad(): # Disable gradient tracking to save memory
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            
            # --- Measure FPS and Inference Time ---
            start_time = time.time()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize() # Wait for GPU to finish for accurate timing
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            num_images += len(images)

            # CORRECT — in evaluate.py, outputs go straight into the metric unfiltered
            # DO NOT add any score/confidence filtering here

            # Move predictions and targets to CPU for torchmetrics
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            outputs = [{k: v.to('cpu') for k, v in o.items()} for o in outputs]

            # Feed ALL predictions into the metric — no confidence threshold applied
            metric.update(outputs, targets)
            
            if batch_idx % 10 == 0:
                print(f"Evaluated [{batch_idx}/{len(val_loader)}] images...")


    if num_images == 0:
        raise RuntimeError("No images were evaluated. Check that WEIGHTS_PATH and val_loader paths are correct.")
    avg_inference_time = total_inference_time / num_images
    fps = 1.0 / avg_inference_time

    mAP_results = metric.compute()

    map_50          = mAP_results['map_50'].item()       # mAP at IoU=0.50
    map_50_95       = mAP_results['map'].item()          # mAP at IoU=0.50:0.95 (strict COCO metric)
    max_recall_300  = mAP_results['mar_300'].item()      # Max average recall @ 300 detections/image

    print("\n--- Evaluation Results ---")
    print(f"FPS:                      {fps:.2f}")
    print(f"Inference Time:           {avg_inference_time:.4f} sec/image")
    print(f"mAP@0.5:                  {map_50:.4f}")
    print(f"mAP@0.5:0.95 (COCO):      {map_50_95:.4f}   ← strict metric, lower is expected")
    print(f"Max Recall @ 300 dets:    {max_recall_300:.4f}   ← not standard recall")
    print("Note: torchmetrics does not expose a single Precision scalar directly.")
    print("      For per-class AP breakdown, check mAP_results['map_per_class'].")
    
    # --- Log to MLflow ---
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")
    

    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("FPS",                  fps)
        mlflow.log_metric("Inference_Time_sec",   avg_inference_time)
        mlflow.log_metric("mAP_0.5",              map_50)
        mlflow.log_metric("mAP_0.5_0.95",         map_50_95)
        mlflow.log_metric("Max_Recall_at_300",    max_recall_300)
        print("\n✅ Successfully logged evaluation metrics to the original MLflow run!")


if __name__ == "__main__":
    main()