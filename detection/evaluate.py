import torch
import time
import mlflow
import torchvision
import argparse # NEW: For command-line arguments
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES = 8 
BATCH_SIZE = 1 
WEIGHTS_PATH = "detection/retinanet_best.pth"

# ─── 2. Model Initialization ───────────────────────────────────
def get_retinanet_model(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )
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
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print(f"Successfully loaded weights from {WEIGHTS_PATH}")
    else:
        print("Error: Weights file not found!")
        return
        
    model.to(device)
    model.eval() # CRITICAL: Set to evaluation mode!

    # Import your Validation Dataloader
    # Assuming dataloader.py has a get_val_loader() function
    from dataloader import get_val_loader
    val_loader = get_val_loader(BATCH_SIZE)

    # Initialize TorchMetrics mAP calculator
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

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

            # Move predictions and targets to CPU for torchmetrics
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            outputs = [{k: v.to('cpu') for k, v in o.items()} for o in outputs]
            
            # Feed into the metric calculator
            metric.update(outputs, targets)
            
            if batch_idx % 10 == 0:
                print(f"Evaluated [{batch_idx}/{len(val_loader)}] images...")

    # --- Calculate Final Metrics ---
    avg_inference_time = total_inference_time / num_images
    fps = 1.0 / avg_inference_time
    mAP_results = metric.compute()

    map_50 = mAP_results['map_50'].item()
    precision = mAP_results['map'].item() # Strict mAP 0.5:0.95 often correlates to precision in this API
    recall = mAP_results['mar_100'].item() # Maximum recall given 100 detections per image

    print("\n--- Evaluation Results ---")
    print(f"FPS:             {fps:.2f}")
    print(f"Inference Time:  {avg_inference_time:.4f} sec/image")
    print(f"mAP@0.5:         {map_50:.4f}")
    print(f"Recall (AR@100): {recall:.4f}")

    # --- Log to MLflow ---
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")
    
    # Using the exact run_id resumes the previous training run to add these metrics!
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metric("FPS", fps)
        mlflow.log_metric("Inference_Time_sec", avg_inference_time)
        mlflow.log_metric("mAP_0.5", map_50)
        mlflow.log_metric("Recall", recall)
        print("\n✅ Successfully logged evaluation metrics to the original MLflow run!")

if __name__ == "__main__":
    main()