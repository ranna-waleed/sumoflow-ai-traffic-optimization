import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import os
import mlflow 
import subprocess 
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader import get_train_loader, get_val_loader

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES = 8 
BATCH_SIZE = 2
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
SAVE_PATH = "detection/RetinaNet/retinanet_best.pth"
UNFREEZE_EPOCH = 15  # ← backbone unfreezes after this epoch

# ─── 2. Model Initialization ───────────────────────────────────
def get_retinanet_model(num_classes):
    print("Loading pre-trained RetinaNet ResNet-50-FPN...")
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

# ─── 3. Training Loop ──────────────────────────────────────────
def main():
    # Setup MLflow Experiment
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")


    model = get_retinanet_model(NUM_CLASSES)
    model.to(device)
    train_loader = get_train_loader(BATCH_SIZE)

    # Freeze backbone — only train the detection head initially
    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    best_map = 0.0
    val_loader = get_val_loader(batch_size=1)
    
    # Start MLflow Run
    with mlflow.start_run(run_name="retinanet_v2_1800_images"):
        # Log Hyperparameters
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "SGD")

        print("Starting fine-tuning on El-Tahrir dataset...")
        
        for epoch in range(NUM_EPOCHS):
            # ── Unfreeze backbone after warm-up ──────────────────────
            if epoch == UNFREEZE_EPOCH:  
                print("Unfreezing backbone for full fine-tuning...")
                for param in model.backbone.parameters():
                    param.requires_grad = True
                # Re-build optimizer to include backbone params with lower LR
                params = [
                    {"params": model.backbone.parameters(), "lr": LEARNING_RATE * 0.1},
                    {"params": model.head.parameters(),     "lr": LEARNING_RATE}
                ]
                optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=0.0005)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH, eta_min=1e-6
                )
    
                print("Optimizer rebuilt with differential learning rates.")
            # ─────────────────────────────────────────────────────────

            model.train()
            epoch_loss = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Guard against NaN — skip batch and warn instead of corrupting weights
                if not torch.isfinite(losses):
                    print(f"Warning: Non-finite loss {losses.item()} at epoch {epoch+1} batch {batch_idx} — skipping batch.")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                losses.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # prevent exploding gradients
                optimizer.step()

                epoch_loss += losses.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {losses.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            
            # Log Metric to MLflow
            mlflow.log_metric("avg_loss", avg_epoch_loss, step=epoch)
                        
            try:
                current_lr = scheduler.get_last_lr()[0]
            except Exception:
                current_lr = LEARNING_RATE * 0.1 if epoch >= UNFREEZE_EPOCH else LEARNING_RATE
            
            mlflow.log_metric("lr", current_lr, step=epoch)

            print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

            # ── Validation mAP check every 5 epochs ──────────────────
            if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
                model.eval()
                val_metric = MeanAveragePrecision(
                    box_format='xyxy',
                    max_detection_thresholds=[1, 10, 300]
                )
                with torch.no_grad():
                    for val_images, val_targets in val_loader:
                        val_images  = [img.to(device) for img in val_images]
                        val_outputs = model(val_images)
                        val_targets = [{k: v.to('cpu') for k, v in t.items()} for t in val_targets]
                        val_outputs = [{k: v.to('cpu') for k, v in o.items()} for o in val_outputs]
                        val_metric.update(val_outputs, val_targets)

                val_results = val_metric.compute()
                current_map = val_results['map_50'].item()
                mlflow.log_metric("val_mAP_50", current_map, step=epoch)
                print(f"Validation mAP@0.5: {current_map:.4f}")

                if current_map > best_map:
                    best_map = current_map
                    torch.save(model.state_dict(), SAVE_PATH)
                    mlflow.log_artifact(SAVE_PATH)
                    print(f"✅ New best model saved! mAP@0.5 = {best_map:.4f}")

                    try:
                        # ⚠️ Disable DVC push during training to prevent Kaggle browser blocks
                        # subprocess.run(f"dvc add {SAVE_PATH} mlruns/", shell=True, check=True)
                        # subprocess.run("dvc push", shell=True, check=True)
                        print("Checkpoint secured in Google Drive!")
                    except subprocess.CalledProcessError as e:
                        print(f"Warning: DVC push failed. Error: {e}")
            # ──────────────────────────────────────────────────────────
                
            scheduler.step()
            if epoch == UNFREEZE_EPOCH:
                print("Backbone unfrozen. Scheduler and optimizer rebuilt for full fine-tuning.")

    print("Training Complete. Best weights secured in Google Drive.")

if __name__ == "__main__":
    main()