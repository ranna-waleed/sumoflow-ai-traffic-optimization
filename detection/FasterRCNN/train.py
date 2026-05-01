"""
Part 2: Train Faster R-CNN
- Loads pretrained Faster R-CNN with RPN (ResNet-50 FPN backbone)
- Trains for 60 epochs with GPU support
- Tracks all metrics via MLflow
- Saves best model weights (lowest validation loss)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import copy
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.RetinaNet.dataloader import (
    get_train_loader, get_val_loader, CLASS_TO_IDX_FASTERRCNN
)

# ── paths ────────────────────────────────────────────────────────────────────
IMGS_DIR   = "detection/dataset/images/train"
XML_DIR    = "detection/dataset/annotations/train"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES  = 8          # 7 vehicle classes + 1 background
NUM_EPOCHS   = 60
BATCH_SIZE   = 2
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 30         # decay LR every N epochs
LR_GAMMA     = 0.1
VAL_SPLIT    = 0.15       # 15% of training data used for validation
BEST_WEIGHTS = os.path.join(OUTPUT_DIR, "best_faster_rcnn.pth")

train_loader = get_train_loader(BATCH_SIZE, class_to_idx=CLASS_TO_IDX_FASTERRCNN)
val_loader   = get_val_loader(batch_size=1, class_to_idx=CLASS_TO_IDX_FASTERRCNN)

def build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    """Load pretrained Faster R-CNN and replace the classification head."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace box predictor to match our class count
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0
    start = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Epoch [{epoch+1:02d}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {losses.item():.4f}")

    elapsed = time.time() - start
    avg_loss = total_loss / len(loader)
    print(f"  → Epoch {epoch+1:02d} avg loss: {avg_loss:.4f}  ({elapsed:.1f}s)")
    return avg_loss

@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Compute average loss on a validation set (model stays in train mode for loss)."""
    model.train()   # Faster R-CNN only computes losses in train mode
    total_loss = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss for loss in loss_dict.values()).item()
    return total_loss / max(len(loader), 1)

def main():
    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── dataset split ──────────────────────────────────────────────────────────
    full_dataset = TahrirTrafficDataset(imgs_dir=IMGS_DIR, xml_dir=XML_DIR)
    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print(f"Train samples: {train_size} | Val samples: {val_size}")

    # ── model, optimizer, scheduler ───────────────────────────────────────────
    model     = build_model(NUM_CLASSES, device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_experiment("FasterRCNN_TahrirTraffic")

    with mlflow.start_run(run_name="v2_backbone_lr0.005_lrstepsize"):
        # Log hyperparameters
        mlflow.log_params({
            "num_classes":  NUM_CLASSES,
            "num_epochs":   NUM_EPOCHS,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "momentum":     MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "lr_step_size": LR_STEP_SIZE,
            "lr_gamma":     LR_GAMMA,
            "backbone":     "resnet50_fpn",
            "device":       str(device),
        })

        best_val_loss  = float("inf")
        best_epoch     = -1

        for epoch in range(NUM_EPOCHS):
            print(f"\n{'='*55}")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"{'='*55}")

            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss   = evaluate_loss(model, val_loader, device)
            scheduler.step()

            print(f"  Val loss: {val_loss:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch + 1)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch + 1
                best_state    = copy.deepcopy(model.state_dict())
                torch.save(best_state, BEST_WEIGHTS)
                print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

        # ── Save final artifacts ───────────────────────────────────────────────
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "last_faster_rcnn.pth"))
        mlflow.log_artifact(BEST_WEIGHTS,  artifact_path="models")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "last_faster_rcnn.pth"), artifact_path="models")

        mlflow.log_params({"best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 6)})

        print(f"\nTraining complete!")
        print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Best weights saved to: {BEST_WEIGHTS}")

if __name__ == "__main__":
    main()