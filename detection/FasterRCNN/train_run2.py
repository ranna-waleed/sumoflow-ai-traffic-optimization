"""
Part 2 - Run 2: Train Faster R-CNN
Changes from Run 1:
- Weighted Random Sampling to fix class imbalance (motorcycle, bicycle)
- Empty annotation images filtered out from training set
- MLflow run name: v2_backbone_weighted_sampling_filtered
"""

import os
import sys
import time
import copy
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ..RetinaNet.dataloader import TahrirTrafficDataset, collate_fn


# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

IMGS_DIR   = "detection/dataset/images/train"
XML_DIR    = "detection/dataset/annotations/train"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── hyperparameters (same as Run 1) ──────────────────────────────────────────
NUM_CLASSES  = 8          # 7 vehicle classes + 1 background
NUM_EPOCHS   = 60
BATCH_SIZE   = 2
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 30         # decay LR every N epochs
LR_GAMMA     = 0.1
VAL_SPLIT    = 0.15 
BEST_WEIGHTS = os.path.join(OUTPUT_DIR, "best_faster_rcnn_run2.pth")


def build_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)


# ── NEW: Filter empty annotation images ──────────────────────────────────────
def filter_empty_images(dataset):
    """Remove images that have zero annotated boxes."""
    valid_indices = []
    empty_count   = 0
    for i in range(len(dataset)):
        _, target = dataset[i]
        if target['boxes'].shape[0] > 0:
            valid_indices.append(i)
        else:
            empty_count += 1
    print(f"  Filtered out {empty_count} empty images → {len(valid_indices)} remaining")
    return Subset(dataset, valid_indices), valid_indices


# ── NEW: Build weighted sampler to fix class imbalance ───────────────────────
def build_weighted_sampler(dataset, num_classes=8):
    """
    Give higher sampling probability to images containing rare classes
    (motorcycle, bicycle) and lower probability to dominant classes (car).
    """
    # Step 1: count how many times each class appears in the dataset
    class_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        _, target = dataset[i]
        for label in target['labels'].tolist():
            class_counts[label] += 1

    print("  Class instance counts:")
    class_names = {1:'car', 2:'bus', 3:'truck', 4:'motorcycle',
                   5:'taxi', 6:'microbus', 7:'bicycle'}
    for idx, count in enumerate(class_counts):
        if idx == 0: continue
        print(f"    {class_names.get(idx, idx):<12}: {int(count)}")

    # Step 2: inverse frequency weight per class
    # Add 1 to avoid division by zero for classes with 0 samples
    class_weights = 1.0 / (class_counts + 1)

    # Step 3: assign each image a weight = max class weight among its labels
    # (prioritises images that contain at least one rare class)
    sample_weights = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target['labels'].tolist()
        if len(labels) == 0:
            sample_weights.append(0.0)
        else:
            w = max(float(class_weights[l]) for l in labels)
            sample_weights.append(w)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True   # allows rare images to be sampled multiple times
    )
    return sampler


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

    elapsed  = time.time() - start
    avg_loss = total_loss / len(loader)
    print(f"  → Epoch {epoch+1:02d} avg loss: {avg_loss:.4f}  ({elapsed:.1f}s)")
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss for loss in loss_dict.values()).item()
    return total_loss / max(len(loader), 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 1: Load full dataset ─────────────────────────────────────────────
    full_dataset = TahrirTrafficDataset(imgs_dir=IMGS_DIR, xml_dir=XML_DIR)
    print(f"\nOriginal dataset size: {len(full_dataset)}")

    # ── Step 2: Filter empty images ───────────────────────────────────────────
    print("\nFiltering empty annotation images...")
    filtered_dataset, _ = filter_empty_images(full_dataset)

    # ── Step 3: Train/val split ───────────────────────────────────────────────
    val_size   = max(1, int(len(filtered_dataset) * VAL_SPLIT))
    train_size = len(filtered_dataset) - val_size
    train_ds, val_ds = random_split(
        filtered_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train samples: {train_size} | Val samples: {val_size}")

    # ── Step 4: Build weighted sampler on train split only ────────────────────
    print("\nBuilding weighted sampler...")
    sampler = build_weighted_sampler(train_ds, num_classes=NUM_CLASSES)

    # NOTE: sampler replaces shuffle=True — do not use both
    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,       # ← weighted sampling
        collate_fn  = collate_fn,
        num_workers = 2,
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 2,
        pin_memory  = True
    )

    # ── Step 5: Model, optimizer, scheduler ──────────────────────────────────
    model     = build_model(NUM_CLASSES, device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    # ── Step 6: MLflow run ────────────────────────────────────────────────────
    mlflow.set_experiment("FasterRCNN_TahrirTraffic")

    with mlflow.start_run(run_name="v2_backbone_weighted_sampling_filtered"):
        mlflow.log_params({
            "num_classes":        NUM_CLASSES,
            "num_epochs":         NUM_EPOCHS,
            "batch_size":         BATCH_SIZE,
            "lr":                 LR,
            "momentum":           MOMENTUM,
            "weight_decay":       WEIGHT_DECAY,
            "lr_step_size":       LR_STEP_SIZE,
            "lr_gamma":           LR_GAMMA,
            "backbone":           "resnet50_fpn_v2",
            "device":             str(device),
            "weighted_sampling":  True,
            "empty_filtered":     True,
            "train_size":         train_size,
            "val_size":           val_size,
        })

        best_val_loss = float("inf")
        best_epoch    = -1

        for epoch in range(NUM_EPOCHS):
            print(f"\n{'='*55}")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"{'='*55}")

            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss   = evaluate_loss(model, val_loader, device)
            scheduler.step()

            print(f"  Val loss: {val_loss:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch + 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch + 1
                best_state    = copy.deepcopy(model.state_dict())
                torch.save(best_state, BEST_WEIGHTS)
                print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

        # ── Save final artifacts ──────────────────────────────────────────────
        last_weights = os.path.join(OUTPUT_DIR, "last_faster_rcnn_run2.pth")
        torch.save(model.state_dict(), last_weights)
        mlflow.log_artifact(BEST_WEIGHTS,  artifact_path="models")
        mlflow.log_artifact(last_weights,  artifact_path="models")
        mlflow.log_params({"best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 6)})

        print(f"\nTraining complete!")
        print(f"Best val loss : {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Best weights  : {BEST_WEIGHTS}")


if __name__ == "__main__":
    main()