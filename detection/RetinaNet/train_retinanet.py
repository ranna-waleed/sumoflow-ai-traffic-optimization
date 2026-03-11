import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import os
import mlflow
import subprocess
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader import get_train_loader, get_val_loader  # get_val_loader now uses val/ not test/

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES    = 8
BATCH_SIZE     = 2
NUM_EPOCHS     = 60
LEARNING_RATE  = 0.001
SAVE_PATH      = "detection/RetinaNet/retinanet_best.pth"
UNFREEZE_EPOCH = 15      # backbone unfreezes after this epoch
WARMUP_EPOCHS  = 3       # linear LR warmup before cosine decay begins

# ─── 2. Model Initialization ───────────────────────────────────
def get_retinanet_model(num_classes):
    """
    Load RetinaNet with pretrained COCO backbone, then replace only the
    classification head for our 7-class problem.

    Focal loss tuning:
      - alpha=0.25 is the default. Increase toward 0.5 to up-weight rare classes
        (motorcycle, bicycle) — try 0.35 if small-class AP is poor after training.
      - gamma=2.0 is the default. Increase to 3.0 to focus harder on hard examples
        in dense scenes like Tahrir Square.
    """
    print("Loading pre-trained RetinaNet ResNet-50-FPN...")
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )

    # Tune focal loss for dense, imbalanced Egyptian traffic
    model.head.classification_head.focal_loss_alpha = 0.35   # was 0.25
    model.head.classification_head.focal_loss_gamma = 2.5    # was 2.0

    in_channels  = model.head.classification_head.conv[0][0].in_channels
    num_anchors  = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    # Re-apply focal loss tuning after head replacement (head init resets them)
    model.head.classification_head.focal_loss_alpha = 0.35
    model.head.classification_head.focal_loss_gamma = 2.5

    return model


def get_warmup_scheduler(optimizer, warmup_epochs, base_lr):
    """Linear warmup: LR ramps from base_lr/10 → base_lr over warmup_epochs."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    return torch.optim.lr_scheduler.LambdaScheduler(optimizer, lr_lambda)


# ─── 3. Training Loop ──────────────────────────────────────────
def main():
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")

    model = get_retinanet_model(NUM_CLASSES)
    model.to(device)

    train_loader = get_train_loader(BATCH_SIZE)
    val_loader   = get_val_loader(batch_size=1)   # ← val/ split, not test/

    # ── Phase 1: Freeze backbone, train head only ───────────────
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.SGD(head_params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Warmup for WARMUP_EPOCHS, then cosine decay for the rest of Phase 1
    warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UNFREEZE_EPOCH - WARMUP_EPOCHS, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    best_map    = 0.0
    phase       = 1   # for logging clarity

    with mlflow.start_run(run_name="retinanet_v2_1800_images_improved"):
        mlflow.log_params({
            "num_epochs":      NUM_EPOCHS,
            "learning_rate":   LEARNING_RATE,
            "batch_size":      BATCH_SIZE,
            "optimizer":       "SGD",
            "unfreeze_epoch":  UNFREEZE_EPOCH,
            "warmup_epochs":   WARMUP_EPOCHS,
            "focal_alpha":     0.35,
            "focal_gamma":     2.5,
        })

        print("Starting fine-tuning on El-Tahrir dataset...")

        for epoch in range(NUM_EPOCHS):

            # ── Phase 2: Unfreeze backbone with differential LR ──
            if epoch == UNFREEZE_EPOCH:
                print("Unfreezing backbone for full fine-tuning...")
                for param in model.backbone.parameters():
                    param.requires_grad = True

                params = [
                    {"params": model.backbone.parameters(), "lr": LEARNING_RATE * 0.1},
                    {"params": model.head.parameters(),     "lr": LEARNING_RATE}
                ]
                optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=0.0005)

                # FIX: Build a fresh cosine scheduler starting from the current LR.
                # Do NOT step this scheduler on the same epoch it's created.
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH, eta_min=1e-6
                )
                phase = 2
                print(f"Optimizer rebuilt | backbone LR={LEARNING_RATE*0.1:.6f} | head LR={LEARNING_RATE:.6f}")

            # ── Training ─────────────────────────────────────────
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses) or losses.item() > 10.0:
                    print(f"⚠️  Non-finite loss at epoch {epoch+1} batch {batch_idx} — skipping.")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss += losses.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Phase {phase} | "
                          f"Batch [{batch_idx}/{len(train_loader)}] | Loss: {losses.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("avg_loss", avg_epoch_loss, step=epoch)

            # FIX: scheduler.step() happens here, AFTER the epoch, NOT before it's rebuilt
            scheduler.step()

            try:
                current_lr = scheduler.get_last_lr()[0]
            except Exception:
                current_lr = LEARNING_RATE
            mlflow.log_metric("lr", current_lr, step=epoch)

            print(f"─── Epoch {epoch+1} done | Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.7f} ───")

            # ── Validation mAP check every 5 epochs ──────────────
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
                        val_targets = [{k: v.cpu() for k, v in t.items()} for t in val_targets]
                        val_outputs = [{k: v.cpu() for k, v in o.items()} for o in val_outputs]
                        val_metric.update(val_outputs, val_targets)

                val_results  = val_metric.compute()
                current_map  = val_results['map_50'].item()
                current_map_coco = val_results['map'].item()

                mlflow.log_metric("val_mAP_50",      current_map,      step=epoch)
                mlflow.log_metric("val_mAP_50_95",   current_map_coco, step=epoch)
                print(f"Val mAP@0.5: {current_map:.4f} | mAP@0.5:0.95: {current_map_coco:.4f}")

                if current_map > best_map:
                    best_map = current_map
                    torch.save(model.state_dict(), SAVE_PATH)
                    mlflow.log_artifact(SAVE_PATH)
                    print(f"✅ New best model saved! mAP@0.5 = {best_map:.4f}")

    print(f"Training complete. Best val mAP@0.5 = {best_map:.4f}")
    print("Weights saved to:", SAVE_PATH)


if __name__ == "__main__":
    main()