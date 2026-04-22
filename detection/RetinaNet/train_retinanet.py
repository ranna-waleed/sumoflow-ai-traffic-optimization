import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import os
import mlflow
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader import get_train_loader, get_val_loader

# ─── 1. Configuration ──────────────────────────────────────────────────────────
NUM_CLASSES    = 7
BATCH_SIZE     = 2
NUM_EPOCHS     = 60
LEARNING_RATE  = 0.001
SAVE_PATH      = "detection/RetinaNet/retinanet_best.pth"
# Backup outside the repo — safe from accidental !cp overwrites
BACKUP_SAVE_PATH = "/kaggle/working/retinanet_best_BACKUP.pth"
UNFREEZE_EPOCH = 15
WARMUP_EPOCHS  = 5

# ─── Resume config ─────────────────────────────────────────────────────────────
# START_EPOCH = the last COMPLETED epoch number shown in training output.
# "Epoch 30 done" → START_EPOCH = 30.   Fresh run → START_EPOCH = 0.
RESUME_WEIGHTS  = "/kaggle/input/datasets/roaaraafat/sumoflowai-best-weights/retinanet_best.pth"
START_EPOCH     = 30
RESUME_BEST_MAP = 0.3874

# ─── Where cosine LR should be at START_EPOCH ─────────────────────────────────
# lr(t) = eta_min + 0.5*(lr_max - eta_min)*(1 + cos(pi*t/T_max))
# backbone: lr_max=0.00005, t=15, T_max=45 → 0.0000378
# cls head: lr_max=0.001,   same ratio    → 0.0007560
# reg head: lr_max=0.0005,  same ratio    → 0.0003780
RESUME_LR_BACKBONE = 0.0000378
RESUME_LR_CLS      = 0.0007560
RESUME_LR_REG      = 0.0003780

# Phase-2 peak LRs (lr_max in cosine formula — used as initial_lr by scheduler)
PHASE2_BASE_LR_BACKBONE = LEARNING_RATE * 0.05   # 0.00005
PHASE2_BASE_LR_CLS      = LEARNING_RATE           # 0.001
PHASE2_BASE_LR_REG      = LEARNING_RATE * 0.5     # 0.0005


# ─── 2. Model ──────────────────────────────────────────────────────────────────
def get_retinanet_model(num_classes):
    print("Loading pre-trained RetinaNet ResNet-50-FPN...")
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )
    model.head.classification_head.focal_loss_alpha = 0.35
    model.head.classification_head.focal_loss_gamma = 2.5

    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    # Re-apply after head replacement (head __init__ resets them to defaults)
    model.head.classification_head.focal_loss_alpha = 0.35
    model.head.classification_head.focal_loss_gamma = 2.5
    return model


# ─── 3. Optimizers ─────────────────────────────────────────────────────────────
def build_phase1_optimizer(model, lr):
    """Phase 1: backbone frozen, heads only."""
    return torch.optim.AdamW([
        {
            "params": [p for p in model.head.classification_head.parameters()
                       if p.requires_grad],
            "lr": lr,
        },
        {
            "params": [p for p in model.head.regression_head.parameters()
                       if p.requires_grad],
            "lr": lr * 0.5,
        },
    ], weight_decay=0.0005)


def build_phase2_optimizer(model, lr,
                           backbone_lr=None, cls_lr=None, reg_lr=None):
    """Phase 2: full fine-tuning with differential LRs."""
    b_lr = backbone_lr if backbone_lr is not None else lr * 0.05
    c_lr = cls_lr      if cls_lr      is not None else lr
    r_lr = reg_lr      if reg_lr      is not None else lr * 0.5

    return torch.optim.AdamW([
        {
            "params": model.backbone.parameters(),
            "lr": b_lr,
        },
        {
            "params": [p for p in model.head.classification_head.parameters()
                       if p.requires_grad],
            "lr": c_lr,
        },
        {
            "params": [p for p in model.head.regression_head.parameters()
                       if p.requires_grad],
            "lr": r_lr,
        },
    ], weight_decay=0.0005)


# ─── 4. Training Loop ──────────────────────────────────────────────────────────
def main():
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training on device: {device}")

    model = get_retinanet_model(NUM_CLASSES)
    model.to(device)

    train_loader = get_train_loader(BATCH_SIZE)
    val_loader   = get_val_loader(batch_size=1)

    # ── Phase 1 setup (always built first; may be replaced below) ─────────────
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = build_phase1_optimizer(model, LEARNING_RATE)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UNFREEZE_EPOCH - WARMUP_EPOCHS, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    best_map     = 0.0
    phase        = 1
    resume_epoch = False

    # ── Resume logic ──────────────────────────────────────────────────────────
    if START_EPOCH > 0 and RESUME_WEIGHTS:
        print(f"\n▶  Loading checkpoint: {RESUME_WEIGHTS}")
        state = torch.load(RESUME_WEIGHTS, map_location=device)
        model.load_state_dict(state)
        print(f"   Weights loaded. Resuming from epoch {START_EPOCH + 1}.")

        best_map     = RESUME_BEST_MAP
        resume_epoch = True
        print(f"   best_map = {best_map:.4f}")

        if START_EPOCH >= UNFREEZE_EPOCH:
            print(f"   Resuming inside Phase 2 (epoch {START_EPOCH+1})")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = build_phase2_optimizer(
                model, LEARNING_RATE,
                backbone_lr=RESUME_LR_BACKBONE,
                cls_lr=RESUME_LR_CLS,
                reg_lr=RESUME_LR_REG,
            )
            print(f"   backbone LR={RESUME_LR_BACKBONE:.7f} | "
                  f"cls LR={RESUME_LR_CLS:.7f} | "
                  f"reg LR={RESUME_LR_REG:.7f}  (direct set, no scheduler stepping)")

            # FIX: stamp initial_lr into param groups before building
            # CosineAnnealingLR with last_epoch > 0, otherwise PyTorch raises:
            # KeyError: "param 'initial_lr' is not specified in param_groups[0]"
            phase2_base_lrs = [
                PHASE2_BASE_LR_BACKBONE,
                PHASE2_BASE_LR_CLS,
                PHASE2_BASE_LR_REG,
            ]
            for group, base_lr in zip(optimizer.param_groups, phase2_base_lrs):
                group["initial_lr"] = base_lr

            epochs_into_phase2 = START_EPOCH - UNFREEZE_EPOCH
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS - UNFREEZE_EPOCH,
                eta_min=1e-6,
                last_epoch=epochs_into_phase2,
            )
            phase = 2

            actual_lrs = scheduler.get_last_lr()
            print(f"   CosineAnnealingLR last_epoch={epochs_into_phase2} "
                  f"(next .step() → epoch {epochs_into_phase2+1} of 45)")
            print(f"   Scheduler LR check → backbone={actual_lrs[0]:.7f} "
                  f"cls={actual_lrs[1]:.7f} reg={actual_lrs[2]:.7f}")
            print(f"   Expected           → backbone={RESUME_LR_BACKBONE:.7f} "
                  f"cls={RESUME_LR_CLS:.7f} reg={RESUME_LR_REG:.7f}\n")

        else:
            optimizer = build_phase1_optimizer(model, LEARNING_RATE)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=WARMUP_EPOCHS
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=UNFREEZE_EPOCH - WARMUP_EPOCHS, eta_min=1e-5,
                last_epoch=max(START_EPOCH - WARMUP_EPOCHS, 0)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[WARMUP_EPOCHS],
                last_epoch=START_EPOCH,
            )
            print(f"   Phase-1 scheduler resumed at step {START_EPOCH}\n")

    # ── MLflow run ────────────────────────────────────────────────────────────
    run_label = (f"retinanet_v3_resume_ep{START_EPOCH+1}"
                 if START_EPOCH > 0 else "retinanet_v2_adamw_per_head_lr")

    with mlflow.start_run(run_name=run_label):
        mlflow.log_params({
            "num_epochs":               NUM_EPOCHS,
            "learning_rate":            LEARNING_RATE,
            "batch_size":               BATCH_SIZE,
            "optimizer":                "AdamW",
            "unfreeze_epoch":           UNFREEZE_EPOCH,
            "warmup_epochs":            WARMUP_EPOCHS,
            "focal_alpha":              0.35,
            "focal_gamma":              2.5,
            "regression_lr_multiplier": 0.5,
            "backbone_lr_multiplier":   0.05,
            "resume_from_epoch":        START_EPOCH,
            "resume_best_map":          RESUME_BEST_MAP,
        })

        print(f"Training epochs {START_EPOCH + 1} → {NUM_EPOCHS} ...\n")

        for epoch in range(START_EPOCH, NUM_EPOCHS):

            # ── Phase 2 transition (fresh-start run only) ─────────────────────
            if epoch == UNFREEZE_EPOCH and START_EPOCH < UNFREEZE_EPOCH:
                print("\nUnfreezing backbone for full fine-tuning...")
                for param in model.backbone.parameters():
                    param.requires_grad = True

                optimizer = build_phase2_optimizer(model, LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH, eta_min=1e-6
                )
                phase = 2
                print(f"Optimizer rebuilt | backbone LR={LEARNING_RATE*0.05:.6f} | "
                      f"cls LR={LEARNING_RATE:.6f} | reg LR={LEARNING_RATE*0.5:.6f}\n")

            # ── Training pass ─────────────────────────────────────────────────
            model.train()
            epoch_loss   = 0.0
            batches_used = 0

            is_first_resume_epoch = resume_epoch and (epoch == START_EPOCH)
            spike_threshold = 5.0 if is_first_resume_epoch else 10.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Mini-warmup: scale LR 10%→100% over first 10 batches of resume epoch
                if is_first_resume_epoch and batch_idx < 10:
                    scale = 0.1 + 0.09 * batch_idx
                    for group in optimizer.param_groups:
                        group["lr"] = group["lr"] * scale if batch_idx == 0 \
                            else group["lr"] / (0.1 + 0.09 * (batch_idx - 1)) * scale

                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses) or losses.item() > spike_threshold:
                    print(f"⚠️  Batch {batch_idx} skipped — loss={losses.item():.2f} "
                          f"(threshold={spike_threshold})")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                losses.backward()

                nan_grad = any(
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    for p in model.parameters() if p.grad is not None
                )
                if nan_grad:
                    print(f"⚠️  Batch {batch_idx} skipped — NaN/Inf gradient detected")
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                # Restore exact cosine LR after mini-warmup completes
                if is_first_resume_epoch and batch_idx == 9:
                    optimizer.param_groups[0]["lr"] = RESUME_LR_BACKBONE
                    optimizer.param_groups[1]["lr"] = RESUME_LR_CLS
                    optimizer.param_groups[2]["lr"] = RESUME_LR_REG
                    print(f"   Mini-warmup complete — LR restored to cosine position")

                epoch_loss   += losses.item()
                batches_used += 1

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Phase {phase} | "
                          f"Batch [{batch_idx}/{len(train_loader)}] | "
                          f"Loss: {losses.item():.4f}")

            resume_epoch = False

            avg_epoch_loss = epoch_loss / max(batches_used, 1)
            mlflow.log_metric("avg_loss",     avg_epoch_loss, step=epoch)
            mlflow.log_metric("batches_used", batches_used,   step=epoch)

            scheduler.step()

            try:
                current_lr = scheduler.get_last_lr()[0]
            except Exception:
                current_lr = LEARNING_RATE
            mlflow.log_metric("lr", current_lr, step=epoch)

            print(f"─── Epoch {epoch+1} done | "
                  f"Loss: {avg_epoch_loss:.4f} | "
                  f"LR: {current_lr:.7f} | "
                  f"Batches used: {batches_used}/{len(train_loader)} ───\n")

            # ── Validation mAP every 5 epochs ─────────────────────────────────
            if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
                model.eval()
                val_metric = MeanAveragePrecision(
                    box_format="xyxy",
                    class_metrics=False,  # keep fast during training
                )
                with torch.no_grad():
                    for val_images, val_targets in val_loader:
                        val_images  = [img.to(device) for img in val_images]
                        val_outputs = model(val_images)
                        val_targets = [{k: v.cpu() for k, v in t.items()}
                                       for t in val_targets]
                        val_outputs = [{k: v.cpu() for k, v in o.items()}
                                       for o in val_outputs]
                        # Remap 1-indexed → 0-indexed for torchmetrics
                        val_targets = [{**t, "labels": t["labels"] - 1}
                                       for t in val_targets]
                        val_outputs = [{**o, "labels": o["labels"] - 1}
                                       for o in val_outputs]
                        val_metric.update(val_outputs, val_targets)

                val_results      = val_metric.compute()
                current_map      = val_results["map_50"].item()
                current_map_coco = val_results["map"].item()

                mlflow.log_metric("val_mAP_50",    current_map,      step=epoch)
                mlflow.log_metric("val_mAP_50_95", current_map_coco, step=epoch)
                print(f"Val mAP@0.5: {current_map:.4f} | "
                      f"mAP@0.5:0.95: {current_map_coco:.4f}")

                if current_map > best_map:
                    best_map = current_map
                    torch.save(model.state_dict(), SAVE_PATH)
                    # Backup outside repo — safe from accidental overwrites
                    torch.save(model.state_dict(), BACKUP_SAVE_PATH)
                    mlflow.log_artifact(SAVE_PATH)
                    print(f"✅ New best model saved! mAP@0.5 = {best_map:.4f}")
                    print(f"   Backup → {BACKUP_SAVE_PATH}")

    print(f"\nTraining complete. Best val mAP@0.5 = {best_map:.4f}")
    print(f"Weights saved to: {SAVE_PATH}")
    print(f"Backup saved to:  {BACKUP_SAVE_PATH}")


if __name__ == "__main__":
    main()