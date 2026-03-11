import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import os
import mlflow
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataloader import get_train_loader, get_val_loader

# ─── 1. Configuration ──────────────────────────────────────────────────────────
NUM_CLASSES    = 8
BATCH_SIZE     = 2
NUM_EPOCHS     = 60
LEARNING_RATE  = 0.001        # AdamW adaptive scaling — no spike risk
SAVE_PATH      = "detection/RetinaNet/retinanet_best.pth"
UNFREEZE_EPOCH = 15           # backbone unfreezes after this epoch
WARMUP_EPOCHS  = 5            # linear LR warmup before cosine decay begins

# ─── Resume config ─────────────────────────────────────────────────────────────
# START_EPOCH = the last COMPLETED epoch number shown in training output.
# "Epoch 30 done" → START_EPOCH = 30.   Fresh run → START_EPOCH = 0.
RESUME_WEIGHTS  = "/kaggle/input/datasets/roaaraafat/sumoflowai-best-weights/retinanet_best.pth"
START_EPOCH     = 30
RESUME_BEST_MAP = 0.3874      # mAP@0.5 of the checkpoint — only overwrite when beaten

# ─── Where cosine LR should be at START_EPOCH ─────────────────────────────────
# Compute manually: CosineAnnealingLR with T_max=45, eta_min=1e-6, after 15 steps.
# epoch 30 is step 15 into Phase 2 (started at epoch 15).
# lr(t) = eta_min + 0.5*(lr_max - eta_min)*(1 + cos(pi*t/T_max))
# For backbone: lr_max=0.00005, t=15, T_max=45
#   = 1e-6 + 0.5*(0.00005-1e-6)*(1 + cos(pi*15/45))
#   = 1e-6 + 0.5*0.000049*(1 + cos(pi/3))
#   = 1e-6 + 0.5*0.000049*(1 + 0.5) = 1e-6 + 0.5*0.000049*1.5 ≈ 0.0000378
# For cls head: lr_max=0.001,    same ratio → ≈ 0.0007560
# For reg head: lr_max=0.0005,   same ratio → ≈ 0.0003780
RESUME_LR_BACKBONE = 0.0000378
RESUME_LR_CLS      = 0.0007560
RESUME_LR_REG      = 0.0003780


# ─── 2. Model ──────────────────────────────────────────────────────────────────
def get_retinanet_model(num_classes):
    """
    Load RetinaNet with pretrained COCO backbone, replace classification head.

    Focal loss tuning (kept from run 7):
      alpha=0.35  — up-weight rare classes (motorcycle, bicycle)
      gamma=2.5   — focus harder on hard examples in dense Tahrir Square scenes
    """
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
            "lr": lr,        # 0.001
        },
        {
            "params": [p for p in model.head.regression_head.parameters()
                       if p.requires_grad],
            "lr": lr * 0.5,  # 0.0005
        },
    ], weight_decay=0.0005)


def build_phase2_optimizer(model, lr,
                           backbone_lr=None, cls_lr=None, reg_lr=None):
    """
    Phase 2: full fine-tuning with differential LRs.

    backbone_lr / cls_lr / reg_lr let the caller inject exact LR values
    (used on resume to restore the cosine position without stepping the
    scheduler, which would cause a v_t=0 explosion).

    KEY CHANGE vs run 7: default backbone LR is lr*0.05 (= 0.00005), halved
    from the previous lr*0.1 (= 0.0001) to prevent AdamW v_t accumulation
    from reaching the explosive threshold that caused the epoch-32 collapse.
    """
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
    resume_epoch = False   # flag: first epoch after resume needs gentle treatment

    # ── Resume logic ──────────────────────────────────────────────────────────
    # IMPORTANT: We do NOT step the scheduler N times to fast-forward it.
    # Stepping a fresh scheduler (with v_t=0) causes enormous effective steps
    # on the first real batch — this is what caused the epoch-31 spikes you saw.
    #
    # Instead we:
    #   1. Load weights.
    #   2. Build Phase-2 optimizer with the exact LR values that cosine decay
    #      would have produced at START_EPOCH (pre-computed above).
    #   3. Build a fresh CosineAnnealingLR with last_epoch set to the correct
    #      position so future scheduler.step() calls continue smoothly.
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

            # Build optimizer with the exact LR values at START_EPOCH.
            # This skips scheduler fast-forwarding entirely — no v_t=0 explosion.
            optimizer = build_phase2_optimizer(
                model, LEARNING_RATE,
                backbone_lr=RESUME_LR_BACKBONE,
                cls_lr=RESUME_LR_CLS,
                reg_lr=RESUME_LR_REG,
            )
            print(f"   backbone LR={RESUME_LR_BACKBONE:.7f} | "
                  f"cls LR={RESUME_LR_CLS:.7f} | "
                  f"reg LR={RESUME_LR_REG:.7f}  (direct set, no scheduler stepping)")

            # Build a fresh CosineAnnealingLR whose internal clock is at the
            # correct position.  Pass last_epoch= steps already done in Phase 2.
            # The scheduler will then produce the right LR on the next .step() call.
            epochs_into_phase2 = START_EPOCH - UNFREEZE_EPOCH  # = 15
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS - UNFREEZE_EPOCH,  # = 45
                eta_min=1e-6,
                last_epoch=epochs_into_phase2,       # ← correct clock position
            )
            phase = 2
            print(f"   CosineAnnealingLR last_epoch={epochs_into_phase2} "
                  f"(next .step() → epoch {epochs_into_phase2+1} of 45)\n")

        else:
            # Resuming inside Phase 1 — rebuild SequentialLR at correct position
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

            # On the very first epoch after resume the AdamW moment buffers are
            # empty (v_t=0).  The first few batches will see large effective step
            # sizes until v_t accumulates.  We tighten the spike threshold to 5.0
            # and add a 10-batch mini-warmup that scales LR from 10% → 100%.
            is_first_resume_epoch = resume_epoch and (epoch == START_EPOCH)
            spike_threshold = 5.0 if is_first_resume_epoch else 10.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Mini-warmup for first 10 batches of the first resume epoch.
                # Scales each param group LR from 10% → 100% over 10 batches.
                if is_first_resume_epoch and batch_idx < 10:
                    scale = 0.1 + 0.09 * batch_idx   # 0.10 → 0.91 → 1.0 at step 10
                    for group in optimizer.param_groups:
                        group["lr"] = group["lr"] * scale if batch_idx == 0 else group["lr"] / (0.1 + 0.09 * (batch_idx - 1)) * scale

                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())

                # Guard 1 — runaway loss
                if not torch.isfinite(losses) or losses.item() > spike_threshold:
                    print(f"⚠️  Batch {batch_idx} skipped — loss={losses.item():.2f} "
                          f"(threshold={spike_threshold})")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                losses.backward()

                # Guard 2 — NaN/Inf gradient
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

                # Restore correct LR after mini-warmup completes
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

            resume_epoch = False   # only applies to the very first epoch

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
                    max_detection_thresholds=[1, 10, 300]
                )
                with torch.no_grad():
                    for val_images, val_targets in val_loader:
                        val_images  = [img.to(device) for img in val_images]
                        val_outputs = model(val_images)
                        val_targets = [{k: v.cpu() for k, v in t.items()} for t in val_targets]
                        val_outputs = [{k: v.cpu() for k, v in o.items()} for o in val_outputs]
                        val_metric.update(val_outputs, val_targets)

                val_results      = val_metric.compute()
                current_map      = val_results["map_50"].item()
                current_map_coco = val_results["map"].item()

                mlflow.log_metric("val_mAP_50", current_map, step=epoch)
                if current_map_coco >= 0:
                    mlflow.log_metric("val_mAP_50_95", current_map_coco, step=epoch)
                print(f"Val mAP@0.5: {current_map:.4f} | mAP@0.5:0.95: {current_map_coco:.4f}")

                if current_map > best_map:
                    best_map = current_map
                    torch.save(model.state_dict(), SAVE_PATH)
                    mlflow.log_artifact(SAVE_PATH)
                    print(f"✅ New best model saved! mAP@0.5 = {best_map:.4f}")

    print(f"\nTraining complete. Best val mAP@0.5 = {best_map:.4f}")
    print("Weights saved to:", SAVE_PATH)


if __name__ == "__main__":
    main()