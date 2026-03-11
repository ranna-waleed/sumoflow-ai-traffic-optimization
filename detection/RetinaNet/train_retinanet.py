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
# Set RESUME_WEIGHTS to the Kaggle input path of the best checkpoint.
# Set START_EPOCH to the number of the last completed epoch (0-indexed).
#   e.g. checkpoint saved at epoch 30 (displayed as "Epoch 30") → START_EPOCH = 30
# Set RESUME_BEST_MAP to the val mAP@0.5 of that checkpoint so we only
#   overwrite it when we actually beat it.
# To train from scratch: set RESUME_WEIGHTS = None and START_EPOCH = 0.
RESUME_WEIGHTS  = "/kaggle/input/datasets/roaaraafat/sumoflowai-best-weights/retinanet_best.pth"
START_EPOCH     = 30          # last completed epoch (0-indexed); loop starts at 30
RESUME_BEST_MAP = 0.3874      # mAP@0.5 of the checkpoint — prevents overwriting with worse


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
    """
    Phase 1: backbone frozen, train heads only.
    Regression head gets lr*0.5 — regression gradients are ~10x larger on
    dense frames so a lower LR prevents runaway updates.
    AdamW replaces SGD — per-parameter adaptive scaling eliminates the
    momentum-driven spikes that killed runs 1-6.
    """
    return torch.optim.AdamW([
        {
            "params": [p for p in model.head.classification_head.parameters()
                       if p.requires_grad],
            "lr": lr,           # 0.001
        },
        {
            "params": [p for p in model.head.regression_head.parameters()
                       if p.requires_grad],
            "lr": lr * 0.5,     # 0.0005
        },
    ], weight_decay=0.0005)


def build_phase2_optimizer(model, lr):
    """
    Phase 2: full fine-tuning with differential LRs.

    KEY CHANGE vs run 7: backbone LR is lr*0.05 (= 0.00005), halved from
    the previous lr*0.1 (= 0.0001).

    Root cause of the epoch-32 catastrophic collapse: AdamW accumulates a
    running average of squared gradients (v_t) per parameter.  After 16
    Phase-2 epochs at backbone LR=0.0001 some v_t values grew very large.
    A degenerate batch then produced a gradient whose effective step size
    (g / sqrt(v_t + eps)) exploded → NaN weights → all subsequent batches
    skipped.  Halving backbone LR to 0.00005 keeps effective steps in a
    safe range for the remaining ~30 epochs without sacrificing adaptation.
    """
    return torch.optim.AdamW([
        {
            "params": model.backbone.parameters(),
            "lr": lr * 0.05,    # 0.00005 — HALVED from run 7's 0.0001
        },
        {
            "params": [p for p in model.head.classification_head.parameters()
                       if p.requires_grad],
            "lr": lr,           # 0.001
        },
        {
            "params": [p for p in model.head.regression_head.parameters()
                       if p.requires_grad],
            "lr": lr * 0.5,     # 0.0005
        },
    ], weight_decay=0.0005)


def reset_optimizer_state(optimizer):
    """
    Wipe all accumulated AdamW first/second-moment buffers.
    Called on resume so stale moments from run 7's Phase 2 (which caused
    the epoch-32 collapse) do not carry into this resumed run.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                optimizer.state[p] = {}
    print("🔄 Optimizer moment state reset — fresh AdamW buffers")


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

    # ── Phase 1 optimizer + scheduler (always built first) ────────────────────
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

    best_map = 0.0
    phase    = 1

    # ── Resume logic ──────────────────────────────────────────────────────────
    # If resuming, we:
    #   1. Load weights
    #   2. Decide if we are in Phase 1 or Phase 2
    #   3. Rebuild the correct optimizer + scheduler and fast-forward its state
    #   4. Reset optimizer moments (prevents carrying stale v_t into this run)
    if START_EPOCH > 0 and RESUME_WEIGHTS:
        print(f"\n▶  Loading checkpoint: {RESUME_WEIGHTS}")
        state = torch.load(RESUME_WEIGHTS, map_location=device)
        model.load_state_dict(state)
        print(f"   Weights loaded. Resuming from epoch {START_EPOCH + 1}.")

        best_map = RESUME_BEST_MAP
        print(f"   best_map = {best_map:.4f} (checkpoint value — will only save when beaten)")

        if START_EPOCH >= UNFREEZE_EPOCH:
            # ── Resuming inside Phase 2 ──────────────────────────────────────
            print(f"   START_EPOCH ({START_EPOCH}) ≥ UNFREEZE_EPOCH ({UNFREEZE_EPOCH})")
            print("   Rebuilding Phase-2 optimizer with halved backbone LR and fresh moments...")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = build_phase2_optimizer(model, LEARNING_RATE)
            reset_optimizer_state(optimizer)   # ← critical: wipe stale v_t

            # Phase-2 cosine scheduler spans epochs UNFREEZE_EPOCH → NUM_EPOCHS
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS - UNFREEZE_EPOCH,
                eta_min=1e-6
            )
            # Fast-forward to the position we were at when training stopped
            epochs_into_phase2 = START_EPOCH - UNFREEZE_EPOCH
            for _ in range(epochs_into_phase2):
                scheduler.step()

            phase = 2
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Scheduler fast-forwarded {epochs_into_phase2} steps.")
            print(f"   backbone LR={LEARNING_RATE*0.05:.6f} | "
                  f"cls LR={LEARNING_RATE:.6f} | "
                  f"reg LR={LEARNING_RATE*0.5:.6f} | "
                  f"cosine position LR={current_lr:.7f}\n")

        else:
            # ── Resuming inside Phase 1 ──────────────────────────────────────
            for _ in range(START_EPOCH):
                scheduler.step()
            reset_optimizer_state(optimizer)
            print(f"   Phase-1 scheduler fast-forwarded {START_EPOCH} steps.\n")

    # ── MLflow run ────────────────────────────────────────────────────────────
    run_label = (f"retinanet_v3_resume_ep{START_EPOCH+1}_halved_backbone_lr"
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
            "backbone_lr_multiplier":   0.05,   # halved vs run 7
            "resume_from_epoch":        START_EPOCH,
            "resume_best_map":          RESUME_BEST_MAP,
        })

        print(f"Training epochs {START_EPOCH + 1} → {NUM_EPOCHS} ...\n")

        for epoch in range(START_EPOCH, NUM_EPOCHS):

            # ── Phase 2 transition (fresh run only — resume handles this above) ──
            if epoch == UNFREEZE_EPOCH and START_EPOCH < UNFREEZE_EPOCH:
                print("\nUnfreezing backbone for full fine-tuning...")
                for param in model.backbone.parameters():
                    param.requires_grad = True

                optimizer = build_phase2_optimizer(model, LEARNING_RATE)
                # No reset needed here — this is a fresh Phase-2 start, no
                # accumulated moments exist yet.

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

            for batch_idx, (images, targets) in enumerate(train_loader):
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())

                # Guard 1 — runaway loss value
                if not torch.isfinite(losses) or losses.item() > 10.0:
                    print(f"⚠️  Batch {batch_idx} skipped — loss={losses.item():.2f}")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                losses.backward()

                # Guard 2 — NaN/Inf gradient (new defence against epoch-32 collapse)
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

                epoch_loss   += losses.item()
                batches_used += 1

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Phase {phase} | "
                          f"Batch [{batch_idx}/{len(train_loader)}] | "
                          f"Loss: {losses.item():.4f}")

            # Guard against all-skipped epoch (NaN weights would give 0/0)
            avg_epoch_loss = epoch_loss / max(batches_used, 1)
            mlflow.log_metric("avg_loss",     avg_epoch_loss, step=epoch)
            mlflow.log_metric("batches_used", batches_used,   step=epoch)

            # scheduler.step() always fires AFTER the full epoch.
            # It is never called on the same epoch the scheduler was just built
            # (Phase-2 transition and resume fast-forward are both handled before
            # the loop body, so the first step() here is safe).
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