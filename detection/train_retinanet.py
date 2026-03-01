import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import os
import mlflow  

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES = 8 
BATCH_SIZE = 2
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
SAVE_PATH = "detection/retinanet_best.pth"

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
    # NEW: Setup MLflow Experiment
    mlflow.set_experiment("SumoFlowAI-Traffic-Detection")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")

    model = get_retinanet_model(NUM_CLASSES)
    model.to(device)

    from dataloader import get_train_loader
    train_loader = get_train_loader(BATCH_SIZE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    best_loss = float('inf')

    # NEW: Start MLflow Run
    with mlflow.start_run(run_name="retinanet_v2_1800_images"):
        # Log Hyperparameters
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "SGD")

        print("Starting fine-tuning on El-Tahrir dataset...")
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {losses.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            
            # NEW: Log Metric to MLflow
            mlflow.log_metric("avg_loss", avg_epoch_loss, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

            print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), SAVE_PATH)
                # NEW: Log model artifact to MLflow
                mlflow.log_artifact(SAVE_PATH)
                print(f"New best model saved and logged to MLflow!")
                
            scheduler.step()

    print("Training Complete. Best weights saved.")

if __name__ == "__main__":
    main()