import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import os

# ─── 1. Configuration ──────────────────────────────────────────
NUM_CLASSES = 8 # 7 vehicles (car, bus, taxi, microbus, bicycle, truck, motorcycle) + 1 background
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_PATH = "detection/retinanet_best.pth"

# ─── 2. Model Initialization ───────────────────────────────────
def get_retinanet_model(num_classes):
    print("Loading pre-trained RetinaNet ResNet-50-FPN...")
    # Load the standard pre-trained model
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )
    
    # Extract the number of input features and anchors for the classifier
    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Replace the classification head to match our 8 classes (7 + background)
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model

# ─── 3. Training Loop ──────────────────────────────────────────
def main():
    # Setup device (GPU if available, otherwise CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")

    model = get_retinanet_model(NUM_CLASSES)
    model.to(device)

    # Note: Import your existing dataloader here! 
    # Assuming your dataloader.py has a function like get_train_loader()
    from dataloader import get_train_loader
    train_loader = get_train_loader(BATCH_SIZE)

    # Define the Optimizer (Stochastic Gradient Descent is very stable for RetinaNet)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    print("Starting fine-tuning on El-Tahrir dataset...")
    
    # To track the best loss and save the best weights
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train() # Set to training mode (enables internal Focal Loss calculation)
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move images and targets to GPU/CPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: RetinaNet returns a dictionary of losses during training
            loss_dict = model(images, targets)
            
            # Combine the classification (Focal Loss) and bounding box regression losses
            losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            #GRADIENT CLIPPING : This prevents the gradients from exploding into NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {losses.item():.4f}")

        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

        # Save the best model weights
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New best model saved to {SAVE_PATH}!")

    print("Training Complete. Best weights saved.")

if __name__ == "__main__":
    main()