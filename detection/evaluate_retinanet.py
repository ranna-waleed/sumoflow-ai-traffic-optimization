import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import cv2
import matplotlib.pyplot as plt
import os
import random

# ─── 1. Configuration & Mapping ────────────────────────────────
NUM_CLASSES = 8
CONFIDENCE_THRESHOLD = 0.50 # Only show boxes if the model is >50% sure
WEIGHTS_PATH = "detection/retinanet_best.pth"
TEST_DIR = "detection/dataset/images/test"

# Map PyTorch integers back to your string labels
IDX_TO_CLASS = {
    1: 'car', 2: 'bus', 3: 'truck', 4: 'motorcycle', 
    5: 'taxi', 6: 'microbus', 7: 'bicycle'
}

# ─── 2. Model Initialization (Same as training) ────────────────
def get_retinanet_model(num_classes):
    # Load the blank architecture (no pretrained weights this time, we use yours!)
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Swap the head to our 8 classes
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    return model

# ─── 3. Inference & Visualization ──────────────────────────────
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Evaluating on device: {device}")

    # Load your trained weights
    model = get_retinanet_model(NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Set model to evaluation mode!

    # Grab a random image from the test set
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith('.png')]
    if not test_images:
        print("No test images found!")
        return
        
    sample_img_name = random.choice(test_images)
    img_path = os.path.join(TEST_DIR, sample_img_name)
    
    # Load image with OpenCV
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PyTorch tensor format (C, H, W) and normalize (0-1)
    img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # Add fake batch dimension

    print(f"Running inference on {sample_img_name}...")
    
    # Run the model! No gradients needed for evaluation
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Filter out weak predictions below our confidence threshold
    mask = predictions['scores'] > CONFIDENCE_THRESHOLD
    boxes = predictions['boxes'][mask]
    labels = predictions['labels'][mask]
    scores = predictions['scores'][mask]
    
    print(f"Found {len(boxes)} vehicles with >{int(CONFIDENCE_THRESHOLD*100)}% confidence.")

    # Draw the boxes!
    img_draw = img_rgb.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = IDX_TO_CLASS.get(label.item(), 'Unknown')
        
        # Draw a neon green rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add a nice background for the text
        text = f"{class_name}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Put the label text in black
        cv2.putText(img_draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save the output image (Requirements for Issue #24)
    save_file = f"detection/pred_{sample_img_name}"
    cv2.imwrite(save_file, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    print(f"Saved evaluation image to {save_file}")

    # Show the image right here in Google Colab!
    plt.figure(figsize=(16, 10))
    plt.imshow(img_draw)
    plt.axis('off')
    plt.title(f"RetinaNet Detections (Loss ~0.74)")
    plt.show()

if __name__ == "__main__":
    main()