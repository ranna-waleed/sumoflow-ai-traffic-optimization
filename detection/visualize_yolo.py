import cv2
import os
import random

IMG_DIR = "detection/dataset/images/raw"
LBL_DIR = "detection/dataset/labels/raw"
CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'taxi', 'microbus', 'bicycle']

# Grab a random image from your 360 frames
imgs = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
img_name = random.choice(imgs)
txt_name = img_name.replace('.png', '.txt')

img_path = os.path.join(IMG_DIR, img_name)
txt_path = os.path.join(LBL_DIR, txt_name)

print(f"Testing frame: {img_name}")
img = cv2.imread(img_path)
h, w, _ = img.shape

# Read the YOLO labels and draw the boxes
if os.path.exists(txt_path):
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])

                # Convert YOLO percentages back to exact pixels
                xmin = int((x_center - box_w / 2) * w)
                ymin = int((y_center - box_h / 2) * h)
                xmax = int((x_center + box_w / 2) * w)
                ymax = int((y_center + box_h / 2) * h)

                # Draw a green rectangle and write the class name
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
                cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the test image
output_path = "detection/test_verification.png"
cv2.imwrite(output_path, img)
print(f"Success! Open '{output_path}' to see if the boxes align.")