import cv2
import os
import numpy as np

IMG_DIR = "detection/dataset/images/raw"
LBL_DIR = "detection/dataset/labels/raw"
os.makedirs(LBL_DIR, exist_ok=True)

# Map our YOLO classes to the EXACT BGR colors from the Extractor
CLASS_COLORS_BGR = {
    0: [255, 0, 255],     # car: Magenta -> BGR [255, 0, 255]
    1: [255, 255, 0],     # bus: Cyan -> BGR [255, 255, 0]
    2: [150, 255, 0],     # truck: Mint Green -> BGR [150, 255, 0]
    3: [0, 150, 255],     # motorcycle: Neon Orange -> BGR [0, 150, 255]
    4: [255, 0, 150],     # taxi: Electric Purple -> BGR [255, 0, 150]
    5: [200, 100, 255],   # microbus: Hot Pink -> BGR [200, 100, 255]
    6: [255, 150, 0]      # bicycle: Bright Blue -> BGR [255, 150, 0]
}

def generate_labels():
    images = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    print(f"Starting OpenCV Auto-Labeling on {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(IMG_DIR, img_name)
        txt_path = os.path.join(LBL_DIR, img_name.replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        yolo_labels = []

        # Scan the image for each specific class color
        for class_id, bgr_color in CLASS_COLORS_BGR.items():
            
            # CHANGED: Widened the tolerance from 5 to 50 to catch SUMO's shading/anti-aliasing!
            lower_bound = np.array([max(0, c - 50) for c in bgr_color])
            upper_bound = np.array([min(255, c + 50) for c in bgr_color])
            
            # Mask out everything except this specific color
            mask = cv2.inRange(img, lower_bound, upper_bound)
            
            # Find the distinct blobs of color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # Get the pixel-perfect bounding box around the blob
                x, y, box_w, box_h = cv2.boundingRect(cnt)
                
                # Filter out tiny noise (increased from 3 to 5 to avoid UI pixels)
                # Lower the threshold so we don't accidentally filter out tiny bikes!
                if box_w < 2 or box_h < 2: 
                    continue
                
                # Convert to YOLO format
                center_x = (x + (box_w / 2)) / w
                center_y = (y + (box_h / 2)) / h
                norm_w = box_w / w
                norm_h = box_h / h
                
                yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Write all perfectly aligned boxes to the YOLO file
        with open(txt_path, 'w') as f:
            f.writelines(yolo_labels)

    print("OpenCV Labeling Complete! Pixel-perfect YOLO labels generated.")

if __name__ == "__main__":
    generate_labels()