import os
import torch
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET

CLASS_TO_IDX = {
    'car': 1, 'bus': 2, 'truck': 3, 'motorcycle': 4,
    'taxi': 5, 'microbus': 6, 'bicycle': 7
}

class TahrirTrafficDataset(Dataset):
    def __init__(self, imgs_dir, xml_dir, augment=False):
        self.imgs_dir = imgs_dir
        self.xml_dir = xml_dir
        self.augment = augment
        self.imgs = list(sorted([f for f in os.listdir(imgs_dir) if f.endswith('.png')]))

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.imgs_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace('.png', '.xml'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        boxes, labels = [], []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                
                if label_name not in CLASS_TO_IDX:
                    print(f"Warning: Unknown class '{label_name}' found in annotation — skipping object.")
                    continue
                labels.append(CLASS_TO_IDX[label_name])
                
                bndbox = obj.find('bndbox')

                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Skip degenerate boxes — RetinaNet will NaN-loss on zero-area boxes
                if xmax <= xmin or ymax <= ymin:
                    print(f"Warning: Degenerate box [{xmin},{ymin},{xmax},{ymax}] in {img_name} — skipping.")
                    labels.pop()  # remove the label we already appended above
                    continue

                boxes.append([xmin, ymin, xmax, ymax])

        # ── Augmentation (training only) ──────────────────────────
        if self.augment and len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            h, w = img.shape[:2]

            # 1. Random horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            # 2. Random brightness / contrast jitter
            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3)   # contrast
                beta  = random.uniform(-20, 20)    # brightness
                img = np.clip(img * alpha + beta, 0, 255)

            # 3. Random HSV hue/saturation shift
            if random.random() > 0.5:
                img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + random.uniform(-18, 18)) % 180
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
                img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

            boxes = boxes.tolist()

        # ── Safe tensor logic ──────────────────────────────────────
        if len(boxes) == 0:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_tensor  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        img_tensor = torch.as_tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

        target = {
            "boxes":    boxes_tensor,
            "labels":   labels_tensor,
            "image_id": torch.tensor([idx])
        }
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_loader(batch_size=2):
    dataset = TahrirTrafficDataset(
        imgs_dir="detection/dataset_v2/images/train",
        xml_dir="detection/dataset_v2/annotations/train",
        augment=True   # ← augmentation ON for training
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,        # parallel CPU workers pre-load batches while GPU trains
        pin_memory=True       # faster CPU→GPU transfer when using CUDA
    )

def get_val_loader(batch_size=1):
    dataset = TahrirTrafficDataset(
        imgs_dir="detection/dataset_v2/images/test",
        xml_dir="detection/dataset_v2/annotations/test",
        augment=False  # ← augmentation OFF for validation
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

if __name__ == "__main__":
    print("Testing PyTorch DataLoader...")
    dataset = TahrirTrafficDataset(
        imgs_dir="detection/dataset_v2/images/train", 
        xml_dir="detection/dataset_v2/annotations/train"
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Fetch one batch to prove it works
    images, targets = next(iter(data_loader))
    print(f"Successfully loaded batch of {len(images)} images!")
    print(f"Sample target labels mapped: {targets[0]['labels']}")