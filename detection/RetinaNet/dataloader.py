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

# ── ImageNet normalization constants (MUST match RetinaNet's pretrained backbone) ──
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TahrirTrafficDataset(Dataset):
    def __init__(self, imgs_dir, xml_dir, augment=False):
        self.imgs_dir = imgs_dir
        self.xml_dir  = xml_dir
        self.augment  = augment
        self.imgs     = list(sorted([f for f in os.listdir(imgs_dir) if f.endswith('.png')]))

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
                    print(f"Warning: Unknown class '{label_name}' in {img_name} — skipping.")
                    continue

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                if xmax <= xmin or ymax <= ymin:
                    print(f"Warning: Degenerate box [{xmin},{ymin},{xmax},{ymax}] in {img_name} — skipping.")
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASS_TO_IDX[label_name])

        # ── Augmentation (training only) ─────────────────────────────────
        if self.augment and len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            h, w  = img.shape[:2]

            # 1. Random horizontal flip
            if random.random() > 0.5:
                img        = img[:, ::-1, :].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            # 2. Random brightness / contrast jitter
            #    FIX: clip to [0, 255] and keep float32 clean BEFORE any cast
            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3)
                beta  = random.uniform(-20, 20)
                img   = np.clip(img * alpha + beta, 0, 255).astype(np.float32)

            # 3. Random HSV hue/saturation shift
            #    FIX: convert to uint8 AFTER previous step guarantees [0,255] range
            if random.random() > 0.5:
                img_u8  = img.astype(np.uint8)
                img_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + random.uniform(-18, 18)) % 180
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
                img = cv2.cvtColor(
                    np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
                ).astype(np.float32)

            # 4. Random vertical flip (mild — helps with top-down SUMO camera view)
            if random.random() > 0.8:
                img        = img[::-1, :, :].copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

            # 5. Random small rotation (±10°) — simulates slight camera angle drift
            if random.random() > 0.85:         # 15% chance  
                angle = random.uniform(-5, 5)  # smaller angle
                cx, cy = w / 2, h / 2
                M      = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                img    = cv2.warpAffine(img, M, (w, h)).astype(np.float32)
                # Transform box corners through the rotation matrix
                corners = np.array([
                    [boxes[:, 0], boxes[:, 1]],  # xmin, ymin
                    [boxes[:, 2], boxes[:, 1]],  # xmax, ymin
                    [boxes[:, 0], boxes[:, 3]],  # xmin, ymax
                    [boxes[:, 2], boxes[:, 3]],  # xmax, ymax
                ], dtype=np.float32)  # (4, 2, N)
                ones    = np.ones((1, corners.shape[2]), dtype=np.float32)
                rotated = []
                for ci in range(4):
                    pts = np.vstack([corners[ci], ones])  # (3, N)
                    rpt = M @ pts                         # (2, N)
                    rotated.append(rpt)
                rotated    = np.array(rotated)   # (4, 2, N)
                new_xmin   = rotated[:, 0, :].min(axis=0).clip(0, w)
                new_xmax   = rotated[:, 0, :].max(axis=0).clip(0, w)
                new_ymin   = rotated[:, 1, :].min(axis=0).clip(0, h)
                new_ymax   = rotated[:, 1, :].max(axis=0).clip(0, h)
                boxes      = np.stack([new_xmin, new_ymin, new_xmax, new_ymax], axis=1)
                # Remove boxes that collapsed after rotation
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes  = boxes[valid]
                labels = [l for l, v in zip(labels, valid) if v]

            boxes = boxes.tolist()

        # ── Normalize with ImageNet mean/std (CRITICAL for pretrained backbone) ──
        img_normalized = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # ── Safe tensor logic ─────────────────────────────────────────────
        if len(boxes) == 0:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_tensor  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        img_tensor = torch.as_tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)

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
        augment=True
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )


def get_val_loader(batch_size=1):
    # FIX: Use the dedicated 'val/' split — NOT 'test/'
    # Test set must remain completely unseen until final evaluation.
    dataset = TahrirTrafficDataset(
        imgs_dir="detection/dataset_v2/images/val",
        xml_dir="detection/dataset_v2/annotations/val",
        augment=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )


def get_test_loader(batch_size=1):
    """Completely held-out set — only call this in evaluate.py for final reporting."""
    dataset = TahrirTrafficDataset(
        imgs_dir="detection/dataset_v2/images/test",
        xml_dir="detection/dataset_v2/annotations/test",
        augment=False
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
    images, targets = next(iter(data_loader))
    print(f"Successfully loaded batch of {len(images)} images!")
    print(f"Image tensor range: [{images[0].min():.3f}, {images[0].max():.3f}]")  # Should be ~[-2, 2]
    print(f"Sample target labels: {targets[0]['labels']}")