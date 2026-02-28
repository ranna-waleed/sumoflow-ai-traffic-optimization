import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET

# Map the 7 classes to integers (PyTorch requires background to be 0, so we shift by +1)
CLASS_TO_IDX = {
    'car': 1, 'bus': 2, 'truck': 3, 'motorcycle': 4, 
    'taxi': 5, 'microbus': 6, 'bicycle': 7
}

class TahrirTrafficDataset(Dataset):
    def __init__(self, imgs_dir, xml_dir):
        self.imgs_dir = imgs_dir
        self.xml_dir = xml_dir
        self.imgs = list(sorted([f for f in os.listdir(imgs_dir) if f.endswith('.png')]))

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.imgs_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace('.png', '.xml'))
        
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        boxes, labels = [], []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                labels.append(CLASS_TO_IDX.get(label_name, 1))
                
                bndbox = obj.find('bndbox')
                boxes.append([
                    float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
                ])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    print("Testing PyTorch DataLoader...")
    dataset = TahrirTrafficDataset(
        imgs_dir="detection/dataset/images/train", 
        xml_dir="detection/dataset/annotations/train"
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Fetch one batch to prove it works
    images, targets = next(iter(data_loader))
    print(f"Successfully loaded batch of {len(images)} images!")
    print(f"Sample target labels mapped: {targets[0]['labels']}")