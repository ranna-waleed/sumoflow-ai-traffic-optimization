"""
Part 1: Environment Setup & Verification
- Verifies all required libraries are installed
- Tests DataLoader on 5 sample images
- Confirms GPU availability
"""

import sys
import os

def check_imports():
    print("=" * 60)
    print("Checking required libraries...")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] torch          {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("[FAIL] torch - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    try:
        import torchvision
        print(f"[OK] torchvision    {torchvision.__version__}")
    except ImportError:
        print("[FAIL] torchvision")

    try:
        import cv2
        print(f"[OK] opencv-python  {cv2.__version__}")
    except ImportError:
        print("[FAIL] opencv-python - run: pip install opencv-python")

    try:
        import mlflow
        print(f"[OK] mlflow         {mlflow.__version__}")
    except ImportError:
        print("[FAIL] mlflow - run: pip install mlflow")

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        import torchmetrics
        print(f"[OK] torchmetrics   {torchmetrics.__version__}")
    except ImportError:
        print("[FAIL] torchmetrics - run: pip install torchmetrics")

    try:
        import matplotlib
        print(f"[OK] matplotlib     {matplotlib.__version__}")
    except ImportError:
        print("[FAIL] matplotlib - run: pip install matplotlib")

    try:
        import numpy
        print(f"[OK] numpy          {numpy.__version__}")
    except ImportError:
        print("[FAIL] numpy - run: pip install numpy")

    print()


def verify_dataloader(n_samples=5):
    print("=" * 60)
    print("Verifying DataLoader on 5 sample images...")
    print("=" * 60)

    # Add parent dir to path so we can import dataloader
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    try:
        from dataloader import TahrirTrafficDataset, collate_fn, CLASS_TO_IDX
    except ImportError:
        print("[ERROR] Could not import dataloader.py — make sure it's in the FastRCNN/ folder or its parent.")
        return False

    import torch
    from torch.utils.data import DataLoader

    IMGS_DIR = "detection/dataset/images/train"
    XML_DIR  = "detection/dataset/annotations/train"

    if not os.path.isdir(IMGS_DIR):
        print(f"[WARNING] Images dir not found: {IMGS_DIR}")
        print("          Update IMGS_DIR / XML_DIR paths in this script to match your folder structure.")
        return False

    dataset = TahrirTrafficDataset(imgs_dir=IMGS_DIR, xml_dir=XML_DIR)
    print(f"[OK] Dataset size: {len(dataset)} images")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for i, (images, targets) in enumerate(loader):
        if i >= n_samples:
            break
        img = images[0]
        tgt = targets[0]
        print(f"  Sample {i+1}: shape={tuple(img.shape)}, "
              f"boxes={tgt['boxes'].shape[0]}, "
              f"labels={tgt['labels'].tolist()}")

    print(f"\n[OK] Successfully verified {min(n_samples, len(dataset))} samples!")
    return True


if __name__ == "__main__":
    check_imports()
    verify_dataloader(n_samples=5)
    print("\nSetup complete. You're ready to train!")