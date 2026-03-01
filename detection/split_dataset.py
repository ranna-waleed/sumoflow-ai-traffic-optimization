import os
import shutil
import random

# ─── Paths ───────────────────────────────────────────
RAW_IMAGES = "detection/dataset_v2/images/raw"
RAW_LABELS = "detection/dataset_v2/labels/raw"


# ─── Clean & Create Split Folders ────────────────────────
for split in ["train", "val", "test"]:
    img_dir = f"detection/dataset_v2/images/{split}"
    lbl_dir = f"detection/dataset_v2/labels/{split}"
    
    # If the folder exists from a previous run, delete it completely to prevent data leakage
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if os.path.exists(lbl_dir):
        shutil.rmtree(lbl_dir)
        
    # Create fresh empty folders
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    
# ─── Get Labeled Images Only ─────────────────────────
all_images = [
    f for f in os.listdir(RAW_IMAGES)
    if f.endswith(".png") and
    os.path.exists(f"{RAW_LABELS}/{f.replace('.png', '.txt')}")
]

print(f"Total labeled images found: {len(all_images)}")

# ─── Shuffle & Split ─────────────────────────────────
random.seed(42)
random.shuffle(all_images)

total     = len(all_images)
train_end = int(total * 0.70)
val_end   = int(total * 0.90)

train = all_images[:train_end]
val   = all_images[train_end:val_end]
test  = all_images[val_end:]

# ─── Copy Files ──────────────────────────────────────
def copy_files(file_list, split):
    for img in file_list:
        label = img.replace(".png", ".txt")
        shutil.copy(
            f"{RAW_IMAGES}/{img}",
            f"detection/dataset_v2/images/{split}/{img}"
        )
        shutil.copy(
            f"{RAW_LABELS}/{label}",
            f"detection/dataset_v2/labels/{split}/{label}"
        )

copy_files(train, "train")
copy_files(val,   "val")
copy_files(test,  "test")

# ─── Summary ─────────────────────────────────────────
print("\nDataset split complete!")
print(f"   Train : {len(train)} images  (70%)")
print(f"   Val   : {len(val)}   images  (20%)")
print(f"   Test  : {len(test)}  images  (10%)")