import os

print("Verifying dataset...\n")

for split in ["train", "val", "test"]:
    img_dir = f"detection/dataset/images/{split}"
    lbl_dir = f"detection/dataset/labels/{split}"

    images = set(f.replace(".png", "") for f in os.listdir(img_dir) if f.endswith(".png"))
    labels = set(f.replace(".txt", "") for f in os.listdir(lbl_dir) if f.endswith(".txt"))

    missing_labels = images - labels
    missing_images = labels - images

    print(f"📂 {split}:")
    print(f"   Images : {len(images)}")
    print(f"   Labels : {len(labels)}")

    if missing_labels:
        print(f"   Missing labels for: {missing_labels}")
    elif missing_images:
        print(f"   Missing images for: {missing_images}")
    else:
        print(f"   Perfect match!")
    print()

print("Verification complete!")