import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ..RetinaNet.dataloader import TahrirTrafficDataset

ds = TahrirTrafficDataset(
    imgs_dir="detection/dataset/images/train",
    xml_dir="detection/dataset/annotations/train"
)

counts = Counter()
empty  = 0

for _, t in ds:
    if t['boxes'].shape[0] == 0:
        empty += 1
    for l in t['labels'].tolist():
        counts[l] += 1

names = {1:'car', 2:'bus', 3:'truck', 4:'motorcycle',
         5:'taxi', 6:'microbus', 7:'bicycle'}

total_instances = sum(counts.values())

print(f"\nDataset: {len(ds)} images")
print(f"Empty images (no boxes): {empty}")
print(f"Total annotated instances: {total_instances}")
print(f"\nClass distribution:")
print(f"  {'Class':<14} {'Count':>6}  {'% of total':>10}")
print(f"  {'-'*34}")
for k, v in sorted(counts.items()):
    pct = v / total_instances * 100
    print(f"  {names[k]:<14} {v:>6}  {pct:>9.1f}%")