"""
Fix dataset label class IDs based on metadata.csv
Synchronizes YOLO annotation files in train/labels and val/labels with true targets.
"""
import csv
import os

DATASET_ROOT = r"d:\D,Drive\Honors 7th Sem\dataset01"
METADATA_CSV = os.path.join(DATASET_ROOT, "metadata.csv")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "weapon_detection", "train", "labels")
VAL_LABELS = os.path.join(DATASET_ROOT, "weapon_detection", "val", "labels")

def fix_labels():
    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} entries from {METADATA_CSV}")

    fixed_count = 0
    total_boxes = 0
    missing_files = 0
    class_box_counts = {}

    for row in rows:
        target = int(row["target"])
        labelfile = row["labelfile"]
        
        train_path = os.path.join(TRAIN_LABELS, labelfile)
        val_path = os.path.join(VAL_LABELS, labelfile)
        
        if os.path.exists(train_path):
            label_path = train_path
        elif os.path.exists(val_path):
            label_path = val_path
        else:
            missing_files += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            coords = parts[1:]
            new_cls = target
            if old_cls != new_cls:
                file_changed = True
            
            new_lines.append(f"{new_cls} " + " ".join(coords) + "\n")
            total_boxes += 1
            class_box_counts[new_cls] = class_box_counts.get(new_cls, 0) + 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        if file_changed:
            fixed_count += 1

    print(f"Fixed {fixed_count} label files ({missing_files} missing).")
    print(f"Total bounding boxes: {total_boxes}")
    print("Class distribution of boxes after fix:")
    names = {
        0: "Automatic Rifle", 1: "Bazooka", 2: "Grenade Launcher",
        3: "Handgun", 4: "Knife", 5: "Shotgun", 6: "SMG",
        7: "Sniper", 8: "Sword"
    }
    for c in sorted(class_box_counts.keys()):
        print(f"  Class {c} ({names.get(c, 'Unknown')}): {class_box_counts[c]} boxes")

if __name__ == "__main__":
    fix_labels()
