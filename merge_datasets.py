"""
merge_datasets.py
=================
Merges three weapon datasets into a single unified 9-class YOLO dataset
saved to:  d:/D,Drive/Honors 7th Sem/merged_dataset/

Datasets merged:
  1. dataset01/weapon_detection  -- 9 weapon classes (0-8), studio images
                                    → included as-is
  2. cctv_dataset/combined_gunsnknifes -- 2 classes (pistol=0, knife=1)
                                    → undersampled to ~1500 images
                                    → remapped: pistol→3 (Handgun), knife→4 (Knife)
  3. archive/Dataset             -- 2 original classes (person=0, weapon=1)
                                    → person boxes DROPPED
                                    → weapon boxes AUTO-CLASSIFIED via weapon_model.pt
                                      (Automatic Rifle=0 or Handgun=3) at conf>=0.50

Run from smart-surveillance-system/ with venv activated:
    python merge_datasets.py
"""
import os, shutil, random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE            = Path(r"d:\D,Drive\Honors 7th Sem")
DATASET01       = BASE / "dataset01" / "weapon_detection"
CCTV_ROOT       = BASE / "cctv_dataset" / "combined_gunsnknifes"
ARCHIVE_ROOT    = BASE / "archive" / "Dataset"
MERGED_ROOT     = BASE / "merged_dataset"
MODEL_PATH      = Path(__file__).resolve().parent / "weapon_model.pt"

# ── Merge settings ─────────────────────────────────────────────────────────────
CCTV_MAX_TOTAL  = 1500          # max images to sample from cctv_dataset
CCTV_VAL_RATIO  = 0.12          # fraction of sampled CCTV images used for val
ARCHIVE_CONF    = 0.50          # min confidence for auto-classification
RANDOM_SEED     = 42

# ── 9-class schema ────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Automatic Rifle", 1: "Bazooka",        2: "Grenade Launcher",
    3: "Handgun",         4: "Knife",           5: "Shotgun",
    6: "SMG",             7: "Sniper",          8: "Sword",
    9: "Lethal Weapon"
}

random.seed(RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dirs(root: Path):
    for split in ["train/images", "train/labels", "val/images", "val/labels"]:
        (root / split).mkdir(parents=True, exist_ok=True)


def copy_image_label(src_img: Path, src_lbl: Path,
                     dst_img_dir: Path, dst_lbl_dir: Path,
                     label_lines: list[str], prefix: str = ""):
    """Copy image and write (possibly remapped) label lines."""
    stem = prefix + src_img.stem
    # avoid name collisions across datasets
    dst_img = dst_img_dir / (stem + src_img.suffix)
    dst_lbl = dst_lbl_dir / (stem + ".txt")
    # handle stem collision
    counter = 1
    while dst_img.exists():
        stem_c = f"{stem}_{counter}"
        dst_img = dst_img_dir / (stem_c + src_img.suffix)
        dst_lbl = dst_lbl_dir / (stem_c + ".txt")
        counter += 1
    shutil.copy2(src_img, dst_img)
    dst_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return dst_img, dst_lbl


def read_label(lbl_path: Path) -> list[str]:
    if not lbl_path.exists():
        return []
    return [l for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def remap_lines(lines: list[str], remap: dict, drop_classes: set = None) -> list[str]:
    """Remap class IDs. Lines with class in drop_classes are removed."""
    out = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if drop_classes and cls in drop_classes:
            continue
        new_cls = remap.get(cls, cls)
        out.append(f"{new_cls} " + " ".join(parts[1:]))
    return out


def audit(root: Path, split: str) -> dict:
    counts = {}
    total_boxes = 0
    lbl_dir = root / split / "labels"
    if not lbl_dir.exists():
        return counts
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
            total_boxes += 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — dataset01  (copied as-is, labels unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_dataset01(merged: Path):
    print("\n[1/3] Merging dataset01 (9 classes, unchanged) ...")
    count = 0
    for split, dst_split in [("train", "train"), ("val", "val")]:
        img_src = DATASET01 / split / "images"
        lbl_src = DATASET01 / split / "labels"
        dst_img = merged / dst_split / "images"
        dst_lbl = merged / dst_split / "labels"
        for img_path in sorted(img_src.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_src / (img_path.stem + ".txt")
            lines = read_label(lbl_path)
            copy_image_label(img_path, lbl_path, dst_img, dst_lbl, lines, prefix="ds01_")
            count += 1
    print(f"   Copied {count} images from dataset01.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — cctv_dataset  (undersample to ~1500, remap pistol→3, knife→4)
# ═══════════════════════════════════════════════════════════════════════════════

CCTV_REMAP = {0: 3, 1: 4}   # pistol→Handgun, knife→Knife

def merge_cctv(merged: Path):
    print(f"\n[2/3] Merging cctv_dataset (undersampled to ~{CCTV_MAX_TOTAL} images) ...")

    # Collect all images from train + val + test
    all_items = []   # list of (img_path, lbl_path)
    for split in ["train", "val", "test"]:
        img_dir = CCTV_ROOT / split / "images"
        lbl_dir = CCTV_ROOT / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            all_items.append((img, lbl))

    # Separate by original class to preserve ratio while undersampling
    pistol_items = []
    knife_items  = []
    for img, lbl in all_items:
        lines = read_label(lbl)
        has_pistol = any(int(l.split()[0]) == 0 for l in lines)
        has_knife  = any(int(l.split()[0]) == 1 for l in lines)
        if has_pistol:
            pistol_items.append((img, lbl))
        elif has_knife:
            knife_items.append((img, lbl))

    # Compute how many of each to keep (preserve ~64:36 pistol:knife ratio)
    total = len(pistol_items) + len(knife_items)
    pistol_ratio = len(pistol_items) / total if total else 0.64
    n_pistol = int(CCTV_MAX_TOTAL * pistol_ratio)
    n_knife  = CCTV_MAX_TOTAL - n_pistol

    random.shuffle(pistol_items)
    random.shuffle(knife_items)
    selected = pistol_items[:n_pistol] + knife_items[:n_knife]
    random.shuffle(selected)

    # Split into train / val
    n_val   = int(len(selected) * CCTV_VAL_RATIO)
    val_set = selected[:n_val]
    train_set = selected[n_val:]

    for items, dst_split in [(train_set, "train"), (val_set, "val")]:
        dst_img = merged / dst_split / "images"
        dst_lbl = merged / dst_split / "labels"
        for img_path, lbl_path in items:
            lines = read_label(lbl_path)
            remapped = remap_lines(lines, CCTV_REMAP)
            if not remapped:
                continue
            copy_image_label(img_path, lbl_path, dst_img, dst_lbl,
                             remapped, prefix="cctv_")

    print(f"   Sampled {len(train_set)} train + {len(val_set)} val from cctv_dataset.")
    print(f"   (pistol->Handgun: {n_pistol}, knife->Knife: {n_knife})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — archive  (map weapon→9 "Lethal Weapon", drop person→0)
# ═══════════════════════════════════════════════════════════════════════════════

ARCHIVE_REMAP = {1: 9}  # weapon -> Lethal Weapon
ARCHIVE_DROP = {0}      # drop person

def merge_archive(merged: Path):
    print("\n[3/3] Merging archive dataset (mapping weapon->9 'Lethal Weapon', dropping person) ...")

    img_dir = ARCHIVE_ROOT / "images"
    lbl_dir = ARCHIVE_ROOT / "labels"

    imgs = sorted([f for f in img_dir.iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    dst_img = merged / "train" / "images"
    dst_lbl = merged / "train" / "labels"

    used = 0
    skipped = 0

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lines = read_label(lbl_path)
        
        remapped = remap_lines(lines, ARCHIVE_REMAP, drop_classes=ARCHIVE_DROP)
        
        if not remapped:
            skipped += 1
            continue

        copy_image_label(img_path, lbl_path, dst_img, dst_lbl, remapped, prefix="arc_")
        used += 1

    print(f"   Used {used} archive images | Skipped {skipped} (no weapon labels)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Weapon Dataset Merge — 9-class unified dataset")
    print("=" * 65)

    if MERGED_ROOT.exists():
        print(f"\n[!] merged_dataset/ already exists at {MERGED_ROOT}")
        shutil.rmtree(MERGED_ROOT)
        print("    Deleted old merged_dataset/.")

    make_dirs(MERGED_ROOT)

    merge_dataset01(MERGED_ROOT)
    merge_cctv(MERGED_ROOT)
    merge_archive(MERGED_ROOT)

    # ── Final audit ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL DATASET AUDIT")
    print("=" * 65)
    total_img = 0
    for split in ["train", "val"]:
        counts = audit(MERGED_ROOT, split)
        n_imgs = len(list((MERGED_ROOT / split / "images").iterdir()))
        total_img += n_imgs
        total_boxes = sum(counts.values())
        print(f"\n  {split.upper()}: {n_imgs} images, {total_boxes} boxes")
        for c in sorted(counts):
            pct = counts[c] / total_boxes * 100 if total_boxes else 0
            bar = "=" * int(pct / 2)
            print(f"    class {c} ({CLASS_NAMES.get(c,'?'):>18}): {counts[c]:>5} boxes  {pct:5.1f}% [{bar}]")

    print(f"\n  Total images: {total_img}")
    print(f"\n  merged_dataset saved to: {MERGED_ROOT}")
    print("  Next step: run  python train_merged_model.py")


if __name__ == "__main__":
    main()
