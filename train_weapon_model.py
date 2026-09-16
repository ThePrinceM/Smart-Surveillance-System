"""
Train YOLOv8n on the custom weapon detection dataset (dataset01).
Run this from the smart-surveillance-system directory with the venv activated:

    python train_weapon_model.py

The best weights will be saved as weapon_model.pt in the project root.
Training takes ~10-30 minutes depending on GPU availability.

NOTE: A label audit is performed before training. If the audit reveals
class imbalance (e.g. everything is class 0), fix_labels.py must be
run first and training restarted.
"""
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "[ERROR] ultralytics not installed. Run: pip install ultralytics"
    )

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_YAML  = PROJECT_ROOT / "dataset.yaml"
OUTPUT_DIR    = PROJECT_ROOT / "runs" / "weapon_train_v2"
BEST_WEIGHTS  = OUTPUT_DIR / "weights" / "best.pt"
DEST_WEIGHTS  = PROJECT_ROOT / "weapon_model.pt"
DATASET_ROOT  = Path(r"d:\D,Drive\Honors 7th Sem\dataset01")
TRAIN_LABELS  = DATASET_ROOT / "weapon_detection" / "train" / "labels"
VAL_LABELS    = DATASET_ROOT / "weapon_detection" / "val" / "labels"

CLASS_NAMES = {
    0: "Automatic Rifle", 1: "Bazooka",        2: "Grenade Launcher",
    3: "Handgun",         4: "Knife",           5: "Shotgun",
    6: "SMG",             7: "Sniper",          8: "Sword",
}

# ─── Label Audit ─────────────────────────────────────────────────────────────

def audit_labels(labels_dir: Path, split_name: str) -> dict:
    counts: dict = {}
    total = 0
    n_files = 0
    if not labels_dir.exists():
        print(f"[WARN] {split_name} labels dir not found: {labels_dir}")
        return counts
    for f in labels_dir.glob("*.txt"):
        n_files += 1
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
            total += 1
    print(f"\n[Audit] {split_name}: {n_files} files, {total} boxes")
    if total == 0:
        print(f"  [WARN] No boxes found in {labels_dir}")
        return counts
    for c in sorted(counts):
        pct = counts[c] / total * 100
        bar = "=" * int(pct / 2)
        print(f"  class {c} ({CLASS_NAMES.get(c, '?'):>18}): {counts[c]:>4} boxes  {pct:5.1f}% [{bar}]")
    # Detect class collapse: any single class > 80% of all boxes
    dominant_ratio = max(counts.values()) / total
    if dominant_ratio > 0.80:
        dominant_cls = max(counts, key=counts.get)
        print(f"\n  [ERROR] CLASS COLLAPSE DETECTED: class {dominant_cls} "
              f"({CLASS_NAMES.get(dominant_cls,'?')}) "
              f"is {dominant_ratio*100:.1f}% of all boxes!")
        print("  Run fix_labels.py to repair the annotations, then re-run this script.\n")
        raise SystemExit("[ERROR] Aborting training: label audit failed (class collapse).")
    return counts

# ─── Main ─────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  Weapon Detection Model Training  (v2 - Fixed Labels)")
print("=" * 65)
print(f"  Dataset config : {DATASET_YAML}")
print(f"  Output dir     : {OUTPUT_DIR}")
print(f"  Final weights  : {DEST_WEIGHTS}")
print("=" * 65)

if not DATASET_YAML.exists():
    raise FileNotFoundError(f"dataset.yaml not found at: {DATASET_YAML}")

# Step 1: Audit labels BEFORE training to catch class collapse early
print("\n[Step 1/3] Auditing dataset labels ...")
audit_labels(TRAIN_LABELS, "TRAIN")
audit_labels(VAL_LABELS, "VAL")
print("\n[Audit] Labels look healthy - proceeding with training.")

# Step 2: Train fresh from yolov8n.pt
# We deliberately do NOT resume from the old weapon_train run because that run
# was trained on all-class-0 labels (before fix_labels.py was applied).
print("\n[Step 2/3] Starting fresh training from yolov8n.pt ...")
model = YOLO("yolov8n.pt")   # start from COCO-pretrained nano backbone

results = model.train(
    data=str(DATASET_YAML),
    epochs=100,             # More epochs for 9-class weapon discrimination
    imgsz=640,
    batch=16,
    name="weapon_train_v2",
    project=str(OUTPUT_DIR.parent),
    patience=20,
    exist_ok=True,
    verbose=True,
    # ── Loss weights ──────────────────────────────────────────────────────
    # Increase cls weight (default=0.5) so the model is penalised heavily
    # for predicting the wrong weapon class. This is the primary fix for the
    # "everything classified as Automatic Rifle" class-collapse problem.
    cls=2.5,                # 5x default - critical for 9-class weapon ID
    box=7.5,                # keep default
    dfl=1.5,                # keep default
    # ── Augmentation ──────────────────────────────────────────────────────
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,              # slight mixup helps generalise across weapon shapes
    # ── LR schedule ───────────────────────────────────────────────────────
    lr0=0.01,
    lrf=0.005,
    warmup_epochs=5,
    # ── Misc ──────────────────────────────────────────────────────────────
    seed=42,
    single_cls=False,       # MUST be False for 9-class training
)

# Step 3: Copy best weights to project root
print("\n[Step 3/3] Copying best weights ...")
if BEST_WEIGHTS.exists():
    shutil.copy2(BEST_WEIGHTS, DEST_WEIGHTS)
    print(f"\nTraining complete!")
    print(f"    Best weights -> {DEST_WEIGHTS}")
    try:
        rd = results.results_dict
        print(f"    mAP50    = {rd.get('metrics/mAP50(B)', '?')}")
        print(f"    Precision = {rd.get('metrics/precision(B)', '?')}")
        print(f"    Recall    = {rd.get('metrics/recall(B)', '?')}")
    except Exception:
        pass
    print("\n    Restart app.py - it will automatically load weapon_model.pt")
else:
    print(f"\n[WARN] best.pt not found at {BEST_WEIGHTS}")
    print("    Check the runs/ directory for last.pt and copy it manually.")

