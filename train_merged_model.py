"""
train_merged_model.py
=====================
Retrains the weapon detection model from scratch on the merged 9-class dataset
(dataset01 + CCTV images + auto-classified archive images).

Prerequisites:
    python merge_datasets.py   ← must be run first

Run from smart-surveillance-system/ with venv activated:
    python train_merged_model.py

The best weights are saved as weapon_model.pt in the project root.
Estimated training time: 10–12 hours (CPU) / 2–4 hours (GPU).
"""
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_YAML  = PROJECT_ROOT / "merged_dataset.yaml"
MERGED_ROOT   = Path(r"d:\D,Drive\Honors 7th Sem\merged_dataset")
OUTPUT_DIR    = PROJECT_ROOT / "runs" / "weapon_train_v3"
BEST_WEIGHTS  = OUTPUT_DIR / "weights" / "best.pt"
DEST_WEIGHTS  = PROJECT_ROOT / "weapon_model.pt"

CLASS_NAMES = {
    0: "Automatic Rifle", 1: "Bazooka",        2: "Grenade Launcher",
    3: "Handgun",         4: "Knife",           5: "Shotgun",
    6: "SMG",             7: "Sniper",          8: "Sword",
    9: "Lethal Weapon"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if not DATASET_YAML.exists():
    raise FileNotFoundError(f"merged_dataset.yaml not found: {DATASET_YAML}")

if not MERGED_ROOT.exists():
    raise SystemExit(
        "[ERROR] merged_dataset/ not found.\n"
        "        Run  python merge_datasets.py  first."
    )

# ── Label audit (abort on class collapse) ─────────────────────────────────────

def audit_labels(split: str) -> dict:
    counts: dict = {}
    total = 0
    lbl_dir = MERGED_ROOT / split / "labels"
    if not lbl_dir.exists():
        print(f"[WARN] {split}/labels not found")
        return counts
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
            total += 1
    n_imgs = len(list((MERGED_ROOT / split / "images").iterdir()))
    print(f"\n[Audit] {split.upper()}: {n_imgs} images, {total} boxes")
    if total == 0:
        return counts
    for c in sorted(counts):
        pct = counts[c] / total * 100
        bar = "=" * int(pct / 2)
        print(f"  class {c} ({CLASS_NAMES.get(c,'?'):>18}): {counts[c]:>5} boxes  {pct:5.1f}% [{bar}]")
    dominant = max(counts.values()) / total if counts else 0
    if dominant > 0.90:
        dominant_cls = max(counts, key=counts.get)
        raise SystemExit(
            f"[ERROR] Class collapse: class {dominant_cls} is {dominant*100:.1f}% of boxes.\n"
            f"        Re-run merge_datasets.py to investigate."
        )
    return counts

print("=" * 65)
print("  Weapon Detection — Merged Dataset Training (v3)")
print("=" * 65)
print(f"  Dataset yaml : {DATASET_YAML}")
print(f"  Merged root  : {MERGED_ROOT}")
print(f"  Output dir   : {OUTPUT_DIR}")
print(f"  Dest weights : {DEST_WEIGHTS}")
print("=" * 65)

print("\n[Step 1/3] Auditing merged dataset labels ...")
train_counts = audit_labels("train")
val_counts   = audit_labels("val")
print("\n[Audit] Labels OK - proceeding with training.")

# ── Training ──────────────────────────────────────────────────────────────────

print("\n[Step 2/3] Starting training from yolov8n.pt ...")
print("  Strategy: Full retrain (not fine-tune) so ALL 9 classes are learned together.")
print("  Augmentation: enhanced HSV/brightness variation to simulate CCTV conditions.\n")

# Always start fresh from COCO backbone — do NOT resume from weapon_model.pt.
# Fine-tuning on CCTV-heavy data would cause catastrophic forgetting of the
# rarer classes (Bazooka, SMG, Sniper, Sword) that only exist in dataset01.
model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=120,             # larger merged dataset warrants more epochs
    imgsz=640,
    batch=16,
    name="weapon_train_v3",
    project=str(OUTPUT_DIR.parent),
    patience=20,
    exist_ok=True,
    verbose=True,

    # ── Loss weights ──────────────────────────────────────────────────────────
    # cls=2.5 keeps multi-class discrimination strong even though CCTV classes
    # (Handgun, Knife) will dominate the box count.
    cls=2.5,
    box=7.5,
    dfl=1.5,

    # ── CCTV-tuned augmentation ───────────────────────────────────────────────
    # Wider brightness/saturation range simulates low-quality CCTV footage
    # (underexposed, washed-out, grainy cameras).
    hsv_h=0.015,
    hsv_s=0.8,              # up from 0.7 — CCTV cameras often desaturate
    hsv_v=0.6,              # up from 0.4 — wide brightness variation
    # Random blur is simulated indirectly via scale + mosaic
    scale=0.6,              # up from 0.5 — objects appear at varied CCTV distances
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,             # slightly higher mixup for cross-quality generalisation
    erasing=0.4,            # random erasing simulates CCTV occlusion

    # ── LR schedule ───────────────────────────────────────────────────────────
    lr0=0.01,
    lrf=0.005,
    warmup_epochs=5,

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed=42,
    single_cls=False,       # MUST be False for 9-class detection
)

# ── Copy best weights ─────────────────────────────────────────────────────────

print("\n[Step 3/3] Copying best weights ...")
if BEST_WEIGHTS.exists():
    # Backup previous model before overwriting
    backup = PROJECT_ROOT / "weapon_model_v2.pt"
    if (PROJECT_ROOT / "weapon_model.pt").exists():
        shutil.copy2(PROJECT_ROOT / "weapon_model.pt", backup)
        print(f"   Previous model backed up → {backup.name}")

    shutil.copy2(BEST_WEIGHTS, DEST_WEIGHTS)
    print(f"\nTraining complete!")
    print(f"    Best weights → {DEST_WEIGHTS}")
    try:
        rd = results.results_dict
        print(f"    mAP50     = {rd.get('metrics/mAP50(B)', '?')}")
        print(f"    Precision = {rd.get('metrics/precision(B)', '?')}")
        print(f"    Recall    = {rd.get('metrics/recall(B)', '?')}")
    except Exception:
        pass
    print("\n    Restart app.py — it will automatically load weapon_model.pt")
else:
    print(f"\n[WARN] best.pt not found at {BEST_WEIGHTS}")
    print("       Check runs/weapon_train_v3/weights/ manually.")
