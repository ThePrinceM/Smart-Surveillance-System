"""
Weapon Detection Module
Uses YOLOv8 for detecting weapons (guns, knives) with smart multi-layer
false-positive filtering that preserves recall on blurry/low-quality CCTV footage.

5-Layer False Positive Filtering:
  Layer 1: Confidence gate (LOW 0.30 — preserves recall on blurry/partial weapons)
  Layer 2: Minimum bounding-box area per class (phone edges = tiny sliver boxes)
  Layer 3: Per-class aspect ratio validation (weapon geometry fingerprinting)
  Layer 4: Temporal 2-frame consistency (real weapons stay in frame; flickering FP don't)
  Layer 5: Target weapon class filter (only pass known weapon class names)
"""
import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path
from threading import Lock
import time


# Set to True to print every raw detection for debugging.
DEBUG_DETECTIONS = False

# Expected weapon class names (must match dataset.yaml exactly)
EXPECTED_WEAPON_NAMES = {
    0: 'Automatic Rifle', 1: 'Bazooka',        2: 'Grenade Launcher',
    3: 'Handgun',         4: 'Knife',           5: 'Shotgun',
    6: 'SMG',             7: 'Sniper',          8: 'Sword',
    9: 'Lethal Weapon'
}

# ---------------------------------------------------------------------------
# Per-class shape constraints for smart false-positive filtering.
# Encodes the real-world geometry of each weapon category.
# min_area : minimum bounding-box pixel area — eliminates tiny slivers
# ar_min   : minimum width/height aspect ratio
# ar_max   : maximum width/height aspect ratio
# ---------------------------------------------------------------------------
CLASS_CONSTRAINTS = {
    # Bladed weapons — knives come in MANY shapes:
    # kitchen knife (wide), dagger (symmetric), cleaver (square), machete (long),
    # box cutter (thin), hunting knife (medium). So we keep a very wide AR range
    # and filter instead by the per-class confidence threshold.
    'knife':            {'min_area': 1200, 'ar_min': 0.04, 'ar_max': 10.0},
    'sword':            {'min_area': 800,  'ar_min': 0.04, 'ar_max': 12.0},
    # Firearms — larger, more rectangular
    'handgun':          {'min_area': 2000, 'ar_min': 0.25, 'ar_max': 4.5},
    'shotgun':          {'min_area': 2500, 'ar_min': 0.05, 'ar_max': 10.0},
    'smg':              {'min_area': 2000, 'ar_min': 0.10, 'ar_max': 7.0},
    'sniper':           {'min_area': 2500, 'ar_min': 0.05, 'ar_max': 12.0},
    'automatic rifle':  {'min_area': 2500, 'ar_min': 0.05, 'ar_max': 10.0},
    'bazooka':          {'min_area': 2500, 'ar_min': 0.05, 'ar_max': 12.0},
    'grenade launcher': {'min_area': 2000, 'ar_min': 0.08, 'ar_max': 8.0},
    'lethal weapon':    {'min_area': 800,  'ar_min': 0.04, 'ar_max': 12.0},
}

# Fallback constraints for any class not explicitly listed
DEFAULT_CONSTRAINTS = {'min_area': 800, 'ar_min': 0.04, 'ar_max': 12.0}

# ---------------------------------------------------------------------------
# Per-class MINIMUM CONFIDENCE overrides.
# The global threshold stays at 0.30 for most weapons (ensures blurry CCTV
# footage still catches real weapons). Knives are the #1 false-positive source
# (phone edges, scissors, rulers) so they get a slightly higher minimum.
# ---------------------------------------------------------------------------
CLASS_CONF_THRESHOLDS = {
    'knife':  0.42,   # higher — phone/remote edges frequently misclassified as knife
    'sword':  0.38,   # moderate — uncommon in real surveillance; raise a bit
    # All other classes use the global confidence_threshold (0.30)
}


class WeaponDetector:
    """Detects weapons using YOLOv8 with smart multi-layer false-positive filtering."""

    def __init__(self,
                 model_path: str = "weapon_model.pt",
                 confidence_threshold: float = 0.30,
                 target_classes: List[str] = None):
        """
        Initialize weapon detector.

        Args:
            model_path: Path to the custom weapon-detection YOLO weights.
            confidence_threshold: Keep LOW (0.30). Blurry real weapons may only reach
                                  0.30-0.45 confidence — false positives are handled
                                  by the shape and temporal filters instead.
            target_classes: List of class names to detect. Defaults to all 10 classes.
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = [cls.lower() for cls in (target_classes or [
            'automatic rifle', 'bazooka', 'grenade launcher',
            'handgun', 'knife', 'shotgun', 'smg', 'sniper', 'sword', 'lethal weapon'
        ])]
        self.keyword_map = {
            'automatic rifle': {'automatic rifle', 'rifle', 'assault rifle', 'ar'},
            'bazooka':         {'bazooka', 'rocket launcher', 'rpg'},
            'grenade launcher': {'grenade launcher', 'grenade'},
            'handgun':         {'handgun', 'pistol', 'gun', 'firearm', 'revolver'},
            'knife':           {'knife', 'dagger', 'blade'},
            'shotgun':         {'shotgun'},
            'smg':             {'smg', 'submachine gun', 'machine gun'},
            'sniper':          {'sniper', 'sniper rifle'},
            'sword':           {'sword', 'saber', 'katana'},
            'lethal weapon':   {'lethal weapon', 'weapon'}
        }

        self.model = None
        self.model_loaded = False
        self.model_path = self._resolve_model_path(model_path)
        self.inference_lock = Lock()

        # Result cache — keyed by camera_id
        self.cache: dict = {}
        self.cache_ttl = 0.20  # seconds (reduced from 0.4 for snappier response)

        # Temporal consistency tracking (Layer 4)
        # pending[camera_id][class_name] = consecutive_frame_count
        # confirmed[camera_id] = set of class names confirmed for >= 2 frames
        self._pending: dict = {}
        self._confirmed: dict = {}

        if self.model_path is None:
            print(f"[WeaponDetector] Could not locate weights file '{model_path}'")
            return

        try:
            try:
                from ultralytics.nn import tasks as u_tasks
                torch.serialization.add_safe_globals([
                    torch.nn.modules.container.Sequential,
                    u_tasks.DetectionModel
                ])
            except Exception as e:
                print(f"[WeaponDetector] Warning: Could not register safe globals: {e}")

            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            print(f"[WeaponDetector] Successfully loaded model: {self.model_path}")
            self._validate_model_classes()

        except Exception as e:
            print(f"[WeaponDetector] Failed to load YOLO model: {e}")
            self.model_loaded = False

    def _resolve_model_path(self, model_path: str) -> Optional[Path]:
        """Resolve weights path relative to project if needed."""
        path = Path(model_path)
        if path.is_file():
            return path
        base_dir = Path(__file__).resolve().parent.parent
        for candidate in [base_dir / model_path, base_dir / "models" / model_path]:
            if candidate.is_file():
                return candidate
        return None

    def _validate_model_classes(self):
        """Warn if loaded model class names don't match expected weapon classes."""
        if self.model is None:
            return
        model_names = getattr(self.model, 'names', {})
        mismatches = []
        for idx, expected in EXPECTED_WEAPON_NAMES.items():
            actual = model_names.get(idx, '<missing>')
            if actual.lower() != expected.lower():
                mismatches.append(f"  class {idx}: expected '{expected}', got '{actual}'")
        if mismatches:
            print("[WeaponDetector] WARNING: Model class names do not match expected weapon names!")
            print("[WeaponDetector] Re-run train_weapon_model.py to fix this.")
            for m in mismatches:
                print(m)
        else:
            print(f"[WeaponDetector] Model class validation OK: {len(model_names)} classes match.")

    def _is_target_class(self, label: str) -> bool:
        """Layer 5: Check if detected class is a known weapon type."""
        label = (label or "").lower()
        if label in self.target_classes:
            return True
        for key in self.target_classes:
            keywords = self.keyword_map.get(key, {key})
            if any(keyword in label for keyword in keywords):
                return True
        return False

    def _passes_shape_filter(self, class_name: str, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Layers 2 & 3: Validate bounding box geometry against per-class constraints.

        Common false positives (phone, remote, scissors) fail these checks because:
        - Phone screen = wide, large-area box (aspect ratio > 4.5 for handguns)
        - Phone edge sliver = very small area (<2000px²) or extreme AR
        - Book/laptop = way too large area for a knife
        """
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect_ratio = w / h

        key = class_name.lower()
        constraints = DEFAULT_CONSTRAINTS
        for cname, cval in CLASS_CONSTRAINTS.items():
            if cname in key or key in cname:
                constraints = cval
                break

        if area < constraints['min_area']:
            if DEBUG_DETECTIONS:
                print(f"[WeaponDetector] Shape FAIL '{class_name}': area={area} < {constraints['min_area']}")
            return False

        if not (constraints['ar_min'] <= aspect_ratio <= constraints['ar_max']):
            if DEBUG_DETECTIONS:
                print(f"[WeaponDetector] Shape FAIL '{class_name}': AR={aspect_ratio:.2f} "
                      f"not in [{constraints['ar_min']}, {constraints['ar_max']}]")
            return False

        return True

    def _update_temporal(self, camera_id: int, seen_classes: set) -> set:
        """
        Layer 4: 2-consecutive-frame rule.

        Real weapons remain visible across frames. Random false positives
        (phone glints, reflections) typically appear for only 1 frame.
        A class must be seen in ≥2 consecutive frames to trigger an alert.
        """
        if camera_id not in self._pending:
            self._pending[camera_id] = {}
            self._confirmed[camera_id] = set()

        pending = self._pending[camera_id]
        confirmed = self._confirmed[camera_id]

        # Decay classes not seen this frame
        for cls in list(pending.keys()):
            if cls not in seen_classes:
                pending[cls] = 0

        # Increment classes seen this frame
        for cls in seen_classes:
            pending[cls] = pending.get(cls, 0) + 1
            if pending[cls] >= 2:
                confirmed.add(cls)

        # Clear confirmed status for classes that disappeared
        for cls in list(confirmed):
            if pending.get(cls, 0) == 0:
                confirmed.discard(cls)

        return set(confirmed)

    def detect_weapons(self, frame: np.ndarray, camera_id: Optional[int] = None) -> Tuple[np.ndarray, bool, List[dict]]:
        """
        Detect weapons with 5-layer smart filtering.

        Returns:
            (annotated_frame, weapon_confirmed_bool, list_of_confirmed_detections)
        """
        if frame is None or not self.model_loaded:
            return frame if frame is not None else None, False, []

        cache_key = camera_id
        if cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached and (time.time() - cached['timestamp'] <= self.cache_ttl):
                return cached['frame'].copy(), cached['weapon_detected'], list(cached['detections'])

        try:
            with self.inference_lock:
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            annotated_frame = frame.copy()
            candidate_detections = []
            seen_classes_this_frame = set()

            result = results[0] if isinstance(results, list) else results
            if hasattr(result, 'boxes') and result.boxes is not None:
                data = result.boxes.data.cpu().numpy()
                names = getattr(result, 'names', self.model.names)
                for det in data:
                    x1, y1, x2, y2, conf, cls_id = det
                    cls_id = int(cls_id)
                    conf = float(conf)
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

                    if DEBUG_DETECTIONS:
                        print(f"[WeaponDetector] RAW: class={cls_id} ({class_name}), conf={conf:.3f}")

                    # Layer 1a: Global confidence gate
                    if conf < self.confidence_threshold:
                        continue

                    # Layer 1b: Per-class confidence gate (e.g. knife needs higher conf)
                    cls_key = class_name.lower()
                    for k, min_conf in CLASS_CONF_THRESHOLDS.items():
                        if k in cls_key or cls_key in k:
                            if conf < min_conf:
                                if DEBUG_DETECTIONS:
                                    print(f"[WeaponDetector] Per-class gate FAIL '{class_name}': "
                                          f"conf={conf:.3f} < class_min={min_conf}")
                                conf = -1  # mark for skip
                            break
                    if conf < 0:
                        continue

                    # Layer 5: Must be a target weapon class
                    if not self._is_target_class(class_name):
                        continue

                    # Layers 2 & 3: Shape geometry validation
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    if not self._passes_shape_filter(class_name, x1i, y1i, x2i, y2i):
                        continue

                    seen_classes_this_frame.add(class_name.lower())
                    candidate_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1i, y1i, x2i, y2i)
                    })

            # Layer 4: Temporal consistency
            tid = camera_id if camera_id is not None else -1
            confirmed_classes = self._update_temporal(tid, seen_classes_this_frame)

            final_detections = []
            weapon_detected = False

            for det in candidate_detections:
                cls_lower = det['class'].lower()
                x1i, y1i, x2i, y2i = det['bbox']

                if cls_lower in confirmed_classes:
                    # Fully confirmed — solid red alert box
                    weapon_detected = True
                    final_detections.append(det)
                    cv2.rectangle(annotated_frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)
                    label_text = f"{det['class'].upper()} {det['confidence']:.2f}"
                    lsz, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame,
                                  (x1i, y1i - lsz[1] - 10), (x1i + lsz[0], y1i),
                                  (0, 0, 255), -1)
                    cv2.putText(annotated_frame, label_text, (x1i, y1i - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # 1st-frame candidate — subtle amber box (visual feedback, no alert)
                    cv2.rectangle(annotated_frame, (x1i, y1i), (x2i, y2i), (0, 165, 255), 1)
                    cv2.putText(annotated_frame,
                                f"? {det['class']} {det['confidence']:.2f}",
                                (x1i, max(0, y1i - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            if cache_key is not None:
                self.cache[cache_key] = {
                    'frame': annotated_frame.copy(),
                    'weapon_detected': weapon_detected,
                    'detections': list(final_detections),
                    'timestamp': time.time()
                }

            return annotated_frame, weapon_detected, final_detections

        except Exception as e:
            print(f"[WeaponDetector] Error during detection: {e}")
            return frame, False, []

    def add_overlay(self, frame: np.ndarray, weapon_detected: bool, detections: List[dict]) -> np.ndarray:
        """Add alert overlay to frame."""
        if frame is None:
            return None

        h, w = frame.shape[:2]

        if weapon_detected:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.putText(frame, "!! WEAPON DETECTED !!", (w // 2 - 210, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            y_off = 100
            for det in detections:
                cv2.putText(frame, f"{det['class'].upper()}: {det['confidence']*100:.1f}%",
                            (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_off += 30
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (250, 60), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.putText(frame, "Status: CLEAR", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model_loaded

    def clear_camera_cache(self, camera_id: int):
        """Clear temporal state and result cache for a specific camera."""
        self.cache.pop(camera_id, None)
        if camera_id in self._pending:
            self._pending[camera_id] = {}
        if camera_id in self._confirmed:
            self._confirmed[camera_id] = set()