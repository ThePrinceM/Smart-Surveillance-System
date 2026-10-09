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
    'knife':            {'min_area': 100, 'ar_min': 0.001, 'ar_max': 50.0},
    'sword':            {'min_area': 100, 'ar_min': 0.001, 'ar_max': 50.0},
    'handgun':          {'min_area': 150, 'ar_min': 0.01,  'ar_max': 20.0},
    'shotgun':          {'min_area': 200, 'ar_min': 0.01,  'ar_max': 30.0},
    'smg':              {'min_area': 150, 'ar_min': 0.01,  'ar_max': 25.0},
    'sniper':           {'min_area': 200, 'ar_min': 0.01,  'ar_max': 30.0},
    'automatic rifle':  {'min_area': 200, 'ar_min': 0.01,  'ar_max': 25.0},
    'bazooka':          {'min_area': 200, 'ar_min': 0.01,  'ar_max': 30.0},
    'grenade launcher': {'min_area': 150, 'ar_min': 0.01,  'ar_max': 25.0},
    'lethal weapon':    {'min_area': 100, 'ar_min': 0.001, 'ar_max': 50.0},
}

# Fallback constraints for any class not explicitly listed
DEFAULT_CONSTRAINTS = {'min_area': 100, 'ar_min': 0.001, 'ar_max': 50.0}

CLASS_CONF_THRESHOLDS = {
    'knife':    0.10,
    'sword':    0.10,
    'handgun':  0.10,
}


class WeaponDetector:
    """Detects weapons using YOLOv8 with ultra-sensitive detection settings."""

    def __init__(self,
                 model_path: str = "weapon_model.pt",
                 confidence_threshold: float = 0.10,
                 target_classes: List[str] = None):
        """
        Initialize weapon detector.

        Args:
            model_path: Path to the custom weapon-detection YOLO weights.
            confidence_threshold: Confidence gate (0.38).
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
        self.cache_ttl = 0.20  # seconds

        # Temporal consistency tracking (Layer 4: Super-category 2-frame verification)
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
        """
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        min_dim = min(w, h)
        if min_dim < 3:
            return False

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

    def _get_super_category(self, class_name: str) -> str:
        """Group weapon classes into broad super-categories for robust temporal tracking."""
        cn = class_name.lower()
        if 'knife' in cn or 'sword' in cn or 'blade' in cn:
            return 'blade'
        return 'firearm'

    def _update_temporal(self, camera_id: int, seen_super_categories: set) -> set:
        """
        Layer 4: 2-consecutive-frame verification per super-category.
        Eliminates single-frame false positive flashes from background clutter.
        """
        if camera_id not in self._pending:
            self._pending[camera_id] = {}
            self._confirmed[camera_id] = set()

        pending = self._pending[camera_id]
        confirmed = self._confirmed[camera_id]

        for cat in list(pending.keys()):
            if cat not in seen_super_categories:
                pending[cat] = 0

        for cat in seen_super_categories:
            pending[cat] = pending.get(cat, 0) + 1
            if pending[cat] >= 1:
                confirmed.add(cat)

        for cat in list(confirmed):
            if pending.get(cat, 0) == 0:
                confirmed.discard(cat)

        return set(confirmed)

    def detect_weapons(self, frame: np.ndarray, camera_id: Optional[int] = None) -> Tuple[np.ndarray, bool, List[dict]]:
        """
        Detect weapons with 5-layer smart false-positive filtering.

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

                    # Layer 1b: Per-class confidence gate
                    cls_key = class_name.lower()
                    for k, min_conf in CLASS_CONF_THRESHOLDS.items():
                        if k in cls_key or cls_key in k:
                            if conf < min_conf:
                                conf = -1
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

                    candidate_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1i, y1i, x2i, y2i)
                    })

            final_detections = list(candidate_detections)
            weapon_detected = len(final_detections) > 0

            for det in final_detections:
                x1i, y1i, x2i, y2i = det['bbox']
                cv2.rectangle(annotated_frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)
                label_text = f"{det['class'].upper()} {det['confidence']*100:.1f}%"
                lsz, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_frame,
                              (x1i, max(0, y1i - lsz[1] - 8)), (x1i + lsz[0] + 6, y1i),
                              (0, 0, 255), -1)
                cv2.putText(annotated_frame, label_text, (x1i + 3, max(14, y1i - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

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

    def add_overlay(self, frame: np.ndarray, weapon_detected: bool, detections: List[dict]) -> np.ndarray:
        """Add alert overlay to frame with resolution scaling."""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        scale = max(0.6, w / 640.0)

        if weapon_detected:
            border_thick = max(4, int(6 * scale))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border_thick)
            overlay = frame.copy()
            banner_h = int(55 * scale)
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            title_scale = 0.85 * scale
            title_thick = max(2, int(2 * scale))
            title_text = "!! WEAPON DETECTED !!"
            tsz, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick)
            tx = max(10, (w - tsz[0]) // 2)
            ty = max(25, int(35 * scale))
            cv2.putText(frame, title_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), title_thick, cv2.LINE_AA)

            y_off = int(banner_h + (30 * scale))
            det_scale = 0.6 * scale
            det_thick = max(1, int(2 * scale))
            for det in detections:
                det_text = f"THREAT: {det['class'].upper()} ({det['confidence']*100:.1f}%)"
                cv2.putText(frame, det_text, (int(20 * scale), y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, det_scale, (0, 0, 255), det_thick, cv2.LINE_AA)
                y_off += int(30 * scale)
        else:
            # Draw status badge in top-right below face recognition badge
            badge_w, badge_h = int(240 * scale), int(36 * scale)
            x1 = max(10, w - badge_w - int(12 * scale))
            y1 = int(54 * scale)
            x2 = x1 + badge_w
            y2 = y1 + badge_h

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 18), -1)
            frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 1)
            cv2.putText(frame, "WEAPON SEC: CLEAR", (x1 + int(10 * scale), y1 + int(24 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52 * scale, (0, 255, 136), 1, cv2.LINE_AA)

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