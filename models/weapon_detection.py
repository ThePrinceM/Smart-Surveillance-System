"""
Weapon Detection Module
Uses YOLOv8 for detecting weapons (guns, knives)
"""
import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path
from threading import Lock
import time


# Set to True to print every raw detection (class + conf) for debugging.
DEBUG_DETECTIONS = False

# Expected weapon class names (must match dataset.yaml exactly)
EXPECTED_WEAPON_NAMES = {
    0: 'Automatic Rifle', 1: 'Bazooka',        2: 'Grenade Launcher',
    3: 'Handgun',         4: 'Knife',           5: 'Shotgun',
    6: 'SMG',             7: 'Sniper',          8: 'Sword',
    9: 'Lethal Weapon'
}


class WeaponDetector:
    """Detects weapons using YOLOv8 object detection"""

    def __init__(self,
                 model_path: str = "weapon_model.pt",
                 confidence_threshold: float = 0.35,
                 target_classes: List[str] = None):
        """
        Initialize weapon detector.

        Args:
            model_path: Path to the custom weapon-detection YOLO weights.
                        Defaults to weapon_model.pt (custom-trained on dataset01).
            confidence_threshold: Minimum confidence for a detection to be reported.
            target_classes: List of class names to detect.
                            Defaults to all 9 weapon classes from dataset01.
        """
        self.confidence_threshold = confidence_threshold
        # All 9 classes present in the custom-trained weapon dataset
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
        self.cache: dict[int, dict] = {}
        self.cache_ttl = 0.4  # seconds
        if self.model_path is None:
            print(f"[WeaponDetector] Could not locate weights file '{model_path}'")
            return
        
        # Try to load YOLO model
        try:
            # Fix PyTorch 2.6+ weights_only issue
            try:
                from ultralytics.nn import tasks as u_tasks
                torch.serialization.add_safe_globals([
                    torch.nn.modules.container.Sequential,
                    u_tasks.DetectionModel
                ])
            except Exception as e:
                print(f"[WeaponDetector] Warning: Could not register safe globals: {e}")
            
            # Import and load YOLO
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            print(f"[WeaponDetector] Successfully loaded model: {self.model_path}")
            # Validate that the loaded model has the expected 9-class weapon names
            self._validate_model_classes()

        except Exception as e:
            print(f"[WeaponDetector] Failed to load YOLO model: {e}")
            print("[WeaponDetector] Weapon detection will be disabled")
            self.model_loaded = False

    def _resolve_model_path(self, model_path: str) -> Optional[Path]:
        """Resolve weights path relative to project if needed."""
        candidates = []
        path = Path(model_path)
        if path.is_file():
            candidates.append(path)
        base_dir = Path(__file__).resolve().parent.parent
        candidates.append(base_dir / model_path)
        candidates.append(base_dir / "models" / model_path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _validate_model_classes(self):
        """Warn if the loaded model's class names don't match the expected 9 weapon classes."""
        if self.model is None:
            return
        model_names = getattr(self.model, 'names', {})
        mismatches = []
        for idx, expected in EXPECTED_WEAPON_NAMES.items():
            actual = model_names.get(idx, '<missing>')
            if actual.lower() != expected.lower():
                mismatches.append(f"  class {idx}: expected '{expected}', got '{actual}'")
        if mismatches:
            print("[WeaponDetector] WARNING: Model class names do not match expected weapon classes!")
            print("[WeaponDetector] This usually means the model was not trained on the correct dataset.")
            print("[WeaponDetector] Re-run train_weapon_model.py to fix this.")
            for m in mismatches:
                print(m)
        else:
            nc = len(model_names)
            print(f"[WeaponDetector] Model class validation OK: {nc} classes match expected weapon names.")

    def _is_target_class(self, label: str) -> bool:
        label = (label or "").lower()
        if label in self.target_classes:
            return True
        for key in self.target_classes:
            keywords = self.keyword_map.get(key, {key})
            if any(keyword in label for keyword in keywords):
                return True
        return False
    
    def detect_weapons(self, frame: np.ndarray, camera_id: Optional[int] = None) -> Tuple[np.ndarray, bool, List[dict]]:
        """
        Detect weapons in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, weapon_detected, detections_list)
        """
        if frame is None or not self.model_loaded:
            return frame if frame is not None else None, False, []

        cache_key = camera_id if camera_id is not None else None
        if cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached and (time.time() - cached['timestamp'] <= self.cache_ttl):
                return cached['frame'].copy(), cached['weapon_detected'], list(cached['detections'])
        
        try:
            with self.inference_lock:
                # Run YOLO inference (align with working final_weapon pipeline)
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            annotated_frame = frame.copy()
            weapon_detected = False
            detections = []
            
            # Process first result set for current frame
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
                        print(f"[WeaponDetector] RAW det: class={cls_id} ({class_name}), conf={conf:.3f}")
                    # Note: conf threshold already applied by model(conf=...) call above;
                    # this guard handles any edge-case detections that slip through.
                    if conf < self.confidence_threshold:
                        continue
                    if not self._is_target_class(class_name):
                        continue
                    weapon_detected = True
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label_text = f"{class_name.upper()} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if cache_key is not None:
                self.cache[cache_key] = {
                    'frame': annotated_frame.copy(),
                    'weapon_detected': weapon_detected,
                    'detections': list(detections),
                    'timestamp': time.time()
                }

            return annotated_frame, weapon_detected, detections
            
        except Exception as e:
            print(f"[WeaponDetector] Error during detection: {e}")
            return frame, False, []
    
    def add_overlay(self, frame: np.ndarray, weapon_detected: bool, detections: List[dict]) -> np.ndarray:
        """
        Add alert overlay to frame
        
        Args:
            frame: Input frame
            weapon_detected: Whether a weapon was detected
            detections: List of detection dictionaries
            
        Returns:
            Frame with overlay
        """
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        
        if weapon_detected:
            # Add red flashing border
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)
            
            # Add alert banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Alert text
            alert_text = "ALERT: WEAPON DETECTED!"
            cv2.putText(frame, alert_text, (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # List detected weapons
            y_offset = 100
            for det in detections:
                text = f"{det['class'].upper()}: {det['confidence']*100:.1f}%"
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        else:
            # Normal status
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (250, 60), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            cv2.putText(frame, "Status: CLEAR", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        return self.model_loaded
