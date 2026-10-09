"""
Crowd Detection Module
Uses YOLOv8 for accurate, real-time person detection and crowd density monitoring
"""
import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional
from pathlib import Path
from threading import Lock
import time


class CrowdDetector:
    """Detects and counts people using YOLOv8 object detection"""

    def __init__(self,
                 model_path: str = "yolov8s.pt",
                 confidence_threshold: float = 0.15,
                 iou_threshold: float = 0.60,
                 smoothing_window: int = 3,
                 **kwargs):
        """
        Initialize crowd detector using YOLOv8.

        Args:
            model_path: Path to the YOLO weights (defaults to yolov8n.pt).
            confidence_threshold: Minimum confidence for detecting a person.
            iou_threshold: NMS IoU threshold for overlapping persons in dense crowds.
            smoothing_window: Number of recent frames to average/smooth the count.
            **kwargs: Backward compatibility for legacy OpenCV Haar cascade args
                      (scale_factor, min_neighbors, min_size, enable_rotations).
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.smoothing_window = max(1, smoothing_window)

        # Backward compatibility notice
        legacy_keys = {'scale_factor', 'min_neighbors', 'min_size', 'enable_rotations'}
        used_legacy = legacy_keys.intersection(kwargs.keys())
        if used_legacy:
            print(f"[CrowdDetector] Note: Legacy cascade parameters {used_legacy} ignored. Using YOLOv8 engine.")

        self.model = None
        self.model_loaded = False
        self.inference_lock = Lock()

        # Multi-camera tracking state
        self.recent_counts = {}
        self.current_counts = {}
        self.cache = {}
        self.cache_ttl = 0.15  # seconds

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resolve weights path
        self.model_path = self._resolve_model_path(model_path)
        self._load_model()

    def _resolve_model_path(self, model_path: str) -> Path:
        """Resolve weights path relative to project if needed."""
        path = Path(model_path)
        if path.is_file():
            return path
        base_dir = Path(__file__).resolve().parent.parent
        candidates = [
            base_dir / model_path,
            base_dir / "models" / model_path
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # If not found locally, let YOLO download/resolve it
        return path

    def _load_model(self):
        """Load YOLOv8 model for person detection."""
        try:
            # Fix PyTorch 2.6+ weights_only issue if needed
            try:
                from ultralytics.nn import tasks as u_tasks
                torch.serialization.add_safe_globals([
                    torch.nn.modules.container.Sequential,
                    u_tasks.DetectionModel
                ])
            except Exception:
                pass

            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            print(f"[CrowdDetector] Successfully loaded YOLO crowd detector from {self.model_path}")
        except Exception as e:
            print(f"[CrowdDetector] Error loading YOLO model: {e}")
            self.model_loaded = False

    def detect_people(self, frame: np.ndarray, camera_id: int = 0) -> Tuple[np.ndarray, int, List[Tuple[int, int, int, int]]]:
        """
        Detect and count people in a frame.

        Args:
            frame: Input image/frame as BGR numpy array
            camera_id: Camera identifier for multi-camera tracking

        Returns:
            Tuple of (annotated_frame, smoothed_people_count, list_of_person_bounding_boxes)
            where each box is (x, y, w, h).
        """
        if frame is None or frame.size == 0:
            return frame, 0, []

        now = time.time()
        # Check cache for recent detection to avoid redundant inference on rapid frame loops
        cached = self.cache.get(camera_id)
        if cached and (now - cached['time']) <= self.cache_ttl:
            annotated_frame = self._render_detections(frame.copy(), cached['boxes'], cached['confs'])
            return annotated_frame, self.current_counts.get(camera_id, cached['count']), cached['boxes']

        verified_boxes = []
        confs = []

        if self.model_loaded and self.model is not None:
            with self.inference_lock:
                try:
                    # Run inference for person class only (class 0 in COCO)
                    results = self.model(
                        frame,
                        classes=[0],
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        device=self.device,
                        verbose=False
                    )

                    if results and len(results) > 0:
                        boxes_data = results[0].boxes
                        if boxes_data is not None and len(boxes_data) > 0:
                            for box in boxes_data:
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                x1, y1, x2, y2 = map(int, xyxy)
                                w = max(1, x2 - x1)
                                h = max(1, y2 - y1)
                                verified_boxes.append((x1, y1, w, h))
                                confs.append(conf)
                except Exception as e:
                    print(f"[CrowdDetector] Inference error on camera {camera_id}: {e}")

        # Update tracking and smoothing
        raw_count = len(verified_boxes)
        if camera_id not in self.recent_counts:
            self.recent_counts[camera_id] = []
            self.current_counts[camera_id] = 0

        self.recent_counts[camera_id].append(raw_count)
        if len(self.recent_counts[camera_id]) > self.smoothing_window:
            self.recent_counts[camera_id].pop(0)

        # Ensure count matches the verified detection boxes exactly
        current_count = raw_count
        self.current_counts[camera_id] = current_count

        # Update cache
        self.cache[camera_id] = {
            'time': now,
            'boxes': verified_boxes,
            'confs': confs,
            'count': current_count
        }

        annotated_frame = self._render_detections(frame.copy(), verified_boxes, confs)
        return annotated_frame, current_count, verified_boxes

    # Backward compatibility alias
    def detect_faces(self, frame: np.ndarray, camera_id: int = 0) -> Tuple[np.ndarray, int, List[Tuple[int, int, int, int]]]:
        """Alias for detect_people to maintain 100% backward compatibility with existing calls."""
        return self.detect_people(frame, camera_id)

    def _render_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], confs: List[float]) -> np.ndarray:
        """Draw modern bounding boxes and confidence tags for detected persons."""
        for i, (x, y, w, h) in enumerate(boxes):
            conf = confs[i] if i < len(confs) else 0.0

            # Box color: vibrant cyan/teal
            color = (255, 200, 0)  # BGR: Sky blue / Cyan
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw corner accents for high-tech surveillance look
            corner_len = min(15, max(4, w // 4), max(4, h // 4))
            accent_color = (0, 255, 255)  # Yellow-cyan accent
            if corner_len > 3:
                # Top-left
                cv2.line(frame, (x, y), (x + corner_len, y), accent_color, 3)
                cv2.line(frame, (x, y), (x, y + corner_len), accent_color, 3)
                # Top-right
                cv2.line(frame, (x + w, y), (x + w - corner_len, y), accent_color, 3)
                cv2.line(frame, (x + w, y), (x + w, y + corner_len), accent_color, 3)
                # Bottom-left
                cv2.line(frame, (x, y + h), (x + corner_len, y + h), accent_color, 3)
                cv2.line(frame, (x, y + h), (x, y + h - corner_len), accent_color, 3)
                # Bottom-right
                cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), accent_color, 3)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), accent_color, 3)

            # Person tag label
            label = f"Person {int(conf * 100)}%" if conf > 0 else "Person"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lw, lh = label_size
            label_y1 = max(0, y - lh - 6)
            label_y2 = y
            cv2.rectangle(frame, (x, label_y1), (x + lw + 6, label_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x, label_y1), (x + lw + 6, label_y2), color, 1)
            cv2.putText(frame, label, (x + 3, label_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def add_overlay(self, frame: np.ndarray, count: int, threshold: int = 8) -> np.ndarray:
        """
        Add sleek, modern status overlay to frame.

        Args:
            frame: Input frame
            count: Number of people detected
            threshold: Crowd density threshold

        Returns:
            Frame with overlay
        """
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        is_high_density = count >= threshold
        is_moderate = count >= max(1, threshold // 2)

        # Draw semi-transparent HUD card in top-left (12, 12) -> (250, 85)
        card_w, card_h = 240, 72
        cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (15, 15, 18), -1)
        # Smooth alpha blend for backdrop
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw outline border for HUD
        border_color = (0, 0, 230) if is_high_density else ((0, 180, 240) if is_moderate else (60, 60, 60))
        cv2.rectangle(frame, (12, 12), (12 + card_w, 12 + card_h), border_color, 1)

        # Text: People Count
        cv2.putText(frame, f"People Count: {count}", (22, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Density Status Badge
        if is_high_density:
            # Red alert
            cv2.putText(frame, "! HIGH DENSITY ALERT !", (22, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 255), 2, cv2.LINE_AA)
            # Prominent outer alert border around entire screen
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        elif is_moderate:
            # Moderate density (amber)
            cv2.putText(frame, f"Moderate Density (Max: {threshold})", (22, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            # Normal density (green)
            cv2.putText(frame, f"Normal Density (Max: {threshold})", (22, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 230, 90), 1, cv2.LINE_AA)

        return frame

    def get_count(self, camera_id: int = 0) -> int:
        """Get current people count for a camera."""
        return self.current_counts.get(camera_id, 0)

    def reset(self, camera_id: int = 0):
        """Reset tracking and cache for a camera."""
        if camera_id in self.recent_counts:
            self.recent_counts[camera_id] = []
        if camera_id in self.current_counts:
            self.current_counts[camera_id] = 0
        if camera_id in self.cache:
            del self.cache[camera_id]
