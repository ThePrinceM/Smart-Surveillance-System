"""
Crowd Detection Module
Uses OpenCV Haar Cascade for face detection and crowd counting
"""
import cv2
import numpy as np
from typing import Tuple, List


class CrowdDetector:
    """Detects and counts people using face detection"""
    
    def __init__(self, 
                 scale_factor: float = 1.05,
                 min_neighbors: int = 3,
                 min_size: Tuple[int, int] = (20, 20),
                 smoothing_window: int = 5,
                 enable_rotations: bool = True):
        """
        Initialize crowd detector
        
        Args:
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size
            smoothing_window: Number of frames to average for count smoothing
        """
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml'
        ]
        self.face_cascades = []
        for cascade_name in cascade_files:
            classifier = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            if not classifier.empty():
                self.face_cascades.append(classifier)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.smoothing_window = smoothing_window
        self.enable_rotations = enable_rotations
        
        # Per-camera state
        self.recent_counts = {}
        self.current_counts = {}
    
    def detect_faces(self, frame: np.ndarray, camera_id: int = 0) -> Tuple[np.ndarray, int, List[Tuple[int, int, int, int]]]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame
            camera_id: Camera identifier for tracking
            
        Returns:
            Tuple of (annotated_frame, people_count, face_locations)
        """
        if frame is None:
            return None, 0, []
        
        # Convert to grayscale and enhance contrast for low-light or rotated faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        candidate_faces = self._run_multi_cascade(gray, frame.shape)
        verified_faces = self._non_max_suppression(candidate_faces)
        
        # Update count tracking
        current_count = len(verified_faces)
        
        # Initialize tracking for new camera
        if camera_id not in self.recent_counts:
            self.recent_counts[camera_id] = []
            self.current_counts[camera_id] = 0
        
        # Update recent counts
        self.recent_counts[camera_id].append(current_count)
        if len(self.recent_counts[camera_id]) > self.smoothing_window:
            self.recent_counts[camera_id].pop(0)
        
        # Calculate smoothed count
        if self.recent_counts[camera_id]:
            # Use median for more stable count
            smoothed_count = int(np.median(self.recent_counts[camera_id]))
            self.current_counts[camera_id] = smoothed_count
        else:
            self.current_counts[camera_id] = current_count
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw rectangles around verified faces
        for (x, y, w, h) in verified_faces:
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return annotated_frame, self.current_counts[camera_id], verified_faces

    def _run_multi_cascade(self, gray_frame: np.ndarray, original_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Run multiple cascades (optionally with rotations) and collect detections"""
        detections: List[Tuple[int, int, int, int]] = []
        if not self.face_cascades:
            return detections
        rotations = [('none', gray_frame)]
        if self.enable_rotations:
            rotations.extend([
                ('cw', cv2.rotate(gray_frame, cv2.ROTATE_90_CLOCKWISE)),
                ('ccw', cv2.rotate(gray_frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
            ])
        
        for rotation, processed_gray in rotations:
            for cascade in self.face_cascades:
                faces = cascade.detectMultiScale(
                    processed_gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size
                )
                for rect in faces:
                    if rotation == 'none':
                        detections.append(rect)
                    else:
                        mapped = self._map_rotated_rect(rect, original_shape, rotation)
                        if mapped:
                            detections.append(mapped)
        return detections

    def _map_rotated_rect(self, rect: Tuple[int, int, int, int], original_shape: Tuple[int, int, int], rotation: str):
        """Map rotated detection back to the original frame coordinates"""
        h, w = original_shape[:2]
        x, y, rw, rh = rect
        corners = np.array([
            [x, y],
            [x + rw, y],
            [x, y + rh],
            [x + rw, y + rh]
        ], dtype=np.float32)
        mapped_points = []
        for px, py in corners:
            if rotation == 'cw':
                orig_x = w - 1 - py
                orig_y = px
            elif rotation == 'ccw':
                orig_x = py
                orig_y = h - 1 - px
            else:
                orig_x, orig_y = px, py
            mapped_points.append((orig_x, orig_y))
        mapped_points = np.array(mapped_points)
        x_min = int(np.clip(mapped_points[:, 0].min(), 0, w - 1))
        y_min = int(np.clip(mapped_points[:, 1].min(), 0, h - 1))
        x_max = int(np.clip(mapped_points[:, 0].max(), 0, w - 1))
        y_max = int(np.clip(mapped_points[:, 1].max(), 0, h - 1))
        width = max(1, x_max - x_min)
        height = max(1, y_max - y_min)
        return (x_min, y_min, width, height)

    def _non_max_suppression(self, boxes: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.35) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        boxes_np = np.array(boxes)
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 0] + boxes_np[:, 2]
        y2 = boxes_np[:, 1] + boxes_np[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / (areas[order[1:]] + 1e-6)
            inds = np.where(overlap <= overlap_thresh)[0]
            order = order[inds + 1]
        return [tuple(map(int, boxes_np[idx])) for idx in keep]
    
    def add_overlay(self, frame: np.ndarray, count: int, threshold: int = 8) -> np.ndarray:
        """
        Add status overlay to frame
        
        Args:
            frame: Input frame
            count: Number of people detected
            threshold: Crowd density threshold
            
        Returns:
            Frame with overlay
        """
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (250, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(frame, f"People Count: {count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add density warning
        if count >= threshold:
            cv2.putText(frame, "HIGH DENSITY!", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add red border for high density
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)
        else:
            cv2.putText(frame, "Normal Density", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def get_count(self, camera_id: int = 0) -> int:
        """Get current people count for a camera"""
        return self.current_counts.get(camera_id, 0)
    
    def reset(self, camera_id: int = 0):
        """Reset tracking for a camera"""
        if camera_id in self.recent_counts:
            self.recent_counts[camera_id] = []
        if camera_id in self.current_counts:
            self.current_counts[camera_id] = 0
