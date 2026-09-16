"""
Face Recognition Module
Uses MTCNN and OpenCV Haar cascades for face detection,
with InceptionResnetV1 (FaceNet) embeddings for deep matching.
Includes automatic EXIF-orientation handling, multi-scale detection,
rotation fallbacks, and center-crop fallback so sample photo uploads
never fail to detect the target face.
"""
from __future__ import annotations

import io
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from typing import Tuple, Optional, List

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
except ImportError:
    InceptionResnetV1 = None
    MTCNN = None


class FaceRecognizer:
    """Face recognition pipeline backed by FaceNet embeddings and MTCNN/Haar detection."""

    def __init__(self, match_threshold: float = 0.50):
        """Initialize recognizer.

        Args:
            match_threshold: Cosine similarity threshold (0-1) for declaring a match.
        """
        self.match_threshold = float(min(max(match_threshold, 0.40), 0.90))
        self.reference_embedding: Optional[np.ndarray] = None
        self.reference_name = "Unknown"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.engine = 'template'

        # Load FaceNet & MTCNN if facenet-pytorch is available
        self.embedder = None
        self.mtcnn = None
        if InceptionResnetV1 is not None and MTCNN is not None:
            try:
                self.mtcnn = MTCNN(
                    keep_all=False,
                    select_largest=True,
                    post_process=False,
                    device=self.device
                )
                self.embedder = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
                self.engine = 'facenet'
                print(f"[FaceRecognizer] FaceNet + MTCNN initialized successfully on {self.device}")
            except Exception as e:
                print(f"[FaceRecognizer] Warning: Failed to load FaceNet/MTCNN: {e}")
                self.embedder = None
                self.mtcnn = None
        else:
            print("[FaceRecognizer] facenet-pytorch not installed. Falling back to template matcher.")

        # Fallback template matcher state (ORB + histogram)
        self.template_threshold = 0.30
        self.reference_template: Optional[np.ndarray] = None
        self.reference_descriptors: Optional[np.ndarray] = None
        self.reference_histogram: Optional[np.ndarray] = None
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Load primary and alternate Haar cascades
        self.cascades: List[cv2.CascadeClassifier] = []
        cascade_names = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_profileface.xml'
        ]
        for cname in cascade_names:
            try:
                path = cv2.data.haarcascades + cname
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.cascades.append(cascade)
            except Exception:
                pass

        if self.cascades:
            self.face_cascade = self.cascades[0]
            print(f"[FaceRecognizer] Loaded {len(self.cascades)} OpenCV Haar cascades for detection")
        else:
            self.face_cascade = None
            print("[FaceRecognizer] Warning: Could not load any Haar cascade")

    # ------------------------------------------------------------------
    # Image preprocessing & helpers
    # ------------------------------------------------------------------
    def _decode_image_bytes(self, image_data: bytes) -> Optional[np.ndarray]:
        """Decodes image bytes into an RGB numpy array, respecting EXIF orientation."""
        try:
            pil_img = Image.open(io.BytesIO(image_data))
            pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
            return np.array(pil_img)
        except Exception:
            pass

        try:
            np_img = np.frombuffer(image_data, np.uint8)
            bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[FaceRecognizer] Error decoding image: {e}")
        return None

    def _resize_if_needed(self, img_rgb: np.ndarray, max_dim: int = 1000) -> np.ndarray:
        """Downscale huge images to speed up detection without losing face details."""
        h, w = img_rgb.shape[:2]
        largest_dim = max(h, w)
        if largest_dim > max_dim:
            scale = max_dim / largest_dim
            new_size = (int(w * scale), int(h * scale))
            img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
        return img_rgb

    def _extract_face_boxes_haar(self, gray: np.ndarray, min_neighbors: int = 3) -> List[Tuple[int, int, int, int]]:
        """Run Haar cascades to detect face bounding boxes."""
        for cascade in self.cascades:
            boxes = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            if len(boxes) > 0:
                return [tuple(map(int, b)) for b in boxes]

        # Retry with histogram equalization
        eq_gray = cv2.equalizeHist(gray)
        for cascade in self.cascades:
            boxes = cascade.detectMultiScale(
                eq_gray,
                scaleFactor=1.08,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            if len(boxes) > 0:
                return [tuple(map(int, b)) for b in boxes]

        return []

    def _prepare_face_rgb(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract and resize a face crop from a BGR video frame."""
        h_frame, w_frame = frame_bgr.shape[:2]
        x, y, w, h = box
        # Slightly expand box by 10% for facial context
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_frame, x + w + margin_x)
        y2 = min(h_frame, y + h + margin_y)

        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))
        return face_rgb

    def _preprocess_tensor(self, face_rgb: np.ndarray) -> torch.Tensor:
        """Convert standard 160x160 RGB uint8 face to FaceNet normalized tensor."""
        tensor = torch.from_numpy(face_rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor.to(self.device)

    def _compute_embedding(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Compute L2-normalized 512-d FaceNet embedding."""
        if self.embedder is None:
            return None
        try:
            tensor = self._preprocess_tensor(face_rgb)
            with torch.no_grad():
                embedding = self.embedder(tensor).cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            return embedding / norm
        except Exception as e:
            print(f"[FaceRecognizer] Error computing embedding: {e}")
            return None

    def _compute_template_features(self, face_gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        keypoints, descriptors = self.orb.detectAndCompute(face_gray, None)
        hist = cv2.calcHist([face_gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return descriptors, hist

    # ------------------------------------------------------------------
    # Robust reference face detection & extraction
    # ------------------------------------------------------------------
    def _extract_best_reference_face(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Multi-stage extraction:
        1. MTCNN deep detector
        2. OpenCV Haar multi-cascade with histogram equalization
        3. 90-degree rotations search
        4. Fallback center crop (for tight portrait photos)
        """
        h, w = img_rgb.shape[:2]

        # 1. MTCNN detector
        if self.mtcnn is not None:
            try:
                pil_img = Image.fromarray(img_rgb)
                boxes, probs = self.mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    # Pick box with highest confidence
                    best_idx = int(np.argmax(probs))
                    if probs[best_idx] is not None and probs[best_idx] > 0.65:
                        box = boxes[best_idx]
                        x1 = max(0, int(box[0]))
                        y1 = max(0, int(box[1]))
                        x2 = min(w, int(box[2]))
                        y2 = min(h, int(box[3]))
                        if (x2 - x1) > 20 and (y2 - y1) > 20:
                            crop = img_rgb[y1:y2, x1:x2]
                            print(f"[FaceRecognizer] MTCNN found face with prob {probs[best_idx]:.2f}")
                            return cv2.resize(crop, (160, 160))
            except Exception as e:
                print(f"[FaceRecognizer] MTCNN extraction notice: {e}")

        # 2. Haar cascade multi-scale detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._extract_face_boxes_haar(gray, min_neighbors=3)
        if boxes:
            # Pick largest face
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
            bx, by, bw, bh = boxes[0]
            crop = img_rgb[by:by+bh, bx:bx+bw]
            print(f"[FaceRecognizer] Haar cascade found face of size {bw}x{bh}")
            return cv2.resize(crop, (160, 160))

        # 3. Rotations check (90°, 180°, 270°)
        for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rot_rgb = cv2.rotate(img_rgb, angle)
            rot_gray = cv2.cvtColor(rot_rgb, cv2.COLOR_RGB2GRAY)
            rot_boxes = self._extract_face_boxes_haar(rot_gray, min_neighbors=3)
            if rot_boxes:
                rot_boxes = sorted(rot_boxes, key=lambda b: b[2] * b[3], reverse=True)
                bx, by, bw, bh = rot_boxes[0]
                crop = rot_rgb[by:by+bh, bx:bx+bw]
                print(f"[FaceRecognizer] Haar cascade found face in rotated orientation")
                return cv2.resize(crop, (160, 160))

        # 4. Fallback: Center crop (user uploaded a sample portrait photo)
        print("[FaceRecognizer] Detectors missed; using center crop fallback for uploaded sample portrait")
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)
        center_crop = img_rgb[margin_h:h-margin_h, margin_w:w-margin_w]
        if center_crop.size > 0:
            return cv2.resize(center_crop, (160, 160))

        return cv2.resize(img_rgb, (160, 160))

    # ------------------------------------------------------------------
    # Reference upload
    # ------------------------------------------------------------------
    def upload_reference_face(self, image_data: bytes, name: str = "Target") -> bool:
        """Upload and store reference face features/embedding from raw image bytes."""
        try:
            img_rgb = self._decode_image_bytes(image_data)
            if img_rgb is None:
                print("[FaceRecognizer] Failed to decode reference image")
                return False

            img_rgb = self._resize_if_needed(img_rgb)
            face_rgb = self._extract_best_reference_face(img_rgb)
            if face_rgb is None:
                print("[FaceRecognizer] Unable to extract face from uploaded image")
                return False

            if self.engine == 'facenet' and self.embedder is not None:
                embedding = self._compute_embedding(face_rgb)
                if embedding is None:
                    print("[FaceRecognizer] Failed to compute FaceNet embedding")
                    return False
                self.reference_embedding = embedding
                print(f"[FaceRecognizer] Successfully stored FaceNet embedding for '{name}'")
            else:
                face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
                descriptors, hist = self._compute_template_features(face_gray)
                if descriptors is None or len(descriptors) < 10:
                    print("[FaceRecognizer] Not enough template features detected")
                    return False
                self.reference_template = face_gray
                self.reference_descriptors = descriptors
                self.reference_histogram = hist
                print(f"[FaceRecognizer] Successfully stored template features for '{name}'")

            self.reference_name = name
            return True

        except Exception as e:
            print(f"[FaceRecognizer] Error uploading reference face: {e}")
            return False

    # ------------------------------------------------------------------
    # Real-time frame matching
    # ------------------------------------------------------------------
    def _match_with_facenet(self, face_rgb: np.ndarray) -> Tuple[bool, float]:
        embedding = self._compute_embedding(face_rgb)
        if embedding is None or self.reference_embedding is None:
            return False, 0.0
        similarity = float(np.dot(self.reference_embedding, embedding))
        return similarity >= self.match_threshold, similarity

    def _match_with_template(self, face_rgb: np.ndarray) -> Tuple[bool, float]:
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        descriptors, hist = self._compute_template_features(face_gray)
        match_ratio = 0.0
        hist_score = 0.0

        if descriptors is not None and self.reference_descriptors is not None:
            matches = self.matcher.match(descriptors, self.reference_descriptors)
            if matches:
                good_matches = [m for m in matches if m.distance < 70]
                match_ratio = len(good_matches) / max(len(self.reference_descriptors), 1)

        if hist is not None and self.reference_histogram is not None:
            hist_score = cv2.compareHist(hist.astype(np.float32), self.reference_histogram.astype(np.float32), cv2.HISTCMP_CORREL)
            hist_score = max(0.0, min(float(hist_score), 1.0))

        combined_score = (match_ratio * 0.7) + (hist_score * 0.3)
        return combined_score >= self.template_threshold, combined_score

    def detect_and_match(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """Detect all faces in the video frame and compare them to the reference face."""
        if frame is None:
            return frame if frame is not None else None, False, 0

        annotated_frame = frame.copy()
        faces = []
        
        # 1. Try MTCNN for robust face detection (handles blur/angles much better than Haar)
        if self.mtcnn is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                boxes, probs = self.mtcnn.detect(pil_img)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if probs[i] is not None and probs[i] > 0.60:
                            x1 = max(0, int(box[0]))
                            y1 = max(0, int(box[1]))
                            x2 = min(frame.shape[1], int(box[2]))
                            y2 = min(frame.shape[0], int(box[3]))
                            w, h = x2 - x1, y2 - y1
                            if w > 20 and h > 20:
                                faces.append((x1, y1, w, h))
            except Exception as e:
                print(f"[FaceRecognizer] Real-time MTCNN error: {e}")

        # 2. Fallback to Haar cascades if MTCNN fails or isn't loaded
        if not faces and self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._extract_face_boxes_haar(gray, min_neighbors=4)

        match_found = False
        num_faces = len(faces)

        for (x, y, w, h) in faces:
            face_rgb = self._prepare_face_rgb(frame, (x, y, w, h))
            if face_rgb is None:
                continue

            if not self.has_reference():
                # Just draw neutral box if no reference is uploaded
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
                cv2.putText(annotated_frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                continue

            if self.engine == 'facenet' and self.reference_embedding is not None:
                is_match, score = self._match_with_facenet(face_rgb)
                confidence = max(0.0, min(99.9, ((score - self.match_threshold) / (1 - self.match_threshold + 1e-6)) * 100.0))
            else:
                is_match, score = self._match_with_template(face_rgb)
                confidence = min(99.9, (score / max(self.template_threshold, 1e-6)) * 100.0)

            if is_match:
                match_found = True
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = f"MATCH: {self.reference_name} ({confidence:.1f}%)"
                cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated_frame, match_found, num_faces

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def clear_reference(self):
        self.reference_embedding = None
        self.reference_template = None
        self.reference_descriptors = None
        self.reference_histogram = None
        self.reference_name = "Unknown"
        print("[FaceRecognizer] Reference face cleared")

    def has_reference(self) -> bool:
        if self.engine == 'facenet':
            return self.reference_embedding is not None
        return self.reference_template is not None

    def add_overlay(self, frame: np.ndarray, match_found: bool, num_faces: int) -> np.ndarray:
        if frame is None:
            return frame

        overlay_frame = frame.copy()
        if match_found:
            color = (0, 255, 0)
            status = f"✓ MATCH FOUND: {self.reference_name}"
        elif num_faces > 0:
            color = (0, 165, 255)
            status = f"Searching... ({num_faces} faces in frame)"
        elif self.has_reference():
            color = (255, 200, 0)
            status = f"Target: {self.reference_name} (Waiting for face)"
        else:
            color = (128, 128, 128)
            status = "No reference face uploaded"

        cv2.rectangle(overlay_frame, (0, 0), (frame.shape[1], 45), color, -1)
        cv2.putText(overlay_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return overlay_frame
