"""
Integrated Surveillance System
Combines crowd detection, face recognition, and weapon detection
"""
import os
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_cors import CORS
import cv2
import numpy as np
import time
from threading import Lock
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env configuration file if present
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    with open(_env_path, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                _k = _k.strip()
                _v = _v.strip().strip("'\"")
                if _k not in os.environ:
                    os.environ[_k] = _v

# Import our custom modules
from utils.camera_manager import camera_manager


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


# LIGHT_MODE skips importing heavy ML dependencies (torch/ultralytics/facenet)
LIGHT_MODE = _env_flag('LIGHT_MODE', False)

from models.crowd_detection import CrowdDetector
if not LIGHT_MODE:
    from models.face_recognition_module import FaceRecognizer
    from models.weapon_detection import WeaponDetector

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'integrated_surveillance_secret_key_2025')

# Enable CORS so hosted front-ends (e.g., Vercel) can call the API
allowed_origins = os.getenv('CORS_ORIGINS', '*').strip()
if not allowed_origins or allowed_origins == '*':
    CORS(app, supports_credentials=True)
else:
    origins = [origin.strip() for origin in allowed_origins.split(',') if origin.strip()]
    CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple user database (override via environment variables for deployments)
users = {}

admin_username = os.getenv('ADMIN_USERNAME', 'admin')
admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
operator_username = os.getenv('OPERATOR_USERNAME', 'operator')
operator_password = os.getenv('OPERATOR_PASSWORD', 'operator123')

if admin_username:
    users[admin_username] = {"password": admin_password}
if operator_username:
    users[operator_username] = {"password": operator_password}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Initialize detection models
if LIGHT_MODE:
    print("[App] LIGHT_MODE enabled: skipping face/weapon model imports and initialization")
else:
    print("[App] Initializing detection models...")
crowd_detector = CrowdDetector(
    model_path="yolov8s.pt",
    confidence_threshold=0.15,
    iou_threshold=0.60,
    smoothing_window=3
)

face_recognizer = None
weapon_detector = None
if not LIGHT_MODE:
    face_recognizer = FaceRecognizer(match_threshold=0.65)
    weapon_detector = WeaponDetector(
        model_path="weapon_model.pt",  # custom-trained on dataset01
        confidence_threshold=0.10,     # maximum sensitivity for presentation demo & partial blades
        target_classes=None            # uses all 10 weapon classes
    )

# Global settings
settings = {
    "crowd_threshold": 8,
    "enable_crowd_detection": True,
    "enable_face_recognition": False if LIGHT_MODE else True,
    "enable_weapon_detection": False if LIGHT_MODE else True,
    "show_overlays": True
}

frame_locks = {}
processed_frames = {}
FRAME_CACHE_TTL = 0.08  # seconds — lower = higher effective FPS


def process_frame(camera_id: int, detection_type: str) -> np.ndarray:
    """
    Process a frame with the specified detection type
    
    Args:
        camera_id: Camera ID (0 or 1)
        detection_type: Type of detection ('crowd', 'face', 'weapon', 'all')
        
    Returns:
        Processed frame as JPEG bytes
    """
    cache_bucket = processed_frames.setdefault(camera_id, {})
    now = time.time()
    cached = cache_bucket.get(detection_type)
    if cached and (now - cached['timestamp']) <= FRAME_CACHE_TTL:
        return cached['frame']

    lock = frame_locks.setdefault((camera_id, detection_type), Lock())
    with lock:
        # Double-check cache after acquiring lock to avoid duplicate work
        cached = cache_bucket.get(detection_type)
        if cached and (time.time() - cached['timestamp']) <= FRAME_CACHE_TTL:
            return cached['frame']

        # Get frame from camera manager
        frame = camera_manager.get_frame(camera_id)
    
        if frame is None:
            # Create placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Camera {camera_id} not available", 
                       (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes() if ret else b''
            cache_bucket[detection_type] = {
                'frame': frame_bytes,
                'timestamp': time.time()
            }
            return frame_bytes
        
        processed_frame = frame.copy()
        
        # Apply detection based on type
        if detection_type == 'crowd' and settings['enable_crowd_detection']:
            processed_frame, count, faces = crowd_detector.detect_faces(processed_frame, camera_id)
            if settings['show_overlays']:
                processed_frame = crowd_detector.add_overlay(processed_frame, count, settings['crowd_threshold'])
        
        elif detection_type == 'face' and settings['enable_face_recognition'] and face_recognizer is not None:
            processed_frame, match_found, num_faces = face_recognizer.detect_and_match(processed_frame)
            if settings['show_overlays']:
                processed_frame = face_recognizer.add_overlay(processed_frame, match_found, num_faces)
        elif detection_type == 'face' and (LIGHT_MODE or face_recognizer is None):
            cv2.putText(processed_frame, "Face recognition disabled (LIGHT_MODE)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        elif detection_type == 'weapon' and settings['enable_weapon_detection'] and weapon_detector is not None:
            try:
                processed_frame, weapon_detected, detections = weapon_detector.detect_weapons(processed_frame, camera_id)
                if settings['show_overlays']:
                    processed_frame = weapon_detector.add_overlay(processed_frame, weapon_detected, detections)
            except Exception as e:
                print(f"[App] Error in weapon detection: {e}")
                # Return original frame if detection fails, so feed doesn't break
                pass
        elif detection_type == 'weapon' and (LIGHT_MODE or weapon_detector is None):
            cv2.putText(processed_frame, "Weapon detection disabled (LIGHT_MODE)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        elif detection_type == 'all':
            # Apply all detections (lightweight overlay mode)
            temp_frame = processed_frame.copy()
            count = 0
            match_found = False
            num_faces = 0
            weapon_detected = False
            detections = []

            # Crowd detection
            if settings['enable_crowd_detection']:
                temp_frame, count, _ = crowd_detector.detect_faces(temp_frame, camera_id)
                if settings['show_overlays']:
                    temp_frame = crowd_detector.add_overlay(temp_frame, count, settings['crowd_threshold'])

            # Face recognition
            if settings['enable_face_recognition'] and face_recognizer is not None:
                temp_frame, match_found, num_faces = face_recognizer.detect_and_match(temp_frame)
                if settings['show_overlays']:
                    temp_frame = face_recognizer.add_overlay(temp_frame, match_found, num_faces)

            # Weapon detection
            if settings['enable_weapon_detection'] and weapon_detector is not None:
                temp_frame, weapon_detected, detections = weapon_detector.detect_weapons(temp_frame, camera_id)
                if settings['show_overlays']:
                    temp_frame = weapon_detector.add_overlay(temp_frame, weapon_detected, detections)

            if LIGHT_MODE:
                cv2.putText(temp_frame, "LIGHT_MODE: ML modules disabled",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            processed_frame = temp_frame
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes() if ret else b''
        cache_bucket[detection_type] = {
            'frame': frame_bytes,
            'timestamp': time.time()
        }
        return frame_bytes


def generate_video_stream(camera_id: int, detection_type: str):
    """Generate MJPEG stream for video feed"""
    while True:
        try:
            frame_bytes = process_frame(camera_id, detection_type)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
        except Exception as e:
            print(f"[Stream] Error generating stream: {e}")
            time.sleep(0.1)


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Main dashboard"""
    return render_template('index.html', 
                         settings=settings,
                         user=current_user)


@app.route('/video_feed/<int:camera_id>/<detection_type>')
def video_feed(camera_id, detection_type):
    """Video feed endpoint"""
    return Response(generate_video_stream(camera_id, detection_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_face', methods=['POST'])
@login_required
def upload_face():
    """Upload reference face for recognition"""
    try:
        if LIGHT_MODE or face_recognizer is None:
            return jsonify({'success': False, 'message': 'Face recognition is disabled in LIGHT_MODE'}), 503
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['face_image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Get name if provided
        name = request.form.get('face_name', 'Target Person')
        
        # Read file data
        image_data = file.read()
        
        # Upload to face recognizer
        success = face_recognizer.upload_reference_face(image_data, name)
        
        if success:
            return jsonify({'success': True, 'message': f'Reference face uploaded: {name}'})
        else:
            return jsonify({'success': False, 'message': 'No face detected in image'})
    
    except Exception as e:
        print(f"[UploadFace] Error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/clear_face', methods=['POST'])
@login_required
def clear_face():
    """Clear reference face"""
    try:
        if LIGHT_MODE or face_recognizer is None:
            return jsonify({'success': False, 'message': 'Face recognition is disabled in LIGHT_MODE'}), 503
        face_recognizer.clear_reference()
        return jsonify({'success': True, 'message': 'Reference face cleared'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/status')
def status():
    """Get system status"""
    status_data = {
        'cameras': {},
        'crowd': {},
        'face': {},
        'weapon': {}
    }

    status_data['mode'] = {
        'light_mode': bool(LIGHT_MODE)
    }
    
    active_camera_ids = list(camera_manager.cameras.keys()) or [0, 1]
    
    # Camera status
    for cam_id in active_camera_ids:
        camera = camera_manager.get_camera(cam_id)
        if camera:
            status_data['cameras'][cam_id] = {
                'name': camera.name,
                'running': camera.running,
                'last_update': camera.last_update
            }
    
    # Crowd detection status — use string keys so JSON and JS both see "0"/"1"
    for cam_id in active_camera_ids:
        count = crowd_detector.get_count(cam_id)
        status_data['crowd'][str(cam_id)] = {
            'count': count,
            'threshold': settings['crowd_threshold'],
            'high_density': count >= settings['crowd_threshold']
        }
    
    # Face recognition status
    if face_recognizer is None:
        status_data['face'] = {
            'enabled': False,
            'reason': 'LIGHT_MODE' if LIGHT_MODE else 'unavailable'
        }
    else:
        status_data['face'] = {
            'enabled': True,
            'has_reference': face_recognizer.has_reference(),
            'reference_name': face_recognizer.reference_name
        }
    
    # Weapon detection status
    if weapon_detector is None:
        status_data['weapon'] = {
            'enabled': False,
            'reason': 'LIGHT_MODE' if LIGHT_MODE else 'unavailable'
        }
    else:
        status_data['weapon'] = {
            'enabled': True,
            'model_loaded': weapon_detector.is_loaded()
        }
    
    return jsonify(status_data)


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def update_settings():
    """Update system settings"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'crowd_threshold' in data:
                settings['crowd_threshold'] = int(data['crowd_threshold'])
            
            if 'enable_crowd_detection' in data:
                settings['enable_crowd_detection'] = bool(data['enable_crowd_detection'])
            
            if 'enable_face_recognition' in data:
                if LIGHT_MODE:
                    settings['enable_face_recognition'] = False
                else:
                    settings['enable_face_recognition'] = bool(data['enable_face_recognition'])
            
            if 'enable_weapon_detection' in data:
                if LIGHT_MODE:
                    settings['enable_weapon_detection'] = False
                else:
                    settings['enable_weapon_detection'] = bool(data['enable_weapon_detection'])
            
            if 'show_overlays' in data:
                settings['show_overlays'] = bool(data['show_overlays'])

            # Flush annotated-frame cache so new settings (threshold etc) take effect immediately
            processed_frames.clear()

            return jsonify({'success': True, 'settings': settings})
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    return jsonify(settings)


@app.before_request
def initialize_cameras():
    """Initialize cameras before first request or skip if disabled."""
    if getattr(initialize_cameras, 'done', False):
        return
    disable_manager = os.getenv('DISABLE_CAMERA_MANAGER', 'false').lower() in {'1', 'true', 'yes'}
    if disable_manager:
        print("[App] Camera manager disabled via DISABLE_CAMERA_MANAGER")
        initialize_cameras.done = True
        return
    print("[App] Initializing cameras...")
    camera_sources = os.getenv('CAMERA_SOURCES', '').strip()
    if camera_sources:
        camera_manager.initialize_from_sources(camera_sources)
    else:
        camera_manager.initialize_default_cameras()
    initialize_cameras.done = True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("INTEGRATED SURVEILLANCE SYSTEM")
    print("=" * 60)
    print("Starting server...")
    print("Configured users:")
    for username in users:
        print(f"  Username: {username}")
    print("=" * 60)
    
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() in {'1', 'true', 'yes'}
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True, use_reloader=False)
