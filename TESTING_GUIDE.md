# 🧪 Quick Testing Guide

## Application is Running! ✅

**Access the dashboard at**: http://localhost:5000

**Login Credentials**:
- Username: `admin`
- Password: `admin123`

---

## 📋 Testing Checklist

### 1. ✅ Test Crowd Detection
1. Navigate to the **"Crowd Detection"** tab
2. You should see both camera feeds
3. Move in front of the camera
4. Watch the people counter update
5. If count exceeds threshold (default: 8), you'll see a red alert

### 2. ✅ Test Face Recognition
1. Navigate to the **"Face Recognition"** tab
2. Click **"Upload Reference Face"**
3. Select a clear photo of a face
4. Enter a name (optional)
5. Click upload
6. The system will now search for that person across both cameras
7. When matched, you'll see:
   - Green rectangle around the matched face
   - Person's name with confidence percentage
8. To clear: Click **"Clear Face"** button

### 3. ✅ Test Weapon Detection
1. Navigate to the **"Weapon Detection"** tab
2. Both camera feeds will scan for weapons (guns, knives)
3. When a weapon-shaped object is detected:
   - Red border around the frame
   - "WEAPON DETECTED" banner
   - Bounding box around the detected object

### 4. ✅ Test "All Detections" View
1. Navigate to the **"All Detections"** tab
2. This shows all three modules working simultaneously:
   - Crowd counting overlays
   - Face matching (if reference face uploaded)
   - Weapon detection alerts
3. You'll see combined results from all modules

---

## 🎨 What You Should See

### Normal Operation:
- Camera feeds updating in real-time (~20-30 FPS)
- Smooth video playback
- Detection overlays appearing when relevant

### When Face Matched:
- **Green rectangles** around matched faces
- **Name + confidence %** displayed
- **Green banner** at top: "✓ MATCH FOUND: [Name]"

### When Weapon Detected:
- **Red border** around entire frame
- **Red banner** at top: "⚠️ WEAPON DETECTED"
- **Bounding box** around detected object

### When Crowd Exceeds Threshold:
- **People count** displayed on frame
- **Red "HIGH DENSITY" banner** if over threshold
- **Count updates** as people move

---

## 🔧 Adjusting Settings

### From the UI (Settings Panel):
- **Crowd Threshold**: Set how many people trigger an alert
- **Enable/Disable** specific modules
- **Show/Hide Overlays**: Toggle visual indicators

### From Code:
- **Face matching sensitivity**: Edit `tolerance` in `models/face_recognition_module.py` (line 14)
  - Lower = stricter (fewer false positives)
  - Higher = more lenient (catches more matches)
- **Weapon detection confidence**: Edit `confidence_threshold` in `app.py` (line 67)
- **Crowd detection sensitivity**: Edit parameters in `app.py` (lines 61-64)

---

## 📊 System Status Endpoint

Check backend health: http://localhost:5000/status

Returns JSON with:
```json
{
  "cameras": {
    "0": {"name": "Main Camera", "running": true},
    "1": {"name": "External Camera", "running": true}
  },
  "crowd": {
    "0": {"count": 2, "threshold": 8, "high_density": false},
    "1": {"count": 1, "threshold": 8, "high_density": false}
  },
  "face": {
    "has_reference": true,
    "reference_name": "John Doe"
  },
  "weapon": {
    "model_loaded": true
  }
}
```

---

## 🐛 Common Issues

### Camera Not Working?
- Close other apps using the camera (Zoom, Teams, etc.)
- Check Device Manager → Cameras
- Restart the application

### Face Not Matching?
- Ensure good lighting
- Upload a clear, frontal face photo
- Try lowering the `tolerance` value for stricter matching

### Slow Performance?
- Reduce camera resolution in `utils/camera_manager.py`
- Disable "Show Overlays" in settings
- Close other resource-intensive apps

---

## 🎯 Next Steps

1. **Test each module individually**
2. **Try uploading your own face photo**
3. **Adjust crowd threshold to see alerts**
4. **Check the "All Detections" view for combined output**
5. **Review logs in the terminal for debugging**

---

**Everything is working! No dlib, no CMake, no 10GB downloads needed!** 🚀
