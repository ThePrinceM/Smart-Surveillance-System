# 🚀 Quick Setup Guide - Integrated Surveillance System

## ⚡ Quick Start (Easiest Method)

### Option 1: Using the Start Script (Recommended)

Simply double-click `start.bat` or run in PowerShell:

```powershell
.\start.ps1
```

This will automatically:
- Create virtual environment
- Install all dependencies
- Copy YOLO model weights
- Start the application

### Option 2: Manual Setup

1. **Navigate to project directory:**
```powershell
cd c:\Users\acer\Downloads\IP_final\integrated_surveillance
```

2. **Create virtual environment:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

4. **Copy YOLO model (if needed):**
```powershell
copy ..\final_weapon\yolov8n.pt .
```

5. **Run the application:**
```powershell
python app.py
```

6. **Access the dashboard:**
   - Open browser: `http://localhost:5000`
   - Login: `admin` / `admin123`

## 📦 What's Included

### Project Files
- ✅ `app.py` - Main Flask application
- ✅ `requirements.txt` - All Python dependencies
- ✅ `README.md` - Complete documentation
- ✅ `start.bat` / `start.ps1` - Quick start scripts

### Modules
- ✅ `models/crowd_detection.py` - People counting
- ✅ `models/face_recognition_module.py` - Face matching
- ✅ `models/weapon_detection.py` - Threat detection
- ✅ `utils/camera_manager.py` - Camera management

### Frontend
- ✅ `templates/index.html` - Modern dashboard with tabs
- ✅ `templates/login.html` - Secure login page

## 🎯 Features at a Glance

| Feature | Description | Status |
|---------|-------------|--------|
| **Crowd Detection** | Real-time people counting | ✅ Ready |
| **Face Recognition** | Upload & match faces | ✅ Ready |
| **Weapon Detection** | Detect guns/knives | ✅ Ready |
| **Unified Dashboard** | Single interface for all | ✅ Ready |
| **Multi-Camera** | Support for 2+ cameras | ✅ Ready |
| **Login System** | Secure authentication | ✅ Ready |
| **Real-time Stats** | Live system statistics | ✅ Ready |
| **Configurable** | Adjust thresholds & settings | ✅ Ready |

## 🔧 System Requirements Check

```powershell
# Check Python version (need 3.8+)
python --version

# Check if cameras work
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works:', cap.isOpened()); cap.release()"

# Check pip
pip --version
```

## 📋 Common Installation Issues & Fixes

### Issue 1: face_recognition Won't Install

**Solution:** Install pre-built wheels

1. Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/
   - `dlib-19.24.0-cp39-cp39-win_amd64.whl` (match your Python version)
   - Or install cmake first: `pip install cmake`

2. Install:
```powershell
pip install dlib-19.24.0-cp39-cp39-win_amd64.whl
pip install face-recognition
```

### Issue 2: PyTorch Installation

**Solution:** Install CPU version explicitly

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Camera Not Detected

**Solutions:**
1. Check Windows camera privacy settings
2. Close other apps using camera (Zoom, Skype, etc.)
3. Try different camera index (0, 1, 2)

### Issue 4: Module Import Errors

**Solution:** Ensure you're in the virtual environment

```powershell
# You should see (venv) in your prompt
.\venv\Scripts\Activate.ps1
```

## 🎮 Using the Dashboard

### 1. Overview Tab
- See all detections simultaneously
- View system statistics
- Monitor both cameras with all features

### 2. Crowd Detection Tab
- Monitor people counts
- Set density thresholds
- View high-density alerts

### 3. Face Recognition Tab
- Upload reference face image
- Real-time face matching
- Named face tracking

### 4. Weapon Detection Tab
- Automatic threat detection
- Visual alerts for weapons
- High-confidence filtering

### 5. Settings Tab
- Enable/disable modules
- Adjust crowd threshold
- Toggle overlays

## 🔐 Default Credentials

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Administrator |
| operator | operator123 | Operator |

**Change these in `app.py` for production use!**

## 📊 Performance Tips

### For Better Performance:
1. **Lower resolution** - Edit `camera_manager.py`, change `width=640, height=480` to `320x240`
2. **Disable unused modules** - Turn off detections you don't need in Settings
3. **Close other apps** - Free up system resources
4. **Use GPU** - Install CUDA-enabled PyTorch for faster weapon detection

### For Better Accuracy:
1. **Good lighting** - Ensure cameras have adequate lighting
2. **Camera positioning** - Position at eye level for faces
3. **Adjust thresholds** - Fine-tune in Settings tab
4. **Clear images** - Use high-quality reference faces

## 🌐 Network Access

To access from other devices on your network:

1. Find your IP address:
```powershell
ipconfig
# Look for IPv4 Address
```

2. Update `app.py`:
```python
app.run(host='0.0.0.0', port=5000, ...)
```

3. Access from other device: `http://YOUR_IP:5000`

## 🐛 Debugging

### Enable Debug Mode
In `app.py`, change:
```python
app.run(..., debug=True)
```

### View Logs
The console will show:
- Camera initialization status
- Detection model loading
- Frame processing errors
- HTTP requests

### Test Individual Components

```powershell
# Test camera manager
python -c "from utils.camera_manager import camera_manager; camera_manager.initialize_default_cameras()"

# Test crowd detection
python -c "from models.crowd_detection import CrowdDetector; cd = CrowdDetector(); print('OK')"

# Test face recognition
python -c "from models.face_recognition_module import FaceRecognizer; fr = FaceRecognizer(); print('OK')"

# Test weapon detection
python -c "from models.weapon_detection import WeaponDetector; wd = WeaponDetector(); print('Loaded:', wd.is_loaded())"
```

## 📁 File Structure Reference

```
integrated_surveillance/
├── app.py                          # Main application
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
├── SETUP_GUIDE.md                  # This file
├── start.bat                       # Windows start script
├── start.ps1                       # PowerShell start script
├── yolov8n.pt                      # YOLO model weights
├── models/
│   ├── __init__.py
│   ├── crowd_detection.py          # Crowd counting
│   ├── face_recognition_module.py  # Face matching
│   └── weapon_detection.py         # Weapon detection
├── utils/
│   ├── __init__.py
│   └── camera_manager.py           # Camera management
├── templates/
│   ├── index.html                  # Main dashboard
│   └── login.html                  # Login page
└── static/
    ├── css/
    └── js/
```

## 🎓 Next Steps

1. ✅ Complete setup using this guide
2. ✅ Test all three detection modules
3. ✅ Upload a reference face
4. ✅ Adjust settings to your needs
5. ✅ Set up for your specific use case

## 💡 Pro Tips

1. **Test Mode**: Start with one detection module enabled to verify cameras work
2. **Reference Faces**: Use passport-style photos for best face recognition
3. **Crowd Threshold**: Start with 5-8 people, adjust based on your space
4. **Weapon Detection**: Requires `yolov8n.pt` - auto-downloaded or copied from old project
5. **Multiple Users**: Add more users in `app.py` users dictionary

## 🆘 Still Having Issues?

1. Check Python version: `python --version` (need 3.8+)
2. Verify in virtual environment: Look for `(venv)` in prompt
3. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
4. Check README.md troubleshooting section
5. Test cameras independently with OpenCV

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] YOLO model file present (`yolov8n.pt`)
- [ ] Cameras detected and working
- [ ] Application starts without errors
- [ ] Can access `http://localhost:5000`
- [ ] Can login with admin credentials
- [ ] All tabs load correctly
- [ ] Camera feeds visible

## 🎉 Success!

Once everything is working:
- All camera feeds should be visible
- You can switch between detection types
- Upload faces and see matches
- Monitor crowd density
- Receive weapon alerts

**Enjoy your integrated surveillance system!** 🚀

---

For detailed information, see README.md
