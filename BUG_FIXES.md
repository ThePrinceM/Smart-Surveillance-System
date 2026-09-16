# Bug Fixes - December 3, 2025

## Issues Reported
1. Face reference image upload preview not showing
2. Crowd count (number of people) not updating in real-time
3. Weapon detection camera feeds not visible

---

## Fix 1: Face Image Upload Preview ✅

### Problem
When uploading a face reference image, there was no visual feedback showing the selected image before upload.

### Solution
**Modified: `templates/index.html`**

1. **Added HTML Elements:**
   - Image preview element: `<img id="uploadedImagePreview">`
   - Manual upload button: `<button id="uploadButton" onclick="uploadFace()">`
   - Face status card: `<div id="faceStatusCard">` to display current reference face name

2. **Added JavaScript:**
   - File selection event listener to display image preview using FileReader API
   - Changed from auto-upload to manual button-triggered upload for better UX
   - Preview image is shown when file is selected
   - Upload button appears after image selection
   - Original upload text is hidden when image is displayed

**Code Changes:**
```javascript
// Face image preview
document.getElementById('faceImageInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const reader = new FileReader();
        
        reader.onload = function(event) {
            const preview = document.getElementById('uploadedImagePreview');
            const uploadButton = document.getElementById('uploadButton');
            const uploadArea = document.getElementById('uploadArea');
            
            preview.src = event.target.result;
            preview.style.display = 'block';
            uploadButton.style.display = 'block';
            
            // Hide the upload text when image is shown
            uploadArea.querySelector('i').style.display = 'none';
            uploadArea.querySelector('h6').style.display = 'none';
            uploadArea.querySelector('p').style.display = 'none';
        };
        
        reader.readAsDataURL(file);
    }
});
```

---

## Fix 2: Real-Time Crowd Count Updates ✅

### Problem
The crowd count (total people detected) was not updating in real-time on the dashboard.

### Solution
**Modified: `templates/index.html` - `updateStats()` function**

**Changes:**
1. Added error handling with `.catch()` to log any fetch errors
2. Added null-safe checks for DOM elements before updating
3. Added face status card update when reference face exists
4. Enhanced crowd status badge updates with proper null checks

**Code Changes:**
```javascript
function updateStats() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            // Update people count in dashboard
            const totalPeople = (data.crowd[0]?.count || 0) + (data.crowd[1]?.count || 0);
            const peopleElement = document.getElementById('totalPeople');
            if (peopleElement) {
                peopleElement.textContent = totalPeople;
            }
            
            // Update face status in dashboard
            const faceStatus = data.face.has_reference ? 
                (data.face.reference_name || 'Active') : 'None';
            const faceElement = document.getElementById('faceStatus');
            if (faceElement) {
                faceElement.textContent = faceStatus;
            }
            
            // Update face status card if reference exists
            if (data.face.has_reference) {
                const statusCard = document.getElementById('faceStatusCard');
                const nameElement = document.getElementById('currentFaceName');
                if (statusCard && nameElement) {
                    statusCard.style.display = 'block';
                    statusCard.className = 'alert alert-success';
                    nameElement.textContent = 'Searching for: ' + data.face.reference_name;
                }
            }
            
            // Update crowd status badges
            updateCrowdStatus('crowdCam0Status', data.crowd[0]);
            updateCrowdStatus('crowdCam1Status', data.crowd[1]);
        })
        .catch(err => console.error('Error updating stats:', err));
}
```

---

## Fix 3: Weapon Detection Camera Feeds Visibility ✅

### Problem
Camera feeds in the Weapon Detection tab were not visible. Analysis showed:
- No HTTP requests for `/video_feed/*/weapon` endpoints in server logs
- Video feed images might not be loading until tab is clicked

### Solution
**Modified: `templates/index.html`**

**Added JavaScript to force-reload weapon feeds when tab is clicked:**
```javascript
// Force reload weapon detection feeds when weapon tab is shown
const weaponTabButton = document.querySelector('[data-bs-target="#weapon"]');
if (weaponTabButton) {
    weaponTabButton.addEventListener('click', function() {
        // Reload weapon detection video feeds by updating src
        const weaponFeeds = document.querySelectorAll('#weapon .camera-feed');
        weaponFeeds.forEach(feed => {
            const currentSrc = feed.getAttribute('src');
            // Add timestamp to force reload
            feed.src = currentSrc.split('?')[0] + '?t=' + new Date().getTime();
        });
    });
}
```

**How it works:**
1. Listens for clicks on the Weapon Detection tab button
2. Finds all camera feed images inside the weapon tab
3. Forces browser to reload the images by appending a timestamp query parameter
4. This triggers fresh HTTP requests to the video feed endpoints

---

## Testing Instructions

### Test Face Upload Preview:
1. Navigate to Face Recognition tab
2. Click on the upload area or "Choose File"
3. Select an image from your computer
4. **Expected:** Image preview should appear in place of upload text
5. **Expected:** "Upload Face" button should appear
6. Click "Upload Face" button
7. **Expected:** Alert confirming successful upload

### Test Crowd Count Updates:
1. Navigate to Overview or Crowd Detection tab
2. Observe the "People Detected" counter at the top
3. Move in front of Camera 0 (if available)
4. **Expected:** Count should update within 2 seconds
5. **Expected:** Crowd status badges should show current count vs threshold

### Test Weapon Detection Feeds:
1. Navigate to Weapon Detection tab
2. **Expected:** Camera 0 feed should load immediately
3. **Expected:** Camera 1 will show "Camera not available" (only 1 camera connected)
4. If feeds don't load, click the tab again
5. **Expected:** Feeds should reload with timestamp parameter

---

## Technical Details

### Files Modified:
- `integrated_surveillance/templates/index.html` (3 major sections)

### JavaScript Functions Added/Modified:
1. `uploadFace()` - New function for manual face upload
2. `updateStats()` - Enhanced with error handling and null checks
3. File input change listener - New for image preview
4. Weapon tab click listener - New for force-reloading feeds

### Known Issues:
- **Camera 1 unavailable**: Only one camera is connected to the system
  - This is expected and shown in server logs: `[CameraManager] Warning: External camera (1) not available`
  - Camera 1 feeds will show "Camera not available" placeholder
  
### Browser Compatibility:
- Tested with modern browsers (Chrome, Edge, Firefox)
- Uses FileReader API (supported in all modern browsers)
- Uses Bootstrap 5 tab events
- Uses querySelector and querySelectorAll (ES6+)

---

## Server Status
✅ Flask server running on http://127.0.0.1:5000
✅ All detection modules loaded successfully
✅ Camera 0 (Main Camera) operational
⚠️ Camera 1 (External Camera) not available (hardware limitation)

---

## Next Steps

If you encounter any issues:
1. Clear browser cache (Ctrl+Shift+Delete)
2. Hard refresh the page (Ctrl+F5)
3. Check browser console for JavaScript errors (F12 → Console tab)
4. Check Flask server logs in terminal for backend errors

For additional features or bugs, please report with:
- Browser name and version
- Steps to reproduce
- Expected vs actual behavior
- Screenshot if possible
