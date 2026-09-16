# Debugging Guide

## 1. Check Browser Console
If features are not working, press `F12` to open Developer Tools and go to the **Console** tab.
- Look for "Status data:" logs. This shows what the backend is sending.
- If you see `Network response was not ok` or `Fetch error`, the backend might be down.

## 2. Crowd Detection Issues
If the count is still 0:
- Ensure the camera has good lighting.
- The detection parameters have been relaxed (minNeighbors=4, scaleFactor=1.1).
- If still 0, try moving closer to the camera.

## 3. Weapon Detection Issues
If camera feeds are blank:
- The system now catches errors in weapon detection to prevent the feed from breaking.
- If the feed is visible but no weapons are detected:
  - Note that the standard `yolov8n.pt` model detects **knives** but might not detect **handguns** accurately without custom training.
  - It detects: 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'.

## 4. Face Upload Issues
If upload fails:
- Ensure the image has a clear, frontal face.
- The system now accepts smaller faces (30x30 pixels).
- Check the server terminal for `[FaceRecognizer]` logs.
