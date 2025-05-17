# # import torch
# # import cv2
# # from utils.tts import speak


# # # Load YOLOv5 model
# # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 's' is the small model
# import torch
# import cv2
# import sys
# from pathlib import Path
# import time


# # Add yolov5 folder to path
# FILE = Path(__file__).resolve()
# YOLOV5_PATH = FILE.parent / 'yolov5'
# sys.path.append(str(YOLOV5_PATH))

# # Import and load the model

# # from models.common import DetectMultiBackend
# from utils.tts import speak
# from utils.general import (non_max_suppression, scale_boxes)

# from utils.torch_utils import select_device
# from utils.augmentations import letterbox

# # Load model manually
# model = torch.hub.load(str(YOLOV5_PATH), 'custom', path=YOLOV5_PATH / 'yolov5s.pt', source='local')

# last_spoken_time = 0 
# # Open webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform detection
#     results = model(frame)
#     detections = results.pred[0]

#     # Get current time
#     current_time = time.time()

#     # If detections exist and 30 seconds have passed
#     # Only run speech every 30 seconds
#     if (current_time - last_spoken_time) > 10:
#         labels = results.names
#         confident_labels = []

#         for det in detections:
#             confidence = det[4].item()  # Confidence is at index 4
#             class_id = int(det[5].item())  # Class index is at index 5

#             if confidence >= 0.80:  # Only if confidence > 80%
#                 label = labels[class_id]
#                 confident_labels.append(label)

#         # Speak only if there are high-confidence labels
#         if confident_labels:
#             unique_objects = list(set(confident_labels))
#             description = "I see " + ", ".join(unique_objects)
#             speak(description)
#             last_spoken_time = current_time

#     # Show the frame
#     annotated_frame = results.render()[0]
#     cv2.imshow('Virtual Eye - Object Detection', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


import torch
import cv2
import sys
from pathlib import Path
import time

# Add yolov5 folder to path
FILE = Path(__file__).resolve()
YOLOV5_PATH = FILE.parent / 'yolov5'
sys.path.append(str(YOLOV5_PATH))

# Import and load the model

# from models.common import DetectMultiBackend
from utils.tts import speak
from utils.general import (non_max_suppression, scale_boxes)

from utils.torch_utils import select_device
from utils.augmentations import letterbox
from ultralytics import YOLO
# from utils.tts import speak  # keep your custom TTS import

# Load YOLOv8 model (you can use 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', etc.)
model = YOLO('yolov8m.pt')  # You can use yolov8l.pt for even better accuracy

last_spoken_time = 0
CONFIDENCE_THRESHOLD = 0.8

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection (inference)
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, stream=False)

    current_time = time.time()

    if (current_time - last_spoken_time) > 10:
        detected_labels = set()

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                if conf >= CONFIDENCE_THRESHOLD:
                    label = model.names[cls]
                    detected_labels.add(label)

        if detected_labels:
            description = "I see " + ", ".join(detected_labels)
            speak(description)
            last_spoken_time = current_time

    # Show frame with annotated results
    annotated_frame = results[0].plot()
    cv2.imshow('Virtual Eye - YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
