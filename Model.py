# ESP32-CAM Simulated Test Pipeline using YOLOv11 + COCO weights

import torch
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import os

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pretrained YOLOv11 model (placeholder with yolov5 API)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov11.pt')
model.to(device)

# Set detection thresholds
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # IOU threshold for NMS
model.classes = [3]  # COCO class ID for motorcycle is 3

# Optionally use webcam instead of folder
use_webcam = True

if use_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
    else:
        print("Press 'q' to quit webcam preview.")
        motorcycle_total = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results.xyxy[0].to('cpu')  # Move detections to CPU for further processing
            detected_motorcycles = [d for d in detections if int(d[5]) == 3]
            motorcycle_total += len(detected_motorcycles)

            for *xyxy, conf, cls in detected_motorcycles:
                label = f"motorcycle {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('Webcam Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total motorcycles detected from webcam: {motorcycle_total}")
else:
    # Simulate ESP32: read images from a local folder
    image_folder = 'sample_images/'  # Put test images here
    image_paths = list(Path(image_folder).glob('*.jpg'))

    if not image_paths:
        print(f"No images found in {image_folder}. Please add some test .jpg images.")
    else:
        motorcycle_total = 0
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Failed to load {image_path}")
                continue

            results = model(frame)
            detections = results.xyxy[0].to('cpu')  # Ensure detections are on CPU
            detected_motorcycles = [d for d in detections if int(d[5]) == 3]
            motorcycle_total += len(detected_motorcycles)

            for *xyxy, conf, cls in detected_motorcycles:
                label = f"motorcycle {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('Detection', frame)
            print(f"Image: {image_path.name}, Motorcycles Detected: {len(detected_motorcycles)}")

            if cv2.waitKey(0) == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"Total motorcycles detected across all images: {motorcycle_total}")