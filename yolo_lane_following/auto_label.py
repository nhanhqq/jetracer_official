import cv2
import os
import numpy as np
from ultralytics import YOLO
import sys

video_path = "19.08.2026_13.55.43_REC.mp4"
model_path = "artifacts/track_yolo26n_sem_cube_best.pt"
out_img_dir = "semantic_dataset/images/train"
out_mask_dir = "semantic_dataset/masks/train"

print("Loading model...")
model = YOLO(model_path, task='semantic')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    sys.exit(1)

frame_idx = 0
saved_idx = 0

print("Starting auto-labeling...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 5 == 0:
        # Resize frame to 224x224
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Predict
        results = model.predict(frame_resized, verbose=False)
        mask = results[0].semantic_mask.data.cpu().numpy()
        
        # Save image and mask
        img_name = f"auto_video_{saved_idx:04d}.jpg"
        mask_name = f"auto_video_{saved_idx:04d}.png"
        
        cv2.imwrite(os.path.join(out_img_dir, img_name), frame_resized)
        cv2.imwrite(os.path.join(out_mask_dir, mask_name), mask)
        
        saved_idx += 1
        if saved_idx % 50 == 0:
            print(f"Processed {saved_idx} frames...")

    frame_idx += 1

print(f"Done! Extracted and labeled {saved_idx} frames.")
