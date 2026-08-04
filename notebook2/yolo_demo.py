import cv2
import sys
from ultralytics import YOLO

# Load the BDD100K trained YOLO model
try:
    model = YOLO('yolov8-bdd100k-weights/best.pt')
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    sys.exit(1)

# BDD100K classes typically have traffic lights, signs, etc.
# The user's target mapping is 0-7, but this BDD100K model has its own mapping.
# We will just print what the model detects.

video_path = 'dashcam_real.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Limit to 60 seconds
max_frames = fps * 60
frame_count = 0

out = cv2.VideoWriter('output_yolo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Processing video with YOLOv8 BDD100K model...")

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Draw results on frame
    annotated_frame = results[0].plot()
    
    out.write(annotated_frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print("Finished processing. Output saved to output_yolo.mp4")
