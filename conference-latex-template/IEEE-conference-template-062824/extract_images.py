import cv2
import numpy as np
import os

video_path = "/home/namphuongtran9196/jetracer_official/yolo_lane_following/artifacts/dataset_images_inference.mp4"
output_path = "/home/namphuongtran9196/jetracer_official/conference-latex-template/IEEE-conference-template-062824/fig2_seg.jpg"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [int(total_frames * 0.1), int(total_frames * 0.35), int(total_frames * 0.6), int(total_frames * 0.85)]

frames = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

cap.release()

if len(frames) == 4:
    # Resize frames if they are not the same size (though they should be)
    h, w = frames[0].shape[:2]
    for i in range(1, 4):
        frames[i] = cv2.resize(frames[i], (w, h))

    # Add a border to each frame
    border_size = 5
    bordered_frames = []
    for frame in frames:
        bordered = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        bordered_frames.append(bordered)

    # Combine into a 2x2 grid
    top_row = np.hstack((bordered_frames[0], bordered_frames[1]))
    bottom_row = np.hstack((bordered_frames[2], bordered_frames[3]))
    grid = np.vstack((top_row, bottom_row))

    cv2.imwrite(output_path, grid)
    print(f"Successfully saved 2x2 grid to {output_path}")
else:
    print(f"Failed to extract 4 frames. Only extracted {len(frames)} frames.")
