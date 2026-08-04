import cv2
import numpy as np

width, height = 640, 480
fps = 30
duration = 20
total_frames = fps * duration

out = cv2.VideoWriter('traffic_light.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for i in range(total_frames):
    # Background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Draw Traffic Light Body
    tl_x1, tl_y1 = 250, 100
    tl_x2, tl_y2 = 350, 350
    cv2.rectangle(frame, (tl_x1, tl_y1), (tl_x2, tl_y2), (50, 50, 50), -1)
    
    # Determine color based on time (10s Red, 10s Green)
    if i < total_frames // 2:
        # Red light on
        cv2.circle(frame, (300, 150), 30, (0, 0, 255), -1) # Red
        cv2.circle(frame, (300, 225), 30, (50, 50, 0), -1) # Dark Yellow
        cv2.circle(frame, (300, 300), 30, (0, 50, 0), -1) # Dark Green
    else:
        # Green light on
        cv2.circle(frame, (300, 150), 30, (0, 0, 50), -1) # Dark Red
        cv2.circle(frame, (300, 225), 30, (50, 50, 0), -1) # Dark Yellow
        cv2.circle(frame, (300, 300), 30, (0, 255, 0), -1) # Green
        
    # Simulate a car moving towards the light by increasing size slightly?
    # Or just keep it simple.
    
    out.write(frame)

out.release()
print("Generated traffic_light.mp4")
