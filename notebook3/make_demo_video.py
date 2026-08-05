import cv2
import numpy as np
import os
import glob

def process_cv_lane(img):
    height, width = img.shape[:2]
    
    # 1. Cắt ROI (Region of Interest)
    roi_top = height // 2
    roi = img[roi_top:height, :]
    
    # 2. Xử lý màu bằng HSV để bắt ĐÚNG màu cam và đỏ của vạch kẻ đường
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Dải màu cam
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Dải màu đỏ (HSV có 2 vùng đỏ: 0-5 và 170-180)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([5, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Gộp mask cam và đỏ
    mask_track = cv2.bitwise_or(mask_orange, mask_red1)
    mask_track = cv2.bitwise_or(mask_track, mask_red2)
    
    # Dọn nhiễu (Morphology)
    kernel = np.ones((5,5), np.uint8)
    mask_track = cv2.morphologyEx(mask_track, cv2.MORPH_OPEN, kernel)
    mask_track = cv2.dilate(mask_track, kernel, iterations=1)
    
    # 4. Tìm các vạch đường trên ảnh mask sạch
    edges = cv2.Canny(mask_track, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=50)
    
    line_image = np.zeros_like(img)
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            
            y1 += roi_top
            y2 += roi_top
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Phân tách vạch trái (slope < -0.3) và vạch phải (slope > 0.3)
            if slope < -0.3 and (x1 < width * 0.6 or x2 < width * 0.6):
                left_lines.append(line)
            elif slope > 0.3 and (x1 > width * 0.4 or x2 > width * 0.4):
                right_lines.append(line)
                
    # 5. Tính điểm đích để bám đường
    target_x = width // 2
    target_y = height // 2 + 50
    
    left_x = []
    right_x = []
    
    if left_lines:
        for l in left_lines: left_x.extend([l[0][0], l[0][2]])
    if right_lines:
        for l in right_lines: right_x.extend([l[0][0], l[0][2]])
        
    avg_left = sum(left_x)//len(left_x) if left_x else 0
    avg_right = sum(right_x)//len(right_x) if right_x else width
    
    if left_lines and right_lines:
        target_x = (avg_left + avg_right) // 2
    elif left_lines:
        target_x = avg_left + 100
    elif right_lines:
        target_x = avg_right - 100

    # Vẽ đúng 2 đường duy nhất bằng cách tính trung bình các điểm
    if left_lines:
        lx1 = int(np.mean([l[0][0] for l in left_lines]))
        ly1 = int(np.mean([l[0][1] for l in left_lines])) + roi_top
        lx2 = int(np.mean([l[0][2] for l in left_lines]))
        ly2 = int(np.mean([l[0][3] for l in left_lines])) + roi_top
        cv2.line(line_image, (lx1, ly1), (lx2, ly2), (0, 0, 255), 5)
        
    if right_lines:
        rx1 = int(np.mean([l[0][0] for l in right_lines]))
        ry1 = int(np.mean([l[0][1] for l in right_lines])) + roi_top
        rx2 = int(np.mean([l[0][2] for l in right_lines]))
        ry2 = int(np.mean([l[0][3] for l in right_lines])) + roi_top
        cv2.line(line_image, (rx1, ry1), (rx2, ry2), (0, 0, 255), 5)

    cv2.circle(line_image, (target_x, target_y), 8, (0, 255, 0), -1)
    result = cv2.addWeighted(img, 0.8, line_image, 1.0, 0)
    return result

image_folder = '/home/jetson/jetracer_official/notebook3/road_following_A/apex'
video_name = '/home/jetson/jetracer_official/notebook3/demo.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# Sort images
images.sort()

if len(images) > 0:
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    # Use mp4v codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 10.0, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        processed_frame = process_cv_lane(frame)
        video.write(processed_frame)

    video.release()
    print("Video saved as", video_name)
else:
    print("No images found!")
