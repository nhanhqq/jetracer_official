import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Lọc ra vùng quan tâm (ROI)
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Vẽ các đường thẳng lên ảnh
    if lines is None:
        return
    img = np.copy(img)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line.flatten()[:4]
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
    return img

def process_frame(image):
    height = image.shape[0]
    width = image.shape[1]
    
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Xóa nhiễu bằng Canny Edge
    canny = cv2.Canny(gray, 100, 200)
    
    # 3. Chọn vùng quan tâm (ROI - thường là nửa dưới khung hình)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2 + 50),
        (width, height)
    ]
    cropped_image = region_of_interest(canny,
                    np.array([region_of_interest_vertices], np.int32))
    
    # 4. Tìm các đoạn thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(cropped_image,
                            rho=6,
                            theta=np.pi/60,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)
                            
    # 5. Vẽ đè lên ảnh gốc
    image_with_lines = draw_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('solidWhiteRight.mp4')

# Lấy thông tin video gốc
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Khởi tạo VideoWriter để lưu kết quả
out = cv2.VideoWriter('output_lane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print("Đang xử lý video... (Sẽ mất khoảng vài chục giây)")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    processed_frame = process_frame(frame)
    out.write(processed_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Hoàn tất! Kết quả được lưu tại output_lane.mp4")
