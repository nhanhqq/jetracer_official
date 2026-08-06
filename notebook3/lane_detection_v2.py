#!/usr/bin/env python3
"""
lane_detection_v2.py — Advanced Lane Detection & Obstacle Avoidance for JetRacer
=================================================================================

Thuật toán phát hiện lane và tránh vật cản nâng cao cho sa bàn JetRacer:
- Sa bàn: đường tối màu, 2 lề (cam/đỏ/vàng/trắng sáng), dải phân cách đứt đoạn ở giữa
- Chống bóng, lóe sáng, ánh sáng chập chờn bằng CLAHE + adaptive thresholding
- Phát hiện vật cản bằng contour + bounding box, logic né vật cản thông minh
- Sử dụng: Canny, HoughLinesP, morphology, sliding window, lọc nhiễu Gaussian/Bilateral

Phương pháp chính:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) → chống sáng/tối cục bộ
2. Multi-channel color detection (HSV + LAB + Grayscale) → detect lề + vạch giữa
3. Canny + HoughLinesP → detect đường thẳng
4. Polynomial line fitting → tạo lane line mượt
5. Contour-based obstacle detection → phát hiện vật cản
6. Steering calculation → tính góc lái
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    """
    Bộ phát hiện lane nâng cao cho sa bàn JetRacer.
    Hỗ trợ phát hiện 2 lề đường + dải phân cách đứt đoạn + tránh vật cản.
    """

    def __init__(self, img_width=224, img_height=224):
        self.img_width = img_width
        self.img_height = img_height

        # --- ROI Settings ---
        # Chỉ xử lý phần dưới ảnh (phần đường phía trước xe)
        self.roi_top_ratio = 0.35  # Lấy từ 35% ảnh trở xuống (nhiều hơn 50% cũ)

        # --- CLAHE for lighting robustness ---
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # --- HSV Thresholds cho lane boundaries (cam/đỏ/vàng/trắng) ---
        # Orange range
        self.hsv_orange_low = np.array([3, 80, 80])
        self.hsv_orange_high = np.array([25, 255, 255])
        # Red range 1 (H=0-10)
        self.hsv_red1_low = np.array([0, 80, 80])
        self.hsv_red1_high = np.array([10, 255, 255])
        # Red range 2 (H=165-180)
        self.hsv_red2_low = np.array([165, 80, 80])
        self.hsv_red2_high = np.array([180, 255, 255])
        # Yellow range
        self.hsv_yellow_low = np.array([15, 60, 100])
        self.hsv_yellow_high = np.array([35, 255, 255])
        # White range (for dashed center line and white lane markers)
        self.hsv_white_low = np.array([0, 0, 180])
        self.hsv_white_high = np.array([180, 60, 255])

        # --- Hough Transform Parameters ---
        self.hough_threshold = 15
        self.hough_min_line_length = 8
        self.hough_max_line_gap = 40

        # --- Obstacle Detection ---
        self.obstacle_min_area = 500
        self.obstacle_max_area = 10000
        self.obstacle_avoidance_offset = 60  # pixels

        # --- Smoothing / History ---
        self.steering_history = deque(maxlen=5)
        self.left_lane_history = deque(maxlen=5)
        self.right_lane_history = deque(maxlen=5)
        self.center_line_history = deque(maxlen=5)
        self.obstacle_history = deque(maxlen=3)

        # --- Lane tracking state ---
        self.last_valid_left_x = None
        self.last_valid_right_x = None
        self.last_valid_center_x = None
        self.frames_without_lanes = 0

    def preprocess_image(self, img):
        """
        Tiền xử lý ảnh để chống bóng, lóe sáng, ánh sáng chập chờn.
        Sử dụng CLAHE trên kênh L (LAB).
        Tối ưu tốc độ: dùng GaussianBlur thay vì bilateralFilter.
        """
        # 1. Gaussian blur nhanh hơn bilateral ~3x
        denoised = cv2.GaussianBlur(img, (3, 3), 0)

        # 2. CLAHE trên LAB color space (cải thiện contrast cục bộ)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def get_roi(self, img):
        """
        Cắt ROI (Region of Interest) — phần dưới ảnh chứa đường.
        """
        h, w = img.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        return img[roi_top:h, :], roi_top

    def detect_lane_boundaries(self, roi_enhanced, roi_original):
        """
        Phát hiện lề đường (cam/đỏ/vàng) bằng HSV color thresholding.
        Trả về mask của boundary lines.
        """
        hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2HSV)

        # Detect orange
        mask_orange = cv2.inRange(hsv, self.hsv_orange_low, self.hsv_orange_high)
        # Detect red (2 ranges)
        mask_red1 = cv2.inRange(hsv, self.hsv_red1_low, self.hsv_red1_high)
        mask_red2 = cv2.inRange(hsv, self.hsv_red2_low, self.hsv_red2_high)
        # Detect yellow
        mask_yellow = cv2.inRange(hsv, self.hsv_yellow_low, self.hsv_yellow_high)

        # Combine all boundary masks
        mask_boundary = cv2.bitwise_or(mask_orange, mask_red1)
        mask_boundary = cv2.bitwise_or(mask_boundary, mask_red2)
        mask_boundary = cv2.bitwise_or(mask_boundary, mask_yellow)

        # Morphological cleanup
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        # Remove noise
        mask_boundary = cv2.morphologyEx(mask_boundary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Fill gaps
        mask_boundary = cv2.morphologyEx(mask_boundary, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        # Thicken lines slightly
        mask_boundary = cv2.dilate(mask_boundary, kernel_small, iterations=1)

        return mask_boundary

    def detect_center_line(self, roi_enhanced, mask_boundary):
        """
        Phát hiện dải phân cách đứt đoạn (trắng/sáng) ở giữa đường.
        Sử dụng kết hợp HSV white detection + adaptive thresholding.
        Chỉ detect trên vùng mặt đường (loại bỏ background/tường/trần).
        """
        h, w = roi_enhanced.shape[:2]
        hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)

        # === Tạo road mask: chỉ giữ vùng mặt đường (tối, saturation thấp) ===
        # Mặt đường tối có V thấp-trung bình, S thấp
        road_mask_dark = cv2.inRange(hsv,
                                      np.array([0, 0, 30]),     # Low: any H, low S, low V
                                      np.array([180, 120, 160]))  # High: any H, med S, med V
        # Mở rộng road mask và fill gaps
        kernel_road = np.ones((9, 9), np.uint8)
        road_mask = cv2.morphologyEx(road_mask_dark, cv2.MORPH_CLOSE, kernel_road, iterations=2)
        road_mask = cv2.dilate(road_mask, kernel_road, iterations=1)

        # Thêm boundary vào road mask (đường nằm giữa 2 boundary)
        boundary_thick = cv2.dilate(mask_boundary, np.ones((11, 11), np.uint8), iterations=2)
        road_mask = cv2.bitwise_or(road_mask, boundary_thick)

        # Chỉ giữ vùng dưới ROI (phần gần xe nhất, tin cậy hơn)
        # Phần trên ROI thường chứa tường/trần/background
        upper_cutoff = int(h * 0.15)  # Bỏ 15% trên cùng của ROI
        road_mask[:upper_cutoff, :] = 0

        # === Method 1: HSV white detection (chỉ trên road surface) ===
        mask_white = cv2.inRange(hsv, self.hsv_white_low, self.hsv_white_high)
        mask_white = cv2.bitwise_and(mask_white, road_mask)

        # === Method 2: Adaptive thresholding (chỉ trên road surface) ===
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=-10  # Stricter threshold
        )
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, road_mask)

        # === Combine methods ===
        mask_center = cv2.bitwise_or(mask_white, adaptive_thresh)

        # Loại bỏ boundary lines khỏi center mask
        boundary_dilated = cv2.dilate(mask_boundary, np.ones((7, 7), np.uint8), iterations=2)
        mask_center = cv2.bitwise_and(mask_center, cv2.bitwise_not(boundary_dilated))

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask_center = cv2.morphologyEx(mask_center, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_center = cv2.morphologyEx(mask_center, cv2.MORPH_CLOSE, kernel, iterations=1)

        # === Lọc contours: chỉ giữ blob phù hợp dải phân cách ===
        contours, _ = cv2.findContours(mask_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filtered = np.zeros_like(mask_center)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 2000:  # Dashed line segments typically small
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Aspect ratio check
                aspect_ratio = max(cw, ch) / (min(cw, ch) + 1)
                if aspect_ratio < 8:
                    # Vị trí: dải phân cách nằm giữa đường (không sát 2 bên)
                    center_blob_x = x + cw // 2
                    if w * 0.15 < center_blob_x < w * 0.85:
                        # Y position: phải nằm trong vùng đường (phần dưới ROI)
                        if y > upper_cutoff:
                            cv2.drawContours(mask_filtered, [cnt], -1, 255, -1)

        return mask_filtered

    def find_lane_lines(self, mask, img_width, line_type='boundary'):
        """
        Tìm lane lines từ mask bằng Canny + HoughLinesP.
        Phân loại thành left lines và right lines dựa trên slope và vị trí.
        """
        # Canny edge detection
        edges = cv2.Canny(mask, 50, 150)

        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        left_lines = []
        right_lines = []
        center_lines = []

        if lines is None:
            return left_lines, right_lines, center_lines

        mid_x = img_width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Filter out near-horizontal lines (noise)
            if abs(slope) < 0.15:
                continue

            avg_x = (x1 + x2) / 2

            if line_type == 'boundary':
                # Boundary lines: left has negative slope (going up-left), right positive
                if slope < -0.2 and avg_x < mid_x * 1.2:
                    left_lines.append((line, slope, length))
                elif slope > 0.2 and avg_x > mid_x * 0.8:
                    right_lines.append((line, slope, length))
            else:
                # Center/divider lines: typically near the middle
                if avg_x > mid_x * 0.3 and avg_x < mid_x * 1.7:
                    center_lines.append((line, slope, length))

        return left_lines, right_lines, center_lines

    def calculate_lane_position(self, left_lines, right_lines, center_lines,
                                 roi_height, roi_width):
        """
        Tính toán vị trí target_x dựa trên các lane lines đã detect.
        Ưu tiên: cả 2 lề > 1 lề + center > 1 lề > history
        """
        target_y = int(roi_height * 0.7)  # Điểm tính toán ở 70% chiều cao ROI

        # Tính trung bình x position cho mỗi nhóm lines
        left_x = self._avg_line_x(left_lines, target_y)
        right_x = self._avg_line_x(right_lines, target_y)
        center_x = self._avg_line_x(center_lines, target_y)

        # Update history
        if left_x is not None:
            self.left_lane_history.append(left_x)
            self.last_valid_left_x = left_x
        if right_x is not None:
            self.right_lane_history.append(right_x)
            self.last_valid_right_x = right_x
        if center_x is not None:
            self.center_line_history.append(center_x)
            self.last_valid_center_x = center_x

        # Smoothed values from history
        smooth_left = np.mean(self.left_lane_history) if self.left_lane_history else left_x
        smooth_right = np.mean(self.right_lane_history) if self.right_lane_history else right_x
        smooth_center = np.mean(self.center_line_history) if self.center_line_history else center_x

        target_x = roi_width // 2  # Default: center

        # Case 1: Cả 2 lề detected → target = giữa 2 lề
        if smooth_left is not None and smooth_right is not None:
            target_x = int((smooth_left + smooth_right) / 2)
            self.frames_without_lanes = 0

        # Case 2: 1 lề + center line → target = giữa lề và center, lệch về phía center
        elif smooth_left is not None and smooth_center is not None:
            target_x = int((smooth_left + smooth_center) / 2)
            self.frames_without_lanes = 0
        elif smooth_right is not None and smooth_center is not None:
            target_x = int((smooth_right + smooth_center) / 2)
            self.frames_without_lanes = 0

        # Case 3: Chỉ 1 lề → offset về phía đường
        elif smooth_left is not None:
            lane_width_estimate = int(roi_width * 0.45)
            target_x = int(smooth_left + lane_width_estimate)
            self.frames_without_lanes = 0
        elif smooth_right is not None:
            lane_width_estimate = int(roi_width * 0.45)
            target_x = int(smooth_right - lane_width_estimate)
            self.frames_without_lanes = 0

        # Case 4: Chỉ center line → bám center
        elif smooth_center is not None:
            target_x = int(smooth_center)
            self.frames_without_lanes = 0

        # Case 5: Không detect được gì → dùng history
        else:
            self.frames_without_lanes += 1
            if self.last_valid_left_x is not None and self.last_valid_right_x is not None:
                target_x = int((self.last_valid_left_x + self.last_valid_right_x) / 2)
            elif self.last_valid_center_x is not None:
                target_x = int(self.last_valid_center_x)

        # Clamp target_x
        target_x = max(10, min(roi_width - 10, target_x))

        return target_x, left_x, right_x, center_x

    def _avg_line_x(self, lines, target_y):
        """
        Tính trung bình x position của một nhóm lines tại target_y.
        Sử dụng weighted average theo chiều dài line.
        """
        if not lines:
            return None

        x_positions = []
        weights = []

        for line_data in lines:
            line, slope, length = line_data
            x1, y1, x2, y2 = line[0]

            # Tính x tại target_y bằng nội suy
            if abs(slope) > 0.1:
                x_at_target = x1 + (target_y - y1) / (slope + 1e-6)
            else:
                x_at_target = (x1 + x2) / 2

            x_positions.append(x_at_target)
            weights.append(length)

        if not x_positions:
            return None

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return np.mean(x_positions)
        weighted_x = sum(x * w for x, w in zip(x_positions, weights)) / total_weight
        return weighted_x

    def detect_obstacle(self, roi_original, mask_boundary, mask_center):
        """
        Phát hiện vật cản trên đường.
        Loại bỏ lane lines từ edge map → contours còn lại = vật cản.
        """
        h, w = roi_original.shape[:2]

        # Grayscale + blur
        gray = cv2.cvtColor(roi_original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Full scene edge detection
        edges_all = cv2.Canny(blurred, 40, 120)

        # Tạo mask loại bỏ lane lines (boundary + center)
        combined_lane_mask = cv2.bitwise_or(mask_boundary, mask_center)
        lane_dilated = cv2.dilate(combined_lane_mask, np.ones((7, 7), np.uint8), iterations=3)

        # Edge mà KHÔNG thuộc lane = potential obstacles
        edges_obstacle = cv2.bitwise_and(edges_all, cv2.bitwise_not(lane_dilated))

        # Morphological close to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        edges_obstacle = cv2.morphologyEx(edges_obstacle, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges_obstacle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.obstacle_min_area < area < self.obstacle_max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Bounding box area check
                bbox_area = cw * ch
                if bbox_area < self.obstacle_min_area or bbox_area > self.obstacle_max_area:
                    continue

                # Aspect ratio check (vật cản thường không quá dẹt)
                aspect = max(cw, ch) / (min(cw, ch) + 1)
                if aspect > 6:
                    continue

                # Vị trí: vật cản phải nằm trong vùng đường (không sát biên)
                center_x = x + cw // 2
                center_y = y + ch // 2
                if center_x < w * 0.05 or center_x > w * 0.95:
                    continue
                if center_y < h * 0.1:
                    continue

                # Fill ratio check (vật cản thường đặc, không rỗng)
                fill_ratio = area / (bbox_area + 1)
                if fill_ratio < 0.15:
                    continue

                obstacles.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'center_x': center_x, 'center_y': center_y,
                    'area': bbox_area
                })

        # Sort by area (biggest first) → likely the real obstacle
        obstacles.sort(key=lambda o: o['area'], reverse=True)

        # Temporal filtering: chỉ nhận obstacle nếu detect liên tục
        if obstacles:
            self.obstacle_history.append(obstacles[0])
        else:
            self.obstacle_history.append(None)

        # Kiểm tra obstacle ổn định (detect >=2 trong 3 frame gần nhất)
        stable_obstacle = None
        valid_count = sum(1 for o in self.obstacle_history if o is not None)
        if valid_count >= 2 and obstacles:
            stable_obstacle = obstacles[0]

        return stable_obstacle, obstacles

    def calculate_steering(self, target_x, img_width, obstacle=None):
        """
        Tính toán steering value từ target_x.
        Áp dụng obstacle avoidance nếu cần.
        """
        # Điều chỉnh target_x nếu có vật cản
        if obstacle is not None:
            obs_cx = obstacle['center_x']
            mid_x = img_width // 2

            if obs_cx < mid_x:
                # Vật cản bên trái → né sang phải
                target_x = min(img_width - 20, target_x + self.obstacle_avoidance_offset)
            else:
                # Vật cản bên phải → né sang trái
                target_x = max(20, target_x - self.obstacle_avoidance_offset)

        # Normalize target_x to steering [-1, 1]
        steering = (target_x / (img_width / 2.0)) - 1.0

        # Add to history for smoothing
        self.steering_history.append(steering)

        # Weighted moving average (recent values have more weight)
        weights = np.linspace(0.3, 1.0, len(self.steering_history))
        smooth_steering = np.average(list(self.steering_history), weights=weights)

        # Clamp
        smooth_steering = max(-1.0, min(1.0, smooth_steering))

        return smooth_steering, target_x

    def process_frame(self, img, draw_debug=True):
        """
        Pipeline xử lý hoàn chỉnh cho 1 frame.

        Args:
            img: BGR image (224x224)
            draw_debug: Vẽ visualization lên ảnh

        Returns:
            result_img: Ảnh với visualization
            steering: Giá trị steering [-1, 1]
            info: Dict chứa thông tin debug
        """
        h, w = img.shape[:2]

        # 1. Preprocess (CLAHE, denoise)
        enhanced = self.preprocess_image(img)

        # 2. Get ROI
        roi_enhanced, roi_top = self.get_roi(enhanced)
        roi_original, _ = self.get_roi(img)
        roi_h, roi_w = roi_enhanced.shape[:2]

        # 3. Detect lane boundaries (cam/đỏ/vàng)
        mask_boundary = self.detect_lane_boundaries(roi_enhanced, roi_original)

        # 4. Detect center dashed line (trắng/sáng)
        mask_center = self.detect_center_line(roi_enhanced, mask_boundary)

        # 5. Find lane lines from boundaries
        left_boundary, right_boundary, _ = self.find_lane_lines(
            mask_boundary, roi_w, 'boundary'
        )

        # 6. Find center divider lines
        _, _, center_divider = self.find_lane_lines(
            mask_center, roi_w, 'center'
        )

        # 7. Calculate lane position
        target_x, left_x, right_x, center_x = self.calculate_lane_position(
            left_boundary, right_boundary, center_divider,
            roi_h, roi_w
        )

        # 8. Detect obstacles
        stable_obstacle, all_obstacles = self.detect_obstacle(
            roi_original, mask_boundary, mask_center
        )

        # 9. Calculate steering
        steering, adjusted_target_x = self.calculate_steering(
            target_x, roi_w, stable_obstacle
        )

        # 10. Draw debug visualization
        result_img = img.copy()
        if draw_debug:
            result_img = self._draw_debug(
                result_img, roi_top,
                left_boundary, right_boundary, center_divider,
                adjusted_target_x, left_x, right_x, center_x,
                stable_obstacle, all_obstacles,
                steering, mask_boundary, mask_center
            )

        info = {
            'steering': steering,
            'target_x': adjusted_target_x,
            'left_x': left_x,
            'right_x': right_x,
            'center_x': center_x,
            'obstacle': stable_obstacle,
            'roi_top': roi_top,
            'frames_without_lanes': self.frames_without_lanes,
            'mask_boundary': mask_boundary,
            'mask_center': mask_center,
        }

        return result_img, steering, info

    def _draw_debug(self, img, roi_top,
                     left_lines, right_lines, center_lines,
                     target_x, left_x, right_x, center_x,
                     obstacle, all_obstacles, steering,
                     mask_boundary, mask_center):
        """
        Vẽ thông tin debug lên ảnh.
        """
        h, w = img.shape[:2]
        overlay = img.copy()

        # Draw lane lines
        for line_data in left_lines:
            line, slope, length = line_data
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top),
                     (255, 0, 0), 2)  # Blue = left boundary

        for line_data in right_lines:
            line, slope, length = line_data
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top),
                     (0, 0, 255), 2)  # Red = right boundary

        for line_data in center_lines:
            line, slope, length = line_data
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top),
                     (0, 255, 255), 2)  # Yellow = center divider

        # Draw lane positions
        roi_h = h - roi_top
        target_y = int(roi_h * 0.7) + roi_top

        if left_x is not None:
            lx = int(left_x)
            cv2.circle(overlay, (lx, target_y), 5, (255, 100, 0), -1)
            cv2.putText(overlay, "L", (lx - 5, target_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1)

        if right_x is not None:
            rx = int(right_x)
            cv2.circle(overlay, (rx, target_y), 5, (0, 100, 255), -1)
            cv2.putText(overlay, "R", (rx - 5, target_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 255), 1)

        if center_x is not None:
            cx = int(center_x)
            cv2.circle(overlay, (cx, target_y), 4, (0, 255, 255), -1)
            cv2.putText(overlay, "C", (cx - 5, target_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Draw target point (green)
        cv2.circle(overlay, (int(target_x), target_y), 8, (0, 255, 0), -1)
        cv2.circle(overlay, (int(target_x), target_y), 10, (0, 200, 0), 2)

        # Draw steering arrow
        arrow_start = (w // 2, h - 10)
        arrow_end = (int(w // 2 + steering * 50), h - 30)
        cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 255, 0), 2)

        # Draw obstacles
        if obstacle is not None:
            ox, oy = obstacle['x'], obstacle['y']
            ow, oh = obstacle['w'], obstacle['h']
            cv2.rectangle(overlay, (ox, oy + roi_top),
                         (ox + ow, oy + oh + roi_top), (0, 165, 255), 2)
            cv2.putText(overlay, "OBS!", (ox, oy + roi_top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # Show avoidance direction
            if obstacle['center_x'] < w // 2:
                cv2.putText(overlay, ">> Right", (5, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            else:
                cv2.putText(overlay, "<< Left", (5, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Draw ROI line
        cv2.line(overlay, (0, roi_top), (w, roi_top), (100, 100, 100), 1)

        # Steering text
        steer_text = f"S:{steering:.2f}"
        cv2.putText(overlay, steer_text, (w - 60, h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Blend overlay
        result = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
        return result

    def process_frame_with_masks(self, img):
        """
        Giống process_frame nhưng trả thêm debug masks cho visualization.
        Dùng để tạo video demo chi tiết.
        """
        result_img, steering, info = self.process_frame(img, draw_debug=True)

        # Tạo ảnh debug nhỏ cho masks
        roi_h = img.shape[0] - info['roi_top']
        mask_boundary_color = cv2.cvtColor(info['mask_boundary'], cv2.COLOR_GRAY2BGR)
        mask_boundary_color[:, :, 0] = 0  # Make it green tint
        mask_boundary_color[:, :, 2] = 0

        mask_center_color = cv2.cvtColor(info['mask_center'], cv2.COLOR_GRAY2BGR)
        mask_center_color[:, :, 0] = 0
        mask_center_color[:, :, 1] = 0  # Make it red tint (actually just blue channel)

        return result_img, steering, info, mask_boundary_color, mask_center_color


# =============================================================================
# Standalone helper functions for quick usage
# =============================================================================

_detector_instance = None


def get_detector(width=224, height=224):
    """Get or create singleton LaneDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LaneDetector(width, height)
    return _detector_instance


def process_single_image(img, draw_debug=True):
    """Process a single image and return result + steering."""
    detector = get_detector(img.shape[1], img.shape[0])
    return detector.process_frame(img, draw_debug)


if __name__ == '__main__':
    import os
    import glob

    # Test trên dataset
    dataset_dir = '/home/jetson/jetracer_official/notebook3/road_following_A/apex'
    images = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))

    if not images:
        print("No images found!")
    else:
        detector = LaneDetector(224, 224)
        print(f"Testing on {len(images)} images...")

        for i, img_path in enumerate(images[:5]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            result, steering, info = detector.process_frame(img)
            print(f"  [{i}] {os.path.basename(img_path)}: "
                  f"steering={steering:.3f}, "
                  f"left={info['left_x']}, "
                  f"right={info['right_x']}, "
                  f"center={info['center_x']}, "
                  f"obstacle={info['obstacle'] is not None}")

        print("Done!")
