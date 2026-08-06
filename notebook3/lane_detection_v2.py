#!/usr/bin/env python3
"""
lane_detection_v2.py — Advanced Lane Detection & Obstacle Avoidance for JetRacer
=================================================================================

Thuật toán phát hiện lane cho sa bàn JetRacer MỚI NHẤT.
Dựa trên phản hồi: "nền đường tối, giải phân cách đứt đoạn và 2 lề có màu sáng CÙNG MÀU với nhau (vàng, cam, đỏ, trắng...)"

=== CHIẾN LƯỢC XỬ LÝ (SPATIAL CLUSTERING) ===
1. Khử màu cứng: Không gán cố định màu đỏ là lề, trắng là phân làn nữa.
2. Nhận diện Bright Markings: Mặt đường luôn là màu tối (Gray/Low-Sat). MỌI vạch kẻ (lề, phân làn)
   sẽ có màu nổi bật (Saturation cao) HOẶC độ sáng cao (Value cao).
3. 1D Spatial Clustering: Gom nhóm tất cả đường line quét được theo tọa độ X tại một điểm chiếu.
   - Nếu có 3 cụm X: Đích thị là [Lề Trái, Phân Làn, Lề Phải].
   - Nếu có 2 cụm X: Dựa vào khoảng cách giữa chúng để đoán là [Lề Trái, Lề Phải] 
     hay [Lề Trái, Phân Làn] hay [Phân Làn, Lề Phải].
   - Nếu có 1 cụm X: Đoán xem nó nằm ở rìa (Lề) hay ở giữa (Phân Làn).
4. Obstacle Avoidance Tự Do: Nếu có vật cản, xe ĐƯỢC PHÉP lấn tuyến (đè giải phân cách bên trái),
   nhưng TUYỆT ĐỐI KHÔNG đè lề (right boundary).
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    def __init__(self, img_width=224, img_height=224):
        self.img_width = img_width
        self.img_height = img_height
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # === ROI: Trapezoid covering the road area only ===
        self.roi_top_ratio = 0.45       
        self.roi_top_width_ratio = 0.70  

        # Road surface thresholds (Dark, Low Saturation)
        self.road_low  = np.array([0,   0,  10])
        self.road_high = np.array([180, 80, 140])

        # === Obstacle Detection ===
        self.obstacle_min_area        = 400
        self.obstacle_max_area        = 8000
        self.obstacle_avoidance_offset = 65  # Đánh lái mạnh hơn để né

        # === Smoothing / History ===
        self.steering_history   = deque(maxlen=7)
        self.left_x_history     = deque(maxlen=7)
        self.right_x_history    = deque(maxlen=7)
        self.center_x_history   = deque(maxlen=7)
        self.obstacle_history   = deque(maxlen=3)

        self.last_left_x   = None
        self.last_right_x  = None
        self.last_center_x = None
        self.frames_lost   = 0

        self._debug_masks = {}

    def preprocess(self, img):
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def get_trapezoid_roi(self, img):
        h, w = img.shape[:2]
        roi_top_y = int(h * self.roi_top_ratio)
        half_top_w = int(w * self.roi_top_width_ratio / 2)
        cx = w // 2

        pts = np.array([[
            [0,              h],
            [w,              h],
            [cx + half_top_w, roi_top_y],
            [cx - half_top_w, roi_top_y],
        ]], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pts, 255)
        roi_masked = cv2.bitwise_and(img, img, mask=mask)
        return roi_masked, roi_top_y, pts, mask

    def get_road_mask(self, img_roi, trap_mask):
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, self.road_low, self.road_high)
        road_mask = cv2.bitwise_and(road_mask, trap_mask)
        k9 = np.ones((9, 9), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k9, iterations=2)
        road_mask = cv2.dilate(road_mask, k9, iterations=2)
        return road_mask

    def detect_all_markings(self, img_roi, road_mask):
        """
        Dùng Canny Edge Detection để tìm MỌI vạch kẻ (lề, đứt đoạn).
        Băng keo kẻ đường có mép rất sắc nét, trong khi bóng mờ phản chiếu (glare)
        thì tạo ra gradient mềm, không có edge sắc.
        Điều này hoạt động với MỌI màu (đỏ, cam, vàng, trắng).
        """
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # Chặn vùng biên giới road_mask (chỉ giữ edges TRONG hoặc SÁT mặt đường)
        road_zone = cv2.dilate(road_mask, np.ones((9, 9), np.uint8), iterations=2)
        markings = cv2.bitwise_and(edges, road_zone)

        # Morphological cleanup nhỏ để nối nét đứt li ti
        k3 = np.ones((3, 3), np.uint8)
        markings = cv2.morphologyEx(markings, cv2.MORPH_CLOSE, k3)
        return markings

    def cluster_lane_lines(self, lane_mask, roi_h, roi_w):
        """
        Dùng HoughLines và Spatial Clustering 1D để phân cụm các làn đường.
        Trả về tọa độ X của left, center, right boundary nếu có.
        """
        # lane_mask đã là ảnh edges từ detect_all_markings
        lines = cv2.HoughLinesP(lane_mask, 1, np.pi/180, 15, minLineLength=15, maxLineGap=25)

        target_y = int(roi_h * 0.75)
        raw_lines = []
        x_points = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Bỏ qua các đường nằm ngang
                if abs(slope) < 0.15:
                    continue
                
                if abs(slope) > 0.05:
                    x_at_y = x1 + (target_y - y1) / slope
                else:
                    x_at_y = (x1 + x2) / 2
                    
                x_points.append((x_at_y, length, line))
                raw_lines.append(line)

        # --- 1D Clustering theo tọa độ X ---
        x_points.sort(key=lambda p: p[0])
        clusters = []
        cluster_lines = []

        for pt in x_points:
            x, length, line = pt
            if not clusters:
                clusters.append([x])
                cluster_lines.append([line])
            else:
                # Nếu cách cụm cũ < 75 pixel (chiều rộng tối đa của băng keo khi ở gần), gom chung
                if x - np.mean(clusters[-1]) < 75:
                    clusters[-1].append(x)
                    cluster_lines[-1].append(line)
                else:
                    clusters.append([x])
                    cluster_lines.append([line])

        lane_xs = [float(np.mean(c)) for c in clusters if len(c) > 0]

        left_x, center_x, right_x = None, None, None

        # Logic suy diễn cụm lane
        if len(lane_xs) >= 3:
            left_x   = lane_xs[0]
            center_x = lane_xs[1]
            right_x  = lane_xs[2]
            
        elif len(lane_xs) == 2:
            d = lane_xs[1] - lane_xs[0]
            if d > roi_w * 0.45:
                # Khoảng cách xa -> Đây là Lề Trái và Lề Phải
                left_x, right_x = lane_xs[0], lane_xs[1]
            elif lane_xs[0] > roi_w * 0.40:
                # Nằm nghiêng về bên phải -> Center và Right
                center_x, right_x = lane_xs[0], lane_xs[1]
            else:
                # Nằm nghiêng về bên trái -> Left và Center
                left_x, center_x = lane_xs[0], lane_xs[1]
                
        elif len(lane_xs) == 1:
            x = lane_xs[0]
            if x < roi_w * 0.35:
                left_x = x
            elif x > roi_w * 0.65:
                right_x = x
            else:
                center_x = x

        return left_x, center_x, right_x, raw_lines

    def calculate_target_x(self, left_x, center_x, right_x, roi_w):
        # Update history
        if left_x is not None:
            self.left_x_history.append(left_x)
            self.last_left_x = left_x
        if right_x is not None:
            self.right_x_history.append(right_x)
            self.last_right_x = right_x
        if center_x is not None:
            self.center_x_history.append(center_x)
            self.last_center_x = center_x

        s_left   = float(np.mean(self.left_x_history))   if self.left_x_history   else None
        s_right  = float(np.mean(self.right_x_history))  if self.right_x_history  else None
        s_center = float(np.mean(self.center_x_history)) if self.center_x_history else None

        target_x  = roi_w // 2
        case_used = "E:history"
        lane_w = int(roi_w * 0.40)

        # Cân bằng target dựa trên các mốc hiện có
        if s_left is not None and s_right is not None:
            target_x = int((s_left + s_right) / 2)
            case_used = "A:Left+Right"
            self.frames_lost = 0
        elif s_left is not None and s_center is not None:
            target_x = int((s_left + s_center) / 2)
            case_used = "B1:Left+Center"
            self.frames_lost = 0
        elif s_right is not None and s_center is not None:
            target_x = int((s_right + s_center) / 2)
            case_used = "B2:Center+Right"
            self.frames_lost = 0
        elif s_left is not None:
            target_x = int(s_left + lane_w)
            case_used = "C1:Left_only"
            self.frames_lost = 0
        elif s_right is not None:
            target_x = int(s_right - lane_w)
            case_used = "C2:Right_only"
            self.frames_lost = 0
        elif s_center is not None:
            target_x = int(s_center + int(roi_w * 0.20))
            case_used = "D:Center_only"
            self.frames_lost = 0
        else:
            self.frames_lost += 1
            if self.last_left_x and self.last_right_x:
                target_x = int((self.last_left_x + self.last_right_x) / 2)
            elif self.last_center_x:
                target_x = int(self.last_center_x + int(roi_w * 0.20))
            elif self.last_right_x:
                target_x = int(self.last_right_x - lane_w)

        return target_x, s_left, s_right, s_center, case_used

    def detect_obstacle(self, roi_img, lane_mask):
        h, w = roi_img.shape[:2]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # Loại bỏ các edges trùng với đường (lane_mask đã là edge, ta dãn nở ra một chút)
        lane_dil = cv2.dilate(lane_mask, np.ones((11, 11), np.uint8), iterations=3)
        edges_obs = cv2.bitwise_and(edges, cv2.bitwise_not(lane_dil))
        edges_obs = cv2.morphologyEx(edges_obs, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges_obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.obstacle_min_area < area < self.obstacle_max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                bbox_area = cw * ch
                aspect = max(cw, ch) / (min(cw, ch) + 1)
                if aspect > 6: continue
                cx_obs = x + cw // 2
                cy_obs = y + ch // 2
                if cx_obs < w * 0.05 or cx_obs > w * 0.95 or cy_obs < h * 0.10: continue
                if area / (bbox_area + 1) < 0.15: continue
                
                obstacles.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'center_x': cx_obs, 'center_y': cy_obs,
                    'area': bbox_area
                })

        obstacles.sort(key=lambda o: o['area'], reverse=True)
        self.obstacle_history.append(obstacles[0] if obstacles else None)
        valid_count = sum(1 for o in self.obstacle_history if o is not None)
        stable_obstacle = obstacles[0] if (valid_count >= 2 and obstacles) else None

        return stable_obstacle

    def calculate_steering(self, target_x, img_width, obstacle=None, right_bound_x=None):
        if obstacle is not None:
            obs_cx = obstacle['center_x']
            mid = img_width // 2
            # Đánh lái dứt khoát né vật cản
            if obs_cx < mid:
                target_x += self.obstacle_avoidance_offset # Trái có vật cản -> quẹo phải
            else:
                target_x -= self.obstacle_avoidance_offset # Phải có vật cản -> quẹo trái

        # TUYỆT ĐỐI KHÔNG ĐÈ LÊN LỀ PHẢI (right boundary)
        safe_margin = 35
        if right_bound_x is not None:
            max_allowed_x = right_bound_x - safe_margin
            if target_x > max_allowed_x:
                target_x = max_allowed_x
                
        # ĐƯỢC PHÉP ĐÈ LÊN GIẢI PHÂN CÁCH BÊN TRÁI (cho target_x chạy thoải mái sang trái)
        target_x = max(15, min(img_width - 15, target_x))

        steering = (target_x / (img_width / 2.0)) - 1.0
        self.steering_history.append(steering)

        weights = np.linspace(0.4, 1.0, len(self.steering_history))
        smooth = float(np.average(list(self.steering_history), weights=weights))
        smooth = max(-1.0, min(1.0, smooth))

        return smooth, target_x

    def process_frame(self, img, draw_debug=True):
        h, w = img.shape[:2]
        enhanced = self.preprocess(img)

        roi_enhanced, roi_top_y, trap_pts, trap_mask = self.get_trapezoid_roi(enhanced)
        roi_original, _, _, _ = self.get_trapezoid_roi(img)

        # 1. Quét mặt đường tối màu
        road_mask = self.get_road_mask(roi_enhanced, trap_mask)

        # 2. Quét TẤT CẢ các vạch kẻ trên đường (sáng, rực rỡ) không phân biệt màu
        lane_mask = self.detect_all_markings(roi_enhanced, road_mask)

        # 3. Phân cụm vạch kẻ theo trục X để biết đâu là Trái, Giữa, Phải
        left_x_raw, center_x_raw, right_x_raw, raw_lines = self.cluster_lane_lines(lane_mask, h, w)

        # 4. Tính toán mục tiêu (Target)
        target_x, s_left, s_right, s_center, case_used = self.calculate_target_x(
            left_x_raw, center_x_raw, right_x_raw, w)

        # 5. Phát hiện vật cản
        stable_obs = self.detect_obstacle(roi_original, lane_mask)

        # 6. Ra quyết định góc lái (Né vật cản thoải mái bên trái, cấm đè lề phải)
        steering, adj_target_x = self.calculate_steering(target_x, w, stable_obs, s_right)

        result_img = img.copy()
        if draw_debug:
            result_img = self._draw_debug(
                result_img, trap_pts, raw_lines,
                adj_target_x, s_left, s_right, s_center,
                stable_obs, steering, case_used,
                road_mask, lane_mask
            )

        self._debug_masks = {
            'lane': lane_mask,
            'road': road_mask,
        }

        info = {
            'steering': steering,
            'target_x': adj_target_x,
            'left_x': s_left,
            'right_x': s_right,
            'center_x': s_center,
            'case': case_used,
            'obstacle': stable_obs,
            'mask_lane': lane_mask,
            'mask_road': road_mask,
            'roi_top': roi_top_y,
        }
        return result_img, steering, info

    def _draw_debug(self, img, trap_pts, raw_lines,
                    target_x, left_x, right_x, center_x,
                    obstacle, steering, case_used,
                    road_mask, lane_mask):
        h, w = img.shape[:2]
        overlay = img.copy()

        # Road mask (Xanh mờ)
        road_viz = np.zeros_like(img)
        road_viz[road_mask > 0] = [0, 40, 0]
        overlay = cv2.addWeighted(overlay, 1.0, road_viz, 0.3, 0)

        # Lane mask chung (Magenta mờ)
        lane_viz = np.zeros_like(img)
        lane_viz[lane_mask > 0] = [255, 0, 255]
        overlay = cv2.addWeighted(overlay, 1.0, lane_viz, 0.4, 0)

        cv2.polylines(overlay, trap_pts, True, (80, 80, 80), 1)

        # Vẽ lines tìm được
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

        target_y_dot = int(h * 0.75)
        if left_x is not None:
            lx = int(left_x)
            cv2.circle(overlay, (lx, target_y_dot), 6, (255, 120, 0), -1)
            cv2.putText(overlay, "L", (lx - 4, target_y_dot - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 120, 0), 1)
        if right_x is not None:
            rx = int(right_x)
            cv2.circle(overlay, (rx, target_y_dot), 6, (0, 120, 255), -1)
            cv2.putText(overlay, "R", (rx - 4, target_y_dot - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 120, 255), 1)
        if center_x is not None:
            cx = int(center_x)
            cv2.circle(overlay, (cx, target_y_dot), 5, (0, 200, 200), -1)
            cv2.putText(overlay, "C", (cx - 4, target_y_dot - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 200), 1)

        # Target & Steering
        cv2.circle(overlay, (int(target_x), target_y_dot), 9, (0, 255, 0), -1)
        arrow_start = (w // 2, h - 10)
        arrow_end = (int(w // 2 + steering * 60), h - 30)
        cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.3)

        if obstacle is not None:
            ox, oy, ow, oh = obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h']
            cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh), (0, 165, 255), 2)
            cv2.putText(overlay, "OBS!", (ox, max(0, oy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        cv2.putText(overlay, f"S:{steering:+.2f}", (w - 65, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)
        cv2.putText(overlay, case_used[:18], (3, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

        return cv2.addWeighted(img, 0.35, overlay, 0.65, 0)

    def process_frame_with_masks(self, img):
        result_img, steering, info = self.process_frame(img, draw_debug=True)
        mask_lane = info['mask_lane']
        mask_color = np.zeros((*mask_lane.shape, 3), dtype=np.uint8)
        mask_color[:, :, 2] = mask_lane # Red channel
        # Fake center mask for compatibility with old demo scripts
        return result_img, steering, info, mask_color, mask_color

_detector_instance = None

def get_detector(width=224, height=224):
    global _detector_instance
    if _detector_instance is None or _detector_instance.img_width != width:
        _detector_instance = LaneDetector(width, height)
    return _detector_instance

def process_single_image(img, draw_debug=True):
    return get_detector(img.shape[1], img.shape[0]).process_frame(img, draw_debug)

if __name__ == '__main__':
    pass
