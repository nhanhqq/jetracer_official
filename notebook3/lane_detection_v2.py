#!/usr/bin/env python3
"""
lane_detection_v2.py — Advanced Lane Detection & Obstacle Avoidance for JetRacer
=================================================================================

Thuật toán phát hiện lane cho sa bàn JetRacer (đường tối màu, lề băng keo đỏ/cam,
vạch phân làn trắng/đứt đoạn ở giữa).

=== SA BÀN THỰC TẾ ===
- Mặt đường: xám tối, low-saturation
- Lề đường (boundary): băng keo đỏ/cam, saturation cao, H=0-20 và H=160-180
- Vạch phân làn (center divider): trắng sáng, đứt đoạn, V cao, S thấp

=== 3 TRƯỜNG HỢP NHẬN LÀN ===
Case A – Chỉ thấy 2 đường (xe đang đi trong 1 làn):
  - Bên TRÁI: vạch đứt đoạn (center divider, trắng)
  - Bên PHẢI: lề đường (boundary tape, đỏ/cam)
  → Target = giữa 2 đường này

Case B – Chỉ thấy 2 đường (xe đi làn sát phải ngoài cùng):
  - Bên TRÁI: vạch đứt đoạn (center divider, trắng)
  - Bên PHẢI: lề ngoài (boundary tape, đỏ/cam)
  → Target = giữa 2 đường này (giống Case A)

Case C – Thấy 3 đường (xe ở giữa đường / giao lộ):
  - Bên TRÁI: lề trái (boundary tape, đỏ/cam)
  - Giữa: vạch phân làn (center divider, trắng)
  - Bên PHẢI: lề phải (boundary tape, đỏ/cam)
  → Target = giữa lề trái và lề phải, hoặc bám center divider

=== NGUYÊN TẮC XỬ LÝ ===
1. Dùng trapezoid ROI để loại bỏ background (tường, trần, người)
2. Detect mặt đường (dark + low-sat) → road mask
3. Chỉ detect lane markings TRONG VÙNG sát cạnh mặt đường
4. Sliding window để tìm vị trí chính xác của từng đường
5. Polynomial fitting để tạo đường mượt
6. Tính steering dựa trên vị trí target giữa các đường detect được
"""

import cv2
import numpy as np
from collections import deque


class LaneDetector:
    """
    Bộ phát hiện lane nâng cao cho sa bàn JetRacer.
    Hỗ trợ 2-3 đường (lề + vạch phân làn), chống nhiễu background.
    """

    def __init__(self, img_width=224, img_height=224):
        self.img_width = img_width
        self.img_height = img_height

        # --- CLAHE for lighting robustness ---
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # === ROI: Trapezoid covering the road area only ===
        # Tránh detect background (tường, trần, người đứng ngoài sa bàn)
        # Top of trapezoid at ~45% image height, wide at bottom
        self.roi_top_ratio = 0.45       # Chỉ lấy phần dưới 55% ảnh
        self.roi_top_width_ratio = 0.70  # Độ rộng tại top của trapezoid (% width)

        # === HSV Color Thresholds ===
        # Boundary tape: đỏ/cam (H=0-20 và H=155-180, S>70, V>50)
        self.boundary_ranges = [
            (np.array([0,   70, 50]),  np.array([20,  255, 255])),   # Red-Orange
            (np.array([155, 70, 50]),  np.array([180, 255, 255])),   # Red wraparound
        ]

        # Center divider: trắng sáng, đứt đoạn (S thấp, V cao)
        self.white_low  = np.array([0,   0, 180])
        self.white_high = np.array([180, 45, 255])

        # Road surface: xám tối, low-saturation
        self.road_low  = np.array([0,   0,  30])
        self.road_high = np.array([180, 80, 145])

        # === Hough Transform Parameters ===
        self.hough_threshold    = 12
        self.hough_min_length   = 10
        self.hough_max_gap      = 35

        # === Sliding Window Parameters ===
        self.n_windows    = 8     # Số cửa sổ theo chiều dọc
        self.window_margin = 20   # Bán kính tìm kiếm (pixels)
        self.min_pix      = 8    # Số pixels tối thiểu để cập nhật vị trí cửa sổ

        # === Obstacle Detection ===
        self.obstacle_min_area        = 400
        self.obstacle_max_area        = 8000
        self.obstacle_avoidance_offset = 55  # pixels

        # === Smoothing / History ===
        self.steering_history   = deque(maxlen=7)
        self.left_x_history     = deque(maxlen=5)
        self.right_x_history    = deque(maxlen=5)
        self.center_x_history   = deque(maxlen=5)
        self.obstacle_history   = deque(maxlen=3)

        # === Lane tracking state ===
        self.last_left_x   = None
        self.last_right_x  = None
        self.last_center_x = None
        self.frames_lost   = 0

        # === Debug ===
        self._debug_masks = {}

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def preprocess(self, img):
        """CLAHE + GaussianBlur để giảm nhiễu và cân bằng ánh sáng."""
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab2 = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # =========================================================================
    # ROI — TRAPEZOID MASK
    # =========================================================================

    def get_trapezoid_roi(self, img):
        """
        Cắt ROI hình thang để loại bỏ background phía trên và 2 góc.
        Trả về (roi_img, roi_top_y, trapezoid_pts).
        """
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

        # Full-image mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pts, 255)

        roi_masked = cv2.bitwise_and(img, img, mask=mask)

        return roi_masked, roi_top_y, pts, mask

    # =========================================================================
    # ROAD SURFACE MASK — để filter background
    # =========================================================================

    def get_road_mask(self, img_roi, trap_mask):
        """
        Detect mặt đường (dark, low-saturation) trong trapezoid ROI.
        Dùng để chỉ tìm lane markings trên mặt đường, không phải background.
        """
        h, w = img_roi.shape[:2]
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)

        road_mask = cv2.inRange(hsv, self.road_low, self.road_high)

        # Giới hạn trong trapezoid
        road_mask = cv2.bitwise_and(road_mask, trap_mask)

        # Morphological: fill holes trong road mask
        k9 = np.ones((9, 9), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k9, iterations=2)
        road_mask = cv2.dilate(road_mask, k9, iterations=2)

        return road_mask

    # =========================================================================
    # LANE BOUNDARY DETECTION (băng keo đỏ/cam)
    # =========================================================================

    def detect_boundary_mask(self, img_roi, road_mask):
        """
        Detect lề đường (băng keo đỏ/cam) CHỈ TRONG VÙNG sát mặt đường.
        Tránh bắt màu tương tự ở background (tường/người/vật khác).
        """
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)

        mask = np.zeros(img_roi.shape[:2], dtype=np.uint8)
        for lo, hi in self.boundary_ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = cv2.bitwise_or(mask, m)

        # Chỉ giữ vùng sát mặt đường (giãn nở road_mask để lấy viền đường)
        boundary_zone = cv2.dilate(road_mask, np.ones((13, 13), np.uint8), iterations=2)
        mask = cv2.bitwise_and(mask, boundary_zone)

        # Cleanup
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
        mask = cv2.dilate(mask, k3, iterations=1)

        return mask

    # =========================================================================
    # CENTER DIVIDER DETECTION (vạch trắng đứt đoạn)
    # =========================================================================

    def detect_center_mask(self, img_roi, road_mask, boundary_mask):
        """
        Detect vạch phân làn trắng đứt đoạn.
        Chỉ tìm trong vùng mặt đường, loại bỏ vùng đã detect boundary.
        """
        h, w = img_roi.shape[:2]
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

        # === Method 1: HSV white range ===
        mask_white = cv2.inRange(hsv, self.white_low, self.white_high)

        # === Method 2: Adaptive thresholding on road surface ===
        g_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adapt = cv2.adaptiveThreshold(
            g_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=17,
            C=-12
        )

        # Chỉ trong road mask
        mask_white = cv2.bitwise_and(mask_white, road_mask)
        adapt      = cv2.bitwise_and(adapt,      road_mask)

        # Kết hợp
        mask_center = cv2.bitwise_or(mask_white, adapt)

        # Loại bỏ vùng đã detect boundary
        boundary_excl = cv2.dilate(boundary_mask, np.ones((9, 9), np.uint8), iterations=2)
        mask_center   = cv2.bitwise_and(mask_center, cv2.bitwise_not(boundary_excl))

        # Morphological cleanup
        k3 = np.ones((3, 3), np.uint8)
        mask_center = cv2.morphologyEx(mask_center, cv2.MORPH_OPEN,  k3, iterations=1)
        mask_center = cv2.morphologyEx(mask_center, cv2.MORPH_CLOSE, k3, iterations=1)

        # === Filter contours: chỉ giữ blob phù hợp vạch đứt đoạn ===
        contours, _ = cv2.findContours(mask_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask_center)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 3000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = max(cw, ch) / (min(cw, ch) + 1)
                if aspect < 10:
                    # Không quá sát 2 bên (lề thường ở sát biên)
                    cx_blob = x + cw // 2
                    if w * 0.10 < cx_blob < w * 0.90:
                        cv2.drawContours(filtered, [cnt], -1, 255, -1)

        return filtered

    # =========================================================================
    # SLIDING WINDOW — tìm lane line position theo chiều dọc
    # =========================================================================

    def sliding_window_x(self, mask, roi_height, init_x=None):
        """
        Dùng sliding window để tìm x-position của một lane marking theo chiều cao.
        Trả về list các (x, y) điểm dọc theo làn đường.
        Returns None nếu không tìm được.
        """
        h, w = mask.shape[:2]

        # Khởi tạo vị trí ban đầu từ histogram của nửa dưới mask
        if init_x is None:
            bottom_half = mask[h // 2:, :]
            histogram   = np.sum(bottom_half, axis=0)
            if histogram.max() == 0:
                return None
            init_x = int(np.argmax(histogram))

        window_height = h // self.n_windows
        current_x     = init_x

        x_points = []
        y_points = []

        for win_idx in range(self.n_windows):
            y_high = h - win_idx * window_height
            y_low  = y_high - window_height
            x_low  = max(0, current_x - self.window_margin)
            x_high = min(w, current_x + self.window_margin)

            # Pixels trong cửa sổ này
            win_mask = mask[y_low:y_high, x_low:x_high]
            good_pix = np.sum(win_mask > 0)

            if good_pix >= self.min_pix:
                ys, xs = np.where(win_mask > 0)
                if len(xs) > 0:
                    new_x = int(np.mean(xs)) + x_low
                    current_x = new_x

            x_points.append(current_x)
            y_points.append(int((y_high + y_low) / 2))

        if len(x_points) < 2:
            return None

        return list(zip(x_points, y_points))

    # =========================================================================
    # LANE LINE POSITION EXTRACTION — Hough-based approach
    # =========================================================================

    def extract_lane_lines(self, mask, img_width, side='both'):
        """
        Tìm vị trí lane line từ mask bằng Hough Transform.
        Phân loại left/right dựa trên slope và vị trí x.

        side: 'left', 'right', 'both', 'center'
        Returns: (left_lines, right_lines) hoặc (center_lines,)
        """
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap
        )

        left_lines   = []
        right_lines  = []
        center_lines = []

        if lines is None:
            return left_lines, right_lines, center_lines

        mid_x = img_width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue

            slope  = (y2 - y1) / (x2 - x1 + 1e-6)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            avg_x  = (x1 + x2) / 2

            # Lọc đường gần nằm ngang (noise)
            if abs(slope) < 0.10:
                continue

            if side in ('both', 'left', 'right'):
                # Boundary lines: left có slope âm (đi lên bên trái), right slope dương
                if slope < -0.15 and avg_x < mid_x * 1.3:
                    left_lines.append((line, slope, length))
                elif slope > 0.15 and avg_x > mid_x * 0.7:
                    right_lines.append((line, slope, length))
                # Đường gần thẳng đứng (khi xe đang thẳng, lề song song)
                elif abs(slope) > 1.5:
                    if avg_x < mid_x:
                        left_lines.append((line, slope, length))
                    else:
                        right_lines.append((line, slope, length))

            elif side == 'center':
                # Center divider: nằm trong vùng giữa
                if mid_x * 0.2 < avg_x < mid_x * 1.8:
                    center_lines.append((line, slope, length))

        return left_lines, right_lines, center_lines

    # =========================================================================
    # LANE POSITION CALCULATION
    # =========================================================================

    def _weighted_x_at_y(self, lines, target_y):
        """Tính x trung bình có trọng số (theo length) tại target_y."""
        if not lines:
            return None
        xs = []
        ws = []
        for line, slope, length in lines:
            x1, y1, x2, y2 = line[0]
            if abs(slope) > 0.05:
                x_at_y = x1 + (target_y - y1) / (slope + 1e-8)
            else:
                x_at_y = (x1 + x2) / 2
            xs.append(x_at_y)
            ws.append(length)
        total_w = sum(ws)
        if total_w == 0:
            return float(np.mean(xs))
        return float(np.average(xs, weights=ws))

    def calculate_target_x(self, boundary_mask, center_mask, roi_h, roi_w):
        """
        Tính target_x dựa trên lane markings detect được.

        Logic:
        - Phân chia boundary_mask thành LEFT và RIGHT boundary
          dựa trên vị trí x của chúng trong ảnh
        - Center mask = vạch phân làn (giữa đường)

        3 trường hợp:
        A) Detect được left_boundary + right_boundary → target = giữa 2 lề
        B) Detect được 1 boundary + center → target = giữa boundary và center
        C) Detect được 1 boundary → offset về phía đường
        D) Chỉ có center → bám center
        """
        target_y = int(roi_h * 0.75)

        # === Tìm boundary lines (left và right) ===
        left_bound_lines, right_bound_lines, _ = self.extract_lane_lines(
            boundary_mask, roi_w, side='both'
        )

        # === Tìm center lines ===
        _, _, center_lines_raw = self.extract_lane_lines(
            center_mask, roi_w, side='center'
        )

        # Tính x tại target_y
        left_bound_x  = self._weighted_x_at_y(left_bound_lines,   target_y)
        right_bound_x = self._weighted_x_at_y(right_bound_lines,  target_y)
        center_x      = self._weighted_x_at_y(center_lines_raw,   target_y)

        # === Sliding window fallback nếu Hough thất bại ===
        if left_bound_x is None and boundary_mask.any():
            # Thử tìm bên trái (histogram trên half trái)
            left_half_mask = boundary_mask.copy()
            left_half_mask[:, roi_w // 2:] = 0
            pts = self.sliding_window_x(left_half_mask, roi_h)
            if pts:
                near_pts = [(x, y) for x, y in pts if abs(y - target_y) < roi_h // 3]
                if near_pts:
                    left_bound_x = float(np.mean([p[0] for p in near_pts]))

        if right_bound_x is None and boundary_mask.any():
            right_half_mask = boundary_mask.copy()
            right_half_mask[:, :roi_w // 2] = 0
            pts = self.sliding_window_x(right_half_mask, roi_h)
            if pts:
                near_pts = [(x, y) for x, y in pts if abs(y - target_y) < roi_h // 3]
                if near_pts:
                    right_bound_x = float(np.mean([p[0] for p in near_pts]))

        if center_x is None and center_mask.any():
            pts = self.sliding_window_x(center_mask, roi_h)
            if pts:
                near_pts = [(x, y) for x, y in pts if abs(y - target_y) < roi_h // 3]
                if near_pts:
                    center_x = float(np.mean([p[0] for p in near_pts]))

        # === Update history ===
        if left_bound_x is not None:
            # Sanity check: left phải thực sự ở bên trái center
            if left_bound_x < roi_w * 0.65:
                self.left_x_history.append(left_bound_x)
                self.last_left_x = left_bound_x
            else:
                left_bound_x = None  # Bác bỏ nếu vị trí sai

        if right_bound_x is not None:
            # Sanity check: right phải thực sự ở bên phải center
            if right_bound_x > roi_w * 0.35:
                self.right_x_history.append(right_bound_x)
                self.last_right_x = right_bound_x
            else:
                right_bound_x = None

        if center_x is not None:
            self.center_x_history.append(center_x)
            self.last_center_x = center_x

        # === Smoothed values ===
        s_left   = float(np.mean(self.left_x_history))   if self.left_x_history   else None
        s_right  = float(np.mean(self.right_x_history))  if self.right_x_history  else None
        s_center = float(np.mean(self.center_x_history)) if self.center_x_history else None

        # === Calculate target_x ===
        target_x  = roi_w // 2  # default: center
        case_used = "default"

        # Validate: left và right không được quá gần nhau
        if s_left is not None and s_right is not None:
            if s_right - s_left < roi_w * 0.15:
                # Quá gần → có thể nhầm, bỏ qua cái yếu hơn
                if len(self.left_x_history) <= len(self.right_x_history):
                    s_left = None
                else:
                    s_right = None

        # CASE A: Cả 2 lề → target = giữa 2 lề
        if s_left is not None and s_right is not None:
            target_x  = int((s_left + s_right) / 2)
            case_used = "A:both_boundary"
            self.frames_lost = 0

        # CASE B1: Left boundary + center divider
        elif s_left is not None and s_center is not None:
            # Target nằm giữa left boundary và center, lệch về phía center
            # (vì center là giữa đường, left là lề trái)
            target_x  = int((s_left + s_center) / 2)
            case_used = "B1:left+center"
            self.frames_lost = 0

        # CASE B2: Right boundary + center divider
        elif s_right is not None and s_center is not None:
            # Target nằm giữa right boundary và center
            target_x  = int((s_right + s_center) / 2)
            case_used = "B2:right+center"
            self.frames_lost = 0

        # CASE C1: Chỉ left boundary → offset sang phải (vào làn)
        elif s_left is not None:
            lane_w = int(roi_w * 0.40)
            target_x  = int(s_left + lane_w)
            case_used = "C1:left_only"
            self.frames_lost = 0

        # CASE C2: Chỉ right boundary → offset sang trái (vào làn)
        elif s_right is not None:
            lane_w = int(roi_w * 0.40)
            target_x  = int(s_right - lane_w)
            case_used = "C2:right_only"
            self.frames_lost = 0

        # CASE D: Chỉ center divider → bám sát phải center
        elif s_center is not None:
            # Xe đi bên phải làn → target hơi phải của center divider
            offset   = int(roi_w * 0.20)
            target_x  = int(s_center + offset)
            case_used = "D:center_only"
            self.frames_lost = 0

        # CASE E: Không detect được gì → dùng history
        else:
            self.frames_lost += 1
            if self.last_left_x is not None and self.last_right_x is not None:
                target_x  = int((self.last_left_x + self.last_right_x) / 2)
                case_used = "E:history_both"
            elif self.last_center_x is not None:
                offset   = int(roi_w * 0.20)
                target_x  = int(self.last_center_x + offset)
                case_used = "E:history_center"
            elif self.last_right_x is not None:
                lane_w = int(roi_w * 0.40)
                target_x  = int(self.last_right_x - lane_w)
                case_used = "E:history_right"
            elif self.last_left_x is not None:
                lane_w = int(roi_w * 0.40)
                target_x  = int(self.last_left_x + lane_w)
                case_used = "E:history_left"

        # Clamp
        target_x = max(15, min(roi_w - 15, target_x))

        return (target_x, left_bound_x, right_bound_x, center_x,
                left_bound_lines, right_bound_lines, center_lines_raw, case_used)

    # =========================================================================
    # OBSTACLE DETECTION
    # =========================================================================

    def detect_obstacle(self, roi_img, boundary_mask, center_mask):
        """Phát hiện vật cản trên đường bằng edge detection."""
        h, w = roi_img.shape[:2]
        gray    = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)

        # Loại bỏ lane markings khỏi edge map
        lane_mask  = cv2.bitwise_or(boundary_mask, center_mask)
        lane_dil   = cv2.dilate(lane_mask, np.ones((7, 7), np.uint8), iterations=3)
        edges_obs  = cv2.bitwise_and(edges, cv2.bitwise_not(lane_dil))

        # Morphological close để kết nối edges
        edges_obs = cv2.morphologyEx(
            edges_obs, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )

        contours, _ = cv2.findContours(edges_obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.obstacle_min_area < area < self.obstacle_max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                bbox_area = cw * ch
                if not (self.obstacle_min_area < bbox_area < self.obstacle_max_area):
                    continue
                aspect = max(cw, ch) / (min(cw, ch) + 1)
                if aspect > 6:
                    continue
                cx_obs = x + cw // 2
                cy_obs = y + ch // 2
                if cx_obs < w * 0.05 or cx_obs > w * 0.95:
                    continue
                if cy_obs < h * 0.10:
                    continue
                fill_ratio = area / (bbox_area + 1)
                if fill_ratio < 0.15:
                    continue
                obstacles.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'center_x': cx_obs, 'center_y': cy_obs,
                    'area': bbox_area
                })

        obstacles.sort(key=lambda o: o['area'], reverse=True)

        self.obstacle_history.append(obstacles[0] if obstacles else None)
        valid_count    = sum(1 for o in self.obstacle_history if o is not None)
        stable_obstacle = obstacles[0] if (valid_count >= 2 and obstacles) else None

        return stable_obstacle, obstacles

    # =========================================================================
    # STEERING CALCULATION
    # =========================================================================

    def calculate_steering(self, target_x, img_width, obstacle=None):
        """Tính steering [-1, 1] từ target_x, áp dụng obstacle avoidance."""
        if obstacle is not None:
            obs_cx = obstacle['center_x']
            mid    = img_width // 2
            if obs_cx < mid:
                target_x = min(img_width - 20, target_x + self.obstacle_avoidance_offset)
            else:
                target_x = max(20,             target_x - self.obstacle_avoidance_offset)

        steering = (target_x / (img_width / 2.0)) - 1.0
        self.steering_history.append(steering)

        weights = np.linspace(0.4, 1.0, len(self.steering_history))
        smooth  = float(np.average(list(self.steering_history), weights=weights))
        smooth  = max(-1.0, min(1.0, smooth))

        return smooth, target_x

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def process_frame(self, img, draw_debug=True):
        """
        Pipeline xử lý hoàn chỉnh cho 1 frame.

        Args:
            img: BGR image (thường 224x224)
            draw_debug: Vẽ visualization lên ảnh

        Returns:
            result_img: Ảnh với visualization
            steering: Giá trị steering [-1, 1]
            info: Dict chứa thông tin debug
        """
        h, w = img.shape[:2]

        # 1. Preprocess
        enhanced = self.preprocess(img)

        # 2. Trapezoid ROI (full-image, không crop)
        roi_enhanced, roi_top_y, trap_pts, trap_mask = self.get_trapezoid_roi(enhanced)
        roi_original, _,         _,        _         = self.get_trapezoid_roi(img)

        # 3. Road surface mask (để filter background)
        road_mask = self.get_road_mask(roi_enhanced, trap_mask)

        # 4. Detect boundary tape (đỏ/cam) — chỉ trên road
        boundary_mask = self.detect_boundary_mask(roi_enhanced, road_mask)

        # 5. Detect center divider (trắng đứt đoạn) — chỉ trên road
        center_mask = self.detect_center_mask(roi_enhanced, road_mask, boundary_mask)

        # 6. Calculate lane positions
        (target_x, left_x, right_x, center_x,
         left_lines, right_lines, center_lines,
         case_used) = self.calculate_target_x(boundary_mask, center_mask, h, w)

        # 7. Detect obstacles
        stable_obs, all_obs = self.detect_obstacle(roi_original, boundary_mask, center_mask)

        # 8. Calculate steering
        steering, adj_target_x = self.calculate_steering(target_x, w, stable_obs)

        # 9. Debug visualization
        result_img = img.copy()
        if draw_debug:
            result_img = self._draw_debug(
                result_img, trap_pts,
                left_lines, right_lines, center_lines,
                adj_target_x, left_x, right_x, center_x,
                stable_obs, steering, case_used,
                road_mask, boundary_mask, center_mask
            )

        # Store for process_frame_with_masks
        self._debug_masks = {
            'boundary': boundary_mask,
            'center':   center_mask,
            'road':     road_mask,
        }

        info = {
            'steering':           steering,
            'target_x':           adj_target_x,
            'left_x':             left_x,
            'right_x':            right_x,
            'center_x':           center_x,
            'case':               case_used,
            'obstacle':           stable_obs,
            'frames_lost':        self.frames_lost,
            'mask_boundary':      boundary_mask,
            'mask_center':        center_mask,
            'mask_road':          road_mask,
            'roi_top':            roi_top_y,
        }

        return result_img, steering, info

    # =========================================================================
    # DEBUG VISUALIZATION
    # =========================================================================

    def _draw_debug(self, img, trap_pts,
                    left_lines, right_lines, center_lines,
                    target_x, left_x, right_x, center_x,
                    obstacle, steering, case_used,
                    road_mask, boundary_mask, center_mask):
        """Vẽ thông tin debug lên ảnh (full-image coordinates)."""
        h, w = img.shape[:2]
        overlay = img.copy()

        # --- Tô nhẹ road mask (xanh lá mờ) ---
        road_viz = np.zeros_like(img)
        road_viz[road_mask > 0] = [0, 40, 0]
        overlay = cv2.addWeighted(overlay, 1.0, road_viz, 0.3, 0)

        # --- Trapezoid ROI outline ---
        cv2.polylines(overlay, trap_pts, True, (80, 80, 80), 1)

        # --- Boundary mask overlay (đỏ mờ) ---
        bound_viz = np.zeros_like(img)
        bound_viz[boundary_mask > 0] = [0, 0, 180]
        overlay = cv2.addWeighted(overlay, 1.0, bound_viz, 0.4, 0)

        # --- Center mask overlay (cyan mờ) ---
        center_viz = np.zeros_like(img)
        center_viz[center_mask > 0] = [180, 180, 0]
        overlay = cv2.addWeighted(overlay, 1.0, center_viz, 0.4, 0)

        # --- Vẽ Hough lines ---
        for line, slope, length in left_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 80, 0), 2)   # Blue = left boundary

        for line, slope, length in right_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 80, 255), 2)   # Red = right boundary

        for line, slope, length in center_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 200), 2)  # Yellow = center

        # --- Lane position dots ---
        target_y_dot = int(h * 0.70)

        if left_x is not None and 0 < int(left_x) < w:
            lx = int(left_x)
            cv2.circle(overlay, (lx, target_y_dot), 6, (255, 120, 0), -1)
            cv2.putText(overlay, "L", (lx - 4, target_y_dot - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 120, 0), 1)

        if right_x is not None and 0 < int(right_x) < w:
            rx = int(right_x)
            cv2.circle(overlay, (rx, target_y_dot), 6, (0, 120, 255), -1)
            cv2.putText(overlay, "R", (rx - 4, target_y_dot - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 120, 255), 1)

        if center_x is not None and 0 < int(center_x) < w:
            cx = int(center_x)
            cv2.circle(overlay, (cx, target_y_dot), 5, (0, 200, 200), -1)
            cv2.putText(overlay, "C", (cx - 4, target_y_dot - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 200), 1)

        # --- Target point (green) ---
        if 0 < int(target_x) < w:
            cv2.circle(overlay, (int(target_x), target_y_dot), 9,  (0, 255, 0), -1)
            cv2.circle(overlay, (int(target_x), target_y_dot), 11, (0, 200, 0), 2)

        # --- Steering arrow ---
        arrow_start = (w // 2, h - 10)
        arrow_end   = (int(w // 2 + steering * 60), h - 30)
        cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.3)

        # --- Obstacle box ---
        if obstacle is not None:
            ox, oy = obstacle['x'], obstacle['y']
            ow, oh = obstacle['w'], obstacle['h']
            cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh), (0, 165, 255), 2)
            cv2.putText(overlay, "OBS!", (ox, max(0, oy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            if obstacle['center_x'] < w // 2:
                cv2.putText(overlay, ">> Right", (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            else:
                cv2.putText(overlay, "<< Left",  (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # --- Text info ---
        steer_color = (0, 255, 0) if abs(steering) < 0.3 else \
                      (0, 200, 255) if abs(steering) < 0.6 else (0, 80, 255)
        cv2.putText(overlay, f"S:{steering:+.2f}", (w - 65, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, steer_color, 1)
        cv2.putText(overlay, case_used[:18], (3, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

        # Blend overlay
        return cv2.addWeighted(img, 0.35, overlay, 0.65, 0)

    # =========================================================================
    # PROCESS WITH MASKS (for demo video)
    # =========================================================================

    def process_frame_with_masks(self, img):
        """
        Giống process_frame nhưng trả thêm debug masks.
        Dùng để tạo video demo chi tiết.
        """
        result_img, steering, info = self.process_frame(img, draw_debug=True)

        mask_boundary = info['mask_boundary']
        mask_center   = info['mask_center']

        # Tạo colored mask images
        mask_b_color = np.zeros((*mask_boundary.shape, 3), dtype=np.uint8)
        mask_b_color[:, :, 1] = mask_boundary  # Green channel

        mask_c_color = np.zeros((*mask_center.shape, 3), dtype=np.uint8)
        mask_c_color[:, :, 0] = mask_center    # Blue
        mask_c_color[:, :, 1] = mask_center    # Green (= cyan)

        return result_img, steering, info, mask_b_color, mask_c_color


# =============================================================================
# Standalone helper functions
# =============================================================================

_detector_instance = None


def get_detector(width=224, height=224):
    """Get or create singleton LaneDetector instance."""
    global _detector_instance
    if _detector_instance is None or \
       _detector_instance.img_width != width or \
       _detector_instance.img_height != height:
        _detector_instance = LaneDetector(width, height)
    return _detector_instance


def process_single_image(img, draw_debug=True):
    """Process a single image and return result + steering."""
    detector = get_detector(img.shape[1], img.shape[0])
    return detector.process_frame(img, draw_debug)


# =============================================================================
# MAIN — Test trên dataset
# =============================================================================

if __name__ == '__main__':
    import os
    import sys
    import glob
    import argparse

    parser = argparse.ArgumentParser(description='Lane Detection V2 Test')
    parser.add_argument('--dataset', type=str,
                        default='/home/jetson/jetracer_official/notebook3/road_following_A/apex',
                        help='Đường dẫn đến thư mục dataset')
    parser.add_argument('--video',   action='store_true', help='Tạo video demo')
    parser.add_argument('--show',    action='store_true', help='Hiện ảnh từng frame')
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.dataset, '*.jpg')))
    if not images:
        print("Không tìm thấy ảnh trong dataset!")
        sys.exit(1)

    print(f"Testing trên {len(images)} ảnh...")
    detector = LaneDetector(224, 224)

    stats = {k: 0 for k in ['total', 'left', 'right', 'center', 'both', 'none']}

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        result, steering, info = detector.process_frame(img, draw_debug=True)
        stats['total'] += 1
        if info['left_x']   is not None: stats['left']   += 1
        if info['right_x']  is not None: stats['right']  += 1
        if info['center_x'] is not None: stats['center'] += 1
        if info['left_x'] is not None and info['right_x'] is not None:
            stats['both'] += 1
        if (info['left_x'] is None and info['right_x'] is None
                and info['center_x'] is None):
            stats['none'] += 1

        fname = os.path.basename(img_path)
        parts = fname.split('_')
        gt_x = 112 + int(parts[0])  # Ground truth apex x
        err  = abs(info['target_x'] - gt_x) if info['target_x'] else 'N/A'

        print(f"  [{i+1:3d}] {fname[:25]:25s} | "
              f"steering={steering:+.3f} | "
              f"target={info['target_x']:3d} gt={gt_x:3d} err={err} | "
              f"case={info['case']} | "
              f"L={info['left_x'] is not None} "
              f"R={info['right_x'] is not None} "
              f"C={info['center_x'] is not None}")

        if args.show:
            cv2.imshow("Lane Detection V2", result)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break

    cv2.destroyAllWindows()

    total = stats['total']
    print("\n" + "=" * 60)
    print(f"  Left detected:  {stats['left']}/{total}  ({100*stats['left']/total:.1f}%)")
    print(f"  Right detected: {stats['right']}/{total} ({100*stats['right']/total:.1f}%)")
    print(f"  Both detected:  {stats['both']}/{total}  ({100*stats['both']/total:.1f}%)")
    print(f"  Center detected:{stats['center']}/{total} ({100*stats['center']/total:.1f}%)")
    print(f"  None detected:  {stats['none']}/{total}  ({100*stats['none']/total:.1f}%)")
    print("=" * 60)
