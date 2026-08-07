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
        self.roi_top_ratio = 0.50        # Chỉ lấy phần dưới ảnh, tránh tường/người/đèn
        self.roi_top_width_ratio = 0.92  # Đủ rộng cho lane dạng hình thang khi camera thấp

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
        self.obstacle_avoidance_offset = 36  # retained for compatibility / fine avoidance

        # === Generic lane detection ===
        # Sa bàn thực tế: road tối, lane sáng hơn hoặc màu bão hòa hơn road.
        # Không giả định lane chỉ đỏ/cam/trắng.
        self.min_lane_width_px = 32
        self.max_lane_width_px = 150
        self.default_lane_width_px = 74
        self.safety_margin_px = 16
        # Pure-pursuit style preview.  On a bend, following only the midpoint
        # at one row reacts too late; project the corridor heading farther into
        # the turn while keeping the final point inside the image safety band.
        self.heading_preview_gain = 3.0
        self.max_heading_preview_px = 72
        self.max_target_step_px = 28

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
        self.last_target_x = img_width // 2
        self.last_lane_pair = None
        self.frames_lost   = 0

        # Lane state: a lane is always a *pair* of neighbouring markings.
        # Keeping this state is essential for a two-lane track: a dashed divider
        # may disappear for a few frames, but that must not make the car select
        # the other lane or drive towards an outside boundary.
        self.active_lane_pair = None
        self._last_detected_pairs = []
        self.lane_pair_match_px = 24
        self.lane_switch_state = None
        self.obstacle_clear_frames = 0
        # At 20 FPS this is ~0.9 s: enough time to pass the obstacle before
        # trying to merge back, instead of oscillating immediately.
        self.return_after_clear_frames = 18

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
        _, sat, val = cv2.split(hsv)

        # Road là vùng tối hơn nền/lề sáng. Nới saturation để chấp nhận road hơi xanh/xám
        # dưới ánh đèn, nhưng loại vùng trắng/sáng ngoài đường.
        # Exposure-adaptive value ceiling.  A fixed V threshold makes the road
        # vanish when overhead lights flare or the camera auto-exposure jumps.
        trap_values = val[trap_mask > 0]
        if trap_values.size:
            road_v_limit = int(np.clip(np.percentile(trap_values, 68) + 28, 160, 205))
        else:
            road_v_limit = 172
        road_mask = ((val < road_v_limit) & (sat < 175)).astype(np.uint8) * 255
        road_mask = cv2.bitwise_and(road_mask, trap_mask)

        k5 = np.ones((5, 5), np.uint8)
        k11 = np.ones((11, 11), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k5, iterations=1)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k11, iterations=2)

        # Chỉ giữ component của road nối với vùng đáy-gần-tâm ảnh. Đây là điểm quan trọng
        # để không nhầm sàn trắng/tường/vật thể phía ngoài sa bàn là lane.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, 8)
        seed_labels = []
        y0 = int(h * 0.72)
        for yy in range(y0, h, 8):
            for xx in range(int(w * 0.25), int(w * 0.75), 8):
                lab = labels[yy, xx]
                if lab != 0:
                    seed_labels.append(int(lab))

        if seed_labels:
            keep_label = max(set(seed_labels), key=seed_labels.count)
        elif n_labels > 1:
            keep_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        else:
            keep_label = 0

        if keep_label != 0:
            road_mask = (labels == keep_label).astype(np.uint8) * 255
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=1)

        return road_mask

    def detect_generic_lane_mask(self, img_roi, road_mask, trap_mask):
        """
        Detect mọi lane marking dựa trên tương phản với road:
        - lane trắng/vàng/cam/xanh/... sáng hơn road
        - hoặc lane màu có saturation cao hơn road
        Sau đó chỉ giữ blob mảnh/line-like nằm trong hoặc sát road component.
        """
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_roi, cv2.COLOR_BGR2LAB)
        _, sat, val = cv2.split(hsv)
        lum = lab[:, :, 0]

        local_bg = cv2.GaussianBlur(lum, (31, 31), 0)
        local_contrast = cv2.subtract(lum, local_bg)

        # Lane màu: ưu tiên saturation cao, tương đối ổn định hơn vùng chói trắng.
        colored_mask = ((sat > 92) & (val > 45)).astype(np.uint8) * 255

        # Lane trắng/sáng: phải vừa sáng hơn nền cục bộ vừa không quá "cháy".
        # Specular highlight thường V rất cao + S thấp + blob vụn/loang, sẽ bị
        # contour filter phía dưới loại mạnh hơn.
        white_mask = (((local_contrast > 16) & (val > 95) & (sat < 135)) |
                      ((val > 185) & (sat < 75))).astype(np.uint8) * 255

        # Lane màu có thể nằm ở sát mép road/lề nên cho phép road dilate nhẹ.
        # Lane trắng/chói phải nằm trong road component thật; không dùng dilate để
        # tránh bắt hắt sáng/sàn/tường nằm sát ngoài sa bàn.
        colored_zone = cv2.dilate(road_mask, np.ones((5, 5), np.uint8), iterations=1)
        white_zone = cv2.erode(road_mask, np.ones((3, 3), np.uint8), iterations=1)

        colored_mask = cv2.bitwise_and(colored_mask, colored_zone)
        white_mask = cv2.bitwise_and(white_mask, white_zone)
        lane_mask = cv2.bitwise_or(colored_mask, white_mask)
        lane_mask = cv2.bitwise_and(lane_mask, trap_mask)

        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        # Filter blob: lane/vạch thường mảnh hoặc có màu rõ. Glare thường là
        # nhiều blob trắng vụn/loang, không ổn định theo hình học lane.
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_colored = np.zeros_like(lane_mask)
        filtered_white = np.zeros_like(lane_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10 or area > 1800:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 2 or ch < 2:
                continue

            roi_s = sat[y:y + ch, x:x + cw]
            contour_mask = np.zeros((ch, cw), dtype=np.uint8)
            shifted = cnt - np.array([[x, y]])
            cv2.drawContours(contour_mask, [shifted], -1, 255, -1)
            mean_sat = float(cv2.mean(roi_s, mask=contour_mask)[0])

            bbox_aspect = max(cw, ch) / (min(cw, ch) + 1)
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            rect_aspect = max(rw, rh) / (min(rw, rh) + 1.0)
            fill_ratio = area / float(cw * ch + 1)

            is_colored_lane = mean_sat > 85 and area >= 12
            is_line_like = (bbox_aspect > 2.0 or rect_aspect > 2.2) and area >= 16
            is_small_dash = area >= 35 and bbox_aspect > 1.35 and fill_ratio > 0.18

            if is_colored_lane:
                cv2.drawContours(filtered_colored, [cnt], -1, 255, -1)
            elif is_line_like or is_small_dash:
                cv2.drawContours(filtered_white, [cnt], -1, 255, -1)

        # Nếu có đủ lane màu, bỏ phần trắng để tránh hắt sáng làm lane mask nhảy.
        # Nếu không đủ lane màu, bật fallback trắng cho sa bàn dùng lane trắng.
        if cv2.countNonZero(filtered_colored) > 90:
            filtered = filtered_colored
        else:
            filtered = cv2.bitwise_or(filtered_colored, filtered_white)

        filtered = cv2.dilate(filtered, np.ones((3, 3), np.uint8), iterations=1)
        return filtered

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

        # Chỉ giữ vùng sát mặt đường. Không dilate quá rộng vì ánh sáng/màu ngoài
        # sa bàn sẽ bị nhận thành vạch giả.
        boundary_zone = cv2.dilate(road_mask, np.ones((7, 7), np.uint8), iterations=1)
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
        # === HSV white range ===
        mask_white = cv2.inRange(hsv, self.white_low, self.white_high)

        # Chỉ trong road mask
        mask_white = cv2.bitwise_and(mask_white, road_mask)
        mask_center = mask_white

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
            if 12 < area < 1800:
                x, y, cw, ch = cv2.boundingRect(cnt)
                bbox_aspect = max(cw, ch) / (min(cw, ch) + 1)
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                rect_aspect = max(rw, rh) / (min(rw, rh) + 1.0)
                fill_ratio = area / float(cw * ch + 1)
                looks_like_dash = (
                    bbox_aspect > 1.6 or rect_aspect > 2.0 or
                    (area > 45 and fill_ratio > 0.20)
                )
                if looks_like_dash:
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

    @staticmethod
    def _line_coords(line):
        """Return x1, y1, x2, y2 for both OpenCV Hough output layouts."""
        coords = np.asarray(line).reshape(-1)
        if coords.size != 4:
            raise ValueError("Invalid Hough line shape: %r" % (np.asarray(line).shape,))
        return tuple(int(value) for value in coords)

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
            x1, y1, x2, y2 = self._line_coords(line)
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
            x1, y1, x2, y2 = self._line_coords(line)
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

    @staticmethod
    def _scanline_peaks(mask, y, half_height=6, min_support=2.0, max_peaks=6):
        """Return separated x peaks of marking pixels around one image row."""
        h, w = mask.shape[:2]
        y0, y1 = max(0, y - half_height), min(h, y + half_height + 1)
        profile = np.sum(mask[y0:y1, :] > 0, axis=0).astype(np.float32)
        if not profile.size:
            return []
        profile = np.convolve(profile, np.ones(7, dtype=np.float32) / 7.0, mode='same')
        peaks = []
        work = profile.copy()
        for _ in range(max_peaks):
            x = int(np.argmax(work))
            if work[x] < min_support:
                break
            peaks.append((float(x), float(work[x])))
            work[max(0, x - 18):min(w, x + 19)] = 0
        return peaks

    def calculate_target_x(self, boundary_mask, center_mask, roi_h, roi_w):
        """
        Tính target_x dựa trên lane markings detect được.

        Phiên bản này không phụ thuộc màu lane. Nó gom mọi vạch sáng/màu vào
        combined_mask, ước lượng vị trí x của từng vạch tại target_y, cluster theo x,
        rồi chọn cặp lane bao quanh tâm xe. Nếu chỉ thấy 1 vạch thì offset theo
        lane width gần nhất/history.
        """
        # Aim ahead of the front bumper.  Looking too close to the bottom of a
        # 224px image makes a normal curve appear as a huge steering error and
        # makes straight-line Hough extrapolation particularly unreliable.
        target_y = int(roi_h * 0.62)
        combined_mask = cv2.bitwise_or(boundary_mask, center_mask)

        edges = cv2.Canny(combined_mask, 40, 120)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=max(8, self.hough_threshold - 4),
            minLineLength=max(8, self.hough_min_length),
            maxLineGap=min(16, self.hough_max_gap)
        )

        candidates = []
        all_lines = []
        # (x at look-ahead, x near the car, line length).  These provide the
        # link between the nearest physical corridor and its curved direction.
        line_geometries = []
        near_y = int(roi_h * 0.84)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = self._line_coords(line)
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) < 2:
                    slope = 99.0
                    x_at_y = (x1 + x2) / 2.0
                else:
                    slope = dy / (dx + 1e-6)
                    if abs(slope) < 0.08:
                        continue
                    x_at_y = x1 + (target_y - y1) / (slope + 1e-6)
                if -20 <= x_at_y <= roi_w + 20:
                    length = float(np.hypot(dx, dy))
                    line_mask = np.zeros_like(combined_mask)
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
                    support_px = cv2.countNonZero(cv2.bitwise_and(combined_mask, line_mask))
                    support_ratio = support_px / max(1.0, length * 3.0)
                    if support_ratio < 0.28:
                        continue
                    candidates.append((float(x_at_y), max(1.0, length)))
                    all_lines.append((line, slope, length))
                    if abs(slope) > 0.05:
                        x_at_near = x1 + (near_y - y1) / (slope + 1e-6)
                    else:
                        x_at_near = (x1 + x2) / 2.0
                    if -30 <= x_at_near <= roi_w + 30:
                        line_geometries.append((float(x_at_y), float(x_at_near), length))

        # Directly measure marking positions in a narrow horizontal band at the
        # look-ahead point.  This is the primary curve-safe signal: unlike a
        # Hough segment it does not project a straight line through a bend.
        band_half_height = max(5, roi_h // 28)
        for x_peak, weight in self._scanline_peaks(
                combined_mask, target_y, band_half_height, min_support=2.0):
            # Give direct local evidence more weight than an extrapolated
            # segment.  It remains combined with Hough for dashed lines.
            candidates.append((x_peak, weight * 10.0))

        # Histogram fallback bắt được vạch đứt/quẹo khi Hough ít line.
        lower = combined_mask[int(roi_h * 0.52):, :]
        histogram = np.sum(lower > 0, axis=0).astype(np.float32)
        if histogram.size:
            histogram = np.convolve(histogram, np.ones(9, dtype=np.float32) / 9.0, mode='same')
            tmp = histogram.copy()
            for _ in range(5):
                peak = int(np.argmax(tmp))
                if tmp[peak] < 4:
                    break
                candidates.append((float(peak), float(tmp[peak]) * 3.0))
                tmp[max(0, peak - 24):min(roi_w, peak + 25)] = 0

        # Cluster x candidates.
        clusters = []
        for x_val, weight in sorted(candidates, key=lambda item: item[0]):
            if not clusters or abs(x_val - clusters[-1][0]) > 22:
                clusters.append([x_val, weight])
            else:
                cx, cw = clusters[-1]
                clusters[-1] = [(cx * cw + x_val * weight) / (cw + weight), cw + weight]

        lane_xs = sorted([
            cx for cx, weight in clusters
            if weight > 8 and 3 < cx < roi_w - 3
        ])

        target_x = self.last_target_x if self.last_target_x is not None else roi_w // 2
        left_x = right_x = center_x = None
        left_lines = []
        right_lines = []
        center_lines = []
        case_used = "generic:none"

        valid_pairs = [
            (lane_xs[i], lane_xs[i + 1])
            for i in range(len(lane_xs) - 1)
            if self.min_lane_width_px <= (lane_xs[i + 1] - lane_xs[i]) <= self.max_lane_width_px
        ]
        # Keep every geometrically valid corridor for obstacle planning.  The
        # narrower stability gate below is only for deciding which lane to
        # follow; using it here would hide the adjacent lane precisely when a
        # lane switch is needed.
        self._last_detected_pairs = [(float(a), float(b)) for a, b in valid_pairs]

        # Nearest-corridor rule.  At the near field, the two boundaries of the
        # lane occupied by the car are the closest marking on each side of the
        # camera centre.  This prevents a visible part of another loop of an
        # F1-style track from winning simply because it is prominent farther
        # ahead in the frame.  Hough segments associate those two local marks
        # with their positions at the look-ahead row.
        near_peaks = self._scanline_peaks(combined_mask, near_y, max(5, roi_h // 32))
        near_left = [x for x, _ in near_peaks if x < roi_w / 2.0]
        near_right = [x for x, _ in near_peaks if x > roi_w / 2.0]
        local_pair = None
        if near_left and near_right:
            local_left, local_right = max(near_left), min(near_right)
            near_width = local_right - local_left
            if self.min_lane_width_px <= near_width <= self.max_lane_width_px * 1.35:
                def project_from_near(boundary_x):
                    projected = [(xt, length) for xt, xn, length in line_geometries
                                 if abs(xn - boundary_x) <= 22]
                    if not projected:
                        return None
                    xs, weights = zip(*projected)
                    return float(np.average(xs, weights=weights))

                local_target_left = project_from_near(local_left)
                local_target_right = project_from_near(local_right)
                if local_target_left is not None and local_target_right is not None:
                    if self.min_lane_width_px * 0.65 <= local_target_right - local_target_left <= self.max_lane_width_px * 1.5:
                        local_pair = (local_target_left, local_target_right)

        # Không nhận cặp lane nhảy quá xa so với vị trí đang bám. Frame đầu
        # chưa có history thì lấy tâm ảnh làm expected center.
        expected_center = float(self.last_target_x) if self.last_target_x is not None else roi_w / 2.0
        stable_pairs = [
            pair for pair in valid_pairs
            if abs(((pair[0] + pair[1]) / 2.0) - expected_center) <= roi_w * 0.30
        ]
        if stable_pairs:
            valid_pairs = stable_pairs
        elif self.last_lane_pair is None:
            valid_pairs = []

        if local_pair is not None:
            chosen_pair = local_pair
            case_used = "lane:nearest_corridor"
            left_x, right_x = chosen_pair
            target_x = int((left_x + right_x) / 2.0)
            self.last_lane_pair = (float(left_x), float(right_x))
            self.active_lane_pair = self.last_lane_pair
            self.left_x_history.append(float(left_x))
            self.right_x_history.append(float(right_x))
            self.last_left_x = float(left_x)
            self.last_right_x = float(right_x)
            self.frames_lost = 0

            extras = [x for x in lane_xs if abs(x - left_x) > 8 and abs(x - right_x) > 8]
            if extras:
                center_x = float(min(extras, key=lambda x: abs(x - roi_w / 2.0)))
                self.center_x_history.append(center_x)
                self.last_center_x = center_x
        elif valid_pairs:
            # Never use image centre as the primary lane identity.  The camera
            # is mounted over the car and the car can legitimately be in either
            # half of the image after a lane change.  Prefer the actively
            # tracked lane; on the first frame fall back to the pair containing
            # the camera centre.
            tracked_pair = self.active_lane_pair or self.last_lane_pair
            if tracked_pair is not None:
                chosen_pair = min(
                    valid_pairs,
                    key=lambda pair: abs(pair[0] - tracked_pair[0]) +
                                     abs(pair[1] - tracked_pair[1])
                )
                case_used = "lane:tracked_pair"
            else:
                image_center = roi_w / 2.0
                containing_pairs = [
                    pair for pair in valid_pairs
                    if pair[0] - 6 <= image_center <= pair[1] + 6
                ]
                if containing_pairs:
                    chosen_pair = min(containing_pairs, key=lambda pair: pair[1] - pair[0])
                    case_used = "lane:initial_pair"
                else:
                    chosen_pair = min(
                        valid_pairs,
                        key=lambda pair: abs(((pair[0] + pair[1]) / 2.0) - image_center)
                    )
                    case_used = "lane:initial_nearest"

            left_x, right_x = chosen_pair
            target_x = int((left_x + right_x) / 2.0)
            self.last_lane_pair = (float(left_x), float(right_x))
            self.active_lane_pair = self.last_lane_pair
            self.left_x_history.append(float(left_x))
            self.right_x_history.append(float(right_x))
            self.last_left_x = float(left_x)
            self.last_right_x = float(right_x)
            self.frames_lost = 0

            # Nếu có vạch thứ 3, coi là center/debug.
            extras = [x for x in lane_xs if abs(x - left_x) > 8 and abs(x - right_x) > 8]
            if extras:
                center_x = float(min(extras, key=lambda x: abs(x - roi_w / 2.0)))
                self.center_x_history.append(center_x)
                self.last_center_x = center_x

        elif len(lane_xs) == 1:
            only_x = float(lane_xs[0])
            lane_w = self.default_lane_width_px
            if self.last_lane_pair is not None:
                lane_w = int(np.clip(self.last_lane_pair[1] - self.last_lane_pair[0],
                                     self.min_lane_width_px, self.max_lane_width_px))

            # With a dashed centre line, a single visible marking is common.
            # Infer which SIDE of the current corridor it belongs to from the
            # tracked pair, not from the image centre (which causes direction
            # flips while changing lane or on a curve).
            tracked_pair = self.active_lane_pair or self.last_lane_pair
            if tracked_pair is not None:
                is_left_boundary = abs(only_x - tracked_pair[0]) <= abs(only_x - tracked_pair[1])
            else:
                is_left_boundary = only_x < roi_w / 2.0

            if is_left_boundary:
                left_x = only_x
                right_x = min(roi_w - self.safety_margin_px, only_x + lane_w)
                target_x = int((left_x + right_x) / 2.0)
                case_used = "generic:left_only"
                self.left_x_history.append(left_x)
                self.last_left_x = left_x
            else:
                right_x = only_x
                left_x = max(self.safety_margin_px, only_x - lane_w)
                target_x = int((left_x + right_x) / 2.0)
                case_used = "generic:right_only"
                self.right_x_history.append(right_x)
                self.last_right_x = right_x
            self.last_lane_pair = (float(left_x), float(right_x))
            self.active_lane_pair = self.last_lane_pair
            self.frames_lost = 0

        else:
            self.frames_lost += 1
            if self.last_lane_pair is not None and self.frames_lost <= 8:
                left_x, right_x = self.last_lane_pair
                target_x = int((left_x + right_x) / 2.0)
                case_used = "generic:history_pair"
            elif self.last_target_x is not None:
                target_x = int(self.last_target_x)
                case_used = "generic:history_target"

        # Curve anticipation: estimate how the selected physical corridor moves
        # from the near field to the look-ahead row.  This preserves lane-pair
        # identity but starts steering before the car reaches a left/right bend.
        if left_x is not None and right_x is not None and line_geometries:
            def matching_near_x(boundary_x):
                matches = [(xn, length) for xt, xn, length in line_geometries
                           if abs(xt - boundary_x) <= 24]
                if not matches:
                    return None
                xs, weights = zip(*matches)
                return float(np.average(xs, weights=weights))

            near_left_x = matching_near_x(left_x)
            near_right_x = matching_near_x(right_x)
            shifts = []
            if near_left_x is not None:
                shifts.append(float(left_x) - near_left_x)
            if near_right_x is not None:
                shifts.append(float(right_x) - near_right_x)
            if shifts:
                # Median is robust when glare contributes a stray Hough segment
                # to one boundary.  A single genuine boundary still works on a
                # dashed-line gap or when the outside shoulder leaves the frame.
                heading_shift = float(np.median(shifts))
                preview = np.clip(self.heading_preview_gain * heading_shift,
                                  -self.max_heading_preview_px,
                                  self.max_heading_preview_px)
                target_x = int(round(target_x + preview))
                case_used += ":curve"

        target_x = max(self.safety_margin_px, min(roi_w - self.safety_margin_px, int(target_x)))
        # A real lane cannot teleport laterally between adjacent camera frames.
        # Clamp isolated glare/reflection spikes before they reach steering.
        if self.last_target_x is not None and self.frames_lost == 0:
            target_x = int(np.clip(target_x,
                                   self.last_target_x - self.max_target_step_px,
                                   self.last_target_x + self.max_target_step_px))
        self.last_target_x = target_x

        # Phân line để debug màu trái/phải/giữa.
        if left_x is not None or right_x is not None or center_x is not None:
            for line, slope, length in all_lines:
                x_at = self._weighted_x_at_y([(line, slope, length)], target_y)
                if x_at is None:
                    continue
                nearest = min(
                    [('L', left_x), ('R', right_x), ('C', center_x)],
                    key=lambda item: abs(x_at - item[1]) if item[1] is not None else 1e9
                )
                if nearest[1] is None or abs(x_at - nearest[1]) > 28:
                    continue
                if nearest[0] == 'L':
                    left_lines.append((line, slope, length))
                elif nearest[0] == 'R':
                    right_lines.append((line, slope, length))
                else:
                    center_lines.append((line, slope, length))

        return (target_x, left_x, right_x, center_x,
                left_lines, right_lines, center_lines, case_used)

    # =========================================================================
    # OBSTACLE DETECTION
    # =========================================================================

    def detect_obstacle(self, roi_img, boundary_mask, center_mask, road_mask=None):
        """Phát hiện vật cản trên đường bằng edge detection."""
        h, w = roi_img.shape[:2]
        hsv_obs = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        sat_obs = hsv_obs[:, :, 1]
        val_obs = hsv_obs[:, :, 2]
        if road_mask is not None and cv2.countNonZero(road_mask) > 50:
            road_pixels = road_mask > 0
            road_median_v = float(np.median(val_obs[road_pixels]))
            road_median_s = float(np.median(sat_obs[road_pixels]))
        else:
            road_median_v = float(np.median(val_obs))
            road_median_s = float(np.median(sat_obs))
        gray    = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)

        # Loại bỏ lane markings khỏi edge map
        lane_mask  = cv2.bitwise_or(boundary_mask, center_mask)
        lane_dil   = cv2.dilate(lane_mask, np.ones((7, 7), np.uint8), iterations=3)
        edges_obs  = cv2.bitwise_and(edges, cv2.bitwise_not(lane_dil))

        if road_mask is not None:
            # Chỉ tìm vật cản trong vùng road đang chạy, bỏ người/tường/sàn ngoài đường.
            obstacle_zone = cv2.erode(road_mask, np.ones((7, 7), np.uint8), iterations=1)
            obstacle_zone[:int(h * 0.12), :] = 0
            edges_obs = cv2.bitwise_and(edges_obs, obstacle_zone)

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

                # Floor seams, wrinkles and reflections create closed Canny
                # contours but have almost the same appearance as the road.
                # A physical obstacle must also form a photometrically distinct
                # region.  This keeps white cups/cones and dark/coloured objects
                # while rejecting the grey vinyl texture seen in test videos.
                contour_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                mean_v = float(cv2.mean(val_obs, mask=contour_mask)[0])
                mean_s = float(cv2.mean(sat_obs, mask=contour_mask)[0])
                value_contrast = abs(mean_v - road_median_v)
                saturation_contrast = mean_s - road_median_s
                if value_contrast < 24 and saturation_contrast < 28:
                    continue
                obstacles.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'center_x': cx_obs, 'center_y': cy_obs,
                    'area': bbox_area,
                    'contrast': max(value_contrast, saturation_contrast),
                })

        obstacles.sort(key=lambda o: o['area'], reverse=True)

        current = obstacles[0] if obstacles else None
        self.obstacle_history.append(current)
        stable_obstacle = None
        if current is not None:
            matching_history = [
                previous for previous in self.obstacle_history
                if previous is not None and
                abs(previous['center_x'] - current['center_x']) <= 26 and
                abs(previous['center_y'] - current['center_y']) <= 30
            ]
            if len(matching_history) >= 2:
                stable_obstacle = current

        return stable_obstacle, obstacles

    # =========================================================================
    # STEERING CALCULATION
    # =========================================================================

    def _pair_matches(self, pair_a, pair_b):
        """True when two lane-pair estimates refer to the same physical lane."""
        return (abs(pair_a[0] - pair_b[0]) <= self.lane_pair_match_px and
                abs(pair_a[1] - pair_b[1]) <= self.lane_pair_match_px)

    def _alternate_lane(self, current_pair):
        """Find the neighbouring lane that shares exactly one marking."""
        alternatives = []
        for pair in self._last_detected_pairs:
            if self._pair_matches(pair, current_pair):
                continue
            shared = (abs(pair[0] - current_pair[0]) <= self.lane_pair_match_px or
                      abs(pair[0] - current_pair[1]) <= self.lane_pair_match_px or
                      abs(pair[1] - current_pair[0]) <= self.lane_pair_match_px or
                      abs(pair[1] - current_pair[1]) <= self.lane_pair_match_px)
            if shared:
                alternatives.append(pair)
        if not alternatives:
            return None
        # A two-lane road normally has one option.  In a junction prefer the
        # option whose centre is closest to the current corridor.
        current_center = (current_pair[0] + current_pair[1]) / 2.0
        return min(alternatives, key=lambda p: abs((p[0] + p[1]) / 2.0 - current_center))

    def plan_lane_target(self, base_target_x, obstacle, img_width):
        """Keep obstacles scoped to the current lane and perform safe lane swaps.

        An object in the other lane is ignored.  An object in our lane triggers
        a switch only when a detected adjacent lane shares a boundary marking.
        After the object has been absent for several frames, the original lane
        is restored if it is still visible.
        """
        # During a lane-change manoeuvre, the local near-field detector can
        # still see the source lane for a few frames.  Keep the commanded
        # destination authoritative until the manoeuvre has completed.
        if self.lane_switch_state is not None:
            destination = self.lane_switch_state['destination']
            refreshed_destination = next(
                (pair for pair in self._last_detected_pairs
                 if self._pair_matches(pair, destination)), None
            )
            if refreshed_destination is not None:
                self.lane_switch_state['destination'] = refreshed_destination
                destination = refreshed_destination
            self.active_lane_pair = destination
            self.last_lane_pair = destination

        active = self.active_lane_pair or self.last_lane_pair
        obstacle_in_active_lane = False
        if obstacle is not None and active is not None:
            obstacle_in_active_lane = (active[0] + 4 <= obstacle['center_x'] <= active[1] - 4)

        if self.lane_switch_state is None:
            if obstacle_in_active_lane:
                destination = self._alternate_lane(active)
                if destination is not None:
                    self.lane_switch_state = {
                        'source': active,
                        'destination': destination,
                    }
                    self.active_lane_pair = destination
                    self.last_lane_pair = destination
                    self.last_target_x = int((destination[0] + destination[1]) / 2.0)
                    self.obstacle_clear_frames = 0
                    return self.last_target_x, 'avoid:switch_lane'

            # No neighbour to switch to: stay inside the current lane and make
            # only a small bounded avoidance correction.
            if obstacle_in_active_lane and active is not None:
                target = (active[0] + active[1]) / 2.0
                if obstacle['center_x'] <= target:
                    target += self.obstacle_avoidance_offset
                else:
                    target -= self.obstacle_avoidance_offset
                target = np.clip(target, active[0] + self.safety_margin_px,
                                 active[1] - self.safety_margin_px)
                return int(target), 'avoid:within_lane'
            return base_target_x, 'follow:active_lane'

        # Already travelling in the alternative lane.  Do not oscillate because
        # of one missed obstacle frame; return only after a sustained clear view.
        source = self.lane_switch_state['source']
        if obstacle_in_active_lane:
            self.obstacle_clear_frames = 0
        elif obstacle is None:
            self.obstacle_clear_frames += 1
        else:
            # An object in the original lane means it is still not safe to return.
            self.obstacle_clear_frames = 0

        target = int((self.active_lane_pair[0] + self.active_lane_pair[1]) / 2.0)
        if self.obstacle_clear_frames >= self.return_after_clear_frames:
            matching_source = next((p for p in self._last_detected_pairs
                                    if self._pair_matches(p, source)), None)
            if matching_source is not None:
                self.active_lane_pair = matching_source
                self.last_lane_pair = matching_source
                self.last_target_x = int((matching_source[0] + matching_source[1]) / 2.0)
                self.lane_switch_state = None
                self.obstacle_clear_frames = 0
                return self.last_target_x, 'avoid:return_lane'
        return target, 'avoid:hold_other_lane'

    def calculate_steering(self, target_x, img_width):
        """Tính steering [-1, 1] từ target_x; target is already lane-safe."""

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

        # 4. Detect lane markings. Giữ fallback màu cũ, nhưng thêm generic mask
        # cho lane vàng/xanh/cam/trắng hoặc lane sáng hơn road.
        color_boundary_mask = self.detect_boundary_mask(roi_enhanced, road_mask)
        generic_lane_mask = self.detect_generic_lane_mask(roi_enhanced, road_mask, trap_mask)
        boundary_mask = cv2.bitwise_or(color_boundary_mask, generic_lane_mask)

        # 5. Detect center divider (trắng/đứt đoạn) — bổ sung vào combined mask.
        center_mask = self.detect_center_mask(roi_enhanced, road_mask, color_boundary_mask)

        # 6. Calculate lane positions
        (target_x, left_x, right_x, center_x,
         left_lines, right_lines, center_lines,
         case_used) = self.calculate_target_x(boundary_mask, center_mask, h, w)

        # 7. Detect obstacles
        stable_obs, all_obs = self.detect_obstacle(roi_original, boundary_mask, center_mask, road_mask)

        # 8. Only react to obstacles inside the active corridor.  A lane change
        # uses the adjacent detected corridor, then holds it until the obstacle
        # has genuinely cleared instead of steering back and forth every frame.
        lane_target_x, lane_action = self.plan_lane_target(target_x, stable_obs, w)
        steering, adj_target_x = self.calculate_steering(lane_target_x, w)
        case_used = f"{case_used}|{lane_action}"

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
            'generic_lane': generic_lane_mask,
        }

        info = {
            'steering':           steering,
            'target_x':           adj_target_x,
            'left_x':             left_x,
            'right_x':            right_x,
            'center_x':           center_x,
            'case':               case_used,
            'lane_action':        lane_action,
            'active_lane_pair':   self.active_lane_pair,
            'lane_switching':     self.lane_switch_state is not None,
            # The caller can stop the motor if markings are gone for this long.
            # Three frames tolerates a dashed line while still failing safe
            # before the car can leave the board.
            'lane_confident':     self.active_lane_pair is not None and self.frames_lost <= 3,
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
            x1, y1, x2, y2 = self._line_coords(line)
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 80, 0), 2)   # Blue = left boundary

        for line, slope, length in right_lines:
            x1, y1, x2, y2 = self._line_coords(line)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 80, 255), 2)   # Red = right boundary

        for line, slope, length in center_lines:
            x1, y1, x2, y2 = self._line_coords(line)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 200), 2)  # Yellow = center

        # --- Lane position dots ---
        target_y_dot = int(h * 0.62)

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
                        default='/home/jetson/jetracer_official/notebook3/old_codes/road_following_A/apex',
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
        gt_x = int(parts[0])  # XYDataset stores the clicked apex in pixels.
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
