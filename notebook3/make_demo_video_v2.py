#!/usr/bin/env python3
"""
make_demo_video_v2.py — Tạo video demo thuật toán lane detection v2
====================================================================

Ghép toàn bộ dataset thành video MP4 và chạy thuật toán lane detection v2
để visualize kết quả detect lane, dải phân cách, vật cản.

Output:
- dataset_raw.mp4: Video gốc từ dataset (không xử lý)
- dataset_lane_v2.mp4: Video đã qua thuật toán lane detection v2
- dataset_debug_v2.mp4: Video debug chi tiết (ảnh gốc + masks + lane overlay)
"""

import cv2
import numpy as np
import os
import sys
import glob
import time

# Thêm thư mục hiện tại vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from lane_detection_v2 import LaneDetector


def create_debug_frame(original, result, mask_boundary, mask_center, steering, info):
    """
    Tạo debug frame gồm 4 ô:
    - Top-left: Original image
    - Top-right: Lane detection result
    - Bottom-left: Boundary mask (green)
    - Bottom-right: Center line mask (cyan)
    + Steering bar ở dưới cùng
    """
    h, w = original.shape[:2]

    # Resize masks to match ROI
    roi_top = info['roi_top']
    roi_h = h - roi_top

    # Tạo mask full size
    mask_b_full = np.zeros((h, w), dtype=np.uint8)
    mask_c_full = np.zeros((h, w), dtype=np.uint8)

    if mask_boundary.shape[0] == roi_h:
        mask_b_full[roi_top:h, :] = mask_boundary[:, :, 1] if len(mask_boundary.shape) == 3 else mask_boundary
    if mask_center.shape[0] == roi_h:
        mask_c_full[roi_top:h, :] = mask_center[:, :, 2] if len(mask_center.shape) == 3 else mask_center

    # Tạo colored masks
    mask_b_color = np.zeros((h, w, 3), dtype=np.uint8)
    mask_b_color[:, :, 1] = mask_b_full  # Green channel

    mask_c_color = np.zeros((h, w, 3), dtype=np.uint8)
    mask_c_color[:, :, 0] = mask_c_full  # Blue channel
    mask_c_color[:, :, 1] = mask_c_full  # Green channel (cyan)

    # Add labels
    cv2.putText(original, "Original", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(result, "Lane Detect", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(mask_b_color, "Boundary", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(mask_c_color, "Center Line", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Combine 2x2 grid
    top_row = np.hstack([original, result])
    bottom_row = np.hstack([mask_b_color, mask_c_color])
    grid = np.vstack([top_row, bottom_row])

    # Add steering bar at bottom
    bar_h = 30
    bar = np.zeros((bar_h, w * 2, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)

    # Draw steering indicator
    bar_center = w
    bar_pos = int(bar_center + steering * w * 0.8)
    cv2.rectangle(bar, (bar_center - 2, 5), (bar_center + 2, bar_h - 5), (100, 100, 100), -1)
    cv2.circle(bar, (bar_pos, bar_h // 2), 8, (0, 255, 0), -1)
    cv2.putText(bar, f"Steering: {steering:.3f}", (10, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Obstacle info
    if info['obstacle'] is not None:
        obs = info['obstacle']
        cv2.putText(bar, f"OBSTACLE @ x={obs['center_x']}", (w + 50, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    final = np.vstack([grid, bar])
    return final


def main():
    # --- Configuration ---
    dataset_dir = '/home/jetson/jetracer_official/notebook3/road_following_A/apex'
    output_dir = '/home/jetson/jetracer_official/notebook3'

    raw_video_path = os.path.join(output_dir, 'dataset_raw.mp4')
    lane_video_path = os.path.join(output_dir, 'dataset_lane_v2.mp4')
    debug_video_path = os.path.join(output_dir, 'dataset_debug_v2.mp4')

    fps = 10.0  # FPS cho video output

    # --- Load images ---
    images = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))
    print(f"Tìm thấy {len(images)} ảnh trong dataset")

    if not images:
        print("ERROR: Không tìm thấy ảnh nào!")
        return

    # Read first image to get dimensions
    first_img = cv2.imread(images[0])
    if first_img is None:
        print("ERROR: Không đọc được ảnh đầu tiên!")
        return
    h, w = first_img.shape[:2]
    print(f"Kích thước ảnh: {w}x{h}")

    # --- Initialize detector ---
    detector = LaneDetector(w, h)

    # --- Initialize video writers ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Raw video
    raw_writer = cv2.VideoWriter(raw_video_path, fourcc, fps, (w, h))

    # Lane detection video (same size as original)
    lane_writer = cv2.VideoWriter(lane_video_path, fourcc, fps, (w, h))

    # Debug video (2x2 grid + steering bar)
    debug_w = w * 2
    debug_h = h * 2 + 30  # 2x2 grid + steering bar
    debug_writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (debug_w, debug_h))

    # --- Process each frame ---
    print("\nĐang xử lý...")
    start_time = time.time()

    steering_values = []
    detection_stats = {
        'total': 0,
        'left_detected': 0,
        'right_detected': 0,
        'center_detected': 0,
        'obstacle_detected': 0,
        'both_lanes': 0,
    }

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Không đọc được {img_path}")
            continue

        # Resize nếu cần
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        # Write raw
        raw_writer.write(img)

        # Process with lane detection v2
        result_img, steering, info, mask_b, mask_c = detector.process_frame_with_masks(img)

        # Write lane detection result
        lane_writer.write(result_img)

        # Create and write debug frame
        debug_frame = create_debug_frame(img.copy(), result_img.copy(),
                                          mask_b, mask_c, steering, info)
        debug_writer.write(debug_frame)

        # Statistics
        steering_values.append(steering)
        detection_stats['total'] += 1
        if info['left_x'] is not None:
            detection_stats['left_detected'] += 1
        if info['right_x'] is not None:
            detection_stats['right_detected'] += 1
        if info['center_x'] is not None:
            detection_stats['center_detected'] += 1
        if info['obstacle'] is not None:
            detection_stats['obstacle_detected'] += 1
        if info['left_x'] is not None and info['right_x'] is not None:
            detection_stats['both_lanes'] += 1

        # Progress
        if (i + 1) % 10 == 0 or i == len(images) - 1:
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed
            print(f"  Frame {i + 1}/{len(images)} | "
                  f"Steering: {steering:+.3f} | "
                  f"L:{info['left_x'] is not None} "
                  f"R:{info['right_x'] is not None} "
                  f"C:{info['center_x'] is not None} | "
                  f"FPS: {fps_actual:.1f}")

    # --- Release writers ---
    raw_writer.release()
    lane_writer.release()
    debug_writer.release()

    elapsed = time.time() - start_time

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("        KẾT QUẢ XỬ LÝ VIDEO DEMO")
    print("=" * 60)
    print(f"\nThời gian xử lý: {elapsed:.1f}s ({detection_stats['total']} frames)")
    print(f"FPS trung bình: {detection_stats['total'] / elapsed:.1f}")
    print(f"\nVideo output:")
    print(f"  Raw:   {raw_video_path}")
    print(f"  Lane:  {lane_video_path}")
    print(f"  Debug: {debug_video_path}")

    total = detection_stats['total']
    print(f"\nThống kê Detection:")
    print(f"  Left lane detected:  {detection_stats['left_detected']}/{total} "
          f"({100 * detection_stats['left_detected'] / total:.1f}%)")
    print(f"  Right lane detected: {detection_stats['right_detected']}/{total} "
          f"({100 * detection_stats['right_detected'] / total:.1f}%)")
    print(f"  Both lanes:          {detection_stats['both_lanes']}/{total} "
          f"({100 * detection_stats['both_lanes'] / total:.1f}%)")
    print(f"  Center line:         {detection_stats['center_detected']}/{total} "
          f"({100 * detection_stats['center_detected'] / total:.1f}%)")
    print(f"  Obstacle:            {detection_stats['obstacle_detected']}/{total} "
          f"({100 * detection_stats['obstacle_detected'] / total:.1f}%)")

    if steering_values:
        print(f"\nSteering Statistics:")
        print(f"  Min: {min(steering_values):.3f}")
        print(f"  Max: {max(steering_values):.3f}")
        print(f"  Mean: {np.mean(steering_values):.3f}")
        print(f"  Std: {np.std(steering_values):.3f}")

    print("\nHoàn tất!")


if __name__ == '__main__':
    main()
