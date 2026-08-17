import json

with open("notebook3/cv_road_following_live.ipynb", "r") as f:
    notebook = json.load(f)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "def process_cv_lane(img):" in "".join(cell["source"]):
        source = "".join(cell["source"])
        
        # We need to insert the obstacle detection logic and avoidance logic.
        new_source = source.replace(
            "    # 4. Tìm viền bằng Canny trên vùng đã Mask\n",
            """    # 4. Tìm viền bằng Canny trên vùng đã Mask
    edges = cv2.Canny(mask, 50, 150)
    
    # --- THÊM: TÌM VẬT CẢN (OBSTACLE DETECTION) ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_all = cv2.Canny(blurred, 50, 150)
    
    # Loại bỏ các đường biên của vạch kẻ đường (để không nhận vạch là vật cản)
    kernel_dilate = np.ones((5,5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel_dilate, iterations=2)
    edges_obs = cv2.bitwise_and(edges_all, edges_all, mask=cv2.bitwise_not(mask_dilated))
    
    # Tìm các đường viền của vật cản
    contours, _ = cv2.findContours(edges_obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacle_detected = False
    obs_center_x = -1
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h # Diện tích bounding box
            if 200 < area < 8000: # Lọc nhiễu, kích thước vật cản hợp lý
                obs_center_x = x + w // 2
                obs_center_y = y + h // 2
                cv2.rectangle(line_image, (x, y + roi_top), (x+w, y+h + roi_top), (0, 255, 255), 2)
                cv2.putText(line_image, "Vat can", (x, y + roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                obstacle_detected = True
                break
    # ---------------------------------------------
    
    # Bỏ dòng tìm edges cũ vì đã gộp ở trên
"""
        )
        
        new_source = new_source.replace(
            "    edges = cv2.Canny(mask, 50, 150)\n    \n    # 5. Phát hiện đường",
            "    # 5. Phát hiện đường"
        )
        
        new_source = new_source.replace(
            "    cv2.circle(line_image, (center_x, center_y), 8, (0, 255, 0), -1)\n",
            """    # --- THÊM: LOGIC NÉ VẬT CẢN ---
    view_center = width // 2
    if obstacle_detected:
        if obs_center_x < view_center:
            # Vật cản bên trái -> Dịch tâm ảo sang phải để né
            center_x += 70
            cv2.putText(line_image, "Tranh Trai -> Re Phai", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Vật cản bên phải -> Dịch tâm ảo sang trái để né
            center_x -= 70
            cv2.putText(line_image, "Tranh Phai -> Re Trai", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # ------------------------------

    cv2.circle(line_image, (center_x, center_y), 8, (0, 255, 0), -1)\n"""
        )
        
        # update cell source
        cell["source"] = [line + "\n" for line in new_source.split("\n")[:-1]]
        # Because we split by \n, the last element is empty if new_source ends with \n, or it doesn't matter much.
        # Better way to split and keep \n:
        import io
        cell["source"] = [line for line in io.StringIO(new_source)]

with open("notebook3/cv_road_following_live.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Done modifying notebook")
