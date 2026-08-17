#!/bin/bash
echo "Bắt đầu thiết lập môi trường Jetson AI Racer Challenge..."

# Tạo môi trường ảo với --system-site-packages để dùng chung OpenCV, TensorRT, CUDA của Jetpack
echo "Đang tạo virtual environment 'jetson_ai_env'..."
python3 -m venv --system-site-packages jetson_ai_env

# Kích hoạt môi trường
source jetson_ai_env/bin/activate

# Cập nhật pip cơ bản
pip install --upgrade pip

# Cài đặt các thư viện từ requirements.txt
echo "Đang cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt

echo "========================================================"
echo "Thiết lập hoàn tất!"
echo "Để sử dụng môi trường, hãy chạy lệnh sau mỗi lần mở terminal:"
echo "source jetson_ai_env/bin/activate"
echo "Để thoát môi trường, gõ: deactivate"
echo "========================================================"
