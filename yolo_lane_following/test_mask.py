import cv2, numpy as np, glob
from ultralytics import YOLO

img_path = glob.glob('semantic_dataset/images/train/*.jpg')[0]
model = YOLO('artifacts/track_yolo26n_sem_cube_best.pt', task='semantic')
results = model.predict(img_path)
mask = results[0].semantic_mask
arr = mask.data.cpu().numpy()
print(arr.shape, arr.dtype)
