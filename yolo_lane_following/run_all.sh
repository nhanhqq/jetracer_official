#!/bin/bash
set -e

echo "Starting training..."
python3 train_semantic.py --model artifacts/track_yolo26n_sem_best.pt --epochs 50

echo "Copying to .pth..."
cp artifacts/track_yolo26n_sem_best.pt artifacts/track_yolo26n_sem_best.pth

echo "Exporting to ONNX..."
python3 export_semantic_onnx.py --model artifacts/track_yolo26n_sem_best.pt

echo "All done!"
