# Training

The deployable baseline segmentation checkpoint was trained in the existing
`yolo_lane_following` pipeline and is copied to `V2/models/`. `pseudo_label.py` is the
V2 label generator. It deliberately separates forbidden white from a bottom-center
connected dark-road component before labels are consumed by a segmentation trainer.

For a final run, manually review `V2/data/pseudo_labels`, split by drive/session (not
random adjacent frames), train a Nano segmentation model at 224px, export it with
`V2/deployment/export_onnx.py`, then compare PyTorch/ONNX/TensorRT outputs on the same
held-out frames. No new training run is claimed in this environment because the
Ultralytics and ONNX Python packages are unavailable.

