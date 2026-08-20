# Jetson Nano real-time setup

Jetson Nano is supported by JetPack 4 only. Do not install the JetPack 5/6 PyTorch wheels from newer Jetson instructions on this board. Check the board first:

```bash
cat /etc/nv_tegra_release
python3 --version
/usr/src/tensorrt/bin/trtexec --help >/dev/null && echo TensorRT_OK
```

The current Ultralytics Jetson guide lists Nano under JetPack 4 and provides a JetPack 4 Docker image. Use that route when possible:

```bash
sudo apt update
sudo apt install -y docker.io nvidia-container-runtime git
sudo systemctl restart docker
sudo usermod -aG docker "$USER"
```

Log out and back in, then from the repository root:

```bash
docker pull ultralytics/ultralytics:latest-jetson-jetpack4
docker run --rm -it --ipc=host --runtime=nvidia --privileged --network=host \
  -v "$PWD":/workspace/jetracer_official \
  -w /workspace/jetracer_official \
  ultralytics/ultralytics:latest-jetson-jetpack4 bash
```

Inside the container, install the camera/UI packages used by the notebook:

```bash
apt-get update && apt-get install -y python3-opencv v4l-utils
python3 -m pip install --upgrade pip
python3 -m pip install jetcam ipywidgets jupyterlab PyYAML
```

Before building, ensure the host user can access both Docker and Jetson GPU
devices. If a previous CUDA/TensorRT process has hung, reboot the Nano first;
restarting only the notebook does not reset the GPU context:

```bash
sudo usermod -aG docker "$USER"
sudo reboot
```

After logging back in, verify `groups` contains `docker` and run the runtime
verification script below. Do not build an engine while a CUDA matmul or
`trtexec` process is still stuck.

## Copy and build the semantic engine

Export ONNX on the workstation that has the trained checkpoint:

```bash
python3 yolo_lane_following/export_semantic_onnx.py
```

Copy these files to the same paths on Nano: `artifacts/track_yolo26n_sem_best.onnx`, `artifacts/track_yolo26n_sem_best.pt`, and the `yolo_lane_following/` package. Build the engine on Nano itself because TensorRT engines are tied to the target GPU, CUDA, and TensorRT versions:

```bash
chmod +x yolo_lane_following/build_semantic_engine_jetson_nano.sh
yolo_lane_following/build_semantic_engine_jetson_nano.sh
```

The script uses the lane-only checkpoint, static batch 1, FP16, `224x224`, and a 512 MiB workspace. It builds with the target's `trtexec`, deserializes the temporary engine, then runs a 10-second benchmark before replacing the old engine. A successful workstation ONNX export is not evidence that the Nano engine is valid until this command succeeds on the actual Nano.

The build has a five-minute build timeout and a one-minute benchmark timeout;
override them with `BUILD_TIMEOUT=600 BENCH_TIMEOUT=120` only if TensorRT timing
is known to be slow. A timeout leaves the previous engine untouched.

After the engine exists, `semantic_lane_live.ipynb` automatically selects:

```text
artifacts/track_yolo26n_sem_nano_fp16.engine
```

Otherwise it falls back to `track_yolo26n_sem_best.pt`, which is useful for diagnosis but is not the preferred real-time backend on Nano.

Run the runtime gate before opening the notebook. It separately proves that
TensorRT can execute the engine and that the Python 3.8/Ultralytics warmup does
not hang:

```bash
chmod +x yolo_lane_following/verify_semantic_runtime.sh
yolo_lane_following/verify_semantic_runtime.sh
```

If step 1 passes but step 3 times out, the engine itself is valid and the
problem is the Python/container CUDA stack; check that the container's
TensorRT, cuBLAS and cuDNN libraries come from the same JetPack release.

On JetPack R32.5.x, do not combine TensorRT 7.1.3 with a container that ships
another cuBLAS build. In particular, the warning
`TensorRT was linked against cuBLAS ... 10.2.3 but loaded ... 10.2.2` is a real
runtime mismatch, not harmless noise; fix the image/library versions before
rebuilding the engine.

## Run the notebook

Inside the container:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

Open `yolo_lane_following/semantic_lane_live.ipynb`, run cells in order, and keep `ARM MOTOR` unchecked during the first test. If CSI initialization fails, run this in another terminal on the Nano:

```bash
sudo systemctl restart nvargus-daemon
```

For a normal start, leave `COMPETITION` off and ARM automatically selects `live`;
driving starts as soon as a valid lane segmentation and safe controller state are available. For the
competition start, turn `COMPETITION` on, then ARM: inference starts but motor
output remains blocked until a bright green circle is detected for three
consecutive frames. Stop, disarming, or toggling the mode resets permission.

The CLI dry-run is useful before opening Jupyter:

```bash
python3 yolo_lane_following/run.py --source camera --display
```

Only after checking the overlay and CSV telemetry should motor output be enabled:

```bash
python3 yolo_lane_following/run.py --source camera --arm
```

Keep the car on a stand for the first armed test, use a physical emergency stop, and recalibrate `steering_gain`, `steering_offset`, `throttle_gain`, and `throttle_max` in `config.yaml` for the actual vehicle.

## Acceptance checks

Before driving on the table:

```bash
python3 -m unittest discover -s yolo_lane_following/tests -v
python3 yolo_lane_following/run.py --source notebook3/test/VIDEO.mp4 \
  --model yolo_lane_following/artifacts/track_yolo26n_sem_nano_fp16.engine \
  --output-video yolo_lane_following/artifacts/nano_semantic_preview.mp4
```

Verify that white shoulders are marked `forbidden`, the orange divider stays the target through turns, an obstacle causes `avoid` or a fail-safe stop, and the target returns to `divider` after the obstacle clears. Do not infer Nano real-time performance from the RTX workstation; retain the `trtexec` output from the target board.
