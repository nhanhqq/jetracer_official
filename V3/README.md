# V3 divider-first lane fusion

V3 keeps `waypoint_lane_fusion/` untouched and uses its fast waypoint model every frame as a fallback. The orange/red divider segmentation is tracked across the image and across frames with a quadratic lookahead fit for tight, long bends; when visible it exclusively owns lateral target x and heading. Dark-road segmentation defines the allowed corridor, white/outside pixels are a hard safety veto, and obstacle masks shift the target away while reducing throttle. Neural segmentation is decimated by `runtime.segmentation_interval` while the waypoint/control loop remains per-frame.

Dry-run replay:

```bash
PYTHONPATH=. python3 -m V3.runtime --source input.mp4 --output-video V3/results/replay.mp4 --log V3/results/replay.csv
```

Supervised CSI use:

```python
from V3.live import LiveRunner
runner = LiveRunner(arm=False)
runner.start()
runner.set_live(True)       # perception/control, motors remain stopped
runner.set_armed(True)      # enable only during supervised low-throttle test
```

Call `runner.set_armed(False)` and `runner.stop()` immediately after the test.

This is offline evidence only. Full CSI FPS and vehicle behavior still require supervised low-throttle validation.
