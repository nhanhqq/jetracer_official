from dataclasses import dataclass
import numpy as np


@dataclass
class IntersectionCommand:
    steering: float
    throttle: float
    state: str


class IntersectionController:
    """Conservative timed MVP. It is never enabled by default for real motors."""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def update(self, maneuver: str, elapsed: float) -> IntersectionCommand:
        if maneuver not in ("left", "right", "straight"):
            return IntersectionCommand(0.0, 0.0, "stop:no_legal_branch")
        duration = float(self.cfg.get("turn_duration_s", {}).get(maneuver, .0))
        if duration <= 0.0 or elapsed > duration:
            return IntersectionCommand(0.0, 0.0, "turn_complete")
        steering = float(self.cfg.get("turn_steering", {}).get(maneuver, 0.0))
        throttle = float(self.cfg.get("speed", {}).get(maneuver, 0.0))
        return IntersectionCommand(float(np.clip(steering, -1, 1)), max(0.0, throttle), "turn:" + maneuver)
