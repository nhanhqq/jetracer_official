import time
from .types import DriveState


class BehaviorStateMachine:
    """Priority: lane lost > obstacle/red light > turn sign > slowdown > normal."""
    def __init__(self, cfg):
        self.c, self.bad, self.good = cfg, 0, 0
        self.state = DriveState.LANE_LOST

    def update(self, waypoint, snapshot, now=None):
        now = time.time() if now is None else now
        if waypoint.confidence < self.c["confidence_stop"]: self.bad += 1; self.good = 0
        elif waypoint.confidence >= self.c["confidence_resume"]: self.good += 1; self.bad = 0
        if self.bad >= self.c["lane_lost_frames"]: self.state = DriveState.LANE_LOST
        if self.state == DriveState.LANE_LOST and self.good < self.c["lane_resume_frames"]: return self.state, 0.0
        fresh = snapshot and now-snapshot.timestamp <= self.c["detection_ttl"]
        labels = {d.label.lower() for d in snapshot.detections} if fresh else set()
        if labels & {"obstacle", "person", "car"}: self.state = DriveState.OBSTACLE
        elif labels & {"red_light", "red light", "stop"}: self.state = DriveState.RED_LIGHT
        elif labels & {"left", "turn_left", "re_trai"}: self.state = DriveState.TURN_LEFT
        elif labels & {"right", "turn_right", "re_phai"}: self.state = DriveState.TURN_RIGHT
        elif labels & {"slow", "slowdown"}: self.state = DriveState.SLOWDOWN
        else: self.state = DriveState.NORMAL
        bias = -self.c["turn_bias"] if self.state == DriveState.TURN_LEFT else self.c["turn_bias"] if self.state == DriveState.TURN_RIGHT else 0.0
        return self.state, bias
