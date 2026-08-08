from dataclasses import dataclass, field
from enum import Enum
from typing import List


class DriveState(str, Enum):
    NORMAL = "NORMAL"
    SLOWDOWN = "SLOWDOWN"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    OBSTACLE = "OBSTACLE"
    RED_LIGHT = "RED_LIGHT"
    LANE_LOST = "LANE_LOST"


@dataclass
class Waypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Detection:
    label: str
    confidence: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass
class DetectionSnapshot:
    detections: List[Detection] = field(default_factory=list)
    timestamp: float = 0.0
    fps: float = 0.0


@dataclass
class DriveCommand:
    steering_raw: float
    steering: float
    throttle: float
    state: DriveState
