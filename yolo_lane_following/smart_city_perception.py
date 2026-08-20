"""Composes existing semantic output with Smart City-only perception."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .crosswalk_detector import CrosswalkEstimate
from .geometry import LaneEstimate
from .intersection_geometry import BranchEstimate
from .sign_perception import SignEstimate
from .traffic_light import TrafficLightEstimate


@dataclass
class SmartCityScene:
    lane: LaneEstimate
    road_mask: np.ndarray
    divider_mask: np.ndarray
    forbidden_mask: np.ndarray
    crosswalk: CrosswalkEstimate
    branches: BranchEstimate
    sign: SignEstimate
    traffic_light: Optional[TrafficLightEstimate] = None
