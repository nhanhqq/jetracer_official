from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Optional


@dataclass
class TrafficDecision:
    state: str
    route: str
    throttle_scale: float
    reason: str


class SmartCityPolicy:
    """Temporal traffic policy with red-light priority over all drive modes."""

    def __init__(self, red_confirm_frames: int = 2, green_confirm_frames: int = 2,
                 sign_confirm_frames: int = 2, default_route: str = "STRAIGHT",
                 forbidden_direction: Optional[str] = None,
                 forbidden_random_seed: Optional[int] = 2608):
        self.red_confirm = max(1, int(red_confirm_frames))
        self.green_confirm = max(1, int(green_confirm_frames))
        self.sign_confirm = max(1, int(sign_confirm_frames))
        self.default_route = default_route.upper()
        self.forbidden_direction = forbidden_direction.upper() if forbidden_direction else None
        self.random = random.Random(forbidden_random_seed)
        self.red_count = 0
        self.green_count = 0
        self.sign_label = None
        self.sign_count = 0
        self.signal = "UNKNOWN"
        self.pending_route = self.default_route

    @staticmethod
    def _best(detections: Iterable[dict], prefix: str):
        aliases = {
            "red_light": ("red_light", "den_do"),
            "green_light": ("green_light", "den_xanh"),
        }.get(prefix, (prefix,))
        selected = [d for d in detections if str(d.get("label", "")).lower().startswith(aliases)]
        return max(selected, key=lambda d: float(d.get("confidence", 0.0)), default=None)

    def update(self, detections: Iterable[dict], lane_valid: bool,
               forbidden_front: float = 0.0) -> TrafficDecision:
        detections = list(detections)
        red = self._best(detections, "red_light")
        green = self._best(detections, "green_light")
        if red:
            self.red_count += 1
            self.green_count = 0
        elif green:
            self.green_count += 1
            self.red_count = 0
        else:
            self.red_count = max(0, self.red_count - 1)
            self.green_count = max(0, self.green_count - 1)
        if self.red_count >= self.red_confirm:
            self.signal = "RED"
        elif self.green_count >= self.green_confirm:
            self.signal = "GREEN"

        # Signs are deliberately lower priority than lights, but forbidden
        # signs have priority over route signs when both are visible.
        sign = self._priority_sign(detections)
        if sign:
            label = str(sign["label"]).lower()
            if label == self.sign_label:
                self.sign_count += 1
            else:
                self.sign_label, self.sign_count = label, 1
            if self.sign_count >= self.sign_confirm:
                if ("left" in label or "re_trai" in label) and "forbidden" not in label:
                    self.pending_route = "LEFT"
                elif ("right" in label or "re_phai" in label) and "forbidden" not in label:
                    self.pending_route = "RIGHT"
                elif ("straight" in label or "di_thang" in label) and "forbidden" not in label:
                    self.pending_route = "STRAIGHT"
                elif label in ("forbidden_sign", "bien_cam"):
                    # The requested competition behavior is to choose one of
                    # the two available turns when a generic forbidden sign
                    # blocks the nominal route. The RNG is seeded for replay.
                    self.pending_route = self.random.choice(("LEFT", "RIGHT"))
                elif label.startswith("forbidden_"):
                    blocked = label.split("_", 1)[1].upper()
                    self.pending_route = self._alternate(blocked)

        if self.signal == "RED":
            return TrafficDecision("STOP_RED", self.pending_route, 0.0, "red_light")
        if not lane_valid:
            return TrafficDecision("STOP_LANE_LOST", self.pending_route, 0.0, "lane_lost")
        if self.pending_route == "UNRESOLVED":
            return TrafficDecision("STOP_FORBIDDEN_UNRESOLVED", self.pending_route, 0.0,
                                   "forbidden_direction_required")
        return TrafficDecision("DRIVE", self.pending_route, 1.0, "green_or_no_signal")

    @staticmethod
    def _is_sign(label: str) -> bool:
        return ("_sign" in label or label in {"bien_cam", "di_thang", "re_phai", "re_trai"}
                or label.startswith("forbidden"))

    @classmethod
    def _priority_sign(cls, detections: Iterable[dict]):
        signs = [d for d in detections if cls._is_sign(str(d.get("label", "")).lower())]
        if not signs:
            return None
        forbidden = [d for d in signs
                     if ("forbidden" in str(d.get("label", "")).lower()
                         or str(d.get("label", "")).lower() == "bien_cam")]
        pool = forbidden or signs
        return max(pool, key=lambda d: float(d.get("confidence", 0.0)))

    @staticmethod
    def _alternate(forbidden: str) -> str:
        return {"LEFT": "RIGHT", "RIGHT": "STRAIGHT", "STRAIGHT": "RIGHT"}.get(forbidden, "STRAIGHT")
