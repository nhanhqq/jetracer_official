"""Latched intersection state machine. Uncertainty stops rather than guesses."""
from dataclasses import dataclass
from typing import Optional

from .decision import choose_branch
from .smart_city_perception import SmartCityScene


@dataclass
class DrivingIntent:
    mode: str
    maneuver: Optional[str]
    speed_limit: float
    state: str


class IntersectionFSM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = "FOLLOW"
        self.maneuver: Optional[str] = None
        self.clear_frames = 0
        self.stable_frames = 0
        self.turn_elapsed = 0.0

    def update(self, scene: SmartCityScene, dt: float) -> DrivingIntent:
        speed = self.cfg["speed"]
        cw = scene.crosswalk
        light = scene.traffic_light or cw.traffic_light
        light_state = light.state if light is not None else "UNKNOWN"
        if self.state == "FOLLOW":
            if cw.present:
                self.state = "APPROACH"
            return DrivingIntent("lane", None, speed["cruise"], self.state)
        if self.state == "APPROACH":
            if cw.y >= float(self.cfg["crosswalk"].get("approach_y", .55)) and light_state != "GREEN":
                self.state = "WAIT_LIGHT"
                return DrivingIntent("stop", None, 0.0, "WAIT_" + light_state)
            if cw.y >= float(self.cfg["crosswalk"].get("decision_y", .70)):
                self.state = "DECIDE"
            return DrivingIntent("lane", None, speed["approach"], self.state)
        if self.state == "WAIT_LIGHT":
            if light_state == "GREEN":
                self.state = "APPROACH"
                return DrivingIntent("lane", None, speed["approach"], self.state)
            return DrivingIntent("stop", None, 0.0, "WAIT_" + light_state)
        if self.state == "DECIDE":
            if light_state != "GREEN":
                self.state = "WAIT_LIGHT"
                return DrivingIntent("stop", None, 0.0, "WAIT_" + light_state)
            if not scene.branches.valid:
                self.state = "FAILSAFE"
                return DrivingIntent("stop", None, 0.0, "FAILSAFE:no_bev_calibration")
            self.maneuver = choose_branch(scene.branches.available, scene.sign.locked,
                                          self.cfg["intersection"]["preferred_order"])
            if self.maneuver is None:
                self.state = "FAILSAFE"
                return DrivingIntent("stop", None, 0.0, "FAILSAFE:no_legal_branch")
            self.state = "TURN"
            self.turn_elapsed = 0.0
        if self.state == "TURN":
            self.turn_elapsed += dt
            return DrivingIntent("turn", self.maneuver, speed[self.maneuver], "TURN_" + self.maneuver.upper())
        if self.state == "REACQUIRE":
            self.clear_frames = self.clear_frames + 1 if not cw.present else 0
            self.stable_frames = self.stable_frames + 1 if scene.lane.valid and scene.lane.confidence >= float(self.cfg["exit"]["lane_confidence"]) else 0
            if self.clear_frames >= int(self.cfg["exit"]["crosswalk_clear_frames"]) and self.stable_frames >= int(self.cfg["exit"]["lane_reacquire_frames"]):
                self.state = "FOLLOW"; self.maneuver = None
            return DrivingIntent("lane", None, speed["reacquire"], self.state)
        return DrivingIntent("stop", None, 0.0, self.state)

    def turn_complete(self) -> None:
        if self.state == "TURN":
            self.state = "REACQUIRE"; self.clear_frames = self.stable_frames = 0
