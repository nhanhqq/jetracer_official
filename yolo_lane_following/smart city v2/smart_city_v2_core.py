"""Core logic shared by the Smart City V2 calibration and route notebooks.

The fixed throttle is deliberately a source-code constant.  Both notebooks use
MotionDriver, so calibration and route execution exercise the same motor path.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


FIXED_THROTTLE = 0.50
START_NODE = 1

POSITIONS = {
    7: (0, 0), 4: (1, 0), 1: (2, 0),
    8: (0, 1), 5: (1, 1), 2: (2, 1),
    9: (0, 2), 6: (1, 2), 3: (2, 2),
}

EDGES = [
    (7, 4), (4, 1), (8, 5), (5, 2), (9, 6), (6, 3),
    (7, 8), (8, 9), (4, 5), (5, 6), (1, 2), (2, 3),
]
EDGE_SET = {frozenset(edge) for edge in EDGES}


def edge_key(a: int, b: int) -> str:
    """Stable undirected key matching the official edge list."""
    for left, right in EDGES:
        if frozenset((a, b)) == frozenset((left, right)):
            return f"{left}-{right}"
    raise ValueError(f"Nodes {a} and {b} are not neighbors")


def turn_key(a: int, b: int, c: int) -> str:
    return f"{a}-{b}-{c}"


def are_neighbors(a: int, b: int) -> bool:
    return a != b and frozenset((a, b)) in EDGE_SET


def classify_turn(a: int, b: int, c: int) -> str:
    """Return LEFT_90/RIGHT_90/STRAIGHT using image-grid vectors.

    Grid y increases downward, so a negative cross product is a physical left
    turn and a positive cross product is a physical right turn.
    """
    if not are_neighbors(a, b) or not are_neighbors(b, c):
        raise ValueError(f"Invalid path fragment {a} -> {b} -> {c}")
    ax, ay = POSITIONS[a]; bx, by = POSITIONS[b]; cx, cy = POSITIONS[c]
    incoming = (bx - ax, by - ay)
    outgoing = (cx - bx, cy - by)
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    if cross == 0 and dot > 0:
        return "STRAIGHT"
    if cross < 0:
        return "LEFT_90"
    if cross > 0:
        return "RIGHT_90"
    raise ValueError("Immediate U-turn is not supported; choose another node")


def default_config() -> dict:
    return {
        "schema_version": 2,
        "motion": {
            "throttle": FIXED_THROTTLE,
            "steering_center": 0.0,
            "steering_gain": -0.65,
            "steering_offset": 0.0,
            "throttle_gain": 0.8,
        },
        "turns": {
            "left_90": {
                "pre_steer_time": 0.15, "steering": -1.0,
                "time": 0.0, "center_settle_time": 0.15,
            },
            "right_90": {
                "pre_steer_time": 0.15, "steering": 1.0,
                "time": 0.0, "center_settle_time": 0.15,
            },
        },
        "edges": {edge_key(a, b): 0.0 for a, b in EDGES},
        "tuning": {"edges": {}, "turns": {}},
        "traffic_light": {
            "roi_bottom": 0.50,
            "min_area": 25.0,
            "max_area": 2500.0,
            "min_circularity": 0.68,
            "confirm_frames": 3,
        },
    }


def validate_config(config: dict, require_measurements: bool = True) -> None:
    throttle = float(config.get("motion", {}).get("throttle", -1.0))
    if not math.isclose(throttle, FIXED_THROTTLE, abs_tol=1e-9):
        raise ValueError(
            f"motion.throttle must remain locked at {FIXED_THROTTLE:.2f}; got {throttle}"
        )
    center = float(config.get("motion", {}).get("steering_center", 99.0))
    if not math.isclose(center, 0.0, abs_tol=1e-9):
        raise ValueError(f"motion.steering_center must remain 0.0; got {center}")
    for name in ("left_90", "right_90"):
        turn = config.get("turns", {}).get(name, {})
        steering, duration = float(turn.get("steering", 99)), float(turn.get("time", 0))
        duration_invalid = duration <= 0 if require_measurements else duration < 0
        if not -1.0 <= steering <= 1.0 or duration_invalid:
            raise ValueError(f"Invalid {name} steering/time")
        for phase_name in ("pre_steer_time", "center_settle_time"):
            phase_time = float(turn.get(phase_name, 0.15))
            if not 0.0 <= phase_time <= 2.0:
                raise ValueError(f"Invalid {name}.{phase_name}")
    missing = [edge_key(a, b) for a, b in EDGES if float(config.get("edges", {}).get(edge_key(a, b), 0)) <= 0]
    if require_measurements and missing:
        raise ValueError("Missing positive edge measurements: " + ", ".join(missing))


def load_config(path: Path, require_measurements: bool = True) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config, require_measurements=require_measurements)
    return config


def save_config(config: dict, path: Path, require_measurements: bool = True) -> None:
    validate_config(config, require_measurements=require_measurements)
    Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class Command:
    kind: str
    duration: float = 0.0
    steering: float = 0.0
    start: Optional[int] = None
    end: Optional[int] = None
    node: Optional[int] = None
    turn: Optional[str] = None
    tune_key: Optional[str] = None
    pre_steer_time: float = 0.0
    center_settle_time: float = 0.0
    center_steering: float = 0.0


def compile_route(route: Sequence[int], config: dict) -> List[Command]:
    validate_config(config, require_measurements=True)
    if not route or route[0] != START_NODE:
        raise ValueError("Every route must start at fixed Node 1")
    if len(route) < 2:
        raise ValueError("Click Node 4 or Node 2 to establish the initial heading")
    for a, b in zip(route, route[1:]):
        if not are_neighbors(a, b):
            raise ValueError(f"Invalid edge {a} -> {b}")

    tuning = config.get("tuning", {})
    edge_tuning = tuning.get("edges", {})
    turn_tuning = tuning.get("turns", {})
    commands: List[Command] = []
    center = float(config["motion"]["steering_center"])

    for index, (a, b) in enumerate(zip(route, route[1:])):
        if index > 0:
            previous = route[index - 1]
            maneuver = classify_turn(previous, a, b)
            if maneuver != "STRAIGHT":
                base = config["turns"][maneuver.lower()]
                key = turn_key(previous, a, b)
                override = turn_tuning.get(key, {})
                commands.append(Command(
                    kind="TURN", node=a, turn=maneuver, tune_key=key,
                    steering=float(override.get("steering", base["steering"])),
                    duration=float(override.get("time", base["time"])),
                    pre_steer_time=float(override.get("pre_steer_time", base.get("pre_steer_time", 0.15))),
                    center_settle_time=float(override.get("center_settle_time", base.get("center_settle_time", 0.15))),
                    center_steering=0.0,
                ))

        key = edge_key(a, b)
        base_duration = float(config["edges"][key])
        override = edge_tuning.get(f"{a}-{b}", {})
        commands.append(Command(
            kind="STRAIGHT", start=a, end=b, tune_key=f"{a}-{b}",
            steering=center,
            duration=float(override.get("time", base_duration)),
        ))
    commands.append(Command(kind="STOP"))
    return commands


def apply_command_tuning(config: dict, commands: Iterable[Command]) -> dict:
    """Persist edited command values as route-specific overrides."""
    config.setdefault("tuning", {}).setdefault("edges", {})
    config.setdefault("tuning", {}).setdefault("turns", {})
    for command in commands:
        if command.kind == "STRAIGHT":
            config["tuning"]["edges"][command.tune_key] = {"time": round(float(command.duration), 4)}
        elif command.kind == "TURN":
            config["tuning"]["turns"][command.tune_key] = {
                "steering": round(float(command.steering), 4),
                "time": round(float(command.duration), 4),
                "pre_steer_time": round(float(command.pre_steer_time), 4),
                "center_settle_time": round(float(command.center_settle_time), 4),
            }
    return config


class MotionDriver:
    """Single motor-command implementation shared by calibration and RUN."""
    def __init__(self, car, armed: Callable[[], bool]):
        self.car = car
        self.armed = armed
        self.lock = threading.RLock()

    def stop(self, center: bool = False) -> None:
        with self.lock:
            self.car.throttle = 0.0
            if center:
                self.car.steering = 0.0

    def start(self, steering: float) -> float:
        if not self.armed():
            self.stop(center=True)
            raise RuntimeError("ARM MOTOR is off")
        steering = float(np.clip(steering, -1.0, 1.0))
        with self.lock:
            self.car.steering = steering
            started = time.perf_counter()
            self.car.throttle = FIXED_THROTTLE
        return started

    def prepare_steering(self, steering: float) -> None:
        """Phase 1/3: move steering servo while throttle is guaranteed zero."""
        if not self.armed():
            self.stop(center=True)
            raise RuntimeError("ARM MOTOR is off")
        with self.lock:
            self.car.throttle = 0.0
            self.car.steering = float(np.clip(steering, -1.0, 1.0))


@dataclass
class LightEstimate:
    state: str
    red_area: float
    green_area: float
    red_circularity: float
    green_circularity: float
    roi_height: int


class TrafficLightDetector:
    """Upper-image HSV + contour area + circularity detector."""
    def __init__(self, config: dict):
        self.config = config
        self.history = deque(maxlen=max(1, int(config.get("confirm_frames", 3))))

    @staticmethod
    def _best_circle(mask: np.ndarray, min_area: float, max_area: float, min_circularity: float) -> Tuple[float, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = best_circularity = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            if min_area <= area <= max_area and circularity >= min_circularity and area > best_area:
                best_area, best_circularity = area, circularity
        return best_area, best_circularity

    def update(self, frame: np.ndarray) -> LightEstimate:
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a BGR camera frame")
        roi_h = max(1, int(frame.shape[0] * float(self.config.get("roi_bottom", 0.50))))
        hsv = cv2.cvtColor(frame[:roi_h], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([12, 255, 255]))
        red |= cv2.inRange(hsv, np.array([168, 100, 100]), np.array([180, 255, 255]))
        green = cv2.inRange(hsv, np.array([38, 80, 80]), np.array([92, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
        minimum = float(self.config.get("min_area", 25.0))
        maximum = float(self.config.get("max_area", 2500.0))
        circularity = float(self.config.get("min_circularity", 0.68))
        red_area, red_circle = self._best_circle(red, minimum, maximum, circularity)
        green_area, green_circle = self._best_circle(green, minimum, maximum, circularity)
        raw = "RED" if red_area > 0 and red_area >= green_area else ("GREEN" if green_area > 0 else "NONE")
        self.history.append(raw)
        confirmed = raw if len(self.history) == self.history.maxlen and all(item == raw for item in self.history) and raw != "NONE" else "NONE"
        return LightEstimate(confirmed, red_area, green_area, red_circle, green_circle, roi_h)


class RouteExecutor:
    """Timed command runner with latched RED pause and GREEN-only resume."""
    def __init__(self, driver: MotionDriver, light_state: Callable[[], str], on_update: Callable[[dict], None]):
        self.driver = driver
        self.light_state = light_state
        self.on_update = on_update
        self.cancel = threading.Event()
        self.red_latched = False

    def emergency_stop(self) -> None:
        self.cancel.set()
        self.driver.stop(center=True)

    def _observe_light(self) -> str:
        signal = self.light_state()
        if signal == "RED":
            self.red_latched = True
        return signal

    def _timed(self, command: Command, index: int) -> None:
        completed = 0.0
        signal = self._observe_light()
        paused = self.red_latched
        if paused:
            self.driver.stop(center=False)
            last = time.perf_counter()
        else:
            last = self.driver.start(command.steering)
        while completed < command.duration:
            if self.cancel.is_set():
                raise RuntimeError("Emergency stop")
            now = time.perf_counter()
            signal = self._observe_light()
            if not paused and signal == "RED":
                completed += max(0.0, now - last)
                self.driver.stop(center=False)
                paused = True
            elif paused and signal == "GREEN":
                self.red_latched = False
                last = self.driver.start(command.steering)
                paused = False
            elif not paused:
                completed += max(0.0, now - last)
                last = now
            self.on_update({
                "state": "PAUSED_RED" if paused else "RUN_COMMAND",
                "index": index,
                "command": asdict(command),
                "completed": min(completed, command.duration),
                "remaining": max(0.0, command.duration - completed),
                "light": signal,
            })
            time.sleep(0.01)
        self.driver.stop(center=False)

    def _wait_stationary_phase(self, duration: float, state: str, index: int, command: Command) -> None:
        started = time.perf_counter()
        while time.perf_counter() - started < duration:
            if self.cancel.is_set():
                raise RuntimeError("Emergency stop")
            signal = self._observe_light()
            self.on_update({
                "state": state, "index": index, "command": asdict(command),
                "completed": time.perf_counter() - started,
                "remaining": max(0.0, duration - (time.perf_counter() - started)),
                "light": signal,
            })
            time.sleep(0.01)

    def _turn_three_phase(self, command: Command, index: int) -> None:
        # Phase 1: wheels reach the calibrated turn angle with throttle=0.
        self.driver.prepare_steering(command.steering)
        self._wait_stationary_phase(command.pre_steer_time, "TURN_PRE_STEER", index, command)
        # Phase 2: powered arc; RED pauses active time and only GREEN resumes it.
        self._timed(command, index)
        # Phase 3: throttle=0 and wheels settle at logical center before next edge.
        self.driver.prepare_steering(command.center_steering)
        self._wait_stationary_phase(command.center_settle_time, "TURN_CENTER", index, command)

    def run(self, commands: Sequence[Command]) -> None:
        self.cancel.clear()
        try:
            for index, command in enumerate(commands, 1):
                if command.kind == "STOP":
                    break
                if command.kind == "TURN":
                    self._turn_three_phase(command, index)
                else:
                    self._timed(command, index)
            self.driver.stop(center=True)
            self.on_update({"state": "FINISH", "index": len(commands), "remaining": 0.0})
        except Exception as exc:
            self.driver.stop(center=True)
            self.on_update({"state": "STOPPED", "error": str(exc)})
