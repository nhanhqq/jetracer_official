"""Closed-loop adaptive steering/throttle and perception-driven recovery."""
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np


class State(Enum):
    NORMAL='NORMAL'; CAUTION='CAUTION'; STOP='STOP'; RECOVERY_REVERSE='RECOVERY_REVERSE'; RECOVERY_TURN='RECOVERY_TURN'; REACQUIRE='REACQUIRE'


@dataclass
class Command:
    steering: float
    throttle: float
    state: str
    warning: str


class Controller:
    def __init__(self, cfg):
        self.c=cfg; self.state=State.CAUTION; self.last_steer=0.; self.last_throttle=0.; self.last_error=0.; self.lost=0; self.good=0; self.started=0

    def _ramp(self, old, new, up, down): return float(old + np.clip(new-old, -down, up))

    def update(self, target, geom, dt):
        c=self.c; dt=float(np.clip(dt,.01,.2)); self.lost = self.lost+1 if not geom.valid or geom.confidence < c['road_confidence_stop'] else 0
        boundary = geom.white_right - geom.white_left
        emergency = geom.white_center > c['white_center_stop'] or self.lost >= 3
        if self.state in (State.STOP, State.RECOVERY_REVERSE, State.RECOVERY_TURN, State.REACQUIRE):
            if self.started == 0: self.started=time.time()
        if emergency and self.state in (State.NORMAL, State.CAUTION): self.state=State.STOP; self.started=time.time(); self.good=0
        elif self.state == State.STOP and geom.valid and geom.confidence > c['road_confidence_caution'] and not geom.white_center > c['white_center_stop']:
            self.state=State.RECOVERY_REVERSE; self.started=time.time()
        elif self.state == State.RECOVERY_REVERSE and time.time()-self.started > .35: self.state=State.RECOVERY_TURN; self.started=time.time()
        elif self.state == State.RECOVERY_TURN and geom.valid and geom.confidence > .45: self.state=State.REACQUIRE; self.good=0
        elif self.state == State.REACQUIRE:
            self.good=self.good+1 if geom.valid and geom.confidence>.55 and geom.white_center<.15 else 0
            if self.good >= int(c['reacquire_frames']): self.state=State.CAUTION; self.started=0
        elif self.state == State.NORMAL and geom.confidence < c['road_confidence_caution']: self.state=State.CAUTION
        elif self.state == State.CAUTION and geom.confidence > .62 and self.lost == 0: self.state=State.NORMAL
        if self.state == State.STOP: return Command(self._ramp(self.last_steer,0,c['max_steering_step'],c['max_steering_step']),0,self.state.value,'ROAD_LOST_OR_WHITE_CENTER')
        if self.state == State.RECOVERY_REVERSE: return Command(0,-min(c['recovery_reverse'], c['throttle_max']),self.state.value,'PERCEPTION_RECOVERY')
        if self.state == State.RECOVERY_TURN:
            # White side is directional; turn toward the visible road.
            s = min(c['recovery_turn'], c['max_steering']) if geom.white_left > geom.white_right else -min(c['recovery_turn'], c['max_steering'])
            return Command(s,0,self.state.value,'RECOVERY_TURN_TO_ROAD')
        if self.state == State.REACQUIRE: return Command(self._ramp(self.last_steer,0,c['max_steering_step'],c['max_steering_step']),0,self.state.value,'REACQUIRE')
        error=float(target.x-.5); derivative=(error-self.last_error)/dt; self.last_error=error
        raw=c['kp']*error+c['kd']*derivative+c['heading_gain']*geom.heading+c['safety_gain']*boundary-c['obstacle_gain']*geom.obstacle_offset
        raw=float(np.clip(raw,-c['max_steering'],c['max_steering']))
        smooth=c['steering_alpha']*raw+(1-c['steering_alpha'])*self.last_steer
        self.last_steer=self._ramp(self.last_steer,smooth,c['max_steering_step'],c['max_steering_step'])
        load=max(abs(self.last_steer),geom.curvature); conf=max(0,min(1,target.confidence));
        desired=c['throttle_cruise']*(1-.72*load)*(.45+.55*conf)*(1-min(1,geom.white_left+geom.white_right)*2)
        desired *= max(.25, 1.0 - min(1.0, geom.obstacle * 4.0))
        desired=float(np.clip(desired, c['throttle_min'], c['throttle_max']))
        if self.state == State.CAUTION: desired=min(desired,c['throttle_cruise']*.55)
        self.last_throttle=self._ramp(self.last_throttle,desired,c['throttle_step_up'],c['throttle_step_down'])
        return Command(self.last_steer,self.last_throttle,self.state.value,geom.warning)
