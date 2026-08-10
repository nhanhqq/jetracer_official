from dataclasses import dataclass
from enum import Enum
import numpy as np

class State(Enum):
    NORMAL='NORMAL'; CAUTION='CAUTION'; RECOVERY_TURN='RECOVERY_TURN'; REVERSE_REACQUIRE='REVERSE_REACQUIRE'; STOP='STOP'
@dataclass
class Command:
    steering: float; throttle: float; state: str; warning: str
class Controller:
    def __init__(self, cfg): self.c=cfg; self.last_s=0.; self.last_t=0.; self.last_e=0.; self.lost=0
    def update(self, target, geom, dt):
        c=self.c; dt=float(np.clip(dt,.01,.2)); self.lost = self.lost+1 if not geom.valid else 0
        if geom.white_center >= c.get('white_hard_ratio',.32):
            # White is forbidden, but a valid visible road still gets a
            # bounded turn toward its corridor rather than a blind stop.
            desired_s=float(np.clip(1.45*(target.x-.5),-c['max_steering'],c['max_steering']))
            self.last_s=float(self.last_s+np.clip(desired_s-self.last_s,-c['max_steering_step'],c['max_steering_step']))
            self.last_t=float(max(0.,self.last_t-c['throttle_step_down']))
            if geom.valid: return Command(self.last_s,self.last_t,State.RECOVERY_TURN.value,'WHITE_FORBIDDEN_RECOVERY')
            return Command(self.last_s,-min(c.get('recovery_reverse',.07),c['throttle_max']),State.REVERSE_REACQUIRE.value,'WHITE_FORBIDDEN_REVERSE')
        if self.lost >= 4: return Command(0.,-min(c.get('recovery_reverse',.07),c['throttle_max']),State.REVERSE_REACQUIRE.value,'ROAD_REACQUIRE')
        e=target.x-.5; raw=c['kp']*e+c['kd']*(e-self.last_e)/dt+c['heading_gain']*target.heading; self.last_e=e
        raw=float(np.clip(raw,-c['max_steering'],c['max_steering'])); smooth=c['steering_alpha']*raw+(1-c['steering_alpha'])*self.last_s
        self.last_s=float(self.last_s+np.clip(smooth-self.last_s,-c['max_steering_step'],c['max_steering_step']))
        load=max(abs(self.last_s),abs(target.heading)); desired=c['throttle_cruise']*(1-.68*load)*(.45+.55*target.confidence)
        if 'white_' in target.source: desired *= .55
        if 'obstacle_avoid' in target.source: desired *= .60
        if target.confidence < .35: desired=min(desired,c['throttle_cruise']*.45)
        desired=float(np.clip(desired,c['throttle_min'],c['throttle_max'])); self.last_t=float(self.last_t+np.clip(desired-self.last_t,-c['throttle_step_down'],c['throttle_step_up']))
        state=State.CAUTION.value if target.confidence < .45 else State.NORMAL.value
        return Command(self.last_s,self.last_t,state,target.source)
