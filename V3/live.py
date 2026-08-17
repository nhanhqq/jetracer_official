"""CSI runner. Motors require both ``arm=True`` and ``set_live(True)``."""
import threading, time
from pathlib import Path
import cv2
from V2.config import load_config
from V3.pipeline import perceive, resize
from V3.divider import DividerTracker
from V3.fusion import fuse
from V3.control import Controller, State
from V3.waypoint import WaypointModel
from V3.perception import Perception

class LiveRunner:
    def __init__(self, config_path=None, arm=False):
        self.cfg=load_config(config_path or Path(__file__).with_name('config.yaml')); self.armed=bool(arm); self.live_enabled=False; self.running=False; self.lock=threading.Lock(); self.last=time.time(); self.last_output=time.time(); self.fps=0.
        self.perception=Perception(self.cfg); self.tracker=DividerTracker(self.cfg['geometry']); self.controller=Controller(self.cfg['control']); self.model=WaypointModel(self.cfg); self.camera=None; self.car=None; self.last_result=None
    def _command(self, frame):
        small=resize(frame,self.cfg); (road,outside,marking),geom,mode=perceive(small,self.perception,self.cfg); divider=self.tracker.update(marking,road,outside); now=time.time(); dt=now-self.last; self.last=now; target=fuse(self.model.predict(small),divider,geom,small.shape[1],small.shape[0],self.cfg['geometry']); return small,geom,target,self.controller.update(target,geom,dt),road,outside,marking,mode,dt
    def _callback(self, change):
        if not self.running or not self.live_enabled: return
        if not self.lock.acquire(False):
            if self.car is not None and time.time()-self.last_output > .15: self.car.stop(); self.car.center_steering()
            return
        try:
            result=self._command(change['new']); self.last_result=result; small,geom,target,cmd,_,_,_,_,dt=result; self.fps=.2/max(1e-3,dt)+.8*self.fps
            if self.car is not None and self.armed and cmd.state not in (State.STOP.value,State.REVERSE_REACQUIRE.value): self.car.set_steering(cmd.steering); self.car.set_throttle(cmd.throttle); self.last_output=time.time()
            elif self.car is not None: self.car.stop(); self.car.center_steering()
        except Exception:
            if self.car is not None: self.car.stop(); self.car.center_steering()
            self.live_enabled=False
        finally: self.lock.release()
    def start(self):
        from jetcam.csi_camera import CSICamera
        from notebook3.basic_motion import JetRacerController
        c=self.cfg['camera']; h=self.cfg['hardware']; self.camera=CSICamera(width=c['width'],height=c['height'],capture_fps=c.get('capture_fps',0)); self.car=JetRacerController(h['steering_gain'],h['steering_offset'],h['throttle_gain'],self.cfg['control']['throttle_max']); self.car.stop(); self.car.center_steering(); self.running=True; self.camera.observe(self._callback,names='value'); self.camera.running=True
    def set_live(self, value):
        self.live_enabled=bool(value) and self.running
        if not self.live_enabled and self.car: self.car.stop(); self.car.center_steering()
    def set_armed(self, value):
        self.armed=bool(value)
        if not self.armed and self.car: self.car.stop(); self.car.center_steering()
    def stop(self):
        self.running=False; self.live_enabled=False
        if self.car: self.car.stop(); self.car.center_steering()
        if self.camera: self.camera.unobserve_all(); self.camera.running=False
