"""Jetson CSI live runner used by the V2 notebook.

The motor remains stopped until ``armed`` and ``running`` are both true. The
waypoint model is loaded from the V2 TensorRT artifact; road perception has a
deterministic CV fallback because old Nano TensorRT cannot parse the inherited
YOLO26 segmentation graph.
"""
import csv, threading, time
from pathlib import Path
import cv2, numpy as np
from .config import load_config
from .segmentation import Segmenter
from .geometry import estimate_geometry
from .fusion import fuse
from .control import Controller, State

class LiveRunner:
    def __init__(self, config_path=None, arm=False):
        self.cfg=load_config(config_path); self.armed=bool(arm); self.running=False; self.camera=None; self.car=None; self.model=None; self.mode='geometry-fallback'; self.lock=threading.Lock(); self.last=time.time(); self.fps=0.; self.widgets=None
        self.controller=Controller(self.cfg['control'])
        self.segmenter=Segmenter(self.cfg)
        self._load_waypoint()

    def _load_waypoint(self):
        root=Path(self.cfg['_root']); engine=root/self.cfg['models'].get('waypoint_engine','models/waypoint_baseline_fp16.engine'); onnx=root/self.cfg['models']['waypoint']
        try:
            from waypoint_lane_fusion.lane_model import TensorRTWaypointModel, OnnxWaypointModel
            if engine.exists(): self.model=TensorRTWaypointModel(engine); self.mode='waypoint-tensorrt-fp16'
            else: self.model=OnnxWaypointModel(onnx); self.mode='waypoint-onnx'
        except Exception as exc:
            self.model=None; self.mode='geometry-fallback:%s' % type(exc).__name__
            self.model_error=str(exc)

    def _command(self, frame):
        small=cv2.resize(frame,(int(self.cfg['camera']['width']),int(self.cfg['camera']['height'])))
        (road,outside,marking), perception_mode=self.segmenter.infer(small)
        geom=estimate_geometry(road,outside,marking,self.cfg['geometry'],self.segmenter.obstacle)
        self.mode='%s+%s' % (self.mode.split('+')[0], perception_mode)
        if self.model is not None:
            point=self.model.predict(small); waypoint=(point.x,point.y,point.confidence)
        else:
            ys,xs=np.nonzero(marking>0); keep=ys>int(marking.shape[0]*.48)
            if np.count_nonzero(keep)>=8:
                coef=np.polyfit(ys[keep],xs[keep],1); waypoint=(float(np.clip(np.polyval(coef,int(marking.shape[0]*.64))/marking.shape[1],0,1)),.64,min(1,np.count_nonzero(keep)/80.))
            else: waypoint=(geom.center_x/small.shape[1],.64,0.)
        target=fuse(waypoint,geom,small.shape[1],small.shape[0]); now=time.time(); dt=now-self.last; self.last=now; command=self.controller.update(target,geom,dt)
        return small,road,outside,marking,geom,target,command,dt

    def _callback(self, change):
        if not self.running or not self.lock.acquire(False): return
        try:
            frame=change['new']; result=self._command(frame); small,road,outside,marking,geom,target,cmd,dt=result
            if self.armed and cmd.state not in (State.STOP.value,State.REACQUIRE.value): self.car.set_steering(cmd.steering); self.car.set_throttle(cmd.throttle)
            else: self.car.stop(); self.car.center_steering()
            instant=1/max(1e-3,dt); self.fps=instant if not self.fps else .2*instant+.8*self.fps
            view=small.copy(); view[road>0]=(30,110,30); view[outside>0]=(220,220,220); view[marking>0]=(0,80,220)
            view[self.segmenter.obstacle>0]=(0,0,255)
            if geom.points: cv2.polylines(view,[np.asarray([(int(x),int(y)) for y,x in geom.points],np.int32)],False,(0,255,0),2)
            cv2.circle(view,(int(target.x*224),int(target.y*224)),5,(255,0,255),-1); cv2.putText(view,'%s %.1ffps conf %.2f steer %.2f gas %.2f'%(cmd.state,self.fps,geom.confidence,cmd.steering,cmd.throttle),(2,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
            if self.widgets: self.widgets['image'].value=bytes(cv2.imencode('.jpg',view)[1]); self.widgets['status'].value='<b>%s | %s | %.1f FPS | conf %.2f</b>'%(self.mode,cmd.state,self.fps,geom.confidence)
        except Exception as exc:
            self.stop();
            if self.widgets: self.widgets['status'].value='<b style="color:red">ERROR: %s</b>'%exc
        finally: self.lock.release()

    def start(self):
        from jetcam.csi_camera import CSICamera
        from notebook3.basic_motion import JetRacerController
        c=self.cfg['camera']; self.camera=CSICamera(width=c['width'],height=c['height'],capture_fps=0); h=self.cfg.get('hardware',{'steering_gain':-.65,'steering_offset':0.,'throttle_gain':.8})
        self.car=JetRacerController(h['steering_gain'],h['steering_offset'],h['throttle_gain'],self.cfg['control']['throttle_max']); self.car.stop(); self.car.center_steering(); self.running=True; self.camera.observe(self._callback,names='value'); self.camera.running=True

    def set_armed(self, value):
        self.armed=bool(value)
        if not self.armed and self.car: self.car.stop(); self.car.center_steering()

    def stop(self):
        self.running=False
        if self.car: self.car.stop(); self.car.center_steering()
        if self.camera:
            self.camera.unobserve_all(); self.camera.running=False
