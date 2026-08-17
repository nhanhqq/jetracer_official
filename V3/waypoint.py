"""Use the original fast waypoint model without changing its source."""
from pathlib import Path
from waypoint_lane_fusion.lane_model import TensorRTWaypointModel, OnnxWaypointModel

class WaypointModel:
    def __init__(self, cfg):
        root=Path(cfg['_root']); models=cfg['models']; engine=root/models.get('waypoint_engine','')
        onnx=root/models.get('waypoint','')
        self.mode='waypoint-fallback'; self.model=None
        if engine.exists():
            try:
                self.model=TensorRTWaypointModel(engine); self.mode='waypoint-tensorrt'
            except Exception as exc:
                self.error='TensorRT: %s' % exc
        if self.model is None:
            try:
                self.model=OnnxWaypointModel(onnx); self.mode='waypoint-onnx'
            except Exception as exc:
                self.error=getattr(self,'error','')+' ONNX: %s' % exc
    def predict(self, frame):
        if self.model is None: return .5,.66,.0
        p=self.model.predict(frame); return p.x,p.y,p.confidence
