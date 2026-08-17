import unittest
import numpy as np
from V2.pseudo_label import make_masks
from V2.geometry import estimate_geometry
from V2.fusion import fuse
from V2.control import Controller, State

CFG={'roi_top':.38,'bands':[.52,.64,.76,.90],'min_road_pixels':180,'min_boundary_pixels':8,'width_shrink_tolerance':.2}
CONTROL={'max_steering':.82,'max_steering_step':.075,'steering_alpha':.3,'kp':.82,'kd':.08,'heading_gain':.38,'safety_gain':.48,'obstacle_gain':.32,'throttle_min':0.,'throttle_cruise':.18,'throttle_max':.24,'throttle_step_up':.006,'throttle_step_down':.04,'road_confidence_stop':.22,'road_confidence_caution':.48,'white_center_stop':.32,'recovery_timeout':4.,'recovery_reverse':.07,'recovery_turn':.34,'reacquire_frames':5}

class V2Tests(unittest.TestCase):
    def test_white_is_not_road(self):
        im=np.full((224,224,3),255,np.uint8); im[110:,:, :]=(35,35,35)
        road,outside,_=make_masks(im)
        self.assertGreater(np.count_nonzero(outside[:110]),0); self.assertLess(np.count_nonzero(road[:100]),100)
    def test_waypoint_outside_is_corrected(self):
        road=np.zeros((224,224),np.uint8); road[110:]=255
        outside=np.zeros_like(road); marking=np.zeros_like(road)
        geom=estimate_geometry(road,outside,marking,CFG); target=fuse((.95,.7,.9),geom,224,224)
        self.assertTrue(target.corrected); self.assertLess(target.x,.8)
    def test_rate_limit_and_recovery(self):
        ctl=Controller(CONTROL); road=np.zeros((224,224),np.uint8); road[100:]=255; outside=np.zeros_like(road); marking=np.zeros_like(road)
        geom=estimate_geometry(road,outside,marking,CFG); target=fuse((.5,.6,.8),geom,224,224); cmd=ctl.update(target,geom,.05)
        self.assertLessEqual(abs(cmd.steering),.07501); self.assertIn(cmd.state,(State.CAUTION.value,State.NORMAL.value))

if __name__=='__main__': unittest.main()
