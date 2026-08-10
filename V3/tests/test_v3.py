import unittest
import numpy as np
from V3.divider import DividerTracker
from V3.fusion import fuse
from V3.control import Controller, State

G={'bands':[.46,.56,.66,.78,.90],'divider_min_pixels':8,'divider_min_confidence':.32,'divider_max_jump':.22,'white_hard_ratio':.32,'white_side_ratio':.12}
C={'max_steering':.82,'max_steering_step':.075,'steering_alpha':.3,'kp':.92,'kd':.07,'heading_gain':.34,'throttle_min':0.,'throttle_cruise':.18,'throttle_max':.24,'throttle_step_up':.006,'throttle_step_down':.04,'road_confidence_stop':.18,'white_hard_ratio':.32,'recovery_reverse':.07}
class Geo:
    valid=True; left_x=30.; right_x=210.; center_x=112.; white_center=0.; white_left=0.; white_right=0.; obstacle=0.; obstacle_offset=0.
class TestV3(unittest.TestCase):
    def test_divider_beats_outer_waypoint(self):
        m=np.zeros((224,224),np.uint8); road=np.zeros_like(m); road[90:]=255
        for y in range(85,210): m[y,max(0,int(70+.45*(y-85))-2):int(70+.45*(y-85))+3]=255
        d=DividerTracker(G).update(m,road); t=fuse((.93,.65,.95),d,Geo(),224,224,G)
        self.assertEqual(t.source,'divider_first'); self.assertLess(t.x,.65)
        d2=type('D',(),{'confidence':.8,'x':140.,'heading':.1})()
        exact=fuse((.93,.65,.95),d2,Geo(),224,224,G)
        self.assertAlmostEqual(exact.x,140./224.,places=6)
    def test_tight_curved_divider_uses_lookahead_curve(self):
        m=np.zeros((224,224),np.uint8); road=np.zeros_like(m); road[70:]=255
        for y in range(70,215):
            x=55 + .0028*(y-70)*(y-70)
            m[y,max(0,int(x)-2):min(224,int(x)+3)]=255
        d=DividerTracker(G).update(m,road)
        expected=55 + .0028*(int(.66*224)-70)**2
        self.assertGreater(d.confidence,.45)
        self.assertLess(abs(d.x-expected),8.)
    def test_temporal_hold_survives_one_weak_mask(self):
        tracker=DividerTracker(G); m=np.zeros((224,224),np.uint8); road=np.ones_like(m)*255; m[80:205,105:110]=255
        first=tracker.update(m,road); weak=np.zeros_like(m); held=tracker.update(weak,road)
        self.assertGreater(first.confidence,.45); self.assertGreaterEqual(held.confidence,.32); self.assertEqual(held.source,'temporal_hold')
    def test_white_forbidden(self):
        g=Geo(); g.white_center=.5; d=DividerTracker(G).update(np.zeros((224,224),np.uint8),np.ones((224,224),np.uint8)); t=fuse((.5,.6,.8),d,g,224,224,G); self.assertTrue('white_recover' in t.source)
    def test_divider_outside_white_is_rejected(self):
        m=np.zeros((224,224),np.uint8); m[80:190,180:185]=255; road=np.ones_like(m)*255; outside=np.zeros_like(m); outside[80:190,170:224]=255
        d=DividerTracker(G).update(m,road,outside); self.assertLess(d.confidence,.32)
    def test_side_white_recovery(self):
        g=Geo(); g.white_left=.25; d=type('D',(),{'confidence':.8,'x':45.,'heading':0.})()
        t=fuse((.15,.6,.9),d,g,224,224,G); self.assertIn('white_left_recover',t.source); self.assertGreater(t.x,.25)
    def test_obstacle_shifts_target_away(self):
        g=Geo(); g.obstacle=.2; g.obstacle_offset=.25; d=type('D',(),{'confidence':.8,'x':145.,'heading':0.})()
        t=fuse((.65,.6,.9),d,g,224,224,G); self.assertIn('obstacle_avoid',t.source); self.assertLess(t.x,.65)
    def test_rate_limit(self):
        ctl=Controller(C); t=type('T',(),{'x':1.,'heading':0.,'confidence':1.,'source':'x'})(); cmd=ctl.update(t,Geo(),.05); self.assertLessEqual(abs(cmd.steering),.07501)
    def test_white_center_turns_back_when_road_is_visible(self):
        g=Geo(); g.white_center=.5; g.center_x=70.; ctl=Controller(C)
        t=type('T',(),{'x':70./224.,'heading':0.,'confidence':.5,'source':'white_recover'})()
        cmd=ctl.update(t,g,.05); self.assertEqual(cmd.state,State.RECOVERY_TURN.value); self.assertLess(cmd.steering,0.)
if __name__=='__main__': unittest.main()
