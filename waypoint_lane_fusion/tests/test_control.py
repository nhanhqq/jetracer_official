import unittest
from waypoint_lane_fusion.behavior import BehaviorStateMachine
from waypoint_lane_fusion.controller import DriveController, WaypointFilter
from waypoint_lane_fusion.types import Detection, DetectionSnapshot, DriveState, Waypoint

C=dict(waypoint_ema=.3,confidence_stop=.4,confidence_resume=.55,lane_lost_frames=2,lane_resume_frames=2,
       kp=.95,kd=.08,heading_gain=.28,steering_alpha=.25,max_steering=.82,max_steering_change=.04,
       throttle_min=.07,throttle_max=.18,curve_penalty=.1,confidence_penalty=.06,accel_rate=.008,
       brake_rate=.035,turn_bias=.1,detection_ttl=.75)

class ControlTests(unittest.TestCase):
    def test_rate_limit_and_acceleration(self):
        ctl=DriveController(C); cmd=ctl.update(Waypoint(.9,.3,1),DriveState.NORMAL,.05)
        self.assertLessEqual(abs(cmd.steering),.04001); self.assertAlmostEqual(cmd.throttle,.008)
    def test_curve_reduces_target_throttle(self):
        straight=DriveController(C); curve=DriveController(C)
        for _ in range(30): a=straight.update(Waypoint(.5,.3,1),DriveState.NORMAL,.05); b=curve.update(Waypoint(.9,.3,1),DriveState.NORMAL,.05)
        self.assertLess(b.throttle,a.throttle)
    def test_lane_loss_hysteresis(self):
        sm=BehaviorStateMachine(C); empty=DetectionSnapshot()
        self.assertEqual(sm.update(Waypoint(.5,.3,.2),empty)[0],DriveState.LANE_LOST)
        sm.update(Waypoint(.5,.3,.8),empty); self.assertEqual(sm.state,DriveState.LANE_LOST)
        self.assertEqual(sm.update(Waypoint(.5,.3,.8),empty)[0],DriveState.NORMAL)
    def test_obstacle_priority(self):
        import time
        sm=BehaviorStateMachine(C); empty=DetectionSnapshot()
        sm.update(Waypoint(.5,.3,.9),empty); sm.update(Waypoint(.5,.3,.9),empty)
        snap=DetectionSnapshot([Detection("obstacle",.9)],time.time(),10)
        self.assertEqual(sm.update(Waypoint(.5,.3,.9),snap)[0],DriveState.OBSTACLE)
    def test_filter_rejects_single_jump(self):
        f=WaypointFilter(.25); f.update(Waypoint(.5,.3,1)); out=f.update(Waypoint(.9,.3,1))
        self.assertAlmostEqual(out.x,.6)

if __name__=="__main__": unittest.main()
