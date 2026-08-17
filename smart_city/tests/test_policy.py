import unittest

from smart_city.policy import SmartCityPolicy


class SmartCityPolicyTests(unittest.TestCase):
    def test_red_has_priority_and_latches(self):
        policy = SmartCityPolicy(red_confirm_frames=2, green_confirm_frames=2)
        self.assertEqual(policy.update([{"label": "red_light", "confidence": .9}], True).state, "DRIVE")
        self.assertEqual(policy.update([{"label": "red_light", "confidence": .9}], True).state, "STOP_RED")
        self.assertEqual(policy.update([], True).state, "STOP_RED")

    def test_green_releases_after_confirmation(self):
        policy = SmartCityPolicy(red_confirm_frames=1, green_confirm_frames=2)
        policy.update([{"label": "red_light", "confidence": .9}], True)
        self.assertEqual(policy.update([{"label": "green_light", "confidence": .9}], True).state, "STOP_RED")
        self.assertEqual(policy.update([{"label": "green_light", "confidence": .9}], True).state, "DRIVE")

    def test_red_wins_if_both_lights_are_detected(self):
        policy = SmartCityPolicy(red_confirm_frames=1, green_confirm_frames=1)
        decision = policy.update([
            {"label": "den_do", "confidence": .60},
            {"label": "den_xanh", "confidence": .99},
        ], True)
        self.assertEqual(decision.state, "STOP_RED")

    def test_light_priority_is_above_forbidden_and_direction_signs(self):
        policy = SmartCityPolicy(red_confirm_frames=1, sign_confirm_frames=1,
                                 forbidden_random_seed=3)
        decision = policy.update([
            {"label": "den_do", "confidence": .40},
            {"label": "bien_cam", "confidence": .99},
            {"label": "re_trai", "confidence": .99},
        ], True)
        self.assertEqual(decision.state, "STOP_RED")

    def test_sign_is_temporally_confirmed(self):
        policy = SmartCityPolicy(sign_confirm_frames=2)
        policy.update([{"label": "left_sign", "confidence": .9}], True)
        self.assertEqual(policy.pending_route, "STRAIGHT")
        policy.update([{"label": "left_sign", "confidence": .9}], True)
        self.assertEqual(policy.pending_route, "LEFT")

    def test_forbidden_sign_selects_a_random_turn(self):
        policy = SmartCityPolicy(sign_confirm_frames=1, forbidden_random_seed=7)
        decision = policy.update([{"label": "bien_cam", "confidence": .9}], True)
        self.assertEqual(decision.state, "DRIVE")
        self.assertIn(decision.route, ("LEFT", "RIGHT"))

    def test_forbidden_sign_has_priority_over_direction_sign(self):
        policy = SmartCityPolicy(sign_confirm_frames=1, forbidden_random_seed=7)
        decision = policy.update([
            {"label": "di_thang", "confidence": .99},
            {"label": "bien_cam", "confidence": .40},
        ], True)
        self.assertIn(decision.route, ("LEFT", "RIGHT"))

    def test_generic_forbidden_sign_without_direction_still_turns_randomly(self):
        policy = SmartCityPolicy(sign_confirm_frames=1, forbidden_random_seed=9)
        decision = policy.update([{"label": "bien_cam", "confidence": .9}], True)
        self.assertEqual(decision.state, "DRIVE")
        self.assertIn(decision.route, ("LEFT", "RIGHT"))


if __name__ == "__main__":
    unittest.main()
