"""Protect the user's paired REC and Scan Mask promotion requirements."""
import unittest

from scripts.evaluate_scanrefer_object_appearance_official import promotion_check


class PromotionRequirements(unittest.TestCase):
    def setUp(self):
        self.protected = dict(rows=9508, rec_hits025=5572, rec_hits050=4797,
                              mask_hits025=5689, mask_hits050=4975, mask_miou=45.93)

    def test_preserving_current_pair_advances_without_stretch(self):
        result = promotion_check(self.protected, dict(self.protected))
        self.assertTrue(result['advance_to_nr3d_sr3d_rec'])
        self.assertTrue(result['nr3d_sr3d_mask_not_a_promotion_gate'])

    def test_one_rec_hit_regression_blocks_each_threshold(self):
        for name in ['rec_hits025', 'rec_hits050']:
            candidate = dict(self.protected)
            candidate[name] -= 1
            self.assertFalse(promotion_check(self.protected, candidate)['advance_to_nr3d_sr3d_rec'])

    def test_stronger_same_run_control_cannot_be_ignored(self):
        protected = dict(self.protected, rec_hits050=4800)
        self.assertFalse(promotion_check(protected, self.protected)['advance_to_nr3d_sr3d_rec'])

    def test_each_scan_mask_floor_is_required(self):
        for name, value in [('mask_hits025', 5580), ('mask_hits050', 4820), ('mask_miou', 44.719)]:
            candidate = dict(self.protected)
            candidate[name] = value
            self.assertFalse(promotion_check(self.protected, candidate)['advance_to_nr3d_sr3d_rec'])


if __name__ == '__main__':
    unittest.main()
