import math

from scripts.evaluate_scanrefer_native_box_transfer_official import promotion_check


def protected_metrics():
    return {'rows': 9508, 'rec_hits025': 5572, 'rec_hits050': 4797,
            'mask_hits025': math.ceil(.587 * 9508), 'mask_hits050': math.ceil(.507 * 9508),
            'mask_miou': 44.72}


def test_existing_floors_allow_transfer_without_waiting_for_stretch_goals():
    values = protected_metrics()
    assert promotion_check(values, dict(values))['advance_to_nr3d_sr3d_rec']


def test_strict_threshold_improvement_does_not_excuse_primary_regression():
    protected = protected_metrics()
    candidate = dict(protected, rec_hits025=5571, rec_hits050=4900)
    assert not promotion_check(protected, candidate)['advance_to_nr3d_sr3d_rec']


def test_scan_mask_floor_still_applies_to_rec_passing_candidate():
    protected = protected_metrics()
    candidate = dict(protected, mask_miou=44.71)
    assert not promotion_check(protected, candidate)['advance_to_nr3d_sr3d_rec']


def test_same_run_protected_control_is_also_preserved():
    candidate = protected_metrics()
    protected = dict(candidate, rec_hits050=4800)
    assert not promotion_check(protected, candidate)['advance_to_nr3d_sr3d_rec']
