import numpy as np
import pytest

from scripts.audit_nr3d_superpoint_training_targets import describe_target


def test_strict_majority_can_remove_all_target_superpoints():
    result = describe_target(np.array([4, 4, 4, 4]), np.array([True, True, False, False]))
    assert result['majority_positive_superpoints'] == 0
    assert result['majority_mask_iou'] == 0
    assert result['optimal_superpoint_mask_iou'] == .5
    assert result['empty_native_slots'] == 4


def test_majority_target_can_miss_050_while_representational_oracle_passes():
    labels = np.array([0] * 20 + [1] * 90)
    target = np.array([True] * 60 + [False] * 50)
    result = describe_target(labels, target)
    assert result['majority_positive_superpoints'] == 1
    assert result['majority_mask_iou'] == pytest.approx(1 / 3)
    assert result['majority_target_point_recall'] == pytest.approx(1 / 3)
    assert result['optimal_superpoint_mask_iou'] == pytest.approx(6 / 11)
