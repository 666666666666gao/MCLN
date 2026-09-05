from itertools import product

import numpy as np

from scripts.nr3d_superpoint_mask_oracle import optimal_superpoint_mask_iou


def test_prefix_oracle_matches_exhaustive_unions_with_mixed_superpoints():
    labels = np.array([4, 4, 4, 9, 9, 20, 20, 20])
    for target_bits in product([False, True], repeat=len(labels)):
        target = np.array(target_bits)
        if not target.any():
            continue
        exhaustive = []
        for choices in product([False, True], repeat=3):
            mask = np.isin(labels, np.unique(labels)[np.array(choices)])
            exhaustive.append(np.logical_and(mask, target).sum() / np.logical_or(mask, target).sum())
        assert optimal_superpoint_mask_iou(labels, target) == max(exhaustive)
