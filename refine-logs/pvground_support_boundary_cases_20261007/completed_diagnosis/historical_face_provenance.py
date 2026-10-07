"""Conservative provenance of cached mask-range faces from identical raw points.

Coordinate candidates are possible old extremal members, not a reconstructed
old foreground mask or a recovered Query. Current predicted masks are unused.
"""
import numpy as np


TOLERANCE_M = 2e-6
FACE_NAMES = ('x-', 'y-', 'z-', 'x+', 'y+', 'z+')


def historical_faces(arrays, cached):
    xyz = arrays['xyz'].astype(np.float64)
    target = arrays['target'].astype(bool)
    superpoint = arrays['superpoint'].astype(np.int64)
    assert target.any() and xyz.shape == (50000, 3)
    assert arrays['root_gt'].tolist() == cached['root_gt']
    assert cached['reference_valid']
    counts = np.bincount(superpoint)
    target_counts = np.bincount(superpoint, weights=target, minlength=len(counts))
    reference = np.asarray(cached['reference_box'], dtype=np.float64)
    truth = np.asarray(cached['root_gt'], dtype=np.float64)
    low, high = reference[:3] - reference[3:] / 2, reference[:3] + reference[3:] / 2
    gt_low, gt_high = truth[:3] - truth[3:] / 2, truth[:3] + truth[3:] / 2
    result = []
    for face, name in enumerate(FACE_NAMES):
        axis = face % 3
        upper = face >= 3
        position = high[axis] if upper else low[axis]
        candidates = np.flatnonzero(np.abs(xyz[:, axis] - position) <= TOLERANCE_M)
        assert candidates.size, (cached['row_id'], name, position)
        possible_target = int(target[candidates].sum())
        ids = np.unique(superpoint[candidates])
        pure_background = int((target_counts[ids] == 0).sum())
        pure_target = int((target_counts[ids] == counts[ids]).sum())
        mixed = len(ids) - pure_background - pure_target
        if possible_target == 0:
            certainty = 'only_background_member_possible'
        elif possible_target == len(candidates):
            certainty = 'only_target_member_possible'
        else:
            certainty = 'target_and_background_members_possible'
        error = position - gt_high[axis] if upper else gt_low[axis] - position
        outside = xyz[target, axis] > position + TOLERANCE_M if upper else xyz[target, axis] < position - TOLERANCE_M
        result.append(dict(face=name, historical_reference_coordinate=position,
            coordinate_candidate_points=int(len(candidates)), possible_target_points=possible_target,
            possible_background_points=int(len(candidates) - possible_target), member_certainty=certainty,
            possible_superpoints=ids.tolist(), possible_pure_background_superpoints=pure_background,
            possible_pure_target_superpoints=pure_target, possible_mixed_superpoints=mixed,
            all_possible_superpoints_pure_background=pure_background == len(ids),
            all_possible_superpoints_mixed=mixed == len(ids),
            signed_gt_outward_error_m=float(error), observed_target_members_beyond_face=int(outside.sum())))
    return dict(row_id=cached['row_id'], historical_query_index=cached['query'],
        historical_reference_box=cached['reference_box'], faces=result,
        current_masks_used=False, historical_foreground_reconstructed=False)
