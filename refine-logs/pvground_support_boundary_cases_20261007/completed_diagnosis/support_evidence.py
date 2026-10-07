"""Offline six-face member evidence; GT is diagnostic only."""
import numpy as np


def analyze_case(arrays):
    xyz = arrays['xyz'].astype(np.float64)
    ids = arrays['superpoint'].astype(np.int64)
    target = arrays['target'].astype(bool)
    gt = arrays['root_gt'].astype(np.float64)
    gt_bounds = np.concatenate((gt[:3] - gt[3:] / 2, gt[:3] + gt[3:] / 2))
    n = arrays['text_logits'].shape[0]
    counts = np.bincount(ids, minlength=n)
    target_counts = np.bincount(ids[target], minlength=n)
    assert xyz.shape == (50000, 3) and target.any() and counts.sum() == 50000
    assert arrays['fused_active'].shape == arrays['query_logits'].shape == (2, n)
    target_bounds = np.concatenate((xyz[target].min(0), xyz[target].max(0)))
    roles = []
    for slot, role in enumerate(('deployed', 'historical')):
        foreground = arrays['fused_active'][slot, ids].astype(bool)
        points = xyz[foreground]
        if not arrays['reference_valid'][slot]:
            roles.append(dict(role=role, query=int(arrays['queries'][slot]), valid_mask_range=False,
                foreground_points=int(foreground.sum()), false_positive_points=int((foreground & ~target).sum()),
                false_negative_points=int((~foreground & target).sum()), target_points=int(target.sum()), faces=[]))
            continue
        assert len(points) > 0
        bounds = np.concatenate((points.min(0), points.max(0)))
        reference = arrays['reference_boxes'][slot].astype(np.float64)
        declared = np.concatenate((reference[:3] - reference[3:] / 2, reference[:3] + reference[3:] / 2))
        assert np.max(np.abs(bounds - declared)) < 2e-6
        faces = []
        for face in range(6):
            axis = face % 3
            point_indices = np.flatnonzero(foreground & (xyz[:, axis] == bounds[face]))
            assert len(point_indices) > 0
            groups = np.unique(ids[point_indices])
            pure_target = groups[target_counts[groups] == counts[groups]]
            pure_background = groups[target_counts[groups] == 0]
            mixed = groups[(target_counts[groups] > 0) & (target_counts[groups] < counts[groups])]
            assert len(pure_target) + len(pure_background) + len(mixed) == len(groups)
            # Positive error denotes excessive extent; negative denotes missing extent.
            error = (gt_bounds[face] - bounds[face]) if face < 3 else (bounds[face] - gt_bounds[face])
            faces.append(dict(face=('x-', 'y-', 'z-', 'x+', 'y+', 'z+')[face], extent=float(bounds[face]),
                gt_box_extent=float(gt_bounds[face]), gt_member_extent=float(target_bounds[face]),
                signed_outward_error_m=float(error), extremal_point_indices=point_indices.tolist(),
                extremal_target_points=int(target[point_indices].sum()), extremal_points=len(point_indices),
                pure_target_superpoints=pure_target.tolist(), pure_background_superpoints=pure_background.tolist(),
                mixed_superpoints=mixed.tolist(),
                superpoint_target_fraction={str(int(g)): float(target_counts[g] / counts[g]) for g in groups},
                query_logits={str(int(g)): float(arrays['query_logits'][slot, g]) for g in groups},
                text_logits={str(int(g)): float(arrays['text_logits'][g]) for g in groups},
                fused_logits={str(int(g)): float(arrays['fused_logits'][slot, g]) for g in groups}))
        roles.append(dict(role=role, query=int(arrays['queries'][slot]), valid_mask_range=True, foreground_points=int(foreground.sum()),
            false_positive_points=int((foreground & ~target).sum()), false_negative_points=int((~foreground & target).sum()),
            target_points=int(target.sum()), faces=faces))
    return dict(roles=roles, target_member_bounds=target_bounds.tolist())
