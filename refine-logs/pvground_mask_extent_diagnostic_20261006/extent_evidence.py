"""Fixed selected-Query Mask extent diagnostics; no score or model changes."""
import numpy as np

QUANTILE = .005


def member_geometry(xyz, superpoint, target_mask):
    ids, inverse, count = np.unique(superpoint, return_inverse=True, return_counts=True)
    low = np.full((len(ids), 3), np.inf, dtype=np.float64)
    high = np.full((len(ids), 3), -np.inf, dtype=np.float64)
    np.minimum.at(low, inverse, xyz)
    np.maximum.at(high, inverse, xyz)
    target_count = np.bincount(inverse, weights=target_mask, minlength=len(ids)).astype(np.int64)
    assert count.sum() == len(xyz) == 50000
    return ids, inverse, count, low, high, target_count


def extent_box(low, high):
    size = high - low
    if not np.all(size > 0):
        return None
    return np.concatenate(((low + high) * .5, size))


def exact_box(active, low, high):
    if not active.any():
        return None
    return extent_box(low[active].min(0), high[active].max(0))


def quantile_box(xyz, point_active):
    count = int(point_active.sum())
    if count == 0:
        return None, np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=np.int64)
    values = np.sort(xyz[point_active].astype(np.float64), axis=0)
    positions = (count - 1) * np.array([QUANTILE, 1 - QUANTILE])
    floor = np.floor(positions).astype(np.int64)
    ceil = np.ceil(positions).astype(np.int64)
    ranks = np.stack((floor, ceil), 1)
    samples = values[ranks]
    fraction = (positions - floor)[:, None]
    limits = samples[:, 0] * (1 - fraction) + samples[:, 1] * fraction
    return extent_box(limits[0], limits[1]), samples, ranks


def box_iou(box, truth):
    if box is None:
        return 0.
    low = np.maximum(box[:3] - box[3:] * .5, truth[:3] - truth[3:] * .5)
    high = np.minimum(box[:3] + box[3:] * .5, truth[:3] + truth[3:] * .5)
    intersection = np.maximum(high - low, 0).prod()
    return float(intersection / (box[3:].prod() + truth[3:].prod() - intersection))
