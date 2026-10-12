"""Compact whole-instance evidence from actual point members, not SP centers.

This is an unintegrated geometry provider, not a trained box refiner. Point
geometry is summarized on CPU for one cloud and its current augmentation.
Native Text/Query Mask logits remain differentiable in the aggregation.
"""
import numpy as np
import torch
from torch.nn import functional as F


def member_statistics(xyz, superpoint_ids, bins=32):
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert superpoint_ids.shape == (len(xyz),)
    native_ids, inverse = np.unique(superpoint_ids, return_inverse=True)
    count = np.bincount(inverse)
    origin = xyz.min(axis=0)
    span = xyz.max(axis=0) - origin
    assert (span > 0).all()
    normalized = (xyz - origin) / span
    bin_ids = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    histogram, mean, second, lower, upper = [], [], [], [], []
    for axis in range(3):
        histogram.append(np.bincount(inverse * bins + bin_ids[:, axis],
                         minlength=len(native_ids) * bins).reshape(len(native_ids), bins))
        mean.append(np.bincount(inverse, weights=normalized[:, axis]) / count)
        second.append(np.bincount(inverse, weights=normalized[:, axis] ** 2) / count)
        minimum = np.full(len(native_ids), np.inf)
        maximum = np.full(len(native_ids), -np.inf)
        np.minimum.at(minimum, inverse, normalized[:, axis])
        np.maximum.at(maximum, inverse, normalized[:, axis])
        lower.append(minimum)
        upper.append(maximum)
    return dict(native_ids=native_ids, count=count, origin=origin, span=span,
                histogram=np.stack(histogram, axis=1),
                mean=np.stack(mean, axis=1), second=np.stack(second, axis=1),
                lower=np.stack(lower, axis=1), upper=np.stack(upper, axis=1))


def mask_range_evidence(text_logits, query_logits, alpha, geometry):
    """Use native scalar-alpha logit fusion and the observed native SP slots.

    SP weight is proportional to sigmoid(fused logit) times actual member count.
    log-sigmoid and softmax implement this normalization without dividing a
    possibly underflowed foreground mass. No foreground threshold or Q pruning
    is used. Extent means are evidence, not claimed final box boundaries.
    """
    assert query_logits.ndim == 2 and text_logits.shape == query_logits.shape
    assert alpha.ndim == 0
    fused = alpha * text_logits + (1 - alpha) * query_logits
    native_ids = torch.as_tensor(geometry['native_ids'], device=fused.device, dtype=torch.long)
    constants = {name: torch.as_tensor(geometry[name], dtype=fused.dtype, device=fused.device)
                 for name in ('count', 'histogram', 'mean', 'second', 'lower', 'upper', 'origin', 'span')}
    count = constants['count']
    observed_logits = fused[:, native_ids]
    weights = (F.logsigmoid(observed_logits) + count.log()).softmax(dim=-1)
    histogram = constants['histogram'] / count[:, None, None]
    profile = (weights @ histogram.reshape(len(count), -1)).reshape(
        len(fused), 3, geometry['histogram'].shape[-1])
    mean = weights @ constants['mean']
    second = weights @ constants['second']
    return dict(axis_profile=profile, mean_normalized=mean,
                second_normalized=second, variance_normalized=second - mean.square(),
                mean_xyz=constants['origin'] + mean * constants['span'],
                member_lower_mean_normalized=weights @ constants['lower'],
                member_upper_mean_normalized=weights @ constants['upper'],
                support_fraction=(observed_logits.sigmoid() @ count) / count.sum())
