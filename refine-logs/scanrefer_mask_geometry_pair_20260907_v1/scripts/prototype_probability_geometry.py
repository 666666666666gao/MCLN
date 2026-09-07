"""Diagnostic inverse-CDF readout; not enabled in MCLN training or deployment.

Use the original sampled points, sigmoid Mask mass, and fixed .005/.995
quantiles. Coordinate sorting is fixed input geometry; derivatives flow through
CDF weights and linear interpolation. This is not an OT sorting implementation.
"""
import torch


def probability_quantile_boxes(coords, superpoint_ids, mask_logits, quantile=.005):
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert superpoint_ids.shape == coords.shape[:1]
    assert mask_logits.ndim == 2 and 0. < quantile < .5
    assert torch.isfinite(coords).all() and torch.isfinite(mask_logits).all()
    point_weights = mask_logits.double().index_select(1, superpoint_ids.long()).sigmoid()
    total = point_weights.sum(dim=1, keepdim=True)
    assert (total > 0).all()
    probabilities = coords.new_tensor([quantile, 1. - quantile], dtype=torch.float64)
    axis_bounds = []
    for axis in range(3):
        values, order = coords[:, axis].double().sort()
        weights = point_weights[:, order]
        cdf = (weights.cumsum(dim=1) - weights / 2.) / total
        cdf = torch.cat([torch.zeros_like(total), cdf, torch.ones_like(total)], dim=1)
        locations = torch.cat([values[:1], values, values[-1:]])
        targets = probabilities[None].expand(mask_logits.shape[0], -1).contiguous()
        upper = torch.searchsorted(cdf.contiguous(), targets)
        lower = upper - 1
        cdf0, cdf1 = cdf.gather(1, lower), cdf.gather(1, upper)
        assert (cdf1 > cdf0).all()
        fraction = (targets - cdf0) / (cdf1 - cdf0)
        bounds = locations[lower] + fraction * (locations[upper] - locations[lower])
        axis_bounds.append(bounds)
    bounds = torch.stack(axis_bounds, dim=-1)
    lower, upper = bounds[:, 0], bounds[:, 1]
    boxes = torch.cat([(lower + upper) / 2., upper - lower], dim=-1)
    assert torch.isfinite(boxes).all() and (boxes[:, 3:] >= 0.).all()
    return boxes.to(coords.dtype), point_weights
