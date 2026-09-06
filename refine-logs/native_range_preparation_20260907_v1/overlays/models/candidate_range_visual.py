"""Matched point budgets for center and spatially distributed candidate evidence."""

import math

import torch

from .candidate_local_visual import CandidateLocalVisual


@torch.no_grad()
def candidate_region_points(xyz, boxes, sampling, query_chunk_size=16):
    """Return 64 slots, validity, and octants; GT is never read.

    Both arms use the same 1.5-half-extent window and 5 cm minimum scale.
    Center takes the nearest64 points. Extent takes up to8 different points
    per octant around eight RoI grid supports at +/-0.5 half extents.
    Empty slots have index0 only as storage; validity excludes their evidence.
    """
    assert sampling in ('center', 'extent') and xyz.shape[1] >= 64
    all_indices, all_valid, all_regions = [], [], []
    for start in range(0, boxes.shape[1], query_chunk_size):
        current = boxes[:, start:start + query_chunk_size]
        scale = (current[..., 3:] * .5).clamp_min(.05)
        relative = (xyz[:, None] - current[:, :, None, :3]) / scale[:, :, None]
        region = ((relative[..., 0] >= 0).long() * 4
                  + (relative[..., 1] >= 0).long() * 2
                  + (relative[..., 2] >= 0).long())
        in_window = (relative.abs() <= 1.5).all(dim=-1)
        if sampling == 'center':
            distance = relative.square().sum(dim=-1).masked_fill(~in_window, float('inf'))
            values, indices = distance.topk(64, dim=-1, largest=False, sorted=True)
            valid = torch.isfinite(values)
            selected_regions = region.gather(-1, indices)
        else:
            indices_parts, valid_parts, region_parts = [], [], []
            for octant in range(8):
                support = relative.new_tensor([.5 if octant & bit else -.5 for bit in (4, 2, 1)])
                distance = (relative - support).square().sum(dim=-1)
                distance = distance.masked_fill(~(in_window & (region == octant)), float('inf'))
                values, selected = distance.topk(8, dim=-1, largest=False, sorted=True)
                indices_parts.append(selected)
                valid_parts.append(torch.isfinite(values))
                region_parts.append(torch.full_like(selected, octant))
            indices = torch.cat(indices_parts, dim=-1)
            valid = torch.cat(valid_parts, dim=-1)
            selected_regions = torch.cat(region_parts, dim=-1)
        all_indices.append(indices.masked_fill(~valid, 0))
        all_valid.append(valid)
        all_regions.append(selected_regions)
    return tuple(torch.cat(parts, dim=1) for parts in (all_indices, all_valid, all_regions))


def _observed_softmax(scores, observed, dim):
    # An empty observed region is represented by zero weights, not a distant point.
    return scores.masked_fill(~observed, torch.finfo(scores.dtype).min).softmax(dim=dim) * observed


class CandidateRangeVisual(CandidateLocalVisual):
    """The two sampling arms share all parameters and the same region readout."""

    def __init__(self, sampling):
        super().__init__()
        assert sampling in ('center', 'extent')
        self.sampling = sampling

    def read_regions(self, query, boxes, xyz, rgb, features, valid, regions):
        relative = xyz - boxes[:, :, None, :3]
        scale = (boxes[..., 3:] * .5).clamp_min(.05)
        memory = self.point_encoder(torch.cat([features, rgb, relative, relative / scale[:, :, None]], dim=-1))
        batch, queries, points, _ = memory.shape
        q = self.query_projection(query).reshape(batch, queries, self.heads, self.head_dim)
        k = self.key_projection(memory).reshape(batch, queries, points, self.heads, self.head_dim)
        v = self.value_projection(memory).reshape(batch, queries, points, self.heads, self.head_dim)
        scores = (q[:, :, None] * k).sum(dim=-1) / math.sqrt(self.head_dim)
        membership = valid[:, :, None, :] & (regions[:, :, None, :] == torch.arange(8, device=regions.device)[None, None, :, None])
        point_weights = _observed_softmax(scores[:, :, None, :, :], membership[..., None], dim=3)
        region_values = (point_weights[..., None] * v[:, :, None, :, :, :]).sum(dim=3)
        counts = membership.sum(dim=3)
        # Mean compatibility lets Query choose among observed spatial regions;
        # the point softmax above chooses evidence within each region.
        region_scores = (scores[:, :, None, :, :] * membership[..., None]).sum(dim=3) / counts.clamp_min(1)[..., None]
        region_weights = _observed_softmax(region_scores, (counts > 0)[..., None], dim=2)
        summary = (region_weights[..., None] * region_values).sum(dim=2).reshape(batch, queries, -1)
        return self.output_projection(summary) * valid.any(dim=-1, keepdim=True)

    def forward(self, query, boxes, point_cloud, sa1_xyz, sa1_features):
        import pointnet2_utils

        indices, valid, regions = candidate_region_points(point_cloud[..., :3], boxes, self.sampling)
        batch, queries, points = indices.shape
        selected = point_cloud.gather(1, indices.reshape(batch, -1, 1).expand(-1, -1, 6))
        xyz = selected[..., :3].contiguous()
        distances, neighbors = pointnet2_utils.three_nn(xyz, sa1_xyz.contiguous())
        reciprocal = 1. / (distances + 1e-8)
        weights = reciprocal / reciprocal.sum(dim=2, keepdim=True)
        features = pointnet2_utils.three_interpolate(sa1_features.contiguous(), neighbors, weights)
        return self.read_regions(query, boxes, xyz.reshape(batch, queries, points, 3),
            selected[..., 3:6].reshape(batch, queries, points, 3),
            features.transpose(1, 2).reshape(batch, queries, points, -1), valid, regions)
