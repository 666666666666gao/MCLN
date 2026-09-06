"""Candidate-conditioned reading of raw point appearance and pretrained SA1 features."""

import math

import torch
from torch import nn


@torch.no_grad()
def nearest_candidate_points(xyz, boxes, count=64, query_chunk_size=32):
    """Select points by distance in the candidate's axis-scaled coordinate frame.

    Raw predicted extents can be nonpositive in the existing model. The 5 cm
    lower bound is only a neighborhood scale; it does not change predicted boxes.
    Chunking avoids a B x 256 x 50000 x 3 distance temporary.
    """
    indices = []
    for start in range(0, boxes.shape[1], query_chunk_size):
        chunk = boxes[:, start:start + query_chunk_size]
        half_extent = (chunk[..., 3:] * .5).clamp_min(.05)
        relative = (xyz[:, None] - chunk[:, :, None, :3]) / half_extent[:, :, None]
        distance = relative.square().sum(dim=-1)
        indices.append(distance.topk(count, dim=-1, largest=False, sorted=True).indices)
    return torch.cat(indices, dim=1)


class CandidateLocalVisual(nn.Module):
    """Read a separate point neighborhood for each language-conditioned query."""

    def __init__(self, d_model=288, point_dim=128, hidden_dim=144, heads=4,
                 points_per_query=64):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.points_per_query = points_per_query
        self.point_encoder = nn.Sequential(nn.Linear(point_dim + 9, hidden_dim), nn.ReLU())
        self.query_projection = nn.Linear(d_model, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def read_points(self, query, boxes, xyz, rgb, features):
        """Arguments use B x Q x K point axes; returns a B x Q x C residual."""
        relative = xyz - boxes[:, :, None, :3]
        scale = (boxes[..., 3:] * .5).clamp_min(.05)
        memory = self.point_encoder(torch.cat([
            features, rgb, relative, relative / scale[:, :, None],
        ], dim=-1))
        batch, queries, points, _ = memory.shape
        q = self.query_projection(query).reshape(batch, queries, self.heads, self.head_dim)
        k = self.key_projection(memory).reshape(batch, queries, points, self.heads, self.head_dim)
        v = self.value_projection(memory).reshape(batch, queries, points, self.heads, self.head_dim)
        weights = ((q[:, :, None] * k).sum(dim=-1) / math.sqrt(self.head_dim)).softmax(dim=2)
        summary = (weights[..., None] * v).sum(dim=2).reshape(batch, queries, -1)
        return self.output_projection(summary)

    def forward(self, query, boxes, point_cloud, sa1_xyz, sa1_features):
        import pointnet2_utils

        indices = nearest_candidate_points(point_cloud[..., :3], boxes, self.points_per_query)
        batch, queries, points = indices.shape
        flat_indices = indices.reshape(batch, -1)
        selected = point_cloud.gather(1, flat_indices[..., None].expand(-1, -1, 6))
        xyz = selected[..., :3].contiguous()
        # Reuse the backbone's existing CUDA interpolation and inverse-distance rule.
        distances, neighbors = pointnet2_utils.three_nn(xyz, sa1_xyz.contiguous())
        reciprocal = 1. / (distances + 1e-8)
        weights = reciprocal / reciprocal.sum(dim=2, keepdim=True)
        features = pointnet2_utils.three_interpolate(sa1_features.contiguous(), neighbors, weights)
        return self.read_points(query, boxes,
            xyz.reshape(batch, queries, points, 3),
            selected[..., 3:6].reshape(batch, queries, points, 3),
            features.transpose(1, 2).reshape(batch, queries, points, -1))
