"""Experimental point-detail residual for native superpoint Mask features.

This is the point-based control for local memory, not a voxel backbone.
It consumes existing SA1 features and the same sampled XYZ/RGB input.
"""

import torch
from torch import nn

from utils.scatter_util import deterministic_scatter_mean_dim0


class PointDetailSuperpointResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(134, 128), nn.ReLU())
        self.output = nn.Linear(128, 288, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, point_features, point_cloud, superpoints, centers):
        """One scene: (N,128), (N,6), (N,), (S,3) -> (S,288)."""
        offsets = point_cloud[:, :3] - centers.index_select(0, superpoints)
        detail = torch.cat((point_features, offsets, point_cloud[:, 3:6]), dim=-1)
        encoded = self.point_encoder(detail)
        pooled = deterministic_scatter_mean_dim0(
            encoded, superpoints, dim_size=centers.shape[0])
        return self.output(pooled)


def interpolate_sa1(point_xyz, anchor_xyz, anchor_features):
    # Loaded only for the native CUDA path; there is no CPU replacement.
    import pointnet2_utils

    distance, indices = pointnet2_utils.three_nn(
        point_xyz.unsqueeze(0).contiguous(), anchor_xyz.unsqueeze(0).contiguous())
    # SA1 anchors are sampled input points, so exact zero distances occur.
    # Preserve the existing PointnetFPModule interpolation convention.
    inverse_distance = 1.0 / (distance + 1e-8)
    weights = inverse_distance / inverse_distance.sum(dim=2, keepdim=True)
    values = pointnet2_utils.three_interpolate(
        anchor_features.unsqueeze(0).contiguous(), indices, weights)
    return values.squeeze(0).transpose(0, 1)


class SuperpointDetailIntervention:
    """Temporary attachment; base state and native Mask heads stay intact."""

    def __init__(self, model, addon):
        self.addon = addon
        self.inputs = self.anchors = None
        self.scene_index = 0
        self.handles = [
            model.register_forward_pre_hook(self.capture_inputs),
            model.backbone_net.sa1.register_forward_hook(self.capture_anchors),
            model.super_grouper.register_forward_hook(self.add_detail),
            model.register_forward_hook(self.check_batch),
        ]

    def capture_inputs(self, module, inputs):
        self.inputs = inputs[0]
        self.anchors = None
        self.scene_index = 0

    def capture_anchors(self, module, inputs, output):
        xyz, features, _ = output
        self.anchors = (xyz.detach(), features.detach())

    def add_detail(self, module, inputs, output):
        scene = self.scene_index
        cloud = self.inputs['point_clouds'][scene]
        superpoints = self.inputs['superpoint'][scene]
        centers = inputs[1].squeeze(0)
        xyz, features = self.anchors
        point_features = interpolate_sa1(cloud[:, :3], xyz[scene], features[scene])
        residual = self.addon(point_features, cloud, superpoints, centers)
        grouped, indices = output
        # A shared residual on both neighbors passes through the native max.
        # Neighborhood membership and relative-position encoding are unchanged.
        grouped = grouped + residual.T.unsqueeze(0).unsqueeze(-1)
        self.scene_index += 1
        return grouped, indices

    def check_batch(self, module, inputs, output):
        assert self.scene_index == self.inputs['point_clouds'].shape[0]
        self.inputs = self.anchors = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.inputs = self.anchors = None
