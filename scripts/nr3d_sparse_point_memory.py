"""Fine sparse neighborhood features pooled directly from points to superpoints."""

import torch
from torch import nn
import spconv.pytorch as sparse

from scripts.nr3d_point_voxel_mapping import point_voxel_mapping
from utils.scatter_util import deterministic_scatter_mean_dim0


class SparsePointSuperpointResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(6, 32), nn.LayerNorm(32), nn.ReLU())
        self.fine = sparse.SparseSequential(
            sparse.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key='fine'),
            nn.LayerNorm(32), nn.ReLU())
        self.down = sparse.SparseSequential(
            sparse.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key='scale'),
            nn.LayerNorm(64), nn.ReLU())
        self.coarse = sparse.SparseSequential(
            sparse.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key='coarse'),
            nn.LayerNorm(64), nn.ReLU())
        self.up = sparse.SparseSequential(
            sparse.SparseInverseConv3d(64, 32, 3, bias=False, indice_key='scale'),
            nn.LayerNorm(32), nn.ReLU())
        self.output = nn.Linear(64, 288, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, point_cloud, superpoints, centers):
        """One scene: same (N,6) sampled points, (N,) SP ids and (S,3) centers."""
        cells, inverse, local, _ = point_voxel_mapping(point_cloud)
        point_features = self.point_encoder(local)
        voxel_features = deterministic_scatter_mean_dim0(point_features, inverse, dim_size=len(cells))
        # Use ZYX coordinates together with the correspondingly ordered shape.
        coordinates = torch.cat((cells.new_zeros((len(cells), 1)), cells[:, [2, 1, 0]]), dim=1).int()
        spatial_shape = (cells.max(dim=0).values + 1)[[2, 1, 0]].tolist()
        source = sparse.SparseConvTensor(voxel_features, coordinates, spatial_shape, batch_size=1)
        fine = self.fine(source)
        restored = self.up(self.coarse(self.down(fine)))
        assert torch.equal(restored.indices, fine.indices)
        neighbors = restored.features + fine.features
        detail = torch.cat((point_features, neighbors.index_select(0, inverse)), dim=-1)
        pooled = deterministic_scatter_mean_dim0(detail, superpoints, dim_size=len(centers))
        return self.output(pooled)


class SparseSuperpointIntervention:
    """Attach the new memory at the existing superpoint feature input."""

    def __init__(self, model, addon):
        self.addon = addon
        self.inputs = None
        self.scene_index = 0
        self.handles = [model.register_forward_pre_hook(self.capture_inputs),
                        model.super_grouper.register_forward_hook(self.add_detail),
                        model.register_forward_hook(self.check_batch)]

    def capture_inputs(self, module, inputs):
        self.inputs = inputs[0]
        self.scene_index = 0

    def add_detail(self, module, inputs, output):
        scene = self.scene_index
        cloud = self.inputs['point_clouds'][scene]
        superpoints = self.inputs['superpoint'][scene]
        centers = inputs[1].squeeze(0)
        residual = self.addon(cloud, superpoints, centers)
        grouped, indices = output
        grouped = grouped + residual.T.unsqueeze(0).unsqueeze(-1)
        self.scene_index += 1
        return grouped, indices

    def check_batch(self, module, inputs, output):
        assert self.scene_index == self.inputs['point_clouds'].shape[0]
        self.inputs = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.inputs = None
