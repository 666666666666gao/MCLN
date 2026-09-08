"""An explicit empty-neighborhood control for the existing PV-Ground VSA.

Grouping and pooling follow OpenPCDet 233f849829b6ac19afb8af8837a0246890908755,
pcdet/ops/pointnet2/pointnet2_stack/{pointnet2_utils,pointnet2_modules}.py.
The only candidate change is masking AFTER MLP/pooling, before source fusion.
This control adds no parameters and is not a learned coverage mechanism.
"""
import torch
from torch import nn
from torch.nn import functional as F
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG


class EmptyPoolMaskedVSA(nn.Module):
    def __init__(self, original, mask_empty):
        super().__init__()
        assert isinstance(original, StackSAModuleMSG)
        assert original.pool_method == 'max_pool'
        assert all(g.use_xyz for g in original.groupers)
        self.groupers = original.groupers
        self.mlps = original.mlps
        self.pool_method = original.pool_method
        self.mask_empty = bool(mask_empty)
        self.train(original.training)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features=None, empty_voxel_set_zeros=True):
        if not self.mask_empty:
            return StackSAModuleMSG.forward(self, xyz, xyz_batch_cnt, new_xyz,
                                            new_xyz_batch_cnt, features, empty_voxel_set_zeros)
        # All five current VSA sources provide features and use relative XYZ.
        assert features is not None
        assert xyz.shape[0] == xyz_batch_cnt.sum()
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum()
        pooled = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            indices, empty = pointnet2_utils.ball_query(
                grouper.radius, grouper.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
            grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, indices, new_xyz_batch_cnt)
            grouped_xyz -= new_xyz.unsqueeze(-1)
            grouped_xyz[empty] = 0
            grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, indices, new_xyz_batch_cnt)
            grouped_features[empty] = 0
            combined = torch.cat([grouped_xyz, grouped_features], dim=1)
            transformed = mlp(combined.permute(1, 0, 2).unsqueeze(0))
            value = F.max_pool2d(transformed, kernel_size=[1, transformed.size(3)]).squeeze(-1)
            value = value.squeeze(0).permute(1, 0)
            pooled.append(value * (~empty).to(value.dtype).unsqueeze(-1))
        return new_xyz, torch.cat(pooled, dim=1)


def install_empty_pool_mask(model, enabled):
    """Keep every pretrained state key while replacing the five VSA readers."""
    vsa = model.backbone_net.vsa
    before = list(vsa.state_dict())
    vsa.SA_rawpoints = EmptyPoolMaskedVSA(vsa.SA_rawpoints, enabled)
    for index in range(len(vsa.SA_layers)):
        vsa.SA_layers[index] = EmptyPoolMaskedVSA(vsa.SA_layers[index], enabled)
    assert list(vsa.state_dict()) == before
