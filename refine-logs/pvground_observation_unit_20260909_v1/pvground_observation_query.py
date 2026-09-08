"""Content and observed support for the pinned six-source PV-Ground reader.

Grouping preserves OpenPCDet 233f849's QueryAndGroup operations. Its CUDA
ball query pads with the first index and stops at nsample. Counts below are
distinct SELECTED indices, not all points within a radius or instance quality.
No post-pooling masking, new sampling, labels, or new losses are introduced.
"""
import torch
from torch import nn
from torch.nn import functional as F
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils

from pvground_source_query import SourceQueryRead, SOURCE_NAMES, SOURCE_WIDTHS


OBSERVATION_WIDTHS = (10, 13, 13, 13, 13, 13)


@torch.no_grad()
def selected_support_state(indices, empty, relative_xyz, radius):
    """[has_support, selected_fraction, min/r, mean/r, max/r, cap_reached]."""
    distinct = indices != indices[:, :1]
    distinct[:, 0] = True
    distinct &= ~empty[:, None]
    count = distinct.sum(-1)
    distance = relative_xyz.detach().square().sum(1).sqrt() / radius
    weight = distinct.to(distance.dtype)
    mean = (distance * weight).sum(-1) / count.clamp_min(1)
    minimum = distance.masked_fill(~distinct, float('inf')).min(-1)[0]
    minimum = torch.where(empty, torch.zeros_like(minimum), minimum)
    maximum = (distance * weight).max(-1)[0]
    return torch.stack([(~empty).to(distance.dtype), count.to(distance.dtype) / indices.shape[1],
                        minimum, mean, maximum, (count == indices.shape[1]).to(distance.dtype)], -1)


class ObservedQueryAndGroup(nn.Module):
    """Same grouping values as the upstream module; record actual query outputs."""
    def __init__(self, original):
        super().__init__()
        assert isinstance(original, pointnet2_utils.QueryAndGroup) and original.use_xyz
        self.radius, self.nsample, self.use_xyz = original.radius, original.nsample, original.use_xyz
        self.train(original.training)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None):
        assert features is not None  # All five pinned VSA sources carry features.
        assert xyz.shape[0] == xyz_batch_cnt.sum()
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum()
        indices, empty = pointnet2_utils.ball_query(
            self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, indices, new_xyz_batch_cnt)
        grouped_xyz -= new_xyz.unsqueeze(-1)
        grouped_xyz[empty] = 0
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, indices, new_xyz_batch_cnt)
        grouped_features[empty] = 0
        result = torch.cat([grouped_xyz, grouped_features], dim=1)
        self.observation = selected_support_state(indices, empty, grouped_xyz, self.radius)
        return result, indices


@torch.no_grad()
def within_voxel_range(xyz, point_cloud_range):
    limits = xyz.new_tensor(point_cloud_range)
    return ((xyz >= limits[:3]) & (xyz < limits[3:])).all(-1).to(xyz.dtype)


@torch.no_grad()
def spatial_observation(aggregate, keypoints, point_cloud_range):
    assert len(aggregate.groupers) == 2
    # Range is the shared voxelization window, NOT validity of the raw-point source.
    return torch.cat([g.observation for g in aggregate.groupers] +
                     [within_voxel_range(keypoints[:, 1:4], point_cloud_range)[:, None]], -1)


@torch.no_grad()
def bev_observation(keypoints, batch_dict, voxel_size, point_cloud_range):
    """Range, interior stencil, four encoded-column flags, four actual weights.

Flags describe occupied columns of the encoded sparse tensor (with convolution
context), not direct raw-point observations. Corners can repeat after clamping;
their flags are not summed into a distinct-support count.
    """
    sparse = batch_dict['encoded_spconv_tensor']
    batch_size, _, height, width = batch_dict['spatial_features'].shape
    assert tuple(sparse.spatial_shape[1:]) == (height, width)
    occupied = torch.zeros((batch_size, height, width), dtype=torch.bool, device=keypoints.device)
    index = sparse.indices.long()
    occupied[index[:, 0], index[:, 2], index[:, 3]] = True
    stride = batch_dict['spatial_features_stride']
    x = (keypoints[:, 1] - point_cloud_range[0]) / voxel_size[0] / stride
    y = (keypoints[:, 2] - point_cloud_range[1]) / voxel_size[1] / stride
    floor_x, floor_y = x.floor().long(), y.floor().long()
    x0, x1 = floor_x.clamp(0, width - 1), (floor_x + 1).clamp(0, width - 1)
    y0, y1 = floor_y.clamp(0, height - 1), (floor_y + 1).clamp(0, height - 1)
    batch = keypoints[:, 0].long()
    flags = [occupied[batch, yy, xx].to(x.dtype) for yy, xx in [(y0, x0), (y1, x0), (y0, x1), (y1, x1)]]
    x0f, x1f, y0f, y1f = x0.to(x.dtype), x1.to(x.dtype), y0.to(y.dtype), y1.to(y.dtype)
    weights = [(x1f-x)*(y1f-y), (x1f-x)*(y-y0f), (x-x0f)*(y1f-y), (x-x0f)*(y-y0f)]
    interior = ((x >= 0) & (x < width-1) & (y >= 0) & (y < height-1)).to(x.dtype)
    return torch.stack([within_voxel_range(keypoints[:, 1:4], point_cloud_range), interior] + flags + weights, -1)


class ObservationQueryRead(SourceQueryRead):
    def __init__(self, fusion_linear, visual_attention):
        super().__init__(fusion_linear, visual_attention)
        # Direct zero parameters consume no random draws beyond control B.
        self.observation_keys = nn.ParameterList(nn.Parameter(torch.zeros(288, width)) for width in OBSERVATION_WIDTHS)
        self.observation_values = nn.ParameterList(nn.Parameter(torch.zeros(288, width)) for width in OBSERVATION_WIDTHS)
        self.use_observation = True

    def forward(self, query, source_features, source_position, observations):
        if not self.enabled:
            return torch.zeros_like(query)
        assert source_features.shape[-1] == sum(SOURCE_WIDTHS)
        assert source_position.shape == source_features.shape[:2] + (288,)
        assert query.shape[1] == source_features.shape[0]
        assert len(observations) == len(SOURCE_NAMES)
        values = [norm(project(part)) for part, project, norm in zip(
            torch.split(source_features, SOURCE_WIDTHS, dim=-1), self.projections, self.norms)]
        keys = list(values)
        if self.use_observation:
            for i, (state, width) in enumerate(zip(observations, OBSERVATION_WIDTHS)):
                assert state.shape == source_features.shape[:2] + (width,)
                keys[i] = keys[i] + F.linear(state, self.observation_keys[i])
                values[i] = values[i] + F.linear(state, self.observation_values[i])
        memory = torch.cat(values, dim=1).transpose(0, 1).contiguous()
        key = torch.cat(keys, dim=1).transpose(0, 1).contiguous()
        positions = source_position.repeat(1, len(SOURCE_WIDTHS), 1).transpose(0, 1)
        evidence = self.attention(query=query, key=key + positions, value=memory, need_weights=False)[0]
        return self.output(evidence)


def install_observation_query_read(model):
    """Install after strict native parent loading; preserve all pretrained keys."""
    last, vsa = model.decoder[-1], model.backbone_net.vsa
    assert last.source_query_read is None and tuple(vsa.SA_layer_names) == SOURCE_NAMES[2:]
    assert vsa.n_output_features == 288 and vsa.n_keypoints == 1024
    before = list(vsa.state_dict())
    for aggregate in [vsa.SA_rawpoints] + list(vsa.SA_layers):
        assert len(aggregate.groupers) == 2
        aggregate.groupers = nn.ModuleList(ObservedQueryAndGroup(g) for g in aggregate.groupers)
        aggregate.groupers.train(aggregate.training)
    assert list(vsa.state_dict()) == before
    last.source_query_read = ObservationQueryRead(vsa.vsa_point_feature_fusion[0], last.cross_v)
    last.source_query_read.train(last.training)
