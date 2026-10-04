"""Whole predicted Mask range plus the existing local support, one native box."""
import numpy as np
import torch
from torch import nn

from pvground_tail_support_box_refiner import TailSupportBoxRefiner
from whole_mask_range import member_statistics, mask_range_evidence


class WholeMaskSupportBoxRefiner(TailSupportBoxRefiner):
    def __init__(self, use_whole_range, d_model=288):
        super().__init__(use_fused_support=True, d_model=d_model)
        self.use_whole_range = use_whole_range
        self.aggregate = nn.Sequential(nn.Linear(d_model + 896 + 3 + 6 + 109, 288), nn.ReLU())

    def whole_range(self, raw_points, coarse_center, coarse_size, end_points):
        evidence, coarse = [], []
        xyz = raw_points[..., :3].detach().cpu().numpy().astype(np.float64)
        for bid in range(len(raw_points)):
            ids = end_points['superpoints'][bid].detach().cpu().numpy()
            geometry = member_statistics(xyz[bid], ids, bins=32)
            values = mask_range_evidence(end_points['last_pred_masks'][bid][0],
                end_points['sp_last_pred_masks'][bid], end_points['adaptive_weights'][bid], geometry)
            evidence.append(torch.cat([values['axis_profile'].flatten(1),
                values['mean_normalized'], values['second_normalized'],
                values['member_lower_mean_normalized'], values['member_upper_mean_normalized'],
                values['support_fraction'][:, None]], dim=-1))
            origin = torch.as_tensor(geometry['origin'], device=coarse_center.device, dtype=coarse_center.dtype)
            span = torch.as_tensor(geometry['span'], device=coarse_center.device, dtype=coarse_center.dtype)
            coarse.append(torch.cat([(coarse_center[bid].detach() - origin) / span,
                                     coarse_size[bid].detach() / span], dim=-1))
        return torch.stack(evidence), torch.stack(coarse)

    def forward(self, query, raw_points, coarse_center, coarse_size, end_points):
        batch, count, _ = query.shape
        assert raw_points.shape == (batch, 50000, 6) and count == 256
        center = coarse_center.detach()
        layout_size = coarse_size.detach().clamp(min=1e-6)
        locations = center[:, :, None] + .5 * layout_size[:, :, None] * self.locations
        indices, distances = self.nearest_members(raw_points[..., :3], locations.reshape(batch, count * 7, 3))
        batch_index = torch.arange(batch, device=query.device)[:, None, None]
        members = raw_points[batch_index, indices].reshape(batch, count, 7, self.neighbors, 6)
        distances = distances.reshape(batch, count, 7, self.neighbors, 1)
        support = self.member_support(indices, end_points, batch, count)
        member_input = torch.cat([members[..., 3:], members[..., :3] - center[:, :, None, None],
            members[..., :3] - locations[:, :, :, None], distances, support], dim=-1)
        feature = torch.relu(self.member(member_input) + self.condition(query)[:, :, None, None])
        pooled = torch.cat([feature.mean(dim=-2), feature.max(dim=-2).values], dim=-1)
        whole, coarse = self.whole_range(raw_points, coarse_center, coarse_size, end_points)
        end_points['whole_mask_range_evidence'] = whole
        if not self.use_whole_range:
            whole = torch.zeros_like(whole)
        combined = torch.cat([query, pooled.reshape(batch, count, 896), layout_size, coarse, whole], dim=-1)
        residual = self.output(self.aggregate(combined))
        end_points['p3_coarse_center'] = coarse_center
        end_points['p3_coarse_size'] = coarse_size
        end_points['p3_neighbor_distances'] = distances.detach()
        return coarse_center + residual[..., :3], coarse_size + residual[..., 3:]


def install_whole_mask_refinement(model, use_whole_range):
    assert model.candidate_box_refiner is None
    model.candidate_box_refiner = WholeMaskSupportBoxRefiner(use_whole_range)
