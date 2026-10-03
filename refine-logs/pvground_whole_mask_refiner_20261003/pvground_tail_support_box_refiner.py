"""Same-tail candidate refinement with optional native predicted Mask evidence."""
import torch
from torch import nn
from pvground_candidate_box_refiner import CandidateAlignedBoxRefiner


class TailSupportBoxRefiner(CandidateAlignedBoxRefiner):
    """The paired arms differ only in four continuous support input channels."""

    def __init__(self, use_fused_support, d_model=288):
        super().__init__(d_model=d_model)
        self.use_fused_support = use_fused_support
        self.member[0] = nn.Linear(14, 64)

    def member_support(self, indices, end_points, batch, count):
        """Read only selected point memberships; never expand all Q x N points."""
        support = []
        point_indices = indices.reshape(batch, count, 7, self.neighbors)
        for bid in range(batch):
            superpoint_ids = end_points['superpoints'][bid][point_indices[bid]]
            query_ids = torch.arange(count, device=indices.device)[:, None, None]
            text_logits = end_points['last_pred_masks'][bid][0, query_ids, superpoint_ids]
            query_logits = end_points['sp_last_pred_masks'][bid][query_ids, superpoint_ids]
            alpha = end_points['adaptive_weights'][bid]
            fused_logits = alpha * text_logits + (1 - alpha) * query_logits
            text_probability = text_logits.sigmoid()
            query_probability = query_logits.sigmoid()
            support.append(torch.stack([
                text_probability, query_probability,
                (text_probability - query_probability).abs(), fused_logits.sigmoid()
            ], dim=-1))
        return torch.stack(support)

    def forward(self, query, raw_points, coarse_center, coarse_size, end_points):
        batch, count, _ = query.shape
        assert raw_points.shape == (batch, 50000, 6)
        center = coarse_center.detach()
        layout_size = coarse_size.detach().clamp(min=1e-6)
        locations = center[:, :, None] + .5 * layout_size[:, :, None] * self.locations
        indices, distances = self.nearest_members(
            raw_points[..., :3], locations.reshape(batch, count * 7, 3))
        batch_index = torch.arange(batch, device=query.device)[:, None, None]
        members = raw_points[batch_index, indices].reshape(batch, count, 7, self.neighbors, 6)
        distances = distances.reshape(batch, count, 7, self.neighbors, 1)
        # Both arms execute the native gather. Zeroing these slots is the paired
        # control intervention; it is not a fallback for missing Mask output.
        support = self.member_support(indices, end_points, batch, count)
        if not self.use_fused_support:
            support = torch.zeros_like(support)
        member_input = torch.cat([
            members[..., 3:], members[..., :3] - center[:, :, None, None],
            members[..., :3] - locations[:, :, :, None], distances, support
        ], dim=-1)
        feature = torch.relu(self.member(member_input) + self.condition(query)[:, :, None, None])
        pooled = torch.cat([feature.mean(dim=-2), feature.max(dim=-2).values], dim=-1)
        evidence = torch.cat([query, pooled.reshape(batch, count, 7 * 128), layout_size], dim=-1)
        residual = self.output(self.aggregate(evidence))
        end_points['p3_coarse_center'] = coarse_center
        end_points['p3_coarse_size'] = coarse_size
        end_points['p3_neighbor_distances'] = distances.detach()
        return coarse_center + residual[..., :3], coarse_size + residual[..., 3:]


def install_tail_support_refinement(model, use_fused_support):
    assert model.candidate_box_refiner is None
    model.candidate_box_refiner = TailSupportBoxRefiner(use_fused_support)
