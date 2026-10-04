"""Controlled six-face distribution adapter; unchanged whole/local support."""
import torch
from torch import nn
from torch.nn import functional as F

from pvground_whole_mask_box_refiner import WholeMaskSupportBoxRefiner


REG_MAX = 32
REG_SCALE = 4.0
SIZE_FLOOR = 1e-6
_step = 3.0 ** (2.0 / (REG_MAX - 2))
# D-FINE-style nonuniform coordinate knots. Face order: xyz-, xyz+.
KNOTS = tuple([-4.0] + [1.0 - _step ** i for i in range(15, 0, -1)]
              + [0.0] + [_step ** i - 1.0 for i in range(1, 16)] + [4.0])


def expected_offset(logits):
    knots = logits.new_tensor(KNOTS)
    probability = logits.softmax(-1)
    # Center the floating-point uniform reference, so zero logits are exactly zero.
    return ((probability - 1.0 / len(KNOTS)) * knots).sum(-1)


def decode_faces(center, size, offset):
    reference = size.clamp(min=SIZE_FLOOR)
    negative, positive = offset[..., :3], offset[..., 3:]
    final_center = center + .5 * (positive - negative) * reference / REG_SCALE
    raw_size = reference * (1.0 + (positive + negative) / REG_SCALE)
    return final_center, raw_size.clamp(min=SIZE_FLOOR), raw_size


def face_targets(center, size, gt_center, gt_size):
    reference = size.clamp(min=SIZE_FLOOR)
    low, high = gt_center - gt_size / 2, gt_center + gt_size / 2
    return torch.cat([(center - low) / reference * REG_SCALE - REG_SCALE / 2,
                      (high - center) / reference * REG_SCALE - REG_SCALE / 2], -1)


def distribution_loss(predictions, batch, indices):
    """Use the actual native final-layer Hungarian assignments, including all GTs."""
    logits, targets = [], []
    for bid, (queries, gt_indices) in enumerate(indices):
        valid = batch['box_label_mask'][bid].bool()
        center = batch['center_label'][bid, valid, :3][gt_indices]
        size = batch['size_gts'][bid, valid][gt_indices]
        logits.append(predictions['boundary_logits'][bid, queries])
        targets.append(face_targets(predictions['p3_coarse_center'][bid, queries].detach(),
                                    predictions['p3_coarse_size'][bid, queries].detach(), center, size))
    logits = torch.cat(logits).reshape(-1, REG_MAX + 1)
    target = torch.cat(targets).detach().reshape(-1)
    assert logits.shape[0] == target.numel() and target.numel() > 0
    assert torch.isfinite(target).all()
    knots = logits.new_tensor(KNOTS)
    outside = (target < knots[0]) | (target > knots[-1])
    target = target.clamp(min=KNOTS[0], max=KNOTS[-1])
    left = (target[:, None] >= knots[None]).sum(-1).sub(1).clamp(0, REG_MAX - 1)
    right_weight = (target - knots[left]) / (knots[left + 1] - knots[left])
    losses = ((1 - right_weight) * F.cross_entropy(logits, left, reduction='none')
              + right_weight * F.cross_entropy(logits, left + 1, reduction='none'))
    # Independent budget: mean over native matched boxes and their six faces.
    return losses.mean(), dict(boundary_matched_boxes=target.numel() // 6,
                               boundary_faces=target.numel(),
                               boundary_targets_finite=True,
                               boundary_target_outside=int(outside.sum()))


class BoundaryBoxRefiner(WholeMaskSupportBoxRefiner):
    def __init__(self, boundary_mode):
        assert boundary_mode in ('residual', 'distribution')
        super().__init__(use_whole_range=True)
        self.boundary_mode = boundary_mode
        self.output = nn.Linear(288, 6 if boundary_mode == 'residual' else 6 * (REG_MAX + 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, raw_points, coarse_center, coarse_size, end_points):
        # Same 1302-input support as the completed whole_range arm.
        batch, count, _ = query.shape
        assert raw_points.shape == (batch, 50000, 6) and count == 256
        center = coarse_center.detach()
        layout_size = coarse_size.detach().clamp(min=SIZE_FLOOR)
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
        combined = torch.cat([query, pooled.reshape(batch, count, 896), layout_size, coarse, whole], dim=-1)
        output = self.output(self.aggregate(combined))
        end_points['whole_mask_range_evidence'] = whole
        end_points['p3_coarse_center'] = coarse_center
        end_points['p3_coarse_size'] = coarse_size
        end_points['p3_neighbor_distances'] = distances.detach()
        if self.boundary_mode == 'residual':
            final_center = coarse_center + output[..., :3]
            raw_size = coarse_size.clamp(min=SIZE_FLOOR) + output[..., 3:]
            final_size = raw_size.clamp(min=SIZE_FLOOR)
        else:
            logits = output.reshape(batch, count, 6, REG_MAX + 1)
            offset = expected_offset(logits)
            final_center, final_size, raw_size = decode_faces(coarse_center, coarse_size, offset)
            end_points['boundary_logits'] = logits
        end_points['boundary_size_floor_count'] = (raw_size <= SIZE_FLOOR).sum().detach()
        end_points['boundary_size_floored'] = (raw_size <= SIZE_FLOOR).detach()
        return final_center, final_size


def install_boundary_refinement(model, boundary_mode):
    assert model.candidate_box_refiner is None
    model.candidate_box_refiner = BoundaryBoxRefiner(boundary_mode)
