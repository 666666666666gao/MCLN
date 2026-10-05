"""Own/fused support moves the reference of the existing six-face head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pvground_boundary_box_refiner import BoundaryBoxRefiner, SIZE_FLOOR
from whole_mask_range import member_statistics, mask_range_evidence


def pack_range(values):
    return torch.cat([values['axis_profile'].flatten(1),
        values['mean_normalized'], values['second_normalized'],
        values['member_lower_mean_normalized'], values['member_upper_mean_normalized'],
        values['support_fraction'][:, None]], -1)


def own_range(query_logits, geometry):
    """Same probability/member weighting, with candidate's own Mask only."""
    slots = torch.as_tensor(geometry['native_ids'], device=query_logits.device, dtype=torch.long)
    constants = {name: torch.as_tensor(geometry[name], device=query_logits.device,
        dtype=query_logits.dtype) for name in ('count', 'histogram', 'mean', 'second', 'lower', 'upper')}
    count = constants['count']
    observed = query_logits[:, slots]
    weight = (F.logsigmoid(observed) + count.log()).softmax(-1)
    profile = (weight @ (constants['histogram'] / count[:, None, None]).flatten(1)).reshape(
        len(query_logits), 3, geometry['histogram'].shape[-1])
    return dict(axis_profile=profile, mean_normalized=weight @ constants['mean'],
        second_normalized=weight @ constants['second'],
        member_lower_mean_normalized=weight @ constants['lower'],
        member_upper_mean_normalized=weight @ constants['upper'],
        support_fraction=(observed.sigmoid() @ count) / count.sum())


class SupportReferenceBoxRefiner(BoundaryBoxRefiner):
    def __init__(self):
        super().__init__('distribution')
        self.reference = nn.Linear(512, 6)
        nn.init.zeros_(self.reference.weight)
        nn.init.zeros_(self.reference.bias)

    def forward(self, query, raw_points, coarse_center, coarse_size, end_points):
        own, fused, normalized = [], [], []
        xyz = raw_points[..., :3].detach().cpu().numpy().astype(np.float64)
        for bid in range(len(query)):
            geometry = member_statistics(xyz[bid], end_points['superpoints'][bid].detach().cpu().numpy(), bins=32)
            own.append(pack_range(own_range(end_points['sp_last_pred_masks'][bid], geometry)))
            fused.append(pack_range(mask_range_evidence(end_points['last_pred_masks'][bid][0],
                end_points['sp_last_pred_masks'][bid], end_points['adaptive_weights'][bid], geometry)))
            origin = coarse_center.new_tensor(geometry['origin'])
            span = coarse_center.new_tensor(geometry['span'])
            normalized.append(torch.cat([(coarse_center[bid].detach() - origin) / span,
                coarse_size[bid].detach() / span], -1))
        own, fused, normalized = torch.stack(own), torch.stack(fused), torch.stack(normalized)
        change = self.reference(torch.cat([query, own, fused, normalized], -1))
        scale = coarse_size.clamp(min=SIZE_FLOOR)
        reference_center = coarse_center + scale * change[..., :3]
        # Additive form preserves the actual raw coarse size at exactly zero.
        reference_size = coarse_size + scale * torch.expm1(change[..., 3:])
        end_points['native_coarse_center'] = coarse_center
        end_points['native_coarse_size'] = coarse_size
        end_points['support_reference_center'] = reference_center
        end_points['support_reference_size'] = reference_size.clamp(min=SIZE_FLOOR)
        end_points['own_mask_range_evidence'] = own
        end_points['support_reference_change'] = change
        return super().forward(query, raw_points, reference_center, reference_size, end_points)


def install_support_reference(model):
    old = model.candidate_box_refiner
    assert isinstance(old, BoundaryBoxRefiner) and old.boundary_mode == 'distribution'
    replacement = SupportReferenceBoxRefiner()
    loaded = replacement.load_state_dict(old.state_dict(), strict=False)
    assert set(loaded.missing_keys) == {'reference.weight', 'reference.bias'}
    assert not loaded.unexpected_keys
    assert all(torch.equal(value, replacement.state_dict()[name]) for name, value in old.state_dict().items())
    assert sum(parameter.numel() for parameter in replacement.parameters()) == 459180
    model.candidate_box_refiner = replacement


def reference_localization_loss(predictions, batch, indices, qualified, set_criterion, roots):
    """Supervise reference coordinates; original matching/extra set are reused."""
    boxes = torch.cat([predictions['support_reference_center'], predictions['support_reference_size']], -1)
    targets = []
    for bid in range(len(boxes)):
        valid = batch['box_label_mask'][bid].bool()
        targets.append({'boxes': torch.cat([batch['center_label'][bid, valid, :3], batch['size_gts'][bid, valid]], -1)})
    number = sum(len(queries) for queries, _ in indices)
    assert number > 0
    matched = set_criterion.loss_boxes({'pred_boxes': boxes}, targets, indices, float(number), None)
    matched_loss = (10 * matched['loss_bbox'] + 2 * matched['loss_giou']) / 7
    extra_loss = boxes.sum() * 0
    for bid, queries in enumerate(qualified):
        if queries.numel() == 0:
            continue
        loss = set_criterion.loss_boxes({'pred_boxes': boxes[bid:bid + 1]}, [{'boxes': roots[bid:bid + 1]}],
            [(queries, torch.zeros_like(queries))], float(queries.numel()), None)
        extra_loss = extra_loss + (10 * loss['loss_bbox'] + 2 * loss['loss_giou']) / 7
    return matched_loss + extra_loss / len(boxes)
