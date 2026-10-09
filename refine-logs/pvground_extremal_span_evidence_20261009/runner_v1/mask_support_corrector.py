"""Candidate-specific correction of the native Query Mask, before logit fusion."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from whole_mask_range import member_statistics


class CandidateMaskSupportCorrector(nn.Module):
    def __init__(self, use_box_geometry):
        super().__init__()
        self.use_box_geometry = use_box_geometry
        self.query_projection = nn.Linear(288, 32)
        self.support_projection = nn.Linear(288, 32)
        self.member = nn.Sequential(nn.Linear(79, 64), nn.GELU(),
                                    nn.Linear(64, 64), nn.GELU())
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def correct_one(self, query, support, center, size, text, own, alpha, geometry):
        assert query.shape == (256, 288) and support.shape == (288, own.shape[1])
        assert text.shape == own.shape and own.shape[0] == 256 and alpha.ndim == 0
        slots = torch.as_tensor(geometry['native_ids'], device=own.device, dtype=torch.long)
        mean = own.new_tensor(geometry['origin'] + geometry['mean'] * geometry['span'])
        lower = own.new_tensor(geometry['origin'] + geometry['lower'] * geometry['span'])
        upper = own.new_tensor(geometry['origin'] + geometry['upper'] * geometry['span'])
        # The parent layout already floors observed nonpositive coarse sizes.
        scale = size.detach().clamp_min(1e-6)[:, None]
        position = torch.cat([(value[None] - center.detach()[:, None]) / scale
                              for value in (mean, lower, upper)], -1)
        if self.use_box_geometry:
            position = position.sign() * position.abs().log1p()
        if not self.use_box_geometry:
            position = torch.zeros_like(position)
        query_content = F.gelu(self.query_projection(query)).unsqueeze(1)
        support_content = F.gelu(self.support_projection(support[:, slots].T)).unsqueeze(0)
        q_prob, t_prob = own[:, slots].sigmoid(), text[:, slots].sigmoid()
        fused_prob = (alpha * text[:, slots] + (1 - alpha) * own[:, slots]).sigmoid()
        mask_state = torch.stack([t_prob, q_prob, fused_prob, (t_prob - q_prob).abs(),
                                  alpha.expand_as(q_prob)], -1)
        count = own.new_tensor(geometry['count']).log1p() / math.log1p(50000)
        member_input = torch.cat([query_content.expand(-1, len(slots), -1),
                                  support_content.expand(256, -1, -1), mask_state,
                                  count[None, :, None].expand(256, -1, -1), position], -1)
        residual = self.output(self.member(member_input)).squeeze(-1)
        delta = torch.zeros_like(own)
        delta[:, slots] = residual
        return own + delta

    def forward(self, query, supports, raw_points, center, size, predictions):
        batch = len(query)
        assert query.shape == (batch, 256, 288) and raw_points.shape == (batch, 50000, 6)
        assert len(supports) == batch
        corrected, geometries = [], []
        xyz = raw_points[..., :3].detach().cpu().numpy().astype(np.float64)
        for bid in range(batch):
            geometry = member_statistics(xyz[bid],
                predictions['superpoints'][bid].detach().cpu().numpy(), bins=32)
            geometries.append(geometry)
            corrected.append(self.correct_one(query[bid], supports[bid], center[bid], size[bid],
                predictions['last_pred_masks'][bid][0], predictions['sp_last_pred_masks'][bid],
                predictions['adaptive_weights'][bid], geometry))
        return corrected, geometries


def install_support_correction(model, use_box_geometry, state=None):
    assert model.candidate_support_corrector is None
    assert model.candidate_box_refiner.reference_mode == 'fused_mask'
    assert torch.count_nonzero(model.candidate_box_refiner.output.weight) == 0
    assert torch.count_nonzero(model.candidate_box_refiner.output.bias) == 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    corrector = CandidateMaskSupportCorrector(use_box_geometry)
    if state is not None:
        corrector.load_state_dict(state, strict=True)
    assert sum(parameter.numel() for parameter in corrector.parameters()) == 27841
    assert len(corrector.state_dict()) == 10
    model.candidate_support_corrector = corrector
    model.eval()
    return corrector
