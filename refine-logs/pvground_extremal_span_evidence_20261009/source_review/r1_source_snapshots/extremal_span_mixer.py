"""Prediction-only boundary provenance and bounded axis-wise geometry mixing.

Prepared source only. The existing frozen parent supplies native regression,
corrected fused-Mask references and the same native Query score. This module
has no score head, GT input, new candidate pruning or new reference fallback.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ExtremalSpanMixer(nn.Module):
    def __init__(self, source_mode):
        super().__init__()
        assert source_mode in ('whole_support', 'extremal_support')
        self.source_mode = source_mode
        self.query_projection = nn.Linear(288, 32)
        self.support_projection = nn.Linear(288, 32)
        self.face_encoder = nn.Sequential(nn.Linear(41, 32), nn.GELU(),
                                          nn.Linear(32, 32), nn.GELU())
        self.axis_decoder = nn.Sequential(nn.Linear(105, 64), nn.GELU(),
                                          nn.Linear(64, 32), nn.GELU())
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def mix_one(self, query, support, native_center, native_size, mask_center,
                mask_size, text, own, alpha, geometry, reference_valid):
        assert query.shape == (256, 288) and support.shape == (288, own.shape[1])
        assert text.shape == own.shape and own.shape[0] == 256 and alpha.ndim == 0
        assert native_center.shape == native_size.shape == mask_center.shape == mask_size.shape == (256, 3)
        assert reference_valid.shape == (256,) and reference_valid.dtype == torch.bool

        # Parents are frozen. The native evaluator/layout already floors the
        # observed nonpositive native sizes; use its same1e-6 definition here.
        native_center = native_center.detach()
        native_size = native_size.detach().clamp_min(1e-6)
        mask_center, mask_size = mask_center.detach(), mask_size.detach()
        assert torch.isfinite(mask_center).all() and (mask_size > 0).all()
        slots = torch.as_tensor(geometry['native_ids'], device=query.device, dtype=torch.long)
        count = query.new_tensor(geometry['count'])
        assert (count > 0).all()
        lower = query.new_tensor(geometry['origin'] + geometry['lower'] * geometry['span'])
        upper = query.new_tensor(geometry['origin'] + geometry['upper'] * geometry['span'])
        mean = query.new_tensor(geometry['origin'] + geometry['mean'] * geometry['span'])
        text, own, alpha = text.detach(), own.detach(), alpha.detach()
        t_prob, q_prob = text[:, slots].sigmoid(), own[:, slots].sigmoid()
        f_prob = (alpha * text[:, slots] + (1 - alpha) * own[:, slots]).sigmoid()
        active = (f_prob > .5) & reference_valid[:, None]
        low = lower[None].expand(256, -1, -1).masked_fill(~active[..., None], float('inf')).min(1).values
        high = upper[None].expand(256, -1, -1).masked_fill(~active[..., None], float('-inf')).max(1).values
        foreground_count = (active.to(query.dtype) * count[None]).sum(1)

        mask_state = torch.stack([t_prob, q_prob, f_prob, (t_prob - q_prob).abs(),
                                  alpha.expand_as(q_prob),
                                  (count.log1p() / math.log1p(50000))[None].expand(256, -1)], -1)
        support_content = F.gelu(self.support_projection(support[:, slots].T.detach()))
        query_content = F.gelu(self.query_projection(query.detach()))
        # Scale is positive by the existing reference/native size definition.
        scale = native_size + mask_size
        tokens, fractions = [], []
        for axis in range(3):
            for positive in (False, True):
                observed = upper[:, axis] if positive else lower[:, axis]
                extreme = high[:, axis] if positive else low[:, axis]
                selected = active
                if self.source_mode == 'extremal_support':
                    # Compare the very tensor reduced above, not center+size
                    # reconstructions. All tied source SPs remain represented.
                    selected = active & (observed[None] == extreme[:, None])
                weights = selected.to(query.dtype) * count[None]
                total = weights.sum(1)
                # Empty evidence is actually observed in39 selected references.
                # Zero numerator/one denominator defines its zero evidence token;
                # it does not create another candidate or choose another box.
                normalized = weights / total.clamp_min(1)[:, None]
                content = normalized @ support_content
                masks = (normalized[..., None] * mask_state).sum(1)
                present = normalized.sum(1)
                face = mask_center[:, axis] + (.5 if positive else -.5) * mask_size[:, axis]
                position = torch.stack([
                    (normalized @ value[:, axis] - face * present) / scale[:, axis]
                    for value in (mean, lower, upper)], -1)
                tokens.append(self.face_encoder(torch.cat([content, masks, position], -1)))
                fractions.append(total / foreground_count.clamp_min(1))
        face_tokens = torch.stack(tokens, 1)
        source_fraction = torch.stack(fractions, 1).reshape(256, 3, 2)
        paired_faces = face_tokens.reshape(256, 3, 64)
        geometry_state = torch.stack([(native_center - mask_center) / scale,
                                      native_size / scale, mask_size / scale], -1)
        axis_identity = torch.eye(3, device=query.device, dtype=query.dtype)[None].expand(256, -1, -1)
        decoder_input = torch.cat([paired_faces, query_content[:, None].expand(-1, 3, -1),
                                   geometry_state, source_fraction,
                                   reference_valid.to(query.dtype)[:, None, None].expand(-1, 3, -1),
                                   axis_identity], -1)
        assert decoder_input.shape == (256, 3, 105)
        raw_gate = self.output(self.axis_decoder(decoder_input)).squeeze(-1)
        # A bounded trust coefficient is the parameterization, not a score gate.
        # The same coefficient for center and size keeps each axis ordered.
        gate = raw_gate.clamp(0, 1)
        center = (1 - gate) * mask_center + gate * native_center
        size = (1 - gate) * mask_size + gate * native_size
        return center, size, dict(axis_gate=gate, raw_axis_gate=raw_gate,
                                  face_tokens=face_tokens, source_fraction=source_fraction,
                                  reference_valid=reference_valid)

    def forward(self, query, supports, predictions, geometries):
        batch = len(query)
        assert query.shape == (batch, 256, 288) and len(supports) == len(geometries) == batch
        centers, sizes, evidence = [], [], []
        for bid in range(batch):
            center, size, current = self.mix_one(query[bid], supports[bid],
                predictions['native_coarse_center'][bid], predictions['native_coarse_size'][bid],
                predictions['last_center'][bid], predictions['last_pred_size'][bid],
                predictions['last_pred_masks'][bid][0], predictions['sp_last_pred_masks'][bid],
                predictions['adaptive_weights'][bid], geometries[bid],
                predictions['mask_reference_valid'][bid])
            centers.append(center)
            sizes.append(size)
            evidence.append(current)
        return torch.stack(centers), torch.stack(sizes), evidence
