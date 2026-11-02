"""Candidate-mask support correction and extremal-span box mixing."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def member_statistics(xyz, superpoint_ids):
    native_ids, inverse = np.unique(superpoint_ids, return_inverse=True)
    count = np.bincount(inverse)
    origin = xyz.min(axis=0)
    span = xyz.max(axis=0) - origin
    normalized = (xyz - origin) / span
    mean, lower, upper = [], [], []
    for axis in range(3):
        mean.append(np.bincount(
            inverse, weights=normalized[:, axis]
        ) / count)
        minimum = np.full(len(native_ids), np.inf)
        maximum = np.full(len(native_ids), -np.inf)
        np.minimum.at(minimum, inverse, normalized[:, axis])
        np.maximum.at(maximum, inverse, normalized[:, axis])
        lower.append(minimum)
        upper.append(maximum)
    return {
        "native_ids": native_ids,
        "count": count,
        "origin": origin,
        "span": span,
        "mean": np.stack(mean, axis=1),
        "lower": np.stack(lower, axis=1),
        "upper": np.stack(upper, axis=1),
    }


class CandidateMaskSupportCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = nn.Linear(288, 32)
        self.support_projection = nn.Linear(288, 32)
        self.member = nn.Sequential(
            nn.Linear(79, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def correct_one(self, query, support, center, size, text, own, alpha,
                    geometry):
        slots = torch.as_tensor(
            geometry["native_ids"], device=own.device, dtype=torch.long
        )
        mean = own.new_tensor(
            geometry["origin"] + geometry["mean"] * geometry["span"]
        )
        lower = own.new_tensor(
            geometry["origin"] + geometry["lower"] * geometry["span"]
        )
        upper = own.new_tensor(
            geometry["origin"] + geometry["upper"] * geometry["span"]
        )
        scale = size.detach().clamp_min(1e-6)[:, None]
        position = torch.cat([
            (value[None] - center.detach()[:, None]) / scale
            for value in (mean, lower, upper)
        ], dim=-1)
        query_content = F.gelu(self.query_projection(query)).unsqueeze(1)
        support_content = F.gelu(
            self.support_projection(support[:, slots].T)
        ).unsqueeze(0)
        text_prob = text[:, slots].sigmoid()
        own_prob = own[:, slots].sigmoid()
        fused_prob = (alpha * text[:, slots] + (1 - alpha) * own[:, slots]).sigmoid()
        mask_state = torch.stack([
            text_prob,
            own_prob,
            fused_prob,
            (text_prob - own_prob).abs(),
            alpha.expand_as(own_prob),
        ], dim=-1)
        count = own.new_tensor(geometry["count"])
        position = torch.cat([
            position,
            (count.log1p() / math.log1p(50000))[None, :, None].expand(
                256, -1, -1
            ),
        ], dim=-1)
        member_input = torch.cat([
            query_content.expand(-1, len(slots), -1),
            support_content.expand(256, -1, -1),
            mask_state,
            position,
        ], dim=-1)
        residual = self.output(self.member(member_input)).squeeze(-1)
        delta = torch.zeros_like(own)
        delta[:, slots] = residual
        return own + delta

    def forward(self, query, supports, raw_points, centers, sizes, predictions):
        xyz = raw_points[..., :3].detach().cpu().numpy().astype(np.float64)
        corrected, geometries = [], []
        for batch_index in range(len(query)):
            geometry = member_statistics(
                xyz[batch_index],
                predictions["superpoints"][batch_index].detach().cpu().numpy(),
            )
            geometries.append(geometry)
            corrected.append(self.correct_one(
                query[batch_index],
                supports[batch_index],
                centers[batch_index],
                sizes[batch_index],
                predictions["last_pred_masks"][batch_index][0],
                predictions["sp_last_pred_masks"][batch_index],
                predictions["adaptive_weights"][batch_index],
                geometry,
            ))
        return corrected, geometries


def mask_extent_reference(native_center, native_size, text_masks, own_masks,
                          alpha, geometries):
    centers, sizes, validities = [], [], []
    for batch_index, geometry in enumerate(geometries):
        slots = torch.as_tensor(
            geometry["native_ids"], device=native_center.device, dtype=torch.long
        )
        fused = (
            alpha[batch_index].detach() * text_masks[batch_index].detach()
            + (1 - alpha[batch_index].detach()) * own_masks[batch_index].detach()
        )
        active = fused[:, slots].sigmoid() > 0.5
        lower = native_center.new_tensor(
            geometry["origin"] + geometry["lower"] * geometry["span"]
        )
        upper = native_center.new_tensor(
            geometry["origin"] + geometry["upper"] * geometry["span"]
        )
        low = lower[None].expand(256, -1, -1).masked_fill(
            ~active[..., None], float("inf")
        ).min(dim=1).values
        high = upper[None].expand(256, -1, -1).masked_fill(
            ~active[..., None], float("-inf")
        ).max(dim=1).values
        valid = active.any(dim=1) & (high > low).all(dim=-1)
        center = torch.where(
            valid[:, None], (low + high) * 0.5, native_center[batch_index]
        )
        size = torch.where(
            valid[:, None], high - low, native_size[batch_index].clamp_min(1e-6)
        )
        centers.append(center)
        sizes.append(size)
        validities.append(valid)
    return torch.stack(centers), torch.stack(sizes), torch.stack(validities)


class ExtremalSpanMixer(nn.Module):
    def __init__(self, source_mode="extremal_support"):
        super().__init__()
        if source_mode not in ("whole_support", "extremal_support"):
            raise ValueError("unsupported span source mode")
        self.source_mode = source_mode
        self.query_projection = nn.Linear(288, 32)
        self.support_projection = nn.Linear(288, 32)
        self.face_encoder = nn.Sequential(
            nn.Linear(41, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.axis_decoder = nn.Sequential(
            nn.Linear(105, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def mix_one(self, query, support, native_center, native_size, mask_center,
                mask_size, text, own, alpha, geometry, reference_valid):
        native_size = native_size.clamp_min(1e-6)
        mask_center = mask_center.detach()
        mask_size = mask_size.detach().clamp_min(1e-6)
        slots = torch.as_tensor(
            geometry["native_ids"], device=query.device, dtype=torch.long
        )
        count = query.new_tensor(geometry["count"])
        lower = query.new_tensor(
            geometry["origin"] + geometry["lower"] * geometry["span"]
        )
        upper = query.new_tensor(
            geometry["origin"] + geometry["upper"] * geometry["span"]
        )
        mean = query.new_tensor(
            geometry["origin"] + geometry["mean"] * geometry["span"]
        )
        text_prob = text[:, slots].sigmoid()
        own_prob = own[:, slots].sigmoid()
        fused_prob = (alpha * text[:, slots] + (1 - alpha) * own[:, slots]).sigmoid()
        active = (fused_prob > 0.5) & reference_valid[:, None]
        low = lower[None].expand(256, -1, -1).masked_fill(
            ~active[..., None], float("inf")
        ).min(dim=1).values
        high = upper[None].expand(256, -1, -1).masked_fill(
            ~active[..., None], float("-inf")
        ).max(dim=1).values
        foreground_count = (active.to(query.dtype) * count[None]).sum(dim=1)

        mask_state = torch.stack([
            text_prob,
            own_prob,
            fused_prob,
            (text_prob - own_prob).abs(),
            alpha.expand_as(own_prob),
            (count.log1p() / math.log1p(50000))[None].expand(256, -1),
        ], dim=-1)
        support_content = F.gelu(self.support_projection(support[:, slots].T))
        query_content = F.gelu(self.query_projection(query))
        scale = native_size + mask_size
        tokens, fractions = [], []
        for axis in range(3):
            for positive in (False, True):
                observed = upper[:, axis] if positive else lower[:, axis]
                extreme = high[:, axis] if positive else low[:, axis]
                selected = active
                if self.source_mode == "extremal_support":
                    selected = active & (observed[None] == extreme[:, None])
                weights = selected.to(query.dtype) * count[None]
                total = weights.sum(dim=1)
                normalized = weights / total.clamp_min(1)[:, None]
                content = normalized @ support_content
                masks = (normalized[..., None] * mask_state).sum(dim=1)
                present = normalized.sum(dim=1)
                face = mask_center[:, axis] + (
                    0.5 if positive else -0.5
                ) * mask_size[:, axis]
                position = torch.stack([
                    (normalized @ value[:, axis] - face * present)
                    / scale[:, axis]
                    for value in (mean, lower, upper)
                ], dim=-1)
                tokens.append(self.face_encoder(torch.cat(
                    [content, masks, position], dim=-1
                )))
                fractions.append(total / foreground_count.clamp_min(1))

        face_tokens = torch.stack(tokens, dim=1)
        source_fraction = torch.stack(fractions, dim=1).reshape(256, 3, 2)
        paired_faces = face_tokens.reshape(256, 3, 64)
        geometry_state = torch.stack([
            (native_center - mask_center) / scale,
            native_size / scale,
            mask_size / scale,
        ], dim=-1)
        axis_identity = torch.eye(
            3, device=query.device, dtype=query.dtype
        )[None].expand(256, -1, -1)
        decoder_input = torch.cat([
            paired_faces,
            query_content[:, None].expand(-1, 3, -1),
            geometry_state,
            source_fraction,
            reference_valid.to(query.dtype)[:, None, None].expand(-1, 3, -1),
            axis_identity,
        ], dim=-1)
        assert decoder_input.shape == (256, 3, 105)
        raw_gate = self.output(self.axis_decoder(decoder_input)).squeeze(-1)
        mask_weight = raw_gate.clamp(0, 1)
        center = (1 - mask_weight) * native_center + mask_weight * mask_center
        size = (1 - mask_weight) * native_size + mask_weight * mask_size
        return center, size, {
            "axis_gate": mask_weight,
            "raw_axis_gate": raw_gate,
            "face_tokens": face_tokens,
            "source_fraction": source_fraction,
            "reference_valid": reference_valid,
        }

    def forward(self, query, supports, predictions, geometries):
        centers, sizes, evidence = [], [], []
        for batch_index in range(len(query)):
            center, size, current = self.mix_one(
                query[batch_index],
                supports[batch_index],
                predictions["native_coarse_center"][batch_index],
                predictions["native_coarse_size"][batch_index],
                predictions["mask_reference_center"][batch_index],
                predictions["mask_reference_size"][batch_index],
                predictions["last_pred_masks"][batch_index][0],
                predictions["sp_last_pred_masks"][batch_index],
                predictions["adaptive_weights"][batch_index],
                geometries[batch_index],
                predictions["mask_reference_valid"][batch_index],
            )
            centers.append(center)
            sizes.append(size)
            evidence.append(current)
        return torch.stack(centers), torch.stack(sizes), evidence
