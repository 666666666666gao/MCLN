"""Shared query-wise mask fusion helpers for MCLN."""

import math

import torch
from torch import nn


QUERY_MASK_SOURCE_EVIDENCE_NAMES = (
    "text_mask_probability_mean",
    "query_mask_probability_mean",
    "text_mask_probability_std",
    "query_mask_probability_std",
    "text_mask_confidence",
    "query_mask_confidence",
    "text_mask_foreground_ratio",
    "query_mask_foreground_ratio",
    "text_query_probability_l1",
    "text_query_hard_disagreement",
)
QUERY_MASK_SOURCE_EVIDENCE_DIM = len(QUERY_MASK_SOURCE_EVIDENCE_NAMES)


def as_query_mask_logits(value, name="mask logits"):
    """Normalize one sample's mask logits to ``[Q,S]``."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a tensor".format(name))
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(
            "{} must have shape [Q,S] or [1,Q,S]".format(name)
        )
    return value


def build_query_mask_source_evidence(text_mask_logits, query_mask_logits):
    """Summarize source-specific mask evidence without target annotations."""
    if not isinstance(text_mask_logits, (list, tuple)) or not isinstance(
            query_mask_logits, (list, tuple)):
        raise ValueError("mask logits must be per-sample lists")
    if not text_mask_logits or len(text_mask_logits) != len(query_mask_logits):
        raise ValueError("text and query mask batches must be nonempty and align")

    rows = []
    expected_query_count = None
    expected_device = None
    for text_row, query_row in zip(text_mask_logits, query_mask_logits):
        text_row = as_query_mask_logits(text_row, "text mask logits")
        query_row = as_query_mask_logits(query_row, "query mask logits")
        if text_row.shape != query_row.shape or text_row.shape[1] < 1:
            raise ValueError("text and query mask rows must align as [Q,S]")
        if (not text_row.is_floating_point()
                or not query_row.is_floating_point()
                or text_row.device != query_row.device):
            raise ValueError("mask source rows must be floating tensors on one device")
        if (not bool(torch.isfinite(text_row).all().item())
                or not bool(torch.isfinite(query_row).all().item())):
            raise ValueError("mask source logits must be finite")
        if expected_query_count is None:
            expected_query_count = text_row.shape[0]
            expected_device = text_row.device
        elif (text_row.shape[0] != expected_query_count
              or text_row.device != expected_device):
            raise ValueError("mask source batches must share query count and device")

        text_probability = text_row.float().sigmoid()
        query_probability = query_row.float().sigmoid()
        text_foreground = text_probability > 0.5
        query_foreground = query_probability > 0.5
        evidence = torch.stack((
            text_probability.mean(dim=1),
            query_probability.mean(dim=1),
            text_probability.std(dim=1, unbiased=False),
            query_probability.std(dim=1, unbiased=False),
            (2.0 * (text_probability - 0.5).abs()).mean(dim=1),
            (2.0 * (query_probability - 0.5).abs()).mean(dim=1),
            text_foreground.float().mean(dim=1),
            query_foreground.float().mean(dim=1),
            (text_probability - query_probability).abs().mean(dim=1),
            (text_foreground != query_foreground).float().mean(dim=1),
        ), dim=-1)
        rows.append(evidence)

    result = torch.stack(rows, dim=0)
    if (result.shape[-1] != QUERY_MASK_SOURCE_EVIDENCE_DIM
            or not bool(torch.isfinite(result).all().item())
            or bool(((result < 0.0) | (result > 1.0)).any().item())):
        raise RuntimeError("mask source evidence contract is invalid")
    return result


def query_fusion_weight(weight, query_count, reference):
    """Return a scalar or ``[Q,1]`` fusion weight for one sample."""
    if not isinstance(query_count, int) or isinstance(query_count, bool) \
            or query_count < 0:
        raise ValueError("query_count must be a non-negative integer")
    if not isinstance(reference, torch.Tensor) \
            or not reference.is_floating_point():
        raise TypeError("reference must be a floating-point tensor")
    weight = torch.as_tensor(
        weight, device=reference.device, dtype=reference.dtype
    )
    if weight.numel() == 1:
        normalized = weight.reshape(())
    else:
        squeezed = weight.squeeze()
        if squeezed.dim() != 1 or squeezed.shape[0] != query_count:
            raise ValueError(
                "adaptive weight must be scalar or query-wise [Q] / [Q,1]"
            )
        normalized = squeezed.reshape(query_count, 1)
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("adaptive mask weight must be finite")
    return normalized.clamp(0.0, 1.0)


def gather_query_fusion_weight(weight, query_indices, query_count, reference):
    """Gather query-wise weights while preserving legacy scalar weights."""
    normalized = query_fusion_weight(weight, query_count, reference)
    if normalized.dim() == 0:
        return normalized
    indices = torch.as_tensor(
        query_indices, dtype=torch.long, device=reference.device
    ).reshape(-1)
    if indices.numel() and bool(
            ((indices < 0) | (indices >= query_count)).any().item()):
        raise IndexError("query index is out of adaptive-weight range")
    return normalized.index_select(0, indices)


def fuse_query_mask_logits(text_logits, query_logits, weight):
    """Fuse one sample's ``[Q,S]`` logits using scalar or query weights."""
    text_logits = as_query_mask_logits(text_logits, "text mask logits")
    query_logits = as_query_mask_logits(query_logits, "query mask logits")
    if text_logits.shape != query_logits.shape:
        raise ValueError("text and query mask logits must align")
    if not text_logits.is_floating_point() or not query_logits.is_floating_point():
        raise TypeError("mask logits must be floating point")
    query_logits = query_logits.to(
        device=text_logits.device, dtype=text_logits.dtype
    )
    alpha = query_fusion_weight(weight, text_logits.shape[0], text_logits)
    return alpha * text_logits + (1.0 - alpha) * query_logits


def batched_query_fusion_weight(weight, batch_size, query_count, reference):
    """Return a weight broadcastable to batched ``[B,Q,S]`` logits."""
    if not isinstance(reference, torch.Tensor) \
            or not reference.is_floating_point():
        raise TypeError("reference must be a floating-point tensor")
    weight = torch.as_tensor(
        weight, device=reference.device, dtype=reference.dtype
    )
    if weight.numel() == 1:
        normalized = weight.reshape(1, 1, 1)
    elif tuple(weight.shape) in ((batch_size,), (batch_size, 1)):
        normalized = weight.reshape(batch_size, 1, 1)
    elif tuple(weight.shape) in (
            (batch_size, query_count), (batch_size, query_count, 1)):
        normalized = weight.reshape(batch_size, query_count, 1)
    elif batch_size == 1 and tuple(weight.shape) in (
            (query_count,), (query_count, 1)):
        normalized = weight.reshape(1, query_count, 1)
    else:
        raise ValueError(
            "alpha must be scalar, per-sample [B], or query-wise [B,Q]"
        )
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("adaptive mask weight must be finite")
    return normalized.clamp(0.0, 1.0)


def fuse_batched_query_mask_logits(text_logits, query_logits, weight):
    """Fuse batched ``[B,Q,S]`` logits with scalar or query weights."""
    if not isinstance(text_logits, torch.Tensor) or not isinstance(
            query_logits, torch.Tensor):
        raise TypeError("mask logits must be tensors")
    if text_logits.dim() != 3 or text_logits.shape != query_logits.shape:
        raise ValueError("mask logits must share shape [B,Q,S]")
    if not text_logits.is_floating_point() or not query_logits.is_floating_point():
        raise TypeError("mask logits must be floating point")
    query_logits = query_logits.to(
        device=text_logits.device, dtype=text_logits.dtype
    )
    alpha = batched_query_fusion_weight(
        weight, text_logits.shape[0], text_logits.shape[1], text_logits
    )
    return alpha * text_logits + (1.0 - alpha) * query_logits


def apply_query_mask_calibration(text_mask_logits, query_mask_logits,
                                 fusion_weights, logit_bias):
    """Apply one network-predicted weight and bias to every query mask."""
    if not isinstance(text_mask_logits, (list, tuple)) or not isinstance(
            query_mask_logits, (list, tuple)):
        raise ValueError("mask logits must be per-sample lists")
    if len(text_mask_logits) != len(query_mask_logits):
        raise ValueError("text and query mask batches must align")
    if (not isinstance(fusion_weights, torch.Tensor)
            or not isinstance(logit_bias, torch.Tensor)
            or fusion_weights.dim() != 2
            or logit_bias.shape != fusion_weights.shape
            or fusion_weights.shape[0] != len(text_mask_logits)
            or not fusion_weights.is_floating_point()
            or not logit_bias.is_floating_point()
            or fusion_weights.device != logit_bias.device
            or not bool(torch.isfinite(fusion_weights).all().item())
            or not bool(torch.isfinite(logit_bias).all().item())
            or bool(((fusion_weights < 0.0)
                     | (fusion_weights > 1.0)).any().item())):
        raise ValueError(
            "fusion weights and logit bias must align as finite [B,Q]"
        )

    calibrated_text = []
    calibrated_query = []
    calibrated_weights = []
    for batch_idx, (text_row, query_row) in enumerate(zip(
            text_mask_logits, query_mask_logits)):
        text_row = as_query_mask_logits(text_row, "text mask logits")
        query_row = as_query_mask_logits(query_row, "query mask logits")
        if text_row.shape != query_row.shape or (
                text_row.shape[0] != fusion_weights.shape[1]):
            raise ValueError("mask rows do not align with calibration queries")
        bias = logit_bias[batch_idx].to(
            device=text_row.device, dtype=text_row.dtype
        ).unsqueeze(-1)
        if query_row.device != text_row.device:
            raise ValueError("text and query mask rows must share a device")
        calibrated_text.append(text_row + bias)
        calibrated_query.append(query_row + bias.to(query_row.dtype))
        calibrated_weights.append(fusion_weights[batch_idx].to(
            device=text_row.device, dtype=text_row.dtype
        ))
    return calibrated_text, calibrated_query, calibrated_weights


def apply_query_superpoint_mask_residual(
        text_mask_logits, query_mask_logits, spatial_residuals):
    """Add one query-superpoint residual to both mask source logits."""
    if not isinstance(text_mask_logits, (list, tuple)) or not isinstance(
            query_mask_logits, (list, tuple)) or not isinstance(
            spatial_residuals, (list, tuple)):
        raise ValueError("mask logits and residuals must be per-sample lists")
    if not (
            len(text_mask_logits) == len(query_mask_logits)
            == len(spatial_residuals)):
        raise ValueError("mask logits and spatial residual batches must align")
    refined_text = []
    refined_query = []
    for text_row, query_row, residual in zip(
            text_mask_logits, query_mask_logits, spatial_residuals):
        text_row = as_query_mask_logits(text_row, "text mask logits")
        query_row = as_query_mask_logits(query_row, "query mask logits")
        residual = as_query_mask_logits(
            residual, "query-superpoint mask residual"
        )
        if (text_row.shape != query_row.shape
                or residual.shape != text_row.shape
                or text_row.device != query_row.device
                or residual.device != text_row.device
                or not residual.is_floating_point()
                or not bool(torch.isfinite(residual).all().item())):
            raise ValueError(
                "mask rows and spatial residual must align as finite [Q,S]"
            )
        refined_text.append(text_row + residual.to(text_row.dtype))
        refined_query.append(query_row + residual.to(query_row.dtype))
    return refined_text, refined_query


class QueryMaskFusionCalibrator(nn.Module):
    """Predict a bounded per-query residual around MCLN's scalar alpha."""

    def __init__(self, d_model, hidden_dim=128, dropout=0.0,
                 max_delta=0.25, detach_inputs=True):
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be positive")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) \
                or not math.isfinite(float(dropout)) \
                or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not isinstance(max_delta, (int, float)) \
                or isinstance(max_delta, bool) \
                or not math.isfinite(float(max_delta)) \
                or not 0.0 < float(max_delta) <= 1.0:
            raise ValueError("max_delta must lie in (0,1]")
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be boolean")

        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.max_delta = float(max_delta)
        self.detach_inputs = detach_inputs
        self.query_norm = nn.LayerNorm(self.d_model)
        self.text_norm = nn.LayerNorm(self.d_model)
        self.box_norm = nn.LayerNorm(6)
        self.encoder = nn.Sequential(
            nn.Linear(self.d_model * 2 + 6, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _pool_text(text_feats, text_padding_mask):
        valid = (~text_padding_mask.bool()).to(text_feats.dtype).unsqueeze(-1)
        return (text_feats * valid).sum(1) / valid.sum(1).clamp(min=1.0)

    @staticmethod
    def _normalize_boxes(boxes):
        center = boxes[..., :3]
        sizes = boxes[..., 3:].clamp(min=1e-4).log()
        center = center - center.mean(dim=1, keepdim=True)
        center_scale = center.square().mean(dim=1, keepdim=True).sqrt().clamp(
            min=1e-4
        )
        center = center / center_scale
        sizes = sizes - sizes.mean(dim=1, keepdim=True)
        return torch.cat((center, sizes), dim=-1)

    def forward(self, query_feats, text_feats, text_padding_mask, boxes,
                base_alpha):
        if not isinstance(query_feats, torch.Tensor) or query_feats.dim() != 3:
            raise ValueError("query_feats must have shape [B,Q,D]")
        if query_feats.shape[-1] != self.d_model:
            raise ValueError("query feature dimension does not match d_model")
        if not isinstance(text_feats, torch.Tensor) or text_feats.dim() != 3 \
                or text_feats.shape[0] != query_feats.shape[0] \
                or text_feats.shape[-1] != self.d_model:
            raise ValueError("text_feats must have shape [B,L,D]")
        if not isinstance(text_padding_mask, torch.Tensor) \
                or text_padding_mask.shape != text_feats.shape[:2]:
            raise ValueError("text_padding_mask must have shape [B,L]")
        if not isinstance(boxes, torch.Tensor) \
                or boxes.shape != query_feats.shape[:2] + (6,):
            raise ValueError("boxes must have shape [B,Q,6]")
        if any(value.device != query_feats.device for value in (
                text_feats, text_padding_mask, boxes)):
            raise ValueError("calibrator inputs must share a device")

        base_alpha = torch.as_tensor(
            base_alpha, device=query_feats.device, dtype=query_feats.dtype
        )
        if base_alpha.numel() != query_feats.shape[0]:
            raise ValueError("base_alpha must contain one scalar per sample")
        base_alpha = base_alpha.reshape(query_feats.shape[0], 1)
        if not bool(torch.isfinite(base_alpha).all().item()):
            raise ValueError("base_alpha must be finite")
        base_alpha = base_alpha.clamp(0.0, 1.0)

        if self.detach_inputs:
            query_feats = query_feats.detach()
            text_feats = text_feats.detach()
            boxes = boxes.detach()
            base_alpha = base_alpha.detach()
        text_context = self._pool_text(text_feats, text_padding_mask)
        text_context = text_context.unsqueeze(1).expand(
            -1, query_feats.shape[1], -1
        )
        encoded = self.encoder(torch.cat((
            self.query_norm(query_feats),
            self.text_norm(text_context),
            self.box_norm(self._normalize_boxes(boxes)),
        ), dim=-1))
        residual = self.max_delta * self.residual_head(encoded).squeeze(-1).tanh()
        weights = (base_alpha + residual).clamp(0.0, 1.0)
        return {
            "weights": weights,
            "residual": residual,
            "base_alpha": base_alpha,
        }
