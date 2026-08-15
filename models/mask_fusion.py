"""Shared query-wise mask fusion helpers for MCLN."""

import math

import torch
from torch import nn
from torch.nn import functional as F


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


class EvidenceGeometryQuerySuperpointMaskRefiner(nn.Module):
    """Zero-residual query-superpoint correction from generic mask evidence.

    The refiner is deliberately dataset agnostic: it consumes only frozen
    query/superpoint features, the two mask-source logits, predicted boxes,
    superpoint centers, and the existing fusion weight.  It never reads class,
    scene, or dataset identifiers.
    """

    COMPONENTS = ("content", "evidence", "geometry", "all")
    EVIDENCE_DIM = 7
    GEOMETRY_DIM = 8

    def __init__(self, d_model=288, hidden_dim=32, max_delta=2.0,
                 components="all", detach_inputs=True):
        super().__init__()
        for name, value in (("d_model", d_model), ("hidden_dim", hidden_dim)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value <= 0):
                raise ValueError("{} must be a positive integer".format(name))
        if (not isinstance(max_delta, (int, float))
                or isinstance(max_delta, bool)
                or not math.isfinite(float(max_delta))
                or float(max_delta) <= 0.0):
            raise ValueError("max_delta must be finite and positive")
        if components not in self.COMPONENTS:
            raise ValueError(
                "components must be one of {}".format(self.COMPONENTS)
            )
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be boolean")

        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.components = components
        self.detach_inputs = detach_inputs
        self.query_norm = nn.LayerNorm(self.d_model)
        self.superpoint_norm = nn.LayerNorm(self.d_model)
        self.query_projection = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.superpoint_projection = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.evidence_coefficients = nn.Linear(
            self.d_model, self.EVIDENCE_DIM
        )
        self.geometry_coefficients = nn.Linear(
            self.d_model, self.GEOMETRY_DIM
        )

        # Every output-producing head starts at exactly zero.  Loading a
        # parent checkpoint therefore preserves its masks bit-for-bit at step
        # zero, while gradients can immediately enter each component.
        nn.init.zeros_(self.query_projection[-1].weight)
        nn.init.zeros_(self.query_projection[-1].bias)
        nn.init.zeros_(self.evidence_coefficients.weight)
        nn.init.zeros_(self.evidence_coefficients.bias)
        nn.init.zeros_(self.geometry_coefficients.weight)
        nn.init.zeros_(self.geometry_coefficients.bias)

    @staticmethod
    def _superpoint_xyz(value, expected_count):
        if not isinstance(value, torch.Tensor):
            raise ValueError("superpoint xyz must be a tensor")
        if value.dim() == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.dim() != 2 or value.shape != (expected_count, 3):
            raise ValueError("superpoint xyz must have shape [S,3] or [1,S,3]")
        return value

    @staticmethod
    def _component_weight(components, name):
        return 1.0 if components in (name, "all") else 0.0

    def forward(self, query_features, superpoint_features, superpoint_xyz,
                boxes, text_mask_logits, query_mask_logits, fusion_weights):
        if (not isinstance(query_features, torch.Tensor)
                or query_features.dim() != 3
                or query_features.shape[-1] != self.d_model
                or not query_features.is_floating_point()):
            raise ValueError("query_features must be floating [B,Q,D]")
        batch_size, query_count, _ = query_features.shape
        if (not isinstance(boxes, torch.Tensor)
                or boxes.shape != (batch_size, query_count, 6)
                or not boxes.is_floating_point()
                or boxes.device != query_features.device):
            raise ValueError("boxes must be floating [B,Q,6] on query device")
        rows = (
            superpoint_features, superpoint_xyz, text_mask_logits,
            query_mask_logits, fusion_weights,
        )
        if any(not isinstance(value, (list, tuple)) for value in rows):
            raise ValueError("superpoint, mask, and fusion inputs must be lists")
        if any(len(value) != batch_size for value in rows):
            raise ValueError("all refiner input batches must align")

        query = query_features.float()
        box_rows = boxes.float()
        if self.detach_inputs:
            query = query.detach()
            box_rows = box_rows.detach()
        if (not bool(torch.isfinite(query).all().item())
                or not bool(torch.isfinite(box_rows).all().item())):
            raise ValueError("query features and boxes must be finite")
        normalized_query = self.query_norm(query)
        query_embedding = self.query_projection(normalized_query)
        evidence_coefficients = self.evidence_coefficients(normalized_query)
        geometry_coefficients = self.geometry_coefficients(normalized_query)
        scale_content = math.sqrt(float(self.hidden_dim))
        scale_evidence = math.sqrt(float(self.EVIDENCE_DIM))
        scale_geometry = math.sqrt(float(self.GEOMETRY_DIM))
        content_weight = self._component_weight(self.components, "content")
        evidence_weight = self._component_weight(self.components, "evidence")
        geometry_weight = self._component_weight(self.components, "geometry")

        residuals = []
        content_abs_means = []
        evidence_abs_means = []
        geometry_abs_means = []
        for batch_idx in range(batch_size):
            feature_row = superpoint_features[batch_idx]
            if (not isinstance(feature_row, torch.Tensor)
                    or feature_row.dim() != 2
                    or feature_row.shape[0] != self.d_model
                    or feature_row.shape[1] <= 0
                    or not feature_row.is_floating_point()
                    or feature_row.device != query_features.device):
                raise ValueError(
                    "each superpoint feature row must be floating [D,S]"
                )
            superpoint_count = feature_row.shape[1]
            xyz_row = self._superpoint_xyz(
                superpoint_xyz[batch_idx], superpoint_count
            )
            if (xyz_row.device != query_features.device
                    or not xyz_row.is_floating_point()):
                raise ValueError("superpoint xyz must be floating on query device")
            text_row = as_query_mask_logits(
                text_mask_logits[batch_idx], "text mask logits"
            )
            source_row = as_query_mask_logits(
                query_mask_logits[batch_idx], "query mask logits"
            )
            if (text_row.shape != (query_count, superpoint_count)
                    or source_row.shape != text_row.shape
                    or text_row.device != query_features.device
                    or source_row.device != query_features.device
                    or not text_row.is_floating_point()
                    or not source_row.is_floating_point()):
                raise ValueError("mask source rows must align as floating [Q,S]")

            feature_row = feature_row.transpose(0, 1).float()
            xyz_row = xyz_row.float()
            text_row = text_row.float()
            source_row = source_row.float()
            alpha = query_fusion_weight(
                fusion_weights[batch_idx], query_count, text_row
            )
            if alpha.dim() == 0:
                alpha = alpha.expand(query_count, 1)
            alpha = alpha.float()
            if self.detach_inputs:
                feature_row = feature_row.detach()
                xyz_row = xyz_row.detach()
                text_row = text_row.detach()
                source_row = source_row.detach()
                alpha = alpha.detach()
            if not all(bool(torch.isfinite(value).all().item()) for value in (
                    feature_row, xyz_row, text_row, source_row, alpha)):
                raise ValueError("all refiner inputs must be finite")

            superpoint_embedding = self.superpoint_projection(
                self.superpoint_norm(feature_row)
            )
            content = torch.matmul(
                query_embedding[batch_idx], superpoint_embedding.transpose(0, 1)
            ) / scale_content

            fused = alpha * text_row + (1.0 - alpha) * source_row
            text_probability = text_row.sigmoid()
            source_probability = source_row.sigmoid()
            fused_probability = fused.sigmoid()
            evidence = torch.stack((
                torch.tanh(text_row / 4.0),
                torch.tanh(source_row / 4.0),
                torch.tanh(fused / 4.0),
                fused_probability,
                1.0 - 2.0 * (fused_probability - 0.5).abs(),
                (text_probability - source_probability).abs(),
                text_probability - source_probability,
            ), dim=-1)
            evidence_score = (
                evidence
                * evidence_coefficients[batch_idx].unsqueeze(1)
            ).sum(dim=-1) / scale_evidence

            center = box_rows[batch_idx, :, :3].unsqueeze(1)
            half_size = box_rows[batch_idx, :, 3:].clamp(
                min=1e-4
            ).unsqueeze(1) * 0.5
            relative = (xyz_row.unsqueeze(0) - center) / half_size
            absolute = relative.abs()
            inside_margin = (1.0 - absolute).amin(
                dim=-1, keepdim=True
            ).tanh()
            radius = torch.sqrt(
                relative.square().sum(dim=-1, keepdim=True) + 1e-6
            )
            geometry = torch.cat((
                torch.tanh(relative / 2.0),
                torch.tanh(absolute / 2.0),
                inside_margin,
                torch.tanh(radius / 2.0),
            ), dim=-1)
            geometry_score = (
                geometry
                * geometry_coefficients[batch_idx].unsqueeze(1)
            ).sum(dim=-1) / scale_geometry

            raw = (
                content_weight * content
                + evidence_weight * evidence_score
                + geometry_weight * geometry_score
            )
            residual = self.max_delta * torch.tanh(raw)
            if not bool(torch.isfinite(residual).all().item()):
                raise RuntimeError("EGQS mask residual became non-finite")
            residuals.append(residual)
            content_abs_means.append(content.detach().abs().mean())
            evidence_abs_means.append(evidence_score.detach().abs().mean())
            geometry_abs_means.append(geometry_score.detach().abs().mean())

        detached = [row.detach() for row in residuals]
        return {
            "residuals": residuals,
            "residual_abs_mean": torch.stack([
                row.abs().mean() for row in detached
            ]).mean(),
            "residual_abs_max": torch.stack([
                row.abs().amax() for row in detached
            ]).amax(),
            "superpoint_std_mean": torch.stack([
                row.std(dim=1, unbiased=False).mean() for row in detached
            ]).mean(),
            "query_std_mean": torch.stack([
                row.std(dim=0, unbiased=False).mean() for row in detached
            ]).mean(),
            "content_abs_mean": torch.stack(content_abs_means).mean(),
            "evidence_abs_mean": torch.stack(evidence_abs_means).mean(),
            "geometry_abs_mean": torch.stack(geometry_abs_means).mean(),
        }


class BoundaryAwareSuperpointGraphMaskRefiner(nn.Module):
    """Zero-initialized mask correction from a frozen superpoint graph.

    The graph is shared by all queries in a scene.  Only generic geometry,
    frozen superpoint features, the current mask-source logits, fusion weight,
    query features, and predicted boxes are consumed.
    """

    GRAPH_MODES = ("spatial", "bilateral")
    BASIS_DIM = 8

    def __init__(self, d_model=288, neighbor_count=8, max_delta=2.0,
                 graph_mode="bilateral", detach_inputs=True):
        super().__init__()
        for name, value in (
                ("d_model", d_model), ("neighbor_count", neighbor_count)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value <= 0):
                raise ValueError("{} must be a positive integer".format(name))
        if (not isinstance(max_delta, (int, float))
                or isinstance(max_delta, bool)
                or not math.isfinite(float(max_delta))
                or float(max_delta) <= 0.0):
            raise ValueError("max_delta must be finite and positive")
        if graph_mode not in self.GRAPH_MODES:
            raise ValueError(
                "graph_mode must be one of {}".format(self.GRAPH_MODES)
            )
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be boolean")

        self.d_model = int(d_model)
        self.neighbor_count = int(neighbor_count)
        self.max_delta = float(max_delta)
        self.graph_mode = graph_mode
        self.detach_inputs = detach_inputs
        self.query_norm = nn.LayerNorm(self.d_model)
        self.graph_coefficients = nn.Linear(self.d_model, self.BASIS_DIM)
        nn.init.zeros_(self.graph_coefficients.weight)
        nn.init.zeros_(self.graph_coefficients.bias)

    @staticmethod
    def _superpoint_xyz(value, expected_count):
        if not isinstance(value, torch.Tensor):
            raise ValueError("superpoint xyz must be a tensor")
        if value.dim() == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.dim() != 2 or value.shape != (expected_count, 3):
            raise ValueError("superpoint xyz must have shape [S,3] or [1,S,3]")
        return value

    def _build_graph(self, xyz_row, feature_row):
        superpoint_count = xyz_row.shape[0]
        if superpoint_count <= 1:
            raise ValueError("graph refiner requires at least two superpoints")
        neighbor_count = min(self.neighbor_count, superpoint_count - 1)
        distances = torch.cdist(
            xyz_row.unsqueeze(0), xyz_row.unsqueeze(0), p=2
        ).squeeze(0)
        diagonal = torch.eye(
            superpoint_count, device=distances.device, dtype=torch.bool
        )
        distances = distances.masked_fill(diagonal, float("inf"))
        neighbor_distances, neighbor_indices = torch.topk(
            distances, k=neighbor_count, dim=1, largest=False, sorted=True
        )
        positive_distances = neighbor_distances[neighbor_distances > 0]
        if positive_distances.numel() == 0:
            distance_scale = neighbor_distances.new_tensor(1.0)
        else:
            distance_scale = positive_distances.median().clamp(min=1e-4)
        edge_logits = -neighbor_distances / distance_scale

        normalized_features = F.normalize(feature_row, p=2, dim=-1, eps=1e-6)
        neighbor_features = normalized_features[neighbor_indices]
        feature_cosine = (
            normalized_features.unsqueeze(1) * neighbor_features
        ).sum(dim=-1).clamp(min=-1.0, max=1.0)
        if self.graph_mode == "bilateral":
            edge_logits = edge_logits + 2.0 * feature_cosine
        edge_weights = torch.softmax(edge_logits, dim=1)
        if not all(bool(torch.isfinite(value).all().item()) for value in (
                neighbor_distances, feature_cosine, edge_weights)):
            raise RuntimeError("superpoint graph contains non-finite values")
        return {
            "indices": neighbor_indices,
            "weights": edge_weights,
            "distance_mean": neighbor_distances.mean(),
            "feature_cosine_mean": feature_cosine.mean(),
            "entropy_mean": -(
                edge_weights * edge_weights.clamp(min=1e-8).log()
            ).sum(dim=1).mean(),
        }

    def forward(self, query_features, superpoint_features, superpoint_xyz,
                boxes, text_mask_logits, query_mask_logits, fusion_weights):
        if (not isinstance(query_features, torch.Tensor)
                or query_features.dim() != 3
                or query_features.shape[-1] != self.d_model
                or not query_features.is_floating_point()):
            raise ValueError("query_features must be floating [B,Q,D]")
        batch_size, query_count, _ = query_features.shape
        if (not isinstance(boxes, torch.Tensor)
                or boxes.shape != (batch_size, query_count, 6)
                or not boxes.is_floating_point()
                or boxes.device != query_features.device):
            raise ValueError("boxes must be floating [B,Q,6] on query device")
        rows = (
            superpoint_features, superpoint_xyz, text_mask_logits,
            query_mask_logits, fusion_weights,
        )
        if any(not isinstance(value, (list, tuple)) for value in rows):
            raise ValueError("superpoint, mask, and fusion inputs must be lists")
        if any(len(value) != batch_size for value in rows):
            raise ValueError("all graph refiner input batches must align")

        query = query_features.float()
        box_rows = boxes.float()
        if self.detach_inputs:
            query = query.detach()
            box_rows = box_rows.detach()
        if (not bool(torch.isfinite(query).all().item())
                or not bool(torch.isfinite(box_rows).all().item())):
            raise ValueError("query features and boxes must be finite")
        coefficients = self.graph_coefficients(self.query_norm(query))
        basis_scale = math.sqrt(float(self.BASIS_DIM))

        residuals = []
        graph_abs_means = []
        distance_means = []
        cosine_means = []
        entropy_means = []
        for batch_idx in range(batch_size):
            feature_row = superpoint_features[batch_idx]
            if (not isinstance(feature_row, torch.Tensor)
                    or feature_row.dim() != 2
                    or feature_row.shape[0] != self.d_model
                    or feature_row.shape[1] <= 1
                    or not feature_row.is_floating_point()
                    or feature_row.device != query_features.device):
                raise ValueError(
                    "each superpoint feature row must be floating [D,S]"
                )
            superpoint_count = feature_row.shape[1]
            xyz_row = self._superpoint_xyz(
                superpoint_xyz[batch_idx], superpoint_count
            )
            if (xyz_row.device != query_features.device
                    or not xyz_row.is_floating_point()):
                raise ValueError("superpoint xyz must be floating on query device")
            text_row = as_query_mask_logits(
                text_mask_logits[batch_idx], "text mask logits"
            )
            source_row = as_query_mask_logits(
                query_mask_logits[batch_idx], "query mask logits"
            )
            if (text_row.shape != (query_count, superpoint_count)
                    or source_row.shape != text_row.shape
                    or text_row.device != query_features.device
                    or source_row.device != query_features.device
                    or not text_row.is_floating_point()
                    or not source_row.is_floating_point()):
                raise ValueError("mask source rows must align as floating [Q,S]")

            feature_row = feature_row.transpose(0, 1).float()
            xyz_row = xyz_row.float()
            text_row = text_row.float()
            source_row = source_row.float()
            alpha = query_fusion_weight(
                fusion_weights[batch_idx], query_count, text_row
            )
            if alpha.dim() == 0:
                alpha = alpha.expand(query_count, 1)
            alpha = alpha.float()
            if self.detach_inputs:
                feature_row = feature_row.detach()
                xyz_row = xyz_row.detach()
                text_row = text_row.detach()
                source_row = source_row.detach()
                alpha = alpha.detach()
            if not all(bool(torch.isfinite(value).all().item()) for value in (
                    feature_row, xyz_row, text_row, source_row, alpha)):
                raise ValueError("all graph refiner inputs must be finite")

            graph = self._build_graph(xyz_row, feature_row)
            neighbor_indices = graph["indices"]
            edge_weights = graph["weights"].unsqueeze(0)
            fused = alpha * text_row + (1.0 - alpha) * source_row
            probability = fused.sigmoid()
            neighbor_logits = fused[:, neighbor_indices]
            neighbor_probabilities = probability[:, neighbor_indices]
            mean_logit = (neighbor_logits * edge_weights).sum(dim=-1)
            mean_probability = (
                neighbor_probabilities * edge_weights
            ).sum(dim=-1)
            logit_diffusion = torch.tanh((mean_logit - fused) / 2.0)
            probability_diffusion = mean_probability - probability
            uncertainty = 4.0 * probability * (1.0 - probability)
            variation = (
                (neighbor_probabilities - probability.unsqueeze(-1)).abs()
                * edge_weights
            ).sum(dim=-1)

            center = box_rows[batch_idx, :, :3].unsqueeze(1)
            half_size = box_rows[batch_idx, :, 3:].clamp(
                min=1e-4
            ).unsqueeze(1) * 0.5
            relative = (xyz_row.unsqueeze(0) - center) / half_size
            inside_margin = (1.0 - relative.abs()).amin(dim=-1).tanh()
            neighbor_inside = (
                inside_margin[:, neighbor_indices] * edge_weights
            ).sum(dim=-1)
            inside_diffusion = neighbor_inside - inside_margin
            signed_variation = variation * (2.0 * mean_probability - 1.0)
            graph_basis = torch.stack((
                logit_diffusion,
                probability_diffusion,
                uncertainty * logit_diffusion,
                uncertainty * probability_diffusion,
                variation,
                signed_variation,
                inside_diffusion,
                uncertainty * inside_diffusion,
            ), dim=-1)
            graph_score = (
                graph_basis * coefficients[batch_idx].unsqueeze(1)
            ).sum(dim=-1) / basis_scale
            residual = self.max_delta * torch.tanh(graph_score)
            if not bool(torch.isfinite(residual).all().item()):
                raise RuntimeError("graph mask residual became non-finite")
            residuals.append(residual)
            graph_abs_means.append(graph_score.detach().abs().mean())
            distance_means.append(graph["distance_mean"].detach())
            cosine_means.append(graph["feature_cosine_mean"].detach())
            entropy_means.append(graph["entropy_mean"].detach())

        detached = [row.detach() for row in residuals]
        return {
            "residuals": residuals,
            "residual_abs_mean": torch.stack([
                row.abs().mean() for row in detached
            ]).mean(),
            "residual_abs_max": torch.stack([
                row.abs().amax() for row in detached
            ]).amax(),
            "superpoint_std_mean": torch.stack([
                row.std(dim=1, unbiased=False).mean() for row in detached
            ]).mean(),
            "query_std_mean": torch.stack([
                row.std(dim=0, unbiased=False).mean() for row in detached
            ]).mean(),
            "graph_abs_mean": torch.stack(graph_abs_means).mean(),
            "graph_neighbor_distance_mean": torch.stack(distance_means).mean(),
            "graph_feature_cosine_mean": torch.stack(cosine_means).mean(),
            "graph_edge_entropy_mean": torch.stack(entropy_means).mean(),
        }
