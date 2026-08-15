"""Pure helpers for query-consistent ScanRefer box and mask selection."""

import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F

from .mask_fusion import batched_query_fusion_weight


MASK_SOURCE_NAMES = ("text", "query", "fused")
JOINT_MASK_SCHEMA_VERSION = "rec-joint-box-mask-v1"
MASK_LOGIT_THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0)
MASK_POLICY_COUNT = len(MASK_SOURCE_NAMES) * len(MASK_LOGIT_THRESHOLDS)
MASK_POLICY_FEATURE_SCHEMA_VERSION = "rec-mask-policy-features-v1"
LEGACY_MASK_POLICY_INDEX = (
    MASK_SOURCE_NAMES.index("fused") * len(MASK_LOGIT_THRESHOLDS)
    + MASK_LOGIT_THRESHOLDS.index(0.0)
)


def build_mask_policy_feature_names():
    """Return the fixed, dataset-agnostic inference feature schema."""
    names = []
    for source in MASK_SOURCE_NAMES:
        for statistic in (
                "logit_mean", "logit_std", "logit_min", "logit_max",
                "probability_mean", "confidence_mean", "entropy_mean"):
            names.append("{}_{}".format(source, statistic))
    for source in MASK_SOURCE_NAMES:
        for threshold in MASK_LOGIT_THRESHOLDS:
            names.append("{}_foreground_at_{:+.1f}".format(source, threshold))
    for left, right in (("text", "query"), ("text", "fused"),
                        ("query", "fused")):
        for threshold in MASK_LOGIT_THRESHOLDS:
            names.append(
                "{}_{}_agreement_at_{:+.1f}".format(left, right, threshold)
            )
    names.append("fusion_text_weight")
    if len(names) != 52 or len(set(names)) != len(names):
        raise RuntimeError("mask policy feature schema is inconsistent")
    return names


def compute_mask_policy_inference_features(
        text_logits, query_logits, alpha, valid_mask,
        logit_thresholds=MASK_LOGIT_THRESHOLDS):
    """Build query-level mask statistics without labels or ground truth."""
    if (not isinstance(text_logits, torch.Tensor)
            or not isinstance(query_logits, torch.Tensor)
            or text_logits.shape != query_logits.shape
            or text_logits.dim() != 3 or text_logits.numel() == 0):
        raise ValueError("mask logits must share non-empty shape [B,K,S]")
    if (not text_logits.is_floating_point()
            or not query_logits.is_floating_point()):
        raise TypeError("mask logits must be floating point")
    if (text_logits.device != query_logits.device
            or not bool(torch.isfinite(text_logits).all().item())
            or not bool(torch.isfinite(query_logits).all().item())):
        raise ValueError("mask logits must be finite on one device")
    batch_size, query_count, _superpoint_count = text_logits.shape
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != (batch_size, query_count)
            or valid_mask.device != text_logits.device
            or not bool(valid_mask.any(dim=1).all().item())):
        raise ValueError("valid_mask must cover at least one query per row")
    thresholds = torch.as_tensor(
        logit_thresholds, device=text_logits.device, dtype=text_logits.dtype
    )
    expected_thresholds = torch.tensor(
        MASK_LOGIT_THRESHOLDS,
        device=text_logits.device,
        dtype=text_logits.dtype,
    )
    if (thresholds.dim() != 1
            or not torch.equal(thresholds, expected_thresholds)):
        raise ValueError("mask policy thresholds differ from frozen schema")

    fusion_weight = _normalized_alpha(
        alpha, batch_size, query_count, text_logits
    )
    if fusion_weight.shape == (batch_size, query_count, 1):
        fusion_feature = fusion_weight
    elif fusion_weight.shape == (batch_size, 1, 1):
        fusion_feature = fusion_weight.expand(-1, query_count, -1)
    elif fusion_weight.shape == (batch_size, query_count):
        fusion_feature = fusion_weight.unsqueeze(-1)
    else:
        raise RuntimeError("fusion weight shape differs from mask contract")
    fused_logits = fuse_mask_logits(text_logits, query_logits, fusion_weight)
    sources = torch.stack((text_logits, query_logits, fused_logits), dim=2)
    probabilities = sources.sigmoid()
    epsilon = torch.finfo(probabilities.dtype).eps
    safe_probabilities = probabilities.clamp(
        min=epsilon, max=1.0 - epsilon
    )
    entropy = -(
        safe_probabilities * safe_probabilities.log()
        + (1.0 - safe_probabilities)
        * (1.0 - safe_probabilities).log()
    )
    statistics = torch.stack((
        sources.mean(dim=-1),
        sources.std(dim=-1, unbiased=False),
        sources.amin(dim=-1),
        sources.amax(dim=-1),
        probabilities.mean(dim=-1),
        (2.0 * (probabilities - 0.5).abs()).mean(dim=-1),
        entropy.mean(dim=-1),
    ), dim=-1).reshape(batch_size, query_count, -1)
    binary = sources.unsqueeze(-1) > thresholds.reshape(1, 1, 1, 1, -1)
    foreground = binary.to(text_logits.dtype).mean(dim=3).reshape(
        batch_size, query_count, -1
    )
    agreements = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        agreements.append(
            binary[:, :, left].eq(binary[:, :, right])
            .to(text_logits.dtype).mean(dim=2)
        )
    agreement = torch.cat(agreements, dim=-1)
    features = torch.cat((
        statistics,
        foreground,
        agreement,
        fusion_feature.to(text_logits.dtype),
    ), dim=-1)
    if features.shape[-1] != len(build_mask_policy_feature_names()):
        raise RuntimeError("mask policy feature dimension drifted")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("mask policy features must be finite")
    return torch.where(
        valid_mask.unsqueeze(-1), features, torch.zeros_like(features)
    ).contiguous()


class JointBoxMaskAdapter(nn.Module):
    """Small frozen-backbone adapter over query/geometry candidates."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.variant_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.box_head = nn.Linear(self.hidden_dim, 2)
        self.mask_iou_head = nn.Linear(self.hidden_dim, 1)
        self.mask_threshold_head = nn.Linear(self.hidden_dim, 2)
        self.log_scale_head = nn.Linear(self.hidden_dim, 1)
        self.mask_policy_head = nn.Linear(
            self.hidden_dim, MASK_POLICY_COUNT
        )
        nn.init.zeros_(self.mask_policy_head.weight)
        nn.init.zeros_(self.mask_policy_head.bias)
        identity_prior = torch.zeros(MASK_POLICY_COUNT)
        identity_prior[LEGACY_MASK_POLICY_INDEX] = 1e-3
        self.register_buffer("mask_policy_identity_prior", identity_prior)

    @staticmethod
    def _masked_mean(values, valid, dimensions):
        weights = valid.to(values.dtype).unsqueeze(-1)
        numerator = (values * weights).sum(dim=dimensions, keepdim=True)
        denominator = weights.sum(dim=dimensions, keepdim=True).clamp(min=1.0)
        return numerator / denominator

    @staticmethod
    def _masked_max(values, valid, dimensions):
        masked = values.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        maximum = masked.amax(dim=dimensions, keepdim=True)
        return torch.where(torch.isfinite(maximum), maximum,
                           torch.zeros_like(maximum))

    def forward(self, features, valid_mask):
        if not isinstance(features, torch.Tensor) or features.dim() != 4:
            raise ValueError("features must have shape [B,K,G,D]")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if features.shape[-1] != self.input_dim:
            raise ValueError("feature dimension does not match adapter")
        if not isinstance(valid_mask, torch.Tensor) \
                or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use bool dtype")
        if valid_mask.shape != features.shape[:3]:
            raise ValueError("valid_mask must have shape [B,K,G]")
        if valid_mask.device != features.device:
            raise ValueError("features and validity must share a device")
        if not bool(valid_mask.reshape(features.shape[0], -1).any(1).all().item()):
            raise ValueError("each row must contain a valid candidate")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("features must be finite")

        local = self.variant_projection(features)
        row_mean = self._masked_mean(local, valid_mask, (1, 2)).expand_as(local)
        row_max = self._masked_max(local, valid_mask, (1, 2)).expand_as(local)
        encoded = self.context_projection(torch.cat((local, row_mean, row_max), -1))
        encoded = encoded.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        query_weights = valid_mask.to(encoded.dtype).unsqueeze(-1)
        query_encoded = (encoded * query_weights).sum(2) / query_weights.sum(
            2
        ).clamp(min=1.0)
        mask_policy_logits = (
            self.mask_policy_head(query_encoded)
            + self.mask_policy_identity_prior.view(1, 1, -1)
        )
        return {
            "box_logits": self.box_head(encoded),
            "mask_iou": self.mask_iou_head(encoded).squeeze(-1).sigmoid(),
            "mask_logits": self.mask_threshold_head(encoded),
            "log_scale": self.log_scale_head(encoded).squeeze(-1).clamp(-5.0, 5.0),
            "mask_policy_logits": mask_policy_logits,
        }


def _calibration_broadcast(parameter, logits):
    if logits.dim() == 3:
        return parameter.unsqueeze(-1)
    if logits.dim() == 4:
        return parameter.unsqueeze(-1).unsqueeze(-1)
    raise ValueError("mask logits must have shape [B,K,S] or [B,K,G,S]")


def calibrate_mask_logits(calibration, text_logits, query_logits,
                          disabled=False, legacy_alpha=None):
    """Apply bounded query-specific calibration to mask logits."""
    if not isinstance(calibration, torch.Tensor) or calibration.dim() != 3 \
            or calibration.shape[-1] != 5:
        raise ValueError("calibration must have shape [B,K,5]")
    if not isinstance(text_logits, torch.Tensor) or not isinstance(
            query_logits, torch.Tensor):
        raise TypeError("mask logits must be tensors")
    if text_logits.shape != query_logits.shape or text_logits.dim() not in (3, 4):
        raise ValueError("mask logits must share shape [B,K,S] or [B,K,G,S]")
    if text_logits.shape[:2] != calibration.shape[:2]:
        raise ValueError("calibration and mask logits must share B,K axes")
    if not text_logits.is_floating_point() or not query_logits.is_floating_point():
        raise TypeError("mask logits must be floating point")
    if calibration.device != text_logits.device or query_logits.device != text_logits.device:
        raise ValueError("calibration and logits must share a device")
    if not bool(torch.isfinite(calibration).all().item()) or not bool(
            torch.isfinite(text_logits).all().item()) or not bool(
            torch.isfinite(query_logits).all().item()):
        raise ValueError("calibration and logits must be finite")

    if disabled:
        if legacy_alpha is None:
            raise ValueError("legacy_alpha is required for disabled calibration")
        alpha = torch.as_tensor(
            legacy_alpha, device=text_logits.device, dtype=text_logits.dtype
        )
        if alpha.numel() not in (1, text_logits.shape[0]):
            raise ValueError("legacy_alpha must be scalar or one value per row")
        if not bool(torch.isfinite(alpha).all().item()):
            raise ValueError("legacy_alpha must be finite")
        if bool(((alpha < 0.0) | (alpha > 1.0)).any().item()):
            raise ValueError("legacy_alpha must lie in [0,1]")
        weight = alpha.reshape(-1, 1).expand(calibration.shape[:2])
        temperature_text = torch.ones_like(weight)
        temperature_query = torch.ones_like(weight)
        bias = torch.zeros_like(weight)
        threshold = torch.zeros_like(weight)
    else:
        raw = calibration.to(text_logits.dtype)
        weight = raw[..., 0].sigmoid()
        temperature_text = 0.25 + 3.75 * raw[..., 1].sigmoid()
        temperature_query = 0.25 + 3.75 * raw[..., 2].sigmoid()
        bias = 2.0 * raw[..., 3].tanh()
        threshold = raw[..., 4].tanh()

    weight_view = _calibration_broadcast(weight, text_logits)
    text_temperature_view = _calibration_broadcast(
        temperature_text, text_logits
    )
    query_temperature_view = _calibration_broadcast(
        temperature_query, text_logits
    )
    bias_view = _calibration_broadcast(bias, text_logits)
    threshold_view = _calibration_broadcast(threshold, text_logits)
    logits = (
        weight_view * (text_logits / text_temperature_view)
        + (1.0 - weight_view) * (query_logits / query_temperature_view)
        + bias_view
    )
    return {
        "logits": logits,
        "binary": logits > threshold_view,
        "weight": weight,
        "temperature_text": temperature_text,
        "temperature_query": temperature_query,
        "bias": bias,
        "threshold": threshold,
    }


def _masked_mean_loss(values, valid_mask):
    weights = valid_mask.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp(min=1.0)


def compute_joint_box_mask_losses(outputs, targets, valid_mask):
    """Return finite named losses for the frozen-backbone joint adapter."""
    required_outputs = {
        "box_logits", "mask_iou", "mask_logits", "log_scale",
        "mask_policy_logits",
    }
    if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
        raise ValueError("adapter outputs are incomplete")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    box_target = torch.as_tensor(
        targets["box_tier"], device=outputs["box_logits"].device,
        dtype=outputs["box_logits"].dtype,
    )
    mask_iou_target = torch.as_tensor(
        targets["mask_iou"], device=outputs["mask_iou"].device,
        dtype=outputs["mask_iou"].dtype,
    )
    hit050 = torch.as_tensor(
        targets["mask_hits050"], device=outputs["mask_logits"].device,
        dtype=outputs["mask_logits"].dtype,
    )
    hit025 = (mask_iou_target > 0.25).to(hit050.dtype)
    mask_target = torch.stack((hit025, hit050), dim=-1)
    mask_policy_ious = torch.as_tensor(
        targets["mask_policy_ious"],
        device=outputs["mask_policy_logits"].device,
        dtype=outputs["mask_policy_logits"].dtype,
    )
    expected_shape = tuple(valid_mask.shape)
    if box_target.shape != expected_shape + (2,):
        raise ValueError("box_tier must have shape [B,K,G,2]")
    if mask_iou_target.shape != expected_shape or hit050.shape != expected_shape:
        raise ValueError("mask targets must have shape [B,K,G]")
    expected_policy_shape = (
        valid_mask.shape[0], valid_mask.shape[1], MASK_POLICY_COUNT
    )
    if mask_policy_ious.shape != expected_policy_shape:
        raise ValueError(
            "mask_policy_ious must have shape [B,K,{}]".format(
                MASK_POLICY_COUNT
            )
        )
    if (not bool(torch.isfinite(mask_policy_ious).all().item())
            or bool(((mask_policy_ious < 0.0)
                     | (mask_policy_ious > 1.0)).any().item())):
        raise ValueError("mask policy IoUs must be finite in [0,1]")

    box_bce = F.binary_cross_entropy_with_logits(
        outputs["box_logits"], box_target, reduction="none"
    )
    mask_bce = F.binary_cross_entropy_with_logits(
        outputs["mask_logits"], mask_target, reduction="none"
    )
    iou_huber = F.smooth_l1_loss(
        outputs["mask_iou"], mask_iou_target, reduction="none"
    )
    probability = outputs["mask_logits"].sigmoid()
    focal = F.binary_cross_entropy_with_logits(
        outputs["mask_logits"], mask_target, reduction="none"
    ) * (probability - mask_target).abs().pow(2)
    dice = 1.0 - (
        2.0 * probability * mask_target + 1.0
    ) / (probability + mask_target + 1.0)

    predicted_delta = outputs["mask_iou"] - outputs["mask_iou"].detach().mean(
        dim=(1, 2), keepdim=True
    )
    target_delta = mask_iou_target - mask_iou_target.mean(
        dim=(1, 2), keepdim=True
    )
    ranking = F.relu(0.05 - predicted_delta * target_delta.sign())
    risky = ((box_target[..., 0] < 0.5) | (box_target[..., 1] < 0.5)).to(
        outputs["mask_iou"].dtype
    )
    switch_risk = outputs["mask_iou"] * risky
    query_valid = valid_mask.any(dim=2)
    policy_logits = outputs["mask_policy_logits"]
    if policy_logits.shape != expected_policy_shape:
        raise ValueError(
            "mask_policy_logits must have shape [B,K,{}]".format(
                MASK_POLICY_COUNT
            )
        )
    policy_target = mask_policy_ious.argmax(dim=-1)
    policy_ce = F.cross_entropy(
        policy_logits.reshape(-1, MASK_POLICY_COUNT),
        policy_target.reshape(-1),
        reduction="none",
    ).reshape_as(query_valid)
    policy_probability = policy_logits.softmax(dim=-1)
    expected_policy_iou = (
        policy_probability * mask_policy_ious
    ).sum(dim=-1)
    policy_regret = (
        mask_policy_ious.max(dim=-1).values - expected_policy_iou
    ).clamp(min=0.0)
    return {
        "box_bce": _masked_mean_loss(box_bce, valid_mask),
        "mask_bce": _masked_mean_loss(mask_bce, valid_mask),
        "mask_iou_huber": _masked_mean_loss(iou_huber, valid_mask),
        "mask_focal": _masked_mean_loss(focal, valid_mask),
        "mask_dice": _masked_mean_loss(dice, valid_mask),
        "ranking": _masked_mean_loss(ranking, valid_mask),
        "switch_risk": _masked_mean_loss(switch_risk, valid_mask),
        "mask_policy_ce": _masked_mean_loss(policy_ce, query_valid),
        "mask_policy_regret": _masked_mean_loss(
            policy_regret, query_valid
        ),
    }


def flat_to_parent_query(flat_indices, variant_count):
    """Map flattened ``[query, variant]`` identities to detector queries."""
    if not isinstance(flat_indices, torch.Tensor):
        raise TypeError("flat_indices must be a tensor")
    if flat_indices.dtype != torch.long:
        raise TypeError("flat_indices must use int64")
    if isinstance(variant_count, bool) or not isinstance(
            variant_count, numbers.Integral):
        raise TypeError("variant_count must be an integer")
    variant_count = int(variant_count)
    if variant_count <= 0:
        raise ValueError("variant_count must be positive")
    if bool((flat_indices < 0).any().item()):
        raise ValueError("flat_indices must be non-negative")
    return torch.div(flat_indices, variant_count, rounding_mode="floor")


def _normalized_alpha(alpha, batch_size, query_count, reference):
    return batched_query_fusion_weight(
        alpha, batch_size, query_count, reference
    )


def fuse_mask_logits(text_logits, query_logits, alpha):
    """Apply the current MCLN logit-space fusion without changing semantics."""
    if not isinstance(text_logits, torch.Tensor) or not isinstance(
            query_logits, torch.Tensor):
        raise TypeError("mask logits must be tensors")
    if text_logits.shape != query_logits.shape or text_logits.dim() != 3:
        raise ValueError("mask logits must share shape [B,K,S]")
    if not text_logits.is_floating_point() or not query_logits.is_floating_point():
        raise TypeError("mask logits must be floating point")
    normalized = _normalized_alpha(
        alpha, text_logits.shape[0], text_logits.shape[1], text_logits
    )
    return normalized * text_logits + (1.0 - normalized) * query_logits


def _binary_entropy_from_logits(logits):
    probabilities = logits.sigmoid()
    epsilon = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(min=epsilon, max=1.0 - epsilon)
    return -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )


def compute_mask_candidate_targets(text_logits, query_logits, alpha,
                                   gt_masks, valid_mask, logit_thresholds):
    """Compute train-only mask targets for all queries, sources and thresholds.

    The returned tensor axes are ``[batch, query, source, threshold]``. Ground
    truth is accepted only by this explicit target helper and is never part of
    the inference feature mapping.
    """
    if not isinstance(text_logits, torch.Tensor) or not isinstance(
            query_logits, torch.Tensor):
        raise TypeError("mask logits must be tensors")
    if text_logits.shape != query_logits.shape or text_logits.dim() != 3:
        raise ValueError("mask logits must share shape [B,K,S]")
    if not text_logits.is_floating_point() or not query_logits.is_floating_point():
        raise TypeError("mask logits must be floating point")
    if text_logits.numel() == 0:
        raise ValueError("mask logits must be non-empty")
    if not bool(torch.isfinite(text_logits).all().item()) or not bool(
            torch.isfinite(query_logits).all().item()):
        raise ValueError("mask logits must be finite")

    batch_size, candidate_count, superpoint_count = text_logits.shape
    if not isinstance(gt_masks, torch.Tensor) or gt_masks.dtype != torch.bool:
        raise TypeError("gt_masks must use bool dtype")
    if gt_masks.shape != (batch_size, superpoint_count):
        raise ValueError("gt_masks must have shape [B,S]")
    if gt_masks.device != text_logits.device:
        raise ValueError("gt_masks must be on the logits device")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    if valid_mask.shape != (batch_size, candidate_count):
        raise ValueError("valid_mask must have shape [B,K]")
    if valid_mask.device != text_logits.device:
        raise ValueError("valid_mask must be on the logits device")
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("each row must contain at least one valid candidate")

    thresholds = torch.as_tensor(
        logit_thresholds, dtype=text_logits.dtype, device=text_logits.device
    )
    if thresholds.dim() != 1 or thresholds.numel() == 0:
        raise ValueError("logit_thresholds must have shape [T]")
    if not bool(torch.isfinite(thresholds).all().item()):
        raise ValueError("logit_thresholds must be finite")

    fused_logits = fuse_mask_logits(text_logits, query_logits, alpha)
    source_logits = torch.stack(
        (text_logits, query_logits, fused_logits), dim=2
    )
    predictions = source_logits.unsqueeze(-1) > thresholds.reshape(
        1, 1, 1, 1, -1
    )
    target = gt_masks.reshape(batch_size, 1, 1, superpoint_count, 1)
    intersections = (predictions & target).sum(dim=3)
    unions = (predictions | target).sum(dim=3)
    expanded_valid = valid_mask.unsqueeze(-1).unsqueeze(-1)
    valid_unions = unions.masked_select(expanded_valid.expand_as(unions))
    if bool((valid_unions == 0).any().item()):
        raise ValueError("mask IoU has an empty union")

    ious = intersections.to(text_logits.dtype) / unions.clamp(min=1).to(
        text_logits.dtype
    )
    ious = ious.masked_fill(~expanded_valid, 0.0)
    selected_count = predictions.sum(dim=3).masked_fill(~expanded_valid, 0)
    foreground_fraction = selected_count.to(text_logits.dtype) / float(
        superpoint_count
    )

    confidence = (
        2.0 * (source_logits.sigmoid() - 0.5).abs()
    ).mean(dim=3).masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    entropy = _binary_entropy_from_logits(source_logits).mean(dim=3)
    entropy = entropy.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    text_binary = predictions[:, :, 0]
    query_binary = predictions[:, :, 1]
    text_query_intersection = (text_binary & query_binary).sum(dim=2)
    text_query_denominator = (
        text_binary.sum(dim=2) + query_binary.sum(dim=2)
    )
    text_query_dice = (
        2.0 * text_query_intersection.to(text_logits.dtype)
        / text_query_denominator.clamp(min=1).to(text_logits.dtype)
    ).masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    return {
        "schema_version": JOINT_MASK_SCHEMA_VERSION,
        "source_names": MASK_SOURCE_NAMES,
        "logit_thresholds": thresholds,
        "ious": ious,
        "hits025": (ious > 0.25) & expanded_valid,
        "hits050": (ious > 0.50) & expanded_valid,
        "selected_superpoint_count": selected_count,
        "foreground_fraction": foreground_fraction,
        "confidence": confidence,
        "entropy": entropy,
        "text_query_dice": text_query_dice,
    }


def compress_point_mask_to_superpoints(point_target, superpoint_ids,
                                       num_superpoints=None):
    """Compress a point mask into exact per-superpoint point counts."""
    if not isinstance(point_target, torch.Tensor) or point_target.dtype != torch.bool:
        raise TypeError("point_target must use bool dtype")
    if not isinstance(superpoint_ids, torch.Tensor) or superpoint_ids.dim() != 1:
        raise ValueError("superpoint_ids must have shape [N]")
    if point_target.dim() != 1 or point_target.shape != superpoint_ids.shape:
        raise ValueError("point target and superpoint IDs must share shape [N]")
    if superpoint_ids.device != point_target.device:
        raise ValueError("point target and superpoint IDs must share a device")
    if torch.is_floating_point(superpoint_ids):
        if not bool(torch.isfinite(superpoint_ids).all().item()) or not torch.equal(
                superpoint_ids, superpoint_ids.round()):
            raise ValueError("superpoint IDs must be finite integers")
    ids = superpoint_ids.long()
    if ids.numel() == 0 or bool((ids < 0).any().item()):
        raise ValueError("superpoint IDs must be non-empty and non-negative")
    inferred = int(ids.max().item()) + 1
    if num_superpoints is None:
        num_superpoints = inferred
    if not isinstance(num_superpoints, int) or isinstance(num_superpoints, bool) \
            or num_superpoints < inferred:
        raise ValueError("num_superpoints does not cover the point mapping")
    point_counts = torch.bincount(ids, minlength=num_superpoints)
    target_counts = torch.bincount(
        ids, weights=point_target.to(torch.float32), minlength=num_superpoints
    ).long()
    if int(target_counts.sum().item()) <= 0:
        raise ValueError("point target must contain foreground")
    return {
        "point_counts": point_counts,
        "target_counts": target_counts,
    }


def compute_weighted_mask_candidate_targets(
        text_logits, query_logits, alpha, point_counts, target_counts,
        valid_mask, logit_thresholds):
    """Compute exact point-level mask metrics from compressed superpoints."""
    if not isinstance(text_logits, torch.Tensor) or not isinstance(
            query_logits, torch.Tensor):
        raise TypeError("mask logits must be tensors")
    if text_logits.shape != query_logits.shape or text_logits.dim() != 3:
        raise ValueError("mask logits must share shape [B,K,S]")
    if not text_logits.is_floating_point() or not query_logits.is_floating_point():
        raise TypeError("mask logits must be floating point")
    if not bool(torch.isfinite(text_logits).all().item()) or not bool(
            torch.isfinite(query_logits).all().item()):
        raise ValueError("mask logits must be finite")
    batch_size, candidate_count, superpoint_count = text_logits.shape
    point_counts = torch.as_tensor(point_counts, device=text_logits.device)
    target_counts = torch.as_tensor(target_counts, device=text_logits.device)
    if point_counts.shape != (batch_size, superpoint_count) or (
            target_counts.shape != point_counts.shape):
        raise ValueError("compressed point counts must have shape [B,S]")
    if torch.is_floating_point(point_counts) or torch.is_floating_point(target_counts):
        raise TypeError("compressed point counts must use integer dtype")
    point_counts = point_counts.long()
    target_counts = target_counts.long()
    if bool((point_counts < 0).any().item()) or bool((target_counts < 0).any().item()) \
            or bool((target_counts > point_counts).any().item()):
        raise ValueError("compressed point counts are invalid")
    if not bool((point_counts.sum(1) > 0).all().item()) or not bool(
            (target_counts.sum(1) > 0).all().item()):
        raise ValueError("each row must contain points and target foreground")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    if valid_mask.shape != (batch_size, candidate_count):
        raise ValueError("valid_mask must have shape [B,K]")
    if valid_mask.device != text_logits.device or not bool(
            valid_mask.any(1).all().item()):
        raise ValueError("each row must contain a valid candidate on the logits device")
    thresholds = torch.as_tensor(
        logit_thresholds, device=text_logits.device, dtype=text_logits.dtype
    )
    if thresholds.dim() != 1 or thresholds.numel() == 0 or not bool(
            torch.isfinite(thresholds).all().item()):
        raise ValueError("logit_thresholds must be a finite vector")

    fused = fuse_mask_logits(text_logits, query_logits, alpha)
    source_logits = torch.stack((text_logits, query_logits, fused), dim=2)
    predictions = source_logits.unsqueeze(-1) > thresholds.reshape(
        1, 1, 1, 1, -1
    )
    point_weights = point_counts.reshape(batch_size, 1, 1, superpoint_count, 1)
    target_weights = target_counts.reshape(
        batch_size, 1, 1, superpoint_count, 1
    )
    intersections = (predictions.long() * target_weights).sum(3)
    selected_points = (predictions.long() * point_weights).sum(3)
    target_points = target_counts.sum(1).reshape(batch_size, 1, 1, 1)
    unions = selected_points + target_points - intersections
    expanded_valid = valid_mask.unsqueeze(-1).unsqueeze(-1)
    valid_unions = unions.masked_select(expanded_valid.expand_as(unions))
    if bool((valid_unions <= 0).any().item()):
        raise ValueError("mask IoU has an empty union")
    ious = intersections.to(text_logits.dtype) / unions.clamp(min=1).to(
        text_logits.dtype
    )
    ious = ious.masked_fill(~expanded_valid, 0.0)
    return {
        "schema_version": JOINT_MASK_SCHEMA_VERSION,
        "source_names": MASK_SOURCE_NAMES,
        "logit_thresholds": thresholds,
        "ious": ious,
        "hits025": (ious > 0.25) & expanded_valid,
        "hits050": (ious > 0.50) & expanded_valid,
        "selected_point_count": selected_points.masked_fill(
            ~expanded_valid, 0
        ),
    }


def iou_tier(ious):
    """Return the strict ScanRefer box tier: zero, @0.25, or @0.50."""
    if not isinstance(ious, torch.Tensor) or not ious.is_floating_point():
        raise TypeError("ious must be a floating-point tensor")
    if not bool(torch.isfinite(ious).all().item()):
        raise ValueError("ious must be finite")
    return (ious > 0.25).long() + (ious > 0.50).long()


def _validate_joint_oracle_inputs(box_ious, mask_ious, baseline_flat_indices,
                                  valid_mask):
    if not isinstance(box_ious, torch.Tensor) or box_ious.dim() != 3:
        raise ValueError("box_ious must have shape [B,K,G]")
    if not box_ious.is_floating_point() or box_ious.numel() == 0:
        raise TypeError("box_ious must be a non-empty floating-point tensor")
    batch_size, candidate_count, variant_count = box_ious.shape
    if not isinstance(mask_ious, torch.Tensor) or not mask_ious.is_floating_point():
        raise TypeError("mask_ious must be a floating-point tensor")
    if mask_ious.shape == (batch_size, candidate_count):
        mask_ious = mask_ious.unsqueeze(-1).expand(
            -1, -1, variant_count
        )
    elif mask_ious.shape != box_ious.shape:
        raise ValueError("mask_ious must have shape [B,K] or [B,K,G]")
    if mask_ious.device != box_ious.device:
        raise ValueError("box and mask IoUs must share a device")
    if not bool(torch.isfinite(box_ious).all().item()) or not bool(
            torch.isfinite(mask_ious).all().item()):
        raise ValueError("box and mask IoUs must be finite")
    if bool(((box_ious < 0.0) | (box_ious > 1.0)).any().item()) or bool(
            ((mask_ious < 0.0) | (mask_ious > 1.0)).any().item()):
        raise ValueError("box and mask IoUs must lie in [0,1]")

    if valid_mask is None:
        valid_mask = torch.ones_like(box_ious, dtype=torch.bool)
    elif not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    elif valid_mask.shape != box_ious.shape:
        raise ValueError("valid_mask must have shape [B,K,G]")
    elif valid_mask.device != box_ious.device:
        raise ValueError("valid_mask must share the IoU device")
    if not bool(valid_mask.reshape(batch_size, -1).any(dim=1).all().item()):
        raise ValueError("each row must contain a valid joint candidate")

    if not isinstance(baseline_flat_indices, torch.Tensor) or (
            baseline_flat_indices.dtype != torch.long):
        raise TypeError("baseline_flat_indices must use int64")
    if baseline_flat_indices.shape != (batch_size,):
        raise ValueError("baseline_flat_indices must have shape [B]")
    if baseline_flat_indices.device != box_ious.device:
        raise ValueError("baseline indices must share the IoU device")
    flat_count = candidate_count * variant_count
    if bool(((baseline_flat_indices < 0)
             | (baseline_flat_indices >= flat_count)).any().item()):
        raise ValueError("baseline index is out of range")
    flat_valid = valid_mask.reshape(batch_size, flat_count)
    baseline_valid = flat_valid.gather(
        1, baseline_flat_indices.unsqueeze(1)
    ).squeeze(1)
    if not bool(baseline_valid.all().item()):
        raise ValueError("baseline joint candidate must be valid")
    return mask_ious, valid_mask


def select_joint_oracle(box_ious, mask_ious, baseline_flat_indices,
                        valid_mask=None):
    """Select the best mask candidate without lowering a sample's box tier."""
    mask_ious, valid_mask = _validate_joint_oracle_inputs(
        box_ious, mask_ious, baseline_flat_indices, valid_mask
    )
    batch_size, candidate_count, variant_count = box_ious.shape
    flat_count = candidate_count * variant_count
    flat_box = box_ious.reshape(batch_size, flat_count)
    flat_mask = mask_ious.reshape(batch_size, flat_count)
    flat_valid = valid_mask.reshape(batch_size, flat_count)
    gather_index = baseline_flat_indices.unsqueeze(1)
    baseline_box = flat_box.gather(1, gather_index).squeeze(1)
    baseline_mask = flat_mask.gather(1, gather_index).squeeze(1)
    eligible = flat_valid & (
        iou_tier(flat_box) >= iou_tier(baseline_box).unsqueeze(1)
    )

    selected = baseline_flat_indices.clone()
    selected_box = baseline_box.clone()
    selected_mask = baseline_mask.clone()
    for flat_index in range(flat_count):
        candidate_box = flat_box[:, flat_index]
        candidate_mask = flat_mask[:, flat_index]
        better = candidate_mask > selected_mask
        equal_mask = candidate_mask == selected_mask
        better_box = candidate_box > selected_box
        equal_box = candidate_box == selected_box
        earlier = flat_index < selected
        replace = eligible[:, flat_index] & (
            better | (equal_mask & (better_box | (equal_box & earlier)))
        )
        selected = torch.where(
            replace, selected.new_full(selected.shape, flat_index), selected
        )
        selected_box = torch.where(replace, candidate_box, selected_box)
        selected_mask = torch.where(replace, candidate_mask, selected_mask)

    baseline_position025 = baseline_box > 0.25
    baseline_position050 = baseline_box > 0.50
    selected_position025 = selected_box > 0.25
    selected_position050 = selected_box > 0.50
    baseline_mask025 = baseline_mask > 0.25
    baseline_mask050 = baseline_mask > 0.50
    selected_mask025 = selected_mask > 0.25
    selected_mask050 = selected_mask > 0.50
    return {
        "selected_flat_index": selected,
        "selected_parent_query": flat_to_parent_query(
            selected, variant_count
        ),
        "selected_variant_index": torch.remainder(selected, variant_count),
        "baseline_flat_index": baseline_flat_indices,
        "baseline_parent_query": flat_to_parent_query(
            baseline_flat_indices, variant_count
        ),
        "baseline_box_iou": baseline_box,
        "selected_box_iou": selected_box,
        "baseline_mask_iou": baseline_mask,
        "selected_mask_iou": selected_mask,
        "eligible_count": eligible.sum(dim=1),
        "position_fix025": selected_position025 & ~baseline_position025,
        "position_fix050": selected_position050 & ~baseline_position050,
        "position_break025": baseline_position025 & ~selected_position025,
        "position_break050": baseline_position050 & ~selected_position050,
        "mask_fix025": selected_mask025 & ~baseline_mask025,
        "mask_fix050": selected_mask050 & ~baseline_mask050,
        "mask_break025": baseline_mask025 & ~selected_mask025,
        "mask_break050": baseline_mask050 & ~selected_mask050,
    }


def _validated_metric_pair(selection, baseline_key, selected_key):
    try:
        baseline = selection[baseline_key]
        selected = selection[selected_key]
    except KeyError as error:
        raise ValueError("selection is missing {}".format(error.args[0]))
    if not isinstance(baseline, torch.Tensor) or not isinstance(
            selected, torch.Tensor):
        raise TypeError("selection metrics must be tensors")
    if baseline.dim() != 1 or selected.shape != baseline.shape:
        raise ValueError("selection metrics must share shape [B]")
    if not baseline.is_floating_point() or not selected.is_floating_point():
        raise TypeError("selection metrics must be floating point")
    if baseline.numel() == 0:
        raise ValueError("selection metrics must be non-empty")
    if not bool(torch.isfinite(baseline).all().item()) or not bool(
            torch.isfinite(selected).all().item()):
        raise ValueError("selection metrics must be finite")
    return baseline.double(), selected.double()


def summarize_joint_oracle(selection):
    """Aggregate baseline and oracle metrics without changing strict formulas."""
    baseline_box, selected_box = _validated_metric_pair(
        selection, "baseline_box_iou", "selected_box_iou"
    )
    baseline_mask, selected_mask = _validated_metric_pair(
        selection, "baseline_mask_iou", "selected_mask_iou"
    )
    if baseline_box.shape != baseline_mask.shape:
        raise ValueError("box and mask metric rows must align")
    row_count = baseline_box.numel()

    def _hits(values, threshold):
        return int((values > threshold).sum().item())

    summary = {"row_count": row_count}
    for name, baseline, selected in (
            ("position", baseline_box, selected_box),
            ("mask", baseline_mask, selected_mask)):
        for suffix, threshold in (("025", 0.25), ("050", 0.50)):
            baseline_hits = _hits(baseline, threshold)
            selected_hits = _hits(selected, threshold)
            summary["baseline_{}_hits{}".format(name, suffix)] = baseline_hits
            summary["selected_{}_hits{}".format(name, suffix)] = selected_hits
            summary["baseline_{}_acc{}".format(name, suffix)] = (
                baseline_hits / float(row_count)
            )
            summary["selected_{}_acc{}".format(name, suffix)] = (
                selected_hits / float(row_count)
            )
            summary["delta_{}_acc{}".format(name, suffix)] = (
                (selected_hits - baseline_hits) / float(row_count)
            )
    summary["baseline_mask_miou"] = float(baseline_mask.mean().item())
    summary["selected_mask_miou"] = float(selected_mask.mean().item())
    summary["delta_mask_miou"] = (
        summary["selected_mask_miou"] - summary["baseline_mask_miou"]
    )
    return summary


def stage0_gate(summary):
    """Apply the immutable train-only Stage 0 headroom thresholds."""
    thresholds = {
        "delta_mask_acc050": 0.03,
        "delta_mask_miou": 0.04,
        "delta_position_acc025": 0.0,
        "delta_position_acc050": 0.0,
    }
    observed = {}
    for name in thresholds:
        if name not in summary:
            raise ValueError("summary is missing {}".format(name))
        try:
            value = float(summary[name])
        except (TypeError, ValueError, OverflowError):
            raise ValueError("{} must be a finite number".format(name))
        if not math.isfinite(value):
            raise ValueError("{} must be a finite number".format(name))
        observed[name] = value
    return {
        "pass": all(
            observed[name] >= threshold
            for name, threshold in thresholds.items()
        ),
        "thresholds": thresholds,
        "observed": observed,
    }
