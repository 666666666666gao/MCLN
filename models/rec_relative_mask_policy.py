"""Dataset-agnostic relative mask-transition policy over a frozen REC query."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.rec_joint_box_mask import LEGACY_MASK_POLICY_INDEX


MASK_POLICY_FEATURE_DIM = 52
GEOMETRY_FEATURE_DIM = 179
QUERY_COUNT = 16
VARIANT_COUNT = 7
ALLOWED_MASK_POLICY_INDICES = tuple(range(4, 15))
RELATIVE_MASK_POLICY_COUNT = len(ALLOWED_MASK_POLICY_INDICES)
LEGACY_ALLOWED_POLICY_POSITION = ALLOWED_MASK_POLICY_INDICES.index(
    LEGACY_MASK_POLICY_INDEX
)
TRANSITION_CLASS_COUNT = 3
TRANSITION_THRESHOLD_COUNT = 2
TRANSITION_BREAK = 0
TRANSITION_NEUTRAL = 1
TRANSITION_RESCUE = 2


class RelativeMaskTransitionPostprocessor(nn.Module):
    """Predict candidate-minus-anchor IoU and threshold transitions."""

    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("V103 requires hidden_dim=128")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.variant_encoder = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + MASK_POLICY_FEATURE_DIM, hidden_dim
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.query_context = nn.TransformerEncoder(layer, num_layers=1)
        self.delta_iou_head = nn.Linear(
            hidden_dim, RELATIVE_MASK_POLICY_COUNT
        )
        self.transition_head = nn.Linear(
            hidden_dim,
            RELATIVE_MASK_POLICY_COUNT
            * TRANSITION_THRESHOLD_COUNT
            * TRANSITION_CLASS_COUNT,
        )

    def forward(self, geometry_features, mask_features, variant_valid):
        if (not isinstance(geometry_features, torch.Tensor)
                or geometry_features.dtype != torch.float32
                or geometry_features.dim() != 4
                or tuple(geometry_features.shape[1:]) != (
                    QUERY_COUNT, VARIANT_COUNT, GEOMETRY_FEATURE_DIM)):
            raise ValueError("geometry_features must have shape [B,16,7,179]")
        batch_size = geometry_features.shape[0]
        if (not isinstance(mask_features, torch.Tensor)
                or mask_features.dtype != torch.float32
                or tuple(mask_features.shape) != (
                    batch_size, QUERY_COUNT, MASK_POLICY_FEATURE_DIM)):
            raise ValueError("mask_features must have shape [B,16,52]")
        if (not isinstance(variant_valid, torch.Tensor)
                or variant_valid.dtype != torch.bool
                or tuple(variant_valid.shape) != (
                    batch_size, QUERY_COUNT, VARIANT_COUNT)):
            raise ValueError("variant_valid must have shape [B,16,7]")
        if len({
                geometry_features.device,
                mask_features.device,
                variant_valid.device,
        }) != 1:
            raise ValueError("V103 inputs must share one device")
        if (not bool(variant_valid.reshape(batch_size, -1).any(dim=1).all())
                or not bool(torch.isfinite(geometry_features).all())
                or not bool(torch.isfinite(mask_features).all())):
            raise ValueError("V103 inputs must be finite with valid candidates")

        query_valid = variant_valid.any(dim=2)
        variant_mask = variant_valid.unsqueeze(-1)
        local = self.variant_encoder(torch.where(
            variant_mask, geometry_features, torch.zeros_like(geometry_features)
        ))
        local = torch.where(variant_mask, local, torch.zeros_like(local))
        count = variant_valid.sum(dim=2, keepdim=True).clamp_min(1)
        mean = local.sum(dim=2) / count.to(local.dtype)
        maximum = local.masked_fill(
            ~variant_mask, -float("inf")
        ).max(dim=2).values
        maximum = torch.where(
            query_valid.unsqueeze(-1), maximum, torch.zeros_like(maximum)
        )
        query_input = torch.cat((mean, maximum, mask_features), dim=-1)
        encoded = self.query_encoder(query_input)
        encoded = torch.where(
            query_valid.unsqueeze(-1), encoded, torch.zeros_like(encoded)
        )
        contextual = self.query_context(
            encoded, src_key_padding_mask=~query_valid
        )
        contextual = torch.where(
            query_valid.unsqueeze(-1), contextual, torch.zeros_like(contextual)
        )

        delta_iou = self.delta_iou_head(contextual).tanh()
        transition_logits = self.transition_head(contextual).reshape(
            batch_size,
            QUERY_COUNT,
            RELATIVE_MASK_POLICY_COUNT,
            TRANSITION_THRESHOLD_COUNT,
            TRANSITION_CLASS_COUNT,
        )
        transition_probabilities = transition_logits.softmax(dim=-1)
        query_mask = query_valid.unsqueeze(-1)
        transition_mask = query_mask.unsqueeze(-1).unsqueeze(-1)
        return {
            "delta_iou": torch.where(
                query_mask, delta_iou, torch.zeros_like(delta_iou)
            ),
            "transition_logits": torch.where(
                transition_mask,
                transition_logits,
                torch.zeros_like(transition_logits),
            ),
            "transition_probabilities": torch.where(
                transition_mask,
                transition_probabilities,
                torch.zeros_like(transition_probabilities),
            ),
            "query_valid": query_valid,
        }


def _validate_ensemble_output(output, reference=None):
    if not isinstance(output, dict):
        raise ValueError("V103 output must be an object")
    delta_iou = output.get("delta_iou")
    probabilities = output.get("transition_probabilities")
    query_valid = output.get("query_valid")
    if (not isinstance(delta_iou, torch.Tensor)
            or delta_iou.dim() != 3
            or delta_iou.shape[-1] != RELATIVE_MASK_POLICY_COUNT
            or not isinstance(probabilities, torch.Tensor)
            or probabilities.shape != delta_iou.shape + (
                TRANSITION_THRESHOLD_COUNT, TRANSITION_CLASS_COUNT)
            or not isinstance(query_valid, torch.Tensor)
            or query_valid.dtype != torch.bool
            or query_valid.shape != delta_iou.shape[:2]
            or delta_iou.device != probabilities.device
            or query_valid.device != delta_iou.device
            or not bool(torch.isfinite(delta_iou).all())
            or not bool(torch.isfinite(probabilities).all())):
        raise ValueError("V103 prediction tensors are malformed")
    if reference is not None:
        if (delta_iou.shape != reference[0]
                or delta_iou.device != reference[1]
                or not torch.equal(query_valid, reference[2])):
            raise ValueError("V103 seed outputs do not align")
    valid_probabilities = probabilities[query_valid]
    if (bool((valid_probabilities < 0.0).any())
            or bool((valid_probabilities > 1.0).any())
            or not torch.allclose(
                valid_probabilities.sum(dim=-1),
                torch.ones_like(valid_probabilities[..., 0]),
                rtol=1e-5,
                atol=1e-6,
            )):
        raise ValueError("V103 transition probabilities are invalid")
    return delta_iou, probabilities, query_valid


def select_relative_mask_policy_ensemble(
        seed_outputs, selected_parent_positions, aggregate_margin=0.02):
    """Apply the frozen three-seed worst-case mask-only selector."""
    if (not isinstance(seed_outputs, (list, tuple))
            or len(seed_outputs) != 3):
        raise ValueError("V103 selector requires exactly three seed outputs")
    validated = []
    reference = None
    for output in seed_outputs:
        current = _validate_ensemble_output(output, reference)
        if reference is None:
            reference = (current[0].shape, current[0].device, current[2])
        validated.append(current)
    delta_reference = validated[0][0]
    query_valid = validated[0][2]
    if (isinstance(aggregate_margin, bool)
            or not isinstance(aggregate_margin, (int, float))
            or not math.isfinite(float(aggregate_margin))
            or float(aggregate_margin) <= 0.0):
        raise ValueError("aggregate_margin must be finite and positive")
    selected_parent_positions = torch.as_tensor(
        selected_parent_positions, device=delta_reference.device
    )
    if (selected_parent_positions.dtype != torch.long
            or selected_parent_positions.shape != (
                delta_reference.shape[0],)
            or bool(((selected_parent_positions < 0)
                     | (selected_parent_positions
                        >= delta_reference.shape[1])).any())):
        raise ValueError("selected parent positions are malformed")
    rows = torch.arange(
        delta_reference.shape[0], device=delta_reference.device
    )
    if not bool(query_valid[rows, selected_parent_positions].all()):
        raise ValueError("selected REC parent query is invalid")

    seed_delta_iou = torch.stack([
        delta[rows, selected_parent_positions]
        for delta, _, _ in validated
    ], dim=0)
    seed_probabilities = torch.stack([
        probabilities[rows, selected_parent_positions]
        for _, probabilities, _ in validated
    ], dim=0)
    seed_effects = (
        seed_probabilities[..., TRANSITION_RESCUE]
        - seed_probabilities[..., TRANSITION_BREAK]
    )
    anchor_mask = torch.zeros(
        RELATIVE_MASK_POLICY_COUNT,
        dtype=torch.bool,
        device=delta_reference.device,
    )
    anchor_mask[LEGACY_ALLOWED_POLICY_POSITION] = True
    seed_delta_iou = torch.where(
        anchor_mask.view(1, 1, -1),
        torch.zeros_like(seed_delta_iou),
        seed_delta_iou,
    )
    seed_effects = torch.where(
        anchor_mask.view(1, 1, -1, 1),
        torch.zeros_like(seed_effects),
        seed_effects,
    )
    seed_aggregate = (
        seed_delta_iou
        + seed_effects[..., 0]
        + 2.0 * seed_effects[..., 1]
    )
    worst_delta_iou = seed_delta_iou.min(dim=0).values
    worst_effects = seed_effects.min(dim=0).values
    worst_aggregate = seed_aggregate.min(dim=0).values
    eligible = (
        worst_delta_iou.gt(0.0)
        & worst_effects[..., 0].gt(0.0)
        & worst_effects[..., 1].gt(0.0)
        & worst_aggregate.ge(float(aggregate_margin))
        & ~anchor_mask.unsqueeze(0)
    )
    scored = worst_aggregate.masked_fill(~eligible, -float("inf"))
    maximum = scored.max(dim=-1, keepdim=True).values
    ties = eligible & scored.eq(maximum)
    local_indices = torch.arange(
        RELATIVE_MASK_POLICY_COUNT, device=delta_reference.device
    ).expand_as(scored)
    proposal_local = local_indices.masked_fill(
        ~ties, RELATIVE_MASK_POLICY_COUNT
    ).min(dim=-1).values
    accepted = eligible.any(dim=-1)
    baseline_local = torch.full_like(
        proposal_local, LEGACY_ALLOWED_POLICY_POSITION
    )
    proposal_local = torch.where(
        accepted, proposal_local, baseline_local
    )
    allowed_mapping = torch.tensor(
        ALLOWED_MASK_POLICY_INDICES,
        dtype=torch.long,
        device=delta_reference.device,
    )
    proposal_original = allowed_mapping[proposal_local]
    selected_original = torch.where(
        accepted,
        proposal_original,
        torch.full_like(proposal_original, LEGACY_MASK_POLICY_INDEX),
    )
    selected_worst_delta = worst_delta_iou[rows, proposal_local]
    selected_worst_effects = worst_effects[rows, proposal_local]
    selected_worst_aggregate = worst_aggregate[rows, proposal_local]
    return {
        "selected_policy_indices": selected_original,
        "proposal_policy_indices": proposal_original,
        "selected_local_policy_positions": proposal_local,
        "accepted": accepted,
        "worst_aggregate_gain": selected_worst_aggregate,
        "worst_delta_iou": selected_worst_delta,
        "worst_effects": selected_worst_effects,
        "selected_parent_positions": selected_parent_positions.clone(),
    }


def select_relative_mask_policy_iou_priority_ensemble(
        seed_outputs, selected_parent_positions, aggregate_margin=0.02):
    """Apply V104 eligibility, then rank by worst delta-IoU first."""
    if (not isinstance(seed_outputs, (list, tuple))
            or len(seed_outputs) != 3):
        raise ValueError("V104 selector requires exactly three seed outputs")
    validated = []
    reference = None
    for output in seed_outputs:
        current = _validate_ensemble_output(output, reference)
        if reference is None:
            reference = (current[0].shape, current[0].device, current[2])
        validated.append(current)
    delta_reference = validated[0][0]
    query_valid = validated[0][2]
    if (isinstance(aggregate_margin, bool)
            or not isinstance(aggregate_margin, (int, float))
            or not math.isfinite(float(aggregate_margin))
            or float(aggregate_margin) <= 0.0):
        raise ValueError("aggregate_margin must be finite and positive")
    selected_parent_positions = torch.as_tensor(
        selected_parent_positions, device=delta_reference.device
    )
    if (selected_parent_positions.dtype != torch.long
            or selected_parent_positions.shape != (
                delta_reference.shape[0],)
            or bool(((selected_parent_positions < 0)
                     | (selected_parent_positions
                        >= delta_reference.shape[1])).any())):
        raise ValueError("selected parent positions are malformed")
    rows = torch.arange(
        delta_reference.shape[0], device=delta_reference.device
    )
    if not bool(query_valid[rows, selected_parent_positions].all()):
        raise ValueError("selected REC parent query is invalid")

    seed_delta_iou = torch.stack([
        delta[rows, selected_parent_positions]
        for delta, _, _ in validated
    ], dim=0)
    seed_probabilities = torch.stack([
        probabilities[rows, selected_parent_positions]
        for _, probabilities, _ in validated
    ], dim=0)
    seed_effects = (
        seed_probabilities[..., TRANSITION_RESCUE]
        - seed_probabilities[..., TRANSITION_BREAK]
    )
    anchor_mask = torch.zeros(
        RELATIVE_MASK_POLICY_COUNT,
        dtype=torch.bool,
        device=delta_reference.device,
    )
    anchor_mask[LEGACY_ALLOWED_POLICY_POSITION] = True
    seed_delta_iou = torch.where(
        anchor_mask.view(1, 1, -1),
        torch.zeros_like(seed_delta_iou),
        seed_delta_iou,
    )
    seed_effects = torch.where(
        anchor_mask.view(1, 1, -1, 1),
        torch.zeros_like(seed_effects),
        seed_effects,
    )
    seed_aggregate = (
        seed_delta_iou
        + seed_effects[..., 0]
        + 2.0 * seed_effects[..., 1]
    )
    worst_delta_iou = seed_delta_iou.min(dim=0).values
    worst_effects = seed_effects.min(dim=0).values
    worst_aggregate = seed_aggregate.min(dim=0).values
    eligible = (
        worst_delta_iou.gt(0.0)
        & worst_effects[..., 0].gt(0.0)
        & worst_effects[..., 1].gt(0.0)
        & worst_aggregate.ge(float(aggregate_margin))
        & ~anchor_mask.unsqueeze(0)
    )

    delta_scored = worst_delta_iou.masked_fill(~eligible, -float("inf"))
    maximum_delta = delta_scored.max(dim=-1, keepdim=True).values
    delta_ties = eligible & delta_scored.eq(maximum_delta)
    aggregate_scored = worst_aggregate.masked_fill(
        ~delta_ties, -float("inf")
    )
    maximum_aggregate = aggregate_scored.max(
        dim=-1, keepdim=True
    ).values
    ties = delta_ties & aggregate_scored.eq(maximum_aggregate)
    local_indices = torch.arange(
        RELATIVE_MASK_POLICY_COUNT, device=delta_reference.device
    ).expand_as(delta_scored)
    proposal_local = local_indices.masked_fill(
        ~ties, RELATIVE_MASK_POLICY_COUNT
    ).min(dim=-1).values
    accepted = eligible.any(dim=-1)
    baseline_local = torch.full_like(
        proposal_local, LEGACY_ALLOWED_POLICY_POSITION
    )
    proposal_local = torch.where(
        accepted, proposal_local, baseline_local
    )
    allowed_mapping = torch.tensor(
        ALLOWED_MASK_POLICY_INDICES,
        dtype=torch.long,
        device=delta_reference.device,
    )
    proposal_original = allowed_mapping[proposal_local]
    selected_original = torch.where(
        accepted,
        proposal_original,
        torch.full_like(proposal_original, LEGACY_MASK_POLICY_INDEX),
    )
    selected_worst_delta = worst_delta_iou[rows, proposal_local]
    selected_worst_effects = worst_effects[rows, proposal_local]
    selected_worst_aggregate = worst_aggregate[rows, proposal_local]
    return {
        "selected_policy_indices": selected_original,
        "proposal_policy_indices": proposal_original,
        "selected_local_policy_positions": proposal_local,
        "accepted": accepted,
        "worst_aggregate_gain": selected_worst_aggregate,
        "worst_delta_iou": selected_worst_delta,
        "worst_effects": selected_worst_effects,
        "selected_parent_positions": selected_parent_positions.clone(),
        "ranking": "worst_delta_iou_then_aggregate",
    }


def _transition_labels(anchor_iou, candidate_iou, threshold):
    anchor_hit = anchor_iou.gt(float(threshold))
    candidate_hit = candidate_iou.gt(float(threshold))
    result = torch.full_like(
        candidate_iou, TRANSITION_NEUTRAL, dtype=torch.long
    )
    result = torch.where(
        anchor_hit & ~candidate_hit,
        torch.full_like(result, TRANSITION_BREAK),
        result,
    )
    return torch.where(
        ~anchor_hit & candidate_hit,
        torch.full_like(result, TRANSITION_RESCUE),
        result,
    )


def compute_relative_mask_policy_loss(
        outputs, policy_ious, query_valid, target_temperature=0.25):
    """Frozen V103 relative regression, transition, listwise, regret loss."""
    if not isinstance(outputs, dict):
        raise ValueError("V103 outputs must be an object")
    delta_iou = outputs.get("delta_iou")
    transition_logits = outputs.get("transition_logits")
    transition_probabilities = outputs.get("transition_probabilities")
    output_query_valid = outputs.get("query_valid")
    if (not isinstance(delta_iou, torch.Tensor)
            or delta_iou.dim() != 3
            or delta_iou.shape[-1] != RELATIVE_MASK_POLICY_COUNT
            or not isinstance(transition_logits, torch.Tensor)
            or transition_logits.shape != delta_iou.shape + (
                TRANSITION_THRESHOLD_COUNT, TRANSITION_CLASS_COUNT)
            or not isinstance(transition_probabilities, torch.Tensor)
            or transition_probabilities.shape != transition_logits.shape
            or not isinstance(output_query_valid, torch.Tensor)
            or output_query_valid.dtype != torch.bool
            or output_query_valid.shape != delta_iou.shape[:2]):
        raise ValueError("V103 loss outputs are malformed")
    policy_ious = torch.as_tensor(
        policy_ious, device=delta_iou.device, dtype=delta_iou.dtype
    )
    if policy_ious.shape != delta_iou.shape[:2] + (15,):
        raise ValueError("policy_ious must have shape [B,16,15]")
    if (not isinstance(query_valid, torch.Tensor)
            or query_valid.dtype != torch.bool
            or query_valid.shape != delta_iou.shape[:2]
            or query_valid.device != delta_iou.device
            or not torch.equal(query_valid, output_query_valid)):
        raise ValueError("query_valid must match V103 outputs")
    if (not math.isfinite(float(target_temperature))
            or not 0.0 < float(target_temperature)):
        raise ValueError("target_temperature must be finite and positive")
    if not bool(torch.isfinite(policy_ious).all()):
        raise ValueError("policy_ious must be finite")

    allowed = torch.tensor(
        ALLOWED_MASK_POLICY_INDICES,
        dtype=torch.long,
        device=delta_iou.device,
    )
    candidate_iou = policy_ious.index_select(dim=-1, index=allowed)
    anchor_iou = policy_ious[..., LEGACY_MASK_POLICY_INDEX].unsqueeze(-1)
    target_delta_iou = candidate_iou - anchor_iou
    transition025 = _transition_labels(anchor_iou, candidate_iou, 0.25)
    transition050 = _transition_labels(anchor_iou, candidate_iou, 0.50)
    target_effect025 = transition025.to(delta_iou.dtype) - 1.0
    target_effect050 = transition050.to(delta_iou.dtype) - 1.0
    predicted_effects = (
        transition_probabilities[..., TRANSITION_RESCUE]
        - transition_probabilities[..., TRANSITION_BREAK]
    )
    anchor_mask = torch.zeros(
        RELATIVE_MASK_POLICY_COUNT,
        dtype=torch.bool,
        device=delta_iou.device,
    )
    anchor_mask[LEGACY_ALLOWED_POLICY_POSITION] = True
    candidate_valid = (
        query_valid.unsqueeze(-1).expand_as(delta_iou)
        & ~anchor_mask.view(1, 1, -1)
    )
    denominator = candidate_valid.sum().clamp_min(1).to(delta_iou.dtype)
    delta_loss = F.smooth_l1_loss(
        delta_iou, target_delta_iou, reduction="none"
    )
    transition025_loss = F.cross_entropy(
        transition_logits[..., 0, :].reshape(-1, TRANSITION_CLASS_COUNT),
        transition025.reshape(-1),
        reduction="none",
    ).reshape_as(delta_iou)
    transition050_loss = F.cross_entropy(
        transition_logits[..., 1, :].reshape(-1, TRANSITION_CLASS_COUNT),
        transition050.reshape(-1),
        reduction="none",
    ).reshape_as(delta_iou)

    neutralized_delta = torch.where(
        anchor_mask.view(1, 1, -1),
        torch.zeros_like(delta_iou),
        delta_iou,
    )
    neutralized_effects = torch.where(
        anchor_mask.view(1, 1, -1, 1),
        torch.zeros_like(predicted_effects),
        predicted_effects,
    )
    predicted_utility = (
        neutralized_delta
        + neutralized_effects[..., 0]
        + 2.0 * neutralized_effects[..., 1]
    )
    target_utility = (
        target_delta_iou + target_effect025 + 2.0 * target_effect050
    )
    target_distribution = (
        target_utility / float(target_temperature)
    ).softmax(dim=-1)
    listwise = -(
        target_distribution * predicted_utility.log_softmax(dim=-1)
    ).sum(dim=-1)
    predicted_distribution = predicted_utility.softmax(dim=-1)
    expected_target_utility = (
        predicted_distribution * target_utility
    ).sum(dim=-1)
    regret = (
        target_utility.max(dim=-1).values - expected_target_utility
    ).clamp_min(0.0)
    query_denominator = query_valid.sum().clamp_min(1).to(delta_iou.dtype)
    components = {
        "delta_iou": (delta_loss * candidate_valid).sum() / denominator,
        "transition025": (
            transition025_loss * candidate_valid
        ).sum() / denominator,
        "transition050": (
            transition050_loss * candidate_valid
        ).sum() / denominator,
        "listwise": (listwise * query_valid).sum() / query_denominator,
        "regret": (regret * query_valid).sum() / query_denominator,
    }
    total = (
        components["delta_iou"]
        + components["transition025"]
        + 2.0 * components["transition050"]
        + 0.5 * components["listwise"]
        + 0.5 * components["regret"]
    )
    return total, components
