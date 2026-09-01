"""Parent-relative deployment for the structured SACR score head."""

import math

import torch
import torch.nn as nn


def _validate_max_delta(max_delta):
    if (
            not isinstance(max_delta, (float, int))
            or isinstance(max_delta, bool)
            or not math.isfinite(float(max_delta))
            or not 0.0 < float(max_delta) <= 0.25):
        raise ValueError("parent-relative SACR max_delta must be in (0,0.25]")


def _validate_promotion_margin(promotion_margin, max_delta):
    if (
            not isinstance(promotion_margin, (float, int))
            or isinstance(promotion_margin, bool)
            or not math.isfinite(float(promotion_margin))
            or not 0.0 <= float(promotion_margin) < float(max_delta)):
        raise ValueError(
            "parent-relative SACR promotion_margin must be in [0,max_delta)"
        )


def build_parent_relative_sacr_context(
        raw_scores, parent_scores, candidate_valid, max_delta,
        promotion_margin=0.0):
    """Build the immutable parent anchor and residual-feasibility contract."""
    if (
            not isinstance(raw_scores, torch.Tensor)
            or raw_scores.dim() != 2
            or not isinstance(parent_scores, torch.Tensor)
            or parent_scores.shape != raw_scores.shape
            or not isinstance(candidate_valid, torch.Tensor)
            or candidate_valid.dtype != torch.bool
            or candidate_valid.shape != raw_scores.shape):
        raise ValueError(
            "parent-relative SACR score inputs must align as [B,Q]"
        )
    if len({
            raw_scores.device,
            parent_scores.device,
            candidate_valid.device}) != 1:
        raise ValueError("parent-relative SACR inputs must share a device")
    _validate_max_delta(max_delta)
    _validate_promotion_margin(promotion_margin, max_delta)

    valid_rows = candidate_valid.any(dim=1)
    masked_parent_scores = parent_scores.float().masked_fill(
        ~candidate_valid, torch.finfo(torch.float32).min
    )
    parent_indices = masked_parent_scores.argmax(dim=1)
    parent_raw = torch.gather(
        raw_scores.float(), 1, parent_indices.unsqueeze(1)
    )
    relative_raw = raw_scores.float() - parent_raw
    query_indices = torch.arange(
        raw_scores.shape[1], device=raw_scores.device
    ).unsqueeze(0)
    parent_valid = (
        candidate_valid & (query_indices == parent_indices.unsqueeze(1))
    )
    non_parent_valid = candidate_valid & ~parent_valid
    parent_score = torch.gather(
        parent_scores.float(), 1, parent_indices.unsqueeze(1)
    )
    promotion_budget = parent_score - parent_scores.float()
    feasible_candidate = (
        non_parent_valid
        & (
            promotion_budget + float(promotion_margin)
            < float(max_delta)
        )
    )
    return {
        "valid_rows": valid_rows,
        "parent_indices": parent_indices,
        "parent_valid_mask": parent_valid,
        "non_parent_valid_mask": non_parent_valid,
        "relative_raw_scores": relative_raw,
        "promotion_budget": promotion_budget,
        "feasible_candidate_mask": feasible_candidate,
    }


class SACRParentRelativeGate(nn.Module):
    """Predict per-sample trust from dataset-agnostic reliability evidence."""

    FEATURE_NAMES = (
        "parse_confidence",
        "anchor_top1_mass",
        "anchor_concentration",
        "relation_active_ratio",
        "parent_score_margin",
        "best_feasible_relative_score",
        "feasible_budget_ease",
        "parent_structured_disagreement",
    )

    def __init__(self, hidden_dim=32, top_k_anchors=16):
        super().__init__()
        if (
                not isinstance(hidden_dim, int)
                or isinstance(hidden_dim, bool)
                or hidden_dim < 1):
            raise ValueError("parent-relative gate hidden_dim must be positive")
        if (
                not isinstance(top_k_anchors, int)
                or isinstance(top_k_anchors, bool)
                or top_k_anchors < 1):
            raise ValueError(
                "parent-relative gate top_k_anchors must be positive"
            )
        self.top_k_anchors = int(top_k_anchors)
        self.network = nn.Sequential(
            nn.Linear(len(self.FEATURE_NAMES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start conservatively without creating a dead zero-gradient gate.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.0)

    def forward(
            self, raw_scores, parent_scores, candidate_valid,
            structured_valid_mask, parse_confidence, anchor_top1_mass,
            anchor_entropy, relation_active_ratio, max_delta,
            promotion_margin=0.0):
        context = build_parent_relative_sacr_context(
            raw_scores,
            parent_scores,
            candidate_valid,
            max_delta,
            promotion_margin=promotion_margin,
        )
        batch_size, query_count = raw_scores.shape
        row_features = (
            parse_confidence,
            anchor_top1_mass,
            anchor_entropy,
            relation_active_ratio,
        )
        if (
                not isinstance(structured_valid_mask, torch.Tensor)
                or structured_valid_mask.dtype != torch.bool
                or structured_valid_mask.shape != (batch_size,)
                or any(
                    not isinstance(value, torch.Tensor)
                    or value.shape != (batch_size,)
                    for value in row_features
                )):
            raise ValueError(
                "parent-relative gate reliability inputs must align as [B]"
            )
        if any(
                value.device != raw_scores.device
                for value in row_features + (structured_valid_mask,)):
            raise ValueError(
                "parent-relative gate reliability inputs must share a device"
            )

        masked_scores = parent_scores.float().masked_fill(
            ~candidate_valid, torch.finfo(torch.float32).min
        )
        top_count = min(2, query_count)
        top_scores = torch.topk(masked_scores, top_count, dim=1).values
        parent_margin = (
            top_scores[:, 0] - top_scores[:, 1]
            if top_count == 2 else top_scores.new_zeros(batch_size)
        )
        if top_count == 2:
            parent_margin = torch.where(
                candidate_valid.sum(dim=1) >= 2,
                parent_margin,
                parent_margin.new_zeros(batch_size),
            )
        feasible = context["feasible_candidate_mask"]
        relative_raw = context["relative_raw_scores"]
        negative_large = torch.finfo(torch.float32).min
        best_relative, best_indices = relative_raw.masked_fill(
            ~feasible, negative_large
        ).max(dim=1)
        has_feasible = feasible.any(dim=1)
        best_relative = torch.where(
            has_feasible, best_relative, best_relative.new_zeros(batch_size)
        )
        positive_large = torch.finfo(torch.float32).max
        min_budget = context["promotion_budget"].masked_fill(
            ~feasible, positive_large
        ).amin(dim=1)
        budget_ease = torch.where(
            has_feasible,
            1.0 - min_budget / float(max_delta),
            min_budget.new_zeros(batch_size),
        ).clamp(min=0.0, max=1.0)
        disagreement = (
            has_feasible
            & (best_indices != context["parent_indices"])
            & (best_relative > 0.0)
        ).float()
        anchor_count = max(min(self.top_k_anchors, query_count), 2)
        anchor_concentration = (
            1.0 - anchor_entropy.float() / math.log(float(anchor_count))
        ).clamp(min=0.0, max=1.0)
        features = torch.stack((
            parse_confidence.float().clamp(min=0.0, max=1.0),
            anchor_top1_mass.float().clamp(min=0.0, max=1.0),
            anchor_concentration,
            relation_active_ratio.float().clamp(min=0.0, max=1.0),
            parent_margin.clamp(min=0.0).tanh(),
            best_relative.tanh(),
            budget_ease,
            disagreement,
        ), dim=1)
        gate_logits = self.network(features).squeeze(1)
        active_rows = (
            structured_valid_mask & context["valid_rows"] & has_feasible
        )
        sample_gate = torch.sigmoid(gate_logits) * active_rows.float()
        return {
            "sample_gate": sample_gate,
            "gate_logits": gate_logits,
            "features": features,
            "active_rows": active_rows,
            **context,
        }


def apply_parent_relative_sacr_refinement(
        raw_scores, parent_scores, candidate_valid, structured_valid_mask,
        sample_gate, max_delta, promotion_margin=0.0):
    """Apply a bounded residual controlled only by a per-sample trust gate."""
    context = build_parent_relative_sacr_context(
        raw_scores,
        parent_scores,
        candidate_valid,
        max_delta,
        promotion_margin=promotion_margin,
    )
    if (
            not isinstance(structured_valid_mask, torch.Tensor)
            or structured_valid_mask.dtype != torch.bool
            or structured_valid_mask.shape != (raw_scores.shape[0],)
            or not isinstance(sample_gate, torch.Tensor)
            or sample_gate.shape != (raw_scores.shape[0],)):
        raise ValueError(
            "parent-relative SACR deployment inputs must align as [B]"
        )
    if (
            structured_valid_mask.device != raw_scores.device
            or sample_gate.device != raw_scores.device):
        raise ValueError("parent-relative SACR inputs must share a device")
    if bool((~torch.isfinite(sample_gate)).any().item()) or bool(
            ((sample_gate < 0.0) | (sample_gate > 1.0)).any().item()):
        raise ValueError("parent-relative SACR sample gate must be in [0,1]")

    active_rows = structured_valid_mask & context["valid_rows"]
    feasible = context["feasible_candidate_mask"]
    sample_gate = (
        sample_gate.float() * active_rows.float() * feasible.any(dim=1).float()
    )
    # Infeasible candidates cannot overtake the parent inside max_delta, so
    # modifying them only adds ranking risk without a possible REC repair.
    apply_mask = active_rows.unsqueeze(1) & (
        context["parent_valid_mask"] | feasible
    )
    residual = (
        float(max_delta)
        * sample_gate.unsqueeze(1)
        * context["relative_raw_scores"].tanh()
    ).masked_fill(~apply_mask, 0.0)
    refined_scores = torch.where(
        apply_mask, parent_scores.float() + residual, parent_scores.float()
    )
    return {
        "scores": refined_scores,
        "residual": residual,
        "sample_gate": sample_gate,
        "apply_mask": apply_mask,
        **context,
    }
