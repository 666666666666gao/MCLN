"""Structured Anchor-Compositional Reasoning score head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACRHead(nn.Module):
    """Score target/attribute compatibility and relation-anchor geometry."""

    def __init__(self, d_model=288, hidden_dim=288, top_m_targets=32,
                 top_k_anchors=16, geo_dim=16):
        super().__init__()
        if min(top_m_targets, top_k_anchors) < 1:
            raise ValueError("SACR top-k settings must be positive")
        if geo_dim < 0:
            raise ValueError("geo_dim must be non-negative")
        self.top_m_targets = int(top_m_targets)
        self.top_k_anchors = int(top_k_anchors)
        self.geo_dim = int(geo_dim)
        self.target_attr_mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.ReLU(),
            # Joint source mixing consumes within-row ranks, so a constant
            # output offset cannot affect this source or its target shortlist.
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.anchor_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            # Anchor probabilities are a softmax over queries; this scalar
            # bias would cancel exactly for every relation slot.
            nn.Linear(hidden_dim, 1, bias=False),
        )
        relation_dim = 3 * d_model + self.geo_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.geo_encoder = (
            nn.Sequential(nn.Linear(11, self.geo_dim), nn.ReLU())
            if self.geo_dim > 0
            else None
        )

    def forward(self, query_feats, pred_boxes, base_scores, slot_dict,
                global_only_mask=None, weak_generic_target_mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            query_feats = query_feats.float()
            pred_boxes = pred_boxes.float()
            base_scores = base_scores.float()
            batch_size, query_count, feature_dim = query_feats.shape
            device = query_feats.device
            coverage = slot_dict.get("coverage_stats", {})
            has_target = coverage.get(
                "has_target",
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            ).to(device=device).bool()
            if global_only_mask is None:
                global_only_mask = torch.zeros_like(has_target)
            if weak_generic_target_mask is None:
                weak_generic_target_mask = torch.zeros_like(has_target)
            global_only_mask = global_only_mask.to(device=device).bool()
            weak_generic_target_mask = weak_generic_target_mask.to(
                device=device
            ).bool()
            structured_valid = has_target & ~global_only_mask

            global_slot = slot_dict["global_slot"].float()
            target_slot = slot_dict["target_slot"].float()
            attr_slot = slot_dict["attr_slot"].float()
            global_query = global_slot.unsqueeze(1).expand(
                batch_size, query_count, feature_dim
            )
            target_query = target_slot.unsqueeze(1).expand_as(global_query)
            target_query = target_query * (
                ~weak_generic_target_mask
            ).float().view(batch_size, 1, 1)
            attr_query = attr_slot.unsqueeze(1).expand_as(global_query)
            target_attr_scores = self.target_attr_mlp(torch.cat((
                query_feats, target_query, attr_query, global_query
            ), dim=-1)).squeeze(-1)
            target_attr_scores = target_attr_scores + 0.1 * self.global_mlp(
                torch.cat((query_feats, global_query), dim=-1)
            ).squeeze(-1)

            top_count = min(self.top_m_targets, query_count)
            top_indices = torch.topk(
                base_scores + target_attr_scores, top_count, dim=1
            ).indices
            relation_scores, anchor_probs = self._relation_scores(
                query_feats,
                pred_boxes,
                top_indices,
                slot_dict["rel_slots"].float(),
                slot_dict["anchor_slots"].float(),
                slot_dict["slot_mask"].to(device=device).bool(),
            )
            valid_values = structured_valid.float().unsqueeze(1)
            target_attr_scores = target_attr_scores * valid_values
            relation_scores = relation_scores * valid_values
            structured_scores = target_attr_scores + relation_scores

            slot_mask = slot_dict["slot_mask"].to(device=device).bool()
            active = slot_mask.float().sum(dim=1).clamp(min=1.0)
            entropy = -(
                anchor_probs * (anchor_probs + 1e-8).log()
            ).sum(dim=-1)
            anchor_entropy = (
                entropy * slot_mask.float()
            ).sum(dim=1) / active
            anchor_top1_mass = (
                anchor_probs.max(dim=-1).values * slot_mask.float()
            ).sum(dim=1) / active
        return {
            "structured_scores": structured_scores,
            "target_attr_scores": target_attr_scores,
            "relation_anchor_scores": relation_scores,
            "structured_valid_mask": structured_valid,
            "anchor_entropy": anchor_entropy,
            "anchor_top1_mass": anchor_top1_mass,
            "relation_active_ratio": slot_mask.float().mean(),
        }

    def _relation_scores(self, query_feats, boxes, top_indices, rel_slots,
                         anchor_slots, slot_mask):
        batch_size, query_count, feature_dim = query_feats.shape
        pair_count = rel_slots.shape[1]
        target_count = top_indices.shape[1]
        anchor_count = min(self.top_k_anchors, query_count)
        query_by_pair = query_feats.unsqueeze(1).expand(
            batch_size, pair_count, query_count, feature_dim
        )
        anchor_by_pair = anchor_slots.unsqueeze(2).expand_as(query_by_pair)
        anchor_logits = self.anchor_mlp(torch.cat((
            query_by_pair, anchor_by_pair
        ), dim=-1)).squeeze(-1)
        anchor_indices = torch.topk(
            anchor_logits, anchor_count, dim=2
        ).indices
        anchor_probs = F.softmax(
            torch.gather(anchor_logits, 2, anchor_indices), dim=2
        )

        target_feats = torch.gather(
            query_feats,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, feature_dim),
        )
        target_boxes = torch.gather(
            boxes, 1, top_indices.unsqueeze(-1).expand(-1, -1, 6)
        )
        anchor_feats = torch.gather(
            query_by_pair,
            2,
            anchor_indices.unsqueeze(-1).expand(-1, -1, -1, feature_dim),
        )
        boxes_by_pair = boxes.unsqueeze(1).expand(
            batch_size, pair_count, query_count, 6
        )
        anchor_boxes = torch.gather(
            boxes_by_pair,
            2,
            anchor_indices.unsqueeze(-1).expand(-1, -1, -1, 6),
        )
        target_feats = target_feats[:, None, :, None, :].expand(
            -1, pair_count, -1, anchor_count, -1
        )
        anchor_feats = anchor_feats[:, :, None, :, :].expand(
            -1, -1, target_count, -1, -1
        )
        relation_feats = rel_slots[:, :, None, None, :].expand(
            -1, -1, target_count, anchor_count, -1
        )
        inputs = [target_feats, anchor_feats, relation_feats]
        if self.geo_encoder is not None:
            geometry = self._pair_geometry(
                target_boxes[:, None, :, None, :],
                anchor_boxes[:, :, None, :, :],
            )
            inputs.append(self.geo_encoder(geometry))
        relation_logits = self.relation_mlp(
            torch.cat(inputs, dim=-1)
        ).squeeze(-1)
        weighted = (
            relation_logits * anchor_probs.unsqueeze(2)
        ).sum(dim=3)
        target_relation = (
            weighted * slot_mask.float().unsqueeze(2)
        ).sum(dim=1)
        result = query_feats.new_zeros(batch_size, query_count)
        result.scatter_(1, top_indices, target_relation)
        return result, anchor_probs

    @staticmethod
    def _pair_geometry(first, second):
        first_center = first[..., :3]
        second_center = second[..., :3]
        delta = first_center - second_center
        distance = delta.square().sum(dim=-1).add(1e-6).sqrt()
        direction = delta / distance.unsqueeze(-1).clamp(min=1e-6)
        first_size = first[..., 3:].abs().clamp(min=1e-6)
        second_size = second[..., 3:].abs().clamp(min=1e-6)
        size_ratio = first_size / second_size
        volume_ratio = first_size.prod(dim=-1) / second_size.prod(dim=-1)
        return torch.cat((
            delta,
            distance.unsqueeze(-1),
            direction,
            volume_ratio.unsqueeze(-1),
            size_ratio,
        ), dim=-1)
