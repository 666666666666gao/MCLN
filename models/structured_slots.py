"""Span-to-slot structured language decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredSlotBuilder(nn.Module):
    """Pool aligned target, attribute, relation, and anchor spans."""

    def __init__(self, d_model=288, pooling="attention", max_pairs=3):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError("pooling must be attention or mean")
        if not isinstance(max_pairs, int) or max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer")
        self.pooling = pooling
        self.max_pairs = max_pairs
        if pooling == "attention":
            # A scalar bias is constant over every token in a span and is
            # therefore cancelled exactly by the pooling softmax.
            self.global_attn = nn.Linear(d_model, 1, bias=False)
            self.target_attn = nn.Linear(d_model, 1, bias=False)
            self.attr_attn = nn.Linear(d_model, 1, bias=False)
            self.rel_attn = nn.Linear(d_model, 1, bias=False)
            self.anchor_attn = nn.Linear(d_model, 1, bias=False)

    @staticmethod
    def _span_masks(spans, token_count):
        starts = spans[..., 0]
        ends = spans[..., 1]
        valid = (starts >= 0) & (ends > starts) & (ends <= token_count)
        positions = torch.arange(token_count, device=spans.device)
        masks = (
            (positions >= starts.unsqueeze(-1))
            & (positions < ends.unsqueeze(-1))
            & valid.unsqueeze(-1)
        )
        return masks, valid

    def _pool_spans(self, token_feats, spans, attention):
        batch_size, token_count, feature_dim = token_feats.shape
        if spans is None:
            return (
                token_feats.new_zeros(batch_size, 1, feature_dim),
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=token_feats.device
                ),
            )
        masks, valid = self._span_masks(spans, token_count)
        mask_values = masks.to(dtype=token_feats.dtype)
        if self.pooling == "attention":
            logits = attention(token_feats).squeeze(-1).unsqueeze(1)
            logits = logits.expand(-1, spans.shape[1], -1)
            logits = logits.masked_fill(~masks, -1e4)
            weights = F.softmax(logits, dim=-1) * mask_values
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(
                min=1e-6
            )
        else:
            weights = mask_values / mask_values.sum(
                dim=-1, keepdim=True
            ).clamp(min=1.0)
        pooled = torch.matmul(weights, token_feats)
        return pooled * valid.unsqueeze(-1).to(pooled.dtype), valid

    def _global_slot(self, token_feats, attention_mask):
        valid = attention_mask.bool()
        if self.pooling == "attention":
            logits = self.global_attn(token_feats).squeeze(-1)
            logits = logits.masked_fill(~valid, -1e4)
            weights = F.softmax(logits, dim=-1) * valid.to(token_feats.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(
                min=1e-6
            )
        else:
            weights = valid.to(token_feats.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(
                min=1.0
            )
        return torch.bmm(weights.unsqueeze(1), token_feats).squeeze(1)

    def forward(self, token_feats, attention_mask, target_spans=None,
                entity_spans=None, attr_spans=None, rel_spans=None,
                anchor_ids=None):
        batch_size, _, feature_dim = token_feats.shape
        global_slot = self._global_slot(token_feats, attention_mask)

        target_candidates = target_spans
        if target_candidates is None:
            target_candidates = entity_spans
        target_reps, target_valid = self._pool_spans(
            token_feats, target_candidates, getattr(self, "target_attn", None)
        )
        target_count = target_valid.sum(dim=1).clamp(min=1).to(
            token_feats.dtype
        )
        target_slot = target_reps.sum(dim=1) / target_count.unsqueeze(-1)
        has_target = target_valid.any(dim=1)
        target_slot = target_slot * has_target.unsqueeze(-1).to(
            target_slot.dtype
        )

        attr_reps, attr_valid = self._pool_spans(
            token_feats, attr_spans, getattr(self, "attr_attn", None)
        )
        num_attrs = attr_valid.sum(dim=1)
        attr_slot = attr_reps.sum(dim=1) / num_attrs.clamp(
            min=1
        ).to(token_feats.dtype).unsqueeze(-1)
        attr_slot = attr_slot * (num_attrs > 0).unsqueeze(-1).to(
            attr_slot.dtype
        )

        rel_slots = token_feats.new_zeros(
            batch_size, self.max_pairs, feature_dim
        )
        anchor_slots = token_feats.new_zeros(
            batch_size, self.max_pairs, feature_dim
        )
        slot_mask = torch.zeros(
            batch_size,
            self.max_pairs,
            dtype=torch.bool,
            device=token_feats.device,
        )
        if rel_spans is not None:
            relation_count = min(rel_spans.shape[1], self.max_pairs)
            rel_reps, rel_valid = self._pool_spans(
                token_feats,
                rel_spans[:, :relation_count],
                getattr(self, "rel_attn", None),
            )
            if entity_spans is not None and anchor_ids is not None:
                ids = anchor_ids[:, :relation_count]
                in_range = (ids >= 0) & (ids < entity_spans.shape[1])
                safe_ids = ids.clamp(0, max(entity_spans.shape[1] - 1, 0))
                anchor_spans = torch.gather(
                    entity_spans,
                    1,
                    safe_ids.unsqueeze(-1).expand(-1, -1, 2),
                )
                anchor_reps, anchor_valid = self._pool_spans(
                    token_feats,
                    anchor_spans,
                    getattr(self, "anchor_attn", None),
                )
                pair_valid = rel_valid & anchor_valid & in_range
            else:
                anchor_reps = rel_reps.new_zeros(rel_reps.shape)
                pair_valid = torch.zeros_like(rel_valid)
            rel_slots[:, :relation_count] = rel_reps * pair_valid.unsqueeze(
                -1
            ).to(rel_reps.dtype)
            anchor_slots[:, :relation_count] = anchor_reps * pair_valid.unsqueeze(
                -1
            ).to(anchor_reps.dtype)
            slot_mask[:, :relation_count] = pair_valid

        return {
            "global_slot": global_slot,
            "target_slot": target_slot,
            "attr_slot": attr_slot,
            "rel_slots": rel_slots,
            "anchor_slots": anchor_slots,
            "slot_mask": slot_mask,
            "coverage_stats": {
                "has_target": has_target,
                "num_attrs": num_attrs,
                "num_pairs": slot_mask.sum(dim=1),
            },
        }
