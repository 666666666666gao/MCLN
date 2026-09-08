"""Isolated comparison of global and candidate-pair text relation readouts.

This prototype is not connected to MCLN.forward or its evaluator. The caller
supplies one shared target selection and the full legal Query memory. Scores
are uncalibrated logits, not predicted IoU or threshold-hit probabilities.
"""

import math

import torch
from torch import nn

from .encoder_decoder_layers import MultiHeadAttentionSpatial, calc_pairwise_locs


class CandidateEdgeDirectScorer(nn.Module):
    """Score K target Queries against all Q legal Queries and a null anchor.

    ``global`` retains the existing conditional spatial attention, with a
    masked full-sentence max pool. ``pair`` replaces only its text context:
    each ordered target-memory pair attends to the same full token sequence.
    Both modes retain the existing five-dimensional geometry and log-sigmoid
    geometric modulation, and share the spatial and final-head architecture.
    """

    def __init__(self, relation_readout, d_model=288, n_head=8, dropout=0.0):
        super().__init__()
        if relation_readout not in ("global", "pair"):
            raise ValueError("relation_readout must be 'global' or 'pair'")
        self.relation_readout = relation_readout
        self.d_model = d_model
        self.spatial = MultiHeadAttentionSpatial(
            d_model=d_model, n_head=n_head, dropout=dropout,
            spatial_multihead=True, spatial_dim=5, spatial_attn_fusion="cond",
        )
        self.null_anchor = nn.Parameter(torch.zeros(1, 1, d_model))
        self.score_head = nn.Linear(d_model, 1)
        if relation_readout == "pair":
            self.pair_query = nn.Sequential(
                nn.Linear(2 * d_model + 5, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.pair_text_attention = nn.MultiheadAttention(
                d_model, n_head, dropout=dropout,
            )

    def _pair_readout(self, targets, memory, geometry, memory_padding,
                      text_feats, text_padding_mask):
        batch_size, target_count, _ = targets.shape
        memory_count = memory.shape[1]
        pair_query = self.pair_query(torch.cat([
            targets.unsqueeze(2).expand(-1, -1, memory_count, -1),
            memory.unsqueeze(1).expand(-1, target_count, -1, -1),
            geometry,
        ], dim=-1))
        relation_text, token_attention = self.pair_text_attention(
            query=pair_query.reshape(batch_size, -1, self.d_model).transpose(0, 1),
            key=text_feats.transpose(0, 1),
            value=text_feats.transpose(0, 1),
            key_padding_mask=text_padding_mask,
            need_weights=True,
        )
        relation_text = relation_text.transpose(0, 1).reshape(
            batch_size, target_count, memory_count, self.d_model,
        )

        # Same theta -> sigmoid -> log modulation as the original cond path;
        # theta now has a separate full-token text context for each (i, j).
        weights = self.spatial.lang_cond_fc(targets.unsqueeze(2) + relation_text)
        weights = weights.reshape(
            batch_size, target_count, memory_count, self.spatial.n_head, 6,
        ).permute(3, 0, 1, 2, 4)
        loc_logits = (
            weights[..., 1:] * geometry.unsqueeze(0)
        ).sum(-1) + weights[..., 0]
        loc_bias = loc_logits.sigmoid().clamp(min=1e-6).log()

        q = self.spatial.w_qs(targets).reshape(
            batch_size, target_count, self.spatial.n_head, -1,
        ).permute(2, 0, 1, 3)
        k = self.spatial.w_ks(memory).reshape(
            batch_size, memory_count, self.spatial.n_head, -1,
        ).permute(2, 0, 1, 3)
        v = self.spatial.w_vs(memory).reshape(
            batch_size, memory_count, self.spatial.n_head, -1,
        ).permute(2, 0, 1, 3)
        attention_logits = torch.einsum("hbkd,hbmd->hbkm", q, k) / math.sqrt(q.shape[-1])
        attention_logits = (attention_logits + loc_bias).masked_fill(
            memory_padding.unsqueeze(0).unsqueeze(2), -float("inf"),
        )
        anchor_attention = attention_logits.softmax(-1)
        output = torch.einsum("hbkm,hbmd->hbkd", anchor_attention, v)
        output = output.permute(1, 2, 0, 3).reshape(
            batch_size, target_count, self.d_model,
        )
        output = self.spatial.layer_norm(
            targets + self.spatial.dropout(self.spatial.fc(output)),
        )
        return output, anchor_attention, token_attention.reshape(
            batch_size, target_count, memory_count, text_feats.shape[1],
        )

    def forward(self, candidate_feats, candidate_boxes, text_feats,
                text_padding_mask, query_indices, valid_query_mask):
        """Read fixed [B,Q,D] features and emit scores on the original Query axis.

        query_indices: [B,K], the caller's shared legal-Top-K preselection.
        valid_query_mask: [B,Q], the actual REC overlap mask, not the selector
            adapter's all-true mask. Invalid target slots remain excluded.
        text_padding_mask: [B,L], True for padding; at least one token is valid.

        Query/box/text tensors are not modified or detached here. Frozen
        backbone execution and training targets belong to the later paired
        experiment contract. No GT target or GT anchor is a forward input.
        """
        batch_size, query_count, feat_dim = candidate_feats.shape
        target_count = query_indices.shape[1]
        targets = candidate_feats.gather(
            1, query_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
        )
        target_valid = valid_query_mask.gather(1, query_indices)
        geometry = calc_pairwise_locs(candidate_boxes[..., :3]).gather(
            1, query_indices[:, :, None, None].expand(-1, -1, query_count, 5),
        )
        geometry = torch.cat([
            geometry, geometry.new_zeros(batch_size, target_count, 1, 5),
        ], dim=2)
        memory = torch.cat([
            candidate_feats, self.null_anchor.expand(batch_size, -1, -1),
        ], dim=1)
        memory_padding = torch.cat([
            ~valid_query_mask,
            valid_query_mask.new_zeros(batch_size, 1),
        ], dim=1)

        token_attention = None
        if self.relation_readout == "global":
            global_text = text_feats.masked_fill(
                text_padding_mask.unsqueeze(-1), -float("inf"),
            ).max(dim=1).values
            features, anchor_attention = self.spatial(
                targets, memory, memory, geometry,
                key_padding_mask=memory_padding, txt_embeds=global_text,
            )
        else:
            features, anchor_attention, token_attention = self._pair_readout(
                targets, memory, geometry, memory_padding,
                text_feats, text_padding_mask,
            )

        candidate_logits = self.score_head(features).squeeze(-1).masked_fill(
            ~target_valid, -float("inf"),
        )
        query_scores = candidate_logits.new_full(
            (batch_size, query_count), -float("inf"),
        ).scatter(1, query_indices, candidate_logits)
        return {
            "query_indices": query_indices,
            "candidate_valid_mask": target_valid,
            "candidate_logits": candidate_logits,
            "query_scores": query_scores,
            "anchor_attention": anchor_attention,
            "null_anchor_attention": anchor_attention[..., -1],
            "pair_token_attention": token_attention,
        }
