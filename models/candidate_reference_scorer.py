"""Compare reference memories with the same candidate-pair scoring parameters."""

import torch

from .candidate_edge_direct_scorer import CandidateEdgeDirectScorer
from .encoder_decoder_layers import calc_pairwise_locs


class CandidateReferenceScorer(CandidateEdgeDirectScorer):
    """Separate target Queries from the supplied reference-memory axis.

    The parameterization is exactly CandidateEdgeDirectScorer. Supplying the
    full Query memory recovers that existing readout; supplying the existing
    object-input features changes only the reference evidence and its geometry.
    Target scores still use the original legal Query axis. Memory features are
    supplied by the caller, with no new encoder, GT anchor assignment or gate.
    """

    def forward(self, candidate_feats, candidate_boxes, text_feats,
                text_padding_mask, query_indices, valid_query_mask,
                memory_feats, memory_boxes, memory_valid_mask):
        batch_size, query_count, feat_dim = candidate_feats.shape
        target_count = query_indices.shape[1]
        targets = candidate_feats.gather(
            1, query_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
        target_valid = valid_query_mask.gather(1, query_indices)
        target_centers = candidate_boxes[..., :3].gather(
            1, query_indices.unsqueeze(-1).expand(-1, -1, 3))
        geometry = calc_pairwise_locs(torch.cat([
            target_centers, memory_boxes[..., :3]], dim=1))[:, :target_count, target_count:]
        geometry = torch.cat([
            geometry, geometry.new_zeros(batch_size, target_count, 1, 5)], dim=2)
        memory = torch.cat([
            memory_feats, self.null_anchor.expand(batch_size, -1, -1)], dim=1)
        memory_padding = torch.cat([
            ~memory_valid_mask, memory_valid_mask.new_zeros(batch_size, 1)], dim=1)
        token_attention = None
        if self.relation_readout == "global":
            global_text = text_feats.masked_fill(
                text_padding_mask.unsqueeze(-1), -float("inf")).max(dim=1).values
            features, anchor_attention = self.spatial(
                targets, memory, memory, geometry,
                key_padding_mask=memory_padding, txt_embeds=global_text)
        else:
            features, anchor_attention, token_attention = self._pair_readout(
                targets, memory, geometry, memory_padding, text_feats, text_padding_mask)
        logits = self.score_head(features).squeeze(-1).masked_fill(~target_valid, -float("inf"))
        query_scores = logits.new_full((batch_size, query_count), -float("inf")).scatter(
            1, query_indices, logits)
        return {"query_indices": query_indices, "candidate_valid_mask": target_valid,
                "candidate_logits": logits, "query_scores": query_scores,
                "anchor_attention": anchor_attention,
                "null_anchor_attention": anchor_attention[..., -1],
                "pair_token_attention": token_attention}
