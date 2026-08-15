"""Semantic candidate-vs-baseline utility critic for REC reranking."""

import torch
from torch import nn

from models.rec_semantic_candidate_critic import (
    V121_AUX_DIM,
    V121_AUX_LATENT_DIM,
    V121_DROPOUT,
    V121_LATENT_DIM,
    V121_QUERY_EMBED_DIM,
    V121_TEXT_DIM,
    V121_THRESHOLD_COUNT,
    V121_VARIANT_EMBED_DIM,
)


V122_CANDIDATE_DIM = 64
V122_PAIR_HIDDEN_DIM = 128
V122_MODEL_GAIN_DIM = 2


def _projection(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
    )


class SemanticPairwiseUtilityCritic(nn.Module):
    """Classify fix versus break for a candidate relative to its baseline."""

    def __init__(
            self, candidate_dim=V122_CANDIDATE_DIM,
            pair_hidden_dim=V122_PAIR_HIDDEN_DIM,
            dropout=V121_DROPOUT):
        super().__init__()
        if (candidate_dim != V122_CANDIDATE_DIM
                or pair_hidden_dim != V122_PAIR_HIDDEN_DIM
                or float(dropout) != V121_DROPOUT):
            raise ValueError("V122 architecture is frozen")
        self.query_projection = _projection(
            V121_QUERY_EMBED_DIM, V121_LATENT_DIM
        )
        self.variant_projection = _projection(
            V121_VARIANT_EMBED_DIM, V121_LATENT_DIM
        )
        self.text_projection = _projection(
            V121_TEXT_DIM, V121_LATENT_DIM
        )
        self.aux_projection = _projection(
            V121_AUX_DIM, V121_AUX_LATENT_DIM
        )
        candidate_input_dim = (
            6 * V121_LATENT_DIM + V121_AUX_LATENT_DIM
        )
        self.candidate_projection = _projection(
            candidate_input_dim, candidate_dim
        )
        pair_input_dim = 5 * candidate_dim + V122_MODEL_GAIN_DIM
        self.utility_head = nn.Sequential(
            nn.Linear(pair_input_dim, pair_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(pair_hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(pair_hidden_dim, V121_THRESHOLD_COUNT),
        )

    @staticmethod
    def _validate(value, width, name, count=None):
        if (not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.dim() != 2
                or value.shape[0] <= 0
                or value.shape[1] != width):
            raise TypeError(
                "V122 {} must be nonempty float32 [N,{}]".format(
                    name, width
                )
            )
        if count is not None and value.shape[0] != count:
            raise ValueError("V122 pair tensors must align")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("V122 {} must be finite".format(name))
        return value.shape[0]

    def _encode_candidate(
            self, query_embedding, variant_embedding,
            text_latent, candidate_aux):
        query = self.query_projection(query_embedding)
        variant = self.variant_projection(variant_embedding)
        aux = self.aux_projection(candidate_aux)
        fused = torch.cat((
            query,
            variant,
            text_latent,
            query * text_latent,
            variant * text_latent,
            torch.abs(query - variant),
            aux,
        ), dim=1)
        return self.candidate_projection(fused)

    def forward(
            self, proposal_query_embedding, proposal_variant_embedding,
            baseline_query_embedding, baseline_variant_embedding,
            target_text, proposal_aux, baseline_aux, model_gain):
        count = self._validate(
            proposal_query_embedding, V121_QUERY_EMBED_DIM,
            "proposal_query_embedding"
        )
        tensors = (
            (proposal_variant_embedding, V121_VARIANT_EMBED_DIM,
             "proposal_variant_embedding"),
            (baseline_query_embedding, V121_QUERY_EMBED_DIM,
             "baseline_query_embedding"),
            (baseline_variant_embedding, V121_VARIANT_EMBED_DIM,
             "baseline_variant_embedding"),
            (target_text, V121_TEXT_DIM, "target_text"),
            (proposal_aux, V121_AUX_DIM, "proposal_aux"),
            (baseline_aux, V121_AUX_DIM, "baseline_aux"),
            (model_gain, V122_MODEL_GAIN_DIM, "model_gain"),
        )
        for value, width, name in tensors:
            self._validate(value, width, name, count=count)
        text = self.text_projection(target_text)
        proposal = self._encode_candidate(
            proposal_query_embedding,
            proposal_variant_embedding,
            text,
            proposal_aux,
        )
        baseline = self._encode_candidate(
            baseline_query_embedding,
            baseline_variant_embedding,
            text,
            baseline_aux,
        )
        pair = torch.cat((
            proposal,
            baseline,
            proposal - baseline,
            torch.abs(proposal - baseline),
            proposal * baseline,
            model_gain,
        ), dim=1)
        return self.utility_head(pair)
