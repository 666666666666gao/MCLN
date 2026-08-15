"""Candidate-level semantic hit critic for REC safety calibration."""

import torch
from torch import nn


V121_QUERY_EMBED_DIM = 128
V121_VARIANT_EMBED_DIM = 128
V121_TEXT_DIM = 64
V121_AUX_DIM = 16
V121_LATENT_DIM = 64
V121_AUX_LATENT_DIM = 32
V121_HIDDEN_DIM = 128
V121_THRESHOLD_COUNT = 2
V121_DROPOUT = 0.1


def _projection(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
    )


class SemanticCandidateHitCritic(nn.Module):
    """Predict threshold hits from contextual object, language, and geometry."""

    def __init__(
            self, latent_dim=V121_LATENT_DIM,
            hidden_dim=V121_HIDDEN_DIM, dropout=V121_DROPOUT):
        super().__init__()
        if (latent_dim != V121_LATENT_DIM
                or hidden_dim != V121_HIDDEN_DIM
                or float(dropout) != V121_DROPOUT):
            raise ValueError("V121 architecture is frozen")
        self.query_projection = _projection(
            V121_QUERY_EMBED_DIM, latent_dim
        )
        self.variant_projection = _projection(
            V121_VARIANT_EMBED_DIM, latent_dim
        )
        self.text_projection = _projection(V121_TEXT_DIM, latent_dim)
        self.aux_projection = _projection(
            V121_AUX_DIM, V121_AUX_LATENT_DIM
        )
        fusion_dim = 6 * latent_dim + V121_AUX_LATENT_DIM
        self.hit_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, V121_THRESHOLD_COUNT),
        )

    def forward(
            self, query_embedding, variant_embedding,
            target_text, candidate_aux):
        tensors = (
            (query_embedding, V121_QUERY_EMBED_DIM, "query_embedding"),
            (variant_embedding, V121_VARIANT_EMBED_DIM, "variant_embedding"),
            (target_text, V121_TEXT_DIM, "target_text"),
            (candidate_aux, V121_AUX_DIM, "candidate_aux"),
        )
        count = None
        for value, width, name in tensors:
            if (not isinstance(value, torch.Tensor)
                    or value.dtype != torch.float32
                    or value.dim() != 2
                    or value.shape[0] <= 0
                    or value.shape[1] != width):
                raise TypeError(
                    "V121 {} must be nonempty float32 [N,{}]".format(
                        name, width
                    )
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("V121 {} must be finite".format(name))
            if count is None:
                count = value.shape[0]
            elif value.shape[0] != count:
                raise ValueError("V121 candidate tensors must align")
        query = self.query_projection(query_embedding)
        variant = self.variant_projection(variant_embedding)
        text = self.text_projection(target_text)
        aux = self.aux_projection(candidate_aux)
        fused = torch.cat((
            query,
            variant,
            text,
            query * text,
            variant * text,
            torch.abs(query - variant),
            aux,
        ), dim=1)
        return self.hit_head(fused)
