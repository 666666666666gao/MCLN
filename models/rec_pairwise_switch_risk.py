"""General proposal-vs-baseline signed-risk head for REC reranking."""

import torch
from torch import nn


V118_FEATURE_DIM = 23
V118_HIDDEN_DIM = 32
V118_DROPOUT = 0.1
V118_OUTPUT_DIM = 2
V118_MIN_STD = 1e-6


class PairwiseSwitchRiskHead(nn.Module):
    """Predict signed utility at REC@0.25 and REC@0.50."""

    def __init__(
            self, feature_mean, feature_std,
            hidden_dim=V118_HIDDEN_DIM, dropout=V118_DROPOUT):
        super().__init__()
        if (not isinstance(feature_mean, torch.Tensor)
                or not isinstance(feature_std, torch.Tensor)
                or feature_mean.dtype != torch.float32
                or feature_std.dtype != torch.float32
                or tuple(feature_mean.shape) != (V118_FEATURE_DIM,)
                or tuple(feature_std.shape) != (V118_FEATURE_DIM,)):
            raise TypeError("V118 normalization must be float32 [23]")
        if (not bool(torch.isfinite(feature_mean).all().item())
                or not bool(torch.isfinite(feature_std).all().item())
                or bool((feature_std < V118_MIN_STD).any().item())):
            raise ValueError("V118 normalization must be finite and positive")
        if hidden_dim != V118_HIDDEN_DIM or float(dropout) != V118_DROPOUT:
            raise ValueError("V118 architecture is frozen at 23-32-2/dropout0.1")
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_std", feature_std.detach().clone())
        self.network = nn.Sequential(
            nn.Linear(V118_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, V118_OUTPUT_DIM),
        )

    def forward(self, features):
        if (not isinstance(features, torch.Tensor)
                or features.dtype != torch.float32
                or features.dim() != 2
                or features.shape[0] <= 0
                or features.shape[1] != V118_FEATURE_DIM):
            raise TypeError("V118 features must be nonempty float32 [N,23]")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("V118 features must be finite")
        return self.network(
            (features - self.feature_mean) / self.feature_std
        )
