"""Three-class proposal-vs-baseline outcome head for REC reranking."""

import torch
from torch import nn


V120_FEATURE_DIM = 23
V120_HIDDEN_DIM = 32
V120_DROPOUT = 0.1
V120_THRESHOLD_COUNT = 2
V120_CLASS_COUNT = 3
V120_MIN_STD = 1e-6


class PairwiseSwitchOutcomeClassifier(nn.Module):
    """Classify break, neutral, or fix at each REC threshold."""

    def __init__(
            self, feature_mean, feature_std,
            hidden_dim=V120_HIDDEN_DIM, dropout=V120_DROPOUT):
        super().__init__()
        if (not isinstance(feature_mean, torch.Tensor)
                or not isinstance(feature_std, torch.Tensor)
                or feature_mean.dtype != torch.float32
                or feature_std.dtype != torch.float32
                or tuple(feature_mean.shape) != (V120_FEATURE_DIM,)
                or tuple(feature_std.shape) != (V120_FEATURE_DIM,)):
            raise TypeError("V120 normalization must be float32 [23]")
        if (not bool(torch.isfinite(feature_mean).all().item())
                or not bool(torch.isfinite(feature_std).all().item())
                or bool((feature_std < V120_MIN_STD).any().item())):
            raise ValueError("V120 normalization must be finite and positive")
        if hidden_dim != V120_HIDDEN_DIM or float(dropout) != V120_DROPOUT:
            raise ValueError("V120 architecture is frozen at 23-32-(2x3)")
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_std", feature_std.detach().clone())
        self.network = nn.Sequential(
            nn.Linear(V120_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(
                hidden_dim, V120_THRESHOLD_COUNT * V120_CLASS_COUNT
            ),
        )

    def forward(self, features):
        if (not isinstance(features, torch.Tensor)
                or features.dtype != torch.float32
                or features.dim() != 2
                or features.shape[0] <= 0
                or features.shape[1] != V120_FEATURE_DIM):
            raise TypeError("V120 features must be nonempty float32 [N,23]")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("V120 features must be finite")
        logits = self.network(
            (features - self.feature_mean) / self.feature_std
        )
        return logits.reshape(
            features.shape[0], V120_THRESHOLD_COUNT, V120_CLASS_COUNT
        )
