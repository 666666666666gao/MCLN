"""Add frozen visual evidence to the semantic part of the existing object memory."""
import torch
from torch import nn
from torch.nn import functional as F


class PretrainedObjectAppearance(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1280, 160, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, object_memory, visual_features, available):
        assert object_memory.shape[:-1] == visual_features.shape[:-1] == available.shape
        assert object_memory.shape[-1] == 288 and visual_features.shape[-1] == 1280
        assert available.dtype == torch.bool
        appearance = self.projection(F.layer_norm(visual_features, (1280,)))
        appearance = appearance * available.unsqueeze(-1)
        # Keep the existing 128-dimensional box position representation intact.
        return torch.cat([object_memory[..., :128], object_memory[..., 128:] + appearance], dim=-1)
