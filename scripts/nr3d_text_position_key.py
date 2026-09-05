"""Minimal key-evidence intervention in the native last text attention.

Position-pooling idea: EG3DVG PECA, commit174e34894aea6513442da6b5dfa9b3e2bf8a1efa.
This independently written addon keeps the original attention implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F


class TextPositionKey(nn.Module):
    def __init__(self, dimension, heads, mode):
        super().__init__()
        assert dimension % heads == 0 and mode in ('text', 'position')
        self.heads = heads
        self.width = dimension // heads
        self.mode = mode
        self.weight = nn.Parameter(torch.zeros(dimension, dimension))

    def split_heads(self, values):
        batch, length, _ = values.shape
        return values.reshape(batch, length, self.heads, self.width).transpose(1, 2)

    def forward(self, projected_query, text, points, positions):
        """Inputs are batch-first; output is native MHA's(B*H,Q,L) logit bias."""
        if self.mode == 'text':
            added_key = self.split_heads(F.linear(text, self.weight))
        else:
            alignment = torch.matmul(self.split_heads(text),
                                     self.split_heads(points).transpose(-1, -2))
            alignment = (alignment * self.width ** -.5).softmax(dim=-1)
            added_key = torch.matmul(alignment, self.split_heads(F.linear(positions, self.weight)))
        bias = torch.matmul(self.split_heads(projected_query) * self.width ** -.5,
                            added_key.transpose(-1, -2))
        batch, heads, queries, tokens = bias.shape
        return bias.reshape(batch * heads, queries, tokens)


class LastTextAttentionIntervention:
    """Temporary runtime attachment; the addon is not registered in base state."""

    def __init__(self, model, addon):
        self.addon = addon
        self.attention = model.decoder[-1].cross_l
        self.original_forward = self.attention.forward
        self.positions = None
        self.points = None
        self.position_calls = 0
        self.decoder_calls = 0
        self.attention_calls = 0
        self.handles = [
            model.pos_embed.register_forward_hook(self.capture_positions),
            model.decoder[-1].register_forward_pre_hook(self.capture_points),
        ]
        self.attention.forward = self.forward

    def capture_positions(self, module, inputs, output):
        self.positions = output.transpose(1, 2)
        self.position_calls += 1

    def capture_points(self, module, inputs):
        self.points = inputs[1]
        self.decoder_calls += 1

    def forward(self, query, key, value, attn_mask, key_padding_mask):
        assert attn_mask is None
        dimension = self.attention.embed_dim
        projected = F.linear(query, self.attention.in_proj_weight[:dimension],
                             self.attention.in_proj_bias[:dimension])
        bias = self.addon(projected.transpose(0, 1), key.transpose(0, 1),
                          self.points, self.positions)
        self.attention_calls += 1
        return self.original_forward(query=query, key=key, value=value,
                                     attn_mask=bias, key_padding_mask=key_padding_mask)

    def remove(self):
        self.attention.forward = self.original_forward
        for handle in self.handles:
            handle.remove()
        self.points = self.positions = None
