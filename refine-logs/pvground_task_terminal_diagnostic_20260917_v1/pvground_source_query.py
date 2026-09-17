"""Ordinary source-separated Query reading control for the pinned PV-Ground.

Reads VSA's six real pre-fusion memories at the existing keypoints. No learned
coverage score, empty-pool mask, new sampling, or post-hoc decision is included.
The original visual attention remains in place. This is a control, not a claim
of a novel attention operator or established performance improvement.
"""
from copy import deepcopy

import torch
from torch import nn


SOURCE_WIDTHS = (128, 128, 128, 128, 256, 256)
SOURCE_NAMES = ('bev', 'raw_points', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4')


class SourceQueryRead(nn.Module):
    def __init__(self, fusion_linear, visual_attention):
        super().__init__()
        assert fusion_linear.weight.shape == (288, sum(SOURCE_WIDTHS))
        assert fusion_linear.bias is None
        assert visual_attention.embed_dim == 288
        self.enabled = True
        self.projections = nn.ModuleList(nn.Linear(width, 288, bias=False) for width in SOURCE_WIDTHS)
        start = 0
        with torch.no_grad():
            for width, projection in zip(SOURCE_WIDTHS, self.projections):
                projection.weight.copy_(fusion_linear.weight[:, start:start + width])
                start += width
        self.norms = nn.ModuleList(nn.LayerNorm(288) for _ in SOURCE_WIDTHS)
        self.attention = deepcopy(visual_attention)
        # No new dropout draw: zero-residual insertion preserves the native RNG stream.
        self.attention.dropout = 0.0
        self.output = nn.Linear(288, 288)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, source_features, source_position):
        # query [Q,B,288]; source_features [B,K,1024]; position [B,K,288].
        if not self.enabled:
            return torch.zeros_like(query)
        assert source_features.shape[-1] == sum(SOURCE_WIDTHS)
        assert source_position.shape == source_features.shape[:2] + (288,)
        assert query.shape[1] == source_features.shape[0]
        values = [norm(project(part)) for part, project, norm in zip(
            torch.split(source_features, SOURCE_WIDTHS, dim=-1), self.projections, self.norms)]
        memory = torch.cat(values, dim=1).transpose(0, 1).contiguous()
        positions = source_position.repeat(1, len(SOURCE_WIDTHS), 1).transpose(0, 1)
        evidence = self.attention(query=query, key=memory + positions, value=memory,
                                  need_weights=False)[0]
        return self.output(evidence)


def install_source_query_read(model):
    """Call after strict native parent loading; added weights are checkpointed normally."""
    last = model.decoder[-1]
    assert last.source_query_read is None
    vsa = model.backbone_net.vsa
    assert tuple(vsa.SA_layer_names) == SOURCE_NAMES[2:]
    assert vsa.n_output_features == 288 and vsa.n_keypoints == 1024
    last.source_query_read = SourceQueryRead(vsa.vsa_point_feature_fusion[0], last.cross_v)
    last.source_query_read.train(last.training)
