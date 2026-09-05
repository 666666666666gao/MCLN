"""Padding interventions must not alter valid visual/query representations."""

import torch

from models.encoder_decoder_layers import BiDecoderLayer, BiEncoderLayer, calc_pairwise_locs


def fixture():
    torch.manual_seed(8)
    text = torch.randn(1, 5, 24)
    padded = torch.cat([text, torch.full((1, 3, 24), 20.0)], dim=1)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    padded_mask = torch.cat([mask, torch.ones(1, 3, dtype=torch.bool)], dim=1)
    return text, padded, mask, padded_mask


def test_encoder_visual_and_valid_text_ignore_padding_values():
    text, padded, mask, padded_mask = fixture()
    layer = BiEncoderLayer(d_model=24, n_heads=4, dim_feedforward=48, dropout=0).eval()
    visual = torch.randn(1, 7, 24)
    position = torch.randn(1, 7, 24)
    geometry = calc_pairwise_locs(torch.randn(1, 7, 3))
    visual_mask = torch.zeros(1, 7, dtype=torch.bool)
    with torch.no_grad():
        before = layer(visual, position, visual_mask, text, mask, spatial_point_xyz=geometry)
        after = layer(visual, position, visual_mask, padded, padded_mask, spatial_point_xyz=geometry)
    assert torch.allclose(before[0], after[0], atol=2e-6, rtol=1e-5)
    assert torch.allclose(before[1], after[1][:, :5], atol=2e-6, rtol=1e-5)


def test_decoder_queries_ignore_padding_values():
    text, padded, mask, padded_mask = fixture()
    layer = BiDecoderLayer(d_model=24, n_heads=4, dim_feedforward=48, dropout=0).eval()
    queries = torch.randn(1, 6, 24)
    visual = torch.randn(1, 7, 24)
    positions = torch.rand(1, 6, 6)
    query_mask = torch.zeros(1, 6, dtype=torch.bool)
    with torch.no_grad():
        before = layer(queries, visual, text, positions, query_mask, mask)
        after = layer(queries, visual, padded, positions, query_mask, padded_mask)
    assert torch.allclose(before, after, atol=2e-6, rtol=1e-5)
