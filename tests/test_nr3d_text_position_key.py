import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.nr3d_text_position_key import TextPositionKey


def fixture(mode):
    torch.manual_seed(0)
    attention = nn.MultiheadAttention(24, 4, dropout=0).eval()
    attention.requires_grad_(False)
    addon = TextPositionKey(24, 4, mode)
    query = torch.randn(5, 2, 24)
    text = torch.randn(7, 2, 24)
    points = torch.randn(2, 11, 24)
    positions = torch.randn(2, 11, 24)
    padding = torch.tensor([[False]*5+[True]*2, [False]*6+[True]])
    return attention, addon, query, text, points, positions, padding


def invoke(attention, addon, query, text, points, positions, padding):
    projected = F.linear(query, attention.in_proj_weight[:24], attention.in_proj_bias[:24])
    bias = addon(projected.transpose(0, 1), text.transpose(0, 1), points, positions)
    return attention(query, text, text, attn_mask=bias, key_padding_mask=padding)


@pytest.mark.parametrize('mode', ['text', 'position'])
def test_zero_start_is_exact_and_native_attention_trains_only_addon(mode):
    attention, addon, query, text, points, positions, padding = fixture(mode)
    original, original_weights = attention(query, text, text, key_padding_mask=padding)
    actual, weights = invoke(attention, addon, query, text, points, positions, padding)
    assert torch.equal(original, actual) and torch.equal(original_weights, weights)
    actual.square().sum().backward()
    assert torch.isfinite(addon.weight.grad).all() and addon.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in attention.parameters())


@pytest.mark.parametrize('mode', ['text', 'position'])
def test_padding_is_ignored_after_activation(mode):
    attention, addon, query, text, points, positions, padding = fixture(mode)
    with torch.no_grad():
        addon.weight.normal_(0, .02)
    original, _ = invoke(attention, addon, query, text, points, positions, padding)
    altered = text.clone()
    altered[padding.T] = torch.randn_like(altered[padding.T]) * 7
    actual, _ = invoke(attention, addon, query, altered, points, positions, padding)
    assert torch.equal(original, actual)


def test_position_pair_order_is_irrelevant_but_position_assignment_matters():
    attention, addon, query, text, points, positions, padding = fixture('position')
    with torch.no_grad():
        addon.weight.normal_(0, .02)
    original, _ = invoke(attention, addon, query, text, points, positions, padding)
    order = torch.randperm(11)
    reordered, _ = invoke(attention, addon, query, text, points[:, order], positions[:, order], padding)
    changed, _ = invoke(attention, addon, query, text, points, positions[:, order], padding)
    assert torch.allclose(original, reordered, atol=1e-6, rtol=1e-6)
    assert (original-changed).abs().max() > 1e-4


@pytest.mark.parametrize('mode', ['text', 'position'])
def test_logit_intervention_equals_explicit_added_projected_key(mode):
    attention, addon, query, text, points, positions, padding = fixture(mode)
    with torch.no_grad():
        addon.weight.normal_(0, .02)
    actual, weights = invoke(attention, addon, query, text, points, positions, padding)
    q = addon.split_heads(F.linear(query, attention.in_proj_weight[:24],
                                  attention.in_proj_bias[:24]).transpose(0, 1))
    k = addon.split_heads(F.linear(text, attention.in_proj_weight[24:48],
                                  attention.in_proj_bias[24:48]).transpose(0, 1))
    v = addon.split_heads(F.linear(text, attention.in_proj_weight[48:],
                                  attention.in_proj_bias[48:]).transpose(0, 1))
    if mode == 'text':
        extra = addon.split_heads(F.linear(text.transpose(0, 1), addon.weight))
    else:
        similarity = torch.einsum('bhld,bhnd->bhln', addon.split_heads(text.transpose(0, 1)),
                                  addon.split_heads(points)) / 6 ** .5
        extra = torch.einsum('bhln,bhnd->bhld', similarity.softmax(-1),
                             addon.split_heads(F.linear(positions, addon.weight)))
    score = torch.einsum('bhqd,bhld->bhql', q, k+extra) / 6 ** .5
    probability = score.masked_fill(padding[:, None, None, :], -float('inf')).softmax(-1)
    expected = torch.einsum('bhql,bhld->bhqd', probability, v).transpose(1, 2).reshape(2, 5, 24)
    expected = attention.out_proj(expected).transpose(0, 1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(weights, probability.mean(1), atol=1e-6, rtol=1e-6)
