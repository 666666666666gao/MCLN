import copy

import pytest
import torch
from torch import nn

from models.candidate_edge_direct_scorer import CandidateEdgeDirectScorer


def inputs():
    torch.manual_seed(21)
    return {
        "candidate_feats": torch.randn(2, 8, 24),
        "candidate_boxes": torch.cat([torch.randn(2, 8, 3), torch.ones(2, 8, 3)], -1),
        "text_feats": torch.randn(2, 6, 24),
        "text_padding_mask": torch.tensor([[False] * 4 + [True] * 2,
                                           [False] * 5 + [True]]),
        "query_indices": torch.tensor([[0, 2, 7], [3, 2, 7]]),
        "valid_query_mask": torch.tensor([[True] * 7 + [False]] * 2),
    }


def scorer(mode):
    torch.manual_seed(9)
    return CandidateEdgeDirectScorer(mode, d_model=24, n_head=4).eval()


class BroadcastGlobalText(nn.Module):
    def forward(self, query, key, value, key_padding_mask, need_weights):
        pooled = key.transpose(0, 1).masked_fill(
            key_padding_mask.unsqueeze(-1), -float("inf"),
        ).max(1).values
        weights = (~key_padding_mask).to(key.dtype)
        weights = weights / weights.sum(-1, keepdim=True)
        return (pooled.unsqueeze(0).expand_as(query),
                weights.unsqueeze(1).expand(-1, query.shape[0], -1))


def test_pair_reduces_to_exact_old_cond_path_when_text_context_is_global():
    data = inputs()
    old, pair = scorer("global"), scorer("pair")
    # Constructors initialize common modules before pair-only parameters.
    for key, value in old.state_dict().items():
        assert torch.equal(value, pair.state_dict()[key])
    pair.pair_text_attention = BroadcastGlobalText()
    expected, actual = old(**data), pair(**data)
    for key in ["candidate_logits", "query_scores", "anchor_attention"]:
        assert torch.allclose(expected[key], actual[key], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["global", "pair"])
def test_fixed_feature_readout_ignores_existing_and_appended_padding(mode):
    data = inputs()
    model = scorer(mode)
    expected = model(**data)
    altered = copy.deepcopy(data)
    altered["text_feats"][data["text_padding_mask"]] = 1000
    altered["text_feats"] = torch.cat([
        altered["text_feats"], torch.full((2, 16, 24), 2000.0),
    ], dim=1)
    altered["text_padding_mask"] = torch.cat([
        data["text_padding_mask"], torch.ones(2, 16, dtype=torch.bool),
    ], dim=1)
    actual = model(**altered)
    assert torch.allclose(expected["query_scores"], actual["query_scores"], atol=1e-6)
    assert torch.allclose(expected["anchor_attention"], actual["anchor_attention"], atol=1e-6)
    if mode == "pair":
        assert torch.count_nonzero(actual["pair_token_attention"][..., -16:]) == 0


@pytest.mark.parametrize("mode", ["global", "pair"])
def test_full_memory_keeps_non_target_anchors_and_excludes_illegal_queries(mode):
    data = inputs()
    model = scorer(mode)
    expected = model(**data)
    assert torch.count_nonzero(expected["anchor_attention"][..., 7]) == 0
    assert torch.isfinite(expected["null_anchor_attention"]).all()
    assert (expected["null_anchor_attention"] > 0).all()
    assert torch.isneginf(expected["candidate_logits"][:, 2]).all()
    assert torch.isneginf(expected["query_scores"][:, 7]).all()

    illegal_change = copy.deepcopy(data)
    illegal_change["candidate_feats"][:, 7] += 100
    illegal_change["candidate_boxes"][:, 7, :3] += 100
    unchanged = model(**illegal_change)
    assert torch.allclose(expected["query_scores"], unchanged["query_scores"], atol=1e-6)

    # Query 1 is a legal memory entry outside every target Top-K.
    anchor_change = copy.deepcopy(data)
    anchor_change["candidate_feats"][:, 1] += torch.arange(24) * 0.5
    changed = model(**anchor_change)
    assert not torch.allclose(expected["candidate_logits"][:, :2],
                              changed["candidate_logits"][:, :2], atol=1e-5)


def test_pair_token_readout_binds_each_memory_query_independently():
    data = inputs()
    model = scorer("pair")
    expected = model(**data)["pair_token_attention"]
    data["candidate_feats"][:, 1] += torch.arange(24) * 0.5
    actual = model(**data)["pair_token_attention"]
    assert not torch.allclose(expected[:, :, 1], actual[:, :, 1], atol=1e-5)
    unchanged_edges = torch.tensor([0, 2, 3, 4, 5, 6, 7, 8])
    assert torch.allclose(expected[:, :, unchanged_edges],
                          actual[:, :, unchanged_edges], atol=1e-6)


@pytest.mark.parametrize("mode", ["global", "pair"])
def test_query_permutation_preserves_identity_and_scattered_scores(mode):
    data = inputs()
    model = scorer(mode)
    expected = model(**data)
    order = torch.tensor([4, 1, 7, 0, 6, 3, 2, 5])
    inverse = order.argsort()
    altered = copy.deepcopy(data)
    for key in ["candidate_feats", "candidate_boxes", "valid_query_mask"]:
        altered[key] = data[key][:, order]
    altered["query_indices"] = inverse[data["query_indices"]]
    actual = model(**altered)
    assert torch.allclose(expected["candidate_logits"], actual["candidate_logits"], atol=1e-6)
    assert torch.allclose(expected["query_scores"], actual["query_scores"][:, inverse], atol=1e-6)
    assert torch.equal(data["candidate_feats"], inputs()["candidate_feats"])


def test_final_score_has_gradient_through_pair_text_geometry_and_null_state():
    model = scorer("pair")
    out = model(**inputs())
    out["candidate_logits"][:, :2].sum().backward()
    for parameter in [model.pair_query[0].weight,
                      model.pair_text_attention.in_proj_weight,
                      model.spatial.lang_cond_fc.weight,
                      model.null_anchor, model.score_head.weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_real_query_dimensions_use_32_targets_and_257_memory_states():
    torch.manual_seed(0)
    model = CandidateEdgeDirectScorer("pair").eval()
    with torch.no_grad():
        output = model(
            candidate_feats=torch.randn(1, 256, 288),
            candidate_boxes=torch.randn(1, 256, 6),
            text_feats=torch.randn(1, 32, 288),
            text_padding_mask=torch.zeros(1, 32, dtype=torch.bool),
            query_indices=torch.arange(32).unsqueeze(0),
            valid_query_mask=torch.ones(1, 256, dtype=torch.bool),
        )
    assert output["anchor_attention"].shape == (8, 1, 32, 257)
    assert output["pair_token_attention"].shape == (1, 32, 257, 32)
    assert output["query_scores"].shape == (1, 256)
    assert torch.isneginf(output["query_scores"][:, 32:]).all()
