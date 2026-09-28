import pytest
import torch

from models.candidate_edge_direct_scorer import CandidateEdgeDirectScorer
from models.candidate_reference_scorer import CandidateReferenceScorer


def inputs():
    torch.manual_seed(3)
    return {"candidate_feats": torch.randn(2, 6, 32),
            "candidate_boxes": torch.randn(2, 6, 6),
            "text_feats": torch.randn(2, 5, 32),
            "text_padding_mask": torch.tensor([[False, False, False, True, True],
                                                 [False, False, False, False, True]]),
            "query_indices": torch.tensor([[3, 0], [2, 4]]),
            "valid_query_mask": torch.tensor([[True, True, True, True, True, False],
                                                [True, True, True, True, False, True]])}


@pytest.mark.parametrize("mode", ["global", "pair"])
def test_explicit_query_memory_recovers_existing_readout_without_new_parameters(mode):
    shared = inputs()
    old = CandidateEdgeDirectScorer(mode, d_model=32, n_head=4).eval()
    new = CandidateReferenceScorer(mode, d_model=32, n_head=4).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    expected = old(**shared)
    actual = new(**shared, memory_feats=shared["candidate_feats"],
                 memory_boxes=shared["candidate_boxes"], memory_valid_mask=shared["valid_query_mask"])
    assert set(new.state_dict()) == set(old.state_dict())
    for key in ("candidate_logits", "query_scores", "anchor_attention"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)


def test_object_slot_permutation_and_masked_padding_preserve_scores():
    shared = inputs()
    head = CandidateReferenceScorer("pair", d_model=32, n_head=4).eval()
    memory = {"memory_feats": torch.randn(2, 4, 32), "memory_boxes": torch.randn(2, 4, 6),
              "memory_valid_mask": torch.tensor([[True, True, False, False], [True, False, True, False]])}
    expected = head(**shared, **memory)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = head(**shared, **{key: value[:, permutation] for key, value in memory.items()})
    torch.testing.assert_close(permuted["query_scores"], expected["query_scores"], rtol=1e-6, atol=1e-6)
    changed = {key: value.clone() for key, value in memory.items()}
    for key in ("memory_feats", "memory_boxes"):
        changed[key][~memory["memory_valid_mask"]] = 10000
    actual = head(**shared, **changed)
    torch.testing.assert_close(actual["query_scores"], expected["query_scores"], rtol=1e-6, atol=1e-6)
    weights = expected["anchor_attention"][..., :-1]
    assert torch.count_nonzero(weights.masked_select(~memory["memory_valid_mask"][None, :, None, :])) == 0


def test_null_reference_and_object_feature_gradient_are_explicit():
    shared = inputs()
    head = CandidateReferenceScorer("pair", d_model=32, n_head=4).eval()
    memory_feats = torch.randn(2, 4, 32, requires_grad=True)
    memory_boxes = torch.randn(2, 4, 6)
    valid = torch.tensor([[True, False, True, False], [False, False, False, False]])
    output = head(**shared, memory_feats=memory_feats, memory_boxes=memory_boxes, memory_valid_mask=valid)
    torch.testing.assert_close(output["null_anchor_attention"][:, 1], torch.ones(4, 2))
    output["candidate_logits"][output["candidate_valid_mask"]].sum().backward()
    assert torch.isfinite(memory_feats.grad).all()
    assert memory_feats.grad[0, valid[0]].abs().sum() > 0
    assert torch.count_nonzero(memory_feats.grad[~valid]) == 0
