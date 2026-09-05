import torch

from scripts.nr3d_candidate_contract import ranked_oracle_profile


def test_top16_failure_can_have_full256_coverage():
    scores = torch.arange(256, 0, -1).float()
    ious = torch.zeros(256)
    ious[40] = .8
    result = ranked_oracle_profile(scores, ious, torch.ones(256, dtype=torch.bool))
    assert not result["top_16"]["hit025"]
    assert not result["top_32"]["hit025"]
    assert result["top_64"]["hit050"]
    assert result["top_256"]["hit050"]
    assert result["box_oracle_query"] == 40


def test_filter_loss_is_separate_from_raw_candidate_coverage():
    scores, ious = torch.tensor([.9, .8]), torch.tensor([.8, .1])
    before = ranked_oracle_profile(scores, ious, torch.tensor([True, True]))
    after = ranked_oracle_profile(scores, ious, torch.tensor([False, True]))
    assert before["top_256"]["hit050"]
    assert not after["top_256"]["hit025"]
    assert after["top_query"] == 1
    assert after["top_16"]["available"] == 1


def test_empty_legal_set_has_no_query_and_thresholds_are_strict():
    scores, ious = torch.tensor([.9, .8]), torch.tensor([.25, .50])
    result = ranked_oracle_profile(scores, ious, torch.tensor([False, False]))
    assert result["top_query"] is None
    assert result["box_oracle_query"] is None
    assert not result["top_256"]["hit025"]
    result = ranked_oracle_profile(scores, ious, torch.tensor([True, True]))
    assert result["top_16"]["hit025"]
    assert not result["top_16"]["hit050"]
