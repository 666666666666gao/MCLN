import math

import torch

import scripts.run_v99_pareto_contextual_hierarchical as v99
from scripts.run_v116_meshsp_primary025_policy_oof import (
    V116_THRESHOLD_WEIGHTS,
    select_v116_proposal,
)


def _logit(probability):
    return math.log(probability / (1.0 - probability))


def test_v116_uses_normalized_four_to_one_primary_threshold_weights():
    assert V116_THRESHOLD_WEIGHTS == (2.4, 0.6)
    assert sum(V116_THRESHOLD_WEIGHTS) == 3.0
    assert V116_THRESHOLD_WEIGHTS[0] / V116_THRESHOLD_WEIGHTS[1] == 4.0


def test_v116_changes_only_the_primary_threshold_ranking_and_masks_padding():
    query_logits = torch.zeros(1, 16, 2, dtype=torch.float32)
    variant_logits = torch.zeros(1, 16, 7, 2, dtype=torch.float32)
    query_valid = torch.zeros(1, 16, dtype=torch.bool)
    variant_valid = torch.zeros(1, 16, 7, dtype=torch.bool)
    query_valid[0, :2] = True
    variant_valid[0, :2, 0] = True

    primary_candidate = torch.tensor([
        _logit(0.80), _logit(0.125)
    ], dtype=torch.float32)
    balanced_candidate = torch.tensor([
        _logit(0.65), _logit(0.99)
    ], dtype=torch.float32)
    query_logits[0, 0] = primary_candidate
    query_logits[0, 1] = balanced_candidate
    variant_logits[0, 0, 0] = primary_candidate
    variant_logits[0, 1, 0] = balanced_candidate
    query_logits[0, 2:] = 100.0
    variant_logits[0, 2:] = 100.0

    standard = v99.select_hierarchical_proposal(
        query_logits, variant_logits, query_valid, variant_valid
    )
    primary = select_v116_proposal(
        query_logits, variant_logits, query_valid, variant_valid
    )
    assert standard["query_indices"].item() == 1
    assert primary["query_indices"].item() == 0
    assert primary["variant_indices"].item() == 0
    assert primary["flat_indices"].item() == 0

