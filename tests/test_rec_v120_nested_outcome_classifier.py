import torch

from models.rec_pairwise_switch_classifier import (
    PairwiseSwitchOutcomeClassifier,
    V120_CLASS_COUNT,
    V120_FEATURE_DIM,
    V120_THRESHOLD_COUNT,
)
from scripts.run_v120_meshsp_nested_outcome_classifier_oof import (
    V120_BREAK_CLASS,
    V120_BREAK_COST,
    V120_CLASS_WEIGHTS,
    V120_EVENT_WEIGHT,
    V120_FIX_CLASS,
    V120_NEUTRAL_CLASS,
    switch_outcome_targets,
)


def test_v120_classifier_shape_is_finite_and_trainable():
    model = PairwiseSwitchOutcomeClassifier(
        torch.zeros(V120_FEATURE_DIM),
        torch.ones(V120_FEATURE_DIM),
    )
    features = torch.randn(8, V120_FEATURE_DIM, requires_grad=True)
    logits = model(features)
    assert tuple(logits.shape) == (
        8, V120_THRESHOLD_COUNT, V120_CLASS_COUNT
    )
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_v120_outcome_targets_use_break_neutral_fix_order():
    records = [
        {"candidate_ious": torch.tensor([0.10, 0.30])},
        {"candidate_ious": torch.tensor([0.30, 0.10])},
        {"candidate_ious": torch.tensor([0.60, 0.30])},
        {"candidate_ious": torch.tensor([0.30, 0.60])},
    ]
    proposals = torch.ones(4, dtype=torch.long)
    baselines = torch.zeros(4, dtype=torch.long)
    targets, counts = switch_outcome_targets(
        records, proposals, baselines
    )
    expected = torch.tensor([
        [V120_FIX_CLASS, V120_NEUTRAL_CLASS],
        [V120_BREAK_CLASS, V120_NEUTRAL_CLASS],
        [V120_NEUTRAL_CLASS, V120_BREAK_CLASS],
        [V120_NEUTRAL_CLASS, V120_FIX_CLASS],
    ])
    assert torch.equal(targets, expected)
    assert counts == {
        "025": {"fix": 1, "break": 1, "neutral": 2},
        "050": {"fix": 1, "break": 1, "neutral": 2},
    }


def test_v120_class_weights_are_frozen_from_event_and_break_costs():
    assert V120_BREAK_COST == 4.0
    assert V120_EVENT_WEIGHT == 4.0
    assert V120_CLASS_WEIGHTS == (16.0, 1.0, 4.0)
