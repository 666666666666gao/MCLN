import torch

from models.rec_pairwise_switch_risk import (
    PairwiseSwitchRiskHead,
    V118_FEATURE_DIM,
)
from scripts.run_v118_meshsp_nested_pairwise_risk_oof import (
    build_pairwise_features,
    signed_switch_targets,
)


def _record(iou_fill=0.0):
    return {
        "query_features": torch.zeros(16, 152, dtype=torch.float32),
        "query_aux_continuous": torch.zeros(16, 4, dtype=torch.float32),
        "variant_aux_continuous": torch.zeros(16, 7, 2, dtype=torch.float32),
        "baseline_scores": torch.zeros(112, dtype=torch.float32),
        "candidate_ious": torch.full(
            (16, 7), float(iou_fill), dtype=torch.float32
        ),
    }


def test_v118_pair_features_are_finite_deployable_and_ignore_iou_labels():
    records = [_record(0.0), _record(1.0)]
    records[0]["query_features"][1, 128:134] = 1.0
    records[1]["query_aux_continuous"][2] = 0.5
    proposals = torch.tensor([7, 15], dtype=torch.long)
    baselines = torch.tensor([0, 14], dtype=torch.long)
    proposal_probability = torch.tensor(
        [[0.8, 0.6], [0.7, 0.5]], dtype=torch.float32
    )
    baseline_probability = torch.tensor(
        [[0.6, 0.4], [0.65, 0.45]], dtype=torch.float32
    )
    head_gain = proposal_probability - baseline_probability
    first = build_pairwise_features(
        records, proposals, baselines, proposal_probability,
        baseline_probability, head_gain,
    )
    records[0]["candidate_ious"].fill_(1.0)
    records[1]["candidate_ious"].zero_()
    second = build_pairwise_features(
        records, proposals, baselines, proposal_probability,
        baseline_probability, head_gain,
    )
    assert first.shape == (2, V118_FEATURE_DIM)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_v118_signed_targets_encode_fix_break_and_neutral():
    records = [_record() for _ in range(4)]
    baselines = torch.zeros(4, dtype=torch.long)
    proposals = torch.ones(4, dtype=torch.long)
    pairs = ((0.1, 0.3), (0.3, 0.1), (0.6, 0.3), (0.3, 0.6))
    for record, (baseline_iou, proposal_iou) in zip(records, pairs):
        record["candidate_ious"].view(-1)[0] = baseline_iou
        record["candidate_ious"].view(-1)[1] = proposal_iou
    targets, counts = signed_switch_targets(records, proposals, baselines)
    assert torch.equal(targets, torch.tensor([
        [1.0, 0.0], [-4.0, 0.0], [0.0, -4.0], [0.0, 1.0]
    ]))
    assert counts["025"] == {"fix": 1, "break": 1, "neutral": 2}
    assert counts["050"] == {"fix": 1, "break": 1, "neutral": 2}


def test_v118_risk_head_has_frozen_shape_and_backpropagates():
    mean = torch.zeros(V118_FEATURE_DIM, dtype=torch.float32)
    std = torch.ones(V118_FEATURE_DIM, dtype=torch.float32)
    model = PairwiseSwitchRiskHead(mean, std).train()
    features = torch.randn(8, V118_FEATURE_DIM, dtype=torch.float32)
    output = model(features)
    assert output.shape == (8, 2)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

