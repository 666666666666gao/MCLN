import torch

from scripts.run_v98_nested_proposal_verifier import (
    MIN_DELTA_025,
    MIN_DELTA_050,
    V97_MARGIN,
    gate,
    pack_verifier_rows,
    target_for_pair,
)


def residual_record(baseline_index=1):
    pair_features = torch.zeros(112, 185)
    pair_valid = torch.ones(112, dtype=torch.bool)
    pair_valid[baseline_index] = False
    candidate_ious = torch.zeros(112)
    candidate_ious[baseline_index] = 0.30
    candidate_ious[2] = 0.60
    candidate_ious[3] = 0.10
    return {
        "pair_features": pair_features,
        "pair_valid": pair_valid,
        "candidate_ious": candidate_ious,
        "baseline_index": baseline_index,
    }


def test_pair_targets_are_strict_break_neutral_fix():
    record = residual_record()
    assert target_for_pair(record, 2) == [1, 2]
    assert target_for_pair(record, 3) == [0, 1]


def test_pack_skips_frozen_v97_abstentions():
    records = [residual_record(), residual_record()]
    proposals = torch.tensor([1, 2])
    packed = pack_verifier_rows(records, proposals, [0, 1])
    assert packed["row_indices"] == [1]
    assert packed["features"].shape == (1, 185)
    assert packed["targets"].tolist() == [[1, 2]]


def test_nested_gate_uses_oracle_headroom_effect_and_positive_bounds():
    diagnostics = {
        "delta_hits025": MIN_DELTA_025,
        "delta_hits050": MIN_DELTA_050,
        "fold_deltas": {
            str(index): {"hits025": 1, "hits050": 1}
            for index in range(5)
        },
        "bootstrap025": {"lower_bound_95": 1},
        "bootstrap050": {"lower_bound_95": 1},
    }
    assert V97_MARGIN == 0.13312220573425293
    assert all(gate(diagnostics).values())
    diagnostics["bootstrap050"]["lower_bound_95"] = 0
    assert gate(diagnostics)["bootstrap050_lower_bound_positive"] is False

