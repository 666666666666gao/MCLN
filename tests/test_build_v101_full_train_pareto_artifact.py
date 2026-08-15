from pathlib import Path

import pytest

import scripts.build_v101_full_train_pareto_artifact as b101


RESULT = Path(
    "experiment_output/historical_e71_geometry/"
    "v101_full_train_pareto_oof_v1.json"
)


def test_v101_oof_evidence_is_bound_and_accepted():
    result = b101._validate_oof_result(RESULT)
    assert result["oof"]["diagnostics"]["sample_count"] == 36665
    assert result["oof"]["diagnostics"]["delta_hits025"] == 159
    assert result["oof"]["diagnostics"]["delta_hits050"] == 520


def test_v101_training_contract_is_full_train_and_frozen():
    contract = b101._training_contract()
    assert contract["training_rows"] == "all_scanrefer_train_rows"
    assert contract["seed"] == 0
    assert contract["epochs"] == 12
    assert contract["objective"] == (
        "bounded_iou_plus_2hit025_plus_hit050_soft_listwise"
    )


def test_v101_evidence_rejects_wrong_digest(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(RESULT.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        b101._validate_oof_result(path)
