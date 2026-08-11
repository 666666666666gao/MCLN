import pytest

from scripts.audit_source_moe_candidate_oracle import audit


def _metrics(rec025, rec050, mask025=70, mask050=60, miou=0.42):
    return {
        "schema": "mcln-retrain-metrics-v1",
        "sample_count": 100,
        "position": {
            "fixed_default": {"hits025": 58, "hits050": 47},
            "learned_selector": {
                "hits025": rec025, "hits050": rec050
            },
        },
        "mask": {
            "hits025": mask025,
            "hits050": mask050,
            "iou_sum": miou * 100,
            "miou": miou,
        },
    }


def _diagnostics(oracle025, oracle050):
    return {
        "schema": "mcln-source-choice-diagnostics-v1",
        "sample_count": 100,
        "gate_candidate_oracle": {
            "hits025": oracle025,
            "hits050": oracle050,
            "iou_sum": 60.0,
            "miou": 0.6,
        },
        "gate_oracle_headroom": {
            "hits025": 10,
            "hits050": 8,
            "rate025": 0.1,
            "rate050": 0.08,
        },
    }


@pytest.mark.parametrize(
    "metrics, diagnostics, expected",
    [
        (_metrics(59, 49), _diagnostics(70, 60), "learned_target_reached"),
        (_metrics(58, 48), _diagnostics(70, 60), "train_contextual_gate"),
        (
            _metrics(58, 48),
            _diagnostics(70, 48),
            "improve_candidate_generation",
        ),
    ],
)
def test_audit_selects_architecture_action(metrics, diagnostics, expected):
    result = audit(
        metrics, diagnostics, expected_sample_count=100
    )

    assert result["decision"] == expected


def test_audit_reports_all_five_deltas_against_baseline():
    result = audit(
        _metrics(58, 48, mask025=71, mask050=59, miou=0.43),
        _diagnostics(70, 60),
        baseline_receipt=_metrics(57, 47, mask025=70, mask050=60, miou=0.42),
        expected_sample_count=100,
    )

    assert set(result["deltas_vs_baseline"]) == {
        "rec_acc025",
        "rec_acc050",
        "mask_acc025",
        "mask_acc050",
        "mask_miou",
    }
    assert result["deltas_vs_baseline"]["rec_acc025"] == pytest.approx(0.01)
    assert result["deltas_vs_baseline"]["mask_acc050"] == pytest.approx(-0.01)
    assert result["mask_guard_pass"] is False


def test_rec_target_with_mask_regression_requires_tradeoff_repair():
    result = audit(
        _metrics(59, 49, mask025=70, mask050=59, miou=0.41),
        _diagnostics(70, 60),
        baseline_receipt=_metrics(
            58, 48, mask025=70, mask050=60, miou=0.42
        ),
        expected_sample_count=100,
    )

    assert result["rec_target_pass"] is True
    assert result["mask_guard_pass"] is False
    assert result["learned_target_pass"] is False
    assert result["decision"] == "repair_mask_tradeoff"


def test_audit_rejects_wrong_full_validation_count():
    with pytest.raises(ValueError, match="expected 9508"):
        audit(_metrics(59, 49), _diagnostics(70, 60))


def test_audit_rejects_internally_inconsistent_oracle_receipt():
    diagnostics = _diagnostics(70, 60)
    diagnostics["gate_oracle_headroom"]["rate050"] = 0.5

    with pytest.raises(ValueError, match="rate050 disagrees"):
        audit(
            _metrics(58, 48), diagnostics, expected_sample_count=100
        )
