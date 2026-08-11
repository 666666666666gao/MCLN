import json

import pytest
import torch

from scripts.audit_training_completion import (
    AUDIT_SCHEMA,
    _write_json_atomic,
    audit_training_completion,
)


def _receipt(sample_count=100):
    return {
        "schema": "mcln-retrain-metrics-v1",
        "sample_count": sample_count,
        "position": {
            "fixed_default": {"hits025": 58, "hits050": 47},
            "learned_selector": {"hits025": 59, "hits050": 49},
        },
        "position_subgroups": {
            "unique": {
                "sample_count": 20,
                "hits025": 15,
                "hits050": 12,
                "acc025": 0.75,
                "acc050": 0.6,
            },
            "multiple": {
                "sample_count": sample_count - 20,
                "hits025": 44,
                "hits050": 37,
                "acc025": 44.0 / (sample_count - 20),
                "acc050": 37.0 / (sample_count - 20),
            },
        },
        "mask": {
            "hits025": 70,
            "hits050": 60,
            "iou_sum": 42.0,
            "miou": 0.42,
            "position_subgroups": {
                "unique": {
                    "sample_count": 20,
                    "hits025": 16,
                    "hits050": 13,
                    "acc025": 0.8,
                    "acc050": 0.65,
                },
                "multiple": {
                    "sample_count": sample_count - 20,
                    "hits025": 54,
                    "hits050": 47,
                    "acc025": 54.0 / (sample_count - 20),
                    "acc050": 47.0 / (sample_count - 20),
                },
            },
        },
    }


def _checkpoint(tmp_path, epoch=80):
    path = tmp_path / "ckpt_epoch_last.pth"
    torch.save({"epoch": epoch, "model": {}}, path)
    return path


def test_completion_gate_accepts_exact_receipt_and_epoch(tmp_path):
    result = audit_training_completion(
        _receipt(), _checkpoint(tmp_path),
        expected_epoch=80, expected_sample_count=100,
        require_position_subgroups=True,
    )

    assert result["schema"] == AUDIT_SCHEMA
    assert result["passed"] is True
    assert result["checkpoint_epoch"] == 80
    assert result["metrics"]["rec_acc025"] == pytest.approx(0.59)
    assert result["metrics"]["mask_miou"] == pytest.approx(0.42)
    assert result["metrics"]["position_subgroups"]["unique"][
        "acc050"
    ] == pytest.approx(0.6)
    assert result["metrics"]["mask_position_subgroups"]["multiple"][
        "acc050"
    ] == pytest.approx(47.0 / 80.0)


def test_completion_gate_requires_position_subgroups_when_requested(tmp_path):
    receipt = _receipt()
    receipt.pop("position_subgroups")

    with pytest.raises(ValueError, match="lacks position subgroups"):
        audit_training_completion(
            receipt, _checkpoint(tmp_path),
            expected_epoch=80, expected_sample_count=100,
            require_position_subgroups=True,
        )


def test_completion_gate_requires_mask_position_subgroups_when_requested(
        tmp_path):
    receipt = _receipt()
    receipt["mask"].pop("position_subgroups")

    with pytest.raises(ValueError, match="lacks mask position subgroups"):
        audit_training_completion(
            receipt, _checkpoint(tmp_path),
            expected_epoch=80, expected_sample_count=100,
            require_position_subgroups=True,
        )


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda receipt: receipt.update(schema="bad"), "schema"),
        (lambda receipt: receipt.update(sample_count=99), "expected 100"),
        (
            lambda receipt: receipt["position"]["learned_selector"].update(
                hits050=60
            ),
            "hits050 exceeds hits025",
        ),
        (
            lambda receipt: receipt["mask"].update(hits050=71),
            "mask hits050 exceeds hits025",
        ),
        (
            lambda receipt: receipt["mask"].update(iou_sum=41.0),
            "disagrees with mean IoU",
        ),
    ],
)
def test_completion_gate_rejects_invalid_receipt(
        tmp_path, mutation, error):
    receipt = _receipt()
    mutation(receipt)

    with pytest.raises(ValueError, match=error):
        audit_training_completion(
            receipt, _checkpoint(tmp_path),
            expected_epoch=80, expected_sample_count=100,
        )


def test_completion_gate_rejects_wrong_checkpoint_epoch(tmp_path):
    with pytest.raises(ValueError, match="does not match expected 80"):
        audit_training_completion(
            _receipt(), _checkpoint(tmp_path, epoch=79),
            expected_epoch=80, expected_sample_count=100,
        )


def test_completion_gate_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        audit_training_completion(
            _receipt(), tmp_path / "missing.pth",
            expected_epoch=80, expected_sample_count=100,
        )


def test_completion_audit_output_replaces_atomically(tmp_path):
    output = tmp_path / "nested" / "audit.json"
    output.parent.mkdir()
    output.write_text('{"old": true}\n', encoding="utf-8")

    _write_json_atomic(output, {"passed": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
    assert list(output.parent.glob("*.tmp")) == []
