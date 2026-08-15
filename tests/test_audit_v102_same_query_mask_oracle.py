import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_v102_same_query_mask_oracle.py"
SPEC = importlib.util.spec_from_file_location("v102_oracle", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_query_values_keep_query_fixed_and_choose_policy_only():
    # [rows, queries, policies]
    values = torch.zeros(2, 2, 15)
    values[0, 0, [0, 1, 12]] = torch.tensor([0.10, 0.20, 0.30])
    values[0, 1, [0, 1, 12]] = torch.tensor([0.40, 0.80, 0.50])
    values[1, 0, [0, 1, 12]] = torch.tensor([0.60, 0.30, 0.20])
    values[1, 1, [0, 1, 12]] = torch.tensor([0.10, 0.20, 0.90])
    selected = torch.tensor([1, 0])
    result = MODULE.selected_query_policy_values(values, selected, legacy_index=12)
    assert torch.equal(result["baseline"], torch.tensor([0.50, 0.20]))
    assert torch.equal(result["oracle"], torch.tensor([0.80, 0.60]))
    assert torch.equal(result["oracle_policy"], torch.tensor([1, 0]))


def test_metric_summary_uses_strict_scanrefer_thresholds():
    summary = MODULE.metric_summary(torch.tensor([0.25, 0.50, 0.51, 0.26]))
    assert summary == {
        "count": 4,
        "hits025": 3,
        "hits050": 1,
        "acc025": 0.75,
        "acc050": 0.25,
        "miou": pytest.approx(0.38),
    }


def test_validate_alignment_rejects_reordered_or_missing_rows():
    with pytest.raises(ValueError, match="dataset index alignment"):
        MODULE.validate_alignment(
            torch.tensor([3, 4]),
            [{"dataset_index": 4}, {"dataset_index": 3}],
        )


def test_policy_metadata_requires_exact_frozen_order():
    good = {
        "source_names": ["text", "query", "fused"],
        "thresholds": [-1.0, -0.5, 0.0, 0.5, 1.0],
    }
    assert MODULE.validate_policy_metadata(good) == 12
    bad = dict(good, thresholds=[0.0])
    with pytest.raises(ValueError, match="threshold"):
        MODULE.validate_policy_metadata(bad)
