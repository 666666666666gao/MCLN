import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "run_v103_relative_mask_oof.py"
)
SPEC = importlib.util.spec_from_file_location("v103_oof", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v103_protocol_constants_are_frozen():
    assert MODULE.SEEDS == (0, 1, 2)
    assert MODULE.EPOCHS == 12
    assert MODULE.TARGET_TEMPERATURE == pytest.approx(0.25)
    assert MODULE.AGGREGATE_MARGIN == pytest.approx(0.02)
    assert MODULE.ALLOWED_MASK_POLICY_INDICES == tuple(range(4, 15))
    assert MODULE.BOOTSTRAP_SAMPLES == 10000


def test_acceptance_gate_requires_every_preregistered_condition():
    metrics = {
        "delta_hits025": 49,
        "delta_hits050": 79,
        "delta_miou": 0.0023,
    }
    folds = [{
        "delta_hits025": 0,
        "delta_hits050": 1,
        "delta_miou": 0.0,
    } for _ in range(5)]
    bootstrap = {
        "delta_acc025_lower_bound_95": 1e-6,
        "delta_acc050_lower_bound_95": 1e-6,
        "delta_miou_lower_bound_95": 1e-6,
    }
    assert MODULE.acceptance_gate(
        metrics, folds, bootstrap, True
    )["passed"] is True
    assert MODULE.acceptance_gate(
        dict(metrics, delta_hits050=78), folds, bootstrap, True
    )["passed"] is False
    assert MODULE.acceptance_gate(
        metrics, folds, bootstrap, False
    )["passed"] is False


def test_metric_delta_keeps_strict_threshold_semantics():
    result = MODULE.metric_delta(
        torch.tensor([0.25, 0.50, 0.9]),
        torch.tensor([0.26, 0.51, 0.8]),
    )
    assert result["delta_hits025"] == 1
    assert result["delta_hits050"] == 1
    assert result["delta_miou"] == pytest.approx(-0.0266666667)
