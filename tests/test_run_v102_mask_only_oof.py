import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v102_mask_only_oof.py"
SPEC = importlib.util.spec_from_file_location("v102_oof", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_streaming_statistics_match_direct_population_values():
    features = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    valid = torch.tensor([[True, False], [True, True]])
    stats = MODULE._streaming_statistics(
        features, valid, torch.tensor([0, 1]), chunk_size=1
    )
    values = features[valid]
    assert stats["count"] == 3
    assert torch.allclose(stats["mean"], values.mean(0))
    assert torch.allclose(stats["std"], values.std(0, unbiased=False))


def test_metric_delta_uses_strict_thresholds_and_signed_miou():
    result = MODULE.metric_delta(
        torch.tensor([0.25, 0.50, 0.9]),
        torch.tensor([0.26, 0.51, 0.8]),
    )
    assert result["delta_hits025"] == 1
    assert result["delta_hits050"] == 1
    assert result["delta_miou"] == pytest.approx(-0.0266666667)


def test_acceptance_gate_requires_every_preregistered_condition():
    metrics = {"delta_hits025": 49, "delta_hits050": 79,
               "delta_miou": 0.0023}
    folds = [{"delta_hits025": 0, "delta_hits050": 1,
              "delta_miou": 0.0} for _ in range(5)]
    bootstrap = {
        "delta_acc025_lower_bound_95": 1e-6,
        "delta_acc050_lower_bound_95": 1e-6,
        "delta_miou_lower_bound_95": 1e-6,
    }
    gate = MODULE.acceptance_gate(metrics, folds, bootstrap, True)
    assert gate["passed"] is True
    broken = dict(metrics, delta_hits050=78)
    assert MODULE.acceptance_gate(
        broken, folds, bootstrap, True
    )["passed"] is False
    assert MODULE.acceptance_gate(
        metrics, folds, bootstrap, False
    )["passed"] is False


def test_bootstrap_rejects_wrong_scene_coverage():
    with pytest.raises(ValueError, match="scene coverage"):
        MODULE.scene_block_bootstrap_lower_bounds(
            torch.zeros(2), torch.ones(2), ["a", "b"], samples=10
        )
