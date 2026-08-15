import pytest
import torch

from scripts.run_v99_pareto_contextual_hierarchical import (
    V97_MARGIN,
    pareto_accept_mask,
)


def test_pareto_gate_requires_both_threshold_gains_and_margin():
    proposals = torch.tensor([1, 2, 3, 4, 5])
    baseline = torch.tensor([0, 0, 0, 0, 5])
    aggregate = torch.tensor([
        V97_MARGIN, V97_MARGIN + 0.1, V97_MARGIN + 0.1,
        V97_MARGIN - 1e-4, V97_MARGIN + 0.1,
    ], dtype=torch.float32)
    heads = torch.tensor([
        [0.01, 0.02], [0.0, 0.2], [0.2, -0.01],
        [0.3, 0.3], [0.3, 0.3],
    ], dtype=torch.float32)
    assert pareto_accept_mask(
        proposals, baseline, aggregate, heads
    ).tolist() == [True, False, False, False, False]


def test_pareto_gate_rejects_nonfinite_gain():
    with pytest.raises(ValueError, match="finite"):
        pareto_accept_mask(
            torch.tensor([1]), torch.tensor([0]),
            torch.tensor([float("nan")], dtype=torch.float32),
            torch.tensor([[0.1, 0.1]], dtype=torch.float32),
        )


def test_pareto_gate_rejects_shape_drift():
    with pytest.raises(ValueError, match="shape or dtype"):
        pareto_accept_mask(
            torch.tensor([1]), torch.tensor([0]),
            torch.tensor([0.2], dtype=torch.float32),
            torch.tensor([0.1, 0.1], dtype=torch.float32),
        )
