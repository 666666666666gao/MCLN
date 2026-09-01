import pytest
import torch

from models.rec_evaluator_filter import build_detector_overlap_valid


def _box(center_x):
    return [float(center_x), 0.0, 0.0, 2.0, 2.0, 2.0]


def test_detector_overlap_valid_supports_flat_and_variant_axes():
    candidates = torch.tensor([[
        [_box(0.0), _box(0.5)],
        [_box(10.0), _box(0.0)],
    ]], dtype=torch.float32)
    valid = torch.tensor([[
        [True, True],
        [True, False],
    ]])
    detected = torch.tensor([[
        _box(0.0), _box(100.0),
    ]], dtype=torch.float32)
    detected_valid = torch.tensor([[True, False]])

    result = build_detector_overlap_valid(
        candidates, valid, detected, detected_valid
    )

    assert result.dtype == torch.bool
    assert result.tolist() == [[
        [True, True],
        [False, False],
    ]]


def test_detector_overlap_valid_returns_empty_for_no_active_detector():
    candidates = torch.tensor([[
        _box(0.0), _box(1.0),
    ]], dtype=torch.float32)
    valid = torch.tensor([[True, True]])
    detected = torch.tensor([[
        _box(0.0), _box(1.0),
    ]], dtype=torch.float32)
    detected_valid = torch.tensor([[False, False]])

    result = build_detector_overlap_valid(
        candidates, valid, detected, detected_valid
    )

    assert not bool(result.any().item())


def test_detector_overlap_valid_treats_degenerate_detector_as_no_overlap():
    candidates = torch.tensor([[
        _box(0.0), _box(10.0),
    ]], dtype=torch.float32)
    valid = torch.tensor([[True, True]])
    detected = torch.tensor([[
        _box(0.0), [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]], dtype=torch.float32)
    detected_valid = torch.tensor([[True, True]])

    result = build_detector_overlap_valid(
        candidates, valid, detected, detected_valid
    )

    assert result.dtype == torch.bool
    assert result.tolist() == [[True, False]]


@pytest.mark.parametrize("mutation", [
    "candidate_shape",
    "candidate_dtype",
    "valid_dtype",
    "device_contract",
    "invalid_size",
    "threshold",
])
def test_detector_overlap_valid_rejects_malformed_inputs(mutation):
    candidates = torch.tensor([[_box(0.0)]], dtype=torch.float32)
    valid = torch.tensor([[True]])
    detected = torch.tensor([[_box(0.0)]], dtype=torch.float32)
    detected_valid = torch.tensor([[True]])
    threshold = 0.25
    if mutation == "candidate_shape":
        candidates = candidates[..., :5]
    elif mutation == "candidate_dtype":
        candidates = candidates.long()
    elif mutation == "valid_dtype":
        valid = valid.float()
    elif mutation == "device_contract":
        detected = detected.to("meta")
    elif mutation == "invalid_size":
        candidates[..., 3] = 0.0
    else:
        threshold = float("nan")

    with pytest.raises((TypeError, ValueError)):
        build_detector_overlap_valid(
            candidates, valid, detected, detected_valid, threshold
        )
