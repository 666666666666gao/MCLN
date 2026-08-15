import math

import torch

from scripts.run_v117_meshsp_calibrated_adapter_oof import (
    threshold_calibration_loss,
)


def _logit(probability):
    return math.log(probability / (1.0 - probability))


def _head_logits(probability025, probability050):
    return torch.tensor([
        _logit(probability025),
        _logit(probability050 / probability025),
    ], dtype=torch.float32)


def _case(inverted=False):
    candidate_ious = torch.full((1, 16, 7), 0.10, dtype=torch.float32)
    candidate_ious[0, 0, 0] = 0.80
    query_valid = torch.ones(1, 16, dtype=torch.bool)
    variant_valid = torch.ones(1, 16, 7, dtype=torch.bool)
    low = _head_logits(0.02, 0.01)
    high = _head_logits(0.98, 0.96)
    query_logits = low.expand(1, 16, 2).clone()
    variant_logits = low.expand(1, 16, 7, 2).clone()
    query_logits[0, 0] = high
    variant_logits[0, 0, 0] = high
    if inverted:
        query_logits = -query_logits
        variant_logits = -variant_logits
    return {
        "outputs": {
            "query_logits": query_logits,
            "variant_logits": variant_logits,
        },
        "candidate_ious": candidate_ious,
        "query_valid": query_valid,
        "variant_valid": variant_valid,
    }


def test_v117_calibration_prefers_absolute_threshold_correct_predictions():
    correct = _case(inverted=False)
    inverted = _case(inverted=True)
    correct_loss, correct_stats = threshold_calibration_loss(**correct)
    inverted_loss, _ = threshold_calibration_loss(**inverted)
    assert correct_loss.item() < inverted_loss.item()
    assert set(correct_stats) == {
        "calibration_query_loss", "calibration_variant_loss"
    }


def test_v117_calibration_is_finite_and_backpropagates_both_levels():
    case = _case(inverted=False)
    query_logits = case["outputs"]["query_logits"].requires_grad_()
    variant_logits = case["outputs"]["variant_logits"].requires_grad_()
    case["outputs"] = {
        "query_logits": query_logits,
        "variant_logits": variant_logits,
    }
    loss, _ = threshold_calibration_loss(**case)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(query_logits.grad).all()
    assert torch.isfinite(variant_logits.grad).all()
    assert query_logits.grad.abs().sum() > 0
    assert variant_logits.grad.abs().sum() > 0

