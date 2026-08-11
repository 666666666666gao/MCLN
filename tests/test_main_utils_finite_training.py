from types import SimpleNamespace

import pytest
import torch

import main_utils
from main_utils import BaseTrainTester


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class _ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return {
            "prediction": self.weight * inputs["point_clouds"].mean()
        }


class _NanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


def _args():
    return SimpleNamespace(
        clip_norm=0.0,
        consistency_loss_scale=1.0,
        mask_loss_scale=1.0,
        num_decoder_layers=6,
        print_freq=100,
        query_points_obj_topk=4,
        source_choice_selector_choice_target="target",
        source_choice_selector_default_source="default",
        source_choice_selector_loss_weight=0.0,
        source_choice_selector_min_iou_gap=0.05,
        use_source_choice_selector=False,
    )


def _batch(value):
    return {
        "point_clouds": torch.tensor([float(value)]),
        "utterances": ["object"],
    }


def _tester():
    tester = BaseTrainTester.__new__(BaseTrainTester)
    tester.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    return tester


@pytest.fixture(autouse=True)
def _single_process_rank(monkeypatch):
    monkeypatch.setattr(main_utils.dist, "get_rank", lambda: 1)


def test_nonfinite_total_loss_rejects_before_backward_or_step():
    model = _ScalarModel()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()

    def criterion(end_points, *_args, **_kwargs):
        loss = end_points["prediction"] * float("nan")
        return loss, dict(end_points, loss=loss)

    with pytest.raises(ValueError, match="total loss.*finite"):
        _tester().train_one_epoch(
            1, [_batch(1.0)], model, criterion, None,
            optimizer, scheduler, _args(),
        )

    assert model.weight.grad is None
    assert optimizer.step_calls == 0
    assert scheduler.step_calls == 0


def test_nonfinite_optimizer_gradient_rejects_before_step():
    model = _ScalarModel()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()

    def criterion(end_points, *_args, **_kwargs):
        loss = _NanGradient.apply(end_points["prediction"])
        return loss, dict(end_points, loss=loss)

    with pytest.raises(ValueError, match="optimizer gradient.*finite"):
        _tester().train_one_epoch(
            1, [_batch(1.0)], model, criterion, None,
            optimizer, scheduler, _args(),
        )

    assert torch.isnan(model.weight.grad)
    assert optimizer.step_calls == 0
    assert scheduler.step_calls == 0


def test_train_one_epoch_returns_exact_finite_loss_means():
    model = _ScalarModel()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()

    def criterion(end_points, *_args, **_kwargs):
        loss = end_points["prediction"]
        return loss, dict(
            end_points,
            mask_loss=loss * 2.0,
            python_loss=float(loss.detach()) + 1.0,
        )

    receipt = _tester().train_one_epoch(
        55, [_batch(1.0), _batch(3.0)], model, criterion, None,
        optimizer, scheduler, _args(),
    )

    assert receipt == {
        "schema": "mcln-train-loss-epoch-v1",
        "batch_count": 2,
        "loss_means": {
            "mask_loss": 4.0,
            "python_loss": 3.0,
            "total_loss": 2.0,
        },
    }
    assert optimizer.step_calls == 2
    assert scheduler.step_calls == 2


def test_train_one_epoch_accepts_one_element_endpoint_loss_tensor():
    model = _ScalarModel()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()

    def criterion(end_points, *_args, **_kwargs):
        loss = end_points["prediction"]
        return loss, dict(
            end_points,
            head_loss=torch.stack([loss * 2.5]),
        )

    receipt = _tester().train_one_epoch(
        55, [_batch(1.0)], model, criterion, None,
        optimizer, scheduler, _args(),
    )

    assert receipt["loss_means"] == {
        "head_loss": 2.5,
        "total_loss": 1.0,
    }
    assert optimizer.step_calls == 1
    assert scheduler.step_calls == 1


@pytest.mark.parametrize("endpoint_loss", [
    torch.tensor([1.0, 2.0]),
    torch.tensor([float("nan")]),
    torch.tensor([float("inf")]),
])
def test_train_one_epoch_rejects_invalid_endpoint_loss_tensor(endpoint_loss):
    model = _ScalarModel()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()

    def criterion(end_points, *_args, **_kwargs):
        loss = end_points["prediction"]
        return loss, dict(end_points, head_loss=endpoint_loss)

    with pytest.raises(ValueError, match="end_points head_loss"):
        _tester().train_one_epoch(
            55, [_batch(1.0)], model, criterion, None,
            optimizer, scheduler, _args(),
        )

    assert model.weight.grad is None
    assert optimizer.step_calls == 0
    assert scheduler.step_calls == 0
