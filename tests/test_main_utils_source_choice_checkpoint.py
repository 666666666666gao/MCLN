import copy
import math
from types import SimpleNamespace

import pytest
import torch

from main_utils import BaseTrainTester, load_checkpoint
from utils.lr_scheduler import GradualWarmupScheduler


class FailingScheduler:
    def load_state_dict(self, _state):
        raise AssertionError("scheduler state should be skipped")


class MutatingFailScheduler:
    def __init__(self):
        self.value = "initial"

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]
        if state.get("fail"):
            raise RuntimeError("scheduler checkpoint is damaged")


class DDPishSourceChoiceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Module()
        self.module.source_choice_selector = torch.nn.Module()
        self.module.source_choice_selector.source_embedding = torch.nn.Embedding(
            3, 16
        )
        self.module.backbone = torch.nn.Linear(2, 2)
        self.module.register_buffer("scalar_temperature", torch.tensor(1.0))


class CheckpointToyMCLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(2, 2)
        self.backbone_net = torch.nn.Linear(2, 2)
        self.x_mask = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)
        self.source_choice_selector = torch.nn.Linear(2, 2)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False


class GateResumeToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, values):
        return self.linear(values)


class QueryMaskResumeToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.query_mask_fusion_calibrator = torch.nn.Linear(3, 2)

    def forward(self, values):
        with torch.no_grad():
            features = self.backbone(values)
        return self.query_mask_fusion_calibrator(features)


def _optimizer_args(**overrides):
    values = {
        "source_choice_selector_train_only": False,
        "use_source_choice_selector": True,
        "frozen": False,
        "small_lr": False,
        "source_choice_selector_lr": 7e-4,
        "mask_head_lr_multiplier": 4.0,
        "lr": 2e-5,
        "lr_backbone": 2e-4,
        "text_encoder_lr": 3e-6,
        "weight_decay": 5e-4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _checkpoint_args(path, reduce_lr=False, **overrides):
    values = {
        "checkpoint_path": str(path),
        "eval": False,
        "reduce_lr": reduce_lr,
        "source_choice_selector_train_only": False,
        "use_source_choice_selector": True,
        "frozen": False,
        "small_lr": False,
        "start_epoch": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exact_gate_only_resume_restores_optimizer_scheduler_and_epoch(
        tmp_path):
    checkpoint_model = GateResumeToy()
    checkpoint_optimizer = torch.optim.AdamW(
        checkpoint_model.parameters(), lr=3e-4
    )
    checkpoint_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        checkpoint_optimizer, milestones=[10], gamma=0.1
    )
    checkpoint_model(torch.randn(2, 3)).sum().backward()
    checkpoint_optimizer.step()
    checkpoint_scheduler.step()
    checkpoint_path = tmp_path / "gate-resume.pth"
    torch.save({
        "config": {"use_source_moe": True},
        "epoch": 1,
        "model": checkpoint_model.state_dict(),
        "optimizer": checkpoint_optimizer.state_dict(),
        "scheduler": checkpoint_scheduler.state_dict(),
    }, checkpoint_path)

    model = GateResumeToy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10], gamma=0.1
    )
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        start_epoch=2,
        checkpoint_start_epoch=None,
        eval=False,
        reduce_lr=False,
        use_source_moe=True,
        use_source_choice_selector=False,
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_resume_optimizer=True,
        frozen=False,
        small_lr=False,
    )

    load_checkpoint(args, model, optimizer, scheduler)

    assert args.start_epoch == 2
    assert len(optimizer.state) == len(checkpoint_optimizer.state)
    assert scheduler.last_epoch == checkpoint_scheduler.last_epoch
    for loaded, expected in zip(
            model.parameters(), checkpoint_model.parameters()):
        assert torch.equal(loaded, expected)


def test_exact_gate_only_resume_rejects_noncontiguous_start_epoch(tmp_path):
    model = GateResumeToy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10], gamma=0.1
    )
    checkpoint_path = tmp_path / "gate-resume.pth"
    torch.save({
        "config": {"use_source_moe": True},
        "epoch": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, checkpoint_path)
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        start_epoch=4,
        checkpoint_start_epoch=None,
        eval=False,
        reduce_lr=False,
        use_source_moe=True,
        use_source_choice_selector=False,
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_resume_optimizer=True,
        frozen=False,
        small_lr=False,
    )

    with pytest.raises(ValueError, match="checkpoint epoch \\+ 1"):
        load_checkpoint(args, model, optimizer, scheduler)


def _query_mask_resume_args(path, **overrides):
    values = {
        "checkpoint_path": str(path),
        "start_epoch": 2,
        "checkpoint_start_epoch": None,
        "eval": False,
        "reduce_lr": False,
        "use_source_moe": False,
        "use_source_choice_selector": False,
        "source_choice_selector_train_only": False,
        "source_moe_train_only": False,
        "source_moe_gate_train_only": False,
        "source_moe_gate_resume_optimizer": False,
        "use_query_mask_fusion_calibrator": True,
        "query_mask_fusion_train_only": True,
        "query_mask_fusion_resume_optimizer": True,
        "query_mask_fusion_lr": 1e-4,
        "query_mask_fusion_hidden_dim": 128,
        "query_mask_fusion_dropout": 0.0,
        "query_mask_fusion_max_delta": 0.1,
        "frozen": False,
        "small_lr": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _query_mask_checkpoint_config():
    return {
        "use_query_mask_fusion_calibrator": True,
        "query_mask_fusion_train_only": True,
        "query_mask_fusion_lr": 1e-4,
        "query_mask_fusion_hidden_dim": 128,
        "query_mask_fusion_dropout": 0.0,
        "query_mask_fusion_max_delta": 0.1,
    }


def test_exact_query_mask_resume_restores_optimizer_scheduler_and_epoch(
        tmp_path):
    checkpoint_model = QueryMaskResumeToy()
    checkpoint_optimizer = torch.optim.AdamW(
        checkpoint_model.query_mask_fusion_calibrator.parameters(), lr=1e-4
    )
    checkpoint_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        checkpoint_optimizer, milestones=[50, 75], gamma=0.1
    )
    checkpoint_model(torch.randn(2, 3)).sum().backward()
    checkpoint_optimizer.step()
    checkpoint_scheduler.step()
    checkpoint_path = tmp_path / "query-mask-resume.pth"
    torch.save({
        "config": _query_mask_checkpoint_config(),
        "epoch": 1,
        "model": checkpoint_model.state_dict(),
        "optimizer": checkpoint_optimizer.state_dict(),
        "scheduler": checkpoint_scheduler.state_dict(),
    }, checkpoint_path)

    model = QueryMaskResumeToy()
    optimizer = torch.optim.AdamW(
        model.query_mask_fusion_calibrator.parameters(), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )

    args = _query_mask_resume_args(checkpoint_path)
    load_checkpoint(args, model, optimizer, scheduler)

    assert args.start_epoch == 2
    assert len(optimizer.state) == len(checkpoint_optimizer.state)
    _assert_nested_equal(
        optimizer.state_dict(), checkpoint_optimizer.state_dict()
    )
    _assert_nested_equal(
        scheduler.state_dict(), checkpoint_scheduler.state_dict()
    )
    for loaded, expected in zip(
            model.parameters(), checkpoint_model.parameters()):
        assert torch.equal(loaded, expected)


def test_exact_query_mask_resume_rejects_noncontiguous_start_epoch(tmp_path):
    model = QueryMaskResumeToy()
    optimizer = torch.optim.AdamW(
        model.query_mask_fusion_calibrator.parameters(), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )
    checkpoint_path = tmp_path / "query-mask-resume.pth"
    torch.save({
        "config": _query_mask_checkpoint_config(),
        "epoch": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint epoch \\+ 1"):
        load_checkpoint(
            _query_mask_resume_args(checkpoint_path, start_epoch=3),
            model,
            optimizer,
            scheduler,
        )


def test_exact_query_mask_resume_rejects_config_drift(tmp_path):
    model = QueryMaskResumeToy()
    optimizer = torch.optim.AdamW(
        model.query_mask_fusion_calibrator.parameters(), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )
    checkpoint_path = tmp_path / "query-mask-resume.pth"
    torch.save({
        "config": _query_mask_checkpoint_config(),
        "epoch": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, checkpoint_path)

    with pytest.raises(ValueError, match="query_mask_fusion_max_delta"):
        load_checkpoint(
            _query_mask_resume_args(
                checkpoint_path, query_mask_fusion_max_delta=0.2
            ),
            model,
            optimizer,
            scheduler,
        )


def _legacy_joint_source_choice_optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "backbone_net" not in name
                    and "text_encoder" not in name
                    and parameter.requires_grad
                ]
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "backbone_net" in name and parameter.requires_grad
                ],
                "lr": 2e-4,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "text_encoder" in name and parameter.requires_grad
                ],
                "lr": 3e-6,
            },
        ],
        lr=2e-5,
        weight_decay=5e-4,
    )


def _take_distinct_adam_step(model, optimizer, scale=1.0):
    for index, (_name, parameter) in enumerate(model.named_parameters(), 1):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(
                parameter, float(index) * float(scale)
            )
    optimizer.step()
    optimizer.zero_grad()


def _assert_optimizer_parameter_states_equal(
    source_model,
    source_optimizer,
    target_model,
    target_optimizer,
):
    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(target_model.named_parameters())
    for name, source_parameter in source_parameters.items():
        if not source_parameter.requires_grad:
            continue
        target_parameter = target_parameters[name]
        source_state = source_optimizer.state[source_parameter]
        target_state = target_optimizer.state[target_parameter]
        assert set(target_state) == set(source_state)
        for key, expected in source_state.items():
            actual = target_state[key]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_selector_only_checkpoint_load_skips_incompatible_optimizer_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    checkpoint_optimizer = torch.optim.Adam(
        [model.weight, model.bias],
        lr=0.001,
    )
    checkpoint_path = tmp_path / "baseline.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": checkpoint_optimizer.state_dict(),
            "scheduler": {"unused": True},
            "epoch": 7,
        },
        checkpoint_path,
    )

    selector_only_optimizer = torch.optim.Adam([model.weight], lr=0.01)
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        eval=False,
        reduce_lr=False,
        source_choice_selector_train_only=True,
    )

    load_checkpoint(
        args,
        model,
        selector_only_optimizer,
        FailingScheduler(),
    )

    assert args.start_epoch == 1


def test_checkpoint_load_skips_mismatched_source_choice_selector_shape(tmp_path):
    model = DDPishSourceChoiceModel()
    initial_embedding = model.state_dict()[
        "module.source_choice_selector.source_embedding.weight"
    ].clone()
    checkpoint_state = model.state_dict()
    checkpoint_state["module.source_choice_selector.source_embedding.weight"] = (
        torch.arange(32, dtype=torch.float32).view(2, 16)
    )
    checkpoint_state["module.backbone.weight"] = torch.full((2, 2), 2.0)
    checkpoint_state["module.scalar_temperature"] = torch.tensor(2.0)
    checkpoint_path = tmp_path / "two_source_selector.pth"
    torch.save(
        {
            "model": checkpoint_state,
            "optimizer": {},
            "scheduler": {},
            "epoch": 70,
        },
        checkpoint_path,
    )

    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        eval=False,
        reduce_lr=True,
        source_choice_selector_train_only=False,
    )

    load_checkpoint(args, model, torch.optim.Adam(model.parameters()), None)

    assert args.start_epoch == 71
    assert torch.equal(
        model.state_dict()["module.backbone.weight"],
        torch.full((2, 2), 2.0),
    )
    assert torch.equal(
        model.state_dict()["module.scalar_temperature"],
        torch.tensor(2.0),
    )
    loaded_embedding = model.state_dict()[
        "module.source_choice_selector.source_embedding.weight"
    ]
    assert loaded_embedding.shape == (3, 16)
    assert torch.equal(
        loaded_embedding[:2],
        torch.arange(32, dtype=torch.float32).view(2, 16),
    )
    assert torch.equal(loaded_embedding[2], initial_embedding[2])


@pytest.mark.parametrize(
    "branch_overrides",
    [
        {"use_source_choice_selector": False},
        {"frozen": True},
        {"small_lr": True},
    ],
    ids=["no-selector", "frozen", "small-lr"],
)
def test_legacy_optimizer_branches_keep_native_same_layout_checkpoint_loading(
    tmp_path,
    branch_overrides,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(**branch_overrides),
        source_model,
    )
    source_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    _take_distinct_adam_step(source_model, source_optimizer)
    source_scheduler.step()
    checkpoint_path = tmp_path / "legacy_native_layout.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "epoch": 4,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(**branch_overrides),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[10],
        gamma=0.1,
    )

    load_checkpoint(
        _checkpoint_args(checkpoint_path, **branch_overrides),
        target_model,
        target_optimizer,
        target_scheduler,
    )

    _assert_optimizer_parameter_states_equal(
        source_model,
        source_optimizer,
        target_model,
        target_optimizer,
    )
    assert [group["lr"] for group in target_optimizer.param_groups] == [
        group["lr"] for group in source_optimizer.param_groups
    ]
    assert target_scheduler.state_dict() == source_scheduler.state_dict()


def test_checkpoint_migrates_legacy_three_group_source_choice_optimizer_state(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = _legacy_joint_source_choice_optimizer(source_model)
    source_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[1],
        gamma=0.1,
    )
    _take_distinct_adam_step(source_model, source_optimizer)
    source_scheduler.step()
    checkpoint_path = tmp_path / "legacy_three_group.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "epoch": 7,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[1],
        gamma=0.1,
    )

    args = _checkpoint_args(checkpoint_path)
    load_checkpoint(
        args,
        target_model,
        target_optimizer,
        target_scheduler,
    )

    assert args.start_epoch == 8
    _assert_optimizer_parameter_states_equal(
        source_model,
        source_optimizer,
        target_model,
        target_optimizer,
    )
    expected_lrs = [2e-6, 2e-5, 8e-6, 7e-5]
    assert [group["lr"] for group in target_optimizer.param_groups] == (
        pytest.approx(expected_lrs)
    )
    assert target_scheduler.last_epoch == source_scheduler.last_epoch
    assert target_scheduler.base_lrs == pytest.approx(
        [2e-5, 2e-4, 8e-5, 7e-4]
    )
    assert target_scheduler._last_lr == pytest.approx(expected_lrs)


def test_checkpoint_migrates_cosine_scheduler_with_nonzero_eta_min(tmp_path):
    source_model = CheckpointToyMCLN()
    source_optimizer = _legacy_joint_source_choice_optimizer(source_model)
    source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        source_optimizer,
        T_max=10,
        eta_min=1e-6,
    )
    for _step in range(3):
        _take_distinct_adam_step(source_model, source_optimizer)
        source_scheduler.step()
    checkpoint_path = tmp_path / "legacy_cosine_three_group.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "epoch": 7,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        target_optimizer,
        T_max=10,
        eta_min=1e-6,
    )

    load_checkpoint(
        _checkpoint_args(checkpoint_path),
        target_model,
        target_optimizer,
        target_scheduler,
    )

    base_lrs = [2e-5, 2e-4, 8e-5, 7e-4]
    cosine_factor = (
        1.0 + math.cos(math.pi * source_scheduler.last_epoch / 10.0)
    ) / 2.0
    expected_lrs = [
        1e-6 + (base_lr - 1e-6) * cosine_factor
        for base_lr in base_lrs
    ]
    assert [group["lr"] for group in target_optimizer.param_groups] == (
        pytest.approx(expected_lrs)
    )
    assert target_scheduler.base_lrs == pytest.approx(base_lrs)
    assert target_scheduler._last_lr == pytest.approx(expected_lrs)
    assert target_scheduler.last_epoch == source_scheduler.last_epoch

    _take_distinct_adam_step(target_model, target_optimizer)
    target_scheduler.step()
    next_factor = (
        1.0 + math.cos(math.pi * target_scheduler.last_epoch / 10.0)
    ) / 2.0
    expected_next_lrs = [
        1e-6 + (base_lr - 1e-6) * next_factor
        for base_lr in base_lrs
    ]
    assert [group["lr"] for group in target_optimizer.param_groups] == (
        pytest.approx(expected_next_lrs)
    )


@pytest.mark.parametrize("scheduler_kind", ["cosine", "multistep"])
def test_checkpoint_migrates_gradual_warmup_after_scheduler_state(
    tmp_path,
    scheduler_kind,
):
    def make_scheduler(optimizer):
        if scheduler_kind == "cosine":
            after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,
                eta_min=1e-6,
            )
        else:
            after_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[1, 3],
                gamma=0.1,
            )
        return GradualWarmupScheduler(
            optimizer,
            multiplier=10,
            warmup_epoch=2,
            after_scheduler=after_scheduler,
        )

    def expected_after_scheduler_lrs(scheduler, base_lrs):
        if scheduler_kind == "cosine":
            factor = (
                1.0
                + math.cos(
                    math.pi * scheduler.last_epoch / scheduler.T_max
                )
            ) / 2.0
            return [
                scheduler.eta_min
                + (base_lr - scheduler.eta_min) * factor
                for base_lr in base_lrs
            ]
        decay_count = sum(
            count
            for milestone, count in scheduler.milestones.items()
            if milestone <= scheduler.last_epoch
        )
        return [
            base_lr * scheduler.gamma ** decay_count
            for base_lr in base_lrs
        ]

    source_model = CheckpointToyMCLN()
    source_optimizer = _legacy_joint_source_choice_optimizer(source_model)
    source_scheduler = make_scheduler(source_optimizer)
    for _step in range(5):
        _take_distinct_adam_step(source_model, source_optimizer)
        source_scheduler.step()

    assert source_scheduler.last_epoch == 5
    assert source_scheduler.after_scheduler.last_epoch == 3
    assert source_scheduler._last_lr != pytest.approx(
        [group["lr"] for group in source_optimizer.param_groups]
    )

    checkpoint_path = tmp_path / (
        "legacy_gradual_warmup_{}.pth".format(scheduler_kind)
    )
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "epoch": 7,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = make_scheduler(target_optimizer)

    load_checkpoint(
        _checkpoint_args(checkpoint_path),
        target_model,
        target_optimizer,
        target_scheduler,
    )

    _assert_optimizer_parameter_states_equal(
        source_model,
        source_optimizer,
        target_model,
        target_optimizer,
    )
    source_state = source_scheduler.state_dict()
    target_state = target_scheduler.state_dict()
    for key, expected in source_state.items():
        if key not in {"base_lrs", "_last_lr", "after_scheduler"}:
            _assert_nested_equal(target_state[key], expected)
    for key, expected in source_state["after_scheduler"].items():
        if key not in {"base_lrs", "_last_lr"}:
            _assert_nested_equal(
                target_state["after_scheduler"][key],
                expected,
            )

    base_lrs = [2e-5, 2e-4, 8e-5, 7e-4]
    expected_lrs = expected_after_scheduler_lrs(
        target_scheduler.after_scheduler,
        base_lrs,
    )
    assert target_scheduler.base_lrs == pytest.approx(base_lrs)
    assert target_scheduler.after_scheduler.base_lrs == pytest.approx(
        base_lrs
    )
    assert target_scheduler._last_lr == pytest.approx(expected_lrs)
    assert target_scheduler.after_scheduler._last_lr == pytest.approx(
        expected_lrs
    )
    assert [group["lr"] for group in target_optimizer.param_groups] == (
        pytest.approx(expected_lrs)
    )

    _take_distinct_adam_step(target_model, target_optimizer)
    target_scheduler.step()
    expected_next_lrs = expected_after_scheduler_lrs(
        target_scheduler.after_scheduler,
        base_lrs,
    )
    assert [group["lr"] for group in target_optimizer.param_groups] == (
        pytest.approx(expected_next_lrs)
    )


def test_checkpoint_rejects_ambiguous_legacy_optimizer_with_actionable_error(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = _legacy_joint_source_choice_optimizer(source_model)
    _take_distinct_adam_step(source_model, source_optimizer)
    optimizer_state = source_optimizer.state_dict()
    removed_identifier = optimizer_state["param_groups"][0]["params"].pop()
    optimizer_state["state"].pop(removed_identifier)
    source_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    checkpoint_path = tmp_path / "ambiguous_three_group.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "epoch": 7,
        },
        checkpoint_path,
    )
    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[10],
        gamma=0.1,
    )

    with pytest.raises(
        ValueError,
        match=r"cannot migrate legacy 3-group.*--reduce_lr",
    ):
        load_checkpoint(
            _checkpoint_args(checkpoint_path),
            target_model,
            target_optimizer,
            target_scheduler,
        )


def test_checkpoint_rejects_equal_count_reordered_strict_optimizer_groups(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        source_model,
    )
    source_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    _take_distinct_adam_step(source_model, source_optimizer)
    optimizer_state = source_optimizer.state_dict()
    optimizer_state["param_groups"][0], optimizer_state["param_groups"][2] = (
        optimizer_state["param_groups"][2],
        optimizer_state["param_groups"][0],
    )
    checkpoint_path = tmp_path / "reordered_four_group.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "epoch": 4,
        },
        checkpoint_path,
    )
    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[10],
        gamma=0.1,
    )

    with pytest.raises(
        ValueError,
        match=r"cannot prove optimizer checkpoint layout.*--reduce_lr",
    ):
        load_checkpoint(
            _checkpoint_args(checkpoint_path),
            target_model,
            target_optimizer,
            target_scheduler,
        )

    assert [group["name"] for group in target_optimizer.param_groups] == [
        "decoder",
        "backbone",
        "mask_head",
        "selector",
    ]
    assert not target_optimizer.state


def test_checkpoint_rejects_equal_count_groups_without_layout_metadata(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        source_model,
    )
    source_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    _take_distinct_adam_step(source_model, source_optimizer)
    optimizer_state = source_optimizer.state_dict()
    for group in optimizer_state["param_groups"]:
        group.pop("name")
        group.pop("parameter_names")
    checkpoint_path = tmp_path / "unverifiable_four_group.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "epoch": 4,
        },
        checkpoint_path,
    )
    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[10],
        gamma=0.1,
    )

    with pytest.raises(
        ValueError,
        match=r"cannot prove optimizer checkpoint layout.*--reduce_lr",
    ):
        load_checkpoint(
            _checkpoint_args(checkpoint_path),
            target_model,
            target_optimizer,
            target_scheduler,
        )

    assert [group["name"] for group in target_optimizer.param_groups] == [
        "decoder",
        "backbone",
        "mask_head",
        "selector",
    ]
    assert not target_optimizer.state


def test_checkpoint_preserves_direct_load_for_matching_optimizer_layout(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(
            lr=3e-5,
            lr_backbone=4e-4,
            source_choice_selector_lr=9e-4,
            mask_head_lr_multiplier=2.0,
        ),
        source_model,
    )
    source_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        source_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    _take_distinct_adam_step(source_model, source_optimizer)
    source_scheduler.step()
    checkpoint_path = tmp_path / "matching_four_group.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "epoch": 4,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[10],
        gamma=0.1,
    )

    load_checkpoint(
        _checkpoint_args(checkpoint_path),
        target_model,
        target_optimizer,
        target_scheduler,
    )

    assert [group["lr"] for group in target_optimizer.param_groups] == [
        group["lr"] for group in source_optimizer.param_groups
    ]
    _assert_optimizer_parameter_states_equal(
        source_model,
        source_optimizer,
        target_model,
        target_optimizer,
    )
    assert target_scheduler.state_dict() == source_scheduler.state_dict()


def test_missing_scheduler_checkpoint_does_not_partially_migrate_optimizer(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = _legacy_joint_source_choice_optimizer(source_model)
    _take_distinct_adam_step(source_model, source_optimizer)
    checkpoint_path = tmp_path / "missing_scheduler.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "epoch": 7,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(),
        target_model,
    )
    target_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        target_optimizer,
        milestones=[10],
        gamma=0.1,
    )
    _take_distinct_adam_step(target_model, target_optimizer, scale=10.0)
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    scheduler_before = copy.deepcopy(target_scheduler.state_dict())

    with pytest.raises(KeyError, match="scheduler"):
        load_checkpoint(
            _checkpoint_args(checkpoint_path),
            target_model,
            target_optimizer,
            target_scheduler,
        )

    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(target_scheduler.state_dict(), scheduler_before)


def test_scheduler_load_failure_rolls_back_native_optimizer_and_scheduler(
    tmp_path,
):
    source_model = CheckpointToyMCLN()
    source_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(use_source_choice_selector=False),
        source_model,
    )
    _take_distinct_adam_step(source_model, source_optimizer)
    checkpoint_path = tmp_path / "damaged_scheduler.pth"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": {"value": "checkpoint", "fail": True},
            "epoch": 4,
        },
        checkpoint_path,
    )

    target_model = CheckpointToyMCLN()
    target_optimizer = BaseTrainTester.get_optimizer(
        _optimizer_args(
            use_source_choice_selector=False,
            lr=9e-5,
            lr_backbone=8e-4,
            text_encoder_lr=7e-6,
        ),
        target_model,
    )
    _take_distinct_adam_step(target_model, target_optimizer)
    target_scheduler = MutatingFailScheduler()
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    scheduler_before = copy.deepcopy(target_scheduler.state_dict())

    with pytest.raises(RuntimeError, match="scheduler checkpoint is damaged"):
        load_checkpoint(
            _checkpoint_args(
                checkpoint_path,
                use_source_choice_selector=False,
            ),
            target_model,
            target_optimizer,
            target_scheduler,
        )

    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(target_scheduler.state_dict(), scheduler_before)


def test_source_choice_diagnostics_are_formatted_for_training_logs():
    assert hasattr(BaseTrainTester, "_format_source_choice_diagnostics")

    stat_dict = {
        "loss": 12.0,
        "source_choice_loss": 0.6,
        "source_choice_target_non_default_ratio": torch.tensor(0.4),
        "source_choice_selected_non_default_ratio": torch.tensor(0.2),
        "source_choice_target_acc": torch.tensor(0.8),
        "source_choice_oracle_acc025": torch.tensor(0.9),
        "source_choice_default_acc025": torch.tensor(0.7),
        "unrelated_ratio": torch.tensor(1.0),
    }

    formatted = BaseTrainTester._format_source_choice_diagnostics(
        stat_dict, denom=2.0
    )

    assert formatted == (
        "source_choice_default_acc025 0.3500 \t"
        "source_choice_oracle_acc025 0.4500 \t"
        "source_choice_selected_non_default_ratio 0.1000 \t"
        "source_choice_target_acc 0.4000 \t"
        "source_choice_target_non_default_ratio 0.2000 \t"
    )
