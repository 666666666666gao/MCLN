import math
import sys
from types import SimpleNamespace

import pytest
import torch

from models.mcln_training_groups import (
    MASK_HEAD_PREFIXES,
    bare_parameter_name,
    build_mcln_optimizer_param_groups,
    parameter_group_name,
)
from main_utils import BaseTrainTester, parse_option


class ToyMCLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(2, 2)
        self.backbone_net = torch.nn.Linear(2, 2)
        self.x_mask = torch.nn.Linear(2, 2)
        self.x_query = torch.nn.Linear(2, 2)
        self.rel_encoder = torch.nn.Linear(2, 2)
        self.swa_layers = torch.nn.Linear(2, 2)
        self.swa_ffn_layers = torch.nn.Linear(2, 2)
        self.out_norm = torch.nn.LayerNorm(2)
        self.out_score = torch.nn.Linear(2, 2)
        self.text_encoder = torch.nn.Linear(2, 2)
        self.source_choice_selector = torch.nn.Linear(2, 2)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False


class SharedAliasParameterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(1, 1, bias=False)
        self.x_mask = torch.nn.Linear(1, 1, bias=False)
        self.x_mask.weight = self.decoder.weight
        self.source_choice_selector = torch.nn.Linear(1, 1)


class ParentRelativeVerifierToyMCLN(ToyMCLN):
    def __init__(self):
        super().__init__()
        self.structured_slot_builder = torch.nn.Linear(2, 2)
        self.sacr_head = torch.nn.Linear(2, 2)
        self.parent_relative_text_verifier = torch.nn.Linear(2, 2)


def optimizer_args(**overrides):
    values = {
        "source_choice_selector_train_only": False,
        "use_source_choice_selector": False,
        "use_parent_relative_text_verifier": False,
        "parent_relative_text_verifier_train_only": False,
        "parent_relative_text_verifier_counterfactual_training": False,
        "parent_relative_text_verifier_detach_inputs": False,
        "parent_relative_text_verifier_lr": 3e-4,
        "eval": False,
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


def test_parameter_classification_uses_only_exact_top_level_prefixes():
    assert bare_parameter_name("module.x_mask.weight") == "x_mask.weight"
    assert bare_parameter_name("decoder.weight") == "decoder.weight"
    assert parameter_group_name("module.source_choice_selector.weight") == (
        "selector"
    )
    assert parameter_group_name(
        "module.joint_query_quality_reranker.residual_head.weight"
    ) == "selector"
    assert parameter_group_name(
        "module.parent_relative_text_verifier.action_head.weight"
    ) == "selector"
    assert parameter_group_name("module.backbone_net.stem.weight") == "backbone"
    assert parameter_group_name("module.text_encoder.layer.weight") == (
        "frozen_text"
    )
    for prefix in MASK_HEAD_PREFIXES:
        assert parameter_group_name("module." + prefix + "weight") == "mask_head"

    assert parameter_group_name("source_choice_selector_extra.weight") == (
        "decoder"
    )
    assert parameter_group_name("decoder.x_mask.weight") == "decoder"
    assert parameter_group_name("backbone_network.weight") == "decoder"
    assert parameter_group_name("text_encoder_extra.weight") == "decoder"


def test_parent_relative_text_verifier_cli_defaults_are_disabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dist_mod.py"])
    args = parse_option()
    assert args.use_parent_relative_text_verifier is False
    assert args.parent_relative_text_verifier_train_only is False
    assert (
        args.parent_relative_text_verifier_counterfactual_training is False
    )


def test_complete_innovation_groups_are_disjoint_and_use_requested_lrs():
    model = ToyMCLN()

    groups = build_mcln_optimizer_param_groups(
        model,
        decoder_lr=2e-5,
        backbone_lr=2e-4,
        selector_lr=7e-4,
        mask_head_lr_multiplier=4.0,
    )

    assert [group["name"] for group in groups] == [
        "decoder",
        "backbone",
        "mask_head",
        "selector",
    ]
    assert [group["lr"] for group in groups] == [2e-5, 2e-4, 8e-5, 7e-4]
    identities = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(identities) == len(set(identities))
    assert set(identities) == expected
    for group in groups:
        assert group["parameter_names"] == tuple(
            sorted(group["parameter_names"])
        )
        assert math.isfinite(group["lr"])
        assert group["lr"] > 0.0


def test_parameter_names_are_aligned_with_serialized_parameter_order():
    model = ToyMCLN()
    names_by_identity = {
        id(parameter): name for name, parameter in model.named_parameters()
    }

    groups = build_mcln_optimizer_param_groups(
        model,
        decoder_lr=2e-5,
        backbone_lr=2e-4,
        selector_lr=7e-4,
        mask_head_lr_multiplier=4.0,
    )

    for group in groups:
        assert group["parameter_names"] == tuple(
            names_by_identity[id(parameter)] for parameter in group["params"]
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("decoder_lr", 0.0),
        ("backbone_lr", -1.0),
        ("selector_lr", float("nan")),
        ("selector_lr", float("inf")),
        ("mask_head_lr_multiplier", True),
        ("mask_head_lr_multiplier", "4"),
    ],
)
def test_group_builder_rejects_invalid_numeric_inputs(keyword, value):
    arguments = {
        "decoder_lr": 2e-5,
        "backbone_lr": 2e-4,
        "selector_lr": 7e-4,
        "mask_head_lr_multiplier": 4.0,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match="finite and positive"):
        build_mcln_optimizer_param_groups(ToyMCLN(), **arguments)


def test_group_builder_rejects_non_finite_derived_mask_head_lr():
    with pytest.raises(ValueError, match="mask_head.*finite and positive"):
        build_mcln_optimizer_param_groups(
            ToyMCLN(),
            decoder_lr=1e308,
            backbone_lr=2e-4,
            selector_lr=7e-4,
            mask_head_lr_multiplier=1e308,
        )


def test_group_builder_rejects_trainable_text_encoder_parameters():
    model = ToyMCLN()
    for parameter in model.text_encoder.parameters():
        parameter.requires_grad = True

    with pytest.raises(ValueError, match="text_encoder.*unexpectedly trainable"):
        build_mcln_optimizer_param_groups(
            model,
            decoder_lr=2e-5,
            backbone_lr=2e-4,
            selector_lr=7e-4,
            mask_head_lr_multiplier=4.0,
        )


def test_group_builder_requires_selector_parameters_by_default():
    model = ToyMCLN()
    for parameter in model.source_choice_selector.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError, match="requires selector parameters"):
        build_mcln_optimizer_param_groups(
            model,
            decoder_lr=2e-5,
            backbone_lr=2e-4,
            selector_lr=7e-4,
            mask_head_lr_multiplier=4.0,
        )


def test_group_builder_rejects_real_shared_alias_with_both_parameter_paths():
    with pytest.raises(
        ValueError,
        match=r"shared trainable parameter.*decoder\.weight.*x_mask\.weight",
    ):
        build_mcln_optimizer_param_groups(
            SharedAliasParameterModel(),
            decoder_lr=2e-5,
            backbone_lr=2e-4,
            selector_lr=7e-4,
            mask_head_lr_multiplier=4.0,
        )


def test_full_joint_optimizer_uses_strict_named_groups_and_requested_lrs():
    model = ToyMCLN()

    optimizer = BaseTrainTester.get_optimizer(
        optimizer_args(use_source_choice_selector=True),
        model,
    )

    assert [group["name"] for group in optimizer.param_groups] == [
        "decoder",
        "backbone",
        "mask_head",
        "selector",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [
        2e-5,
        2e-4,
        8e-5,
        7e-4,
    ]
    identities = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(identities) == len(set(identities))
    assert set(identities) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_selector_only_optimizer_keeps_legacy_three_group_layout():
    model = ToyMCLN()

    optimizer = BaseTrainTester.get_optimizer(
        optimizer_args(source_choice_selector_train_only=True),
        model,
    )

    assert len(optimizer.param_groups) == 3
    assert [group["lr"] for group in optimizer.param_groups] == [
        7e-4,
        2e-4,
        3e-6,
    ]
    selected = optimizer.param_groups[0]["params"]
    assert selected
    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in model.source_choice_selector.parameters()
    }
    assert all(parameter.requires_grad for parameter in selected)


def test_parent_relative_text_verifier_requires_train_only_during_training():
    model = ToyMCLN()

    with pytest.raises(
            ValueError,
            match="requires parent_relative_text_verifier_train_only"):
        BaseTrainTester.get_optimizer(
            optimizer_args(
                use_parent_relative_text_verifier=True,
                parent_relative_text_verifier_train_only=False,
                eval=False,
            ),
            model,
        )


def test_counterfactual_parent_training_requires_verifier_train_only_mode():
    model = ToyMCLN()

    with pytest.raises(ValueError, match="counterfactual Parent supervision"):
        BaseTrainTester.get_optimizer(
            optimizer_args(
                use_parent_relative_text_verifier=True,
                parent_relative_text_verifier_train_only=False,
                parent_relative_text_verifier_counterfactual_training=True,
            ),
            model,
        )


def test_parent_relative_text_verifier_optimizer_confines_trainable_parameters():
    model = ParentRelativeVerifierToyMCLN()

    optimizer = BaseTrainTester.get_optimizer(
        optimizer_args(
            use_parent_relative_text_verifier=True,
            parent_relative_text_verifier_train_only=True,
        ),
        model,
    )

    expected_prefixes = (
        "structured_slot_builder.",
        "sacr_head.",
        "parent_relative_text_verifier.",
    )
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(expected_prefixes)
    }
    selected = {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    }
    assert selected == expected
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert all(
        parameter.requires_grad == (id(parameter) in expected)
        for parameter in model.parameters()
    )


def test_parent_relative_text_verifier_rejects_detached_train_only_inputs():
    model = ParentRelativeVerifierToyMCLN()

    with pytest.raises(ValueError, match="requires attached"):
        BaseTrainTester.get_optimizer(
            optimizer_args(
                use_parent_relative_text_verifier=True,
                parent_relative_text_verifier_train_only=True,
                parent_relative_text_verifier_detach_inputs=True,
            ),
            model,
        )


def test_frozen_optimizer_keeps_legacy_three_group_layout():
    model = ToyMCLN()

    optimizer = BaseTrainTester.get_optimizer(
        optimizer_args(frozen=True, use_source_choice_selector=True),
        model,
    )

    assert len(optimizer.param_groups) == 3
    assert [group["lr"] for group in optimizer.param_groups] == [
        2e-5,
        2e-4,
        3e-6,
    ]
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if "x_mask" in name or "x_query" in name or "seed_decoder" in name
    }
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == expected


def test_small_lr_optimizer_keeps_legacy_four_group_layout():
    model = ToyMCLN()

    optimizer = BaseTrainTester.get_optimizer(
        optimizer_args(small_lr=True, use_source_choice_selector=True),
        model,
    )

    assert len(optimizer.param_groups) == 4
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2e-5, 2e-7, 2e-6, 3e-8]
    )
    selector_ids = {
        id(parameter) for parameter in model.source_choice_selector.parameters()
    }
    assert selector_ids.issubset(
        {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    )


def test_no_selector_optimizer_keeps_legacy_three_group_layout():
    model = ToyMCLN()
    del model.source_choice_selector

    optimizer = BaseTrainTester.get_optimizer(optimizer_args(), model)

    assert len(optimizer.param_groups) == 3
    assert [group["lr"] for group in optimizer.param_groups] == [
        2e-5,
        2e-4,
        3e-6,
    ]


def test_cli_defaults_expose_joint_mask_training_controls(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_utils.py"])

    args = parse_option()

    assert args.mask_head_lr_multiplier == 1.0
    assert args.mask_loss_scale == 1.0
    assert args.consistency_loss_scale == 1.0


def test_compute_loss_forwards_mask_and_consistency_scales():
    received = {}

    def fake_criterion(end_points, decoder_layers, set_criterion, **kwargs):
        received.update(kwargs)
        return torch.tensor(0.0), end_points

    args = SimpleNamespace(
        num_decoder_layers=6,
        query_points_obj_topk=4,
        use_source_choice_selector=True,
        source_choice_selector_loss_weight=0.5,
        source_choice_selector_default_source="default",
        source_choice_selector_choice_target=(
            "precision_gain_default_sourcewise_focal_bce"
        ),
        source_choice_selector_min_iou_gap=0.03,
        mask_loss_scale=2.0,
        consistency_loss_scale=0.25,
    )

    BaseTrainTester._compute_loss({}, fake_criterion, None, args)

    assert received["mask_loss_scale"] == 2.0
    assert received["consistency_loss_scale"] == 0.25
