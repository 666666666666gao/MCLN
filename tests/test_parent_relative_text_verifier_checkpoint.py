from types import SimpleNamespace

import pytest
import torch

from main_utils import load_checkpoint, prepare_source_moe_gate_checkpoint_config


class _TinyFPRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.structured_slot_builder = torch.nn.Linear(2, 2)
        self.sacr_head = torch.nn.Linear(2, 2)
        self.parent_relative_text_verifier = torch.nn.Linear(2, 2)


def _checkpoint_config():
    return {
        "use_parent_relative_text_verifier": True,
        "use_source_choice_selector": True,
        "use_source_moe": False,
        "butd_cls": True,
        "eval_use_selector_choice_scores": True,
        "source_choice_selector_sources": (
            "default,default_rank_blend_contrastive010"
        ),
        "source_choice_selector_default_source": "default",
        "source_choice_selector_hidden_dim": 288,
        "source_choice_selector_lr": 1.25e-4,
        "source_choice_selector_loss_weight": 0.5,
        "source_choice_selector_choice_target": (
            "precision_gain_default_sourcewise_focal_bce"
        ),
        "source_choice_selector_min_iou_gap": 0.03,
        "sacr_hidden_dim": 288,
        "sacr_max_pairs": 3,
        "sacr_top_m_targets": 32,
        "sacr_top_k_anchors": 16,
        "sacr_geo_dim": 16,
        "sacr_min_parse_confidence": 0.0,
        "parent_relative_text_verifier_top_k": 5,
        "parent_relative_text_verifier_max_candidates": 10,
        "parent_relative_text_verifier_hidden_dim": 256,
        "parent_relative_text_verifier_heads": 4,
        "parent_relative_text_verifier_dropout": 0.1,
        "parent_relative_text_verifier_max_parent_score_gap": 0.25,
        "parent_relative_text_verifier_promotion_margin": 1e-4,
        "parent_relative_text_verifier_min_parse_confidence": 0.5,
        "parent_relative_text_verifier_min_anchor_mass": 0.5,
        "parent_relative_text_verifier_promotion_epsilon": 1e-4,
        "parent_relative_text_verifier_detach_inputs": False,
    }


def _write_checkpoint(path, config=None, complete=True):
    state = {
        "module.structured_slot_builder.probe": torch.ones(1),
        "module.sacr_head.probe": torch.ones(1),
    }
    if complete:
        state["module.parent_relative_text_verifier.probe"] = torch.ones(1)
    torch.save({
        "config": _checkpoint_config() if config is None else config,
        "model": state,
    }, str(path))


def _eval_args(path):
    return SimpleNamespace(
        checkpoint_path=str(path),
        eval=True,
        use_parent_relative_text_verifier=True,
        use_source_choice_selector=False,
        use_source_moe=False,
        use_sacr_score_refiner=False,
        use_sacr_source=False,
        butd_cls=True,
    )


def _train_args(path):
    args = _eval_args(path)
    args.eval = False
    args.parent_relative_text_verifier_train_only = True
    return args


def test_eval_inherits_complete_checkpoint_architecture(tmp_path):
    checkpoint_path = tmp_path / "fpr_tv.pth"
    _write_checkpoint(checkpoint_path)

    args = prepare_source_moe_gate_checkpoint_config(
        _eval_args(checkpoint_path)
    )

    assert args.use_source_choice_selector is True
    assert args.use_source_moe is False
    assert args.eval_use_selector_choice_scores is True
    assert args.parent_relative_text_verifier_top_k == 5
    assert args.parent_relative_text_verifier_max_candidates == 10
    assert args.parent_relative_text_verifier_hidden_dim == 256
    assert args.parent_relative_text_verifier_detach_inputs is False


def test_eval_rejects_partial_verifier_state(tmp_path):
    checkpoint_path = tmp_path / "partial.pth"
    _write_checkpoint(checkpoint_path, complete=False)

    with pytest.raises(ValueError, match="complete trained verifier"):
        prepare_source_moe_gate_checkpoint_config(
            _eval_args(checkpoint_path)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("eval_use_selector_choice_scores", False),
        ("source_choice_selector_default_source", "wrong"),
        ("source_choice_selector_choice_target", "wrong"),
        ("source_choice_selector_min_iou_gap", 0.04),
    ],
)
def test_eval_rejects_non_v99_parent_config(tmp_path, field, value):
    checkpoint_path = tmp_path / "wrong_parent.pth"
    config = _checkpoint_config()
    config[field] = value
    _write_checkpoint(checkpoint_path, config=config)

    with pytest.raises(ValueError, match="V99 .* mismatch"):
        prepare_source_moe_gate_checkpoint_config(
            _eval_args(checkpoint_path)
        )


def test_train_only_inherits_exact_v99_parent_config(tmp_path):
    checkpoint_path = tmp_path / "v99_parent.pth"
    config = _checkpoint_config()
    config["use_parent_relative_text_verifier"] = False
    _write_checkpoint(checkpoint_path, config=config, complete=False)

    args = prepare_source_moe_gate_checkpoint_config(
        _train_args(checkpoint_path)
    )

    assert args.use_source_choice_selector is True
    assert args.eval_use_selector_choice_scores is True
    assert args.source_choice_selector_default_source == "default"
    assert args.source_choice_selector_choice_target == (
        "precision_gain_default_sourcewise_focal_bce"
    )
    assert args.source_choice_selector_min_iou_gap == 0.03


def test_train_only_rejects_non_v99_parent_config(tmp_path):
    checkpoint_path = tmp_path / "wrong_v99_parent.pth"
    config = _checkpoint_config()
    config["use_parent_relative_text_verifier"] = False
    config["source_choice_selector_sources"] = "default"
    _write_checkpoint(checkpoint_path, config=config, complete=False)

    with pytest.raises(ValueError, match="V99 .* mismatch"):
        prepare_source_moe_gate_checkpoint_config(
            _train_args(checkpoint_path)
        )


def test_train_and_eval_require_formal_butd_cls_filter(tmp_path):
    checkpoint_path = tmp_path / "missing_filter.pth"
    config = _checkpoint_config()
    config["butd_cls"] = False
    _write_checkpoint(checkpoint_path, config=config)
    with pytest.raises(ValueError, match="did not use butd_cls"):
        prepare_source_moe_gate_checkpoint_config(
            _eval_args(checkpoint_path)
        )

    config["butd_cls"] = True
    _write_checkpoint(checkpoint_path, config=config)
    runtime = _eval_args(checkpoint_path)
    runtime.butd_cls = False
    with pytest.raises(ValueError, match="runtime butd_cls"):
        prepare_source_moe_gate_checkpoint_config(runtime)


def _load_args(path, use_verifier):
    return SimpleNamespace(
        checkpoint_path=str(path),
        eval=True,
        reduce_lr=False,
        start_epoch=1,
        source_choice_selector_train_only=False,
        use_source_choice_selector=bool(use_verifier),
        use_parent_relative_text_verifier=bool(use_verifier),
        use_sacr_score_refiner=False,
        butd_cls=bool(use_verifier),
    )


def _optimizer_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10], gamma=0.1
    )
    return optimizer, scheduler


def test_eval_loads_exact_verifier_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "exact_fpr_tv.pth"
    source = _TinyFPRModel()
    torch.save({
        "config": _checkpoint_config(),
        "epoch": 1,
        "model": source.state_dict(),
    }, str(checkpoint_path))
    target = _TinyFPRModel()
    optimizer, scheduler = _optimizer_scheduler(target)

    load_checkpoint(
        _load_args(checkpoint_path, use_verifier=True),
        target,
        optimizer,
        scheduler,
    )

    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)


def test_eval_rejects_inexact_verifier_model_state(tmp_path):
    checkpoint_path = tmp_path / "inexact_fpr_tv.pth"
    source = _TinyFPRModel()
    state = source.state_dict()
    del state["parent_relative_text_verifier.bias"]
    torch.save({
        "config": _checkpoint_config(),
        "epoch": 1,
        "model": state,
    }, str(checkpoint_path))
    target = _TinyFPRModel()
    optimizer, scheduler = _optimizer_scheduler(target)

    with pytest.raises(ValueError, match="exact full model"):
        load_checkpoint(
            _load_args(checkpoint_path, use_verifier=True),
            target,
            optimizer,
            scheduler,
        )


@pytest.mark.parametrize("corruption", ["leading_shape", "dtype"])
def test_eval_rejects_verifier_tensor_shape_or_dtype_drift(
        tmp_path, corruption):
    checkpoint_path = tmp_path / "drifted_fpr_tv.pth"
    source = _TinyFPRModel()
    state = source.state_dict()
    key = "parent_relative_text_verifier.weight"
    if corruption == "leading_shape":
        state[key] = torch.cat([state[key], state[key][:1]], dim=0)
    else:
        state[key] = state[key].to(torch.float64)
    torch.save({
        "config": _checkpoint_config(),
        "epoch": 1,
        "model": state,
    }, str(checkpoint_path))
    target = _TinyFPRModel()
    optimizer, scheduler = _optimizer_scheduler(target)

    with pytest.raises(ValueError, match="shape/dtype"):
        load_checkpoint(
            _load_args(checkpoint_path, use_verifier=True),
            target,
            optimizer,
            scheduler,
        )


def test_default_off_loads_legacy_eval_checkpoint_unchanged(tmp_path):
    checkpoint_path = tmp_path / "legacy_eval.pth"
    source = torch.nn.Linear(2, 2)
    torch.save({
        "config": {},
        "epoch": 1,
        "model": source.state_dict(),
    }, str(checkpoint_path))
    target = torch.nn.Linear(2, 2)
    optimizer, scheduler = _optimizer_scheduler(target)

    load_checkpoint(
        _load_args(checkpoint_path, use_verifier=False),
        target,
        optimizer,
        scheduler,
    )

    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)
