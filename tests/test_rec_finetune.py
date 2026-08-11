import pytest
import torch

from models.losses import compute_hungarian_loss
from models.rec_candidate_adapter import (
    FEATURE_SCHEMA_VERSION,
    attach_candidate_targets as _attach_candidate_targets,
    build_rec_candidate_batch as _build_rec_candidate_batch,
)
from models.rec_geometry_reranker import (
    build_rec_geometry_model_inputs as _build_rec_geometry_model_inputs,
)
from models.rec_mask_geometry import (
    attach_rec_mask_geometry_targets as _attach_rec_mask_geometry_targets,
    build_rec_mask_geometry_candidates as _build_rec_mask_geometry_candidates,
)
from models.rec_reranker import QueryReranker
import models.rec_finetune as rec_finetune
from models.rec_finetune import (
    CALIBRATION_STEPS,
    MCLN_TRAINABLE_PREFIXES,
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_CALIBRATION_INTERVAL,
    PRODUCTION_MAX_STEPS,
    PRODUCTION_TRAIN_SAMPLE_COUNT,
    build_rec_finetune_optimizer,
    calibration_steps,
    clip_rec_finetune_gradients,
    configure_rec_finetune_trainability,
    natural_batch_count,
    set_rec_finetune_eval_mode,
    set_rec_finetune_train_mode,
)


class FakeSetCriterion:
    def __call__(self, _output, _target):
        return {
            "loss_ce": torch.tensor(2.0),
            "loss_bbox": torch.tensor(3.0),
            "loss_giou": torch.tensor(4.0),
            "loss_mask": torch.tensor(1.0),
            "loss_dice": torch.tensor(2.0),
            "sp_loss_mask": torch.tensor(3.0),
            "sp_loss_dice": torch.tensor(4.0),
            "corresponding_loss_mask": torch.tensor(5.0),
            "corresponding_loss_dice": torch.tensor(6.0),
            "adaptive_weight_loss_mask": torch.tensor(7.0),
            "adaptive_weight_loss_dice": torch.tensor(8.0),
        }, None


def _minimal_end_points():
    end_points = {
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "sem_cls_label": torch.zeros(1, 1, dtype=torch.long),
        "gt_masks": torch.zeros(1, 1, 1),
        "positive_map": torch.zeros(1, 1, 1),
        "modify_positive_map": torch.zeros(1, 1, 1),
        "pron_positive_map": torch.zeros(1, 1, 1),
        "other_entity_map": torch.zeros(1, 1, 1),
        "rel_positive_map": torch.zeros(1, 1, 1),
        "box_label_mask": torch.ones(1, 1, dtype=torch.bool),
        "auxi_entity_positive_map": torch.zeros(1, 1, 1),
        "auxi_box": torch.zeros(1, 6),
        "superpoints": torch.zeros(1, 1, dtype=torch.long),
        "language_dataset": ["nr3d"],
        "last_pred_masks": torch.zeros(1, 1, 1),
        "sp_last_pred_masks": torch.zeros(1, 1, 1),
        "adaptive_weights": torch.ones(1),
        "super_xyz_list": [],
    }
    for prefix in ("proposal_", "last_"):
        end_points[f"{prefix}center"] = torch.zeros(1, 1, 3)
        end_points[f"{prefix}pred_size"] = torch.ones(1, 1, 3)
        end_points[f"{prefix}sem_cls_scores"] = torch.zeros(1, 1, 1)
    return end_points


def test_hungarian_loss_scales_are_separate_and_default_compatible():
    loss, _ = compute_hungarian_loss(
        _minimal_end_points(),
        num_decoder_layers=1,
        set_criterion=FakeSetCriterion(),
        mask_loss_scale=0.25,
        consistency_loss_scale=3.0,
    )

    detector_loss = 21.0
    supervised_mask_loss = 2 * (10 * 1 + 2 * 2 + 5 * 3 + 4 + 10 * 7 + 2 * 8)
    consistency_loss = 2 * (10 * 5 + 2 * 6)
    expected = detector_loss + 0.25 * supervised_mask_loss + 3.0 * consistency_loss
    assert torch.equal(loss, torch.tensor(expected))

    default_loss, _ = compute_hungarian_loss(
        _minimal_end_points(),
        num_decoder_layers=1,
        set_criterion=FakeSetCriterion(),
    )
    previous_total = detector_loss + supervised_mask_loss + consistency_loss
    assert torch.equal(default_loss, torch.tensor(previous_total))


def test_hungarian_loss_normalizes_singleton_component_total_to_scalar():
    class SingletonComponentCriterion(FakeSetCriterion):
        def __init__(self):
            self.singleton = torch.tensor([6.0], requires_grad=True)

        def __call__(self, output, target):
            losses, indices = super().__call__(output, target)
            losses["corresponding_loss_dice"] = self.singleton
            return losses, indices

    criterion = SingletonComponentCriterion()
    loss, _ = compute_hungarian_loss(
        _minimal_end_points(),
        num_decoder_layers=1,
        set_criterion=criterion,
    )

    assert loss.ndim == 0
    assert torch.equal(loss, torch.tensor(383.0))
    loss.backward()
    assert torch.equal(criterion.singleton.grad, torch.tensor([4.0]))


def test_query_mask_fusion_only_loss_skips_full_criterion_and_backpropagates():
    class QueryFusionOnlyCriterion:
        def __init__(self):
            self.mask = torch.tensor(7.0, requires_grad=True)
            self.dice = torch.tensor(8.0, requires_grad=True)

        def __call__(self, _output, _target):
            raise AssertionError("full multi-task criterion must be skipped")

        def forward_query_mask_fusion(self, _output, _target):
            return {
                "adaptive_weight_loss_mask": self.mask,
                "adaptive_weight_loss_dice": self.dice,
            }, None

    criterion = QueryFusionOnlyCriterion()
    loss, end_points = compute_hungarian_loss(
        _minimal_end_points(),
        num_decoder_layers=1,
        set_criterion=criterion,
        mask_loss_scale=0.25,
        query_mask_fusion_train_only=True,
    )

    assert torch.equal(loss, torch.tensor(21.5))
    assert torch.equal(end_points["source_moe_rank_loss"], torch.tensor(0.0))
    loss.backward()
    assert torch.equal(criterion.mask.grad, torch.tensor(2.5))
    assert torch.equal(criterion.dice.grad, torch.tensor(0.5))


def test_build_rec_finetune_forward_is_public():
    assert callable(rec_finetune.build_rec_finetune_forward)


def _rec_forward_fixture():
    torch.manual_seed(41)
    batch_size = 1
    num_queries = 256
    num_tokens = 5
    projection_dim = 64
    num_superpoints = 6
    num_points = 12

    end_points = {
        "last_center": (
            torch.randn(batch_size, num_queries, 3) * 0.2
        ).requires_grad_(),
        "last_pred_size": (
            1.0 + torch.rand(batch_size, num_queries, 3) * 0.2
        ).requires_grad_(),
        "last_sem_cls_scores": torch.randn(
            batch_size, num_queries, num_tokens
        ),
        "last_proj_queries": torch.randn(
            batch_size, num_queries, projection_dim, requires_grad=True
        ),
        "proj_tokens": torch.randn(
            batch_size, num_tokens, projection_dim
        ),
        "last_pred_masks": [
            torch.randn(1, num_queries, num_superpoints)
        ],
        "sp_last_pred_masks": [
            torch.randn(num_queries, num_superpoints)
        ],
        "adaptive_weights": [torch.tensor(0.5)],
    }
    coordinates = torch.linspace(-1.0, 1.0, num_points).view(
        1, num_points, 1
    ).expand(batch_size, -1, 3).clone()
    inputs = {
        "point_clouds": torch.cat([
            coordinates,
            torch.zeros(batch_size, num_points, 3),
        ], dim=-1),
        "superpoints": (
            torch.arange(num_points) % num_superpoints
        ).view(batch_size, num_points),
    }
    for key in (
            "positive_map", "modify_positive_map", "pron_positive_map",
            "other_entity_map", "rel_positive_map"):
        inputs[key] = torch.zeros(batch_size, 1, num_tokens)
    inputs["positive_map"][..., 0] = 1.0

    candidate_batch = _build_rec_candidate_batch(
        end_points, inputs, topk_per_source=8, max_candidates=16
    )
    geometry_batch = _build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch
    )
    parent_dim = candidate_batch["features"].shape[-1]
    geometry_dim = (
        parent_dim + geometry_batch["geometry_features"].shape[-1] + 2
    )
    parent_artifact = {
        "adapter_schema_version": FEATURE_SCHEMA_VERSION,
        "candidate_rule": {"topk_per_source": 8, "max_candidates": 16},
        "input_dim": parent_dim,
        "feature_names": list(candidate_batch["feature_names"]),
        "feature_mean": torch.zeros(parent_dim),
        "feature_std": torch.ones(parent_dim),
        "score_mode": "rank_blend",
        "reranker_weight": 0.9,
    }
    variant_configs = [
        dict(value) for value in geometry_batch["variant_configs"]
    ]
    geometry_artifact = {
        "input_dim": geometry_dim,
        "feature_names": (
            list(candidate_batch["feature_names"])
            + list(geometry_batch["geometry_feature_names"])
            + ["parent_score", "parent_is_deployed_top1"]
        ),
        "feature_mean": torch.zeros(geometry_dim),
        "feature_std": torch.ones(geometry_dim),
        "variant_names": [value["name"] for value in variant_configs],
        "variant_configs": variant_configs,
        "regressed_variant_index": 0,
        "min_points": geometry_batch["min_points"],
        "max_point_fraction": geometry_batch["max_point_fraction"],
        "geometry_weight": 1.0,
    }
    root_box = candidate_batch["boxes"][:, 0].detach()
    other_box = candidate_batch["boxes"][:, 1].detach()
    targets = {
        "center_label": torch.stack([
            root_box[:, :3], other_box[:, :3]
        ], dim=1).squeeze(2).clone().requires_grad_(),
        "size_gts": torch.stack([
            root_box[:, 3:], other_box[:, 3:]
        ], dim=1).squeeze(2).clone().requires_grad_(),
        "box_label_mask": torch.ones(batch_size, 2, dtype=torch.bool),
    }
    return {
        "end_points": end_points,
        "inputs": inputs,
        "targets": targets,
        "candidate_batch": candidate_batch,
        "parent": QueryReranker(parent_dim, hidden_dim=16, dropout=0.0),
        "parent_artifact": parent_artifact,
        "geometry": QueryReranker(geometry_dim, hidden_dim=16, dropout=0.0),
        "geometry_artifact": geometry_artifact,
    }


def test_rec_finetune_forward_isolates_gt_detaches_targets_and_backpropagates(
        monkeypatch):
    fixture = _rec_forward_fixture()
    forbidden = {
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "candidate_ious", "geometry_ious", "threshold_labels",
    }
    calls = []

    def candidate_builder(end_points, inputs, **kwargs):
        assert forbidden.isdisjoint(end_points)
        assert forbidden.isdisjoint(inputs)
        calls.append("candidate")
        return _build_rec_candidate_batch(end_points, inputs, **kwargs)

    def attach_parent(candidate_batch, targets, root_only=False):
        assert calls == ["candidate", "geometry", "geometry_inputs"]
        assert root_only is True
        calls.append("parent_targets")
        return _attach_candidate_targets(
            candidate_batch, targets, root_only=root_only
        )

    def geometry_builder(end_points, inputs, candidate_batch,
                         variant_config=None):
        assert forbidden.isdisjoint(end_points)
        assert forbidden.isdisjoint(inputs)
        assert forbidden.isdisjoint(candidate_batch)
        calls.append("geometry")
        return _build_rec_mask_geometry_candidates(
            end_points,
            inputs,
            candidate_batch,
            variant_config=variant_config,
        )

    def geometry_inputs(*args, **kwargs):
        calls.append("geometry_inputs")
        return _build_rec_geometry_model_inputs(*args, **kwargs)

    def attach_geometry(geometry_batch, targets, root_only=True):
        assert calls == [
            "candidate", "geometry", "geometry_inputs", "parent_targets",
        ]
        assert root_only is True
        calls.append("geometry_targets")
        return _attach_rec_mask_geometry_targets(
            geometry_batch, targets, root_only=root_only
        )

    monkeypatch.setattr(
        rec_finetune, "build_rec_candidate_batch", candidate_builder,
        raising=False,
    )
    monkeypatch.setattr(
        rec_finetune, "attach_candidate_targets", attach_parent,
        raising=False,
    )
    monkeypatch.setattr(
        rec_finetune, "build_rec_mask_geometry_candidates", geometry_builder,
        raising=False,
    )
    monkeypatch.setattr(
        rec_finetune, "build_rec_geometry_model_inputs", geometry_inputs,
        raising=False,
    )
    monkeypatch.setattr(
        rec_finetune, "attach_rec_mask_geometry_targets", attach_geometry,
        raising=False,
    )

    state = rec_finetune.build_rec_finetune_forward(
        fixture["end_points"],
        fixture["inputs"],
        fixture["targets"],
        fixture["parent"],
        fixture["parent_artifact"],
        fixture["geometry"],
        fixture["geometry_artifact"],
    )

    assert calls == [
        "candidate", "geometry", "geometry_inputs", "parent_targets",
        "geometry_targets",
    ]
    assert fixture["parent"].training is True
    assert fixture["geometry"].training is True
    assert state["parent_candidate_ious"].requires_grad is False
    assert state["geometry_candidate_ious"].requires_grad is False
    assert forbidden.isdisjoint(state["parent_model_inputs"])
    assert forbidden.isdisjoint(state["geometry_model_inputs"])
    assert forbidden.isdisjoint(state["geometry_batch"])
    assert torch.isfinite(state["parent_model_inputs"]["features"]).all()
    assert torch.isfinite(state["geometry_model_inputs"]["features"]).all()
    geometry_valid = state["geometry_model_inputs"]["valid_mask"]
    assert torch.equal(
        state["geometry_model_inputs"]["features"][~geometry_valid],
        torch.zeros_like(
            state["geometry_model_inputs"]["features"][~geometry_valid]
        ),
    )
    assert state["parent_state"]["compact_scores"].requires_grad is False
    assert state["runtime_outputs"][
        "rec_geometry_scores"
    ].requires_grad is False

    (state["parent_loss"] + state["geometry_loss"]).backward()

    for model in (fixture["parent"], fixture["geometry"]):
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert all(torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
    for key in ("last_center", "last_pred_size", "last_proj_queries"):
        gradient = fixture["end_points"][key].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert fixture["targets"]["center_label"].grad is None
    assert fixture["targets"]["size_gts"].grad is None
    assert fixture["end_points"]["proj_tokens"].grad is None


def test_rec_finetune_forward_routes_geometry_to_tier_pairwise(monkeypatch):
    fixture = _rec_forward_fixture()
    original = rec_finetune.compute_rec_reranker_loss
    observed_alphas = []

    def record_alpha(*args, **kwargs):
        observed_alphas.append(kwargs.get("tier_pairwise_alpha"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rec_finetune, "compute_rec_reranker_loss", record_alpha
    )

    state = rec_finetune.build_rec_finetune_forward(
        fixture["end_points"],
        fixture["inputs"],
        fixture["targets"],
        fixture["parent"],
        fixture["parent_artifact"],
        fixture["geometry"],
        fixture["geometry_artifact"],
    )

    assert observed_alphas == [0.0, 1.0]
    assert torch.equal(
        state["parent_loss_stats"]["loss_ranking"],
        state["parent_loss_stats"]["loss_listwise"],
    )
    assert torch.equal(
        state["geometry_loss_stats"]["loss_ranking"],
        state["geometry_loss_stats"]["loss_best_tier_pairwise"],
    )


def test_ranking_objective_contract_is_defensive_and_cannot_change_forward(
        monkeypatch):
    expected = {
        "parent": {
            "name": "single-best-iou-listwise-v1",
            "tier_pairwise_alpha": 0.0,
        },
        "geometry": {
            "name": "best-tier-pairwise-v1",
            "tier_pairwise_alpha": 1.0,
            "thresholds": [0.25, 0.50],
            "threshold_operator": "strict_gt",
            "positive_policy": "all_valid_candidates_in_best_tier",
            "negative_policy": "all_valid_candidates_below_best_tier",
            "loss": "softplus(negative_logit-positive_logit)",
            "pair_reduction": "mean_within_row",
            "row_reduction": "mean_over_informative_rows",
            "no_pair_policy": "differentiable_zero",
        },
    }
    first = rec_finetune.rec_finetune_ranking_objective_contract()
    assert first == expected
    first["geometry"]["tier_pairwise_alpha"] = 0.0
    assert rec_finetune.rec_finetune_ranking_objective_contract() == expected

    monkeypatch.setattr(
        rec_finetune,
        "rec_finetune_ranking_objective_contract",
        lambda: {
            "parent": {"tier_pairwise_alpha": 1.0},
            "geometry": {"tier_pairwise_alpha": 0.0},
        },
    )
    fixture = _rec_forward_fixture()
    original = rec_finetune.compute_rec_reranker_loss
    observed_alphas = []

    def record_alpha(*args, **kwargs):
        observed_alphas.append(kwargs.get("tier_pairwise_alpha"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rec_finetune, "compute_rec_reranker_loss", record_alpha
    )
    rec_finetune.build_rec_finetune_forward(
        fixture["end_points"],
        fixture["inputs"],
        fixture["targets"],
        fixture["parent"],
        fixture["parent_artifact"],
        fixture["geometry"],
        fixture["geometry_artifact"],
    )

    assert observed_alphas == [0.0, 1.0]


def test_rec_finetune_forward_eval_matches_existing_runtime_exactly():
    fixture = _rec_forward_fixture()
    fixture["parent"].eval()
    fixture["geometry"].eval()

    state = rec_finetune.build_rec_finetune_forward(
        fixture["end_points"],
        fixture["inputs"],
        fixture["targets"],
        fixture["parent"],
        fixture["parent_artifact"],
        fixture["geometry"],
        fixture["geometry_artifact"],
    )

    # Runtime wrappers freeze their models, so they intentionally run last.
    from train_dist_mod import (
        build_rec_geometry_runtime_outputs,
        build_rec_reranker_outputs,
    )

    parent_outputs = build_rec_reranker_outputs(
        fixture["end_points"],
        fixture["inputs"],
        fixture["parent"],
        fixture["parent_artifact"],
    )
    runtime_outputs = build_rec_geometry_runtime_outputs(
        fixture["end_points"],
        fixture["inputs"],
        parent_outputs,
        fixture["geometry"],
        fixture["geometry_artifact"],
    )

    assert torch.equal(
        state["parent_state"]["query_scores"],
        parent_outputs["query_scores"],
    )
    assert set(state["runtime_outputs"]) == set(runtime_outputs)
    for key, expected in runtime_outputs.items():
        actual = state["runtime_outputs"][key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected), key
        else:
            assert actual == expected


@pytest.mark.parametrize(
    ("scale_name", "bad_scale"),
    [
        ("mask_loss_scale", -0.1),
        ("mask_loss_scale", float("nan")),
        ("mask_loss_scale", float("inf")),
        ("consistency_loss_scale", -0.1),
        ("consistency_loss_scale", float("nan")),
        ("consistency_loss_scale", float("inf")),
    ],
)
def test_hungarian_loss_rejects_invalid_scales(scale_name, bad_scale):
    with pytest.raises(ValueError, match=scale_name):
        compute_hungarian_loss(
            _minimal_end_points(),
            num_decoder_layers=1,
            set_criterion=FakeSetCriterion(),
            **{scale_name: bad_scale},
        )


class FakeMCLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.Dropout(p=0.5),
        )
        self.decoder_query_proj = torch.nn.Linear(3, 3)
        self.proposal_head = torch.nn.Linear(3, 2)
        self.prediction_heads = torch.nn.ModuleList([
            torch.nn.Linear(3, 2),
        ])
        self.backbone_net = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.BatchNorm1d(3),
            torch.nn.Dropout(p=0.5),
        )
        self.text_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.Dropout(p=0.5),
        )
        self.x_mask = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.BatchNorm1d(3),
        )


def _fake_models():
    return (
        FakeMCLN(),
        torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Dropout(p=0.2)),
        torch.nn.Sequential(torch.nn.Linear(4, 1), torch.nn.Dropout(p=0.3)),
    )


def _parameter_ids(parameters):
    return tuple(id(parameter) for parameter in parameters)


def test_rec_finetune_constants_are_the_approved_production_contract():
    assert MCLN_TRAINABLE_PREFIXES == (
        "decoder.",
        "decoder_query_proj.",
        "proposal_head.",
        "prediction_heads.",
    )
    assert PRODUCTION_TRAIN_SAMPLE_COUNT == 33040
    assert PRODUCTION_BATCH_SIZE == 18
    assert PRODUCTION_MAX_STEPS == 1836
    assert PRODUCTION_CALIBRATION_INTERVAL == 306
    assert CALIBRATION_STEPS == (0, 306, 612, 918, 1224, 1530, 1836)


def test_configure_rec_finetune_trainability_returns_exact_disjoint_groups():
    mcln, parent, geometry = _fake_models()
    expected_mcln_names = (
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder_query_proj.weight",
        "decoder_query_proj.bias",
        "proposal_head.weight",
        "proposal_head.bias",
        "prediction_heads.0.weight",
        "prediction_heads.0.bias",
    )

    groups = configure_rec_finetune_trainability(mcln, parent, geometry)

    assert tuple(groups) == (
        "mcln_names",
        "mcln_parameters",
        "parent_names",
        "parent_parameters",
        "geometry_names",
        "geometry_parameters",
    )
    assert groups["mcln_names"] == expected_mcln_names
    assert groups["parent_names"] == ("0.weight", "0.bias")
    assert groups["geometry_names"] == ("0.weight", "0.bias")

    mcln_by_name = dict(mcln.named_parameters())
    parent_by_name = dict(parent.named_parameters())
    geometry_by_name = dict(geometry.named_parameters())
    assert _parameter_ids(groups["mcln_parameters"]) == tuple(
        id(mcln_by_name[name]) for name in expected_mcln_names
    )
    assert _parameter_ids(groups["parent_parameters"]) == tuple(
        id(parameter) for parameter in parent_by_name.values()
    )
    assert _parameter_ids(groups["geometry_parameters"]) == tuple(
        id(parameter) for parameter in geometry_by_name.values()
    )

    assert all(parameter.requires_grad
               for parameter in groups["mcln_parameters"])
    assert all(parameter.requires_grad for parameter in parent.parameters())
    assert all(parameter.requires_grad for parameter in geometry.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in mcln.named_parameters()
        if name not in expected_mcln_names
    )
    identity_sets = [
        set(_parameter_ids(groups[key]))
        for key in (
            "mcln_parameters",
            "parent_parameters",
            "geometry_parameters",
        )
    ]
    assert all(identity_sets)
    assert identity_sets[0].isdisjoint(identity_sets[1])
    assert identity_sets[0].isdisjoint(identity_sets[2])
    assert identity_sets[1].isdisjoint(identity_sets[2])


def test_configure_rec_finetune_trainability_rejects_a_missing_prefix_closed():
    mcln, parent, geometry = _fake_models()
    mcln.prediction_heads = torch.nn.ModuleList([torch.nn.Dropout(p=0.5)])

    with pytest.raises(ValueError, match="prediction_heads"):
        configure_rec_finetune_trainability(mcln, parent, geometry)

    assert all(not parameter.requires_grad for model in (mcln, parent, geometry)
               for parameter in model.parameters())


@pytest.mark.parametrize("empty_group", ["parent", "geometry"])
def test_configure_rec_finetune_trainability_rejects_empty_groups_closed(
        empty_group):
    mcln, parent, geometry = _fake_models()
    if empty_group == "parent":
        parent = torch.nn.Module()
    else:
        geometry = torch.nn.Module()

    with pytest.raises(ValueError, match="{}.*non-empty".format(empty_group)):
        configure_rec_finetune_trainability(mcln, parent, geometry)

    assert all(not parameter.requires_grad for model in (mcln, parent, geometry)
               for parameter in model.parameters())


def test_configure_rec_finetune_trainability_rejects_overlap_closed():
    mcln, _parent, geometry = _fake_models()
    parent = torch.nn.Module()
    parent.register_parameter("shared", mcln.decoder[0].weight)

    with pytest.raises(ValueError, match="overlap"):
        configure_rec_finetune_trainability(mcln, parent, geometry)

    assert all(not parameter.requires_grad for model in (mcln, parent, geometry)
               for parameter in model.parameters())


def test_configure_rec_finetune_rejects_cross_boundary_parameter_alias_closed():
    mcln, parent, geometry = _fake_models()
    mcln.backbone_net.register_parameter(
        "shared_decoder_weight", mcln.decoder[0].weight
    )

    with pytest.raises(ValueError, match="parameter.*allowlist boundary"):
        configure_rec_finetune_trainability(mcln, parent, geometry)

    assert all(not parameter.requires_grad for model in (mcln, parent, geometry)
               for parameter in model.parameters())


def test_rec_finetune_train_mode_rejects_cross_boundary_module_alias_closed():
    mcln, parent, geometry = _fake_models()
    mcln.frozen_alias = mcln.decoder

    with pytest.raises(AssertionError, match="module.*train/eval boundary"):
        set_rec_finetune_train_mode(mcln, parent, geometry)

    assert all(not parameter.requires_grad for model in (mcln, parent, geometry)
               for parameter in model.parameters())
    assert all(module.training is False for model in (mcln, parent, geometry)
               for module in model.modules())


def test_rec_finetune_modes_keep_frozen_mcln_dropout_and_batchnorm_in_eval():
    mcln, parent, geometry = _fake_models()
    configure_rec_finetune_trainability(mcln, parent, geometry)

    set_rec_finetune_train_mode(mcln, parent, geometry)

    assert mcln.training is False
    assert mcln.decoder.training is True
    assert mcln.decoder[1].training is True
    assert mcln.decoder_query_proj.training is True
    assert mcln.proposal_head.training is True
    assert mcln.prediction_heads.training is True
    assert mcln.prediction_heads[0].training is True
    assert mcln.backbone_net.training is False
    assert mcln.backbone_net[1].training is False
    assert mcln.backbone_net[2].training is False
    assert mcln.text_encoder.training is False
    assert mcln.text_encoder[1].training is False
    assert mcln.x_mask.training is False
    assert mcln.x_mask[1].training is False
    assert parent.training is True
    assert geometry.training is True

    set_rec_finetune_eval_mode(mcln, parent, geometry)

    assert all(module.training is False for model in (mcln, parent, geometry)
               for module in model.modules())


def test_build_rec_finetune_optimizer_is_fresh_with_exact_named_groups():
    mcln, parent, geometry = _fake_models()
    groups = configure_rec_finetune_trainability(mcln, parent, geometry)

    optimizer = build_rec_finetune_optimizer(groups)
    fresh_optimizer = build_rec_finetune_optimizer(groups)

    assert type(optimizer) is torch.optim.AdamW
    assert optimizer is not fresh_optimizer
    assert len(optimizer.state) == 0
    assert len(fresh_optimizer.state) == 0
    assert [
        (group["name"], group["lr"], group["weight_decay"])
        for group in optimizer.param_groups
    ] == [
        ("mcln_decoder_box", 2e-5, 5e-4),
        ("parent_reranker", 1e-3, 1e-4),
        ("geometry_reranker", 3e-4, 1e-4),
    ]
    assert [
        _parameter_ids(group["params"])
        for group in optimizer.param_groups
    ] == [
        _parameter_ids(groups["mcln_parameters"]),
        _parameter_ids(groups["parent_parameters"]),
        _parameter_ids(groups["geometry_parameters"]),
    ]


def test_clip_rec_finetune_gradients_uses_each_exact_group_and_clip(monkeypatch):
    mcln, parent, geometry = _fake_models()
    groups = configure_rec_finetune_trainability(mcln, parent, geometry)
    calls = []
    returned_norms = iter((0.25, 0.5, 0.75))

    def fake_clip(parameters, max_norm):
        calls.append((_parameter_ids(parameters), max_norm))
        return torch.tensor(next(returned_norms))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    diagnostics = clip_rec_finetune_gradients(groups)

    assert calls == [
        (_parameter_ids(groups["mcln_parameters"]), 0.1),
        (_parameter_ids(groups["parent_parameters"]), 1.0),
        (_parameter_ids(groups["geometry_parameters"]), 1.0),
    ]
    assert diagnostics == {
        "mcln_decoder_box": 0.25,
        "parent_reranker": 0.5,
        "geometry_reranker": 0.75,
    }


def test_clip_rec_finetune_gradients_rejects_nonfinite_norm(monkeypatch):
    mcln, parent, geometry = _fake_models()
    groups = configure_rec_finetune_trainability(mcln, parent, geometry)
    monkeypatch.setattr(
        rec_finetune.torch.nn.utils,
        "clip_grad_norm_",
        lambda _parameters, _max_norm: torch.tensor(float("inf")),
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        clip_rec_finetune_gradients(groups)


def test_natural_batch_count_and_calibration_steps_include_final_remainders():
    assert natural_batch_count(33040, 18) == 1836
    assert natural_batch_count(19, 18) == 2
    assert calibration_steps(1836, 306) == CALIBRATION_STEPS
    assert calibration_steps(10, 4) == (0, 4, 8, 10)
    assert calibration_steps(3, 5) == (0, 3)


@pytest.mark.parametrize(
    ("sample_count", "batch_size"),
    [
        (0, 1),
        (-1, 1),
        (1, 0),
        (1, -1),
        (True, 1),
        (1, False),
        (1.0, 1),
        (1, "1"),
    ],
)
def test_natural_batch_count_rejects_non_positive_strict_integers(
        sample_count, batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        natural_batch_count(sample_count, batch_size)


@pytest.mark.parametrize(
    ("max_steps", "interval"),
    [
        (0, 1),
        (-1, 1),
        (1, 0),
        (1, -1),
        (True, 1),
        (1, False),
        (1.0, 1),
        (1, "1"),
    ],
)
def test_calibration_steps_rejects_non_positive_strict_integers(
        max_steps, interval):
    with pytest.raises(ValueError, match="positive integer"):
        calibration_steps(max_steps, interval)


def _calibration_transition_state(indices=(7, 2, 19),
                                  selected=(0.25, 0.50, 0.75),
                                  oracle=(0.30, 0.60, 0.80)):
    return rec_finetune.CalibrationDiagnosticsTransitionState(
        expected_indices=indices,
        selected_ious=selected,
        geometry_oracle_ious=oracle,
    )


def test_calibration_selected_output_sha256_is_public_and_canonical():
    assert callable(rec_finetune.calibration_selected_output_sha256)
    state = _calibration_transition_state()

    digest = rec_finetune.calibration_selected_output_sha256(state)

    assert digest == (
        "e820cf3552a1d59de2ea8120afa2dd922ab257d902d6cdca13773b53ce2b5718"
    )


def test_calibration_selected_output_sha256_ignores_oracle_only_changes():
    first = _calibration_transition_state(
        oracle=(0.30, 0.60, 0.80)
    )
    second = _calibration_transition_state(
        oracle=(0.90, 0.95, 1.00)
    )

    assert rec_finetune.calibration_selected_output_sha256(first) == (
        rec_finetune.calibration_selected_output_sha256(second)
    )


@pytest.mark.parametrize(
    "changed",
    [
        _calibration_transition_state(indices=(7, 2, 20)),
        _calibration_transition_state(indices=(2, 7, 19)),
        _calibration_transition_state(
            indices=(19, 7, 2),
            selected=(0.75, 0.25, 0.50),
            oracle=(0.80, 0.30, 0.60),
        ),
        _calibration_transition_state(
            selected=(0.50, 0.25, 0.75),
            oracle=(0.80, 0.80, 0.80),
        ),
        _calibration_transition_state(
            selected=(0.25, float.fromhex("0x1.0000000000001p-1"), 0.75),
            oracle=(0.30, 0.60, 0.80),
        ),
    ],
    ids=(
        "dataset-index", "index-order", "ordered-rows",
        "selected-order", "selected-iou-one-ulp",
    ),
)
def test_calibration_selected_output_sha256_binds_each_ordered_row(changed):
    baseline = _calibration_transition_state()

    assert rec_finetune.calibration_selected_output_sha256(changed) != (
        rec_finetune.calibration_selected_output_sha256(baseline)
    )


@pytest.mark.parametrize(
    "invalid",
    [
        object(),
        _calibration_transition_state(indices=(7, 7, 19)),
        _calibration_transition_state(selected=(0.25, 0.50)),
        _calibration_transition_state(selected=(0.25, float("nan"), 0.75)),
        _calibration_transition_state(selected=(0.25, float("inf"), 0.75)),
        _calibration_transition_state(
            oracle=(0.30, float("-inf"), 0.80)
        ),
        _calibration_transition_state(
            selected=(0.25, 0.50, 0.75),
            oracle=(0.10, 0.60, 0.80),
        ),
    ],
    ids=(
        "wrong-type", "duplicate-index", "length-mismatch",
        "nan-selected", "infinite-selected", "infinite-oracle",
        "oracle-below-selected",
    ),
)
def test_calibration_selected_output_sha256_rejects_invalid_state(invalid):
    with pytest.raises(ValueError):
        rec_finetune.calibration_selected_output_sha256(invalid)
