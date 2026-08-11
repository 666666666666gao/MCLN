import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from main_utils import parse_option
from models.rec_geometry_reranker import _stable_masked_rank_normalize
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    build_hierarchical_feature_names,
)
from train_dist_mod import (
    TrainTester,
    build_rec_geometry_runtime_outputs,
    load_rec_hierarchical_runtime_artifact,
    validate_rec_hierarchical_runtime_provenance,
)
from test_rec_geometry_runtime import (
    RecordingGeometryReranker,
    _geometry_artifact,
    _geometry_batch,
    _parent_outputs,
)
from test_train_scanrefer_rec_hierarchical_reranker import (
    _staged_hierarchical_artifact,
)


class RecordingHierarchy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(
            self, query_features, variant_features, query_aux_continuous,
            query_aux_binary, variant_aux_continuous, variant_aux_binary,
            query_valid, variant_valid):
        self.calls.append({
            "query_features": query_features.detach().cpu().clone(),
            "variant_features": variant_features.detach().cpu().clone(),
            "query_aux_continuous": (
                query_aux_continuous.detach().cpu().clone()
            ),
            "query_aux_binary": query_aux_binary.detach().cpu().clone(),
            "variant_aux_continuous": (
                variant_aux_continuous.detach().cpu().clone()
            ),
            "variant_aux_binary": variant_aux_binary.detach().cpu().clone(),
            "query_valid": query_valid.detach().cpu().clone(),
            "variant_valid": variant_valid.detach().cpu().clone(),
            "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": tuple(
                parameter.requires_grad for parameter in self.parameters()
            ),
        })
        batch_size = query_features.shape[0]
        query_logits = query_features.new_full((batch_size, 16, 2), -10.0)
        variant_logits = variant_features.new_full(
            (batch_size, 16, 7, 2), -10.0
        )
        query_logits[:, 0] = 10.0
        variant_logits[:, 0, 0] = 10.0
        return {
            "query_logits": query_logits + self.anchor * 0.0,
            "variant_logits": variant_logits + self.anchor * 0.0,
            "query_embedding": query_features.new_zeros(
                batch_size, 16, 1
            ),
            "variant_embedding": variant_features.new_zeros(
                batch_size, 16, 7, 1
            ),
        }


def _hierarchical_runtime_artifact(geometry_artifact, margin=0.1):
    staged = _staged_hierarchical_artifact()[3]
    return {
        "feature_names": build_hierarchical_feature_names(
            geometry_artifact["feature_names"]
        ),
        "normalization": copy.deepcopy(staged["normalization"]),
        "selection": {"margin": float(margin)},
    }


def _hierarchical_parent_outputs():
    parent_outputs = _parent_outputs(batch_size=1)
    candidate_batch = parent_outputs["candidate_batch"]
    candidate_batch["default_scores"] = torch.linspace(
        0.9, 0.1, 16
    ).unsqueeze(0)
    candidate_batch["default_top1_query_index"] = candidate_batch[
        "query_indices"
    ][:, 0].clone()
    return parent_outputs


def _denormalize(call, artifact, field):
    group = artifact["normalization"]["groups"][field]
    mean = group["mean"].reshape(
        *((1,) * (call[field].dim() - 1)), -1
    )
    std = group["std"].reshape(
        *((1,) * (call[field].dim() - 1)), -1
    )
    return call[field] * std + mean


def test_main_parser_exposes_hierarchical_checkpoint_and_enable_flag(
        monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_dist_mod.py",
        "--rec_reranker_checkpoint", "parent.pth",
        "--eval_use_rec_reranker_scores",
        "--rec_geometry_reranker_checkpoint", "geometry.pth",
        "--eval_use_rec_geometry_reranker_scores",
        "--rec_hierarchical_reranker_checkpoint", "hierarchical.pth",
        "--eval_use_rec_hierarchical_reranker_scores",
    ])

    args = parse_option()

    assert args.rec_hierarchical_reranker_checkpoint == "hierarchical.pth"
    assert args.eval_use_rec_hierarchical_reranker_scores is True


def test_disabled_hierarchical_path_is_bitwise_identical(monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    artifact = _geometry_artifact(parent_outputs, weight=1.0)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )

    implicit = build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, RecordingGeometryReranker(), artifact
    )
    explicit = build_rec_geometry_runtime_outputs(
        {},
        {},
        parent_outputs,
        RecordingGeometryReranker(),
        artifact,
        hierarchical_model=None,
        hierarchical_artifact=None,
    )

    assert set(implicit) == set(explicit)
    for key in implicit:
        if isinstance(implicit[key], torch.Tensor):
            assert torch.equal(implicit[key], explicit[key])
        else:
            assert implicit[key] == explicit[key]


def test_enabled_hierarchy_rebuilds_exact_inputs_and_only_promotes_scores(
        monkeypatch):
    parent_outputs = _hierarchical_parent_outputs()
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_artifact = _geometry_artifact(parent_outputs, weight=1.0)
    hierarchical_artifact = _hierarchical_runtime_artifact(
        geometry_artifact
    )
    forbidden = {
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "candidate_ious", "geometry_ious", "threshold_labels",
    }
    end_points = {name: object() for name in forbidden}
    inputs = {"target_mask": object(), "center_label": object()}
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )
    baseline = build_rec_geometry_runtime_outputs(
        end_points,
        inputs,
        parent_outputs,
        RecordingGeometryReranker(),
        geometry_artifact,
    )
    hierarchy = RecordingHierarchy().train()

    enabled = build_rec_geometry_runtime_outputs(
        end_points,
        inputs,
        parent_outputs,
        RecordingGeometryReranker(),
        geometry_artifact,
        hierarchical_model=hierarchy,
        hierarchical_artifact=hierarchical_artifact,
    )

    assert set(enabled) == set(baseline) == {
        "rec_reranker_scores",
        "rec_geometry_runtime_mode",
        "rec_geometry_boxes",
        "rec_geometry_scores",
        "rec_geometry_valid_mask",
        "rec_geometry_fallback_index",
    }
    for key in set(enabled) - {"rec_geometry_scores"}:
        if isinstance(enabled[key], torch.Tensor):
            assert torch.equal(enabled[key], baseline[key])
        else:
            assert enabled[key] == baseline[key]
    assert baseline["rec_geometry_scores"].argmax(dim=1).item() == 111
    assert enabled["rec_geometry_scores"].argmax(dim=1).item() == 0
    promoted = torch.nextafter(
        baseline["rec_geometry_scores"].max(), torch.tensor(float("inf"))
    )
    assert enabled["rec_geometry_scores"][0, 0] == promoted
    assert torch.equal(
        enabled["rec_geometry_scores"][0, 1:],
        baseline["rec_geometry_scores"][0, 1:],
    )
    assert forbidden.isdisjoint(enabled)

    assert len(hierarchy.calls) == 1
    call = hierarchy.calls[0]
    assert call["training"] is False
    assert call["grad_enabled"] is False
    assert call["requires_grad"] == (False,)
    assert call["query_features"].shape == (1, 16, 152)
    assert call["variant_features"].shape == (1, 16, 7, 25)
    assert call["query_aux_continuous"].shape == (1, 16, 4)
    assert call["query_aux_binary"].shape == (1, 16, 2)
    assert call["variant_aux_continuous"].shape == (1, 16, 7, 2)
    assert call["variant_aux_binary"].shape == (1, 16, 7, 2)
    assert call["query_valid"].all()
    assert call["variant_valid"].all()

    raw_query = _denormalize(
        call, hierarchical_artifact, "query_features"
    )
    raw_variant = _denormalize(
        call, hierarchical_artifact, "variant_features"
    )
    raw_query_aux = _denormalize(
        call, hierarchical_artifact, "query_aux_continuous"
    )
    raw_variant_aux = _denormalize(
        call, hierarchical_artifact, "variant_aux_continuous"
    )
    candidate_batch = parent_outputs["candidate_batch"]
    assert torch.allclose(
        raw_query, candidate_batch["features"], atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        raw_variant,
        geometry_batch["geometry_features"],
        atol=5e-6,
        rtol=1e-5,
    )
    default_rank = _stable_masked_rank_normalize(
        candidate_batch["default_scores"], candidate_batch["valid_mask"]
    )
    parent_rank = _stable_masked_rank_normalize(
        parent_outputs["compact_scores"], candidate_batch["valid_mask"]
    )
    expected_query_aux = torch.stack((
        candidate_batch["default_scores"],
        default_rank,
        parent_outputs["compact_scores"],
        parent_rank,
    ), dim=-1)
    assert torch.allclose(
        raw_query_aux, expected_query_aux, atol=1e-6, rtol=1e-5
    )
    ranking = torch.arange(112, dtype=torch.float32).reshape(1, 16, 7)
    geometry_rank = _stable_masked_rank_normalize(
        ranking.reshape(1, 112),
        geometry_batch["valid_mask"].reshape(1, 112),
    ).reshape(1, 16, 7)
    assert torch.allclose(
        raw_variant_aux[..., 0], ranking, atol=5e-6, rtol=1e-5
    )
    assert torch.allclose(
        raw_variant_aux[..., 1], geometry_rank, atol=5e-6, rtol=1e-5
    )
    assert call["query_aux_binary"][0, 0, 0]
    assert call["query_aux_binary"][0, 1, 1]
    assert call["variant_aux_binary"][0, 15, 6].tolist() == [True, True]


@pytest.mark.parametrize("partial", ("model", "artifact"))
def test_runtime_builder_rejects_partial_hierarchical_context(
        partial, monkeypatch):
    parent_outputs = _hierarchical_parent_outputs()
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_artifact = _geometry_artifact(parent_outputs, weight=1.0)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )

    with pytest.raises(ValueError, match="partial hierarchy"):
        build_rec_geometry_runtime_outputs(
            {},
            {},
            parent_outputs,
            RecordingGeometryReranker(),
            geometry_artifact,
            hierarchical_model=(
                RecordingHierarchy() if partial == "model" else None
            ),
            hierarchical_artifact=(
                _hierarchical_runtime_artifact(geometry_artifact)
                if partial == "artifact" else None
            ),
        )


def test_runtime_builder_rejects_residual_and_hierarchy_together(monkeypatch):
    parent_outputs = _hierarchical_parent_outputs()
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_artifact = _geometry_artifact(parent_outputs, weight=1.0)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_rec_geometry_runtime_outputs(
            {},
            {},
            parent_outputs,
            RecordingGeometryReranker(),
            geometry_artifact,
            residual_model=torch.nn.Linear(1, 1),
            residual_artifact={},
            hierarchical_model=RecordingHierarchy(),
            hierarchical_artifact=_hierarchical_runtime_artifact(
                geometry_artifact
            ),
        )


def _deployed_hierarchical_stack():
    _records, model, geometry_names, artifact = \
        _staged_hierarchical_artifact()
    artifact = copy.deepcopy(artifact)
    artifact["deployable"] = True
    model._artifact_sha256 = "c" * 64
    parent = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    geometry = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    parent._artifact_sha256 = artifact["input_sha256"]["parent"]
    geometry._artifact_sha256 = artifact["input_sha256"]["geometry"]
    geometry_artifact = {
        "feature_names": geometry_names,
        "checkpoint_sha256": artifact["input_sha256"]["backbone"],
    }
    return parent, geometry, geometry_artifact, model, artifact


def test_hierarchical_runtime_provenance_binds_live_frozen_stack():
    parent, geometry, geometry_artifact, model, artifact = \
        _deployed_hierarchical_stack()

    projection = validate_rec_hierarchical_runtime_provenance(
        parent,
        geometry,
        geometry_artifact,
        model,
        artifact,
        torch.device("cpu"),
    )

    assert projection == {
        "parent_sha256": artifact["input_sha256"]["parent"],
        "geometry_sha256": artifact["input_sha256"]["geometry"],
        "hierarchical_sha256": "c" * 64,
        "margin": artifact["selection"]["margin"],
        "normalization_sha256": artifact["normalization_sha256"],
        "scene_fold_sha256": artifact["scene_fold_sha256"],
        "oof_record_sha256": artifact["oof_record_sha256"],
    }

    changed = copy.deepcopy(artifact)
    changed["input_sha256"]["parent"] = "d" * 64
    with pytest.raises(ValueError, match="parent"):
        validate_rec_hierarchical_runtime_provenance(
            parent,
            geometry,
            geometry_artifact,
            model,
            changed,
            torch.device("cpu"),
        )


def test_runtime_loader_passes_live_parent_geometry_and_deployable_contract(
        monkeypatch):
    parent, geometry, geometry_artifact, model, artifact = \
        _deployed_hierarchical_stack()
    calls = []

    def load(path, device, expected_geometry_feature_names,
             expected_deployable, parent_sha256, geometry_sha256):
        calls.append({
            "path": path,
            "device": device,
            "geometry_names": expected_geometry_feature_names,
            "expected_deployable": expected_deployable,
            "parent_sha256": parent_sha256,
            "geometry_sha256": geometry_sha256,
        })
        return model, artifact

    monkeypatch.setattr(
        "scripts.train_scanrefer_rec_hierarchical_reranker."
        "load_hierarchical_artifact",
        load,
    )
    monkeypatch.setattr(
        "train_dist_mod.validate_rec_hierarchical_runtime_provenance",
        lambda *args: calls.append({"validate": args}),
    )

    loaded = load_rec_hierarchical_runtime_artifact(
        "hierarchical.pth",
        torch.device("cpu"),
        parent,
        geometry,
        geometry_artifact,
    )

    assert loaded == (model, artifact)
    assert calls[0] == {
        "path": "hierarchical.pth",
        "device": torch.device("cpu"),
        "geometry_names": geometry_artifact["feature_names"],
        "expected_deployable": True,
        "parent_sha256": parent._artifact_sha256,
        "geometry_sha256": geometry._artifact_sha256,
    }
    assert "validate" in calls[1]


def test_train_tester_stable_loads_hierarchy_once_and_revalidates_each_use(
        monkeypatch):
    tester = TrainTester.__new__(TrainTester)
    tester.rec_reranker = torch.nn.Linear(1, 1)
    tester.rec_geometry_reranker = torch.nn.Linear(1, 1)
    tester.rec_geometry_reranker_artifact = {"geometry": True}
    tester.rec_hierarchical_reranker = None
    tester.rec_hierarchical_reranker_artifact = None
    hierarchy = RecordingHierarchy().eval().requires_grad_(False)
    artifact = {"hierarchical": True}
    calls = []

    def load(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return hierarchy, artifact

    def validate(*args, **kwargs):
        calls.append(("validate", args, kwargs))
        return {"margin": 0.1}

    monkeypatch.setattr(
        "train_dist_mod.load_rec_hierarchical_runtime_artifact", load
    )
    monkeypatch.setattr(
        "train_dist_mod.validate_rec_hierarchical_runtime_provenance",
        validate,
    )
    args = SimpleNamespace(
        rec_hierarchical_reranker_checkpoint="hierarchical.pth"
    )

    first = tester._ensure_rec_hierarchical_runtime_loaded(
        args, torch.device("cpu")
    )
    second = tester._ensure_rec_hierarchical_runtime_loaded(
        args, torch.device("cpu")
    )

    assert first == (hierarchy, artifact)
    assert second == (hierarchy, artifact)
    assert [call[0] for call in calls] == [
        "load", "validate", "validate"
    ]


def test_train_tester_requires_parent_geometry_and_excludes_residual():
    tester = TrainTester.__new__(TrainTester)
    end_points = {"last_center": torch.zeros(1, 1, 3)}
    base = {
        "eval_use_rec_reranker_scores": True,
        "eval_use_rec_geometry_reranker_scores": False,
        "eval_use_rec_selective_residual_scores": False,
        "eval_use_rec_hierarchical_reranker_scores": True,
        "rec_reranker_checkpoint": "parent.pth",
        "rec_geometry_reranker_checkpoint": "geometry.pth",
        "rec_selective_residual_checkpoint": None,
        "rec_hierarchical_reranker_checkpoint": "hierarchical.pth",
    }

    with pytest.raises(ValueError, match="hierarchical.*geometry"):
        tester._attach_rec_reranker_scores(
            end_points,
            {},
            SimpleNamespace(**base),
            batch_idx=0,
            num_batches=1,
        )

    base["eval_use_rec_geometry_reranker_scores"] = True
    base["eval_use_rec_selective_residual_scores"] = True
    base["rec_selective_residual_checkpoint"] = "residual.pth"
    with pytest.raises(ValueError, match="mutually exclusive"):
        tester._attach_rec_reranker_scores(
            end_points,
            {},
            SimpleNamespace(**base),
            batch_idx=0,
            num_batches=1,
        )


def test_main_eval_attaches_hierarchy_before_ground_truth_merge():
    tester = TrainTester.__new__(TrainTester)
    tester._to_gpu = lambda value: value
    tester._get_inputs = lambda _value: {"train": False}
    tester._compute_loss = lambda end_points, *_args: (
        torch.tensor(0.0), end_points
    )
    tester._accumulate_stats = lambda stats, _end_points: stats
    tester.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    seen = []
    forbidden = {
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "candidate_ious", "geometry_ious", "threshold_labels",
    }

    def attach(end_points, _inputs, _args, *, batch_idx, num_batches):
        assert forbidden.isdisjoint(end_points)
        seen.append((set(end_points), batch_idx, num_batches))

    tester._attach_rec_reranker_scores = attach
    args = SimpleNamespace(
        eval_use_rec_reranker_scores=False,
        eval_use_rec_geometry_reranker_scores=False,
        eval_use_rec_selective_residual_scores=False,
        eval_use_rec_hierarchical_reranker_scores=True,
        print_freq=100,
    )
    batch_data = {
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "box_label_mask": torch.ones(1, 1, dtype=torch.bool),
        "gt_masks": torch.ones(1, 1),
    }

    _stats, result = tester._main_eval_branch(
        0,
        batch_data,
        [None],
        lambda _inputs: {"last_center": torch.zeros(1, 1, 3)},
        {},
        None,
        None,
        args,
    )

    assert seen == [({"last_center"}, 0, 1)]
    assert set(batch_data).issubset(result)


def test_grounding_evaluator_uses_flat_geometry_axis_for_hierarchy(
        monkeypatch):
    tester = TrainTester.__new__(TrainTester)
    tester.logger = object()
    captured = {}

    def evaluator(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr("train_dist_mod.GroundingEvaluator", evaluator)
    args = SimpleNamespace(
        butd_cls=False,
        model="MCLN",
        eval_use_selector_choice_scores=False,
        eval_use_rec_reranker_scores=True,
        eval_use_rec_geometry_reranker_scores=False,
        eval_use_rec_selective_residual_scores=False,
        eval_use_rec_hierarchical_reranker_scores=True,
    )

    tester._build_grounding_evaluator(args, ("last_",))

    assert captured["eval_use_rec_geometry_reranker_scores"] is True
