import sys
from types import SimpleNamespace

import pytest
import torch

from main_utils import parse_option
from scripts.train_scanrefer_rec_selective_residual import (
    AUTHORITATIVE_BACKBONE_SHA256,
    build_selective_pair_feature_names,
)
from train_dist_mod import (
    TrainTester,
    build_rec_geometry_runtime_outputs,
    load_rec_selective_residual_runtime_artifact,
    validate_rec_selective_residual_runtime_provenance,
)
from test_rec_geometry_runtime import (
    _geometry_artifact,
    _geometry_batch,
    _parent_outputs,
)


class HeadRecordingGeometry(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, features, valid_mask):
        self.calls.append({
            "features": features.detach().cpu().clone(),
            "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": tuple(
                parameter.requires_grad for parameter in self.parameters()
            ),
        })
        ranking = torch.arange(
            features.shape[1], dtype=features.dtype, device=features.device
        ).unsqueeze(0).expand(features.shape[0], -1)
        threshold = torch.stack((features[..., 0], features[..., 1]), dim=-1)
        estimate = features[..., 2].sigmoid()
        return {
            "ranking_logits": ranking + self.anchor * 0.0,
            "threshold_logits": threshold,
            "iou_estimate": estimate,
        }


class RecordingResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, pair_features, pair_valid):
        self.calls.append({
            "features": pair_features.detach().cpu().clone(),
            "valid": pair_valid.detach().cpu().clone(),
            "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": tuple(
                parameter.requires_grad for parameter in self.parameters()
            ),
        })
        logits = pair_features.new_zeros(
            pair_features.shape[0], pair_features.shape[1], 2, 3
        )
        logits[..., 0] = 10.0
        logits[:, 0, :, 0] = 0.0
        logits[:, 0, :, 2] = 10.0
        return logits + self.anchor * 0.0


def _residual_artifact(geometry_artifact, margin=0.1):
    return {
        "schema": "rec-selective-residual-v1",
        "deployable": True,
        "validation_data_accessed": False,
        "input_dim": 185,
        "feature_names": build_selective_pair_feature_names(
            geometry_artifact["feature_names"]
        ),
        "selection": {"margin": float(margin)},
        "input_sha256": {
            "backbone": AUTHORITATIVE_BACKBONE_SHA256,
            "parent": "a" * 64,
            "geometry": "b" * 64,
        },
    }


def test_disabled_residual_path_is_bitwise_identical(monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    artifact = _geometry_artifact(parent_outputs, weight=1.0)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )

    implicit = build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, HeadRecordingGeometry(), artifact
    )
    explicit = build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, HeadRecordingGeometry(), artifact,
        residual_model=None, residual_artifact=None,
    )

    assert set(implicit) == set(explicit)
    for key in implicit:
        if isinstance(implicit[key], torch.Tensor):
            assert torch.equal(implicit[key], explicit[key])
        else:
            assert implicit[key] == explicit[key]


def test_enabled_residual_promotes_one_candidate_without_output_schema_drift(
        monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_artifact = _geometry_artifact(parent_outputs, weight=1.0)
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
    baseline_geometry = HeadRecordingGeometry()
    enabled_geometry = HeadRecordingGeometry()
    residual = RecordingResidual().train()

    baseline = build_rec_geometry_runtime_outputs(
        end_points, inputs, parent_outputs, baseline_geometry,
        geometry_artifact,
    )
    enabled = build_rec_geometry_runtime_outputs(
        end_points,
        inputs,
        parent_outputs,
        enabled_geometry,
        geometry_artifact,
        residual_model=residual,
        residual_artifact=_residual_artifact(geometry_artifact),
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
    assert len(residual.calls) == 1
    call = residual.calls[0]
    assert call["features"].shape == (1, 112, 185)
    assert call["valid"].shape == (1, 112)
    assert call["training"] is False
    assert call["grad_enabled"] is False
    assert call["requires_grad"] == (False,)
    normalized = enabled_geometry.calls[0]["features"]
    assert torch.equal(
        call["features"][0, 0, :179],
        normalized[0, 0] - normalized[0, 111],
    )
    assert call["features"][0, 111].eq(0.0).all()
    assert forbidden.isdisjoint(enabled)


@pytest.mark.parametrize("partial", ["model", "artifact"])
def test_runtime_builder_rejects_partial_residual_context(
        partial, monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_artifact = _geometry_artifact(parent_outputs, weight=1.0)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )
    model = RecordingResidual() if partial == "model" else None
    artifact = (
        _residual_artifact(geometry_artifact)
        if partial == "artifact" else None
    )

    with pytest.raises(ValueError, match="partial residual"):
        build_rec_geometry_runtime_outputs(
            {}, {}, parent_outputs, HeadRecordingGeometry(), geometry_artifact,
            residual_model=model, residual_artifact=artifact,
        )


def test_runtime_builder_rejects_residual_feature_schema_drift(monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_artifact = _geometry_artifact(parent_outputs, weight=1.0)
    residual_artifact = _residual_artifact(geometry_artifact)
    residual_artifact["feature_names"][0] = "changed"
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )

    with pytest.raises(ValueError, match="feature schema"):
        build_rec_geometry_runtime_outputs(
            {}, {}, parent_outputs, HeadRecordingGeometry(), geometry_artifact,
            residual_model=RecordingResidual(),
            residual_artifact=residual_artifact,
        )


def test_main_parser_exposes_residual_checkpoint_and_enable_flag(
        monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_dist_mod.py",
        "--rec_reranker_checkpoint", "parent.pth",
        "--eval_use_rec_reranker_scores",
        "--rec_geometry_reranker_checkpoint", "geometry.pth",
        "--eval_use_rec_geometry_reranker_scores",
        "--rec_selective_residual_checkpoint", "residual.pth",
        "--eval_use_rec_selective_residual_scores",
    ])

    args = parse_option()

    assert args.rec_selective_residual_checkpoint == "residual.pth"
    assert args.eval_use_rec_selective_residual_scores is True


def test_train_tester_requires_all_parent_geometry_residual_flags():
    tester = TrainTester.__new__(TrainTester)
    end_points = {"last_center": torch.zeros(1, 1, 3)}
    args = SimpleNamespace(
        eval_use_rec_reranker_scores=True,
        eval_use_rec_geometry_reranker_scores=False,
        eval_use_rec_selective_residual_scores=True,
        rec_reranker_checkpoint="parent.pth",
        rec_geometry_reranker_checkpoint="geometry.pth",
        rec_selective_residual_checkpoint="residual.pth",
    )

    with pytest.raises(ValueError, match="requires.*geometry"):
        tester._attach_rec_reranker_scores(
            end_points, {}, args, batch_idx=0, num_batches=1
        )


def test_train_tester_stable_loads_residual_once(monkeypatch):
    tester = TrainTester.__new__(TrainTester)
    tester.rec_reranker = torch.nn.Linear(1, 1)
    tester.rec_reranker_artifact = {"parent": True}
    tester.rec_geometry_reranker = torch.nn.Linear(1, 1)
    tester.rec_geometry_reranker_artifact = {"geometry": True}
    tester._rec_geometry_runtime_projection = {"allow_tf32": True}
    tester.rec_selective_residual = None
    tester.rec_selective_residual_artifact = None
    residual = RecordingResidual().eval().requires_grad_(False)
    artifact = {"residual": True}
    calls = []

    def load(path, device, **kwargs):
        calls.append((path, device, kwargs))
        return residual, artifact

    monkeypatch.setattr(
        "train_dist_mod.load_rec_selective_residual_runtime_artifact", load
    )
    args = SimpleNamespace(
        rec_selective_residual_checkpoint="residual.pth"
    )

    first = tester._ensure_rec_selective_residual_runtime_loaded(
        args, torch.device("cpu")
    )
    second = tester._ensure_rec_selective_residual_runtime_loaded(
        args, torch.device("cpu")
    )

    assert first == (residual, artifact)
    assert second == (residual, artifact)
    assert len(calls) == 1
    assert calls[0][0] == "residual.pth"


def test_residual_runtime_provenance_rejects_parent_binding():
    parent = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    geometry = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    residual = RecordingResidual().eval().requires_grad_(False)
    parent._artifact_sha256 = "a" * 64
    geometry._artifact_sha256 = "b" * 64
    residual._artifact_sha256 = "c" * 64
    geometry_artifact = {"feature_names": [
        "geometry_{:03d}".format(index) for index in range(179)
    ]}
    artifact = _residual_artifact(geometry_artifact)
    artifact["input_sha256"]["parent"] = "d" * 64
    artifact["model_state_dict"] = residual.state_dict()

    with pytest.raises(ValueError, match="parent"):
        validate_rec_selective_residual_runtime_provenance(
            parent,
            geometry,
            geometry_artifact,
            residual,
            artifact,
            torch.device("cpu"),
        )


def test_runtime_loader_passes_frozen_parent_and_geometry_hashes(monkeypatch):
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    parent._artifact_sha256 = "a" * 64
    geometry._artifact_sha256 = "b" * 64
    residual = RecordingResidual()
    artifact = {"artifact": True}
    calls = []

    def load(path, device, parent_sha256, geometry_sha256):
        calls.append((path, device, parent_sha256, geometry_sha256))
        return residual, artifact

    monkeypatch.setattr(
        "scripts.train_scanrefer_rec_selective_residual."
        "load_selective_residual_artifact",
        load,
    )
    monkeypatch.setattr(
        "train_dist_mod.validate_rec_selective_residual_runtime_provenance",
        lambda *args: calls.append(("validate", args)),
    )

    loaded = load_rec_selective_residual_runtime_artifact(
        "residual.pth",
        torch.device("cpu"),
        parent,
        geometry,
        {"geometry": True},
    )

    assert loaded == (residual, artifact)
    assert calls[0] == (
        "residual.pth", torch.device("cpu"), "a" * 64, "b" * 64
    )
    assert calls[1][0] == "validate"
