from types import SimpleNamespace

import pytest
import torch

from models.rec_geometry_reranker import build_deployed_parent_state
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
)
from models.rec_joint_box_mask import MASK_POLICY_COUNT
from train_dist_mod import (
    TrainTester,
    build_rec_geometry_runtime_outputs,
    load_rec_geometry_runtime_artifacts,
    validate_parent_inference_runtime_compatibility,
    validate_rec_geometry_runtime_environment,
    validate_rec_geometry_runtime_outputs,
    validate_rec_geometry_runtime_provenance,
)


BASE_DIM = 152
GEOMETRY_DIM = 25
NUM_CANDIDATES = 16
NUM_VARIANTS = 7
NUM_QUERIES = 256


class RecordingGeometryReranker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, features, valid_mask):
        self.calls.append({
            "shape": tuple(features.shape),
            "dtype": features.dtype,
            "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": tuple(
                value.requires_grad for value in self.parameters()
            ),
        })
        logits = torch.arange(
            features.shape[1], dtype=features.dtype, device=features.device
        ).unsqueeze(0).expand(features.shape[0], -1)
        return {"ranking_logits": logits + self.anchor * 0.0}


class ForbiddenGeometryReranker(torch.nn.Module):
    def forward(self, _features, _valid_mask):
        raise AssertionError("zero-weight runtime must not call geometry scorer")


class FixedJointAdapter(torch.nn.Module):
    def __init__(self, policy_index=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.policy_index = int(policy_index)

    def forward(self, features, valid_mask):
        batch_size = features.shape[0]
        mask_iou = torch.full(
            (batch_size, 16, 7), 0.5,
            dtype=features.dtype, device=features.device,
        )
        box_logits = torch.full(
            (batch_size, 16, 7, 2), 5.0,
            dtype=features.dtype, device=features.device,
        )
        policy = torch.zeros(
            batch_size, 16, MASK_POLICY_COUNT,
            dtype=features.dtype, device=features.device,
        )
        policy[..., self.policy_index] = 1.0
        return {
            "mask_iou": mask_iou + self.anchor * 0.0,
            "box_logits": box_logits,
            "mask_policy_logits": policy,
        }


def _parent_outputs(batch_size=2):
    base_names = tuple("base_{:03d}".format(index) for index in range(BASE_DIM))
    query_indices = torch.tensor([
        [5, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [7, 3, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    ], dtype=torch.long)[:batch_size]
    valid = torch.ones(batch_size, NUM_CANDIDATES, dtype=torch.bool)
    compact_scores = torch.linspace(
        0.75, 0.0, NUM_CANDIDATES
    ).unsqueeze(0).expand(batch_size, -1).clone()
    compact_scores[:, 0] = 1.0
    compact_scores[:, 1] = 1.0
    parent_state = build_deployed_parent_state(
        compact_scores, query_indices, valid, NUM_QUERIES
    )
    candidate_batch = {
        "features": torch.randn(batch_size, NUM_CANDIDATES, BASE_DIM),
        "feature_names": base_names,
        "query_indices": query_indices,
        "valid_mask": valid,
        "num_queries": NUM_QUERIES,
        "boxes": torch.cat([
            torch.randn(batch_size, NUM_CANDIDATES, 3),
            torch.ones(batch_size, NUM_CANDIDATES, 3),
        ], dim=-1),
    }
    return {
        "candidate_batch": candidate_batch,
        "compact_scores": compact_scores,
        "query_scores": parent_state["query_scores"],
    }


def _geometry_batch(parent_outputs):
    candidate_batch = parent_outputs["candidate_batch"]
    batch_size = candidate_batch["features"].shape[0]
    valid = candidate_batch["valid_mask"].unsqueeze(2).expand(
        -1, -1, NUM_VARIANTS
    ).clone()
    boxes = torch.zeros(batch_size, NUM_CANDIDATES, NUM_VARIANTS, 6)
    boxes[..., :3] = torch.arange(
        NUM_CANDIDATES * NUM_VARIANTS, dtype=torch.float32
    ).reshape(1, NUM_CANDIDATES, NUM_VARIANTS, 1).expand(
        batch_size, -1, -1, 3
    )
    boxes[..., 3:] = 1.0
    configs = tuple(dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS)
    return {
        "boxes": boxes,
        "valid_mask": valid,
        "geometry_features": torch.randn(
            batch_size, NUM_CANDIDATES, NUM_VARIANTS, GEOMETRY_DIM
        ),
        "geometry_feature_names": REC_MASK_GEOMETRY_FEATURE_NAMES,
        "variant_names": tuple(value["name"] for value in configs),
        "variant_configs": configs,
        "min_points": 5,
        "max_point_fraction": 0.5,
        "mask_diagnostics": ({"must_not_escape": True},) * batch_size,
    }


def _geometry_artifact(parent_outputs, weight=0.5):
    base_names = list(parent_outputs["candidate_batch"]["feature_names"])
    configs = [dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    return {
        "input_dim": BASE_DIM + GEOMETRY_DIM + 2,
        "feature_names": (
            base_names + list(REC_MASK_GEOMETRY_FEATURE_NAMES)
            + ["parent_score", "parent_is_deployed_top1"]
        ),
        "feature_mean": torch.zeros(BASE_DIM + GEOMETRY_DIM + 2),
        "feature_std": torch.ones(BASE_DIM + GEOMETRY_DIM + 2),
        "variant_names": [value["name"] for value in configs],
        "variant_configs": configs,
        "regressed_variant_index": 0,
        "min_points": 5,
        "max_point_fraction": 0.5,
        "geometry_weight": float(weight),
        "num_queries": NUM_QUERIES,
    }


def _runtime_contract():
    return {
        "schema": "rec-parent-inference-contract",
        "version": 1,
        "device_type": "cuda",
        "device_index": 0,
        "local_batch_size": 12,
        "world_size": 1,
        "row_order": "dataset-index-contiguous",
        "remainder_policy": "natural-remainder",
        "feature_source": "bound-base-cache-features",
        "dtype": "float32",
        "autocast": False,
        "allow_tf32": True,
        "eval": True,
        "no_grad": True,
        "score_builder": "normalized-query-reranker-rank-blend",
        "score_builder_version": 1,
        "canonical_query_tie_policy": "score-desc-query-index-asc-v1",
        "content_digest_version": "ordered-identity-raw-float32-sha256-v1",
        "row_count": 36665,
        "score_content_sha256": "a" * 64,
    }


def test_parent_runtime_contract_uses_only_inference_compatibility_projection():
    contract = _runtime_contract()
    contract.update({
        "feature_source": "a-train-only-source-that-is-not-live-val",
        "row_count": 9508,
        "score_content_sha256": "b" * 64,
        "content_digest_version": "different-train-only-digest-version",
        "row_order": "different-val-row-identity",
    })

    projection = validate_parent_inference_runtime_compatibility(contract)

    assert projection["device_type"] == "cuda"
    assert projection["local_batch_size"] == 12
    assert "row_count" not in projection
    assert "score_content_sha256" not in projection
    assert "feature_source" not in projection


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("device_type", "cpu"),
        ("device_index", 1),
        ("local_batch_size", 8),
        ("world_size", 2),
        ("remainder_policy", "drop-last"),
        ("dtype", "float16"),
        ("autocast", True),
        ("allow_tf32", False),
        ("eval", False),
        ("no_grad", False),
        ("score_builder", "other"),
        ("score_builder_version", 2),
        ("canonical_query_tie_policy", "unstable"),
    ],
)
def test_parent_runtime_contract_rejects_incompatible_execution(field, bad_value):
    contract = _runtime_contract()
    contract[field] = bad_value

    with pytest.raises(ValueError, match="runtime-compatible"):
        validate_parent_inference_runtime_compatibility(contract)


def test_geometry_runtime_environment_requires_batch12_world1_cuda0(monkeypatch):
    args = SimpleNamespace(batch_size=12)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    validate_rec_geometry_runtime_environment(
        args,
        actual_batch_size=12,
        device=torch.device("cuda:0"),
        batch_idx=0,
        num_batches=2,
    )
    validate_rec_geometry_runtime_environment(
        args,
        actual_batch_size=5,
        device=torch.device("cuda:0"),
        batch_idx=1,
        num_batches=2,
    )

    with pytest.raises(ValueError, match="only the final batch"):
        validate_rec_geometry_runtime_environment(
            args,
            actual_batch_size=5,
            device=torch.device("cuda:0"),
            batch_idx=0,
            num_batches=2,
        )

    args.batch_size = 8
    with pytest.raises(ValueError, match="batch_size=12"):
        validate_rec_geometry_runtime_environment(
            args,
            actual_batch_size=8,
            device=torch.device("cuda:0"),
            batch_idx=0,
            num_batches=1,
        )
    args.batch_size = 12
    with pytest.raises(ValueError, match="at most 12"):
        validate_rec_geometry_runtime_environment(
            args,
            actual_batch_size=13,
            device=torch.device("cuda:0"),
            batch_idx=0,
            num_batches=1,
        )
    with pytest.raises(ValueError, match="cuda:0"):
        validate_rec_geometry_runtime_environment(
            args,
            actual_batch_size=12,
            device=torch.device("cuda:1"),
            batch_idx=0,
            num_batches=1,
        )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world_size=1"):
        validate_rec_geometry_runtime_environment(
            args,
            actual_batch_size=12,
            device=torch.device("cuda:0"),
            batch_idx=0,
            num_batches=1,
        )


def test_zero_weight_is_exact_parent_axis_without_geometry_work(monkeypatch):
    parent_outputs = _parent_outputs()
    artifact = _geometry_artifact(parent_outputs, weight=0.0)

    def forbidden_builder(*_args, **_kwargs):
        raise AssertionError("zero-weight runtime must not build geometry")

    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        forbidden_builder,
    )

    outputs = build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, ForbiddenGeometryReranker(), artifact
    )

    assert set(outputs) == {
        "rec_reranker_scores", "rec_geometry_runtime_mode",
    }
    assert outputs["rec_geometry_runtime_mode"] == "parent_query_axis"
    assert torch.equal(
        outputs["rec_reranker_scores"], parent_outputs["query_scores"]
    )


def test_nonzero_geometry_reuses_parent_candidates_and_attaches_flat_axis(
        monkeypatch):
    parent_outputs = _parent_outputs()
    geometry_batch = _geometry_batch(parent_outputs)
    artifact = _geometry_artifact(parent_outputs, weight=0.5)
    scorer = RecordingGeometryReranker().train()
    seen = []
    forbidden = {
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "candidate_ious", "geometry_ious", "threshold_labels",
        "rejection_codes", "variant_rejection_codes",
    }

    def geometry_builder(end_points, inputs, candidate_batch, variant_config):
        assert forbidden.isdisjoint(end_points)
        assert forbidden.isdisjoint(inputs)
        assert forbidden.isdisjoint(candidate_batch)
        assert candidate_batch is parent_outputs["candidate_batch"]
        assert variant_config["variants"] == artifact["variant_configs"]
        seen.append(candidate_batch)
        return geometry_batch

    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates", geometry_builder
    )

    outputs = build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, scorer, artifact
    )

    assert seen == [parent_outputs["candidate_batch"]]
    assert len(scorer.calls) == 1
    assert scorer.calls[0] == {
        "shape": (2, 112, 179),
        "dtype": torch.float32,
        "training": False,
        "grad_enabled": False,
        "requires_grad": (False,),
    }
    assert outputs["rec_geometry_runtime_mode"] == "flat_geometry_axis"
    assert outputs["rec_geometry_boxes"].shape == (2, 112, 6)
    assert outputs["rec_geometry_scores"].shape == (2, 112)
    assert outputs["rec_geometry_valid_mask"].shape == (2, 112)
    assert outputs["rec_geometry_fallback_index"].shape == (2,)
    assert outputs["rec_geometry_fallback_index"].tolist() == [7, 7]
    assert torch.equal(
        outputs["rec_reranker_scores"], parent_outputs["query_scores"]
    )
    assert forbidden.isdisjoint(outputs)


def test_joint_geometry_runtime_attaches_query_bound_mask_policy(monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    artifact = _geometry_artifact(parent_outputs, weight=0.5)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )
    joint = FixedJointAdapter(policy_index=4).eval().requires_grad_(False)
    joint_artifact = {
        "feature_mean": torch.zeros(179),
        "feature_std": torch.ones(179),
        "switch_margin": 0.02,
        "box_margin": 0.05,
    }

    outputs = build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, RecordingGeometryReranker(), artifact,
        joint_model=joint, joint_artifact=joint_artifact,
    )

    selected = outputs["rec_geometry_scores"].argmax(dim=1)
    assert torch.equal(outputs["rec_joint_selected_flat_index"], selected)
    assert torch.equal(
        outputs["rec_joint_selected_parent_position"],
        torch.div(selected, 7, rounding_mode="floor"),
    )
    assert outputs["rec_joint_mask_policy_index"].tolist() == [4]
    assert outputs["rec_joint_mask_source_index"].tolist() == [0]
    assert outputs["rec_joint_mask_threshold_index"].tolist() == [4]
    assert outputs["rec_joint_mask_threshold"].tolist() == [1.0]
    assert not {
        "gt_masks", "candidate_ious", "geometry_ious", "target_iou"
    }.intersection(outputs)


def test_geometry_runtime_rejects_noncanonical_box_layout(monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    geometry_batch["boxes"] = geometry_batch["boxes"].reshape(1, 112, 6)
    artifact = _geometry_artifact(parent_outputs, weight=0.5)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )

    with pytest.raises(ValueError, match="geometry boxes"):
        build_rec_geometry_runtime_outputs(
            {}, {}, parent_outputs, RecordingGeometryReranker(), artifact
        )


def test_geometry_runtime_disables_outer_autocast_and_grad_for_full_pipeline(
        monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    artifact = _geometry_artifact(parent_outputs, weight=0.5)
    execution = []

    def geometry_builder(*_args, **_kwargs):
        execution.append((
            torch.is_autocast_cpu_enabled(), torch.is_grad_enabled()
        ))
        return geometry_batch

    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates", geometry_builder
    )
    initial_autocast = torch.is_autocast_cpu_enabled()
    with torch.cpu.amp.autocast(enabled=True):
        build_rec_geometry_runtime_outputs(
            {}, {}, parent_outputs, RecordingGeometryReranker().train(), artifact
        )
        assert torch.is_autocast_cpu_enabled() is True

    assert torch.is_autocast_cpu_enabled() is initial_autocast
    assert execution == [(False, False)]


def _valid_runtime_outputs(monkeypatch):
    parent_outputs = _parent_outputs(batch_size=1)
    geometry_batch = _geometry_batch(parent_outputs)
    artifact = _geometry_artifact(parent_outputs, weight=0.5)
    monkeypatch.setattr(
        "train_dist_mod.build_rec_mask_geometry_candidates",
        lambda *_args, **_kwargs: geometry_batch,
    )
    return build_rec_geometry_runtime_outputs(
        {}, {}, parent_outputs, RecordingGeometryReranker(), artifact
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["rec_geometry_scores"].__setitem__((0, 0),
                                                                 float("nan")),
         "valid geometry scores"),
        (lambda value: value["rec_geometry_scores"].__setitem__((0, 0),
                                                                 float("inf")),
         "valid geometry scores"),
        (lambda value: (
            value["rec_geometry_valid_mask"].__setitem__((0, 111), False),
            value["rec_geometry_scores"].__setitem__((0, 111), 0.0),
        ), "invalid geometry scores"),
        (lambda value: value["rec_geometry_boxes"].__setitem__(
            (0, 0, 0), float("nan")
        ), "valid geometry boxes"),
        (lambda value: (
            value["rec_geometry_valid_mask"].__setitem__((0, 111), False),
            value["rec_geometry_scores"].__setitem__(
                (0, 111), -float("inf")
            ),
            value["rec_geometry_boxes"].__setitem__(
                (0, 111, 0), float("nan")
            ),
        ), "geometry boxes"),
        (lambda value: value["rec_geometry_boxes"].__setitem__(
            (0, 0, 3), 0.0
        ), "positive size"),
        (lambda value: value["rec_geometry_valid_mask"].__setitem__(
            (0, slice(None)), False
        ), "at least one valid"),
        (lambda value: value["rec_geometry_fallback_index"].__setitem__(0, 112),
         "fallback"),
        (lambda value: (
            value["rec_geometry_valid_mask"].__setitem__((0, 7), False),
            value["rec_geometry_scores"].__setitem__((0, 7), -float("inf")),
        ), "fallback"),
        (lambda value: value.__setitem__(
            "rec_geometry_boxes", value["rec_geometry_boxes"].double()
        ), "float32"),
        (lambda value: value.__setitem__(
            "rec_geometry_scores", value["rec_geometry_scores"].double()
        ), "float32"),
        (lambda value: value.__setitem__(
            "rec_reranker_scores", value["rec_reranker_scores"].double()
        ), "float32"),
    ],
)
def test_geometry_runtime_outputs_fail_closed(monkeypatch, mutation, match):
    outputs = _valid_runtime_outputs(monkeypatch)
    mutation(outputs)

    with pytest.raises(ValueError, match=match):
        validate_rec_geometry_runtime_outputs(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_geometry_runtime_outputs_reject_cross_device_tensors(monkeypatch):
    outputs = _valid_runtime_outputs(monkeypatch)
    outputs["rec_geometry_boxes"] = outputs["rec_geometry_boxes"].cuda()

    with pytest.raises(ValueError, match="same device"):
        validate_rec_geometry_runtime_outputs(outputs)


def test_parent_axis_runtime_outputs_require_float32():
    outputs = {
        "rec_reranker_scores": torch.ones(
            1, NUM_QUERIES, dtype=torch.float64
        ),
        "rec_geometry_runtime_mode": "parent_query_axis",
    }

    with pytest.raises(ValueError, match="float32"):
        validate_rec_geometry_runtime_outputs(outputs)


def test_parent_axis_runtime_outputs_require_256_query_scores():
    outputs = {
        "rec_reranker_scores": torch.zeros(2, NUM_QUERIES - 1),
        "rec_geometry_runtime_mode": "parent_query_axis",
    }

    with pytest.raises(ValueError, match="256"):
        validate_rec_geometry_runtime_outputs(outputs)


def test_flat_axis_runtime_outputs_require_256_parent_query_scores(monkeypatch):
    outputs = _valid_runtime_outputs(monkeypatch)
    outputs["rec_reranker_scores"] = outputs["rec_reranker_scores"][
        :, :NUM_QUERIES - 1
    ]

    with pytest.raises(ValueError, match="256"):
        validate_rec_geometry_runtime_outputs(outputs)


def test_runtime_loader_reads_each_artifact_once_without_parent_reload(
        monkeypatch):
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    parent_artifact = {"parent": True}
    geometry_artifact = {"geometry": True}
    calls = []

    def load_parent(path, device):
        calls.append(("parent", path, str(device)))
        return parent, parent_artifact

    def load_geometry(path, device, **kwargs):
        calls.append(("geometry", path, str(device), kwargs))
        return geometry, geometry_artifact

    def validate_geometry(artifact, parent=None, **_kwargs):
        calls.append(("validate", artifact, parent))

    monkeypatch.setattr(
        "scripts.train_rec_geometry_reranker.load_parent_reranker_snapshot",
        load_parent,
    )
    monkeypatch.setattr(
        "scripts.train_rec_geometry_reranker.load_geometry_reranker_artifact",
        load_geometry,
    )
    monkeypatch.setattr(
        "scripts.train_rec_geometry_reranker.validate_geometry_artifact",
        validate_geometry,
    )

    loaded = load_rec_geometry_runtime_artifacts(
        "parent.pth", "geometry.pth", device="cpu"
    )

    assert loaded == (parent, parent_artifact, geometry, geometry_artifact)
    assert calls == [
        ("parent", "parent.pth", "cpu"),
        ("geometry", "geometry.pth", "cpu", {}),
        ("validate", geometry_artifact, (parent, parent_artifact)),
    ]


def test_train_tester_reuses_one_stable_artifact_load_across_batches(
        monkeypatch):
    tester = TrainTester.__new__(TrainTester)
    tester.rec_reranker = None
    tester.rec_reranker_artifact = None
    tester.rec_geometry_reranker = None
    tester.rec_geometry_reranker_artifact = None
    tester._rec_geometry_runtime_projection = None
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    parent_artifact = {"parent": True}
    geometry_artifact = {"geometry": True}
    calls = []
    projection = {"allow_tf32": True}

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return parent, parent_artifact, geometry, geometry_artifact

    monkeypatch.setattr(
        "train_dist_mod.load_rec_geometry_runtime_artifacts", load
    )
    monkeypatch.setattr(
        "train_dist_mod.validate_rec_geometry_runtime_provenance",
        lambda *_args, **_kwargs: projection,
    )
    args = SimpleNamespace(
        rec_reranker_checkpoint="parent.pth",
        rec_geometry_reranker_checkpoint="geometry.pth",
    )

    first = tester._ensure_rec_geometry_runtime_loaded(args, torch.device("cpu"))
    second = tester._ensure_rec_geometry_runtime_loaded(args, torch.device("cpu"))

    assert first is projection
    assert second is projection
    assert calls == [(('parent.pth', 'geometry.pth'), {
        "device": torch.device("cpu")
    })]


def test_train_tester_rejects_partial_geometry_artifact_state():
    tester = TrainTester.__new__(TrainTester)
    tester.rec_reranker = torch.nn.Linear(1, 1)
    tester.rec_reranker_artifact = None
    tester.rec_geometry_reranker = None
    tester.rec_geometry_reranker_artifact = None

    with pytest.raises(ValueError, match="partial artifact state"):
        tester._ensure_rec_geometry_runtime_loaded(
            SimpleNamespace(), torch.device("cpu")
        )


def test_geometry_runtime_provenance_rejects_actual_parent_sha_mismatch():
    parent = torch.nn.Linear(1, 1)
    parent._artifact_sha256 = "a" * 64
    geometry = torch.nn.Linear(1, 1)
    geometry._artifact_sha256 = "c" * 64
    artifact = {
        "parent_artifact_sha256": "b" * 64,
        "parent_inference_contract": _runtime_contract(),
    }

    with pytest.raises(ValueError, match="parent artifact SHA"):
        validate_rec_geometry_runtime_provenance(
            SimpleNamespace(), parent, {}, geometry, artifact,
            torch.device("cuda:0")
        )


def test_main_eval_branch_attaches_geometry_before_ground_truth_merge():
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
        "rejection_codes", "variant_rejection_codes",
    }

    def attach(end_points, _inputs, _args, *, batch_idx, num_batches):
        assert forbidden.isdisjoint(end_points)
        seen.append((set(end_points), batch_idx, num_batches))

    tester._attach_rec_reranker_scores = attach
    args = SimpleNamespace(
        eval_use_rec_reranker_scores=False,
        eval_use_rec_geometry_reranker_scores=True,
        print_freq=100,
    )
    batch_data = {
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "box_label_mask": torch.ones(1, 1, dtype=torch.bool),
        "gt_masks": torch.ones(1, 1),
    }

    _, result = tester._main_eval_branch(
        0, batch_data, [None], lambda _inputs: {"last_center": torch.zeros(1, 1, 3)},
        {}, None, None, args
    )

    assert seen == [({"last_center"}, 0, 1)]
    assert set(batch_data).issubset(result)
