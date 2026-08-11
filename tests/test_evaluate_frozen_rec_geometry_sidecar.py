import copy
import hashlib
import inspect
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.evaluate_frozen_rec_geometry_sidecar as frozen


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
ROOT = Path(__file__).resolve().parents[1]


class RaisingScorer(torch.nn.Module):
    def forward(self, features, valid_mask):
        raise AssertionError("geometry scorer must not run")


class FixedScorer(torch.nn.Module):
    def __init__(self, winning_index):
        super().__init__()
        self.winning_index = int(winning_index)
        self.calls = 0

    def forward(self, features, valid_mask):
        self.calls += 1
        logits = features.new_zeros(valid_mask.shape)
        logits[:, self.winning_index] = 10.0
        return {"ranking_logits": logits}


def _artifact(weight=0.35, parent_sha=SHA_A):
    parent_contract = {
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
        "canonical_query_tie_policy": (
            "score-desc-query-index-asc-v1"
        ),
        "content_digest_version": (
            "ordered-identity-raw-float32-sha256-v1"
        ),
        "row_count": 4,
        "score_content_sha256": SHA_F,
    }
    return {
        "geometry_weight": float(weight),
        "regressed_variant_index": 0,
        "feature_mean": torch.zeros(179),
        "feature_std": torch.ones(179),
        "parent_artifact_sha256": parent_sha,
        "checkpoint_sha256": SHA_B,
        "model_inputs": {
            "use_color": True,
            "use_height": False,
            "use_multiview": False,
            "butd": True,
            "butd_gt": False,
            "butd_cls": False,
        },
        "candidate_rule": {"topk_per_source": 8, "max_candidates": 16},
        "target_iou_policy": "root_only",
        "evaluator_filter_policy": "evaluator-valid-no-gt-filter-v1",
        "filter_non_gt_boxes": False,
        "geometry_cache_schema_version": 1,
        "geometry_schema_version": "rec-mask-geometry-v1",
        "base_cache_schema_version": 1,
        "base_feature_schema_version": "rec-query-v1",
        "variant_names": ["variant_{}".format(index) for index in range(7)],
        "variant_configs": [
            {"name": "variant_{}".format(index)} for index in range(7)
        ],
        "min_points": 5,
        "max_point_fraction": 0.5,
        "backbone_config": {"num_target": 256},
        "model_config": {"input_dim": 179, "hidden_dim": 16, "dropout": 0.1},
        "epoch": 4,
        "calibration_metrics": {
            "sample_count": 1,
            "hits025": 1,
            "hits050": 1,
            "score": 1.0,
        },
        "training_args": {"model_seed": 0},
        "flat_parent_prior_version": (
            "score-desc-query-index-asc-regressed-first-v2"
        ),
        "parent_inference_contract": parent_contract,
        "scene_split": {"split_seed": 0, "sample_count": 4},
        "score_mode": "parent-flat-rank-blend-v1",
        "tie_policy": "score-desc-flat-index-asc-v1",
        "train_base_cache_content_digest": SHA_A,
        "train_base_cache_manifest_digest": SHA_B,
        "train_geometry_cache_content_digest": SHA_C,
        "train_geometry_immutable_metadata_digest": SHA_D,
        "train_parent_score_content_sha256": SHA_F,
    }


def _parent_state(batch_size=1):
    query_indices = torch.zeros(batch_size, 16, dtype=torch.long)
    query_indices[:, 0] = 5
    query_indices[:, 1] = 1
    query_indices[:, 2:] = torch.arange(2, 16)
    valid = torch.zeros(batch_size, 16, dtype=torch.bool)
    valid[:, :2] = True
    compact = torch.full((batch_size, 16), -100.0)
    compact[:, :2] = 0.5
    query_scores = torch.full((batch_size, 256), -float("inf"))
    query_scores.scatter_(1, query_indices[:, :2], compact[:, :2])
    query_order = torch.arange(256).unsqueeze(0).expand(batch_size, -1).clone()
    query_order[:, :2] = torch.tensor([1, 5])
    remaining = [index for index in range(256) if index not in (1, 5)]
    query_order[:, 2:] = torch.tensor(remaining)
    return {
        "compact_scores": compact,
        "query_scores": query_scores,
        "query_indices": query_indices,
        "candidate_valid": valid,
        "query_order": query_order,
        "top1_query_index": torch.ones(batch_size, dtype=torch.long),
        "parent_top1_mask": query_indices.eq(1) & valid,
    }


def _valid_record():
    sample_count = frozen.FROZEN_VAL_SAMPLE_COUNT
    parent_contract = {
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
        "canonical_query_tie_policy": (
            "score-desc-query-index-asc-v1"
        ),
        "content_digest_version": (
            "ordered-identity-raw-float32-sha256-v1"
        ),
        "row_count": sample_count,
        "score_content_sha256": SHA_F,
    }
    return {
        "schema": frozen.FROZEN_RECORD_SCHEMA,
        "version": frozen.FROZEN_RECORD_VERSION,
        "sample_count": sample_count,
        "hits025": 7000,
        "hits050": 5000,
        "acc025": 7000 / sample_count,
        "acc050": 5000 / sample_count,
        "parent_hits025": 6900,
        "parent_hits050": 4900,
        "parent_acc025": 6900 / sample_count,
        "parent_acc050": 4900 / sample_count,
        "fixes025": 200,
        "breaks025": 100,
        "fixes050": 200,
        "breaks050": 100,
        "geometry_weight": 0.35,
        "selected_artifact_sha256": SHA_A,
        "parent_artifact_sha256": SHA_B,
        "backbone_checkpoint_sha256": SHA_C,
        "selection_record_sha256": SHA_D,
        "sidecar_evaluator_sha256": SHA_D,
        "record_schema_sha256": frozen.FROZEN_RECORD_SCHEMA_SHA256,
        "base_cache_content_sha256": SHA_E,
        "base_cache_manifest_sha256": SHA_F,
        "geometry_cache_content_sha256": SHA_A,
        "geometry_cache_manifest_sha256": SHA_B,
        "geometry_cache_immutable_metadata_sha256": SHA_C,
        "val_parent_score_content_sha256": SHA_F,
        "parent_inference_contract": parent_contract,
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
    }


def _write_selection(path, selected_sha, artifact=None,
                     selected_filename="selected.pth",
                     uses_validation=False):
    artifact = _artifact() if artifact is None else artifact
    candidate = {
        "calibration_metrics": copy.deepcopy(
            artifact["calibration_metrics"]
        ),
        "candidate_order": 0,
        "eligible_no_regression": True,
        "epoch": artifact["epoch"],
        "filename": selected_filename,
        "geometry_weight": artifact["geometry_weight"],
        "model_config": copy.deepcopy(artifact["model_config"]),
        "selection_score": artifact["calibration_metrics"]["score"],
        "sha256": selected_sha,
        "training_args": copy.deepcopy(artifact["training_args"]),
    }
    payload = {
        "candidate_count": 1,
        "candidates": [candidate],
        "code_sha256": {
            name: frozen.sha256_file(ROOT / name)
            for name in (
                "models/rec_geometry_reranker.py",
                "models/rec_mask_geometry.py",
                "scripts/train_rec_geometry_reranker.py",
            )
        },
        "common_train_provenance": {
            key: copy.deepcopy(artifact[key])
            for key in (
                "flat_parent_prior_version",
                "parent_artifact_sha256",
                "parent_inference_contract",
                "scene_split",
                "score_mode",
                "tie_policy",
                "train_base_cache_content_digest",
                "train_base_cache_manifest_digest",
                "train_geometry_cache_content_digest",
                "train_geometry_immutable_metadata_digest",
                "train_parent_score_content_sha256",
            )
        },
        "created_at_utc": "2026-07-15T00:00:00Z",
        "selection_data_scope": "train fit/calibration scenes only",
        "selection_rule": {
            "eligibility": (
                "acc025 >= frozen parent acc025 AND acc050 >= frozen "
                "parent acc050"
            ),
            "objective": (
                "min(acc025 / 0.60, acc050 / 0.47) + "
                "0.1 * (acc025 + acc050)"
            ),
            "tie_break": (
                "lower candidate_order (declared primary/sweep order)"
            ),
        },
        "selection_schema_version": 1,
        "selection_uses_validation": uses_validation,
        "winner": {
            "calibration_metrics": copy.deepcopy(
                candidate["calibration_metrics"]
            ),
            "candidate_order": 0,
            "epoch": candidate["epoch"],
            "geometry_weight": candidate["geometry_weight"],
            "selected_filename": selected_filename,
            "selected_sha256": selected_sha,
            "selection_score": candidate["selection_score"],
            "source_filename": candidate["filename"],
            "source_sha256": candidate["sha256"],
        },
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload


def test_weight_zero_uses_canonical_parent_without_calling_geometry_scorer():
    parameters = tuple(inspect.signature(
        frozen.select_frozen_geometry_indices
    ).parameters)
    assert not any("iou" in name.lower() or "gt" in name.lower()
                   for name in parameters)
    scorer = RaisingScorer().eval()
    features = torch.zeros(1, 112, 179)
    valid = torch.zeros(1, 112, dtype=torch.bool)
    valid[:, 0] = True
    valid[:, 7] = True

    with torch.no_grad():
        selected, parent = frozen.select_frozen_geometry_indices(
            scorer,
            _artifact(weight=0.0),
            features,
            valid,
            _parent_state(),
        )

    assert selected.tolist() == [7]
    assert parent.tolist() == [7]


def test_nonzero_scoring_uses_only_the_frozen_artifact_weight():
    scorer = FixedScorer(winning_index=7).eval()
    features = torch.zeros(1, 112, 179)
    valid = torch.zeros(1, 112, dtype=torch.bool)
    valid[:, 0] = True
    valid[:, 7] = True

    with torch.no_grad():
        selected, parent = frozen.select_frozen_geometry_indices(
            scorer,
            _artifact(weight=1.0),
            features,
            valid,
            _parent_state(),
        )

    assert scorer.calls == 1
    assert selected.tolist() == [7]
    assert parent.tolist() == [7]
    assert "weight" not in inspect.signature(
        frozen.select_frozen_geometry_indices
    ).parameters


def test_metric_accumulator_reports_only_exact_aggregate_counts():
    ious = torch.tensor([
        [0.60, 0.10],
        [0.10, 0.60],
        [0.60, 0.30],
        [0.30, 0.60],
    ])
    counts = frozen.empty_metric_counts()

    frozen.accumulate_metric_counts(
        counts,
        selected_indices=torch.zeros(4, dtype=torch.long),
        parent_indices=torch.ones(4, dtype=torch.long),
        candidate_ious=ious,
    )
    metrics = frozen.finalize_metric_counts(counts)

    assert metrics == {
        "sample_count": 4,
        "hits025": 3,
        "hits050": 2,
        "parent_hits025": 3,
        "parent_hits050": 2,
        "fixes025": 1,
        "breaks025": 1,
        "fixes050": 2,
        "breaks050": 2,
        "acc025": 0.75,
        "acc050": 0.5,
        "parent_acc025": 0.75,
        "parent_acc050": 0.5,
    }
    assert not any("oracle" in key or "iou" in key for key in metrics)


def test_frozen_evaluation_batches_twelve_with_a_natural_remainder(monkeypatch):
    batch_sizes = []
    materializations = []

    def materialize(rows, parent, device, local_batch_size):
        materializations.append((len(rows), str(device), local_batch_size))

    def build_batch(rows, parent):
        batch_size = len(rows)
        batch_sizes.append(batch_size)
        ious = torch.zeros(batch_size, 112)
        ious[:, 7] = 0.6
        return {
            "features": torch.zeros(batch_size, 112, 179),
            "valid_mask": torch.nn.functional.one_hot(
                torch.full((batch_size,), 7), 112
            ).bool(),
            "candidate_ious": ious,
            "parent_state": _parent_state(batch_size),
        }

    monkeypatch.setattr(frozen, "materialize_parent_scores", materialize)
    monkeypatch.setattr(frozen, "build_geometry_training_batch", build_batch)
    scorer = RaisingScorer().eval()

    metrics = frozen.evaluate_selected_geometry_artifact(
        scorer,
        _artifact(weight=0.0),
        list(range(25)),
        parent=(torch.nn.Identity(), {}),
        device="cpu",
    )

    assert materializations == [(25, "cpu", 12)]
    assert batch_sizes == [12, 12, 1]
    assert metrics["sample_count"] == 25
    assert metrics["hits025"] == metrics["hits050"] == 25


def test_preflight_binds_actual_selection_artifact_and_parent_hashes(
        monkeypatch, tmp_path):
    selected_path = tmp_path / "selected.pth"
    parent_path = tmp_path / "parent.pth"
    selection_path = tmp_path / "selection.json"
    selected_path.write_bytes(b"selected artifact bytes")
    parent_path.write_bytes(b"parent artifact bytes")
    selected_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    artifact = _artifact(weight=0.35, parent_sha=parent_sha)
    _write_selection(
        selection_path, selected_sha, artifact, selected_path.name
    )
    geometry_model = torch.nn.Linear(1, 1).eval()
    parent_model = torch.nn.Linear(1, 1).eval()
    geometry_model._artifact_sha256 = selected_sha
    parent_model._artifact_sha256 = parent_sha

    monkeypatch.setattr(
        frozen,
        "load_geometry_reranker_artifact",
        lambda path, device: (geometry_model, artifact),
    )
    monkeypatch.setattr(
        frozen,
        "load_parent_reranker_snapshot",
        lambda path, device: (parent_model, {"artifact": "parent"}),
    )
    monkeypatch.setattr(frozen, "validate_geometry_artifact", lambda *a, **k: {})

    result = frozen.preflight_frozen_inputs(
        selection_path,
        selected_path,
        parent_path,
        device="cpu",
    )

    assert result["selected_artifact_sha256"] == selected_sha
    assert result["parent_artifact_sha256"] == parent_sha
    assert result["selection_record_sha256"] == hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    assert result["selection_uses_validation"] is False


@pytest.mark.parametrize("mutation, match", [
    (lambda selection, artifact, parent_sha: selection.update(
        selection_uses_validation=True), "validation"),
    (lambda selection, artifact, parent_sha: selection["winner"].update(
        selected_sha256=SHA_F), "winner"),
    (lambda selection, artifact, parent_sha: selection.update(
        candidate_count=2), "candidate"),
    (lambda selection, artifact, parent_sha: selection[
        "common_train_provenance"
    ].update(parent_artifact_sha256=SHA_F), "provenance"),
    (lambda selection, artifact, parent_sha: (
        selection["candidates"][0].update(selection_score=99.0),
        selection["winner"].update(selection_score=99.0),
    ), "artifact"),
    (lambda selection, artifact, parent_sha: selection[
        "selection_rule"
    ].update(objective="arbitrary validation objective"), "rule"),
    (lambda selection, artifact, parent_sha: selection[
        "code_sha256"
    ].update({"models/rec_geometry_reranker.py": SHA_F}), "code"),
    (lambda selection, artifact, parent_sha: selection[
        "candidates"
    ][0]["calibration_metrics"].update(
        selection_uses_validation=True
    ), "validation"),
    (lambda selection, artifact, parent_sha: artifact.update(
        parent_artifact_sha256=SHA_F), "parent"),
    (lambda selection, artifact, parent_sha: artifact["model_inputs"].update(
        butd_gt=True), "ground truth"),
    (lambda selection, artifact, parent_sha: artifact.update(
        filter_non_gt_boxes=True), "ground truth"),
])
def test_preflight_rejects_selection_or_no_gt_contract_mismatch(
        monkeypatch, tmp_path, mutation, match):
    selected_path = tmp_path / "selected.pth"
    parent_path = tmp_path / "parent.pth"
    selection_path = tmp_path / "selection.json"
    selected_path.write_bytes(b"selected")
    parent_path.write_bytes(b"parent")
    selected_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    artifact = _artifact(parent_sha=parent_sha)
    selection = _write_selection(
        selection_path, selected_sha, artifact, selected_path.name
    )
    mutation(selection, artifact, parent_sha)
    selection_path.write_text(
        json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    geometry_model = torch.nn.Linear(1, 1).eval()
    parent_model = torch.nn.Linear(1, 1).eval()
    geometry_model._artifact_sha256 = selected_sha
    parent_model._artifact_sha256 = parent_sha
    monkeypatch.setattr(
        frozen, "load_geometry_reranker_artifact",
        lambda path, device: (geometry_model, artifact)
    )
    monkeypatch.setattr(
        frozen, "load_parent_reranker_snapshot",
        lambda path, device: (parent_model, {})
    )
    monkeypatch.setattr(frozen, "validate_geometry_artifact", lambda *a, **k: {})

    with pytest.raises(ValueError, match=match):
        frozen.preflight_frozen_inputs(
            selection_path, selected_path, parent_path, device="cpu"
        )


def test_preflight_recomputes_the_winner_from_every_actual_candidate(
        monkeypatch, tmp_path):
    selected_path = tmp_path / "selected.pth"
    stronger_path = tmp_path / "stronger.pth"
    parent_path = tmp_path / "parent.pth"
    selection_path = tmp_path / "selection.json"
    selected_path.write_bytes(b"selected candidate")
    stronger_path.write_bytes(b"stronger candidate")
    parent_path.write_bytes(b"parent")
    selected_sha = frozen.sha256_file(selected_path)
    stronger_sha = frozen.sha256_file(stronger_path)
    parent_sha = frozen.sha256_file(parent_path)
    selected_artifact = _artifact(parent_sha=parent_sha)
    selected_artifact["calibration_metrics"].update({
        "acc025": 0.7,
        "acc050": 0.6,
        "parent_acc025": 0.5,
        "parent_acc050": 0.4,
        "score": frozen.calibration_score(0.7, 0.6),
    })
    selection = _write_selection(
        selection_path,
        selected_sha,
        selected_artifact,
        selected_path.name,
    )
    stronger_artifact = copy.deepcopy(selected_artifact)
    stronger_artifact["calibration_metrics"].update({
        "acc025": 0.9,
        "acc050": 0.8,
        "score": frozen.calibration_score(0.9, 0.8),
    })
    stronger = copy.deepcopy(selection["candidates"][0])
    stronger.update({
        "candidate_order": 1,
        "filename": stronger_path.name,
        "sha256": stronger_sha,
        "calibration_metrics": copy.deepcopy(
            stronger_artifact["calibration_metrics"]
        ),
        "selection_score": stronger_artifact[
            "calibration_metrics"
        ]["score"],
    })
    selection["candidate_count"] = 2
    selection["candidates"].append(stronger)
    selection_path.write_text(
        json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    parent_model = torch.nn.Linear(1, 1).eval()
    parent_model._artifact_sha256 = parent_sha

    def load_geometry(path, device):
        path = Path(path)
        artifact = (
            stronger_artifact if path.name == stronger_path.name
            else selected_artifact
        )
        model = torch.nn.Linear(1, 1).eval()
        model._artifact_sha256 = frozen.sha256_file(path)
        return model, artifact

    monkeypatch.setattr(frozen, "load_geometry_reranker_artifact", load_geometry)
    monkeypatch.setattr(
        frozen,
        "load_parent_reranker_snapshot",
        lambda path, device: (parent_model, {}),
    )
    monkeypatch.setattr(frozen, "validate_geometry_artifact", lambda *a, **k: {})

    with pytest.raises(ValueError, match="selection rule"):
        frozen.preflight_frozen_inputs(
            selection_path, selected_path, parent_path, device="cpu"
        )


def test_recomputed_selection_winner_uses_score_then_lower_order():
    def candidate(order, acc025, acc050):
        score = frozen.calibration_score(acc025, acc050)
        return {
            "candidate_order": order,
            "eligible_no_regression": True,
            "selection_score": score,
            "calibration_metrics": {
                "acc025": acc025,
                "acc050": acc050,
                "parent_acc025": 0.5,
                "parent_acc050": 0.4,
                "score": score,
            },
        }

    lower = candidate(0, 0.7, 0.6)
    higher = candidate(1, 0.9, 0.8)
    assert frozen.recompute_selection_winner([lower, higher]) == 1

    tied_later = candidate(1, 0.7, 0.6)
    assert frozen.recompute_selection_winner([lower, tied_later]) == 0


def test_val_bundle_uses_each_strict_loader_once_and_requires_9508(
        monkeypatch, tmp_path):
    base_path = tmp_path / "val"
    geometry_path = tmp_path / "geometry_val"
    base_path.mkdir()
    geometry_path.mkdir()
    rows = [object()] * frozen.FROZEN_VAL_SAMPLE_COUNT
    binding = {
        "path": str(base_path.resolve()),
        "content_sha256": SHA_A,
        "manifest_sha256": SHA_B,
    }
    base_manifest = {"split": "val"}
    geometry_manifest = {
        "split": "val",
        "base_cache_binding": copy.deepcopy(binding),
        "sample_count": frozen.FROZEN_VAL_SAMPLE_COUNT,
    }
    calls = []

    def load_base(path, split):
        calls.append(("base", split))
        return rows, base_manifest, binding

    def load_geometry(path, split, base_snapshot=None):
        calls.append(("geometry", split))
        assert base_snapshot == (rows, base_manifest, binding)
        return rows, geometry_manifest

    monkeypatch.setattr(frozen, "load_bound_candidate_cache", load_base)
    monkeypatch.setattr(frozen, "load_geometry_cache", load_geometry)
    monkeypatch.setattr(
        frozen, "join_base_and_geometry_rows",
        lambda *args, **kwargs: (
            rows if kwargs.get("verified_base_binding") == binding
            else pytest.fail("join must reuse the verified base binding")
        ),
    )

    result = frozen.load_frozen_val_bundle(base_path, geometry_path)

    assert calls == [("base", "val"), ("geometry", "val")]
    assert len(result["rows"]) == frozen.FROZEN_VAL_SAMPLE_COUNT

    monkeypatch.setattr(
        frozen,
        "join_base_and_geometry_rows",
        lambda *args, **kwargs: rows[:-1],
    )
    with pytest.raises(ValueError, match="9,508"):
        frozen.load_frozen_val_bundle(base_path, geometry_path)


def test_runtime_contract_is_exact_cuda0_tf32_world1(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        frozen, "_distributed_world_size", lambda: 1
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        assert frozen.validate_production_runtime("cuda:0") == torch.device(
            "cuda:0"
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        with pytest.raises(ValueError, match="TF32"):
            frozen.validate_production_runtime("cuda:0")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_claim_precedes_first_val_load_and_survives_failure(
        monkeypatch, tmp_path):
    claim_path = tmp_path / "registry" / "geometry_val_sidecar_once.claim"
    monkeypatch.setattr(
        frozen, "FROZEN_EVALUATION_CLAIM_PATH", claim_path, raising=False
    )
    output = tmp_path / "record.json"
    selected = tmp_path / "selected.pth"
    parent = tmp_path / "parent.pth"
    selection = tmp_path / "selection.json"
    selected.write_bytes(b"selected")
    parent.write_bytes(b"parent")
    selection.write_text("{}", encoding="utf-8")
    events = []
    context = {
        "selected_path": selected,
        "parent_path": parent,
        "selection_path": selection,
        "selected_artifact_sha256": frozen.sha256_file(selected),
        "parent_artifact_sha256": frozen.sha256_file(parent),
        "selection_record_sha256": frozen.sha256_file(selection),
        "sidecar_evaluator_sha256": SHA_C,
        "record_schema_sha256": frozen.FROZEN_RECORD_SCHEMA_SHA256,
        "selection_uses_validation": False,
        "geometry_model": torch.nn.Identity(),
        "geometry_artifact": _artifact(),
        "parent": (torch.nn.Identity(), {}),
    }

    monkeypatch.setattr(
        frozen, "preflight_frozen_inputs",
        lambda *args, **kwargs: events.append("preflight") or context,
    )
    monkeypatch.setattr(
        frozen, "validate_production_runtime",
        lambda device: events.append("runtime") or torch.device("cpu"),
    )

    def fail_load(*args):
        assert frozen.claim_path_for().is_file()
        events.append("val-load")
        raise RuntimeError("synthetic loader failure")

    monkeypatch.setattr(frozen, "load_frozen_val_bundle", fail_load)
    args = SimpleNamespace(
        selection_record=str(selection),
        selected_artifact=str(selected),
        parent_artifact=str(parent),
        base_cache="synthetic-base-val",
        geometry_cache="synthetic-geometry-val",
        output=str(output),
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        frozen.run_frozen_sidecar_evaluation(args)
    assert events == ["preflight", "runtime", "val-load"]
    assert frozen.claim_path_for().is_file()
    claim = json.loads(
        frozen.claim_path_for().read_text(encoding="utf-8")
    )
    assert claim["sidecar_evaluator_sha256"] == SHA_C
    assert claim["record_schema_sha256"] == frozen.FROZEN_RECORD_SCHEMA_SHA256

    with pytest.raises(FileExistsError, match="claim"):
        frozen.run_frozen_sidecar_evaluation(args)
    assert events.count("val-load") == 1

    args.output = str(tmp_path / "different-output.json")
    with pytest.raises(FileExistsError, match="claim"):
        frozen.run_frozen_sidecar_evaluation(args)
    assert events.count("val-load") == 1


@pytest.mark.parametrize(
    "link_kind,different_directory",
    [
        ("copy", False),
        ("hardlink", False),
        ("copy", True),
        ("hardlink", True),
    ],
)
def test_fixed_claim_rejects_selection_copy_or_hardlink_bypass(
        monkeypatch, tmp_path, link_kind, different_directory):
    claim_path = tmp_path / "registry" / "geometry_val_sidecar_once.claim"
    monkeypatch.setattr(
        frozen, "FROZEN_EVALUATION_CLAIM_PATH", claim_path, raising=False
    )
    selected = tmp_path / "selected.pth"
    parent = tmp_path / "parent.pth"
    selection = tmp_path / "selection.json"
    selected.write_bytes(b"selected")
    parent.write_bytes(b"parent")
    selection.write_text("{}", encoding="utf-8")
    alternate_directory = tmp_path / "elsewhere" if different_directory else tmp_path
    alternate_directory.mkdir(exist_ok=True)
    alternate = alternate_directory / "selection-copy.json"
    if link_kind == "copy":
        shutil.copyfile(selection, alternate)
    else:
        os.link(selection, alternate)

    def preflight(selection_path, *_args, **_kwargs):
        selection_path = Path(selection_path)
        return {
            "selected_path": selected,
            "parent_path": parent,
            "selection_path": selection_path,
            "selected_artifact_sha256": frozen.sha256_file(selected),
            "parent_artifact_sha256": frozen.sha256_file(parent),
            "selection_record_sha256": frozen.sha256_file(selection_path),
            "sidecar_evaluator_sha256": SHA_C,
            "record_schema_sha256": frozen.FROZEN_RECORD_SCHEMA_SHA256,
            "selection_uses_validation": False,
            "geometry_model": torch.nn.Identity(),
            "geometry_artifact": _artifact(),
            "parent": (torch.nn.Identity(), {}),
        }

    val_loads = []

    def fail_val_load(*_args):
        val_loads.append(True)
        raise RuntimeError("synthetic val load")

    monkeypatch.setattr(frozen, "preflight_frozen_inputs", preflight)
    monkeypatch.setattr(
        frozen, "validate_production_runtime", lambda _device: torch.device("cpu")
    )
    monkeypatch.setattr(frozen, "load_frozen_val_bundle", fail_val_load)
    args = SimpleNamespace(
        selection_record=str(selection),
        selected_artifact=str(selected),
        parent_artifact=str(parent),
        base_cache="synthetic-base-val",
        geometry_cache="synthetic-geometry-val",
        output=str(tmp_path / "record.json"),
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="synthetic val load"):
        frozen.run_frozen_sidecar_evaluation(args)
    args.selection_record = str(alternate)
    args.output = str(tmp_path / "record-copy.json")
    with pytest.raises(FileExistsError, match="claim"):
        frozen.run_frozen_sidecar_evaluation(args)
    assert len(val_loads) == 1
    assert frozen.claim_path_for() == claim_path


def test_existing_output_refuses_before_claim_or_val_load(
        monkeypatch, tmp_path):
    claim_path = tmp_path / "registry" / "geometry_val_sidecar_once.claim"
    monkeypatch.setattr(
        frozen, "FROZEN_EVALUATION_CLAIM_PATH", claim_path, raising=False
    )
    output = tmp_path / "record.json"
    output.write_text("already frozen", encoding="utf-8")
    selected = tmp_path / "selected.pth"
    parent = tmp_path / "parent.pth"
    selection = tmp_path / "selection.json"
    selected.write_bytes(b"selected")
    parent.write_bytes(b"parent")
    selection.write_text("{}", encoding="utf-8")
    context = {
        "selected_path": selected,
        "parent_path": parent,
        "selection_path": selection,
        "selected_artifact_sha256": frozen.sha256_file(selected),
        "parent_artifact_sha256": frozen.sha256_file(parent),
        "selection_record_sha256": frozen.sha256_file(selection),
        "sidecar_evaluator_sha256": SHA_C,
        "record_schema_sha256": SHA_D,
        "selection_uses_validation": False,
        "geometry_model": torch.nn.Identity(),
        "geometry_artifact": _artifact(),
        "parent": (torch.nn.Identity(), {}),
    }
    monkeypatch.setattr(
        frozen, "preflight_frozen_inputs", lambda *a, **k: context
    )
    monkeypatch.setattr(
        frozen, "validate_production_runtime", lambda device: torch.device("cpu")
    )
    monkeypatch.setattr(
        frozen,
        "load_frozen_val_bundle",
        lambda *a: pytest.fail("val loader must not run"),
    )
    args = SimpleNamespace(
        selection_record=str(selection),
        selected_artifact=str(selected),
        parent_artifact=str(parent),
        base_cache="synthetic-base-val",
        geometry_cache="synthetic-geometry-val",
        output=str(output),
        device="cuda:0",
    )

    with pytest.raises(FileExistsError, match="output"):
        frozen.run_frozen_sidecar_evaluation(args)

    assert not frozen.claim_path_for().exists()


def test_immutable_json_publication_is_exclusive_atomic_and_read_only(
        monkeypatch, tmp_path):
    output = tmp_path / "record.json"
    payload = {"b": 2, "a": 1}
    readbacks = []
    original_read = frozen._read_stable_file

    def recording_read(path, label):
        if Path(path) == output:
            readbacks.append(label)
        return original_read(path, label)

    monkeypatch.setattr(frozen, "_read_stable_file", recording_read)

    frozen.publish_immutable_json(output, payload)

    assert output.read_bytes() == b'{"a":1,"b":2}\n'
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert not list(tmp_path.glob("record.json.tmp.*"))
    assert readbacks == ["immutable output"]
    with pytest.raises(FileExistsError):
        frozen.publish_immutable_json(output, payload)


@pytest.mark.parametrize("tamper", ["corrupt", "replace"])
def test_immutable_publication_rejects_post_link_corruption_or_replacement(
        monkeypatch, tmp_path, tamper):
    output = tmp_path / "record.json"
    original_link = frozen.os.link

    def tampering_link(source, destination):
        original_link(source, destination)
        destination = Path(destination)
        os.chmod(str(destination), 0o644)
        if tamper == "corrupt":
            destination.write_bytes(b"corrupt\n")
        else:
            data = destination.read_bytes()
            destination.unlink()
            destination.write_bytes(data)
        os.chmod(str(destination), 0o444)

    monkeypatch.setattr(frozen.os, "link", tampering_link)

    with pytest.raises(RuntimeError, match="read-back"):
        frozen.publish_immutable_json(output, {"a": 1})


@pytest.mark.parametrize("changed_group", ["candidate", "code", "sidecar"])
def test_publish_precheck_rehashes_candidates_code_and_sidecar(
        tmp_path, changed_group):
    selection = tmp_path / "selection.json"
    selected = tmp_path / "selected.pth"
    parent = tmp_path / "parent.pth"
    candidate = tmp_path / "candidate.pth"
    code = tmp_path / "code.py"
    sidecar = tmp_path / "sidecar.py"
    for path in (selection, selected, parent, candidate, code, sidecar):
        path.write_bytes(path.name.encode("ascii"))

    def snapshot(path):
        return {"path": path, "sha256": frozen.sha256_file(path)}

    preflight = {
        "selection_path": selection,
        "selection_record_sha256": frozen.sha256_file(selection),
        "selected_path": selected,
        "selected_artifact_sha256": frozen.sha256_file(selected),
        "parent_path": parent,
        "parent_artifact_sha256": frozen.sha256_file(parent),
        "candidate_snapshots": (snapshot(candidate),),
        "code_snapshots": (snapshot(code),),
        "sidecar_snapshot": snapshot(sidecar),
    }
    changed = {
        "candidate": candidate,
        "code": code,
        "sidecar": sidecar,
    }[changed_group]
    changed.write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed"):
        frozen._require_unchanged_preflight_files(preflight)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=True),
    lambda value: value.update(sample_count=0),
    lambda value: value.update(hits025=2),
    lambda value: value.update(acc050=0.25),
    lambda value: value.update(selection_uses_validation=True),
    lambda value: value.update(inference_uses_ground_truth=True),
    lambda value: value.update(selected_artifact_sha256="bad"),
    lambda value: value["parent_inference_contract"].update(
        local_batch_size=8
    ),
])
def test_frozen_record_schema_and_metric_identities_are_exact(mutation):
    record = _valid_record()
    mutation(record)

    with pytest.raises(ValueError):
        frozen.validate_frozen_record(record)

    assert frozen.validate_frozen_record(_valid_record()) == _valid_record()


def test_frozen_record_rejects_an_internally_consistent_nonfull_sample():
    record = _valid_record()
    record["sample_count"] = frozen.FROZEN_VAL_SAMPLE_COUNT - 1
    denominator = float(record["sample_count"])
    record["acc025"] = record["hits025"] / denominator
    record["acc050"] = record["hits050"] / denominator
    record["parent_acc025"] = record["parent_hits025"] / denominator
    record["parent_acc050"] = record["parent_hits050"] / denominator
    record["parent_inference_contract"]["row_count"] = record[
        "sample_count"
    ]

    with pytest.raises(ValueError, match="9,508"):
        frozen.validate_frozen_record(record)


def test_success_record_binds_all_hashes_and_never_contains_diagnostics(
        monkeypatch, tmp_path):
    claim_path = tmp_path / "registry" / "geometry_val_sidecar_once.claim"
    monkeypatch.setattr(
        frozen, "FROZEN_EVALUATION_CLAIM_PATH", claim_path, raising=False
    )
    output = tmp_path / "record.json"
    selected = tmp_path / "selected.pth"
    parent_path = tmp_path / "parent.pth"
    selection = tmp_path / "selection.json"
    selected.write_bytes(b"selected")
    parent_path.write_bytes(b"parent")
    selection.write_text("{}", encoding="utf-8")
    selected_sha = frozen.sha256_file(selected)
    parent_sha = frozen.sha256_file(parent_path)
    selection_sha = frozen.sha256_file(selection)
    artifact = _artifact(weight=0.35, parent_sha=parent_sha)
    parent_model = torch.nn.Identity()
    context = {
        "selected_path": selected,
        "parent_path": parent_path,
        "selection_path": selection,
        "selected_artifact_sha256": selected_sha,
        "parent_artifact_sha256": parent_sha,
        "selection_record_sha256": selection_sha,
        "sidecar_evaluator_sha256": SHA_D,
        "record_schema_sha256": frozen.FROZEN_RECORD_SCHEMA_SHA256,
        "selection_uses_validation": False,
        "geometry_model": torch.nn.Identity(),
        "geometry_artifact": artifact,
        "parent": (parent_model, {}),
    }
    parent_contract = copy.deepcopy(
        _valid_record()["parent_inference_contract"]
    )
    bundle = {
        "rows": [object()] * frozen.FROZEN_VAL_SAMPLE_COUNT,
        "base_manifest": {"split": "val"},
        "base_binding": {
            "content_sha256": SHA_C,
            "manifest_sha256": SHA_D,
        },
        "geometry_manifest": {
            "cache_content_digest": SHA_E,
            "immutable_metadata_digest": SHA_F,
        },
        "geometry_manifest_sha256": SHA_A,
    }
    metrics = {
        "sample_count": frozen.FROZEN_VAL_SAMPLE_COUNT,
        "hits025": 7000,
        "hits050": 5000,
        "parent_hits025": 6900,
        "parent_hits050": 4900,
        "fixes025": 200,
        "breaks025": 100,
        "fixes050": 200,
        "breaks050": 100,
        "acc025": 7000 / frozen.FROZEN_VAL_SAMPLE_COUNT,
        "acc050": 5000 / frozen.FROZEN_VAL_SAMPLE_COUNT,
        "parent_acc025": 6900 / frozen.FROZEN_VAL_SAMPLE_COUNT,
        "parent_acc050": 4900 / frozen.FROZEN_VAL_SAMPLE_COUNT,
    }
    parent_contract["row_count"] = frozen.FROZEN_VAL_SAMPLE_COUNT
    parent_contract["score_content_sha256"] = SHA_B
    monkeypatch.setattr(frozen, "preflight_frozen_inputs", lambda *a, **k: context)
    monkeypatch.setattr(
        frozen, "validate_production_runtime", lambda device: torch.device("cpu")
    )
    monkeypatch.setattr(frozen, "load_frozen_val_bundle", lambda *a: bundle)
    monkeypatch.setattr(frozen, "validate_frozen_val_provenance", lambda *a: None)
    monkeypatch.setattr(
        frozen, "evaluate_selected_geometry_artifact", lambda *a, **k: metrics
    )
    monkeypatch.setattr(
        frozen,
        "sealed_parent_materialization_metadata",
        lambda parent: (parent_contract, SHA_B),
    )
    args = SimpleNamespace(
        selection_record=str(selection),
        selected_artifact=str(selected),
        parent_artifact=str(parent_path),
        base_cache="synthetic-base-val",
        geometry_cache="synthetic-geometry-val",
        output=str(output),
        device="cuda:0",
    )

    record = frozen.run_frozen_sidecar_evaluation(args)

    assert set(record) == set(frozen.FROZEN_RECORD_FIELDS)
    assert record["selected_artifact_sha256"] == selected_sha
    assert record["parent_artifact_sha256"] == parent_sha
    assert record["selection_record_sha256"] == selection_sha
    assert record["selection_uses_validation"] is False
    assert record["inference_uses_ground_truth"] is False
    assert not any(
        "oracle" in key or "selected_iou" in key or "grid" in key
        for key in record
    )
    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_cli_has_no_weight_grid_or_overwrite_surface(tmp_path):
    args = frozen.parse_args([
        "--base-cache", "synthetic-base-val",
        "--geometry-cache", "synthetic-geometry-val",
        "--parent-artifact", "parent.pth",
        "--selected-artifact", "selected.pth",
        "--selection-record", "selection.json",
        "--output", str(tmp_path / "record.json"),
    ])

    assert args.device == "cuda:0"
    assert not hasattr(args, "geometry_weight")
    assert not hasattr(args, "geometry_weights")
    assert not hasattr(args, "overwrite")


def test_record_schema_digest_binds_version_and_ordered_fields():
    expected = frozen.canonical_json_sha256({
        "schema": frozen.FROZEN_RECORD_SCHEMA,
        "version": frozen.FROZEN_RECORD_VERSION,
        "ordered_fields": list(frozen.FROZEN_RECORD_FIELDS),
    })
    assert frozen.FROZEN_RECORD_SCHEMA_SHA256 == expected


def test_frozen_record_rejects_wrong_well_formed_schema_digest():
    record = _valid_record()
    wrong_digest = "0" * 64
    assert wrong_digest != frozen.FROZEN_RECORD_SCHEMA_SHA256
    record["record_schema_sha256"] = wrong_digest

    with pytest.raises(ValueError, match="schema digest"):
        frozen.validate_frozen_record(record)
