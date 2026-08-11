import copy
import hashlib
import json
import os
import random
import stat
from pathlib import Path

import pytest
import torch

from models.rec_selective_residual import (
    SelectiveResidualModel,
    build_residual_scene_folds,
    build_selective_pair_targets,
)
from scripts.train_rec_geometry_reranker import (
    DEFAULT_GEOMETRY_WEIGHTS,
    GEOMETRY_INPUT_DIM,
    _stable_flat_top1_indices,
    _stable_rank_normalize_once,
    build_geometry_training_batch,
    evaluate_geometry_blends,
    materialize_parent_scores,
)
from scripts.train_rec_reranker import normalize_features
from scripts.train_scanrefer_rec_selective_residual import (
    AUTHORITATIVE_BACKBONE_SHA256,
    AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
    AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
    AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
    RESIDUAL_BATCH_SIZE,
    RESIDUAL_EPOCHS,
    RESIDUAL_LEARNING_RATE,
    build_cache_calibration_baseline,
    build_selective_pair_feature_names,
    build_selective_residual_artifact,
    calibration_gate,
    capture_immutable_artifact_identities,
    canonical_residual_rows_sha256,
    canonical_selected_iou_sha256,
    cross_fit_selective_residual,
    evaluate_selective_residual_policy,
    load_selective_residual_artifact,
    load_residual_training_inputs,
    materialize_residual_rows,
    parse_args,
    publish_selective_residual_experiment,
    refit_selective_residual,
    run_selective_residual_training,
    save_selective_residual_artifact,
    split_residual_records,
    validate_selective_residual_artifact,
)
from test_train_rec_geometry_reranker import (
    BASE_FEATURE_NAMES,
    GEOMETRY_FEATURE_NAMES,
    _joined_row,
    _parent,
)


class RecordingGeometryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, features, valid_mask):
        self.calls.append({
            "autocast": torch.is_autocast_enabled(),
            "cpu_autocast": torch.is_autocast_cpu_enabled(),
            "dtype": features.dtype,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": tuple(
                parameter.requires_grad for parameter in self.parameters()
            ),
            "training": self.training,
        })
        ranking = features[..., len(BASE_FEATURE_NAMES)] + self.anchor * 0.0
        threshold = torch.stack((features[..., 0], features[..., 1]), dim=-1)
        iou_estimate = features[..., 2].sigmoid()
        return {
            "ranking_logits": ranking.masked_fill(~valid_mask, -1e4),
            "threshold_logits": threshold,
            "iou_estimate": iou_estimate,
        }


def _geometry_artifact():
    return {
        "checkpoint_sha256": AUTHORITATIVE_BACKBONE_SHA256,
        "parent_artifact_sha256": AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        "feature_mean": torch.zeros(GEOMETRY_INPUT_DIM, dtype=torch.float32),
        "feature_std": torch.ones(GEOMETRY_INPUT_DIM, dtype=torch.float32),
        "feature_names": list(
            BASE_FEATURE_NAMES
            + GEOMETRY_FEATURE_NAMES
            + ("parent_score", "parent_is_deployed_top1")
        ),
        "geometry_weight": 1.0,
        "input_dim": GEOMETRY_INPUT_DIM,
        "regressed_variant_index": 0,
    }


def _materialized(rows=None):
    rows = rows or [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    geometry_model = RecordingGeometryModel()
    records = materialize_residual_rows(
        rows,
        parent,
        geometry_model,
        _geometry_artifact(),
        batch_size=1,
        device="cpu",
    )
    return rows, parent, geometry_model, records


def test_materialization_returns_canonical_cpu_records_and_frozen_calls():
    rows, parent, geometry_model, records = _materialized()

    assert len(records) == 2
    assert [record["dataset_index"] for record in records] == [0, 1]
    assert set(records[0]) == {
        "dataset_index",
        "scan_id",
        "target_id",
        "pair_features",
        "pair_valid",
        "candidate_ious",
        "baseline_index",
        "baseline_scores",
        "query_positions",
        "variant_indices",
    }
    for record in records:
        assert record["pair_features"].shape == (112, 185)
        assert record["pair_features"].dtype == torch.float32
        assert record["pair_features"].device.type == "cpu"
        assert record["pair_valid"].shape == (112,)
        assert record["pair_valid"].dtype == torch.bool
        assert record["candidate_ious"].shape == (112,)
        assert record["candidate_ious"].dtype == torch.float32
        assert record["baseline_scores"].shape == (112,)
        assert record["baseline_scores"].dtype == torch.float32
        assert record["query_positions"].dtype == torch.long
        assert record["variant_indices"].dtype == torch.long
        assert not record["pair_valid"][record["baseline_index"]]
    assert parent[0].training is False
    assert all(not parameter.requires_grad for parameter in parent[0].parameters())
    assert geometry_model.training is False
    assert all(
        not parameter.requires_grad for parameter in geometry_model.parameters()
    )
    assert len(geometry_model.calls) == len(rows)
    assert all(call == {
        "autocast": False,
        "cpu_autocast": False,
        "dtype": torch.float32,
        "grad_enabled": False,
        "requires_grad": (False,),
        "training": False,
    } for call in geometry_model.calls)


def test_materialization_matches_exact_geometry_weight_one_branch():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    _, _, _, records = _materialized(copy.deepcopy(rows))

    expected_rows = copy.deepcopy(rows)
    expected_parent = _parent()
    materialize_parent_scores(expected_rows, expected_parent, device="cpu")
    batch = build_geometry_training_batch(expected_rows, expected_parent)
    normalized = normalize_features(
        batch["features"],
        batch["valid_mask"],
        _geometry_artifact()["feature_mean"],
        _geometry_artifact()["feature_std"],
    )
    expected_model = RecordingGeometryModel().eval().requires_grad_(False)
    with torch.no_grad():
        outputs = expected_model(normalized, batch["valid_mask"])
    expected_scores = _stable_rank_normalize_once(
        outputs["ranking_logits"], batch["valid_mask"]
    ).masked_fill(~batch["valid_mask"], -float("inf"))
    expected_indices = _stable_flat_top1_indices(
        expected_scores, batch["valid_mask"]
    )

    assert torch.equal(
        torch.stack([record["baseline_scores"] for record in records]),
        expected_scores,
    )
    assert [record["baseline_index"] for record in records] == \
        expected_indices.tolist()

    evaluator_rows = copy.deepcopy(rows)
    metrics = evaluate_geometry_blends(
        RecordingGeometryModel(),
        evaluator_rows,
        _geometry_artifact()["feature_mean"],
        _geometry_artifact()["feature_std"],
        _parent(),
        geometry_weights=DEFAULT_GEOMETRY_WEIGHTS,
        batch_size=1,
        device="cpu",
    )
    selected_ious = tuple(
        float(record["candidate_ious"][record["baseline_index"]].item())
        for record in records
    )
    assert selected_ious == pytest.approx(metrics[1.0]["selected_ious"])


def test_training_ious_change_targets_but_not_features_or_baseline():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["geometry"]["geometry_ious"].copy_(
            1.0 - row["geometry"]["geometry_ious"]
        )
    _, _, _, original = _materialized(rows)
    _, _, _, modified = _materialized(changed)

    original_targets = build_selective_pair_targets(
        torch.stack([record["candidate_ious"] for record in original]),
        torch.stack([record["pair_valid"] for record in original])
        | torch.nn.functional.one_hot(
            torch.tensor([record["baseline_index"] for record in original]),
            num_classes=112,
        ).bool(),
        torch.tensor([record["baseline_index"] for record in original]),
    )
    modified_targets = build_selective_pair_targets(
        torch.stack([record["candidate_ious"] for record in modified]),
        torch.stack([record["pair_valid"] for record in modified])
        | torch.nn.functional.one_hot(
            torch.tensor([record["baseline_index"] for record in modified]),
            num_classes=112,
        ).bool(),
        torch.tensor([record["baseline_index"] for record in modified]),
    )

    assert not torch.equal(original_targets, modified_targets)
    for before, after in zip(original, modified):
        assert torch.equal(before["pair_features"], after["pair_features"])
        assert torch.equal(before["pair_valid"], after["pair_valid"])
        assert before["baseline_index"] == after["baseline_index"]
        assert torch.equal(before["baseline_scores"], after["baseline_scores"])
        assert not torch.equal(before["candidate_ious"], after["candidate_ious"])


def test_materialization_digests_are_ordered_deterministic_and_tamper_bound():
    _, _, _, first = _materialized()
    _, _, _, second = _materialized()

    first_sha = canonical_residual_rows_sha256(first)
    assert first_sha == canonical_residual_rows_sha256(second)
    assert len(first_sha) == 64
    selected = torch.tensor([
        record["candidate_ious"][record["baseline_index"]]
        for record in first
    ], dtype=torch.float32)
    expected_selected_sha = hashlib.sha256(
        selected.contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    assert canonical_selected_iou_sha256(first) == expected_selected_sha

    changed = copy.deepcopy(first)
    changed[0]["pair_features"][0, 0] += 1.0
    assert canonical_residual_rows_sha256(changed) != first_sha


@pytest.mark.parametrize(
    "mutation",
    ["reordered", "missing_index", "identity", "feature_dtype", "schema"],
)
def test_materialization_rejects_row_order_identity_dtype_and_schema(mutation):
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    artifact = _geometry_artifact()
    if mutation == "reordered":
        rows.reverse()
    elif mutation == "missing_index":
        rows[1]["base"]["dataset_index"] = 2
        rows[1]["geometry"]["dataset_index"] = 2
    elif mutation == "identity":
        rows[0]["geometry"]["target_id"] += 1
    elif mutation == "feature_dtype":
        rows[0]["geometry"]["geometry_features"] = rows[0][
            "geometry"
        ]["geometry_features"].double()
    elif mutation == "schema":
        artifact["feature_names"][0] = "changed"

    with pytest.raises((TypeError, ValueError)):
        materialize_residual_rows(
            rows,
            _parent(),
            RecordingGeometryModel(),
            artifact,
            batch_size=1,
            device="cpu",
        )


@pytest.mark.parametrize("mutation", ["training", "requires_grad", "dtype"])
def test_materialization_rejects_mutating_or_malformed_geometry_model(mutation):
    class BadGeometryModel(RecordingGeometryModel):
        def forward(self, features, valid_mask):
            outputs = super().forward(features, valid_mask)
            if mutation == "training":
                self.train()
            elif mutation == "requires_grad":
                self.anchor.requires_grad_(True)
            elif mutation == "dtype":
                outputs = {
                    key: value.double() for key, value in outputs.items()
                }
            return outputs

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        materialize_residual_rows(
            [_joined_row(0, "scene_a")],
            _parent(),
            BadGeometryModel(),
            _geometry_artifact(),
            batch_size=1,
            device="cpu",
        )


def _create_input_paths(tmp_path):
    base_cache = tmp_path / "train_cache"
    geometry_cache = tmp_path / "geometry_train"
    base_cache.mkdir()
    geometry_cache.mkdir()
    parent_path = tmp_path / "parent.pth"
    geometry_path = tmp_path / "geometry.pth"
    parent_path.write_bytes(b"parent")
    geometry_path.write_bytes(b"geometry")
    return base_cache, geometry_cache, parent_path, geometry_path


def _install_loader_stubs(monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    paths = _create_input_paths(tmp_path)
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    base_manifest = {
        "split": "train",
        "checkpoint_sha256": AUTHORITATIVE_BACKBONE_SHA256,
        "sample_count": 2,
    }
    geometry_manifest = {
        "split": "train",
        "sample_count": 2,
        "base_cache_binding": {
            "content_sha256": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        },
        "cache_content_digest": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "immutable_metadata_digest": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }
    parent = _parent()
    parent[0]._artifact_sha256 = AUTHORITATIVE_PARENT_ARTIFACT_SHA256
    geometry_model = RecordingGeometryModel()
    geometry_model._artifact_sha256 = AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256
    artifact = _geometry_artifact()
    calls = []
    lifecycle = []

    def load_training(base_cache, geometry_cache, parent_path):
        calls.append(("load_training", base_cache, geometry_cache, parent_path))
        return rows, base_manifest, geometry_manifest, parent

    def load_geometry(path, **kwargs):
        calls.append(("load_geometry", path, kwargs))
        return geometry_model, artifact

    hashes = {
        str(paths[2].resolve()): AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        str(paths[3].resolve()): AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    }
    monkeypatch.setattr(trainer, "load_geometry_training_data", load_training)
    monkeypatch.setattr(trainer, "load_geometry_reranker_artifact", load_geometry)
    monkeypatch.setattr(
        trainer, "_stable_file_sha256", lambda path: hashes[str(path)]
    )
    return paths, rows, base_manifest, geometry_manifest, calls


def test_train_only_loader_binds_all_immutable_inputs(monkeypatch, tmp_path):
    paths, rows, base_manifest, geometry_manifest, calls = \
        _install_loader_stubs(monkeypatch, tmp_path)

    loaded = load_residual_training_inputs(*paths, device="cpu")

    assert loaded["joined_rows"] is rows
    assert loaded["base_manifest"] is base_manifest
    assert loaded["geometry_manifest"] is geometry_manifest
    assert loaded["input_sha256"] == {
        "backbone": AUTHORITATIVE_BACKBONE_SHA256,
        "parent": AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        "geometry": AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        "geometry_cache_content": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }
    assert [call[0] for call in calls] == [
        "load_training", "load_geometry"
    ]


@pytest.mark.parametrize(
    "forbidden", ["val", "validation", "official", "claim", "receipt"]
)
def test_train_paths_block_forbidden_components(
        forbidden, monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    paths = list(_create_input_paths(tmp_path))
    blocked = tmp_path / forbidden / "train"
    blocked.mkdir(parents=True)
    paths[0] = blocked
    monkeypatch.setattr(
        trainer,
        "load_geometry_training_data",
        lambda *_args, **_kwargs: pytest.fail("loader must not be called"),
    )

    with pytest.raises(ValueError, match="forbidden"):
        load_residual_training_inputs(*paths, device="cpu")


@pytest.mark.parametrize(
    "mutation",
    [
        "base_split",
        "geometry_split",
        "backbone_sha",
        "base_content",
        "geometry_content",
        "geometry_metadata",
        "parent_sha",
        "geometry_sha",
    ],
)
def test_train_only_loader_rejects_manifest_or_artifact_tamper(
        mutation, monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    paths, _rows, base_manifest, geometry_manifest, _calls = \
        _install_loader_stubs(monkeypatch, tmp_path)
    if mutation == "base_split":
        base_manifest["split"] = "other"
    elif mutation == "geometry_split":
        geometry_manifest["split"] = "other"
    elif mutation == "backbone_sha":
        base_manifest["checkpoint_sha256"] = "0" * 64
    elif mutation == "base_content":
        geometry_manifest["base_cache_binding"]["content_sha256"] = "0" * 64
    elif mutation == "geometry_content":
        geometry_manifest["cache_content_digest"] = "0" * 64
    elif mutation == "geometry_metadata":
        geometry_manifest["immutable_metadata_digest"] = "0" * 64
    elif mutation == "parent_sha":
        original = trainer._stable_file_sha256
        monkeypatch.setattr(
            trainer,
            "_stable_file_sha256",
            lambda path: "0" * 64 if Path(path) == paths[2].resolve()
            else original(path),
        )
    elif mutation == "geometry_sha":
        original = trainer._stable_file_sha256
        monkeypatch.setattr(
            trainer,
            "_stable_file_sha256",
            lambda path: "0" * 64 if Path(path) == paths[3].resolve()
            else original(path),
        )

    with pytest.raises(ValueError):
        load_residual_training_inputs(*paths, device="cpu")


def _synthetic_residual_record(index, scan_id, kind):
    alternative_index = 7
    pair_features = torch.zeros(112, 185, dtype=torch.float32)
    pair_valid = torch.zeros(112, dtype=torch.bool)
    pair_valid[alternative_index] = True
    candidate_ious = torch.zeros(112, dtype=torch.float32)
    baseline_scores = torch.full((112,), -float("inf"), dtype=torch.float32)
    baseline_scores[0] = 1.0
    baseline_scores[alternative_index] = 0.0
    if kind == "fix":
        pair_features[alternative_index, 0] = 10.0
        candidate_ious[0] = 0.10
        candidate_ious[alternative_index] = 0.90
    elif kind == "break":
        pair_features[alternative_index, 0] = -10.0
        candidate_ious[0] = 0.90
        candidate_ious[alternative_index] = 0.10
    else:
        raise ValueError(kind)
    query_positions = torch.arange(112, dtype=torch.long).div(
        7, rounding_mode="floor"
    )
    variant_indices = torch.arange(112, dtype=torch.long).remainder(7)
    return {
        "dataset_index": index,
        "scan_id": scan_id,
        "target_id": index,
        "pair_features": pair_features,
        "pair_valid": pair_valid,
        "candidate_ious": candidate_ious,
        "baseline_index": 0,
        "baseline_scores": baseline_scores,
        "query_positions": query_positions,
        "variant_indices": variant_indices,
    }


def _separable_fit_records(scene_count=15):
    records = []
    for scene_index in range(scene_count):
        scan_id = "fit_scene_{:02d}".format(scene_index)
        for kind in ("fix", "fix", "break"):
            records.append(_synthetic_residual_record(
                len(records), scan_id, kind
            ))
    return records


def _label_summary_records():
    specifications = (
        (0.40, (0.60, 0.10, 0.40)),
        (0.10, (0.40, 0.10, 0.60)),
        (0.60, (0.10, 0.40, 0.60)),
    )
    records = []
    alternative_indices = (1, 7, 8)
    for index, (baseline_iou, alternative_ious) in enumerate(specifications):
        record = _synthetic_residual_record(
            index, "label_scene_{:02d}".format(index), "fix"
        )
        record["pair_valid"].zero_()
        record["pair_valid"][list(alternative_indices)] = True
        record["candidate_ious"].zero_()
        record["candidate_ious"][0] = baseline_iou
        record["candidate_ious"][list(alternative_indices)] = torch.tensor(
            alternative_ious, dtype=torch.float32
        )
        record["baseline_scores"].fill_(-float("inf"))
        record["baseline_scores"][0] = 1.0
        record["baseline_scores"][list(alternative_indices)] = 0.0
        records.append(record)
    return records


def test_residual_training_label_summary_partitions_query_and_iou_tiers():
    import scripts.train_scanrefer_rec_selective_residual as trainer

    summary = trainer.summarize_residual_training_labels(
        _label_summary_records()
    )

    assert summary == {
        "all": {
            "0.25": {"break": 2, "neutral": 5, "fix": 2, "total": 9},
            "0.50": {"break": 2, "neutral": 5, "fix": 2, "total": 9},
        },
        "same_query": {
            "0.25": {"break": 1, "neutral": 1, "fix": 1, "total": 3},
            "0.50": {"break": 1, "neutral": 1, "fix": 1, "total": 3},
        },
        "different_query": {
            "0.25": {"break": 1, "neutral": 4, "fix": 1, "total": 6},
            "0.50": {"break": 1, "neutral": 4, "fix": 1, "total": 6},
        },
    }
    for group in summary.values():
        assert set(group) == {"0.25", "0.50"}
        for threshold in group.values():
            assert set(threshold) == {"break", "neutral", "fix", "total"}
            assert sum(threshold[name] for name in (
                "break", "neutral", "fix"
            )) == threshold["total"]
    json.dumps(summary, allow_nan=False, sort_keys=True)


def test_oof_pair_gain_summary_uses_fixed_nearest_rank_quantiles():
    import scripts.train_scanrefer_rec_selective_residual as trainer

    pair_gain = torch.zeros(2, 112, dtype=torch.float32)
    pair_valid = torch.zeros(2, 112, dtype=torch.bool)
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    pair_gain[0, :3] = values[:3]
    pair_gain[1, :3] = values[3:]
    pair_valid[:, :3] = True

    summary = trainer.summarize_oof_pair_gain(pair_gain, pair_valid)

    assert trainer.RESIDUAL_GAIN_QUANTILES == (
        0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0,
    )
    assert set(summary) == {"valid", "positive"}
    assert summary["valid"]["count"] == 6
    assert summary["positive"]["count"] == 3
    valid_statistics = summary["valid"]["statistics"]
    positive_statistics = summary["positive"]["statistics"]
    assert set(valid_statistics) == {
        "minimum",
        "maximum",
        "mean",
        "population_standard_deviation",
        "nearest_rank_quantiles",
    }
    assert valid_statistics["minimum"] == -2.0
    assert valid_statistics["maximum"] == 4.0
    assert valid_statistics["mean"] == pytest.approx(values.mean().item())
    assert valid_statistics["population_standard_deviation"] == \
        pytest.approx(values.std(unbiased=False).item())
    assert valid_statistics["nearest_rank_quantiles"] == [
        {"quantile": 0.0, "value": -2.0},
        {"quantile": 0.01, "value": -2.0},
        {"quantile": 0.05, "value": -2.0},
        {"quantile": 0.25, "value": -1.0},
        {"quantile": 0.50, "value": 0.0},
        {"quantile": 0.75, "value": 2.0},
        {"quantile": 0.95, "value": 4.0},
        {"quantile": 0.99, "value": 4.0},
        {"quantile": 1.0, "value": 4.0},
    ]
    assert positive_statistics["nearest_rank_quantiles"] == [
        {"quantile": 0.0, "value": 1.0},
        {"quantile": 0.01, "value": 1.0},
        {"quantile": 0.05, "value": 1.0},
        {"quantile": 0.25, "value": 1.0},
        {"quantile": 0.50, "value": 2.0},
        {"quantile": 0.75, "value": 4.0},
        {"quantile": 0.95, "value": 4.0},
        {"quantile": 0.99, "value": 4.0},
        {"quantile": 1.0, "value": 4.0},
    ]
    json.dumps(summary, allow_nan=False, sort_keys=True)


def test_oof_pair_gain_summary_does_not_fabricate_empty_positive_stats():
    import scripts.train_scanrefer_rec_selective_residual as trainer

    pair_gain = torch.zeros(1, 112, dtype=torch.float32)
    pair_gain[0, :2] = torch.tensor([-2.0, 0.0])
    pair_valid = torch.zeros(1, 112, dtype=torch.bool)
    pair_valid[0, :2] = True

    summary = trainer.summarize_oof_pair_gain(pair_gain, pair_valid)

    assert summary["valid"]["count"] == 2
    assert summary["valid"]["statistics"] is not None
    assert summary["positive"] == {"count": 0, "statistics": None}


@pytest.mark.parametrize("mutation", (
    "gain_dtype", "valid_dtype", "shape", "nonfinite_valid",
))
def test_oof_pair_gain_summary_rejects_noncanonical_inputs(mutation):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    pair_gain = torch.zeros(1, 112, dtype=torch.float32)
    pair_valid = torch.ones(1, 112, dtype=torch.bool)
    if mutation == "gain_dtype":
        pair_gain = pair_gain.double()
    elif mutation == "valid_dtype":
        pair_valid = pair_valid.long()
    elif mutation == "shape":
        pair_valid = pair_valid[:, :-1]
    elif mutation == "nonfinite_valid":
        pair_gain[0, 0] = float("nan")

    with pytest.raises(ValueError):
        trainer.summarize_oof_pair_gain(pair_gain, pair_valid)


def test_cross_fit_is_scene_disjoint_complete_and_exactly_ten_epochs():
    records = _separable_fit_records()
    optimizer_batches = []
    selection_inputs = []

    from models.rec_selective_residual import choose_selective_configuration

    def observe(event):
        optimizer_batches.append(copy.deepcopy(event))

    def select(candidates):
        selection_inputs.extend(copy.deepcopy(candidates))
        return choose_selective_configuration(candidates)

    result = cross_fit_selective_residual(
        records,
        device="cpu",
        batch_observer=observe,
        selector=select,
    )

    assert RESIDUAL_EPOCHS == 10
    assert RESIDUAL_BATCH_SIZE == 256
    assert RESIDUAL_LEARNING_RATE == 3e-4
    assert len(result["configurations"]) == 12
    assert len({
        (record["hidden_dim"], record["weight_decay"], record["break_cost"])
        for record in result["configurations"]
    }) == 12
    for configuration_index, configuration in enumerate(
            result["configurations"]):
        assert configuration["configuration_index"] == configuration_index
        assert "oof_pair_gain" not in configuration
        assert configuration["gain_summary"]["valid"]["count"] == len(records)
        assert configuration["prediction_count"] == len(records)
        assert len(configuration["folds"]) == 5
        for fold in configuration["folds"]:
            labels = fold["training_labels"]
            for threshold in ("0.25", "0.50"):
                assert labels["all"][threshold]["total"] == \
                    fold["fit_row_count"]
                assert labels["same_query"][threshold]["total"] == 0
                assert labels["different_query"][threshold]["total"] == \
                    fold["fit_row_count"]
    assert result["choice"]["eligible"] is True
    assert result["choice"]["delta_hits025"] > 0
    assert result["choice"]["delta_hits050"] >= 0
    assert selection_inputs
    assert {scan_id for candidate in selection_inputs
            for scan_id in candidate["scan_ids"]} == {
        record["scan_id"] for record in records
    }
    assert all("calibration" not in key.lower()
               for candidate in selection_inputs for key in candidate)

    mapping = result["scene_folds"]
    expected_batches = 12 * 5 * RESIDUAL_EPOCHS
    assert len(optimizer_batches) == expected_batches
    for event in optimizer_batches:
        assert event["phase"] == "cross_fit"
        assert event["epoch"] in range(RESIDUAL_EPOCHS)
        assert all(
            mapping[scan_id] != event["held_out_fold"]
            for scan_id in event["scan_ids"]
        )
    for config_index in range(12):
        for fold in range(5):
            assert {
                event["epoch"] for event in optimizer_batches
                if event["config_index"] == config_index
                and event["held_out_fold"] == fold
            } == set(range(RESIDUAL_EPOCHS))


def _calibration_metrics(hits025=3524, hits050=3315):
    return {
        "sample_count": 3625,
        "hits025": hits025,
        "hits050": hits050,
        "baseline_hits025": 3461,
        "baseline_hits050": 3315,
        "oracle_hits025": 3606,
        "oracle_hits050": 3588,
        "candidate_iou_sha256": "a" * 64,
        "row_materialization_sha256": "b" * 64,
    }


def _calibration_baseline():
    return {
        "sample_count": 3625,
        "hits025": 3461,
        "hits050": 3315,
        "oracle_hits025": 3606,
        "oracle_hits050": 3588,
        "candidate_iou_sha256": "a" * 64,
        "row_materialization_sha256": "b" * 64,
    }


def test_cache_calibration_gate_has_exact_boundaries_and_invariants():
    baseline = _calibration_baseline()

    passing = calibration_gate(_calibration_metrics(), baseline)
    below025 = calibration_gate(_calibration_metrics(hits025=3523), baseline)
    below050 = calibration_gate(_calibration_metrics(hits050=3314), baseline)

    assert passing.passed is True
    assert passing.failures == ()
    assert passing.required_hits025 == 3524
    assert passing.required_hits050 == 3315
    assert below025.passed is False
    assert "hits025" in below025.failures
    assert below050.passed is False
    assert "hits050" in below050.failures


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("sample_count", 3624),
        ("baseline_hits025", 3460),
        ("baseline_hits050", 3314),
        ("oracle_hits025", 3605),
        ("oracle_hits050", 3587),
        ("candidate_iou_sha256", "c" * 64),
        ("row_materialization_sha256", "d" * 64),
    ],
)
def test_cache_calibration_gate_rejects_frozen_state_drift(field, bad_value):
    metrics = _calibration_metrics()
    metrics[field] = bad_value

    result = calibration_gate(metrics, _calibration_baseline())

    assert result.passed is False
    assert field in result.failures


class SignedGainModel(torch.nn.Module):
    def forward(self, pair_features, pair_valid):
        logits = pair_features.new_zeros(
            pair_features.shape[0], pair_features.shape[1], 2, 3
        )
        positive = pair_features[..., 0] > 0.0
        negative = pair_features[..., 0] < 0.0
        for head_index in range(2):
            logits[:, :, head_index, 2][positive] = 10.0
            logits[:, :, head_index, 0][negative] = 10.0
        return logits.masked_fill(~pair_valid[:, :, None, None], 0.0)


def test_refit_uses_all_fit_scenes_for_exactly_ten_epochs():
    records = _separable_fit_records()
    events = []
    choice = {
        "eligible": True,
        "hidden_dim": 0,
        "weight_decay": 1e-3,
        "break_cost": 8.0,
        "margin": 0.1,
        "margin_percentile": 95.0,
    }

    model = refit_selective_residual(
        records, choice, device="cpu", batch_observer=events.append
    )

    assert isinstance(model, SelectiveResidualModel)
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert len(events) == RESIDUAL_EPOCHS
    assert {event["epoch"] for event in events} == set(range(10))
    assert all(event["phase"] == "refit" for event in events)
    assert all(set(event["scan_ids"]) <= {
        record["scan_id"] for record in records
    } for event in events)


def test_policy_evaluation_reports_exact_cache_diagnostics_and_digests():
    records = _separable_fit_records()
    baseline = build_cache_calibration_baseline(records)

    metrics = evaluate_selective_residual_policy(
        SignedGainModel(), records, margin=0.1, device="cpu"
    )

    assert metrics["sample_count"] == 45
    assert metrics["baseline_hits025"] == 15
    assert metrics["baseline_hits050"] == 15
    assert metrics["hits025"] == 45
    assert metrics["hits050"] == 45
    assert metrics["oracle_hits025"] == 45
    assert metrics["oracle_hits050"] == 45
    assert metrics["fixes025"] == metrics["fixes050"] == 30
    assert metrics["breaks025"] == metrics["breaks050"] == 0
    assert metrics["switches"] == 30
    assert metrics["abstentions"] == 15
    assert metrics["wrong_query_recoveries025"] == 30
    assert metrics["wrong_variant_recoveries025"] == 0
    assert metrics["recoverable_misses025"] == 30
    assert metrics["recovered_misses025"] == 30
    assert metrics["recoverable_miss_coverage025"] == 1.0
    assert metrics["bootstrap025"]["lower_bound_95"] == 30
    assert metrics["bootstrap050"]["lower_bound_95"] == 30
    assert set(metrics["per_scene_deltas"]) == {
        record["scan_id"] for record in records
    }
    assert all(value == {"hits025": 2, "hits050": 2}
               for value in metrics["per_scene_deltas"].values())
    assert metrics["candidate_iou_sha256"] == baseline[
        "candidate_iou_sha256"
    ]
    assert metrics["row_materialization_sha256"] == baseline[
        "row_materialization_sha256"
    ]
    assert len(metrics["selected_iou_sha256"]) == 64
    assert metrics["baseline_selected_iou_sha256"] == \
        canonical_selected_iou_sha256(records)


def test_fixed_scene_split_is_disjoint_and_preserves_canonical_row_order():
    records = _separable_fit_records(scene_count=20)

    split = split_residual_records(records)

    fit = split["fit_records"]
    calibration = split["calibration_records"]
    fit_scenes = {record["scan_id"] for record in fit}
    calibration_scenes = {record["scan_id"] for record in calibration}
    assert len(fit_scenes) == 18
    assert len(calibration_scenes) == 2
    assert fit_scenes.isdisjoint(calibration_scenes)
    assert [record["dataset_index"] for record in fit] == sorted(
        record["dataset_index"] for record in fit
    )
    assert [record["dataset_index"] for record in calibration] == sorted(
        record["dataset_index"] for record in calibration
    )
    assert split["metadata"]["split_seed"] == 0
    assert split["metadata"]["calibration_fraction"] == 0.1


def _authoritative_joined_identity_rows(monkeypatch, trainer):
    scenes = ["identity_scene_{:03d}".format(index) for index in range(562)]
    shuffled = list(scenes)
    random.Random(0).shuffle(shuffled)
    calibration_scenes = sorted(shuffled[:56])
    fit_scenes = sorted(set(scenes).difference(calibration_scenes))
    row_counts = {scene_id: 1 for scene_id in scenes}
    row_counts[fit_scenes[0]] += 33040 - len(fit_scenes)
    row_counts[calibration_scenes[0]] += 3625 - len(calibration_scenes)
    rows = []
    for scene_id in scenes:
        for _ in range(row_counts[scene_id]):
            dataset_index = len(rows)
            rows.append({
                "base": {
                    "dataset_index": dataset_index,
                    "scan_id": scene_id,
                },
                "geometry": {
                    "dataset_index": dataset_index,
                    "scan_id": scene_id,
                },
            })
    metadata = {
        "split_seed": 0,
        "calibration_fraction": 0.10,
        "scene_count": 562,
        "fit_scene_count": 506,
        "calibration_scene_count": 56,
        "sample_count": 36665,
        "fit_sample_count": 33040,
        "calibration_sample_count": 3625,
        "fit_scene_sha256": trainer.canonical_json_sha256(fit_scenes),
        "calibration_scene_sha256": trainer.canonical_json_sha256(
            calibration_scenes
        ),
        "mapping_sha256": trainer.canonical_json_sha256({
            "fit": fit_scenes,
            "calibration": calibration_scenes,
        }),
    }
    monkeypatch.setattr(trainer, "AUTHORITATIVE_SPLIT_SEED0", metadata)
    return rows, set(fit_scenes), set(calibration_scenes), metadata


def test_joined_row_split_is_authoritative_and_identity_only(
        monkeypatch):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    rows, fit_scenes, calibration_scenes, metadata = \
        _authoritative_joined_identity_rows(monkeypatch, trainer)

    split = trainer.split_residual_joined_rows(rows)

    assert split["metadata"] == metadata
    assert len(split["fit_rows"]) == 33040
    assert len(split["calibration_rows"]) == 3625
    assert {row["base"]["scan_id"] for row in split["fit_rows"]} == \
        fit_scenes
    assert {row["base"]["scan_id"]
            for row in split["calibration_rows"]} == calibration_scenes
    assert fit_scenes.isdisjoint(calibration_scenes)
    for key in ("fit_rows", "calibration_rows"):
        indices = [row["base"]["dataset_index"] for row in split[key]]
        assert indices == sorted(indices)


def _artifact_input_sha256():
    return {
        "backbone": AUTHORITATIVE_BACKBONE_SHA256,
        "parent": AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        "geometry": AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        "geometry_cache_content": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }


def _artifact_selection():
    return {
        "eligible": True,
        "selected": "residual",
        "hidden_dim": 0,
        "weight_decay": 1e-3,
        "break_cost": 8.0,
        "margin_percentile": 95.0,
        "margin": 0.1,
        "switches": 30,
        "delta_hits025": 30,
        "delta_hits050": 30,
    }


def _staged_artifact():
    scene_folds = build_residual_scene_folds([
        "artifact_scene_{:02d}".format(index) for index in range(10)
    ])
    feature_names = build_selective_pair_feature_names(
        _geometry_artifact()["feature_names"]
    )
    return build_selective_residual_artifact(
        model=SelectiveResidualModel(hidden_dim=0),
        selection=_artifact_selection(),
        scene_folds=scene_folds,
        feature_names=feature_names,
        input_sha256=_artifact_input_sha256(),
        row_materialization_sha256="b" * 64,
        oof_record={
            "prediction_count": 45,
            "pair_gain_sha256": "c" * 64,
            "delta_hits025": 30,
            "delta_hits050": 30,
        },
        calibration_record=_calibration_metrics(),
        calibration_baseline=_calibration_baseline(),
    )


def test_staged_artifact_round_trips_strictly_and_binds_provenance(tmp_path):
    artifact = _staged_artifact()
    feature_names = build_selective_pair_feature_names(
        _geometry_artifact()["feature_names"]
    )
    path = tmp_path / "selected_residual.pth"

    save_selective_residual_artifact(path, artifact)
    model, loaded = load_selective_residual_artifact(
        path,
        device="cpu",
        parent_sha256=AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        geometry_sha256=AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    )
    validated = validate_selective_residual_artifact(
        loaded,
        AUTHORITATIVE_BACKBONE_SHA256,
        AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        feature_names,
    )

    assert validated == {"input_dim": 185, "hidden_dim": 0, "dropout": 0.1}
    assert artifact["schema"] == "rec-selective-residual-v1"
    assert artifact["deployable"] is False
    assert artifact["validation_data_accessed"] is False
    assert loaded["input_sha256"] == _artifact_input_sha256()
    assert isinstance(model, SelectiveResidualModel)
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(FileExistsError):
        save_selective_residual_artifact(path, artifact)
    with pytest.raises(ValueError, match="parent"):
        load_selective_residual_artifact(
            path, parent_sha256="0" * 64,
            geometry_sha256=AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "deployable",
        "validation",
        "backbone",
        "feature_names",
        "fold_hash",
        "margin",
        "calibration",
        "state",
    ],
)
def test_staged_artifact_rejects_contract_tampering(mutation):
    artifact = _staged_artifact()
    feature_names = list(artifact["feature_names"])
    if mutation == "deployable":
        artifact["deployable"] = True
    elif mutation == "validation":
        artifact["validation_data_accessed"] = True
    elif mutation == "backbone":
        artifact["input_sha256"]["backbone"] = "0" * 64
    elif mutation == "feature_names":
        artifact["feature_names"][0] = "changed"
    elif mutation == "fold_hash":
        artifact["scene_fold_sha256"] = "0" * 64
    elif mutation == "margin":
        artifact["selection"]["margin"] = 0.2
    elif mutation == "calibration":
        artifact["calibration_record"]["hits025"] = 3523
    elif mutation == "state":
        artifact["model_state_dict"]["head.bias"] = torch.zeros(5)

    with pytest.raises((TypeError, ValueError)):
        validate_selective_residual_artifact(
            artifact,
            AUTHORITATIVE_BACKBONE_SHA256,
            AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
            AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
            feature_names,
        )


def _receipt_label_summary(pair_count):
    return {
        "all": {
            "0.25": {
                "break": 0, "neutral": pair_count, "fix": 0,
                "total": pair_count,
            },
            "0.50": {
                "break": 0, "neutral": pair_count, "fix": 0,
                "total": pair_count,
            },
        },
        "same_query": {
            "0.25": {"break": 0, "neutral": 0, "fix": 0, "total": 0},
            "0.50": {"break": 0, "neutral": 0, "fix": 0, "total": 0},
        },
        "different_query": {
            "0.25": {
                "break": 0, "neutral": pair_count, "fix": 0,
                "total": pair_count,
            },
            "0.50": {
                "break": 0, "neutral": pair_count, "fix": 0,
                "total": pair_count,
            },
        },
    }


def _zero_gain_summary(trainer, count):
    quantiles = [
        {"quantile": value, "value": 0.0}
        for value in trainer.RESIDUAL_GAIN_QUANTILES
    ]
    return {
        "valid": {
            "count": count,
            "statistics": {
                "minimum": 0.0,
                "maximum": 0.0,
                "mean": 0.0,
                "population_standard_deviation": 0.0,
                "nearest_rank_quantiles": quantiles,
            },
        },
        "positive": {"count": 0, "statistics": None},
    }


def _no_switch_diagnostic(config, sample_count, hits025, hits050):
    predicates = {
        "not_no_switch": False,
        "all_folds_nonnegative025": True,
        "all_folds_nonnegative050": True,
        "pooled_delta025_positive": False,
        "bootstrap025_lower_bound_positive": False,
        "bootstrap050_lower_bound_nonnegative": True,
    }
    return {
        "hidden_dim": config["hidden_dim"],
        "weight_decay": config["weight_decay"],
        "break_cost": config["break_cost"],
        "margin_percentile": None,
        "margin": None,
        "no_switch": True,
        "sample_count": sample_count,
        "switches": 0,
        "abstentions": sample_count,
        "switch_rate": 0.0,
        "baseline": {
            "0.25": {"hits": hits025},
            "0.50": {"hits": hits050},
        },
        "proposed": {
            "0.25": {"hits": hits025},
            "0.50": {"hits": hits050},
        },
        "effects": {
            "0.25": {
                "fixes": 0,
                "breaks": 0,
                "neutral_switches": 0,
                "kept_correct": hits025,
                "kept_wrong": sample_count - hits025,
            },
            "0.50": {
                "fixes": 0,
                "breaks": 0,
                "neutral_switches": 0,
                "kept_correct": hits050,
                "kept_wrong": sample_count - hits050,
            },
        },
        "delta_hits025": 0,
        "delta_hits050": 0,
        "fold_deltas": {
            str(fold): {"hits025": 0, "hits050": 0}
            for fold in range(5)
        },
        "bootstrap025": {
            "confidence": 0.95,
            "delta_hits": 0,
            "lower_bound_95": 0,
            "replicates": 10000,
            "scene_count": 506,
            "seed": 0,
        },
        "bootstrap050": {
            "confidence": 0.95,
            "delta_hits": 0,
            "lower_bound_95": 0,
            "replicates": 10000,
            "scene_count": 506,
            "seed": 0,
        },
        "eligibility_predicates": predicates,
        "failed_predicates": sorted(
            name for name, passed in predicates.items() if not passed
        ),
        "eligible": False,
        "selected": "baseline",
    }


def _rejected_result_context(trainer, protected):
    sample_count = 33040
    pair_count = sample_count * 3
    hits025 = 20000
    hits050 = 16000
    scene_folds = {
        "receipt_fit_scene_{:03d}".format(index): index % 5
        for index in range(506)
    }
    held_scene_counts = (102, 101, 101, 101, 101)
    held_row_counts = (6608, 6608, 6608, 6608, 6608)
    grid = [
        {
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "break_cost": break_cost,
        }
        for hidden_dim in trainer.RESIDUAL_HIDDEN_DIMS
        for weight_decay in trainer.RESIDUAL_WEIGHT_DECAYS
        for break_cost in trainer.RESIDUAL_BREAK_COSTS
    ]
    diagnostics = [
        _no_switch_diagnostic(
            config, sample_count, hits025, hits050
        ) for config in grid
    ]
    configurations = []
    for configuration_index, config in enumerate(grid):
        folds = []
        for fold in range(5):
            fit_row_count = sample_count - held_row_counts[fold]
            fit_pair_count = fit_row_count * 3
            folds.append({
                "fold": fold,
                "fit_scene_count": 506 - held_scene_counts[fold],
                "fit_row_count": fit_row_count,
                "fit_pair_count": fit_pair_count,
                "held_scene_count": held_scene_counts[fold],
                "held_row_count": held_row_counts[fold],
                "training_labels": _receipt_label_summary(fit_pair_count),
            })
        record = copy.deepcopy(config)
        record.update({
            "configuration_index": configuration_index,
            "folds": folds,
            "gain_summary": _zero_gain_summary(trainer, pair_count),
            "oof_pair_gain_sha256": "{:064x}".format(
                configuration_index + 1
            ),
            "prediction_count": sample_count,
        })
        configurations.append(record)
    choice = {
        "candidate_count": len(diagnostics),
        "eligible_candidate_count": 0,
        "candidate_diagnostics": diagnostics,
        "eligible": False,
        "reason": "no-eligible-configuration",
        "selected": "baseline",
    }
    protected_sha = {
        name: protected[name]["sha256"]
        for name in ("backbone", "parent", "geometry")
    }
    return {
        "input_sha256": {
            "backbone": protected_sha["backbone"],
            "parent": protected_sha["parent"],
            "geometry": protected_sha["geometry"],
            "base_cache_content": "a" * 64,
            "geometry_cache_content": "b" * 64,
            "geometry_metadata": "c" * 64,
        },
        "split": copy.deepcopy(trainer.AUTHORITATIVE_SPLIT_SEED0),
        "fit_joined_identity_sha256": "d" * 64,
        "fit_materialization_sha256": "e" * 64,
        "oof": {
            "baseline": {
                "sample_count": sample_count,
                "hits025": hits025,
                "hits050": hits050,
                "oracle_hits025": 25000,
                "oracle_hits050": 22000,
                "candidate_iou_sha256": "f" * 64,
                "row_materialization_sha256": "e" * 64,
                "baseline_selected_iou_sha256": "1" * 64,
            },
            "scene_folds": scene_folds,
            "scene_fold_sha256": trainer.canonical_scene_fold_sha256(
                scene_folds
            ),
            "configuration_count": len(configurations),
            "configurations": configurations,
            "policy_candidate_count": len(diagnostics),
            "choice": choice,
        },
        "calibration": {
            "status": "not_run",
            "reason": "oof_selection_rejected",
        },
    }


def _eligible_result_context(trainer, protected):
    context = _rejected_result_context(trainer, protected)
    choice = context["oof"]["choice"]
    diagnostic = choice["candidate_diagnostics"][0]
    diagnostic.update({
        "margin_percentile": 95.0,
        "margin": 0.1,
        "no_switch": False,
        "switches": 30,
        "abstentions": diagnostic["sample_count"] - 30,
        "switch_rate": 30.0 / diagnostic["sample_count"],
        "delta_hits025": 30,
        "delta_hits050": 30,
        "eligible": True,
        "selected": "residual",
    })
    predicates = {
        name: True for name in diagnostic["eligibility_predicates"]
    }
    diagnostic["eligibility_predicates"] = predicates
    diagnostic["failed_predicates"] = []
    for threshold in ("0.25", "0.50"):
        baseline_hits = diagnostic["baseline"][threshold]["hits"]
        diagnostic["proposed"][threshold]["hits"] = baseline_hits + 30
        diagnostic["effects"][threshold] = {
            "fixes": 30,
            "breaks": 0,
            "neutral_switches": 0,
            "kept_correct": baseline_hits,
            "kept_wrong": (
                diagnostic["sample_count"] - baseline_hits - 30
            ),
        }
    for fold in diagnostic["fold_deltas"].values():
        fold.update(hits025=6, hits050=6)
    for bootstrap in (diagnostic["bootstrap025"],
                      diagnostic["bootstrap050"]):
        bootstrap["delta_hits"] = 30
        bootstrap["lower_bound_95"] = 30
    winner = copy.deepcopy(diagnostic)
    winner.update({
        "candidate_count": choice["candidate_count"],
        "eligible_candidate_count": 1,
        "candidate_diagnostics": choice["candidate_diagnostics"],
    })
    context["oof"]["choice"] = winner
    baseline = _calibration_baseline()
    record = _calibration_metrics()
    gate = calibration_gate(record, baseline)
    context["calibration"] = {
        "status": "run",
        "baseline": baseline,
        "record": record,
        "gate": {
            "passed": gate.passed,
            "failures": list(gate.failures),
            "required_hits025": gate.required_hits025,
            "required_hits050": gate.required_hits050,
            "observed_hits025": gate.observed_hits025,
            "observed_hits050": gate.observed_hits050,
        },
    }
    return context


def _protected_files(tmp_path):
    paths = {}
    for name in ("backbone", "parent", "geometry"):
        path = tmp_path / "{}.pth".format(name)
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o444)
        paths[name] = path
    return paths


def test_rejected_v2_receipt_preserves_measured_oof_diagnostics(tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected_paths = _protected_files(tmp_path)
    protected = capture_immutable_artifact_identities(protected_paths)
    context = _rejected_result_context(trainer, protected)

    receipt = trainer.build_selective_residual_result_receipt(
        context,
        artifact_binding=None,
        protected_before=protected,
        protected_after=protected,
    )

    assert receipt["schema"] == "rec-selective-residual-result-receipt-v2"
    assert receipt["version"] == 2
    assert receipt["selected"] == "baseline"
    assert receipt["deployable"] is False
    assert receipt["report_only"] is False
    assert receipt["eligible_for_model_selection"] is True
    assert receipt["validation_data_accessed"] is False
    assert receipt["calibration"] == {
        "status": "not_run",
        "reason": "oof_selection_rejected",
    }
    assert receipt["oof"]["choice"]["eligible"] is False
    assert len(receipt["oof"]["choice"]["candidate_diagnostics"]) == 12
    assert receipt["oof"]["baseline"]["hits025"] > 0
    assert receipt["oof"]["baseline"]["hits050"] > 0
    assert all(
        fold["fit_pair_count"] == 3 * fold["fit_row_count"]
        for configuration in receipt["oof"]["configurations"]
        for fold in configuration["folds"]
    )
    assert trainer.validate_selective_residual_result_receipt(receipt) == \
        receipt
    json.dumps(receipt, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("mutation", (
    "input_digest",
    "split_digest",
    "fit_identity_digest",
    "fit_materialization_digest",
    "scene_fold_digest",
    "gain_digest",
    "baseline_count",
    "configuration_count",
    "policy_candidate_count",
    "choice_candidate_count",
    "candidate_baseline_count",
    "fold_count",
    "fit_pair_count",
    "calibration_status",
    "protected_snapshot",
    "report_only",
    "eligible_for_model_selection",
    "validation_data_accessed",
))
def test_v2_result_receipt_rejects_nested_tampering(mutation, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected_paths = _protected_files(tmp_path)
    protected = capture_immutable_artifact_identities(protected_paths)
    receipt = trainer.build_selective_residual_result_receipt(
        _rejected_result_context(trainer, protected),
        artifact_binding=None,
        protected_before=protected,
        protected_after=protected,
    )
    if mutation == "input_digest":
        receipt["input_sha256"]["backbone"] = "0" * 64
    elif mutation == "split_digest":
        receipt["split"]["mapping_sha256"] = "0" * 64
    elif mutation == "fit_identity_digest":
        receipt["fit_joined_identity_sha256"] = "0" * 64
    elif mutation == "fit_materialization_digest":
        receipt["fit_materialization_sha256"] = "0" * 64
    elif mutation == "scene_fold_digest":
        receipt["oof"]["scene_fold_sha256"] = "0" * 64
    elif mutation == "gain_digest":
        receipt["oof"]["configurations"][0][
            "oof_pair_gain_sha256"
        ] = "0" * 64
    elif mutation == "baseline_count":
        receipt["oof"]["baseline"]["hits050"] = 20001
    elif mutation == "configuration_count":
        receipt["oof"]["configuration_count"] = 11
    elif mutation == "policy_candidate_count":
        receipt["oof"]["policy_candidate_count"] += 1
    elif mutation == "choice_candidate_count":
        receipt["oof"]["choice"]["candidate_count"] += 1
    elif mutation == "candidate_baseline_count":
        receipt["oof"]["choice"]["candidate_diagnostics"][0][
            "baseline"
        ]["0.25"]["hits"] += 1
    elif mutation == "fold_count":
        receipt["oof"]["configurations"][0]["folds"].pop()
    elif mutation == "fit_pair_count":
        receipt["oof"]["configurations"][0]["folds"][0][
            "fit_pair_count"
        ] += 1
    elif mutation == "calibration_status":
        receipt["calibration"]["status"] = "run"
    elif mutation == "protected_snapshot":
        receipt["protected_after"]["backbone"]["inode"] += 1
    elif mutation == "report_only":
        receipt["report_only"] = True
    elif mutation == "eligible_for_model_selection":
        receipt["eligible_for_model_selection"] = False
    elif mutation == "validation_data_accessed":
        receipt["validation_data_accessed"] = True

    with pytest.raises((TypeError, ValueError)):
        trainer.validate_selective_residual_result_receipt(receipt)


def test_failed_publication_writes_only_readonly_baseline_receipt(tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected = _protected_files(tmp_path)
    before = capture_immutable_artifact_identities(protected)
    context = _rejected_result_context(trainer, before)
    output = tmp_path / "failed_experiment"

    publication = publish_selective_residual_experiment(
        output,
        artifact=None,
        result_context=context,
        protected_paths=protected,
        protected_before=before,
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "result-receipt.json"
    ]
    receipt_path = output / "result-receipt.json"
    assert os.stat(str(receipt_path)).st_mode & 0o777 == 0o444
    receipt = json.loads(receipt_path.read_text("ascii"))
    assert receipt["selected"] == "baseline"
    assert receipt["deployable"] is False
    assert receipt["artifact"] is None
    assert receipt["schema"] == "rec-selective-residual-result-receipt-v2"
    assert receipt["calibration"]["status"] == "not_run"
    assert receipt["oof"]["baseline"]["hits025"] == 20000
    assert receipt["validation_data_accessed"] is False
    assert receipt["protected_before"] == receipt["protected_after"]
    assert publication["artifact_path"] is None
    assert capture_immutable_artifact_identities(protected) == before


def test_success_publication_stages_non_deployable_artifact_and_keeps_best(
        tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected = _protected_files(tmp_path)
    before = capture_immutable_artifact_identities(protected)
    context = _eligible_result_context(trainer, before)
    output = tmp_path / "passed_experiment"

    publication = publish_selective_residual_experiment(
        output,
        artifact=_staged_artifact(),
        result_context=context,
        protected_paths=protected,
        protected_before=before,
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "result-receipt.json", "selected_residual.pth"
    ]
    receipt = json.loads((output / "result-receipt.json").read_text("ascii"))
    assert receipt["selected"] == "staged_residual"
    assert receipt["deployable"] is False
    assert receipt["artifact"]["sha256"] == publication["artifact_sha256"]
    loaded_model, loaded = load_selective_residual_artifact(
        publication["artifact_path"], device="cpu"
    )
    assert isinstance(loaded_model, SelectiveResidualModel)
    assert loaded["deployable"] is False
    assert capture_immutable_artifact_identities(protected) == before


def test_output_reservation_is_exclusive_and_rejects_symlink_components(
        tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    output = tmp_path / "reserved"
    reservation = trainer.reserve_selective_residual_output(output)

    assert reservation["path"] == str(output.absolute())
    assert output.is_dir()
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        trainer.reserve_selective_residual_output(output)

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        trainer.reserve_selective_residual_output(linked_parent / "run")
    assert not (real_parent / "run").exists()


def test_training_reserves_raw_output_before_protected_artifact_access(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    protected_accessed = []

    def capture(_paths):
        protected_accessed.append(True)
        pytest.fail("protected artifacts must not be accessed")

    monkeypatch.setattr(
        trainer, "capture_immutable_artifact_identities", capture
    )
    monkeypatch.setattr(
        trainer,
        "load_residual_training_inputs",
        lambda *_args, **_kwargs: pytest.fail("inputs must not be loaded"),
    )
    monkeypatch.setattr(
        trainer, "AUTHORITATIVE_BACKBONE_PATH", tmp_path / "backbone.pth"
    )

    with pytest.raises(ValueError, match="symlink"):
        run_selective_residual_training(
            tmp_path / "train",
            tmp_path / "geometry_train",
            tmp_path / "parent.pth",
            tmp_path / "geometry.pth",
            linked_parent / "run",
            device="cuda:0",
        )

    assert protected_accessed == []
    assert not (real_parent / "run").exists()


def test_publication_closes_reserved_directory_if_preflight_fails(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    output = tmp_path / "preflight_failure"
    reservation = trainer.reserve_selective_residual_output(output)
    opened = {}
    original = trainer._open_reserved_directory

    def open_directory(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        opened["descriptor"] = descriptor
        return descriptor

    monkeypatch.setattr(trainer, "_open_reserved_directory", open_directory)

    with pytest.raises(ValueError, match="protected paths"):
        publish_selective_residual_experiment(
            output,
            artifact=None,
            result_context={},
            protected_paths={},
            reservation=reservation,
        )

    with pytest.raises(OSError):
        os.fstat(opened["descriptor"])


@pytest.mark.parametrize("target_name", (
    "selected_residual.pth", "result-receipt.json",
))
def test_publication_never_overwrites_target_created_after_reservation(
        target_name, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected_paths = _protected_files(tmp_path)
    protected = capture_immutable_artifact_identities(protected_paths)
    output = tmp_path / "race_output"
    reservation = trainer.reserve_selective_residual_output(output)
    target = output / target_name
    target.write_bytes(b"existing-target")
    context = (
        _eligible_result_context(trainer, protected)
        if target_name == "selected_residual.pth"
        else _rejected_result_context(trainer, protected)
    )
    artifact = (
        _staged_artifact()
        if target_name == "selected_residual.pth" else None
    )

    with pytest.raises(FileExistsError):
        publish_selective_residual_experiment(
            output,
            artifact=artifact,
            result_context=context,
            protected_paths=protected_paths,
            protected_before=protected,
            reservation=reservation,
        )

    assert target.read_bytes() == b"existing-target"
    if target_name == "selected_residual.pth":
        assert not (output / "result-receipt.json").exists()
    assert capture_immutable_artifact_identities(protected_paths) == protected


def test_interrupted_artifact_write_leaves_no_completion_receipt(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected_paths = _protected_files(tmp_path)
    protected = capture_immutable_artifact_identities(protected_paths)
    output = tmp_path / "interrupted_artifact"
    reservation = trainer.reserve_selective_residual_output(output)
    original = trainer._exclusive_write_bytes

    def interrupt(directory_fd, reserved, name, payload, mode=0o444):
        if name == "selected_residual.pth":
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(descriptor, b"partial-artifact")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise RuntimeError("injected artifact interruption")
        return original(directory_fd, reserved, name, payload, mode=mode)

    monkeypatch.setattr(trainer, "_exclusive_write_bytes", interrupt)

    with pytest.raises(RuntimeError, match="artifact interruption"):
        publish_selective_residual_experiment(
            output,
            artifact=_staged_artifact(),
            result_context=_eligible_result_context(trainer, protected),
            protected_paths=protected_paths,
            protected_before=protected,
            reservation=reservation,
        )

    assert (output / "selected_residual.pth").read_bytes() == \
        b"partial-artifact"
    assert not (output / "result-receipt.json").exists()
    assert capture_immutable_artifact_identities(protected_paths) == protected


def test_interrupted_receipt_write_leaves_no_completion_receipt(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    protected_paths = _protected_files(tmp_path)
    protected = capture_immutable_artifact_identities(protected_paths)
    output = tmp_path / "interrupted_receipt"
    reservation = trainer.reserve_selective_residual_output(output)
    original = trainer._exclusive_write_bytes

    def interrupt(directory_fd, reserved, name, payload, mode=0o444):
        if name.startswith(".result-receipt.json.pending"):
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(descriptor, b"partial-receipt")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise RuntimeError("injected receipt interruption")
        return original(directory_fd, reserved, name, payload, mode=mode)

    monkeypatch.setattr(trainer, "_exclusive_write_bytes", interrupt)

    with pytest.raises(RuntimeError, match="receipt interruption"):
        publish_selective_residual_experiment(
            output,
            artifact=None,
            result_context=_rejected_result_context(trainer, protected),
            protected_paths=protected_paths,
            protected_before=protected,
            reservation=reservation,
        )

    assert not (output / "result-receipt.json").exists()
    assert any(path.name.startswith(".result-receipt.json.pending")
               for path in output.iterdir())
    assert capture_immutable_artifact_identities(protected_paths) == protected


def test_trainer_cli_has_only_fixed_inputs_output_and_cuda0():
    args = parse_args([
        "--base-cache", "/tmp/train",
        "--geometry-cache", "/tmp/geometry_train",
        "--parent-artifact", "/tmp/parent.pth",
        "--geometry-artifact", "/tmp/geometry.pth",
        "--output-dir", "/tmp/fresh_output",
        "--device", "cuda:0",
    ])

    assert vars(args) == {
        "base_cache": "/tmp/train",
        "geometry_cache": "/tmp/geometry_train",
        "parent_artifact": "/tmp/parent.pth",
        "geometry_artifact": "/tmp/geometry.pth",
        "output_dir": "/tmp/fresh_output",
        "device": "cuda:0",
    }
    with pytest.raises(SystemExit):
        parse_args([
            "--base-cache", "/tmp/train",
            "--geometry-cache", "/tmp/geometry_train",
            "--parent-artifact", "/tmp/parent.pth",
            "--geometry-artifact", "/tmp/geometry.pth",
            "--output-dir", "/tmp/fresh_output",
            "--device", "cuda:0",
            "--hidden-dim", "64",
        ])
    with pytest.raises(SystemExit):
        parse_args([
            "--base-cache", "/tmp/train",
            "--geometry-cache", "/tmp/geometry_train",
            "--parent-artifact", "/tmp/parent.pth",
            "--geometry-artifact", "/tmp/geometry.pth",
            "--output-dir", "/tmp/fresh_output",
            "--device", "cpu",
        ])


def test_training_orchestration_evaluates_calibration_exactly_once(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    joined_rows, fit_scenes, calibration_scenes, _ = \
        _authoritative_joined_identity_rows(monkeypatch, trainer)
    calls = []
    lifecycle = []
    geometry_artifact = _geometry_artifact()
    choice = _artifact_selection()
    frozen_choice = copy.deepcopy(choice)
    selection_frozen = {"value": False}

    monkeypatch.setattr(
        trainer,
        "capture_immutable_artifact_identities",
        lambda paths: {"captured": sorted(paths)},
    )
    reservation = {
        "path": str((tmp_path / "fresh_output").absolute()),
        "device": 1,
        "inode": 2,
    }

    def reserve(output):
        lifecycle.append("reserve")
        return reservation

    def load(*args, **kwargs):
        lifecycle.append("load")
        return {
            "joined_rows": joined_rows,
            "parent": object(),
            "geometry_model": object(),
            "geometry_artifact": geometry_artifact,
            "input_sha256": _artifact_input_sha256(),
        }

    monkeypatch.setattr(trainer, "reserve_selective_residual_output", reserve)
    monkeypatch.setattr(trainer, "load_residual_training_inputs", load)

    def materialize(received, *args, **kwargs):
        received_scenes = {row["base"]["scan_id"] for row in received}
        if received_scenes == fit_scenes:
            kind = "fit"
            assert len(received) == 33040
            assert selection_frozen["value"] is False
        elif received_scenes == calibration_scenes:
            kind = "calibration"
            assert len(received) == 3625
            assert selection_frozen["value"] is True
            assert choice == frozen_choice
        else:
            pytest.fail("materializer received a mixed or incomplete split")
        assert kwargs["require_contiguous"] is False
        calls.append(("materialize_{}".format(kind), received))
        return received

    monkeypatch.setattr(trainer, "materialize_residual_rows", materialize)
    monkeypatch.setattr(
        trainer,
        "split_residual_records",
        lambda *_args, **_kwargs: pytest.fail(
            "materialized records must not be split"
        ),
    )

    def cross_fit(received, device):
        calls.append(("cross_fit", received))
        selection_frozen["value"] = True
        return {
            "choice": choice,
            "scene_folds": {"fit": 0},
            "scene_fold_sha256": "d" * 64,
            "policy_candidate_count": 1,
            "configurations": [{
                "hidden_dim": 0,
                "weight_decay": 1e-3,
                "break_cost": 8.0,
                "oof_pair_gain_sha256": "c" * 64,
            }],
        }

    def refit(received, selected, device):
        calls.append(("refit", received, selected))
        return SelectiveResidualModel(hidden_dim=0)

    def evaluate(model, received, margin, device):
        calls.append(("evaluate", received, margin))
        return _calibration_metrics()

    monkeypatch.setattr(trainer, "cross_fit_selective_residual", cross_fit)
    monkeypatch.setattr(trainer, "refit_selective_residual", refit)
    monkeypatch.setattr(
        trainer, "build_cache_calibration_baseline",
        lambda received: _calibration_baseline()
    )
    monkeypatch.setattr(
        trainer, "evaluate_selective_residual_policy", evaluate
    )
    monkeypatch.setattr(
        trainer, "canonical_residual_rows_sha256", lambda _rows: "b" * 64
    )
    monkeypatch.setattr(
        trainer, "build_selective_residual_artifact",
        lambda **kwargs: {"staged": kwargs},
    )

    def publish(output, artifact, result_context, protected_paths,
                protected_before, reservation):
        assert reservation is not None
        calls.append(("publish", artifact, result_context))
        return {"published": True, "artifact": artifact}

    monkeypatch.setattr(
        trainer, "publish_selective_residual_experiment", publish
    )
    monkeypatch.setattr(
        trainer, "AUTHORITATIVE_BACKBONE_PATH", tmp_path / "backbone.pth"
    )

    result = run_selective_residual_training(
        tmp_path / "train",
        tmp_path / "geometry_train",
        tmp_path / "parent.pth",
        tmp_path / "geometry.pth",
        tmp_path / "fresh_output",
        device="cuda:0",
    )

    assert result["published"] is True
    assert lifecycle == ["reserve", "load"]
    assert [call[0] for call in calls] == [
        "materialize_fit",
        "cross_fit",
        "refit",
        "materialize_calibration",
        "evaluate",
        "publish",
    ]
    fit_records = calls[0][1]
    calibration_records = calls[3][1]
    assert calls[1][1] is fit_records
    assert calls[2][1] is fit_records
    assert calls[4][1] is calibration_records
    assert sum(call[0] == "evaluate" for call in calls) == 1
    assert calls[-1][1]["staged"]["calibration_record"] == \
        _calibration_metrics()
    assert calls[-1][2]["calibration"]["status"] == "run"
    assert choice == frozen_choice


def test_training_orchestration_stops_before_refit_when_oof_rejects(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_selective_residual as trainer

    joined_rows, fit_scenes, calibration_scenes, _ = \
        _authoritative_joined_identity_rows(monkeypatch, trainer)
    materialized_scene_sets = []
    lifecycle = []
    monkeypatch.setattr(
        trainer,
        "capture_immutable_artifact_identities",
        lambda paths: {"captured": sorted(paths)},
    )
    reservation = {
        "path": str((tmp_path / "fresh_output").absolute()),
        "device": 1,
        "inode": 2,
    }

    def reserve(output):
        lifecycle.append("reserve")
        return reservation

    def load(*args, **kwargs):
        lifecycle.append("load")
        return {
            "joined_rows": joined_rows,
            "parent": object(),
            "geometry_model": object(),
            "geometry_artifact": _geometry_artifact(),
            "input_sha256": _artifact_input_sha256(),
        }

    monkeypatch.setattr(trainer, "reserve_selective_residual_output", reserve)
    monkeypatch.setattr(trainer, "load_residual_training_inputs", load)

    def materialize(received, *args, **kwargs):
        received_scenes = {row["base"]["scan_id"] for row in received}
        materialized_scene_sets.append(received_scenes)
        assert received_scenes == fit_scenes
        assert received_scenes.isdisjoint(calibration_scenes)
        assert len(received) == 33040
        assert kwargs["require_contiguous"] is False
        return received

    monkeypatch.setattr(
        trainer, "materialize_residual_rows", materialize
    )
    monkeypatch.setattr(
        trainer,
        "split_residual_records",
        lambda *_args, **_kwargs: pytest.fail(
            "materialized records must not be split"
        ),
    )
    monkeypatch.setattr(
        trainer,
        "cross_fit_selective_residual",
        lambda received, device: {
            "choice": {"eligible": False, "selected": "baseline"},
            "scene_folds": {},
            "scene_fold_sha256": "d" * 64,
            "policy_candidate_count": 0,
            "configurations": [],
        },
    )
    monkeypatch.setattr(
        trainer,
        "refit_selective_residual",
        lambda *args, **kwargs: pytest.fail("refit must not run"),
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_selective_residual_policy",
        lambda *args, **kwargs: pytest.fail("calibration must not run"),
    )
    monkeypatch.setattr(
        trainer,
        "build_cache_calibration_baseline",
        lambda received: {
            "sample_count": len(received),
            "hits025": 1,
            "hits050": 1,
            "oracle_hits025": 1,
            "oracle_hits050": 1,
            "candidate_iou_sha256": "a" * 64,
            "row_materialization_sha256": "b" * 64,
            "baseline_selected_iou_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        trainer, "canonical_residual_rows_sha256", lambda _rows: "b" * 64
    )

    published = {}

    def publish(output, artifact, result_context, protected_paths,
                protected_before, reservation):
        assert reservation is not None
        published.update(artifact=artifact, context=result_context)
        return {"published": True}

    monkeypatch.setattr(
        trainer, "publish_selective_residual_experiment", publish
    )
    monkeypatch.setattr(
        trainer, "AUTHORITATIVE_BACKBONE_PATH", tmp_path / "backbone.pth"
    )

    result = run_selective_residual_training(
        tmp_path / "train",
        tmp_path / "geometry_train",
        tmp_path / "parent.pth",
        tmp_path / "geometry.pth",
        tmp_path / "fresh_output",
        device="cuda:0",
    )

    assert result == {"published": True}
    assert lifecycle == ["reserve", "load"]
    assert published["artifact"] is None
    assert published["context"]["calibration"] == {
        "status": "not_run",
        "reason": "oof_selection_rejected",
    }
    assert materialized_scene_sets == [fit_scenes]
