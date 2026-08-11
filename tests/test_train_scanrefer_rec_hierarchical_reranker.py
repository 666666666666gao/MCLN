import copy
import io
import json
import math
import os
import stat
from pathlib import Path

import pytest
import torch

from models.rec_hierarchical_reranker import (
    HierarchicalQueryVariantReranker,
    build_hierarchical_scene_folds,
    build_hierarchical_targets,
    canonical_hierarchical_scene_fold_sha256,
    choose_hierarchical_configuration,
)
from scripts.train_scanrefer_rec_selective_residual import (
    capture_immutable_artifact_identities as residual_capture_identities,
    load_residual_training_inputs as residual_load_inputs,
    split_residual_joined_rows as residual_split_rows,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    HIERARCHICAL_MATERIALIZATION_BATCH_SIZE,
    HIERARCHICAL_BATCH_SIZE,
    HIERARCHICAL_DROPOUT,
    HIERARCHICAL_EPOCHS,
    HIERARCHICAL_GAIN_QUANTILES,
    HIERARCHICAL_GRAD_CLIP_NORM,
    HIERARCHICAL_LEARNING_RATE,
    HIERARCHICAL_MODEL_SEED,
    HIERARCHICAL_MIN_STD,
    HIERARCHICAL_MODEL_BATCH_FIELDS,
    HIERARCHICAL_NORMALIZATION_SCHEMA,
    HIERARCHICAL_NORMALIZATION_GROUPS,
    HIERARCHICAL_RECORD_FIELDS,
    _serialize_hierarchical_artifact,
    build_hierarchical_artifact,
    canonical_hierarchical_candidate_iou_sha256,
    canonical_hierarchical_deployable_sha256,
    canonical_hierarchical_rows_sha256,
    build_hierarchical_cache_calibration_baseline,
    build_hierarchical_feature_names,
    build_hierarchical_result_receipt,
    build_hierarchical_policy_candidate,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    cross_fit_hierarchical_reranker,
    load_residual_training_inputs,
    materialize_hierarchical_rows,
    evaluate_hierarchical_cache_policy,
    hierarchical_calibration_gate,
    hierarchical_calibration_gate_receipt,
    load_hierarchical_artifact,
    normalize_hierarchical_batch,
    nearest_rank_hierarchical_margin,
    parse_args,
    publish_hierarchical_experiment,
    refit_hierarchical_reranker,
    reserve_hierarchical_output,
    run_hierarchical_training,
    save_hierarchical_artifact,
    summarize_hierarchical_training_labels,
    split_residual_joined_rows,
    validate_hierarchical_artifact,
    validate_hierarchical_result_receipt,
)
from scripts.train_rec_geometry_reranker import AUTHORITATIVE_SPLIT_SEED0
from test_train_rec_geometry_reranker import _joined_row, _parent
from test_train_scanrefer_rec_selective_residual import (
    RecordingGeometryModel,
    _geometry_artifact,
)


def _materialized(rows=None, batch_size=1):
    if rows is None:
        rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    parent = _parent()
    geometry_model = RecordingGeometryModel()
    records = materialize_hierarchical_rows(
        rows,
        parent,
        geometry_model,
        _geometry_artifact(),
        batch_size=batch_size,
        device="cpu",
    )
    return rows, parent, geometry_model, records


def test_hierarchical_trainer_reuses_strict_loader_split_and_snapshots():
    assert load_residual_training_inputs is residual_load_inputs
    assert split_residual_joined_rows is residual_split_rows
    assert capture_immutable_artifact_identities is residual_capture_identities
    assert HIERARCHICAL_MATERIALIZATION_BATCH_SIZE == 256


def test_materialization_returns_exact_canonical_cpu_contract():
    rows, parent, geometry_model, records = _materialized()

    assert len(records) == 2
    assert [record["dataset_index"] for record in records] == [0, 1]
    assert set(records[0]) == set(HIERARCHICAL_RECORD_FIELDS)
    assert HIERARCHICAL_RECORD_FIELDS == (
        "dataset_index",
        "scan_id",
        "target_id",
        "query_features",
        "variant_features",
        "query_aux_continuous",
        "query_aux_binary",
        "variant_aux_continuous",
        "variant_aux_binary",
        "query_valid",
        "variant_valid",
        "candidate_ious",
        "baseline_index",
        "baseline_scores",
    )
    shapes = {
        "query_features": ((16, 152), torch.float32),
        "variant_features": ((16, 7, 25), torch.float32),
        "query_aux_continuous": ((16, 4), torch.float32),
        "query_aux_binary": ((16, 2), torch.bool),
        "variant_aux_continuous": ((16, 7, 2), torch.float32),
        "variant_aux_binary": ((16, 7, 2), torch.bool),
        "query_valid": ((16,), torch.bool),
        "variant_valid": ((16, 7), torch.bool),
        "candidate_ious": ((16, 7), torch.float32),
        "baseline_scores": ((112,), torch.float32),
    }
    for record in records:
        for field, (shape, dtype) in shapes.items():
            value = record[field]
            assert value.device.type == "cpu"
            assert value.shape == shape
            assert value.dtype == dtype
        assert record["query_valid"].tolist() == [True] * 16
        assert record["variant_valid"].all()
        assert record["query_aux_binary"][:, 0].sum().item() == 1
        assert record["query_aux_binary"][:, 1].sum().item() == 1
        assert record["variant_aux_binary"][..., 0].sum().item() == 1
        assert record["variant_aux_binary"][..., 1].sum().item() == 1
        assert record["variant_valid"].reshape(-1)[
            record["baseline_index"]
        ]
        assert record["baseline_scores"].argmax().item() == record[
            "baseline_index"
        ]
        targets = build_hierarchical_targets(
            record["candidate_ious"].unsqueeze(0),
            record["variant_valid"].unsqueeze(0),
        )
        assert torch.equal(targets["query_valid"][0], record["query_valid"])

    assert torch.equal(
        records[0]["query_features"], rows[0]["base"]["features"]
    )
    assert torch.equal(
        records[0]["variant_features"],
        rows[0]["geometry"]["geometry_features"],
    )
    assert torch.equal(
        records[0]["query_aux_continuous"][:, 0],
        rows[0]["base"]["default_scores"],
    )
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


def test_materialization_preserves_query_major_variant_minor_axes():
    rows, _parent_value, _model, records = _materialized()
    record = records[0]

    for query_index in range(16):
        assert torch.equal(
            record["query_features"][query_index],
            rows[0]["base"]["features"][query_index],
        )
        for variant_index in range(7):
            assert torch.equal(
                record["variant_features"][query_index, variant_index],
                rows[0]["geometry"]["geometry_features"][
                    query_index, variant_index
                ],
            )
            flat_index = query_index * 7 + variant_index
            assert record["baseline_scores"][flat_index].isfinite()


def test_training_ious_change_only_labels_and_label_bound_digests():
    rows = [_joined_row(0, "scene_a"), _joined_row(1, "scene_b")]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["geometry"]["geometry_ious"].copy_(
            1.0 - row["geometry"]["geometry_ious"]
        )
    _, _, _, original = _materialized(rows)
    _, _, _, modified = _materialized(changed)

    original_targets = build_hierarchical_targets(
        torch.stack([record["candidate_ious"] for record in original]),
        torch.stack([record["variant_valid"] for record in original]),
    )
    modified_targets = build_hierarchical_targets(
        torch.stack([record["candidate_ious"] for record in modified]),
        torch.stack([record["variant_valid"] for record in modified]),
    )

    assert not torch.equal(
        original_targets["variant_targets"],
        modified_targets["variant_targets"],
    )
    for before, after in zip(original, modified):
        for field in HIERARCHICAL_RECORD_FIELDS:
            if field == "candidate_ious":
                assert not torch.equal(before[field], after[field])
            elif isinstance(before[field], torch.Tensor):
                assert torch.equal(before[field], after[field])
            else:
                assert before[field] == after[field]
    assert canonical_hierarchical_rows_sha256(original) != \
        canonical_hierarchical_rows_sha256(modified)
    assert canonical_hierarchical_candidate_iou_sha256(original) != \
        canonical_hierarchical_candidate_iou_sha256(modified)
    assert canonical_hierarchical_deployable_sha256(original) == \
        canonical_hierarchical_deployable_sha256(modified)


def test_materialization_digests_are_ordered_and_tamper_bound():
    _, _, _, first = _materialized()
    _, _, _, repeated = _materialized()

    full_sha = canonical_hierarchical_rows_sha256(first)
    deployable_sha = canonical_hierarchical_deployable_sha256(first)
    label_sha = canonical_hierarchical_candidate_iou_sha256(first)

    assert full_sha == canonical_hierarchical_rows_sha256(repeated)
    assert deployable_sha == canonical_hierarchical_deployable_sha256(repeated)
    assert label_sha == canonical_hierarchical_candidate_iou_sha256(repeated)
    assert all(len(value) == 64 for value in (
        full_sha, deployable_sha, label_sha,
    ))
    changed = copy.deepcopy(first)
    changed[0]["query_features"][0, 0] += 1.0
    assert canonical_hierarchical_rows_sha256(changed) != full_sha
    assert canonical_hierarchical_deployable_sha256(changed) != deployable_sha
    assert canonical_hierarchical_candidate_iou_sha256(changed) == label_sha


@pytest.mark.parametrize(
    "mutation",
    ["reordered", "missing_index", "identity", "feature_dtype", "schema"],
)
def test_materialization_rejects_row_or_artifact_contract_changes(mutation):
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
        artifact["feature_names"][152] = "changed"

    with pytest.raises((TypeError, ValueError)):
        materialize_hierarchical_rows(
            rows,
            _parent(),
            RecordingGeometryModel(),
            artifact,
            batch_size=1,
            device="cpu",
        )


@pytest.mark.parametrize("mutation", ["training", "requires_grad", "dtype"])
def test_materialization_rejects_mutating_geometry_model(mutation):
    class BadGeometryModel(RecordingGeometryModel):
        def forward(self, features, valid_mask):
            outputs = super().forward(features, valid_mask)
            if mutation == "training":
                self.train()
            elif mutation == "requires_grad":
                self.anchor.requires_grad_(True)
            elif mutation == "dtype":
                outputs = {
                    name: value.double() for name, value in outputs.items()
                }
            return outputs

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        materialize_hierarchical_rows(
            [_joined_row(0, "scene_a")],
            _parent(),
            BadGeometryModel(),
            _geometry_artifact(),
            batch_size=1,
            device="cpu",
        )


def _model_batch(records):
    return {
        field: torch.stack([record[field] for record in records])
        for field in HIERARCHICAL_MODEL_BATCH_FIELDS
    }


def _mask_non_top1_query(record, query_index=8):
    assert record["query_aux_binary"][query_index].sum().item() == 0
    assert record["variant_aux_binary"][query_index].sum().item() == 0
    record["query_valid"][query_index] = False
    record["variant_valid"][query_index] = False
    for field in ("query_features", "query_aux_continuous"):
        record[field][query_index].zero_()
    record["query_aux_binary"][query_index].zero_()
    for field in ("variant_features", "variant_aux_continuous"):
        record[field][query_index].zero_()
    record["variant_aux_binary"][query_index].zero_()
    start = query_index * 7
    record["baseline_scores"][start:start + 7] = -float("inf")


def test_normalization_fits_exact_valid_population_statistics():
    _, _, _, records = _materialized()
    records = copy.deepcopy(records)
    _mask_non_top1_query(records[0])
    assert not records[0]["variant_aux_binary"][0, 6].any()
    records[0]["variant_valid"][0, 6] = False
    records[0]["variant_features"][0, 6].zero_()
    records[0]["variant_aux_continuous"][0, 6].zero_()
    records[0]["baseline_scores"][6] = -float("inf")

    statistics = fit_hierarchical_normalization(records)

    assert HIERARCHICAL_NORMALIZATION_SCHEMA == \
        "rec-hierarchical-normalization-v1"
    assert HIERARCHICAL_MIN_STD == 1e-6
    assert set(statistics) == {
        "schema", "minimum_std", "groups", "sha256"
    }
    assert statistics["schema"] == HIERARCHICAL_NORMALIZATION_SCHEMA
    assert statistics["minimum_std"] == HIERARCHICAL_MIN_STD
    assert set(statistics["groups"]) == set(HIERARCHICAL_NORMALIZATION_GROUPS)
    assert len(statistics["sha256"]) == 64
    masks = {
        "query_features": "query_valid",
        "variant_features": "variant_valid",
        "query_aux_continuous": "query_valid",
        "variant_aux_continuous": "variant_valid",
    }
    for group_name, mask_name in masks.items():
        values = torch.cat([
            record[group_name][record[mask_name]].to(torch.float64)
            for record in records
        ], dim=0)
        expected_mean = values.mean(dim=0).to(torch.float32)
        expected_std = values.std(dim=0, unbiased=False).clamp(
            min=HIERARCHICAL_MIN_STD
        ).to(torch.float32)
        group = statistics["groups"][group_name]
        assert group["count"] == values.shape[0]
        assert group["mean"].dtype == torch.float32
        assert group["std"].dtype == torch.float32
        assert group["mean"].device.type == "cpu"
        assert group["std"].device.type == "cpu"
        assert torch.allclose(group["mean"], expected_mean, atol=1e-6)
        assert torch.allclose(group["std"], expected_std, atol=1e-6)
        assert len(group["feature_names"]) == values.shape[1]
        assert len(set(group["feature_names"])) == values.shape[1]


def test_fold_local_normalization_is_bitwise_isolated_from_held_scenes():
    rows = [
        _joined_row(0, "fit_a", 0.0),
        _joined_row(1, "fit_b", 0.5),
        _joined_row(2, "held", 100.0),
    ]
    _, _, _, records = _materialized(rows, batch_size=3)
    fit_records = records[:2]
    held_record = records[2]

    before_stats = fit_hierarchical_normalization(fit_records)
    before = normalize_hierarchical_batch(
        _model_batch(fit_records), before_stats
    )
    held_record["query_features"].add_(1e9)
    held_record["variant_features"].sub_(1e9)
    held_record["query_aux_continuous"].mul_(1e6)
    held_record["variant_aux_continuous"].mul_(-1e6)
    after_stats = fit_hierarchical_normalization(fit_records)
    after = normalize_hierarchical_batch(
        _model_batch(fit_records), after_stats
    )

    assert before_stats["sha256"] == after_stats["sha256"]
    for group_name in HIERARCHICAL_NORMALIZATION_GROUPS:
        assert torch.equal(
            before_stats["groups"][group_name]["mean"],
            after_stats["groups"][group_name]["mean"],
        )
        assert torch.equal(
            before_stats["groups"][group_name]["std"],
            after_stats["groups"][group_name]["std"],
        )
    for field in HIERARCHICAL_MODEL_BATCH_FIELDS:
        assert torch.equal(before[field], after[field])


def test_normalization_zero_fills_invalid_and_preserves_binary_fields():
    _, _, _, records = _materialized()
    records = copy.deepcopy(records)
    _mask_non_top1_query(records[0])
    statistics = fit_hierarchical_normalization(records)
    batch = _model_batch(records)

    normalized = normalize_hierarchical_batch(batch, statistics)

    assert set(normalized) == set(HIERARCHICAL_MODEL_BATCH_FIELDS)
    assert torch.equal(
        normalized["query_aux_binary"], batch["query_aux_binary"]
    )
    assert torch.equal(
        normalized["variant_aux_binary"], batch["variant_aux_binary"]
    )
    assert torch.equal(normalized["query_valid"], batch["query_valid"])
    assert torch.equal(normalized["variant_valid"], batch["variant_valid"])
    for field in ("query_features", "query_aux_continuous"):
        assert torch.equal(
            normalized[field][~batch["query_valid"]],
            torch.zeros_like(normalized[field][~batch["query_valid"]]),
        )
    for field in ("variant_features", "variant_aux_continuous"):
        assert torch.equal(
            normalized[field][~batch["variant_valid"]],
            torch.zeros_like(normalized[field][~batch["variant_valid"]]),
        )
    assert all(
        torch.isfinite(normalized[field]).all()
        for field in (
            "query_features",
            "variant_features",
            "query_aux_continuous",
            "variant_aux_continuous",
        )
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_label",
        "query_shape",
        "variant_shape",
        "query_dtype",
        "binary_dtype",
        "mask_dtype",
        "inconsistent_mask",
        "nonfinite_valid",
    ],
)
def test_normalization_rejects_non_model_or_malformed_batches(mutation):
    _, _, _, records = _materialized()
    statistics = fit_hierarchical_normalization(records)
    batch = _model_batch(records)
    if mutation == "extra_label":
        batch["candidate_ious"] = torch.stack([
            record["candidate_ious"] for record in records
        ])
    elif mutation == "query_shape":
        batch["query_features"] = batch["query_features"][:, :-1]
    elif mutation == "variant_shape":
        batch["variant_features"] = batch["variant_features"][:, :, :-1]
    elif mutation == "query_dtype":
        batch["query_features"] = batch["query_features"].double()
    elif mutation == "binary_dtype":
        batch["query_aux_binary"] = batch["query_aux_binary"].float()
    elif mutation == "mask_dtype":
        batch["variant_valid"] = batch["variant_valid"].long()
    elif mutation == "inconsistent_mask":
        batch["query_valid"][0, 0] = False
    elif mutation == "nonfinite_valid":
        batch["variant_features"][0, 0, 0, 0] = float("nan")

    with pytest.raises((TypeError, ValueError)):
        normalize_hierarchical_batch(batch, statistics)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "sha", "mean", "std", "count", "feature_names"],
)
def test_normalization_rejects_tampered_or_different_statistics(mutation):
    _, _, _, records = _materialized()
    statistics = fit_hierarchical_normalization(records)
    changed = copy.deepcopy(statistics)
    if mutation == "schema":
        changed["schema"] = "other-schema"
    elif mutation == "sha":
        changed["sha256"] = "0" * 64
    elif mutation == "mean":
        changed["groups"]["query_features"]["mean"][0] += 1.0
    elif mutation == "std":
        changed["groups"]["variant_features"]["std"][0] = 0.0
    elif mutation == "count":
        changed["groups"]["query_aux_continuous"]["count"] = 0
    elif mutation == "feature_names":
        changed["groups"]["variant_aux_continuous"][
            "feature_names"
        ][0] = "changed"

    with pytest.raises((TypeError, ValueError)):
        normalize_hierarchical_batch(_model_batch(records), changed)


def test_hierarchical_training_constants_are_fixed():
    assert HIERARCHICAL_EPOCHS == 12
    assert HIERARCHICAL_BATCH_SIZE == 256
    assert HIERARCHICAL_LEARNING_RATE == 3e-4
    assert HIERARCHICAL_GRAD_CLIP_NORM == 1.0
    assert HIERARCHICAL_DROPOUT == 0.1
    assert HIERARCHICAL_MODEL_SEED == 0


def test_nearest_rank_margin_uses_only_positive_proposal_gains():
    gains = torch.tensor([-3.0, 0.0, 1.0, 2.0, 4.0])

    assert nearest_rank_hierarchical_margin(gains, 50.0) == 2.0
    assert nearest_rank_hierarchical_margin(gains, 95.0) == 4.0
    assert nearest_rank_hierarchical_margin(gains, 0.0) == 1.0
    assert nearest_rank_hierarchical_margin(
        torch.tensor([-1.0, 0.0]), 50.0
    ) is None


def _policy_diagnostic_records():
    rows = [
        _joined_row(index, "policy_scene_{:02d}".format(index), index / 10.0)
        for index in range(6)
    ]
    _, _, _, records = _materialized(rows, batch_size=6)
    records = copy.deepcopy(records)
    proposals = []
    for index, record in enumerate(records):
        record["candidate_ious"].zero_()
        baseline = record["baseline_index"]
        if index % 2 == 0:
            proposal = 0
            assert proposal // 7 != baseline // 7
        else:
            proposal = baseline - 1
            assert proposal // 7 == baseline // 7
            assert proposal != baseline
        record["candidate_ious"].reshape(-1)[proposal] = 1.0
        proposals.append(proposal)
    return records, torch.tensor(proposals, dtype=torch.long)


def test_policy_candidate_records_query_and_variant_recoveries_exactly():
    records, proposals = _policy_diagnostic_records()
    config = {
        "hidden_dim": 64,
        "weight_decay": 1e-3,
        "false_positive_cost": 4.0,
    }

    candidate = build_hierarchical_policy_candidate(
        records,
        proposals,
        torch.ones(6, dtype=torch.float32),
        config,
        percentile=95.0,
        margin=0.5,
    )
    choice = choose_hierarchical_configuration([candidate])

    assert candidate["transition_diagnostics"] == {
        "selected_query_changes": 3,
        "same_query_variant_changes": 3,
        "wrong_query_recoveries025": 3,
        "wrong_query_recoveries050": 3,
        "wrong_variant_recoveries025": 3,
        "wrong_variant_recoveries050": 3,
    }
    assert choice["eligible"] is True
    assert choice["transition_diagnostics"] == candidate[
        "transition_diagnostics"
    ]
    assert choice["effects"]["0.25"]["fixes"] == 6
    assert choice["effects"]["0.50"]["fixes"] == 6


def test_hierarchical_training_label_summary_reconciles_masks():
    records, _proposals = _policy_diagnostic_records()

    summary = summarize_hierarchical_training_labels(records)

    assert set(summary) == {"query", "variant"}
    for level, valid_field in (
            ("query", "query_valid"), ("variant", "variant_valid")):
        expected = sum(
            int(record[valid_field].sum().item()) for record in records
        )
        for threshold in ("0.25", "0.50"):
            counts = summary[level][threshold]
            assert set(counts) == {"positive", "negative", "total"}
            assert counts["positive"] + counts["negative"] == expected
            assert counts["total"] == expected


def _small_crossfit_records():
    rows = []
    for index in range(10):
        row = _joined_row(
            index,
            "crossfit_scene_{:02d}".format(index),
            feature_offset=(index % 3) * 0.25,
        )
        ious = row["geometry"]["geometry_ious"]
        ious.zero_()
        positive_query = index % 2
        ious[positive_query, 0] = 0.75
        rows.append(row)
    _, _, _, records = _materialized(rows, batch_size=10)
    return records


def test_cross_fit_is_scene_disjoint_complete_and_exactly_repeatable():
    records = _small_crossfit_records()
    optimizer_events = []
    normalization_events = []

    first = cross_fit_hierarchical_reranker(
        records,
        device="cpu",
        batch_observer=lambda event: optimizer_events.append(
            copy.deepcopy(event)
        ),
        normalization_observer=lambda event: normalization_events.append(
            copy.deepcopy(event)
        ),
    )
    second = cross_fit_hierarchical_reranker(records, device="cpu")

    assert first == second
    assert len(first["configurations"]) == 8
    assert len({
        (
            record["hidden_dim"],
            record["weight_decay"],
            record["false_positive_cost"],
        )
        for record in first["configurations"]
    }) == 8
    assert first["scene_fold_sha256"] == second["scene_fold_sha256"]
    assert "calibration" not in first
    for configuration_index, configuration in enumerate(
            first["configurations"]):
        assert configuration["configuration_index"] == configuration_index
        assert configuration["prediction_count"] == len(records)
        assert len(configuration["oof_proposal_sha256"]) == 64
        assert len(configuration["oof_gain_sha256"]) == 64
        assert "oof_proposals" not in configuration
        assert "oof_gain" not in configuration
        assert len(configuration["folds"]) == 5
        for fold in configuration["folds"]:
            assert fold["fit_row_count"] + fold["held_row_count"] == len(
                records
            )
            assert fold["normalization_sha256"]
            assert fold["training_labels"]["query"]["0.25"][
                "total"
            ] == fold["fit_query_count"]
            assert fold["training_labels"]["variant"]["0.25"][
                "total"
            ] == fold["fit_variant_count"]

    mapping = first["scene_folds"]
    assert len(normalization_events) == 8 * 5
    assert len(optimizer_events) == 8 * 5 * HIERARCHICAL_EPOCHS
    for event in normalization_events + optimizer_events:
        assert event["phase"] == "cross_fit"
        assert all(
            mapping[scan_id] != event["held_out_fold"]
            for scan_id in event["scan_ids"]
        )
    for config_index in range(8):
        for fold in range(5):
            assert {
                event["epoch"] for event in optimizer_events
                if event["config_index"] == config_index
                and event["held_out_fold"] == fold
            } == set(range(HIERARCHICAL_EPOCHS))
            matching_stats = [
                event for event in normalization_events
                if event["config_index"] == config_index
                and event["held_out_fold"] == fold
            ]
            assert len(matching_stats) == 1
    assert first["choice"]["selected"] in {"baseline", "hierarchical"}
    assert first["policy_candidate_count"] >= 8


def _eligible_hierarchical_choice():
    return {
        "eligible": True,
        "selected": "hierarchical",
        "hidden_dim": 64,
        "weight_decay": 1e-4,
        "false_positive_cost": 2.0,
        "margin_percentile": 95.0,
        "margin": 0.25,
    }


def test_refit_uses_all_fit_rows_and_one_all_fit_normalization():
    records = _small_crossfit_records()
    batch_events = []
    normalization_events = []

    model, statistics = refit_hierarchical_reranker(
        records,
        _eligible_hierarchical_choice(),
        device="cpu",
        batch_observer=lambda event: batch_events.append(copy.deepcopy(event)),
        normalization_observer=lambda event: normalization_events.append(
            copy.deepcopy(event)
        ),
    )

    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    repeated_statistics = fit_hierarchical_normalization(records)
    assert statistics["sha256"] == repeated_statistics["sha256"]
    for group_name in HIERARCHICAL_NORMALIZATION_GROUPS:
        assert torch.equal(
            statistics["groups"][group_name]["mean"],
            repeated_statistics["groups"][group_name]["mean"],
        )
        assert torch.equal(
            statistics["groups"][group_name]["std"],
            repeated_statistics["groups"][group_name]["std"],
        )
    assert len(normalization_events) == 1
    assert normalization_events[0] == {
        "phase": "refit",
        "scan_ids": tuple(record["scan_id"] for record in records),
        "normalization_sha256": statistics["sha256"],
    }
    assert len(batch_events) == HIERARCHICAL_EPOCHS
    assert {event["epoch"] for event in batch_events} == set(
        range(HIERARCHICAL_EPOCHS)
    )
    assert all(event["phase"] == "refit" for event in batch_events)
    assert all(
        set(event["scan_ids"]) == {
            record["scan_id"] for record in records
        }
        for event in batch_events
    )


def test_refit_rejects_unfrozen_or_out_of_grid_oof_choice_before_fitting():
    records = _small_crossfit_records()
    for mutation in ("ineligible", "hidden_dim", "margin"):
        choice = _eligible_hierarchical_choice()
        if mutation == "ineligible":
            choice["eligible"] = False
        elif mutation == "hidden_dim":
            choice["hidden_dim"] = 96
        else:
            choice["margin"] = 0.0
        events = []
        with pytest.raises(ValueError):
            refit_hierarchical_reranker(
                records,
                choice,
                device="cpu",
                batch_observer=lambda event: events.append(event),
            )
        assert events == []


def _cache_policy_records():
    rows = [
        _joined_row(index, "cache_scene_{:02d}".format(index))
        for index in range(5)
    ]
    _, _, _, records = _materialized(rows, batch_size=5)
    records = copy.deepcopy(records)
    proposed_indices = []
    baseline_ious = (0.50, 0.25, 0.10, 0.75, 0.00)
    proposed_ious = (0.75, 0.50, 0.25, 0.10, 0.30)
    for record, baseline_iou, proposed_iou in zip(
            records, baseline_ious, proposed_ious):
        baseline_index = record["baseline_index"]
        flat_valid = record["variant_valid"].reshape(-1)
        proposed_index = next(
            index for index in flat_valid.nonzero(
                as_tuple=False
            ).reshape(-1).tolist()
            if index != baseline_index
        )
        record["candidate_ious"].zero_()
        record["candidate_ious"].reshape(-1)[baseline_index] = baseline_iou
        record["candidate_ious"].reshape(-1)[proposed_index] = proposed_iou
        proposed_indices.append(proposed_index)
    return records, torch.tensor(proposed_indices, dtype=torch.long)


class _FixedHierarchicalProposalModel(torch.nn.Module):
    def __init__(self, proposed_indices):
        super().__init__()
        self.register_buffer("proposed_indices", proposed_indices.clone())

    def forward(self, **batch):
        batch_size = batch["query_features"].shape[0]
        assert batch_size == self.proposed_indices.numel()
        query_logits = torch.full(
            (batch_size, 16, 2), -10.0, dtype=torch.float32,
            device=batch["query_features"].device,
        )
        variant_logits = torch.full(
            (batch_size, 16, 7, 2), -10.0, dtype=torch.float32,
            device=batch["query_features"].device,
        )
        for row, flat_index in enumerate(self.proposed_indices.tolist()):
            query_index, variant_index = divmod(flat_index, 7)
            query_logits[row, query_index] = 10.0
            variant_logits[row, query_index, variant_index] = 10.0
        return {
            "query_logits": query_logits,
            "variant_logits": variant_logits,
        }


def test_cache_baseline_and_policy_use_strict_thresholds_and_bound_digests():
    records, proposed_indices = _cache_policy_records()
    baseline = build_hierarchical_cache_calibration_baseline(records)

    assert baseline["sample_count"] == 5
    assert baseline["hits025"] == 2
    assert baseline["hits050"] == 1
    assert baseline["oracle_hits025"] == 4
    assert baseline["oracle_hits050"] == 2
    assert baseline["candidate_iou_sha256"] == \
        canonical_hierarchical_candidate_iou_sha256(records)
    assert baseline["row_materialization_sha256"] == \
        canonical_hierarchical_rows_sha256(records)

    statistics = fit_hierarchical_normalization(records)
    metrics = evaluate_hierarchical_cache_policy(
        _FixedHierarchicalProposalModel(proposed_indices),
        records,
        statistics,
        margin=0.25,
        device="cpu",
    )

    assert metrics["sample_count"] == 5
    assert metrics["hits025"] == 3
    assert metrics["hits050"] == 1
    assert metrics["baseline_hits025"] == 2
    assert metrics["baseline_hits050"] == 1
    assert metrics["oracle_hits025"] == 4
    assert metrics["oracle_hits050"] == 2
    assert metrics["fixes025"] == 2
    assert metrics["breaks025"] == 1
    assert metrics["fixes050"] == 1
    assert metrics["breaks050"] == 1
    assert metrics["switches"] == 5
    assert metrics["abstentions"] == 0
    assert metrics["candidate_iou_sha256"] == baseline[
        "candidate_iou_sha256"
    ]
    assert metrics["row_materialization_sha256"] == baseline[
        "row_materialization_sha256"
    ]


def _authoritative_cache_gate_values():
    baseline = {
        "sample_count": 3625,
        "hits025": 3461,
        "hits050": 3315,
        "oracle_hits025": 3606,
        "oracle_hits050": 3588,
        "candidate_iou_sha256": "1" * 64,
        "row_materialization_sha256": "2" * 64,
    }
    metrics = {
        "sample_count": 3625,
        "hits025": 3524,
        "hits050": 3315,
        "baseline_hits025": 3461,
        "baseline_hits050": 3315,
        "oracle_hits025": 3606,
        "oracle_hits050": 3588,
        "candidate_iou_sha256": "1" * 64,
        "row_materialization_sha256": "2" * 64,
    }
    return metrics, baseline


def test_hierarchical_cache_gate_is_fixed_and_tamper_bound():
    metrics, baseline = _authoritative_cache_gate_values()

    passed = hierarchical_calibration_gate(metrics, baseline)

    assert passed.passed is True
    assert passed.failures == ()
    assert passed.required_hits025 == 3524
    assert passed.required_hits050 == 3315
    for field in (
            "hits025", "hits050", "candidate_iou_sha256",
            "row_materialization_sha256"):
        changed_metrics = copy.deepcopy(metrics)
        if field == "hits025":
            changed_metrics[field] = 3523
        elif field == "hits050":
            changed_metrics[field] = 3314
        else:
            changed_metrics[field] = "f" * 64
        failed = hierarchical_calibration_gate(changed_metrics, baseline)
        assert failed.passed is False
        assert field in failed.failures

    changed_baseline = copy.deepcopy(baseline)
    changed_baseline["oracle_hits025"] -= 1
    failed = hierarchical_calibration_gate(metrics, changed_baseline)
    assert failed.passed is False
    assert "oracle_hits025" in failed.failures
    assert "authoritative_oracle_hits025" in failed.failures


def _hierarchical_input_sha256():
    return {
        "backbone": (
            "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
        ),
        "parent": (
            "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
        ),
        "geometry": (
            "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f"
        ),
        "base_cache_content": (
            "411ec7d5d80a7be9596de20b348667d529e6a8f568b8ab0c0e0922b8719f9045"
        ),
        "geometry_cache_content": (
            "2f099adb04823c8a4bdfb32040431c8b9150b6da39a617640c9b871f52ba3750"
        ),
        "geometry_metadata": (
            "6965b4a21daf52a25b7793e1df4fbff3ca26ed9e5db011cc847f2b601eb8c062"
        ),
    }


def _staged_hierarchical_artifact():
    records = _small_crossfit_records()
    model = HierarchicalQueryVariantReranker(
        hidden_dim=64, dropout=HIERARCHICAL_DROPOUT
    ).eval().requires_grad_(False)
    normalization = fit_hierarchical_normalization(records)
    choice = _eligible_hierarchical_choice()
    choice.update({
        "switches": 5,
        "delta_hits025": 3,
        "delta_hits050": 1,
    })
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    metrics, baseline = _authoritative_cache_gate_values()
    metrics.update({
        "fixes025": 100,
        "breaks025": 37,
        "fixes050": 0,
        "breaks050": 0,
        "switches": 200,
        "abstentions": 3425,
        "switch_rate": 200 / 3625.0,
    })
    geometry_names = _geometry_artifact()["feature_names"]
    artifact = build_hierarchical_artifact(
        model=model,
        selection=choice,
        normalization=normalization,
        scene_folds=scene_folds,
        geometry_feature_names=geometry_names,
        input_sha256=_hierarchical_input_sha256(),
        row_materialization_sha256=canonical_hierarchical_rows_sha256(
            records
        ),
        candidate_iou_sha256=(
            canonical_hierarchical_candidate_iou_sha256(records)
        ),
        oof_record={
            "prediction_count": len(records),
            "proposal_sha256": "a" * 64,
            "gain_sha256": "b" * 64,
            "delta_hits025": 3,
            "delta_hits050": 1,
        },
        calibration_record=metrics,
        calibration_baseline=baseline,
    )
    return records, model, geometry_names, artifact


def test_staged_artifact_binds_model_normalization_features_oof_and_cache():
    records, model, geometry_names, artifact = _staged_hierarchical_artifact()

    assert artifact["schema"] == "rec-hierarchical-query-variant-v1"
    assert artifact["version"] == 1
    assert artifact["deployable"] is False
    assert artifact["validation_data_accessed"] is False
    assert artifact["model_config"] == {
        "hidden_dim": 64,
        "dropout": HIERARCHICAL_DROPOUT,
    }
    assert artifact["normalization_sha256"] == artifact[
        "normalization"
    ]["sha256"]
    assert artifact["feature_names"] == build_hierarchical_feature_names(
        geometry_names
    )
    assert artifact["feature_names"]["query_features"] == geometry_names[:152]
    assert artifact["feature_names"]["variant_features"] == \
        geometry_names[152:177]
    assert artifact["scene_fold_sha256"]
    assert artifact["oof_record"]["proposal_sha256"] == "a" * 64
    assert artifact["oof_record"]["gain_sha256"] == "b" * 64
    assert artifact["calibration_record"]["hits025"] == 3524
    assert artifact["calibration_baseline"]["hits025"] == 3461
    for name, value in model.state_dict().items():
        assert value.device.type == "cpu"
        assert torch.equal(artifact["model_state_dict"][name], value)
    assert validate_hierarchical_artifact(
        artifact, expected_geometry_feature_names=geometry_names
    )["hidden_dim"] == 64

    batch = normalize_hierarchical_batch(_model_batch(records), artifact[
        "normalization"
    ])
    with torch.no_grad():
        expected = model(**batch)
    assert set(expected) == {
        "query_logits", "variant_logits", "query_embedding",
        "variant_embedding",
    }


def test_staged_artifact_saves_exclusively_and_strictly_reloads(tmp_path):
    records, model, geometry_names, artifact = _staged_hierarchical_artifact()
    output = tmp_path / "selected_hierarchical.pth"

    save_hierarchical_artifact(output, artifact)
    loaded_model, loaded_artifact = load_hierarchical_artifact(
        output,
        device="cpu",
        expected_geometry_feature_names=geometry_names,
    )

    assert output.stat().st_mode & 0o777 == 0o444
    assert loaded_model.training is False
    assert all(
        not parameter.requires_grad for parameter in loaded_model.parameters()
    )
    assert len(loaded_model._artifact_sha256) == 64
    assert loaded_artifact["normalization_sha256"] == artifact[
        "normalization_sha256"
    ]
    for name, value in model.state_dict().items():
        assert torch.equal(loaded_model.state_dict()[name], value)
        assert torch.equal(loaded_artifact["model_state_dict"][name], value)
    batch = normalize_hierarchical_batch(
        _model_batch(records), artifact["normalization"]
    )
    with torch.no_grad():
        expected = model(**batch)
        actual = loaded_model(**batch)
    for field in expected:
        assert torch.equal(actual[field], expected[field])

    with pytest.raises(FileExistsError):
        save_hierarchical_artifact(output, artifact)


def test_deployed_artifact_serialization_requires_explicit_policy():
    _records, _model, geometry_names, artifact = \
        _staged_hierarchical_artifact()
    deployed = copy.deepcopy(artifact)
    deployed["deployable"] = True

    with pytest.raises(ValueError, match="top-level policy"):
        _serialize_hierarchical_artifact(deployed)

    payload = _serialize_hierarchical_artifact(
        deployed, expected_deployable=True
    )
    reloaded = torch.load(io.BytesIO(payload), map_location="cpu")

    assert reloaded["deployable"] is True
    assert validate_hierarchical_artifact(
        reloaded,
        expected_geometry_feature_names=geometry_names,
        expected_deployable=True,
    )["hidden_dim"] == 64


def test_staged_artifact_loader_rejects_symlink_entry_path(tmp_path):
    _records, _model, geometry_names, artifact = \
        _staged_hierarchical_artifact()
    target = tmp_path / "target.pth"
    link = tmp_path / "linked.pth"
    save_hierarchical_artifact(target, artifact)
    link.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink"):
        load_hierarchical_artifact(
            link,
            device="cpu",
            expected_geometry_feature_names=geometry_names,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_field",
        "deployable",
        "validation_access",
        "input_sha",
        "feature_names",
        "normalization",
        "normalization_sha",
        "scene_fold",
        "oof_proposal",
        "selection",
        "calibration",
        "state",
    ),
)
def test_staged_artifact_rejects_metadata_and_tensor_tampering(mutation):
    _records, _model, geometry_names, artifact = \
        _staged_hierarchical_artifact()
    changed = copy.deepcopy(artifact)
    if mutation == "extra_field":
        changed["unexpected"] = True
    elif mutation == "deployable":
        changed["deployable"] = True
    elif mutation == "validation_access":
        changed["validation_data_accessed"] = True
    elif mutation == "input_sha":
        changed["input_sha256"]["geometry"] = "f" * 64
    elif mutation == "feature_names":
        changed["feature_names"]["query_features"][0] = "changed"
    elif mutation == "normalization":
        changed["normalization"]["groups"]["query_features"][
            "mean"
        ][0] += 1.0
    elif mutation == "normalization_sha":
        changed["normalization_sha256"] = "f" * 64
    elif mutation == "scene_fold":
        first_scene = sorted(changed["scene_folds"])[0]
        changed["scene_folds"][first_scene] = (
            changed["scene_folds"][first_scene] + 1
        ) % 5
    elif mutation == "oof_proposal":
        changed["oof_record"]["proposal_sha256"] = "f" * 64
    elif mutation == "selection":
        changed["selection"]["false_positive_cost"] = 3.0
    elif mutation == "calibration":
        changed["calibration_record"]["hits025"] = 3523
    else:
        changed["model_state_dict"].pop(
            next(iter(changed["model_state_dict"]))
        )

    with pytest.raises((TypeError, ValueError)):
        validate_hierarchical_artifact(
            changed, expected_geometry_feature_names=geometry_names
        )


def _receipt_gain_statistics(count, value):
    return {
        "count": count,
        "statistics": {
            "minimum": value,
            "maximum": value,
            "mean": value,
            "population_standard_deviation": 0.0,
            "nearest_rank_quantiles": [
                {"quantile": quantile, "value": value}
                for quantile in HIERARCHICAL_GAIN_QUANTILES
            ],
        },
    }


def _receipt_training_labels(query_count, variant_count):
    result = {}
    for level, total in (("query", query_count), ("variant", variant_count)):
        positive025 = total // 2
        positive050 = total // 3
        result[level] = {
            "0.25": {
                "positive": positive025,
                "negative": total - positive025,
                "total": total,
            },
            "0.50": {
                "positive": positive050,
                "negative": total - positive050,
                "total": total,
            },
        }
    return result


def _receipt_configurations():
    configurations = []
    grid = [
        (hidden_dim, weight_decay, false_positive_cost)
        for hidden_dim in (64, 128)
        for weight_decay in (1e-4, 1e-3)
        for false_positive_cost in (2.0, 4.0)
    ]
    held_rows = (6608, 6608, 6608, 6608, 6608)
    held_scenes = (102, 101, 101, 101, 101)
    for configuration_index, config in enumerate(grid):
        folds = []
        for fold_index in range(5):
            fit_rows = 33040 - held_rows[fold_index]
            fit_scenes = 506 - held_scenes[fold_index]
            fit_queries = fit_rows * 16
            fit_variants = fit_rows * 112
            folds.append({
                "fold": fold_index,
                "fit_scene_count": fit_scenes,
                "fit_row_count": fit_rows,
                "fit_query_count": fit_queries,
                "fit_variant_count": fit_variants,
                "held_scene_count": held_scenes[fold_index],
                "held_row_count": held_rows[fold_index],
                "normalization_sha256": format(
                    configuration_index * 5 + fold_index + 1, "064x"
                ),
                "normalization_counts": {
                    "query_features": fit_queries,
                    "variant_features": fit_variants,
                    "query_aux_continuous": fit_queries,
                    "variant_aux_continuous": fit_variants,
                },
                "training_labels": _receipt_training_labels(
                    fit_queries, fit_variants
                ),
            })
        configurations.append({
            "hidden_dim": config[0],
            "weight_decay": config[1],
            "false_positive_cost": config[2],
            "configuration_index": configuration_index,
            "folds": folds,
            "gain_summary": {
                "all": _receipt_gain_statistics(33040, 0.5),
                "positive": _receipt_gain_statistics(100, 0.75),
            },
            "oof_proposal_sha256": format(
                100 + configuration_index, "064x"
            ),
            "oof_gain_sha256": format(200 + configuration_index, "064x"),
            "prediction_count": 33040,
        })
    return configurations


def _receipt_threshold_effects(
        sample_count, baseline_hits, fixes, breaks, neutral_switches):
    switches = fixes + breaks + neutral_switches
    kept_correct = baseline_hits - breaks
    kept_wrong = sample_count - switches - kept_correct
    return {
        "fixes": fixes,
        "breaks": breaks,
        "neutral_switches": neutral_switches,
        "kept_correct": kept_correct,
        "kept_wrong": kept_wrong,
    }


def _receipt_candidate(no_switch, hidden_dim=64, weight_decay=1e-4,
                       false_positive_cost=2.0):
    sample_count = 33040
    baseline025 = 19000
    baseline050 = 15000
    if no_switch:
        switches = 0
        delta025 = 0
        delta050 = 0
        percentile = None
        margin = None
        effects025 = _receipt_threshold_effects(
            sample_count, baseline025, 0, 0, 0
        )
        effects050 = _receipt_threshold_effects(
            sample_count, baseline050, 0, 0, 0
        )
        fold_deltas = {
            str(fold): {"hits025": 0, "hits050": 0}
            for fold in range(5)
        }
        transition = {
            "selected_query_changes": 0,
            "same_query_variant_changes": 0,
            "wrong_query_recoveries025": 0,
            "wrong_query_recoveries050": 0,
            "wrong_variant_recoveries025": 0,
            "wrong_variant_recoveries050": 0,
        }
    else:
        switches = 10
        delta025 = 5
        delta050 = 0
        percentile = 95.0
        margin = 0.25
        effects025 = _receipt_threshold_effects(
            sample_count, baseline025, 5, 0, 5
        )
        effects050 = _receipt_threshold_effects(
            sample_count, baseline050, 0, 0, 10
        )
        fold_deltas = {
            str(fold): {"hits025": 1, "hits050": 0}
            for fold in range(5)
        }
        transition = {
            "selected_query_changes": 5,
            "same_query_variant_changes": 5,
            "wrong_query_recoveries025": 3,
            "wrong_query_recoveries050": 0,
            "wrong_variant_recoveries025": 2,
            "wrong_variant_recoveries050": 0,
        }
    bootstrap025 = {
        "confidence": 0.95,
        "delta_hits": delta025,
        "lower_bound_95": 0,
        "replicates": 10000,
        "scene_count": 506,
        "seed": 0,
    }
    bootstrap050 = {
        "confidence": 0.95,
        "delta_hits": delta050,
        "lower_bound_95": 0,
        "replicates": 10000,
        "scene_count": 506,
        "seed": 0,
    }
    predicates = {
        "not_no_switch": not no_switch,
        "all_folds_nonnegative025": True,
        "all_folds_nonnegative050": True,
        "pooled_delta025_positive": delta025 > 0,
        "bootstrap025_lower_bound_nonnegative": True,
        "bootstrap050_lower_bound_nonnegative": True,
    }
    eligible = all(predicates.values())
    return {
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "false_positive_cost": false_positive_cost,
        "margin_percentile": percentile,
        "margin": margin,
        "no_switch": no_switch,
        "sample_count": sample_count,
        "switches": switches,
        "abstentions": sample_count - switches,
        "switch_rate": switches / float(sample_count),
        "baseline": {
            "0.25": {"hits": baseline025},
            "0.50": {"hits": baseline050},
        },
        "proposed": {
            "0.25": {"hits": baseline025 + delta025},
            "0.50": {"hits": baseline050 + delta050},
        },
        "effects": {"0.25": effects025, "0.50": effects050},
        "delta_hits025": delta025,
        "delta_hits050": delta050,
        "fold_deltas": fold_deltas,
        "bootstrap025": bootstrap025,
        "bootstrap050": bootstrap050,
        "eligibility_predicates": predicates,
        "failed_predicates": sorted(
            name for name, passed in predicates.items() if not passed
        ),
        "eligible": eligible,
        "selected": "hierarchical" if eligible else "baseline",
        "transition_diagnostics": transition,
    }


def _receipt_choice(eligible):
    diagnostics = []
    for hidden_dim in (64, 128):
        for weight_decay in (1e-4, 1e-3):
            for false_positive_cost in (2.0, 4.0):
                diagnostics.append(_receipt_candidate(
                    True, hidden_dim, weight_decay, false_positive_cost
                ))
    if eligible:
        winner = _receipt_candidate(False)
        diagnostics.append(winner)
        choice = copy.deepcopy(winner)
        choice.update({
            "candidate_count": len(diagnostics),
            "eligible_candidate_count": 1,
            "candidate_diagnostics": diagnostics,
        })
        return choice
    return {
        "candidate_count": len(diagnostics),
        "eligible_candidate_count": 0,
        "candidate_diagnostics": diagnostics,
        "eligible": False,
        "reason": "no-eligible-configuration",
        "selected": "baseline",
    }


def _receipt_protected_snapshot():
    inputs = _hierarchical_input_sha256()
    return {
        name: {
            "path": "/protected/{}.pth".format(name),
            "device": 1,
            "inode": index + 1,
            "mode": 0o444,
            "size": 1024 + index,
            "mtime_ns": 100 + index,
            "ctime_ns": 200 + index,
            "sha256": inputs[name],
        }
        for index, name in enumerate(("backbone", "parent", "geometry"))
    }


def _hierarchical_result_context(eligible=True, calibration_passed=True):
    configurations = _receipt_configurations()
    scene_folds = {
        "fit_scene_{:04d}".format(index): index % 5
        for index in range(506)
    }
    choice = _receipt_choice(eligible)
    baseline = {
        "sample_count": 33040,
        "hits025": 19000,
        "hits050": 15000,
        "oracle_hits025": 21000,
        "oracle_hits050": 18000,
        "candidate_iou_sha256": "3" * 64,
        "row_materialization_sha256": "4" * 64,
        "baseline_selected_iou_sha256": "5" * 64,
    }
    if eligible:
        metrics, calibration_baseline = _authoritative_cache_gate_values()
        if not calibration_passed:
            metrics["hits025"] = 3523
        gate = hierarchical_calibration_gate(metrics, calibration_baseline)
        calibration = {
            "status": "run",
            "baseline": calibration_baseline,
            "record": metrics,
            "gate": hierarchical_calibration_gate_receipt(gate),
        }
    else:
        calibration = {
            "status": "not_run",
            "reason": "oof_selection_rejected",
        }
    return {
        "input_sha256": _hierarchical_input_sha256(),
        "split": copy.deepcopy(AUTHORITATIVE_SPLIT_SEED0),
        "fit_joined_identity_sha256": "6" * 64,
        "fit_materialization_sha256": "4" * 64,
        "fit_deployable_sha256": "7" * 64,
        "fit_candidate_iou_sha256": "3" * 64,
        "fit_normalization_sha256": "8" * 64 if eligible else None,
        "oof": {
            "baseline": baseline,
            "scene_folds": scene_folds,
            "scene_fold_sha256": (
                canonical_hierarchical_scene_fold_sha256(scene_folds)
            ),
            "configuration_count": 8,
            "configurations": configurations,
            "policy_candidate_count": choice["candidate_count"],
            "choice": choice,
        },
        "calibration": calibration,
    }


def test_result_receipt_strictly_binds_success_and_oof_rejection():
    protected = _receipt_protected_snapshot()
    success_context = _hierarchical_result_context()

    success = build_hierarchical_result_receipt(
        success_context,
        artifact_binding={
            "name": "selected_hierarchical.pth",
            "sha256": "9" * 64,
        },
        protected_before=protected,
        protected_after=copy.deepcopy(protected),
    )

    assert success["schema"] == "rec-hierarchical-result-receipt-v1"
    assert success["selected"] == "staged_hierarchical"
    assert success["deployable"] is False
    assert success["validation_data_accessed"] is False
    assert success["oof"]["configuration_count"] == 8
    assert success["oof"]["policy_candidate_count"] == 9
    assert success["calibration"]["gate"]["passed"] is True
    assert validate_hierarchical_result_receipt(success) == success

    rejected_context = _hierarchical_result_context(eligible=False)
    rejected = build_hierarchical_result_receipt(
        rejected_context,
        artifact_binding=None,
        protected_before=protected,
        protected_after=copy.deepcopy(protected),
    )
    assert rejected["selected"] == "baseline"
    assert rejected["artifact"] is None
    assert rejected["calibration"] == {
        "status": "not_run",
        "reason": "oof_selection_rejected",
    }
    assert validate_hierarchical_result_receipt(rejected) == rejected


@pytest.mark.parametrize(
    "mutation",
    (
        "fold_count",
        "fold_rows",
        "normalization_count",
        "configuration_count",
        "configuration_digest",
        "candidate_count",
        "eligibility",
        "choice_digest",
        "fit_normalization",
        "calibration_gate",
        "artifact",
        "protected",
    ),
)
def test_result_receipt_rejects_fold_policy_calibration_and_snapshot_tamper(
        mutation):
    protected = _receipt_protected_snapshot()
    receipt = build_hierarchical_result_receipt(
        _hierarchical_result_context(),
        artifact_binding={
            "name": "selected_hierarchical.pth",
            "sha256": "9" * 64,
        },
        protected_before=protected,
        protected_after=copy.deepcopy(protected),
    )
    changed = copy.deepcopy(receipt)
    if mutation == "fold_count":
        changed["oof"]["configurations"][0]["folds"].pop()
    elif mutation == "fold_rows":
        changed["oof"]["configurations"][0]["folds"][0][
            "held_row_count"
        ] -= 1
    elif mutation == "normalization_count":
        changed["oof"]["configurations"][0]["folds"][0][
            "normalization_counts"
        ]["query_features"] -= 1
    elif mutation == "configuration_count":
        changed["oof"]["configuration_count"] = 7
    elif mutation == "configuration_digest":
        changed["oof"]["configuration_summaries_sha256"] = "f" * 64
    elif mutation == "candidate_count":
        changed["oof"]["policy_candidate_count"] -= 1
    elif mutation == "eligibility":
        changed["oof"]["choice"]["candidate_diagnostics"][-1][
            "eligibility_predicates"
        ]["pooled_delta025_positive"] = False
    elif mutation == "choice_digest":
        changed["oof"]["choice_sha256"] = "f" * 64
    elif mutation == "fit_normalization":
        changed["fit_normalization_sha256"] = "f" * 64
    elif mutation == "calibration_gate":
        changed["calibration"]["gate"]["passed"] = False
    elif mutation == "artifact":
        changed["artifact"] = None
    else:
        changed["protected_after"]["geometry"]["size"] += 1

    with pytest.raises(ValueError):
        validate_hierarchical_result_receipt(changed)


def _publication_protected(monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_hierarchical_reranker as trainer

    protected_paths = {}
    for name in ("backbone", "parent", "geometry"):
        path = tmp_path / "{}.pth".format(name)
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o444)
        protected_paths[name] = path
    snapshot = _receipt_protected_snapshot()
    monkeypatch.setattr(
        trainer,
        "capture_immutable_artifact_identities",
        lambda paths: copy.deepcopy(snapshot),
    )
    return protected_paths, snapshot


def test_hierarchical_output_reservation_is_fresh_and_rejects_symlinks(
        tmp_path):
    output = tmp_path / "reserved"
    reservation = reserve_hierarchical_output(output)

    assert reservation["path"] == str(output.absolute())
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        reserve_hierarchical_output(output)

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        reserve_hierarchical_output(linked_parent / "run")
    assert not (real_parent / "run").exists()


def test_rejected_hierarchical_publication_writes_completion_receipt_last(
        monkeypatch, tmp_path):
    protected_paths, protected = _publication_protected(
        monkeypatch, tmp_path
    )
    output = tmp_path / "rejected"

    publication = publish_hierarchical_experiment(
        output,
        artifact=None,
        result_context=_hierarchical_result_context(eligible=False),
        protected_paths=protected_paths,
        protected_before=protected,
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "result-receipt.json"
    ]
    receipt_path = output / "result-receipt.json"
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    receipt = json.loads(receipt_path.read_text("ascii"))
    assert receipt["selected"] == "baseline"
    assert receipt["artifact"] is None
    assert receipt["validation_data_accessed"] is False
    assert publication["artifact_path"] is None


def test_passed_hierarchical_publication_stages_readonly_artifact(
        monkeypatch, tmp_path):
    protected_paths, protected = _publication_protected(
        monkeypatch, tmp_path
    )
    _records, _model, geometry_names, artifact = \
        _staged_hierarchical_artifact()
    output = tmp_path / "passed"

    publication = publish_hierarchical_experiment(
        output,
        artifact=artifact,
        result_context=_hierarchical_result_context(),
        protected_paths=protected_paths,
        protected_before=protected,
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "result-receipt.json", "selected_hierarchical.pth"
    ]
    receipt = json.loads(
        (output / "result-receipt.json").read_text("ascii")
    )
    assert receipt["selected"] == "staged_hierarchical"
    assert receipt["artifact"]["sha256"] == publication[
        "artifact_sha256"
    ]
    loaded_model, loaded = load_hierarchical_artifact(
        publication["artifact_path"],
        device="cpu",
        expected_geometry_feature_names=geometry_names,
    )
    assert loaded_model.training is False
    assert loaded["deployable"] is False


@pytest.mark.parametrize(
    "target_name",
    ("selected_hierarchical.pth", "result-receipt.json"),
)
def test_hierarchical_publication_never_overwrites_racing_target(
        target_name, monkeypatch, tmp_path):
    protected_paths, protected = _publication_protected(
        monkeypatch, tmp_path
    )
    output = tmp_path / "race"
    reservation = reserve_hierarchical_output(output)
    target = output / target_name
    target.write_bytes(b"existing-target")
    eligible = target_name == "selected_hierarchical.pth"
    artifact = _staged_hierarchical_artifact()[3] if eligible else None

    with pytest.raises(FileExistsError):
        publish_hierarchical_experiment(
            output,
            artifact=artifact,
            result_context=_hierarchical_result_context(eligible=eligible),
            protected_paths=protected_paths,
            protected_before=protected,
            reservation=reservation,
        )

    assert target.read_bytes() == b"existing-target"
    if eligible:
        assert not (output / "result-receipt.json").exists()


@pytest.mark.parametrize("stage", ("artifact", "receipt"))
def test_interrupted_hierarchical_write_leaves_no_completion_receipt(
        stage, monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_hierarchical_reranker as trainer

    protected_paths, protected = _publication_protected(
        monkeypatch, tmp_path
    )
    output = tmp_path / "interrupted_{}".format(stage)
    reservation = reserve_hierarchical_output(output)
    original = trainer._exclusive_write_bytes

    def interrupt(directory_fd, reserved, name, payload, mode=0o444):
        selected_stage = (
            name == "selected_hierarchical.pth" if stage == "artifact"
            else name.startswith(".result-receipt.json.pending")
        )
        if selected_stage:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(
                    descriptor,
                    "partial-{}".format(stage).encode("ascii"),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise RuntimeError("injected {} interruption".format(stage))
        return original(
            directory_fd, reserved, name, payload, mode=mode
        )

    monkeypatch.setattr(trainer, "_exclusive_write_bytes", interrupt)
    eligible = stage == "artifact"
    artifact = _staged_hierarchical_artifact()[3] if eligible else None
    with pytest.raises(RuntimeError, match="interruption"):
        publish_hierarchical_experiment(
            output,
            artifact=artifact,
            result_context=_hierarchical_result_context(eligible=eligible),
            protected_paths=protected_paths,
            protected_before=protected,
            reservation=reservation,
        )

    assert not (output / "result-receipt.json").exists()


def test_interrupted_hierarchical_receipt_link_never_completes(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_hierarchical_reranker as trainer

    protected_paths, protected = _publication_protected(
        monkeypatch, tmp_path
    )
    output = tmp_path / "interrupted_link"
    reservation = reserve_hierarchical_output(output)

    def interrupt_link(*args, **kwargs):
        raise RuntimeError("injected link interruption")

    monkeypatch.setattr(trainer.os, "link", interrupt_link)
    with pytest.raises(RuntimeError, match="link interruption"):
        publish_hierarchical_experiment(
            output,
            artifact=None,
            result_context=_hierarchical_result_context(eligible=False),
            protected_paths=protected_paths,
            protected_before=protected,
            reservation=reservation,
        )

    assert not (output / "result-receipt.json").exists()
    assert (output / ".result-receipt.json.pending").exists()


def test_hierarchical_trainer_cli_exposes_only_fixed_protocol_arguments():
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
            "--device", "cpu",
        ])
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


def _run_orchestration_fixture(
        monkeypatch, tmp_path, eligible=True, calibration_passed=True):
    import scripts.train_scanrefer_rec_hierarchical_reranker as trainer

    fit_rows = [object()] * 33040
    calibration_rows = [object()] * 3625
    fit_records = ["fit-records"]
    calibration_records = ["calibration-records"]
    lifecycle = []
    calls = []
    publications = []
    artifacts = []
    output = tmp_path / "fresh_output"
    reservation = {
        "path": str(output.absolute()),
        "device": 1,
        "inode": 2,
    }
    choice = (
        _eligible_hierarchical_choice()
        if eligible else {
            "eligible": False,
            "selected": "baseline",
            "reason": "no-eligible-configuration",
        }
    )
    if eligible:
        choice.update({
            "switches": 5,
            "delta_hits025": 3,
            "delta_hits050": 1,
        })
    frozen_choice = copy.deepcopy(choice)
    fit_baseline = copy.deepcopy(
        _hierarchical_result_context(eligible=False)["oof"]["baseline"]
    )
    calibration_metrics, calibration_baseline = \
        _authoritative_cache_gate_values()
    if not calibration_passed:
        calibration_metrics["hits025"] = 3523

    def reserve(received):
        lifecycle.append("reserve")
        assert Path(received) == output
        return reservation

    def capture(paths):
        lifecycle.append("capture")
        return _receipt_protected_snapshot()

    def load(*args, **kwargs):
        lifecycle.append("load")
        return {
            "joined_rows": ["joined"],
            "parent": object(),
            "geometry_model": object(),
            "geometry_artifact": _geometry_artifact(),
            "input_sha256": _hierarchical_input_sha256(),
        }

    def split(rows):
        lifecycle.append("split")
        assert rows == ["joined"]
        return {
            "fit_rows": fit_rows,
            "calibration_rows": calibration_rows,
            "metadata": copy.deepcopy(AUTHORITATIVE_SPLIT_SEED0),
        }

    def materialize(rows, *args, **kwargs):
        assert kwargs["require_contiguous"] is False
        assert kwargs["batch_size"] == HIERARCHICAL_MATERIALIZATION_BATCH_SIZE
        if rows is fit_rows:
            calls.append("materialize_fit")
            return fit_records
        if rows is calibration_rows:
            assert choice == frozen_choice
            assert "cross_fit" in calls
            calls.append("materialize_calibration")
            return calibration_records
        pytest.fail("materializer received rows outside the identity split")

    def cross_fit(records, device):
        assert records is fit_records
        calls.append("cross_fit")
        return {
            "choice": choice,
            "scene_folds": {
                "fit_scene_{:02d}".format(index): index % 5
                for index in range(10)
            },
            "scene_fold_sha256": "d" * 64,
            "policy_candidate_count": 1,
            "configurations": [{
                "hidden_dim": 64,
                "weight_decay": 1e-4,
                "false_positive_cost": 2.0,
                "oof_proposal_sha256": "a" * 64,
                "oof_gain_sha256": "b" * 64,
                "prediction_count": 33040,
            }],
        }

    normalization = {"sha256": "8" * 64}
    model = object()

    def refit(records, selected, device):
        assert records is fit_records
        assert selected == frozen_choice
        calls.append("refit")
        return model, normalization

    def baseline(records):
        if records is fit_records:
            return copy.deepcopy(fit_baseline)
        assert records is calibration_records
        return copy.deepcopy(calibration_baseline)

    def evaluate(received_model, records, statistics, margin, device):
        assert received_model is model
        assert records is calibration_records
        assert statistics is normalization
        assert margin == choice["margin"]
        calls.append("evaluate")
        return copy.deepcopy(calibration_metrics)

    def build_artifact(**kwargs):
        calls.append("build_artifact")
        artifacts.append(kwargs)
        return {"staged": kwargs}

    def publish(output_dir, artifact, result_context, protected_paths,
                protected_before, reservation):
        calls.append("publish")
        publications.append({
            "artifact": artifact,
            "context": copy.deepcopy(result_context),
            "reservation": reservation,
        })
        return {"published": True, "artifact": artifact}

    monkeypatch.setattr(trainer, "reserve_hierarchical_output", reserve)
    monkeypatch.setattr(
        trainer, "capture_immutable_artifact_identities", capture
    )
    monkeypatch.setattr(trainer, "load_residual_training_inputs", load)
    monkeypatch.setattr(trainer, "split_residual_joined_rows", split)
    monkeypatch.setattr(trainer, "materialize_hierarchical_rows", materialize)
    monkeypatch.setattr(
        trainer, "canonical_residual_joined_identity_sha256",
        lambda rows: "6" * 64,
    )
    monkeypatch.setattr(
        trainer, "canonical_hierarchical_deployable_sha256",
        lambda records: "7" * 64,
    )
    monkeypatch.setattr(
        trainer, "build_hierarchical_cache_calibration_baseline", baseline
    )
    monkeypatch.setattr(trainer, "cross_fit_hierarchical_reranker", cross_fit)
    monkeypatch.setattr(trainer, "refit_hierarchical_reranker", refit)
    monkeypatch.setattr(
        trainer, "evaluate_hierarchical_cache_policy", evaluate
    )
    monkeypatch.setattr(trainer, "build_hierarchical_artifact", build_artifact)
    monkeypatch.setattr(
        trainer, "publish_hierarchical_experiment", publish
    )
    monkeypatch.setattr(
        trainer, "AUTHORITATIVE_BACKBONE_PATH", tmp_path / "backbone.pth"
    )

    result = run_hierarchical_training(
        tmp_path / "train",
        tmp_path / "geometry_train",
        tmp_path / "parent.pth",
        tmp_path / "geometry.pth",
        output,
        device="cuda:0",
    )
    return {
        "result": result,
        "calls": calls,
        "lifecycle": lifecycle,
        "publications": publications,
        "artifacts": artifacts,
        "choice": frozen_choice,
    }


def test_training_orchestration_refits_and_calibrates_exactly_once(
        monkeypatch, tmp_path):
    observed = _run_orchestration_fixture(monkeypatch, tmp_path)

    assert observed["result"]["published"] is True
    assert observed["lifecycle"][:3] == ["reserve", "capture", "load"]
    assert observed["calls"] == [
        "materialize_fit",
        "cross_fit",
        "refit",
        "materialize_calibration",
        "evaluate",
        "build_artifact",
        "publish",
    ]
    assert len(observed["artifacts"]) == 1
    artifact = observed["artifacts"][0]
    assert artifact["selection"] == observed["choice"]
    assert artifact["normalization"]["sha256"] == "8" * 64
    assert artifact["calibration_record"]["hits025"] == 3524
    context = observed["publications"][0]["context"]
    assert context["fit_normalization_sha256"] == "8" * 64
    assert context["calibration"]["status"] == "run"
    assert context["calibration"]["gate"]["passed"] is True


@pytest.mark.parametrize(
    "eligible,calibration_passed,expected_calls",
    (
        (
            False,
            False,
            ["materialize_fit", "cross_fit", "publish"],
        ),
        (
            True,
            False,
            [
                "materialize_fit",
                "cross_fit",
                "refit",
                "materialize_calibration",
                "evaluate",
                "publish",
            ],
        ),
    ),
)
def test_training_orchestration_never_advances_a_failed_gate(
        eligible, calibration_passed, expected_calls, monkeypatch, tmp_path):
    observed = _run_orchestration_fixture(
        monkeypatch,
        tmp_path,
        eligible=eligible,
        calibration_passed=calibration_passed,
    )

    assert observed["calls"] == expected_calls
    assert observed["artifacts"] == []
    assert observed["publications"][0]["artifact"] is None
    context = observed["publications"][0]["context"]
    if not eligible:
        assert context["fit_normalization_sha256"] is None
        assert context["calibration"] == {
            "status": "not_run",
            "reason": "oof_selection_rejected",
        }
    else:
        assert context["fit_normalization_sha256"] == "8" * 64
        assert context["calibration"]["status"] == "run"
        assert context["calibration"]["gate"]["passed"] is False


def test_training_reserves_output_before_any_protected_or_cache_access(
        monkeypatch, tmp_path):
    import scripts.train_scanrefer_rec_hierarchical_reranker as trainer

    accessed = []

    def reject_reservation(_output):
        accessed.append("reserve")
        raise ValueError("injected reservation rejection")

    monkeypatch.setattr(
        trainer, "reserve_hierarchical_output", reject_reservation
    )
    monkeypatch.setattr(
        trainer,
        "capture_immutable_artifact_identities",
        lambda _paths: pytest.fail("protected files must not be accessed"),
    )
    monkeypatch.setattr(
        trainer,
        "load_residual_training_inputs",
        lambda *_args, **_kwargs: pytest.fail("caches must not be loaded"),
    )

    with pytest.raises(ValueError, match="reservation rejection"):
        run_hierarchical_training(
            tmp_path / "train",
            tmp_path / "geometry_train",
            tmp_path / "parent.pth",
            tmp_path / "geometry.pth",
            tmp_path / "fresh_output",
            device="cuda:0",
        )
    assert accessed == ["reserve"]
