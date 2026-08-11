import json

import pytest
import torch

from scripts.audit_scanrefer_mask_geometry import (
    _prepare_output_staging,
    _publish_output_bundle,
    assert_candidate_cache_parity,
    build_cache_replay_groups,
    cache_row_to_panel_record,
    extract_default_variant_diagnostics,
    load_selected_cache_rows,
    load_train_cache_panel_records,
    parse_args,
    select_baseline_stratified_panel,
    summarize_geometry_metrics,
    summarize_variant_diagnostics,
)


def _cache_row(dataset_index, scan_id, default_iou):
    return {
        "dataset_index": dataset_index,
        "scan_id": scan_id,
        "target_id": dataset_index + 100,
        "default_top1_query_index": 7,
        "query_indices": torch.tensor([3, 7, 9]),
        "valid_mask": torch.tensor([True, True, True]),
        "candidate_ious": torch.tensor([0.1, default_iou, 0.2]),
    }


def test_cache_row_to_panel_record_finds_default_candidate_and_bucket():
    assert cache_row_to_panel_record(
        _cache_row(4, "scene0004_00", 0.25)
    ) == {
        "dataset_index": 4,
        "scan_id": "scene0004_00",
        "target_id": 104,
        "default_position": 1,
        "default_iou": 0.25,
        "bucket": "fail025",
    }
    assert cache_row_to_panel_record(
        _cache_row(5, "scene0005_00", 0.50)
    )["bucket"] == "mid"
    assert cache_row_to_panel_record(
        _cache_row(6, "scene0006_00", 0.50001)
    )["bucket"] == "pass050"


def test_cache_row_to_panel_record_rejects_missing_or_duplicate_default():
    missing = _cache_row(0, "scene", 0.1)
    missing["default_top1_query_index"] = 99
    with pytest.raises(ValueError, match="default Top-1"):
        cache_row_to_panel_record(missing)

    duplicate = _cache_row(0, "scene", 0.1)
    duplicate["query_indices"] = torch.tensor([7, 7, 9])
    with pytest.raises(ValueError, match="exactly once"):
        cache_row_to_panel_record(duplicate)


def _panel_records():
    records = []
    index = 0
    for scene_idx in range(4):
        scan_id = "scene{:04d}_00".format(scene_idx)
        for iou in (0.10, 0.30, 0.70, 0.80, 0.90):
            records.append(cache_row_to_panel_record(
                _cache_row(index, scan_id, iou)
            ))
            index += 1
    # This scene is ineligible because it has no fail025 row.
    for iou in (0.30, 0.40, 0.70, 0.80):
        records.append(cache_row_to_panel_record(
            _cache_row(index, "scene9999_00", iou)
        ))
        index += 1
    return records


def test_stratified_panel_is_deterministic_and_covers_each_bucket():
    first = select_baseline_stratified_panel(
        _panel_records(), scene_count=2, expressions_per_scene=4, seed=17
    )
    second = select_baseline_stratified_panel(
        list(reversed(_panel_records())),
        scene_count=2,
        expressions_per_scene=4,
        seed=17,
    )

    assert first == second
    assert len(first) == 8
    selected_scenes = {row["scan_id"] for row in first}
    assert len(selected_scenes) == 2
    assert "scene9999_00" not in selected_scenes
    for scan_id in selected_scenes:
        rows = [row for row in first if row["scan_id"] == scan_id]
        assert len(rows) == 4
        assert {row["bucket"] for row in rows} == {
            "fail025", "mid", "pass050"
        }
        assert len({row["dataset_index"] for row in rows}) == 4


def test_stratified_panel_fails_closed_when_too_few_scenes_are_eligible():
    with pytest.raises(ValueError, match="eligible scenes"):
        select_baseline_stratified_panel(
            _panel_records(), scene_count=5, expressions_per_scene=4, seed=0
        )


def test_geometry_summary_uses_strict_thresholds_fallback_and_pool_oracle():
    regressed_ious = torch.tensor([
        [0.20, 0.10],
        [0.40, 0.60],
        [0.70, 0.10],
    ])
    regressed_valid = torch.ones_like(regressed_ious, dtype=torch.bool)
    geometry_ious = torch.tensor([
        [[0.20, 0.30], [0.10, 0.55]],
        [[0.40, 0.20], [0.60, 0.70]],
        [[0.70, 0.80], [0.10, 0.90]],
    ])
    geometry_valid = torch.tensor([
        [[True, True], [True, True]],
        [[True, True], [True, True]],
        [[True, False], [True, False]],
    ])

    summary = summarize_geometry_metrics(
        regressed_ious=regressed_ious,
        regressed_valid=regressed_valid,
        geometry_ious=geometry_ious,
        geometry_valid=geometry_valid,
        default_positions=torch.tensor([0, 0, 0]),
        variant_names=("regressed", "mask"),
    )

    assert summary["sample_count"] == 3
    assert summary["baseline_default"]["acc025"] == pytest.approx(2 / 3)
    assert summary["baseline_default"]["acc050"] == pytest.approx(1 / 3)
    assert summary["baseline_oracle"]["acc025"] == pytest.approx(2 / 3)
    assert summary["baseline_oracle"]["acc050"] == pytest.approx(2 / 3)

    mask = summary["variants"]["mask"]
    assert mask["fallback_acc025"] == pytest.approx(2 / 3)
    assert mask["fallback_acc050"] == pytest.approx(1 / 3)
    assert mask["raw_acc025"] == pytest.approx(1 / 3)
    assert mask["raw_acc050"] == pytest.approx(0.0)
    assert mask["fixes025"] == 1
    assert mask["breaks025"] == 1
    assert mask["fixes050"] == 0
    assert mask["breaks050"] == 0
    assert mask["invalid_count"] == 1
    assert mask["invalid_rate"] == pytest.approx(1 / 3)
    assert mask["augmented_oracle_acc025"] == pytest.approx(1.0)
    assert mask["augmented_oracle_acc050"] == pytest.approx(1.0)
    assert mask["oracle_fixes025"] == 1
    assert mask["oracle_fixes050"] == 1

    assert summary["combined_oracle"]["acc025"] == pytest.approx(1.0)
    assert summary["combined_oracle"]["acc050"] == pytest.approx(1.0)
    assert summary["combined_oracle"]["breaks025"] == 0
    assert summary["combined_oracle"]["breaks050"] == 0


def test_geometry_summary_does_not_count_iou_equal_to_threshold():
    summary = summarize_geometry_metrics(
        regressed_ious=torch.tensor([[0.25, 0.50]]),
        regressed_valid=torch.tensor([[True, True]]),
        geometry_ious=torch.tensor([[[0.25], [0.50]]]),
        geometry_valid=torch.tensor([[[True], [True]]]),
        default_positions=torch.tensor([0]),
        variant_names=("regressed",),
    )

    assert summary["baseline_default"]["acc025"] == 0.0
    assert summary["baseline_oracle"]["acc050"] == 0.0
    assert summary["combined_oracle"]["acc050"] == 0.0


def _write_synthetic_cache(path, split="train", fingerprint="abc"):
    path.mkdir()
    rows = [
        _cache_row(0, "scene0000_00", 0.1),
        _cache_row(1, "scene0001_00", 0.4),
    ]
    torch.save({"rows": rows}, path / "shard_000000.pt")
    manifest = {
        "cache_schema_version": 1,
        "feature_schema_version": "rec-query-v1",
        "split": split,
        "checkpoint_sha256": fingerprint,
        "checkpoint_epoch": 71,
        "target_iou_policy": "root_only",
        "sample_count": len(rows),
        "dataset_size": len(rows),
        "source_dataset_size": len(rows),
        "deterministic": True,
        "feature_dim": 152,
        "feature_names": ["feature_{}".format(idx) for idx in range(152)],
        "model_inputs": {"use_color": True},
        "backbone_config": {"model": "MCLN"},
        "candidate_rule": {"topk_per_source": 8, "max_candidates": 3},
        "shards": ["shard_000000.pt"],
    }
    with (path / "manifest.json").open("w") as handle:
        json.dump(manifest, handle)
    return rows


def test_train_cache_loader_validates_provenance_and_recovers_selected_rows(
        tmp_path):
    cache_path = tmp_path / "cache"
    source_rows = _write_synthetic_cache(cache_path)

    manifest, records = load_train_cache_panel_records(
        cache_path, expected_checkpoint_sha256="abc"
    )
    selected = load_selected_cache_rows(
        cache_path, manifest, dataset_indices={1}
    )

    assert manifest["sample_count"] == 2
    assert [row["dataset_index"] for row in records] == [0, 1]
    assert set(selected) == {1}
    assert torch.equal(selected[1]["candidate_ious"], source_rows[1][
        "candidate_ious"
    ])


@pytest.mark.parametrize(
    "split,fingerprint,target_policy,error",
    [
        ("val", "abc", "root_only", "train split"),
        ("train", "wrong", "root_only", "checkpoint"),
        ("train", "abc", "all_targets", "root_only"),
    ],
)
def test_train_cache_loader_rejects_wrong_provenance(
        tmp_path, split, fingerprint, target_policy, error):
    cache_path = tmp_path / "cache"
    _write_synthetic_cache(cache_path, split=split, fingerprint=fingerprint)
    manifest_path = cache_path / "manifest.json"
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    manifest["target_iou_policy"] = target_policy
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match=error):
        load_train_cache_panel_records(
            cache_path, expected_checkpoint_sha256="abc"
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda manifest: manifest.pop("cache_schema_version"), "schema"),
        (lambda manifest: manifest.update({"deterministic": False}),
         "deterministic"),
        (lambda manifest: manifest.update({"source_dataset_size": 3}),
         "complete"),
        (lambda manifest: manifest.update({"feature_dim": 151}),
         "feature"),
    ],
)
def test_train_cache_loader_rejects_incomplete_manifest(
        tmp_path, mutation, error):
    cache_path = tmp_path / "cache"
    _write_synthetic_cache(cache_path)
    manifest_path = cache_path / "manifest.json"
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    mutation(manifest)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match=error):
        load_train_cache_panel_records(
            cache_path, expected_checkpoint_sha256="abc"
        )


def test_variant_diagnostics_decode_rejections_and_distribution_stats():
    diagnostics = summarize_variant_diagnostics(
        rejection_codes=torch.tensor([
            [0, 0],
            [0, 1],
            [0, 4 | 8],
            [0, 128],
        ]),
        default_variant_valid=torch.tensor([
            [True, True],
            [True, False],
            [True, False],
            [True, False],
        ]),
        selected_point_fractions=torch.tensor([
            [0.0, 0.2],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.4],
        ]),
        final_volume_ratios=torch.tensor([
            [1.0, 1.5],
            [1.0, 0.0],
            [1.0, 3.0],
            [1.0, 0.5],
        ]),
        variant_names=("regressed", "mask"),
    )

    mask = diagnostics["mask"]
    assert mask["rejections"] == {
        "empty": 1,
        "too_few": 0,
        "full": 1,
        "over_fraction": 1,
        "nonfinite_logits": 0,
        "nonfinite_coordinates": 0,
        "nonfinite_box": 0,
        "degenerate": 1,
    }
    assert mask["foreground_fraction"]["count"] == 4
    assert mask["foreground_fraction"]["mean"] == pytest.approx(0.4)
    assert mask["foreground_fraction"]["median"] == pytest.approx(0.3)
    assert mask["final_volume_ratio"]["count"] == 1
    assert mask["final_volume_ratio"]["mean"] == pytest.approx(1.5)


def _parity_candidate_batch():
    return {
        "query_indices": torch.tensor([[3, 7, 9]]),
        "valid_mask": torch.tensor([[True, True, True]]),
        "boxes": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 3]),
        "candidate_ious": torch.tensor([[0.1, 0.4, 0.2]]),
        "features": torch.tensor([[[0.1, 0.2]] * 3]),
        "default_scores": torch.tensor([[0.3, 0.8, 0.1]]),
        "contrastive_scores": torch.tensor([[0.4, 0.7, 0.2]]),
        "default_top1_query_index": torch.tensor([7]),
    }


def test_candidate_cache_parity_checks_identity_queries_boxes_and_ious():
    candidate_batch = _parity_candidate_batch()
    cached_row = _cache_row(5, "scene0005_00", 0.4)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()

    assert_candidate_cache_parity(
        candidate_batch,
        cached_rows={5: cached_row},
        dataset_indices=[5],
        scan_ids=["scene0005_00"],
        target_ids=torch.tensor([105]),
    )

    candidate_batch["boxes"][0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="boxes"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0005_00"],
            target_ids=torch.tensor([105]),
        )

    candidate_batch = _parity_candidate_batch()
    candidate_batch["features"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0005_00"],
            target_ids=torch.tensor([105]),
        )


def test_candidate_cache_parity_rejects_default_bucket_drift():
    candidate_batch = _parity_candidate_batch()
    candidate_batch["candidate_ious"][0, 1] = 0.2495
    cached_row = _cache_row(5, "scene0005_00", 0.2505)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()

    with pytest.raises(ValueError, match="bucket"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0005_00"],
            target_ids=torch.tensor([105]),
            atol=0.01,
            rtol=0.0,
        )


def test_candidate_cache_parity_reports_finite_feature_drift_diagnostically():
    candidate_batch = _parity_candidate_batch()
    cached_row = _cache_row(5, "scene0005_00", 0.4)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()
    candidate_batch["features"][0, 0, 0] += 0.2

    differences = assert_candidate_cache_parity(
        candidate_batch,
        cached_rows={5: cached_row},
        dataset_indices=[5],
        scan_ids=["scene0005_00"],
        target_ids=torch.tensor([105]),
    )

    assert differences["features"] == pytest.approx(0.2, abs=1e-6)

    candidate_batch = _parity_candidate_batch()
    candidate_batch["boxes"][0, 0, 0] += 0.008
    with pytest.raises(ValueError, match="boxes"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0005_00"],
            target_ids=torch.tensor([105]),
        )


def test_candidate_cache_parity_reports_finite_score_drift_diagnostically():
    candidate_batch = _parity_candidate_batch()
    cached_row = _cache_row(5, "scene0005_00", 0.4)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()
    candidate_batch["default_scores"][0, 0] += 0.2
    candidate_batch["contrastive_scores"][0, 0] -= 0.2

    differences = assert_candidate_cache_parity(
        candidate_batch,
        cached_rows={5: cached_row},
        dataset_indices=[5],
        scan_ids=["scene0005_00"],
        target_ids=torch.tensor([105]),
    )

    assert differences["default_scores"] == pytest.approx(0.2, abs=1e-6)
    assert differences["contrastive_scores"] == pytest.approx(
        0.2, abs=1e-6
    )

    candidate_batch["default_scores"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0005_00"],
            target_ids=torch.tensor([105]),
        )


def test_candidate_cache_parity_bounds_iou_drift_separately_from_boxes():
    candidate_batch = _parity_candidate_batch()
    cached_row = _cache_row(5, "scene0005_00", 0.4)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()
    candidate_batch["candidate_ious"][0, 0] += 0.008

    differences = assert_candidate_cache_parity(
        candidate_batch,
        cached_rows={5: cached_row},
        dataset_indices=[5],
        scan_ids=["scene0005_00"],
        target_ids=torch.tensor([105]),
    )
    assert differences["candidate_ious"] == pytest.approx(0.008, abs=1e-6)

    candidate_batch["candidate_ious"][0, 0] += 0.02
    with pytest.raises(ValueError, match="candidate_ious"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0005_00"],
            target_ids=torch.tensor([105]),
        )


def test_candidate_cache_identity_only_reports_drift_without_rejecting_it():
    candidate_batch = _parity_candidate_batch()
    cached_row = _cache_row(5, "scene0005_00", 0.4)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()

    # A post-cache candidate implementation may reorder a boundary query while
    # preserving the selected row's scene/target identity.
    candidate_batch["query_indices"][0] = torch.tensor([3, 9, 7])
    candidate_batch["default_top1_query_index"][0] = 9
    candidate_batch["default_scores"][0, 1] += 0.25

    diagnostics = assert_candidate_cache_parity(
        candidate_batch,
        cached_rows={5: cached_row},
        dataset_indices=[5],
        scan_ids=["scene0005_00"],
        target_ids=torch.tensor([105]),
        identity_only=True,
    )

    assert diagnostics["query_identity_drift_count"] == 1
    assert diagnostics["default_query_identity_drift_count"] == 1
    assert diagnostics["default_scores"] == pytest.approx(0.25, abs=1e-6)


def test_candidate_cache_identity_only_still_rejects_row_identity_drift():
    candidate_batch = _parity_candidate_batch()
    cached_row = _cache_row(5, "scene0005_00", 0.4)
    cached_row["boxes"] = candidate_batch["boxes"][0].clone()
    for key in ("features", "default_scores", "contrastive_scores"):
        cached_row[key] = candidate_batch[key][0].clone()

    with pytest.raises(ValueError, match="scan identity"):
        assert_candidate_cache_parity(
            candidate_batch,
            cached_rows={5: cached_row},
            dataset_indices=[5],
            scan_ids=["scene0006_00"],
            target_ids=torch.tensor([105]),
            identity_only=True,
        )


def test_cache_replay_groups_restore_original_contiguous_batches():
    groups = build_cache_replay_groups(
        selected_indices=[5, 6, 9],
        source_dataset_size=11,
        extraction_batch_size=4,
    )

    assert groups == [
        {
            "batch_indices": (4, 5, 6, 7),
            "selected_indices": (5, 6),
            "selected_positions": (1, 2),
            "replay_boundary": 0,
        },
        {
            "batch_indices": (8, 9, 10),
            "selected_indices": (9,),
            "selected_positions": (1,),
            "replay_boundary": 0,
        },
    ]


def test_cache_replay_groups_honor_resume_boundaries_with_overlap():
    groups = build_cache_replay_groups(
        selected_indices=[33279, 33280, 33291],
        source_dataset_size=33300,
        extraction_batch_size=12,
        replay_boundaries=(0, 33280),
    )

    assert groups == [
        {
            "batch_indices": tuple(range(33276, 33288)),
            "selected_indices": (33279,),
            "selected_positions": (3,),
            "replay_boundary": 0,
        },
        {
            "batch_indices": tuple(range(33280, 33292)),
            "selected_indices": (33280, 33291),
            "selected_positions": (0, 11),
            "replay_boundary": 33280,
        },
    ]


def test_extract_default_variant_diagnostics_maps_quantile_and_volume():
    geometry = {
        "boxes": torch.tensor([[  # B=1, K=2, G=2
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
            ],
        ]]),
        "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "variant_configs": (
            {
                "name": "regressed", "source": "regressed",
                "logit_threshold": 0.0, "quantile": 0.0,
                "regressed_weight": 1.0,
            },
            {
                "name": "fused_q", "source": "fused",
                "logit_threshold": 0.0, "quantile": 0.005,
                "regressed_weight": 0.0,
            },
        ),
        "mask_diagnostics": ({
            "fused_t0": {
                "quantiles": torch.tensor([0.0, 0.005]),
                "rejection_codes": torch.tensor([[1, 2], [4, 0]]),
                "selected_point_fractions": torch.tensor([0.0, 0.2]),
            },
        },),
    }

    extracted = extract_default_variant_diagnostics(
        geometry, default_positions=torch.tensor([1])
    )

    assert extracted["rejection_codes"].tolist() == [[0, 0]]
    assert torch.allclose(
        extracted["selected_point_fractions"], torch.tensor([[0.0, 0.2]])
    )
    assert torch.allclose(
        extracted["final_volume_ratios"], torch.tensor([[1.0, 1.0]])
    )
    assert extracted["default_variant_valid"].tolist() == [[True, True]]


def test_parse_args_validates_panel_and_runtime_sizes(tmp_path):
    required = [
        "--data-root", str(tmp_path),
        "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--train-cache", str(tmp_path / "cache"),
        "--output-dir", str(tmp_path / "output"),
    ]
    args = parse_args(required)
    assert args.scene_count == 64
    assert args.expressions_per_scene == 4
    assert args.batch_size == 4
    assert args.cache_extraction_batch_size == 4
    assert args.cache_replay_boundaries == [0]
    assert args.min_points == 5
    assert args.max_point_fraction == pytest.approx(0.5)

    with pytest.raises(SystemExit):
        parse_args(required + ["--scene-count", "0"])
    with pytest.raises(SystemExit):
        parse_args(required + ["--max-point-fraction", "1.1"])


def test_output_bundle_keeps_old_result_until_complete_publish(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    (output / "summary.json").write_text("old")

    final_path, staging = _prepare_output_staging(output, overwrite=True)
    assert final_path == output
    assert (output / "summary.json").read_text() == "old"
    for name in ("selection.json", "summary.json", "rows.pt"):
        (staging / name).write_text("new")

    _publish_output_bundle(final_path, staging)

    assert not staging.exists()
    assert (output / "summary.json").read_text() == "new"
    assert sorted(path.name for path in output.iterdir()) == [
        "rows.pt", "selection.json", "summary.json"
    ]


def test_output_bundle_refuses_existing_result_without_overwrite(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()

    with pytest.raises(ValueError, match="overwrite"):
        _prepare_output_staging(output, overwrite=False)
