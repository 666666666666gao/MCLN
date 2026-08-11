import copy
import json
from pathlib import Path

import pytest
import torch

from models.rec_reranker import QueryReranker
from scripts.train_rec_reranker import (
    ARTIFACT_VERSION,
    calibration_score,
    choose_best_reranker_blend,
    compute_feature_stats,
    deterministic_scene_split,
    evaluate_reranker,
    load_candidate_cache,
    load_reranker_artifact,
    load_training_cache,
    normalize_backbone_config,
    normalize_features,
    parse_args,
    save_reranker_artifact,
    select_best_reranker_blend,
    train_reranker,
)


def test_backbone_config_normalization_is_legacy_compatible_and_fail_closed():
    legacy = {
        "model": "MCLN",
        "num_target": 256,
        "num_decoder_layers": 6,
        "self_position_embedding": "loc_learned",
        "self_attend": True,
        "use_soft_token_loss": True,
        "use_contrastive_align": True,
        "detect_intermediate": True,
        "use_source_choice_selector": False,
        "source_choice_selector_sources": "default,mask_text",
        "source_choice_selector_hidden_dim": 288,
    }

    normalized = normalize_backbone_config(legacy)

    assert "use_source_moe" not in legacy
    assert normalized["use_source_moe"] is False
    assert normalized["source_moe_query_layers"] == 1
    assert normalized["source_moe_use_fallback_gate"] is False
    assert normalized["source_moe_gate_candidate_top_k"] == 8
    assert normalized["source_moe_gate_uncertainty_weight"] == pytest.approx(0.0)
    assert normalized["source_moe_gate_use_evidence_features"] is False
    partial = dict(legacy, use_source_moe=True)
    with pytest.raises(ValueError, match="incomplete"):
        normalize_backbone_config(partial)

    complete_moe = dict(legacy)
    complete_moe.update({
        "use_source_moe": True,
        "source_moe_shared_source": "default",
        "source_moe_top_k": 1,
        "source_moe_balance_loss_weight": 0.01,
        "source_moe_query_layers": 1,
        "source_moe_query_heads": 4,
        "source_moe_query_dropout": 0.1,
        "source_moe_query_max_delta": 0.25,
    })
    with pytest.raises(ValueError, match="fallback-gate config is incomplete"):
        normalize_backbone_config(dict(
            complete_moe, source_moe_use_fallback_gate=True
        ))

    complete_gate = dict(complete_moe, **{
        "source_moe_use_fallback_gate": True,
        "source_moe_gate_hidden_dim": 128,
        "source_moe_gate_candidate_top_k": 8,
        "source_moe_gate_break_cost": 2.0,
        "source_moe_gate_decision_margin": 0.0,
        "source_moe_gate_mask_utility_weight": 0.25,
    })
    normalized_old_gate = normalize_backbone_config(complete_gate)
    assert normalized_old_gate[
        "source_moe_gate_use_evidence_features"
    ] is False
    normalized_cascade = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_absolute_quality_correction"
        ),
    ))
    assert normalized_cascade["source_moe_gate_action_mode"] == (
        "cascade_absolute_quality_correction"
    )
    normalized_opportunity = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_opportunity_quality_correction"
        ),
    ))
    assert normalized_opportunity["source_moe_gate_action_mode"] == (
        "cascade_opportunity_quality_correction"
    )
    normalized_verified = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_opportunity_verified_correction"
        ),
    ))
    assert normalized_verified["source_moe_gate_action_mode"] == (
        "cascade_opportunity_verified_correction"
    )
    normalized_joint = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode="cascade_joint_risk_correction",
    ))
    assert normalized_joint["source_moe_gate_action_mode"] == (
        "cascade_joint_risk_correction"
    )
    normalized_v21 = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_v19_fallback_set_correction"
        ),
    ))
    assert normalized_v21["source_moe_gate_action_mode"] == (
        "cascade_v19_fallback_set_correction"
    )
    normalized_v22 = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_v19_rich_set_correction"
        ),
    ))
    assert normalized_v22["source_moe_gate_action_mode"] == (
        "cascade_v19_rich_set_correction"
    )
    normalized_v23 = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_v23_dense_quality_correction"
        ),
        source_moe_gate_uncertainty_weight=0.5,
    ))
    assert normalized_v23["source_moe_gate_action_mode"] == (
        "cascade_v23_dense_quality_correction"
    )
    assert normalized_v23["source_moe_gate_uncertainty_weight"] == pytest.approx(
        0.5
    )
    with pytest.raises(ValueError, match="numeric config"):
        normalize_backbone_config(dict(
            complete_gate, source_moe_gate_uncertainty_weight=-0.1
        ))
    normalized_v26 = normalize_backbone_config(dict(
        complete_gate,
        source_moe_gate_action_mode=(
            "cascade_v26_prior_restored_pairwise_correction"
        ),
    ))
    assert normalized_v26["source_moe_gate_action_mode"] == (
        "cascade_v26_prior_restored_pairwise_correction"
    )
    with pytest.raises(ValueError, match="require the fallback gate"):
        normalize_backbone_config(dict(
            complete_moe,
            source_moe_gate_use_evidence_features=True,
        ))


def _row(index, scan_id, features, candidate_ious, valid_mask=None,
         default_position=0):
    features = torch.as_tensor(features, dtype=torch.float32)
    candidate_ious = torch.as_tensor(candidate_ious, dtype=torch.float32)
    num_candidates = features.shape[0]
    if valid_mask is None:
        valid_mask = torch.ones(num_candidates, dtype=torch.bool)
    else:
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
    query_indices = torch.arange(num_candidates, dtype=torch.long) + 10
    default_scores = torch.linspace(
        1.0, 0.0, steps=num_candidates, dtype=torch.float32
    )
    default_scores[default_position] = 2.0
    return {
        "dataset_index": int(index),
        "scan_id": scan_id,
        "target_id": index,
        "features": features,
        "boxes": torch.zeros(num_candidates, 6),
        "query_indices": query_indices,
        "valid_mask": valid_mask,
        "default_scores": default_scores,
        "contrastive_scores": -default_scores,
        "candidate_ious": candidate_ious,
        "default_top1_query_index": int(query_indices[default_position]),
    }


def _write_cache(cache_dir, rows, manifest_updates=None, shard_count=2):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_dim = int(rows[0]["features"].shape[-1])
    num_candidates = int(rows[0]["features"].shape[0])
    shard_count = max(1, min(int(shard_count), len(rows)))
    shard_names = []
    start = 0
    for shard_index in range(shard_count):
        remaining = len(rows) - start
        slots = shard_count - shard_index
        stop = start + (remaining + slots - 1) // slots
        shard_name = "shard_{:06d}.pt".format(shard_index)
        torch.save({"rows": rows[start:stop]}, cache_dir / shard_name)
        shard_names.append(shard_name)
        start = stop
    manifest = {
        "cache_schema_version": 1,
        "feature_schema_version": "rec-query-v1",
        "checkpoint_sha256": "synthetic-checkpoint-sha256",
        "checkpoint_epoch": 71,
        "split": "train",
        "candidate_rule": {
            "topk_per_source": 8,
            "max_candidates": num_candidates,
        },
        "feature_dim": feature_dim,
        "feature_names": [
            "feature_{}".format(index) for index in range(feature_dim)
        ],
        "target_iou_policy": "root_only",
        "model_inputs": {
            "use_color": True,
            "use_height": False,
            "use_multiview": False,
            "butd": True,
            "butd_gt": False,
            "butd_cls": False,
        },
        "backbone_config": {
            "model": "MCLN",
            "num_target": 256,
            "num_decoder_layers": 6,
            "self_position_embedding": "loc_learned",
            "self_attend": True,
            "use_soft_token_loss": True,
            "use_contrastive_align": True,
            "detect_intermediate": True,
            "use_source_choice_selector": True,
            "source_choice_selector_sources": (
                "default,default_rank_blend_contrastive010"
            ),
            "source_choice_selector_hidden_dim": 288,
        },
        "dataset_size": len(rows),
        "source_dataset_size": len(rows),
        "deterministic": True,
        "sample_count": len(rows),
        "shards": shard_names,
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _basic_rows(num_scenes=12, rows_per_scene=2):
    rows = []
    for scene_index in range(num_scenes):
        for _ in range(rows_per_scene):
            index = len(rows)
            rows.append(_row(
                index,
                "scene{:04d}_00".format(scene_index),
                [[-1.0, float(scene_index)], [1.0, float(scene_index)]],
                [0.1, 0.8],
            ))
    return rows


def test_scene_split_is_deterministic_disjoint_and_nonempty_when_possible():
    rows = _basic_rows()

    fit_a, calibration_a = deterministic_scene_split(rows, seed=19)
    fit_b, calibration_b = deterministic_scene_split(rows, seed=19)

    fit_scenes = {row["scan_id"] for row in fit_a}
    calibration_scenes = {row["scan_id"] for row in calibration_a}
    assert fit_scenes
    assert calibration_scenes
    assert fit_scenes.isdisjoint(calibration_scenes)
    assert fit_scenes | calibration_scenes == {
        row["scan_id"] for row in rows
    }
    assert [row["dataset_index"] for row in fit_a] == [
        row["dataset_index"] for row in fit_b
    ]
    assert [row["dataset_index"] for row in calibration_a] == [
        row["dataset_index"] for row in calibration_b
    ]
    assert len(calibration_scenes) == 1

    two_scene_rows = _basic_rows(num_scenes=2, rows_per_scene=1)
    two_fit, two_calibration = deterministic_scene_split(
        two_scene_rows, seed=19
    )
    assert len(two_fit) == 1
    assert len(two_calibration) == 1


def test_scene_split_keeps_a_single_scene_in_fit_partition():
    rows = _basic_rows(num_scenes=1, rows_per_scene=3)

    fit_rows, calibration_rows = deterministic_scene_split(rows, seed=3)

    assert fit_rows == rows
    assert calibration_rows == []


def test_feature_stats_use_only_valid_fit_candidates_and_clamp_std():
    fit_rows = [
        _row(
            0,
            "scene0000_00",
            [[1.0, 5.0], [3.0, 5.0], [999.0, 999.0]],
            [0.1, 0.8, 0.0],
            valid_mask=[True, True, False],
        )
    ]
    calibration_row = _row(
        1,
        "scene0001_00",
        [[101.0, 205.0], [103.0, 205.0], [105.0, 205.0]],
        [0.1, 0.8, 0.2],
    )

    mean, std = compute_feature_stats(fit_rows, min_std=1e-4)

    assert torch.allclose(mean, torch.tensor([2.0, 5.0]))
    assert torch.allclose(std, torch.tensor([1.0, 1e-4]))
    assert not torch.equal(mean, calibration_row["features"].mean(dim=0))


def test_normalization_zeroes_invalid_padding_and_standardizes_valid_values():
    features = torch.tensor([[[1.0, 5.0], [3.0, 5.0], [777.0, 888.0]]])
    valid_mask = torch.tensor([[True, True, False]])
    mean = torch.tensor([2.0, 5.0])
    std = torch.tensor([1.0, 1e-4])

    normalized = normalize_features(features, valid_mask, mean, std)

    assert torch.allclose(normalized[0, 0], torch.tensor([-1.0, 0.0]))
    assert torch.allclose(normalized[0, 1], torch.tensor([1.0, 0.0]))
    assert torch.equal(normalized[0, 2], torch.zeros(2))


@pytest.mark.parametrize(
    "manifest_update,error",
    [
        ({"split": "val"}, "training split"),
        ({"cache_schema_version": 99}, "cache schema"),
        ({"feature_schema_version": "other-schema"}, "feature schema"),
        ({"target_iou_policy": "all_targets"}, "target IoU policy"),
    ],
)
def test_training_cache_rejects_non_train_schema_or_policy(
        tmp_path, manifest_update, error):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, _basic_rows(num_scenes=2), manifest_update)

    with pytest.raises(ValueError, match=error):
        load_training_cache(cache_dir)


def test_candidate_cache_loads_val_and_rejects_cross_split(tmp_path):
    cache_dir = tmp_path / "cache"
    rows = _basic_rows(num_scenes=2)
    manifest = _write_cache(
        cache_dir, rows, manifest_updates={"split": "val"}
    )

    loaded_rows, loaded_manifest = load_candidate_cache(
        cache_dir, expected_split="val"
    )

    assert loaded_manifest == manifest
    assert [row["dataset_index"] for row in loaded_rows] == list(
        range(len(rows))
    )
    with pytest.raises(ValueError, match="split"):
        load_candidate_cache(cache_dir, expected_split="train")

    train_cache_dir = tmp_path / "train-cache"
    _write_cache(train_cache_dir, rows)
    with pytest.raises(ValueError, match="split"):
        load_candidate_cache(train_cache_dir, expected_split="val")

    with pytest.raises(ValueError, match="expected_split"):
        load_candidate_cache(cache_dir, expected_split="test")


def test_training_cache_rejects_noncontiguous_shard_names(tmp_path):
    cache_dir = tmp_path / "cache"
    manifest = _write_cache(
        cache_dir, _basic_rows(num_scenes=2), shard_count=1
    )
    old_path = cache_dir / manifest["shards"][0]
    new_path = cache_dir / "shard_000001.pt"
    old_path.rename(new_path)
    manifest["shards"] = [new_path.name]
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="non-contiguous shard"):
        load_training_cache(cache_dir)


def test_training_cache_rejects_malformed_shards_and_rows(tmp_path):
    malformed_shard_dir = tmp_path / "malformed-shard"
    manifest = _write_cache(
        malformed_shard_dir, _basic_rows(num_scenes=2), shard_count=1
    )
    torch.save(
        {"not_rows": []}, malformed_shard_dir / manifest["shards"][0]
    )
    with pytest.raises(ValueError, match="row list"):
        load_training_cache(malformed_shard_dir)

    malformed_row_dir = tmp_path / "malformed-row"
    rows = _basic_rows(num_scenes=2)
    rows[0] = dict(rows[0])
    rows[0]["features"] = torch.zeros(2, 3)
    _write_cache(malformed_row_dir, rows, shard_count=1)
    manifest_path = malformed_row_dir / "manifest.json"
    malformed_manifest = json.loads(manifest_path.read_text())
    malformed_manifest["feature_dim"] = 2
    malformed_manifest["feature_names"] = ["feature_0", "feature_1"]
    manifest_path.write_text(json.dumps(malformed_manifest))
    with pytest.raises(ValueError, match="features"):
        load_training_cache(malformed_row_dir)


def test_training_cache_loads_cpu_rows_and_checks_sample_count(tmp_path):
    rows = _basic_rows(num_scenes=3)
    cache_dir = tmp_path / "cache"
    manifest = _write_cache(cache_dir, rows)

    loaded_rows, loaded_manifest = load_training_cache(cache_dir)

    assert loaded_manifest == manifest
    assert [row["dataset_index"] for row in loaded_rows] == list(
        range(len(rows))
    )
    assert all(
        not value.is_cuda
        for row in loaded_rows for value in row.values()
        if isinstance(value, torch.Tensor)
    )

    manifest["sample_count"] += 1
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sample count"):
        load_training_cache(cache_dir)


def test_training_cache_rejects_an_incomplete_extraction(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_cache(
        cache_dir,
        _basic_rows(num_scenes=2),
        manifest_updates={"dataset_size": 100, "source_dataset_size": 100},
    )

    with pytest.raises(ValueError, match="incomplete"):
        load_training_cache(cache_dir)


def test_training_cache_rejects_a_completed_limited_extraction(tmp_path):
    cache_dir = tmp_path / "cache"
    rows = _basic_rows(num_scenes=2)
    _write_cache(
        cache_dir,
        rows,
        manifest_updates={"source_dataset_size": len(rows) + 100},
    )

    with pytest.raises(ValueError, match="limited"):
        load_training_cache(cache_dir)


def test_artifact_round_trip_restores_identical_logits_and_metadata(tmp_path):
    torch.manual_seed(5)
    model = QueryReranker(input_dim=2, hidden_dim=8, dropout=0.0).eval()
    manifest = _write_cache(
        tmp_path / "cache", _basic_rows(num_scenes=2), shard_count=1
    )
    mean = torch.tensor([1.5, -2.0])
    std = torch.tensor([0.5, 4.0])
    features = torch.tensor([
        [[1.0, -1.0], [2.0, 3.0], [99.0, 99.0]]
    ])
    valid_mask = torch.tensor([[True, True, False]])
    normalized = normalize_features(features, valid_mask, mean, std)
    with torch.no_grad():
        expected = model(normalized, valid_mask)["ranking_logits"]

    output = tmp_path / "reranker.pth"
    artifact = save_reranker_artifact(
        output,
        model,
        mean,
        std,
        manifest,
        epoch=7,
        calibration_metrics={"acc025": 0.75, "acc050": 0.5},
        training_args={"seed": 5, "lr": 0.01},
    )
    restored, loaded = load_reranker_artifact(output, device="cpu")
    with torch.no_grad():
        actual = restored(normalized, valid_mask)["ranking_logits"]

    assert torch.equal(actual, expected)
    assert loaded["artifact_version"] == ARTIFACT_VERSION
    assert loaded["adapter_schema_version"] == "rec-query-v1"
    assert loaded["input_dim"] == 2
    assert loaded["feature_names"] == ["feature_0", "feature_1"]
    assert loaded["candidate_rule"] == manifest["candidate_rule"]
    assert loaded["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert loaded["target_iou_policy"] == "root_only"
    assert loaded["model_inputs"] == manifest["model_inputs"]
    assert loaded["backbone_config"] == manifest["backbone_config"]
    assert loaded["score_mode"] == "rank_blend"
    assert loaded["reranker_weight"] == 1.0
    assert loaded["epoch"] == 7
    assert loaded["calibration_metrics"]["acc050"] == 0.5
    assert loaded["training_args"]["seed"] == 5
    assert torch.equal(artifact["feature_mean"], mean)
    assert not list(tmp_path.glob("*.tmp"))


def test_calibration_blend_can_fall_back_to_the_default_ranking():
    rows = [
        _row(
            index,
            "scene{:04d}_00".format(index),
            [[-1.0, 0.0], [1.0, 0.0]],
            [0.8, 0.1],
            default_position=0,
        )
        for index in range(4)
    ]

    class WrongReranker(torch.nn.Module):
        def forward(self, features, valid_mask):
            logits = features.new_tensor([[0.0, 1.0]]).expand(
                features.shape[0], -1
            )
            return {"ranking_logits": logits.masked_fill(~valid_mask, -1e4)}

    mean, std = compute_feature_stats(rows)
    weight, metrics = select_best_reranker_blend(
        WrongReranker(),
        rows,
        mean,
        std,
        reranker_weights=(0.0, 1.0),
        batch_size=4,
        device="cpu",
    )

    assert weight == 0.0
    assert metrics["acc025"] == 1.0
    assert metrics["acc050"] == 1.0


def test_blend_selection_never_regresses_either_default_threshold():
    baseline = {
        "acc025": 0.58,
        "acc050": 0.463,
        "score": calibration_score(0.58, 0.463),
    }
    scalar_better_but_acc050_worse = {
        "acc025": 0.60,
        "acc050": 0.455,
        "score": calibration_score(0.60, 0.455),
    }

    weight, metrics = choose_best_reranker_blend({
        0.0: baseline,
        1.0: scalar_better_but_acc050_worse,
    })

    assert scalar_better_but_acc050_worse["score"] > baseline["score"]
    assert weight == 0.0
    assert metrics == baseline


def test_artifact_loader_rejects_missing_required_provenance(tmp_path):
    torch.manual_seed(5)
    model = QueryReranker(input_dim=2, hidden_dim=8, dropout=0.0).eval()
    manifest = _write_cache(
        tmp_path / "cache", _basic_rows(num_scenes=2), shard_count=1
    )
    output = tmp_path / "valid.pth"
    artifact = save_reranker_artifact(
        output,
        model,
        torch.zeros(2),
        torch.ones(2),
        manifest,
        epoch=3,
        calibration_metrics={"acc025": 0.75, "acc050": 0.5},
        training_args={"seed": 5},
    )

    for key in ("epoch", "calibration_metrics", "training_args"):
        malformed = copy.deepcopy(artifact)
        malformed.pop(key)
        malformed_path = tmp_path / "missing-{}.pth".format(key)
        torch.save(malformed, malformed_path)
        with pytest.raises(ValueError, match=key):
            load_reranker_artifact(malformed_path, device="cpu")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("checkpoint_sha256", "", "checkpoint fingerprint"),
        ("feature_names", ["duplicate", "duplicate"], "feature names"),
        ("candidate_rule", {}, "candidate rule"),
    ],
)
def test_artifact_loader_rejects_malformed_runtime_metadata(
        tmp_path, field, value, error):
    torch.manual_seed(5)
    model = QueryReranker(input_dim=2, hidden_dim=8, dropout=0.0).eval()
    manifest = _write_cache(
        tmp_path / "cache", _basic_rows(num_scenes=2), shard_count=1
    )
    output = tmp_path / "valid.pth"
    artifact = save_reranker_artifact(
        output,
        model,
        torch.zeros(2),
        torch.ones(2),
        manifest,
        epoch=3,
        calibration_metrics={"acc025": 0.75, "acc050": 0.5},
        training_args={"seed": 5},
    )
    artifact[field] = value
    malformed_path = tmp_path / "malformed.pth"
    torch.save(artifact, malformed_path)

    with pytest.raises(ValueError, match=error):
        load_reranker_artifact(malformed_path, device="cpu")


def test_synthetic_training_improves_both_calibration_thresholds(tmp_path):
    rows = []
    for scene_index in range(20):
        scene_offset = (scene_index % 5) * 0.02
        rows.append(_row(
            scene_index,
            "scene{:04d}_00".format(scene_index),
            [
                [-1.0, scene_offset],
                [1.0, scene_offset],
                [0.0, scene_offset],
            ],
            [0.10, 0.80, 0.20],
            default_position=0,
        ))
    cache_dir = tmp_path / "train-cache"
    _write_cache(cache_dir, rows, shard_count=3)
    output = tmp_path / "trained-reranker.pth"

    result = train_reranker(
        train_cache=cache_dir,
        output=output,
        seed=13,
        hidden_dim=12,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        batch_size=8,
        max_epochs=30,
        patience=6,
        device="cpu",
    )

    assert output.is_file()
    assert result["calibration_metrics"]["acc025"] > result[
        "calibration_metrics"
    ]["default_acc025"]
    assert result["calibration_metrics"]["acc050"] > result[
        "calibration_metrics"
    ]["default_acc050"]
    assert result["calibration_metrics"]["acc025"] == 1.0
    assert result["calibration_metrics"]["acc050"] == 1.0
    assert result["score_mode"] == "rank_blend"
    assert 0.0 <= result["reranker_weight"] <= 1.0

    restored, artifact = load_reranker_artifact(output, device="cpu")
    loaded_rows, _ = load_training_cache(cache_dir)
    _, calibration_rows = deterministic_scene_split(loaded_rows, seed=13)
    restored_metrics = evaluate_reranker(
        restored,
        calibration_rows,
        artifact["feature_mean"],
        artifact["feature_std"],
        batch_size=8,
        device="cpu",
    )
    assert restored_metrics == artifact["calibration_metrics"]


def test_calibration_score_and_cli_contract(tmp_path):
    assert calibration_score(0.60, 0.47) == pytest.approx(1.107)
    args = parse_args([
        "--train-cache", str(tmp_path / "cache"),
        "--output", str(tmp_path / "reranker.pth"),
        "--seed", "9",
        "--hidden-dim", "16",
        "--dropout", "0.2",
        "--lr", "0.005",
        "--weight-decay", "0.01",
        "--batch-size", "4",
        "--max-epochs", "12",
        "--patience", "3",
        "--device", "cpu",
    ])

    assert args.train_cache == str(tmp_path / "cache")
    assert args.output == str(tmp_path / "reranker.pth")
    assert args.seed == 9
    assert args.hidden_dim == 16
    assert args.dropout == 0.2
    assert args.lr == 0.005
    assert args.weight_decay == 0.01
    assert args.batch_size == 4
    assert args.max_epochs == 12
    assert args.patience == 3
    assert args.device == "cpu"
