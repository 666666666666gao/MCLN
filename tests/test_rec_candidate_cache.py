import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.cache_scanrefer_rec_candidates import (
    append_cache_shard,
    build_cache_rows,
    cache_resume_state,
    checkpoint_sha256,
    _cache_metadata,
    _prepare_model_config,
    compute_batch_metric_counts,
    initialize_cache,
    load_manifest,
    oracle_gate_exit_code,
    parse_args,
    strip_module_prefix,
)


def _metadata(fingerprint="checkpoint-a"):
    return {
        "cache_schema_version": 1,
        "feature_schema_version": "rec-query-v1",
        "checkpoint_sha256": fingerprint,
        "split": "train",
        "candidate_rule": {
            "topk_per_source": 8,
            "max_candidates": 16,
        },
        "feature_dim": 12,
        "feature_names": ["f{}".format(idx) for idx in range(12)],
        "dataset_size": 3,
    }


def _row(index):
    return {
        "dataset_index": index,
        "scan_id": "scene{:04d}_00".format(index),
        "features": torch.full((2, 3), float(index)),
        "valid_mask": torch.tensor([True, False]),
        "candidate_ious": torch.tensor([0.6, 0.0]),
    }


def test_strip_module_prefix_removes_exactly_one_prefix():
    state = {
        "module.layer.weight": torch.tensor([1.0]),
        "module.module.layer.bias": torch.tensor([2.0]),
        "plain": torch.tensor([3.0]),
    }

    stripped = strip_module_prefix(state)

    assert set(stripped) == {"layer.weight", "module.layer.bias", "plain"}
    assert stripped["layer.weight"].item() == 1.0


def test_strip_module_prefix_rejects_key_collision():
    state = {
        "module.value": torch.tensor([1.0]),
        "value": torch.tensor([2.0]),
    }

    with pytest.raises(ValueError, match="collision"):
        strip_module_prefix(state)


def test_checkpoint_sha256_reads_file_contents(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"known checkpoint bytes")

    digest = checkpoint_sha256(checkpoint)

    assert digest == hashlib.sha256(b"known checkpoint bytes").hexdigest()


def test_cache_manifest_and_shards_round_trip_atomically(tmp_path):
    output_dir = tmp_path / "cache"
    manifest = initialize_cache(output_dir, _metadata())

    manifest = append_cache_shard(output_dir, manifest, [_row(0), _row(1)])
    loaded = load_manifest(output_dir)

    assert loaded == manifest
    assert loaded["sample_count"] == 2
    assert loaded["shards"] == ["shard_000000.pt"]
    assert (output_dir / "shard_000000.pt").is_file()
    assert not list(output_dir.glob("*.tmp"))
    payload = torch.load(output_dir / loaded["shards"][0])
    assert [row["dataset_index"] for row in payload["rows"]] == [0, 1]


def test_cache_resume_uses_last_manifested_shard(tmp_path):
    output_dir = tmp_path / "cache"
    manifest = initialize_cache(output_dir, _metadata())
    manifest = append_cache_shard(output_dir, manifest, [_row(0), _row(1)])
    manifest = append_cache_shard(output_dir, manifest, [_row(2)])

    resumed = initialize_cache(output_dir, _metadata())
    sample_count, next_shard = cache_resume_state(output_dir, resumed)

    assert sample_count == 3
    assert next_shard == 2


def test_cache_resume_rejects_missing_manifested_shard(tmp_path):
    output_dir = tmp_path / "cache"
    manifest = initialize_cache(output_dir, _metadata())
    manifest = append_cache_shard(output_dir, manifest, [_row(0)])
    (output_dir / manifest["shards"][0]).unlink()

    with pytest.raises(ValueError, match="missing shard"):
        cache_resume_state(output_dir, manifest)


def test_initialize_cache_refuses_mixed_checkpoint_fingerprints(tmp_path):
    output_dir = tmp_path / "cache"
    manifest = initialize_cache(output_dir, _metadata("first"))
    append_cache_shard(output_dir, manifest, [_row(0)])

    with pytest.raises(ValueError, match="metadata"):
        initialize_cache(output_dir, _metadata("second"))


def test_initialize_cache_overwrite_resets_known_cache_files(tmp_path):
    output_dir = tmp_path / "cache"
    manifest = initialize_cache(output_dir, _metadata("first"))
    append_cache_shard(output_dir, manifest, [_row(0)])
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("keep")
    unrelated_tmp = output_dir / "notes.tmp"
    unrelated_tmp.write_text("keep this too")

    reset = initialize_cache(output_dir, _metadata("second"), overwrite=True)

    assert reset["checkpoint_sha256"] == "second"
    assert reset["sample_count"] == 0
    assert reset["shards"] == []
    assert unrelated.read_text() == "keep"
    assert unrelated_tmp.read_text() == "keep this too"
    assert not list(output_dir.glob("shard_*.pt"))


def test_compute_batch_metric_counts_uses_strict_thresholds():
    candidate_ious = torch.tensor([
        [0.25, 0.50, 0.80],
        [0.30, 0.40, 0.10],
    ])
    valid_mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    query_indices = torch.tensor([[4, 2, 1], [3, 0, 2]])
    default_top1 = torch.tensor([4, 3])

    counts = compute_batch_metric_counts(
        candidate_ious, valid_mask, query_indices, default_top1
    )

    assert counts == {
        "sample_count": 2,
        "default_hits025": 1,
        "default_hits050": 0,
        "oracle_hits025": 2,
        "oracle_hits050": 0,
    }


def test_build_cache_rows_slices_candidate_tensors_and_identity_fields():
    candidate_batch = {
        "features": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "boxes": torch.arange(36, dtype=torch.float32).reshape(2, 3, 6),
        "query_indices": torch.tensor([[4, 2, 1], [3, 0, 2]]),
        "valid_mask": torch.tensor([
            [True, True, False], [True, True, True]
        ]),
        "default_scores": torch.tensor([[0.9, 0.7, 0.1], [0.8, 0.4, 0.3]]),
        "contrastive_scores": torch.tensor([
            [0.2, 0.8, 0.1], [0.5, 0.4, 0.9]
        ]),
        "candidate_ious": torch.tensor([
            [0.6, 0.2, 0.0], [0.1, 0.7, 0.4]
        ]),
        "default_top1_query_index": torch.tensor([4, 3]),
    }
    batch_data = {
        "scan_ids": ["scene0001_00", "scene0002_00"],
        "target_id": torch.tensor([7, 11]),
    }

    rows = build_cache_rows([20, 21], batch_data, candidate_batch)

    assert len(rows) == 2
    assert rows[0]["dataset_index"] == 20
    assert rows[0]["scan_id"] == "scene0001_00"
    assert rows[0]["target_id"] == 7
    assert rows[0]["default_top1_query_index"] == 4
    assert torch.equal(rows[1]["features"], candidate_batch["features"][1])
    assert torch.equal(
        rows[1]["candidate_ious"], candidate_batch["candidate_ious"][1]
    )
    assert all(
        not value.is_cuda
        for row in rows for value in row.values()
        if isinstance(value, torch.Tensor)
    )


def test_cache_cli_contract_and_oracle_gate_exit_code(tmp_path):
    args = parse_args([
        "--split", "val",
        "--data-root", str(tmp_path / "data"),
        "--checkpoint", str(tmp_path / "model.pth"),
        "--output-dir", str(tmp_path / "cache"),
        "--batch-size", "2",
        "--num-workers", "0",
        "--shard-size", "8",
        "--max-candidates", "12",
        "--limit", "10",
        "--device", "cpu",
        "--overwrite",
        "--require-oracle", "0.62", "0.50",
    ])

    assert args.split == "val"
    assert args.max_candidates == 12
    assert args.require_oracle == [0.62, 0.50]
    assert args.overwrite
    assert oracle_gate_exit_code(
        {"oracle_acc025": 0.62, "oracle_acc050": 0.50},
        args.require_oracle,
    ) == 0
    assert oracle_gate_exit_code(
        {"oracle_acc025": 0.61999, "oracle_acc050": 0.80},
        args.require_oracle,
    ) == 2


def test_cache_metadata_records_root_only_rec_target_policy():
    args = SimpleNamespace(split="val", max_candidates=16)
    config = SimpleNamespace(
        data_root="/tmp/scanrefer-data/",
        model="MCLN",
        num_target=256,
        num_decoder_layers=6,
        self_position_embedding="loc_learned",
        self_attend=True,
        use_soft_token_loss=True,
        use_contrastive_align=True,
        detect_intermediate=True,
        use_source_choice_selector=True,
        source_choice_selector_sources="default,default_rank_blend_contrastive010",
        source_choice_selector_hidden_dim=288,
        use_color=True,
        use_height=False,
        use_multiview=False,
        butd=True,
        butd_gt=False,
        butd_cls=False,
    )

    metadata = _cache_metadata(
        args,
        "fingerprint",
        {"epoch": 71},
        config,
        dataset_size=10,
        source_dataset_size=10,
        feature_dim=12,
        feature_names=["f{}".format(index) for index in range(12)],
    )

    assert metadata["target_iou_policy"] == "root_only"
    assert metadata["data_root"] == "/tmp/scanrefer-data"
    assert metadata["backbone_config"] == {
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
    }


def test_moe_checkpoint_config_is_rebuilt_and_recorded_in_cache_metadata():
    raw_config = SimpleNamespace(
        model="MCLN",
        use_source_moe=True,
        source_choice_selector_sources="default,contrastive_text,mask_text",
        source_moe_shared_source="default",
        source_moe_top_k=1,
        source_moe_balance_loss_weight=0.02,
        source_moe_query_layers=2,
        source_moe_query_heads=8,
        source_moe_query_dropout=0.05,
        source_moe_query_max_delta=0.4,
        source_moe_use_fallback_gate=True,
        source_moe_gate_hidden_dim=96,
        source_moe_gate_candidate_top_k=6,
        source_moe_gate_break_cost=2.5,
        source_moe_gate_decision_margin=0.1,
        source_moe_gate_mask_utility_weight=0.2,
        source_moe_gate_uncertainty_weight=0.5,
        source_moe_gate_use_evidence_features=True,
    )
    checkpoint = {"epoch": 72, "config": raw_config}
    config = _prepare_model_config(checkpoint, "/tmp/scanrefer-data/")

    metadata = _cache_metadata(
        SimpleNamespace(split="val", max_candidates=16),
        "fingerprint",
        checkpoint,
        config,
        dataset_size=10,
        source_dataset_size=10,
        feature_dim=12,
        feature_names=["f{}".format(index) for index in range(12)],
    )

    backbone = metadata["backbone_config"]
    assert backbone["use_source_moe"] is True
    assert backbone["source_moe_shared_source"] == "default"
    assert backbone["source_moe_top_k"] == 1
    assert backbone["source_moe_balance_loss_weight"] == pytest.approx(0.02)
    assert backbone["source_moe_query_layers"] == 2
    assert backbone["source_moe_query_heads"] == 8
    assert backbone["source_moe_query_dropout"] == pytest.approx(0.05)
    assert backbone["source_moe_query_max_delta"] == pytest.approx(0.4)
    assert backbone["source_moe_use_fallback_gate"] is True
    assert backbone["source_moe_gate_hidden_dim"] == 96
    assert backbone["source_moe_gate_candidate_top_k"] == 6
    assert backbone["source_moe_gate_break_cost"] == pytest.approx(2.5)
    assert backbone["source_moe_gate_decision_margin"] == pytest.approx(0.1)
    assert backbone["source_moe_gate_mask_utility_weight"] == pytest.approx(0.2)
    assert backbone["source_moe_gate_uncertainty_weight"] == pytest.approx(0.5)
    assert backbone["source_moe_gate_use_evidence_features"] is True
