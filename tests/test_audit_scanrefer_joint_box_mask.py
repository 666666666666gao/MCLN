import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.audit_scanrefer_joint_box_mask as audit

from models.rec_joint_box_mask import (
    summarize_joint_oracle,
    stage0_gate,
)
from scripts.audit_scanrefer_joint_box_mask import (
    _build_parent_geometry_targets,
    _assert_historical_cache_identity,
    _publish_staging_no_clobber,
    parse_args,
    run_audit,
    semantic_mask_query_indices,
    validate_manifest,
)


def test_joint_audit_uses_cache_only_as_historical_identity_panel(monkeypatch):
    observed = {}

    def fake_parity(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"query_identity_drift_count": 1}

    monkeypatch.setattr(audit, "assert_candidate_cache_parity", fake_parity)
    result = _assert_historical_cache_identity(
        "candidate", "cache", [3], ["scene0003_00"], torch.tensor([4])
    )

    assert result == {"query_identity_drift_count": 1}
    assert observed["kwargs"]["identity_only"] is True


def _selection_result():
    return {
        "baseline_box_iou": torch.tensor([0.20, 0.60, 0.30, 0.70]),
        "selected_box_iou": torch.tensor([0.30, 0.60, 0.30, 0.80]),
        "baseline_mask_iou": torch.tensor([0.20, 0.40, 0.50, 0.60]),
        "selected_mask_iou": torch.tensor([0.30, 0.60, 0.70, 0.80]),
    }


def test_summarize_joint_oracle_reports_exact_counts_and_deltas():
    summary = summarize_joint_oracle(_selection_result())

    assert summary["row_count"] == 4
    assert summary["baseline_position_hits025"] == 3
    assert summary["selected_position_hits025"] == 4
    assert summary["delta_position_acc025"] == pytest.approx(0.25)
    assert summary["delta_position_acc050"] == pytest.approx(0.0)
    assert summary["delta_mask_acc025"] == pytest.approx(0.25)
    assert summary["delta_mask_acc050"] == pytest.approx(0.5)
    assert summary["delta_mask_miou"] == pytest.approx(0.175)


def test_stage0_gate_uses_preregistered_thresholds():
    summary = {
        "delta_position_acc025": 0.0,
        "delta_position_acc050": 0.0,
        "delta_mask_acc050": 0.03,
        "delta_mask_miou": 0.04,
    }

    gate = stage0_gate(summary)

    assert gate["pass"] is True
    assert gate["thresholds"] == {
        "delta_mask_acc050": 0.03,
        "delta_mask_miou": 0.04,
        "delta_position_acc025": 0.0,
        "delta_position_acc050": 0.0,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("delta_position_acc025", -1e-8),
        ("delta_position_acc050", -1e-8),
        ("delta_mask_acc050", 0.029999),
        ("delta_mask_miou", 0.039999),
    ],
)
def test_stage0_gate_fails_when_any_threshold_is_missed(field, value):
    summary = {
        "delta_position_acc025": 0.0,
        "delta_position_acc050": 0.0,
        "delta_mask_acc050": 0.03,
        "delta_mask_miou": 0.04,
    }
    summary[field] = value

    assert stage0_gate(summary)["pass"] is False


def test_joint_audit_defaults_are_the_approved_panel():
    args = parse_args([
        "--checkpoint", "base.pth",
        "--parent-checkpoint", "parent.pth",
        "--geometry-checkpoint", "geometry.pth",
        "--train-cache", "cache",
        "--output-dir", "out",
    ])
    assert args.scene_count == 64
    assert args.expressions_per_scene == 16
    assert args.selection_seed == 0
    assert args.logit_thresholds == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_protected_artifact_contract_is_explicit_and_immutable():
    expected = {
        "checkpoint": {
            "path": "/root/autodl-tmp/DATA_ROOT/output/preserved_best/"
                    "mcln_pair_sweep/"
                    "mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_"
                    "0.57993.pth",
            "sha256": "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208",
            "size": 794125833,
            "mode": 0o444,
        },
        "parent": {
            "path": "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/"
                    "e71_top16/artifacts/"
                    "reranker_h256_d010_lr1e3_seed0_final_contract.pth",
            "sha256": "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b",
            "size": 611713,
            "mode": 0o444,
        },
        "geometry": {
            "path": "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/"
                    "e71_top16/geometry_artifacts/selected_geometry_reranker.pth",
            "sha256": "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f",
            "size": 704449,
            "mode": 0o444,
        },
    }

    assert dict(audit.PROTECTED_ARTIFACT_CONTRACT) == expected


def test_artifact_snapshot_rejects_writable_protected_file_before_inference(
        monkeypatch, tmp_path):
    payload = b"protected"
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(payload)
    path.chmod(0o600)
    contract = {
        "checkpoint": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "mode": 0o444,
        },
    }
    monkeypatch.setattr(audit, "PROTECTED_ARTIFACT_CONTRACT", contract)

    with pytest.raises(ValueError, match="mode|0444"):
        audit._artifact_snapshot(
            path, contract["checkpoint"]["sha256"], "checkpoint"
        )


def test_artifact_snapshot_rejects_byte_identical_noncanonical_path(
        monkeypatch, tmp_path):
    payload = b"protected"
    canonical = tmp_path / "canonical.pth"
    alternate = tmp_path / "alternate.pth"
    canonical.write_bytes(payload)
    alternate.write_bytes(payload)
    canonical.chmod(0o444)
    alternate.chmod(0o444)
    contract = {
        "checkpoint": {
            "path": str(canonical.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "mode": 0o444,
        },
    }
    monkeypatch.setattr(audit, "PROTECTED_ARTIFACT_CONTRACT", contract)

    with pytest.raises(ValueError, match="path|canonical"):
        audit._artifact_snapshot(
            alternate, contract["checkpoint"]["sha256"], "checkpoint"
        )


def test_joint_audit_rejects_validation_manifest():
    with pytest.raises(ValueError, match="train"):
        validate_manifest({"split": "val", "checkpoint_sha256": "a" * 64},
                          "a" * 64)


def _complete_cache_manifest():
    return {
        "split": "train",
        "checkpoint_sha256": "a" * 64,
        "sample_count": 2,
        "dataset_size": 2,
        "source_dataset_size": 2,
        "cache_schema_version": 1,
        "feature_schema_version": "rec-query-v1",
        "target_iou_policy": "root_only",
        "deterministic": True,
        "candidate_rule": {"topk_per_source": 8, "max_candidates": 16},
        "shards": ["shard_000000.pt"],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_schema_version", True),
        ("target_iou_policy", "all_targets"),
        ("deterministic", False),
        ("candidate_rule", {"topk_per_source": 8}),
        ("shards", []),
    ],
)
def test_joint_audit_rejects_cache_schema_drift(field, value):
    manifest = _complete_cache_manifest()
    manifest[field] = value

    with pytest.raises(ValueError, match="cache|candidate|schema|deterministic"):
        validate_manifest(manifest, "a" * 64)


def test_joint_audit_builds_geometry_before_attaching_targets(monkeypatch):
    calls = []
    candidate_batch = {"features": torch.zeros(1, 1, 1)}
    parent = {"candidate_batch": candidate_batch}
    runtime = {"rec_geometry_scores": torch.zeros(1, 1)}

    def build_parent(*_args):
        calls.append("parent")
        return parent

    def build_geometry(*args):
        calls.append("geometry")
        assert args[2] is parent
        assert "candidate_ious" not in args[2]["candidate_batch"]
        return runtime

    def attach_targets(candidates, batch_data, root_only):
        calls.append("targets")
        assert candidates is candidate_batch
        assert batch_data == {"center_label": "target-only"}
        assert root_only is True
        return dict(candidates, candidate_ious=torch.ones(1, 1))

    monkeypatch.setattr(
        "train_dist_mod.build_rec_reranker_outputs", build_parent
    )
    monkeypatch.setattr(
        "train_dist_mod.build_rec_geometry_runtime_outputs", build_geometry
    )
    monkeypatch.setattr(
        "models.rec_candidate_adapter.attach_candidate_targets", attach_targets
    )

    actual = _build_parent_geometry_targets(
        selected_end_points={},
        selected_inputs={},
        selected_batch={"center_label": "target-only"},
        parent_model=object(),
        parent_artifact={},
        geometry_model=object(),
        geometry_artifact={},
    )

    assert calls == ["parent", "geometry", "targets"]
    assert actual == (parent, runtime, {
        "features": candidate_batch["features"],
        "candidate_ious": torch.ones(1, 1),
    })


def test_joint_audit_publish_is_atomic_no_clobber(tmp_path):
    staging = tmp_path / "run.staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}\n")
    final_path = tmp_path / "run"
    final_path.mkdir()

    with pytest.raises(FileExistsError):
        _publish_staging_no_clobber(staging, final_path)

    assert staging.is_dir()
    assert final_path.is_dir()
    assert list(final_path.iterdir()) == []


def test_joint_audit_cleans_interrupted_staging(monkeypatch, tmp_path):
    output_dir = tmp_path / "interrupted"

    def fail_run(_args, staging):
        (Path(staging) / "partial.txt").write_text("partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(
        "scripts.audit_scanrefer_joint_box_mask._run", fail_run
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        run_audit(SimpleNamespace(output_dir=str(output_dir)))

    assert not output_dir.exists()
    assert list(tmp_path.glob("interrupted.staging.*")) == []


def test_semantic_mask_query_selection_matches_evaluator_formula():
    end_points = {
        "proj_tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "last_proj_queries": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    }
    maps = {
        "positive_map": torch.tensor([[[0.0, 1.0]]]),
        "modify_positive_map": torch.zeros(1, 1, 2),
        "pron_positive_map": torch.zeros(1, 1, 2),
        "other_entity_map": torch.zeros(1, 1, 2),
        "rel_positive_map": torch.zeros(1, 1, 2),
    }
    assert semantic_mask_query_indices(end_points, maps).tolist() == [1]


def _write_bytes(path, payload, mode=0o640):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def _make_project_assets(tmp_path, include_cls_results=True):
    project = tmp_path / "project_assets"
    _write_bytes(
        project / "data" / "meta_data" / "scannetv2-labels.combined.tsv",
        b"raw_category\tid\nchair\t1\n",
        0o644,
    )
    _write_bytes(
        project / "data" / "class_embeddings3d.npy", b"class embeddings",
        0o644,
    )
    if include_cls_results:
        _write_bytes(
            project / "data" / "cls_results.json", b"{}\n", 0o644
        )
    return project


def _make_train_data_root(tmp_path, scene_ids, scanrefer_name="scanrefer",
                          extra_scene_ids=()):
    data_root = tmp_path / "data"
    scanrefer = data_root / scanrefer_name
    _write_bytes(
        scanrefer / "ScanRefer_filtered_train.json", b"[]\n", 0o640
    )
    _write_bytes(
        scanrefer / "ScanRefer_filtered_train.txt",
        ("\n".join(scene_ids) + "\n").encode("ascii"),
        0o644,
    )
    _write_bytes(data_root / "train_v3scans.pkl", b"train scans", 0o600)
    tokenizer_root = data_root / "roberta-base"
    for asset_name in (
            "tokenizer.json", "vocab.json", "merges.txt",
            "tokenizer_config.json"):
        _write_bytes(
            tokenizer_root / asset_name,
            (asset_name + " bytes").encode("ascii"),
            0o644,
        )
    all_scene_ids = list(scene_ids) + [
        scene_id for scene_id in extra_scene_ids if scene_id not in scene_ids
    ]
    for index, scene_id in enumerate(all_scene_ids):
        _write_bytes(
            data_root / "superpoints" / "train"
            / (scene_id + "_superpoint.pth"),
            ("superpoint-{}".format(index)).encode("ascii"),
            0o640,
        )
        _write_bytes(
            data_root / "group_free_pred_bboxes"
            / "group_free_pred_bboxes_train" / (scene_id + ".npy"),
            ("boxes-{}".format(index)).encode("ascii"),
            0o644,
        )
    return data_root


def _write_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = _complete_cache_manifest()
    manifest["shards"] = ["shard_000000.pt", "shard_000001.pt"]
    manifest_bytes = json.dumps(
        manifest, indent=1, sort_keys=False
    ).encode("utf-8") + b"\n"
    _write_bytes(cache_dir / "manifest.json", manifest_bytes, 0o640)
    _write_bytes(cache_dir / "shard_000000.pt", b"first shard", 0o600)
    _write_bytes(cache_dir / "shard_000001.pt", b"second shard", 0o640)
    return cache_dir, manifest, manifest_bytes


def test_dataset_bindings_hash_real_train_files_and_detect_mutation(tmp_path):
    scene_ids = ["scene0001_00", "scene0000_00"]
    data_root = _make_train_data_root(tmp_path, scene_ids)

    bindings = audit._build_dataset_bindings(
        data_root, scene_ids, project_root=_make_project_assets(tmp_path)
    )

    assert bindings["scanrefer_directory"] == "scanrefer"
    assert list(bindings["scenes"]) == sorted(scene_ids)
    annotation_path = data_root / "scanrefer" / "ScanRefer_filtered_train.json"
    annotation = bindings["annotation_json"]
    assert annotation["path"] == str(annotation_path.resolve())
    assert annotation["sha256"] == hashlib.sha256(b"[]\n").hexdigest()
    assert annotation["size"] == 3
    assert annotation["mode"] == 0o640
    assert "val" not in annotation_path.name
    audit._verify_dataset_bindings(bindings, sorted(scene_ids))

    annotation_path.write_bytes(b"mutated\n")
    with pytest.raises(RuntimeError, match="changed|binding|SHA-256|size"):
        audit._verify_dataset_bindings(bindings, sorted(scene_ids))


def test_dataset_bindings_follow_runtime_scanrefer_case_fallback(tmp_path):
    scene_ids = ["scene0000_00"]
    data_root = _make_train_data_root(
        tmp_path, scene_ids, scanrefer_name="ScanRefer"
    )

    bindings = audit._build_dataset_bindings(
        data_root, scene_ids, project_root=_make_project_assets(tmp_path)
    )

    assert bindings["scanrefer_directory"] == "ScanRefer"
    assert "/ScanRefer/ScanRefer_filtered_train.json" in (
        bindings["annotation_json"]["path"]
    )


def test_dataset_bindings_reject_extra_top_level_validation_binding(tmp_path):
    scene_ids = ["scene0000_00"]
    data_root = _make_train_data_root(tmp_path, scene_ids)
    bindings = audit._build_dataset_bindings(
        data_root, scene_ids, project_root=_make_project_assets(tmp_path)
    )
    audit._verify_dataset_bindings(bindings, scene_ids)
    validation_path = _write_bytes(
        data_root / "scanrefer" / "ScanRefer_filtered_val.json",
        b"[]\n",
        0o640,
    )
    bindings["validation_annotation_json"] = audit._stable_file_binding(
        validation_path, "ScanRefer validation annotations"
    )

    with pytest.raises(ValueError, match="dataset.*keys|dataset.*bindings"):
        audit._verify_dataset_bindings(bindings, scene_ids)


def test_dataset_bindings_reject_extra_per_scene_validation_binding(tmp_path):
    scene_ids = ["scene0000_00"]
    data_root = _make_train_data_root(tmp_path, scene_ids)
    bindings = audit._build_dataset_bindings(
        data_root, scene_ids, project_root=_make_project_assets(tmp_path)
    )
    audit._verify_dataset_bindings(bindings, scene_ids)
    validation_path = _write_bytes(
        data_root / "superpoints" / "val" / "scene0000_00_superpoint.pth",
        b"validation superpoint",
        0o640,
    )
    bindings["scenes"][scene_ids[0]][
        "validation_superpoint"
    ] = audit._stable_file_binding(
        validation_path, "scene0000_00 validation superpoint"
    )

    with pytest.raises(ValueError, match="scene.*keys|scene.*bindings"):
        audit._verify_dataset_bindings(bindings, scene_ids)


def test_dataset_bindings_cover_replay_neighbor_and_all_train_superpoints(
        tmp_path):
    selected_scene_ids = ["scene0000_00"]
    replay_scene_ids = ["scene0000_00", "scene0001_00"]
    all_train_scene_ids = replay_scene_ids + ["scene0002_00"]
    data_root = _make_train_data_root(
        tmp_path, selected_scene_ids, extra_scene_ids=all_train_scene_ids[1:]
    )
    project_root = _make_project_assets(tmp_path)

    bindings = audit._build_dataset_bindings(
        data_root,
        selected_scene_ids,
        replay_scene_ids=replay_scene_ids,
        project_root=project_root,
    )

    assert sorted(bindings["scenes"]) == replay_scene_ids
    assert sorted(bindings["superpoints_train"]) == all_train_scene_ids
    assert sorted(bindings["tokenizer_assets"]) == [
        "added_tokens.json", "merges.txt", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ]
    assert sorted(bindings["project_assets"]) == [
        "data/class_embeddings3d.npy",
        "data/cls_results.json",
        "data/meta_data/scannetv2-labels.combined.tsv",
    ]
    audit._verify_dataset_bindings(
        bindings, selected_scene_ids, replay_scene_ids
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "roberta-base/tokenizer.json",
        "project_assets/data/class_embeddings3d.npy",
    ],
)
def test_dataset_asset_bindings_detect_live_mutation(tmp_path, relative_path):
    selected_scene_ids = ["scene0000_00"]
    data_root = _make_train_data_root(tmp_path, selected_scene_ids)
    project_root = _make_project_assets(tmp_path)
    bindings = audit._build_dataset_bindings(
        data_root, selected_scene_ids, project_root=project_root
    )
    audit._verify_dataset_bindings(bindings, selected_scene_ids)
    if relative_path.startswith("roberta-base/"):
        path = data_root / relative_path
    else:
        path = tmp_path / relative_path
    path.write_bytes(b"mutated asset\n")

    with pytest.raises(RuntimeError, match="changed|binding|SHA-256|size"):
        audit._verify_dataset_bindings(bindings, selected_scene_ids)


@pytest.mark.parametrize(
    "scene_ids",
    [
        ["scene0000_00", "scene0000_00"],
        ["../scene0000_00"],
        ["scene0000_00/../../val"],
        ["scene1_00"],
        ["scene0000_00\n"],
    ],
)
def test_dataset_bindings_reject_duplicate_or_unsafe_scene_ids(
        tmp_path, scene_ids):
    with pytest.raises(ValueError, match="scene"):
        audit._build_dataset_bindings(tmp_path / "unused", scene_ids)


def test_train_cache_binding_hashes_manifest_bytes_and_every_shard(tmp_path):
    cache_dir, manifest, manifest_bytes = _write_cache(tmp_path)

    binding = audit._build_train_cache_binding(cache_dir, manifest)

    assert binding["manifest"]["sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    assert binding["logical_manifest_sha256"] == audit._manifest_sha256(
        manifest
    )
    assert list(binding["shards"]) == manifest["shards"]
    assert binding["shards"]["shard_000000.pt"]["size"] == len(
        b"first shard"
    )
    audit._verify_train_cache_binding(binding)

    (cache_dir / "shard_000001.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed|binding|SHA-256|size"):
        audit._verify_train_cache_binding(binding)


def test_train_cache_binding_rejects_loaded_manifest_byte_mismatch(tmp_path):
    cache_dir, manifest, _manifest_bytes = _write_cache(tmp_path)
    loaded_manifest = dict(manifest)
    loaded_manifest["sample_count"] = manifest["sample_count"] + 1

    with pytest.raises(ValueError, match="manifest.*bytes|bytes.*manifest"):
        audit._build_train_cache_binding(cache_dir, loaded_manifest)


def test_train_cache_binding_rejects_type_coercing_manifest_mismatch(tmp_path):
    cache_dir, manifest, _manifest_bytes = _write_cache(tmp_path)
    loaded_manifest = dict(manifest)
    loaded_manifest["cache_schema_version"] = True

    with pytest.raises(ValueError, match="manifest.*bytes|bytes.*manifest"):
        audit._build_train_cache_binding(cache_dir, loaded_manifest)


@pytest.mark.parametrize(
    "shards",
    [
        ["../shard_000000.pt"],
        ["shard_000001.pt"],
        ["shard_000000.pt", "shard_000000.pt"],
        [["shard_000000.pt"]],
    ],
)
def test_train_cache_binding_rejects_unsafe_or_noncontiguous_shards(
        tmp_path, shards):
    cache_dir, manifest, _manifest_bytes = _write_cache(tmp_path)
    manifest["shards"] = shards
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="shard"):
        audit._build_train_cache_binding(cache_dir, manifest)


def _make_project_tree(tmp_path):
    project = tmp_path / "project"
    _write_bytes(
        project / "scripts" / "audit_scanrefer_joint_box_mask.py",
        b"AUDIT = True\n",
        0o640,
    )
    _write_bytes(project / "train_dist_mod.py", b"TRAIN = True\n", 0o600)
    _write_bytes(
        project / "models" / "rec_joint_box_mask.py",
        b"MODEL = True\n",
        0o644,
    )
    _write_bytes(project / "package" / "module.py", b"VALUE = 1\n", 0o644)
    _write_bytes(project / "package" / "native.so", b"ELF-bytes", 0o755)
    _write_bytes(
        project / "package" / "__pycache__" / "ignored.py",
        b"IGNORE = True\n",
        0o644,
    )
    return project


def test_source_snapshot_copies_all_python_and_shared_objects(tmp_path):
    project = _make_project_tree(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()

    snapshot, source_bindings = audit._build_source_snapshot(
        staging, root=project
    )

    expected = [
        "models/rec_joint_box_mask.py",
        "package/module.py",
        "package/native.so",
        "scripts/audit_scanrefer_joint_box_mask.py",
        "train_dist_mod.py",
    ]
    assert list(snapshot["files"]) == expected
    assert list(source_bindings) == expected
    assert not (staging / "source_snapshot/package/__pycache__").exists()
    for relative_path in expected:
        copied = staging / "source_snapshot" / relative_path
        assert copied.read_bytes() == (project / relative_path).read_bytes()
        assert snapshot["files"][relative_path]["sha256"] == hashlib.sha256(
            copied.read_bytes()
        ).hexdigest()
    canonical = json.dumps(
        snapshot["files"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert snapshot["aggregate_sha256"] == hashlib.sha256(canonical).hexdigest()
    audit._verify_code_snapshot(staging, snapshot)


def test_source_snapshot_rejects_symlinked_code(tmp_path):
    project = _make_project_tree(tmp_path)
    (project / "package" / "linked.py").symlink_to(
        project / "package" / "module.py"
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="symlink"):
        audit._build_source_snapshot(staging, root=project)


def test_source_snapshot_rejects_symlinked_pycache_directory(tmp_path):
    project = _make_project_tree(tmp_path)
    pycache = project / "package" / "__pycache__"
    (pycache / "ignored.py").unlink()
    pycache.rmdir()
    pycache.symlink_to(project / "package", target_is_directory=True)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="symlink"):
        audit._build_source_snapshot(staging, root=project)


def _write_complete_receipt(tmp_path):
    scene_ids = ["scene0000_00"]
    data_root = _make_train_data_root(tmp_path, scene_ids)
    project_assets = _make_project_assets(tmp_path)
    dataset_inputs = audit._build_dataset_bindings(
        data_root, scene_ids, project_root=project_assets
    )
    cache_dir, cache_manifest, _manifest_bytes = _write_cache(tmp_path)
    train_cache = audit._build_train_cache_binding(cache_dir, cache_manifest)
    project = _make_project_tree(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    code_snapshot, _source_bindings = audit._build_source_snapshot(
        staging, root=project
    )
    selection = {
        "schema": audit.AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "population_estimate": False,
        "rows": [{
            "dataset_index": 0,
            "scan_id": scene_ids[0],
            "target_id": 0,
        }],
        "replay_input_scene_ids": scene_ids,
        "train_cache_manifest_sha256": train_cache[
            "logical_manifest_sha256"
        ],
    }
    summary = {
        "schema": audit.AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "population_estimate": False,
        "elapsed_seconds": 1.25,
        "sample_count": 1,
        "stage0_gate": {"pass": True},
    }
    (staging / "selection.json").write_text(json.dumps(selection) + "\n")
    (staging / "summary.json").write_text(json.dumps(summary) + "\n")
    torch.save({
        "schema": audit.AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "logit_thresholds": (-1.0, 0.0, 1.0),
        "source_names": ("mask_text",),
        "rows": [{
            "dataset_index": 0,
            "scan_id": scene_ids[0],
            "target_id": 0,
            "query_indices": torch.tensor([0], dtype=torch.long),
            "candidate_valid": torch.tensor([True]),
            "geometry_valid": torch.tensor([True]),
            "geometry_ious": torch.tensor([[0.5]], dtype=torch.float32),
            "candidate_mask_ious": torch.tensor(
                [[[0.5]]], dtype=torch.float32
            ),
            "legacy_semantic_query": 0,
            "legacy_mask_iou": 0.5,
            "geometry_parent_mask_iou": 0.5,
            "baseline_flat_index": 0,
            "baseline_box_iou": 0.5,
            "joint_oracle_flat_index": 0,
            "joint_oracle_mask_iou": 0.5,
            "selected_box_iou": 0.5,
            "selected_mask_iou": 0.5,
            "selected_uses_joint_query": False,
        }],
    }, staging / "rows.pt")
    (staging / "stdout.log").write_text("audited\n")
    outputs = audit._build_output_bindings(staging, [
        "selection.json", "rows.pt", "summary.json", "stdout.log",
    ])
    protected = {}
    for name in ("checkpoint", "parent", "geometry"):
        path = _write_bytes(
            tmp_path / "protected" / (name + ".pth"),
            name.encode("ascii"),
            0o600,
        )
        protected[name] = audit._stable_file_binding(path, name)
    source_sha256 = {
        name: code_snapshot["files"][name]["sha256"] for name in (
            "models/rec_joint_box_mask.py",
            "scripts/audit_scanrefer_joint_box_mask.py",
        )
    }
    receipt = {
        "schema": audit.AUDIT_SCHEMA_VERSION,
        "split": "train",
        "validation_data_accessed": False,
        "population_estimate": False,
        "elapsed_seconds": summary["elapsed_seconds"],
        "selected_train_scene_ids": scene_ids,
        "replay_input_scene_ids": scene_ids,
        "dataset_inputs": dataset_inputs,
        "train_cache": train_cache,
        "train_cache_manifest_sha256": train_cache[
            "logical_manifest_sha256"
        ],
        "protected_before": protected,
        "protected_after": protected,
        "outputs": outputs,
        "outputs_sha256": {
            name: binding["sha256"] for name, binding in outputs.items()
        },
        "code_snapshot": code_snapshot,
        "source_sha256": source_sha256,
        "stage0_gate": summary["stage0_gate"],
    }
    (staging / "manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    audit._verify_staged_manifest(staging)
    return staging


@pytest.mark.parametrize(
    "relative_path",
    ["summary.json", "source_snapshot/train_dist_mod.py"],
)
def test_staged_manifest_verifier_detects_output_and_snapshot_tampering(
        tmp_path, relative_path):
    staging = _write_complete_receipt(tmp_path)
    (staging / relative_path).write_bytes(b"tampered\n")

    with pytest.raises(RuntimeError, match="changed|binding|SHA-256|size"):
        audit._verify_staged_manifest(staging)


def test_staged_manifest_rejects_invalid_rows_bytes_after_hash_refresh(tmp_path):
    staging = _write_complete_receipt(tmp_path)
    (staging / "rows.pt").write_bytes(b"not a torch payload")
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outputs = audit._build_output_bindings(staging, [
        "selection.json", "rows.pt", "summary.json", "stdout.log",
    ])
    manifest["outputs"] = outputs
    manifest["outputs_sha256"] = {
        name: binding["sha256"] for name, binding in outputs.items()
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="rows|torch|mapping"):
        audit._verify_staged_manifest(staging)


@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), -0.1, True])
def test_staged_manifest_rejects_invalid_elapsed_seconds(tmp_path, elapsed):
    staging = _write_complete_receipt(tmp_path)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["elapsed_seconds"] = elapsed
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="elapsed_seconds"):
        audit._verify_staged_manifest(staging)


def test_staged_manifest_rejects_non_string_selection_scene_id(tmp_path):
    staging = _write_complete_receipt(tmp_path)
    selection_path = staging / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["rows"].append({"scan_id": None})
    selection_path.write_text(json.dumps(selection) + "\n")
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outputs = audit._build_output_bindings(staging, [
        "selection.json", "rows.pt", "summary.json", "stdout.log",
    ])
    manifest["outputs"] = outputs
    manifest["outputs_sha256"] = {
        name: binding["sha256"] for name, binding in outputs.items()
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="scene"):
        audit._verify_staged_manifest(staging)
