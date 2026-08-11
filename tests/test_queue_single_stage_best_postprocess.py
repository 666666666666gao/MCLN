from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "queue_single_stage_best_postprocess.sh"


def _text():
    return SCRIPT.read_text(encoding="utf-8")


def test_queue_waits_by_pid_and_requires_full_training_receipt():
    text = _text()
    assert 'tail --pid="${train_pid}" -f /dev/null' in text
    assert "audit_training_completion.py" in text
    assert 'eval_metrics_epoch_${FINAL_EPOCH}.json' in text
    assert '--expected-sample-count "${EXPECTED_SAMPLE_COUNT}"' in text


def test_pipeline_rebuilds_single_stage_parent_and_geometry_artifacts():
    text = _text()
    assert "cache_scanrefer_rec_candidates.py" in text
    assert "train_rec_reranker.py" in text
    assert "audit_scanrefer_mask_geometry.py" in text
    assert "cache_scanrefer_rec_mask_geometry.py" in text
    assert "train_rec_geometry_reranker.py" in text
    assert "--portable-provenance" in text
    assert "--audit-train-cache" in text
    parallel = text.split(
        "training parent reranker and mask audit in parallel", 1
    )[1].split("extracting train/val geometry caches", 1)[0]
    assert "CUDA_VISIBLE_DEVICES=2" in parallel
    assert "CUDA_VISIBLE_DEVICES=3" in parallel
    assert parallel.count("independent_pids+=(\"$!\")") == 2
    assert 'for pid in "${independent_pids[@]}"' in parallel


def test_official_eval_is_single_stage_and_provenance_bound():
    text = _text()
    evaluation = text.split(
        "running contract-bound official single-stage", 1
    )[1]
    assert "--butd " not in evaluation
    assert "--butd_gt" not in evaluation
    assert "--butd_cls" not in evaluation
    assert "--rec_reranker_checkpoint" in evaluation
    assert "--rec_geometry_reranker_checkpoint" in evaluation
    assert "--eval_use_rec_geometry_reranker_scores" in evaluation
    assert "--expected_eval_sample_count" in evaluation
    assert "CUDA_VISIBLE_DEVICES=0 " in evaluation
    assert "--nproc_per_node 1" in evaluation
    assert "--batch_size 12" in evaluation


def test_joint_branch_trains_and_deploys_real_mask_policy_after_baseline():
    text = _text()
    joint = text.split(
        "extracting complete train-only joint Mask policy cache", 1
    )[1]
    assert "cache_scanrefer_joint_box_mask.py" in joint
    assert "train_scanrefer_joint_box_mask.py" in joint
    assert '--joint-cache "${JOINT_TRAIN}"' in joint
    assert '--rec_joint_box_mask_checkpoint "${JOINT_ARTIFACT}"' in joint
    assert "--eval_use_rec_joint_box_mask" in joint
    assert "joint_status == 2" in joint
    assert 'receipt.get("selection") != "baseline"' in joint
    assert "joint_official_eval_subgroup_audit.json" in joint
    assert "require_position_subgroups=True" in joint
