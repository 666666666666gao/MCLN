from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "queue_single_stage_joint_query_after_postprocess.sh"
LAUNCHER = ROOT / "scripts" / "train_scanrefer_single_stage_joint_query_quality.sh"


def _queue_text():
    return QUEUE.read_text(encoding="utf-8")


def test_queue_waits_for_postprocess_without_log_polling():
    text = _queue_text()
    assert 'tail --pid="${postprocess_pid}" -f /dev/null' in text
    assert '"${POST_ROOT}/parent_reranker.pth"' in text
    assert '"${POST_ROOT}/geometry_reranker.pth"' in text
    assert '"${POST_ROOT}/official_eval.log"' in text
    assert "best_checkpoint.sha256" in text


def test_queue_runs_four_selector_smokes_before_formal_training():
    text = _queue_text()
    assert 'RUN_FORMAL_AFTER_SMOKE="${RUN_FORMAL_AFTER_SMOKE:-0}"' in text
    assert 'for gpu in 0 1 2 3; do' in text
    assert "train_scanrefer_single_stage_joint_query_quality.sh" in text
    assert "--profile v43_selector" in text
    assert "--require-candidate-mask" in text
    assert "--require-lovasz-variant" in text
    smoke_audit = text.index("scripts/audit_training_completion.py")
    formal_launch = text.index(
        "starting four-GPU formal single-stage joint-query run"
    )
    assert "--require-position-subgroups" in text[smoke_audit:formal_launch]
    assert text.index("job_pids=()") < text.index(
        "starting four-GPU formal single-stage joint-query run"
    )


def test_formal_queue_uses_all_gpus_and_full_completion_audits():
    text = _queue_text()
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in text
    assert 'FORMAL_MAX_EPOCH="${FORMAL_MAX_EPOCH:-80}"' in text
    assert "audit_training_completion.py" in text
    assert "--require-position-subgroups" in text
    assert "audit_source_moe_checkpoint.py" in text
    assert 'expected_sha="$(awk' in text


def test_single_stage_launcher_requires_checkpoint_and_forbids_gate_evidence():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "CHECKPOINT_PATH=<single-stage checkpoint> is required" in text
    assert "export MODEL_STAGE=single" in text
    assert "export SOURCE_ARBITER=selector" in text
    assert "export JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE=0" in text
