from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / \
    "scripts/queue_double_stage_v50_sacr_after_v49.sh"


def test_v50_queue_waits_for_formal_v49_without_log_polling():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "queue_double_stage_v49_formal_after_smoke.sh" in text
    assert 'tail --pid="${PREDECESSOR_PID}" -f /dev/null' in text
    assert "formal_completion_audit.json" in text
    assert "formal_v49_checkpoint_audit.json" in text


def test_v50_queue_keeps_sacr_out_of_frozen_parent_source_pool():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"parent_source_names": [' in text
    assert '"joint_source_names": [' in text
    assert "JOINT_QUERY_QUALITY_SOURCE_NAMES=default,contrastive_text,mask_text,sacr_structured" in text
    assert "USE_SACR_SOURCE=1" in text
    assert "--profile v50_sacr" in text


def test_v50_queue_gates_four_gpu_smoke_before_formal_training():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("CUDA_VISIBLE_DEVICES=0,1,2,3") == 2
    assert text.count("NPROC_PER_NODE=4") == 2
    assert "v50_sacr_smoke_summary.json" in text
    assert "mcln-v50-sacr-selection-v1" in text
    assert "max(selected_mix, 0.25)" in text
    assert '"source_mix_query_focus_weight": 0.75' in text
    assert text.count(
        "JOINT_QUERY_QUALITY_SOURCE_MIX_QUERY_FOCUS_WEIGHT="
    ) == 2
    assert text.count(
        "JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY=0"
    ) == 2
    assert text.count(
        "--expected-source-mix-query-focus-weight"
    ) == 2
    assert "audit_sacr_structured_data.py" in text
    assert "for structured_split in train val" in text
    assert "sacr_structured_data_${structured_split}_audit.json" in text
    assert "audit_joint_query_initialization.py" in text
    assert "--require-position-subgroups" in text
    assert "-name '*.pth'" in text
