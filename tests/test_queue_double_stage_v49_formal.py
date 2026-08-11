from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / \
    "scripts/queue_double_stage_v49_formal_after_smoke.sh"


def test_formal_queue_waits_for_v49_and_uses_four_gpu_two_stage_training():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "V49 smoke panel did not pass" in text
    assert "MODEL_STAGE=two" in text
    assert "SOURCE_ARBITER=moe" in text
    assert "NPROC_PER_NODE=4" in text
    assert "JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING=1" in text
    assert "JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT" in text
    assert "JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE=0.25" in text


def test_formal_queue_requires_complete_subgroup_receipt():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--require-position-subgroups" in text
    assert "selected_v49_formal_config.json" in text
    assert "mcln-v49-formal-selection-v1" in text


def test_formal_queue_audits_the_trained_v49_module_and_optimizer_steps():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "formal_steps_per_epoch" in text
    assert "formal_optimizer_step" in text
    assert "formal steps-per-epoch changed" in text
    assert "scripts/audit_source_moe_checkpoint.py" in text
    assert "--profile v49" in text
    assert "formal_v49_checkpoint_audit.json" in text
    assert "--expected-source-mix-loss-weight" in text
    assert "--expected-source-mix-alignment-temperature 0.25" in text
    assert "int(lovasz_weight) / 100.0" in text
    assert "int(lovasz_weight) / 1000.0" not in text
