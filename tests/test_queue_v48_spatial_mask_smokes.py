from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "queue_v48_spatial_mask_smokes_after_v47.sh"


def test_v48_queue_waits_for_v47_without_polling_its_logs():
    text = QUEUE.read_text(encoding="utf-8")
    assert "queue_single_stage_joint_query_after_postprocess.sh" in text
    assert 'tail --pid="${predecessor_pid}" -f /dev/null' in text
    assert "no log polling" in text
    assert "single_v47_base_panel_summary.json" in text
    assert "single_v47_candidate_panel_summary.json" in text
    assert 'summary.get("pass") is not True' in text


def test_v48_queue_runs_four_spatial_smokes_and_no_unvalidated_formal_run():
    text = QUEUE.read_text(encoding="utf-8")
    assert "for gpu in 0 1 2 3" in text
    assert "JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER=1" in text
    assert "JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM=32" in text
    assert "JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA=2.0" in text
    assert "MODEL_STAGE=two" in text
    assert "SOURCE_ARBITER=moe" in text
    assert "FORMAL_MAX_EPOCH" not in text


def test_v48_queue_is_fail_closed_and_cleans_only_debug_weights():
    text = QUEUE.read_text(encoding="utf-8")
    assert "PROTECTED_V19_SHA256" in text
    assert "--profile v48" in text
    assert "--require-position-subgroups" in text
    assert "--require-candidate-mask" in text
    assert text.count("--require-lovasz-variant") == 2
    assert "v48_spatial_mask_smoke_summary.json" in text
    assert "-mindepth 1 -maxdepth 1 -type f -name '*.pth'" in text
