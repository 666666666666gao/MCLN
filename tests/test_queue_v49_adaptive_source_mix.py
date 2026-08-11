from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / (
    "queue_v49_adaptive_source_mix_smokes_after_v48.sh"
)


def test_v49_queue_is_four_gpu_chained_and_fail_closed():
    text = QUEUE.read_text(encoding="utf-8")
    assert "queue_v48_spatial_mask_smokes_after_v47.sh" in text
    assert "tail --pid=\"${predecessor_pid}\" -f /dev/null" in text
    assert "V48_SUMMARY" in text
    assert 'summary.get("pass") is not True' in text
    assert "for gpu in 0 1 2 3" in text
    assert "JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING=1" in text
    assert "selected_v48_mask_config.json" in text
    assert "mcln-v49-mask-selection-v1" in text
    assert 'source_mix_loss_codes=("000" "010" "025" "050")' in text
    assert "JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT" in text
    assert "JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE=0.25" in text
    assert "--expected-source-mix-loss-weight" in text
    assert "--expected-source-mix-alignment-temperature 0.25" in text
    assert "int(lovasz_code) / 100.0" in text
    assert "int(lovasz_code) / 1000.0" not in text
    assert "--profile v49" in text
    assert "--require-position-subgroups" in text
    assert "v49_adaptive_source_mix_smoke_summary.json" in text
    assert "-name '*.pth'" in text
