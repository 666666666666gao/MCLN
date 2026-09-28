import pytest

from scripts.summarize_nr3d_pair_readout import scene_cluster_intervals


def test_cluster_interval_keeps_expression_weighting_and_actual_protected_mask_path():
    rows = []
    for scene, count, pair_box, reference_box, pair_mask in [
            ("large", 9, .8, .1, .75), ("small", 1, .1, .8, .25)]:
        for _ in range(count):
            rows.append({"scan_id": scene, "protected_mask_iou": .5, "scores": {
                "pair": {"box_iou": pair_box, "mask_iou": pair_mask},
                "global": {"box_iou": reference_box, "mask_iou": .5},
                # Its REC Query mask is intentionally different from its actual Mask selection.
                "protected": {"box_iou": reference_box, "mask_iou": .1}}})
    result = scene_cluster_intervals(rows)
    assert result["scenes"] == 2 and result["rows"] == 10
    # Expression-weighted net is 8/10; averaging the two scene accuracies gives zero.
    for reference in ["global", "protected"]:
        differences = result["pair_minus_" + reference]
        assert differences["rec025"]["estimate"] == pytest.approx(80)
        assert differences["rec050"]["estimate"] == pytest.approx(80)
        assert differences["mask_mean_iou"]["estimate"] == pytest.approx(20)
        assert differences["rec025"]["percentile_95_interval"] == [-100.0, 100.0]
    assert result["screening_gates_changed"] is False
