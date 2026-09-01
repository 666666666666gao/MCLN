from types import SimpleNamespace

import pytest
import torch

from models.density_aware_target_box_audit import (
    DensityAwareTargetBoxAuditAccumulator,
    scene_sample_identity_digest,
)


def _audit_end_points(detector_overlap=True):
    centers = torch.tensor([
        [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ])
    sizes = torch.ones(2, 2, 3)
    target_centers = torch.zeros(2, 1, 3)
    target_sizes = torch.ones(2, 1, 3)
    detector_centers = centers.clone()
    if not detector_overlap:
        detector_centers.fill_(20.0)
    detected_boxes = torch.cat(
        [detector_centers, torch.ones(2, 2, 3)], dim=-1
    )
    point_labels = torch.full((2, 300), -1, dtype=torch.long)
    point_labels[0, :100] = 0
    point_labels[1, :] = 0
    return {
        "last_center": centers,
        "last_pred_size": sizes,
        "selected_source_scores": torch.tensor([
            [10.0, 5.0],
            [10.0, 5.0],
        ]),
        "density_scene_audit_sample_index": torch.tensor([10, 11]),
        "point_instance_label": point_labels,
        "sample_dataset": ["nr3d", "nr3d"],
        "center_label": target_centers,
        "size_gts": target_sizes,
        "box_label_mask": torch.ones(2, 1, dtype=torch.bool),
        "all_detected_boxes": detected_boxes,
        "all_detected_bbox_label_mask": torch.ones(
            2, 2, dtype=torch.bool
        ),
        "density_scene_audit_last_match_indices": [
            (torch.tensor([1]), torch.tensor([0])),
            (torch.tensor([0]), torch.tensor([0])),
        ],
    }


def test_scene_audit_reports_deployment_filtered_selected_top16_and_match():
    accumulator = DensityAwareTargetBoxAuditAccumulator()
    accumulator.update(_audit_end_points())
    metrics = accumulator.finalize(
        expected_sample_count=2,
        expected_identity_sha256=scene_sample_identity_digest([10, 11]),
    )

    assert metrics["sample_count"] == 2
    assert metrics["sample_identity_unique_count"] == 2
    sparse = metrics["slices"]["active_sparse"]
    assert sparse["sample_count"] == 1
    assert sparse["selected_hits025"] == 0
    assert sparse["top16_hits025"] == 1
    assert sparse["matched_hits025"] == 1
    assert sparse["matched_iou_mean"] == pytest.approx(1.0)
    assert sparse["matched_center_l1_mean"] == pytest.approx(0.0)
    assert sparse["matched_size_l1_mean"] == pytest.approx(0.0)

    dense = metrics["slices"]["dense"]
    assert dense["sample_count"] == 1
    assert dense["selected_hits025"] == 1
    assert dense["top16_hits050"] == 1
    assert dense["matched_hits050"] == 1
    assert metrics["slices"]["zero_point"]["sample_count"] == 0


def test_scene_audit_counts_detector_empty_row_as_ranking_miss():
    accumulator = DensityAwareTargetBoxAuditAccumulator()
    accumulator.update(_audit_end_points(detector_overlap=False))
    metrics = accumulator.finalize(
        expected_sample_count=2,
        expected_identity_sha256=scene_sample_identity_digest([10, 11]),
    )
    overall = metrics["slices"]["overall"]
    assert overall["selected_hits025"] == 0
    assert overall["top16_hits025"] == 0
    assert overall["matched_hits025"] == 2


def test_scene_audit_rejects_duplicate_or_drifted_row_identity():
    accumulator = DensityAwareTargetBoxAuditAccumulator()
    end_points = _audit_end_points()
    end_points["density_scene_audit_sample_index"] = torch.tensor([10, 10])
    accumulator.update(end_points)
    with pytest.raises(ValueError, match="unique"):
        accumulator.finalize(
            expected_sample_count=2,
            expected_identity_sha256=scene_sample_identity_digest([10, 11]),
        )


def test_scene_audit_control_checkpoint_uses_protected_e57_gate(
        tmp_path, monkeypatch):
    import hashlib
    import main_utils

    checkpoint_path = tmp_path / "protected_e57.pth"
    torch.save({"epoch": 57, "model": {"weight": torch.ones(1)}},
               checkpoint_path)
    expected = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        main_utils, "DENSITY_TARGET_BOX_SCENE_AUDIT_E57_SHA256", expected
    )
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        density_aware_target_box_loss_weight=0.0,
        density_aware_target_box_scene_disjoint_audit=True,
        density_aware_target_box_checkpoint_sha256=expected,
        fpr_scene_disjoint_audit=False,
        restore_e57_lr_to_initial=False,
    )
    payload = main_utils._load_checkpoint_payload(args)
    assert payload["epoch"] == 57
    assert args.density_aware_target_box_consumed_checkpoint_sha256 == expected
    assert args.density_aware_target_box_consumed_checkpoint_epoch == 57


def test_scene_audit_lifecycle_is_explicitly_audit_only():
    from main_utils import BaseTrainTester

    source = __import__("inspect").getsource(BaseTrainTester.main)
    assert "density_scene_audit" in source
    assert "_save_density_scene_audit_role_receipt" in source
    assert source.index("if density_scene_audit:") < source.index(
        "if max_train_batches > 0:"
    )
