import importlib.util
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "density_aware_target_box.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "density_aware_target_box_under_test", str(_MODULE_PATH)
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_density_aware_target_box_loss = (
    _MODULE.compute_density_aware_target_box_loss
)


def _target(box):
    return {"boxes": torch.tensor([box], dtype=torch.float32)}


def test_sparse_target_uses_final_hungarian_target_zero_match_only():
    pred_boxes = torch.zeros(1, 3, 6, requires_grad=True)
    with torch.no_grad():
        pred_boxes[0, 1] = torch.tensor([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    point_labels = torch.cat([
        torch.zeros(128, dtype=torch.long),
        torch.full((128,), -1, dtype=torch.long),
    ]).unsqueeze(0)
    result = compute_density_aware_target_box_loss(
        pred_boxes=pred_boxes,
        targets=[_target([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])],
        match_indices=[(
            torch.tensor([2, 1]),
            torch.tensor([1, 0]),
        )],
        point_instance_label=point_labels,
        sample_datasets=["nr3d"],
    )

    assert result["density_aware_target_box_loss"].item() == pytest.approx(
        7.8, abs=1e-6
    )
    assert result[
        "density_aware_target_box_active_row_ratio"
    ].item() == pytest.approx(1.0)
    assert result[
        "density_aware_target_box_target_point_count_mean"
    ].item() == pytest.approx(128.0)
    assert result[
        "density_aware_target_box_sparsity_weight_mean"
    ].item() == pytest.approx(0.5)

    result["density_aware_target_box_loss"].backward()
    assert torch.equal(pred_boxes.grad[0, 0], torch.zeros(6))
    assert torch.equal(pred_boxes.grad[0, 2], torch.zeros(6))
    assert torch.allclose(
        pred_boxes.grad[0, 1],
        torch.tensor([1.0, 1.0, 1.0, 0.2, 0.2, 0.2]),
    )


def test_density_weight_normalizes_active_rows_without_gt_gradient():
    pred_boxes = torch.zeros(2, 1, 6, requires_grad=True)
    with torch.no_grad():
        pred_boxes[0, 0, 0] = 2.0
        pred_boxes[1, 0, 0] = 4.0
    labels = torch.full((2, 256), -1, dtype=torch.long)
    labels[0, :64] = 0
    labels[1, :192] = 0
    target_boxes = [
        torch.zeros(1, 6, requires_grad=True),
        torch.zeros(1, 6, requires_grad=True),
    ]
    result = compute_density_aware_target_box_loss(
        pred_boxes=pred_boxes,
        targets=[{"boxes": value} for value in target_boxes],
        match_indices=[
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([0]), torch.tensor([0])),
        ],
        point_instance_label=labels,
        sample_datasets="sr3d",
    )

    # Weights are 0.75 and 0.25: (2*.75 + 4*.25) / 1 == 2.5.
    assert result["density_aware_target_box_loss"].item() == pytest.approx(2.5)
    result["density_aware_target_box_loss"].backward()
    assert pred_boxes.grad[0, 0, 0].item() == pytest.approx(0.75)
    assert pred_boxes.grad[1, 0, 0].item() == pytest.approx(0.25)
    assert target_boxes[0].grad is None
    assert target_boxes[1].grad is None


def test_dense_and_scannet_rows_return_backward_safe_zero():
    pred_boxes = torch.randn(2, 2, 6, requires_grad=True)
    labels = torch.full((2, 256), -1, dtype=torch.long)
    labels[0, :] = 0
    labels[1, :1] = 0
    result = compute_density_aware_target_box_loss(
        pred_boxes=pred_boxes,
        targets=[
            _target([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            _target([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ],
        match_indices=[
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([0]), torch.tensor([0])),
        ],
        point_instance_label=labels,
        sample_datasets=["nr3d", "scannet"],
    )

    assert result["density_aware_target_box_loss"].item() == pytest.approx(0.0)
    assert result[
        "density_aware_target_box_active_row_ratio"
    ].item() == pytest.approx(0.0)
    assert result[
        "density_aware_target_box_referring_row_count"
    ].item() == pytest.approx(1.0)
    result["density_aware_target_box_loss"].backward()
    assert torch.equal(pred_boxes.grad, torch.zeros_like(pred_boxes.grad))


@pytest.mark.parametrize("target_point_count", [0, 16, 256])
def test_any_referring_row_without_exact_target_zero_match_fails_closed(
        target_point_count):
    pred_boxes = torch.zeros(1, 2, 6, requires_grad=True)
    labels = torch.full((1, 256), -1, dtype=torch.long)
    labels[0, :target_point_count] = 0
    with pytest.raises(ValueError, match="target 0 exactly once"):
        compute_density_aware_target_box_loss(
            pred_boxes=pred_boxes,
            targets=[{
                "boxes": torch.zeros(2, 6, dtype=torch.float32)
            }],
            match_indices=[(
                torch.tensor([0]),
                torch.tensor([1]),
            )],
            point_instance_label=labels,
            sample_datasets=["scanrefer"],
        )


def test_threshold_and_size_coefficient_are_not_exposed_parameters():
    parameters = inspect.signature(
        compute_density_aware_target_box_loss
    ).parameters
    assert "point_threshold" not in parameters
    assert "size_loss_weight" not in parameters
    assert _MODULE.TARGET_POINT_THRESHOLD == 256
    assert _MODULE.TARGET_SIZE_LOSS_WEIGHT == pytest.approx(0.2)


def test_density_audit_checkpoint_load_is_bound_to_same_fd_sha(tmp_path):
    from main_utils import _load_checkpoint_payload

    checkpoint_path = tmp_path / "protected_e57.pth"
    torch.save({"epoch": 57, "model": {"weight": torch.ones(1)}},
               checkpoint_path)
    expected = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        density_aware_target_box_loss_weight=1.0,
        density_aware_target_box_checkpoint_sha256=expected,
        fpr_scene_disjoint_audit=False,
        restore_e57_lr_to_initial=False,
    )
    checkpoint = _load_checkpoint_payload(args)
    assert checkpoint["epoch"] == 57
    assert args.density_aware_target_box_consumed_checkpoint_sha256 == expected
    assert args.density_aware_target_box_consumed_checkpoint_epoch == 57

    args.density_aware_target_box_checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_checkpoint_payload(args)


def test_pointnet_checkpoint_load_is_bound_to_same_fd_sha(tmp_path):
    from models.mcln import _load_pointnet_checkpoint_payload

    checkpoint_path = tmp_path / "groupfree.pth"
    payload = {"weight": torch.arange(3)}
    torch.save(payload, checkpoint_path)
    expected = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    loaded = _load_pointnet_checkpoint_payload(
        str(checkpoint_path), expected
    )
    assert torch.equal(loaded["weight"], payload["weight"])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_pointnet_checkpoint_payload(
            str(checkpoint_path), "0" * 64
        )


def test_default_off_integration_is_guarded_before_auxiliary_call(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1] / "models" / "losses.py"
    ).read_text(encoding="utf-8")
    assert "density_aware_target_box_loss_weight=0.0" in source
    assert "if density_aware_target_box_loss_weight > 0:" in source
    assert source.count("compute_density_aware_target_box_loss(") == 1

    main_source = (
        Path(__file__).resolve().parents[1] / "main_utils.py"
    ).read_text(encoding="utf-8")
    assert "--density_aware_target_box_loss_weight" in main_source
    assert "type=float, default=0.0" in main_source

    from models import losses as losses_module

    def forbidden_auxiliary_call(**_kwargs):
        raise AssertionError("default-off path called density auxiliary")

    monkeypatch.setattr(
        losses_module,
        "compute_density_aware_target_box_loss",
        forbidden_auxiliary_call,
    )

    class FakeSetCriterion:
        def __call__(self, output, _targets):
            zero = output["pred_boxes"].sum() * 0.0
            return ({
                "loss_ce": zero + 1.0,
                "loss_bbox": zero + 2.0,
                "loss_giou": zero + 3.0,
                "loss_mask": zero,
                "loss_dice": zero,
                "sp_loss_mask": zero,
                "sp_loss_dice": zero,
                "corresponding_loss_mask": zero,
                "corresponding_loss_dice": zero,
                "adaptive_weight_loss_mask": zero,
                "adaptive_weight_loss_dice": zero,
            }, [(torch.tensor([0]), torch.tensor([0]))])

    def make_end_points():
        result = {
            "center_label": torch.zeros(1, 1, 3),
            "size_gts": torch.ones(1, 1, 3),
            "sem_cls_label": torch.zeros(1, 1, dtype=torch.long),
            "gt_masks": torch.zeros(1, 1, 4),
            "positive_map": torch.zeros(1, 1, 256),
            "modify_positive_map": torch.zeros(1, 1, 256),
            "pron_positive_map": torch.zeros(1, 1, 256),
            "other_entity_map": torch.zeros(1, 1, 256),
            "rel_positive_map": torch.zeros(1, 1, 256),
            "box_label_mask": torch.ones(1, 1),
            "auxi_entity_positive_map": torch.zeros(1, 1, 256),
            "auxi_box": torch.zeros(1, 6),
            "language_dataset": ["nr3d"],
            "sample_dataset": ["nr3d"],
            "point_instance_label": torch.zeros(1, 4, dtype=torch.long),
            "superpoints": torch.zeros(1, 4, dtype=torch.long),
        }
        for prefix in ("proposal_", "last_"):
            result[prefix + "center"] = torch.zeros(1, 1, 3)
            result[prefix + "pred_size"] = torch.ones(1, 1, 3)
            result[prefix + "sem_cls_scores"] = torch.zeros(1, 1, 2)
            result[prefix + "pred_masks"] = torch.zeros(1, 1, 4)
        return result

    implicit_loss, implicit_end_points = losses_module.compute_hungarian_loss(
        make_end_points(), 1, FakeSetCriterion()
    )
    explicit_loss, explicit_end_points = losses_module.compute_hungarian_loss(
        make_end_points(), 1, FakeSetCriterion(),
        density_aware_target_box_loss_weight=0.0,
    )
    assert torch.equal(implicit_loss, explicit_loss)
    assert implicit_end_points.keys() == explicit_end_points.keys()
    assert not any(
        key.startswith("density_aware_target_box_")
        for key in implicit_end_points
    )
    for key in implicit_end_points:
        left = implicit_end_points[key]
        right = explicit_end_points[key]
        if torch.is_tensor(left):
            assert torch.equal(left, right), key
        else:
            assert left == right, key

    _, audit_end_points = losses_module.compute_hungarian_loss(
        make_end_points(), 1, FakeSetCriterion(),
        density_scene_audit_return_match_indices=True,
    )
    assert "density_scene_audit_last_match_indices" in audit_end_points
    source_indices, target_indices = audit_end_points[
        "density_scene_audit_last_match_indices"
    ][0]
    assert torch.equal(source_indices, torch.tensor([0]))
    assert torch.equal(target_indices, torch.tensor([0]))
