import importlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _trainer():
    return importlib.import_module("scripts.train_scanrefer_joint_mask_head")


class _RootMaskDataset(Dataset):
    def __init__(self, mask_kind="tensor", split="train"):
        self.split = split
        self.metadata = {"source": "base"}
        self.length_calls = 0
        self.item_calls = []
        values = np.arange(132 * 5, dtype=np.int16).reshape(132, 5)
        if mask_kind == "tensor":
            self.masks = [torch.from_numpy(values + offset).clone()
                          for offset in (0, 1000)]
        else:
            self.masks = [(values + offset).copy()
                          for offset in (0, 1000)]
        self.last_mapping = None

    def __len__(self):
        self.length_calls += 1
        return len(self.masks)

    def __getitem__(self, index):
        self.item_calls.append(index)
        self.last_mapping = {
            "gt_masks": self.masks[index],
            "metadata": self.metadata,
        }
        return self.last_mapping


@pytest.mark.parametrize("mask_kind", ["tensor", "numpy"])
def test_root_mask_train_view_crops_before_default_collate_and_owns_storage(
        mask_kind):
    trainer = _trainer()
    base = _RootMaskDataset(mask_kind=mask_kind)
    view = trainer.RootMaskTrainDatasetView(base)

    assert len(view) == len(base)
    sample = view[0]
    assert sample is not base.last_mapping
    assert sample["metadata"] is base.metadata
    assert sample["dataset_index"] == 0
    assert sample["gt_masks"].shape == (1, 5)
    assert sample["gt_masks"].dtype == base.masks[0].dtype
    if isinstance(sample["gt_masks"], torch.Tensor):
        assert sample["gt_masks"].is_contiguous()
        sample["gt_masks"].fill_(-7)
        assert not bool(base.masks[0].eq(-7).any().item())
    else:
        assert sample["gt_masks"].flags.c_contiguous
        sample["gt_masks"].fill(-7)
        assert not bool(np.any(base.masks[0] == -7))

    batch = next(iter(DataLoader(view, batch_size=2, shuffle=False)))
    assert batch["gt_masks"].shape == (2, 1, 5)
    assert batch["gt_masks"].shape[1] != 132
    assert torch.equal(batch["dataset_index"], torch.tensor([0, 1]))


class _SingleItemDataset(Dataset):
    def __init__(self, item, split="train"):
        self.split = split
        self.item = item
        self.item_calls = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.item_calls += 1
        return self.item


@pytest.mark.parametrize("split", [None, "val", "test", 1])
def test_root_mask_train_view_rejects_non_train_dataset_without_access(split):
    trainer = _trainer()
    base = _SingleItemDataset({"gt_masks": torch.ones(1, 3)})
    if split is None:
        del base.split
    else:
        base.split = split

    with pytest.raises(ValueError, match="train"):
        trainer.RootMaskTrainDatasetView(base)

    assert base.item_calls == 0


@pytest.mark.parametrize(
    "item,message",
    [
        ([], "mapping"),
        ({}, "gt_masks"),
        ({"gt_masks": [[1, 0]]}, "Tensor or numpy"),
        ({"gt_masks": torch.ones(3)}, "shape"),
        ({"gt_masks": torch.ones(0, 3)}, "shape"),
        ({"gt_masks": torch.ones(2, 0)}, "shape"),
    ],
)
def test_root_mask_train_view_rejects_invalid_sample_masks(item, message):
    trainer = _trainer()
    view = trainer.RootMaskTrainDatasetView(_SingleItemDataset(item))

    with pytest.raises((TypeError, ValueError), match=message):
        view[0]


def test_root_mask_train_view_rejects_conflicting_dataset_index():
    trainer = _trainer()
    view = trainer.RootMaskTrainDatasetView(_SingleItemDataset({
        "gt_masks": np.ones((2, 3), dtype=np.uint8),
        "dataset_index": 9,
    }))

    with pytest.raises(ValueError, match="dataset_index"):
        view[0]


def test_root_mask_train_view_does_not_truncate_conflicting_dataset_index():
    trainer = _trainer()
    view = trainer.RootMaskTrainDatasetView(_SingleItemDataset({
        "gt_masks": torch.ones(2, 3),
        "dataset_index": 0.5,
    }))

    with pytest.raises(ValueError, match="dataset_index"):
        view[0]


def test_root_mask_train_view_normalizes_matching_dataset_index_to_int():
    trainer = _trainer()
    view = trainer.RootMaskTrainDatasetView(_SingleItemDataset({
        "gt_masks": torch.ones(2, 3, dtype=torch.bool),
        "dataset_index": np.int64(0),
    }))

    sample = view[0]

    assert type(sample["dataset_index"]) is int
    assert sample["dataset_index"] == 0


def _candidate_batch():
    return {
        "query_indices": torch.tensor([
            [5, 2, 7, 1],
            [4, 8, 6, 0],
            [3, 1, 9, 6],
        ], dtype=torch.long),
        "valid_mask": torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
            [False, True, True, True],
        ]),
        "candidate_ious": torch.tensor([
            [0.60, 0.80, 0.80, 0.99],
            [0.25, 0.10, 0.99, 0.99],
            [0.99, 0.70, 0.70, 0.60],
        ], dtype=torch.float64),
    }


def test_root_query_selection_is_strict_stable_and_never_uses_invalid_slots():
    trainer = _trainer()

    selected = trainer.select_root_box_supervision_queries(
        _candidate_batch(), threshold=0.25
    )

    assert set(selected) == {
        "selected_query_indices",
        "selected_candidate_slots",
        "selected_candidate_ious",
        "eligible_mask",
    }
    assert selected["selected_query_indices"].dtype == torch.long
    assert selected["selected_candidate_slots"].dtype == torch.long
    assert selected["selected_candidate_ious"].dtype == torch.float64
    assert selected["eligible_mask"].dtype == torch.bool
    assert selected["selected_query_indices"].tolist() == [2, -1, 1]
    assert selected["selected_candidate_slots"].tolist() == [1, -1, 1]
    assert selected["selected_candidate_ious"].tolist() == [0.8, 0.0, 0.7]
    assert selected["eligible_mask"].tolist() == [True, False, True]


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_root_query_selection_rejects_non_finite_valid_iou(non_finite):
    trainer = _trainer()
    candidates = _candidate_batch()
    candidates["candidate_ious"][0, 1] = non_finite

    with pytest.raises(ValueError, match="finite"):
        trainer.select_root_box_supervision_queries(candidates)


@pytest.mark.parametrize(
    "field,replacement,error_type,message",
    [
        ("query_indices", torch.zeros(3, 4, dtype=torch.int32),
         TypeError, "int64"),
        ("valid_mask", torch.ones(3, 4, dtype=torch.uint8),
         TypeError, "bool"),
        ("candidate_ious", torch.ones(3, 4, dtype=torch.long),
         TypeError, "floating"),
        ("candidate_ious", torch.ones(3, 3), ValueError, "shape"),
        ("query_indices", torch.tensor([
            [-1, 2, 7, 1], [4, 8, 6, 0], [3, 1, 9, 6]
        ]), ValueError, "non-negative"),
        ("valid_mask", torch.empty(3, 4, dtype=torch.bool, device="meta"),
         ValueError, "device"),
    ],
)
def test_root_query_selection_validates_candidate_contract(
        field, replacement, error_type, message):
    trainer = _trainer()
    candidates = _candidate_batch()
    candidates[field] = replacement

    with pytest.raises(error_type, match=message):
        trainer.select_root_box_supervision_queries(candidates)


@pytest.mark.parametrize("threshold", [0.0, 0.250001, 0.5])
def test_root_query_selection_rejects_non_fixed_threshold(threshold):
    trainer = _trainer()

    with pytest.raises(ValueError, match="0.25"):
        trainer.select_root_box_supervision_queries(
            _candidate_batch(), threshold=threshold
        )


def _root_loss_inputs(second_row_eligible=True):
    logits = [
        torch.tensor([
            [-1.5, 0.2, 1.1],
            [0.7, -0.4, 1.6],
            [1.2, 0.3, -1.1],
        ], dtype=torch.float64, requires_grad=True),
        torch.tensor([
            [-0.8, 1.3, 0.1, -1.7],
            [0.4, -0.6, 1.5, 0.2],
        ], dtype=torch.float64, requires_grad=True),
    ]
    candidate_ious = torch.tensor([
        [0.80, 0.50],
        [0.90, 0.40] if second_row_eligible else [0.25, 0.10],
    ], dtype=torch.float64)
    candidates = {
        "query_indices": torch.tensor([[1, 2], [0, 1]]),
        "valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "candidate_ious": candidate_ious,
    }
    gt_masks = torch.tensor([
        [[1, 0, 1, 0, 1, 1, 0]],
        [[0, 1, 1, 0, 0, 1, 1]],
    ], dtype=torch.bool)
    superpoint = torch.tensor([
        [0, 0, 1, 2, 2, 2, 0],
        [0, 1, 1, 2, 3, 3, 3],
    ], dtype=torch.long)
    return logits, candidates, gt_masks, superpoint


def _explicit_point_loss(logits, selection, gt_masks, superpoint):
    focals = []
    dices = []
    for batch_index, eligible in enumerate(
            selection["eligible_mask"].tolist()):
        if not eligible:
            continue
        query_index = int(
            selection["selected_query_indices"][batch_index].item()
        )
        point_logits = logits[batch_index][query_index].index_select(
            0, superpoint[batch_index]
        )
        target = gt_masks[batch_index, 0].to(point_logits.dtype)
        probability = point_logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            point_logits, target, reduction="none"
        )
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = 0.25 * target + 0.75 * (1.0 - target)
        focals.append((alpha_t * bce * (1.0 - p_t).pow(2.0)).mean())
        intersection = (probability * target).sum()
        dices.append(1.0 - (2.0 * intersection + 1.0) / (
            probability.sum() + target.sum() + 1.0
        ))
    focal = torch.stack(focals).mean()
    dice = torch.stack(dices).mean()
    return {"focal": focal, "dice": dice, "loss": focal + dice}


def test_point_count_root_mask_loss_matches_explicit_points_and_backward():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs()
    selection = trainer.select_root_box_supervision_queries(candidates)

    actual = trainer.compute_point_count_weighted_root_mask_loss(
        logits, selection, gt_masks, superpoint
    )
    expected = _explicit_point_loss(
        logits, selection, gt_masks, superpoint
    )

    assert set(actual) == {"loss", "focal", "dice", "eligible_count"}
    assert type(actual["eligible_count"]) is int
    assert actual["eligible_count"] == 2
    for key in ("focal", "dice", "loss"):
        assert actual[key].shape == ()
        assert torch.isfinite(actual[key])
        torch.testing.assert_close(actual[key], expected[key])

    actual["loss"].backward()
    for row, selected_query in zip(logits, (1, 0)):
        assert row.grad is not None
        assert bool(torch.isfinite(row.grad).all().item())
        assert float(row.grad[selected_query].abs().sum().item()) > 0.0
        unselected = torch.cat([
            row.grad[:selected_query], row.grad[selected_query + 1:]
        ], dim=0)
        assert torch.equal(unselected, torch.zeros_like(unselected))


@pytest.mark.parametrize(
    "dtype,point_total",
    [
        (torch.bfloat16, 257),
        (torch.float16, 2049),
    ],
)
def test_point_count_root_mask_loss_accumulates_low_precision_in_float32(
        dtype, point_total):
    trainer = _trainer()
    logits = [torch.tensor(
        [[1.25], [-0.75]], dtype=dtype, requires_grad=True
    )]
    candidates = {
        "query_indices": torch.tensor([[0]], dtype=torch.long),
        "valid_mask": torch.tensor([[True]]),
        "candidate_ious": torch.tensor([[0.80]], dtype=torch.float32),
    }
    positive_total = point_total // 2
    gt_masks = torch.zeros(1, 1, point_total, dtype=torch.bool)
    gt_masks[0, 0, :positive_total] = True
    superpoint = torch.zeros(1, point_total, dtype=torch.long)
    selection = trainer.select_root_box_supervision_queries(candidates)

    actual = trainer.compute_point_count_weighted_root_mask_loss(
        logits, selection, gt_masks, superpoint
    )

    point_logits = logits[0][0].index_select(0, superpoint[0]).float()
    target = gt_masks[0, 0].float()
    probability = point_logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(
        point_logits, target, reduction="none"
    )
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = 0.25 * target + 0.75 * (1.0 - target)
    expected_focal = (
        alpha_t * bce * (1.0 - p_t).pow(2.0)
    ).mean()
    intersection = (probability * target).sum()
    expected_dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum() + target.sum() + 1.0
    )
    expected = {
        "focal": expected_focal,
        "dice": expected_dice,
        "loss": expected_focal + expected_dice,
    }

    for key in ("focal", "dice", "loss"):
        assert actual[key].dtype in (torch.float32, torch.float64)
        torch.testing.assert_close(
            actual[key].float(), expected[key], rtol=1e-5, atol=1e-5
        )
    actual["loss"].backward()
    assert logits[0].grad is not None
    assert logits[0].grad.dtype == dtype
    assert bool(torch.isfinite(logits[0].grad).all().item())
    assert float(logits[0].grad[0].abs().sum().item()) > 0.0
    assert torch.equal(logits[0].grad[1], torch.zeros_like(logits[0].grad[1]))


def test_point_count_root_mask_loss_ignores_ineligible_rows_exactly():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs(
        second_row_eligible=False
    )
    selection = trainer.select_root_box_supervision_queries(candidates)

    actual = trainer.compute_point_count_weighted_root_mask_loss(
        logits, selection, gt_masks, superpoint
    )
    expected = _explicit_point_loss(
        logits, selection, gt_masks, superpoint
    )

    assert selection["eligible_mask"].tolist() == [True, False]
    assert actual["eligible_count"] == 1
    for key in ("focal", "dice", "loss"):
        torch.testing.assert_close(actual[key], expected[key])
    actual["loss"].backward()
    assert logits[0].grad is not None
    assert logits[1].grad is None


def test_point_count_root_mask_loss_raises_typed_error_without_eligible_row():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs(
        second_row_eligible=False
    )
    candidates["candidate_ious"][0] = torch.tensor([0.25, 0.20])
    selection = trainer.select_root_box_supervision_queries(candidates)

    with pytest.raises(RuntimeError, match="eligible") as error:
        trainer.compute_point_count_weighted_root_mask_loss(
            logits, selection, gt_masks, superpoint
        )

    assert type(error.value) is trainer.NoEligibleRootMaskSupervisionError


@pytest.mark.parametrize(
    "replacement,message",
    [
        (torch.tensor([
            [-1, 0, 1, 2, 2, 2, 0],
            [0, 1, 1, 2, 3, 3, 3],
        ]), "non-negative"),
        (torch.tensor([
            [0, 0, 1, 3, 2, 2, 0],
            [0, 1, 1, 2, 3, 3, 3],
        ]), "range"),
        (torch.zeros(2, 7, dtype=torch.int32), "int64"),
    ],
)
def test_point_count_root_mask_loss_validates_superpoint_ids(
        replacement, message):
    trainer = _trainer()
    logits, candidates, gt_masks, _superpoint = _root_loss_inputs()
    selection = trainer.select_root_box_supervision_queries(candidates)

    with pytest.raises((TypeError, ValueError), match=message):
        trainer.compute_point_count_weighted_root_mask_loss(
            logits, selection, gt_masks, replacement
        )


def test_point_count_root_mask_loss_rejects_non_finite_unselected_logit():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs()
    logits[0] = logits[0].detach().clone()
    logits[0][0, 0] = float("nan")
    logits[0].requires_grad_(True)
    selection = trainer.select_root_box_supervision_queries(candidates)

    with pytest.raises(ValueError, match="finite"):
        trainer.compute_point_count_weighted_root_mask_loss(
            logits, selection, gt_masks, superpoint
        )


def test_point_count_root_mask_loss_rejects_selected_query_out_of_range():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs()
    candidates["query_indices"][0, 0] = 99
    selection = trainer.select_root_box_supervision_queries(candidates)

    with pytest.raises(ValueError, match="query.*range"):
        trainer.compute_point_count_weighted_root_mask_loss(
            logits, selection, gt_masks, superpoint
        )


def test_point_count_root_mask_loss_validates_root_mask_shape():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs()
    selection = trainer.select_root_box_supervision_queries(candidates)

    with pytest.raises(ValueError, match=r"\[B,1,N\]"):
        trainer.compute_point_count_weighted_root_mask_loss(
            logits, selection, gt_masks[:, 0], superpoint
        )


class _ReadTrackingMapping(dict):
    def __init__(self, *args, **kwargs):
        super(_ReadTrackingMapping, self).__init__(*args, **kwargs)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super(_ReadTrackingMapping, self).__getitem__(key)


def test_train_only_root_mask_supervision_composes_without_other_reads():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs()
    end_points = _ReadTrackingMapping({
        "sp_last_pred_masks": logits,
        "model_forward": object(),
        "validation_outputs": object(),
    })
    batch = _ReadTrackingMapping({
        "gt_masks": gt_masks,
        "superpoint": superpoint,
        "validation_loader": object(),
    })
    expected_selection = trainer.select_root_box_supervision_queries(
        candidates
    )
    expected_loss = trainer.compute_point_count_weighted_root_mask_loss(
        logits, expected_selection, gt_masks, superpoint
    )

    actual = trainer.compute_train_only_root_mask_supervision(
        end_points, batch, candidates
    )

    assert set(actual) == {
        "loss", "focal", "dice", "eligible_count", "selection"
    }
    assert end_points.reads == ["sp_last_pred_masks"]
    assert batch.reads == ["gt_masks", "superpoint"]
    for key in ("loss", "focal", "dice"):
        torch.testing.assert_close(actual[key], expected_loss[key])
    assert actual["eligible_count"] == expected_loss["eligible_count"]
    for key, value in expected_selection.items():
        assert torch.equal(actual["selection"][key], value)


def test_train_only_root_mask_supervision_propagates_no_eligible_signal():
    trainer = _trainer()
    logits, candidates, gt_masks, superpoint = _root_loss_inputs(
        second_row_eligible=False
    )
    candidates["candidate_ious"][0] = torch.tensor([0.25, 0.10])

    with pytest.raises(trainer.NoEligibleRootMaskSupervisionError):
        trainer.compute_train_only_root_mask_supervision(
            {"sp_last_pred_masks": logits},
            {"gt_masks": gt_masks, "superpoint": superpoint},
            candidates,
        )
