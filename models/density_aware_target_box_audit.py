"""Deployment-aligned metrics for the density target-box short audit."""

import hashlib
import json
import math

import torch

from .rec_evaluator_filter import build_detector_overlap_valid
from .source_moe import compute_query_box_ious


AUDIT_SLICE_NAMES = (
    "overall",
    "active_sparse",
    "dense",
    "zero_point",
)


def scene_sample_identity_digest(sample_ids):
    """Return the frozen order-independent identity of one row set."""
    values = list(sample_ids)
    if any(
            not isinstance(value, int) or isinstance(value, bool)
            or value < 0 for value in values):
        raise ValueError("scene-audit sample ids must be nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("scene-audit sample ids must be unique")
    encoded = json.dumps(
        sorted(values), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_slice_metrics():
    return {
        "sample_count": 0,
        "target_point_count_sum": 0,
        "selected_hits025": 0,
        "selected_hits050": 0,
        "top16_hits025": 0,
        "top16_hits050": 0,
        "matched_iou_sum": 0.0,
        "matched_hits025": 0,
        "matched_hits050": 0,
        "matched_center_l1_sum": 0.0,
        "matched_size_l1_sum": 0.0,
    }


def _validated_scalar(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def _finalize_slice(values):
    count = int(values["sample_count"])
    if count < 0:
        raise ValueError("audit slice sample count cannot be negative")
    integer_fields = (
        "selected_hits025",
        "selected_hits050",
        "top16_hits025",
        "top16_hits050",
        "matched_hits025",
        "matched_hits050",
    )
    for name in integer_fields:
        observed = int(values[name])
        if observed < 0 or observed > count:
            raise ValueError("{} is outside its sample denominator".format(name))
    for prefix in ("selected", "top16", "matched"):
        if values[prefix + "_hits050"] > values[prefix + "_hits025"]:
            raise ValueError("{} audit hits are not nested".format(prefix))

    result = dict(values)
    denominator = float(count) if count else 1.0
    result.update({
        "target_point_count_mean": (
            float(values["target_point_count_sum"]) / denominator
        ),
        "selected_acc025": float(values["selected_hits025"]) / denominator,
        "selected_acc050": float(values["selected_hits050"]) / denominator,
        "top16_acc025": float(values["top16_hits025"]) / denominator,
        "top16_acc050": float(values["top16_hits050"]) / denominator,
        "matched_iou_mean": float(values["matched_iou_sum"]) / denominator,
        "matched_acc025": float(values["matched_hits025"]) / denominator,
        "matched_acc050": float(values["matched_hits050"]) / denominator,
        "matched_center_l1_mean": (
            float(values["matched_center_l1_sum"]) / denominator
        ),
        "matched_size_l1_mean": (
            float(values["matched_size_l1_sum"]) / denominator
        ),
    })
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("final audit metric {} is non-finite".format(name))
    return result


class DensityAwareTargetBoxAuditAccumulator(object):
    """Accumulate exact sparse/dense proposal diagnostics on one holdout."""

    def __init__(self):
        self._slices = {
            name: _empty_slice_metrics() for name in AUDIT_SLICE_NAMES
        }
        self._sample_ids = []

    @staticmethod
    def _required_tensor(end_points, name, dimensions=None):
        value = end_points.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError("audit input {} must be a tensor".format(name))
        if dimensions is not None and value.dim() != dimensions:
            raise ValueError(
                "audit input {} must have {} dimensions".format(
                    name, dimensions
                )
            )
        return value

    @staticmethod
    def _sample_datasets(end_points, batch_size):
        values = end_points.get("sample_dataset")
        if isinstance(values, str):
            values = [values] * batch_size
        if not isinstance(values, (list, tuple)) or len(values) != batch_size:
            raise ValueError("audit sample_dataset must align with batch size")
        normalized = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("audit dataset labels must be non-empty strings")
            normalized.append(value.strip().lower())
        if any(value != "nr3d" for value in normalized):
            raise ValueError("density scene audit accepts only Nr3D rows")
        return normalized

    @staticmethod
    def _matched_query_indices(match_indices, batch_size, query_count):
        if not isinstance(match_indices, (list, tuple)):
            raise ValueError("audit Hungarian assignments must be a sequence")
        if len(match_indices) != batch_size:
            raise ValueError("audit Hungarian assignments must align with batch")
        result = []
        for source_indices, target_indices in match_indices:
            if (
                    not isinstance(source_indices, torch.Tensor)
                    or not isinstance(target_indices, torch.Tensor)
                    or source_indices.dim() != 1
                    or target_indices.dim() != 1
                    or source_indices.numel() != target_indices.numel()):
                raise ValueError("audit Hungarian assignment is malformed")
            positions = (target_indices.long() == 0).nonzero(
                as_tuple=False
            ).reshape(-1)
            if positions.numel() != 1:
                raise ValueError("GT target slot 0 must be matched exactly once")
            query_index = int(source_indices[int(positions[0].item())].item())
            if query_index < 0 or query_index >= query_count:
                raise ValueError("audit matched query is outside the query axis")
            result.append(query_index)
        return result

    def update(self, end_points):
        if not isinstance(end_points, dict):
            raise ValueError("density scene audit end_points must be a dict")
        centers = self._required_tensor(end_points, "last_center", 3)
        sizes = self._required_tensor(end_points, "last_pred_size", 3)
        scores = self._required_tensor(
            end_points, "selected_source_scores", 2
        )
        if centers.shape != sizes.shape or centers.shape[-1] != 3:
            raise ValueError("audit centers and sizes must align [B,Q,3]")
        batch_size, query_count = centers.shape[:2]
        if scores.shape != (batch_size, query_count):
            raise ValueError("audit selected scores must align [B,Q]")
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError("audit selected scores must be finite")
        if not bool(torch.isfinite(centers).all().item()):
            raise ValueError("audit centers must be finite")
        if not bool(torch.isfinite(sizes).all().item()):
            raise ValueError("audit sizes must be finite")
        if bool((sizes <= 0.0).any().item()):
            raise ValueError("audit candidate sizes must be positive")
        self._sample_datasets(end_points, batch_size)

        row_ids = self._required_tensor(
            end_points, "density_scene_audit_sample_index", 1
        )
        if row_ids.shape[0] != batch_size:
            raise ValueError("audit row identities must align with batch")
        row_ids = [int(value) for value in row_ids.detach().cpu().tolist()]
        if any(value < 0 for value in row_ids):
            raise ValueError("audit row identities must be nonnegative")
        self._sample_ids.extend(row_ids)

        point_labels = self._required_tensor(
            end_points, "point_instance_label", 2
        )
        if point_labels.shape[0] != batch_size:
            raise ValueError("audit point labels must align with batch")
        point_counts = (point_labels.detach() == 0).sum(dim=1)

        gt_centers = self._required_tensor(end_points, "center_label", 3)
        gt_sizes = self._required_tensor(end_points, "size_gts", 3)
        gt_valid = self._required_tensor(end_points, "box_label_mask", 2).bool()
        if (
                gt_centers.shape != gt_sizes.shape
                or gt_centers.shape[-1] < 3
                or gt_valid.shape != gt_centers.shape[:2]
                or gt_centers.shape[0] != batch_size):
            raise ValueError("audit GT box tensors are inconsistent")
        if not bool(gt_valid[:, 0].all().item()):
            raise ValueError("audit requires GT target slot 0 in every row")
        target_boxes = torch.cat(
            [gt_centers[..., :3], gt_sizes[..., :3]], dim=-1
        )
        target_valid = torch.zeros_like(gt_valid)
        target_valid[:, 0] = True

        candidate_boxes = torch.cat([centers, sizes], dim=-1)
        candidate_valid = build_detector_overlap_valid(
            candidate_boxes,
            torch.ones_like(scores, dtype=torch.bool),
            self._required_tensor(end_points, "all_detected_boxes", 3),
            self._required_tensor(
                end_points, "all_detected_bbox_label_mask", 2
            ).bool(),
            iou_threshold=0.25,
        )
        target_ious = compute_query_box_ious(
            candidate_boxes, target_boxes, target_valid
        )
        matched_queries = self._matched_query_indices(
            end_points.get("density_scene_audit_last_match_indices"),
            batch_size,
            query_count,
        )

        for row_index in range(batch_size):
            point_count = int(point_counts[row_index].item())
            if point_count < 0:
                raise ValueError("audit target point count cannot be negative")
            if point_count == 0:
                slice_names = ("overall", "zero_point")
            elif point_count < 256:
                slice_names = ("overall", "active_sparse")
            else:
                slice_names = ("overall", "dense")

            valid_indices = candidate_valid[row_index].nonzero(
                as_tuple=False
            ).reshape(-1).detach().cpu().tolist()
            ranked_indices = sorted(
                (int(index) for index in valid_indices),
                key=lambda index: (
                    -float(scores[row_index, index].item()), index
                ),
            )
            if ranked_indices:
                selected_iou = _validated_scalar(
                    target_ious[row_index, ranked_indices[0]].item(),
                    "selected IoU",
                )
                top16_iou = max(
                    _validated_scalar(
                        target_ious[row_index, index].item(), "Top-16 IoU"
                    )
                    for index in ranked_indices[:16]
                )
            else:
                selected_iou = 0.0
                top16_iou = 0.0

            matched_query = matched_queries[row_index]
            matched_iou = _validated_scalar(
                target_ious[row_index, matched_query].item(), "matched IoU"
            )
            predicted = candidate_boxes[row_index, matched_query]
            target = target_boxes[row_index, 0]
            center_l1 = _validated_scalar(
                (predicted[:3] - target[:3]).abs().sum().item(),
                "matched center L1",
            )
            size_l1 = _validated_scalar(
                (predicted[3:] - target[3:]).abs().sum().item(),
                "matched size L1",
            )

            for slice_name in slice_names:
                values = self._slices[slice_name]
                values["sample_count"] += 1
                values["target_point_count_sum"] += point_count
                values["selected_hits025"] += int(selected_iou > 0.25)
                values["selected_hits050"] += int(selected_iou > 0.50)
                values["top16_hits025"] += int(top16_iou > 0.25)
                values["top16_hits050"] += int(top16_iou > 0.50)
                values["matched_iou_sum"] += matched_iou
                values["matched_hits025"] += int(matched_iou > 0.25)
                values["matched_hits050"] += int(matched_iou > 0.50)
                values["matched_center_l1_sum"] += center_l1
                values["matched_size_l1_sum"] += size_l1

    def finalize(self, expected_sample_count, expected_identity_sha256):
        if (
                not isinstance(expected_sample_count, int)
                or isinstance(expected_sample_count, bool)
                or expected_sample_count <= 0):
            raise ValueError("expected audit sample count must be positive")
        if len(self._sample_ids) != expected_sample_count:
            raise ValueError(
                "audit evaluated {} rows, expected {}".format(
                    len(self._sample_ids), expected_sample_count
                )
            )
        observed_identity = scene_sample_identity_digest(self._sample_ids)
        if observed_identity != expected_identity_sha256:
            raise ValueError("audit holdout row identity drifted")
        slices = {
            name: _finalize_slice(self._slices[name])
            for name in AUDIT_SLICE_NAMES
        }
        if (
                slices["active_sparse"]["sample_count"]
                + slices["dense"]["sample_count"]
                + slices["zero_point"]["sample_count"]
                != slices["overall"]["sample_count"]):
            raise ValueError("density audit slices do not partition the holdout")
        return {
            "schema": "mcln-density-target-box-scene-metrics-v1",
            "sample_count": expected_sample_count,
            "sample_identity_count": len(self._sample_ids),
            "sample_identity_unique_count": len(set(self._sample_ids)),
            "sample_identity_sha256": observed_identity,
            "candidate_filter": "butd-cls-overlap-iou-strictly-greater-0.25",
            "ranking": "selected-source-scores-desc-query-index-asc",
            "top_k": 16,
            "slices": slices,
        }
