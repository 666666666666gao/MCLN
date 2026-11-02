"""Per-example audit of the current evaluator's root semantic Top1 branch."""

import hashlib
import json

import torch


def root_semantic_scores(end_points):
    similarities = torch.matmul(
        end_points["last_proj_queries"],
        end_points["proj_tokens"].transpose(-1, -2),
    )
    probabilities = (similarities / 0.07).softmax(-1)
    scores = probabilities.new_zeros(
        probabilities.shape[0], probabilities.shape[1], 256
    )
    scores[..., :probabilities.shape[-1]] = probabilities
    main = (end_points["positive_map"][:, 0] > 0).to(scores.dtype)
    total = (scores * main[:, None]).sum(-1)
    for name, sign in (
        ("modify_positive_map", 1),
        ("pron_positive_map", 1),
        ("rel_positive_map", 1),
        ("other_entity_map", -1),
    ):
        total = total + sign * (
            scores * end_points[name][:, :1]
        ).sum(-1)
    return total


class PVGroundSemanticAudit:
    def __init__(self, stream, iou_function, box_converter):
        self.stream = stream
        self.iou_function = iou_function
        self.box_converter = box_converter
        self.samples = 0
        self.hits = {0.25: 0, 0.5: 0}

    def _ious(self, gt, boxes):
        values, _ = self.iou_function(
            self.box_converter(gt[None]),
            self.box_converter(boxes),
        )
        return values[0]

    @torch.no_grad()
    def update(self, end_points, gate_evidence):
        # ScanRefer uses no detector-overlap score filter in current config.
        scores = root_semantic_scores(end_points)
        ordered = scores.argsort(1, True)
        boxes = torch.cat([
            end_points["last_center"], end_points["last_pred_size"],
        ], dim=-1)
        gt = torch.cat([
            end_points["center_label"][:, 0, :3],
            end_points["size_gts"][:, 0],
        ], dim=-1)
        for index in range(len(boxes)):
            selected = int(ordered[index, 0])
            ious = self._ious(gt[index], boxes[index])
            chosen_iou = float(ious[selected])
            point_bytes = (
                end_points["point_clouds"][index].detach().cpu().contiguous()
                .numpy().tobytes()
            )
            row = {
                "kind": "semantic_sample", "sample_index": self.samples,
                "scan_id": end_points["scan_ids"][index],
                "target_id": int(end_points["target_id"][index]),
                "utterance": end_points["utterances"][index],
                "point_sha256": hashlib.sha256(point_bytes).hexdigest(),
                "is_unique": bool(end_points["is_unique"][index]),
                "selected_raw_gate": gate_evidence[index]["raw"][selected].tolist(),
                "selected_clamped_gate": gate_evidence[index]["clamped"][selected].tolist(),
                "selected_query": selected, "selected_iou": chosen_iou,
                "oracle_iou": float(ious.max()),
                "oracle_query": int(ious.argmax()),
                "top10_queries": ordered[index, :10].tolist(),
                "top10_ious": ious[ordered[index, :10]].tolist(),
                "first_good_rank025": self._first_good(ious, ordered[index], 0.25),
                "first_good_rank050": self._first_good(ious, ordered[index], 0.5),
            }
            for name in (
                "reader_query_l2", "reader_residual_l2_before_bypass",
                "support_delta_abs_mean_before_bypass",
                "support_delta_abs_max_before_bypass",
                "support_foreground_flips_before_bypass",
            ):
                row["selected_" + name] = gate_evidence[index][name][selected].item()
            row["support_observed_members"] = gate_evidence[index][
                "support_observed_members"
            ]
            for name, center, size in (
                ("native", "native_coarse_center", "native_coarse_size"),
                ("reference", "mask_reference_center", "mask_reference_size"),
                ("final", "last_center", "last_pred_size"),
            ):
                candidate = torch.cat([
                    end_points[center][index, selected],
                    end_points[size][index, selected],
                ])
                row[name + "_box"] = candidate.tolist()
                row[name + "_center_error"] = (
                    candidate[:3] - gt[index, :3]
                ).tolist()
                row[name + "_size_error"] = (
                    candidate[3:] - gt[index, 3:]
                ).tolist()
                row[name + "_iou"] = float(
                    self._ious(gt[index], candidate[None])[0]
                )
            row["gt_box"] = gt[index].tolist()
            row["selected_reference_valid"] = bool(
                end_points["mask_reference_valid"][index, selected]
            )
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            for threshold in self.hits:
                self.hits[threshold] += int(chosen_iou > threshold)
            self.samples += 1
        self.stream.flush()

    @staticmethod
    def _first_good(ious, ordered, threshold):
        ranks = torch.nonzero(ious[ordered] > threshold).flatten()
        return int(ranks[0]) + 1 if len(ranks) else None

    def verify(self, evaluator):
        for threshold, hits in self.hits.items():
            key = ("last_", threshold, 1, "bbf")
            if evaluator.dets[key] != hits or evaluator.gts[key] != self.samples:
                raise ValueError("per-example semantic audit disagrees with evaluator")
