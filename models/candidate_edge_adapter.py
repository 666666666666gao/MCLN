"""Bind the relation-readout prototype to existing MCLN inference inputs."""

import torch

from .rec_evaluator_filter import build_detector_overlap_valid
from .source_choice_adapter import compute_default_source_scores


def build_candidate_edge_inputs(end_points, inputs, decoder_query_last, top_k=32):
    """Use final unprojected Queries, post-encoder text and legal Default Top-K.

    ``decoder_query_last`` is the 288-dimensional tensor before ``x_query``;
    ``source_choice_candidate_feats`` is a different 64-dimensional projection.
    The detector boxes are the existing model inputs. Root GT boxes, target IDs
    and IoU labels are not read. The returned full memory and compact target
    mapping are shared by the global and pair readouts.
    """
    boxes = torch.cat([
        end_points["last_center"],
        end_points["last_pred_size"].clamp(min=1e-6),
    ], dim=-1)
    valid = build_detector_overlap_valid(
        boxes,
        torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device),
        inputs["det_boxes"], inputs["det_bbox_label_mask"],
        iou_threshold=0.25,
    )
    default_scores = compute_default_source_scores(end_points, inputs)
    query_indices = default_scores.masked_fill(
        ~valid, -float("inf"),
    ).argsort(dim=1, descending=True)[:, :top_k]
    return {
        "candidate_feats": decoder_query_last,
        "candidate_boxes": boxes,
        "text_feats": end_points["text_memory"],
        "text_padding_mask": end_points["text_attention_mask"],
        "query_indices": query_indices,
        "valid_query_mask": valid,
    }
