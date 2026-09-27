"""Root-matched V99-style quality target for the native MCLN score."""

import torch

from .source_moe import compute_query_box_ious


def root_matched_quality_loss(scores, final_boxes, gt_boxes, matches,
                              sample_datasets, score_temperature):
    """Rank the matched root against existing no-object negatives only.

    Unmatched high-IoU queries remain outside this first control because the
    native soft-token and contrastive losses still label them no-object.
    """
    assert score_temperature > 0
    rows = []
    for row, source_name in enumerate(sample_datasets):
        if source_name != 'scanrefer':
            continue
        source, target = matches[row]
        root_query = source[target == 0]
        assert root_query.numel() == 1
        root_query = int(root_query[0])
        iou = compute_query_box_ious(
            final_boxes[row:row + 1].detach(),
            gt_boxes[row:row + 1, :1].detach(),
            torch.ones(1, 1, dtype=torch.bool, device=final_boxes.device),
        )[0]
        if float(iou[root_query]) <= 0.25:
            continue
        eligible = iou < 0.25
        eligible[source.to(eligible.device)] = False
        eligible[root_query] = True
        quality = iou + 2 * (iou > 0.25).float() + (iou > 0.5).float()
        target_distribution = (
            quality[eligible] / 0.25
        ).softmax(dim=0).detach()
        predicted_log_distribution = (
            scores[row, eligible] / score_temperature
        ).log_softmax(dim=0)
        rows.append(-(
            target_distribution * predicted_log_distribution
        ).sum())
    if not rows:
        return scores.sum() * 0
    return torch.stack(rows).mean()
