"""GT-only least-squares geometry diagnostic, never an inference policy or IoU upper bound."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONTROL = Path(r'C:\Users\gb\.codex\tmp\pvground_mask_box_controls_20261009')
N = 9508


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(boxes, gt):
    low = np.maximum(boxes[:, :3] - boxes[:, 3:] / 2, gt[:, :3] - gt[:, 3:] / 2)
    high = np.minimum(boxes[:, :3] + boxes[:, 3:] / 2, gt[:, :3] + gt[:, 3:] / 2)
    inter = np.maximum(high - low, 0).prod(axis=1)
    return inter / (boxes[:, 3:].prod(axis=1) + gt[:, 3:].prod(axis=1) - inter)


def main():
    source = CONTROL / 'SELECTED_BOX_ROWS.jsonl'
    base_summary = json.loads((CONTROL / 'SUMMARY.json').read_bytes())
    audit = json.loads((CONTROL / 'audit/EXPERIMENT_AUDIT.json').read_bytes())
    assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
    assert sha(source) == base_summary['output_rows_sha256']
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == N and [row['row_id'] for row in rows] == list(range(N))
    gt = np.asarray([row['root_box'] for row in rows], dtype=np.float64)
    native = np.asarray([row['boxes']['native_regression'] for row in rows], dtype=np.float64)
    mask = np.asarray([row['boxes']['mask_reference'] for row in rows], dtype=np.float64)
    assert (native[:, 3:] > 0).all() and (mask[:, 3:] > 0).all()
    delta, target = native - mask, gt - mask
    numerator = delta[:, :3] * target[:, :3] + delta[:, 3:] * target[:, 3:]
    denominator = delta[:, :3] ** 2 + delta[:, 3:] ** 2
    # Observed zero-length spans (including the archived39 invalid references)
    # have no geometric degree of freedom; coefficient0 yields the same box.
    alpha = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    alpha = np.clip(alpha, 0, 1)
    diagnostic = mask + np.concatenate([alpha, alpha], axis=1) * delta
    assert np.isfinite(diagnostic).all() and (diagnostic[:, 3:] > 0).all()
    mask_iou, diag_iou = iou(mask, gt), iou(diagnostic, gt)
    old_loss = ((mask - gt) ** 2).sum(axis=1)
    new_loss = ((diagnostic - gt) ** 2).sum(axis=1)
    assert (new_loss <= old_loss + 1e-12).all()
    result = dict(status='EXECUTED_GT_ONLY_CONSTRAINED_LEAST_SQUARES_AXIS_SPAN_DIAGNOSTIC',
        time_cst=datetime.datetime.now().astimezone().isoformat(), rows=N, source_snapshot='20261006_initial_formal',
        source_rows_sha256=sha(source), source_controls_summary_sha256=sha(CONTROL / 'SUMMARY.json'),
        source_controls_audit_sha256=sha(CONTROL / 'audit/EXPERIMENT_AUDIT.json'), source_script_sha256=sha(Path(__file__)),
        gt_used_to_compute_coefficients=True, deployment_valid=False, iou_upper_bound=False,
        candidate_query_changes=0, positive_size_by_axis_shared_convex_coefficients=True,
        zero_length_axis_spans=int((denominator == 0).sum()),
        both_boxes_identical_rows=int((denominator == 0).all(axis=1).sum()),
        coefficient_zero_axes=int((alpha == 0).sum()), coefficient_one_axes=int((alpha == 1).sum()),
        interior_coefficient_axes=int(((alpha > 0) & (alpha < 1)).sum()),
        objective='Per-axis squared error in center and size, equal coordinate weights; not IoU maximization.',
        mask_baseline_hits=[int((mask_iou > t).sum()) for t in (.25, .5)],
        diagnostic_gt_conditioned_hits=[int((diag_iou > t).sum()) for t in (.25, .5)],
        comparisons={str(t):dict(repairs=int(((mask_iou <= t) & (diag_iou > t)).sum()),
                                 damages=int(((mask_iou > t) & (diag_iou <= t)).sum()),
                                 net=int((diag_iou > t).sum() - (mask_iou > t).sum())) for t in (.25, .5)},
        neural_forwards=0, optimizer_updates=0, ssh_queries=0, new_weights=0,
        retained_best_changed=False, full_goal_complete=False, three_effective_contributions=False,
        limits=[
            'Uses validation GT to choose coefficients offline; forbidden as inference, new-method performance, or model promotion.',
            'Least-squares error decreases by construction but IoU and threshold hits need not increase per row.',
            'Not the optimum IoU over the permitted family and not a geometric oracle upper bound.',
            'Historical one-snapshot diagnostic of representational space; a trainable head must infer weights from prediction-only evidence.',
            'A convex axis span cannot reach a GT range beyond both priors; this does not resolve instance-selection errors by itself.',
            'The same coefficient applies to center and size for each axis; this retains positive sizes without independent-face crossing fixes.'
        ])
    (ROOT / 'AXIS_SPAN_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
