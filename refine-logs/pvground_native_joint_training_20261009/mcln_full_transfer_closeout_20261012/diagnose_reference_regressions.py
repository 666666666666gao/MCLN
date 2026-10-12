"""Actual saved-row diagnosis of the observed .25/.50 reference tradeoff."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
raw = (root / 'full_paired_rows.jsonl').read_bytes()
binding = json.loads((root / 'full_ROWS_TRANSFER.json').read_bytes())
assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
assert len(rows) == 9508
gt = np.asarray([row['native']['root_gt_box'] for row in rows], dtype=np.float64)
groups = np.searchsorted(np.quantile(gt[:, 3:].prod(-1), [.25, .5, .75]), gt[:, 3:].prod(-1), side='right')
summary = {}
for before, after in (('native', 'mask_reference'), ('mask_reference', 'full_transfer')):
    pairs = {}
    before_iou = np.asarray([row[before]['selected_iou'] for row in rows])
    after_iou = np.asarray([row[after]['selected_iou'] for row in rows])
    boxes = np.asarray([row[after]['selected_box'] for row in rows], dtype=np.float64)
    fused_iou = np.asarray([row[after]['selected_fused_mask_iou'] for row in rows])
    for threshold, name in ((.25, '025'), (.5, '050')):
        broken = (before_iou > threshold) & (after_iou <= threshold)
        repaired = (before_iou <= threshold) & (after_iou > threshold)
        selected = boxes[broken]
        truth = gt[broken]
        predicted_size = np.maximum(selected[:, 3:], 1e-6)
        gt_size = np.maximum(truth[:, 3:], 1e-6)
        lower_error = ((truth[:, :3] - gt_size * .5) - (selected[:, :3] - predicted_size * .5)) / gt_size
        upper_error = ((selected[:, :3] + predicted_size * .5) - (truth[:, :3] + gt_size * .5)) / gt_size
        outward = np.maximum(np.concatenate([lower_error, upper_error], -1), 0).max(-1)
        inward = np.maximum(-np.concatenate([lower_error, upper_error], -1), 0).max(-1)
        pairs[name] = dict(repaired=int(repaired.sum()), broken=int(broken.sum()),
            broken_with_fused_mask_iou_gt_050=int((broken & (fused_iou > .5)).sum()),
            broken_fused_mask_iou_median=float(np.median(fused_iou[broken])),
            broken_predicted_to_gt_volume_ratio_median=float(np.median(predicted_size.prod(-1) / gt_size.prod(-1))),
            broken_max_normalized_outward_face_error_median=float(np.median(outward)),
            broken_max_normalized_inward_face_error_median=float(np.median(inward)),
            gt_volume_quartiles=[dict(quartile=g+1, rows=int((groups == g).sum()),
                                     repaired=int((repaired & (groups == g)).sum()),
                                     broken=int((broken & (groups == g)).sum())) for g in range(4)])
    summary[before + '_to_' + after] = pairs
result = dict(status='ACTUAL_MCLN_REFERENCE_REGRESSION_ROWS_DIAGNOSED_CPU_ONLY', rows=9508,
              paired_rows_sha256=binding['sha256'], pairs=summary,
              qualification_uses_GT_offline_only=True, added_neural_forwards=0,
              inference_changed=False, GT_overlap_is_not_physical_identity_proof=True,
              cause_proven=False, full_goal_complete=False)
(root / 'REFERENCE_REGRESSION_DIAGNOSIS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
