"""Read the two actual saved training batches; do not replay or change training."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
source = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit')
records = []
bindings = []
for step in (1, 2):
    directory = source / ('preflight_support_step_' + str(step))
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    assert receipt['step'] == step and len(receipt['rows']) == 8
    for witness in receipt['rows']:
        path = directory / Path(witness['path']).name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == witness['bytes'] and digest == witness['sha256']
        bindings.append(dict(path=str(path), bytes=len(raw), sha256=digest))
        with np.load(str(path), allow_pickle=False) as a:
            assert len(a['actual_gt_boxes']) == len(a['actual_gt_masks']) == 1
            q = witness['selected_query']
            observed = np.flatnonzero(a['observed_queries'] == q)
            assert observed.size == 1 and a['scores'][q] == a['scores'].max()
            own = a['observed_query_logits'][int(observed[0])]
            text = a['observed_text_logits'][int(observed[0])]
            mapping = a['superpoints']
            gt = a['actual_gt_masks'][0] > .5
            values = {}
            for name, foreground in (('own', own[mapping] > 0),
                                     ('fused', (a['alpha'] * text[mapping] + (1 - a['alpha']) * own[mapping]) > 0)):
                values[name] = float(np.logical_and(foreground, gt).sum()) / int(np.logical_or(foreground, gt).sum())
            matched = bool((a['matched_queries'] == q).any())
            records.append(dict(step=step, training_row_id=witness['row_id'], scan_id=witness['scene'],
                                selected_query=q, selected_query_is_matched=matched,
                                matched_queries=a['matched_queries'].tolist(),
                                selected_own_mask_iou=values['own'], selected_fused_mask_iou=values['fused'],
                                own_and_fused_masks_qualified=values['own'] > .5 and values['fused'] > .5,
                                GT_count=1))
summary = dict(rows=16, batches=2,
              selected_query_unmatched=sum(not row['selected_query_is_matched'] for row in records),
              own_and_fused_qualified=sum(row['own_and_fused_masks_qualified'] for row in records),
              selected_query_unmatched_own_and_fused_qualified=sum(
                  not row['selected_query_is_matched'] and row['own_and_fused_masks_qualified'] for row in records))
result = dict(status='CPU_READ_OF_SAVED_ACTUAL_TWO_TRAINING_BATCHES', summary=summary,
              records=records, input_bindings=bindings, new_neural_forwards=0, new_optimizer_updates=0,
              remote_queries=0, active_training_changes=0,
              limitations=['Two saved augmented training batches, not a full training or validation prevalence estimate.',
                           'These are preflight frozen-parent assignments; they are not the later native MCLN matching state.',
                           'Mask qualification is GT overlap, not independent full-language identity proof.',
                           'No new backward pass or training intervention is performed.'])
(root / 'SAVED_TRAINING_WINNER_RESPONSIBILITY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary))
