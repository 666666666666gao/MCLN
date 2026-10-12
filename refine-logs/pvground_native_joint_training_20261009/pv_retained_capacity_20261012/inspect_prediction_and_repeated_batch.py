"""CPU-only diagnostics of saved predictions; no neural replay or new targets."""
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
source = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit')
original = root / 'SAVED_TRAINING_WINNER_RESPONSIBILITY.json'
responsibility = json.loads(original.read_bytes())
receipts = [json.loads((source / ('preflight_support_step_' + str(step)) / 'receipt.json').read_bytes())
            for step in (1, 2)]
equal_fields = ('raw_points', 'superpoints', 'actual_gt_boxes', 'actual_gt_masks')
repeated = []
for first, second in zip(receipts[0]['rows'], receipts[1]['rows']):
    assert (first['row_id'], first['scene']) == (second['row_id'], second['scene'])
    paths = [source / ('preflight_support_step_' + str(step)) / Path(row['path']).name
             for step, row in zip((1, 2), (first, second))]
    for path, row in zip(paths, (first, second)):
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    with np.load(str(paths[0]), allow_pickle=False) as a, np.load(str(paths[1]), allow_pickle=False) as b:
        assert all(np.array_equal(a[key], b[key]) for key in equal_fields)
        assert hashlib.sha256(a['raw_points'].tobytes()).hexdigest() == first['raw_points_sha256']
        assert first['raw_points_sha256'] == second['raw_points_sha256']
    repeated.append(dict(row_id=first['row_id'], scan_id=first['scene'], input_and_GT_equal=True))

rows_path = source / 'formal/rows.jsonl'
rows_bytes = rows_path.read_bytes()
receipt = json.loads((source / 'formal/receipt.json').read_bytes())
assert hashlib.sha256(rows_bytes).hexdigest() == receipt['rows_sha256']
rows = [json.loads(line) for line in rows_bytes.decode('utf-8').splitlines()]
assert len(rows) == 9508
gate = np.asarray([row['arms']['extremal_support']['axis_gate'] for row in rows])
native_iou = np.asarray([row['controls']['native']['iou'] for row in rows])
reference_iou = np.asarray([row['controls']['mask']['iou'] for row in rows])
final_iou = np.asarray([row['arms']['extremal_support']['iou'] for row in rows])
quantiles = [0, .01, .1, .5, .9, .99, 1]
groups = []
for threshold in (.25, .5):
    groups.append(dict(threshold=threshold,
        native_qualified_reference_unqualified=int(((native_iou > threshold) & (reference_iou <= threshold)).sum()),
        native_qualified_reference_unqualified_final_unqualified=int(((native_iou > threshold) & (reference_iou <= threshold) & (final_iou <= threshold)).sum()),
        reference_qualified_native_unqualified=int(((reference_iou > threshold) & (native_iou <= threshold)).sum()),
        reference_qualified_native_unqualified_final_unqualified=int(((reference_iou > threshold) & (native_iou <= threshold) & (final_iou <= threshold)).sum()),
        final_repairs_reference=int(((reference_iou <= threshold) & (final_iou > threshold)).sum()),
        final_breaks_reference=int(((reference_iou > threshold) & (final_iou <= threshold)).sum())))

result = dict(status='CPU_SAVED_PV_PREDICTION_AND_REPEATED_BATCH_DIAGNOSTIC',
    formal_rows=9508, source_rows_sha256=hashlib.sha256(rows_bytes).hexdigest(),
    original_responsibility_path=str(original), original_responsibility_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    training_observations=16, unique_training_rows=len(repeated),
    same_augmented_batch_reused_for_two_updates=True, input_and_GT_equal_fields=list(equal_fields),
    repeated_training_rows=repeated, original_observation_summary=responsibility['summary'],
    correction='The original 16 row records are two observations of the same eight augmented inputs, not two independent training batches.',
    axis_names=['x', 'y', 'z'], quantile_positions=quantiles,
    actual_axis_gate_quantiles=np.quantile(gate, quantiles, axis=0).tolist(),
    actual_axis_gate_std=gate.std(0).tolist(),
    gate_axes_at_zero=int((gate == 0).sum()), gate_axes_at_one=int((gate == 1).sum()),
    threshold_groups=groups,
    new_neural_forwards=0, new_optimizer_updates=0, remote_queries=0, active_training_changes=0,
    limitations=[
        'Actual predicted gate range is not the representational range, an optimization cause, or a learnability guarantee.',
        'Endpoint qualification uses validation GT for diagnosis only; it supplies no inference rule or training data.',
        'Five qualified unmatched selected-query observations are not five independent training expressions; no full-dataset prevalence claim.',
        'The saved preflight uses frozen-parent matching, not the later native MCLN training assignments.',
        'This source cache predates the later native E0; aggregate count equality alone does not establish rowwise equivalence.'
    ])
(root / 'PREDICTION_AND_REPEATED_BATCH_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ('training_observations', 'unique_training_rows', 'actual_axis_gate_quantiles', 'threshold_groups')}))
