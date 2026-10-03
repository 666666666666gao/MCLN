"""Verify the completed control's real rows; do not claim formal accuracy."""
import datetime
import hashlib
import json
import math
from pathlib import Path

from analyze_complete import read_evaluation


local = Path(__file__).parent
root = local/'control_complete'
assert not (root/'ANALYSIS.json').exists()
intake = json.loads((root/'INTAKE.json').read_bytes())
receipt = json.loads((root/'receipt.json').read_bytes())
spec = json.loads((root/'spec.json').read_bytes())
assert receipt['training_steps'] == 3723 and receipt['fit_rows'] == 29778
assert receipt['holdout_rows'] == 6887 and receipt['formal_rows'] == 0
assert receipt['semantic_consistency'] is spec['semantic_consistency'] is False
assert receipt['fresh_optimizer'] and receipt['fit_seen_exactly_once']
assert receipt['frozen_parameters_unchanged']
assert receipt['base_terminal_sha256'] == spec['base_terminal_sha256']
assert receipt['terminal_sha256'] == intake['checkpoint']['sha256']
assert receipt['script_sha256'] == hashlib.sha256((local/'run.py').read_bytes()).hexdigest()
for name, expected in intake['files'].items():
    raw = (root/name).read_bytes()
    assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
raw = (root/'train.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['train_log_sha256']
records = [json.loads(line) for line in raw.splitlines()]
assert [row['step'] for row in records] == list(range(1, 3724))
assert all(row['total_steps'] == 3723 and not row['semantic_consistency'] for row in records)
assert all(row['loss_contrastive_correction'] == 0 for row in records)
assert all(math.isfinite(value) for row in records for key, value in row.items()
           if key.startswith('loss') or key == 'grad_norm')
seen = [identity for row in records for identity in row['rows']]
assert len(seen) == len(set(seen)) == 29778
initial_rows, initial_metrics = read_evaluation(root/'initial', 'initial', 6887)
final_rows, final_metrics = read_evaluation(root/'terminal', 'terminal', 6887)
assert not set(seen).intersection(row['row_id'] for row in initial_rows)
assert initial_metrics['bbs']['rec_hits25'] == 6176 and initial_metrics['bbs']['rec_hits50'] == 5602
for before, after in zip(initial_rows, final_rows):
    for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
        assert before[key] == after[key]
transitions = {}
for mode in ('bbs', 'bbf'):
    transitions[mode] = {}
    for threshold in (.25, .5):
        repairs = sum(a[mode]['iou'] <= threshold < b[mode]['iou'] for a, b in zip(initial_rows, final_rows))
        damages = sum(b[mode]['iou'] <= threshold < a[mode]['iou'] for a, b in zip(initial_rows, final_rows))
        recomputed = dict(fixes=repairs, breaks=damages, net=repairs-damages)
        assert recomputed == receipt['transitions'][mode][str(threshold)]
        transitions[mode][str(threshold)] = recomputed
result = dict(status='control_native_rows_verified',
              time_cst=datetime.datetime.now().astimezone().isoformat(),
              training_steps=3723, fit_rows=29778, module_holdout_rows=6887,
              all_step_records_finite=True, unique_fit_rows_verified=True,
              fit_holdout_disjoint=True, final_and_initial_inputs_equal=True,
              initial=initial_metrics, terminal=final_metrics, transitions=transitions,
              primary_mode='bbs', formal9508_evaluated=False,
              checkpoint=intake['checkpoint'], paired_method_evaluated=False,
              optimizer_updates_executed=0, GPU_forward_executed=False,
              interpretation='Seen-scene module holdout only; no current paired/formal accuracy or method gain.',
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'ANALYSIS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ['status','training_steps','module_holdout_rows','transitions','formal9508_evaluated']}, indent=2))
