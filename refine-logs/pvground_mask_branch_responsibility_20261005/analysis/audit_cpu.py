"""Independent saved-evidence audit; NumPy/stdlib only, no model or remote call."""
import ast
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
complete = root / 'complete'
prior = root.parent / 'pvground_mask_geometry_responsibility_20261005'


def read_json(path):
    return json.loads(path.read_bytes())


def arrays_at(directory):
    parts = []
    for batch in range(8):
        with np.load(directory / ('batch_%02d.npz' % batch), allow_pickle=False) as data:
            parts.append({name: data[name] for name in data.files})
    assert all(set(part) == set(parts[0]) for part in parts)
    return {name: np.concatenate([part[name] for part in parts], axis=0)
            for name in parts[0]}


intake = read_json(complete / 'INTAKE.json')
for entry in intake['files']:
    raw = (complete / entry['name']).read_bytes()
    assert len(raw) == entry['bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
assert intake['total_bytes'] == sum(entry['bytes'] for entry in intake['files'])
assert {entry['name'] for entry in intake['files']} == {
    path.name for path in complete.iterdir() if path.name != 'INTAKE.json'}
assert intake['weights_copied'] == 0 and not intake['model_replayed']

receipt = read_json(complete / 'receipt.json')
status = read_json(complete / 'status.json')
wait = read_json(root / 'wait.json')
summary = read_json(root / 'analysis/SUMMARY.json')
assert status['status'] == 'complete' and status['exit_code'] == 0
assert status == wait['status'] and wait['observer_closed'] and not wait['controller_alive']
assert wait['exitcode'] == 0 and status['protected_parent_hashes_exact']
assert (complete / 'controller.exit').read_text().strip() == '0'
assert (complete / 'probe.exit').read_text().strip() == '0'
assert receipt['status'] == 'pass' and receipt['rows'] == 64 and receipt['batches'] == 8
assert receipt['optimizer_steps'] == receipt['weight_files_created'] == 0
assert not receipt['optimizer_constructed'] and not receipt['accuracy_result']
assert receipt['model_state_unchanged'] and receipt['model_gradients_absent']
for key, name in [('rows_sha256', 'rows.jsonl'), ('runner_sha256', 'run_mask_branch_probe.py'),
                  ('spec_sha256', 'spec.json')]:
    assert receipt[key] == hashlib.sha256((complete / name).read_bytes()).hexdigest()
for name in ['run_mask_branch_probe.py', 'spec.json', 'controller.py', 'EXPERIMENT_PLAN.md']:
    assert (root / name).read_bytes() == (complete / name).read_bytes()

rows = [json.loads(line) for line in (complete / 'rows.jsonl').read_text().splitlines()]
old_rows = [json.loads(line) for line in (prior / 'complete/rows.jsonl').read_text().splitlines()]
current = arrays_at(complete)
old = arrays_at(prior / 'complete')
assert len(rows) == len(old_rows) == len(set(row['row_id'] for row in rows)) == 64
assert current['box_iou'].shape == (64, 256)
assert current['boxes'].shape == (64, 256, 6)
assert current['root_box'].shape == (64, 6)
assert current['row_id'].tolist() == [row['row_id'] for row in rows]
assert np.array_equal(current['row_id'], old['row_id'])
assert np.array_equal(current['root_box'], old['root_box'])
for row, old_row in zip(rows, old_rows):
    for key in ['row_id', 'scan_id', 'point_sha256', 'root_box']:
        assert row[key] == old_row[key]
assert len({row['scan_id'] for row in rows}) == len({row['scan_id'].split('_')[0] for row in rows}) == 60
assert all(np.isfinite(array).all() for array in current.values())
assert np.all(current['boxes'][..., 3:] > 0) and np.all(current['root_box'][:, 3:] > 0)
qualified = {}
ratio_error = {}
for branch in ['text', 'query', 'fused']:
    intersection = current[branch + '_intersection']
    union = current[branch + '_union']
    assert intersection.dtype == union.dtype == np.dtype('int64')
    assert intersection.shape == union.shape == (64, 256)
    assert np.all((intersection >= 0) & (intersection <= union) & (union > 0) & (union <= 50000))
    ratio_error[branch] = float(np.max(np.abs(intersection / union - current[branch + '_iou'])))
    assert ratio_error[branch] < 1e-7
    qualified[branch] = 2 * intersection > union
    assert np.array_equal(qualified[branch], current[branch + '_iou'] > .5)
    if branch == 'text':
        assert np.all(intersection == intersection[:, :1]) and np.all(union == union[:, :1])

boxes = current['boxes'].astype(np.float64)
truth = current['root_box'].astype(np.float64)[:, None, :]
extent = np.maximum(0, np.minimum(boxes[..., :3] + boxes[..., 3:] / 2, truth[..., :3] + truth[..., 3:] / 2)
                    - np.maximum(boxes[..., :3] - boxes[..., 3:] / 2, truth[..., :3] - truth[..., 3:] / 2))
intersection = extent.prod(axis=-1)
box_iou = intersection / (boxes[..., 3:].prod(axis=-1) + truth[..., 3:].prod(axis=-1) - intersection)
box_error = float(np.max(np.abs(box_iou - current['box_iou'])))
assert np.array_equal(box_iou > .5, current['box_iou'] > .5)

unmatched = current['matched_slot'] < 0
boxpoor = box_iou <= .5
pool = unmatched & boxpoor & qualified['fused']
own = pool & qualified['query']
text_only_qualification = pool & ~qualified['query'] & qualified['text']
neither = pool & ~qualified['query'] & ~qualified['text']
query_all = unmatched & boxpoor & qualified['query']
counts = dict(fused_mask_only_unmatched=int(pool.sum()), own_query_confirmed=int(own.sum()),
              query_not_qualified_text_qualified=int(text_only_qualification.sum()),
              neither_branch_qualified=int(neither.sum()), own_query_mask_only_unmatched=int(query_all.sum()),
              rows_with_own_query_confirmed=int(own.any(axis=1).sum()))
assert counts == receipt['totals'] == summary['counts']
assert counts['fused_mask_only_unmatched'] == counts['own_query_confirmed'] + counts['query_not_qualified_text_qualified'] + counts['neither_branch_qualified']
selection = current['bbs'].argmax(axis=1)
assert np.all((current['bbs'] == current['bbs'].max(axis=1, keepdims=True)).sum(axis=1) == 1)
for index, row in enumerate(rows):
    assert row['batch_index'] == index // 8 and row['candidates'] == 256
    assert row['root_box'] == current['root_box'][index].tolist()
    assert row['valid_native_GT_slots'] == [0]
    selected = int(selection[index])
    assert selected == row['selected_query']
    assert row['selected_box_iou'] == float(current['box_iou'][index, selected])
    assert row['alpha'] == float(current['alpha'][index]) and 0 <= row['alpha'] <= 1
    for branch in qualified:
        assert row['selected_branch_ious'][branch] == float(current[branch + '_iou'][index, selected])
    expected_slots = np.full(256, -1, dtype=np.int16)
    expected_slots[row['native_matched_queries']] = row['native_matched_slots']
    assert np.array_equal(expected_slots, current['matched_slot'][index])
    row_counts = dict(fused_mask_only_unmatched=int(pool[index].sum()), own_query_confirmed=int(own[index].sum()),
                      query_not_qualified_text_qualified=int(text_only_qualification[index].sum()),
                      neither_branch_qualified=int(neither[index].sum()), own_query_mask_only_unmatched=int(query_all[index].sum()))
    assert row_counts == row['counts']

differences = dict(prior_matched_slot_differences=int(np.count_nonzero(old['matched_slot'] != current['matched_slot'])),
                   prior_box_threshold_differences=int(np.count_nonzero((old['box_iou'] > .5) != (current['box_iou'] > .5))),
                   prior_fused_threshold_differences=int(np.count_nonzero((old['mask_iou'] > .5) != qualified['fused'])),
                   prior_max_box_iou_difference=float(np.max(np.abs(old['box_iou'] - current['box_iou']))),
                   prior_max_fused_iou_difference=float(np.max(np.abs(old['mask_iou'] - current['fused_iou']))))
assert all(summary[key] == value for key, value in differences.items())
assert summary['rows_text_mask_qualified'] == int(qualified['text'][:, 0].sum())
assert summary['own_query_fraction_of_fused_mask_only_unmatched'] == counts['own_query_confirmed'] / counts['fused_mask_only_unmatched']
assert summary['alpha_min'] == float(current['alpha'].min()) and summary['alpha_max'] == float(current['alpha'].max())

log_records = [json.loads(line.split(' ', 1)[1]) for line in (complete / 'probe.log').read_text().splitlines()
               if line.startswith('MASK_BRANCH_BATCH ')]
assert [item['index'] for item in log_records] == list(range(8))
running = Counter()
for batch_index, record in enumerate(log_records):
    for row in rows[batch_index * 8:(batch_index + 1) * 8]:
        running.update(row['counts'])
        running['rows_with_own_query_confirmed'] += row['counts']['own_query_confirmed'] > 0
    assert dict(running) == record['totals']

parsed = []
for name in ['run_mask_branch_probe.py', 'analyze_probe.py', 'controller.py', 'collect_probe_authorized.py',
             'observe_probe_authorized.py', 'launch_probe_authorized.py', 'prepare_probe.py', 'publish_branch.py']:
    ast.parse((root / name).read_text(encoding='utf-8'), feature_version=(3, 7))
    parsed.append(name)
setup = (prior / 'run_mask_geometry_probe.py').read_text(encoding='utf-8').split('    # Fixed first eight fit batches')[0]
setup = setup.replace('"""Frozen4506 R fit with training-only native final-IoU differences.\n\nImplementation draft; source review and two-step preflight precede launch.\n"""',
                      '"""Read-only candidate Query/Text/fused Mask qualification on augmentedfit64."""')
generated = setup + (root / 'probe_body.py').read_text(encoding='utf-8') + "\n\nif __name__ == '__main__':\n    main()\n"
assert generated == (root / 'run_mask_branch_probe.py').read_text(encoding='utf-8')
runner_tree = ast.parse(generated)
call_names = [ast.unparse(node.func) for node in ast.walk(runner_tree) if isinstance(node, ast.Call)]
assert not any(name.startswith('torch.optim') or name in ['torch.save', 'torch.autograd.grad'] or name.endswith('.backward') for name in call_names)

result = dict(status='PASS', generated_at=datetime.datetime.now().astimezone().isoformat(), numpy_version=np.__version__,
              intake_files_verified=len(intake['files']), intake_bytes_verified=intake['total_bytes'],
              rows=64, unique_scan_ids=60, unique_physical_scene_prefixes=60, candidates=16384,
              all_target_slots_root_only=True, counts=counts, integer_ratio_max_error=ratio_error,
              independent_float64_box_iou_max_error=box_error, independent_box_threshold_differences=0,
              **differences, prior_selected_query_differences=sum(int(row['selected_query'] != old_row['selected_query']) for row, old_row in zip(rows, old_rows)),
              current_selected_top_ties=0, row_json_and_batch_logs_exact=True, generation_recipe_exact=True,
              python37_parsed=parsed, model_forwards=0, gpu_runs=0, optimizer_steps=0, remote_calls=0,
              raw_point_mask_cpu_replay=False,
              limitation='CPU recomputes all saved integer intersections/unions and boxes. Raw point masks are not archived; GPU selected-point expansion equality is a reviewed runtime assertion, not a CPU replay.')
(root / 'analysis/AUDIT_CPU.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
