"""Prepare an isolated evaluator that reports the same bbs Query's box and mask."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prior = root / 'referit_author_entry_20261010'
destination = root / 'referit_same_query_20261010'
assert not destination.exists()
warm = json.loads((prior / 'WARM_SOURCE_HASHES.json').read_bytes())
author_evaluator = root.parent / 'runtime_binding/model_source/src/grounding_evaluator.py'
assert hashlib.sha256(author_evaluator.read_bytes()).hexdigest() == warm['src/grounding_evaluator.py']
for path in (prior / 'source').rglob('*.py'):
    target = destination / 'source' / path.relative_to(prior / 'source')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
for dataset in ('nr3d', 'sr3d'):
    target = destination / dataset / 'init.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((prior / dataset / 'init.json').read_bytes())
old = author_evaluator.read_text(encoding='utf-8')
start = old.index('    def evaluate_bbox_by_pos_align(')
end = old.index('    def evaluate_bbox_by_sem_align(', start)
body = old[start:end]
loop = '        # Highest scoring box -> iou\n        for bid in range(len(positive_map)):'
assert body.count(loop) == 1
body = body.replace(loop,
    '        end_points[prefix + "bbs_selected_query"] = torch.empty(\n'
    '            len(positive_map), dtype=torch.long, device=sem_scores.device)\n\n' + loop)
rank = '            top = scores.argsort(1, True)[:, :10]\n'
assert body.count(rank) == 1
body = body.replace(rank, rank +
    '            end_points[prefix + "bbs_selected_query"][bid] = top[0, 0]\n')
new = old[:start] + body + old[end:]
for before in ('            top = scores.argsort(1, True)[:, :1]  # top-1 mask',
               '            top = scores.argsort(1, True)[:, :1]\n'):
    assert new.count(before) == 1
    after = '            top = end_points[prefix + "bbs_selected_query"][bid].reshape(1, 1)'
    if before.endswith('\n'):
        after += '\n'
    new = new.replace(before, after)
ast.parse(new, feature_version=(3, 7))
target = destination / 'source/src/grounding_evaluator.py'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(new, encoding='utf-8')
warm['src/grounding_evaluator.py'] = hashlib.sha256(target.read_bytes()).hexdigest()
(destination / 'WARM_SOURCE_HASHES.json').write_text(json.dumps(warm, indent=2) + '\n')
(destination / 'EVALUATOR_CHANGE.diff').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), fromfile='author/src/grounding_evaluator.py',
    tofile='same_query/src/grounding_evaluator.py')), encoding='utf-8')
receipt = dict(status='AUTHOR_BBS_SAME_QUERY_MASK_REPORTING_SOURCE_PREPARED',
    prepared_from=str(prior), evaluator_original=str(author_evaluator),
    evaluator_original_sha256=hashlib.sha256(author_evaluator.read_bytes()).hexdigest(),
    evaluator_prepared_sha256=warm['src/grounding_evaluator.py'],
    changes_relative_to_author_entry=['src/grounding_evaluator.py'],
    original_bbs_formula_and_zero_score_filter_preserved=True,
    bbf_box_diagnostic_unchanged=True, model_and_losses_unchanged=True,
    mask_counter_meaning='Mask of the one Query actually selected by last/bbs',
    init_specs_unchanged=True, source_deployed=False, CPU_check_completed=False,
    real_data_evaluation_completed=False, GPU_training_admission=False,
    current_training_queries=0, active_ScanRefer_source_changed=False,
    formal_accuracy=None, final_method_selection_pending=True, full_goal_complete=False)
(destination / 'SOURCE_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status=receipt['status'], changed=receipt['changes_relative_to_author_entry'],
    source_deployed=False, formal_accuracy=None)))
