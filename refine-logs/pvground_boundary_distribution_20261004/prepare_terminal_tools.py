"""Adapt the completed comparison's text collector and row analyzer, not its model."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
previous = local.parent / 'pvground_range_head_only_20261004'
collector = (previous / 'collect_terminal.py').read_text(encoding='utf-8')
collector = collector.replace("local/'whole_range_spec.json'", "local/'distribution_spec.json'")
collector = collector.replace('pvground_range_head_only_fit_20261004', 'pvground_boundary_fit_20261004')
collector = collector.replace("'local_range'", "'residual'").replace("'whole_range'", "'distribution'")
collector = collector.replace("source pair's", "boundary pair's")
collector_path = local / 'collect_terminal.py'
assert not collector_path.exists()
collector_path.write_text(collector, encoding='utf-8')
source = (previous / 'analyze_terminal.py').read_text(encoding='utf-8')
runner_sha = hashlib.sha256((local / 'run_boundary_fit.py').read_bytes()).hexdigest()
source = source.replace('closed local/whole source pair', 'closed residual/distribution boundary pair')
source = source.replace('run_range_head_only.py', 'run_boundary_fit.py')
source = source.replace('e4fdf9bddb90e1709f439a80e55cbe86136f5cd059fb76190a104df28d748b95', runner_sha)
source = source.replace('local_range', 'residual').replace('whole_range_hits', 'distribution_hits')
source = source.replace('whole_range_errors_with_good_full256_candidate', 'distribution_errors_with_good_full256_candidate')
source = source.replace("'whole_range'", "'distribution'")
source = source.replace("changed = {'root', 'support_arm', 'use_whole_range', 'preflight_root'}",
    "changed = {'root', 'preflight_root', 'boundary_mode', 'boundary_loss_weight', 'head_parameters', 'paired_control_root'}")
source = source.replace("assert spec['support_arm'] == arm and spec['use_whole_range'] == (arm == 'distribution')",
    "assert spec['support_arm'] == 'whole_range' and spec['use_whole_range']\n"
    "        assert spec['boundary_mode'] == arm\n"
    "        assert spec['head_parameters'] == (400614 if arm == 'residual' else 456102)\n"
    "        assert spec['boundary_loss_weight'] == (0.0 if arm == 'residual' else 1/7)")
source = source.replace("loaded['head_parameters'] == 400614", "loaded['head_parameters'] == spec['head_parameters']")
source = source.replace("fit['head_parameters'] == 400614", "fit['head_parameters'] == spec['head_parameters']")
source = source.replace("train_receipts, orders = {}, {}", "train_receipts, orders, boundary_training = {}, {}, {}")
needle = "        orders[arm] = [r['rows'] for r in records]"
insert = '''        assert all(math.isfinite(r['boundary_loss']) and math.isfinite(r['boundary_weighted_loss']) for r in records)
        assert all(math.isclose(r['boundary_weighted_loss'], r['boundary_loss'] * spec['boundary_loss_weight'], rel_tol=1e-6, abs_tol=1e-7) for r in records)
        boundary_training[arm] = dict(size_floor_axis_count_sum=sum(r['size_floor_axis_count'] for r in records),
            mean_boundary_loss=sum(r['boundary_loss'] for r in records)/len(records))
        if arm == 'distribution':
            assert all(r['boundary_targets_finite'] and r['boundary_faces'] == 6*r['boundary_matched_boxes'] for r in records)
            assert all(r['boundary_matched_boxes'] == r['matched_queries'] for r in records)
            boundary_training[arm].update(matched_faces=sum(r['boundary_faces'] for r in records),
                targets_outside=sum(r['boundary_target_outside'] for r in records),
                steps_with_outside_targets=sum(r['boundary_target_outside'] > 0 for r in records))
'''
assert source.count(needle) == 1
source = source.replace(needle, insert + needle)
begin = source.index('    limits = [')
end = source.index('    summary = ', begin)
source = source[:begin] + '''    limits = [
        'Original G is fixed in eval mode and each arm starts from its protected parent with fresh AdamW; all256 candidates remain.',
        'Both use identical1302-D whole/global plus local support. Residual400614 versus distribution456102 parameters, face decoding and DFL/7 change together: representation-plus-supervision adaptation, not an isolated DFL effect or completed face-conditioned token method.',
        'Both apply the common1e-6 reference/final size floor. It preserves zero-head agreement with the same numerical protocol; it is not exact preservation of untreated negative native dimensions.',
        'Actual cross-process initial Query, continuous box/Mask-IoU and threshold differences are recorded, not assumed bitwise equal.',
        'The6887 module holdout scenes were seen by author pretraining;9508 formal rows are development validation. Single seed2027 gives no multi-seed significance estimate.',
        'Boundary targets use every actual final native Hungarian correspondence and training GT; none enter the inference head. Out-of-range targets saturate at the declared endpoints and are counted.',
        'Final native box/GIoU losses supervise the deployed frame. The first experiment keeps native scoring unchanged; no geometry-quality callback, P2, contrastive expansion, V99 teacher or dual ranking.',
        'Same-Query coarse/final evidence comes from this frozen-G forward; between-arm repairs are not an independently trained reranking ablation.',
        'CPU reconstruction verifies selected boxes and thresholds. Raw Masks and full candidate boxes are not reconstructed; their recorded native scalar IoUs and coverage are only recounted.',
        'Distribution entropy is a model statistic, not calibrated localization quality. Face offsets describe this parameterization, not guaranteed true instance boundaries.',
        'Formal runner restored the terminal delta and parent identity before retention. This analyzer never replays model/optimizer tensors, downloads or deletes weights.',
        'No new Nr3D or Sr3D result exists. Three complete proposed contribution claims and the full target remain pending actual evidence.',
    ]
''' + source[end:]
source = source.replace('trainable_parameters=400614,',
    "trainable_parameters={arm: specs[arm]['head_parameters'] for arm in arms}, boundary_training=boundary_training,")
source = source.replace('# PV-Ground frozen-G local / whole complete range-source comparison',
    '# PV-Ground frozen-G residual / distribution boundary comparison')
source = source.replace('only the same 400614-parameter head learns. Native+G losses are computed, and native final-box losses train the head.',
    'only the declared boundary head learns. Both use whole-range support; distribution adds six-face distribution targets with a separate1/7 coefficient. Native+G final-box losses remain.')
source = source.replace('Local hits', 'Residual hits').replace('Whole hits', 'Distribution hits')
source = source.replace('Whole minus local', 'Distribution minus residual')
source = source.replace('Whole repairs vs local', 'Distribution repairs vs residual')
source = source.replace('Whole damages vs local', 'Distribution damages vs residual')
analyzer_path = local / 'analyze_terminal.py'
assert not analyzer_path.exists()
analyzer_path.write_text(source, encoding='utf-8')
for path in (collector_path, analyzer_path):
    ast.parse(path.read_text(encoding='utf-8'))
tree = ast.parse(collector)
remote = next(n for n in tree.body if isinstance(n, ast.Assign) and
    any(isinstance(target, ast.Name) and target.id == 'probe' for target in n.targets))
compile(ast.literal_eval(remote.value), 'collector remote', 'exec')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (collector_path, analyzer_path)},
    raw_parent_tools={name: hashlib.sha256((previous/name).read_bytes()).hexdigest()
        for name in ('collect_terminal.py', 'analyze_terminal.py')},
    execution_scope='PREPARATION_ONLY_LOCAL_AST', remote_calls=0, model_forwards=0, updates=0,
    analysis_executed=False, training_source_changed=False)
(local / 'terminal_tool_preparation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
