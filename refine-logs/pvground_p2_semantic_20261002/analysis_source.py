"""Recount semantic-only P2 against the sealed G and joint-P2 results."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

previous = Path(r'C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete')
analyzer_spec = importlib.util.spec_from_file_location('sealed_pair_analysis', previous / 'source/analyze.py')
sealed = importlib.util.module_from_spec(analyzer_spec)
analyzer_spec.loader.exec_module(sealed)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists(), 'Do not overwrite an existing analysis.'
intake = sealed.read_json(args.root / 'INTAKE.json')
assert intake['status'] == 'pass'
status = sealed.read_json(args.root / 'status.json')
assert status['status'] == 'complete'
assert [item['stage'] for item in status['completed']] == [
    'preflight/preflight', 'semantic/train', 'semantic/formal']
method_root = args.root / 'semantic'
method_spec = sealed.read_json(method_root / 'spec.json')
method_train = sealed.read_json(method_root / 'receipt.json')
control_spec = sealed.read_json(previous / 'g_control/spec.json')
control_train = sealed.read_json(previous / 'g_control/receipt.json')
assert method_spec['p2_routing'] == 'semantic_only'
for key in ('base_terminal_sha256', 'seed', 'batch_size', 'lr', 'lr_backbone',
            'fit_passes', 'primary_mode', 'assignment_threshold',
            'source_query_module_sha256', 'observation_module_sha256',
            'task_module_sha256', 'assignment_module_sha256'):
    assert method_spec[key] == control_spec[key], key
for receipt in (method_train, control_train):
    assert receipt['status'] == 'complete' and receipt['training_steps'] == 3723
    assert receipt['fit_rows'] == 29778 and receipt['holdout_rows'] == 6887
    assert receipt['fit_seen_exactly_once'] and receipt['fresh_optimizer']
    assert receipt['frozen_parameters_unchanged']
assert method_train['initial_rec_inputs_match_control']
assert method_train['fit_batch_row_order_matches_control']
assert method_train['terminal_sha256'] == intake['files']['terminal_checkpoint']['sha256']
for relative, expected in intake['files'].items():
    if relative != 'terminal_checkpoint':
        raw = (args.root / relative).read_bytes()
        assert len(raw) == expected['bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], relative

def updates(directory):
    rows = [json.loads(line) for line in (directory / 'train.jsonl').read_bytes().splitlines()]
    assert len(rows) == 3723 and [row['step'] for row in rows] == list(range(1, 3724))
    consumed = [item for row in rows for item in row['rows']]
    assert len(consumed) == 29778 and len(set(consumed)) == 29778
    return [row['rows'] for row in rows]

assert updates(method_root) == updates(previous / 'g_control')

def iou_groups(control, method):
    matrix = [[0] * 3 for _ in range(3)]
    def group(value):
        return 0 if value > .5 else 1 if value > .25 else 2
    for before, after in zip(control, method):
        matrix[group(before['bbs']['iou'])][group(after['bbs']['iou'])] += 1
    assert sum(sum(row) for row in matrix) == len(control)
    return {'labels': ['strict_gt050', 'loose_only_025_to050', 'miss_le025'],
            'control_rows_method_columns': matrix,
            'limit': 'IoU endpoint groups do not establish physical instance identity or a causal loss mechanism.'}

phases = {}
for stage, count in (('initial', 6887), ('terminal', 6887), ('formal', 9508)):
    control_rows, control_metrics = sealed.read_evaluation(previous / 'g_control' / stage, stage, count)
    joint_rows, joint_metrics = sealed.read_evaluation(previous / 'g_p2' / stage, stage, count)
    method_rows, method_metrics = sealed.read_evaluation(method_root / stage, stage, count)
    if stage == 'initial':
        for before, after in zip(control_rows, method_rows):
            for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
                assert before[key] == after[key], (key, before['row_id'])
            for key in ('query', 'box', 'iou'):
                assert before['bbs'][key] == after['bbs'][key], (key, before['row_id'])
    phases[stage] = {'rows': count, 'g_control': control_metrics,
                     'joint_p2': joint_metrics, 'semantic_p2': method_metrics,
                     'vs_g_control': {mode: sealed.compare_rows(control_rows, method_rows, mode)
                                      for mode in ('bbs', 'bbf')},
                     'vs_joint_p2': {mode: sealed.compare_rows(joint_rows, method_rows, mode)
                                     for mode in ('bbs', 'bbf')},
                     'bbs_iou_groups_vs_g': iou_groups(control_rows, method_rows),
                     'bbs_iou_groups_vs_joint_p2': iou_groups(joint_rows, method_rows)}
formal = phases['formal']
hits = formal['semantic_p2']['bbs']
delta = formal['vs_g_control']['bbs']
summary = {'primary_mode': 'bbs', 'finished_cst': status['finished_cst'],
           'phases': phases, 'historical_g_formal_hits': [5615, 4495],
           'scanrefer_target_hits_preserving_historical_g': [5615, 4754],
           'preserves_historical_g_both_thresholds': hits['rec_hits25'] >= 5615 and hits['rec_hits50'] >= 4495,
           'target_pass': hits['rec_hits25'] >= 5615 and hits['rec_hits50'] >= 4754,
           'strict_increment_preserving_loose_vs_control': delta['25']['net'] >= 0 and delta['50']['net'] > 0,
           'limits': [
               'All three trained arms start from the original G endpoint with fresh AdamW.',
               'The sealed G/joint-P2 arms are reused; no G control was retrained.',
               'Both boxes and scores can change across trained arms; this is not a fixed-box comparison.',
               'The semantic-only routing removes direct P2 geometry residual; shared/upstream parameters still train.',
               'Initial REC identity is checked, while historical initial Mask differences remain unexplained.',
               'Module holdout scenes were seen by author pretraining; formal9508 is development validation.',
               'Candidate oracle uses GT for offline diagnosis, and bbf stays secondary.',
               'One seed does not establish cross-seed significance or Nr3D/Sr3D effectiveness.']}
lines = ['# PV-Ground semantic-only P2 continuation', '',
         'Same original G start, fresh AdamW, seed2027, batch8, LR1e-5, 3723 updates/29778 fit rows once.', '',
         '| Set / mode | G control | Joint P2 | Semantic-only P2 | Delta vs G |',
         '|---|---:|---:|---:|---:|']
for stage, phase in phases.items():
    for mode in ('bbs', 'bbf'):
        g, j, m = [phase[name][mode] for name in ('g_control', 'joint_p2', 'semantic_p2')]
        d = phase['vs_g_control'][mode]
        lines.append('| {} / {} | {} / {} | {} / {} | {} / {} | {:+d} / {:+d} |'.format(
            stage, mode, g['rec_hits25'], g['rec_hits50'], j['rec_hits25'], j['rec_hits50'],
            m['rec_hits25'], m['rec_hits50'], d['25']['net'], d['50']['net']))
lines += ['', '| Formal bbs comparison | Threshold | Repairs | Damages | Net |',
          '|---|---|---:|---:|---:|']
for comparison in ('vs_g_control', 'vs_joint_p2'):
    for threshold in ('25', '50'):
        paired = formal[comparison]['bbs'][threshold]
        lines.append('| {} | @{} | {} | {} | {:+d} |'.format(
            comparison, threshold, paired['repairs'], paired['damages'], paired['net']))
lines += ['', '| Formal bbs model | Full-256 @0.25 / @0.50 |', '|---|---:|']
for model in ('g_control', 'joint_p2', 'semantic_p2'):
    metric = formal[model]['bbs']
    lines.append('| {} | {} / {} |'.format(model, metric['oracle_hits25'][-1], metric['oracle_hits50'][-1]))
for key in ('bbs_iou_groups_vs_g', 'bbs_iou_groups_vs_joint_p2'):
    grouped = formal[key]
    lines += ['', key + ': reference rows, semantic-only P2 columns.', '',
              '| Reference IoU | >0.50 | (0.25, 0.50] | <=0.25 |', '|---|---:|---:|---:|']
    for label, values in zip(grouped['labels'], grouped['control_rows_method_columns']):
        lines.append('| {} | {} | {} | {} |'.format(label, *values))
    lines += ['', grouped['limit']]
lines += ['', 'Historical G remains5615/4495; development target preserves5615 loose hits and requires4754 strict hits.',
          'Target passed: {}. Same-budget strict increment preserving loose: {}.'.format(
              summary['target_pass'], summary['strict_increment_preserving_loose_vs_control']), '',
          'Interpretation limits:', ''] + ['- ' + value for value in summary['limits']]
args.output.mkdir()
(args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
(args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'formal_bbs_hits': [hits['rec_hits25'], hits['rec_hits50']],
                  'delta_vs_g_control': [delta['25']['net'], delta['50']['net']],
                  'target_pass': summary['target_pass']}))
