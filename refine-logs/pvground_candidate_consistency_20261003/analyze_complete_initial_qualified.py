"""Compare the completed G/consistent pair from its native, GT-evaluated row records."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(path.read_bytes())


def read_evaluation(directory, stage, count):
    receipt = read_json(directory/'receipt.json')
    assert receipt['status'] == 'pass' and receipt['stage'] == stage
    assert receipt['rows'] == count
    assert receipt['formal_rows'] == (9508 if stage == 'formal' else 0)
    source = directory/'rows.jsonl'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == count and len({row['row_id'] for row in rows}) == count
    metrics = {}
    for mode in ('bbs', 'bbf'):
        values = [row[mode] for row in rows]
        for value in values:
            assert math.isfinite(value['iou']) and 0 <= value['iou'] <= 1
            assert math.isfinite(value['mask_iou']) and 0 <= value['mask_iou'] <= 1
            for key, threshold in (('oracle25', .25), ('oracle50', .5)):
                oracle = value[key]
                assert len(oracle) == 4 and all(x in (0, 1) for x in oracle)
                assert oracle == sorted(oracle)
                assert oracle[-1] >= (value['iou'] > threshold)
        observed = receipt['metrics'][mode]
        recomputed = {
            'rec_hits25': sum(value['iou'] > .25 for value in values),
            'rec_hits50': sum(value['iou'] > .5 for value in values),
            'mask_hits25': sum(value['mask_iou'] > .25 for value in values),
            'mask_hits50': sum(value['mask_iou'] > .5 for value in values),
            'mask_iou_sum': sum(value['mask_iou'] for value in values),
        }
        for key in ('rec_hits25', 'rec_hits50', 'mask_hits25', 'mask_hits50'):
            assert recomputed[key] == observed[key]
        assert abs(recomputed['mask_iou_sum']-observed['mask_iou_sum']) < 1e-6
        recomputed['mask_miou'] = recomputed['mask_iou_sum']/count*100
        assert abs(recomputed['mask_miou']-observed['mask_miou']) < 1e-6
        for label, key in (('25', 'oracle25'), ('50', 'oracle50')):
            recomputed['oracle_hits'+label] = [sum(value[key][i] for value in values) for i in range(4)]
        metrics[mode] = recomputed
    return rows, metrics


def compare_rows(control, method, mode):
    assert len(control) == len(method)
    for before, after in zip(control, method):
        for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
            assert before[key] == after[key], (key, before['row_id'])
    result = {}
    for label, threshold, oracle_key in (('25', .25, 'oracle25'), ('50', .5, 'oracle50')):
        pairs = [(a[mode], b[mode]) for a, b in zip(control, method)]
        repairs = sum(a['iou'] <= threshold < b['iou'] for a, b in pairs)
        damages = sum(b['iou'] <= threshold < a['iou'] for a, b in pairs)
        control_hits = sum(a['iou'] > threshold for a, _ in pairs)
        method_hits = sum(b['iou'] > threshold for _, b in pairs)
        assert method_hits-control_hits == repairs-damages
        result[label] = {
            'control_hits': control_hits, 'consistent_hits': method_hits,
            'repairs': repairs, 'damages': damages, 'net': repairs-damages,
            'control_errors_with_good_full256_candidate': sum(a['iou'] <= threshold and a[oracle_key][-1] for a, _ in pairs),
            'consistent_errors_with_good_full256_candidate': sum(b['iou'] <= threshold and b[oracle_key][-1] for _, b in pairs),
            'repairs_from_control_coverable_errors': sum(a['iou'] <= threshold < b['iou'] and a[oracle_key][-1] for a, b in pairs),
            'repairs_from_control_missing_candidate_errors': sum(a['iou'] <= threshold < b['iou'] and not a[oracle_key][-1] for a, b in pairs),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status = read_json(args.root/'pair_status.json')
    assert status['status'] == 'complete', 'paired run has not completed'
    expected = [('g_control', 'train'), ('g_consistent', 'train'), ('g_control', 'formal'), ('g_consistent', 'formal')]
    assert [(row['arm'], row['mode']) for row in status['completed']] == expected
    specs = {arm: read_json(args.root/arm/'spec.json') for arm in ('g_control', 'g_consistent')}
    assert not specs['g_control']['p2'] and not specs['g_consistent']['p2']
    assert not specs['g_control']['semantic_consistency']
    assert specs['g_consistent']['semantic_consistency']
    assert {k:v for k,v in specs['g_control'].items() if k not in ('root', 'semantic_consistency')} == {
        k:v for k,v in specs['g_consistent'].items() if k not in ('root', 'semantic_consistency')}
    fit_orders = {}
    qualification = {}
    for arm in ('g_control', 'g_consistent'):
        train = read_json(args.root/arm/'receipt.json')
        assert train['status'] == 'complete' and train['training_steps'] == 3723
        assert train['fit_rows'] == 29778 and train['holdout_rows'] == 6887 and train['formal_rows'] == 0
        assert train['fit_seen_exactly_once'] and train['frozen_parameters_unchanged'] and train['fresh_optimizer']
        assert train['p2'] == specs[arm]['p2']
        assert train['semantic_consistency'] == specs[arm]['semantic_consistency']
        assert train['base_terminal_sha256'] == specs[arm]['base_terminal_sha256']
        assert train['script_sha256'] == hashlib.sha256((args.root/'source/run.py').read_bytes()).hexdigest()
        raw = (args.root/arm/'train.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == train['train_log_sha256']
        records = [json.loads(line) for line in raw.splitlines()]
        assert [row['step'] for row in records] == list(range(1, 3724))
        assert all(row['total_steps'] == 3723 for row in records)
        assert all(row['semantic_consistency'] == specs[arm]['semantic_consistency'] for row in records)
        for row in records:
            assert all(math.isfinite(value) for key, value in row.items()
                       if key.startswith('loss') or key == 'grad_norm')
        fit_orders[arm] = [row['rows'] for row in records]
        seen = [row_id for row in fit_orders[arm] for row_id in row]
        assert len(seen) == 29778 and len(Counter(seen)) == 29778
        if arm == 'g_control':
            assert all(row['loss_contrastive_correction'] == 0 for row in records)
        else:
            qualification[arm] = {
                'updates_with_nonzero_contrastive_correction': sum(row['loss_contrastive_correction'] != 0 for row in records),
                'contrastive_correction_min': min(row['loss_contrastive_correction'] for row in records),
                'contrastive_correction_max': max(row['loss_contrastive_correction'] for row in records),
            }
    assert fit_orders['g_control'] == fit_orders['g_consistent']
    phases = {}
    for stage, count in (('initial', 6887), ('terminal', 6887), ('formal', 9508)):
        control, cm = read_evaluation(args.root/'g_control'/stage, stage, count)
        method, pm = read_evaluation(args.root/'g_consistent'/stage, stage, count)
        phases[stage] = {'rows':count, 'g_control':cm, 'g_consistent':pm,
                         'paired':{mode:compare_rows(control, method, mode) for mode in ('bbs','bbf')}}
        if stage == 'initial':
            # The original exact-summary check failed; preserve and disclose it.
            for mode in ('bbs', 'bbf'):
                for metric in ('iou', 'mask_iou'):
                    for threshold in (.25, .5):
                        assert all((a[mode][metric] > threshold) == (b[mode][metric] > threshold)
                                   for a, b in zip(control, method)), (mode, metric, threshold)
            initial_diagnostic = args.root.parent/'analysis_initial_comparison/SUMMARY.json'
            startup_comparison = read_json(initial_diagnostic)
            startup_comparison['diagnostic_sha256'] = hashlib.sha256(initial_diagnostic.read_bytes()).hexdigest()
            assert (cm['bbs']['rec_hits25'], cm['bbs']['rec_hits50']) == (6176, 5602)
    formal = phases['formal']
    hits = formal['g_consistent']['bbs']
    delta = formal['paired']['bbs']
    summary = {
        'primary_mode':'bbs', 'finished_cst':status['finished_cst'], 'phases':phases,
        'same_budget_strict_increment_preserving_loose':delta['25']['net'] >= 0 and delta['50']['net'] > 0,
        'historical_g_formal_hits':[5615,4495], 'scanrefer_development_target_hits':[5615,4754],
        'preserves_historical_g_both_thresholds':hits['rec_hits25'] >= 5615 and hits['rec_hits50'] >= 4495,
        'scanrefer_development_target_pass':hits['rec_hits25'] >= 5615 and hits['rec_hits50'] >= 4754,
        'nr3d_sr3d_new_method_results_available':False,
        'same_fit_batch_order': True, 'training_qualification': qualification,
        'starting_output_comparison': startup_comparison,
        'bitwise_paired_comparison': False,
        'interpretation_limits':[
            'The original exact starting-summary assertion failed and is archived. Inputs/GT and initial REC/Mask threshold bitmaps match; selected boxes, some queries, continuous Mask IoU and Top-k oracle vectors differ across processes. This is a qualified same-budget empirical comparison, not bitwise-paired evidence.',
            'The 6887 module holdout includes scenes seen by author pretraining.',
            'The 9508 formal set is development validation.',
            'Candidate coverage uses GT only for offline diagnosis.',
            'Both frames and scores can change across these trained models; paired repairs are not fixed-box reranking gains.',
            'bbs and bbf are reported separately; primary selection remains bbs.',
            'One seed provides no cross-seed significance estimate.',
            'Expanding positive correspondences with the original matched denominator also changes effective objective weight.',
            'No candidates are dropped from forward inference; training qualification is not a deployable GT filter.',
        ],
    }
    candidates = [
        ('original_g', 5615, 4495),
        ('g_control', formal['g_control']['bbs']['rec_hits25'], formal['g_control']['bbs']['rec_hits50']),
        ('g_consistent', hits['rec_hits25'], hits['rec_hits50']),
    ]
    leader = max(candidates, key=lambda row: (row[2], row[1]))
    summary['retention_recommendation'] = {
        'primary_metric': 'formal9508/last/bbs/Acc@0.50',
        'score_leader': leader[0], 'leader_hits25': leader[1], 'leader_hits50': leader[2],
        'nonleading_pair_endpoints': [row[0] for row in candidates[1:] if row[0] != leader[0]],
        'original_g_remains_required_parent': True,
        'deletion_executed': False,
    }
    lines = ['# PV-Ground G / G+consistent fixed-budget comparison', '',
             'Primary mode: `bbs`. Both arms reuse trained G, start fresh AdamW, and consume 29778 fit rows once (3723 updates).',
             'Seed2027, batch8, core/backbone LR1e-5, weight decay5e-4, clip0.1. The shared G adaptation history is additional to this pass.',
             'Only final contrastive correspondence replacement changes; both arms retain G CE replacement and all 256 candidates.', '',
             '| Set / mode | Rows | G @0.25 / @0.50 | G+consistent @0.25 / @0.50 | Delta hits |',
             '|---|---:|---:|---:|---:|']
    for stage, phase in phases.items():
        for mode in ('bbs','bbf'):
            c, p, d = phase['g_control'][mode], phase['g_consistent'][mode], phase['paired'][mode]
            lines.append('| {} / {} | {} | {} / {} | {} / {} | {:+d} / {:+d} |'.format(
                stage,mode,phase['rows'],c['rec_hits25'],c['rec_hits50'],p['rec_hits25'],p['rec_hits50'],d['25']['net'],d['50']['net']))
    lines.extend(['', '| Formal bbs threshold | Repairs | Damages | G coverable errors | Consistent coverable errors |',
                  '|---|---:|---:|---:|---:|'])
    for label in ('25','50'):
        row = delta[label]
        lines.append('| @{} | {} | {} | {} | {} |'.format(label,row['repairs'],row['damages'],
            row['control_errors_with_good_full256_candidate'],row['consistent_errors_with_good_full256_candidate']))
    lines.extend(['', 'The ScanRefer development target preserves historical G loose hits (5615) and requires strict hits >=4754 (50%).',
                  'Target passed: {}. Same-budget strict increment preserving loose: {}.'.format(
                      summary['scanrefer_development_target_pass'],summary['same_budget_strict_increment_preserving_loose']), '',
                  'Score leader for retention: {} ({} / {} hits). Original G remains a required parent. No weights were deleted by this analysis.'.format(*leader), '',
                  'Interpretation limits:', ''] + ['- '+value for value in summary['interpretation_limits']])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'formal_bbs_delta_hits':[delta['25']['net'],delta['50']['net']],
                      'scanrefer_development_target_pass':summary['scanrefer_development_target_pass']}))


if __name__ == '__main__':
    main()
