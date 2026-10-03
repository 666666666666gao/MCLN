"""Compare normalization to the two completed arms, without model execution."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_difference(before, after):
    result = {}
    for mode in ('bbs', 'bbf'):
        values = [(a[mode], b[mode]) for a, b in zip(before, after)]
        result[mode] = {
            'selected_query_changes': sum(a['query'] != b['query'] for a, b in values),
            'selected_box_changes': sum(a['box'] != b['box'] for a, b in values),
            'max_abs_selected_box_difference_m': max(abs(x-y) for a, b in values for x, y in zip(a['box'], b['box'])),
            'selected_iou_changes': sum(a['iou'] != b['iou'] for a, b in values),
            'selected_mask_iou_changes': sum(a['mask_iou'] != b['mask_iou'] for a, b in values),
            'max_abs_mask_iou_difference': max(abs(a['mask_iou']-b['mask_iou']) for a, b in values),
            'threshold_bitmap_difference_counts': {
                metric + str(threshold): sum((a[metric] > threshold) != (b[metric] > threshold) for a, b in values)
                for metric in ('iou', 'mask_iou') for threshold in (.25, .5)},
            'oracle_vector_changes': {key: sum(a[key] != b[key] for a, b in values) for key in ('oracle25', 'oracle50')},
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--comparators', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve any earlier analysis'
    helper_path = args.comparators.parent / 'analyze_complete_initial_qualified.py'
    assert sha(helper_path) == 'a5d01e0cf5babeda9f440f8a3ab815ffb27b6a9606a01f89c9b6cd402589ce07'
    helper_spec = importlib.util.spec_from_file_location('completed_pair_metrics', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    intake = read_json(args.root / 'INTAKE.json')
    assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
    assert not intake['controller_alive'] and not intake['checkpoints_downloaded']
    assert [row['mode'] for row in intake['status']['completed']] == ['train', 'formal']
    for relative, expected in intake['files'].items():
        assert (args.root / relative).stat().st_size == expected['bytes']
        assert sha(args.root / relative) == expected['sha256'], relative
    old_status = read_json(args.comparators / 'pair_status.json')
    assert old_status['status'] == 'complete'
    assert [(row['arm'], row['mode']) for row in old_status['completed']] == [
        ('g_control', 'train'), ('g_consistent', 'train'), ('g_control', 'formal'), ('g_consistent', 'formal')]
    directories = {name: args.comparators / name for name in ('g_control', 'g_consistent')}
    directories['normalized'] = args.root / 'normalized'
    specs = {name: read_json(directory / 'spec.json') for name, directory in directories.items()}
    changed_fields = {'root', 'comparison', 'consistency_module_sha256', 'semantic_consistency',
                      'contrastive_normalization', 'completed_comparators'}
    common = {key: value for key, value in specs['g_control'].items() if key not in changed_fields}
    assert all({key: value for key, value in spec.items() if key not in changed_fields} == common for spec in specs.values())
    assert not specs['g_control']['semantic_consistency']
    assert specs['g_consistent']['semantic_consistency'] and specs['normalized']['semantic_consistency']
    assert specs['normalized']['contrastive_normalization'] == 'expanded_correspondence_count'
    assert not any(spec['p2'] for spec in specs.values())
    assert sha(args.root / 'source/run.py') == sha(args.comparators / 'source/run.py')
    assert sha(args.root / 'source/run.py') == '919fc36d61b8f21a7285f55c153717f05baa4e969ca43ccbc25171e165bc7392'
    assert sha(args.root / 'source/pvground_candidate_consistency.py') == specs['normalized']['consistency_module_sha256']
    assert sha(args.comparators / 'source/pvground_candidate_consistency.py') == specs['g_consistent']['consistency_module_sha256']
    fit_orders, qualification, train_receipts = {}, {}, {}
    for name, directory in directories.items():
        imports = read_json(directory / 'imports.json')
        reference = read_json(directories['g_control'] / 'imports.json')
        assert {key: value for key, value in imports['sha256'].items() if key != 'pvground_candidate_consistency'} == {
            key: value for key, value in reference['sha256'].items() if key != 'pvground_candidate_consistency'}
        assert imports['sha256']['pvground_candidate_consistency'] == specs[name]['consistency_module_sha256']
        load = read_json(directory / 'load.json')
        assert load['status'] == 'pass' and load['fresh_optimizer'] and load['p2_states'] == 0
        assert load['base_terminal'] == specs[name]['base_terminal']
        train = read_json(directory / 'receipt.json')
        train_receipts[name] = train
        assert train['status'] == 'complete' and train['training_steps'] == 3723
        assert train['fit_rows'] == 29778 and train['holdout_rows'] == 6887 and train['formal_rows'] == 0
        assert train['fit_seen_exactly_once'] and train['frozen_parameters_unchanged'] and train['fresh_optimizer']
        assert train['base_terminal_sha256'] == specs[name]['base_terminal_sha256']
        assert train['spec_sha256'] == sha(directory / 'spec.json')
        assert train['script_sha256'] == sha(args.root / 'source/run.py')
        assert train['semantic_consistency'] == specs[name]['semantic_consistency'] and not train['p2']
        assert (directory / 'train.exit').read_text().strip() == '0'
        assert (directory / 'formal.exit').read_text().strip() == '0'
        raw = (directory / 'train.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == train['train_log_sha256']
        records = [json.loads(line) for line in raw.splitlines()]
        assert [row['step'] for row in records] == list(range(1, 3724))
        assert all(row['total_steps'] == 3723 and row['semantic_consistency'] == specs[name]['semantic_consistency'] for row in records)
        for row in records:
            assert all(math.isfinite(value) for key, value in row.items() if key.startswith('loss') or key == 'grad_norm')
        fit_orders[name] = [row['rows'] for row in records]
        seen = [row_id for batch in fit_orders[name] for row_id in batch]
        assert len(seen) == 29778 and len(Counter(seen)) == 29778
        if name == 'g_control':
            assert all(row['loss_contrastive_correction'] == 0 for row in records)
            continue
        for row in records:
            assert row['contrastive_reassigned_queries'] == row['reassigned_queries']
            assert row['contrastive_reconstruction_error'] <= 1e-6
            assert abs(row['loss_contrastive_correction'] - (.5 / 7) * (
                row['contrastive_replaced'] - row['contrastive_native'])) <= 1e-5
            if name == 'normalized':
                assert row['expanded_count_normalization'] and not row['semantic_target_only']
                assert row['contrastive_native_denominator'] == row['matched_queries']
                assert row['contrastive_expanded_denominator'] == row['matched_queries'] + row['contrastive_reassigned_queries']
                assert row['contrastive_native_denominator'] > 0
            else:
                assert row['semantic_target_only']
        qualification[name] = {
            'extra_correspondences_total': sum(row['contrastive_reassigned_queries'] for row in records),
            'updates_with_negative_expanded_loss': sum(row['contrastive_replaced'] < 0 for row in records),
            'updates_with_nonzero_correction': sum(row['loss_contrastive_correction'] != 0 for row in records),
            'correction_min': min(row['loss_contrastive_correction'] for row in records),
            'correction_max': max(row['loss_contrastive_correction'] for row in records),
            'max_reconstruction_error': max(row['contrastive_reconstruction_error'] for row in records),
        }
        if name == 'normalized':
            qualification[name]['native_denominator_total'] = sum(row['contrastive_native_denominator'] for row in records)
            qualification[name]['expanded_denominator_total'] = sum(row['contrastive_expanded_denominator'] for row in records)
            qualification[name]['expanded_to_native_ratio_min'] = min(row['contrastive_expanded_denominator'] / row['contrastive_native_denominator'] for row in records)
            qualification[name]['expanded_to_native_ratio_max'] = max(row['contrastive_expanded_denominator'] / row['contrastive_native_denominator'] for row in records)
    assert fit_orders['normalized'] == fit_orders['g_control'] == fit_orders['g_consistent']
    phases, startup = {}, {}
    for stage, count in (('initial', 6887), ('terminal', 6887), ('formal', 9508)):
        evaluations = {name: helper.read_evaluation(directory / stage, stage, count) for name, directory in directories.items()}
        pairs = {}
        for name in ('g_control', 'g_consistent'):
            pairs[name] = {mode: helper.compare_rows(evaluations[name][0], evaluations['normalized'][0], mode) for mode in ('bbs', 'bbf')}
            for mode in ('bbs', 'bbf'):
                for label in ('25', '50'):
                    row = pairs[name][mode][label]
                    row['normalized_hits'] = row.pop('consistent_hits')
                    row['normalized_errors_with_good_full256_candidate'] = row.pop('consistent_errors_with_good_full256_candidate')
            if stage == 'initial':
                startup[name] = initial_difference(evaluations[name][0], evaluations['normalized'][0])
        if stage != 'formal':
            for name, (_, metrics) in evaluations.items():
                for mode in ('bbs', 'bbf'):
                    for key, value in train_receipts[name][stage][mode].items():
                        assert abs(metrics[mode][key] - value) < 1e-6, (name, stage, mode, key)
        phases[stage] = {'rows': count, 'metrics': {name: value[1] for name, value in evaluations.items()}, 'vs_normalized': pairs}
    formal = phases['formal']
    hit = formal['metrics']['normalized']['bbs']
    candidates = [('original_g', 5615, 4495)] + [
        (name, formal['metrics'][name]['bbs']['rec_hits25'], formal['metrics'][name]['bbs']['rec_hits50'])
        for name in ('g_control', 'g_consistent', 'normalized')]
    leader = max(candidates, key=lambda row: (row[2], row[1], row[0] == 'original_g'))
    startup_equal = all(all(not count for count in mode['threshold_bitmap_difference_counts'].values())
                        for comparison in startup.values() for mode in comparison.values())
    limits = [
        'Same original-G weight, fresh optimizer and fit order; cross-process continuous starting outputs can differ. Actual initial REC/Mask bitmap and candidate/query differences are reported, without assuming parity.',
        'Normalization changes the scale of the entire expanded contrastive term, including original and background contributions; it is not a pure label-only or extra-positive-only change.',
        'The 6887 module holdout has author-pretraining-seen scenes; the 9508 formal set is development validation.',
        'GT candidate coverage is offline evidence, not a deployable filter or verified instance identity.',
        'Both boxes and scores can change through shared training. Repairs are not fixed-box reranking gains.',
        'All 256 candidates remain available; bbs and bbf are separate outputs, with bbs primary.',
        'One fixed seed supplies no cross-seed significance estimate; no new Nr3D or Sr3D results exist.',
        'This analysis recounts the saved native IoU/Mask-IoU records; it does not independently reconstruct all Masks or full candidate geometry from raw data.',
    ]
    summary = {
        'finished_cst': intake['status']['finished_cst'], 'primary_mode': 'bbs', 'phases': phases,
        'training_qualification': qualification, 'same_fit_batch_order': True,
        'starting_output_comparison': startup, 'initial_threshold_bitmaps_equal': startup_equal,
        'bitwise_paired_comparison': False, 'historical_g_formal_hits': [5615, 4495],
        'delta_from_original_g': [hit['rec_hits25'] - 5615, hit['rec_hits50'] - 4495],
        'scanrefer_development_target_hits': [5615, 4754],
        'scanrefer_development_target_pass': hit['rec_hits25'] >= 5615 and hit['rec_hits50'] >= 4754,
        'preserves_historical_g_both_thresholds': hit['rec_hits25'] >= 5615 and hit['rec_hits50'] >= 4495,
        'nr3d_sr3d_new_method_results_available': False, 'interpretation_limits': limits,
        'input_hashes': {'metrics_helper': sha(helper_path), 'intake': sha(args.root / 'INTAKE.json')},
        'retention_recommendation': {'primary_metric': 'formal9508/last/bbs/Acc@0.50',
            'score_leader': leader[0], 'leader_hits25': leader[1], 'leader_hits50': leader[2],
            'normalized_endpoint_keep': leader[0] == 'normalized',
            'original_g_remains_required_parent': True, 'deletion_executed': False},
    }
    lines = ['# PV-Ground candidate normalization: completed three-arm comparison', '',
        'All arms start from original G with fresh AdamW, seed2027, batch8, LR1e-5, weight decay5e-4 and clip0.1; 29778 fit rows are consumed once in identical order (3723 updates).',
        'Existing G CE replacement, original model, native score and 256 candidates remain unchanged. Only the expanded final contrastive denominator changes from N to N+A relative to g_consistent.', '',
        '| Set / mode | Rows | G control hits | Expanded N hits | Expanded N+A hits | N+A delta vs control | N+A delta vs expanded N |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage, phase in phases.items():
        for mode in ('bbs', 'bbf'):
            values = [phase['metrics'][name][mode] for name in ('g_control', 'g_consistent', 'normalized')]
            deltas = [phase['vs_normalized'][name][mode] for name in ('g_control', 'g_consistent')]
            lines.append('| {} / {} | {} | {} | {} | {} | {:+d} / {:+d} | {:+d} / {:+d} |'.format(
                stage, mode, phase['rows'], *['{} / {}'.format(value['rec_hits25'], value['rec_hits50']) for value in values],
                deltas[0]['25']['net'], deltas[0]['50']['net'], deltas[1]['25']['net'], deltas[1]['50']['net']))
    lines += ['', '| Formal bbs comparison | Threshold | Repairs | Damages | Net |', '|---|---:|---:|---:|---:|']
    for name in ('g_control', 'g_consistent'):
        for label in ('25', '50'):
            row = formal['vs_normalized'][name]['bbs'][label]
            lines.append('| N+A vs {} | @{} | {} | {} | {:+d} |'.format(name, label, row['repairs'], row['damages'], row['net']))
    lines += ['', '| Formal mode / arm | Mask @0.25 hits | Mask @0.50 hits | Mask mIoU % |', '|---|---:|---:|---:|']
    for mode in ('bbs', 'bbf'):
        for name in ('g_control', 'g_consistent', 'normalized'):
            value = formal['metrics'][name][mode]
            lines.append('| {} / {} | {} | {} | {:.8f} |'.format(mode, name, value['mask_hits25'], value['mask_hits50'], value['mask_miou']))
    lines += ['', 'Normalized delta from original G (5615/4495): {:+d}/{:+d} hits. Development target 5615/4754 passed: {}.'.format(
        *summary['delta_from_original_g'], summary['scanrefer_development_target_pass']),
        'Initial threshold bitmaps all equal: {}. Continuous differences and query/coverage differences are retained in SUMMARY.json.'.format(startup_equal),
        'Metric-best for weight retention: {} ({}/{} hits). Original G remains a necessary parent. This analyzer deletes no weights.'.format(*leader), '',
        'Interpretation limits:', ''] + ['- ' + value for value in limits]
    args.output.mkdir(parents=True)
    (args.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), delta_from_original_g=summary['delta_from_original_g'],
        target_pass=summary['scanrefer_development_target_pass'], weight_leader=leader[0])))


if __name__ == '__main__':
    main()
