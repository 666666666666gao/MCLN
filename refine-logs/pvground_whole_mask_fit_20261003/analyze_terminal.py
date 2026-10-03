"""Analyze the closed local/whole source pair from verified native row evidence."""
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


def box_iou(box, target):
    assert len(box) == len(target) == 6
    assert all(math.isfinite(x) for x in box+target)
    assert all(x > 0 for x in box[3:]+target[3:])
    extent = [max(0., min(box[i]+box[i+3]/2, target[i]+target[i+3]/2)
        - max(box[i]-box[i+3]/2, target[i]-target[i+3]/2)) for i in range(3)]
    intersection = extent[0]*extent[1]*extent[2]
    return intersection/(box[3]*box[4]*box[5]+target[3]*target[4]*target[5]-intersection)


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def refinement(rows, mode):
    values = [row[mode] for row in rows]
    movements = []
    cpu_final, cpu_coarse = [], []
    for row, value in zip(rows, values):
        assert 0 <= value['query'] < 256
        assert math.isfinite(value['coarse_iou']) and 0 <= value['coarse_iou'] <= 1
        cpu_final.append(box_iou(value['box'], row['root_box']))
        cpu_coarse.append(box_iou(value['coarse_box'], row['root_box']))
        movements.append(max(abs((value['box'][axis]+sign*value['box'][axis+3]/2)
            - (value['coarse_box'][axis]+sign*value['coarse_box'][axis+3]/2))
            for axis in range(3) for sign in (-1, 1)))
    result = {
        'selected_max_face_displacement_m': {str(q): quantile(movements, q) for q in (.5, .9, .99, 1.)},
        'selected_rows_max_face_displacement_under_1cm': sum(x < .01 for x in movements),
        'mean_selected_coarse_iou': sum(v['coarse_iou'] for v in values)/len(rows),
        'mean_selected_final_iou': sum(v['iou'] for v in values)/len(rows),
        'cpu_final_threshold_changes': sum((u > t) != (v['iou'] > t)
            for u, v in zip(cpu_final, values) for t in (.25, .5)),
        'cpu_coarse_threshold_changes': sum((u > t) != (v['coarse_iou'] > t)
            for u, v in zip(cpu_coarse, values) for t in (.25, .5)),
    }
    assert result['cpu_final_threshold_changes'] == 0
    for label, threshold in (('25', .25), ('50', .5)):
        coarse = sum(v['coarse_iou'] > threshold for v in values)
        final = sum(v['iou'] > threshold for v in values)
        repairs = sum(v['coarse_iou'] <= threshold < v['iou'] for v in values)
        damages = sum(v['iou'] <= threshold < v['coarse_iou'] for v in values)
        assert final-coarse == repairs-damages
        full = sum(v['oracle'+label][-1] for v in values)
        missing = sum(not v['oracle'+label][-1] for v in values)
        coverable_errors = sum(v['iou'] <= threshold and v['oracle'+label][-1] for v in values)
        assert coverable_errors+missing == len(rows)-final
        result[label] = dict(coarse_selected_hits=coarse, final_selected_hits=final,
            repairs=repairs, damages=damages, net=repairs-damages,
            coarse_full256_hits=sum(v['coarse_oracle'+label] for v in values),
            final_full256_hits=full, errors_with_good_full256_candidate=coverable_errors,
            errors_without_good_full256_candidate=missing)
    result['selected_box_mask_at50'] = {
        'both_good': sum(v['iou'] > .5 and v['mask_iou'] > .5 for v in values),
        'box_only_good': sum(v['iou'] > .5 and v['mask_iou'] <= .5 for v in values),
        'mask_only_good': sum(v['iou'] <= .5 and v['mask_iou'] > .5 for v in values),
        'both_bad': sum(v['iou'] <= .5 and v['mask_iou'] <= .5 for v in values),
    }
    buckets = [0]*5
    for value in values:
        if value['iou'] > .5:
            continue
        bits = value['oracle50']
        index = next((i for i, bit in enumerate(bits) if bit), 4)
        buckets[index] += 1
    result['strict_error_first_good_candidate_rank_bucket'] = dict(zip(
        ('2_16', '17_32', '33_64', '65_256', 'none_full256'), buckets))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    helper_path = Path(__file__).parent.parent/'pvground_candidate_consistency_20261003/analyze_complete_initial_qualified.py'
    assert sha(helper_path) == 'a5d01e0cf5babeda9f440f8a3ab815ffb27b6a9606a01f89c9b6cd402589ce07'
    helper_spec = importlib.util.spec_from_file_location('native_row_metrics', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    intake = read_json(args.root/'INTAKE.json')
    status = intake['status']
    assert status['status'] == 'complete' and intake['controller_exit'] == 0
    assert not intake['controller_alive'] and intake['weights_downloaded'] == 0
    assert [(r['arm'], r['mode']) for r in status['completed']] == [
        ('local_range', 'train'), ('local_range', 'formal'), ('whole_range', 'train'), ('whole_range', 'formal')]
    for relative, item in intake['files'].items():
        path = args.root/relative
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], relative
    assert sha(args.root/'run_whole_mask_fit.py') == 'd862a861d05f0411ed76c18b78684b485fe1fdfaa8a0c00d3250ce9a9e8cfeac'
    arms = ('local_range', 'whole_range')
    specs = {arm: read_json(args.root/arm/'spec.json') for arm in arms}
    changed = {'root', 'support_arm', 'use_whole_range', 'preflight_root'}
    assert {k: v for k, v in specs[arms[0]].items() if k not in changed} == {
        k: v for k, v in specs[arms[1]].items() if k not in changed}
    train_receipts, orders = {}, {}
    for arm in arms:
        directory, spec = args.root/arm, specs[arm]
        assert spec['support_arm'] == arm and spec['use_whole_range'] == (arm == 'whole_range')
        assert spec['batch_size'] == 8 and spec['fit_passes'] == 1 and spec['seed'] == 2027
        assert spec['lr'] == spec['lr_backbone'] == 1e-5 and not spec['p2']
        assert spec['p3'] and spec['fused_support'] and spec['semantic_assignment']
        assert spec['primary_mode'] == 'bbs' and spec['primary_threshold'] == .5
        assert spec['base_terminal_sha256'] == intake['original_g_sha256']
        assert spec['env_spec_sha256'] == intake['environment_spec_sha256']
        for name, digest in spec['whole_range_files'].items():
            assert sha(directory/name) == digest
        for name, key in (
                ('pvground_candidate_box_refiner.py', 'p3_module_sha256'),
                ('pvground_tail_support_box_refiner.py', 'tail_module_sha256'),
                ('pvground_tail_preflight.py', 'tail_preflight_module_sha256'),
                ('pvground_semantic_assignment.py', 'assignment_module_sha256'),
                ('pvground_source_query.py', 'source_query_module_sha256'),
                ('pvground_observation_query.py', 'observation_module_sha256'),
                ('pvground_task_observation_query.py', 'task_module_sha256')):
            assert sha(directory/name) == spec[key]
        imported = read_json(directory/'imports.json')
        assert imported['sha256'] == read_json(args.root/arms[0]/'imports.json')['sha256']
        loaded = read_json(directory/'load.json')
        assert loaded['status'] == 'pass' and loaded['fresh_optimizer'] and not loaded['p2']
        assert loaded['g_delta_tensors'] == 1072 and loaded['p3_states'] == 10
        assert loaded['head_parameters'] == 400614 and loaded['base_terminal'] == spec['base_terminal']
        fit = read_json(directory/'receipt.json')
        train_receipts[arm] = fit
        assert fit['status'] == 'complete' and fit['training_steps'] == 3723
        assert fit['fit_rows'] == 29778 and fit['holdout_rows'] == 6887 and fit['formal_rows'] == 0
        assert fit['fit_seen_exactly_once'] and fit['frozen_parameters_unchanged'] and fit['fresh_optimizer']
        assert fit['whole_range_files'] == spec['whole_range_files'] and fit['use_whole_range'] == spec['use_whole_range']
        assert fit['head_parameters'] == 400614 and fit['tail_after_native_masks'] and fit['primary_mode'] == 'bbs'
        assert fit['base_terminal_sha256'] == spec['base_terminal_sha256']
        assert fit['script_sha256'] == sha(args.root/'run_whole_mask_fit.py')
        assert fit['spec_sha256'] == sha(directory/'spec.json')
        assert (directory/'train.exit').read_text().strip() == (directory/'formal.exit').read_text().strip() == '0'
        log = directory/'train.jsonl'
        assert sha(log) == fit['train_log_sha256']
        records = [json.loads(line) for line in log.read_bytes().splitlines()]
        assert [r['step'] for r in records] == list(range(1, 3724))
        assert all(r['total_steps'] == 3723 for r in records)
        assert all(math.isfinite(v) for r in records for k, v in r.items() if k.startswith('loss') or k == 'grad_norm')
        orders[arm] = [r['rows'] for r in records]
        seen = [row_id for batch in orders[arm] for row_id in batch]
        assert len(seen) == len(Counter(seen)) == 29778
        retained = read_json(directory/'weight_retention.json')
        assert retained['terminal_sha256'] == fit['terminal_sha256']
        assert retained['parent_sha256_after'] == intake['original_g_sha256']
        assert not retained['local_weight_archive_created'] and not retained['model_or_optimizer_replayed']
    assert orders[arms[0]] == orders[arms[1]]
    assert train_receipts['whole_range']['fit_batch_row_order_matches_control']
    phases, formal_rows = {}, {}
    for stage, count in (('initial', 6887), ('terminal', 6887), ('formal', 9508)):
        evaluated = {arm: helper.read_evaluation(args.root/arm/stage, stage, count) for arm in arms}
        paired = {mode: helper.compare_rows(evaluated[arms[0]][0], evaluated[arms[1]][0], mode) for mode in ('bbs', 'bbf')}
        for mode in paired.values():
            for value in mode.values():
                value['whole_range_hits'] = value.pop('consistent_hits')
                value['whole_range_errors_with_good_full256_candidate'] = value.pop('consistent_errors_with_good_full256_candidate')
        phases[stage] = dict(rows=count, metrics={arm: value[1] for arm, value in evaluated.items()}, paired=paired)
        if stage == 'initial':
            comparison_path = args.root/'whole_range/initial_range_comparison.py'
            assert sha(comparison_path) == sha(args.root/'local_range/initial_range_comparison.py')
            comparison_spec = importlib.util.spec_from_file_location('range_start_comparison', comparison_path)
            comparator = importlib.util.module_from_spec(comparison_spec)
            comparison_spec.loader.exec_module(comparator)
            actual_start = comparator.compare_initial_rows(evaluated['whole_range'][0], evaluated['local_range'][0])
            assert actual_start == read_json(args.root/'whole_range/initial_control_comparison.json')
            assert actual_start == train_receipts['whole_range']['initial_pair_comparison']
        if stage != 'formal':
            for arm, (_, metrics) in evaluated.items():
                for mode in ('bbs', 'bbf'):
                    for key, value in train_receipts[arm][stage][mode].items():
                        assert abs(metrics[mode][key]-value) < 1e-6
        else:
            formal_rows = {arm: value[0] for arm, value in evaluated.items()}
    formal = phases['formal']
    leader = dict(name='original_g', bbs_hits25=5615, bbs_hits50=4495)
    for arm in arms:
        metric = formal['metrics'][arm]['bbs']
        if metric['rec_hits50'] > leader['bbs_hits50']:
            leader = dict(name=arm, bbs_hits25=metric['rec_hits25'], bbs_hits50=metric['rec_hits50'])
    assert all(status['retained_best'][key] == value for key, value in leader.items())
    diagnostics = {arm: {mode: refinement(formal_rows[arm], mode) for mode in ('bbs', 'bbf')} for arm in arms}
    limits = [
        'Both arms start from original G with fresh AdamW. The only planned source change is the 109-D whole-instance range versus zeros; all 256 candidates remain available.',
        'The actual cross-process initial query, continuous box/Mask-IoU and threshold differences are recorded. Same-forward zero-head exactness does not establish cross-process bitwise pairing.',
        'The 6887 module holdout has scenes seen by author pretraining; the 9508 formal set is development validation.',
        'Full-256 coverage uses GT offline and does not certify physical instance identity or deployable achievable accuracy.',
        'Both boxes and scores can change during shared training. Between-model repairs are not fixed-box reranking gains.',
        'Within-model coarse/final comparison fixes the selected Query in a jointly trained model; the coarse box is not an independent baseline.',
        'CPU reconstruction verifies selected box thresholds. Mask and full-candidate coverage recount saved native values; raw Masks and all candidate boxes are not independently reconstructed.',
        'Terminal weights were restored by the formal runner and hashes were checked before controller retention. This analyzer does not replay model or optimizer tensors, download or delete weights.',
        'One fixed seed supplies no cross-seed significance estimate. No new Nr3D/Sr3D, six-face distribution, geometry-quality readback or V99 teacher result exists.',
    ]
    summary = dict(finished_cst=status['finished_cst'], primary_mode='bbs', phases=phases,
        same_fit_batch_order=True, starting_output_comparison=actual_start, bitwise_paired_comparison=False,
        same_query_refinement=diagnostics, historical_g_formal_hits=[5615, 4495],
        delta_from_original_g={arm: [formal['metrics'][arm]['bbs']['rec_hits25']-5615,
            formal['metrics'][arm]['bbs']['rec_hits50']-4495] for arm in arms},
        scanrefer_development_target_hits=[5615, 4754], retained_best=status['retained_best'],
        scanrefer_development_target_pass=leader['bbs_hits25'] >= 5615 and leader['bbs_hits50'] >= 4754,
        interpretation_limits=limits, analysis_weight_deletions=0,
        input_hashes=dict(intake=sha(args.root/'INTAKE.json'), metrics_helper=sha(helper_path)))
    assert summary['scanrefer_development_target_pass'] == status['scanrefer_target_pass']
    lines = ['# PV-Ground local / whole complete range-source comparison', '',
        'Original G, fresh AdamW, seed2027, batch8, LR1e-5/core and backbone, weight decay5e-4, clip0.1.',
        'Each arm consumes 29778 fit rows once in identical order: 3723 optimizer updates. Both use the same 400614-parameter head and native+G losses.', '',
        '| Set / mode | Rows | Local hits @.25 / .50 | Whole hits @.25 / .50 | Whole minus local |',
        '|---|---:|---:|---:|---:|']
    for stage, phase in phases.items():
        for mode in ('bbs', 'bbf'):
            a, b, paired = phase['metrics']['local_range'][mode], phase['metrics']['whole_range'][mode], phase['paired'][mode]
            lines.append('| {} / {} | {} | {} / {} | {} / {} | {:+d} / {:+d} |'.format(stage, mode,
                phase['rows'], a['rec_hits25'], a['rec_hits50'], b['rec_hits25'], b['rec_hits50'], paired['25']['net'], paired['50']['net']))
    lines += ['', '| Formal bbs threshold | Whole repairs vs local | Whole damages vs local | Net |', '|---|---:|---:|---:|']
    for label, value in formal['paired']['bbs'].items():
        lines.append('| @{} | {} | {} | {:+d} |'.format(label, value['repairs'], value['damages'], value['net']))
    lines += ['', '| Formal arm / mode | Mask @.25 hits | Mask @.50 hits | Mask mIoU % |', '|---|---:|---:|---:|']
    for arm in arms:
        for mode in ('bbs', 'bbf'):
            metric = formal['metrics'][arm][mode]
            lines.append('| {} / {} | {} | {} | {:.8f} |'.format(arm, mode, metric['mask_hits25'], metric['mask_hits50'], metric['mask_miou']))
    lines += ['', '| Formal bbs arm | Same-Query coarse/final @.50 | Repairs / damages | Full256 good / missing errors | Median max-face move mm |', '|---|---:|---:|---:|---:|']
    for arm in arms:
        diagnostic = diagnostics[arm]['bbs']
        value = diagnostic['50']
        lines.append('| {} | {} / {} | {} / {} | {} / {} | {:.6f} |'.format(arm,
            value['coarse_selected_hits'], value['final_selected_hits'], value['repairs'], value['damages'],
            value['errors_with_good_full256_candidate'], value['errors_without_good_full256_candidate'],
            diagnostic['selected_max_face_displacement_m']['0.5']*1000))
    lines += ['', 'Retained metric best: {} ({}/{}). Target5615/4754 passed: {}.'.format(
        leader['name'], leader['bbs_hits25'], leader['bbs_hits50'], summary['scanrefer_development_target_pass']),
        'Original-G deltas: {}. Controller retention is verified separately; this analyzer deletes no weights.'.format(summary['delta_from_original_g']), '',
        'Interpretation limits:', ''] + ['- '+value for value in limits]
    args.output.mkdir(parents=True)
    (args.output/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), retained_best=leader['name'],
        delta_from_original_g=summary['delta_from_original_g'], target_pass=summary['scanrefer_development_target_pass'])))


if __name__ == '__main__':
    main()
