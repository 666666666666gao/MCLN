"""Recount the closed full pair from archived predictions, without a model replay."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / 'complete_fit'
OUTPUT = ROOT / 'analysis'
ARMS = ('content', 'box_conditioned')
BOX_NAMES = ('parent',) + ARMS
STAGES = ('initial_formal', 'formal')
N = 9508


def read_json(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def box_iou(boxes, truth):
    boxes = np.asarray(boxes, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    low = np.maximum(boxes[..., :3] - boxes[..., 3:] / 2, truth[..., :3] - truth[..., 3:] / 2)
    high = np.minimum(boxes[..., :3] + boxes[..., 3:] / 2, truth[..., :3] + truth[..., 3:] / 2)
    intersection = np.maximum(high - low, 0).prod(axis=-1)
    union = boxes[..., 3:].prod(axis=-1) + truth[..., 3:].prod(axis=-1) - intersection
    return intersection / union


def compare(before, after):
    assert len(before) == len(after) == N
    return {str(t): dict(before_hits=sum(value > t for value in before),
        after_hits=sum(value > t for value in after),
        repairs=sum(old <= t < new for old, new in zip(before, after)),
        damages=sum(new <= t < old for old, new in zip(before, after)),
        net=sum(value > t for value in after) - sum(value > t for value in before))
        for t in (.25, .5)}


def align(before, after):
    assert len(before) == len(after) == N
    for old, new in zip(before, after):
        assert all(old[key] == new[key] for key in
            ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))


def recount(stage, rows):
    directory = COMPLETE / stage
    receipt = read_json(directory / 'receipt.json')
    assert receipt['status'] == 'pass' and receipt['rows'] == receipt['formal_rows'] == N
    assert receipt['frozen_parent_forward_per_batch'] == 1
    assert receipt['all256_retained'] and receipt['same_selected_query_box_and_mask']
    assert receipt['primary_mode'] == 'bbs'
    assert digest(directory / 'rows.jsonl') == receipt['rows_sha256']
    assert [row['row_id'] for row in rows] == list(range(N))
    assert all(row['parent_forwards'] == row['final_semantic_head_calls'] == 1 for row in rows)
    flips = {name: {str(t): 0 for t in (.25, .5)} for name in BOX_NAMES}
    coverage = {name: {str(t): 0 for t in (.25, .5)} for name in BOX_NAMES}
    oracle_mismatches = {arm: {str(t): 0 for t in (.25, .5)} for arm in ARMS}
    available_unselected = {arm: {str(t): 0 for t in (.25, .5)} for arm in ARMS}
    missing_qualified = {arm: {str(t): 0 for t in (.25, .5)} for arm in ARMS}
    offset = batches = 0
    for path in sorted(directory.glob('batch_*.npz')):
        assert path.name == 'batch_%05d.npz' % offset
        with np.load(path, allow_pickle=False) as batch:
            assert set(batch.files) == {'row_ids', 'root_gt', 'scores', *BOX_NAMES}
            n = len(batch['row_ids'])
            assert n == min(8, N - offset)
            assert np.array_equal(batch['row_ids'], np.arange(offset, offset + n))
            assert batch['root_gt'].shape == (n, 6)
            assert batch['scores'].shape == (n, 256) and np.isfinite(batch['scores']).all()
            current = rows[offset:offset + n]
            truth = np.asarray([row['root_box'] for row in current])
            np.testing.assert_array_equal(batch['root_gt'], truth)
            indices = np.arange(n)
            queries = np.asarray([row['query'] for row in current])
            assert np.array_equal(batch['scores'][indices, queries], batch['scores'].max(axis=1))
            for name in BOX_NAMES:
                boxes = batch[name]
                assert boxes.shape == (n, 256, 6)
                assert np.isfinite(boxes).all() and (boxes[..., 3:] > 0).all()
                if name == 'parent':
                    saved_boxes = [row['parent_box'] for row in current]
                    saved_ious = [row['parent_iou'] for row in current]
                else:
                    saved_boxes = [row['arms'][name]['box'] for row in current]
                    saved_ious = [row['arms'][name]['iou'] for row in current]
                np.testing.assert_array_equal(boxes[indices, queries], saved_boxes)
                ious = box_iou(boxes, truth[:, None, :])
                for threshold in (.25, .5):
                    qualified = (ious > threshold).any(axis=1)
                    selected = ious[indices, queries] > threshold
                    coverage[name][str(threshold)] += int(qualified.sum())
                    flips[name][str(threshold)] += sum(bool(value) != bool(saved > threshold)
                        for value, saved in zip(selected, saved_ious))
                    if name in ARMS:
                        label = 'oracle25' if threshold == .25 else 'oracle50'
                        oracle_mismatches[name][str(threshold)] += sum(
                            bool(value) != bool(row['arms'][name][label][-1])
                            for value, row in zip(qualified, current))
                        available_unselected[name][str(threshold)] += int((qualified & ~selected).sum())
                        missing_qualified[name][str(threshold)] += int((~qualified).sum())
            if stage == 'initial_formal':
                for arm in ARMS:
                    np.testing.assert_array_equal(batch[arm], batch['parent'])
            offset += n
            batches += 1
    assert offset == N and batches == 1189
    table = []
    for arm in ARMS:
        values = [row['arms'][arm]['iou'] for row in rows]
        hits = [sum(value > threshold for value in values) for threshold in (.25, .5)]
        metric = receipt['metrics'][arm]
        assert hits == [metric['rec_hits25'], metric['rec_hits50']]
        masks = [row['arms'][arm]['mask_iou'] for row in rows]
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in masks)
        assert sum(value > .25 for value in masks) == metric['mask_hits25']
        assert sum(value > .5 for value in masks) == metric['mask_hits50']
        assert abs(sum(masks) / N * 100 - metric['mask_miou']) < 1e-9
        parent_to_final = compare([row['parent_iou'] for row in rows], values)
        for threshold, suffix in ((.25, '25'), (.5, '50')):
            assert parent_to_final[str(threshold)]['repairs'] == metric['repairs' + suffix]
            assert parent_to_final[str(threshold)]['damages'] == metric['damages' + suffix]
        table.append(dict(arm=arm, stage=stage, optimizer_updates=0 if stage == 'initial_formal' else 3723,
            rec_hits25=hits[0], rec_hits50=hits[1], accuracy25=hits[0] / N * 100, accuracy50=hits[1] / N * 100,
            selected_mask_ious_stored_recount=dict(hits25=metric['mask_hits25'],hits50=metric['mask_hits50'],miou=metric['mask_miou']),
            same_forward_parent_to_final=parent_to_final,
            selected_mask_good_box_bad=sum(mask > .5 >= value for mask, value in zip(masks, values)),
            invalid_selected_reference=sum(not row['arms'][arm]['reference_valid'] for row in rows),
            passes_updated_dual_target=hits[0] >= 5620 and hits[1] >= 4764))
    assert sum(row['parent_iou'] > .25 for row in rows) == receipt['parent_hits25']
    assert sum(row['parent_iou'] > .5 for row in rows) == receipt['parent_hits50']
    return table, dict(npz_count=batches, independent_box_iou_values=N * 256 * len(BOX_NAMES),
        cpu_full256_coverage=coverage, cpu_selected_threshold_flips=flips,
        cpu_full256_oracle_label_mismatches=oracle_mismatches,
        qualified_candidate_available_but_unselected=available_unselected,
        no_qualified_candidate=missing_qualified, selected_query_has_max_native_score=True)


def main():
    assert not OUTPUT.exists()
    intake = read_json(COMPLETE / 'INTAKE.json')
    assert intake['status'] == 'CLOSED_FIT_ARTIFACTS_COLLECTED' and intake['weights_copied'] == 0
    status = read_json(COMPLETE / 'fit_status.json')
    assert status['status'] == 'complete' and status['protected_parents_exact']
    assert status['optimizer_steps_per_arm'] == 3723
    assert (COMPLETE / 'fit_controller.exit').read_text().strip() == '0'
    for item in intake['files']:
        path = COMPLETE / item['name']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    fit = read_json(COMPLETE / 'receipt.json')
    assert fit['status'] == 'complete' and fit['training_steps_per_arm'] == 3723
    assert fit['fit_rows_per_arm'] == 29778 and fit['holdout_rows'] == 6887
    assert fit['fit_seen_exactly_once_per_arm'] and fit['parent_and_box_head_states_exact']
    training = read_rows(COMPLETE / 'train.jsonl')
    assert len(training) == 3723 and [row['step'] for row in training] == list(range(1, 3724))
    seen = [identity for row in training for identity in row['rows']]
    assert len(seen) == len(set(seen)) == 29778 and len(training[-1]['rows']) == 2
    assert all(row['frozen_parent_forwards_per_batch'] == row['final_semantic_head_calls'] == 1 for row in training)
    assert all(row['expanded_positive_queries'] == 0 for row in training)
    for row in training:
        for arm in ARMS:
            value = row['arms'][arm]
            assert value['native_coefficients'] == [5, 1, 10, 2]
            assert value['expanded_positive_queries'] == 0 and value['independent_head_gradients']
            assert np.isfinite(value['loss']) and np.isfinite(value['gradient_norm'])
    for arm in ARMS:
        restored = read_json(COMPLETE / arm / 'formal_restore.json')
        assert restored['status'] == 'pass'
        assert restored['optimizer']['all_keys_moments_steps_and_groups_exact']
    protected_path = ROOT.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal/rows.jsonl'
    protected = read_rows(protected_path)
    assert len(protected) == N
    assert [sum(row['bbs']['iou'] > t for row in protected) for t in (.25, .5)] == [5598, 4848]
    table = [dict(arm='protected_parent',stage='protected',rec_hits25=5598,rec_hits50=4848,
        accuracy25=5598 / N * 100,accuracy50=4848 / N * 100,passes_updated_dual_target=False)]
    stages, checks = {}, {}
    for stage in STAGES:
        stages[stage] = read_rows(COMPLETE / stage / 'rows.jsonl')
        values, checks[stage] = recount(stage, stages[stage])
        table.extend(values)
    before, after = stages['initial_formal'], stages['formal']
    align(before, after)
    align(protected, after)
    candidates = [row for row in table if row['stage'] in ('protected','formal')]
    metric_best = max(candidates,key=lambda row:(row['passes_updated_dual_target'],
        row['rec_hits50'],row['rec_hits25'],row['arm']=='protected_parent'))
    comparisons = dict(same_forward_content_to_box_conditioned=compare(
        [row['arms']['content']['iou'] for row in after],
        [row['arms']['box_conditioned']['iou'] for row in after]),
        independent_pass_initial_to_final={arm: compare(
            [row['arms'][arm]['iou'] for row in before],
            [row['arms'][arm]['iou'] for row in after]) for arm in ARMS},
        historical_protected_to_final={arm: compare([row['bbs']['iou'] for row in protected],
            [row['arms'][arm]['iou'] for row in after]) for arm in ARMS})
    parent_drift = dict(query_changes=sum(old['query'] != new['query'] for old,new in zip(before,after)),
        selected_parent_box_changes=sum(old['parent_box'] != new['parent_box'] for old,new in zip(before,after)),
        initial_hits=[sum(row['parent_iou'] > t for row in before) for t in (.25,.5)],
        final_hits=[sum(row['parent_iou'] > t for row in after) for t in (.25,.5)],
        scope='independent native passes; frozen states do not imply identical floating-point outputs')
    array_drift = {key: dict(changed_elements=0,changed_expressions=0,max_abs_difference=0.0)
        for key in ('scores','parent')}
    for path in sorted((COMPLETE/'initial_formal').glob('batch_*.npz')):
        with np.load(path,allow_pickle=False) as initial, np.load(COMPLETE/'formal'/path.name,allow_pickle=False) as final:
            np.testing.assert_array_equal(initial['row_ids'],final['row_ids'])
            np.testing.assert_array_equal(initial['root_gt'],final['root_gt'])
            n=len(initial['row_ids'])
            for key,counts in array_drift.items():
                changed=initial[key] != final[key]
                counts['changed_elements'] += int(changed.sum())
                counts['changed_expressions'] += int(changed.reshape(n,-1).any(axis=1).sum())
                counts['max_abs_difference'] = max(counts['max_abs_difference'],
                    float(np.abs(initial[key].astype(np.float64)-final[key].astype(np.float64)).max()))
    summary = dict(status='ACTUAL_CLOSED_FULL_FORMAL_CPU_RECOUNT',formal_rows_per_arm=N,
        fit_rows_per_arm=29778,optimizer_updates_per_arm=3723,seed=2027,table=table,
        checks=checks,comparisons=comparisons,frozen_parent_cross_pass_drift=parent_drift,
        metric_best_candidate=metric_best,cross_pass_all256_parent_and_scores_drift=array_drift,
        selection_rule='Only protected checkpoint and trained terminals are eligible. Joint5620/4764 gate first, then strict hits, wide hits; exact ties retain protected parent. Zero-output initial evaluations are diagnostics, not new checkpoint gains.',
        target_passing_trained_candidates=[row for row in candidates if row['stage']=='formal' and row['passes_updated_dual_target']],
        weights_deleted=0,model_or_optimizer_replayed=False,
        new_module_gain_requires_same_forward_parent_and_paired_control_comparison=True,
        three_effective_contributions_established=False,full_goal_complete=False,
        actual_intake_sha256=digest(COMPLETE/'INTAKE.json'))
    OUTPUT.mkdir()
    (OUTPUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=summary['status'],metric_best_candidate=metric_best,
        target_passing_trained_candidates=len(summary['target_passing_trained_candidates']),full_goal_complete=False)),flush=True)


if __name__ == '__main__':
    main()
