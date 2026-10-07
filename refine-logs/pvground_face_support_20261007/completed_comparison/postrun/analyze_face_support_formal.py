"""CPU recount of the closed shared-parent pair; never rerun a model."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / 'complete_fit'
OUTPUT = ROOT / 'analysis'
ARMS = ('face_center', 'face_region')
STAGES = ('initial_formal', 'formal')
BOX_NAMES = ('original_prior', 'reference', 'final_face_center', 'final_face_region')
N = 9508


def read_json(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def box_iou(boxes, truth):
    boxes = np.asarray(boxes, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    low = np.maximum(boxes[..., :3] - boxes[..., 3:] / 2, truth[..., :3] - truth[..., 3:] / 2)
    high = np.minimum(boxes[..., :3] + boxes[..., 3:] / 2, truth[..., :3] + truth[..., 3:] / 2)
    intersection = np.maximum(high - low, 0).prod(axis=-1)
    union = boxes[..., 3:].prod(axis=-1) + truth[..., 3:].prod(axis=-1) - intersection
    return intersection / union


def compare(values_before, values_after):
    assert len(values_before) == len(values_after) == N
    return {str(t): dict(before_hits=sum(value > t for value in values_before),
                        after_hits=sum(value > t for value in values_after),
                        repairs=sum(old <= t < new for old, new in zip(values_before, values_after)),
                        damages=sum(new <= t < old for old, new in zip(values_before, values_after)),
                        net=sum(new > t for new in values_after) - sum(old > t for old in values_before))
            for t in (.25, .5)}


def align(before, after):
    assert len(before) == len(after) == N
    for old, new in zip(before, after):
        assert all(old[key] == new[key] for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))


def recount(stage, rows):
    directory = COMPLETE / stage
    receipt = read_json(directory / 'receipt.json')
    assert receipt['status'] == 'pass' and receipt['rows'] == receipt['formal_rows'] == N
    assert receipt['shared_parent_forward_per_batch'] == 1
    assert receipt['native_selected_query_and_masks_shared'] and receipt['primary_mode'] == 'bbs'
    assert digest(directory / 'rows.jsonl') == receipt['rows_sha256']
    assert [row['row_id'] for row in rows] == list(range(N))
    assert all(row['frozen_parent_forwards'] == row['native_head_calls'] == 1 for row in rows)
    flips = {name: {str(t): 0 for t in (.25, .5)} for name in BOX_NAMES}
    coverage = {name: {str(t): 0 for t in (.25, .5)} for name in BOX_NAMES}
    oracle_mismatches = {arm: {str(t): 0 for t in (.25, .5)} for arm in ARMS}
    hit_available_unselected = {arm: {str(t): 0 for t in (.25, .5)} for arm in ARMS}
    missing_qualified = {arm: {str(t): 0 for t in (.25, .5)} for arm in ARMS}
    invalid_selected = invalid_all = offset = batches = 0
    for path in sorted(directory.glob('batch_*.npz')):
        assert path.name == 'batch_%05d.npz' % offset
        with np.load(path, allow_pickle=False) as batch:
            assert set(batch.files) == {'row_ids', 'root_gt', 'original_prior', 'reference', 'reference_valid',
                                        'scores', 'final_face_center', 'final_face_region'}
            n = len(batch['row_ids'])
            assert n == min(8, N - offset)
            assert np.array_equal(batch['row_ids'], np.arange(offset, offset + n))
            assert batch['root_gt'].shape == (n, 6)
            assert batch['scores'].shape == batch['reference_valid'].shape == (n, 256)
            assert np.isfinite(batch['scores']).all()
            current = rows[offset:offset + n]
            truth = np.asarray([row['root_box'] for row in current])
            np.testing.assert_array_equal(batch['root_gt'], truth)
            indices = np.arange(n)
            queries = np.asarray([row['query'] for row in current])
            assert np.array_equal(batch['scores'][indices, queries], batch['scores'].max(axis=1))
            for name in BOX_NAMES:
                boxes = batch[name]
                assert boxes.shape == (n, 256, 6) and np.isfinite(boxes).all() and (boxes[..., 3:] > 0).all()
                if name == 'original_prior':
                    saved_boxes = [row['coarse_box'] for row in current]
                    saved_ious = [row['coarse_iou'] for row in current]
                elif name == 'reference':
                    saved_boxes = [row['reference_box'] for row in current]
                    saved_ious = [row['reference_iou'] for row in current]
                else:
                    arm = name[len('final_'):]
                    saved_boxes = [row['arms'][arm]['box'] for row in current]
                    saved_ious = [row['arms'][arm]['iou'] for row in current]
                np.testing.assert_array_equal(boxes[indices, queries], saved_boxes)
                ious = box_iou(boxes, truth[:, None, :])
                for threshold in (.25, .5):
                    valid = (ious > threshold).any(axis=1)
                    coverage[name][str(threshold)] += int(valid.sum())
                    selected = ious[indices, queries] > threshold
                    flips[name][str(threshold)] += sum(bool(value) != bool(saved > threshold)
                                                      for value, saved in zip(selected, saved_ious))
                    if name.startswith('final_'):
                        label = 'oracle25' if threshold == .25 else 'oracle50'
                        oracle_mismatches[arm][str(threshold)] += sum(bool(value) != bool(row['arms'][arm][label][-1])
                                                                      for value, row in zip(valid, current))
                        hit_available_unselected[arm][str(threshold)] += int((valid & ~selected).sum())
                        missing_qualified[arm][str(threshold)] += int((~valid).sum())
            if stage == 'initial_formal':
                for arm in ARMS:
                    np.testing.assert_array_equal(batch['final_' + arm], batch['reference'])
            invalid_selected += int((~batch['reference_valid'][indices, queries]).sum())
            invalid_all += int((~batch['reference_valid']).sum())
            assert [bool(value) for value in batch['reference_valid'][indices, queries]] == [row['reference_valid'] for row in current]
            offset += n
            batches += 1
    assert offset == N and batches == 1189
    mask = dict(hits25=sum(row['mask_iou'] > .25 for row in rows), hits50=sum(row['mask_iou'] > .5 for row in rows),
                miou=sum(row['mask_iou'] for row in rows) / N * 100)
    table = []
    for arm in ARMS:
        values = [row['arms'][arm]['iou'] for row in rows]
        hits = [sum(value > threshold for value in values) for threshold in (.25, .5)]
        assert hits == [receipt['metrics'][arm]['rec_hits25'], receipt['metrics'][arm]['rec_hits50']]
        assert mask['hits25'] == receipt['metrics'][arm]['mask_hits25']
        assert mask['hits50'] == receipt['metrics'][arm]['mask_hits50']
        assert abs(mask['miou'] - receipt['metrics'][arm]['mask_miou']) < 1e-9
        table.append(dict(arm=arm, stage=stage, rec_hits25=hits[0], rec_hits50=hits[1],
            accuracy25=hits[0] / N * 100, accuracy50=hits[1] / N * 100,
            optimizer_updates=0 if stage == 'initial_formal' else 3723,
            zero_update_architecture=stage == 'initial_formal', mask_metrics_stored_recount=mask,
            internal_reference_to_final=compare([row['reference_iou'] for row in rows], values),
            internal_prior_to_final=compare([row['coarse_iou'] for row in rows], values),
            selected_mask_good_box_bad=sum(row['mask_iou'] > .5 >= row['arms'][arm]['iou'] for row in rows)))
    for threshold, key in ((.25, 'same_forward_reference_hits25'), (.5, 'same_forward_reference_hits50')):
        assert sum(row['reference_iou'] > threshold for row in rows) == receipt[key]
    return table, dict(npz_count=batches, independent_box_iou_values=N * 256 * len(BOX_NAMES),
        cpu_full256_coverage=coverage, cpu_selected_threshold_flips_by_box_type=flips,
        cpu_full256_oracle_label_mismatches=oracle_mismatches,
        qualified_candidate_available_but_unselected=hit_available_unselected,
        no_qualified_candidate=missing_qualified, invalid_reference_selected=invalid_selected,
        invalid_reference_all256=invalid_all, selected_query_has_max_native_score=True)


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
    assert fit['status'] == 'complete' and fit['training_steps_per_arm'] == 3723 and fit['fit_rows_per_arm'] == 29778
    assert fit['fit_seen_exactly_once_per_arm'] and fit['parent_and_zero_R_states_exact']
    assert fit['head_parameters_per_arm'] == 456102 and fit['deployed_heads_per_model'] == 1
    assert fit['reference_keep_weight'] == 0 and fit['extra_geometry_weight'] == 1
    training = read_rows(COMPLETE / 'train.jsonl')
    assert len(training) == 3723 and [row['step'] for row in training] == list(range(1, 3724))
    fit_ids = [identity for row in training for identity in row['rows']]
    assert len(fit_ids) == len(set(fit_ids)) == 29778 and len(training[-1]['rows']) == 2
    parent = read_rows(ROOT.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal/rows.jsonl')
    assert [sum(row['bbs']['iou'] > t for row in parent) for t in (.25, .5)] == [5598, 4848]
    table = [dict(arm='protected_geometry_parent', stage='protected', optimizer_updates=0,
                  rec_hits25=5598, rec_hits50=4848, zero_update_architecture=True)]
    data = {}
    stages = {}
    for stage in STAGES:
        rows = read_rows(COMPLETE / stage / 'rows.jsonl')
        align(parent, rows)
        entries, stages[stage] = recount(stage, rows)
        for entry in entries:
            entry['parent_delta'] = compare([row['bbs']['iou'] for row in parent],
                                           [row['arms'][entry['arm']]['iou'] for row in rows])
            entry['parent_selected_query_changes'] = sum(old['bbs']['query'] != new['query'] for old, new in zip(parent, rows))
        table.extend(entries)
        data[stage] = rows
    align(data['initial_formal'], data['formal'])
    common_drift = {key: sum(old[key] != new[key] for old, new in zip(data['initial_formal'], data['formal']))
                    for key in ('query', 'mask_iou', 'coarse_box', 'reference_box', 'reference_valid')}
    shared_drift = {key: dict(different_elements=0, expressions_changed=0, maximum_absolute_difference=0.0)
                    for key in ('scores', 'original_prior', 'reference', 'reference_valid')}
    for path in sorted((COMPLETE / 'initial_formal').glob('batch_*.npz')):
        with np.load(path, allow_pickle=False) as initial, np.load(COMPLETE / 'formal' / path.name, allow_pickle=False) as terminal:
            np.testing.assert_array_equal(initial['row_ids'], terminal['row_ids'])
            np.testing.assert_array_equal(initial['root_gt'], terminal['root_gt'])
            n = len(initial['row_ids'])
            for key, counts in shared_drift.items():
                assert initial[key].shape == terminal[key].shape and initial[key].dtype == terminal[key].dtype
                changed = initial[key] != terminal[key]
                counts['different_elements'] += int(changed.sum())
                counts['expressions_changed'] += int(changed.reshape(n, -1).any(axis=1).sum())
                maximum = np.abs(initial[key].astype(np.float64) - terminal[key].astype(np.float64)).max()
                counts['maximum_absolute_difference'] = max(counts['maximum_absolute_difference'], float(maximum))
    arm_effect = {stage: compare([row['arms'][ARMS[0]]['iou'] for row in data[stage]],
                                [row['arms'][ARMS[1]]['iou'] for row in data[stage]]) for stage in STAGES}
    adaptation = {arm: compare([row['arms'][arm]['iou'] for row in data['initial_formal']],
                               [row['arms'][arm]['iou'] for row in data['formal']]) for arm in ARMS}
    eligible = [row for row in table if row['stage'] in ('protected', 'formal')]
    best = max(eligible, key=lambda row: (row['rec_hits25'] >= 5620 and row['rec_hits50'] >= 4764,
        row['rec_hits50'], row['rec_hits25'], row['arm'] == 'protected_geometry_parent'))
    summary = dict(status='ACTUAL_CLOSED_SHARED_PARENT_FACE_PAIR_RECOUNTED', table=table,
        cpu_stage_recounts=stages, arms_compared=arm_effect, initial_to_trained=adaptation,
        cross_stage_selected_common_drift=common_drift,
        cross_stage_all256_shared_array_drift=shared_drift,
        training_extra_scope={arm: dict(candidate_occurrences=sum(row['arms'][arm]['extra_counts']['extra_candidates'] for row in training),
            outside_face_occurrences=sum(row['arms'][arm]['extra_counts']['extra_boundary_outside'] for row in training)) for arm in ARMS},
        metric_best_candidate=best, formal_rows_per_result=N, seed=2027, no_multiseed=True,
        strict_target_gap=max(0, 4764 - best['rec_hits50']), wide_target_gap=max(0, 5620 - best['rec_hits25']),
        same_checkpoint_scan_gate_passed=best['rec_hits25'] >= 5620 and best['rec_hits50'] >= 4764,
        three_effective_modules_established=False, retained_hidden_prior_updates=11169,
        retained_hidden_total_updates_at_terminal=14892, reset_output_total_updates_at_terminal=3723,
        source_reset_output_tensors=2, parent_and_R_frozen=True,
        cpu_scope='All256 stored prior/reference/two final boxes, scores and real root boxes; no raw Mask recomputation or model replay',
        independent_iou_values=sum(stage['independent_box_iou_values'] for stage in stages.values()),
        fresh_terminal_audit_pending=True, selected_checkpoint_restore_pending=True, weight_retention_executed=False,
        selection_rule='Only the protected checkpoint and two actual trained terminals are eligible. Initial-formal neutral rows are diagnostics, with no new checkpoint. Joint5620/4764 gate first, then strict hits then wide hits; exact ties retain protected parent. If none passes jointly, same ordering among eligible candidates.')
    OUTPUT.mkdir()
    (OUTPUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(table=[{key: row[key] for key in ('arm', 'stage', 'rec_hits25', 'rec_hits50')} for row in table],
        metric_best={key: best[key] for key in ('arm', 'stage', 'rec_hits25', 'rec_hits50')}, audit_pending=True, weights_changed=False)))


if __name__ == '__main__':
    main()
