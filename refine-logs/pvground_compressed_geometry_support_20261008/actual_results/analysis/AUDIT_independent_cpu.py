"""Fresh auditor's offline checks. No executor analysis import, torch, or network."""
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

W = Path(__file__).resolve().parents[1]
C = W / 'complete_fit'
N = W.parent / 'pvground_mask_support_correction_20261008_v2'
P = N / 'complete_fit/PV-Ground'
H = W.parent / 'pvground_final_quality_20261005/runtime_bundle'
OUT = W / 'analysis'
ARMS = ('content', 'box_conditioned')
BOXES = ('parent', *ARMS)
STAGES = ('initial_formal', 'formal')
COUNT = 9508
reviewed = {}


def read(path, scope='data_contents'):
    path = path.resolve()
    raw = path.read_bytes()
    reviewed[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                               bytes=len(raw), review_scope=scope)
    return raw


def obj(path):
    return json.loads(read(path))


def lines(path):
    return [json.loads(row) for row in read(path).decode('utf-8').splitlines()]


def hits(values):
    return {str(t): int(np.count_nonzero(np.asarray(values) > t)) for t in (.25, .5)}


def overlap(boxes, gt, dtype):
    b = boxes.astype(dtype)
    g = gt.astype(dtype)[:, None, :]
    b0, b1 = b[..., :3] - b[..., 3:] * .5, b[..., :3] + b[..., 3:] * .5
    g0, g1 = g[..., :3] - g[..., 3:] * .5, g[..., :3] + g[..., 3:] * .5
    side = np.clip(np.minimum(b1, g1) - np.maximum(b0, g0), 0, None)
    intersection = side[..., 0] * side[..., 1] * side[..., 2]
    volume_b = b[..., 3] * b[..., 4] * b[..., 5]
    volume_g = g[..., 3] * g[..., 4] * g[..., 5]
    ratio = intersection / (volume_b + volume_g - intersection)
    assert np.isfinite(ratio).all() and (ratio >= 0).all()
    return ratio


def transition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {str(t): dict(before=int((a > t).sum()), after=int((b > t).sum()),
                        repairs=int(((a <= t) & (b > t)).sum()),
                        damages=int(((a > t) & (b <= t)).sum())) for t in (.25, .5)}


def check_stage(stage):
    rows = lines(C / stage / 'rows.jsonl')
    receipt = obj(C / stage / 'receipt.json')
    assert len(rows) == COUNT
    assert [r['row_id'] for r in rows] == list(range(COUNT))
    assert receipt['rows'] == receipt['formal_rows'] == COUNT and receipt['status'] == 'pass'
    assert receipt['rows_sha256'] == reviewed[str((C / stage / 'rows.jsonl').resolve())]['sha256']
    assert receipt['primary_mode'] == 'bbs' and receipt['all256_retained']
    assert receipt['same_selected_query_box_and_mask'] and receipt['frozen_parent_forward_per_batch'] == 1
    assert all(r['parent_forwards'] == r['final_semantic_head_calls'] == 1 for r in rows)
    selected = {name: [] for name in BOXES}
    cover = {name: {str(t): 0 for t in (.25, .5)} for name in BOXES}
    fp32_cover = {name: {str(t): 0 for t in (.25, .5)} for name in BOXES}
    selected_flips = {name: {str(t): 0 for t in (.25, .5)} for name in BOXES}
    max_errors = {name: 0.0 for name in BOXES}
    ranges = {name: {precision: dict(maximum=0.0, greater_than_one=0) for precision in ('float32', 'float64')} for name in BOXES}
    oracle_mismatches = {name: {str(t): {str(k): 0 for k in (16, 32, 64, 256)} for t in (.25, .5)} for name in ARMS}
    oracle_examples = []
    boundary_ties = {str(k): 0 for k in (1, 16, 32, 64)}
    score_arrays, parent_arrays = [], []
    initial_array_disagreement = 0
    arm_array_changed_values = arm_array_changed_rows = 0
    paths = sorted((C / stage).glob('batch_*.npz'))
    assert len(paths) == 1189
    offset = 0
    for path in paths:
        raw = read(path, 'all_npz_arrays_recomputed_and_verified')
        assert path.name == f'batch_{offset:05d}.npz'
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            assert set(z.files) == {'row_ids', 'root_gt', 'scores', *BOXES}
            n = min(8, COUNT - offset)
            np.testing.assert_array_equal(z['row_ids'], np.arange(offset, offset + n))
            truth = z['root_gt']
            current = rows[offset:offset + n]
            assert truth.shape == (n, 6) and (truth[:, 3:] > 0).all()
            np.testing.assert_array_equal(truth, [r['root_box'] for r in current])
            score = z['scores']
            assert score.shape == (n, 256) and np.isfinite(score).all()
            q = np.array([r['query'] for r in current])
            assert ((q >= 0) & (q < 256)).all()
            np.testing.assert_array_equal(score[np.arange(n), q], score.max(1))
            ranking = np.argsort(-score, axis=1, kind='stable')
            sorted_scores = np.take_along_axis(score, ranking, axis=1)
            for k in (1, 16, 32, 64):
                boundary_ties[str(k)] += int((sorted_scores[:, k - 1] == sorted_scores[:, k]).sum())
            score_arrays.append(score.copy())
            parent_arrays.append(z['parent'].copy())
            changed = z['content'] != z['box_conditioned']
            arm_array_changed_values += int(changed.sum())
            arm_array_changed_rows += int(changed.reshape(n, -1).any(1).sum())
            if stage == 'initial_formal':
                initial_array_disagreement += int(changed.sum())
            for name in BOXES:
                boxes = z[name]
                assert boxes.shape == (n, 256, 6)
                assert np.isfinite(boxes).all() and (boxes[..., 3:] > 0).all()
                saved_boxes = [r['parent_box'] if name == 'parent' else r['arms'][name]['box'] for r in current]
                native = np.array([r['parent_iou'] if name == 'parent' else r['arms'][name]['iou'] for r in current])
                np.testing.assert_array_equal(boxes[np.arange(n), q], saved_boxes)
                iou64 = overlap(boxes, truth, np.float64)
                iou32 = overlap(boxes, truth, np.float32)
                for precision, values in (('float64', iou64), ('float32', iou32)):
                    ranges[name][precision]['maximum'] = max(ranges[name][precision]['maximum'], float(values.max()))
                    ranges[name][precision]['greater_than_one'] += int((values > 1).sum())
                choose = iou64[np.arange(n), q]
                selected[name].extend(choose.tolist())
                max_errors[name] = max(max_errors[name], float(np.abs(choose - native).max()))
                for t in (.25, .5):
                    cover[name][str(t)] += int((iou64 > t).any(1).sum())
                    fp32_cover[name][str(t)] += int((iou32 > t).any(1).sum())
                    selected_flips[name][str(t)] += int(((choose > t) != (native > t)).sum())
                    if name in ARMS:
                        ordered_iou = np.take_along_axis(iou64, ranking, axis=1)
                        label = 'oracle25' if t == .25 else 'oracle50'
                        for j, k in enumerate((16, 32, 64, 256)):
                            actual = (ordered_iou[:, :k] > t).any(1)
                            saved = np.array([r['arms'][name][label][j] for r in current], dtype=bool)
                            mismatch = actual != saved
                            oracle_mismatches[name][str(t)][str(k)] += int(mismatch.sum())
                            for idx in np.flatnonzero(mismatch):
                                if len(oracle_examples) < 30:
                                    oracle_examples.append(dict(row_id=current[idx]['row_id'], arm=name,
                                                                threshold=t, topk=k, cpu=bool(actual[idx]), native=bool(saved[idx])))
            offset += n
    assert offset == COUNT and initial_array_disagreement == 0
    native = {name: np.array([r['parent_iou'] if name == 'parent' else r['arms'][name]['iou'] for r in rows]) for name in BOXES}
    metrics = {}
    for name in BOXES:
        assert np.isfinite(native[name]).all() and (native[name] >= 0).all()
        expected = {str(t): receipt[f'parent_hits{suffix}'] if name == 'parent' else receipt['metrics'][name][f'rec_hits{suffix}']
                    for t, suffix in ((.25, '25'), (.5, '50'))}
        assert hits(native[name]) == expected
        metrics[name] = dict(native_selected_hits=hits(native[name]), cpu_selected_hits=hits(selected[name]),
                             native_selected_iou_range=dict(minimum=float(native[name].min()), maximum=float(native[name].max()),
                                                            greater_than_one=int((native[name] > 1).sum()),
                                                            greater_than_one_rows=np.flatnonzero(native[name] > 1).tolist()),
                             full256_oracle_hits=cover[name], cpu_float32_full256_oracle_hits=fp32_cover[name])
        if name in ARMS:
            masks = np.array([r['arms'][name]['mask_iou'] for r in rows])
            assert np.isfinite(masks).all() and ((masks >= 0) & (masks <= 1)).all()
            m = receipt['metrics'][name]
            assert hits(masks) == {'0.25': m['mask_hits25'], '0.5': m['mask_hits50']}
            assert abs(float(masks.mean()) * 100 - m['mask_miou']) < 1e-9
            changes = transition(native['parent'], native[name])
            for t, suffix in ((.25, '25'), (.5, '50')):
                assert changes[str(t)]['repairs'] == m['repairs' + suffix]
                assert changes[str(t)]['damages'] == m['damages' + suffix]
            metrics[name].update(mask_saved_iou_recount=dict(hits=hits(masks), miou=float(masks.mean()) * 100),
                                 same_forward_parent_transition=changes,
                                 invalid_selected_references=sum(not r['arms'][name]['reference_valid'] for r in rows))
    report = dict(rows=COUNT, scan_count=len({r['scan_id'] for r in rows}),
                  physical_scene_count=len({r['scan_id'].split('_')[0] for r in rows}),
                  npz_count=len(paths), recomputed_candidate_ious=COUNT * 256 * 3,
                  metrics=metrics, selected_cpu_native_threshold_mismatches=selected_flips,
                  max_selected_iou_absolute_error=max_errors, candidate_iou_range=ranges, oracle_mismatches=oracle_mismatches,
                  oracle_mismatch_examples=oracle_examples, score_boundary_ties=boundary_ties,
                  arm_all256_changed_values=arm_array_changed_values, arm_all256_changed_expressions=arm_array_changed_rows,
                  same_forward_control_treatment_transition=transition(native['content'], native['box_conditioned']))
    return report, rows, native, dict(scores=np.concatenate(score_arrays), parent=np.concatenate(parent_arrays))


def main():
    start = time.monotonic()
    spec = obj(W / 'pair_spec.json')
    summary = obj(OUT / 'SUMMARY.json')
    goals = obj(W / 'CURRENT_RESEARCH_GOALS.json')
    intake = obj(C / 'INTAKE.json')
    assert intake['status'] == 'CLOSED_FIT_ARTIFACTS_COLLECTED'
    assert intake['files_copied'] == len(intake['files']) == 2423
    assert len({r['name'] for r in intake['files']}) == 2423
    assert intake['total_bytes'] == sum(r['bytes'] for r in intake['files']) == 230030167
    assert intake['weights_copied'] == 0 and not intake['model_or_optimizer_replayed']
    expected_files = {r['name'] for r in intake['files']}
    actual_files = {p.relative_to(C).as_posix() for p in C.rglob('*') if p.is_file()}
    assert actual_files == expected_files | {'INTAKE.json'}
    for item in intake['files']:
        path = C / item['name']
        assert C.resolve() in path.resolve().parents
        raw = read(path, 'archive_inventory_bytes_and_sha256')
        assert len(raw) == item['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['name']
    manifest = obj(W / 'collector_recovery/REMOTE_MANIFEST.json')
    assert manifest['files'] == intake['files']
    recovery = obj(W / 'collector_recovery/RECOVERY_COMPLETE.json')
    partial = obj(W / 'collector_recovery/PARTIAL_CLASSIFICATION.json')
    assert partial['verified_files'] + partial['remaining_files'] == 2423
    assert partial['verified_bytes'] + partial['remaining_bytes'] == 230030167
    assert recovery['resumed_files'] == partial['remaining_files'] == 294
    assert recovery['verified_original_files_retained'] == partial['verified_files'] == 2129
    assert summary['actual_intake_sha256'] == reviewed[str((C / 'INTAKE.json').resolve())]['sha256']
    status = obj(C / 'fit_status.json')
    wait = obj(W / 'fit_wait.json')
    assert status == wait['terminal']['status'] and wait['observer_closed']
    assert not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0
    assert status['status'] == 'complete' and status['exit_code'] == 0 and status['protected_parents_exact']
    assert status['completed_modes'] == ['initial_formal', 'train', 'formal']
    for name in ('fit_controller.exit', 'initial_formal.exit', 'train.exit', 'formal.exit'):
        assert read(C / name).decode().strip() == '0'
    fit = obj(C / 'receipt.json')
    assert fit['spec_sha256'] == reviewed[str((W / 'pair_spec.json').resolve())]['sha256']
    assert fit['training_steps_per_arm'] == status['optimizer_steps_per_arm'] == 3723
    assert fit['fit_rows_per_arm'] == 29778 and fit['holdout_rows'] == 6887
    assert fit['status'] == 'complete' and fit['fit_seen_exactly_once_per_arm'] and fit['parent_and_box_head_states_exact']
    source_bindings = []
    for name, expected in spec['new_runner_files'].items():
        raw = read(W / name, 'semantic_source_review')
        assert hashlib.sha256(raw).hexdigest() == expected
        assert read(C / name, 'archived_source_identical_to_reviewed_source') == raw
        source_bindings.append(name)
    assert read(C / 'pair_spec.json') == read(W / 'pair_spec.json')
    assert read(C / 'controller.py') == read(W / 'controller.py', 'semantic_source_review')
    imports = obj(C / 'imports.json')
    imported_paths = {'models.pv_ground': P / 'models/pv_ground.py', 'models.losses': P / 'models/losses.py',
                      'main_utils': P / 'main_utils.py', 'prepare_data': P / 'prepare_data.py',
                      'native_evaluator': P / 'src/grounding_evaluator.py',
                      'src.joint_det_dataset': W.parent / 'pvground_g_p2_20261002/complete/source/joint_det_dataset.py'}
    for name, path in imported_paths.items():
        assert hashlib.sha256(read(path, 'import_binding_and_semantic_source_review')).hexdigest() == imports['sha256'][name]
    for name, expected in spec['runner_files'].items():
        raw = read(H / name, 'helper_source_hash_binding')
        assert hashlib.sha256(raw).hexdigest() == expected
    train = lines(C / 'train.jsonl')
    assert len(train) == 3723 and [r['step'] for r in train] == list(range(1, 3724))
    sample_ids = [i for r in train for i in r['rows']]
    assert len(sample_ids) == len(set(sample_ids)) == 29778
    assert [len(r['rows']) for r in train] == [8] * 3722 + [2]
    training = {}
    loss_error = {arm: 0.0 for arm in ARMS}
    for r in train:
        n = len(r['rows'])
        assert r['frozen_parent_forwards_per_batch'] == r['final_semantic_head_calls'] == 1
        assert r['expanded_positive_queries'] == 0
        assigned = r['original_parent_assignment']
        assert len(assigned) == n
        for a in assigned:
            assert len(a['queries']) == len(a['actual_gt_ids'])
            assert len(set(a['queries'])) == len(a['queries'])
            assert all(0 <= q < 256 for q in a['queries'])
        for arm in ARMS:
            a = r['arms'][arm]
            assert a['native_coefficients'] == [5, 1, 10, 2] and a['correspondence'] == 'frozen_parent_actual_gt'
            assert a['independent_head_gradients'] and a['expanded_positive_queries'] == 0
            assert a['matched_queries'] == sum(len(x['queries']) for x in assigned)
            assert a['valid_gt'] >= a['matched_queries'] > 0
            for x in ('query_focal', 'query_dice', 'fused_focal', 'fused_dice', 'loss', 'gradient_norm'):
                assert math.isfinite(a[x]) and a[x] >= 0
            expected_loss = 5 * a['query_focal'] + a['query_dice'] + 10 * a['fused_focal'] + 2 * a['fused_dice']
            loss_error[arm] = max(loss_error[arm], abs(expected_loss - a['loss']))
            assert math.isclose(expected_loss, a['loss'], rel_tol=2e-6, abs_tol=2e-6)
    for arm in ARMS:
        values = np.array([r['arms'][arm]['gradient_norm'] for r in train])
        loss = np.array([r['arms'][arm]['loss'] for r in train])
        training[arm] = dict(updates=len(train), samples=len(sample_ids), clipped_updates=int((values > .1).sum()),
                             gradient_min=float(values.min()), gradient_max=float(values.max()),
                             loss_min=float(loss.min()), loss_max=float(loss.max()), loss_mean=float(loss.mean()),
                             max_coefficient_reconstruction_error=loss_error[arm])
        restored = obj(C / arm / 'formal_restore.json')
        assert restored['status'] == 'pass' and restored['optimizer']['all_keys_moments_steps_and_groups_exact']
        assert restored['optimizer']['moment_and_step_states'] == 10 and restored['optimizer']['param_groups'] == 1
    holdout = {}
    for stage in ('initial', 'terminal'):
        rows = lines(C / stage / 'rows.jsonl')
        receipt = obj(C / stage / 'receipt.json')
        assert len(rows) == receipt['rows'] == 6887 and receipt['formal_rows'] == 0
        assert len({r['row_id'] for r in rows}) == 6887
        assert not set(sample_ids).intersection(r['row_id'] for r in rows)
        assert set(sample_ids) | {r['row_id'] for r in rows} == set(range(36665))
        assert receipt['rows_sha256'] == reviewed[str((C / stage / 'rows.jsonl').resolve())]['sha256']
        for arm in ARMS:
            assert hits([r['arms'][arm]['iou'] for r in rows]) == {
                '0.25': receipt['metrics'][arm]['rec_hits25'], '0.5': receipt['metrics'][arm]['rec_hits50']}
        holdout[stage] = rows
    for a, b in zip(holdout['initial'], holdout['terminal']):
        assert all(a[k] == b[k] for k in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))
    stages, allrows, native, arrays = {}, {}, {}, {}
    for stage in STAGES:
        stages[stage], allrows[stage], native[stage], arrays[stage] = check_stage(stage)
    historical = lines(N / 'complete_fit/formal/rows.jsonl')
    retained = obj(N / 'postrun/RETAINED_BEST.json')
    assert len(historical) == COUNT
    assert hits([r['arms']['content']['iou'] for r in historical]) == {'0.25': 5598, '0.5': 4856}
    for a, b, c in zip(historical, allrows['initial_formal'], allrows['formal']):
        assert all(a[k] == b[k] == c[k] for k in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))
    assert retained['checkpoint_sha256'] == spec['warm_support_terminal_sha256'] == summary['protected_model_checkpoint_sha256']
    assert summary['protected_formal_rows_sha256'] == reviewed[str((N / 'complete_fit/formal/rows.jsonl').resolve())]['sha256']
    drift = {}
    for key in ('scores', 'parent'):
        a, b = arrays['initial_formal'][key], arrays['formal'][key]
        changed = a != b
        drift[key] = dict(changed_elements=int(changed.sum()), changed_expressions=int(changed.reshape(COUNT, -1).any(1).sum()),
                          max_abs_difference=float(np.abs(a.astype(float) - b.astype(float)).max()))
    drift['queries'] = sum(a['query'] != b['query'] for a, b in zip(allrows['initial_formal'], allrows['formal']))
    drift['selected_parent_boxes'] = sum(a['parent_box'] != b['parent_box'] for a, b in zip(allrows['initial_formal'], allrows['formal']))
    assert {k: drift[k] for k in ('scores', 'parent')} == summary['cross_pass_all256_parent_and_scores_drift']
    assert drift['queries'] == summary['frozen_parent_cross_pass_drift']['query_changes']
    assert drift['selected_parent_boxes'] == summary['frozen_parent_cross_pass_drift']['selected_parent_box_changes']
    historical_hits = hits([r['arms']['content']['iou'] for r in historical])
    candidates = [('protected_parent', historical_hits)] + [(arm, hits(native['formal'][arm])) for arm in ARMS]
    gate = spec['candidate_gate_hits']
    assert gate == [5620, 4764]
    key = lambda entry: (entry[1]['0.25'] >= gate[0] and entry[1]['0.5'] >= gate[1], entry[1]['0.5'], entry[1]['0.25'], entry[0] == 'protected_parent')
    winner = max(candidates, key=key)
    assert winner[0] == summary['metric_best_candidate']['arm']
    latest = [math.floor(COUNT * .595) + 1, math.floor(COUNT * .51) + 1]
    assert latest == [goals['scanrefer']['minimum_hits025'], goals['scanrefer']['minimum_hits050']] == [5658, 4850]
    for row in summary['table']:
        if row['stage'] == 'protected':
            h = historical_hits
        else:
            h = hits(native[row['stage']][row['arm']])
        assert [row['rec_hits25'], row['rec_hits50']] == [h['0.25'], h['0.5']]
        assert math.isclose(row['accuracy25'], 100 * h['0.25'] / COUNT, abs_tol=1e-12)
        assert math.isclose(row['accuracy50'], 100 * h['0.5'] / COUNT, abs_tol=1e-12)
    report = dict(status='PASS_DETERMINISTIC_CHECKS_WITH_EXPLICIT_SCOPE_LIMITS', generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=sys.version, numpy=np.__version__, command='uv run --offline --with numpy python -B -X utf8 analysis/AUDIT_independent_cpu.py',
                  execution_scope='ACTUAL_CLOSED_TRAINED_PAIR', neural_forwards=0, torch_imports=0, ssh_calls=0,
                  archive=dict(files=2423, bytes=230030167, npz_files=2378, exact_inventory=True, every_hash_and_size_matches=True),
                  source_bindings=source_bindings, training=training,
                  fit_id_partition=dict(unique_samples=len(set(sample_ids)), total_source_train_rows=36665, holdout_rows=6887, exact_disjoint_exhaustive_ids=True,
                                        caveat='Original raw split manifest and raw dataset were not independently rebuilt.'),
                  stages=stages, original_candidate_ious_recomputed=COUNT * 256 * 3 * 2,
                  identity_alignment=dict(historical_initial_final_rows=COUNT, keys=['row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'], all_exact=True),
                  cross_pass_drift=drift, independent_pass_initial_to_final={arm: transition(native['initial_formal'][arm], native['formal'][arm]) for arm in ARMS},
                  historical_to_final={arm: transition([r['arms']['content']['iou'] for r in historical], native['formal'][arm]) for arm in ARMS},
                  selection=dict(historical_gate=gate, metric_best=winner[0], metric_best_hits=winner[1], current_user_minimum_hits=latest,
                                 historical_gate_met=False, current_user_joint_goal_met=False,
                                 best_tie_between_new_arms=True, new_arm_tiebreak='ARMS order selects content; historic rule does not separately prescribe new-arm ties'),
                  raw_mask_recomputed=False, raw_dataset_rebuilt=False, gpu_restoration_executed=False, terminal_weights_independently_loaded=False,
                  elapsed_seconds=time.monotonic() - start)
    (OUT / 'AUDIT_independent_cpu.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (OUT / 'AUDIT_machine_reviewed_files.json').write_text(json.dumps(list(reviewed.values()), indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'archive', 'training', 'original_candidate_ious_recomputed', 'selection', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
