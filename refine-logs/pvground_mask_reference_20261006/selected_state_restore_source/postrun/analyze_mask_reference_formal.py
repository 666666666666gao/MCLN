"""Recount closed actual artifacts; keep zero-update and trained results separate."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / 'complete_fit'
OUTPUT = ROOT / 'analysis'
ARMS = ('native_reference', 'fused_mask_reference')
STAGES = ('initial_formal', 'formal')


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


def pair(before, after):
    assert len(before) == len(after) == 9508
    for old, new in zip(before, after):
        assert all(old[key] == new[key] for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))


def difference(before, after):
    pair(before, after)
    result = {'selected_query_changes': sum(old['bbs']['query'] != new['bbs']['query'] for old, new in zip(before, after))}
    for threshold in (.25, .5):
        repair = sum(old['bbs']['iou'] <= threshold < new['bbs']['iou'] for old, new in zip(before, after))
        damage = sum(new['bbs']['iou'] <= threshold < old['bbs']['iou'] for old, new in zip(before, after))
        result[str(threshold)] = {'repairs': repair, 'damages': damage, 'net': repair - damage}
    return result


def internal(rows, before, after):
    result = {}
    for threshold in (.25, .5):
        repair = sum(row['bbs'][before] <= threshold < row['bbs'][after] for row in rows)
        damage = sum(row['bbs'][after] <= threshold < row['bbs'][before] for row in rows)
        result[str(threshold)] = dict(before_hits=sum(row['bbs'][before] > threshold for row in rows),
            after_hits=sum(row['bbs'][after] > threshold for row in rows), repairs=repair, damages=damage, net=repair-damage)
    return result


def recount(arm, stage, rows):
    directory = COMPLETE / arm / stage
    receipt = read_json(directory / 'receipt.json')
    assert receipt['status'] == 'pass' and receipt['rows'] == receipt['formal_rows'] == 9508
    assert digest(directory / 'rows.jsonl') == receipt['rows_sha256']
    assert [row['row_id'] for row in rows] == list(range(9508))
    hits = [sum(row['bbs']['iou'] > threshold for row in rows) for threshold in (.25, .5)]
    assert hits == [receipt['metrics']['bbs']['rec_hits25'], receipt['metrics']['bbs']['rec_hits50']]
    selected_flips = {name: {str(t): 0 for t in (.25, .5)} for name in ('original_prior', 'reference', 'final')}
    oracle_mismatches = {str(t): 0 for t in (.25, .5)}
    totals = {name: {str(t): 0 for t in (.25, .5)} for name in ('original_prior', 'reference', 'final')}
    invalid_selected = invalid_all = mask_good_box_bad = 0
    row_offset = 0
    for path in sorted(directory.glob('batch_*.npz')):
        with np.load(path, allow_pickle=False) as batch:
            n = len(batch['row_ids'])
            assert np.array_equal(batch['row_ids'], np.arange(row_offset, row_offset+n))
            assert batch['root_gt'].shape == (n, 6)
            assert batch['scores'].shape == batch['reference_valid'].shape == (n, 256)
            assert np.isfinite(batch['scores']).all()
            current_rows = rows[row_offset:row_offset+n]
            truth = np.asarray([row['root_box'] for row in current_rows])
            np.testing.assert_array_equal(batch['root_gt'], truth)
            queries = np.asarray([row['bbs']['query'] for row in current_rows])
            indices = np.arange(n)
            assert np.array_equal(batch['scores'][indices, queries], batch['scores'].max(axis=1))
            for name, saved_box, saved_iou in (('original_prior', 'coarse_box', 'coarse_iou'),
                                             ('reference', 'reference_box', 'reference_iou'),
                                             ('final', 'box', 'iou')):
                boxes = batch[name]
                assert boxes.shape == (n, 256, 6) and np.isfinite(boxes).all() and (boxes[..., 3:] > 0).all()
                np.testing.assert_array_equal(boxes[indices, queries], [row['bbs'][saved_box] for row in current_rows])
                ious = box_iou(boxes, truth[:, None, :])
                for threshold in (.25, .5):
                    totals[name][str(threshold)] += int((ious > threshold).any(axis=1).sum())
                    selected_flips[name][str(threshold)] += sum(bool(value > threshold) != bool(row['bbs'][saved_iou] > threshold)
                        for value, row in zip(ious[indices, queries], current_rows))
                    if name == 'final':
                        oracle_mismatches[str(threshold)] += sum(bool(value) != bool(row['bbs']['oracle'+('25' if threshold==.25 else '50')][-1])
                            for value, row in zip((ious > threshold).any(axis=1), current_rows))
            if stage == 'initial_formal':
                np.testing.assert_array_equal(batch['final'], batch['reference'])
            if arm == 'native_reference':
                np.testing.assert_array_equal(batch['reference'], batch['original_prior'])
            invalid_all += int((~batch['reference_valid']).sum())
            invalid_selected += int((~batch['reference_valid'][indices, queries]).sum())
            assert [bool(value) for value in batch['reference_valid'][indices, queries]] == [row['bbs']['reference_valid'] for row in current_rows]
            mask_good_box_bad += sum(row['bbs']['mask_iou'] > .5 >= row['bbs']['iou'] for row in current_rows)
            row_offset += n
    assert row_offset == 9508
    return dict(arm=arm, stage=stage, optimizer_updates=0 if stage=='initial_formal' else 3723,
        zero_update_architecture=stage=='initial_formal', rec_hits25=hits[0], rec_hits50=hits[1],
        accuracy25=hits[0]/9508*100, accuracy50=hits[1]/9508*100,
        mask_metrics_stored_recount=dict(hits25=sum(row['bbs']['mask_iou']>.25 for row in rows),
            hits50=sum(row['bbs']['mask_iou']>.5 for row in rows), miou=sum(row['bbs']['mask_iou'] for row in rows)/9508*100),
        internal_prior_to_reference=internal(rows, 'coarse_iou', 'reference_iou'),
        internal_reference_to_final=internal(rows, 'reference_iou', 'iou'),
        cpu_full256_oracle=totals, cpu_selected_threshold_flips_by_box_type=selected_flips,
        cpu_full256_oracle_label_mismatches=oracle_mismatches,
        selected_mask_good_box_bad=mask_good_box_bad, invalid_reference_selected=invalid_selected,
        invalid_reference_all256=invalid_all, selected_query_has_max_native_score=True)


def main():
    assert not OUTPUT.exists()
    status = read_json(COMPLETE / 'fit_status.json')
    assert status['status'] == 'complete' and status['protected_parents_exact']
    assert (COMPLETE / 'fit_controller.exit').read_text().strip() == '0'
    intake = read_json(COMPLETE / 'INTAKE.json')
    assert intake['weights_copied'] == 0
    for item in intake['files']:
        path = COMPLETE / item['name']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    parent = read_rows(ROOT.parent / 'pvground_auxiliary_target_20261005/complete/control/formal/rows.jsonl')
    assert [sum(row['bbs']['iou'] > threshold for row in parent) for threshold in (.25,.5)] == [5616,4511]
    table = [dict(arm='protected_geometry_parent',stage='protected',optimizer_updates=3723,
                  total_geometry_fit_updates=11169,zero_update_architecture=False,rec_hits25=5616,rec_hits50=4511)]
    data = {}
    training = {}
    for arm in ARMS:
        fit = read_json(COMPLETE / arm / 'receipt.json')
        assert fit['status']=='complete' and fit['training_steps']==3723 and fit['fit_rows']==29778
        assert fit['fit_seen_exactly_once'] and fit['parent_and_zero_R_states_exact'] and fit['fresh_optimizer']
        assert fit['head_state_tensors']==10 and fit['head_parameters']==456102
        training[arm] = read_rows(COMPLETE / arm / 'train.jsonl')
        assert len(training[arm])==3723 and [row['step'] for row in training[arm]]==list(range(1,3724))
        fit_ids=[identity for row in training[arm] for identity in row['rows']]
        assert len(fit_ids)==len(set(fit_ids))==29778 and len(training[arm][-1]['rows'])==2
        for stage in STAGES:
            data[arm,stage] = read_rows(COMPLETE / arm / stage / 'rows.jsonl')
            pair(parent,data[arm,stage])
            entry=recount(arm,stage,data[arm,stage])
            entry['parent_delta']=difference(parent,data[arm,stage])
            table.append(entry)
    assert [row['rows'] for row in training[ARMS[0]]]==[row['rows'] for row in training[ARMS[1]]]
    comparisons={stage:difference(data[ARMS[0],stage],data[ARMS[1],stage]) for stage in STAGES}
    adaptation={arm:difference(data[arm,'initial_formal'],data[arm,'formal']) for arm in ARMS}
    qualified={arm:dict(extra_candidate_occurrences=sum(row['extra_counts']['extra_candidates'] for row in training[arm]),
        extra_outside_face_occurrences=sum(row['extra_counts']['extra_boundary_outside'] for row in training[arm])) for arm in ARMS}
    best=max(table,key=lambda row:(row['rec_hits50'],row['arm']=='protected_geometry_parent',row['rec_hits25']))
    summary=dict(status='ACTUAL_CLOSED_ALL256_REFERENCE_PAIR_RECOUNTED',table=table,arms_compared=comparisons,
        initial_to_trained=adaptation,training_scope=qualified,metric_best_candidate=best,
        strict_target_gap=max(0,4754-best['rec_hits50']),formal_rows_per_result=9508,
        seed=2027,fit_order_exact=True,retained_hidden_prior_updates=11169,
        retained_hidden_total_updates_at_terminal=14892,reset_output_total_updates_at_terminal=3723,
        source_reset_output_tensors=2,parent_and_R_frozen=True,
        cpu_scope='all256 stored prior/reference/final boxes, scores, root boxes; no raw Mask recomputation',
        fresh_terminal_audit_pending=True,selected_checkpoint_restore_pending=True,weight_retention_executed=False)
    OUTPUT.mkdir()
    (OUTPUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(table=[{key:row[key] for key in ('arm','stage','rec_hits25','rec_hits50')} for row in table],
        metric_best={key:best[key] for key in ('arm','stage','rec_hits25','rec_hits50')},audit_pending=True,weights_changed=False)))


if __name__ == '__main__':
    main()
