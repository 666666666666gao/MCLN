"""Recount actual saved predictions against dataset root boxes, without NN replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ('parent', 'shared_text', 'candidate_fused')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def box_iou(boxes, truth):
    assert boxes.dtype == truth.dtype == np.float32
    assert np.isfinite(boxes).all() and np.isfinite(truth).all()
    assert (boxes[..., 3:] > 0).all() and (truth[..., 3:] > 0).all()
    low = np.maximum(boxes[..., :3] - boxes[..., 3:] / 2,
                     truth[:, None, :3] - truth[:, None, 3:] / 2)
    high = np.minimum(boxes[..., :3] + boxes[..., 3:] / 2,
                      truth[:, None, :3] + truth[:, None, 3:] / 2)
    intersection = np.maximum(high - low, 0).prod(-1)
    overlap = intersection / (boxes[..., 3:].prod(-1) + truth[:, 3:].prod(-1)[:, None] - intersection)
    assert np.isfinite(overlap).all()
    return overlap


def differences(first, second):
    result = {}
    for threshold, label in ((.25, '025'), (.5, '050')):
        before, after = first > threshold, second > threshold
        repair, damage = int((~before & after).sum()), int((before & ~after).sum())
        result[label] = dict(repairs=repair, damages=damage, net=repair - damage)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--split-protocol', type=Path, required=True)
    parser.add_argument('--historical-rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    intake = json.loads((args.actual / 'INTAKE.json').read_bytes())
    assert intake['exitcode'] == 0
    for entry in intake['files']:
        path = args.actual / entry['name']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['name']
    root = args.actual / 'campaign'
    campaign = json.loads((root / 'receipt.json').read_bytes())
    assert campaign['status'] == 'complete' and campaign['parent_state_exact']
    assert campaign['optimizer_steps_per_arm'] == 3723
    assert campaign['fit_examples_per_arm'] == 29778 and campaign['fit_examples_seen_exactly_once']
    assert campaign['preflight_updates_not_carried'] and not campaign['full_goal_complete']
    partitions = json.loads(args.split_protocol.read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    training = rows(root / 'train.jsonl')
    assert len(training) == 3723 and [r['step'] for r in training] == list(range(1, 3724))
    trained = [index for record in training for index in record['row_ids']]
    assert Counter(trained) == Counter(partitions['fit']) and len(set(trained)) == 29778
    assert all(len(record['row_ids']) == (2 if record['step'] == 3723 else 8) for record in training)
    assert all(record['parent_forwards'] == record['parent_native_head_calls'] == 1 for record in training)
    for record in training:
        assert set(record['arms']) == set(ARMS[1:])
        assert all(np.isfinite(item[key]) for item in record['arms'].values()
                   for key in ('loss', 'final_native_ce', 'g_correction', 'gradient_norm'))
    phases, data, selected = {}, {}, {}
    for stage, ids in (('initial_formal', list(range(9508))),
                       ('holdout', partitions['holdout']), ('formal', list(range(9508)))):
        directory = root / stage
        records = rows(directory / 'rows.jsonl')
        assert len(records) == len(ids) and [r['row_id'] for r in records] == ids
        native = json.loads((directory / 'receipt.json').read_bytes())
        assert native == campaign[stage] and native['rows'] == len(ids)
        assert native['shared_boxes_and_masks'] and native['all256_retained']
        boxes = np.load(directory / 'boxes.npy', mmap_mode='r')
        assert boxes.shape == (len(ids), 256, 6) and boxes.dtype == np.float32
        truth = np.asarray([r['root_box'] for r in records], dtype=np.float32)
        overlap = box_iou(boxes, truth)
        for threshold, key in ((.25, 'all256_oracle25'), (.5, 'all256_oracle50')):
            assert np.array_equal((overlap > threshold).any(-1), [r[key] for r in records])
        metrics, selected[stage], scores = {}, {}, {}
        for arm in ARMS:
            score = np.load(directory / (arm + '_scores.npy'), mmap_mode='r')
            assert score.shape == (len(ids), 256) and score.dtype == np.float32 and np.isfinite(score).all()
            scores[arm] = score
            query = np.asarray([r['arms'][arm]['query'] for r in records], dtype=np.int64)
            assert ((query >= 0) & (query < 256)).all()
            chosen = overlap[np.arange(len(ids)), query]
            saved = np.asarray([r['arms'][arm]['iou'] for r in records])
            assert np.max(np.abs(chosen - saved)) < 1e-5
            assert all(np.array_equal(chosen > t, saved > t) for t in (.25, .5))
            assert np.array_equal(boxes[np.arange(len(ids)), query], np.asarray([r['arms'][arm]['box'] for r in records], dtype=np.float32))
            assert np.array_equal(score[np.arange(len(ids)), query], score.max(-1))
            assert np.array_equal(score[np.arange(len(ids)), query], np.asarray([r['arms'][arm]['score'] for r in records], dtype=np.float32))
            mask = np.asarray([r['arms'][arm]['mask_iou'] for r in records])
            assert np.isfinite(mask).all() and ((mask >= 0) & (mask <= 1)).all()
            hits = [int((chosen > t).sum()) for t in (.25, .5)]
            assert hits == [native['metrics'][arm]['hits25'], native['metrics'][arm]['hits50']]
            assert [int((mask > t).sum()) for t in (.25, .5)] == [native['metrics'][arm]['mask_hits25'], native['metrics'][arm]['mask_hits50']]
            assert abs(mask.mean() * 100 - native['metrics'][arm]['mask_miou']) < 1e-9
            selected[stage][arm] = chosen
            ranked = np.argsort(-score, axis=-1, kind='stable')
            sorted_iou = np.take_along_axis(overlap, ranked, axis=-1)
            sorted_scores = np.take_along_axis(score, ranked, axis=-1)
            rank_checks = {}
            for position, count in enumerate((16, 32, 64, 256)):
                disagreements = {str(t): int(np.count_nonzero((sorted_iou[:, :count] > t).any(-1) != np.asarray([r['arms'][arm][key][position] for r in records])))
                                 for t, key in ((.25, 'oracle25'), (.5, 'oracle50'))}
                rank_checks[str(count)] = dict(cpu_vs_saved_oracle_disagreements=disagreements,
                    boundary_tie_rows=int((sorted_scores[:, count - 1] == sorted_scores[:, count]).sum()) if count < 256 else 0)
            metrics[arm] = dict(hits=hits, percent=[v / len(ids) * 100 for v in hits],
                selected_box_iou_max_abs_difference=float(np.max(np.abs(chosen - saved))),
                mask_hits_from_saved_point_iou=[int((mask > t).sum()) for t in (.25, .5)],
                mask_miou_from_saved_point_iou=float(mask.mean() * 100),
                topk_numpy_stable_sort_checks=rank_checks,
                full256_qualified_but_unselected=[int(((overlap > t).any(-1) & (chosen <= t)).sum()) for t in (.25, .5)],
                selected_query_tied_for_maximum_rows=int(((score == score.max(-1, keepdims=True)).sum(-1) > 1).sum()))
        contrasts = {arm + '_vs_parent': differences(selected[stage]['parent'], selected[stage][arm]) for arm in ARMS[1:]}
        contrasts['candidate_fused_vs_shared_text'] = differences(selected[stage]['shared_text'], selected[stage]['candidate_fused'])
        for arm in ARMS[1:]:
            for label, suffix in (('025', '25'), ('050', '50')):
                values = contrasts[arm + '_vs_parent'][label]
                assert values['repairs'] == native['metrics'][arm]['repairs' + suffix]
                assert values['damages'] == native['metrics'][arm]['damages' + suffix]
        if stage == 'initial_formal':
            assert all(np.array_equal(score, scores['parent']) for score in scores.values())
            assert all(np.array_equal(chosen, selected[stage]['parent']) for chosen in selected[stage].values())
        phases[stage] = dict(rows=len(ids), metrics=metrics, contrasts=contrasts,
            full256_oracle_hits=[int((overlap > t).any(-1).sum()) for t in (.25, .5)],
            candidate_iou_recomputations=int(overlap.size), optimizer_steps_per_arm=native['optimizer_steps_per_arm'])
        data[stage] = (records, boxes, scores)
    identity_keys = ('row_id', 'scan_id', 'target_id', 'text', 'root_box', 'point_sha256',
                     'detector_boxes_sha256', 'detector_class_ids_sha256', 'detector_label_mask_sha256', 'superpoint_sha256')
    input_differences = {key: sum(a[key] != b[key] for a, b in zip(data['initial_formal'][0], data['formal'][0])) for key in identity_keys}
    assert not any(input_differences.values())
    historical = rows(args.historical_rows)
    assert len(historical) == 9508 and [r['row_id'] for r in historical] == list(range(9508))
    history_identity = {key: sum(a[key] != b[key] for a, b in zip(historical, data['initial_formal'][0]))
                        for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256')}
    historical_hits = [sum(r['arms']['content']['iou'] > t for r in historical) for t in (.25, .5)]
    assert historical_hits == [5599, 4859]
    decisions = {arm: dict(hits=phases['formal']['metrics'][arm]['hits'],
        delta_to_retained_best=[a - b for a, b in zip(phases['formal']['metrics'][arm]['hits'], historical_hits)],
        joint_development_thresholds_passed=all(a >= b for a, b in zip(phases['formal']['metrics'][arm]['hits'], (5658, 4850)))) for arm in ARMS[1:]}
    report = dict(status='SAVED_PREDICTION_CPU_RECOUNT_COMPLETE', phases=phases,
        fit_examples=29778, updates_per_arm=3723, single_seed=2027,
        initial_to_formal_input_differences=input_differences,
        historical_input_differences=history_identity,
        parent_initial_to_formal= differences(selected['initial_formal']['parent'], selected['formal']['parent']),
        parent_all256_boxes_max_absolute_drift=float(np.max(np.abs(data['initial_formal'][1] - data['formal'][1]))),
        parent_all256_scores_max_absolute_drift=float(np.max(np.abs(data['initial_formal'][2]['parent'] - data['formal'][2]['parent']))),
        decisions_pending_recovery_and_fresh_review=decisions,
        best_updated=False, full_goal_complete=False,
        input_files=dict(intake=sha(args.actual / 'INTAKE.json'), split_protocol=sha(args.split_protocol), historical_rows=sha(args.historical_rows)),
        limits=['Root GT boxes are saved dataset labels; no independent raw dataset reconstruction in this CPU script.',
                'Mask results are recounted from saved point-mask IoUs, not raw point masks.',
                'NumPy and Torch ordering of tied scores can differ; top-k disagreements are reported, not silently accepted.',
                'Same-frame contrasts isolate final scores on one frozen parent; historical forward drift is reported separately.',
                'Single seed and repeatedly used development validation do not establish statistical stability or three effective contributions.'])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], formal=phases['formal'], best_updated=False)), flush=True)


if __name__ == '__main__':
    main()
