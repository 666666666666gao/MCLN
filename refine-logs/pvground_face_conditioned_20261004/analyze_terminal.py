"""CPU recount and direct common-start comparison after the actual run closes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

local = Path(__file__).resolve().parent
previous = local.parent / 'pvground_boundary_distribution_20261004'
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert not intake['controller_alive']
for relative, item in intake['files'].items():
    raw = (local / 'complete' / relative).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
analysis = local / 'analysis'
analysis.mkdir()

def read_rows(root, stage, expected):
    receipt = json.loads((root / stage / 'receipt.json').read_bytes())
    raw = (root / stage / 'rows.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == receipt['rows'] == expected
    assert len({row['row_id'] for row in rows}) == expected
    return receipt, rows

def iou(box, target):
    assert len(box) == len(target) == 6
    assert all(math.isfinite(x) for x in box + target)
    assert all(x > 0 for x in box[3:] + target[3:])
    overlap = [max(0., min(box[axis] + box[axis+3]/2, target[axis] + target[axis+3]/2)
        - max(box[axis] - box[axis+3]/2, target[axis] - target[axis+3]/2)) for axis in range(3)]
    intersection = math.prod(overlap)
    return intersection / (math.prod(box[3:]) + math.prod(target[3:]) - intersection)

root = local / 'complete/face_conditioned'
control_root = previous / 'complete/distribution'
summaries = {}
for stage, count in (('initial', 6887), ('terminal', 6887), ('formal', 9508)):
    receipt, rows = read_rows(root, stage, count)
    control_receipt, controls = read_rows(control_root, stage, count)
    assert [row['row_id'] for row in rows] == [row['row_id'] for row in controls]
    if stage == 'formal':
        assert [row['row_id'] for row in rows] == list(range(9508))
    for row, control in zip(rows, controls):
        for field in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
            assert row[field] == control[field], (stage, row['row_id'], field)
    modes = {}
    for mode in ('bbs', 'bbf'):
        changes = 0
        rec = Counter()
        refinements = Counter()
        between = Counter()
        coarse_changed = 0
        query_changed = 0
        first_good = Counter()
        mask_differences = 0
        moves = []
        for row, control in zip(rows, controls):
            predicted, baseline = row[mode], control[mode]
            final = iou(predicted['box'], row['root_box'])
            coarse = iou(predicted['coarse_box'], row['root_box'])
            assert abs(final - predicted['iou']) < 1e-5
            assert abs(coarse - predicted['coarse_iou']) < 1e-5
            for label, threshold in (('25', .25), ('50', .5)):
                changes += int((final > threshold) != (predicted['iou'] > threshold))
                rec[label] += int(final > threshold)
                refinements['coarse_hits'+label] += int(coarse > threshold)
                refinements['repairs'+label] += int(coarse <= threshold < final)
                refinements['damages'+label] += int(final <= threshold < coarse)
                between['repairs'+label] += int(baseline['iou'] <= threshold < final)
                between['damages'+label] += int(final <= threshold < baseline['iou'])
                rank = predicted['first_good_rank'+label]
                assert (rank is not None) == bool(predicted['oracle'+label][-1])
                if rank is not None:
                    assert 1 <= rank <= 256
                for k, exists in zip((16, 32, 64, 256), predicted['oracle'+label]):
                    assert bool(exists) == (rank is not None and rank <= k)
            if final <= .5:
                rank = predicted['first_good_rank50']
                group = 'missing' if rank is None else '2-16' if rank <= 16 else '17-32' if rank <= 32 else '33-64' if rank <= 64 else '65-256'
                if rank is not None:
                    assert rank > 1
                first_good[group] += 1
            coarse_changed += int(predicted['coarse_box'] != baseline['coarse_box'])
            query_changed += int(predicted['query'] != baseline['query'])
            mask_differences += int(predicted['mask_iou'] != baseline['mask_iou'])
            box, before = predicted['box'], predicted['coarse_box']
            moves.append(max(abs(box[axis] + sign*box[axis+3]/2 - before[axis] - sign*before[axis+3]/2)
                for axis in range(3) for sign in (-1, 1))*1000)
        assert changes == 0
        assert rec['25'] == receipt['metrics'][mode]['rec_hits25']
        assert rec['50'] == receipt['metrics'][mode]['rec_hits50']
        modes[mode] = dict(rec_hits25=rec['25'], rec_hits50=rec['50'],
            control_hits25=control_receipt['metrics'][mode]['rec_hits25'],
            control_hits50=control_receipt['metrics'][mode]['rec_hits50'],
            native_metrics=receipt['metrics'][mode], cpu_threshold_changes=changes,
            same_query_refinement=dict(refinements), repairs_damages_vs_control=dict(between),
            first_good_strict_errors=dict(first_good), max_face_move_median_mm=statistics.median(moves),
            cross_process_coarse_box_differences=coarse_changed, query_differences=query_changed,
            mask_scalar_differences=mask_differences)
    summaries[stage] = modes
training = [json.loads(line) for line in (root / 'train.jsonl').read_bytes().splitlines()]
assert len(training) == 3723
seen = [rid for record in training for rid in record['rows']]
assert len(seen) == len(set(seen)) == 29778
assert all(len(record['rows']) == 8 for record in training[:-1]) and len(training[-1]['rows']) == 2
result = dict(status='CPU_RECOUNT_COMPLETE', primary_mode='bbs', primary_threshold=.5,
    stages=summaries, fit_rows=29778, optimizer_updates=3723, seed=2027,
    original_g_hits25=5615, original_g_hits50=4495, prior_best_hits50=4506,
    scanrefer_target_pass=summaries['formal']['bbs']['rec_hits25'] >= 5615 and summaries['formal']['bbs']['rec_hits50'] >= 4754,
    architecture_and_capacity_changed_together=True, same_family_audit_pending=True)
(analysis / 'SUMMARY.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
formal = summaries['formal']['bbs']
lines = ['# Face-conditioned boundary decoder: actual bounded result', '',
    'OriginalG fixed/eval, common originalG initialization; all256 candidates, same whole/local predicted support.',
    'New64737-parameter face-conditioned head vs completed456102-parameter flat distribution head; architecture and capacity change together.',
    'Each seed2027 run29778 fit rows once, effectivebatch8/tail2,3723 updates, native final-box loss plus matched DFL/7.',
    '6887 module rows were pretrained-seen scenes;9508 formal rows are development validation. No teacher or extra deployed score.', '',
    '| Stage / mode | New hits.25/.50 | Flat control.25/.50 |', '|---|---:|---:|']
for stage in ('initial', 'terminal', 'formal'):
    for mode in ('bbs', 'bbf'):
        item = summaries[stage][mode]
        lines.append('| '+stage+'/'+mode+' | '+str(item['rec_hits25'])+'/'+str(item['rec_hits50'])+' | '+str(item['control_hits25'])+'/'+str(item['control_hits50'])+' |')
lines += ['', 'CPU selected-box threshold recount:0 changes. Raw Masks and full candidate boxes are not downloaded; oracle ranks are GT diagnostics only.',
    'Cross-process numeric differences are counted in SUMMARY, not assumed absent. Same-checkpoint target5615/4754 '+('passed' if result['scanrefer_target_pass'] else 'unmet')+'.',
    'Primary formal bbs changes vs originalG: '+str(formal['rec_hits25']-5615)+' / '+str(formal['rec_hits50']-4495)+'.',
    'Fresh experiment-audit is pending; this analyzer is evidence computation, not an independent integrity verdict.']
(analysis / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], formal_bbs=[formal['rec_hits25'],formal['rec_hits50']],
    strict_difference_vs_best=formal['rec_hits50']-4506, target_pass=result['scanrefer_target_pass'])))
