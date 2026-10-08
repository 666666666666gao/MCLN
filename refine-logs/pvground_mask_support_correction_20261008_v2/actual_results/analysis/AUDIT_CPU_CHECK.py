"""Independent artifact-only verifier. No project imports or model execution."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np

R = Path(__file__).resolve().parents[1]
C = R / 'complete_fit'
ARMS = ('content', 'box_conditioned')
NAMES = ('parent',) + ARMS
N = 9508
HASHES = {}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    s = h.hexdigest()
    HASHES[str(path)] = 'sha256:' + s
    return s

def doc(path):
    sha(path)
    return json.loads(path.read_bytes())

def rows(path):
    sha(path)
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]

def hits(a):
    a = np.asarray(a)
    return {str(t): int((a > t).sum()) for t in (.25, .5)}

def comparison(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    return {str(t): {'before': int((a > t).sum()), 'after': int((b > t).sum()),
        'repairs': int(((a <= t) & (b > t)).sum()), 'damages': int(((a > t) & (b <= t)).sum()),
        'net': int((b > t).sum() - (a > t).sum()),
        'repair_row_ids': np.flatnonzero((a <= t) & (b > t)).tolist(),
        'damage_row_ids': np.flatnonzero((a > t) & (b <= t)).tolist()} for t in (.25, .5)}

def iou_cpu(boxes, truth):
    # Build each cuboid's opposing corners in float64, then intersect cuboids.
    b = np.asarray(boxes, dtype=np.float64)
    g = np.asarray(truth, dtype=np.float64)
    b0, b1 = b[..., :3] - b[..., 3:] / 2, b[..., :3] + b[..., 3:] / 2
    g0, g1 = g[..., :3] - g[..., 3:] / 2, g[..., :3] + g[..., 3:] / 2
    intersect = np.prod(np.maximum(0, np.minimum(b1, g1) - np.maximum(b0, g0)), axis=-1)
    vb, vg = np.prod(b1 - b0, axis=-1), np.prod(g1 - g0, axis=-1)
    return intersect / (vb + vg - intersect)

report = {}
intake = doc(C / 'INTAKE.json')
files = intake['files']
expected = {x['name']: x for x in files}
assert len(expected) == len(files)
for name, item in expected.items():
    p = C / name
    assert p.is_file() and p.stat().st_size == item['bytes'], name
    assert sha(p) == item['sha256'], name
actual = {p.relative_to(C).as_posix() for p in C.rglob('*') if p.is_file()}
report['intake'] = {'file_count': len(files), 'bytes': sum(x['bytes'] for x in files),
    'missing': sorted(set(expected) - actual), 'extras': sorted(actual - set(expected)),
    'weights_in_intake': [name for name in expected if name.endswith(('.pth', '.pt', '.tmp'))],
    'recorded_copied': intake['files_copied'], 'all_declared_bytes_and_sha256_exact': True,
    'time_cst': intake['time_cst']}
assert report['intake']['file_count'] == intake['files_copied']
assert report['intake']['bytes'] == intake['total_bytes']
assert report['intake']['extras'] == ['INTAKE.json']
spec = doc(C / 'pair_spec.json')
port = doc(C / 'source_port.json')
assert sha(C / 'source_port.json') == spec['source_port_sha256']
for name, h in spec['new_runner_files'].items():
    assert sha(C / name) == h
for name, h in port['files'].items():
    assert sha(C / 'PV-Ground' / name) == h
helper = R.parent / 'pvground_final_quality_20261005' / 'runtime_bundle'
for name, h in spec['runner_files'].items():
    assert sha(helper / name) == h
imports = doc(C / 'imports.json')
report['import_binding'] = {'native_evaluator_matches_runtime_digest':
    sha(C / 'PV-Ground/src/grounding_evaluator.py') == imports['sha256']['native_evaluator'],
    'bundled_dataset_source_matches_import_digest':
    sha(C / 'PV-Ground/src/joint_det_dataset.py') == imports['sha256']['src.joint_det_dataset'],
    'runtime_dataset_source_sha256': imports['sha256']['src.joint_det_dataset']}
report['phase_exits'] = {name: (C / (name + '.exit')).read_text().strip()
    for name in ('initial_formal', 'train', 'formal', 'fit_controller')}
assert set(report['phase_exits'].values()) == {'0'}
status = doc(C / 'fit_status.json')
assert status['status'] == 'complete' and status['completed_modes'] == ['initial_formal', 'train', 'formal']
train = rows(C / 'train.jsonl')
assert [v['step'] for v in train] == list(range(1, 3724))
seen = [i for v in train for i in v['rows']]
assert len(seen) == len(set(seen)) == 29778
assert all(len(v['rows']) == (2 if v['step'] == 3723 else 8) for v in train)
assert all(v['frozen_parent_forwards_per_batch'] == v['final_semantic_head_calls'] == 1 for v in train)
assert all(len(v['original_parent_assignment']) == len(v['rows']) for v in train)
target_ids = Counter()
query_ids = Counter()
reconstruction = {arm: 0.0 for arm in ARMS}
gradient = {arm: [] for arm in ARMS}
for v in train:
    assert v['expanded_positive_queries'] == 0
    for m in v['original_parent_assignment']:
        assert len(m['queries']) == len(m['actual_gt_ids']) == 1
        assert m['actual_gt_ids'] == [0] and 0 <= m['queries'][0] < 256
        target_ids.update(m['actual_gt_ids']); query_ids.update(m['queries'])
    for arm in ARMS:
        a = v['arms'][arm]
        assert a['valid_gt'] == a['matched_queries'] == len(v['rows'])
        assert a['native_coefficients'] == [5, 1, 10, 2]
        assert a['expanded_positive_queries'] == 0 and a['independent_head_gradients']
        assert all(math.isfinite(a[k]) for k in ('loss', 'gradient_norm', 'query_focal', 'query_dice', 'fused_focal', 'fused_dice'))
        reconstructed = 5 * a['query_focal'] + a['query_dice'] + 10 * a['fused_focal'] + 2 * a['fused_dice']
        reconstruction[arm] = max(reconstruction[arm], abs(a['loss'] - reconstructed))
        gradient[arm].append(a['gradient_norm'])
holdouts = {s: rows(C / s / 'rows.jsonl') for s in ('initial', 'terminal')}
holdout_ids = [r['row_id'] for r in holdouts['initial']]
assert len(holdout_ids) == len(set(holdout_ids)) == 6887
assert set(seen).isdisjoint(holdout_ids)
assert set(seen) | set(holdout_ids) == set(range(36665))
assert all(a['row_id'] == b['row_id'] and a['root_box'] == b['root_box'] and
    a['point_sha256'] == b['point_sha256'] for a, b in zip(holdouts['initial'], holdouts['terminal']))
report['training'] = {'steps_each': len(train), 'unique_fit_rows_each': len(seen),
    'holdout_rows': len(holdout_ids), 'fit_holdout_disjoint_and_union36665': True,
    'matched_target_ids': dict(target_ids), 'distinct_matched_query_ids': len(query_ids),
    'max_stored_weighted_loss_reconstruction_error': reconstruction,
    'gradient_norms': {a: {'min': min(g), 'max': max(g), 'nonzero_steps': sum(x > 0 for x in g)} for a, g in gradient.items()},
    'formal_restore_receipts': {a: doc(C / a / 'formal_restore.json') for a in ARMS}}
data = {s: rows(C / s / 'rows.jsonl') for s in ('initial_formal', 'formal')}
historic_path = R.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal/rows.jsonl'
historical = rows(historic_path)
report['formal'] = {}
native = {}
cpu = {}
for stage, rr in data.items():
    assert len(rr) == N and [r['row_id'] for r in rr] == list(range(N))
    assert all(r['parent_forwards'] == r['final_semantic_head_calls'] == 1 for r in rr)
    native[stage] = {'parent': np.asarray([r['parent_iou'] for r in rr])}
    native[stage].update({a: np.asarray([r['arms'][a]['iou'] for r in rr]) for a in ARMS})
    cpu[stage] = {a: [] for a in NAMES}
    stats = {a: {'coverage': {str(t): 0 for t in (.25, .5)},
        'threshold_flip_row_ids': {str(t): [] for t in (.25, .5)},
        'max_native_iou_abs_difference': 0.0, 'changed_all256_boxes_vs_parent': 0,
        'changed_selected_boxes_vs_parent': 0} for a in NAMES}
    oracle = {a: {str(t): [0] * 4 for t in (.25, .5)} for a in ARMS}
    oracle_bad = {a: {str(t): [[] for _ in range(4)] for t in (.25, .5)} for a in ARMS}
    max_tie_rows = []
    offset = 0
    for path in sorted((C / stage).glob('batch_*.npz')):
        assert path.name == 'batch_%05d.npz' % offset
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'row_ids', 'root_gt', 'scores', *NAMES}
            qrows = rr[offset:offset + len(z['row_ids'])]
            n = len(qrows)
            assert n == min(8, N-offset)
            assert np.array_equal(z['row_ids'], np.arange(offset, offset+n))
            truth = np.array([r['root_box'] for r in qrows])
            assert z['root_gt'].shape == (n, 6) and np.array_equal(truth, z['root_gt'])
            scores = z['scores']; q = np.array([r['query'] for r in qrows]); idx = np.arange(n)
            assert scores.shape == (n, 256) and np.isfinite(scores).all()
            assert np.array_equal(scores[idx, q], scores.max(axis=1))
            tied = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) > 1
            max_tie_rows += (np.flatnonzero(tied) + offset).tolist()
            order = np.argsort(-scores, axis=1, kind='stable')
            for a in NAMES:
                boxes = z[a]
                assert boxes.shape == (n, 256, 6) and np.isfinite(boxes).all() and (boxes[..., 3:] > 0).all()
                saved = [r['parent_box'] if a == 'parent' else r['arms'][a]['box'] for r in qrows]
                assert np.array_equal(boxes[idx, q], saved)
                values = iou_cpu(boxes, truth[:, None, :])
                selected = values[idx, q]
                assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1+1e-10).all()
                cpu[stage][a].extend(selected.tolist())
                sn = native[stage][a][offset:offset+n]
                stats[a]['max_native_iou_abs_difference'] = max(stats[a]['max_native_iou_abs_difference'], float(np.abs(sn-selected).max()))
                changed = (boxes != z['parent']).any(axis=-1)
                stats[a]['changed_all256_boxes_vs_parent'] += int(changed.sum())
                stats[a]['changed_selected_boxes_vs_parent'] += int(changed[idx, q].sum())
                if stage == 'initial_formal': assert np.array_equal(boxes, z['parent'])
                for t in (.25, .5):
                    stats[a]['coverage'][str(t)] += int((values > t).any(axis=1).sum())
                    stats[a]['threshold_flip_row_ids'][str(t)] += (np.flatnonzero((sn>t) != (selected>t)) + offset).tolist()
                    if a in ARMS:
                        for j,k in enumerate((16,32,64,256)):
                            flags = (np.take_along_axis(values, order[:, :k], axis=1) > t).any(axis=1)
                            stored = [r['arms'][a]['oracle25' if t == .25 else 'oracle50'][j] for r in qrows]
                            oracle[a][str(t)][j] += int(flags.sum())
                            oracle_bad[a][str(t)][j] += (np.flatnonzero(flags != stored) + offset).tolist()
            offset += n
    assert offset == N
    receipt = doc(C / stage / 'receipt.json')
    assert sha(C / stage / 'rows.jsonl') == receipt['rows_sha256']
    table = []
    for a in NAMES:
        row = {'name': a, 'native_box_hits': hits(native[stage][a]), 'independent_cpu_box_hits': hits(cpu[stage][a])}
        if a in ARMS:
            masks = [r['arms'][a]['mask_iou'] for r in rr]
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in masks)
            row['stored_mask_iou_recount'] = {'hits': hits(masks), 'miou_percent': sum(masks)/N*100}
            row['selected_reference_invalid'] = sum(not r['arms'][a]['reference_valid'] for r in rr)
            row['stored_mask_gt_05_box_le_05'] = int(sum(m>.5 and b<=.5 for m,b in zip(masks,native[stage][a])))
            rec = receipt['metrics'][a]
            assert row['native_box_hits'] == {'0.25': rec['rec_hits25'], '0.5': rec['rec_hits50']}
            assert row['stored_mask_iou_recount']['hits'] == {'0.25':rec['mask_hits25'], '0.5':rec['mask_hits50']}
            assert abs(row['stored_mask_iou_recount']['miou_percent'] - rec['mask_miou']) < 1e-9
        table.append(row)
    report['formal'][stage] = {'rows':N, 'npz_batches':len(list((C/stage).glob('batch_*.npz'))),
        'independent_all256_ious':N*256*3, 'unique_scans':len(set(r['scan_id'] for r in rr)),
        'table':table, 'box_checks':stats, 'max_score_tie_row_ids':max_tie_rows,
        'cpu_topk_16_32_64_256_oracle_hits':oracle,
        'cpu_oracle_label_mismatch_row_ids':oracle_bad}
align_keys = ('row_id','scan_id','target_id','root_box','point_sha256')
assert len(historical) == N
assert all(all(a[k] == b[k] == h[k] for k in align_keys)
    for a,b,h in zip(data['initial_formal'],data['formal'],historical))
report['alignment'] = {'initial_final_historical_all9508_keys_exact':list(align_keys),
    'formal_holdout_scan_overlap': sorted(set(r['scan_id'] for r in data['formal']) & set(r['scan_id'] for r in holdouts['initial']))}
hp = np.array([r['bbs']['iou'] for r in historical])
assert hits(hp) == {'0.25':5598,'0.5':4848}
report['comparisons'] = {
    'same_forward_parent_to_arm': {a:comparison(native['formal']['parent'],native['formal'][a]) for a in ARMS},
    'same_forward_content_to_box_conditioned':comparison(native['formal']['content'],native['formal']['box_conditioned']),
    'initial_to_final': {a:comparison(native['initial_formal'][a],native['formal'][a]) for a in NAMES},
    'historical_to_final': {a:comparison(hp,native['formal'][a]) for a in NAMES}}
report['holdout_table'] = {s:{a:hits([r['arms'][a]['iou'] for r in rr]) for a in ARMS} for s,rr in holdouts.items()}
report['cross_pass_drift'] = {
    'selected_query_changes':sum(a['query'] != b['query'] for a,b in zip(data['initial_formal'],data['formal'])),
    'selected_parent_box_changes':sum(a['parent_box'] != b['parent_box'] for a,b in zip(data['initial_formal'],data['formal'])),
    'initial_parent_vs_historical_selected_box_changes':sum(a['parent_box'] != h['bbs']['box'] for a,h in zip(data['initial_formal'],historical)),
    'final_parent_vs_historical_selected_box_changes':sum(a['parent_box'] != h['bbs']['box'] for a,h in zip(data['formal'],historical)),
    'all256': {k:{'changed_elements':0,'changed_expressions':0,'max_abs_difference':0.0} for k in ('scores','parent')}}
for path in sorted((C/'initial_formal').glob('batch_*.npz')):
    with np.load(path,allow_pickle=False) as a,np.load(C/'formal'/path.name,allow_pickle=False) as b:
        for k,v in report['cross_pass_drift']['all256'].items():
            changed = a[k] != b[k]
            v['changed_elements'] += int(changed.sum())
            v['changed_expressions'] += int(changed.reshape(len(a['row_ids']),-1).any(axis=1).sum())
            v['max_abs_difference'] = max(v['max_abs_difference'],float(np.abs(a[k].astype('float64')-b[k].astype('float64')).max()))
eligible = [('protected_parent', hp)] + [(a,native['formal'][a]) for a in ARMS]
selection = []
for a,v in eligible:
    h=hits(v); gate=h['0.25']>=5620 and h['0.5']>=4764
    selection.append({'arm':a,'hits':h,'dual_target_pass':gate,
        'ordering_key':[gate,h['0.5'],h['0.25'],a=='protected_parent']})
report['selection'] = {'eligible':selection,'metric_best':max(selection,key=lambda v:tuple(v['ordering_key']))['arm'],
    'target_passing_count':sum(v['dual_target_pass'] for v in selection)}
dataset_archive = R.parent/'pvground_g_p2_20261002/complete/source'
assert sha(dataset_archive/'joint_det_dataset.py') == imports['sha256']['src.joint_det_dataset']
dataset_manifest = doc(dataset_archive/'input_manifest.json')
split = doc(dataset_archive/'split_protocol.json')
assert sha(dataset_archive/'split_protocol.json') == dataset_manifest['split_protocol_sha256']
assert set(split['row_ids']['fit']) == set(seen)
assert split['row_ids']['holdout'] == holdout_ids
report['import_binding']['actual_dataset_archive_matches_runtime_digest'] = True
report['training']['archived_split_protocol_ids_match'] = True
for p in [R/'analysis/SUMMARY.json', R/'fit_launch.json',R/'fit_wait.json',
    R/'observer_original63704_closed_transport_failure.json', R/'fit_observer_transport_recovery_inspection.json',
    R/'fit_observer_resumed_wait.json',R/'collect_closed_fit_authorized.py',R/'postrun/analyze_support_correction_formal.py',
    R/'postrun/ANALYSIS_AND_INTAKE_SOURCE_REVIEW.json',R/'preflight_complete/preflight.json',
    R.parent/'pvground_referit_mask_reference_20261006/current_research_goals.json',Path(__file__)]:
    sha(p)
report['audited_input_hashes']=HASHES
(R/'analysis/AUDIT_CPU_EVIDENCE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
compact={k:v for k,v in report.items() if k not in ('audited_input_hashes','comparisons')}
compact['comparisons']={k:({a:{t:{i:j for i,j in b.items() if not i.endswith('_row_ids')} for t,b in x.items()} for a,x in v.items()}
    if k!='same_forward_content_to_box_conditioned' else {t:{i:j for i,j in x.items() if not i.endswith('_row_ids')} for t,x in v.items()})
    for k,v in report['comparisons'].items()}
print(json.dumps(compact,indent=2))
