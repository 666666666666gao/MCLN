"""Audit-owned stdlib-only recount. Never imports or executes primary source."""
from pathlib import Path
from collections import Counter
import ast
import datetime
import hashlib
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / 'TERMINAL_AUDIT_REQUEST.txt'
PATHS = [Path(s) for s in REQUEST.read_text(encoding='utf-8').splitlines() if s.startswith('D:')]
assert len(PATHS) == 70 and len(set(PATHS)) == 70
raw = {p: p.read_bytes() for p in PATHS}
texts = {p: v.decode('utf-8-sig') for p, v in raw.items()}
identities = []
seen_hashes = {}
parsed = {}
for i, p in enumerate(PATHS, 1):
    digest = hashlib.sha256(raw[p]).hexdigest()
    item = dict(id=f'P{i:02}', path=str(p), bytes=len(raw[p]), lines=len(texts[p].splitlines()), sha256=digest)
    if digest in seen_hashes:
        item['identical_bytes_to'] = seen_hashes[digest]
    seen_hashes.setdefault(digest, item['id'])
    if p.suffix == '.json':
        parsed[p] = json.loads(texts[p])
        item['read_method'] = 'full bytes, UTF-8 text, full JSON parse'
    elif p.suffix == '.jsonl':
        parsed[p] = [json.loads(s) for s in texts[p].splitlines()]
        item['records_parsed'] = len(parsed[p])
        item['read_method'] = 'full bytes, every NDJSON record parsed and recounted'
    elif p.suffix == '.py':
        tree = ast.parse(texts[p], filename=str(p))
        item['functions'] = [{'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno}
                             for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        item['read_method'] = 'full bytes, UTF-8 text and AST parse; source reviewed, not executed'
    else:
        item['read_method'] = 'full bytes and UTF-8 text'
    identities.append(item)

def sha(p):
    return hashlib.sha256(raw[p]).hexdigest()

def obj(relative):
    return parsed[ROOT / relative]

def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True

assert all(finite(v) for v in parsed.values())

def iou(a, b):
    assert len(a) == len(b) == 6 and all(math.isfinite(x) for x in a + b)
    assert all(x > 0 for x in a[3:] + b[3:])
    overlap = [max(0.0, min(a[k] + a[k+3]/2, b[k] + b[k+3]/2) -
                   max(a[k] - a[k+3]/2, b[k] - b[k+3]/2)) for k in range(3)]
    intersection = math.prod(overlap)
    return intersection / (math.prod(a[3:]) + math.prod(b[3:]) - intersection)

def transitions(before, after, threshold):
    return dict(repairs=sum(a <= threshold < b for a, b in zip(before, after)),
                damages=sum(b <= threshold < a for a, b in zip(before, after)))

face_root = ROOT / 'complete/face_conditioned'
flat_root = ROOT.parent / 'pvground_boundary_distribution_20261004/complete/distribution'
stage_rows, evaluations = {}, {}
for arm, directory in [('face', face_root), ('flat', flat_root)]:
    evaluations[arm] = {}
    for stage, expected in [('initial', 6887), ('terminal', 6887), ('formal', 9508)]:
        path = directory / stage / 'rows.jsonl'
        receipt = parsed[directory / stage / 'receipt.json']
        rows = parsed[path]
        stage_rows[arm, stage] = rows
        assert len(rows) == receipt['rows'] == expected
        assert sha(path) == receipt['rows_sha256']
        ids = [r['row_id'] for r in rows]
        assert len(set(ids)) == expected and ids == sorted(ids)
        assert receipt['formal_rows'] == (expected if stage == 'formal' else 0)
        if stage == 'formal':
            assert ids == list(range(9508))
        modes = {}
        for mode in ('bbs', 'bbf'):
            final = [iou(r[mode]['box'], r['root_box']) for r in rows]
            coarse = [iou(r[mode]['coarse_box'], r['root_box']) for r in rows]
            masks = [r[mode]['mask_iou'] for r in rows]
            assert all(0 <= x <= 1 for x in masks)
            assert all(type(r[mode]['query']) is int and 0 <= r[mode]['query'] < 256 for r in rows)
            counts = {f'rec_hits{label}': sum(x > threshold for x in final)
                      for label, threshold in [('25', .25), ('50', .5)]}
            counts.update({f'mask_hits{label}': sum(x > threshold for x in masks)
                           for label, threshold in [('25', .25), ('50', .5)]})
            counts.update(mask_iou_sum=sum(masks), mask_miou=sum(masks)/expected*100)
            assert counts == receipt['metrics'][mode]
            errors = [abs(u-r[mode]['iou']) for u, r in zip(final, rows)]
            coarse_errors = [abs(u-r[mode]['coarse_iou']) for u, r in zip(coarse, rows)]
            threshold_changes = sum((u > t) != (r[mode]['iou'] > t)
                                    for u, r in zip(final, rows) for t in (.25, .5))
            coarse_threshold_changes = sum((u > t) != (r[mode]['coarse_iou'] > t)
                                           for u, r in zip(coarse, rows) for t in (.25, .5))
            assert threshold_changes == coarse_threshold_changes == 0
            assert max(errors + coarse_errors) < 1e-5
            same = {}
            oracle = {}
            first_good = Counter()
            witness = {}
            for label, t in [('25', .25), ('50', .5)]:
                same[label] = dict(coarse_hits=sum(u > t for u in coarse), **transitions(coarse, final, t))
                oracle[label] = dict(final_counts=[sum(r[mode]['oracle'+label][j] for r in rows) for j in range(4)],
                                     coarse_full_count=sum(r[mode]['coarse_oracle'+label] for r in rows))
                witness[label] = {'repairs': [], 'damages': []}
                for line, (r, a, b) in enumerate(zip(rows, coarse, final), 1):
                    x = r[mode]
                    flags = x['oracle'+label]
                    assert len(flags) == 4 and all(v in (0, 1) for v in flags) and flags == sorted(flags)
                    if arm == 'face':
                        rank = x['first_good_rank'+label]
                        assert rank is None or type(rank) is int and 1 <= rank <= 256
                        assert flags == [int(rank is not None and rank <= k) for k in (16, 32, 64, 256)]
                        assert (rank == 1) == (b > t)
                    ref = dict(line=line, row_id=r['row_id'], scan_id=r['scan_id'], query=x['query'], coarse_iou=a, final_iou=b)
                    if a <= t < b:
                        witness[label]['repairs'].append(ref)
                    if b <= t < a:
                        witness[label]['damages'].append(ref)
            if arm == 'face':
                for r, b in zip(rows, final):
                    if b <= .5:
                        rank = r[mode]['first_good_rank50']
                        group = 'missing' if rank is None else '2-16' if rank <= 16 else '17-32' if rank <= 32 else '33-64' if rank <= 64 else '65-256'
                        first_good[group] += 1
            decode_errors = []
            moves = []
            for r in rows:
                x = r[mode]
                box, before, offsets = x['box'], x['coarse_box'], x['face_offsets']
                assert len(offsets) == len(x['face_entropy']) == 6
                assert all(-4.000001 <= value <= 4.000001 for value in offsets)
                assert all(-1e-6 <= value <= math.log(33) + 1e-6 for value in x['face_entropy'])
                reference = [max(v, 1e-6) for v in before[3:]]
                decoded = [before[k] + .5*(offsets[k+3]-offsets[k])*reference[k]/4 for k in range(3)]
                decoded += [max(reference[k]*(1+(offsets[k+3]+offsets[k])/4), 1e-6) for k in range(3)]
                decode_errors.append(max(abs(a-b) for a, b in zip(decoded, box)))
                moves.append(max(abs(box[k]+sign*box[k+3]/2-before[k]-sign*before[k+3]/2)
                                 for k in range(3) for sign in (-1, 1))*1000)
            assert max(decode_errors) < 1e-5
            modes[mode] = dict(metrics=counts, acc25_pct=counts['rec_hits25']/expected*100,
                acc50_pct=counts['rec_hits50']/expected*100, cpu_threshold_changes=threshold_changes,
                cpu_coarse_threshold_changes=coarse_threshold_changes, max_selected_iou_error=max(errors),
                max_coarse_iou_error=max(coarse_errors), max_decode_error=max(decode_errors),
                max_selected_iou_error_record=dict(line=errors.index(max(errors))+1, row_id=rows[errors.index(max(errors))]['row_id']),
                same_query_refinement=same, scalar_oracle_recount=oracle,
                first_good_strict_errors=dict(first_good), selected_size_floor_axes=sum(r[mode]['size_floor_axes'] for r in rows),
                selected_size_floor_rows=sum(r[mode]['size_floor_axes'] > 0 for r in rows),
                selected_face_offsets_min=min(v for r in rows for v in r[mode]['face_offsets']),
                selected_face_offsets_max=max(v for r in rows for v in r[mode]['face_offsets']),
                entropy_mean=statistics.fmean(v for r in rows for v in r[mode]['face_entropy']),
                max_face_move_median_mm=statistics.median(moves), threshold_transition_witnesses=witness)
        evaluations[arm][stage] = dict(rows=expected, unique_scans=len({r['scan_id'] for r in rows}),
            physical_scenes=len({r['scan_id'].split('_')[0] for r in rows}), rows_sha256=sha(path), modes=modes)

comparisons = {}
for stage in ('initial', 'terminal', 'formal'):
    face, flat = stage_rows['face', stage], stage_rows['flat', stage]
    for a, b in zip(face, flat):
        assert all(a[k] == b[k] for k in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))
    modes = {}
    for mode in ('bbs', 'bbf'):
        differences = {}
        for field in ('query', 'box', 'coarse_box', 'iou', 'coarse_iou', 'mask_iou'):
            refs = [dict(line=i, row_id=a['row_id'], face=a[mode][field], flat=b[mode][field])
                    for i, (a, b) in enumerate(zip(face, flat), 1) if a[mode][field] != b[mode][field]]
            differences[field] = dict(count=len(refs), examples=refs[:8])
        modes[mode] = dict(differences=differences,
            max_coarse_coordinate_difference=max(abs(x-y) for a, b in zip(face, flat)
                                                for x, y in zip(a[mode]['coarse_box'], b[mode]['coarse_box'])),
            outcomes={label: transitions([b[mode]['iou'] for b in flat], [a[mode]['iou'] for a in face], t)
                      for label, t in [('25', .25), ('50', .5)]})
    comparisons[stage] = dict(input_identities_exact=True, modes=modes)

within = {}
for arm in ('face', 'flat'):
    before, after = stage_rows[arm, 'initial'], stage_rows[arm, 'terminal']
    assert all(all(a[k] == b[k] for k in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))
               for a, b in zip(before, after))
    within[arm] = {mode: dict(query_changes=sum(a[mode]['query'] != b[mode]['query'] for a,b in zip(before,after)),
        coarse_box_changes=sum(a[mode]['coarse_box'] != b[mode]['coarse_box'] for a,b in zip(before,after)),
        mask_scalar_changes=[dict(line=i,row_id=a['row_id'],initial=a[mode]['mask_iou'],terminal=b[mode]['mask_iou'])
                             for i,(a,b) in enumerate(zip(before,after),1) if a[mode]['mask_iou'] != b[mode]['mask_iou']],
        outcomes={label: transitions([a[mode]['iou'] for a in before], [b[mode]['iou'] for b in after], t)
                  for label,t in [('25',.25),('50',.5)]}) for mode in ('bbs','bbf')}

train = parsed[face_root / 'train.jsonl']
ids = [i for record in train for i in record['rows']]
assert len(train) == 3723 and len(ids) == len(set(ids)) == 29778
assert [record['step'] for record in train] == list(range(1,3724))
assert all(record['total_steps'] == 3723 for record in train)
assert [len(record['rows']) for record in train] == [8]*3722 + [2]
holdout_ids = {r['row_id'] for r in stage_rows['face','initial']}
assert not set(ids) & holdout_ids and set(ids) | holdout_ids == set(range(36665))
assert all(r['boundary_targets_finite'] and r['boundary_faces'] == 6*r['boundary_matched_boxes']
           and r['boundary_matched_boxes'] == len(r['rows']) for r in train)
assert all(a['cumulative_seconds'] < b['cumulative_seconds'] for a,b in zip(train,train[1:]))
numeric = [k for k,v in train[0].items() if type(v) in (int,float)]
training = dict(records=len(train), unique_fit_rows=len(set(ids)), every_fit_row_once=True,
    batches=dict(Counter(map(lambda r: len(r['rows']),train))), last_batch=train[-1]['rows'],
    step_sequence_exact=True, holdout_overlap=0, train_plus_holdout_exact_0_to_36664=True,
    order_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
    flat_train_log_in_supplied_set=False, common_order_evidence='runner:628-629 assertion and fit receipt; flat train.jsonl was not supplied',
    all_numeric_finite=True, boundary_matched_boxes=sum(r['boundary_matched_boxes'] for r in train),
    boundary_faces=sum(r['boundary_faces'] for r in train),
    boundary_outside_faces=sum(r['boundary_target_outside'] for r in train),
    boundary_outside_steps=sum(r['boundary_target_outside'] > 0 for r in train),
    outside_step_examples=[dict(line=i,step=r['step'],rows=r['rows'],outside=r['boundary_target_outside'])
                           for i,r in enumerate(train,1) if r['boundary_target_outside'] > 0][:10],
    floor_axes_sum=sum(r['size_floor_axis_count'] for r in train),
    floor_steps=sum(r['size_floor_axis_count'] > 0 for r in train),
    raw_loss_equation_max_abs_error=max(abs(r['loss']-r['loss_native']-r['loss_assignment_correction']-r['boundary_weighted_loss']) for r in train),
    DFL_div7_max_abs_error=max(abs(r['boundary_weighted_loss']-r['boundary_loss']/7) for r in train),
    ranges={k:dict(min=min(r[k] for r in train),max=max(r[k] for r in train)) for k in numeric})

intake = obj('complete/INTAKE.json')
for rel,item in intake['files'].items():
    p = ROOT / 'complete' / rel
    assert p in raw and sha(p) == item['sha256'] and len(raw[p]) == item['bytes']
imports = obj('complete/face_conditioned/imports.json')
source_dir = ROOT.parent / 'pvground_fused_support_20261002/complete_tail_fused_retry/source/imported'
for name,digest in imports['sha256'].items():
    assert sha(source_dir/(name+'.py')) == digest
spec = obj('face_fit_spec.json')
flat_spec = parsed[ROOT.parent/'pvground_boundary_distribution_20261004/distribution_spec.json']
spec_differences = {k:dict(face=spec.get(k),flat=flat_spec.get(k)) for k in sorted(set(spec)|set(flat_spec)) if spec.get(k)!=flat_spec.get(k)}
for key,name in [('source_query_module_sha256','pvground_source_query.py'),('observation_module_sha256','pvground_observation_query.py'),
                 ('task_module_sha256','pvground_task_observation_query.py'),('assignment_module_sha256','pvground_semantic_assignment.py'),
                 ('p3_module_sha256','pvground_candidate_box_refiner.py'),('tail_module_sha256','pvground_tail_support_box_refiner.py'),
                 ('tail_preflight_module_sha256','pvground_tail_preflight.py')]:
    assert spec[key] == sha(face_root/name)
for name,digest in spec['whole_range_files'].items():
    assert sha(face_root/name) == digest
fit = obj('complete/face_conditioned/receipt.json')
assert fit['train_log_sha256'] == sha(face_root/'train.jsonl')
assert fit['script_sha256'] == sha(ROOT/'run_face_fit.py')
assert fit['spec_sha256'] == sha(ROOT/'face_fit_spec.json')
assert fit['initial'] == parsed[face_root/'initial/receipt.json']['metrics']
assert fit['terminal'] == parsed[face_root/'terminal/receipt.json']['metrics']
retention = obj('complete/face_conditioned/weight_retention.json')
status = obj('complete/status.json')
assert intake['status'] == status == json.loads(texts[ROOT/'complete/controller.log'])
assert retention == status['last_weight_retention']
assert fit['terminal_sha256'] == retention['terminal_sha256'] == retention['deleted'][0]['sha256']
assert retention['parent_sha256_after'] == spec['base_terminal_sha256'] == intake['original_g_sha256']
assert retention['formal_rows_sha256'] == sha(face_root/'formal/rows.jsonl')
assert retention['retained_best'] == status['retained_best']
assert status['retained_best']['bbs_hits50'] == evaluations['flat']['formal']['modes']['bbs']['metrics']['rec_hits50']
assert status['status'] == 'complete' and not intake['controller_alive'] and intake['controller_exit'] == 0
assert [(x['arm'],x['mode']) for x in status['completed']] == [('face_conditioned','train'),('face_conditioned','formal')]
for p in [ROOT/'complete/controller.exit',face_root/'train.exit',face_root/'formal.exit']:
    assert texts[p].strip() == '0'
assert intake['weights_downloaded'] == intake['weight_downloads'] == 0
assert not retention['local_weight_archive_created'] and intake['weights_retained'] == {}

logs = {}
for filename in ('train.log','formal.log'):
    events, plain = [], []
    for line,s in enumerate(texts[face_root/filename].splitlines(),1):
        if ' {' in s:
            label,payload=s.split(' ',1)
            value=json.loads(payload)
            assert finite(value)
            events.append(dict(line=line,event=label,value=value))
            if label == 'PVG_TRAIN_PROGRESS':
                assert value == train[value['step']-1]
            if label == 'PVG_EVAL_COMPLETE':
                assert value == parsed[face_root/value['stage']/'receipt.json']
            if label == 'PVG_FINETUNE_COMPLETE':
                assert value == fit
            if label == 'WHOLE_RANGE_INITIAL_PAIR_COMPARISON':
                assert value == obj('complete/face_conditioned/initial_control_comparison.json')
        else:
            plain.append(dict(line=line,text=s))
    logs[filename]=dict(event_counts=dict(Counter(x['event'] for x in events)),plain_lines=plain,
                       event_line_references=[dict(line=x['line'],event=x['event']) for x in events])

summary=obj('analysis/SUMMARY.json')
for stage in ('initial','terminal','formal'):
    for mode in ('bbs','bbf'):
        expected=summary['stages'][stage][mode]
        computed=evaluations['face'][stage]['modes'][mode]
        compared=comparisons[stage]['modes'][mode]
        assert expected['native_metrics'] == computed['metrics']
        assert expected['first_good_strict_errors'] == computed['first_good_strict_errors']
        assert expected['max_face_move_median_mm'] == computed['max_face_move_median_mm']
        for field,key in [('coarse_box','cross_process_coarse_box_differences'),('query','query_differences'),('mask_iou','mask_scalar_differences')]:
            assert expected[key] == compared['differences'][field]['count']
        for label in ('25','50'):
            assert expected['rec_hits'+label] == computed['metrics']['rec_hits'+label]
            assert expected['control_hits'+label] == evaluations['flat'][stage]['modes'][mode]['metrics']['rec_hits'+label]
            assert expected['same_query_refinement']['coarse_hits'+label] == computed['same_query_refinement'][label]['coarse_hits']
            for word in ('repairs','damages'):
                assert expected['same_query_refinement'][word+label] == computed['same_query_refinement'][label][word]
                assert expected['repairs_damages_vs_control'][word+label] == compared['outcomes'][label][word]

components=dict(member=14*64+64+64*64+64,condition=288*64+64,face_embedding=6*64,
                face_condition=11*64+64,range_member=4*64+64+64*64+64,
                attention=3*64*64+3*64+64*64+64,norm=2*64,
                face_update=64*128+128+128*64+64,output=64*33+33)
flat_components=dict(member=components['member'],condition=components['condition'],aggregate=1302*288+288,output=288*198+198)
assert sum(components.values())==64737 and sum(flat_components.values())==456102
step=3.0**(2.0/30)
knots=[-4.0]+[1-step**i for i in range(15,0,-1)]+[0.0]+[step**i-1 for i in range(1,16)]+[4.0]
assert len(knots)==33 and knots==sorted(knots) and all(abs(a+b)<1e-12 for a,b in zip(knots,reversed(knots)))

for p,digest in zip(PATHS,[x['sha256'] for x in identities]):
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
report=dict(status='PASS',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    execution_scope='local_cpu_stdlib_only',python=sys.version,primary_files_read=70,
    primary_bytes_read=sum(x['bytes'] for x in identities),primary_artifacts_unchanged=True,
    model_imports=0,model_forwards=0,optimizer_updates=0,weight_reads=0,network_calls=0,
    ndjson_records_parsed=sum(x.get('records_parsed',0) for x in identities),
    native_metric_recount='selected final/coarse boxes recomputed against saved dataset root boxes; saved scalar masks/oracles recounted',
    raw_mask_replay=False,full_candidate_box_replay=False,selection_logits_replay=False,model_optimizer_replay=False,
    evaluations=evaluations,face_vs_flat=comparisons,initial_to_terminal=within,training=training,
    identity_checks=dict(intake_files=len(intake['files']),native_import_sources=6,module_spec_hashes=12,
        original_g_weight_not_read=True,official_pv_weight_not_read=True,V99_chain_not_supplied=True),
    spec_differences=spec_differences,logs=logs,closure_and_retention=dict(controller_status=status,
        collection_timestamp=intake['time_cst'],claims_local_archive_created=False,claims_weight_downloads=0,
        independent_remote_readback=False,receipt_consistency=True),
    source_arithmetic=dict(face_components=components,face_parameters=sum(components.values()),face_tensors=25,
        flat_components=flat_components,flat_parameters=sum(flat_components.values()),flat_tensors=10,
        nonuniform_knots=knots,face_order=['x-','y-','z-','x+','y+','z+'],face_local_ids=[2,4,6,1,3,5],face_axes=[0,1,2,0,1,2]),
    prior_summary_reproduced=True)
(ROOT/'analysis/EXPERIMENT_AUDIT_READ_IDENTITIES.json').write_text(json.dumps(dict(request_path=str(REQUEST),
    request_sha256=hashlib.sha256(REQUEST.read_bytes()).hexdigest(),inputs=identities),indent=2)+'\n',encoding='utf-8')
(ROOT/'analysis/EXPERIMENT_AUDIT_CPU.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],primary_files=70,ndjson_records=report['ndjson_records_parsed'],
    formal={a:report['evaluations'][a]['formal']['modes']['bbs']['metrics'] for a in ('face','flat')},
    training={k:training[k] for k in ('records','unique_fit_rows','boundary_outside_faces','boundary_faces','boundary_outside_steps','floor_axes_sum','floor_steps')},
    max_box_iou_error=max(v['max_selected_iou_error'] for a in evaluations.values() for s in a.values() for v in s['modes'].values()),
    files=['EXPERIMENT_AUDIT_CPU.json','EXPERIMENT_AUDIT_READ_IDENTITIES.json']),indent=2))
