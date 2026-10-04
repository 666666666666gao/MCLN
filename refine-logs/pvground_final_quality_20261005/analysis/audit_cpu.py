"""Independent standard-library audit of saved rows; no model or network imports."""
from pathlib import Path
import ast
import collections
import csv
import datetime
import difflib
import hashlib
import json
import math
import statistics

ROOT = Path('C:/Users/gb/.codex/tmp/pvground_final_quality_20261005')
TMP = ROOT.parent
CONTROL = TMP / 'pvground_geometry_readback_20261004/formal_draft'
REV = TMP / 'pvground_geometry_readback_20261004/revision2'
PARENT = TMP / 'pvground_boundary_distribution_20261004/complete/distribution/formal'
NATIVE = TMP / 'pvground_runtime_bundle_20260908_v1/PV-Ground'
DATA = TMP / 'pvground_g_p2_20261002/complete/source'
OUT = ROOT / 'analysis'
hashes = {}
assertions = []


def raw(path):
    path = Path(path)
    data = path.read_bytes()
    hashes[path.as_posix()] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return data


def read(path):
    return json.loads(raw(path))


def lines(path):
    return [json.loads(s) for s in raw(path).decode('utf-8').splitlines()]


def verify(condition, name):
    assert condition, name
    assertions.append(name)


for filename in ['EXPERIMENT_PLAN.md', 'quality_fit_spec.json', 'fit_launch.json', 'fit_wait.json',
                 'SOURCE_REVIEW.md', 'SOURCE_REVIEW.json', 'controller.py', 'native_final_quality.py',
                 'collect_formal_authorized.py', 'analyze_closed_formal.py', 'check_closed_resources.py',
                 'CLOSED_RESOURCES.json', 'PARENT_MASK_BOUNDARY_RELATION.json',
                 'analysis/SUMMARY.json', 'analysis/FORMAL_METRICS.csv', 'analysis/RESULTS.md']:
    raw(ROOT / filename)
for path in sorted((ROOT / 'runtime_bundle').glob('*.py')):
    ast.parse(raw(path).decode('utf-8'), feature_version=(3, 7))
for path in [CONTROL/'runtime_bundle/run_readback_fit.py', REV/'readback_remote_source_receipt.json',
             REV/'complete_preflight/source_port.json', REV/'source_preview/PV-Ground/models/pv_ground.py',
             REV/'source_preview/PV-Ground/models/modules.py', NATIVE/'models/losses.py',
             NATIVE/'src/grounding_evaluator.py', NATIVE/'src/joint_det_dataset.py',
             DATA/'joint_det_dataset.py', DATA/'input_manifest.json', DATA/'split_protocol.json']:
    raw(path)

intake_checks = {}
for base, prefix in [(ROOT/'complete', ''), (ROOT/'preflight_complete', ''), (CONTROL/'complete', 'evidence_visible/')]:
    intake = read(base/'INTAKE.json')
    count = 0
    for relative, expected in intake['files'].items():
        if prefix and not relative.startswith(prefix):
            continue
        payload = raw(base/relative)
        verify(len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256'],
               'intake identity: ' + (base/relative).as_posix())
        if relative.endswith('.json'):
            json.loads(payload)
        if relative.endswith('.exit'):
            verify(int(payload) == 0, 'exit zero: ' + (base/relative).as_posix())
        count += 1
    terminal = intake['actual_terminal'] if 'actual_terminal' in intake else intake['remote_terminal']
    verify(terminal['status']['status'] == 'complete' and not terminal['controller_alive'] and terminal['exitcode'] == 0,
           'closed controller: ' + base.as_posix())
    intake_checks[base.as_posix()] = {'verified_files': count, 'closed_exit_zero': True}

spec = read(ROOT/'complete/quality/spec.json')
control_spec = read(CONTROL/'complete/evidence_visible/spec.json')
fit = read(ROOT/'complete/quality/receipt.json')
oldfit = read(CONTROL/'complete/evidence_visible/receipt.json')
preflight = read(ROOT/'complete/preflight/preflight.json')
source_review = read(ROOT/'SOURCE_REVIEW.json')
imports = read(ROOT/'complete/quality/imports.json')
port = read(REV/'complete_preflight/source_port.json')
remote_source = read(REV/'readback_remote_source_receipt.json')
env_path = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_runtime_20260908_v1/witness_repair_v3/env_spec.json')
env = read(env_path)
env_canonical = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
verify(env_canonical == spec['env_spec_sha256'], 'canonical environment identity')
verify(raw(ROOT/'native_final_quality.py') == raw(ROOT/'runtime_bundle/native_final_quality.py'), 'quality helper copies identical')
verify(raw(ROOT/'quality_fit_spec.json') == raw(ROOT/'complete/quality/spec.json'), 'planned and executed fit specs identical')
verify(raw(ROOT/'SOURCE_REVIEW.json') == raw(ROOT/'complete/SOURCE_REVIEW.json'), 'review receipt copy identical')
verify(raw(ROOT/'fit_launch.json') == raw(ROOT/'complete/fit_launch.json'), 'launch receipt copy identical')
for name, sha in spec['runner_files'].items():
    verify(hashes[(ROOT/'runtime_bundle'/name).as_posix()]['sha256'] == sha, 'pinned runtime helper: '+name)
inherited = [name for name in control_spec['runner_files'] if name != 'run_readback_fit.py']
verify(all(control_spec['runner_files'][name] == spec['runner_files'][name] for name in inherited), 'inherited helper pinning identical')
verify(fit['script_sha256'] == preflight['runner_sha256'] == spec['runner_files']['run_final_quality_fit.py'], 'executed runner receipt hashes')
verify(oldfit['script_sha256'] == hashes[(CONTROL/'runtime_bundle/run_readback_fit.py').as_posix()]['sha256'], 'executed control runner hash')
for component, path in [('evaluator', NATIVE/'src/grounding_evaluator.py'), ('models.losses', NATIVE/'models/losses.py'),
                        ('models.pv_ground', REV/'source_preview/PV-Ground/models/pv_ground.py'),
                        ('src.joint_det_dataset', DATA/'joint_det_dataset.py')]:
    verify(imports['sha256'][component] == hashes[path.as_posix()]['sha256'], 'actual import identity: '+component)
verify(hashes[(REV/'complete_preflight/source_port.json').as_posix()]['sha256'] == spec['source_port_sha256'] == remote_source['source_port_sha256'], 'source-port receipt identity')
for name in ['models/pv_ground.py', 'models/modules.py']:
    verify(hashes[(REV/'source_preview/PV-Ground'/name).as_posix()]['sha256'] == port['files'][name] == remote_source['overrides'][name], 'model preview identity: '+name)
verify(hashes[(ROOT/'complete/quality/spec.json').as_posix()]['sha256'] == fit['spec_sha256'], 'fit receipt spec identity')
verify(hashes[(CONTROL/'complete/evidence_visible/spec.json').as_posix()]['sha256'] == oldfit['spec_sha256'], 'control receipt spec identity')
expected_spec_changes = {'root', 'runner_files', 'quality_weight', 'control_root'}
spec_changes = [key for key in sorted(set(spec)|set(control_spec)) if spec.get(key) != control_spec.get(key)]
verify(set(spec_changes) == expected_spec_changes, 'only stated control spec changes')
manifest = read(DATA/'input_manifest.json')
split = read(DATA/'split_protocol.json')
verify(hashes[(DATA/'split_protocol.json').as_posix()]['sha256'] == manifest['split_protocol_sha256'], 'dataset split manifest identity')


def iou(box, gt):
    verify_shape = len(box) == len(gt) == 6 and all(math.isfinite(v) for v in box+gt)
    assert verify_shape and all(v > 0 for v in box[3:]+gt[3:])
    a_low = [box[k]-box[k+3]*0.5 for k in range(3)]
    a_high = [box[k]+box[k+3]*0.5 for k in range(3)]
    b_low = [gt[k]-gt[k+3]*0.5 for k in range(3)]
    b_high = [gt[k]+gt[k+3]*0.5 for k in range(3)]
    overlap = math.prod(max(0, min(a_high[k], b_high[k])-max(a_low[k], b_low[k])) for k in range(3))
    return overlap/(math.prod(box[3:])+math.prod(gt[3:])-overlap)


def effect(old, new, oldscope='bbs', newscope='bbs'):
    assert len(old) == len(new)
    assert all(all(a[k] == b[k] for k in ['row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256']) for a,b in zip(old,new))
    result = {}
    for threshold in [0.25, 0.5]:
        repair = sum(iou(a[oldscope]['box'],a['root_box']) <= threshold < iou(b[newscope]['box'],b['root_box']) for a,b in zip(old,new))
        damage = sum(iou(b[newscope]['box'],b['root_box']) <= threshold < iou(a[oldscope]['box'],a['root_box']) for a,b in zip(old,new))
        result[str(threshold)] = {'repairs': repair, 'damages': damage, 'net': repair-damage,
                                 'selected_query_changes': sum(a[oldscope]['query'] != b[newscope]['query'] for a,b in zip(old,new))}
    return result


def stage_summary(records, receipt, mode='bbs', diagnostic=True):
    assert len(records) == receipt['rows']
    scopes = [('selected', mode, 'box', 'iou'), ('coarse_same_query', mode, 'coarse_box', 'coarse_iou')]
    if diagnostic:
        scopes.append(('bypass_fixed_frame', 'bypass_fixed_frame', 'box', 'iou'))
    recon = {}
    for label, scope, box_key, iou_key in scopes:
        u = [iou(r[scope][box_key],r['root_box']) for r in records]
        changes = [sum((value > t) != (r[scope][iou_key] > t) for value,r in zip(u,records)) for t in [0.25,0.5]]
        assert changes == [0,0], (label,changes)
        recon[label] = {'hits25': sum(v>0.25 for v in u), 'hits50': sum(v>0.5 for v in u),
                        'threshold_disagreements25_50': changes,
                        'max_absolute_iou_error': max(abs(v-r[scope][iou_key]) for v,r in zip(u,records))}
    masks = [r[mode]['mask_iou'] for r in records]
    assert all(math.isfinite(v) and 0<=v<=1 for v in masks)
    metrics = dict(rec_hits25=recon['selected']['hits25'],rec_hits50=recon['selected']['hits50'],
                   mask_hits25=sum(v>.25 for v in masks),mask_hits50=sum(v>.5 for v in masks),
                   mask_miou=statistics.mean(masks)*100)
    for key,value in metrics.items():
        assert math.isclose(value,receipt['metrics'][mode][key],rel_tol=0,abs_tol=1e-9),(key,value)
    metrics.update(rec_acc25=metrics['rec_hits25']/len(records)*100, rec_acc50=metrics['rec_hits50']/len(records)*100)
    availability = {}
    for t,key in [(0.25,'oracle25'),(0.5,'oracle50')]:
        flags = [r[mode][key] for r in records]
        assert all(len(v)==4 and all(x in (0,1) for x in v) and v==sorted(v) for v in flags)
        assert all(not(r[mode]['iou']>t) or v[0] for r,v in zip(records,flags))
        availability[str(t)] = {'topk':[16,32,64,256], 'oracle_hits':[sum(v[k] for v in flags) for k in range(4)],
                                'errors_with_good_full256':sum(r[mode]['iou']<=t and v[-1] for r,v in zip(records,flags)),
                                'errors_without_good_full256':sum(r[mode]['iou']<=t and not v[-1] for r,v in zip(records,flags))}
    direct = effect(records,records,'bypass_fixed_frame') if diagnostic else None
    if diagnostic:
        assert all(r['same_forward_geometry_exact'] and r['native_head_calls']==1 and r['diagnostic_native_head_replay_calls']==1 for r in records)
        for t,d in direct.items():
            exp = receipt['fixed_frame_readback_effect'][t]
            assert [d['repairs'],d['damages'],d['net']] == [exp['fixes'],exp['damages'],exp['net']]
    return dict(rows=len(records), scenes=len({r['scan_id'] for r in records}), physical_spaces=len({r['scan_id'].split('_')[0] for r in records}),
                target_objects=len({(r['scan_id'],r['target_id']) for r in records}), metrics=metrics,
                box_reconstruction=recon, fixed_frame_readback_effect=direct, candidate_availability=availability,
                mask_scope='Reaggregated stored IoUs; raw predicted/GT Mask tensors are absent.',
                availability_scope='Reaggregated stored flags and checked monotonicity; all256 candidate boxes/ranks are absent.')


orders, trainchecks, rows, stages, holdout_transitions = {}, {}, {}, {}, {}
for arm, base in [('final_quality',ROOT/'complete/quality'), ('native_g_control',CONTROL/'complete/evidence_visible')]:
    train = lines(base/'train.jsonl')
    receipt = read(base/'receipt.json')
    verify(hashes[(base/'train.jsonl').as_posix()]['sha256'] == receipt['train_log_sha256'], 'train receipt hash: '+arm)
    order = [rid for step in train for rid in step['rows']]
    verify([step['step'] for step in train] == list(range(1,3724)), 'sequential updates: '+arm)
    verify(all(len(r['rows'])==(2 if r['step']==3723 else 8) for r in train), 'batch sizes: '+arm)
    verify(len(order)==len(set(order))==29778 and set(order)==set(split['row_ids']['fit']), 'complete one-pass split: '+arm)
    verify(all(r['native_head_calls']==1 and r['geometry_and_mask_same_frame_exact'] and
               r['call_order']==['refiner','readback','native_semantic_head'] for r in train), 'all train same-frame flags: '+arm)
    verify(all(all(math.isfinite(r[k]) for k in ['loss','native_loss','assignment_correction','gradient_norm']) for r in train), 'finite train scalars: '+arm)
    orders[arm] = order
    trainchecks[arm] = dict(updates=len(train),rows=len(order),distinct_rows=len(set(order)),batch8=3722,batch2=1,
                           effective_batch=8,accumulation_steps=1,order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
    trainchecks[arm]['matched_gt_minus_batch_histogram'] = dict(collections.Counter(r['assignment_counts']['matched_queries']-len(r['rows']) for r in train))
    if arm == 'final_quality':
        verify(all(r['quality_weight']==1.0 and math.isfinite(r['quality_loss']) for r in train), 'fixed finite quality objective')
        verify(all(r['quality_counts']['pool_queries']==r['assignment_counts']['reassigned_queries']+len(r['rows']) and
                   r['quality_counts']['pool_queries']==sum(r['quality_counts']['pool_per_sample']) and
                   len(r['quality_counts']['pool_per_sample'])==len(r['rows']) and
                   all(1<=n<=256 for n in r['quality_counts']['pool_per_sample']) and
                   r['quality_counts']['final_iou_detached'] and r['quality_counts']['other_matched_queries_excluded'] for r in train), 'all real pool counts match original G plus one root per sample')
        quality_curve = dict(first100_mean=statistics.mean(r['quality_loss'] for r in train[:100]),
                             last100_mean=statistics.mean(r['quality_loss'] for r in train[-100:]),
                             min=min(r['quality_loss'] for r in train),max=max(r['quality_loss'] for r in train))
        trainchecks[arm].update(quality_curve=quality_curve, singleton_sample_count=sum(n==1 for r in train for n in r['quality_counts']['pool_per_sample']),
                               max_float_loss_decomposition_error=max(abs(r['loss']-r['native_loss']-r['assignment_correction']-r['quality_loss']) for r in train))
    rows[arm], stages[arm] = {}, {}
    for stage in ['initial','terminal','formal']:
        data = lines(base/stage/'rows.jsonl')
        rec = read(base/stage/'receipt.json')
        verify(hashes[(base/stage/'rows.jsonl').as_posix()]['sha256']==rec['rows_sha256'], 'stage row receipt hash: '+arm+'/'+stage)
        verify([r['row_id'] for r in data] == (list(range(9508)) if stage=='formal' else split['row_ids']['holdout']), 'stage full ordered traversal: '+arm+'/'+stage)
        rows[arm][stage] = data
        stages[arm][stage] = stage_summary(data,rec)
    holdout_transitions[arm] = effect(rows[arm]['initial'],rows[arm]['terminal'])
    for t,e in holdout_transitions[arm].items():
        assert [e['repairs'],e['damages'],e['net']] == [receipt['transitions'][t][k] for k in ['fixes','damages','net']]
    restore, retention = read(base/'formal_restore.json'), read(base/'weight_retention.json')
    verify(restore['terminal_sha256']==receipt['terminal_sha256']==retention['terminal_sha256']==retention['deleted'][0]['sha256'], 'terminal restore/delete identity: '+arm)
    verify(restore['strict_model_restore'] and restore['optimizer']['all_keys_moments_steps_and_groups_exact'] and restore['restored_steps']==3723,
           'formal restoration receipt: '+arm)
verify(orders['final_quality']==orders['native_g_control'], 'every reused-control batch order identical')

parent_rows = lines(PARENT/'rows.jsonl')
parent_receipt = read(PARENT/'receipt.json')
verify(hashes[(PARENT/'rows.jsonl').as_posix()]['sha256']==parent_receipt['rows_sha256'], 'parent rows receipt identity')
parent_stages = {m:stage_summary(parent_rows,parent_receipt,m,False) for m in ['bbs','bbf']}
comparisons = {s:effect(rows['native_g_control'][s],rows['final_quality'][s]) for s in ['initial','terminal','formal']}
versus_parent = {arm:effect(parent_rows,rows[arm]['formal']) for arm in rows}
volume_order = sorted(range(9508),key=lambda i:(math.prod(parent_rows[i]['root_box'][3:]),i))
groups = []
for q in range(4):
    ix=volume_order[q*2377:(q+1)*2377]
    a,b=[rows['native_g_control']['formal'][i] for i in ix],[rows['final_quality']['formal'][i] for i in ix]
    groups.append(dict(volume_quartile=q+1,rows=len(ix),control_hits50=sum(iou(r['bbs']['box'],r['root_box'])>.5 for r in a),
                       quality_hits50=sum(iou(r['bbs']['box'],r['root_box'])>.5 for r in b),quality_vs_control=effect(a,b)['0.5']))


def cross_process(a,b,scope_a='bbs',scope_b='bbs'):
    assert all(all(x[k]==y[k] for k in ['row_id','scan_id','target_id','root_box','point_sha256']) for x,y in zip(a,b))
    same=[(x,y) for x,y in zip(a,b) if x[scope_a]['query']==y[scope_b]['query']]
    return dict(rows=len(a),same_selected_query=len(same),changed_query_rows=[x['row_id'] for x,y in zip(a,b) if x[scope_a]['query']!=y[scope_b]['query']],
                same_query_exact_box_rows=sum(x[scope_a]['box']==y[scope_b]['box'] for x,y in same),
                max_same_query_box_coordinate_difference=max(abs(u-v) for x,y in same for u,v in zip(x[scope_a]['box'],y[scope_b]['box'])),
                same_query_threshold_disagreements={str(t):sum((iou(x[scope_a]['box'],x['root_box'])>t)!=(iou(y[scope_b]['box'],y['root_box'])>t) for x,y in same) for t in [.25,.5]})


roundoff = {'initial_control_quality':cross_process(rows['native_g_control']['initial'],rows['final_quality']['initial'])}
for arm in rows:
    roundoff[arm+'_formal_bypass_vs_parent']=cross_process(parent_rows,rows[arm]['formal'],'bbs','bypass_fixed_frame')

# Parent selected-query boundary diagnostic, rederived from stored centers/sizes.
relation = read(ROOT/'PARENT_MASK_BOUNDARY_RELATION.json')
quadrants = {}
for box_ok in [True,False]:
    for mask_ok in [True,False]:
        subset=[r for r in parent_rows if (iou(r['bbs']['box'],r['root_box'])>.5)==box_ok and (r['bbs']['mask_iou']>.5)==mask_ok]
        errors,moves=[],[]
        toward=denominator=repairs=damages=0
        for r in subset:
            def faces(box):
                return [box[i]-box[i+3]/2 for i in range(3)]+[box[i]+box[i+3]/2 for i in range(3)]
            coarse,final,gt=faces(r['bbs']['coarse_box']),faces(r['bbs']['box']),faces(r['root_box'])
            errors.append(max(abs(x-y) for x,y in zip(final,gt)))
            moves.append(max(abs(x-y) for x,y in zip(final,coarse)))
            for c,f,g in zip(coarse,final,gt):
                if f!=c:
                    denominator+=1
                    toward+=(f-c)*(g-c)>0
            old,new=iou(r['bbs']['coarse_box'],r['root_box'])>.5,iou(r['bbs']['box'],r['root_box'])>.5
            repairs+=not old and new
            damages+=old and not new
        item=dict(rows=len(subset),box_qualified=box_ok,mask_qualified=mask_ok,median_max_face_error_m=statistics.median(errors),
                  median_max_face_move_m=statistics.median(moves),coarse_hits50=sum(iou(r['bbs']['coarse_box'],r['root_box'])>.5 for r in subset),
                  final_hits50=sum(iou(r['bbs']['box'],r['root_box'])>.5 for r in subset),same_query_repairs50=repairs,same_query_damages50=damages,
                  moving_faces_direction_toward_gt=toward,moving_faces_direction_denominator=denominator)
        quadrants[str((box_ok,mask_ok))]=item
verify(quadrants==relation['quadrants'], 'parent mask/boundary quadrant values')

summary=read(OUT/'SUMMARY.json')
verify(comparisons==summary['quality_vs_control'], 'all pairwise summary values')
verify(versus_parent==summary['versus_protected_parent'], 'all parent comparison values')
verify(groups==summary['formal_gt_volume_groups'], 'all GT-volume quartile values')
for arm in stages:
    for stage,rec in stages[arm].items():
        expected=summary['stages'][arm][stage]
        verify(all(math.isclose(v,expected[k],rel_tol=0,abs_tol=1e-9) for k,v in rec['metrics'].items()), 'summary stage metrics: '+arm+'/'+stage)
        verify(rec['fixed_frame_readback_effect']==expected['fixed_frame_readback_effect'], 'summary bypass effect: '+arm+'/'+stage)
        verify(rec['candidate_availability']==expected['candidate_availability'], 'summary saved coverage flags: '+arm+'/'+stage)
verify(all(math.isclose(v,summary['quality_curve'][k],rel_tol=0,abs_tol=1e-15) for k,v in quality_curve.items()),'reported quality curve values')
with (OUT/'FORMAL_METRICS.csv').open(newline='') as stream:
    table=list(csv.DictReader(stream))
for row in table:
    expected=parent_stages['bbs']['metrics'] if row['system']=='protected_geometry_parent' else stages[row['system']]['formal']['metrics']
    verify(all(math.isclose(float(v),expected[k],rel_tol=0,abs_tol=1e-12) for k,v in row.items() if k!='system'),'CSV numeric identity: '+row['system'])

resources=read(ROOT/'CLOSED_RESOURCES.json')
expected_parents={spec['geometry_terminal']:spec['geometry_terminal_sha256'],spec['base_terminal']:spec['base_terminal_sha256'],
                  env['weight_dirs']['scanrefer']['path']:spec['checkpoint_sha256']}
verify({p:v['sha256'] for p,v in resources['protected_weights'].items()}==expected_parents,'post-cleanup three required parent identities')
verify(resources['nonbest_weight_absent']==spec['root']+'/terminal.pth','post-cleanup new nonbest absence receipt')
verify(resources['gpu_compute_processes']=='' and not resources['inference_or_optimizer_replayed'],'post-cleanup no compute process receipt')
verify(preflight['optimizer_steps']==2 and not preflight['accuracy_result'] and preflight['weight_files_created']==0,'actual two-update no-accuracy preflight')
for witness in preflight['witnesses']:
    q=witness['quality_route']
    verify(q['independent_pairwise_error']==0 and q['independent_logit_gradient_error']<1e-7 and
           q['nonpool_direct_gradient_zero'] and q['other_matched_direct_gradient_zero'] and
           q['equal_score_quality_direction_verified'] and q['final_geometry_target_detached'], 'preflight quality witness '+str(witness['previous_updates']))

for stage in ['fit','preflight']:
    verify(json.loads(raw(ROOT/'complete'/(stage+'_controller.log'))) == read(ROOT/'complete'/(stage+'_status.json')), 'controller log/status equality: '+stage)
for base in [ROOT/'complete/quality', CONTROL/'complete/evidence_visible']:
    entries=[]
    for log in ['train.log','formal.log']:
        for text in raw(base/log).decode('utf-8').splitlines():
            if text.startswith('READBACK_EVAL_COMPLETE '):
                receipt=json.loads(text.split(' ',1)[1])
                verify(receipt==read(base/receipt['stage']/'receipt.json'),'evaluation log/receipt equality: '+base.name+'/'+receipt['stage'])
            if text.startswith('READBACK_FIT_COMPLETE '):
                verify(json.loads(text.split(' ',1)[1])==read(base/'receipt.json'),'fit log/receipt equality: '+base.name)
preflight_logs=[json.loads(s.split(' ',1)[1]) for s in raw(ROOT/'complete/preflight/preflight.log').decode('utf-8').splitlines() if s.startswith('FINAL_QUALITY_PREFLIGHT_COMPLETE ')]
verify(preflight_logs==[preflight],'preflight log/receipt equality')
verify(read(ROOT/'complete/preflight/imports.json')==imports==read(CONTROL/'complete/evidence_visible/imports.json'),'preflight/new/control import receipts identical')

LEGACY=TMP/'pvground_fused_support_20261002/candidate_audit/full_gt_scope_v2'
raw(ROOT/'recount_legacy_mask_match_roles.py')
raw(TMP/'pvground_fused_support_20261002/full_candidate_audit_body.py')
legacy_summary=read(ROOT/'LEGACY_MASK_QUALIFIED_MATCH_ROLE.json')
legacy_receipt=read(LEGACY/'receipt.json')
legacy_rows=lines(LEGACY/'rows.jsonl')
verify(hashes[(LEGACY/'rows.jsonl').as_posix()]['sha256']==legacy_receipt['rows_sha256']==legacy_summary['source_rows_sha256'],'legacy diagnostic row hash')
verify(len(legacy_rows)==9508 and [r['row_id'] for r in legacy_rows]==list(range(9508)),'legacy diagnostic full traversal')
verify(all(r['valid_native_GT_slots']==[0] and r['candidate_count']==256 for r in legacy_rows),'legacy validation matcher root-only GT slots')
verify(all(all(a[k]==b[k] for k in ['row_id','scan_id','target_id','root_box','point_sha256']) for a,b in zip(legacy_rows,parent_rows)), 'legacy/current formal row and GT identity')
legacy_subset=[r for r in legacy_rows if r['selected_mask_iou']>.5 and r['selected_iou']<=.5]
legacy_role=lambda r:'matched_root' if r['selected_matched_slot']==0 else 'unmatched' if r['selected_matched_slot']==-1 else 'matched_other'
role_counts=dict(collections.Counter(legacy_role(r) for r in legacy_subset))
verify(len(legacy_subset)==legacy_summary['selected_mask_good_box_bad'] and role_counts==legacy_summary['selected_match_roles'],'legacy selected Mask-good/Box-bad match roles')
verify({role:[r['row_id'] for r in legacy_subset if legacy_role(r)==role] for role in role_counts}==legacy_summary['row_ids_by_role'],'legacy role row IDs')
legacy_hits=[sum(r['selected_iou']>t for r in legacy_rows) for t in [.25,.5]]
verify(legacy_hits==legacy_summary['source_formal_hits']==[legacy_receipt['rec_hits'][s] for s in ['25','50']],'legacy stored selected IoU counts')
verify(sum(not r['root_in_scene_detection_GT'] for r in legacy_rows)==legacy_receipt['root_excluded_from_scene_detection_GT_rows'],'legacy detection-vocabulary exclusion count')
legacy_check=dict(rows=9508,stored_selected_hits=legacy_hits,selected_mask_good_box_bad=len(legacy_subset),roles=role_counts,
                   root_only_validation_matches=True,raw_mask_or_match_replay=False,source_body_identity_not_pinned_in_receipt=True,
                   scope='Historical tail_fused validation diagnostic only; no actual training responsibility, physical identity, protected-parent responsibility, or efficacy conclusion.')

report=dict(status='PASS',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),executor='standard-library Python 3.12; CPU only',
            neural_network_or_optimizer_replayed=False,network_used=False,weights_loaded=False,
            intake_checks=intake_checks,source_identity=dict(runtime_modules=17,inherited_helper_pins_identical=len(inherited),spec_changes=spec_changes,
              env_canonical_sha256=env_canonical,dataset_import_matches_historical_collected_loader=True,
              runtime_bundle_dataset_is_actual_import=False,input_manifest_file_not_directly_pinned_by_new_spec=True),
            train=trainchecks,training_order_exact=True,stages=stages,parent_stages=parent_stages,
            quality_vs_control=comparisons,versus_parent=versus_parent,holdout_transitions=holdout_transitions,
            formal_gt_volume_groups=groups,cross_process_saved_box_differences=roundoff,parent_mask_boundary_relation=quadrants,
            legacy_mask_qualified_match_roles=legacy_check,
            restored_and_cleanup_receipts_consistent=True,protected_parents_checked_by_remote_receipt=resources['protected_weights'],
            historical_v99_preservation='Not rehashed in supplied terminal receipts; no deletion path targets it.',
            scope_limitations=['Raw ScanRefer/ScanNet annotations and point/Mask tensors were not replayed.',
              'All256 availability, native ranks, model state immutability and optimizer restoration are code/receipt evidence; saved tensors are incomplete.',
              'Input manifest is an authorized historical local copy; new run validates its child source/split checksums but does not record the manifest file SHA.',
              'Independent complete forwards are not bitwise identical; same-frame assertions have narrower scope.',
              'One seed and a reused control on developer validation cannot establish general causal efficacy.'],
            checks_passed=len(assertions),checks=assertions)
(OUT/'AUDIT_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(OUT/'AUDITED_INPUT_SHA256.json').write_text(json.dumps(dict(generated_at=report['generated_at'],algorithm='sha256',files=hashes),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',assertions=len(assertions),input_files=len(hashes),formal_metrics={k:v['formal']['metrics'] for k,v in stages.items()},
                      compared_to_control=comparisons['formal'],roundoff=roundoff),indent=2))
