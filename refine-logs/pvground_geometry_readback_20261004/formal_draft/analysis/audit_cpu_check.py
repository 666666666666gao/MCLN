"""Independent terminal audit. Stdlib-only; no experiment imports or mutations."""
import ast
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/formal_draft')
TMP = ROOT.parent.parent
COMPLETE = ROOT / 'complete'
OUT = ROOT / 'analysis'
HASHES = {}


def raw(path):
    path = Path(path)
    data = path.read_bytes()
    HASHES[path.as_posix()] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return data


def js(path):
    return json.loads(raw(path))


def rows(path):
    return [json.loads(line) for line in raw(path).decode('utf-8').splitlines()]


def iou(box, gt):
    assert len(box) == len(gt) == 6
    assert all(math.isfinite(x) for x in box + gt)
    assert min(box[3:] + gt[3:]) > 0
    intersection = 1.0
    for axis in range(3):
        left = max(box[axis] - box[axis+3]/2, gt[axis] - gt[axis+3]/2)
        right = min(box[axis] + box[axis+3]/2, gt[axis] + gt[axis+3]/2)
        intersection *= max(right-left, 0)
    return intersection / (math.prod(box[3:]) + math.prod(gt[3:]) - intersection)


def pair(before, after, threshold, before_field='bbs'):
    assert len(before) == len(after)
    repairs, damages, changes, coverable, missing = [], [], [], [], []
    for old, new in zip(before, after):
        assert all(old[k] == new[k] for k in ('row_id','scan_id','target_id','root_box','point_sha256'))
        a = iou(old[before_field]['box'], old['root_box']) > threshold
        b = iou(new['bbs']['box'], new['root_box']) > threshold
        if old[before_field]['query'] != new['bbs']['query']:
            changes.append(new['row_id'])
        if b and not a:
            repairs.append(new['row_id'])
            good = old['bbs']['oracle25' if threshold == .25 else 'oracle50'][-1]
            (coverable if good else missing).append(new['row_id'])
        if a and not b:
            damages.append(new['row_id'])
    return dict(repairs=len(repairs), damages=len(damages), net=len(repairs)-len(damages),
                selected_query_changes=len(changes), repairs_from_coverable_errors=len(coverable),
                repairs_from_missing_candidate_errors=len(missing), repair_row_ids=repairs,
                damage_row_ids=damages, changed_row_ids=changes)


def compact_pair(result):
    return {k:v for k,v in result.items() if not k.endswith('_ids')}


def row_check(data, receipt, readback=True):
    assert len(data) == receipt['rows']
    fields = ['bbs','bypass_fixed_frame'] if readback else ['bbs']
    result = {'rows':len(data), 'scans':len({x['scan_id'] for x in data}),
              'physical_scenes':len({x['scan_id'].split('_')[0] for x in data})}
    max_error = {key:0. for key in fields + ['coarse']}
    changes = {key:0 for key in max_error}
    for row in data:
        assert 0 <= row['bbs']['query'] < 256
        assert math.isfinite(row['bbs']['mask_iou']) and 0 <= row['bbs']['mask_iou'] <= 1
        if readback:
            assert row['same_forward_geometry_exact'] and row['native_head_calls'] == 1
            assert row['diagnostic_native_head_replay_calls'] == 1
            assert 0 <= row['bypass_fixed_frame']['query'] < 256
            if row['bbs']['query'] == row['bypass_fixed_frame']['query']:
                assert row['bbs']['box'] == row['bypass_fixed_frame']['box']
        for field in fields:
            value = iou(row[field]['box'], row['root_box'])
            max_error[field] = max(max_error[field], abs(value-row[field]['iou']))
            changes[field] += sum((value > t) != (row[field]['iou'] > t) for t in (.25,.5))
        value = iou(row['bbs']['coarse_box'], row['root_box'])
        max_error['coarse'] = max(max_error['coarse'],abs(value-row['bbs']['coarse_iou']))
        changes['coarse'] += sum((value > t) != (row['bbs']['coarse_iou'] > t) for t in (.25,.5))
        for field, threshold in [('oracle25',.25),('oracle50',.5)]:
            oracle = row['bbs'][field]
            assert len(oracle) == 4 and all(x in (0,1) for x in oracle)
            assert oracle == sorted(oracle)
            if iou(row['bbs']['box'],row['root_box']) > threshold:
                assert all(oracle)
        assert all(a >= b for a,b in zip(row['bbs']['oracle25'],row['bbs']['oracle50']))
    assert all(v == 0 for v in changes.values())
    metric = receipt['metrics']['bbs']
    for t,suffix in [(.25,'25'),(.5,'50')]:
        hits = sum(iou(row['bbs']['box'], row['root_box']) > t for row in data)
        assert hits == metric['rec_hits'+suffix]
        result['rec_hits'+suffix] = hits
        result['rec_acc'+suffix] = hits/len(data)*100
        result['mask_hits'+suffix] = sum(row['bbs']['mask_iou'] > t for row in data)
        assert result['mask_hits'+suffix] == metric['mask_hits'+suffix]
    result['mask_miou'] = sum(row['bbs']['mask_iou'] for row in data)/len(data)*100
    assert result['mask_miou'] == metric['mask_miou']
    result['float64_iou_max_absolute_difference'] = max_error
    result['iou_threshold_disagreements'] = changes
    result['full256_coverage_from_saved_flags'] = {
        str(t):dict(topk=[16,32,64,256], oracle_hits=[sum(x['bbs'][key][i] for x in data) for i in range(4)],
                    errors_with_good_full256=sum(iou(x['bbs']['box'],x['root_box']) <= t and x['bbs'][key][-1] for x in data),
                    errors_without_good_full256=sum(iou(x['bbs']['box'],x['root_box']) <= t and not x['bbs'][key][-1] for x in data))
        for t,key in [(.25,'oracle25'),(.5,'oracle50')]}
    if readback:
        result['fixed_frame_readback_effect'] = {str(t):pair(data,data,t,'bypass_fixed_frame') for t in (.25,.5)}
        for t,p in result['fixed_frame_readback_effect'].items():
            assert p['repairs'] == receipt['fixed_frame_readback_effect'][t]['fixes']
            assert p['damages'] == receipt['fixed_frame_readback_effect'][t]['damages']
            assert p['net'] == receipt['fixed_frame_readback_effect'][t]['net']
    return result


intake = js(COMPLETE/'INTAKE.json')
for rel, ident in intake['files'].items():
    raw(COMPLETE/rel)
    assert HASHES[(COMPLETE/rel).as_posix()] == ident, rel
all_complete = [p for p in COMPLETE.rglob('*') if p.is_file()]
assert set(p.relative_to(COMPLETE).as_posix() for p in all_complete) == set(intake['files']) | {'INTAKE.json'}
for path in all_complete:
    if path.suffix == '.json':
        js(path)
    if path.suffix == '.exit':
        assert raw(path).strip() == b'0'
status = js(COMPLETE/'status.json')
wait = js(ROOT/'wait.json')
assert status == intake['remote_terminal']['status'] == wait['terminal']['status']
assert status['status'] == 'complete' and not status['scanrefer_target_pass']
assert intake['remote_terminal']['exitcode'] == 0 and not intake['remote_terminal']['controller_alive']
assert not intake['downloaded_weights'] and not intake['inference_or_optimizer_replayed']
assert json.loads(raw(COMPLETE/'controller.log').decode().strip()) == status

split_path = TMP/'pvground_g_p2_20261002/complete/source/split_protocol.json'
split = js(split_path)
manifest_path = TMP/'pvground_g_p2_20261002/complete/source/input_manifest.json'
manifest = js(manifest_path)
assert HASHES[split_path.as_posix()]['sha256'] == manifest['split_protocol_sha256']
assert len(split['row_ids']['fit']) == 29778 and len(split['row_ids']['holdout']) == 6887
assert set(split['row_ids']['fit']).isdisjoint(split['row_ids']['holdout'])
assert set(split['row_ids']['fit']) | set(split['row_ids']['holdout']) == set(range(36665))

stages, data_by_key, orders, train_report, identity_report = {}, {}, {}, {}, {}
for arm in ('evidence_hidden','evidence_visible'):
    base = COMPLETE/arm
    spec = js(base/'spec.json')
    assert raw(base/'spec.json') == raw(ROOT/(arm+'_fit_spec.json'))
    fit = js(base/'receipt.json')
    restored = js(base/'formal_restore.json')
    retained = js(base/'weight_retention.json')
    load = js(base/'load.json')
    imports = js(base/'imports.json')
    assert spec['seed'] == 2027 and spec['batch_size'] == 8 and spec['updates'] == 3723 and spec['fit_passes'] == 1
    assert fit['status'] == 'complete' and fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
    assert fit['frozen_parent_states_exact'] and fit['fit_seen_exactly_once'] and fit['fresh_optimizer']
    assert load['readback_parameters'] == fit['readback_parameters'] == 96672
    assert load['readback_state_tensors'] == fit['readback_state_tensors'] == 23
    assert fit['terminal_sha256'] == restored['terminal_sha256'] == retained['terminal_sha256']
    assert restored['optimizer'] == {'all_keys_moments_steps_and_groups_exact':True,'moment_and_step_states':23,'param_groups':1}
    assert restored['strict_model_restore'] and restored['restored_steps'] == 3723
    assert retained['required_parent_chain_preserved'] and len(retained['deleted']) == 1
    assert retained['deleted'][0]['sha256'] == fit['terminal_sha256']
    assert retained['deleted'][0]['path'] == spec['root'] + '/terminal.pth'
    assert retained['retained_best'] == status['retained_best']
    assert not list(base.glob('*.pth'))
    assert fit['spec_sha256'] == load['spec_sha256'] == HASHES[(base/'spec.json').as_posix()]['sha256']
    for name,digest in spec['runner_files'].items():
        path=ROOT/'runtime_bundle'/name
        raw(path)
        assert HASHES[path.as_posix()]['sha256'] == digest
    assert raw(ROOT/'run_readback_fit.py') == raw(ROOT/'runtime_bundle/run_readback_fit.py')
    assert fit['script_sha256'] == HASHES[(ROOT/'run_readback_fit.py').as_posix()]['sha256']
    tr = rows(base/'train.jsonl')
    assert len(tr) == 3723 and [x['step'] for x in tr] == list(range(1,3724))
    assert all(x['total_steps'] == 3723 and len(x['rows']) == (2 if x['step'] == 3723 else 8) for x in tr)
    assert all(x['native_head_calls'] == 1 and x['geometry_and_mask_same_frame_exact'] for x in tr)
    assert all(x['call_order'] == ['refiner','readback','native_semantic_head'] for x in tr)
    assert all(math.isfinite(x[k]) for x in tr for k in ('loss','native_loss','assignment_correction','gradient_norm','seconds','cumulative_seconds'))
    assert all(x['seconds'] > 0 and x['gradient_norm'] >= 0 for x in tr)
    assert all(a['cumulative_seconds'] < b['cumulative_seconds'] for a,b in zip(tr,tr[1:]))
    orders[arm] = [i for x in tr for i in x['rows']]
    assert Counter(orders[arm]) == Counter(split['row_ids']['fit'])
    assert HASHES[(base/'train.jsonl').as_posix()]['sha256'] == fit['train_log_sha256']
    train_report[arm] = dict(steps=len(tr),rows=len(orders[arm]),unique_rows=len(set(orders[arm])),
        full_batches=3722,tail_batch=2,order_sha256=hashlib.sha256(json.dumps(orders[arm],separators=(',',':')).encode()).hexdigest(),
        first_rows=tr[0]['rows'],last_rows=tr[-1]['rows'],total_reassigned_queries=sum(x['assignment_counts']['reassigned_queries'] for x in tr),
        loss_sum_max_roundoff=max(abs(x['loss']-x['native_loss']-x['assignment_correction']) for x in tr),
        correction_formula_max_roundoff=max(abs(x['assignment_correction'] - (x['assignment_counts']['replaced_ce_new']-x['assignment_counts']['replaced_ce_old'])*(.5/7)) for x in tr))
    stages[arm] = {}
    for stage,n in [('initial',6887),('terminal',6887),('formal',9508)]:
        data = rows(base/stage/'rows.jsonl')
        receipt = js(base/stage/'receipt.json')
        assert n == len(data) and receipt['status'] == 'pass'
        assert receipt['rows_sha256'] == HASHES[(base/stage/'rows.jsonl').as_posix()]['sha256']
        assert [x['row_id'] for x in data] == (list(range(9508)) if stage == 'formal' else split['row_ids']['holdout'])
        stages[arm][stage] = row_check(data,receipt)
        data_by_key[arm+'/'+stage] = data
    loglines = raw(base/'train.log').decode().splitlines()
    progresses=[json.loads(s[len('READBACK_TRAIN_PROGRESS '):]) for s in loglines if s.startswith('READBACK_TRAIN_PROGRESS ')]
    assert all(x == tr[x['step']-1] for x in progresses)
    assert len(progresses) == 59
    completed=[json.loads(s[len('READBACK_FIT_COMPLETE '):]) for s in loglines if s.startswith('READBACK_FIT_COMPLETE ')]
    assert completed == [fit]
    formal_events=[json.loads(s[len('READBACK_EVAL_COMPLETE '):]) for s in raw(base/'formal.log').decode().splitlines() if s.startswith('READBACK_EVAL_COMPLETE ')]
    assert formal_events == [js(base/'formal/receipt.json')]
    for t in (.25,.5):
        diff = pair(data_by_key[arm+'/initial'],data_by_key[arm+'/terminal'],t)
        assert {key:diff[value] for key,value in [('fixes','repairs'),('damages','damages'),('net','net')]} == fit['transitions'][str(t)]

assert orders['evidence_hidden'] == orders['evidence_visible']
hidden_spec=js(COMPLETE/'evidence_hidden/spec.json')
visible_spec=js(COMPLETE/'evidence_visible/spec.json')
spec_differences=[k for k in hidden_spec if hidden_spec[k] != visible_spec[k]]
assert set(spec_differences) == {'root','preflight_root','use_geometry_evidence'}
parent_dir = TMP/'pvground_boundary_distribution_20261004/complete/distribution/formal'
parent = rows(parent_dir/'rows.jsonl')
parent_receipt = js(parent_dir/'receipt.json')
assert HASHES[(parent_dir/'rows.jsonl').as_posix()]['sha256'] == parent_receipt['rows_sha256']
parent_checks = row_check(parent,parent_receipt,False)
assert parent_checks['rec_hits25'] == 5616 and parent_checks['rec_hits50'] == 4506
comparisons = {stage:{str(t):pair(data_by_key['evidence_hidden/'+stage],data_by_key['evidence_visible/'+stage],t) for t in (.25,.5)} for stage in ('initial','terminal','formal')}
for stage in comparisons:
    a,b=data_by_key['evidence_hidden/'+stage],data_by_key['evidence_visible/'+stage]
    comparisons[stage]['full256_oracle_row_disagreements']={str(t):sum(x['bbs'][key][-1]!=y['bbs'][key][-1] for x,y in zip(a,b)) for t,key in [(.25,'oracle25'),(.5,'oracle50')]}
vs_parent = {arm:{str(t):pair(parent,data_by_key[arm+'/formal'],t) for t in (.25,.5)} for arm in stages}
volume_order = sorted(range(9508),key=lambda i:(math.prod(parent[i]['root_box'][3:]),i))
groups=[]
for q in range(4):
    ids=volume_order[q*2377:(q+1)*2377]
    a=[data_by_key['evidence_hidden/formal'][i] for i in ids]
    b=[data_by_key['evidence_visible/formal'][i] for i in ids]
    groups.append(dict(volume_quartile=q+1,rows=len(ids),hidden_hits50=sum(iou(x['bbs']['box'],x['root_box'])>.5 for x in a),
        visible_hits50=sum(iou(x['bbs']['box'],x['root_box'])>.5 for x in b),visible_vs_hidden=compact_pair(pair(a,b,.5))))

summary = js(OUT/'SUMMARY.json')
for arm in stages:
    for stage in stages[arm]:
        s=stages[arm][stage]
        expected=summary['stages'][arm][stage]
        for key in ('rows','rec_hits25','rec_hits50','mask_hits25','mask_hits50','mask_miou'):
            assert s[key] == expected[key],(arm,stage,key)
        for key in ('rec_acc25','rec_acc50'):
            assert abs(s[key]-expected[key]) < 1e-12
        assert s['full256_coverage_from_saved_flags'] == expected['candidate_availability']
        assert {t:compact_pair(p) for t,p in s['fixed_frame_readback_effect'].items()} == expected['fixed_frame_readback_effect']
for stage in comparisons:
    assert {k:(compact_pair(v) if k != 'full256_oracle_row_disagreements' else v) for k,v in comparisons[stage].items()} == summary['visible_vs_hidden'][stage]
assert {arm:{t:compact_pair(v) for t,v in d.items()} for arm,d in vs_parent.items()} == summary['versus_protected_parent']
assert groups == summary['formal_gt_volume_groups']
assert summary['parent_rows_sha256'] == HASHES[(parent_dir/'rows.jsonl').as_posix()]['sha256']
csv_rows=list(csv.DictReader(raw(OUT/'FORMAL_METRICS.csv').decode().splitlines()))
assert len(csv_rows)==3
for row,expected in zip(csv_rows,summary['table']):
    assert row['system']==expected['system']
    assert all(float(row[k])==v for k,v in expected.items() if k!='system')
raw(OUT/'RESULTS.md')

imported=js(COMPLETE/'evidence_hidden/imports.json')
local_imports={
    'evaluator':TMP/'pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py',
    'main_utils':TMP/'pvground_runtime_bundle_20260908_v1/PV-Ground/main_utils.py',
    'models.losses':TMP/'pvground_runtime_bundle_20260908_v1/PV-Ground/models/losses.py',
    'models.pv_ground':ROOT.parent/'revision2/source_preview/PV-Ground/models/pv_ground.py',
    'prepare_data':TMP/'pvground_runtime_bundle_20260908_v1/PV-Ground/prepare_data.py',
    'src.joint_det_dataset':TMP/'pvground_g_p2_20261002/complete/source/joint_det_dataset.py'}
for name,path in local_imports.items():
    raw(path)
    assert HASHES[path.as_posix()]['sha256'] == imported['sha256'][name], name
    identity_report[name]=dict(local_path=path.as_posix(),remote_path=imported['files'][name],sha256=imported['sha256'][name])
port_path=ROOT.parent/'revision2/complete_preflight/source_port.json'
port=js(port_path)
assert HASHES[port_path.as_posix()]['sha256'] == hidden_spec['source_port_sha256']
module_path=ROOT.parent/'revision2/source_preview/PV-Ground/models/modules.py'
raw(module_path)
assert HASHES[module_path.as_posix()]['sha256']==port['files']['models/modules.py']
initial_env=js(TMP/'pvground_runtime_bundle_20260908_v1/env_spec.json')
env_path=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_runtime_20260908_v1/witness_repair_v3/env_spec.json')
env=js(env_path)
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==hidden_spec['env_spec_sha256']
cfg_path=TMP/'pvground_runtime_bundle_20260908_v1/PV-Ground/wandb_config.yaml'
raw(cfg_path)
assert HASHES[cfg_path.as_posix()]['sha256']==port['files']['wandb_config.yaml']
source_check=js(ROOT/'FORMAL_DRAFT_SOURCE_CHECK.json')
review=js(ROOT/'READBACK_FORMAL_SOURCE_REVIEW.json')
assert raw(COMPLETE/'FORMAL_DRAFT_SOURCE_CHECK.json')==raw(ROOT/'FORMAL_DRAFT_SOURCE_CHECK.json')
assert raw(COMPLETE/'READBACK_FORMAL_SOURCE_REVIEW.json')==raw(ROOT/'READBACK_FORMAL_SOURCE_REVIEW.json')
raw(ROOT/'controller.py')
assert HASHES[(ROOT/'controller.py').as_posix()]['sha256']==next(x['sha256'] for x in review['reviewed_files'] if x['path'].endswith('/controller.py'))
for path in [ROOT/'analyze_closed_formal.py',ROOT/'EXPERIMENT_PLAN_READBACK_FIT.md',
             Path('C:/Users/gb/.codex_mcln_g0_20260905/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')]:
    raw(path)
ast_files=list((ROOT/'runtime_bundle').glob('*.py'))+[ROOT/'run_readback_fit.py',ROOT/'controller.py',ROOT/'analyze_closed_formal.py',module_path]+list(local_imports.values())
for path in ast_files:
    ast.parse(raw(path).decode('utf-8-sig'),filename=path.as_posix())

# Additional provenance reads: dataset helper identity, original G, actual V2 receipts.
appearance_path=TMP/'pvground_g_p2_20261002/complete/source/appearance_source_manifest.json'
appearance=js(appearance_path)
assert HASHES[appearance_path.as_posix()]['sha256']==manifest['source_manifest_sha256']
visual_path=Path('C:/Users/gb/.codex_mcln_g0_20260905/src/visual_data_handlers.py')
raw(visual_path)
assert HASHES[visual_path.as_posix()]['sha256']==appearance['files']['src/visual_data_handlers.py']
original_g_path=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scanrefer_finetune_20260918_semantic_assignment_v1/pvground_semantic_assignment.py')
assert raw(original_g_path)==raw(ROOT/'runtime_bundle/pvground_semantic_assignment.py')
preflight_root=ROOT.parent/'revision2/complete_preflight'
preflight_intake=js(preflight_root/'INTAKE.json')
for rel,identity in preflight_intake['files'].items():
    raw(preflight_root/rel)
    assert HASHES[(preflight_root/rel).as_posix()]==identity
preflight_report={}
for arm in ('evidence_hidden','evidence_visible'):
    pf=js(preflight_root/arm/'preflight.json')
    assert pf['status']=='pass' and pf['optimizer_steps']==2 and pf['batch_size']==8
    assert pf['geometry_provider_and_g_states_exact'] and pf['zero_residual_native_semantic_exact']
    assert pf['readback_parameters']==96672 and pf['readback_state_tensors']==23
    assert pf['weight_files_created']==0 and not pf['accuracy_result']
    assert all(x['native_bbs']['candidates']==256 and x['native_bbs']['score_exact'] and x['native_bbs']['rank_exact'] and x['native_bbs']['semantic_logit_gradient_exact'] for x in pf['steps'])
    preflight_report[arm]=dict(status=pf['status'],optimizer_steps=2,accuracy_result=False,
        disabled_repeat_exact=pf['disabled_repeat_differences']['exact'],
        zero_cross_forward_exact=pf['zero_cross_forward_differences']['exact'],
        terminal_independently_reexecuted=False,
        native_bbs_score_rank_gradient_exact_receipt=True)
raw(ROOT.parent/'revision2/run_readback_preflight.py')

def same_query_drift(first,second,second_field='bbs'):
    paired_rows=[(a,b) for a,b in zip(first,second) if a['bbs']['query']==b[second_field]['query']]
    threshold_disagreements={str(t):sum((iou(a['bbs']['box'],a['root_box'])>t)!=(iou(b[second_field]['box'],b['root_box'])>t) for a,b in paired_rows) for t in (.25,.5)}
    assert not any(threshold_disagreements.values())
    return dict(rows=len(paired_rows),box_coordinate_differing_rows=sum(a['bbs']['box']!=b[second_field]['box'] for a,b in paired_rows),
        max_abs_coordinate_difference=max(abs(x-y) for a,b in paired_rows for x,y in zip(a['bbs']['box'],b[second_field]['box'])),
        exact_iou_differing_rows=sum(a['bbs']['iou']!=b[second_field]['iou'] for a,b in paired_rows),
        same_query_threshold_disagreements=threshold_disagreements)
cross_forward={stage:same_query_drift(data_by_key['evidence_hidden/'+stage],data_by_key['evidence_visible/'+stage]) for stage in ('initial','terminal','formal')}
cross_forward['parent_vs_hidden_bypass']=same_query_drift(parent,data_by_key['evidence_hidden/formal'],'bypass_fixed_frame')
cross_forward['parent_vs_visible_bypass']=same_query_drift(parent,data_by_key['evidence_visible/formal'],'bypass_fixed_frame')
holdout_scenes={x['scan_id'].split('_')[0] for x in data_by_key['evidence_hidden/initial']}
formal_scenes={x['scan_id'].split('_')[0] for x in parent}
assert holdout_scenes.isdisjoint(formal_scenes)
assert all(int(hashlib.sha256((manifest['split_salt']+'\0'+scene).encode()).hexdigest()[:8],16)%5==0 for scene in holdout_scenes)
with (OUT/'AUDIT_PAIRED_CHANGES.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=['change','row_id','jsonl_line','scan_id','target_id','hidden_query','visible_query','hidden_cpu_iou','visible_cpu_iou'])
    writer.writeheader()
    for change,key in [('repair','repair_row_ids'),('damage','damage_row_ids')]:
        for rid in comparisons['formal']['0.5'][key]:
            a,b=data_by_key['evidence_hidden/formal'][rid],data_by_key['evidence_visible/formal'][rid]
            writer.writerow(dict(change=change,row_id=rid,jsonl_line=rid+1,scan_id=a['scan_id'],target_id=a['target_id'],
                hidden_query=a['bbs']['query'],visible_query=b['bbs']['query'],hidden_cpu_iou=iou(a['bbs']['box'],a['root_box']),visible_cpu_iou=iou(b['bbs']['box'],b['root_box'])))

output=dict(status='PASS_RECORDED_DATA_CPU_CHECKS',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Independent Python stdlib reconstruction from recorded boxes and recorded dataset GT rows; no raw dataset, tensors, weights, inference, SSH, or author analyzer execution.',
    python=sys.version,executable=sys.executable,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    intact_intake_files=len(intake['files']),intake_bytes=sum(x['bytes'] for x in intake['files'].values()),
    rows_recomputed=sum(x['rows'] for d in stages.values() for x in d.values())+parent_checks['rows'],
    training_logs=train_report,same_training_order=True,spec_differences=spec_differences,
    stages=stages,parent=parent_checks,visible_vs_hidden=comparisons,versus_parent=vs_parent,formal_gt_volume_groups=groups,
    local_import_identity_matches=identity_report,syntax_files=len(ast_files),summary_csv_match=True,
    preflight_receipt_checks=preflight_report,preflight_intake_files=len(preflight_intake['files']),
    original_G_helper_byte_identical=True,dataset_visual_helper_manifest_hash_exact=True,
    recorded_same_query_cross_forward_drift=cross_forward,
    holdout_physical_scene_salt_checked=106,holdout_formal_physical_scene_overlap=0,
    environment_identity=dict(exact_archive=env_path.as_posix(),expected_canonical_sha256=hidden_spec['env_spec_sha256'],
        initial_archived_bundle_canonical_sha256=hashlib.sha256(json.dumps(initial_env,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        initial_archive_is_execution_environment=False,exact_execution_environment_archive_found=True),
    limitations=['Raw ScanNet/ScanRefer original data not available at supplied remote paths; GT rows verified as source-derived and consistent across runs, not re-extracted from original scenes.',
      'Mask IoU and full256 oracle flags are aggregated and checked for consistency only; raw Mask tensors/full256 boxes/logits not in terminal export.',
      'Model, optimizer and deletion exactness checked through source assertions plus original runtime receipts; weights deliberately absent and no independent reload was performed.'])
(OUT/'AUDIT_CPU_CHECK.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
(OUT/'AUDITED_INPUT_SHA256.json').write_text(json.dumps(HASHES,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=output['status'],intake_files=output['intact_intake_files'],rows=output['rows_recomputed'],formal={a: {k:stages[a]['formal'][k] for k in ('rec_hits25','rec_hits50','rec_acc25','rec_acc50')} for a in stages},formal_pair=compact_pair(comparisons['formal']['0.5']),volume_groups=groups),indent=2))
