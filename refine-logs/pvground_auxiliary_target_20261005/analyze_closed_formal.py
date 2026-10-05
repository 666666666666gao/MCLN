"""Analyze the actual closed same-start pair, without model/weight replay."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from geometry_result_metrics import paired,summarize

local=Path(__file__).resolve().parent
complete=local/'complete'
intake=json.loads((complete/'INTAKE.json').read_bytes())
terminal=intake['remote_terminal']
assert not terminal['controller_alive'] and terminal['exitcode']==0 and terminal['status']['status']=='complete'
assert terminal['status']['protected_parents_exact']
assert not intake['downloaded_weights'] and not intake['inference_or_optimizer_replayed']
analysis=local/'analysis'
assert not analysis.exists()

def read(relative,rows=False):
    raw=(complete/relative).read_bytes()
    identity=intake['files'][relative]
    assert len(raw)==identity['bytes'] and hashlib.sha256(raw).hexdigest()==identity['sha256']
    return [json.loads(line) for line in raw.decode('utf-8').splitlines()] if rows else json.loads(raw)

stages,rows_by_arm,orders,training={},{},{},{}
for arm in ('control','member_target'):
    fit=read(arm+'/receipt.json')
    restore=read(arm+'/formal_restore.json')
    spec=read(arm+'_spec.json')
    assert fit['status']=='complete' and fit['training_steps']==3723 and fit['fit_rows']==29778
    assert fit['parent_and_zero_R_states_exact'] and fit['fit_seen_exactly_once'] and fit['fresh_optimizer']
    assert fit['head_parameters']==456102 and fit['head_state_tensors']==10
    assert fit['physical_batch']==fit['effective_batch']==8 and fit['accumulation']==1 and fit['last_batch_rows']==2
    assert restore['strict_model_restore'] and restore['optimizer']['all_keys_moments_steps_and_groups_exact']
    assert restore['terminal_sha256']==fit['terminal_sha256']
    assert spec['extra_geometry_weight']==1.0
    assert spec['auxiliary_target_mode']==('native_gt' if arm=='control' else 'member_gt')
    assert fit['extra_geometry_weight']==1.0
    logs=read(arm+'/train.jsonl',rows=True)
    assert [entry['step'] for entry in logs]==list(range(1,3724))
    assert all(len(entry['rows'])==(2 if entry['step']==3723 else 8) for entry in logs)
    assert all(entry['extra_geometry_weight']==spec['extra_geometry_weight'] and entry['auxiliary_target_mode']==spec['auxiliary_target_mode'] for entry in logs)
    assert all(all(math.isfinite(entry[key]) for key in ('loss','native_loss','G_correction','matched_boundary_loss','extra_geometry_loss','gradient_norm')) for entry in logs)
    assert all(sum(entry['extra_counts']['extra_row_counts'])==entry['extra_counts']['extra_candidates'] for entry in logs)
    orders[arm]=[row for entry in logs for row in entry['rows']]
    assert len(orders[arm])==len(set(orders[arm]))==29778
    training[arm]=dict(total_extra_candidate_roles=sum(entry['extra_counts']['extra_candidates'] for entry in logs),
        empty_expression_rows=sum(count==0 for entry in logs for count in entry['extra_counts']['extra_row_counts']),
        total_extra_boundary_outside=sum(entry['extra_counts']['extra_boundary_outside'] for entry in logs),
        extra_geometry_loss_first100_mean=statistics.mean(entry['extra_geometry_loss'] for entry in logs[:100]),
        extra_geometry_loss_last100_mean=statistics.mean(entry['extra_geometry_loss'] for entry in logs[-100:]),
        fit_receipt=fit,restore_receipt=restore)
    stages[arm],rows_by_arm[arm]={},{}
    for stage,count in (('initial',6887),('terminal',6887),('formal',9508)):
        rows=read(arm+'/'+stage+'/rows.jsonl',rows=True)
        receipt=read(arm+'/'+stage+'/receipt.json')
        assert receipt['status']=='pass' and receipt['rows']==len(rows)==count
        assert all(row['same_forward_geometry_exact'] and row['native_head_calls']==1
            and row['diagnostic_native_head_replay_calls']==0 for row in rows)
        stages[arm][stage]=summarize(rows,receipt['metrics']['bbs'])
        rows_by_arm[arm][stage]=rows
    assert [row['row_id'] for row in rows_by_arm[arm]['formal']]==list(range(9508))
    for threshold in (.25,.5):
        effect=paired(rows_by_arm[arm]['initial'],rows_by_arm[arm]['terminal'],threshold)
        assert all(effect[key]==fit['transitions'][str(threshold)][key] for key in ('repairs','damages','net'))
assert orders['control']==orders['member_target']
comparisons={stage:{str(t):paired(rows_by_arm['control'][stage],rows_by_arm['member_target'][stage],t)
    for t in (.25,.5)} for stage in ('initial','terminal','formal')}
parent=local.parent/'pvground_query_supported_geometry_20261005/complete/query_supported/formal'
parent_raw=(parent/'rows.jsonl').read_bytes()
parent_receipt=json.loads((parent/'receipt.json').read_bytes())
assert hashlib.sha256(parent_raw).hexdigest()==parent_receipt['rows_sha256']
parent_rows=[json.loads(line) for line in parent_raw.decode('utf-8').splitlines()]
parent_metric=summarize(parent_rows,parent_receipt['metrics']['bbs'])
assert [parent_metric['rec_hits25'],parent_metric['rec_hits50']]==[5614,4509]
versus_parent={arm:{str(t):paired(parent_rows,rows_by_arm[arm]['formal'],t) for t in (.25,.5)} for arm in stages}
volume_order=sorted(range(9508),key=lambda index:(math.prod(parent_rows[index]['root_box'][3:]),index))
groups=[]
for quartile in range(4):
    indices=volume_order[quartile*2377:(quartile+1)*2377]
    before=[rows_by_arm['control']['formal'][index] for index in indices]
    after=[rows_by_arm['member_target']['formal'][index] for index in indices]
    groups.append(dict(volume_quartile=quartile+1,rows=len(indices),
        control_hits50=sum(row['bbs']['iou']>.5 for row in before),member_target_hits50=sum(row['bbs']['iou']>.5 for row in after),
        strategy_vs_control=paired(before,after,.5)))
keys=('rec_hits25','rec_hits50','rec_acc25','rec_acc50')
table=[dict(system='protected_geometry_parent',**{key:parent_metric[key] for key in keys})]
table.extend(dict(system=arm,**{key:stages[arm]['formal'][key] for key in keys}) for arm in stages)
best=max(table,key=lambda row:(row['rec_hits50'],row['rec_hits25'],row['system']=='protected_geometry_parent'))
target_pass_by_system={row['system']:row['rec_hits25']>=5615 and row['rec_hits50']>=4754 for row in table}
summary=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_CLOSED_ROWS_ANALYZED',
    actual_finished_cst=terminal['status']['finished_cst'],table=table,stages=stages,
    strategy_vs_control=comparisons,versus_protected_parent=versus_parent,formal_gt_volume_groups=groups,
    training=training,training_order_exact=True,updates_per_arm=3723,samples_per_arm=29778,
    physical_batch=8,effective_batch=8,accumulation=1,tail_batch_rows=2,
    geometry_parent_updates=7446,geometry_total_updates_at_terminal=11169,
    metric_best_candidate=best,scanrefer_target_pass=any(target_pass_by_system.values()),
    target_pass_by_system=target_pass_by_system,metric_best_target_pass=target_pass_by_system[best['system']],
    parent_rows_sha256=hashlib.sha256(parent_raw).hexdigest(),integrity_review_pending=True,
    weight_cleanup_pending=True,downloaded_weights=0,inference_or_optimizer_replayed=False,
    scope='Single seed2027; two actual same-start full fits, parents and zeroR frozen; existing boundary head only. Both extra geometry weights1; native/member coordinates change auxiliary qualification and localization targets together, while native GT matching/losses/evaluation remain unchanged. Native9508 is development validation,6887 is pretrained-seen module holdout. CPU independently checks selected Box/GT thresholds; Masks and full candidate coverage are scalar recounts, not raw-array replay. GT qualification and volume grouping are offline only. Candidate support/overlap is not physical identity proof. Cached-upstream preflight exactness does not prove cross-CUDA full-forward bitwise equality. No Nr3D/Sr3D new result or complete three-module novelty proof.')
analysis.mkdir()
(analysis/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
with (analysis/'FORMAL_METRICS.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=['system',*keys]);writer.writeheader();writer.writerows(table)
lines=['# Native/member auxiliary geometry targets: closed result','',
    'Finished CST: '+summary['actual_finished_cst']+'. Primary native last/bbs Acc@0.50.','',
    '| System | Hits@.25 | Hits@.50 | Acc@.25% | Acc@.50% |','|---|---:|---:|---:|---:|']
for row in table:
    lines.append('| {system} | {rec_hits25} | {rec_hits50} | {rec_acc25:.4f} | {rec_acc50:.4f} |'.format(**row))
for label,effect in (('Strategy vs same-budget control',comparisons['formal']['0.5']),
    ('Strategy vs protected parent',versus_parent['member_target']['0.5']),
    ('Strategy same-Query coarse/final refinement',stages['member_target']['formal']['same_query_refinement']['0.5'])):
    lines.extend(['','{}: repairs {}, damages {}, net {:+d}.'.format(label,effect['repairs'],effect['damages'],effect['net'])])
lines.extend(['','Same29778 rows/order once,each3723 updates/B8/tailB2. Only geometry head updated.',
    'Fresh terminal integrity review and weight retention pending; this analyzer never loads/deletes/archives weights.',summary['scope'],''])
(analysis/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(status=summary['status'],table=table,target_pass=summary['scanrefer_target_pass'],integrity_review_pending=True)),flush=True)
