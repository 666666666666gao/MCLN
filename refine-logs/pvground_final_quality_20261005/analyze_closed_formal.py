"""Analyze actual closed rows against the completed control, without NN replay."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


local = Path(__file__).resolve().parent
complete = local/'complete'
control = local.parent/'pvground_geometry_readback_20261004/formal_draft/complete'
parent = local.parent/'pvground_boundary_distribution_20261004/complete/distribution/formal'
intakes = {root:json.loads((root/'INTAKE.json').read_bytes()) for root in (complete,control)}
terminal = intakes[complete]['remote_terminal']
assert terminal['status']['status']=='complete' and terminal['exitcode']==0
assert not terminal['controller_alive']
assert all(record['downloaded_weights']==0 and not record['inference_or_optimizer_replayed'] for record in intakes.values())
analysis = local/'analysis'
assert not analysis.exists()


def read(root, relative, rows=False):
    raw = (root/relative).read_bytes()
    identity = intakes[root]['files'][relative]
    assert len(raw)==identity['bytes'] and hashlib.sha256(raw).hexdigest()==identity['sha256']
    return [json.loads(line) for line in raw.decode('utf-8').splitlines()] if rows else json.loads(raw)


def cpu_iou(box, truth):
    assert len(box)==len(truth)==6 and all(math.isfinite(v) for v in box+truth)
    assert all(v>0 for v in box[3:]+truth[3:])
    intersection = math.prod(max(0.,min(box[a]+box[a+3]/2,truth[a]+truth[a+3]/2)
        - max(box[a]-box[a+3]/2,truth[a]-truth[a+3]/2)) for a in range(3))
    return intersection/(math.prod(box[3:])+math.prod(truth[3:])-intersection)


def paired(before, after, threshold):
    assert len(before)==len(after)
    repairs = damages = changes = 0
    for a,b in zip(before,after):
        assert all(a[key]==b[key] for key in ('row_id','scan_id','target_id','root_box','point_sha256'))
        old,new = a['bbs']['iou']>threshold,b['bbs']['iou']>threshold
        repairs += not old and new
        damages += old and not new
        changes += a['bbs']['query']!=b['bbs']['query']
    return dict(repairs=repairs,damages=damages,net=repairs-damages,selected_query_changes=changes)


def summarize(rows, receipt):
    assert len(rows)==receipt['rows']
    metric = receipt['metrics']['bbs']
    threshold_changes = 0
    for row in rows:
        assert row['same_forward_geometry_exact'] and row['native_head_calls']==1
        assert row['diagnostic_native_head_replay_calls']==1
        for scope,box_key,iou_key in (('bbs','box','iou'),('bbs','coarse_box','coarse_iou'),('bypass_fixed_frame','box','iou')):
            prediction = row[scope]
            u = cpu_iou(prediction[box_key],row['root_box'])
            threshold_changes += sum((u>t)!=(prediction[iou_key]>t) for t in (.25,.5))
        assert math.isfinite(row['bbs']['mask_iou']) and 0<=row['bbs']['mask_iou']<=1
    assert threshold_changes==0
    hits = [sum(row['bbs']['iou']>t for row in rows) for t in (.25,.5)]
    assert hits==[metric['rec_hits25'],metric['rec_hits50']]
    mask_hits = [sum(row['bbs']['mask_iou']>t for row in rows) for t in (.25,.5)]
    assert mask_hits==[metric['mask_hits25'],metric['mask_hits50']]
    assert abs(statistics.mean(row['bbs']['mask_iou'] for row in rows)*100-metric['mask_miou'])<1e-6
    direct,coverage = {},{}
    for t,key in ((.25,'oracle25'),(.5,'oracle50')):
        bypass = [dict(row,bbs=dict(row['bbs'],**row['bypass_fixed_frame'])) for row in rows]
        direct[str(t)] = paired(bypass,rows,t)
        expected = receipt['fixed_frame_readback_effect'][str(t)]
        assert [direct[str(t)][k] for k in ('repairs','damages','net')]==[expected[k] for k in ('fixes','damages','net')]
        coverage[str(t)] = dict(topk=[16,32,64,256],
            oracle_hits=[sum(row['bbs'][key][k] for row in rows) for k in range(4)],
            errors_with_good_full256=sum(row['bbs']['iou']<=t and row['bbs'][key][-1] for row in rows),
            errors_without_good_full256=sum(row['bbs']['iou']<=t and not row['bbs'][key][-1] for row in rows))
    return dict(rows=len(rows),rec_hits25=hits[0],rec_hits50=hits[1],rec_acc25=100*hits[0]/len(rows),
        rec_acc50=100*hits[1]/len(rows),mask_hits25=mask_hits[0],mask_hits50=mask_hits[1],
        mask_miou=metric['mask_miou'],cpu_box_threshold_changes=threshold_changes,
        fixed_frame_readback_effect=direct,candidate_availability=coverage)


stages,rows_by_arm,training_orders,receipts = {},{},{},{}
for arm,root,prefix in (('native_g_control',control,'evidence_visible'),('final_quality',complete,'quality')):
    fit = read(root,prefix+'/receipt.json')
    restore = read(root,prefix+'/formal_restore.json')
    retention = read(root,prefix+'/weight_retention.json')
    spec = read(root,prefix+'/spec.json')
    assert fit['status']=='complete' and fit['training_steps']==3723 and fit['fit_rows']==29778
    assert fit['frozen_parent_states_exact'] and fit['fit_seen_exactly_once']
    assert restore['strict_model_restore'] and restore['restored_steps']==3723
    assert retention['required_parent_chain_preserved'] and retention['cpu_box_threshold_changes']==0
    assert not retention['local_weight_archive_created']
    assert spec['batch_size']==8 and spec['updates']==3723 and spec['fit_passes']==1 and spec['use_geometry_evidence']
    train = read(root,prefix+'/train.jsonl',rows=True)
    assert [row['step'] for row in train]==list(range(1,3724))
    assert all(len(row['rows'])==(2 if row['step']==3723 else 8) for row in train)
    order = [sample for row in train for sample in row['rows']]
    assert len(order)==len(set(order))==29778
    training_orders[arm]=order
    stages[arm],rows_by_arm[arm]={},{}
    for stage,count in (('initial',6887),('terminal',6887),('formal',9508)):
        rows = read(root,prefix+'/'+stage+'/rows.jsonl',rows=True)
        receipt = read(root,prefix+'/'+stage+'/receipt.json')
        assert receipt['status']=='pass' and len(rows)==count
        stages[arm][stage]=summarize(rows,receipt)
        rows_by_arm[arm][stage]=rows
    assert [row['row_id'] for row in rows_by_arm[arm]['formal']]==list(range(9508))
    receipts[arm]=dict(fit=fit,restore=restore,retention=retention)
    if arm=='final_quality':
        assert spec['quality_weight']==1.0
        assert all(row['quality_weight']==1.0 and math.isfinite(row['quality_loss']) for row in train)
        quality_curve=dict(first100_mean=statistics.mean(row['quality_loss'] for row in train[:100]),
            last100_mean=statistics.mean(row['quality_loss'] for row in train[-100:]),
            min=min(row['quality_loss'] for row in train),max=max(row['quality_loss'] for row in train),
            scope='Training loss only; not accuracy or proof of method efficacy.')
assert training_orders['native_g_control']==training_orders['final_quality']
comparisons = {stage:{str(t):paired(rows_by_arm['native_g_control'][stage],rows_by_arm['final_quality'][stage],t)
    for t in (.25,.5)} for stage in ('initial','terminal','formal')}
parent_raw = (parent/'rows.jsonl').read_bytes()
parent_rows = [json.loads(line) for line in parent_raw.decode('utf-8').splitlines()]
parent_receipt = json.loads((parent/'receipt.json').read_bytes())
assert hashlib.sha256(parent_raw).hexdigest()==parent_receipt['rows_sha256']
assert len(parent_rows)==9508
parent_hits = [sum(cpu_iou(row['bbs']['box'],row['root_box'])>t for row in parent_rows) for t in (.25,.5)]
assert parent_hits==[5616,4506]
versus_parent = {arm:{str(t):paired(parent_rows,rows_by_arm[arm]['formal'],t) for t in (.25,.5)} for arm in stages}
volume_order = sorted(range(9508),key=lambda i:(math.prod(parent_rows[i]['root_box'][3:]),i))
groups=[]
for quartile in range(4):
    indices=volume_order[quartile*2377:(quartile+1)*2377]
    before=[rows_by_arm['native_g_control']['formal'][i] for i in indices]
    after=[rows_by_arm['final_quality']['formal'][i] for i in indices]
    groups.append(dict(volume_quartile=quartile+1,rows=len(indices),
        control_hits50=sum(row['bbs']['iou']>.5 for row in before),quality_hits50=sum(row['bbs']['iou']>.5 for row in after),
        quality_vs_control=paired(before,after,.5)))
table=[dict(system='protected_geometry_parent',rec_hits25=5616,rec_hits50=4506,rec_acc25=100*5616/9508,rec_acc50=100*4506/9508)]
for arm in stages:
    table.append(dict(system=arm,**{key:stages[arm]['formal'][key] for key in ('rec_hits25','rec_hits50','rec_acc25','rec_acc50')}))
summary=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),primary='native last/bbs Acc@0.50',
    table=table,stages=stages,quality_vs_control=comparisons,versus_protected_parent=versus_parent,
    formal_gt_volume_groups=groups,training_order_exact=True,updates_per_arm=3723,effective_batch=8,
    samples_per_arm=29778,fit_seen_once=True,quality_curve=quality_curve,
    actual_finished_cst=terminal['status']['finished_cst'],retained_best=terminal['status']['retained_best'],
    target_pass=terminal['status']['scanrefer_target_pass'],parent_rows_sha256=hashlib.sha256(parent_raw).hexdigest(),
    integrity_review_pending=True,downloaded_weights=0,inference_or_optimizer_replayed=False,
    scope='One new quality arm; reused completed native+G control. Single seed2027. 6887 pretrained-seen holdout separately from native9508. Fixed-frame head replay is a forward diagnostic, not independently trained ablation. Parent states exact/frozen; independent complete-forward tensors not claimed bitwise identical. GT coverage/volume groups offline only. No Nr/Sr results or novelty proof.')
analysis.mkdir()
(analysis/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
with (analysis/'FORMAL_METRICS.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
lines=['# Final native-quality supervision: actual closed result','',
    'Actual finished CST: '+summary['actual_finished_cst']+'. Primary: native last/bbs Acc@0.50.','',
    '| System | Hits@.25 | Hits@.50 | Acc@.25% | Acc@.50% |','|---|---:|---:|---:|---:|']
for row in table:
    lines.append('| {system} | {rec_hits25} | {rec_hits50} | {rec_acc25:.4f} | {rec_acc50:.4f} |'.format(**row))
for label,effect in (('Quality vs completed control',comparisons['formal']['0.5']),
    ('Quality vs protected parent',versus_parent['final_quality']['0.5']),
    ('Quality same-frame R forward',stages['final_quality']['formal']['fixed_frame_readback_effect']['0.5'])):
    lines+=['','{}: repairs {}, damages {}, net {:+d}.'.format(label,effect['repairs'],effect['damages'],effect['net'])]
lines+=['','All256 / one native score / same Query Box+Mask. Actual row order matched; B8/29778 once/3723 updates.',
    'Only the fixed-weight quality term differs from the completed control; parents remain frozen/eval.',
    'Independent terminal integrity review pending. This analysis is not reviewer PASS.',summary['scope'],'']
(analysis/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(status='ACTUAL_CLOSED_ROWS_ANALYZED',table=table,target_pass=summary['target_pass'],integrity_review_pending=True)),flush=True)
