"""Read only actually closed rows; separate reference and final geometry."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

root=Path(__file__).resolve().parents[1]
complete=root/'complete'
output=root/'analysis'
assert not output.exists()
status=json.loads((complete/'fit_status.json').read_bytes())
assert status['status']=='complete' and status['protected_parents_exact']
assert (complete/'fit_controller.exit').read_text().strip()=='0'
intake=json.loads((complete/'INTAKE.json').read_bytes())
for name,entry in intake['files'].items():
    raw=(complete/name).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def iou(box,truth):
    intersection=1.0
    predicted=target=1.0
    for axis in range(3):
        low=max(box[axis]-box[axis+3]/2,truth[axis]-truth[axis+3]/2)
        high=min(box[axis]+box[axis+3]/2,truth[axis]+truth[axis+3]/2)
        intersection*=max(0.0,high-low)
        predicted*=box[axis+3]
        target*=truth[axis+3]
    return intersection/(predicted+target-intersection)


def pair(old,new):
    assert len(old)==len(new)==9508
    for before,after in zip(old,new):
        assert all(before[key]==after[key] for key in ('row_id','scan_id','target_id','root_box','point_sha256'))


def difference(old,new):
    pair(old,new)
    value={'selected_query_changes':sum(a['bbs']['query']!=b['bbs']['query'] for a,b in zip(old,new))}
    for threshold in (.25,.5):
        repair=sum(a['bbs']['iou']<=threshold<b['bbs']['iou'] for a,b in zip(old,new))
        damage=sum(b['bbs']['iou']<=threshold<a['bbs']['iou'] for a,b in zip(old,new))
        value[str(threshold)]={'repairs':repair,'damages':damage,'net':repair-damage}
    return value


def transition(data,before,after,threshold):
    repair=sum(row['bbs'][before]<=threshold<row['bbs'][after] for row in data)
    damage=sum(row['bbs'][after]<=threshold<row['bbs'][before] for row in data)
    return {'before_hits':sum(row['bbs'][before]>threshold for row in data),
        'after_hits':sum(row['bbs'][after]>threshold for row in data),
        'repairs':repair,'damages':damage,'net':repair-damage,'scope':'same_selected_Query_internal'}


parent=rows(root.parent/'pvground_auxiliary_target_20261005/complete/control/formal/rows.jsonl')
assert len(parent)==9508 and sum(row['bbs']['iou']>.5 for row in parent)==4511
table=[{'system':'protected_geometry_parent','rec_hits25':5616,'rec_hits50':4511}]
systems={}
all_rows={'protected_geometry_parent':parent}
for arm in ('control','support_reference'):
    data=rows(complete/arm/'formal/rows.jsonl')
    assert [row['row_id'] for row in data]==list(range(9508))
    pair(parent,data)
    formal=json.loads((complete/arm/'formal/receipt.json').read_bytes())
    fit=json.loads((complete/arm/'receipt.json').read_bytes())
    training=rows(complete/arm/'train.jsonl')
    assert formal['rows']==9508 and formal['status']=='pass'
    assert fit['training_steps']==3723 and fit['fit_rows']==29778 and fit['fit_seen_exactly_once']
    assert len(training)==3723 and [line['step'] for line in training]==list(range(1,3724))
    fit_ids=[identity for line in training for identity in line['rows']]
    assert len(fit_ids)==len(set(fit_ids))==29778 and len(training[-1]['rows'])==2
    hits={threshold:sum(row['bbs']['iou']>threshold for row in data) for threshold in (.25,.5)}
    assert [hits[.25],hits[.5]]==[formal['metrics']['bbs']['rec_hits25'],formal['metrics']['bbs']['rec_hits50']]
    flips={str(threshold):0 for threshold in (.25,.5)}
    for row in data:
        for box_name,stored_name in (('box','iou'),('coarse_box','coarse_iou'),('reference_box','reference_iou')):
            independent=iou(row['bbs'][box_name],row['root_box'])
            for threshold in (.25,.5):
                flips[str(threshold)]+=int((independent>threshold)!=(row['bbs'][stored_name]>threshold))
    entry={'system':arm,'rec_hits25':hits[.25],'rec_hits50':hits[.5]}
    table.append(entry)
    systems[arm]={'formal':entry,'parent_delta':difference(parent,data),
        'cpu_selected_coarse_reference_box_threshold_flip_counts':flips,
        'internal_reference':{str(t):transition(data,'coarse_iou','reference_iou',t) for t in (.25,.5)},
        'internal_final':{str(t):transition(data,'reference_iou','iou',t) for t in (.25,.5)},
        'mask_hits50':sum(row['bbs']['mask_iou']>.5 for row in data),
        'selected_mask_good_box_bad':sum(row['bbs']['mask_iou']>.5>=row['bbs']['iou'] for row in data),
        'full256_strict_scalar_oracle':sum(row['bbs']['oracle50'][-1] for row in data),
        'extra_roles':sum(line['extra_counts']['extra_candidates'] for line in training),
        'extra_outside_faces':sum(line['extra_counts']['extra_boundary_outside'] for line in training),
        'extra_loss_first100':statistics.mean(line['extra_geometry_loss'] for line in training[:100]),
        'extra_loss_last100':statistics.mean(line['extra_geometry_loss'] for line in training[-100:]),
        'reference_loss_first100':statistics.mean(line['reference_localization_loss'] for line in training[:100]),
        'reference_loss_last100':statistics.mean(line['reference_localization_loss'] for line in training[-100:]),
        'state_tensors':fit['head_state_tensors'],'parameters':fit['head_parameters']}
    all_rows[arm]=data
assert [line['rows'] for line in rows(complete/'control/train.jsonl')]==[
    line['rows'] for line in rows(complete/'support_reference/train.jsonl')]
winner=max(table,key=lambda item:(item['rec_hits50'],item['system']=='protected_geometry_parent',item['rec_hits25']))
summary={'status':'ACTUAL_CLOSED_REFERENCE_ROWS_ANALYZED','table':table,'systems':systems,
    'support_vs_control':difference(all_rows['control'],all_rows['support_reference']),
    'metric_best_candidate':winner,'strict_target_gap':max(0,4754-winner['rec_hits50']),
    'accuracy_scope':'9508_ScanRefer_development_native_last_bbs',
    'mask_and_full256_scope':'stored_scalar_recount_only','single_seed':2027,
    'fit_order_exact':True,'geometry_parent_fit_updates':11169,'total_geometry_fit_updates':14892,
    'formal_complete':True,'fresh_terminal_audit_pending':True,'weight_retention_executed':False,
    'scanrefer_target_pass':any(item['rec_hits25']>=5544 and item['rec_hits50']>=4754 for item in table),
    'metric_best_target_pass':winner['rec_hits25']>=5544 and winner['rec_hits50']>=4754}
output.mkdir()
(output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'table':table,'best':winner,'audit_pending':True,'weights_changed':False}))
