"""CPU recount of saved selected boxes; no neural model or answer changes."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
intake_path = root/'OBS_native_control_ct3_INTAKE.json'
intake = json.loads(intake_path.read_bytes())
assert not intake['controller_alive'] and not intake['child_alive']
assert intake['remote_status']['status']=='complete' and intake['remote_status']['exit_code']==0
captured = root/'OBS_native_control_ct3_files'
directory = root/'control_cpu_recount'
assert not directory.exists()
directory.mkdir()
source_hashes = {str(intake_path):hashlib.sha256(intake_path.read_bytes()).hexdigest()}
for entry in intake['files']:
    path = captured/entry['name']
    raw = path.read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    source_hashes[str(path)] = entry['sha256']
metric_path, = captured.rglob('native_metrics.jsonl')
metrics = [json.loads(s) for s in metric_path.read_text().splitlines()]
assert [(m['epoch'],m['hits025'],m['hits050']) for m in metrics]==[(0,5677,4920),(1,5655,4620),(2,5616,4596)]

def iou(boxes, targets):
    boxes = np.asarray(boxes,dtype=np.float64)
    targets = np.asarray(targets,dtype=np.float64)
    assert boxes.shape==targets.shape==(9508,6)
    assert np.isfinite(boxes).all() and np.isfinite(targets).all()
    assert (boxes[:,3:]>0).all() and (targets[:,3:]>0).all()
    lower = np.maximum(boxes[:,:3]-boxes[:,3:]/2,targets[:,:3]-targets[:,3:]/2)
    upper = np.minimum(boxes[:,:3]+boxes[:,3:]/2,targets[:,:3]+targets[:,3:]/2)
    intersection = np.maximum(upper-lower,0).prod(axis=1)
    union = boxes[:,3:].prod(axis=1)+targets[:,3:].prod(axis=1)-intersection
    assert (union>0).all()
    return intersection/union

def change(previous,current):
    return dict(repairs=int((~previous&current).sum()),breaks=int((previous&~current).sum()),
        net_hits=int(current.sum()-previous.sum()))

reports = []
initial_rows = None
initial_iou = None
for metric in metrics:
    epoch = metric['epoch']
    row_path, = captured.rglob('native_rows_epoch%03d.jsonl'%epoch)
    diagnostic_path, = captured.rglob('native_diagnostic_epoch%03d.json'%epoch)
    diagnostic = json.loads(diagnostic_path.read_bytes())
    assert hashlib.sha256(row_path.read_bytes()).hexdigest()==diagnostic['row_file_sha256']
    assert row_path.stat().st_size==diagnostic['row_file_bytes']
    rows = [json.loads(s) for s in row_path.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==metric['rows']==9508
    assert [r['row_index'] for r in rows]==list(range(9508))
    targets = [r['root_gt_box'] for r in rows]
    values = {label:iou([r[key] for r in rows],targets) for label,key in
        (('final','selected_box'),('native','native_box'),('mask_endpoint','mask_endpoint_box'))}
    logged = {label:np.asarray([r[label+'_iou'] for r in rows]) for label in values}
    support = np.asarray([2*r['own_mask_intersection']>r['own_mask_union'] and
        2*r['fused_mask_intersection']>r['fused_mask_union'] for r in rows],dtype=bool)
    assert all(0<=r['selected_query']<256 for r in rows)
    for r in rows:
        assert 0<=r['own_mask_intersection']<=r['own_mask_union']
        assert 0<=r['fused_mask_intersection']<=r['fused_mask_union']
    report = dict(epoch=epoch,rows=9508,selected_own_and_fused_support_qualified=int(support.sum()),
        max_cpu64_vs_logged_iou_difference={label:float(np.abs(values[label]-logged[label]).max()) for label in values},
        logged_iou_above_one={label:int((logged[label]>1).sum()) for label in values})
    assert report['selected_own_and_fused_support_qualified']==diagnostic['selected_own_and_fused_support_qualified']
    if initial_rows is None:
        initial_rows,initial_iou = rows,values['final']
    identical = []
    for a,b in zip(initial_rows,rows):
        assert all(a[key]==b[key] for key in ('row_index','scan_id','target_id','utterance_sha256','root_gt_box'))
        identical.append(a['point_cloud_sha256']==b['point_cloud_sha256'] and a['gt_mask_sha256']==b['gt_mask_sha256'])
    report['exact_same_input_fingerprint_rows_as_E0'] = sum(identical)
    report['selected_slot_changes_vs_E0_not_instance_identity'] = sum(a['selected_query']!=b['selected_query'] for a,b in zip(initial_rows,rows))
    for suffix,threshold in (('025',.25),('050',.5)):
        hit = values['final']>threshold
        decisions = {label:int(((values[label]>threshold)!=(logged[label]>threshold)).sum()) for label in values}
        assert all(n==0 for n in decisions.values())
        assert int(hit.sum())==metric['hits'+suffix]==diagnostic[suffix]['hits']
        native_change = change(values['native']>threshold,hit)
        mask_change = change(values['mask_endpoint']>threshold,hit)
        assert native_change['repairs']==diagnostic[suffix]['final_vs_native_repairs']
        assert native_change['breaks']==diagnostic[suffix]['final_vs_native_breaks']
        assert mask_change['repairs']==diagnostic[suffix]['final_vs_mask_endpoint_repairs']
        assert mask_change['breaks']==diagnostic[suffix]['final_vs_mask_endpoint_breaks']
        # These totals use logged coverage scalars, not all256 candidate boxes.
        coverage = sum(r['best256_iou']>threshold for r in rows)
        errors_with_candidate = sum(r['final_iou']<=threshold<r['best256_iou'] for r in rows)
        assert coverage==diagnostic[suffix]['full256_covered']
        assert errors_with_candidate==diagnostic[suffix]['errors_with_qualified_candidate']
        assert all(0<=r['qualified256_count'+suffix]<=256 for r in rows)
        assert all((r['first_qualified_rank'+suffix] is None)==(r['qualified256_count'+suffix]==0) for r in rows)
        report[suffix] = dict(hits=int(hit.sum()),accuracy_percent=100*float(hit.mean()),
            cpu64_vs_logged_threshold_disagreements=decisions,final_vs_native=native_change,
            final_vs_mask_endpoint=mask_change,versus_E0=change(initial_iou>threshold,hit),
            support_qualified_selected_box_wrong=int((support&~hit).sum()),
            logged_full256_coverage=coverage,logged_errors_with_qualified_candidate=errors_with_candidate)
    with (directory/('epoch%03d_selected_rows.csv'%epoch)).open('x',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_index','scan_id','target_id','utterance_sha256','selected_query','final_iou_cpu64',
            'native_iou_cpu64','mask_endpoint_iou_cpu64','E0_selected_iou_cpu64','own_and_fused_mask_qualified_logged_counts'])
        for idx,r in enumerate(rows):
            writer.writerow([idx,r['scan_id'],r['target_id'],r['utterance_sha256'],r['selected_query'],values['final'][idx],
                values['native'][idx],values['mask_endpoint'][idx],initial_iou[idx],bool(support[idx])])
    reports.append(report)
result = dict(scope='CLOSED_CONTROL_SAVED_SELECTED_BOX_COORDINATE_RECOUNT',evaluation_type='real_gt',
    completed_cst=datetime.datetime.now().astimezone().isoformat(),metrics=metrics,epochs=reports,
    source_hashes=source_hashes,inputs_not_loaded=True,masks_not_recomputed=True,
    full256_candidate_boxes_not_available=True,original_matching_eligibility_not_recomputed=True,
    model_checkpoint_not_loaded=True,neural_forwards=0,optimizer_updates=0,remote_queries=0,
    new_metric_or_best_promoted=False,full_goal_complete=False)
(directory/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(scope=result['scope'],epochs=[dict(epoch=x['epoch'],hits025=x['025']['hits'],hits050=x['050']['hits']) for x in reports])))
